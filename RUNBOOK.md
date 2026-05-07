# Runbook

Operational guide for running the Drift Triage Co-Pilot stack from a clean clone.

---

## Prerequisites

- Docker Desktop installed and running
- `git`, `uv` (optional — only needed for local dev outside Docker)
- An Anthropic API key if you want LLM-driven agent nodes (not required for CI or queue tests)

---

## 1. First-Time Setup (clean clone)

```bash
git clone <repo-url>
cd drift-triage-copilot

cp .env.example .env
```

Open `.env` and fill in any secrets you need:

| Variable | Required | Default in docker-compose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Optional (LLM calls) | — |
| `MLFLOW_TRACKING_URI` | No | `sqlite:///mlflow.db` |
| `AGENT_CHECKPOINTER` | No | `postgres` (set in compose) |

All other variables (`DATABASE_URL`, `REDIS_URL`, service URLs, artifact paths) are already set correctly in `docker-compose.yml` for inter-container networking. You do not need to change them.

---

## 2. Prepare ML Artifacts

The model service needs trained artifacts in `artifacts/` before it can serve predictions. If they are not present (fresh clone), run the ML pipeline first:

```bash
# Install dependencies locally (or skip and use docker exec after compose up)
pip install uv
uv sync

# 1. Download dataset
python scripts/fetch_data.py

# 2. Create train / val / test splits
python scripts/make_splits.py

# 3. Train model, tune threshold, register in MLflow
python scripts/train_register_model.py
```

This produces:
- `artifacts/models/bank_marketing_pipeline.joblib` — fitted sklearn pipeline
- `artifacts/reports/threshold.json` — operating threshold (recall >= 0.75 on val)
- `artifacts/model_cards/bank_marketing_model_card.md`
- `mlruns/` — MLflow tracking directory

> If you already have `artifacts/` from a previous run, skip this step.

---

## 3. Start the Stack

```bash
docker compose up --build
```

First build takes ~2 minutes. Subsequent starts are faster.

Services and their ports:

| Service | Port | URL |
|---|---|---|
| Model Service | 8000 | http://localhost:8000 |
| Agent Service | 8001 | http://localhost:8001 |
| Dashboard | 8501 | http://localhost:8501 |
| Redis | 6379 | (internal) |
| Postgres | 5432 | (internal) |

All services are healthy when you see no restart loops in `docker compose logs`.

### Health checks

```bash
curl http://localhost:8000/health   # {"status": "ok", "service": "drift-triage-model-service"}
curl http://localhost:8001/health   # {"status": "ok", "service": "drift-triage-agent"}
```

---

## 4. Send a Prediction

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1
  }' | python -m json.tool
```

Do **not** include `duration` — it is a leakage feature and the service will return HTTP 422 if you send it.

---

## 5. Simulate Drift

Generate enough predictions to fill the rolling window and trigger a drift alert:

```bash
# Send 150 predictions with shifted feature distributions
python scripts/simulate_drift.py --mode drift --count 150
```

Then trigger a drift check (this also fires the webhook to the agent):

```bash
curl -s -X POST http://localhost:8000/drift/check | python -m json.tool
```

When severity changes (e.g., NONE → HIGH), the model service posts to
`http://agent-service:8001/webhooks/drift` and the agent opens an investigation.

---

## 6. Inspect the Agent

```bash
# List all investigations
curl -s http://localhost:8001/investigations | python -m json.tool

# Get one investigation by ID
curl -s http://localhost:8001/investigations/<investigation_id> | python -m json.tool
```

---

## 7. Human-in-the-Loop Approval

Open the dashboard at http://localhost:8501 and go to the **Approvals** tab.

When the agent recommends an action that touches Production (retrain or rollback), a pending approval appears here. Submit a decision (approve / reject) before the agent can proceed.

Approvals can also be managed via API:

```bash
# View pending approvals
curl -s http://localhost:8000/approvals/pending | python -m json.tool

# Submit a decision
curl -s -X POST http://localhost:8000/approvals/decision \
  -H "Content-Type: application/json" \
  -d '{"approval_id": "<id>", "decision": "approved", "decided_by": "operator"}' \
  | python -m json.tool
```

---

## 8. Queue and DLQ

The worker container (`drift-worker`) polls Redis for jobs dispatched by the agent.

```bash
# Check queue depth
curl -s http://localhost:8000/queue/status | python -m json.tool
```

Jobs: `replay_test`, `retrain`, `rollback`. Each retries up to 3 times (exponential backoff: 1 s, 2 s, 4 s) before landing in the dead-letter queue.

---

## 9. MLflow Registry

```bash
# Current production model
curl -s http://localhost:8000/registry/production | python -m json.tool

# Promote a candidate (requires approved=True — set by HIL flow)
curl -s -X POST http://localhost:8000/registry/promote \
  -H "Content-Type: application/json" \
  -d '{"version": "2", "approved": true}' | python -m json.tool
```

MLflow UI (local, not in Docker):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```

---

## 10. Run Tests

```bash
uv sync
python -m pytest tests/ -v
```

Expected: **21 passed**. No API key required — trajectory snapshot tests use a deterministic mock (no LLM calls in CI).

---

## 11. Stop the Stack

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop containers and delete Postgres volume (full reset)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| model-service exits at startup | Missing `artifacts/models/bank_marketing_pipeline.joblib` | Run `scripts/train_register_model.py` first |
| agent-service exits at startup | Postgres not ready | Wait 10 s and run `docker compose restart agent-service` |
| Drift check returns no change | Not enough predictions in rolling window | Run `simulate_drift.py --count 150` |
| Worker job lands in DLQ | Task crashed 3 times | Check `docker compose logs worker`; inspect DLQ via `/queue/status` |
| 422 on `/predict` | `duration` field included in payload | Remove `duration` from request |
