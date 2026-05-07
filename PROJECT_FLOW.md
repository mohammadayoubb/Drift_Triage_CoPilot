# Drift Triage Co-Pilot — Project Flow

End-to-end architecture, data flow, and service wiring for the Week 5 MLOps assignment.

---

## Stack Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        docker compose up                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │ postgres │   │    redis     │   │  mlruns/     │               │
│  │  :5432   │   │   :6379      │   │  (volume)    │               │
│  └────┬─────┘   └──────┬───────┘   └──────────────┘               │
│       │                │                                            │
│  ┌────▼────────────────▼──────────────────────────────────────┐    │
│  │              model-service  :8000                          │    │
│  │  FastAPI · sklearn pipeline · drift monitor · MLflow       │    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                               │ POST /webhooks/drift                │
│  ┌────────────────────────────▼───────────────────────────────┐    │
│  │              agent-service  :8001                          │    │
│  │  FastAPI · LangGraph supervisor · Postgres checkpoints     │    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                               │ enqueue job                        │
│  ┌────────────────────────────▼───────────────────────────────┐    │
│  │                    worker                                  │    │
│  │  async_queue · replay_test · retrain · rollback · DLQ     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  dashboard  :8501                           │   │
│  │  Streamlit · registry · drift · investigations · HIL inbox  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Service Responsibilities

### model-service (`service/`)

Entry point: `uvicorn service.main:app --port 8000`

| Responsibility | Code |
|---|---|
| Prediction serving | `service/api/routes_predictions.py` |
| Pydantic input validation (blocks `duration`) | `service/schemas.py` |
| sklearn pipeline + threshold | `service/model/loader.py` |
| Rolling window prediction log | `service/drift/report_builder.py` |
| PSI on numeric features | `service/drift/psi.py` |
| Chi-square on categorical features | `service/drift/chi_square.py` |
| Severity classification (LOW/MEDIUM/HIGH/CRITICAL) | `service/drift/drift_monitor.py` |
| Drift webhook dispatch | `service/drift/webhook_client.py` |
| MLflow registry routes | `service/api/routes_registry.py` |
| Approval inbox (create / list / decide) | `service/api/routes_approvals.py` |
| Queue status endpoint | `service/api/routes_queue.py` |

Artifact paths (set in `docker-compose.yml`, loaded via `service/config/settings.py`):

```
artifacts/models/bank_marketing_pipeline.joblib   ← fitted sklearn pipeline
artifacts/reports/threshold.json                  ← {"threshold": 0.385, ...}
```

---

### agent-service (`src/agent/`)

Entry point: `uvicorn agent.main:app --port 8001` with `PYTHONPATH=/app/src`

| Responsibility | Code |
|---|---|
| Webhook receiver | `src/agent/api/routes.py` → `POST /webhooks/drift` |
| Create/read/list investigations | `src/agent/persistence/investigation_store.py` |
| LangGraph graph execution | `src/agent/graph/runner.py` |
| Supervisor topology | `src/agent/graph/supervisor.py` |
| Triage sub-agent | `src/agent/graph/nodes/triage.py` |
| Action sub-agent + queue dispatch | `src/agent/graph/nodes/action.py` |
| Comms sub-agent | `src/agent/graph/nodes/comms.py` |
| Postgres / memory checkpointer | `src/agent/persistence/checkpoints.py` |
| Prompt files | `src/agent/prompts/*.md` |

State machine inside the LangGraph graph:

```
START
  │
  ▼
supervisor  ──── status == "opened"          ──→  triage  ──┐
  │                                                          │ sets status = "triaged"
  │         ──── status == "triaged"         ──→  action  ──┤
  │                                                          │ sets status = "action_recommended"
  │         ──── status == "action_recommended" → comms   ──┤
  │                                                          │ sets status = "resolved"
  └─────────────── status == "resolved"      ──→  END
```

Each sub-agent returns to the supervisor after it runs. The supervisor reads `state["status"]` to pick the next step. `comms_node` sets `status = "resolved"` so the graph terminates.

Checkpoint flow:
- `AGENT_CHECKPOINTER=postgres` → state survives container restarts
- `AGENT_CHECKPOINTER=memory` → used in CI and local dev (no Postgres needed)

---

### worker (`async_queue/`)

Entry point: `python -m async_queue.worker`

| Responsibility | Code |
|---|---|
| Job dequeue loop | `async_queue/worker.py` |
| Exponential backoff retry (3 attempts, 2^n s) | `async_queue/worker.py` |
| Dead-letter queue on final failure | `async_queue/dlq.py` |
| Idempotency (Redis key per job_id) | `async_queue/idempotency.py` |
| `replay_test` handler | `async_queue/tasks.py` |
| `retrain` handler | `async_queue/tasks.py` |
| `rollback` handler | `async_queue/tasks.py` |

Job schema (`async_queue/job_models.py`):

```json
{
  "job_id": "uuid",
  "job_type": "replay_test | retrain | rollback",
  "payload": { "investigation_id": "inv_..." },
  "idempotency_key": "inv_...-replay_test"
}
```

`replay_test` loads the model from `artifacts/models/`, runs `predict_proba` on `data/processed/test.csv`, computes AUC / F1 / recall, and writes a timestamped report to `artifacts/reports/replay_reports/`.

---

### dashboard (`dashboard/app.py`)

Entry point: `streamlit run dashboard/app.py --server.port 8501`

Talks to model-service (`:8000`) and agent-service (`:8001`) via HTTP.

Tabs:
1. **Health** — service status for both APIs
2. **Registry** — current production model, promote candidate
3. **Drift** — current drift report (severity, affected features)
4. **Investigations** — list and detail view of agent investigations
5. **Approvals** — HIL inbox; submit approve / reject decisions

---

## Data Flow: Prediction → Drift → Agent → Resolution

```
Client
  │
  │  POST /predict  {age, job, marital, ...}   (no duration field)
  ▼
model-service
  ├── Pydantic validates input schema
  ├── sklearn pipeline: preprocess → predict_proba
  ├── Apply threshold (0.385) → binary label
  ├── Log prediction to rolling window (last 100)
  └── Return {prediction, probability}

  (after N predictions)

  POST /drift/check  (or automatic on rolling window fill)
  ├── PSI computed on numeric features vs. reference training stats
  ├── Chi-square computed on categorical features
  ├── Severity assigned: LOW / MEDIUM / HIGH / CRITICAL
  └── If severity changed → POST http://agent-service:8001/webhooks/drift
        body: DriftEvent (contract v1.0)

agent-service
  ├── Creates InvestigationRecord (JSON store + Postgres checkpoint)
  ├── Runs LangGraph graph:
  │     supervisor → triage → supervisor → action → supervisor → comms → END
  │
  │   triage_node:
  │     - Reads drift_event fields (numeric_drift_summary, categorical_drift_summary)
  │     - Classifies affected features
  │     - Sets triage_result, status = "triaged"
  │
  │   action_node:
  │     - Picks recommended_action based on severity + triage result
  │       · CRITICAL/HIGH → replay_test job enqueued to Redis
  │       · touches_production=True → requires_human_approval = True
  │     - Sets status = "action_recommended"
  │
  │   comms_node:
  │     - Writes comms_summary (human-readable investigation summary)
  │     - Sets status = "resolved"
  │
  └── Updates InvestigationRecord with final state

worker (if queue job was dispatched)
  ├── Dequeues job from Redis (idempotency check first)
  ├── Runs handler (replay_test / retrain / rollback)
  ├── On failure: retry up to 3× with backoff, then push to DLQ
  └── Writes results to artifacts/reports/

Human (via dashboard Approvals tab)
  ├── Sees pending approval for Production-touching action
  ├── Submits decision: approved / rejected
  └── POST /approvals/decision → model-service stores decision
        → agent or operator calls POST /registry/promote with approved=True
```

---

## Contract

The drift event schema is versioned at `contracts/drift_event.v1.json`.

Key fields the model-service sends and the agent expects:

```json
{
  "contract_version": "1.0",
  "event_id": "uuid",
  "event_time": "ISO-8601",
  "model_name": "bank-marketing-classifier",
  "model_version": "1",
  "severity": "HIGH",
  "previous_severity": "LOW",
  "affected_features": ["euribor3m", "poutcome"],
  "drift_type": "numeric_and_categorical",
  "numeric_drift_summary": { "euribor3m": { "psi": 0.42 } },
  "categorical_drift_summary": { "poutcome": { "p_value": 0.003 } }
}
```

Schema changes to this contract are breaking. Increment `contract_version` and update both sides.

---

## Repository Layout

```
drift-triage-copilot/
│
├── service/                  ← Model service (FastAPI)
│   ├── api/                  ← Route modules (predictions, drift, registry, approvals, queue)
│   ├── config/settings.py    ← Env-var config (MODEL_PATH, AGENT_WEBHOOK_URL, etc.)
│   ├── drift/                ← PSI, chi-square, severity, webhook client
│   ├── model/                ← sklearn pipeline loader + threshold
│   ├── storage/              ← Approval store, drift state, prediction log
│   └── main.py
│
├── src/
│   ├── agent/                ← LangGraph agent (FastAPI)
│   │   ├── api/routes.py     ← /webhooks/drift, /investigations
│   │   ├── graph/
│   │   │   ├── supervisor.py ← StateGraph topology
│   │   │   ├── state.py      ← AgentState TypedDict
│   │   │   ├── runner.py     ← graph.invoke() wrapper
│   │   │   └── nodes/        ← triage.py, action.py, comms.py
│   │   ├── persistence/
│   │   │   ├── checkpoints.py          ← Postgres / memory saver
│   │   │   └── investigation_store.py  ← JSON file CRUD
│   │   ├── schemas/
│   │   │   ├── drift_event.py    ← Pydantic model matching contract v1
│   │   │   └── investigation.py  ← InvestigationRecord, response models
│   │   └── prompts/          ← triage.md, action.md, comms.md, supervisor.md
│   │
│   ├── ml/                   ← ML pipeline scripts
│   │   ├── clean.py          ← Drop duration, flag pdays==999
│   │   ├── split.py          ← 60/20/20 stratified split
│   │   ├── pipeline.py       ← sklearn ColumnTransformer + LogisticRegression
│   │   ├── train.py          ← Fit pipeline, tune threshold, evaluate on test
│   │   ├── threshold.py      ← Highest threshold with recall >= 0.75
│   │   ├── registry.py       ← MLflow log_and_register_model
│   │   └── model_card.py     ← SHA256 hash + env fingerprint
│   │
│   └── common/               ← Shared utilities
│       ├── logging.py        ← Structured logging config
│       ├── paths.py          ← Project path constants
│       └── hashing.py        ← Artifact hashing
│
├── async_queue/              ← Redis queue
│   ├── worker.py             ← Poll loop + exponential backoff retry
│   ├── tasks.py              ← replay_test, retrain, rollback handlers
│   ├── producer.py           ← Enqueue jobs
│   ├── dlq.py                ← Dead-letter queue
│   ├── idempotency.py        ← Redis-backed deduplication
│   └── job_models.py         ← QueueJob Pydantic model
│
├── dashboard/app.py          ← Streamlit UI (5 tabs)
│
├── scripts/                  ← One-off ML scripts
│   ├── fetch_data.py         ← Download UCI dataset via kagglehub
│   ├── make_splits.py        ← Produce data/processed/{train,val,test}.csv
│   ├── train_register_model.py ← Full train → threshold → MLflow register
│   ├── simulate_drift.py     ← Send shifted predictions to trigger drift
│   └── generate_predictions.py ← Send normal-distribution predictions
│
├── contracts/
│   └── drift_event.v1.json   ← Versioned webhook contract
│
├── tests/
│   ├── agent/
│   │   ├── test_agent_api.py           ← Webhook + investigation endpoints
│   │   └── test_trajectory_snapshots.py ← Graph routing (no LLM key needed)
│   ├── ml/
│   │   ├── test_model_fidelity.py      ← 1e-12 tolerance replay test
│   │   └── fixtures/reference_proba.npy
│   ├── test_approvals.py
│   ├── test_imports.py
│   ├── test_queue_models.py
│   ├── test_request_schema.py
│   └── test_worker.py
│
├── artifacts/                ← Generated by ML pipeline (gitignored)
│   ├── models/               ← bank_marketing_pipeline.joblib
│   ├── reports/              ← threshold.json, replay reports
│   └── model_cards/
│
├── data/                     ← Gitignored at runtime
│   ├── raw/                  ← bank-additional-full.csv
│   ├── processed/            ← train.csv, val.csv, test.csv
│   └── reference/            ← reference_stats.json (from training split)
│
├── docker-compose.yml        ← 6 services: postgres, redis, model-service, agent-service, worker, dashboard
├── Dockerfile                ← uv sync --frozen from pyproject.toml
├── pyproject.toml            ← Single source of dependencies (uv)
├── .env.example              ← Copy to .env before first run
├── RUNBOOK.md                ← This stack's operational guide
├── ARCH.md                   ← Architecture narrative
├── Decisions.md              ← Key design decisions and trade-offs
└── PROJECT_FLOW.md           ← This file
```

---

## CI Pipeline (`.github/workflows/ci.yml`)

Triggers on every push. Python 3.12. Uses `uv sync` (not requirements.txt).

Steps:
1. Start Redis service container (`:6379`)
2. Install dependencies via `uv sync`
3. Compile all modules (`python -m compileall service agent async_queue approvals dashboard tests src`)
4. Run `pytest tests/ -v` with:
   - `REDIS_URL=redis://localhost:6379/0`
   - `AGENT_CHECKPOINTER=memory` (no Postgres in CI)

The trajectory snapshot tests run without an LLM API key — the graph is deterministic (no LLM calls in CI mode). The model fidelity test requires `artifacts/models/bank_marketing_pipeline.joblib` and `tests/ml/fixtures/reference_proba.npy` to be present.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Drift detection | PSI (numeric) + chi-square (categorical) | Standard MLOps drift metrics; PSI is threshold-free |
| Operating threshold rule | Highest threshold with recall >= 0.75 | Preserve true-positive rate for campaign targeting |
| Leakage removal | Drop `duration` | Recorded after call ends — not available at prediction time |
| `pdays==999` handling | Binary flag `pdays_was_999` | 999 is a sentinel meaning "never contacted" — not a numeric value |
| Agent topology | LangGraph supervisor with 3 sub-agents | Assignment requirement; supervisor pattern enables conditional routing |
| Checkpoint store | Postgres in prod, InMemorySaver in CI | Postgres survives restarts; memory avoids Postgres dependency in tests |
| Queue idempotency | Redis key per `job_id` | Prevents double-processing on retry |
| Promotion gate | `approved=True` required on `/registry/promote` | No Production change without explicit HIL decision |
| Contract versioning | `contracts/drift_event.v1.json` | Schema changes are breaking — version in filename and field |
| Dependency management | `uv` + `pyproject.toml` | `requirements.txt` is superseded and excluded from install |
