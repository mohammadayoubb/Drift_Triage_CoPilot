# Drift Triage CoPilot — Code Review Study Guide

---

## 1. ARCHITECTURE SUMMARY (memorize this)

### The One-Paragraph Version

> A LogisticRegression model is trained on the UCI Bank Marketing dataset and served via FastAPI. Every prediction is logged to a rolling window. A drift checker runs PSI on numeric features and chi-square on categorical features. When severity changes, it webhooks a LangGraph agent. The agent's supervisor routes through three nodes: triage (what drifted?), action (what to do?), comms (write the summary). If drift is critical, the action node pauses and waits for a human to approve before dispatching jobs. Approved jobs go into a Redis queue where a background worker executes replay, retrain, or rollback. Every production model change requires explicit human approval. A Streamlit dashboard shows everything in real time.

### The Architecture in 6 Lines

```
Predictions → predictions.jsonl → Drift Checker (PSI + chi²)
    ↓ if severity changes
DriftEvent → LangGraph Agent (triage → action → comms)
    ↓ if CRITICAL: interrupt() → human approves → 
Redis Queue → Worker (replay_test → retrain → promote approval)
    ↓ human approves promotion →
MLflow Registry: Staging → Production
```

### 6 Services in Docker Compose

| Service | Port | What it does |
|---------|------|-------------|
| `model-service` | 8000 | FastAPI: predict, drift, registry, approvals, queue |
| `agent-service` | 8001 | LangGraph: receive drift webhook, run investigation |
| `worker` | — | Polls Redis, executes replay/retrain/rollback |
| `dashboard` | 8501 | Streamlit: 5-tab UI for full demo |
| `postgres` | 5432 | LangGraph checkpoint storage (agent state) |
| `redis` | 6379 | Job queue (drift_jobs list) |

---

## 2. FILE-BY-FILE CHEAT SHEET

### ML Training
| File | One-line job |
|------|-------------|
| `src/ml/constants.py` | Single source of truth: 10 numeric features, 10 categorical, random_state=42, recall_target=0.75 |
| `src/ml/pipeline.py` | sklearn: ColumnTransformer (impute+scale/encode) + LogisticRegression(class_weight="balanced") |
| `src/ml/train.py` | Full pipeline: load → train → tune threshold → eval → save artifacts → register MLflow |
| `scripts/make_splits.py` | raw CSV → 60/20/20 train/val/test, drops `duration`, engineers `pdays_was_999` |
| `scripts/simulate_drift.py` | Sends normal or drifted prediction batches to `/predict` for demo |
| `scripts/train_register_model.py` | Entry point: calls `train_model()` |

### Model Artifacts
| File | What's inside |
|------|--------------|
| `artifacts/models/bank_marketing_pipeline.joblib` | Serialized sklearn pipeline (7.6 KB) |
| `artifacts/reports/metrics.json` | AUC=0.801, Recall=0.748, F1=0.371, Precision=0.247 |
| `artifacts/reports/threshold.json` | Operating threshold = 0.385 |
| `artifacts/reports/input_schema.json` | 20-feature schema with dtypes |
| `data/reference/reference_stats.json` | Reference distributions for PSI + chi-square |

### Service Layer
| File | One-line job |
|------|-------------|
| `service/main.py` | FastAPI app, registers all 8 route groups |
| `service/config/settings.py` | Env-backed config: MODEL_PATH, AGENT_WEBHOOK_URL, DRIFT_WINDOW_SIZE=100 |
| `service/model/loader.py` | Loads joblib pipeline + threshold from disk |
| `service/model/predictor.py` | Wraps pipeline: `predict()` → {prediction, probability_yes, threshold_used} |
| `service/validation/request_schema.py` | Pydantic: forbids `duration`, `y`, `pdays_was_999` as inputs |
| `service/api/routes_predictions.py` | POST /predict: validate → predict → log to JSONL |
| `service/api/routes_drift.py` | GET /drift/status (no notify) + POST /drift/check (full + notify) |
| `service/api/routes_registry.py` | GET /registry/production, /candidate + POST /registry/promote (requires approved=True) |
| `service/api/routes_approvals.py` | CRUD for approval requests and decisions |
| `service/api/routes_demo.py` | POST /demo/reset: archive state, clear Redis, reset drift baseline |

### Drift Detection
| File | One-line job |
|------|-------------|
| `service/drift/psi.py` | PSI on numeric features: 10-bin deciles, thresholds <0.1/0.1-0.25/>0.25 |
| `service/drift/chi_square.py` | Chi-square on categoricals: scipy, p<0.001=HIGH, p<0.05=MEDIUM |
| `service/drift/drift_monitor.py` | Runs both detectors, aggregates to overall severity |
| `service/drift/drift_service.py` | Orchestrates check, compares to saved state, webhooks agent if changed |
| `service/storage/prediction_store.py` | Append/read `data/predictions.jsonl` |
| `service/storage/drift_state_store.py` | Persist last severity to `data/drift_state.json` |
| `service/storage/reference_store.py` | Load `data/reference/reference_stats.json` |

### LangGraph Agent
| File | One-line job |
|------|-------------|
| `src/agent/graph/state.py` | AgentState TypedDict: investigation_id, severity, status, triage_result, recommended_action, requires_human_approval |
| `src/agent/graph/supervisor.py` | Deterministic router: opened→triage, triaged→action, action_recommended→comms |
| `src/agent/graph/nodes/triage.py` | Top-5 numeric (by PSI) + top-5 categorical (by chi²), optional GPT-4o-mini analysis |
| `src/agent/graph/nodes/action.py` | CRITICAL→interrupt(), HIGH→enqueue replay_test, LOW/MEDIUM→no action |
| `src/agent/graph/nodes/comms.py` | GPT-4o-mini summary or deterministic fallback, sets status="resolved" |
| `src/agent/graph/runner.py` | Entry point: `run_investigation_graph()`, `@lru_cache` singleton graph |
| `src/agent/persistence/checkpoints.py` | InMemorySaver (dev/CI) or PostgresSaver (docker, env: AGENT_CHECKPOINTER) |
| `src/agent/persistence/investigation_store.py` | Thread-safe JSONL store, investigation_id = thread_id |
| `src/agent/api/routes.py` | POST /webhooks/drift: create investigation → run graph → detect __interrupt__ → request approval |

### Async Queue
| File | One-line job |
|------|-------------|
| `async_queue/job_models.py` | QueueJob: job_id, job_type, payload, idempotency_key |
| `async_queue/producer.py` | RPUSH QueueJob JSON to Redis list `drift_jobs` |
| `async_queue/worker.py` | LPOP → idempotency check → execute with retry (3×, exponential) → DLQ → auto-chain |
| `async_queue/tasks.py` | run_replay_test(), run_retrain(), run_rollback() |
| `async_queue/idempotency.py` | Redis keys: `idempotency:{key}` (done) and `queued:{key}` (enqueued) |
| `async_queue/dlq.py` | RPUSH failed jobs to `drift_jobs_dlq` after max retries |
| `async_queue/status.py` | Queue depth, running jobs, DLQ, completed — for dashboard |

### Human Approval
| File | One-line job |
|------|-------------|
| `approvals/approval_models.py` | ApprovalRequest + ApprovalDecision Pydantic models |
| `approvals/approval_store.py` | Append-only JSONL at `data/approvals.jsonl`; pending(), history(), find methods |
| `approvals/approval_service.py` | create_request() (dedup), decide() (dispatch job if approved), list_pending/history |

### Dashboard
| File | One-line job |
|------|-------------|
| `dashboard/app.py` | Streamlit, 5 tabs: Demo Control / Drift & Agent / Human Approval / Queue / Model & CI |
| `dashboard/api_client.py` | HTTP wrappers to model-service (:8000) and agent-service (:8001) |

---

## 3. WHAT HAPPENS WHEN I CLICK / RUN X

### `python scripts/simulate_drift.py --mode drift --count 200`
1. Sends 200 POST /predict requests with artificially shifted data
2. `euribor3m` values are shifted to modern rates (high PSI guaranteed)
3. `contact` distribution skewed toward "cellular" only (high chi-square p-value guaranteed)
4. Each prediction is logged to `data/predictions.jsonl`
5. After 100+ records, `/drift/check` will detect CRITICAL severity

### `POST /drift/check`
1. Loads last 100 records from `data/predictions.jsonl`
2. Runs PSI on all 10 numeric features vs `reference_stats.json`
3. Runs chi-square on all 10 categorical features
4. Aggregates to overall severity (CRITICAL = high numeric + categorical)
5. Compares to `data/drift_state.json` (last known severity)
6. If changed: builds DriftEvent → POSTs to `http://agent:8001/webhooks/drift`
7. Saves new severity to `data/drift_state.json` only if webhook succeeds

### `POST /webhooks/drift` (received by agent)
1. Creates investigation record → `data/investigations.jsonl`
2. Calls `run_investigation_graph(event, investigation_id, thread_id)`
3. LangGraph: supervisor → triage node
   - Extracts top drifted features (PSI rank for numeric, chi² rank for categorical)
   - Calls GPT-4o-mini if API key present (graceful fallback if not)
   - Sets `status="triaged"`
4. LangGraph: supervisor → action node
   - **CRITICAL:** calls `interrupt()` → graph **pauses** → returns `__interrupt__` in state
   - **HIGH:** enqueues `replay_test` to Redis → sets `status="action_recommended"` → continues
5. API layer detects `__interrupt__`:
   - Updates investigation status to `"approval_pending"`
   - POSTs to `model-service /approvals/request` with investigation_id + action_type

### Clicking "Approve" on workflow approval (Tab 3)
1. Dashboard calls `POST /approvals/decision` with `approved=True, approved_by="name"`
2. `approval_service.decide()` runs:
   - Checks not already decided (double-click guard)
   - Maps `action_type="replay_test_set_then_open_retrain_candidate"` → `job_type="replay_test"`
   - Checks `queued:` idempotency namespace (prevents double-queue)
   - Creates `QueueJob` → `QueueProducer().enqueue()` → RPUSH to Redis `drift_jobs`
3. Records decision in `data/approvals.jsonl`

### Worker processes `replay_test` job
1. LPOP from `drift_jobs`
2. Checks `idempotency:` namespace → not seen → proceed
3. HSET to `drift_jobs_running`
4. Calls `run_replay_test()`:
   - Loads production model + threshold
   - Runs predictions on `data/processed/test.csv`
   - Computes AUC, F1, recall, precision
   - Writes timestamped report to `artifacts/reports/replay_reports/`
5. Marks seen in `idempotency:` namespace
6. RPUSH to `drift_jobs_completed`
7. **Auto-enqueues `retrain` job** with `idempotency_key="{investigation_id}-retrain"`

### Worker processes `retrain` job
1. Calls `run_retrain()`:
   - Calls `train_model()` on current data
   - MLflow logs new run
   - Transitions new version to **Staging** stage
2. **Auto-creates promotion approval** via `ApprovalService.create_request(action_type="promote_candidate_model")`

### Clicking "Approve" on promotion approval (Tab 3)
1. Dashboard calls `POST /registry/promote` with `{"candidate_version": "2", "approved": true}`
2. `mlflow_registry.promote_to_production()` transitions Staging → Production
3. Previous Production archived to "Archived" stage

### `POST /demo/reset`
1. Archives runtime state to `artifacts/archive/reset_<timestamp>/`
   - Copies predictions.jsonl, approvals.jsonl, investigations.jsonl
2. Clears Redis: deletes `drift_jobs`, `drift_jobs_running`, `drift_jobs_completed`, `drift_jobs_dlq`, all idempotency keys
3. Resets drift baseline to "LOW" in `data/drift_state.json`
4. Nothing is deleted — everything is archived

---

## 4. LIKELY REVIEWER QUESTIONS + STRONG ANSWERS

### ML Questions

**Q: Why LogisticRegression instead of a tree-based model?**
> "Three reasons: interpretability (coefficients are directly readable), calibrated probabilities (we apply a custom threshold so we need reliable probability estimates, not just rankings), and fast retraining (the auto-retrain pipeline runs during live operations — LR trains in seconds vs minutes for ensemble models). The trade-off is lower raw AUC, but 0.801 is adequate for this use case."

**Q: Why is the threshold 0.385 instead of 0.5?**
> "The model is trained on a class-imbalanced dataset — only about 11% of contacts result in a subscription. At the default 0.5 threshold, the model is too conservative and misses many real subscribers. We tune the threshold on the validation set: we find the highest threshold where recall stays at or above 0.75. That gives us 0.385. We accept lower precision (0.247 — more false positives) to guarantee we capture at least 75% of real subscribers."

**Q: Why drop `duration`?**
> "Duration is how long the phone call lasted. It perfectly predicts the outcome — short calls mean no interest, long calls mean a sale. But it's only known after the call ends, which is after the decision to call was made. Including it would be data leakage: great training metrics, broken real-world performance."

**Q: What is `pdays_was_999`?**
> "In the raw data, `pdays` (days since last contact) has the value 999 when a client was never previously contacted. That's a sentinel value that can't be compared numerically to real day counts. We engineer a binary flag `pdays_was_999` to capture that meaning explicitly before scaling."

**Q: What does `class_weight='balanced'` do?**
> "It automatically adjusts the loss function to up-weight the minority class (subscribers, ~11% of data) inversely proportional to its frequency. Without this, the model would learn to predict 'no' for almost everyone and still score well on accuracy. Balanced weighting forces it to take the minority class seriously."

### Drift Questions

**Q: How does PSI work exactly?**
> "PSI measures distribution shift for numeric features. We take the reference distribution and split it into 10 equal-frequency buckets (deciles). We record what percentage of reference values fall in each bucket. Then we run the same bucketing on current live predictions. PSI = sum of (current_pct - reference_pct) × log(current_pct / reference_pct) across all buckets. Values below 0.1 mean stable; 0.1 to 0.25 is moderate shift worth monitoring; above 0.25 is critical drift. The log term penalizes large relative differences even in small buckets."

**Q: How does chi-square drift detection work?**
> "Chi-square goodness of fit. We take the reference category distribution for a feature — say, 40% cellular, 60% telephone for `contact`. We scale those proportions to the current sample size to get expected counts. We then compare to observed counts with χ² = sum of (observed - expected)² / expected. The resulting p-value tells us whether the observed distribution could have come from the reference by chance. p < 0.05 means statistically significant drift. In our simulation, `contact` shifts to nearly 100% cellular — impossible by chance from the reference distribution."

**Q: Why two different methods for numeric vs categorical?**
> "PSI is designed for continuous distributions where binning makes sense and you want a threshold-based severity score that doesn't depend on sample size. Chi-square is designed for categorical count data where you want a hypothesis test. Using PSI for categoricals or chi-square for numerics would give less reliable signals."

**Q: What triggers agent notification? Every drift check?**
> "No — only when severity *changes*. We persist the last-known severity to `data/drift_state.json`. If the severity is already CRITICAL and we check again, the agent is not notified again. This prevents alert storms. The agent is only notified on transitions: LOW→HIGH, HIGH→CRITICAL, etc. We also only persist the new severity if the webhook delivery succeeded — so a failed delivery means the next check will retry."

### Agent Questions

**Q: Why LangGraph instead of a simple function?**
> "Three reasons: stateful resumption, deterministic testability, and future extensibility. The key feature we use is `interrupt()` — LangGraph can pause a running graph mid-execution, checkpoint the exact state, and resume it later when a human provides approval. A plain function can't do this. The supervisor's routing is also trivially unit-testable because it's pure deterministic logic — no LLM involved. And adding new nodes (e.g., a rollback investigation node) is additive, not a refactor."

**Q: What does `interrupt()` actually do in LangGraph?**
> "It's a built-in LangGraph primitive that pauses graph execution and returns a special `__interrupt__` key in the output state. The graph's state is checkpointed to Postgres at that point. Our API layer detects this `__interrupt__` key in the response, records the investigation as `approval_pending`, and requests a human approval. When a human approves, the graph can be resumed from the checkpoint — it picks up exactly where it paused."

**Q: How is agent state persisted across restarts?**
> "LangGraph checkpoints state at every node boundary using a configurable backend. In our Docker deployment, `AGENT_CHECKPOINTER=postgres` uses PostgresSaver, which writes state to the Postgres container. The checkpoint key is `thread_id`, which we set equal to `investigation_id`. So even if the agent service restarts, a graph in `approval_pending` state can be resumed using the same thread_id."

**Q: What happens if OpenAI API key is missing?**
> "The triage and comms nodes degrade gracefully. Triage returns a deterministic summary of the top drifted features without LLM interpretation. Comms returns a template-based summary listing the features and recommended action. The supervisor routing, interrupt logic, and queue dispatch are completely independent of the LLM — those work with or without an API key."

### Queue Questions

**Q: Why Redis instead of a database queue?**
> "Redis LPOP/RPUSH gives us an atomic O(1) FIFO queue without transaction overhead. We don't need persistence guarantees for the queue itself (jobs can be re-dispatched if they fail via the retry mechanism) and we do need low latency for the polling worker. The idempotency layer using Redis keys ensures we don't double-process even if the queue is replayed."

**Q: What prevents a job from running twice?**
> "Idempotency keys. Every `QueueJob` has an `idempotency_key` set to `{investigation_id}-{job_type}`. Before processing, the worker checks a Redis key `idempotency:{key}` — if it exists, the job is skipped. After successful completion, the worker marks the key. There's also a second namespace `queued:{key}` used by the approval service before enqueuing, so even if someone double-clicks Approve, the second job isn't added to the queue."

**Q: What happens if a job fails?**
> "The worker retries up to 3 times with exponential backoff: 2 seconds, then 4 seconds, then 8 seconds. If all three attempts fail, the job is pushed to the dead-letter queue (`drift_jobs_dlq`). The dashboard shows DLQ jobs in Tab 4 so an operator can investigate. Nothing auto-retries from the DLQ — that's a deliberate safety decision."

**Q: What's the chain of jobs after an approval?**
> "Human approves → replay_test enqueued. Worker completes replay_test → auto-enqueues retrain. Worker completes retrain → auto-creates a second approval for promotion. Human approves promotion → dashboard calls /registry/promote directly. So the full chain is: approval → replay_test → [auto] retrain → [auto] promotion approval → approval → registry update."

### Approval Questions

**Q: How do you prevent someone from promoting a model without approval?**
> "Two enforcement points. First, at the LangGraph layer: the action node calls `interrupt()` for CRITICAL drift, which literally halts the agent graph — nothing proceeds until a human decision is recorded. Second, at the registry layer: `promote_to_production()` in `mlflow_registry.py` requires `approved=True` explicitly in the call. The route `POST /registry/promote` won't call the function unless `approved=True` is in the request body. You can't accidentally promote."

**Q: Where are approvals stored?**
> "Append-only JSONL at `data/approvals.jsonl`. Every approval request and every decision is its own line — records are never updated or deleted, only appended. This gives us a complete audit trail. The store has methods to query pending approvals (requests without matching decisions), history (all requests with their final status merged), and deduplication (won't create two pending approvals for the same investigation + action_type)."

---

## 5. WEAK SPOTS — HONEST ACKNOWLEDGEMENTS

Reviewers respect honesty more than defensiveness. For each weakness, know why it exists and what the tradeoff was.

### 1. Two Failing Tests in CI
> "Two tests are currently failing. One is a floating-point tolerance issue in the model fidelity test — the tolerance of 1e-12 is too strict for joblib round-trip serialization; the model is functionally identical. The other is a test fixture missing a required field in the approval store test. Neither affects runtime behavior, but I should have caught these before the code review. The fix for both is a one-liner."

### 2. Agent State Not Persistent in Local Dev
> "In local development, the LangGraph checkpointer defaults to in-memory, so agent state is lost on restart. In Docker Compose, Postgres is properly configured and state survives restarts. I could improve this by defaulting to a SQLite checkpointer for local dev so persistence works without Docker."

### 3. Clean Clone Needs Manual Setup for Drift Demo
> "On a clean clone, `data/predictions.jsonl` is empty, so the drift detector returns `insufficient_data` until at least 100 predictions have been logged. The demo requires running `simulate_drift.py` first. This should be automated in a startup script or documented in README — which is currently minimal."

### 4. README is Nearly Empty
> "The README only has the project title. There are no setup instructions, no architecture overview, and no demo walkthrough. For a production system, this would be a blocker. I prioritized building the working system but didn't document it properly."

### 5. Some File Paths Are Hardcoded
> "Store paths like `data/predictions.jsonl` and `data/approvals.jsonl` are hardcoded as defaults in the store classes. They work in Docker because the volume mount puts them at the right relative path, but if you run the service from a different working directory it would fail. These should be env-parameterizable."

### 6. `requirements.txt` is Stale
> "The project uses `pyproject.toml` with `uv` as the authoritative dependency source, but there's also an older `requirements.txt` that may have outdated pinned versions. Anyone who uses pip install -r requirements.txt instead of uv sync might get different package versions."

---

## 6. DEMO SCRIPT — STEP BY STEP

Follow this exactly. Each step includes what you should see.

---

### Step 0: Start the Stack
```bash
docker compose up --build
```
Wait until you see all services healthy. Open browser: `http://localhost:8501`

**Expected:** Streamlit dashboard loads with 5 tabs. Tab 1 shows "Demo Control."

---

### Step 1: Check Health
In Tab 1 (Demo Control), click **Health Check** button.

**Expected:** Green status message: model-service and agent-service both "ok"

---

### Step 2: Reset Demo State
Click **Reset Demo State**.

**Expected:** Confirmation message. Redis cleared, drift state reset to LOW, all runtime data archived.

_Say: "This is our clean slate. Every demo reset archives the previous state to a timestamped folder — nothing is deleted."_

---

### Step 3: Send Normal Traffic
Click **Send Drift Batch → Normal Mode**, count 150.

**Expected:** "150 predictions sent (normal distribution)"

Check `GET /drift/status` or click drift status in Tab 2.

**Expected:** severity=LOW, records_available=150

_Say: "The model is serving predictions normally. Each prediction is logged to a rolling window of 100 records."_

---

### Step 4: Send Drifted Traffic
Click **Send Drift Batch → Drift Mode**, count 200.

**Expected:** "200 predictions sent (drifted distribution)"

_Say: "We're now injecting drift. The euribor3m values are shifted to simulate modern interest rate levels — the model was trained when rates were around 4-5%. We're also shifting the contact distribution to nearly all cellular, simulating a behavior change in how the bank reaches customers."_

---

### Step 5: Run Drift Check
Click **Run Drift Check**.

**Expected:**
- Overall severity: CRITICAL
- euribor3m: PSI > 0.25 (CRITICAL)
- contact: chi-square p-value < 0.001 (HIGH)
- event_sent: true

_Say: "PSI fires on euribor3m because the distribution shape has shifted significantly from the training reference. Chi-square fires on contact because the category proportions are statistically impossible given the reference distribution. The severity change from LOW to CRITICAL triggers a webhook to the agent service."_

---

### Step 6: View Agent Investigation
Click Tab 2 (Drift & Agent).

**Expected:** One investigation with status `approval_pending`. Triage result shows top drifted features.

_Say: "The LangGraph agent received the drift event and ran its investigation graph. The triage node identified the top drifted features. The action node saw CRITICAL severity, called `interrupt()` to pause the graph, and requested human approval. The graph is now waiting."_

---

### Step 7: Approve Workflow (First Approval)
Click Tab 3 (Human Approval).

**Expected:** One pending approval: "Run replay test set then open retrain candidate"

Type your name in reviewer field. Click **Approve**.

**Expected:** Approval recorded. Success message.

_Say: "No production change happens without a human decision. The approval service maps this action to a `replay_test` queue job and dispatches it to Redis. The idempotency key prevents this job from ever running twice even if the button is clicked multiple times."_

---

### Step 8: Watch Queue Execute
Click Tab 4 (Queue / DLQ).

**Expected:** See `replay_test` job move from pending → running → completed. Then `retrain` job appears and completes.

Wait ~30-60 seconds for both jobs.

_Say: "The worker is running in a separate container, polling Redis. It executed the replay test first — running the current production model on the held-out test set to get a fresh metrics snapshot. After that completed, it automatically enqueued a retrain job. Retrain has now completed and registered a new candidate model in MLflow Staging."_

---

### Step 9: Approve Promotion (Second Approval)
Return to Tab 3 (Human Approval).

**Expected:** New pending approval: "Promote candidate model" with replay metrics and retrain metrics shown side-by-side.

Review the metrics. Click **Approve**.

**Expected:** Approval recorded. Model version incremented.

_Say: "The worker automatically created this approval after retraining succeeded. We can compare the candidate's metrics to the current production model before deciding. Approving calls /registry/promote with the candidate version. The registry transitions it from Staging to Production in MLflow."_

---

### Step 10: Verify Registry Updated
Click Tab 5 (Model & CI).

**Expected:** Production model version incremented (e.g., v1 → v2). New AUC/recall metrics displayed.

```bash
# Also verify via API
curl http://localhost:8000/registry/production
```

_Say: "The full loop is complete. We detected drift, investigated it with the agent, got human approval, ran replay and retrain, and promoted the new model to production — all tracked and auditable."_

---

### Fallback: If Anything Breaks
- **Drift check returns `insufficient_data`:** Not enough predictions. Run `simulate_drift.py --mode normal --count 150` again.
- **Agent investigation not appearing:** Check agent-service logs: `docker compose logs agent-service`
- **Queue jobs not processing:** Check worker logs: `docker compose logs worker`. If Redis connection issue, restart: `docker compose restart worker`
- **Approval buttons not working:** Check that model-service is healthy: `curl http://localhost:8000/health`
- **Nuclear option:** `POST /demo/reset` and start from Step 2

---

## QUICK FACTS TO KNOW COLD

| Question | Answer |
|----------|--------|
| Dataset | UCI Bank Marketing, 41k rows, 2008-2010 Portugal |
| Target | `y` — did client subscribe to term deposit? Binary 0/1 |
| Model | LogisticRegression, balanced class weights |
| Features | 20 total: 10 numeric, 10 categorical |
| Dropped feature | `duration` (data leakage) |
| Engineered feature | `pdays_was_999` (sentinel value flag) |
| Operating threshold | 0.385 (tuned for recall ≥ 0.75) |
| AUC | 0.801 |
| Recall | 0.748 |
| PSI formula | Σ(curr% - ref%) × ln(curr%/ref%) |
| PSI CRITICAL | > 0.25 |
| Chi-square CRITICAL | p < 0.001 |
| Drift window size | 100 predictions |
| Graph nodes | supervisor, triage, action, comms |
| CRITICAL → | interrupt() → human approval |
| HIGH → | auto-dispatch replay_test |
| Job chain | replay_test → retrain → promote approval |
| Max retries | 3 (backoff: 2s, 4s, 8s) |
| After max retries | Dead-letter queue |
| Approval storage | append-only JSONL (data/approvals.jsonl) |
| Agent state storage | Postgres (docker) / memory (dev/CI) |
| CI tests | 24 total, 22 passing |
