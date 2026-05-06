# Design Decisions

## 1. FastAPI for Model and Agent Services

FastAPI was used because it provides:

- clean HTTP APIs
- Pydantic validation
- automatic OpenAPI documentation
- simple local and Docker deployment

## 2. Pydantic Request Validation

Prediction inputs are validated before inference.

This prevents:

- missing columns
- extra fields
- target leakage
- malformed requests
- unstructured runtime errors

## 3. Logistic Regression Model

Logistic Regression was used because it is:

- interpretable
- fast
- suitable as a baseline classifier
- easy to package in an sklearn pipeline

The operating threshold was chosen separately from the default 0.5 threshold to support recall-focused behavior.

## 4. Dropping `duration`

The `duration` column was dropped because it leaks future information.

In a real prediction setting, call duration is not known before the call finishes.

## 5. PSI and Chi-Square for Drift

Numeric drift is measured with PSI.

Categorical drift is measured with chi-square tests.

This matches the project requirement and provides clear feature-level drift explanations.

## 6. JSONL Local Storage

Local JSONL storage is used for early development because it is simple and inspectable.

Runtime files include:

- predictions
- investigations
- approvals
- drift state

These files are ignored by Git.

## 7. Redis for Slow Tools

Slow tools are not run directly inside the agent request path.

Instead, they are queued through Redis.

This prevents the agent from blocking and supports future replay, retrain, and rollback workflows.

## 8. Idempotency

Queue jobs include idempotency keys.

This prevents duplicate execution when the same logical job is submitted more than once.

## 9. Dead-Letter Queue

Failed jobs move to the DLQ.

This ensures failures are visible and inspectable instead of silently disappearing.

## 10. Human-in-the-Loop Approval

Any Production-changing action requires explicit human approval.

The system must never promote or rollback a model automatically.

## 11. Docker Compose

Docker Compose is used so the full stack can start from a clean clone.

The stack includes:

- model service
- agent service
- Redis
- worker
- dashboard

## 12. CI Tests

CI focuses on regression protection for:

- imports
- request schema
- queue models
- agent API
- approvals
- worker behavior