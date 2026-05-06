# Architecture

## Objective

Drift Triage Co-Pilot is a self-healing MLOps platform for a Bank Marketing binary classifier.

It serves predictions, monitors drift, triggers agent investigations, queues slow remediation tools, and requires human approval before any Production-changing action.

## Main Components

### Model Service

FastAPI service responsible for:

- prediction serving
- request validation
- prediction logging
- drift monitoring
- MLflow registry routes
- approval routes
- queue status routes

### ML Model

The model is a Logistic Regression classifier trained on the UCI Bank Marketing dataset.

The service uses:

- trained sklearn pipeline
- model metadata
- operating threshold
- schema artifacts
- model card

### Drift Monitor

The drift monitor compares recent prediction traffic against reference training data.

It uses:

- PSI for numeric features
- chi-square tests for categorical features

When severity changes, a drift event is created.

### Agent Service

The agent receives drift events through a webhook.

It creates investigations and decides whether to:

- continue monitoring
- enqueue a replay test
- request human approval

### Redis Queue

Redis is used for slow tools such as:

- replay test
- retrain
- rollback

The queue supports:

- idempotency
- worker execution
- dead-letter queue

### Human Approval System

Production-changing actions require explicit human approval.

The system stores approval requests and decisions locally for the dashboard.

### Streamlit Dashboard

The dashboard shows:

- service health
- registry state
- drift status
- queue and DLQ status
- human approval inbox

### Docker Compose

Docker Compose runs:

- model service
- agent service
- Redis
- worker
- dashboard

## End-to-End Flow

1. User sends prediction request.
2. Model service validates input with Pydantic.
3. Model returns prediction and probability.
4. Prediction is logged.
5. Drift monitor compares live traffic to reference data.
6. Drift severity is calculated.
7. If severity changes, a drift event is sent to the agent.
8. Agent creates an investigation.
9. Agent either queues a slow job or requests approval.
10. Dashboard displays system state and pending approvals.

## Safety Rule

No Production model change is made without human approval.