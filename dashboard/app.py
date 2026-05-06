"""
app.py

Streamlit dashboard for the Drift Triage Co-Pilot project.

It surfaces:
- model service health
- registry state
- drift status
- queue and DLQ status
- human approval inbox
"""

import streamlit as st

from dashboard.api_client import  (
    get_health,
    get_registry_state,
    get_drift_status,
    get_queue_status,
    get_pending_approvals,
    submit_approval_decision,
)


st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    layout="wide"
)

st.title("Drift Triage Co-Pilot Dashboard")

st.caption(
    "Observability and human-in-the-loop control surface for the MLOps drift system."
)


# -------------------------
# Service Health
# -------------------------

st.header("Service Health")

try:
    health = get_health()
    st.success(f"Model service status: {health.get('status')}")
    st.json(health)
except Exception as error:
    st.error(f"Model service unavailable: {error}")


# -------------------------
# Registry State
# -------------------------

st.header("Model Registry State")

try:
    registry_state = get_registry_state()

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", registry_state.get("model_name", "unknown"))
    col2.metric("Version", registry_state.get("model_version", "none"))
    col3.metric("Stage", registry_state.get("stage", "unknown"))

    st.json(registry_state)

except Exception as error:
    st.error(f"Could not load registry state: {error}")


# -------------------------
# Drift Status
# -------------------------

st.header("Drift Status")

try:
    drift_status = get_drift_status()

    report = drift_status.get("report", drift_status)

    severity = report.get("severity", "unknown")
    status = report.get("status", "unknown")
    affected_features = report.get("affected_features", [])

    col1, col2, col3 = st.columns(3)

    col1.metric("Drift Status", status)
    col2.metric("Severity", severity)
    col3.metric("Affected Features", len(affected_features))

    if affected_features:
        st.write("Affected features:")
        st.write(affected_features)

    with st.expander("Full drift report"):
        st.json(drift_status)

except Exception as error:
    st.error(f"Could not load drift status: {error}")


# -------------------------
# Queue Status
# -------------------------

st.header("Queue + DLQ Status")

try:
    queue_status = get_queue_status()

    col1, col2 = st.columns(2)

    col1.metric("Queue Depth", queue_status.get("queue_depth", 0))
    col2.metric("DLQ Depth", queue_status.get("dlq_depth", 0))

    st.json(queue_status)

except Exception as error:
    st.error(f"Could not load queue status: {error}")


# -------------------------
# Human Approval Inbox
# -------------------------

st.header("Human Approval Inbox")

try:
    approvals = get_pending_approvals()
    pending = approvals.get("pending", [])

    if not pending:
        st.info("No pending approvals.")
    else:
        for approval in pending:
            approval_id = approval.get("approval_id")

            with st.expander(f"Approval request: {approval_id}"):
                st.json(approval)

                approved_by = st.text_input(
                    "Approved by",
                    value="demo_user",
                    key=f"approved_by_{approval_id}"
                )

                decision_reason = st.text_input(
                    "Decision reason",
                    value="Reviewed in dashboard",
                    key=f"decision_reason_{approval_id}"
                )

                col1, col2 = st.columns(2)

                if col1.button("Approve", key=f"approve_{approval_id}"):
                    result = submit_approval_decision(
                        approval_id=approval_id,
                        approved=True,
                        approved_by=approved_by,
                        decision_reason=decision_reason,
                    )
                    st.success("Approval recorded.")
                    st.json(result)

                if col2.button("Reject", key=f"reject_{approval_id}"):
                    result = submit_approval_decision(
                        approval_id=approval_id,
                        approved=False,
                        approved_by=approved_by,
                        decision_reason=decision_reason,
                    )
                    st.warning("Rejection recorded.")
                    st.json(result)

except Exception as error:
    st.error(f"Could not load approvals: {error}")


# -------------------------
# Refresh
# -------------------------

st.divider()

if st.button("Refresh Dashboard"):
    st.rerun()