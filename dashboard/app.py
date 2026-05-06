"""
app.py

Streamlit dashboard for the Drift Triage Co-Pilot project.

This dashboard presents:
- system health
- model registry state
- drift status
- queue/DLQ status
- human approval inbox
"""

import streamlit as st

from api_client import (
    get_health,
    get_registry_state,
    get_drift_status,
    get_queue_status,
    get_pending_approvals,
    submit_approval_decision,
)


st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    page_icon="🧭",
    layout="wide",
)


# -------------------------
# Styling
# -------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.25rem;
        font-size: 2.4rem;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
    }

    .card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }

    .status-ok {
        color: #16a34a;
        font-weight: 700;
    }

    .status-bad {
        color: #dc2626;
        font-weight: 700;
    }

    .severity-low {
        color: #16a34a;
        font-weight: 700;
    }

    .severity-medium {
        color: #ca8a04;
        font-weight: 700;
    }

    .severity-high {
        color: #ea580c;
        font-weight: 700;
    }

    .severity-critical {
        color: #dc2626;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Header
# -------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧭 Drift Triage Co-Pilot</h1>
        <p>
            A self-healing MLOps control room for model drift detection,
            agent investigation, async remediation, and human approval.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Helper functions
# -------------------------

def severity_class(severity: str) -> str:
    severity = (severity or "unknown").lower()

    if severity == "critical":
        return "severity-critical"
    if severity == "high":
        return "severity-high"
    if severity == "medium":
        return "severity-medium"
    return "severity-low"


def render_card(title: str, body):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    body()
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Top-level system summary
# -------------------------

health = None
registry_state = None
drift_status = None
queue_status = None
approvals = None

try:
    health = get_health()
except Exception:
    health = None

try:
    registry_state = get_registry_state()
except Exception:
    registry_state = None

try:
    drift_status = get_drift_status()
except Exception:
    drift_status = None

try:
    queue_status = get_queue_status()
except Exception:
    queue_status = None

try:
    approvals = get_pending_approvals()
except Exception:
    approvals = None


report = {}
if drift_status:
    report = drift_status.get("report", drift_status)

severity = report.get("severity", "unknown")
pending_approvals = approvals.get("pending", []) if approvals else []


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Model Service",
    "OK" if health else "DOWN",
)

col2.metric(
    "Production Version",
    registry_state.get("model_version", "none") if registry_state else "unknown",
)

col3.metric(
    "Drift Severity",
    severity,
)

col4.metric(
    "Pending Approvals",
    len(pending_approvals),
)


# -------------------------
# Main dashboard tabs
# -------------------------

tab_overview, tab_drift, tab_queue, tab_approvals, tab_raw = st.tabs(
    [
        "Overview",
        "Drift Monitor",
        "Queue + DLQ",
        "Human Approval",
        "Raw Data",
    ]
)


# -------------------------
# Overview tab
# -------------------------

with tab_overview:
    left, right = st.columns(2)

    with left:
        def health_body():
            if health:
                st.success("Model service is healthy.")
                st.json(health)
            else:
                st.error("Model service is unavailable.")

        render_card("Service Health", health_body)

    with right:
        def registry_body():
            if registry_state:
                st.write("Current model registry state:")
                st.json(registry_state)
            else:
                st.error("Could not load model registry state.")

        render_card("Model Registry", registry_body)

    st.markdown("### Demo Flow")
    st.info(
        """
        1. Generate prediction traffic.
        2. Trigger drift detection.
        3. Review the drift severity.
        4. Confirm agent investigation.
        5. Approve or reject human approval requests.
        6. Inspect queue and DLQ behavior.
        """
    )


# -------------------------
# Drift tab
# -------------------------

with tab_drift:
    def drift_body():
        if not drift_status:
            st.error("Could not load drift status.")
            return

        status = report.get("status", "unknown")
        affected_features = report.get("affected_features", [])

        c1, c2, c3 = st.columns(3)

        c1.metric("Report Status", status)
        c2.markdown(
            f"""
            <div style="font-size: 0.9rem; color: #64748b;">Severity</div>
            <div class="{severity_class(severity)}" style="font-size: 1.8rem;">
                {severity}
            </div>
            """,
            unsafe_allow_html=True,
        )
        c3.metric("Affected Features", len(affected_features))

        if affected_features:
            st.markdown("#### Affected Features")
            st.write(", ".join(affected_features))

        with st.expander("Numeric PSI Results"):
            st.json(report.get("numeric_results", {}))

        with st.expander("Categorical Chi-Square Results"):
            st.json(report.get("categorical_results", {}))

    render_card("Drift Monitor", drift_body)


# -------------------------
# Queue tab
# -------------------------

with tab_queue:
    def queue_body():
        if not queue_status:
            st.error("Could not load queue status.")
            return

        c1, c2 = st.columns(2)

        c1.metric("Queue Depth", queue_status.get("queue_depth", 0))
        c2.metric("DLQ Depth", queue_status.get("dlq_depth", 0))

        st.caption("Queue is used for slow tools like replay, retrain, and rollback.")
        st.json(queue_status)

    render_card("Async Queue + Dead Letter Queue", queue_body)


# -------------------------
# Approvals tab
# -------------------------

with tab_approvals:
    def approvals_body():
        if approvals is None:
            st.error("Could not load approvals.")
            return

        if not pending_approvals:
            st.success("No pending approvals.")
            return

        st.warning(f"{len(pending_approvals)} approval request(s) pending.")

        for approval in pending_approvals:
            approval_id = approval.get("approval_id", "unknown")

            with st.expander(f"Approval Request: {approval_id}", expanded=True):
                st.json(approval)

                approved_by = st.text_input(
                    "Reviewer",
                    value="demo_user",
                    key=f"reviewer_{approval_id}",
                )

                decision_reason = st.text_input(
                    "Decision reason",
                    value="Reviewed during dashboard demo",
                    key=f"reason_{approval_id}",
                )

                c1, c2 = st.columns(2)

                if c1.button("Approve", key=f"approve_{approval_id}"):
                    result = submit_approval_decision(
                        approval_id=approval_id,
                        approved=True,
                        approved_by=approved_by,
                        decision_reason=decision_reason,
                    )
                    st.success("Approval recorded.")
                    st.json(result)

                if c2.button("Reject", key=f"reject_{approval_id}"):
                    result = submit_approval_decision(
                        approval_id=approval_id,
                        approved=False,
                        approved_by=approved_by,
                        decision_reason=decision_reason,
                    )
                    st.warning("Rejection recorded.")
                    st.json(result)

    render_card("Human-in-the-Loop Approval Inbox", approvals_body)


# -------------------------
# Raw tab
# -------------------------

with tab_raw:
    st.markdown("### Raw API Responses")

    raw_col1, raw_col2 = st.columns(2)

    with raw_col1:
        st.markdown("#### Health")
        st.json(health or {})

        st.markdown("#### Registry")
        st.json(registry_state or {})

    with raw_col2:
        st.markdown("#### Drift")
        st.json(drift_status or {})

        st.markdown("#### Queue")
        st.json(queue_status or {})

    st.markdown("#### Approvals")
    st.json(approvals or {})


# -------------------------
# Refresh
# -------------------------

st.divider()

if st.button("Refresh Dashboard"):
    st.rerun()