"""
app.py

Streamlit dashboard for AutoHeal.
"""

import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).parent.parent

from api_client import (
    get_health,
    get_registry_state,
    get_drift_status,
    post_drift_check,
    get_queue_status,
    get_pending_approvals,
    get_approval_history,
    submit_approval_decision,
    post_predict,
    post_demo_reset,
    get_investigations,
    get_candidate_model,
    post_promote_model,
)

st.set_page_config(
    page_title="AutoHeal",
    page_icon="🩺",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 1.4rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0 0 .15rem; font-size: 2rem; }
.hero p  { color: #cbd5e1; font-size: .95rem; margin: 0; }
.card {
    background: white;
    padding: 1.1rem 1.3rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(15,23,42,.07);
    border: 1px solid #e2e8f0;
    margin-bottom: .9rem;
}
.card-title { font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: .5rem; }
.kv-row { display: flex; gap: 1rem; margin-bottom: .3rem; font-size: .9rem; }
.kv-label { color: #64748b; min-width: 160px; }
.kv-value { color: #0f172a; font-weight: 600; }
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: .8rem;
    font-weight: 700;
    text-transform: uppercase;
}
.pill-green  { background:#dcfce7; color:#15803d; }
.pill-yellow { background:#fef9c3; color:#854d0e; }
.pill-orange { background:#ffedd5; color:#9a3412; }
.pill-red    { background:#fee2e2; color:#991b1b; }
.pill-blue   { background:#dbeafe; color:#1d4ed8; }
.pill-gray   { background:#f1f5f9; color:#475569; }
.pill-purple { background:#ede9fe; color:#6d28d9; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🏦 AutoHeal</h1>
  <p>MLOps control room · drift detection · agent investigation · human-in-the-loop approval</p>
</div>
""", unsafe_allow_html=True)

if "demo_banner" in st.session_state:
    st.warning(st.session_state["demo_banner"], icon="🔔")

# ── Reference stats ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_reference_stats():
    try:
        df = pd.read_csv(_PROJECT_ROOT / "data" / "reference.csv")
        e = df["euribor3m"]
        c = df["contact"].value_counts(normalize=True)
        return {
            "euribor3m": {
                "mean": round(float(e.mean()), 3),
                "std":  round(float(e.std()), 3),
                "min":  round(float(e.min()), 3),
                "max":  round(float(e.max()), 3),
                "p25":  round(float(e.quantile(.25)), 3),
                "p50":  round(float(e.quantile(.50)), 3),
                "p75":  round(float(e.quantile(.75)), 3),
            },
            "contact": {
                "cellular":  round(float(c.get("cellular", 0)), 3),
                "telephone": round(float(c.get("telephone", 0)), 3),
            },
            "n": len(df),
        }
    except Exception:
        return None

REF = _load_reference_stats()

# ── Helpers ───────────────────────────────────────────────────────────────────

_SEV_PILL = {
    "critical": "pill-red", "high": "pill-orange", "medium": "pill-yellow",
    "low": "pill-green", "local": "pill-blue", "none": "pill-gray", "unknown": "pill-gray",
}

def severity_pill(s):
    s = (s or "unknown").lower()
    return f'<span class="pill {_SEV_PILL.get(s, "pill-gray")}">{s}</span>'

def kv(label, value):
    return (f'<div class="kv-row"><span class="kv-label">{label}</span>'
            f'<span class="kv-value">{value}</span></div>')

def safe(fn):
    try: return fn()
    except: return None

def fmt_pct(v): return f"{v*100:.1f}%"

def fmt4(v) -> str:
    return f"{v:.4f}" if v is not None else "—"

def _fmt_pval(v: float) -> str:
    return "< 0.001" if v < 0.001 else f"{v:.4f}"

def inv_display_name(inv_id: str, severity: str = "") -> str:
    sev = (severity or "").lower()
    label = sev.capitalize() if sev in ("critical", "high", "medium", "low") else "Drift"
    short = inv_id.replace("-", "")[:6].upper() if inv_id else "??????"
    return f"{label} Drift Investigation #{short}"

_SEV_STYLE = {
    "SEVERE": "background-color: #fee2e2; color: #991b1b",
    "HIGH":   "background-color: #fee2e2; color: #991b1b",
    "MEDIUM": "background-color: #fef9c3; color: #854d0e",
    "LOW":    "background-color: #dcfce7; color: #15803d",
}

def _sev_color(val):
    return _SEV_STYLE.get(val, "")

_ACTION_LABELS = {
    "replay_test_set_then_open_retrain_candidate": (
        "Replay test set, evaluate current model behavior, "
        "then open retraining candidate if needed."
    ),
    "monitor_and_replay_test_set": (
        "Monitor model performance and replay the test set to assess drift impact."
    ),
    "promote_candidate_model":   "Promote candidate model to Production.",
    "no_action":                 "No action required — continue monitoring.",
    "rejected_by_human":         "Action rejected by human reviewer.",
}

_APPROVAL_TYPE_LABEL = {
    "replay_test_set_then_open_retrain_candidate": "Workflow Approval — Run Replay / Retrain",
    "promote_candidate_model":                     "Model Promotion Approval",
}

_STATUS_PILL_HTML = {
    "approved": '<span class="pill pill-green">approved</span>',
    "rejected": '<span class="pill pill-red">rejected</span>',
    "pending":  '<span class="pill pill-yellow">pending</span>',
}

# ── Reference distributions ───────────────────────────────────────────────────

_DRIFT_EURIBOR_MEAN = 3.2
_DRIFT_EURIBOR_STD  = 0.25
_DRIFT_CELLULAR_PCT = 0.92
_DROP_COLS = {"pdays_was_999", "y"}

if REF:
    _ref_e = REF["euribor3m"]
    _ref_c = REF["contact"]
else:
    _ref_e = {"mean": 3.616, "std": 1.737, "min": 0.634, "max": 5.045,
              "p25": 1.344, "p50": 4.857, "p75": 4.961}
    _ref_c = {"cellular": 0.636, "telephone": 0.364}

# ── Simulation helpers ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_sim_source() -> pd.DataFrame:
    df = pd.read_csv(_PROJECT_ROOT / "data" / "reference.csv")
    return df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

_SIM_DF = _load_sim_source()


def _sample_row(idx: int) -> dict:
    row = _SIM_DF.iloc[idx % len(_SIM_DF)]
    return {
        col: (int(v) if isinstance(v, np.integer) else
              float(v) if isinstance(v, np.floating) else v)
        for col, v in row.items()
    }


def _make_normal_row(rng: random.Random, idx: int) -> dict:
    return _sample_row(rng.randint(0, len(_SIM_DF) - 1))


def _make_drift_row(rng: random.Random, idx: int) -> dict:
    row = _sample_row(rng.randint(0, len(_SIM_DF) - 1))
    row["euribor3m"] = round(max(0.5, min(5.1, rng.gauss(_DRIFT_EURIBOR_MEAN, _DRIFT_EURIBOR_STD))), 3)
    row["contact"]   = "cellular" if rng.random() < _DRIFT_CELLULAR_PCT else "telephone"
    return row

# ── Fetch state ───────────────────────────────────────────────────────────────

health           = safe(get_health)
registry_state   = safe(get_registry_state)
drift_status     = safe(get_drift_status)
queue_status     = safe(get_queue_status)
approvals        = safe(get_pending_approvals)
approval_history = safe(get_approval_history)
investigations   = safe(get_investigations)
candidate_model  = safe(get_candidate_model)

report  = (drift_status or {}).get("report", drift_status or {})
severity = (
    "low" if report.get("status") == "insufficient_data"
    else report.get("severity", "unknown")
)
pending_approvals       = (approvals or {}).get("pending", [])
history_list            = (approval_history or {}).get("history", [])
inv_list                = (investigations or {}).get("investigations", []) \
                          if isinstance(investigations, dict) else []
completed_jobs          = (queue_status or {}).get("recent_completed_jobs", [])
running_jobs            = (queue_status or {}).get("running_jobs", [])

# ── Top metrics ───────────────────────────────────────────────────────────────

tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
tm1.metric("Model Service",     "OK ✓" if health else "DOWN ✗")
tm2.metric("Version",           (registry_state or {}).get("model_version") or "—")
tm3.metric("Drift Severity",    severity.upper())
tm4.metric("Drifted Features",  len(report.get("affected_features", [])))
tm5.metric("Pending Approvals", len(pending_approvals))
tm6.metric("Investigations",    len(inv_list))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["Demo Control", "Drift & Agent", "Human Approval", "Queue / DLQ", "Model & CI"])
tab_demo, tab_drift_agent, tab_approval, tab_queue, tab_model = tabs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMO CONTROL
# ══════════════════════════════════════════════════════════════════════════════

with tab_demo:

    hs1, hs2 = st.columns(2)

    with hs1:
        st.markdown('<div class="card"><div class="card-title">Service Health</div>', unsafe_allow_html=True)
        if health:
            st.success("Model service reachable.", icon="✅")
            st.markdown(
                kv("Status",  health.get("status", "—")) +
                kv("Service", health.get("service", "—")),
                unsafe_allow_html=True,
            )
        else:
            st.error("Model service unreachable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with hs2:
        st.markdown('<div class="card"><div class="card-title">Production Model</div>', unsafe_allow_html=True)
        if registry_state:
            st.markdown(
                kv("Name",    registry_state.get("model_name", "—")) +
                kv("Version", registry_state.get("model_version") or "—") +
                kv("AUC",     fmt4(registry_state.get("auc"))) +
                kv("Recall",  fmt4(registry_state.get("recall"))),
                unsafe_allow_html=True,
            )
        else:
            st.error("Registry unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Step 1: Reset ─────────────────────────────────────────────────────────
    st.markdown("#### Reset Demo State")
    st.warning(
        "Reset before every demo run — clears predictions, drift state, approvals, and Redis queues.",
        icon="⚠️",
    )
    if st.button("🔄 Reset Demo State", type="primary"):
        try:
            result = post_demo_reset()
            st.success(f"Reset complete. Archive: {result.get('archive_path', '—')}")
            st.cache_data.clear()
            st.session_state.pop("demo_banner", None)
            st.session_state.pop("post_approval_message", None)
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.divider()

    # ── Step 2: Send batch ────────────────────────────────────────────────────
    st.markdown("#### Send Drift Batch")
    st.caption(
        "Send synthetic predictions. "
        "**Drift** mode injects a modern population — euribor3m ≈ 3.2, 92% cellular contact."
    )

    b1, b2, b3 = st.columns([2, 2, 1])
    sim_mode  = b1.radio("Mode", ["Normal (baseline)", "Drift (modern simulation)"], horizontal=True)
    sim_count = b2.slider("Entries", 10, 200, 150, 10)
    b3.markdown("<br>", unsafe_allow_html=True)
    run_sim   = b3.button(f"Send {sim_count}", type="primary", use_container_width=True)

    is_drift = "Drift" in sim_mode

    if run_sim:
        rng = random.Random(99 if is_drift else 42)
        make_row = _make_drift_row if is_drift else _make_normal_row
        progress = st.progress(0, text="Sending…")
        ok, errs = 0, []
        sent_eurib, sent_contact = [], []

        for i in range(sim_count):
            row = make_row(rng, i)
            sent_eurib.append(row["euribor3m"])
            sent_contact.append(row["contact"])
            try:
                post_predict(row)
                ok += 1
            except Exception as exc:
                errs.append(str(exc))
            progress.progress((i + 1) / sim_count, text=f"{i+1}/{sim_count}")

        progress.empty()

        if errs:
            st.warning(f"Sent {ok}/{sim_count}. {len(errs)} failed: {errs[0]}")
        else:
            st.success(f"✓ Sent {ok} {'drift' if is_drift else 'normal'} entries.")

        sc1, sc2 = st.columns(2)
        with sc1:
            arr = np.array(sent_eurib)
            st.dataframe(pd.DataFrame({
                "Stat":       ["Mean", "Std", "Min", "Max"],
                "Reference":  [_ref_e["mean"], _ref_e["std"], _ref_e["min"], _ref_e["max"]],
                "This batch": [round(arr.mean(), 3), round(arr.std(), 3),
                               round(arr.min(), 3),  round(arr.max(), 3)],
            }), use_container_width=True, hide_index=True)

        with sc2:
            cnt   = Counter(sent_contact)
            total = len(sent_contact)
            st.dataframe(pd.DataFrame({
                "Channel":    ["cellular", "telephone"],
                "Reference":  [fmt_pct(_ref_c["cellular"]), fmt_pct(_ref_c["telephone"])],
                "This batch": [fmt_pct(cnt.get("cellular", 0) / total),
                               fmt_pct(cnt.get("telephone", 0) / total)],
                "Count":      [cnt.get("cellular", 0), cnt.get("telephone", 0)],
            }), use_container_width=True, hide_index=True)

    with st.expander("Feature distribution reference"):
        fe1, fe2 = st.columns(2)
        with fe1:
            st.markdown("**`euribor3m` — Euribor 3-Month Rate**")
            st.caption("Training data is bimodal (2008 crisis). Modern ECB rates fall in the gap → high PSI.")
            st.dataframe(pd.DataFrame({
                "Stat":      ["Mean", "Std", "Min", "Max", "p25", "p50", "p75"],
                "Reference": [_ref_e["mean"], _ref_e["std"], _ref_e["min"], _ref_e["max"],
                              _ref_e["p25"],  _ref_e["p50"], _ref_e["p75"]],
                "Drift Sim": [f"~{_DRIFT_EURIBOR_MEAN}", f"~{_DRIFT_EURIBOR_STD}",
                              "≈ 2.3", "≈ 4.1", "≈ 2.9", f"~{_DRIFT_EURIBOR_MEAN}", "≈ 3.5"],
            }), use_container_width=True, hide_index=True)
        with fe2:
            st.markdown("**`contact` — Communication Channel**")
            st.caption("2008–10: 36% landline. Modern mobile penetration: >95% cellular.")
            st.dataframe(pd.DataFrame({
                "Channel":          ["cellular", "telephone"],
                "Reference":        [fmt_pct(_ref_c["cellular"]), fmt_pct(_ref_c["telephone"])],
                "Drift Sim":        [fmt_pct(_DRIFT_CELLULAR_PCT), fmt_pct(1 - _DRIFT_CELLULAR_PCT)],
                "Subscription rate": ["14.7%", "5.4%"],
            }), use_container_width=True, hide_index=True)

    st.divider()

    # ── Step 3: Run drift check ───────────────────────────────────────────────
    st.markdown("#### Run Drift Check")
    st.caption("Runs the detector against the last 100 predictions. Notifies the agent if severity changed.")

    dc1, dc2 = st.columns([1, 4])
    check_btn = dc1.button("Run Drift Check", type="secondary", use_container_width=True)

    if check_btn:
        st.info("Drift check started. The agent may take a few seconds to investigate.")
        try:
            with st.status("Running drift check…", expanded=True) as _status:
                _status.write("Loading prediction window")
                time.sleep(0.15)
                _status.write("Computing PSI and chi-square drift")
                time.sleep(0.2)

                res    = post_drift_check()
                rep    = res.get("report", res)
                sev    = rep.get("severity", "unknown")
                stat   = rep.get("status", "—")
                aff    = rep.get("affected_features", [])
                sent   = res.get("event_sent", False)
                n_av   = rep.get("records_available")
                n_req  = rep.get("records_required")
                wh     = res.get("webhook_response") or {}
                inv_id = wh.get("investigation_id")
                req_hil = wh.get("requires_human_approval", False)

                if sent:
                    _status.write("Severity changed: notifying agent")
                    time.sleep(0.15)
                    _status.write("Agent investigation running")
                    time.sleep(0.25)
                    if req_hil:
                        _status.write("HIL approval request being created")
                        time.sleep(0.15)

                if stat == "insufficient_data":
                    _status.update(label="Insufficient data — send more predictions first", state="error")
                elif wh.get("error"):
                    _status.update(label="Agent webhook failed", state="error")
                else:
                    _status.update(label="Drift check complete ✓", state="complete")

            if stat == "insufficient_data":
                st.warning(f"Need **{n_req}** predictions, have **{n_av}**. Send more predictions first.")
            else:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Severity",         sev.upper())
                r2.metric("Drifted Features", len(aff))
                r3.metric("Agent notified",   "Yes ✓" if sent else "No")
                r4.metric("Approval needed",  "Yes ✓" if req_hil else "No")

                if aff:
                    st.markdown("**Drifted:** " + "  ".join(f"`{f}`" for f in aff))
                else:
                    st.success("No drift detected.")

                if req_hil:
                    st.warning(
                        "Human approval required. Go to the **Human Approval** tab to dispatch "
                        "the recommended action.",
                        icon="⚠️",
                    )
                    st.session_state["demo_banner"] = (
                        "Critical drift detected. Agent investigation opened. "
                        "Human approval required."
                    )
                    st.rerun()

                elif not sent:
                    if wh.get("error"):
                        st.error(
                            f"**Agent webhook failed** — {wh['error']}  \n"
                            f"{wh.get('hint', '')}  \n\n"
                            "Check that the agent service is running on port 8001."
                        )
                    else:
                        st.info(
                            "Agent was **not** notified — severity hasn't changed since last check. "
                            "Click **Reset Demo State** above, then run again."
                        )
                else:
                    st.rerun()

        except Exception as e:
            st.error(f"Drift check failed: {e}")

    # ── Single prediction ─────────────────────────────────────────────────────
    with st.expander("Submit a single custom prediction (advanced)"):
        with st.form("single_entry"):
            st.markdown("**Numeric**")
            s1, s2, s3, s4, s5 = st.columns(5)
            s_age      = s1.number_input("age",          0, 120,  42)
            s_campaign = s2.number_input("campaign",     0,  50,   1)
            s_pdays    = s3.number_input("pdays",        0, 999, 999)
            s_prev     = s4.number_input("previous",     0,  20,   0)
            s_eurib    = s5.number_input("euribor3m", value=3.2, format="%.3f")

            s6, s7, s8, s9 = st.columns(4)
            s_nr  = s6.number_input("nr.employed",    value=5191.0, format="%.1f")
            s_emp = s7.number_input("emp.var.rate",   value=1.1,    format="%.1f")
            s_cpi = s8.number_input("cons.price.idx", value=93.994, format="%.3f")
            s_cci = s9.number_input("cons.conf.idx",  value=-36.4,  format="%.1f")

            st.markdown("**Categorical**")
            cat1, cat2, cat3, cat4, cat5 = st.columns(5)
            s_job  = cat1.selectbox("job",        ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                                    "management", "retired", "self-employed", "services",
                                                    "student", "technician", "unemployed", "unknown"])
            s_mar  = cat2.selectbox("marital",    ["divorced", "married", "single", "unknown"])
            s_edu  = cat3.selectbox("education",  ["basic.4y", "basic.6y", "basic.9y", "high.school",
                                                    "illiterate", "professional.course",
                                                    "university.degree", "unknown"])
            s_def  = cat4.selectbox("default",    ["no", "yes", "unknown"])
            s_hous = cat5.selectbox("housing",    ["no", "yes", "unknown"])

            cat6, cat7, cat8, cat9, cat10 = st.columns(5)
            s_loan = cat6.selectbox("loan",        ["no", "yes", "unknown"])
            s_cont = cat7.selectbox("contact",     ["cellular", "telephone"])
            s_mon  = cat8.selectbox("month",       ["jan", "feb", "mar", "apr", "may", "jun",
                                                     "jul", "aug", "sep", "oct", "nov", "dec"])
            s_dow  = cat9.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"])
            s_pout = cat10.selectbox("poutcome",   ["failure", "nonexistent", "success"])

            submit_single = st.form_submit_button("Submit", type="primary")

        if submit_single:
            payload = {
                "age": s_age, "campaign": s_campaign, "pdays": s_pdays, "previous": s_prev,
                "euribor3m": s_eurib, "nr.employed": s_nr, "emp.var.rate": s_emp,
                "cons.price.idx": s_cpi, "cons.conf.idx": s_cci,
                "job": s_job, "marital": s_mar, "education": s_edu, "default": s_def,
                "housing": s_hous, "loan": s_loan, "contact": s_cont, "month": s_mon,
                "day_of_week": s_dow, "poutcome": s_pout,
            }
            try:
                r = post_predict(payload)
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction",  r.get("prediction", "—").upper())
                c2.metric("Probability", f'{r.get("probability_yes", 0):.4f}')
                c3.metric("Threshold",   r.get("threshold_used", "—"))
            except Exception as e:
                st.error(f"Failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIFT & AGENT
# ══════════════════════════════════════════════════════════════════════════════

with tab_drift_agent:

    # ── Drift status ──────────────────────────────────────────────────────────
    st.markdown("### Drift Status")

    if not drift_status:
        st.error("Could not load drift status.")
    else:
        sval    = report.get("status", "unknown")
        aff     = report.get("affected_features", [])
        num_res = report.get("numeric_results", {})
        cat_res = report.get("categorical_results", {})
        n_avail = report.get("records_available")
        n_req_r = report.get("records_required")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Severity",         severity.upper())
        d2.metric("Drifted Features", len(aff))
        d3.metric("Status",           sval)
        if n_avail is not None:
            d4.metric("Window",       f"{n_avail} / {n_req_r}")

        if sval == "insufficient_data":
            st.warning(
                f"Need **{n_req_r}** predictions, have **{n_avail}**. "
                "Use the Demo Control tab to send predictions."
            )
        else:
            if aff:
                st.markdown("**Drifted features:** " + "  ".join(f"`{f}`" for f in aff))
            else:
                st.success("No drift detected.")

            st.caption(
                "ℹ️ Small demo batches may cause incidental movement in other features. "
                "The intentional demo drift focuses on **euribor3m** and **contact**."
            )

            if num_res:
                st.markdown("#### Numeric Features — PSI (Population Stability Index)")
                st.caption("PSI < 0.1: no drift · 0.1–0.25: moderate · > 0.25: significant")
                rows = [
                    {"Feature": f, "PSI": round(d.get("psi", 0), 4),
                     "Severity": d.get("severity", "—").upper(),
                     "Drifted": "✅" if d.get("is_drifted") else "❌"}
                    for f, d in num_res.items()
                ]
                df_n = pd.DataFrame(rows).sort_values("PSI", ascending=False)
                st.dataframe(df_n.style.map(_sev_color, subset=["Severity"]),
                             use_container_width=True, hide_index=True)

            if cat_res:
                st.markdown("#### Categorical Features — Chi-Square Test")
                st.caption("p-value < 0.05: distribution significantly different from reference")
                rows = [
                    {"Feature": f, "Chi²": round(d.get("chi_square", 0), 2),
                     "p-value": _fmt_pval(d.get("p_value", 1.0)),
                     "Severity": d.get("severity", "—").upper(),
                     "Drifted": "✅" if d.get("is_drifted") else "❌"}
                    for f, d in cat_res.items()
                ]
                df_c = pd.DataFrame(rows).sort_values("Chi²", ascending=False)
                st.dataframe(df_c.style.map(_sev_color, subset=["Severity"]),
                             use_container_width=True, hide_index=True)

        with st.expander("Developer details — raw drift JSON"):
            st.json(drift_status or {})

    st.divider()

    # ── Agent investigations ───────────────────────────────────────────────────
    st.markdown("### Agent Investigations")

    if investigations is None:
        st.warning(
            "Agent service not reachable (port 8001). "
            "Start the agent container to see investigations."
        )
    elif not inv_list:
        st.info("No investigations yet. Send drift predictions and run a drift check in Demo Control.")
    else:
        latest_inv    = inv_list[0]
        latest_status = latest_inv.get("status", "")

        # Only show "agent is paused" if approval has NOT been decided yet
        latest_inv_id = latest_inv.get("investigation_id", "")
        latest_replay_approval = next(
            (h for h in history_list
             if h.get("investigation_id") == latest_inv_id
             and h.get("action_type") == "replay_test_set_then_open_retrain_candidate"),
            None,
        )
        latest_approval_status = (latest_replay_approval or {}).get("status")

        if latest_status == "approval_pending" and not latest_approval_status:
            st.warning(
                "The agent is paused — waiting for human approval. "
                "Go to the **Human Approval** tab to review and dispatch.",
                icon="⚠️",
            )
        elif latest_approval_status == "approved":
            st.info(
                "Workflow approved. Replay test job has been dispatched. "
                "Check **Queue / DLQ** for status."
            )
        elif latest_approval_status == "rejected":
            st.info("Workflow approval was rejected — no production action taken.")
        elif latest_status == "resolved":
            st.success("Latest investigation resolved.")
        elif latest_status:
            st.info(f"Agent is investigating — status: **{latest_status}**")

        if "seen_inv_ids" not in st.session_state:
            st.session_state.seen_inv_ids = set()
        for inv in inv_list:
            iid = inv.get("investigation_id", "")
            if iid and iid not in st.session_state.seen_inv_ids:
                st.toast(f"Investigation opened: {inv_display_name(iid, inv.get('severity',''))}", icon="🔍")
                st.session_state.seen_inv_ids.add(iid)

        for inv in inv_list:
            inv_id        = inv.get("investigation_id", "—")
            sev           = inv.get("severity", "unknown")
            status_i      = inv.get("status", "—")
            created       = inv.get("created_at_utc", "—")
            updated       = inv.get("updated_at_utc", "—")
            action        = inv.get("recommended_action") or {}
            action_raw    = action.get("action_type", "—")
            action_reason = action.get("reason", "")
            needs_approval = inv.get("requires_human_approval", False)
            triage_done   = inv.get("triage_result") is not None
            action_done   = bool(action)
            action_label  = _ACTION_LABELS.get(action_raw, action_raw)

            # Cross-reference approval history and completed jobs for this investigation
            replay_approval = next(
                (h for h in history_list
                 if h.get("investigation_id") == inv_id
                 and h.get("action_type") == "replay_test_set_then_open_retrain_candidate"),
                None,
            )
            replay_decision = (replay_approval or {}).get("status")

            completed_job = next(
                (j for j in completed_jobs
                 if j.get("investigation_id") == inv_id
                 and j.get("job_type") == "replay_test"),
                None,
            )
            running_job = next(
                (j for j in running_jobs if j.get("investigation_id") == inv_id),
                None,
            )
            completed_retrain_job = next(
                (j for j in completed_jobs
                 if j.get("investigation_id") == inv_id
                 and j.get("job_type") == "retrain"),
                None,
            )
            retrain_running = any(
                j.get("investigation_id") == inv_id and j.get("job_type") == "retrain"
                for j in running_jobs
            )
            promote_approval = next(
                (h for h in history_list
                 if h.get("investigation_id") == inv_id
                 and h.get("action_type") == "promote_candidate_model"),
                None,
            )
            promote_decision = (promote_approval or {}).get("status")

            comms_raw = inv.get("comms_summary")
            if comms_raw and comms_raw != "None":
                summary = comms_raw
            elif replay_decision == "approved" and completed_job:
                summary = "Replay test completed successfully. See Queue / DLQ for metrics."
            elif replay_decision == "approved":
                summary = "Workflow approved. Replay test job in progress."
            elif replay_decision == "rejected":
                summary = "Workflow approval rejected. No production action taken."
            elif status_i == "approval_pending":
                summary = "Critical drift investigation completed. Human approval is required."
            elif status_i == "resolved":
                summary = "Investigation resolved. No further action needed."
            else:
                summary = "Agent investigation in progress."

            is_pending = (status_i == "approval_pending" or needs_approval) and not replay_decision

            display_name = inv_display_name(inv_id, sev)
            with st.expander(
                f"{display_name}  ·  {status_i}",
                expanded=is_pending,
            ):
                if is_pending:
                    st.warning(
                        "The agent is paused. No retrain or production action will run "
                        "until a human approves.",
                        icon="⚠️",
                    )

                # Timeline with cross-referenced state
                st.markdown("**Investigation Timeline**")
                timeline = [
                    ("✅", "Drift alert received"),
                    ("✅", "Investigation opened"),
                    ("✅" if triage_done else "⏳", "Agent triaged drift severity"),
                    ("✅" if action_done else "⏳",  "Recommended action selected"),
                ]

                if replay_decision == "approved":
                    timeline.append(("✅", "Human approval received — workflow dispatched"))
                    if completed_job:
                        timeline.append(("✅", "Replay test completed — model evaluated"))
                        if retrain_running:
                            timeline.append(("⏳", "Candidate model training in progress…"))
                        elif completed_retrain_job:
                            mv = (completed_retrain_job.get("result_summary") or {}).get("model_version")
                            mv_label = f" — v{mv}" if mv else ""
                            timeline.append(("✅", f"Candidate model trained{mv_label}"))
                            if promote_decision == "approved":
                                timeline.append(("✅", "Candidate promoted to Production"))
                            elif promote_decision == "rejected":
                                timeline.append(("❌", "Promotion rejected — production unchanged"))
                            else:
                                timeline.append(("⏳", "Awaiting promotion approval — go to Human Approval tab"))
                        else:
                            timeline.append(("⏳", "Retrain job queued — waiting for worker"))
                    elif running_job:
                        started = running_job.get("started_at_utc", "")
                        timeline.append(("⏳", f" Replay test running (started {started})"))
                    else:
                        timeline.append(("⏳", "Replay test queued — waiting for worker"))
                elif replay_decision == "rejected":
                    timeline.append(("❌", "Action rejected — no queue job dispatched"))
                elif status_i == "resolved":
                    timeline.append(("✅", "Investigation resolved"))
                elif is_pending:
                    timeline.append(("⏳", "Awaiting human approval — go to Human Approval tab"))
                    timeline.append(("⏳", "Queue dispatch pending approval"))

                for icon, label in timeline:
                    st.markdown(f"{icon}&nbsp;&nbsp;{label}")

                st.divider()

                st.markdown(
                    kv("Severity", severity_pill(sev)) +
                    kv("Status",   status_i) +
                    (kv("Opened",  created) if created and created != "—" else "") +
                    (kv("Updated", updated) if updated and updated != "—" else "") +
                    kv("Action",   action_label) +
                    kv("Summary",  summary),
                    unsafe_allow_html=True,
                )
                if action_reason:
                    st.caption(f"Reason: {action_reason}")

                with st.expander("Developer details"):
                    st.write("**Investigation ID:**", inv_id)
                    if replay_approval:
                        st.write("**Approval ID:**", replay_approval.get("approval_id", "—"))
                    if completed_job:
                        st.json(completed_job)

        with st.expander("Developer details — raw investigations JSON"):
            st.json(investigations or {})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HUMAN APPROVAL
# ══════════════════════════════════════════════════════════════════════════════

with tab_approval:
    st.markdown("### Human Approval Inbox")

    # Show post-decision navigation hint
    if "post_approval_message" in st.session_state:
        msg = st.session_state.pop("post_approval_message")
        if msg["type"] == "success":
            st.success(msg["text"])
        else:
            st.info(msg["text"])
        if msg.get("hint"):
            st.info(f"→ {msg['hint']}", icon="👉")

    # ── Pending approvals ──────────────────────────────────────────────────────
    if approvals is None:
        st.error("Could not load approvals. Check that the model service is reachable.")
    elif not pending_approvals:
        orphaned = [inv for inv in inv_list if inv.get("status") == "approval_pending"]
        if orphaned:
            st.warning(
                "Investigation is waiting for approval but no approval request was found. "
                "Check approval creation wiring.",
                icon="⚠️",
            )
            for inv in orphaned:
                st.caption(
                    f"Orphaned: {inv_display_name(inv.get('investigation_id',''), inv.get('severity',''))} "
                    f"· {inv.get('severity','—').upper()}"
                )
        else:
            st.success("No pending approvals.")
            st.caption(
                "CRITICAL drift pauses here for human approval before any production-touching action."
            )
    else:
        # Separate by approval type for clarity
        workflow_approvals   = [a for a in pending_approvals
                                if a.get("action_type") != "promote_candidate_model"]
        promotion_approvals  = [a for a in pending_approvals
                                if a.get("action_type") == "promote_candidate_model"]

        # ── Approval type 1: Workflow ──────────────────────────────────────────
        if workflow_approvals:
            st.warning(
                f"{len(workflow_approvals)} workflow approval request(s) pending — "
    
                ,icon="⚠️",
            )
            for appr in workflow_approvals:
                aid      = appr.get("approval_id", "unknown")
                atype    = appr.get("action_type", "—")
                inv_id_a = appr.get("investigation_id", "—")
                reason   = appr.get("reason", "—")
                created  = appr.get("created_at_utc")

                matched_inv  = next((i for i in inv_list if i.get("investigation_id") == inv_id_a), {})
                severity_a   = matched_inv.get("severity", "")
                action_label = _ACTION_LABELS.get(atype, atype)
                type_label   = _APPROVAL_TYPE_LABEL.get(atype, atype)
                display_name_a = inv_display_name(inv_id_a, severity_a)

                with st.expander(
                    f"**{type_label}**  ·  {display_name_a}",
                    expanded=True,
                ):
                    st.markdown(
                        (kv("Severity", severity_pill(severity_a)) if severity_a else "") +
                        kv("Action",           action_label) +
                        kv("Why required",     reason) +
                        (kv("Created",         created) if created and created != "—" else ""),
                        unsafe_allow_html=True,
                    )
                    rev = st.text_input("Reviewer",        "demo_user",             key=f"rev_{aid}")
                    rsn = st.text_input("Decision reason", "Reviewed in dashboard", key=f"rsn_{aid}")
                    ab, rb = st.columns(2)
                    if ab.button(
                        "✅ Approve and dispatch replay/retrain workflow",
                        key=f"app_{aid}",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            submit_approval_decision(aid, True, rev, rsn)
                            st.session_state.pop("demo_banner", None)
                            st.session_state["post_approval_message"] = {
                                "type":  "success",
                                "text":  "✅ Workflow approved. Replay test job dispatched to Redis queue.",
                                "hint":  "Check the **Queue / DLQ** tab to monitor job progress.",
                            }
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Approval submission failed: {exc}")
                    if rb.button(
                        "❌ Reject",
                        key=f"rej_{aid}",
                        use_container_width=True,
                    ):
                        try:
                            submit_approval_decision(aid, False, rev, rsn)
                            st.session_state.pop("demo_banner", None)
                            st.session_state["post_approval_message"] = {
                                "type": "info",
                                "text": "Workflow rejected. No queue job dispatched.",
                                "hint": "",
                            }
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Rejection submission failed: {exc}")

                    with st.expander("Developer details"):
                        st.write("**Approval ID:**", aid)
                        st.write("**Investigation ID:**", inv_id_a)

        # ── Approval type 2: Model Promotion ──────────────────────────────────
        if promotion_approvals:
            st.warning(
                f"{len(promotion_approvals)} model promotion approval(s) pending — "
                ,
                icon="⚠️",
            )
            cand_version = (candidate_model or {}).get("candidate_version")
            cand_msg     = (candidate_model or {}).get("message", "")
            prod_version = (registry_state or {}).get("model_version", "—")

            for appr in promotion_approvals:
                aid_p    = appr.get("approval_id", "unknown")
                inv_id_p = appr.get("investigation_id", "—")
                reason_p = appr.get("reason", "—")
                created_p = appr.get("created_at_utc")
                payload_p = appr.get("payload") or {}
                replay_m  = payload_p.get("replay_metrics") or {}

                display_name_p = inv_display_name(inv_id_p, "")

                with st.expander(
                    f"**Model Promotion Approval**  ·  {display_name_p}",
                    expanded=True,
                ):
                    st.markdown(
                        kv("Why required", reason_p) +
                        kv("Current production", f"v{prod_version}") +
                        kv("Candidate version",
                           f"v{cand_version}" if cand_version else
                           '<span style="color:#94a3b8">No candidate registered yet</span>') +
                        (kv("Created", created_p) if created_p and created_p != "—" else ""),
                        unsafe_allow_html=True,
                    )

                    if replay_m:
                        st.markdown("**Current Model — Replay Test Results (before retraining)**")
                        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                        rc1.metric("AUC",       fmt4(replay_m.get("auc")))
                        rc2.metric("F1",        fmt4(replay_m.get("f1")))
                        rc3.metric("Precision", fmt4(replay_m.get("precision")))
                        rc4.metric("Recall",    fmt4(replay_m.get("recall")))
                        rc5.metric("Threshold", fmt4(replay_m.get("threshold")))

                    retrain_m = payload_p.get("retrain_metrics") or {}
                    if retrain_m:
                        st.markdown("**Candidate Model — Metrics After Retraining**")
                        rr1, rr2, rr3, rr4, rr5 = st.columns(5)
                        rr1.metric("AUC",       fmt4(retrain_m.get("auc")))
                        rr2.metric("F1",        fmt4(retrain_m.get("f1")))
                        rr3.metric("Precision", fmt4(retrain_m.get("precision")))
                        rr4.metric("Recall",    fmt4(retrain_m.get("recall")))
                        rr5.metric("Threshold", fmt4(retrain_m.get("threshold")))

                    if not cand_version:
                        st.info(
                            cand_msg or "No candidate model registered yet — production model unchanged.",
                            icon="ℹ️",
                        )

                    rev_p = st.text_input("Reviewer",        "demo_user",             key=f"rev_p_{aid_p}")
                    rsn_p = st.text_input("Decision reason", "Reviewed in dashboard", key=f"rsn_p_{aid_p}")
                    ap, rp = st.columns(2)

                    approve_label = (
                        f"✅ Approve — promote v{cand_version} to Production"
                        if cand_version else
                        "✅ Approve (no candidate — acknowledge only)"
                    )
                    if ap.button(
                        approve_label,
                        key=f"app_p_{aid_p}",
                        type="primary",
                        use_container_width=True,
                        disabled=False,
                    ):
                        try:
                            submit_approval_decision(aid_p, True, rev_p, rsn_p)
                            promo_result = None
                            if cand_version:
                                try:
                                    promo_result = post_promote_model(cand_version)
                                except Exception as pe:
                                    promo_result = {"status": "error", "error": str(pe)}
                            st.session_state["post_approval_message"] = {
                                "type": "success",
                                "text": (
                                    f"✅ Promotion approved. "
                                    + (f"v{cand_version} promoted to Production."
                                       if promo_result and promo_result.get("status") == "approved"
                                       else "No candidate model to promote — production unchanged.")
                                ),
                                "hint": "Check the **Model & CI** tab for the updated registry state.",
                            }
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Approval submission failed: {exc}")

                    if rp.button(
                        "❌ Reject — keep current production model",
                        key=f"rej_p_{aid_p}",
                        use_container_width=True,
                    ):
                        try:
                            submit_approval_decision(aid_p, False, rev_p, rsn_p)
                            st.session_state["post_approval_message"] = {
                                "type": "info",
                                "text": "Promotion rejected. Current production model unchanged.",
                                "hint": "",
                            }
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Rejection submission failed: {exc}")

                    with st.expander("Developer details"):
                        st.write("**Approval ID:**", aid_p)
                        st.write("**Investigation ID:**", inv_id_p)
                        if candidate_model:
                            st.json(candidate_model)

    st.divider()

    # ── Decision history ───────────────────────────────────────────────────────
    st.markdown("### Decision History")

    if not history_list:
        st.info("No approval history yet.")
    else:
        for entry in history_list:
            eid      = entry.get("approval_id", "—")
            status_h = entry.get("status", "pending")
            atype_h  = entry.get("action_type", "—")
            inv_id_h = entry.get("investigation_id", "—")
            created_h = entry.get("created_at_utc", "—")
            decided_h = entry.get("decided_at_utc", "—")
            by_h      = entry.get("approved_by", "—")
            dreason_h = entry.get("decision_reason", "—")
            alabel_h  = _APPROVAL_TYPE_LABEL.get(atype_h, _ACTION_LABELS.get(atype_h, atype_h))
            spill     = _STATUS_PILL_HTML.get(
                status_h,
                f'<span class="pill pill-gray">{status_h}</span>',
            )
            dname_h   = inv_display_name(inv_id_h, "")

            with st.expander(
                f"{alabel_h[:55]}  ·  {dname_h}  ·  {status_h.upper()}",
                expanded=False,
            ):
                st.markdown(
                    kv("Status",   spill) +
                    kv("Action",   alabel_h) +
                    (kv("Created", created_h) if created_h and created_h != "—" else "") +
                    (kv("Decided", decided_h) if decided_h and decided_h != "—" else "") +
                    (kv("By",      by_h)      if by_h and by_h != "—" else "") +
                    (kv("Reason",  dreason_h) if dreason_h and dreason_h != "—" else ""),
                    unsafe_allow_html=True,
                )
                with st.expander("Developer details"):
                    st.write("**Approval ID:**", eid)
                    st.write("**Investigation ID:**", inv_id_h)

    with st.expander("Developer details — raw approvals JSON"):
        st.json({"pending": pending_approvals, "history": history_list})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUEUE / DLQ
# ══════════════════════════════════════════════════════════════════════════════

with tab_queue:
    st.markdown("### Queue Status")

    if not queue_status:
        st.error("Could not load queue status.")
    else:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Pending",           queue_status.get("queue_depth",     0))
        q2.metric("Running",           queue_status.get("running_depth",   0))
        q3.metric("Completed",         queue_status.get("completed_depth", 0))
        q4.metric("Dead-letter (DLQ)", queue_status.get("dlq_depth",       0))
        st.caption(
            "Values read live from Redis. "
            "The worker dispatches `replay_test` jobs after CRITICAL drift human approval."
        )

        # ── Running jobs ───────────────────────────────────────────────────────
        st.markdown("#### Running Jobs")
        if running_jobs:
            for rj in running_jobs:
                job_type_r = rj.get("job_type", "—")
                _type_msgs = {
                    "replay_test":       "Evaluating",
                    "retrain":           "Training candidate model — this may take a minute.",
                    "retrain_candidate": "Training candidate model — this may take a minute.",
                    "rollback":          "Model rollback is running.",
                }
                type_msg = _type_msgs.get(job_type_r, f"Job running: {job_type_r}")
                st.info(
                    f"**{type_msg}**  \n"
                    f"Investigation: {inv_display_name(rj.get('investigation_id') or '', '')}  \n"
                    f"Started: {rj.get('started_at_utc', '—')}",
                    icon="⏳",
                )
        else:
            st.caption("No jobs currently running.")

        # ── Completed jobs ─────────────────────────────────────────────────────
        st.markdown("#### Completed Jobs")
        if completed_jobs:
            rows = []
            for job in completed_jobs:
                rs      = job.get("result_summary") or {}
                metrics = rs.get("metrics") or {}
                rows.append({
                    "Investigation": inv_display_name(
                        job.get("investigation_id") or "",
                        "",
                    ),
                    "Type":      job.get("job_type", "—"),
                    "Completed": job.get("completed_at_utc", "—"),
                    "AUC":       fmt4(metrics.get("auc")),
                    "F1":        fmt4(metrics.get("f1")),
                    "Precision": fmt4(metrics.get("precision")),
                    "Recall":    fmt4(metrics.get("recall")),
                    "Threshold": fmt4(metrics.get("threshold")),
                    "Report":    (rs.get("report_path") or "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No completed jobs yet. Jobs appear here after the worker processes them.")

        # ── Dead-letter queue ──────────────────────────────────────────────────
        st.markdown("#### Dead-Letter Queue")
        dlq_depth = queue_status.get("dlq_depth", 0)
        if dlq_depth > 0:
            st.error(f"{dlq_depth} job(s) in the dead-letter queue.")
            recent_dlq = queue_status.get("recent_dlq_jobs", [])
            if recent_dlq:
                dlq_rows = []
                for item in recent_dlq:
                    dlq_rows.append({
                        "Investigation": inv_display_name(item.get("investigation_id") or "", ""),
                        "Type":       item.get("job_type", "—"),
                        "Failed at":  item.get("failed_at_utc", "—"),
                        "Error":      str(item.get("error") or item.get("raw") or "—")[:120],
                    })
                st.dataframe(pd.DataFrame(dlq_rows), use_container_width=True, hide_index=True)
        else:
            st.success("Dead-letter queue is empty.")

    with st.expander("Developer details — raw queue JSON"):
        st.json(queue_status or {})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL & CI
# ══════════════════════════════════════════════════════════════════════════════

with tab_model:
    st.markdown("### Production Model")

    if not registry_state:
        st.error("Registry unavailable.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC",       fmt4(registry_state.get("auc")))
        m2.metric("Recall",    fmt4(registry_state.get("recall")))
        m3.metric("Threshold", registry_state.get("threshold", "—"))
        m4.metric("Version",   registry_state.get("model_version") or "—")

        st.markdown('<div class="card"><div class="card-title">Registry Details</div>', unsafe_allow_html=True)
        st.markdown(
            kv("Model",     registry_state.get("model_name", "—")) +
            kv("Version",   registry_state.get("model_version") or "—") +
            kv("Stage",     severity_pill(registry_state.get("stage", "—"))) +
            kv("Threshold", registry_state.get("threshold", "—")) +
            kv("AUC",       fmt4(registry_state.get("auc"))) +
            kv("Recall",    fmt4(registry_state.get("recall"))),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Candidate model ────────────────────────────────────────────────────────
    st.markdown("### Candidate Model (Staging)")
    if candidate_model:
        cand_v = candidate_model.get("candidate_version")
        if cand_v:
            st.success(f"Candidate model registered: **v{cand_v}** (Staging)")
            st.markdown(
                kv("Version",  cand_v) +
                kv("Run ID",   candidate_model.get("run_id", "—")) +
                kv("Stage",    candidate_model.get("stage", "—")),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                candidate_model.get("message", "No candidate model registered yet."),
                icon="ℹ️",
            )
    else:
        st.info("Registry unavailable — cannot check for candidate models.")

    with st.expander("Developer details — raw registry JSON"):
        st.json({"production": registry_state or {}, "candidate": candidate_model or {}})

    st.divider()

    st.markdown("### Demo Scenario Context")
    st.info(
        "**Dataset:** Portuguese bank marketing data (2008–2010).  \n"
        "**Drift scenario:** Simulate a modern (2024) live population — higher Euribor 3M rates "
        "reflecting ECB hike cycle, and near-universal mobile phone ownership shifting contact "
        "from landline to cellular.  \n\n"
        "| Feature | Training (2008–10) | Simulated Drift (Modern) | Significance |\n"
        "|---|---|---|---|\n"
        "| `euribor3m` | Bimodal: 0.6–2.0 or 4.5–5.0 | Concentrated 2.8–3.8 (ECB neutral rate) | **PSI > 1.0** |\n"
        "| `contact` | 64% cellular, 36% telephone | 92% cellular, 8% telephone | **χ² highly significant** |\n\n"
        "Subscribers contacted via **cellular** have **2.7× higher subscription rate** (14.7% vs 5.4%).  \n"
        "Demo flow: **Demo Control → Drift & Agent → Human Approval → Queue / DLQ → Model & CI**."
    )

    st.divider()

    st.markdown("### CI Snapshot Tests")
    st.info(
        "CI runs snapshot tests on every push to verify that the drift detector output "
        "matches a known-good reference. Tests live in `tests/` and run via GitHub Actions on every PR."
    )

# ── Refresh ───────────────────────────────────────────────────────────────────

st.divider()
if st.button("↻ Refresh Dashboard"):
    st.rerun()
