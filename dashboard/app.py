"""
app.py

Streamlit dashboard for AutoHeal.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Absolute path to project root — works regardless of working directory or Docker mount
_PROJECT_ROOT = Path(__file__).parent.parent

from api_client import (
    get_health,
    get_registry_state,
    get_drift_status,
    post_drift_check,
    get_queue_status,
    get_pending_approvals,
    submit_approval_decision,
    post_predict,
    post_demo_reset,
    get_investigations,
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
.kv-label { color: #64748b; min-width: 140px; }
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
.stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: .9rem 1rem;
    text-align: center;
}
.stat-label { font-size: .78rem; color: #64748b; margin-bottom: .2rem; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
.stat-sub   { font-size: .75rem; color: #94a3b8; }
.shift-arrow { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🩺 AutoHeal</h1>
  <p>MLOps control room · drift detection · agent investigation · human-in-the-loop approval</p>
</div>
""", unsafe_allow_html=True)

# ── Reference stats (loaded once from file, no API call) ──────────────────────

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
    "critical":"pill-red","high":"pill-orange","medium":"pill-yellow",
    "low":"pill-green","local":"pill-blue","none":"pill-gray","unknown":"pill-gray",
}
def severity_pill(s):
    s = (s or "unknown").lower()
    return f'<span class="pill {_SEV_PILL.get(s,"pill-gray")}">{s}</span>'

def kv(label, value):
    return (f'<div class="kv-row"><span class="kv-label">{label}</span>'
            f'<span class="kv-value">{value}</span></div>')

def safe(fn):
    try: return fn()
    except: return None

def fmt_pct(v): return f"{v*100:.1f}%"

# ── Fetch state ───────────────────────────────────────────────────────────────

health         = safe(get_health)
registry_state = safe(get_registry_state)
drift_status   = safe(get_drift_status)
queue_status   = safe(get_queue_status)
approvals      = safe(get_pending_approvals)
investigations = safe(get_investigations)

report            = (drift_status or {}).get("report", drift_status or {})
# insufficient_data means no predictions yet — no drift detected, so display LOW
severity = (
    "low" if report.get("status") == "insufficient_data"
    else report.get("severity", "unknown")
)
pending_approvals = (approvals or {}).get("pending", [])
inv_list          = (investigations or {}).get("investigations", []) \
                    if isinstance(investigations, dict) else []

# ── Top metrics ───────────────────────────────────────────────────────────────

tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
tm1.metric("Model Service",      "OK ✓" if health else "DOWN ✗")
tm2.metric("Version",            (registry_state or {}).get("model_version") or "—")
tm3.metric("Drift Severity",     severity.upper())
tm4.metric("Drifted Features",   len(report.get("affected_features", [])))
tm5.metric("Pending Approvals",  len(pending_approvals))
tm6.metric("Investigations",     len(inv_list))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["Overview", "Simulate & Test", "Drift Monitor", "Queue + DLQ", "Approvals", "Agent"])
tab_overview, tab_simulate, tab_drift, tab_queue, tab_approvals, tab_agent = tabs

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="card"><div class="card-title">Service Health</div>', unsafe_allow_html=True)
        if health:
            st.success("Model service is reachable.")
            st.markdown(kv("Status", health.get("status","—")) + kv("Service", health.get("service","—")), unsafe_allow_html=True)
        else:
            st.error("Model service is unreachable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="card"><div class="card-title">Production Model</div>', unsafe_allow_html=True)
        if registry_state:
            st.markdown(
                kv("Model",     registry_state.get("model_name","—")) +
                kv("Version",   registry_state.get("model_version") or "—") +
                kv("Stage",     severity_pill(registry_state.get("stage","—"))) +
                kv("Threshold", registry_state.get("threshold","—")) +
                kv("AUC",       f'{registry_state["auc"]:.4f}' if registry_state.get("auc") else "—") +
                kv("Recall",    f'{registry_state["recall"]:.4f}' if registry_state.get("recall") else "—"),
                unsafe_allow_html=True,
            )
        else:
            st.error("Registry unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Demo scenario context
    st.markdown("#### Demo Scenario")
    st.info(
        "**Dataset context:** Portuguese bank marketing data (2008–2010).  \n"
        "**Drift scenario:** Simulate a modern (2024) live population — higher Euribor 3M rates "
        "reflecting ECB hike cycle, and near-universal mobile phone ownership shifting contact "
        "from landline to cellular.  \n\n"
        "| Feature | Training (2008–10) | Simulated Drift (Modern) | Significance |\n"
        "|---|---|---|---|\n"
        "| `euribor3m` | Bimodal: 0.6–2.0 or 4.5–5.0 | Concentrated 2.8–3.8 (ECB neutral rate) | **PSI > 1.0** |\n"
        "| `contact` | 64% cellular, 36% telephone | 92% cellular, 8% telephone | **χ² highly significant** |\n\n"
        "Subscribers contacted via **cellular** have **2.7× higher subscription rate** (14.7% vs 5.4%).  \n"
        "Demo flow: Reset → Send 150 drift predictions → Trigger check → Review agent investigation → Approve/Reject."
    )

    st.markdown("#### Demo Control")
    st.warning("**Reset before every demo** — clears predictions, drift state, and investigations for a clean run.", icon="⚠️")
    if st.button("🔄 Reset Demo State", type="primary"):
        try:
            result = post_demo_reset()
            st.success(f"Reset complete. {len(result.get('cleared', []))} items cleared.")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATE & TEST
# ══════════════════════════════════════════════════════════════════════════════

# ── Reference distributions (shown to user for context) ───────────────────────
_REF_EURIBOR_MEAN = 3.616
_REF_EURIBOR_MIN  = 0.634
_REF_EURIBOR_MAX  = 5.045
_REF_CELLULAR_PCT = 0.636
_REF_TELEPHONE_PCT = 0.364

_DRIFT_EURIBOR_MEAN = 3.2
_DRIFT_EURIBOR_STD  = 0.25
_DRIFT_CELLULAR_PCT = 0.92

# Client-facing features only (strip internal/target columns before sending)
_DROP_COLS = {"pdays_was_999", "y"}

@st.cache_data(show_spinner=False)
def _load_sim_source() -> pd.DataFrame:
    """Load the training/reference split as the simulation source.

    Using reference data (not test data) ensures that non-overridden features
    match the reference distribution exactly, so only euribor3m and contact
    show drift when we override them — no false positives on correlated features.
    """
    df = pd.read_csv(_PROJECT_ROOT / "data" / "reference.csv")
    return df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

_SIM_DF = _load_sim_source()


def _sample_row(seed: int, idx: int) -> dict:
    """Return one reference row as a plain dict, deterministically by index."""
    row = _SIM_DF.iloc[idx % len(_SIM_DF)]
    return {
        col: (int(v) if isinstance(v, (np.integer,)) else
              float(v) if isinstance(v, (np.floating,)) else v)
        for col, v in row.items()
    }


def _make_normal_row(rng: random.Random, idx: int) -> dict:
    """Reference row — all features authentic to training distribution, no overrides."""
    return _sample_row(rng.randint(0, len(_SIM_DF) - 1), idx)


def _make_drift_row(rng: random.Random, idx: int) -> dict:
    """Reference row with only euribor3m and contact overridden to simulate modern drift."""
    row = _sample_row(rng.randint(0, len(_SIM_DF) - 1), idx)
    row["euribor3m"] = round(max(0.5, min(5.1, rng.gauss(_DRIFT_EURIBOR_MEAN, _DRIFT_EURIBOR_STD))), 3)
    row["contact"]   = "cellular" if rng.random() < _DRIFT_CELLULAR_PCT else "telephone"
    return row


with tab_simulate:

    # ── Distribution Reference Card ──────────────────────────────────────────
    st.markdown("### Feature Distribution: Reference vs Drift")

    if REF:
        ref_e = REF["euribor3m"]
        ref_c = REF["contact"]
    else:
        ref_e = {"mean":3.616,"std":1.737,"min":0.634,"max":5.045,"p25":1.344,"p50":4.857,"p75":4.961}
        ref_c = {"cellular":0.636,"telephone":0.364}

    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown("**`euribor3m` — Euribor 3-Month Rate**")
        st.caption("Training data is bimodal (2008 crisis): low-rate cluster + high-rate cluster. "
                   "Near-zero observations between 2.0–4.0. Modern ECB rates fall exactly in that gap → high PSI.")
        df_e = pd.DataFrame({
            "Stat": ["Mean", "Std Dev", "Min", "Max", "p25", "p50 (median)", "p75"],
            "Reference (2008–10)": [
                ref_e["mean"], ref_e["std"], ref_e["min"], ref_e["max"],
                ref_e["p25"], ref_e["p50"], ref_e["p75"]
            ],
            "Drift Simulation": [
                f"~{_DRIFT_EURIBOR_MEAN}",
                f"~{_DRIFT_EURIBOR_STD}",
                "≈ 2.3",
                "≈ 4.1",
                "≈ 2.9",
                f"~{_DRIFT_EURIBOR_MEAN}",
                "≈ 3.5",
            ],
        })
        st.dataframe(df_e, use_container_width=True, hide_index=True)

    with ec2:
        st.markdown("**`contact` — Communication Channel**")
        st.caption("In 2008–10 Portugal, 36% of contacts were via landline. "
                   "Modern mobile penetration (>95%) shifts this heavily toward cellular. "
                   "Cellular contacts subscribe at 2.7× the rate of telephone contacts.")
        df_c = pd.DataFrame({
            "Channel": ["cellular", "telephone"],
            "Reference (2008–10)": [fmt_pct(ref_c["cellular"]), fmt_pct(ref_c["telephone"])],
            "Drift Simulation":    [fmt_pct(_DRIFT_CELLULAR_PCT), fmt_pct(1-_DRIFT_CELLULAR_PCT)],
            "Δ":                   [
                f"+{fmt_pct(_DRIFT_CELLULAR_PCT - ref_c['cellular'])}",
                f"-{fmt_pct(ref_c['telephone'] - (1-_DRIFT_CELLULAR_PCT))}",
            ],
            "Subscription rate": ["14.7%", "5.4%"],
        })
        st.dataframe(df_c, use_container_width=True, hide_index=True)

    st.divider()

    # ── Single Entry ─────────────────────────────────────────────────────────
    with st.expander("Submit a single custom prediction"):
        with st.form("single_entry"):
            st.markdown("**Numeric**")
            s1, s2, s3, s4, s5 = st.columns(5)
            s_age      = s1.number_input("age",         0, 120, 42)
            s_campaign = s2.number_input("campaign",    0,  50,  1)
            s_pdays    = s3.number_input("pdays",       0, 999, 999, help="999 = not previously contacted")
            s_prev     = s4.number_input("previous",    0,  20,  0)
            s_eurib    = s5.number_input("euribor3m", value=3.2, format="%.3f")

            s6, s7, s8, s9 = st.columns(4)
            s_nr   = s6.number_input("nr.employed",   value=5191.0, format="%.1f")
            s_emp  = s7.number_input("emp.var.rate",  value=1.1,    format="%.1f")
            s_cpi  = s8.number_input("cons.price.idx",value=93.994, format="%.3f")
            s_cci  = s9.number_input("cons.conf.idx", value=-36.4,  format="%.1f")

            st.markdown("**Categorical**")
            cat1, cat2, cat3, cat4, cat5 = st.columns(5)
            s_job  = cat1.selectbox("job",        ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"])
            s_mar  = cat2.selectbox("marital",    ["divorced","married","single","unknown"])
            s_edu  = cat3.selectbox("education",  ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"])
            s_def  = cat4.selectbox("default",    ["no","yes","unknown"])
            s_hous = cat5.selectbox("housing",    ["no","yes","unknown"])

            cat6, cat7, cat8, cat9, cat10 = st.columns(5)
            s_loan = cat6.selectbox("loan",        ["no","yes","unknown"])
            s_cont = cat7.selectbox("contact",     ["cellular","telephone"])
            s_mon  = cat8.selectbox("month",       ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
            s_dow  = cat9.selectbox("day_of_week", ["mon","tue","wed","thu","fri"])
            s_pout = cat10.selectbox("poutcome",   ["failure","nonexistent","success"])

            submit_single = st.form_submit_button("Submit", type="primary")

        if submit_single:
            p = {"age":s_age,"campaign":s_campaign,"pdays":s_pdays,"previous":s_prev,
                 "euribor3m":s_eurib,"nr.employed":s_nr,"emp.var.rate":s_emp,
                 "cons.price.idx":s_cpi,"cons.conf.idx":s_cci,
                 "job":s_job,"marital":s_mar,"education":s_edu,"default":s_def,
                 "housing":s_hous,"loan":s_loan,"contact":s_cont,"month":s_mon,
                 "day_of_week":s_dow,"poutcome":s_pout}
            try:
                r = post_predict(p)
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction",  r.get("prediction","—").upper())
                c2.metric("Probability", f'{r.get("probability_yes",0):.4f}')
                c3.metric("Threshold",   r.get("threshold_used","—"))
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # ── Batch Simulator ──────────────────────────────────────────────────────
    st.markdown("### Batch Simulator")
    st.caption(
        "Send a batch of synthetic predictions. "
        "**Normal** replicates the 2008–10 training distribution. "
        "**Drift** injects a modern population — euribor3m in the 2.8–3.8 range, 92% cellular contact."
    )

    b1, b2, b3 = st.columns([2, 2, 1])
    sim_mode  = b1.radio("Mode", ["Normal (baseline)", "Drift (modern simulation)"], horizontal=True)
    sim_count = b2.slider("Entries", 10, 200, 150, 10)
    b3.markdown("<br>", unsafe_allow_html=True)
    run_sim   = b3.button(f"Send {sim_count}", type="primary", use_container_width=True)

    is_drift = "Drift" in sim_mode

    if run_sim:
        rng = random.Random(42 if not is_drift else 99)
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
            progress.progress((i+1)/sim_count, text=f"{i+1}/{sim_count}")

        progress.empty()

        if errs:
            st.warning(f"Sent {ok}/{sim_count} entries. {len(errs)} failed: {errs[0]}")
        else:
            st.success(f"✓ Sent {ok} {'drift' if is_drift else 'normal'} entries to the model.")

        # Show live stats vs reference
        st.markdown("#### Sent Batch — Distribution Summary")
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            st.markdown("**euribor3m**")
            arr = np.array(sent_eurib)
            df_stats = pd.DataFrame({
                "Stat": ["Mean", "Std", "Min", "Max"],
                "Reference": [ref_e["mean"], ref_e["std"], ref_e["min"], ref_e["max"]],
                "This batch": [round(arr.mean(),3), round(arr.std(),3), round(arr.min(),3), round(arr.max(),3)],
            })
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        with s_col2:
            st.markdown("**contact**")
            from collections import Counter
            cnt = Counter(sent_contact)
            total = len(sent_contact)
            df_contact = pd.DataFrame({
                "Channel":    ["cellular", "telephone"],
                "Reference":  [fmt_pct(ref_c["cellular"]), fmt_pct(ref_c["telephone"])],
                "This batch": [fmt_pct(cnt.get("cellular",0)/total), fmt_pct(cnt.get("telephone",0)/total)],
                "Count":      [cnt.get("cellular",0), cnt.get("telephone",0)],
            })
            st.dataframe(df_contact, use_container_width=True, hide_index=True)

    st.divider()

    # ── Drift Check ──────────────────────────────────────────────────────────
    st.markdown("### Trigger Drift Check")
    st.caption("Runs the detector against the last 100 predictions. Notifies the agent if severity changed from last check.")

    dc1, dc2 = st.columns([1, 4])
    check_btn = dc1.button("Run Drift Check", type="secondary", use_container_width=True)

    if check_btn:
        try:
            res  = post_drift_check()
            rep  = res.get("report", res)
            sev  = rep.get("severity", "unknown")
            stat = rep.get("status", "—")
            aff  = rep.get("affected_features", [])
            sent = res.get("event_sent", False)
            n_av = rep.get("records_available")
            n_req = rep.get("records_required")

            if stat == "insufficient_data":
                st.warning(f"Need **{n_req}** predictions, have **{n_av}**. Send more predictions first.")
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Severity",         sev.upper())
                r2.metric("Drifted Features", len(aff))
                r3.metric("Agent notified",   "Yes ✓" if sent else "No (severity unchanged)")

                if aff:
                    st.markdown("**Drifted:** " + "  ".join(f"`{f}`" for f in aff))
                else:
                    st.success("No drift detected.")

                if not sent and stat == "ok":
                    wh = res.get("webhook_response", {})
                    if wh.get("error"):
                        st.error(
                            f"**Agent webhook failed** — {wh['error']}  \n"
                            f"{wh.get('hint', '')}  \n\n"
                            "Check that the agent service is running on port 8001."
                        )
                    else:
                        st.info(
                            "Agent was **not** notified — severity hasn't changed since last check. "
                            "Click **Reset Demo State** on the Overview tab, then run again."
                        )
        except Exception as e:
            st.error(f"Drift check failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# DRIFT MONITOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_drift:
    if not drift_status:
        st.error("Could not load drift status.")
    else:
        sval     = report.get("status", "unknown")
        aff      = report.get("affected_features", [])
        num_res  = report.get("numeric_results", {})
        cat_res  = report.get("categorical_results", {})
        n_avail  = report.get("records_available")
        n_req    = report.get("records_required")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Status",           sval)
        d2.metric("Severity",         severity.upper())
        d3.metric("Drifted Features", len(aff))
        if n_avail is not None:
            d4.metric("Window",       f"{n_avail} / {n_req}")

        if sval == "insufficient_data":
            st.warning(f"Need **{n_req}** predictions, have **{n_avail}**. Use Simulate & Test tab.")
        else:
            if aff:
                st.markdown("**Drifted features:** " + "  ".join(f"`{f}`" for f in aff))

            _SEV_BG = {"CRITICAL":"#fee2e2","HIGH":"#ffedd5","MEDIUM":"#fef9c3","LOW":"#dcfce7"}
            def _sev_color(val):
                return f"background-color: {_SEV_BG.get(val,'')}"

            if num_res:
                st.markdown("#### Numeric Features — PSI (Population Stability Index)")
                st.caption("PSI < 0.1: no drift · 0.1–0.25: moderate · > 0.25: significant")
                rows = [{"Feature": f, "PSI": round(d.get("psi",0),4),
                         "Severity": d.get("severity","—").upper(),
                         "Drifted": "✓" if d.get("is_drifted") else ""}
                        for f, d in num_res.items()]
                df_n = pd.DataFrame(rows).sort_values("PSI", ascending=False)
                st.dataframe(df_n.style.applymap(_sev_color, subset=["Severity"]),
                             use_container_width=True, hide_index=True)

            if cat_res:
                st.markdown("#### Categorical Features — Chi-Square Test")
                st.caption("p-value < 0.05: distribution significantly different from reference")
                rows = [{"Feature": f, "Chi²": round(d.get("chi_square",0),2),
                         "p-value": round(d.get("p_value",1),4),
                         "Severity": d.get("severity","—").upper(),
                         "Drifted": "✓" if d.get("is_drifted") else ""}
                        for f, d in cat_res.items()]
                df_c = pd.DataFrame(rows).sort_values("Chi²", ascending=False)
                st.dataframe(df_c.style.applymap(_sev_color, subset=["Severity"]),
                             use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# QUEUE + DLQ
# ══════════════════════════════════════════════════════════════════════════════

with tab_queue:
    if not queue_status:
        st.error("Could not load queue status.")
    else:
        q1, q2, q3 = st.columns(3)
        q1.metric("Queue",       queue_status.get("queue","—"))
        q2.metric("Queue depth", queue_status.get("queue_depth", 0))
        q3.metric("DLQ depth",   queue_status.get("dlq_depth", 0))
        if queue_status.get("dlq_depth", 0) > 0:
            st.error(f"{queue_status['dlq_depth']} job(s) in the dead-letter queue.")
        else:
            st.success("Dead-letter queue is empty.")
        st.caption("The agent dispatches `replay_test` jobs on CRITICAL/HIGH drift. Workers process them with exponential-backoff retries.")

# ══════════════════════════════════════════════════════════════════════════════
# APPROVALS
# ══════════════════════════════════════════════════════════════════════════════

with tab_approvals:
    if approvals is None:
        st.error("Could not load approvals.")
    elif not pending_approvals:
        st.success("No pending approvals.")
        st.caption("CRITICAL drift will pause the agent here for human approval before any production-touching action (retrain + promote).")
    else:
        st.warning(f"{len(pending_approvals)} approval request(s) pending.")
        for appr in pending_approvals:
            aid   = appr.get("approval_id","unknown")
            atype = appr.get("action_type", (appr.get("recommended_action") or {}).get("action_type","—"))
            with st.expander(f"Request {aid[:8]}…  ·  {atype}", expanded=True):
                st.markdown(
                    kv("Approval ID", aid) + kv("Action", atype) +
                    kv("Reason", appr.get("reason","—")) + kv("Created", appr.get("created_at","—")),
                    unsafe_allow_html=True,
                )
                rev = st.text_input("Reviewer",       "demo_user",              key=f"rev_{aid}")
                rsn = st.text_input("Decision reason","Reviewed in dashboard",  key=f"rsn_{aid}")
                ab, rb = st.columns(2)
                if ab.button("✓ Approve", key=f"app_{aid}", type="primary"):
                    submit_approval_decision(aid, True, rev, rsn)
                    st.success("Approved — agent will proceed with retrain.")
                if rb.button("✗ Reject",  key=f"rej_{aid}"):
                    submit_approval_decision(aid, False, rev, rsn)
                    st.warning("Rejected — action downgraded to monitor-only.")

# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown("### Agent Investigations")
    if investigations is None:
        st.warning("Agent service not reachable (port 8001). Start the agent container to see investigations.")
    elif not inv_list:
        st.info("No investigations yet. Trigger a drift event to create one.")
    else:
        for inv in inv_list:
            inv_id   = inv.get("investigation_id","—")
            sev      = inv.get("severity","unknown")
            status_i = inv.get("status","—")
            with st.expander(f"{inv_id[:12]}…  ·  {sev.upper()}  ·  {status_i}", expanded=False):
                st.markdown(
                    kv("Investigation ID", inv_id) +
                    kv("Severity",   severity_pill(sev)) +
                    kv("Status",     status_i) +
                    kv("Opened",     inv.get("opened_at","—")) +
                    kv("Action",     (inv.get("recommended_action") or {}).get("action_type","—")) +
                    kv("Summary",    inv.get("comms_summary","—")),
                    unsafe_allow_html=True,
                )
    with st.expander("Raw JSON"):
        st.json(investigations or {})

# ── Refresh ───────────────────────────────────────────────────────────────────

st.divider()
if st.button("↻ Refresh Dashboard"):
    st.rerun()
