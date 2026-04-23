"""
app.py — Elective C-Section Planning Interface (version 5 Mar 23, 2026)

Tab 1 : Sequential policy simulation
          pooled-EDF vs Optimal dedicated-EDF(μᴱ) for each specified μᴱ.
          Three metrics per comparison: case-loss fractions, case losses,
          total case-loss cost (+ cost-savings delta).
          User confirms before each new μᴱ is run.

Tab 2 : Summary
          Three summary plots across all completed policies, then full
          detailed breakdown (wait times, histograms, N(t)) for all policies.

Note: simulation time unit is WEEKS. Lead times entered in days are converted
      to weeks (÷7) after sampling, before being stored in the arrival sequence.
"""

from __future__ import annotations

import os
import datetime
import json
import math
import io
import zipfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from concurrent.futures import ProcessPoolExecutor
from sim_core import (
    sweep_gamma,
    _pdf_cdf_lead_grid,
    generate_common_trace_v7_style,
    shift_early_lead_times,
    simulate_pooled_queue_full,
    simulate_reservation_2queue_full,
    _run_stats_core,
    _run_one_gamma_worker,
)

SESSIONS_DIR = os.path.join(_THIS_DIR, "saved sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Cache CSV serialisation so reruns don't re-encode large DataFrames."""
    return df.to_csv(index=False).encode()


def _traces_to_json(traces) -> str:
    """Serialise a list of trace-runs (each run is a List[PatientTrace]) to JSON."""
    return json.dumps([
        [[tp.pid, tp.cohort, tp.arrival, tp.lead_time, tp.deadline, tp.svc]
         for tp in trace]
        for trace in traces
    ])


def _json_to_traces(raw: str):
    """Deserialise traces from JSON (list of runs, each run is a List[PatientTrace])."""
    from sim_core import PatientTrace
    return [
        [PatientTrace(pid=r[0], cohort=r[1], arrival=r[2],
                      lead_time=r[3], deadline=r[4], svc=r[5])
         for r in trace_data]
        for trace_data in json.loads(raw)
    ]


def _session_to_zip() -> bytes:
    """
    Serialise all tab1 results to a portable ZIP archive.
    Structure:
      config.json           — scalar settings + pooled_mu_E
      saved_traces.json     — the N baseline PatientTrace sequences
      edf_baseline/         — pooled-EDF policy files
      result_<mu>/info.json + hyb/ — Hybrid data per μᴱ
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        cfg = {k: st.session_state[k] for k in [
            "tab1_means_config", "tab1_gamma_config", "tab1_params_base",
            "tab1_n_runs", "tab1_seed0", "tab1_T_max", "tab1_T0",
            "tab1_lambdaE", "tab1_lambdaL",
            "tab1_pooled_mu_E",
        ] if k in st.session_state}
        zf.writestr("config.json", json.dumps(cfg, default=str, ensure_ascii=False))

        # Save baseline traces so additional μᴱ runs can be added after a reload
        if "tab1_saved_traces" in st.session_state:
            zf.writestr("saved_traces.json",
                        _traces_to_json(st.session_state["tab1_saved_traces"]))

        def _write_policy(prefix: str, data: dict):
            meta = {k: v for k, v in data.items() if k not in ("stats", "df_all", "hists")}
            zf.writestr(f"{prefix}/meta.json", json.dumps(meta, default=str, ensure_ascii=False))
            zf.writestr(f"{prefix}/stats.csv",    data["stats"].to_csv())
            zf.writestr(f"{prefix}/patients.csv", data["df_all"].to_csv(index=False))
            hists_raw = [[list(ev) for ev in run] for run in data.get("hists", [])]
            zf.writestr(f"{prefix}/hists.json", json.dumps(hists_raw))

        if "tab1_edf" in st.session_state:
            _write_policy("edf_baseline", st.session_state["tab1_edf"])

        for res in st.session_state.get("tab1_results", []):
            mu_key = int(res["mu_E"])
            zf.writestr(f"result_{mu_key}/info.json",
                        json.dumps({"mu_E": res["mu_E"], "gamma_star": res["gamma_star"]}))
            _write_policy(f"result_{mu_key}/hyb", res["hyb"])

    buf.seek(0)
    return buf.getvalue()


def _zip_to_session(zip_bytes: bytes) -> dict:
    """Deserialise a ZIP archive back into a dict for st.session_state."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        cfg   = json.loads(zf.read("config.json"))
        result.update(cfg)

        if "saved_traces.json" in names:
            result["tab1_saved_traces"] = _json_to_traces(
                zf.read("saved_traces.json").decode())

        def _read_policy(prefix: str) -> dict:
            meta   = json.loads(zf.read(f"{prefix}/meta.json"))
            stats  = pd.read_csv(io.StringIO(zf.read(f"{prefix}/stats.csv").decode()), index_col=0)
            df_all = pd.read_csv(io.StringIO(zf.read(f"{prefix}/patients.csv").decode()))
            hists_raw = json.loads(zf.read(f"{prefix}/hists.json"))
            hists = [[tuple(ev) for ev in run] for run in hists_raw]
            return {**meta, "stats": stats, "df_all": df_all, "hists": hists}

        if "edf_baseline/meta.json" in names:
            result["tab1_edf"] = _read_policy("edf_baseline")

        tab1_results = []
        mu_keys = sorted({n.split("/")[0] for n in names if n.startswith("result_")}, key=lambda x: float(x.replace("result_", "")))
        for mk in mu_keys:
            info  = json.loads(zf.read(f"{mk}/info.json"))
            hyb_d = _read_policy(f"{mk}/hyb")
            tab1_results.append(dict(
                mu_E=info["mu_E"], gamma_star=info["gamma_star"], hyb=hyb_d))
        result["tab1_results"] = tab1_results

    # ── Queue widget values for restoration on next rerun (via _restore_ prefix) ──
    # Widget keys cannot be set after instantiation, so we use a staging prefix.
    # The block at the top of the script copies these to the real widget keys
    # before any widget is rendered.
    _R = "_restore_"
    pb = result.get("tab1_params_base", {})
    gc = result.get("tab1_gamma_config", {})
    if pb:
        result[f"{_R}sb_lambdaE"]  = float(pb.get("λE",     16.87))
        result[f"{_R}sb_lambdaL"]  = float(pb.get("λL",      6.24))
        result[f"{_R}sb_N"]        = int(pb.get("N",         18))
        result[f"{_R}sb_T_max"]    = float(pb.get("T_max",   152))
        result[f"{_R}sb_T0"]       = float(pb.get("warmup",  100))
        result[f"{_R}sb_p_ns"]     = float(pb.get("p_ns",    0.0))
        ep = pb.get("early_params", {}); lp = pb.get("late_params", {})
        result[f"{_R}early_min"]   = float(pb.get("early_min", 3.0))
        result[f"{_R}early_max"]   = float(pb.get("early_max", 10.0))
        if "mean" in ep: result[f"{_R}early_mean"] = float(ep["mean"])
        if "sd"   in ep: result[f"{_R}early_sd"]   = float(ep["sd"])
        result[f"{_R}late_min"]    = float(pb.get("late_min", 3.0))
        result[f"{_R}late_max"]    = float(pb.get("late_max", 10.0))
        if "mean" in lp: result[f"{_R}late_mean"]  = float(lp["mean"])
        if "sd"   in lp: result[f"{_R}late_sd"]    = float(lp["sd"])
    if gc:
        result[f"{_R}t1_cE"]    = float(gc.get("cE",    1.0))
        result[f"{_R}t1_cL"]    = float(gc.get("cL",    2.0))
        result[f"{_R}t1_v"]     = float(gc.get("v",     0.1))
        result[f"{_R}t1_Delta_pct"] = float(gc.get("Delta", 0.2)) * 100.0
        result[f"{_R}t1_gmin"]  = float(gc.get("gmin",  0.0))
        result[f"{_R}t1_gmax"]  = float(gc.get("gmax",  1.0))
        result[f"{_R}t1_gstep"] = float(gc.get("gstep", 0.01))
    if "tab1_seed0" in result:
        result[f"{_R}sb_seed0"] = int(result["tab1_seed0"])
    if "tab1_n_runs" in result:
        result[f"{_R}t1_n_runs"] = int(result["tab1_n_runs"])
    if "tab1_means_config" in result:
        result[f"{_R}t1_means_text"] = ", ".join(
            f"{m:.0f}" for m in result["tab1_means_config"])

    return result

# ── Tab 2 cache ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_mean_sweep(means_t, gammas_t, params_json_, seed0_, cE_, cL_, Delta_):
    pb = json.loads(params_json_)
    rows = []
    for mu_E in means_t:
        adj = _adj_early_lead(float(mu_E))
        pb_mu = dict(pb); pb_mu.update(adj); pb_mu["n_runs"] = pb["n_runs"]
        df_sw = sweep_gamma(gammas_t, pb_mu, seed0_, cE_, cL_, Delta_)
        # C is already count-based — select γ* directly on C (no scaling)
        feas = df_sw[df_sw["ineq_level"] <= 1e-12]
        best_row = (feas.loc[feas["C"].idxmin()] if len(feas) > 0
                    else df_sw.loc[df_sw["ineq_level"].idxmin()])
        rows.append({
            "mean_E": float(mu_E), "adj_min": adj["early_min"], "adj_max": adj["early_max"],
            "gamma_star": float(best_row["gamma"]),
            "C_count_opt": float(best_row["C"]), "C_rate_opt": float("nan"),
            "pE_opt": float(best_row["pE"]), "pL_opt": float(best_row["pL"]),
            "AE_opt": float(best_row["AE"]), "AL_opt": float(best_row["AL"]),
            "phi_opt": float(best_row["phi"]), "ineq_level_opt": float(best_row["ineq_level"]),
            "feasible": bool(best_row["ineq_level"] <= 1e-12),
        })
    return pd.DataFrame(rows)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Elective C-Section Planning Interface", layout="wide")
st.title("Elective C-Section Planning Interface")

# ── Restore widget values from a loaded session (must run before any widget) ─
# _zip_to_session stores values under "_restore_<key>" so they can be applied
# here, before widgets are instantiated, avoiding StreamlitAPIException.
for _rk in [k for k in list(st.session_state.keys()) if k.startswith("_restore_")]:
    st.session_state[_rk[len("_restore_"):]] = st.session_state.pop(_rk)

# ── Initialize widget defaults (only if key not already set by restore above) ─
for _k, _v in [
    ("sb_lambdaE", 16.87), ("sb_lambdaL", 6.24), ("sb_N", 18),
    ("sb_T_max", 152.0),   ("sb_T0", 100.0), ("sb_p_ns", 0.0), ("sb_seed0", 42),
    ("t1_cE", 1.0), ("t1_cL", 2.0), ("t1_Delta_pct", 20.0), ("t1_v", 0.1),
    ("t1_gmin", 0.0), ("t1_gmax", 1.0), ("t1_gstep", 0.01),
    ("t1_means_text", "70"), ("t1_n_runs", 50),
    ("early_min", 3.0), ("early_max", 10.0), ("early_mean", 7.0), ("early_sd", 4.0),
    ("late_min",  3.0), ("late_max",  10.0), ("late_mean",  7.0), ("late_sd",  4.0),
]:
    st.session_state.setdefault(_k, _v)

# Computed before sidebar so sidebar widgets can use it for disabled=
_tab1_locked = bool(st.session_state.get("tab1_results"))

# ── Constants ────────────────────────────────────────────────────────────────
_BASE_MEAN      = 7.0
_BASE_MIN       = 3.0
_BASE_MAX       = 10.0
_LEAD_SD        = 4.0
_LEAD_MIN_FLOOR = 0.5

def _adj_early_lead(mean_E: float, sd: float = _LEAD_SD,
                    base_min: float = _BASE_MIN, base_max: float = _BASE_MAX,
                    base_mean: float = _BASE_MEAN) -> Dict[str, Any]:
    delta   = mean_E - base_mean
    adj_min = max(_LEAD_MIN_FLOOR, base_min + delta)
    adj_max = max(base_max + delta, adj_min + 0.5)
    return dict(early_F="Lognormal (truncated)",
                early_params={"mean": mean_E, "sd": sd},
                early_min=float(adj_min), early_max=float(adj_max))

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def _sb_header(label: str) -> None:
    st.sidebar.markdown(
        f'<div style="background:#1b4f8a;color:#ffffff;padding:5px 10px;'
        f'border-radius:5px;margin:14px 0 5px 0;font-size:0.78em;'
        f'font-weight:700;letter-spacing:0.09em;text-transform:uppercase;">'
        f'{label}</div>', unsafe_allow_html=True)

st.sidebar.title("Parameters")

_sb_header("Cost & Equity")
t1_cE    = st.sidebar.number_input("Early abandon cost (cᴱ)",      min_value=0.0, step=0.1,  format="%.3f", key="t1_cE",    disabled=_tab1_locked)
t1_cL    = st.sidebar.number_input("Late abandon cost (cᴸ)",       min_value=0.0, step=0.1,  format="%.3f", key="t1_cL",    disabled=_tab1_locked)
t1_v     = st.sidebar.number_input("Reserved-slot value (v)",       min_value=0.0, step=0.01, format="%.3f", key="t1_v",     disabled=_tab1_locked)
t1_Delta_pct = st.sidebar.number_input("Equity tolerance Δ (%)",    min_value=0.0, step=1.0,  format="%.1f", key="t1_Delta_pct", disabled=_tab1_locked)

_sb_header("Lead time")
means_text_t1 = st.sidebar.text_input(
    "Allocation to planned date of delivery means (μᴱ) for early engagers (days)",
    key="t1_means_text", disabled=_tab1_locked)
try:
    means_list_t1 = sorted(set(float(x.strip()) for x in means_text_t1.split(",") if x.strip()))
    if not means_list_t1: raise ValueError
    if len(means_list_t1) > 1:
        st.sidebar.caption(
            f"{len(means_list_t1)} values: " + ", ".join(f"{m:.0f}" for m in means_list_t1))
except Exception:
    st.sidebar.error("Please enter valid comma-separated numbers.")
    means_list_t1 = []
st.sidebar.caption(
    "Enter multiple comma-separated values to evaluate several μᴱ at once.")

st.sidebar.markdown("**Baseline Lead Time**")
with st.sidebar.expander("Early engagers", expanded=False):
    early_F = st.selectbox("Distribution",
        ["Lognormal (truncated)", "Uniform", "Normal (truncated)", "Deterministic"],
        index=0, key="early_F_sel")
    early_params: Dict[str, float] = {}
    early_min = st.number_input("Min (days)", 0.0, 10000.0, step=0.5, key="early_min")
    early_max = st.number_input("Max (days)", early_min+1e-6, 10000.0, step=0.5, key="early_max")
    if early_F.startswith("Lognormal") or early_F.startswith("Normal"):
        early_params["mean"] = st.number_input("Mean (days)", 0.01, 10000.0, step=0.1, key="early_mean")
        early_params["sd"]   = st.number_input("SD (days)",   0.01, 10000.0, step=0.1, key="early_sd")
    elif early_F == "Deterministic":
        early_params["c"] = st.number_input("Constant (days)", 0.01, 10000.0, value=7.0, step=0.1, key="early_c")

with st.sidebar.expander("Late engagers", expanded=False):
    late_F = st.selectbox("Distribution",
        ["Lognormal (truncated)", "Uniform", "Normal (truncated)", "Deterministic"],
        index=0, key="late_F_sel")
    late_params: Dict[str, float] = {}
    late_min = st.number_input("Min (days)", 0.0, 10000.0, step=0.5, key="late_min")
    late_max = st.number_input("Max (days)", late_min+1e-6, 10000.0, step=0.5, key="late_max")
    if late_F.startswith("Lognormal") or late_F.startswith("Normal"):
        late_params["mean"] = st.number_input("Mean (days)", 0.01, 10000.0, step=0.1, key="late_mean")
        late_params["sd"]   = st.number_input("SD (days)",   0.01, 10000.0, step=0.1, key="late_sd")
    elif late_F == "Deterministic":
        late_params["c"] = st.number_input("Constant (days)", 0.01, 10000.0, value=7.0, step=0.1, key="late_c")

_sb_header("Service Time")
def _svc_ui(prefix: str, default_p2: float = 0.03) -> Dict[str, Any]:
    with st.sidebar.expander(f"{prefix} engagers", expanded=False):
        dist = st.selectbox("Distribution", ["Discrete (1 or 2 slots)", "Exponential"],
                            index=0, key=f"svc_dist_{prefix}")
        cfg: Dict[str, Any] = {"dist": dist}
        if dist.startswith("Discrete"):
            cfg["svc1"] = st.number_input("1-slot duration", 0.001, 10000.0, value=1.0, step=0.1, key=f"svc1_{prefix}")
            cfg["svc2"] = st.number_input("2-slot duration", cfg["svc1"]+0.001, 10000.0, value=2.0, step=0.1, key=f"svc2_{prefix}")
            cfg["p2"]   = st.slider("P(2-slot)", 0.0, 1.0, float(default_p2), 0.01, key=f"p2_{prefix}")
        else:
            cfg["mu"] = st.number_input("Exponential rate μ", 1e-6, 10000.0, value=5.0, step=0.1, key=f"mu_{prefix}")
    return cfg

svc_E = _svc_ui("Early")
svc_L = _svc_ui("Late")

_sb_header("γ* Search")
t1_gstep = st.sidebar.number_input("γ Grid Step", min_value=0.01, max_value=1.0,
                                    step=0.01, format="%.2f",
                                    key="t1_gstep", disabled=_tab1_locked)
t1_gmin = st.sidebar.number_input("γ min", 0.0, 1.0, step=0.05, key="t1_gmin",
                                   disabled=_tab1_locked)
t1_gmax = st.sidebar.number_input("γ max", 0.0, 1.0, step=0.05, key="t1_gmax",
                                   disabled=_tab1_locked)

_sb_header("Experiment Settings")
t1_n_runs = st.sidebar.number_input("Replications", 1, 500, step=5, key="t1_n_runs",
                                     disabled=_tab1_locked)
T_max     = st.sidebar.number_input("Total time T (weeks)", 1, 10000, step=52, key="sb_T_max")
T0_warmup = st.sidebar.number_input("Warm-up period T₀ (weeks)", 0, int(T_max), step=10, key="sb_T0")
p_ns   = st.sidebar.slider("No-show probability p", 0.0, 1.0, step=0.01, key="sb_p_ns")
seed0  = st.sidebar.number_input("Base random seed", 0, 10_000_000, step=1, key="sb_seed0")

# ═══════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════════════
def _run_stats(df: pd.DataFrame, kpi: Dict) -> Dict:  # thin wrapper kept for compatibility
    se_e = df[(df.cohort=='Early') & (df.status=='SERVED')]
    ab_e = df[(df.cohort=='Early') & (df.status.isin(['ABANDONED', 'NO_SHOW']))]
    se_l = df[(df.cohort=='Late')  & (df.status=='SERVED')]
    ab_l = df[(df.cohort=='Late')  & (df.status.isin(['ABANDONED', 'NO_SHOW']))]
    early_df = df[df.cohort=='Early']; late_df = df[df.cohort=='Late']
    p_ae = len(ab_e)/len(early_df) if len(early_df) else np.nan
    p_al = len(ab_l)/len(late_df)  if len(late_df)  else np.nan
    return dict(
        service_rate=kpi['service_rate'], abandon_rate=kpi['abandon_rate'], util_1=kpi['util_1'],
        es_wait_active=se_e.wait_active.mean() if len(se_e) else np.nan,
        es_slack=se_e.slack.mean()             if len(se_e) else np.nan,
        es_wait_total=se_e.wait_total.mean()   if len(se_e) else np.nan,
        ea_wait_active=ab_e.wait_active.mean() if len(ab_e) else np.nan,
        ea_wait_total=ab_e.wait_total.mean()   if len(ab_e) else np.nan,
        ls_wait_active=se_l.wait_active.mean() if len(se_l) else np.nan,
        ls_slack=se_l.slack.mean()             if len(se_l) else np.nan,
        ls_wait_total=se_l.wait_total.mean()   if len(se_l) else np.nan,
        la_wait_active=ab_l.wait_active.mean() if len(ab_l) else np.nan,
        la_wait_total=ab_l.wait_total.mean()   if len(ab_l) else np.nan,
        e_wait_active=early_df.wait_active.mean(), e_slack=early_df.slack.mean(),
        e_wait_total=early_df.wait_total.mean(),
        l_wait_active=late_df.wait_active.mean(), l_slack=late_df.slack.mean(),
        l_wait_total=late_df.wait_total.mean(),
        prop_served_early=len(se_e)/len(early_df) if len(early_df) else np.nan,
        prop_aband_early=p_ae,
        prop_served_late=len(se_l)/len(late_df) if len(late_df) else np.nan,
        prop_aband_late=p_al,
        aband_ratio_e_over_l=(p_ae/p_al if (not np.isnan(p_ae) and not np.isnan(p_al) and p_al>0) else np.nan),
    )


def _mk_data(stats, df_all, hists, **extra):
    return dict(
        pE=float(stats.loc["prop_aband_early", "mean"]),
        pL=float(stats.loc["prop_aband_late",  "mean"]),
        pE_sd=float(stats.loc["prop_aband_early", "std"]),
        pL_sd=float(stats.loc["prop_aband_late",  "std"]),
        stats=stats, df_all=df_all, hists=hists, **extra)


def _scale_trace_lead_times(trace, scale=1.0/7):
    """
    Return a new trace with lead_time and deadline scaled by `scale`.
    Used to convert day-based sampled lead times to weeks (scale=1/7).
    Arrival times and service times are unchanged.
    """
    from sim_core import PatientTrace
    return [
        PatientTrace(pid=tp.pid, cohort=tp.cohort, arrival=tp.arrival,
                     lead_time=tp.lead_time * scale,
                     deadline=tp.arrival + tp.lead_time * scale + tp.svc,
                     svc=tp.svc)
        for tp in trace
    ]


def _gamma_sweep_on_traces(saved_traces, mu_E, pooled_mu_E, gc, pb, s0, T, T0):
    """
    Two-phase γ* search:
      Phase 1 — stochastic-rounding sweep over γ grid to find true γ*.
                Cost per rep: cE*ab_E + cL*ab_L - v*γ*N*(T-T0).
      Phase 2 — re-run all reps with floor(γ*N) deterministic allocation
                to produce the display metrics.
    """
    shift = (mu_E - pooled_mu_E) / 7   # mu_E and pooled_mu_E in days → shift in weeks
    N  = int(pb["N"]); pns = float(pb.get("p_ns", 0.0))
    cE = float(gc["cE"]); cL = float(gc["cL"]); v = float(gc.get("v", 0.0)); Delta = float(gc["Delta"])

    gammas = np.unique(np.clip(
        np.round(np.arange(gc["gmin"], gc["gmax"] + 1e-12, gc["gstep"]), 10), 0.0, 1.0))

    # Shift traces once, reuse for all γ values
    shifted = [shift_early_lead_times(trace, shift) for trace in saved_traces]

    # ── Stochastic-rounding sweep to find γ* ────────────────────────────
    tasks = [(shifted, float(g), N, T0, pns, s0, T, cE, cL, Delta, v) for g in gammas]

    gamma_summary = {}
    _ON_CLOUD = os.path.exists("/mount/src")
    if _ON_CLOUD:
        _results_iter = map(_run_one_gamma_worker, tasks)
        _pool_ctx = None
    else:
        n_workers = min(len(gammas), os.cpu_count() or 4)
        _pool_ctx = ProcessPoolExecutor(max_workers=n_workers)
        _results_iter = _pool_ctx.map(_run_one_gamma_worker, tasks)
    try:
        for g, costs, phis, runs, dfs, hists in _results_iter:
            mean_C   = float(np.mean(costs)) if costs else float("nan")
            mean_phi = float(np.mean(phis))  if phis  else float("nan")
            ineq     = max(0.0, abs(mean_phi) - Delta)
            stats_h  = pd.DataFrame(runs).agg(["mean", "std"]).T
            gamma_summary[float(g)] = dict(
                C=mean_C, phi=mean_phi, ineq=ineq,
                stats=stats_h, df_all=pd.concat(dfs, ignore_index=True), hists=hists)
    finally:
        if _pool_ctx is not None:
            _pool_ctx.shutdown()

    # Pick γ* — minimise C; break ties by smallest γ
    feas = {g: d for g, d in gamma_summary.items() if d["ineq"] <= 1e-12}
    gamma_star = (min(feas, key=lambda g: (feas[g]["C"], g)) if feas
                  else min(gamma_summary, key=lambda g: (gamma_summary[g]["ineq"], g)))

    best = gamma_summary[gamma_star]
    return gamma_star, _mk_data(best["stats"], best["df_all"], best["hists"],
                                mu_E=mu_E, gamma_star=gamma_star)


def _run_edf_and_first_hybrid(mu_E, gc, label):
    """
    1. Generate N traces from baseline params (no μᴱ adjustment).
    2. Run pooled-EDF on each → save tab1_edf and tab1_saved_traces.
    3. γ* search + Hybrid results in ONE pass using saved traces with
       early lead times shifted by (mu_E − pooled_mu_E).
    """
    pb = st.session_state["tab1_params_base"]
    s0 = st.session_state["tab1_seed0"]
    T  = st.session_state["tab1_T_max"]
    T0 = st.session_state["tab1_T0"]
    n  = st.session_state["tab1_n_runs"]
    N  = int(pb["N"]); pns = float(pb.get("p_ns", 0.0))

    pb_trace = dict(pb)
    pb_trace["lamE"] = float(pb["λE"]); pb_trace["lamL"] = float(pb["λL"])
    pooled_mu_E = float(pb.get("early_params", {}).get("mean", 7.0))

    # Step 1 — generate traces + run EDF
    saved_traces = []
    edf_runs, edf_dfs, edf_hists = [], [], []
    for i in range(n):
        trace = generate_common_trace_v7_style(pb_trace, seed=s0 + i)
        trace = _scale_trace_lead_times(trace)   # days → weeks
        saved_traces.append(trace)
        df_e, kpi_e, hist_e = simulate_pooled_queue_full(
            trace, "EDF", N, T0, pns, s0 + 200_000 + i, T)
        edf_dfs.append(df_e.assign(run=i)); edf_hists.append(hist_e)
        edf_runs.append(_run_stats(df_e, kpi_e))

    stats_e  = pd.DataFrame(edf_runs).agg(["mean", "std"]).T
    edf_data = _mk_data(stats_e, pd.concat(edf_dfs, ignore_index=True), edf_hists)
    st.session_state["tab1_edf"]          = edf_data
    st.session_state["tab1_saved_traces"] = saved_traces
    st.session_state["tab1_pooled_mu_E"]  = pooled_mu_E

    # Step 2 — γ* search + Hybrid (one pass on saved traces)
    gamma_star, hyb_data = _gamma_sweep_on_traces(
        saved_traces, mu_E, pooled_mu_E, gc, pb, s0, T, T0)
    st.session_state["tab1_results"].append(
        dict(mu_E=mu_E, gamma_star=gamma_star, hyb=hyb_data))


def _run_hybrid_only(mu_E, gc, label):
    """
    γ* search + Hybrid results in ONE pass using the already-saved baseline traces.
    Early lead times are shifted by (mu_E − pooled_mu_E).
    """
    pb           = st.session_state["tab1_params_base"]
    s0           = st.session_state["tab1_seed0"]
    T            = st.session_state["tab1_T_max"]
    T0           = st.session_state["tab1_T0"]
    saved_traces = st.session_state["tab1_saved_traces"]
    pooled_mu_E  = st.session_state["tab1_pooled_mu_E"]

    gamma_star, hyb_data = _gamma_sweep_on_traces(
        saved_traces, mu_E, pooled_mu_E, gc, pb, s0, T, T0)
    st.session_state["tab1_results"].append(
        dict(mu_E=mu_E, gamma_star=gamma_star, hyb=hyb_data))

# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════
def _ann_headcounts(lE, lL):
    """Annual expected patients: 52 weeks × weekly demand rate."""
    return 52.0 * lE, 52.0 * lL


def _policy_metrics(data: Dict, ann_E: float, ann_L: float, cE: float, cL: float,
                    slot_reward: float = 0.0):
    """Return (pE, pL, pE_sd, pL_sd, loss_E, loss_L, cost, cost_sd) from a stored data dict.
    slot_reward: pre-computed v×Nᴱ×52 (display) or v×Nᴱ×(T−T₀) (search), subtracted from cost for dedicated-EDF only.
    Nᴱ = floor(γ*×N) is used throughout (floor rounding)."""
    pE, pL       = data["pE"], data["pL"]
    pE_sd, pL_sd = data["pE_sd"], data["pL_sd"]
    loss_E = pE * ann_E;  loss_L = pL * ann_L
    cost   = cE * loss_E + cL * loss_L - slot_reward
    cost_sd = math.sqrt((cE*ann_E*pE_sd)**2 + (cL*ann_L*pL_sd)**2)
    return pE, pL, pE_sd, pL_sd, loss_E, loss_L, cost, cost_sd


def _comparison_charts(edf_data: Dict, hyb_data: Dict,
                       lE: float, lL: float,
                       cE: float, cL: float,
                       v: float = 0.0, N: int = 18,
                       T: float = 500.0, T0: float = 100.0):
    """
    Four charts for one (EDF, Hybrid(μᴱ)) pair:
      Row 1 — case-loss fractions | case losses
      Row 2 — total case-loss cost (absolute) | cost savings vs EDF (delta)
    """
    mu_E   = hyb_data["mu_E"]
    gstar  = hyb_data["gamma_star"]
    NE_star = gstar * N
    NL_star = N - NE_star
    lbl_h  = f"Optimal dedicated-EDF (μᴱ={mu_E:.0f}, γ*={gstar:.3f})"

    ann_E, ann_L = _ann_headcounts(lE, lL)
    pE_e, pL_e, pE_e_sd, pL_e_sd, lE_e, lL_e, cost_e, cost_e_sd = _policy_metrics(edf_data, ann_E, ann_L, cE, cL)
    _slot_reward_h = v * NE_star * 52
    pE_h, pL_h, pE_h_sd, pL_h_sd, lE_h, lL_h, cost_h, cost_h_sd = _policy_metrics(hyb_data, ann_E, ann_L, cE, cL, slot_reward=_slot_reward_h)
    aband_cost_h = cE * lE_h + cL * lL_h  # Hybrid case-loss cost (no slot reward)

    # Short x-axis labels (policies); cohort = bar colour
    x_labels = ["pooled-EDF", f"dedicated-EDF (μᴱ={mu_E:.0f})"]
    COLOR_E = "#4878d0"
    COLOR_L = "#ee854a"
    _leg = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)

    # ── Row 1 ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Case-loss fractions**")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Early engagers", x=x_labels,
                             y=[pE_e*100, pE_h*100],
                             error_y=dict(type='data', array=[pE_e_sd*100, pE_h_sd*100], visible=True),
                             marker_color=COLOR_E))
        fig.add_trace(go.Bar(name="Late engagers", x=x_labels,
                             y=[pL_e*100, pL_h*100],
                             error_y=dict(type='data', array=[pL_e_sd*100, pL_h_sd*100], visible=True),
                             marker_color=COLOR_L))
        fig.update_layout(barmode="group", yaxis_title="Case-loss fraction (%)",
                          height=360, margin=dict(t=10,b=30), legend=_leg)
        st.plotly_chart(fig, use_container_width=True, key=f"frac_{mu_E}")

    with c2:
        st.markdown("**Estimated annual case losses**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Early engagers", x=x_labels,
                              y=[lE_e, lE_h],
                              error_y=dict(type='data', array=[pE_e_sd*ann_E, pE_h_sd*ann_E], visible=True),
                              marker_color=COLOR_E))
        fig2.add_trace(go.Bar(name="Late engagers", x=x_labels,
                              y=[lL_e, lL_h],
                              error_y=dict(type='data', array=[pL_e_sd*ann_L, pL_h_sd*ann_L], visible=True),
                              marker_color=COLOR_L))
        fig2.update_layout(barmode="group",
                           yaxis_title="Estimated annual case losses",
                           height=360, margin=dict(t=10,b=30), legend=_leg)
        st.plotly_chart(fig2, use_container_width=True, key=f"loss_{mu_E}")

    # ── Row 2 ──────────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(f"**Total case-loss cost  (cᴱ={cE:.2f}, cᴸ={cL:.2f})**")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Early component", x=x_labels,
                              y=[cE*lE_e, cE*lE_h],
                              error_y=dict(type='data', array=[cE*pE_e_sd*ann_E, cE*pE_h_sd*ann_E], visible=True),
                              marker_color=COLOR_E))
        fig3.add_trace(go.Bar(name="Late component", x=x_labels,
                              y=[cL*lL_e, cL*lL_h],
                              error_y=dict(type='data', array=[cL*pL_e_sd*ann_L, cL*pL_h_sd*ann_L], visible=True),
                              marker_color=COLOR_L))
        fig3.update_layout(barmode="stack",
                           yaxis_title="Annual case-loss cost",
                           height=360, margin=dict(t=10,b=30), legend=_leg)
        st.plotly_chart(fig3, use_container_width=True, key=f"cost_{mu_E}")

    with c4:
        savings = cost_e - cost_h
        pct     = 100*savings/cost_e if cost_e > 0 else 0.0
        sav_sd  = math.sqrt(cost_e_sd**2 + cost_h_sd**2)
        color_h = "#2e7d32" if savings >= 0 else "#c62828"
        sign    = "↓" if savings >= 0 else "↑"
        st.markdown(
            f"**Total cost C(γ): EDF vs dedicated-EDF** &nbsp;&nbsp;"
            f'<span style="color:{color_h};font-size:1.05em">'
            f'{sign} {abs(savings):.1f} cost units/yr ({abs(pct):.1f}%)</span>',
            unsafe_allow_html=True,
        )
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=x_labels,
            y=[cost_e, cost_h],
            error_y=dict(type='data', array=[cost_e_sd, cost_h_sd], visible=True),
            marker_color=["#4878d0", color_h],
            showlegend=False,
        ))
        fig4.add_annotation(
            x=x_labels[1],
            y=cost_h + cost_h_sd * 1.4,
            text=f"{sign}{abs(savings):.1f} ({abs(pct):.1f}%)",
            showarrow=False,
            font=dict(size=12, color=color_h),
        )
        fig4.update_layout(
            yaxis_title="Annual total cost C(γ) (cost units/yr)",
            height=360, margin=dict(t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True, key=f"totalcost_{mu_E}")

    # ── Numeric summary ────────────────────────────────────────────────────
    with st.expander("Numeric summary table"):
        summary = pd.DataFrame({
            "Metric": [
                "p_E — early abandon. fraction (%)",
                "p_L — late abandon. fraction (%)",
                f"Case losses — Early (52×λᴱ={ann_E:.0f}/yr)",
                f"Case losses — Late  (52×λᴸ={ann_L:.0f}/yr)",
                "Case losses — Total",
                "Case-loss cost (cost units/yr)",
                "Total cost C(γ) (cost units/yr)",
                "Avg queue wait — Early served (weeks)",
                "Avg slack — Early served (weeks)",
                "Nᴱ — reserved Early slots",
                "Nᴸ — reserved Late slots",
            ],
            "pooled-EDF": [
                f"{pE_e*100:.2f} ± {pE_e_sd*100:.2f}",
                f"{pL_e*100:.2f} ± {pL_e_sd*100:.2f}",
                f"{lE_e:.1f} ± {pE_e_sd*ann_E:.1f}",
                f"{lL_e:.1f} ± {pL_e_sd*ann_L:.1f}",
                f"{lE_e+lL_e:.1f}",
                f"{cost_e:.1f} ± {cost_e_sd:.1f}",
                f"{cost_e:.1f} ± {cost_e_sd:.1f}",
                f"{edf_data['stats'].loc['es_wait_active','mean']:.3f} ± {edf_data['stats'].loc['es_wait_active','std']:.3f}",
                f"{edf_data['stats'].loc['es_slack','mean']:.3f} ± {edf_data['stats'].loc['es_slack','std']:.3f}",
                "—",
                "—",
            ],
            lbl_h: [
                f"{pE_h*100:.2f} ± {pE_h_sd*100:.2f}",
                f"{pL_h*100:.2f} ± {pL_h_sd*100:.2f}",
                f"{lE_h:.1f} ± {pE_h_sd*ann_E:.1f}",
                f"{lL_h:.1f} ± {pL_h_sd*ann_L:.1f}",
                f"{lE_h+lL_h:.1f}",
                f"{aband_cost_h:.1f} ± {cost_h_sd:.1f}",
                f"{cost_h:.1f} ± {cost_h_sd:.1f}",
                f"{hyb_data['stats'].loc['es_wait_active','mean']:.3f} ± {hyb_data['stats'].loc['es_wait_active','std']:.3f}",
                f"{hyb_data['stats'].loc['es_slack','mean']:.3f} ± {hyb_data['stats'].loc['es_slack','std']:.3f}",
                f"{NE_star:.2f}",
                f"{NL_star:.2f}",
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# HTML REPORT EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def _generate_html_report() -> str:
    """
    Build a self-contained interactive HTML report from current session results.
    All Plotly figures are embedded with Plotly.js so the file works offline.
    """
    import plotly.io as pio

    results_list_ = st.session_state.get("tab1_results", [])
    edf_data_     = st.session_state.get("tab1_edf", {})
    lE_           = st.session_state.get("tab1_lambdaE", float(λE))
    lL_           = st.session_state.get("tab1_lambdaL", float(λL))
    gc_           = st.session_state.get("tab1_gamma_config", {})
    cE_           = gc_.get("cE", 1.0); cL_ = gc_.get("cL", 2.0); v_ = gc_.get("v", 0.0)
    Delta_        = gc_.get("Delta", 0.2)
    ann_E_, ann_L_ = _ann_headcounts(lE_, lL_)
    pb_           = st.session_state.get("tab1_params_base", {})
    nr_           = st.session_state.get("tab1_n_runs", 50)
    T_            = float(st.session_state.get("tab1_T_max", float(T_max)))
    T0_           = float(st.session_state.get("tab1_T0", float(T0_warmup)))
    _N_html       = int(pb_.get("N", 18))
    COLOR_E = "#4878d0"; COLOR_L = "#ee854a"
    _leg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

    parts = []
    first_fig = [True]

    def _fig_html(fig, responsive=True):
        inc = "cdn" if first_fig[0] else False
        first_fig[0] = False
        return pio.to_html(fig, full_html=False, include_plotlyjs=inc,
                           config={"responsive": responsive})

    def _scroll_fig_html(fig):
        """Wide multi-column figure wrapped in a horizontal-scroll div."""
        return f'<div style="overflow-x:auto;width:100%">{_fig_html(fig, responsive=False)}</div>'

    def _tbl(df: pd.DataFrame) -> str:
        return df.to_html(index=False, border=0,
                          classes="rtbl", escape=False)

    def _tbl_metrics(df: pd.DataFrame) -> str:
        """Like _tbl but preserves the index as a 'Metric' column."""
        df2 = df.reset_index()
        df2.rename(columns={df2.columns[0]: "Metric"}, inplace=True)
        return df2.to_html(index=False, border=0, classes="rtbl", escape=False)

    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Elective C-Section Simulation Report</title>
<style>
  body{font-family:Arial,sans-serif;padding:30px 40px;max-width:1400px;margin:auto;color:#222;}
  h1{color:#1b4f8a;} h2{color:#1b4f8a;border-bottom:2px solid #1b4f8a;padding-bottom:4px;}
  h3{color:#333;} .section{margin-bottom:40px;}
  .rtbl{border-collapse:collapse;width:100%;font-size:0.88em;}
  .rtbl th{background:#1b4f8a;color:#fff;padding:6px 10px;text-align:left;}
  .rtbl td{padding:5px 10px;border-bottom:1px solid #ddd;}
  .rtbl tr:hover td{background:#f0f4ff;}
  .badge{display:inline-block;background:#e8f4e8;border-left:4px solid #2e7d32;
         padding:6px 14px;border-radius:4px;margin:6px 0;font-weight:bold;}
  .row2{display:flex;gap:20px;flex-wrap:wrap;}
  .row2>div{flex:1;min-width:660px;}
</style>
</head><body>
""")
    parts.append(f"<h1>Elective C-Section Simulation Report</h1>")
    parts.append(f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
                 f"Replications: {nr_} &nbsp;|&nbsp; T={T_:.0f} wks, T₀={T0_:.0f} wks &nbsp;|&nbsp; "
                 f"λᴱ={lE_:.3f}, λᴸ={lL_:.3f}, N={pb_.get('N','?')} &nbsp;|&nbsp; "
                 f"cᴱ={cE_:.2f}, cᴸ={cL_:.2f}, v={v_:.3f} &nbsp;|&nbsp; "
                 f"Equity tolerance Δ={Delta_*100:.1f}%</p>")

    # ── Per-μᴱ comparison sections ─────────────────────────────────────────
    x_lbl = ["pooled-EDF"]
    for res in results_list_:
        x_lbl.append(f"dedicated-EDF (μᴱ={res['mu_E']:.0f})")

    for res in results_list_:
        mu_E_ = res["mu_E"]; gstar_ = res["gamma_star"]
        hyb_  = res["hyb"]
        lbl_h = f"Optimal dedicated-EDF (μᴱ={mu_E_:.0f}, γ*={gstar_:.3f})"
        _NE_html_r = gstar_ * _N_html; _NL_html_r = _N_html - _NE_html_r
        parts.append(f'<div class="section"><h2>pooled-EDF vs {lbl_h}</h2>')
        parts.append(f'<div class="badge">{lbl_h} — γ* = {gstar_:.3f} &nbsp;|&nbsp; '
                     f'N<sup>E</sup> = {_NE_html_r:.2f} &nbsp; N<sup>L</sup> = {_NL_html_r:.2f}</div>')

        pE_e, pL_e, pE_e_sd, pL_e_sd, lE_e, lL_e, cost_e, cost_e_sd = \
            _policy_metrics(edf_data_, ann_E_, ann_L_, cE_, cL_)
        _slot_rw_html = v_ * _NE_html_r * 52
        pE_h, pL_h, pE_h_sd, pL_h_sd, lE_h, lL_h, cost_h, cost_h_sd = \
            _policy_metrics(hyb_, ann_E_, ann_L_, cE_, cL_, slot_reward=_slot_rw_html)
        x2 = ["pooled-EDF", f"dedicated-EDF (μᴱ={mu_E_:.0f})"]

        fig_frac = go.Figure()
        fig_frac.add_trace(go.Bar(name="Early engagers", x=x2,
                                  y=[pE_e*100, pE_h*100],
                                  error_y=dict(type='data', array=[pE_e_sd*100, pE_h_sd*100], visible=True),
                                  marker_color=COLOR_E))
        fig_frac.add_trace(go.Bar(name="Late engagers", x=x2,
                                  y=[pL_e*100, pL_h*100],
                                  error_y=dict(type='data', array=[pL_e_sd*100, pL_h_sd*100], visible=True),
                                  marker_color=COLOR_L))
        fig_frac.update_layout(barmode="group", yaxis_title="Case-loss fraction (%)",
                               title="Case-loss fractions", height=360, width=640, legend=_leg)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(name="Early engagers", x=x2,
                                  y=[lE_e, lE_h],
                                  error_y=dict(type='data', array=[pE_e_sd*ann_E_, pE_h_sd*ann_E_], visible=True),
                                  marker_color=COLOR_E))
        fig_loss.add_trace(go.Bar(name="Late engagers", x=x2,
                                  y=[lL_e, lL_h],
                                  error_y=dict(type='data', array=[pL_e_sd*ann_L_, pL_h_sd*ann_L_], visible=True),
                                  marker_color=COLOR_L))
        fig_loss.update_layout(barmode="group",
                               yaxis_title="Estimated annual case losses",
                               title="Estimated annual case losses", height=360, width=640, legend=_leg)

        aband_cost_h_ = cE_ * lE_h + cL_ * lL_h  # Hybrid case-loss cost (no slot reward)
        savings = cost_e - cost_h
        pct     = 100*savings/cost_e if cost_e > 0 else 0.0
        sav_sd  = math.sqrt(cost_e_sd**2 + cost_h_sd**2)
        color_s = "#2e7d32" if savings >= 0 else "#c62828"
        sign_s  = "↓" if savings >= 0 else "↑"
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(name="Early component", x=x2,
                                  y=[cE_*lE_e, cE_*lE_h],
                                  error_y=dict(type='data', array=[cE_*pE_e_sd*ann_E_, cE_*pE_h_sd*ann_E_], visible=True),
                                  marker_color=COLOR_E))
        fig_cost.add_trace(go.Bar(name="Late component", x=x2,
                                  y=[cL_*lL_e, cL_*lL_h],
                                  error_y=dict(type='data', array=[cL_*pL_e_sd*ann_L_, cL_*pL_h_sd*ann_L_], visible=True),
                                  marker_color=COLOR_L))
        fig_cost.update_layout(barmode="stack", yaxis_title="Annual case-loss cost (cost units/yr)",
                               title=f"Total case-loss cost  (cᴱ={cE_:.2f}, cᴸ={cL_:.2f})",
                               height=360, width=640, legend=_leg)

        fig_sav = go.Figure()
        fig_sav.add_trace(go.Bar(x=x2, y=[cost_e, cost_h],
                                 error_y=dict(type='data', array=[cost_e_sd, cost_h_sd], visible=True),
                                 marker_color=[COLOR_E, color_s],
                                 showlegend=False))
        fig_sav.add_annotation(x=x2[1], y=cost_h + cost_h_sd * 1.4,
                               text=f"{sign_s}{abs(savings):.1f} cost units/yr ({abs(pct):.1f}%)",
                               showarrow=False, font=dict(size=12, color=color_s))
        fig_sav.update_layout(yaxis_title="Annual total cost C(γ) (cost units/yr)", showlegend=False,
                              title=f"Total cost C(γ): EDF vs dedicated-EDF (v={v_:.3f})",
                              height=360, width=640)

        parts.append('<div class="row2">')
        parts.append(f'<div>{_fig_html(fig_frac, responsive=False)}</div>')
        parts.append(f'<div>{_fig_html(fig_loss, responsive=False)}</div>')
        parts.append('</div><div class="row2">')
        parts.append(f'<div>{_fig_html(fig_cost, responsive=False)}</div>')
        parts.append(f'<div>{_fig_html(fig_sav, responsive=False)}</div>')
        parts.append('</div></div>')

    # ── Summary section ─────────────────────────────────────────────────────
    if results_list_:
        parts.append('<div class="section"><h2>Summary — All Policies</h2>')
        all_pol = [("pooled-EDF", edf_data_)] + \
                  [(f"dedicated-EDF (μᴱ={r['mu_E']:.0f}, γ*={r['gamma_star']:.3f})", r["hyb"])
                   for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        mu_labels_  = [p for p, _ in all_pol]
        fig_labels_ = ["pooled-EDF"] + [
            f"dedicated-EDF (μᴱ={r['mu_E']:.0f})"
            for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        metrics_   = []
        for pname, data in all_pol:
            if "dedicated-EDF" in pname:
                _gstar_html = data.get("gamma_star", 0.0)
                _NE_s = _gstar_html * _N_html; _NL_s = _N_html - _NE_s
                _slot_rw_s  = v_ * _NE_s * 52
            else:
                _slot_rw_s = 0.0; _NE_s = None; _NL_s = None
            pE, pL, pE_sd, pL_sd, lossE, lossL, cost, cost_sd = \
                _policy_metrics(data, ann_E_, ann_L_, cE_, cL_, slot_reward=_slot_rw_s)
            aband_cost_ = cE_ * lossE + cL_ * lossL  # pure case-loss cost (no slot reward)
            _st = data.get("stats")
            _ewa = float(_st.loc["es_wait_active", "mean"]) if (_st is not None and "es_wait_active" in _st.index) else float("nan")
            _ewa_sd = float(_st.loc["es_wait_active", "std"]) if (_st is not None and "es_wait_active" in _st.index) else float("nan")
            _esl = float(_st.loc["es_slack", "mean"]) if (_st is not None and "es_slack" in _st.index) else float("nan")
            _esl_sd = float(_st.loc["es_slack", "std"]) if (_st is not None and "es_slack" in _st.index) else float("nan")
            metrics_.append(dict(name=pname, pE=pE, pL=pL, pE_sd=pE_sd, pL_sd=pL_sd,
                                 lossE=lossE, lossL=lossL, cost=cost, cost_sd=cost_sd,
                                 aband_cost=aband_cost_, NE=_NE_s, NL=_NL_s,
                                 es_wait_active=_ewa, es_wait_active_sd=_ewa_sd,
                                 es_slack=_esl, es_slack_sd=_esl_sd))

        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Bar(name="Early", x=fig_labels_,
                                y=[m["pE"]*100 for m in metrics_],
                                error_y=dict(type='data', array=[m["pE_sd"]*100 for m in metrics_], visible=True),
                                marker_color=COLOR_E))
        fig_s1.add_trace(go.Bar(name="Late", x=fig_labels_,
                                y=[m["pL"]*100 for m in metrics_],
                                error_y=dict(type='data', array=[m["pL_sd"]*100 for m in metrics_], visible=True),
                                marker_color=COLOR_L))
        fig_s1.update_layout(barmode="group", yaxis_title="Case-loss fraction (%)",
                             title="Case-loss fractions — all policies", height=400, legend=_leg)

        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Bar(name="Early", x=fig_labels_,
                                y=[m["lossE"] for m in metrics_],
                                error_y=dict(type='data', array=[m["pE_sd"]*ann_E_ for m in metrics_], visible=True),
                                marker_color=COLOR_E))
        fig_s2.add_trace(go.Bar(name="Late", x=fig_labels_,
                                y=[m["lossL"] for m in metrics_],
                                error_y=dict(type='data', array=[m["pL_sd"]*ann_L_ for m in metrics_], visible=True),
                                marker_color=COLOR_L))
        fig_s2.update_layout(barmode="group",
                             yaxis_title="Estimated annual case losses",
                             title="Annual case losses — all policies", height=400, legend=_leg)

        fig_s3 = go.Figure()
        fig_s3.add_trace(go.Bar(name="Early component", x=fig_labels_,
                                y=[cE_*m["lossE"] for m in metrics_], marker_color=COLOR_E))
        fig_s3.add_trace(go.Bar(name="Late component", x=fig_labels_,
                                y=[cL_*m["lossL"] for m in metrics_], marker_color=COLOR_L))
        fig_s3.update_layout(barmode="stack", yaxis_title="Annual case-loss cost (cost units/yr)",
                             title=f"Total case-loss cost — all policies (cᴱ={cE_:.2f}, cᴸ={cL_:.2f})",
                             height=400, legend=_leg)

        fig_s4 = go.Figure()
        fig_s4.add_trace(go.Bar(x=fig_labels_,
                                y=[m["cost"] for m in metrics_],
                                error_y=dict(type='data', array=[m["cost_sd"] for m in metrics_], visible=True),
                                marker_color=[COLOR_E] + [COLOR_L]*(len(metrics_)-1),
                                showlegend=False))
        fig_s4.update_layout(yaxis_title="Annual total cost C(γ) (cost units/yr)",
                             title=f"Total cost C(γ) — all policies (v={v_:.3f})",
                             height=400)

        parts.append(_fig_html(fig_s1))
        parts.append(_fig_html(fig_s2))
        parts.append(_fig_html(fig_s3))
        parts.append(_fig_html(fig_s4))

        # Summary table
        edf_cost_base_ = _policy_metrics(edf_data_, ann_E_, ann_L_, cE_, cL_)[6]
        rows_sum_ = []
        for m in metrics_:
            saving = (f"{edf_cost_base_ - m['cost']:+.1f}"
                      if "dedicated-EDF" in m["name"] else "—")
            _ewa_str = f"{m['es_wait_active']:.3f} ± {m['es_wait_active_sd']:.3f}" if not math.isnan(m['es_wait_active']) else "—"
            _esl_str = f"{m['es_slack']:.3f} ± {m['es_slack_sd']:.3f}" if not math.isnan(m['es_slack']) else "—"
            rows_sum_.append({
                "Policy": m["name"],
                "p_E (%)": f"{m['pE']*100:.2f} ± {m['pE_sd']*100:.2f}",
                "p_L (%)": f"{m['pL']*100:.2f} ± {m['pL_sd']*100:.2f}",
                f"Loss Early (52×λᴱ={ann_E_:.0f}/yr)": f"{m['lossE']:.1f}",
                f"Loss Late (52×λᴸ={ann_L_:.0f}/yr)": f"{m['lossL']:.1f}",
                "Nᴱ": f"{m['NE']:.2f}" if m["NE"] is not None else "—",
                "Nᴸ": f"{m['NL']:.2f}" if m["NL"] is not None else "—",
                "Avg queue wait — Early served (wks)": _ewa_str,
                "Avg slack — Early served (wks)": _esl_str,
                "Case-loss cost (cost units/yr)": f"{m['aband_cost']:.1f} ± {m['cost_sd']:.1f}",
                "Total cost C(γ) (cost units/yr)": f"{m['cost']:.1f} ± {m['cost_sd']:.1f}",
                "Cost saving vs EDF": saving,
            })
        parts.append("<h3>Summary table</h3>")
        parts.append(_tbl(pd.DataFrame(rows_sum_)))
        parts.append('</div>')

    # ── Detailed Results — All Policies ─────────────────────────────────────
    if results_list_:
        all_pol_det = [("pooled-EDF", edf_data_)] + \
                      [(f"Optimal dedicated-EDF (μᴱ={r['mu_E']:.0f}, γ*={r['gamma_star']:.3f})", r["hyb"])
                       for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        results_det = {pname: data["stats"]
                       for pname, data in all_pol_det if data}
        policy_names_det = list(results_det.keys())

        parts.append('<div class="section"><h2>Detailed Results — All Policies</h2>')

        # Primary metrics table
        def _fmt_pct_h(m, s): return "—" if pd.isna(m) else f"{m:.2%} ± {s:.2%}"
        def _ctable_h(keys, labels, fmt_fn):
            rows = {}
            for k in keys:
                rows[labels[k]] = {
                    pn: fmt_fn(results_det[pn].loc[k, "mean"], results_det[pn].loc[k, "std"])
                    for pn in policy_names_det if k in results_det[pn].index}
            return pd.DataFrame(rows).T

        parts.append(f"<h3>Primary metrics (mean ± SD over {nr_} runs)</h3>")
        parts.append(_tbl_metrics(_ctable_h(
            ["service_rate", "abandon_rate", "util_1"],
            {"service_rate": "Service rate", "abandon_rate": "Case-loss rate", "util_1": "Utilisation"},
            _fmt_pct_h)))

        parts.append('</div>')

    parts.append("</body></html>")
    return "".join(parts)


st.header("Policy Simulation")
st.markdown(
    "This tool compares two approaches to scheduling planned (elective) C-sections. "
    "The **standard approach** (pooled-EDF) allocates all operating slots to a shared waiting list, "
    "giving priority to the patient whose planned delivery date is soonest. "
    "The **reservation approach** (Optimal dedicated-EDF) sets aside a dedicated fraction of slots for women who book early."
)

# ── Save / Load session ───────────────────────────────────────────────────
with st.expander("💾  Save / Load session", expanded=False):
    st.caption(
        "Sessions are saved as ZIP files to the **saved sessions/** folder inside the app folder."
    )
    sv_col, ld_col = st.columns(2)
    with sv_col:
        st.markdown("**Save** current results")
        if st.session_state.get("tab1_results"):
            _n = len(st.session_state.get("tab1_results", []))
            _sname = st.text_input(
                "Session name", value="session_1",
                key="save_session_name",
                placeholder="Enter a name (without extension)")
            if st.button(f"💾 Save session ({_n} μᴱ value(s))", key="save_session_btn"):
                _sname_clean = _sname.strip().replace("/", "_").replace("\\", "_") or "session"
                _spath = os.path.join(SESSIONS_DIR, f"{_sname_clean}.zip")
                with open(_spath, "wb") as _f:
                    _f.write(_session_to_zip())
                st.success(f"Saved → `saved sessions/{_sname_clean}.zip`")
        else:
            st.info("No results to save yet — run a simulation first.")
    with ld_col:
        st.markdown("**Load** a saved session")
        _existing = sorted(
            f for f in os.listdir(SESSIONS_DIR) if f.endswith(".zip"))
        if _existing:
            _sel = st.selectbox(
                "Choose session", _existing, key="load_session_sel",
                label_visibility="collapsed")
            if st.button("↑ Load session", key="restore_btn"):
                with open(os.path.join(SESSIONS_DIR, _sel), "rb") as _f:
                    _loaded = _zip_to_session(_f.read())
                for _k, _v in _loaded.items():
                    st.session_state[_k] = _v
                _n = len(_loaded.get("tab1_results", []))
                st.success(f"Restored — {_n} μᴱ value(s) loaded.")
                st.rerun()
        else:
            st.info("No saved sessions found.")

results_exist = bool(st.session_state.get("tab1_results"))

with st.container(border=True):
    st.markdown("**Demand & Slots**")
    _d_left, _d_right = st.columns(2)
    with _d_left:
        λE = st.number_input("Demand for early engagers (λᴱ)", 0.0, 500.0,
                             step=0.001, format="%.3f",
                             key="sb_lambdaE", disabled=results_exist)
        λL = st.number_input("Demand for late engagers (λᴸ)",  0.0, 500.0,
                             step=0.001, format="%.3f",
                             key="sb_lambdaL", disabled=results_exist)
    with _d_right:
        N_servers = st.number_input("Slots N", 1, 500, step=1,
                                    key="sb_N", disabled=results_exist)

params_base = dict(
    T_max=float(T_max), warmup=float(T0_warmup),
    p_ns=float(p_ns),
    λE=float(λE), λL=float(λL), N=int(N_servers),
    early_F=early_F, early_params=dict(early_params),
    early_min=float(early_min), early_max=float(early_max),
    late_F=late_F, late_params=dict(late_params),
    late_min=float(late_min), late_max=float(late_max),
    svc_E=dict(svc_E), svc_L=dict(svc_L),
)

# ── Buttons ───────────────────────────────────────────────────────────────
run_clicked = False

btn_col, reset_col = st.columns([5, 1])
with btn_col:
    if not results_exist and means_list_t1:
        _lbl = (f"▶  Run pooled-EDF + Optimal dedicated-EDF (μᴱ = {means_list_t1[0]:.0f})"
                if len(means_list_t1) == 1
                else f"▶  Run pooled-EDF + Optimal dedicated-EDF ({len(means_list_t1)} μᴱ values)")
        run_clicked = st.button(_lbl, key="t1_run", type="primary")
    elif results_exist:
        st.success(f"✓ Run complete ({len(st.session_state.get('tab1_results', []))} μᴱ value(s)).")

with reset_col:
    if results_exist:
        if st.button("↺ Reset", key="t1_reset"):
            for k in list(st.session_state.keys()):
                if k.startswith("tab1_"): del st.session_state[k]
            st.session_state.pop("_html_cache", None)
            st.session_state.pop("_html_cache_sig", None)
            st.rerun()

def _save_config():
    st.session_state["tab1_means_config"] = means_list_t1
    st.session_state["tab1_gamma_config"] = dict(
        gmin=t1_gmin, gmax=t1_gmax, gstep=float(t1_gstep),
        cE=float(t1_cE), cL=float(t1_cL), v=float(t1_v), Delta=float(t1_Delta_pct) / 100.0)
    st.session_state["tab1_params_base"] = dict(params_base)
    st.session_state["tab1_n_runs"]  = int(t1_n_runs)
    st.session_state["tab1_seed0"]   = int(seed0)
    st.session_state["tab1_T_max"]   = float(T_max)
    st.session_state["tab1_T0"]      = float(T0_warmup)
    st.session_state["tab1_lambdaE"] = float(λE)
    st.session_state["tab1_lambdaL"] = float(λL)
    st.session_state["tab1_results"]  = []
    # tab1_edf and tab1_baseline_pb_trace are set by _run_edf_and_first_hybrid

# ── Handle run ────────────────────────────────────────────────────────────
if run_clicked and means_list_t1:
    _save_config()
    gc      = st.session_state["tab1_gamma_config"]
    n_total = len(means_list_t1)
    for i, mu in enumerate(means_list_t1):
        lbl = f"[{i+1}/{n_total}]"
        if i == 0:
            with st.spinner(f"Running pooled-EDF + Optimal dedicated-EDF (μᴱ = {mu:.0f})  {lbl}…"):
                _run_edf_and_first_hybrid(mu, gc, lbl)
        else:
            with st.spinner(f"Running Optimal dedicated-EDF  (μᴱ = {mu:.0f})  {lbl}…"):
                _run_hybrid_only(mu, gc, lbl)
    st.rerun()

# ── Display accumulated results ───────────────────────────────────────────
if results_exist:
    results_list = st.session_state["tab1_results"]
    edf_base = st.session_state["tab1_edf"]   # single EDF baseline
    lE_s  = st.session_state.get("tab1_lambdaE", float(λE))
    lL_s  = st.session_state.get("tab1_lambdaL", float(λL))
    gc_s  = st.session_state.get("tab1_gamma_config", {})
    cE_s  = gc_s.get("cE", 1.0); cL_s = gc_s.get("cL", 2.0); v_s = gc_s.get("v", 0.0)
    ann_E, ann_L = _ann_headcounts(lE_s, lL_s)
    _pb_s  = st.session_state.get("tab1_params_base", {})
    _N_s   = int(_pb_s.get("N", 18))
    _T_s   = float(st.session_state.get("tab1_T_max",  T_max))
    _T0_s  = float(st.session_state.get("tab1_T0",     T0_warmup))
    nr_s   = st.session_state.get("tab1_n_runs", 50)

    for i, res in enumerate(results_list):
        is_latest = (i == len(results_list) - 1)
        _gs_exp = res['gamma_star']
        _NE_exp = _gs_exp * _N_s; _NL_exp = _N_s - _NE_exp
        with st.expander(
            f"pooled-EDF  vs  Optimal dedicated-EDF  (μᴱ = {res['mu_E']:.0f})"
            f"  with  γ* = {_gs_exp:.3f},  Nᴱ = {_NE_exp:.2f},  Nᴸ = {_NL_exp:.2f}",
            expanded=is_latest
        ):
            _comparison_charts(edf_base, res["hyb"], lE_s, lL_s, cE_s, cL_s,
                               v=v_s, N=_N_s, T=_T_s, T0=_T0_s)

    st.divider()

    # ── Comparative Overview — All Policies ───────────────────────────────
    all_policies = [("pooled-EDF", edf_base)]
    for res in sorted(results_list, key=lambda r: r["mu_E"]):
        all_policies.append(
            (f"dedicated-EDF (μᴱ={res['mu_E']:.0f}, γ*={res['gamma_star']:.3f})", res["hyb"]))
    fig_labels = ["pooled-EDF"] + [
        f"dedicated-EDF (μᴱ={res['mu_E']:.0f})"
        for res in sorted(results_list, key=lambda r: r["mu_E"])]

    metrics = []
    for pname, data in all_policies:
        if "dedicated-EDF" in pname:
            _gstar = data.get("gamma_star", 0.0)
            _NE = _gstar * _N_s; _NL = _N_s - _NE
            _slot_rw = v_s * _NE * 52
        else:
            _slot_rw = 0.0; _NE = None; _NL = None
        pE, pL, pE_sd, pL_sd, lossE, lossL, cost, cost_sd = _policy_metrics(
            data, ann_E, ann_L, cE_s, cL_s, slot_reward=_slot_rw)
        aband_cost = cE_s * lossE + cL_s * lossL  # pure case-loss cost (no slot reward)
        _st = data.get("stats")
        _ewa = float(_st.loc["es_wait_active", "mean"]) if (_st is not None and "es_wait_active" in _st.index) else float("nan")
        _ewa_sd = float(_st.loc["es_wait_active", "std"]) if (_st is not None and "es_wait_active" in _st.index) else float("nan")
        _esl = float(_st.loc["es_slack", "mean"]) if (_st is not None and "es_slack" in _st.index) else float("nan")
        _esl_sd = float(_st.loc["es_slack", "std"]) if (_st is not None and "es_slack" in _st.index) else float("nan")
        metrics.append(dict(
            name=pname, pE=pE, pL=pL, pE_sd=pE_sd, pL_sd=pL_sd,
            lossE=lossE, lossL=lossL, cost=cost, cost_sd=cost_sd,
            aband_cost=aband_cost, NE=_NE, NL=_NL,
            es_wait_active=_ewa, es_wait_active_sd=_ewa_sd,
            es_slack=_esl, es_slack_sd=_esl_sd))

    st.subheader("Comparative Overview — All Policies")

    _sum_leg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10))

    # ── Plot 1: Case-loss fractions ──────────────────────────────────────
    st.markdown("**Plot 1 — Case-loss fractions (%)**")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="Early engagers", x=fig_labels,
        y=[m["pE"]*100 for m in metrics],
        error_y=dict(type='data', array=[m["pE_sd"]*100 for m in metrics], visible=True),
        marker_color="#4878d0"))
    fig1.add_trace(go.Bar(
        name="Late engagers", x=fig_labels,
        y=[m["pL"]*100 for m in metrics],
        error_y=dict(type='data', array=[m["pL_sd"]*100 for m in metrics], visible=True),
        marker_color="#ee854a"))
    fig1.update_layout(barmode="group", yaxis_title="Case-loss fraction (%)",
                       xaxis_title="Policy", height=380,
                       margin=dict(t=10,b=30), legend=_sum_leg)
    st.plotly_chart(fig1, use_container_width=True, key="sum_frac")

    # ── Plot 2: Case losses ──────────────────────────────────────────────
    st.markdown("**Plot 2 — Estimated annual case losses**")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Early engagers", x=fig_labels,
        y=[m["lossE"] for m in metrics],
        error_y=dict(type='data', array=[m["pE_sd"]*ann_E for m in metrics], visible=True),
        marker_color="#4878d0"))
    fig2.add_trace(go.Bar(
        name="Late engagers", x=fig_labels,
        y=[m["lossL"] for m in metrics],
        error_y=dict(type='data', array=[m["pL_sd"]*ann_L for m in metrics], visible=True),
        marker_color="#ee854a"))
    fig2.update_layout(barmode="group",
                       yaxis_title="Estimated annual case losses",
                       xaxis_title="Policy", height=380,
                       margin=dict(t=10,b=30), legend=_sum_leg)
    st.plotly_chart(fig2, use_container_width=True, key="sum_loss")

    # ── Plot 3: Total case-loss cost (stacked by component) ──────────────
    st.markdown(f"**Plot 3 — Total case-loss cost  (cᴱ={cE_s:.2f}, cᴸ={cL_s:.2f})**")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Early component", x=fig_labels,
        y=[cE_s*m["lossE"] for m in metrics],
        marker_color="#4878d0"))
    fig3.add_trace(go.Bar(
        name="Late component", x=fig_labels,
        y=[cL_s*m["lossL"] for m in metrics],
        marker_color="#ee854a"))
    fig3.update_layout(
        barmode="stack",
        yaxis_title="Annual case-loss cost (cost units/yr)",
        xaxis_title="Policy", height=380,
        margin=dict(t=10,b=30), legend=_sum_leg)
    st.plotly_chart(fig3, use_container_width=True, key="sum_cost")

    # ── Plot 4: Total cost C(γ) per policy ───────────────────────────────
    st.markdown(f"**Plot 4 — Total cost C(γ)  (v={v_s:.3f})**")
    _edf_cost = next(m["cost"] for m in metrics if "dedicated-EDF" not in m["name"])
    fig4_sum = go.Figure()
    fig4_sum.add_trace(go.Bar(
        x=fig_labels,
        y=[m["cost"] for m in metrics],
        error_y=dict(type='data', array=[m["cost_sd"] for m in metrics], visible=True),
        marker_color=["#4878d0" if "dedicated-EDF" not in m["name"] else "#2e7d32" for m in metrics],
        showlegend=False,
    ))
    for lbl, m in zip(fig_labels, metrics):
        if "dedicated-EDF" not in m["name"]:
            continue
        sav = _edf_cost - m["cost"]
        pct = 100 * sav / _edf_cost if _edf_cost > 0 else 0.0
        sign = "↓" if sav >= 0 else "↑"
        col  = "#2e7d32" if sav >= 0 else "#c62828"
        fig4_sum.add_annotation(
            x=lbl,
            y=m["cost"] + m["cost_sd"] * 1.4,
            text=f"{sign}{abs(sav):.1f} ({abs(pct):.1f}%)",
            showarrow=False,
            font=dict(size=10, color=col),
        )
    fig4_sum.update_layout(
        yaxis_title="Annual total cost C(γ) (cost units/yr)",
        xaxis_title="Policy", height=400,
        margin=dict(t=30, b=30))
    st.plotly_chart(fig4_sum, use_container_width=True, key="sum_totalcost")

    # ── Summary table ────────────────────────────────────────────────────
    with st.expander("Summary table"):
        edf_cost_base = _policy_metrics(edf_base, ann_E, ann_L, cE_s, cL_s)[6]
        rows_sum = []
        for m in metrics:
            saving = (f"{edf_cost_base - m['cost']:+.1f}"
                      if "dedicated-EDF" in m["name"] else "—")
            ne_str = f"{m['NE']:.2f}" if m["NE"] is not None else "—"
            nl_str = f"{m['NL']:.2f}" if m["NL"] is not None else "—"
            _ewa_str = f"{m['es_wait_active']:.3f} ± {m['es_wait_active_sd']:.3f}" if not math.isnan(m['es_wait_active']) else "—"
            _esl_str = f"{m['es_slack']:.3f} ± {m['es_slack_sd']:.3f}" if not math.isnan(m['es_slack']) else "—"
            rows_sum.append({
                "Policy": m["name"],
                "p_E (%)": f"{m['pE']*100:.2f} ± {m['pE_sd']*100:.2f}",
                "p_L (%)": f"{m['pL']*100:.2f} ± {m['pL_sd']*100:.2f}",
                f"Case losses — Early (52×λᴱ={ann_E:.0f}/yr)": f"{m['lossE']:.1f}",
                f"Case losses — Late  (52×λᴸ={ann_L:.0f}/yr)": f"{m['lossL']:.1f}",
                "Nᴱ": ne_str,
                "Nᴸ": nl_str,
                "Avg queue wait — Early served (wks)": _ewa_str,
                "Avg slack — Early served (wks)": _esl_str,
                "Case-loss cost (cost units/yr)": f"{m['aband_cost']:.1f} ± {m['cost_sd']:.1f}",
                "Total cost C(γ) (cost units/yr)": f"{m['cost']:.1f} ± {m['cost_sd']:.1f}",
                "Cost saving vs pooled-EDF": saving,
            })
        st.dataframe(pd.DataFrame(rows_sum), use_container_width=True, hide_index=True)

    # ── Primary metrics (mean ± SD) ──────────────────────────────────────
    st.subheader(f"Primary metrics  (mean ± SD over {nr_s} runs)")
    def _fmt_pct(m, s): return "—" if pd.isna(m) else f"{m:.2%} ± {s:.2%}"
    _pm_keys = ["service_rate", "abandon_rate", "util_1"]
    _pm_labels = {"service_rate": "Service rate",
                  "abandon_rate": "Case-loss rate",
                  "util_1":       "Utilisation"}
    _pm_rows = {}
    for _k in _pm_keys:
        _pm_rows[_pm_labels[_k]] = {}
        for _pname, _data in all_policies:
            if not _data: continue
            _s = _data.get("stats")
            if _s is not None and _k in _s.index:
                _pm_rows[_pm_labels[_k]][_pname] = _fmt_pct(_s.loc[_k, "mean"], _s.loc[_k, "std"])
    st.dataframe(pd.DataFrame(_pm_rows).T, use_container_width=True)

    st.divider()
    # Cache HTML so it is generated only once per result set, not on every render.
    _html_sig = str(len(results_list)) + "_" + str([r["mu_E"] for r in results_list])
    if st.session_state.get("_html_cache_sig") != _html_sig:
        st.session_state["_html_cache"] = _generate_html_report().encode("utf-8")
        st.session_state["_html_cache_sig"] = _html_sig
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇  Export interactive HTML report",
        data=st.session_state["_html_cache"],
        file_name=f"simulation_report_{_ts}.html",
        mime="text/html",
        key="t1_export_html",
    )

