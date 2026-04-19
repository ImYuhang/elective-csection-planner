"""
app.py — Elective C-Section Planning Interface (version 5 Mar 23, 2026)

Tab 1 : Sequential policy simulation
          EDF (Pooled) vs Optimal Hybrid-EDF(μᴱ) for each specified μᴱ.
          Three metrics per comparison: abandonment fractions, case losses,
          total abandonment cost (+ cost-savings delta).
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
      edf_baseline/         — EDF pooled policy files
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
    ("t1_means_text", "7, 14, 21, 28, 35, 70"), ("t1_n_runs", 50),
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

_sb_header("Demand & Slots")
λE        = st.sidebar.number_input("Demand for early engagers (λᴱ)", 0.0, 500.0, step=0.001, format="%.3f", key="sb_lambdaE")
λL        = st.sidebar.number_input("Demand for late engagers (λᴸ)",  0.0, 500.0, step=0.001, format="%.3f", key="sb_lambdaL")
N_servers = st.sidebar.number_input("Slots N", 1, 500, step=1, key="sb_N")

_sb_header("Cost & Equity")
t1_cE    = st.sidebar.number_input("Early abandon cost (cᴱ)",      min_value=0.0, step=0.1,  format="%.3f", key="t1_cE",    disabled=_tab1_locked)
t1_cL    = st.sidebar.number_input("Late abandon cost (cᴸ)",       min_value=0.0, step=0.1,  format="%.3f", key="t1_cL",    disabled=_tab1_locked)
t1_v     = st.sidebar.number_input("Reserved-slot value (v)",       min_value=0.0, step=0.01, format="%.3f", key="t1_v",     disabled=_tab1_locked)
t1_Delta_pct = st.sidebar.number_input("Equity tolerance Δ (%)",    min_value=0.0, step=1.0,  format="%.1f", key="t1_Delta_pct", disabled=_tab1_locked)

_sb_header("Baseline Lead Time")
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
t1_gmin = st.sidebar.number_input("γ min", 0.0, 1.0, step=0.05, key="t1_gmin",
                                   disabled=_tab1_locked)
t1_gmax = st.sidebar.number_input("γ max", 0.0, 1.0, step=0.05, key="t1_gmax",
                                   disabled=_tab1_locked)

_sb_header("Experiment Settings")
T_max     = st.sidebar.number_input("Total time T (weeks)", 1, 10000, step=52, key="sb_T_max")
T0_warmup = st.sidebar.number_input("Warm-up period T₀ (weeks)", 0, int(T_max), step=10, key="sb_T0")
p_ns   = st.sidebar.slider("No-show probability p", 0.0, 1.0, step=0.01, key="sb_p_ns")
seed0  = st.sidebar.number_input("Base random seed", 0, 10_000_000, step=1, key="sb_seed0")

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
    n_workers = min(len(gammas), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for g, costs, phis, runs, dfs, hists in pool.map(_run_one_gamma_worker, tasks):
            mean_C   = float(np.mean(costs)) if costs else float("nan")
            mean_phi = float(np.mean(phis))  if phis  else float("nan")
            ineq     = max(0.0, abs(mean_phi) - Delta)
            stats_h  = pd.DataFrame(runs).agg(["mean", "std"]).T
            gamma_summary[float(g)] = dict(
                C=mean_C, phi=mean_phi, ineq=ineq,
                stats=stats_h, df_all=pd.concat(dfs, ignore_index=True), hists=hists)

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
    2. Run EDF pooled on each → save tab1_edf and tab1_saved_traces.
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
    slot_reward: pre-computed v×Nᴱ×52 (display) or v×Nᴱ×(T−T₀) (search), subtracted from cost for Hybrid-EDF only.
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
      Row 1 — abandonment fractions | case losses
      Row 2 — total abandonment cost (absolute) | cost savings vs EDF (delta)
    """
    mu_E   = hyb_data["mu_E"]
    gstar  = hyb_data["gamma_star"]
    NE_star = gstar * N
    NL_star = N - NE_star
    lbl_h  = f"Optimal Hybrid-EDF (μᴱ={mu_E:.0f}, γ*={gstar:.3f})"

    ann_E, ann_L = _ann_headcounts(lE, lL)
    pE_e, pL_e, pE_e_sd, pL_e_sd, lE_e, lL_e, cost_e, cost_e_sd = _policy_metrics(edf_data, ann_E, ann_L, cE, cL)
    _slot_reward_h = v * NE_star * 52
    pE_h, pL_h, pE_h_sd, pL_h_sd, lE_h, lL_h, cost_h, cost_h_sd = _policy_metrics(hyb_data, ann_E, ann_L, cE, cL, slot_reward=_slot_reward_h)
    aband_cost_h = cE * lE_h + cL * lL_h  # Hybrid abandonment cost (no slot reward)

    # Short x-axis labels (policies); cohort = bar colour
    x_labels = ["EDF (Pooled)", f"Hybrid (μᴱ={mu_E:.0f})"]
    COLOR_E = "#4878d0"
    COLOR_L = "#ee854a"
    _leg = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)

    # ── Row 1 ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Abandonment fractions**")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Early engagers", x=x_labels,
                             y=[pE_e*100, pE_h*100],
                             error_y=dict(type='data', array=[pE_e_sd*100, pE_h_sd*100], visible=True),
                             marker_color=COLOR_E))
        fig.add_trace(go.Bar(name="Late engagers", x=x_labels,
                             y=[pL_e*100, pL_h*100],
                             error_y=dict(type='data', array=[pL_e_sd*100, pL_h_sd*100], visible=True),
                             marker_color=COLOR_L))
        fig.update_layout(barmode="group", yaxis_title="Abandonment fraction (%)",
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
        st.markdown(f"**Total abandonment cost  (cᴱ={cE:.2f}, cᴸ={cL:.2f})**")
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
                           yaxis_title="Annual abandonment cost",
                           height=360, margin=dict(t=10,b=30), legend=_leg)
        st.plotly_chart(fig3, use_container_width=True, key=f"cost_{mu_E}")

    with c4:
        savings = cost_e - cost_h
        pct     = 100*savings/cost_e if cost_e > 0 else 0.0
        sav_sd  = math.sqrt(cost_e_sd**2 + cost_h_sd**2)
        color_h = "#2e7d32" if savings >= 0 else "#c62828"
        sign    = "↓" if savings >= 0 else "↑"
        st.markdown(
            f"**Total cost C(γ): EDF vs Hybrid-EDF** &nbsp;&nbsp;"
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
                "Abandonment cost (cost units/yr)",
                "Total cost C(γ) (cost units/yr)",
                "Avg queue wait — Early served (weeks)",
                "Avg slack — Early served (weeks)",
                "Nᴱ — reserved Early slots",
                "Nᴸ — reserved Late slots",
            ],
            "EDF (Pooled)": [
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
# Full detailed render (adapted from version 3 Tab 1)
# ═══════════════════════════════════════════════════════════════════════════
def _render_detailed(results: dict, warmup_T: float, T_max_: float, n_runs_: int,
                     params_lead: Dict):
    """
    results: {policy_name: (stats_df, df_all, hists_list)}
    params_lead: dict with early_min/max/F/params, late_min/max/F/params
    """
    import re as _re
    import plotly.io as _pio
    import streamlit.components.v1 as _components

    def _short_name(pname):
        """Strip γ* from policy name for use in figure subplot titles."""
        m = _re.search(r"μᴱ=([\d.]+)", pname)
        if m:
            return f"Hybrid (μᴱ={float(m.group(1)):.0f})"
        return pname  # EDF (Pooled)

    def _scroll_chart(fig, height_px):
        """Render a wide Plotly figure inside a horizontally scrollable div."""
        html_str = _pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        _components.html(
            f'<div style="overflow-x:auto;width:100%;padding-bottom:8px">{html_str}</div>',
            height=height_px + 30, scrolling=False)

    policy_names = list(results.keys())
    n_pol        = len(policy_names)
    short_names  = [_short_name(p) for p in policy_names]

    # γ* callouts
    _N_det = int(st.session_state.get("tab1_params_base", {}).get("N", 18))
    for pname in policy_names:
        if "Optimal Hybrid-EDF" not in pname: continue
        m = _re.search(r"γ\*=([\d.]+)", pname)
        if m:
            _gs = float(m.group(1))
            _NE = _gs * _N_det; _NL = _N_det - _NE
            st.markdown(
                f'<div style="background:#e8f4e8;border-left:4px solid #2e7d32;'
                f'padding:8px 14px;border-radius:4px;margin-bottom:6px;">'
                f'<b>{pname}</b> — γ* = {_gs:.3f} &nbsp;|&nbsp; '
                f'Nᴱ = {_NE:.2f} &nbsp; Nᴸ = {_NL:.2f}</div>',
                unsafe_allow_html=True)

    st.markdown("""<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] >
div > div[data-testid="stVerticalBlock"] {
    border:1.5px solid rgba(100,120,180,0.28);border-radius:10px;
    padding:0.7rem 0.9rem !important;background:rgba(245,247,253,0.55);}
</style>""", unsafe_allow_html=True)

    def _fmt_pct(m, s): return "—" if pd.isna(m) else f"{m:.2%} ± {s:.2%}"
    def _fmt_num(m, s): return "—" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"

    def _ctable(keys, labels, fmt_fn):
        rows = {}
        for k in keys:
            rows[labels[k]] = {
                pn: fmt_fn(results[pn][0].loc[k,'mean'], results[pn][0].loc[k,'std'])
                for pn in policy_names if k in results[pn][0].index}
        return pd.DataFrame(rows).T

    st.subheader(f"Primary metrics  (mean ± SD over {n_runs_} runs)")
    st.dataframe(_ctable(
        ['service_rate','abandon_rate','util_1'],
        {'service_rate':'Service rate','abandon_rate':'Abandon rate','util_1':'Utilisation'},
        _fmt_pct), use_container_width=True)

    st.subheader("Wait times & slack by cohort × outcome")
    for label, keys, lmap in [
        ("**A) Early served**",
         ['es_wait_active','es_slack','es_wait_total'],
         {'es_wait_active':'Wait active','es_slack':'Slack','es_wait_total':'Wait total'}),
        ("**B) Early abandoned**",
         ['ea_wait_active','ea_wait_total'],
         {'ea_wait_active':'Wait active (to deadline)','ea_wait_total':'Wait total'}),
        ("**C) Late served**",
         ['ls_wait_active','ls_slack','ls_wait_total'],
         {'ls_wait_active':'Wait active','ls_slack':'Slack','ls_wait_total':'Wait total'}),
        ("**D) Late abandoned**",
         ['la_wait_active','la_wait_total'],
         {'la_wait_active':'Wait active (to deadline)','la_wait_total':'Wait total'}),
    ]:
        st.markdown(label)
        st.dataframe(_ctable(keys, lmap, _fmt_num), use_container_width=True)

    st.markdown("**Aggregated by cohort**")
    st.dataframe(_ctable(
        ['e_wait_active','e_slack','e_wait_total','l_wait_active','l_slack','l_wait_total'],
        {'e_wait_active':'Early: Wait active','e_slack':'Early: Slack','e_wait_total':'Early: Wait total',
         'l_wait_active':'Late: Wait active','l_slack':'Late: Slack','l_wait_total':'Late: Wait total'},
        _fmt_num), use_container_width=True)

    st.markdown("**Composition**")
    st.dataframe(_ctable(
        ['prop_served_early','prop_aband_early','prop_served_late','prop_aband_late','aband_ratio_e_over_l'],
        {'prop_served_early':'Served among Early','prop_aband_early':'Abandoned among Early',
         'prop_served_late':'Served among Late','prop_aband_late':'Abandoned among Late',
         'aband_ratio_e_over_l':'Abandonment ratio (E/L)'},
        _fmt_pct), use_container_width=True)

    _W = 340   # fixed pixel width per policy subplot

    with st.expander("Queue wait (active) histogram"):
        from plotly.subplots import make_subplots as _msp
        _fig = _msp(rows=1, cols=n_pol, subplot_titles=short_names)
        for _i, _pname in enumerate(policy_names, start=1):
            _, _df, _ = results[_pname]
            _show = (_i == 1)
            for _coh, _col in [("Early", "#4878d0"), ("Late", "#aec7e8")]:
                _v = _df[(_df.cohort == _coh) & (_df.status == "SERVED")]["wait_active"].dropna().to_numpy() * 7
                _fig.add_trace(go.Histogram(x=_v, name=_coh, marker_color=_col,
                                            nbinsx=50, opacity=0.75, showlegend=_show),
                               row=1, col=_i)
            _fig.update_xaxes(title_text="Active wait (days)", row=1, col=_i)
        _fig.update_layout(barmode="overlay", height=340,
                           width=max(700, n_pol * _W),
                           margin=dict(t=50, b=30, l=50, r=20))
        _scroll_chart(_fig, 340)

    with st.expander("Slack histogram (served only)"):
        from plotly.subplots import make_subplots as _msp
        _fig = _msp(rows=1, cols=n_pol, subplot_titles=short_names)
        for _i, _pname in enumerate(policy_names, start=1):
            _, _df, _ = results[_pname]
            _show = (_i == 1)
            for _coh, _col in [("Early", "#4878d0"), ("Late", "#aec7e8")]:
                _v = _df[(_df.cohort == _coh) & (_df.status == "SERVED")]["slack"].dropna().to_numpy() * 7
                _fig.add_trace(go.Histogram(x=_v, name=_coh, marker_color=_col,
                                            nbinsx=50, opacity=0.75, showlegend=_show),
                               row=1, col=_i)
            _fig.update_xaxes(title_text="Residual slack (days)", row=1, col=_i)
        _fig.update_layout(barmode="overlay", height=340,
                           width=max(700, n_pol * _W),
                           margin=dict(t=50, b=30, l=50, r=20))
        _scroll_chart(_fig, 340)

    with st.expander("System size N(t)"):
        from plotly.subplots import make_subplots as _msp
        grid_n = 600; t_grid = np.linspace(0.0, T_max_, grid_n)
        def _eval_h(hist, tg):
            if not hist: return np.zeros_like(tg)
            times = np.array([x for x,_ in hist]); vals = np.array([v for _,v in hist])
            return vals[np.clip(np.searchsorted(times, tg, side='right')-1, 0, len(vals)-1)]

        st.markdown("**Mean ± SD across runs**")
        _fig = _msp(rows=1, cols=n_pol, subplot_titles=short_names)
        for _i, _pname in enumerate(policy_names, start=1):
            _, _, _hists = results[_pname]
            if _hists:
                M = np.vstack([_eval_h(h, t_grid) for h in _hists])
                mn = M.mean(0); sd = M.std(0, ddof=0)
                _show = (_i == 1)
                _fig.add_trace(go.Scatter(x=t_grid, y=mn+sd, mode='lines',
                                          line=dict(width=0), showlegend=False), row=1, col=_i)
                _fig.add_trace(go.Scatter(x=t_grid, y=np.maximum(0, mn-sd), mode='lines',
                                          fill='tonexty', fillcolor='rgba(100,149,237,0.25)',
                                          line=dict(width=0), name='±1 SD', showlegend=_show), row=1, col=_i)
                _fig.add_trace(go.Scatter(x=t_grid, y=mn, mode='lines',
                                          line=dict(color='red'), name='Mean N(t)', showlegend=_show), row=1, col=_i)
                _fig.add_vline(x=warmup_T, line_dash="dash", line_color="red",
                               annotation_text="Warm-up end", row=1, col=_i)
            _fig.update_xaxes(title_text="Time (weeks)", row=1, col=_i)
            _fig.update_yaxes(title_text="N(t)", row=1, col=_i)
        _fig.update_layout(height=320, width=max(700, n_pol * _W),
                           margin=dict(t=50, b=30, l=50, r=20))
        _scroll_chart(_fig, 320)
        st.caption("Shaded = ±1 SD across runs.")

        st.markdown("**Last run (sample path)**")
        _fig2 = _msp(rows=1, cols=n_pol, subplot_titles=short_names)
        for _i, _pname in enumerate(policy_names, start=1):
            _, _, _hists = results[_pname]
            if _hists:
                _hdf = pd.DataFrame(_hists[-1], columns=['time', 'system_size'])
                _fig2.add_trace(go.Scatter(x=_hdf['time'], y=_hdf['system_size'],
                                           mode='lines', line_shape='hv',
                                           line=dict(color='#4878d0'), showlegend=False),
                                row=1, col=_i)
                _fig2.add_vline(x=warmup_T, line_dash="dash", line_color="red",
                                annotation_text="Warm-up end", row=1, col=_i)
            _fig2.update_xaxes(title_text="Time (weeks)", row=1, col=_i)
            _fig2.update_yaxes(title_text="N(t)", row=1, col=_i)
        _fig2.update_layout(height=300, width=max(700, n_pol * _W),
                            margin=dict(t=50, b=30, l=50, r=20))
        _scroll_chart(_fig2, 300)

    with st.expander("Lead-time & service-time distributions"):
        e_min = params_lead["early_min"]; e_max = params_lead["early_max"]
        e_F   = params_lead["early_F"];   e_p   = params_lead["early_params"]
        l_min = params_lead["late_min"];  l_max = params_lead["late_max"]
        l_F   = params_lead["late_F"];    l_p   = params_lead["late_params"]
        xs_e, pdf_e, _ = _pdf_cdf_lead_grid(e_min, e_max, e_F, e_p)
        xs_l, pdf_l, _ = _pdf_cdf_lead_grid(l_min, l_max, l_F, l_p)
        col1, col2 = st.columns(2)
        with col1:
            fe = px.line(pd.DataFrame({"T":xs_e,"pdf":pdf_e}), x="T", y="pdf",
                         title="Theoretical PDF — Early lead time",
                         labels={"T":"Lead time (days)","pdf":"pdf"})
            st.plotly_chart(fe, use_container_width=True, key="det_pdf_e")
        with col2:
            fl = px.line(pd.DataFrame({"T":xs_l,"pdf":pdf_l}), x="T", y="pdf",
                         title="Theoretical PDF — Late lead time",
                         labels={"T":"Lead time (days)","pdf":"pdf"})
            st.plotly_chart(fl, use_container_width=True, key="det_pdf_l")

        first_pname = policy_names[0]
        _, df_first, _ = results[first_pname]
        st.markdown(f"**Empirical PDFs** (from *{first_pname}*)")
        _cidx = [0]
        def _pdf_emp(series, title, xlab):
            s = series.dropna().to_numpy()
            if s.size == 0: st.info(f"No data: {title}"); return
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s, histnorm="probability density", nbinsx=60))
            fig.update_layout(title=title, xaxis_title=xlab, yaxis_title="pdf",
                              bargap=0.05, height=260, margin=dict(t=30,b=20))
            st.plotly_chart(fig, use_container_width=True, key=f"det_emp_{_cidx[0]}")
            _cidx[0] += 1
        col1, col2 = st.columns(2)
        with col1:
            _pdf_emp(df_first.loc[df_first.cohort=="Early","lead_time"] * 7, "Empirical — Early lead time","Lead time (days)")
            _pdf_emp(df_first.loc[df_first.cohort=="Early","svc"],       "Empirical — Early service time","Service time (slots)")
        with col2:
            _pdf_emp(df_first.loc[df_first.cohort=="Late","lead_time"] * 7,  "Empirical — Late lead time","Lead time (days)")
            _pdf_emp(df_first.loc[df_first.cohort=="Late","svc"],        "Empirical — Late service time","Service time (slots)")


# ═══════════════════════════════════════════════════════════════════════════
# TABS
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
    x_lbl = ["EDF (Pooled)"]
    for res in results_list_:
        x_lbl.append(f"Hybrid (μᴱ={res['mu_E']:.0f})")

    for res in results_list_:
        mu_E_ = res["mu_E"]; gstar_ = res["gamma_star"]
        hyb_  = res["hyb"]
        lbl_h = f"Optimal Hybrid-EDF (μᴱ={mu_E_:.0f}, γ*={gstar_:.3f})"
        _NE_html_r = gstar_ * _N_html; _NL_html_r = _N_html - _NE_html_r
        parts.append(f'<div class="section"><h2>EDF (Pooled) vs {lbl_h}</h2>')
        parts.append(f'<div class="badge">{lbl_h} — γ* = {gstar_:.3f} &nbsp;|&nbsp; '
                     f'N<sup>E</sup> = {_NE_html_r:.2f} &nbsp; N<sup>L</sup> = {_NL_html_r:.2f}</div>')

        pE_e, pL_e, pE_e_sd, pL_e_sd, lE_e, lL_e, cost_e, cost_e_sd = \
            _policy_metrics(edf_data_, ann_E_, ann_L_, cE_, cL_)
        _slot_rw_html = v_ * _NE_html_r * 52
        pE_h, pL_h, pE_h_sd, pL_h_sd, lE_h, lL_h, cost_h, cost_h_sd = \
            _policy_metrics(hyb_, ann_E_, ann_L_, cE_, cL_, slot_reward=_slot_rw_html)
        x2 = ["EDF (Pooled)", f"Hybrid (μᴱ={mu_E_:.0f})"]

        fig_frac = go.Figure()
        fig_frac.add_trace(go.Bar(name="Early engagers", x=x2,
                                  y=[pE_e*100, pE_h*100],
                                  error_y=dict(type='data', array=[pE_e_sd*100, pE_h_sd*100], visible=True),
                                  marker_color=COLOR_E))
        fig_frac.add_trace(go.Bar(name="Late engagers", x=x2,
                                  y=[pL_e*100, pL_h*100],
                                  error_y=dict(type='data', array=[pL_e_sd*100, pL_h_sd*100], visible=True),
                                  marker_color=COLOR_L))
        fig_frac.update_layout(barmode="group", yaxis_title="Abandonment fraction (%)",
                               title="Abandonment fractions", height=360, width=640, legend=_leg)

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

        aband_cost_h_ = cE_ * lE_h + cL_ * lL_h  # Hybrid abandonment cost (no slot reward)
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
        fig_cost.update_layout(barmode="stack", yaxis_title="Annual abandonment cost (cost units/yr)",
                               title=f"Total abandonment cost  (cᴱ={cE_:.2f}, cᴸ={cL_:.2f})",
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
                              title=f"Total cost C(γ): EDF vs Hybrid-EDF (v={v_:.3f})",
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
        all_pol = [("EDF (Pooled)", edf_data_)] + \
                  [(f"Hybrid (μᴱ={r['mu_E']:.0f}, γ*={r['gamma_star']:.3f})", r["hyb"])
                   for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        mu_labels_  = [p for p, _ in all_pol]
        fig_labels_ = ["EDF (Pooled)"] + [
            f"Hybrid (μᴱ={r['mu_E']:.0f})"
            for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        metrics_   = []
        for pname, data in all_pol:
            if "Hybrid" in pname:
                _gstar_html = data.get("gamma_star", 0.0)
                _NE_s = _gstar_html * _N_html; _NL_s = _N_html - _NE_s
                _slot_rw_s  = v_ * _NE_s * 52
            else:
                _slot_rw_s = 0.0; _NE_s = None; _NL_s = None
            pE, pL, pE_sd, pL_sd, lossE, lossL, cost, cost_sd = \
                _policy_metrics(data, ann_E_, ann_L_, cE_, cL_, slot_reward=_slot_rw_s)
            aband_cost_ = cE_ * lossE + cL_ * lossL  # pure abandonment cost (no slot reward)
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
        fig_s1.update_layout(barmode="group", yaxis_title="Abandonment fraction (%)",
                             title="Abandonment fractions — all policies", height=400, legend=_leg)

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
        fig_s3.update_layout(barmode="stack", yaxis_title="Annual abandonment cost (cost units/yr)",
                             title=f"Total abandonment cost — all policies (cᴱ={cE_:.2f}, cᴸ={cL_:.2f})",
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
                      if "Hybrid" in m["name"] else "—")
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
                "Abandonment cost (cost units/yr)": f"{m['aband_cost']:.1f} ± {m['cost_sd']:.1f}",
                "Total cost C(γ) (cost units/yr)": f"{m['cost']:.1f} ± {m['cost_sd']:.1f}",
                "Cost saving vs EDF": saving,
            })
        parts.append("<h3>Summary table</h3>")
        parts.append(_tbl(pd.DataFrame(rows_sum_)))
        parts.append('</div>')

    # ── Detailed Results — All Policies ─────────────────────────────────────
    if results_list_:
        from plotly.subplots import make_subplots as _msp
        all_pol_det = [("EDF (Pooled)", edf_data_)] + \
                      [(f"Optimal Hybrid-EDF (μᴱ={r['mu_E']:.0f}, γ*={r['gamma_star']:.3f})", r["hyb"])
                       for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        results_det = {pname: (data["stats"], data["df_all"], data.get("hists", []))
                       for pname, data in all_pol_det if data}
        policy_names_det = list(results_det.keys())
        short_titles_det = ["EDF"] + [
            f"Hybrid (μᴱ={r['mu_E']:.0f})"
            for r in sorted(results_list_, key=lambda r: r["mu_E"])]
        n_pol_det = len(policy_names_det)
        _W_det = 380

        parts.append('<div class="section"><h2>Detailed Results — All Policies</h2>')

        # Primary metrics table
        def _fmt_pct_h(m, s): return "—" if pd.isna(m) else f"{m:.2%} ± {s:.2%}"
        def _fmt_num_h(m, s): return "—" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"
        def _ctable_h(keys, labels, fmt_fn):
            rows = {}
            for k in keys:
                rows[labels[k]] = {
                    pn: fmt_fn(results_det[pn][0].loc[k, "mean"], results_det[pn][0].loc[k, "std"])
                    for pn in policy_names_det if k in results_det[pn][0].index}
            return pd.DataFrame(rows).T

        parts.append(f"<h3>Primary metrics (mean ± SD over {nr_} runs)</h3>")
        parts.append(_tbl_metrics(_ctable_h(
            ["service_rate", "abandon_rate", "util_1"],
            {"service_rate": "Service rate", "abandon_rate": "Abandon rate", "util_1": "Utilisation"},
            _fmt_pct_h)))

        parts.append("<h3>Wait times &amp; slack by cohort &times; outcome</h3>")
        for _sec_label, _keys, _lmap in [
            ("<b>A) Early served</b>",
             ["es_wait_active", "es_slack", "es_wait_total"],
             {"es_wait_active": "Wait active", "es_slack": "Slack", "es_wait_total": "Wait total"}),
            ("<b>B) Early abandoned</b>",
             ["ea_wait_active", "ea_wait_total"],
             {"ea_wait_active": "Wait active (to deadline)", "ea_wait_total": "Wait total"}),
            ("<b>C) Late served</b>",
             ["ls_wait_active", "ls_slack", "ls_wait_total"],
             {"ls_wait_active": "Wait active", "ls_slack": "Slack", "ls_wait_total": "Wait total"}),
            ("<b>D) Late abandoned</b>",
             ["la_wait_active", "la_wait_total"],
             {"la_wait_active": "Wait active (to deadline)", "la_wait_total": "Wait total"}),
        ]:
            parts.append(f"<p>{_sec_label}</p>")
            parts.append(_tbl_metrics(_ctable_h(_keys, _lmap, _fmt_num_h)))

        parts.append("<p><b>Aggregated by cohort</b></p>")
        parts.append(_tbl_metrics(_ctable_h(
            ["e_wait_active", "e_slack", "e_wait_total", "l_wait_active", "l_slack", "l_wait_total"],
            {"e_wait_active": "Early: Wait active", "e_slack": "Early: Slack", "e_wait_total": "Early: Wait total",
             "l_wait_active": "Late: Wait active",  "l_slack": "Late: Slack",  "l_wait_total": "Late: Wait total"},
            _fmt_num_h)))

        parts.append("<p><b>Composition</b></p>")
        parts.append(_tbl_metrics(_ctable_h(
            ["prop_served_early", "prop_aband_early", "prop_served_late", "prop_aband_late", "aband_ratio_e_over_l"],
            {"prop_served_early": "Served among Early", "prop_aband_early": "Abandoned among Early",
             "prop_served_late": "Served among Late",   "prop_aband_late": "Abandoned among Late",
             "aband_ratio_e_over_l": "Abandonment ratio (E/L)"},
            _fmt_pct_h)))

        def _binned_bar(fig, df, col, filter_mask=None, n_bins=50, row=1, subplot_col=1, show_legend=True, scale=1.0):
            """Add pre-binned histogram traces (go.Bar) to avoid embedding raw data.
            scale: multiply values before binning (e.g. 7 to convert weeks→days)."""
            for _coh, _clr in [("Early", "#4878d0"), ("Late", "#aec7e8")]:
                mask = df.cohort == _coh
                if filter_mask is not None:
                    mask = mask & filter_mask
                _v = df[mask][col].dropna().to_numpy() * scale
                if _v.size == 0:
                    continue
                counts, edges = np.histogram(_v, bins=n_bins)
                centres = 0.5 * (edges[:-1] + edges[1:])
                fig.add_trace(go.Bar(x=centres, y=counts, name=_coh, marker_color=_clr,
                                     opacity=0.75, showlegend=show_legend,
                                     width=(edges[1] - edges[0]) * 0.95),
                              row=row, col=subplot_col)

        # Queue wait histogram
        fig_wh = _msp(rows=1, cols=n_pol_det, subplot_titles=short_titles_det)
        for _i, _pname in enumerate(policy_names_det, start=1):
            _, _df, _ = results_det[_pname]
            _binned_bar(fig_wh, _df, "wait_active", show_legend=(_i == 1), subplot_col=_i, scale=7,
                        filter_mask=(_df.status == "SERVED"))
            fig_wh.update_xaxes(title_text="Active wait (days)", row=1, col=_i)
        fig_wh.update_layout(barmode="overlay", height=360,
                             width=max(800, n_pol_det * _W_det),
                             title="Queue wait (active) histogram",
                             margin=dict(t=60, b=40, l=50, r=20))
        parts.append(_scroll_fig_html(fig_wh))

        # Slack histogram
        fig_sh = _msp(rows=1, cols=n_pol_det, subplot_titles=short_titles_det)
        for _i, _pname in enumerate(policy_names_det, start=1):
            _, _df, _ = results_det[_pname]
            _binned_bar(fig_sh, _df, "slack", filter_mask=(_df.status == "SERVED"),
                        show_legend=(_i == 1), subplot_col=_i, scale=7)
            fig_sh.update_xaxes(title_text="Residual slack (days)", row=1, col=_i)
        fig_sh.update_layout(barmode="overlay", height=360,
                             width=max(800, n_pol_det * _W_det),
                             title="Slack histogram (served only)",
                             margin=dict(t=60, b=40, l=50, r=20))
        parts.append(_scroll_fig_html(fig_sh))

        # N(t) mean ± SD
        grid_n_h = 600; t_grid_h = np.linspace(0.0, T_, grid_n_h)
        def _eval_h_html(hist, tg):
            if not hist: return np.zeros_like(tg)
            times = np.array([x for x, _ in hist]); vals = np.array([v for _, v in hist])
            return vals[np.clip(np.searchsorted(times, tg, side="right") - 1, 0, len(vals) - 1)]

        fig_nt = _msp(rows=1, cols=n_pol_det, subplot_titles=short_titles_det)
        for _i, _pname in enumerate(policy_names_det, start=1):
            _, _, _hists = results_det[_pname]
            if _hists:
                M = np.vstack([_eval_h_html(h, t_grid_h) for h in _hists])
                mn = M.mean(0); sd = M.std(0, ddof=0)
                _show = (_i == 1)
                fig_nt.add_trace(go.Scatter(x=t_grid_h, y=mn + sd, mode="lines",
                                            line=dict(width=0), showlegend=False), row=1, col=_i)
                fig_nt.add_trace(go.Scatter(x=t_grid_h, y=np.maximum(0, mn - sd), mode="lines",
                                            fill="tonexty", fillcolor="rgba(100,149,237,0.25)",
                                            line=dict(width=0), name="±1 SD", showlegend=_show), row=1, col=_i)
                fig_nt.add_trace(go.Scatter(x=t_grid_h, y=mn, mode="lines",
                                            line=dict(color="red"), name="Mean N(t)", showlegend=_show), row=1, col=_i)
                fig_nt.add_vline(x=T0_, line_dash="dash", line_color="red",
                                 annotation_text="Warm-up end", row=1, col=_i)
            fig_nt.update_xaxes(title_text="Time (weeks)", row=1, col=_i)
            fig_nt.update_yaxes(title_text="N(t)", row=1, col=_i)
        fig_nt.update_layout(height=360, width=max(800, n_pol_det * _W_det),
                             title="System size N(t) — mean ± SD",
                             margin=dict(t=60, b=40, l=50, r=20))
        parts.append(_scroll_fig_html(fig_nt))

        # N(t) last run (sample path)
        fig_nt2 = _msp(rows=1, cols=n_pol_det, subplot_titles=short_titles_det)
        for _i, _pname in enumerate(policy_names_det, start=1):
            _, _, _hists = results_det[_pname]
            if _hists:
                _hdf = pd.DataFrame(_hists[-1], columns=["time", "system_size"])
                fig_nt2.add_trace(go.Scatter(x=_hdf["time"], y=_hdf["system_size"],
                                             mode="lines", line_shape="hv",
                                             line=dict(color="#4878d0"), showlegend=False),
                                  row=1, col=_i)
                fig_nt2.add_vline(x=T0_, line_dash="dash", line_color="red",
                                  annotation_text="Warm-up end", row=1, col=_i)
            fig_nt2.update_xaxes(title_text="Time (weeks)", row=1, col=_i)
            fig_nt2.update_yaxes(title_text="N(t)", row=1, col=_i)
        fig_nt2.update_layout(height=340, width=max(800, n_pol_det * _W_det),
                              title="System size N(t) — last run (sample path)",
                              margin=dict(t=60, b=40, l=50, r=20))
        parts.append(_scroll_fig_html(fig_nt2))

        parts.append('</div>')

    parts.append("</body></html>")
    return "".join(parts)


tab1, tab2 = st.tabs(["Tab 1: Policy Simulation", "Tab 2: Summary"])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Policy Simulation")
    st.markdown(
        "This tool compares two approaches to scheduling planned (elective) C-sections. "
        "The **standard approach** (EDF Pooled) allocates all operating slots to a shared waiting list, "
        "giving priority to the patient whose planned delivery date is soonest. "
        "The **reservation approach** (Optimal Hybrid-EDF) sets aside a dedicated fraction of slots for women who book early.\n\n"
        "Results are shown after each scenario is completed — confirm before continuing to the next."
    )

    # ── Save / Load session ───────────────────────────────────────────────
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
        cfg_left, cfg_right = st.columns([3, 2], gap="large")
        with cfg_left:
            st.markdown("**Optimal Hybrid-EDF(μᴱ) — Configuration**")
            means_text_t1 = st.text_input(
                "Allocation to planned date of delivery means (μᴱ) (comma-separated)",
                key="t1_means_text",
                disabled=results_exist)
            try:
                means_list_t1 = sorted(set(float(x.strip()) for x in means_text_t1.split(",") if x.strip()))
                if not means_list_t1: raise ValueError
                st.caption(f"{len(means_list_t1)} values: " + ", ".join(f"{m:.0f}" for m in means_list_t1))
            except Exception:
                st.error("Please enter valid comma-separated numbers.")
                means_list_t1 = []

            st.markdown("**γ Grid Step**")
            _gs_col, _ = st.columns([1, 1])
            with _gs_col:
                t1_gstep = st.number_input("Step", min_value=0.01, max_value=1.0,
                                           step=0.01, format="%.2f",
                                           key="t1_gstep", disabled=results_exist)

        with cfg_right:
            st.markdown("**Replications**")
            t1_n_runs = st.number_input("Number of replications", 1, 500, step=5,
                                        key="t1_n_runs", disabled=results_exist)
            st.caption("Cost & Equity and γ range are set in the sidebar.")

    # ── Buttons ───────────────────────────────────────────────────────────
    means_cfg  = st.session_state.get("tab1_means_config", means_list_t1)
    n_done     = len(st.session_state.get("tab1_results", []))
    run_first  = False
    run_next   = False
    run_all    = False

    btn_col, run_all_col, reset_col = st.columns([4, 1, 1])
    with btn_col:
        if not results_exist:
            if means_list_t1:
                run_first = st.button(
                    f"▶  Run EDF pooled + Optimal Hybrid-EDF (μᴱ = {means_list_t1[0]:.0f})",
                    key="t1_run_first", type="primary")
        else:
            if n_done < len(means_cfg):
                next_mu = means_cfg[n_done]
                run_next = st.button(
                    f"▶  Run Optimal Hybrid-EDF (μᴱ = {next_mu:.0f})",
                    key=f"t1_cont_{n_done}", type="primary")
            else:
                st.success(f"✓ All {len(means_cfg)} μᴱ values completed.")

    with run_all_col:
        if (not results_exist and means_list_t1) or (results_exist and n_done < len(means_cfg)):
            run_all = st.button("▶▶  Run All",
                                key=f"t1_run_all_{n_done}")

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

    # ── Handle first run (mutually exclusive with run_all) ────────────────
    if run_first and not run_all and means_list_t1:
        _save_config()
        gc  = st.session_state["tab1_gamma_config"]
        mu0 = means_list_t1[0]
        with st.spinner(f"Running EDF (Pooled) + Optimal Hybrid-EDF (μᴱ = {mu0:.0f})  [1/{len(means_list_t1)}]…"):
            _run_edf_and_first_hybrid(mu0, gc, "[1/1]")
        st.rerun()

    # ── Handle continue (mutually exclusive with run_all) ─────────────────
    if run_next and not run_all and results_exist:
        gc  = st.session_state["tab1_gamma_config"]
        mu_ = means_cfg[n_done]
        with st.spinner(f"Running Optimal Hybrid-EDF  (μᴱ = {mu_:.0f})  [{n_done+1}/{len(means_cfg)}]…"):
            _run_hybrid_only(mu_, gc, f"[{n_done+1}/{len(means_cfg)}]")
        st.rerun()

    # ── Handle run all ────────────────────────────────────────────────────
    if run_all and not run_first and not run_next and means_list_t1:
        if not results_exist:
            _save_config()
        gc        = st.session_state["tab1_gamma_config"]
        n_so_far  = len(st.session_state["tab1_results"])
        remaining = st.session_state["tab1_means_config"][n_so_far:]
        n_total   = len(st.session_state["tab1_means_config"])
        for i, mu in enumerate(remaining):
            lbl = f"[{n_so_far+i+1}/{n_total}]"
            if n_so_far == 0 and i == 0:
                with st.spinner(f"Running EDF (Pooled) + Optimal Hybrid-EDF (μᴱ = {mu:.0f})  {lbl}…"):
                    _run_edf_and_first_hybrid(mu, gc, lbl)
            else:
                with st.spinner(f"Running Optimal Hybrid-EDF  (μᴱ = {mu:.0f})  {lbl}…"):
                    _run_hybrid_only(mu, gc, lbl)
        st.rerun()

    # ── Display accumulated results ───────────────────────────────────────
    if results_exist:
        results_list = st.session_state["tab1_results"]
        edf_base = st.session_state["tab1_edf"]   # single EDF baseline
        lE_s  = st.session_state.get("tab1_lambdaE", float(λE))
        lL_s  = st.session_state.get("tab1_lambdaL", float(λL))
        gc_s  = st.session_state.get("tab1_gamma_config", {})
        cE_s  = gc_s.get("cE", 1.0); cL_s = gc_s.get("cL", 2.0); v_s = gc_s.get("v", 0.0)
        _pb_s  = st.session_state.get("tab1_params_base", {})
        _N_s   = int(_pb_s.get("N", 18))
        _T_s   = float(st.session_state.get("tab1_T_max",  T_max))
        _T0_s  = float(st.session_state.get("tab1_T0",     T0_warmup))

        for i, res in enumerate(results_list):
            is_latest = (i == len(results_list) - 1)
            _gs_exp = res['gamma_star']
            _pb_exp = st.session_state.get("tab1_params_base", {})
            _N_exp  = int(_pb_exp.get("N", 18))
            _NE_exp = _gs_exp * _N_exp; _NL_exp = _N_exp - _NE_exp
            with st.expander(
                f"EDF (Pooled)  vs  Optimal Hybrid-EDF  (μᴱ = {res['mu_E']:.0f})"
                f"  with  γ* = {_gs_exp:.3f},  Nᴱ = {_NE_exp:.2f},  Nᴸ = {_NL_exp:.2f}",
                expanded=is_latest
            ):
                _comparison_charts(edf_base, res["hyb"], lE_s, lL_s, cE_s, cL_s,
                                   v=v_s, N=_N_s, T=_T_s, T0=_T0_s)

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



# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — SUMMARY
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Summary")

    if not st.session_state.get("tab1_results"):
        st.info("Run Tab 1 first — results will appear here once at least one μᴱ has been completed.")
    else:
        results_list = st.session_state["tab1_results"]
        lE_s  = st.session_state.get("tab1_lambdaE", float(λE))
        lL_s  = st.session_state.get("tab1_lambdaL", float(λL))
        gc_s  = st.session_state.get("tab1_gamma_config", {})
        cE_s  = gc_s.get("cE", 1.0); cL_s = gc_s.get("cL", 2.0); v_s = gc_s.get("v", 0.0)
        ann_E, ann_L = _ann_headcounts(lE_s, lL_s)
        T_s  = float(st.session_state.get("tab1_T_max",  float(T_max)))
        T0_s = float(st.session_state.get("tab1_T0",     float(T0_warmup)))
        nr_s = st.session_state.get("tab1_n_runs", 50)
        pb_s = st.session_state.get("tab1_params_base", dict(params_base))
        _N_t2 = int(pb_s.get("N", 18))

        # Build ordered policy list: single EDF baseline then one Hybrid per μᴱ
        tab1_edf = st.session_state.get("tab1_edf", {})
        all_policies = [("EDF (Pooled)", tab1_edf)]
        for res in sorted(results_list, key=lambda r: r["mu_E"]):
            all_policies.append(
                (f"Hybrid (μᴱ={res['mu_E']:.0f}, γ*={res['gamma_star']:.3f})", res["hyb"]))
        mu_labels = [p for p, _ in all_policies]
        # Shorter labels for figures only (no γ*)
        fig_labels = ["EDF (Pooled)"] + [
            f"Hybrid (μᴱ={res['mu_E']:.0f})"
            for res in sorted(results_list, key=lambda r: r["mu_E"])]

        # ── Compute metrics for all policies ─────────────────────────────
        metrics = []
        for pname, data in all_policies:
            if "Hybrid" in pname:
                _gstar_t2 = data.get("gamma_star", 0.0)
                _NE_t2 = _gstar_t2 * _N_t2; _NL_t2 = _N_t2 - _NE_t2
                _slot_rw  = v_s * _NE_t2 * 52
            else:
                _slot_rw = 0.0; _gstar_t2 = None; _NE_t2 = None; _NL_t2 = None
            pE, pL, pE_sd, pL_sd, lossE, lossL, cost, cost_sd = _policy_metrics(
                data, ann_E, ann_L, cE_s, cL_s, slot_reward=_slot_rw)
            aband_cost = cE_s * lossE + cL_s * lossL  # pure abandonment cost (no slot reward)
            _st = data.get("stats")
            _ewa = float(_st.loc["es_wait_active", "mean"]) if (_st is not None and "es_wait_active" in _st.index) else float("nan")
            _ewa_sd = float(_st.loc["es_wait_active", "std"]) if (_st is not None and "es_wait_active" in _st.index) else float("nan")
            _esl = float(_st.loc["es_slack", "mean"]) if (_st is not None and "es_slack" in _st.index) else float("nan")
            _esl_sd = float(_st.loc["es_slack", "std"]) if (_st is not None and "es_slack" in _st.index) else float("nan")
            metrics.append(dict(
                name=pname, pE=pE, pL=pL, pE_sd=pE_sd, pL_sd=pL_sd,
                lossE=lossE, lossL=lossL, cost=cost, cost_sd=cost_sd,
                aband_cost=aband_cost, NE=_NE_t2, NL=_NL_t2,
                es_wait_active=_ewa, es_wait_active_sd=_ewa_sd,
                es_slack=_esl, es_slack_sd=_esl_sd))

        COLOR_PALETTE = ["#4878d0","#ee854a","#6acc65","#d65f5f","#956cb4","#8c613c","#dc7ec0"]
        colors = {m["name"]: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i,m in enumerate(metrics)}

        st.subheader("Comparative Overview — All Policies")

        _sum_leg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10))

        # ── Plot 1: Abandonment fractions ──────────────────────────────────
        st.markdown("**Plot 1 — Abandonment fractions (%)**")
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
        fig1.update_layout(barmode="group", yaxis_title="Abandonment fraction (%)",
                           xaxis_title="Policy", height=380,
                           margin=dict(t=10,b=30), legend=_sum_leg)
        st.plotly_chart(fig1, use_container_width=True, key="sum_frac")

        # ── Plot 2: Case losses ────────────────────────────────────────────
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

        # ── Plot 3: Total abandonment cost (stacked by component) ─────────
        st.markdown(f"**Plot 3 — Total abandonment cost  (cᴱ={cE_s:.2f}, cᴸ={cL_s:.2f})**")
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
            yaxis_title="Annual abandonment cost (cost units/yr)",
            xaxis_title="Policy", height=380,
            margin=dict(t=10,b=30), legend=_sum_leg)
        st.plotly_chart(fig3, use_container_width=True, key="sum_cost")

        # ── Plot 4: Total cost C(γ) per policy ────────────────────────────
        st.markdown(f"**Plot 4 — Total cost C(γ)  (v={v_s:.3f})**")
        _edf_cost_t2 = next(m["cost"] for m in metrics if "Hybrid" not in m["name"])
        fig4_sum = go.Figure()
        fig4_sum.add_trace(go.Bar(
            x=fig_labels,
            y=[m["cost"] for m in metrics],
            error_y=dict(type='data', array=[m["cost_sd"] for m in metrics], visible=True),
            marker_color=["#4878d0" if "Hybrid" not in m["name"] else "#2e7d32" for m in metrics],
            showlegend=False,
        ))
        for lbl, m in zip(fig_labels, metrics):
            if "Hybrid" not in m["name"]:
                continue
            sav_t2 = _edf_cost_t2 - m["cost"]
            pct_t2 = 100 * sav_t2 / _edf_cost_t2 if _edf_cost_t2 > 0 else 0.0
            sign_t2 = "↓" if sav_t2 >= 0 else "↑"
            col_t2  = "#2e7d32" if sav_t2 >= 0 else "#c62828"
            fig4_sum.add_annotation(
                x=lbl,
                y=m["cost"] + m["cost_sd"] * 1.4,
                text=f"{sign_t2}{abs(sav_t2):.1f} ({abs(pct_t2):.1f}%)",
                showarrow=False,
                font=dict(size=10, color=col_t2),
            )
        fig4_sum.update_layout(
            yaxis_title="Annual total cost C(γ) (cost units/yr)",
            xaxis_title="Policy", height=400,
            margin=dict(t=30, b=30))
        st.plotly_chart(fig4_sum, use_container_width=True, key="sum_totalcost")

        # ── Summary table ─────────────────────────────────────────────────
        with st.expander("Summary table"):
            edf_cost_base = _policy_metrics(tab1_edf, ann_E, ann_L, cE_s, cL_s)[6]
            rows_sum = []
            for m in metrics:
                saving = (f"{edf_cost_base - m['cost']:+.1f}"
                          if "Hybrid" in m["name"] else "—")
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
                    "Abandonment cost (cost units/yr)": f"{m['aband_cost']:.1f} ± {m['cost_sd']:.1f}",
                    "Total cost C(γ) (cost units/yr)": f"{m['cost']:.1f} ± {m['cost_sd']:.1f}",
                    "Cost saving vs EDF (Pooled)": saving,
                })
            st.dataframe(pd.DataFrame(rows_sum), use_container_width=True, hide_index=True)

        # ── Full detailed results (version 3 style) ───────────────────────
        st.divider()
        st.subheader("Detailed Results — All Policies")

        # Reconstruct results dict for _render_detailed (same order as all_policies)
        results_dict = {}
        for pname, data in all_policies:
            if data:
                results_dict[pname] = (data["stats"], data["df_all"], data.get("hists", []))

        params_lead = dict(
            early_min=pb_s.get("early_min", 3.0), early_max=pb_s.get("early_max", 10.0),
            early_F=pb_s.get("early_F", "Lognormal (truncated)"),
            early_params=pb_s.get("early_params", {"mean":7.0,"sd":4.0}),
            late_min=pb_s.get("late_min", 3.0), late_max=pb_s.get("late_max", 10.0),
            late_F=pb_s.get("late_F", "Lognormal (truncated)"),
            late_params=pb_s.get("late_params", {"mean":7.0,"sd":4.0}),
        )
        _render_detailed(results_dict, T0_s, T_s, nr_s, params_lead)

