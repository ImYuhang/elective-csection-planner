"""
sim_core.py — Shared simulation engine for version 3 Mar 24, 2026.

Contains:
  - Shared helpers (distribution sampling, PDF grid)
  - SimFCFS  : verbatim from FCFS_v7.py  (pooled queue, FCFS priority)
  - SimEDF   : verbatim from EDF_v7.py / EDF_2m2q_v7.py (pooled or dedicated queue, EDF priority)
  - combine_histories : verbatim from EDF_2m2q_v7.py
  - sweep_gamma       : verbatim from p1.py  (γ-sweep for Reservation policy)
  - Trace-based CRN functions : verbatim from p2.py (for policy comparison with shared arrivals)
  - Reporting helpers : verbatim from p2.py
"""

from __future__ import annotations

import math
import heapq
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SQRT2PI = math.sqrt(2.0 * math.pi)
EPS = 1e-9

# =====================================================================
# Shared helpers
# =====================================================================

def _lognormal_mu_sigma_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
    mean = max(1e-9, float(mean))
    sd   = max(1e-9, float(sd))
    sigma2 = math.log(1.0 + (sd * sd) / (mean * mean))
    sigma  = math.sqrt(max(1e-12, sigma2))
    mu     = math.log(mean) - 0.5 * sigma2
    return mu, sigma


def _pdf_cdf_lead_grid(lo: float, hi: float, F: str, params: Dict[str, float], n: int = 800):
    """Compute theoretical PDF/CDF for lead-time distribution on a grid."""
    lo = float(max(lo, EPS))
    hi = float(max(hi, lo + EPS))
    xs  = np.linspace(lo, hi, n)
    pdf = np.zeros_like(xs)
    cdf = np.zeros_like(xs)

    if F == "Deterministic":
        c = float(params.get("c", (lo + hi) / 2))
        c = float(min(max(c, lo), hi))
        width = (hi - lo) / max(1, (n - 1))
        idx = int(np.argmin(np.abs(xs - c)))
        pdf[idx] = 1.0 / max(width, 1e-12)
        cdf = (xs >= c).astype(float)
        return xs, pdf, cdf

    if F == "Uniform":
        pdf[:] = 1.0 / (hi - lo)
        cdf = (xs - lo) / (hi - lo)
        cdf = np.clip(cdf, 0.0, 1.0)
        return xs, pdf, cdf

    if F.startswith("Normal"):
        mean = float(params.get("mean", (lo + hi) / 2))
        sd   = float(params.get("sd",   max(1e-3, (hi - lo) / 4)))
        raw  = (1.0 / (sd * SQRT2PI)) * np.exp(-(xs - mean) ** 2 / (2.0 * sd * sd))
        area = np.trapz(raw, xs)
        pdf  = raw / area if area > 0 else raw
    else:
        # Lognormal (truncated)
        mean = float(params.get("mean", (lo + hi) / 2))
        sd   = float(params.get("sd",   max(1e-3, mean / 2)))
        mu, sigma = _lognormal_mu_sigma_from_mean_sd(mean, sd)
        raw  = (1.0 / (xs * sigma * SQRT2PI)) * np.exp(-(np.log(xs) - mu) ** 2 / (2.0 * sigma * sigma))
        raw[~np.isfinite(raw)] = 0.0
        area = np.trapz(raw, xs)
        pdf  = raw / area if area > 0 else raw

    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * (xs[1:] - xs[:-1]) / 2.0)
    cdf = np.concatenate([[0.0], cdf])
    cdf = np.clip(cdf, 0.0, 1.0)
    return xs, pdf, cdf


# =====================================================================
# SimFCFS — verbatim from FCFS_v7.py
# Priority queue tuples: (book, pid)   [2-tuple]
# =====================================================================

@dataclass(order=True)
class PatientFCFS:
    book:     float
    id:       int
    cohort:   str
    deadline: float
    status:   str = "WAITING"
    start:    Optional[float] = None
    end:      Optional[float] = None
    svc:      Optional[float] = None


class SimFCFS:
    """
    Two independent Poisson arrival streams (λE, λL) feeding ONE queue.
    Policy: FCFS — pick the waiting job with the EARLIEST booking/arrival time.
    If current_time + service_time > deadline, job ABANDONS (not served).
    — verbatim from FCFS_v7.py —
    """

    def __init__(self, pars: Dict[str, Any], seed: int = 42):
        self.pars = pars
        self.rng  = np.random.default_rng(seed)
        self.events: List[Tuple[float, int, str, Any]] = []
        self.pq: List[Tuple[float, int]] = []       # (book, pid)
        self.patients: Dict[int, PatientFCFS] = {}
        self.busy_segments: List[Tuple[float, float]] = []

        self.N = int(self.pars.get("N", 1))
        self.free_servers: List[int] = list(range(self.N))
        heapq.heapify(self.free_servers)
        self.server_mode: List[str]           = ["FREE"] * self.N
        self.server_pid:  List[Optional[int]] = [None]  * self.N
        self._seq = 0

        self.system_size_history: List[Tuple[float, int]] = [(0.0, 0)]
        self.current_system_size = 0

        self.T_max   = float(self.pars["T_max"])
        self.warmup  = float(self.pars["warmup"])
        self.p_ns    = float(self.pars["p_ns"])
        self.λE      = float(self.pars["λE"])
        self.λL      = float(self.pars["λL"])

        self.early_F      = self.pars["early_F"]
        self.early_params = dict(self.pars["early_params"])
        self.early_min    = float(self.pars["early_min"])
        self.early_max    = float(self.pars["early_max"])

        self.late_F      = self.pars["late_F"]
        self.late_params = dict(self.pars["late_params"])
        self.late_min    = float(self.pars["late_min"])
        self.late_max    = float(self.pars["late_max"])

        self.svc_E = dict(self.pars["svc_E"])
        self.svc_L = dict(self.pars["svc_L"])

        self.t   = 0.0
        self.pid = 0

        if self.λE > 0:
            self._sched(self.rng.exponential(1.0 / self.λE), 'ARR_E', None)
        if self.λL > 0:
            self._sched(self.rng.exponential(1.0 / self.λL), 'ARR_L', None)

    def _sched(self, t, typ, payload):
        self._seq += 1
        heapq.heappush(self.events, (t, self._seq, typ, payload))

    def _next_arrival(self, rate):
        return self.t + self.rng.exponential(1.0 / rate)

    def _draw_trunc_lognormal(self, mean, sd, lo, hi):
        mu, sigma = _lognormal_mu_sigma_from_mean_sd(mean, sd)
        lo = max(lo, EPS); hi = max(hi, lo + EPS)
        for _ in range(400):
            x = self.rng.lognormal(mean=mu, sigma=sigma)
            if lo <= x <= hi:
                return float(x)
        return float(self.rng.uniform(lo, hi))

    def _draw_trunc_normal(self, mean, sd, lo, hi):
        lo = max(lo, -1e18); hi = max(hi, lo + EPS); sd = max(sd, 1e-9)
        for _ in range(400):
            x = self.rng.normal(loc=mean, scale=sd)
            if lo <= x <= hi:
                return float(x)
        return float(self.rng.uniform(lo, hi))

    def _lead_time(self, cohort: str) -> float:
        if cohort == "Early":
            F, p, lo, hi = self.early_F, self.early_params, self.early_min, self.early_max
        else:
            F, p, lo, hi = self.late_F,  self.late_params,  self.late_min,  self.late_max
        lo = max(lo, EPS); hi = max(hi, lo + EPS)
        if F == "Deterministic":
            c = float(p.get("c", (lo + hi) / 2))
            return float(min(max(c, lo), hi))
        if F == "Uniform":
            return float(self.rng.uniform(lo, hi))
        if F.startswith("Normal"):
            return self._draw_trunc_normal(float(p.get("mean", (lo+hi)/2)), float(p.get("sd", max(1e-3, (hi-lo)/4))), lo, hi)
        mean_ = float(p.get("mean", (lo+hi)/2))
        return self._draw_trunc_lognormal(mean_, float(p.get("sd", max(1e-3, mean_/2))), lo, hi)

    def _service_time(self, cohort: str) -> float:
        cfg = self.svc_E if cohort == "Early" else self.svc_L
        if cfg["dist"].startswith("Discrete"):
            svc1 = float(cfg.get("svc1", 1.0)); svc2 = float(cfg.get("svc2", 2.0)); p2 = float(cfg.get("p2", 0.03))
            return float(svc2 if (self.rng.random() < p2) else svc1)
        mu = float(cfg.get("mu", 5.0)); mu = max(mu, 1e-12)
        return float(self.rng.exponential(1.0 / mu))

    def _enqueue(self, cohort: str):
        lead = self._lead_time(cohort)
        svc  = self._service_time(cohort)
        for _ in range(200):
            if svc < lead:
                break
            svc = self._service_time(cohort)
        else:
            svc = max(EPS, lead * (1.0 - 1e-6))

        p = PatientFCFS(book=self.t, id=self.pid, cohort=cohort, deadline=self.t + lead, status="WAITING", svc=svc)
        self.patients[self.pid] = p
        heapq.heappush(self.pq, (p.book, p.id))   # FCFS: (book, pid)
        self.pid += 1
        self._sched(p.deadline + EPS, 'DEAD', p.id)
        self.current_system_size += 1
        self.system_size_history.append((self.t, self.current_system_size))

    def _pop_valid(self) -> Optional[PatientFCFS]:
        while self.pq:
            b, pid = heapq.heappop(self.pq)        # FCFS: unpack 2-tuple
            p = self.patients[pid]
            if p.status == "ABANDONED" or p.status not in ("WAITING",):
                continue
            return p
        return None

    def _dispatch(self):
        while self.free_servers:
            p = self._pop_valid()
            if p is None:
                return
            svc = float(p.svc)
            if self.rng.random() < self.p_ns:
                p.status = "NO_SHOW"; p.start = self.t
                srv = heapq.heappop(self.free_servers)
                self.server_mode[srv] = "BLOCKED"; self.server_pid[srv] = None
                self._sched(self.t + svc, 'UNBLOCK', srv)
                continue
            if self.t + svc > p.deadline:
                p.status = "ABANDONED"
                self.current_system_size -= 1
                self.system_size_history.append((self.t, self.current_system_size))
                continue
            p.status = "IN_SERVICE"; p.start = self.t
            srv = heapq.heappop(self.free_servers)
            self.server_mode[srv] = "SERVING"; self.server_pid[srv] = p.id
            self._sched(self.t + svc, 'DONE', (srv, p.id))

    def run(self, T_max: float, warmup: float = 0) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, int]]]:
        self.T_max = T_max; self.warmup = warmup
        while self.events:
            t, _, typ, payload = heapq.heappop(self.events)
            self.t = t
            if typ == 'ARR_E':
                if self.t <= self.T_max:
                    self._enqueue('Early')
                    if self.λE > 0: self._sched(self._next_arrival(self.λE), 'ARR_E', None)
                self._dispatch()
            elif typ == 'ARR_L':
                if self.t <= self.T_max:
                    self._enqueue('Late')
                    if self.λL > 0: self._sched(self._next_arrival(self.λL), 'ARR_L', None)
                self._dispatch()
            elif typ == 'DEAD':
                p = self.patients.get(payload)
                if p and p.status == "WAITING":
                    p.status = "ABANDONED"
                    self.current_system_size -= 1
                    self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()
            elif typ == 'UNBLOCK':
                srv = int(payload)
                self.server_mode[srv] = "FREE"; self.server_pid[srv] = None
                heapq.heappush(self.free_servers, srv)
                self.current_system_size -= 1
                self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()
            elif typ == 'DONE':
                srv, pid = payload; srv = int(srv); pid = int(pid)
                p = self.patients.get(pid)
                if p and p.status == "IN_SERVICE" and self.server_pid[srv] == pid:
                    p.status = "SERVED"; p.end = self.t
                    self.busy_segments.append((p.start, p.end))
                    self.server_mode[srv] = "FREE"; self.server_pid[srv] = None
                    heapq.heappush(self.free_servers, srv)
                    self.current_system_size -= 1
                    self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()

        rows = []
        for p in self.patients.values():
            if p.book < self.warmup:
                continue
            if p.status not in ('SERVED', 'ABANDONED', 'NO_SHOW'):
                continue
            if p.status == 'SERVED':
                wa = p.start - p.book; wt = wa; slk = p.deadline - p.end
            elif p.status == 'ABANDONED':
                wa = p.deadline - p.book; wt = wa; slk = math.nan
            else:
                wa = math.nan; wt = math.nan; slk = math.nan
            rows.append({'cohort': p.cohort, 'status': p.status, 'booking_time': p.book,
                         'lead_time': p.deadline - p.book, 'svc': p.svc,
                         'wait_active': wa, 'wait_total': wt, 'slack': slk})
        df = pd.DataFrame(rows)
        denom         = len(df)
        served_cnt    = int((df.status == 'SERVED').sum())
        abandoned_cnt = int(df.status.isin(['ABANDONED', 'NO_SHOW']).sum())
        service_rate  = (served_cnt / denom)    if denom else math.nan
        abandon_rate  = (abandoned_cnt / denom) if denom else math.nan
        busy_after = 0.0
        for s, e in self.busy_segments:
            lo = max(s, self.warmup); hi = min(e, self.T_max)
            if hi > lo: busy_after += (hi - lo)
        util_1 = busy_after / max(1e-9, self.N * (self.T_max - self.warmup))
        kpi = {'service_rate': service_rate, 'abandon_rate': abandon_rate, 'util_1': util_1}
        return df, kpi, self.system_size_history


# =====================================================================
# SimEDF — verbatim from EDF_v7.py / EDF_2m2q_v7.py
# Priority queue tuples: (deadline, book, pid)   [3-tuple]
# =====================================================================

@dataclass(order=True)
class PatientEDF:
    book:     float
    id:       int
    cohort:   str
    deadline: float
    status:   str = "WAITING"
    start:    Optional[float] = None
    end:      Optional[float] = None
    svc:      Optional[float] = None


class SimEDF:
    """
    Two independent Poisson arrival streams (λE, λL) feeding ONE queue.
    Policy: EDF — pick the waiting job with the EARLIEST absolute deadline.
    If current_time + service_time > deadline, job ABANDONS (not served).
    — verbatim from EDF_v7.py / EDF_2m2q_v7.py —
    """

    def __init__(self, pars: Dict[str, Any], seed: int = 42):
        self.pars = pars
        self.rng  = np.random.default_rng(seed)
        self.events: List[Tuple[float, int, str, Any]] = []
        self.pq: List[Tuple[float, float, int]] = []    # (deadline, book, pid)
        self.patients: Dict[int, PatientEDF] = {}
        self.busy_segments: List[Tuple[float, float]] = []

        self.N = int(self.pars.get("N", 1))
        self.free_servers: List[int] = list(range(self.N))
        heapq.heapify(self.free_servers)
        self.server_mode: List[str]           = ["FREE"] * self.N
        self.server_pid:  List[Optional[int]] = [None]  * self.N
        self._seq = 0

        self.system_size_history: List[Tuple[float, int]] = [(0.0, 0)]
        self.current_system_size = 0

        self.T_max   = float(self.pars["T_max"])
        self.warmup  = float(self.pars["warmup"])
        self.p_ns    = float(self.pars["p_ns"])
        self.λE      = float(self.pars["λE"])
        self.λL      = float(self.pars["λL"])

        self.early_F      = self.pars["early_F"]
        self.early_params = dict(self.pars["early_params"])
        self.early_min    = float(self.pars["early_min"])
        self.early_max    = float(self.pars["early_max"])

        self.late_F      = self.pars["late_F"]
        self.late_params = dict(self.pars["late_params"])
        self.late_min    = float(self.pars["late_min"])
        self.late_max    = float(self.pars["late_max"])

        self.svc_E = dict(self.pars["svc_E"])
        self.svc_L = dict(self.pars["svc_L"])

        self.t   = 0.0
        self.pid = 0

        if self.λE > 0:
            self._sched(self.rng.exponential(1.0 / self.λE), 'ARR_E', None)
        if self.λL > 0:
            self._sched(self.rng.exponential(1.0 / self.λL), 'ARR_L', None)

    def _sched(self, t, typ, payload):
        self._seq += 1
        heapq.heappush(self.events, (t, self._seq, typ, payload))

    def _next_arrival(self, rate):
        return self.t + self.rng.exponential(1.0 / rate)

    def _draw_trunc_lognormal(self, mean, sd, lo, hi):
        mu, sigma = _lognormal_mu_sigma_from_mean_sd(mean, sd)
        lo = max(lo, EPS); hi = max(hi, lo + EPS)
        for _ in range(400):
            x = self.rng.lognormal(mean=mu, sigma=sigma)
            if lo <= x <= hi:
                return float(x)
        return float(self.rng.uniform(lo, hi))

    def _draw_trunc_normal(self, mean, sd, lo, hi):
        lo = max(lo, -1e18); hi = max(hi, lo + EPS); sd = max(sd, 1e-9)
        for _ in range(400):
            x = self.rng.normal(loc=mean, scale=sd)
            if lo <= x <= hi:
                return float(x)
        return float(self.rng.uniform(lo, hi))

    def _lead_time(self, cohort: str) -> float:
        if cohort == "Early":
            F, p, lo, hi = self.early_F, self.early_params, self.early_min, self.early_max
        else:
            F, p, lo, hi = self.late_F,  self.late_params,  self.late_min,  self.late_max
        lo = max(lo, EPS); hi = max(hi, lo + EPS)
        if F == "Deterministic":
            c = float(p.get("c", (lo + hi) / 2))
            return float(min(max(c, lo), hi))
        if F == "Uniform":
            return float(self.rng.uniform(lo, hi))
        if F.startswith("Normal"):
            return self._draw_trunc_normal(float(p.get("mean", (lo+hi)/2)), float(p.get("sd", max(1e-3, (hi-lo)/4))), lo, hi)
        mean_ = float(p.get("mean", (lo+hi)/2))
        return self._draw_trunc_lognormal(mean_, float(p.get("sd", max(1e-3, mean_/2))), lo, hi)

    def _service_time(self, cohort: str) -> float:
        cfg = self.svc_E if cohort == "Early" else self.svc_L
        if str(cfg["dist"]).startswith("Discrete"):
            svc1 = float(cfg.get("svc1", 1.0)); svc2 = float(cfg.get("svc2", 2.0)); p2 = float(cfg.get("p2", 0.03))
            return float(svc2 if (self.rng.random() < p2) else svc1)
        mu = float(cfg.get("mu", 5.0)); mu = max(mu, 1e-12)
        return float(self.rng.exponential(1.0 / mu))

    def _enqueue(self, cohort: str):
        lead = self._lead_time(cohort)
        svc  = self._service_time(cohort)
        for _ in range(200):
            if svc < lead:
                break
            svc = self._service_time(cohort)
        else:
            svc = max(EPS, lead * (1.0 - 1e-6))

        p = PatientEDF(book=self.t, id=self.pid, cohort=cohort, deadline=self.t + lead, status="WAITING", svc=svc)
        self.patients[self.pid] = p
        heapq.heappush(self.pq, (p.deadline, p.book, p.id))   # EDF: (deadline, book, pid)
        self.pid += 1
        self._sched(p.deadline + EPS, 'DEAD', p.id)
        self.current_system_size += 1
        self.system_size_history.append((self.t, self.current_system_size))

    def _pop_valid(self) -> Optional[PatientEDF]:
        while self.pq:
            d, b, pid = heapq.heappop(self.pq)                # EDF: unpack 3-tuple
            p = self.patients[pid]
            if p.status == "ABANDONED" or p.status not in ("WAITING",):
                continue
            return p
        return None

    def _dispatch(self):
        while self.free_servers:
            p = self._pop_valid()
            if p is None:
                return
            svc = float(p.svc)
            if self.rng.random() < self.p_ns:
                p.status = "NO_SHOW"; p.start = self.t
                srv = heapq.heappop(self.free_servers)
                self.server_mode[srv] = "BLOCKED"; self.server_pid[srv] = None
                self._sched(self.t + svc, 'UNBLOCK', srv)
                continue
            if self.t + svc > p.deadline:
                p.status = "ABANDONED"
                self.current_system_size -= 1
                self.system_size_history.append((self.t, self.current_system_size))
                continue
            p.status = "IN_SERVICE"; p.start = self.t
            srv = heapq.heappop(self.free_servers)
            self.server_mode[srv] = "SERVING"; self.server_pid[srv] = p.id
            self._sched(self.t + svc, 'DONE', (srv, p.id))

    def run(self, T_max: float, warmup: float = 0) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, int]]]:
        self.T_max = T_max; self.warmup = warmup
        while self.events:
            t, _, typ, payload = heapq.heappop(self.events)
            self.t = t
            if typ == 'ARR_E':
                if self.t <= self.T_max:
                    self._enqueue('Early')
                    if self.λE > 0: self._sched(self._next_arrival(self.λE), 'ARR_E', None)
                self._dispatch()
            elif typ == 'ARR_L':
                if self.t <= self.T_max:
                    self._enqueue('Late')
                    if self.λL > 0: self._sched(self._next_arrival(self.λL), 'ARR_L', None)
                self._dispatch()
            elif typ == 'DEAD':
                p = self.patients.get(payload)
                if p and p.status == "WAITING":
                    p.status = "ABANDONED"
                    self.current_system_size -= 1
                    self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()
            elif typ == 'UNBLOCK':
                srv = int(payload)
                self.server_mode[srv] = "FREE"; self.server_pid[srv] = None
                heapq.heappush(self.free_servers, srv)
                self.current_system_size -= 1
                self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()
            elif typ == 'DONE':
                srv, pid = payload; srv = int(srv); pid = int(pid)
                p = self.patients.get(pid)
                if p and p.status == "IN_SERVICE" and self.server_pid[srv] == pid:
                    p.status = "SERVED"; p.end = self.t
                    self.busy_segments.append((p.start, p.end))
                    self.server_mode[srv] = "FREE"; self.server_pid[srv] = None
                    heapq.heappush(self.free_servers, srv)
                    self.current_system_size -= 1
                    self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()

        rows = []
        for p in self.patients.values():
            if p.book < self.warmup:
                continue
            if p.status not in ('SERVED', 'ABANDONED', 'NO_SHOW'):
                continue
            if p.status == 'SERVED':
                wa = p.start - p.book; wt = wa; slk = p.deadline - p.end
            elif p.status == 'ABANDONED':
                wa = p.deadline - p.book; wt = wa; slk = math.nan
            else:
                wa = math.nan; wt = math.nan; slk = math.nan
            rows.append({'cohort': p.cohort, 'status': p.status, 'booking_time': p.book,
                         'lead_time': p.deadline - p.book, 'svc': p.svc,
                         'wait_active': wa, 'wait_total': wt, 'slack': slk})
        df = pd.DataFrame(rows)
        denom         = len(df)
        served_cnt    = int((df.status == 'SERVED').sum())
        abandoned_cnt = int(df.status.isin(['ABANDONED', 'NO_SHOW']).sum())
        service_rate  = (served_cnt / denom)    if denom else math.nan
        abandon_rate  = (abandoned_cnt / denom) if denom else math.nan
        busy_after = 0.0
        for s, e in self.busy_segments:
            lo = max(s, self.warmup); hi = min(e, self.T_max)
            if hi > lo: busy_after += (hi - lo)
        if self.N <= 0:
            util_1 = math.nan
        else:
            util_1 = busy_after / max(1e-9, self.N * (self.T_max - self.warmup))
        kpi = {'service_rate': service_rate, 'abandon_rate': abandon_rate, 'util_1': util_1}
        return df, kpi, self.system_size_history

    def run_counts(self, T_max: float, warmup: float = 0) -> Dict[str, int]:
        """
        Fast path for γ-sweeps: run the same simulation but only return outcome counts.
        — verbatim from p1.py —
        """
        self.T_max = float(T_max); self.warmup = float(warmup)
        while self.events:
            t, _, typ, payload = heapq.heappop(self.events)
            self.t = float(t)
            if typ == "ARR_E":
                if self.t <= self.T_max:
                    self._enqueue("Early")
                    if self.λE > 0: self._sched(self._next_arrival(self.λE), "ARR_E", None)
                self._dispatch()
            elif typ == "ARR_L":
                if self.t <= self.T_max:
                    self._enqueue("Late")
                    if self.λL > 0: self._sched(self._next_arrival(self.λL), "ARR_L", None)
                self._dispatch()
            elif typ == "DEAD":
                p = self.patients.get(int(payload))
                if p and p.status == "WAITING":
                    p.status = "ABANDONED"
                    self.current_system_size -= 1
                    self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()
            elif typ == "UNBLOCK":
                srv = int(payload)
                self.server_mode[srv] = "FREE"; self.server_pid[srv] = None
                heapq.heappush(self.free_servers, srv)
                self.current_system_size -= 1
                self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()
            elif typ == "DONE":
                srv, pid = payload; srv = int(srv); pid = int(pid)
                p = self.patients.get(pid)
                if p and p.status == "IN_SERVICE" and self.server_pid[srv] == pid:
                    p.status = "SERVED"; p.end = self.t
                    self.busy_segments.append((float(p.start), float(p.end)))
                    self.server_mode[srv] = "FREE"; self.server_pid[srv] = None
                    heapq.heappush(self.free_servers, srv)
                    self.current_system_size -= 1
                    self.system_size_history.append((self.t, self.current_system_size))
                self._dispatch()

        out = {"Early_served": 0, "Early_abandoned": 0, "Early_no_show": 0,
               "Late_served": 0,  "Late_abandoned": 0,  "Late_no_show": 0}
        for p in self.patients.values():
            if p.book < self.warmup:
                continue
            if p.status == "SERVED":
                out["Early_served"   if p.cohort == "Early" else "Late_served"]   += 1
            elif p.status == "ABANDONED":
                out["Early_abandoned" if p.cohort == "Early" else "Late_abandoned"] += 1
            elif p.status == "NO_SHOW":
                out["Early_no_show"  if p.cohort == "Early" else "Late_no_show"]  += 1
        return out


# =====================================================================
# Reservation utilities — verbatim from EDF_2m2q_v7.py
# =====================================================================

def combine_histories(
    h1: List[Tuple[float, int]], h2: List[Tuple[float, int]]
) -> List[Tuple[float, int]]:
    """
    Sum two right-continuous step functions given as (time, value) lists.
    O(len(h1) + len(h2)) two-pointer merge.
    — verbatim from EDF_2m2q_v7.py —
    """
    if not h1 and not h2: return []
    if not h1: return list(h2)
    if not h2: return list(h1)

    i, j = 0, 0
    cur1, cur2 = h1[0][1], h2[0][1]
    res: List[Tuple[float, int]] = [(0.0, int(cur1 + cur2))]

    t1 = h1[1][0] if len(h1) > 1 else float('inf')
    t2 = h2[1][0] if len(h2) > 1 else float('inf')

    while (i + 1) < len(h1) or (j + 1) < len(h2):
        t_next = t1 if t1 < t2 else t2
        if (i + 1) < len(h1) and math.isclose(h1[i+1][0], t_next, rel_tol=0.0, abs_tol=0.0):
            while (i + 1) < len(h1) and math.isclose(h1[i+1][0], t_next, rel_tol=0.0, abs_tol=0.0):
                i += 1; cur1 = h1[i][1]
            t1 = h1[i+1][0] if (i+1) < len(h1) else float('inf')
        if (j + 1) < len(h2) and math.isclose(h2[j+1][0], t_next, rel_tol=0.0, abs_tol=0.0):
            while (j + 1) < len(h2) and math.isclose(h2[j+1][0], t_next, rel_tol=0.0, abs_tol=0.0):
                j += 1; cur2 = h2[j][1]
            t2 = h2[j+1][0] if (j+1) < len(h2) else float('inf')
        if res and math.isclose(res[-1][0], t_next, rel_tol=0.0, abs_tol=0.0):
            res[-1] = (t_next, int(cur1 + cur2))
        else:
            res.append((t_next, int(cur1 + cur2)))
    return res


def run_two_queue_replication(
    params_base: Dict[str, Any],
    gamma: float,
    N_total: int,
    seed0: int,
    rep_index: int,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, int]]]:
    """
    One replication of the two-dedicated-queue EDF policy.
    Stochastic rounding + two independent SimEDF runs.
    — verbatim from EDF_2m2q_v7.py —
    """
    x = float(gamma) * N_total; k = int(math.floor(x)); f = x - k
    rng_split = np.random.default_rng(seed0 + 10_000 + rep_index)
    NE = k + (1 if (rng_split.random() < f) else 0)
    NE = int(min(max(NE, 0), N_total))
    NL = int(N_total - NE)

    pars_E = dict(params_base); pars_E["λL"] = 0.0; pars_E["N"] = NE
    pars_L = dict(params_base); pars_L["λE"] = 0.0; pars_L["N"] = NL

    T_max  = float(params_base["T_max"])
    warmup = float(params_base["warmup"])

    dfE, kpiE, histE = SimEDF(pars_E, seed0 + 2 * rep_index).run(T_max, warmup=warmup)
    dfL, kpiL, histL = SimEDF(pars_L, seed0 + 2 * rep_index + 1).run(T_max, warmup=warmup)

    df = pd.concat([dfE, dfL], ignore_index=True)

    denom         = len(df)
    served_cnt    = int((df.status == 'SERVED').sum())
    abandoned_cnt = int(df.status.isin(['ABANDONED', 'NO_SHOW']).sum())
    service_rate  = (served_cnt / denom)    if denom else math.nan
    abandon_rate  = (abandoned_cnt / denom) if denom else math.nan

    if N_total <= 0:
        util_total = math.nan
    else:
        util_total = 0.0; wsum = 0
        if NE > 0 and not pd.isna(kpiE["util_1"]):
            util_total += kpiE["util_1"] * NE; wsum += NE
        if NL > 0 and not pd.isna(kpiL["util_1"]):
            util_total += kpiL["util_1"] * NL; wsum += NL
        util_total = (util_total / wsum) if wsum > 0 else math.nan

    kpi = {'service_rate': service_rate, 'abandon_rate': abandon_rate, 'util_1': util_total}
    combined_hist = combine_histories(histE, histL)
    return df, kpi, combined_hist


# =====================================================================
# γ-sweep — verbatim from p1.py, parallelised across (γ, rep) pairs
# =====================================================================

def _sweep_one_rep(args: tuple) -> tuple:
    """
    Top-level worker for ProcessPoolExecutor.
    CRN across γ: a shared patient trace is generated from seed0+i (depends
    only on the replication index, NOT on γ).  All γ values in the same
    replication therefore see identical arrival times, lead times, and service
    times; only the server allocation NE/NL changes with γ.

    Cost is count-based: cE × actual_aband_E + cL × actual_aband_L
    (not the rate proxy λ × p).

    args = (g, i, N_total, params_base, seed0)
    returns (g, pE, pL, aband_E_count, aband_L_count)
    """
    g, i, N_total, params_base, seed0 = args
    g = float(g)

    # ── Build trace-generator params (translate λ key names) ───────────
    pb_trace         = dict(params_base)
    pb_trace["lamE"] = float(params_base["λE"])
    pb_trace["lamL"] = float(params_base["λL"])
    warmup = float(params_base["warmup"])
    p_ns   = float(params_base.get("p_ns", 0.0))

    # ── Generate shared trace for this replication (CRN across γ) ──────
    # seed depends only on i — identical trace for every γ in this rep.
    # Offset by 500_000 so γ* search patients are independent of final runs
    # (which use seed0 + i), giving a proper out-of-sample evaluation.
    trace  = generate_common_trace_v7_style(pb_trace, seed=seed0 + 500_000 + i)
    traceE = [tp for tp in trace if tp.cohort == "Early"]
    traceL = [tp for tp in trace if tp.cohort == "Late"]

    # Count eligible patients (post-warmup) for count-based cost
    n_early = sum(1 for tp in traceE if tp.arrival >= warmup)
    n_late  = sum(1 for tp in traceL if tp.arrival >= warmup)

    # ── Stochastic rounding for server split (unbiased) ─────────────────
    x = g * N_total; k = int(math.floor(x)); f = x - k
    rng_split = np.random.default_rng(seed0 + 10_000 + i + int(round(1e6 * g)))
    NE = k + (1 if (rng_split.random() < f) else 0)
    NE = int(min(max(NE, 0), N_total)); NL = int(N_total - NE)

    # ── Run EDF on each cohort's dedicated sub-trace ───────────────────
    # No-show seeds do NOT include g (preserves CRN across γ)
    pE, _ = simulate_pooled_queue_v7_style(
        traceE, "EDF", NE, warmup, p_ns, seed=seed0 + 200_000 + i)
    _, pL = simulate_pooled_queue_v7_style(
        traceL, "EDF", NL, warmup, p_ns, seed=seed0 + 300_000 + i)

    # ── Actual abandonment counts ───────────────────────────────────────
    # pE = aband_E / n_early  →  aband_E = pE × n_early  (exact integer)
    aband_E = (float(pE) * n_early if (not math.isnan(float(pE)) and n_early > 0)
               else float("nan"))
    aband_L = (float(pL) * n_late  if (not math.isnan(float(pL)) and n_late  > 0)
               else float("nan"))

    return g, float(pE), float(pL), aband_E, aband_L


def sweep_gamma(
    gammas: tuple,
    params_base: dict,
    seed0: int,
    cE: float,
    cL: float,
    Delta: float,
    n_workers: int = 0,
) -> pd.DataFrame:
    """
    Run the two-dedicated-queue EDF simulation over a grid of γ values.

    CRN across γ: each replication i uses the same patient trace for all γ,
    so differences in outcome are attributable to server allocation only.

    Cost is count-based: for each replication, C_i = cE×aband_E_i + cL×aband_L_i.
    γ* minimises mean(C_i) subject to mean |pE − pL| ≤ Δ.

    n_workers: number of parallel processes (0 = all available CPUs).
    """
    N_total = int(params_base["N"])
    n_runs  = int(params_base["n_runs"])
    workers = n_workers if n_workers > 0 else (os.cpu_count() or 1)

    # One task per (gamma, replication) pair — all independent
    tasks = [
        (float(g), i, N_total, params_base, seed0)
        for g in gammas
        for i in range(n_runs)
    ]

    # Collect raw (pE, pL, aband_E, aband_L) per gamma
    raw: Dict[float, List[tuple]] = {float(g): [] for g in gammas}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for g_out, pE, pL, aband_E, aband_L in pool.map(_sweep_one_rep, tasks):
            raw[g_out].append((pE, pL, aband_E, aband_L))

    # Aggregate per gamma
    rows = []
    lamE = float(params_base["λE"]); lamL = float(params_base["λL"])
    for g in gammas:
        g = float(g)
        pairs = raw[g]
        pE = float(np.nanmean([p[0] for p in pairs]))
        pL = float(np.nanmean([p[1] for p in pairs]))

        # Count-based cost: mean over reps of (cE×aband_E + cL×aband_L)
        C_runs = [cE * p[2] + cL * p[3] for p in pairs
                  if not (math.isnan(p[2]) or math.isnan(p[3]))]
        C = float(np.mean(C_runs)) if C_runs else float("nan")

        # Rate-based quantities kept for display/diagnostics only
        AE = lamE * pE if not math.isnan(pE) else float("nan")
        AL = lamL * pL if not math.isnan(pL) else float("nan")

        phi        = pE - pL if (not math.isnan(pE) and not math.isnan(pL)) else float("nan")
        ineq_level = abs(phi) - Delta if not math.isnan(phi) else float("nan")
        rel_ineq   = (ineq_level / Delta) * 100.0 if (Delta > 0 and not math.isnan(ineq_level)) else float("nan")

        rows.append({"gamma": g, "pE": pE, "pL": pL, "AE": AE, "AL": AL, "C": C,
                     "phi": phi, "ineq_level": ineq_level, "rel_ineq_pct": rel_ineq})

    return pd.DataFrame(rows)


# =====================================================================
# Trace-based CRN simulation — verbatim from p2.py (Tab 3 only)
# =====================================================================

def _draw_trunc_lognormal_p2(rng, mean, sd, lo, hi):
    mu, sigma = _lognormal_mu_sigma_from_mean_sd(mean, sd)
    lo = max(float(lo), EPS); hi = max(float(hi), lo + EPS)
    for _ in range(400):
        x = float(rng.lognormal(mean=mu, sigma=sigma))
        if lo <= x <= hi: return x
    return float(rng.uniform(lo, hi))

def _draw_trunc_normal_p2(rng, mean, sd, lo, hi):
    lo = float(max(lo, -1e18)); hi = float(max(hi, lo + EPS)); sd = float(max(sd, 1e-9))
    for _ in range(400):
        x = float(rng.normal(loc=mean, scale=sd))
        if lo <= x <= hi: return x
    return float(rng.uniform(lo, hi))

def _draw_lead_time_p2(rng, F, params, lo, hi):
    lo = float(max(lo, EPS)); hi = float(max(hi, lo + EPS))
    if F == "Deterministic":
        c = float(params.get("c", (lo + hi) / 2))
        return float(min(max(c, lo), hi))
    if F == "Uniform":
        return float(rng.uniform(lo, hi))
    if F.startswith("Normal"):
        return _draw_trunc_normal_p2(rng, float(params.get("mean", (lo+hi)/2)), float(params.get("sd", max(1e-3, (hi-lo)/4))), lo, hi)
    mean_ = float(params.get("mean", (lo+hi)/2))
    return _draw_trunc_lognormal_p2(rng, mean_, float(params.get("sd", max(1e-3, mean_/2))), lo, hi)

def _draw_service_time_p2(rng, cfg):
    dist = str(cfg.get("dist", "Discrete (1 or 2 slots)"))
    if dist.startswith("Discrete"):
        svc1 = float(cfg.get("svc1", 1.0)); svc2 = float(cfg.get("svc2", 2.0)); p2 = float(cfg.get("p2", 0.03))
        return float(svc2 if (rng.random() < p2) else svc1)
    mu = float(cfg.get("mu", 5.0)); mu = max(mu, 1e-12)
    return float(rng.exponential(1.0 / mu))


@dataclass
class PatientTrace:
    pid:       int
    cohort:    str
    arrival:   float
    lead_time: float
    deadline:  float
    svc:       float


def generate_common_trace_v7_style(pars: Dict[str, Any], seed: int) -> List[PatientTrace]:
    """
    Generate a common arrival trace shared across all 3 policies (CRN).
    — verbatim from p2.py —
    """
    rng   = np.random.default_rng(int(seed))
    T_max = float(pars["T_max"])
    lamE  = float(pars["lamE"]); lamL = float(pars["lamL"])

    early_F      = str(pars["early_F"]);  early_params = dict(pars["early_params"])
    early_min    = float(pars["early_min"]); early_max = float(pars["early_max"])
    late_F       = str(pars["late_F"]);   late_params  = dict(pars["late_params"])
    late_min     = float(pars["late_min"]);  late_max  = float(pars["late_max"])
    svc_E        = dict(pars["svc_E"]); svc_L = dict(pars["svc_L"])

    out: List[PatientTrace] = []
    pid = 0

    t = 0.0
    while lamE > 0:
        t += float(rng.exponential(1.0 / lamE))
        if t > T_max: break
        lead = _draw_lead_time_p2(rng, early_F, early_params, early_min, early_max)
        svc  = _draw_service_time_p2(rng, svc_E)
        out.append(PatientTrace(pid=pid, cohort="Early", arrival=t, lead_time=lead, deadline=0.0, svc=svc))  # deadline set after unit scaling
        pid += 1

    t = 0.0
    while lamL > 0:
        t += float(rng.exponential(1.0 / lamL))
        if t > T_max: break
        lead = _draw_lead_time_p2(rng, late_F, late_params, late_min, late_max)
        svc  = _draw_service_time_p2(rng, svc_L)
        out.append(PatientTrace(pid=pid, cohort="Late", arrival=t, lead_time=lead, deadline=0.0, svc=svc))  # deadline set after unit scaling
        pid += 1

    out.sort(key=lambda p: (p.arrival, p.pid))
    return out


def shift_early_lead_times(trace: List[PatientTrace], delta: float) -> List[PatientTrace]:
    """
    Return a new trace with each Early patient's lead_time and deadline shifted
    by delta.  Late patients are returned unchanged.
    Service time is capped below lead_time if the shift would violate svc < lt.
    """
    if delta == 0.0:
        return trace
    result: List[PatientTrace] = []
    for tp in trace:
        if tp.cohort == "Early":
            new_lt = max(EPS, tp.lead_time + delta)
            result.append(PatientTrace(
                pid=tp.pid, cohort=tp.cohort, arrival=tp.arrival,
                lead_time=new_lt, deadline=tp.arrival + new_lt + tp.svc, svc=tp.svc))
        else:
            result.append(tp)
    return result


@dataclass
class _SimPatient:
    pid:      int
    cohort:   str
    arrival:  float
    deadline: float
    svc:      float
    status:   str = "WAITING"


def simulate_pooled_queue_v7_style(
    trace: List[PatientTrace],
    policy: str,
    N: int,
    warmup: float,
    p_ns: float,
    seed: int,
) -> Tuple[float, float]:
    """
    Discrete-event sim on a shared trace, pooled queue (FCFS or EDF).
    — verbatim from p2.py —
    Priority key: FCFS → (arrival, deadline, pid),  EDF → (deadline, arrival, pid)
    """
    rng = np.random.default_rng(int(seed))
    patients: Dict[int, _SimPatient] = {
        tp.pid: _SimPatient(tp.pid, tp.cohort, tp.arrival, tp.deadline, tp.svc)
        for tp in trace
    }
    events: List[Tuple[float, int, str, Any]] = []
    seq = 0
    for tp in trace:
        seq += 1; heapq.heappush(events, (tp.arrival, seq, "ARR", tp.pid))
        seq += 1; heapq.heappush(events, (tp.deadline + EPS, seq, "DEAD", tp.pid))

    wait: list = []
    busy = 0

    def push_wait(pid_):
        p_ = patients[pid_]
        if policy == "EDF":
            heapq.heappush(wait, (p_.deadline, p_.arrival, pid_))
        else:
            heapq.heappush(wait, (p_.arrival, p_.deadline, pid_))

    def pop_wait(now):
        while wait:
            _, _, pid_ = heapq.heappop(wait)
            p_ = patients[pid_]
            if p_.status != "WAITING": continue
            if now > p_.deadline + 1e-12:
                p_.status = "ABANDONED"; continue
            return pid_
        return None

    def dispatch(now):
        nonlocal busy, seq
        while busy < N:
            pid_ = pop_wait(now)
            if pid_ is None: return
            p_ = patients[pid_]
            svc_ = float(p_.svc)
            if rng.random() < float(p_ns):
                p_.status = "NO_SHOW"; busy += 1
                seq += 1; heapq.heappush(events, (now + svc_, seq, "UNBLOCK", None))
                continue
            if now + svc_ > p_.deadline + 1e-12:
                p_.status = "ABANDONED"; continue
            p_.status = "IN_SERVICE"; busy += 1
            seq += 1; heapq.heappush(events, (now + svc_, seq, "DONE", pid_))

    while events:
        now, _, typ, payload = heapq.heappop(events)
        if typ == "ARR":
            push_wait(int(payload)); dispatch(now)
        elif typ == "DEAD":
            p_ = patients[int(payload)]
            if p_.status == "WAITING": p_.status = "ABANDONED"
            dispatch(now)
        elif typ == "UNBLOCK":
            busy = max(0, busy - 1); dispatch(now)
        elif typ == "DONE":
            p_ = patients[int(payload)]
            if p_.status == "IN_SERVICE": p_.status = "SERVED"
            busy = max(0, busy - 1); dispatch(now)

    def frac_abandon(cohort_):
        arr = [p for p in patients.values() if p.cohort == cohort_ and p.arrival >= warmup]
        if not arr: return float("nan")
        return float(np.mean([1.0 if p.status in ("ABANDONED", "NO_SHOW") else 0.0 for p in arr]))

    return frac_abandon("Early"), frac_abandon("Late")


def simulate_reservation_2queue_v7_style(
    trace: List[PatientTrace],
    gamma: float,
    N_total: int,
    warmup: float,
    p_ns: float,
    seed_split: int,
    seed_run: int,
) -> Tuple[float, float]:
    """
    Two dedicated queues, EDF in each, no cross-serving, stochastic rounding.
    — verbatim from p2.py —
    """
    gamma = float(min(max(gamma, 0.0), 1.0))
    x = gamma * N_total; k = int(math.floor(x)); f = x - k
    rng_s = np.random.default_rng(int(seed_split))
    NE = k + (1 if (rng_s.random() < f) else 0)
    NE = int(min(max(NE, 0), N_total)); NL = int(N_total - NE)

    traceE = [tp for tp in trace if tp.cohort == "Early"]
    traceL = [tp for tp in trace if tp.cohort == "Late"]

    pE, _ = simulate_pooled_queue_v7_style(traceE, "EDF", NE, warmup, p_ns, seed_run + 0)
    _, pL  = simulate_pooled_queue_v7_style(traceL, "EDF", NL, warmup, p_ns, seed_run + 1)
    return pE, pL


def simulate_pooled_queue_full(
    trace: List[PatientTrace],
    policy: str,
    N: int,
    warmup: float,
    p_ns: float,
    seed: int,
    T_max: float,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, int]]]:
    """
    Extended version of simulate_pooled_queue_v7_style that returns the
    full patient-level DataFrame, kpi dict, and N(t) history.
    Output schema matches SimEDF.run() exactly so _run_stats() works unchanged.
    """
    rng = np.random.default_rng(int(seed))

    @dataclass
    class _P:
        pid: int; cohort: str; arrival: float; deadline: float
        svc: float; lead_time: float
        status: str = "WAITING"
        start: Optional[float] = None
        end_t: Optional[float] = None

    patients: Dict[int, _P] = {
        tp.pid: _P(tp.pid, tp.cohort, tp.arrival, tp.deadline, tp.svc, tp.lead_time)
        for tp in trace
    }
    events: List[Tuple[float, int, str, Any]] = []
    seq = 0
    for tp in trace:
        seq += 1; heapq.heappush(events, (tp.arrival,          seq, "ARR",  tp.pid))
        seq += 1; heapq.heappush(events, (tp.deadline + EPS,   seq, "DEAD", tp.pid))

    wait: list = []
    busy = 0
    busy_segments: List[Tuple[float, float]] = []
    system_size_history: List[Tuple[float, int]] = [(0.0, 0)]
    current_size = 0

    def push_wait(pid_):
        p_ = patients[pid_]
        if policy == "EDF":
            heapq.heappush(wait, (p_.deadline, p_.arrival, pid_))
        else:
            heapq.heappush(wait, (p_.arrival, p_.deadline, pid_))

    def pop_wait(now):
        nonlocal current_size
        while wait:
            _, _, pid_ = heapq.heappop(wait)
            p_ = patients[pid_]
            if p_.status != "WAITING": continue
            if now > p_.deadline + 1e-12:
                p_.status = "ABANDONED"
                current_size -= 1; system_size_history.append((now, current_size))
                continue
            return pid_
        return None

    def dispatch(now):
        nonlocal busy, seq, current_size
        while busy < N:
            pid_ = pop_wait(now)
            if pid_ is None: return
            p_ = patients[pid_]
            if rng.random() < float(p_ns):
                p_.status = "NO_SHOW"; p_.start = now; busy += 1
                seq += 1; heapq.heappush(events, (now + p_.svc, seq, "UNBLOCK", None))
                current_size -= 1; system_size_history.append((now, current_size))
                continue
            if now + p_.svc > p_.deadline + 1e-12:
                p_.status = "ABANDONED"
                current_size -= 1; system_size_history.append((now, current_size))
                continue
            p_.status = "IN_SERVICE"; p_.start = now; busy += 1
            seq += 1; heapq.heappush(events, (now + p_.svc, seq, "DONE", pid_))

    while events:
        now, _, typ, payload = heapq.heappop(events)
        if typ == "ARR":
            p_ = patients[int(payload)]
            if p_.arrival <= T_max:
                current_size += 1; system_size_history.append((now, current_size))
                push_wait(int(payload))
            dispatch(now)
        elif typ == "DEAD":
            p_ = patients[int(payload)]
            if p_.status == "WAITING":
                p_.status = "ABANDONED"
                current_size -= 1; system_size_history.append((now, current_size))
            dispatch(now)
        elif typ == "UNBLOCK":
            busy = max(0, busy - 1); dispatch(now)
        elif typ == "DONE":
            p_ = patients[int(payload)]
            if p_.status == "IN_SERVICE":
                p_.status = "SERVED"; p_.end_t = now
                busy_segments.append((p_.start, now))
                busy = max(0, busy - 1)
                current_size -= 1; system_size_history.append((now, current_size))
            dispatch(now)

    rows = []
    for p in patients.values():
        if p.arrival < warmup: continue
        if p.status not in ("SERVED", "ABANDONED", "NO_SHOW"): continue
        if p.status == "SERVED":
            wa = p.start - p.arrival; wt = wa; slk = p.deadline - p.end_t
        elif p.status == "ABANDONED":
            wa = p.deadline - p.arrival; wt = wa; slk = math.nan
        else:
            wa = math.nan; wt = math.nan; slk = math.nan
        rows.append({"cohort": p.cohort, "status": p.status,
                     "booking_time": p.arrival, "lead_time": p.lead_time,
                     "svc": p.svc, "wait_active": wa, "wait_total": wt, "slack": slk})

    df = pd.DataFrame(rows)
    denom = len(df)
    served_cnt    = int((df.status == "SERVED").sum())   if denom else 0
    abandoned_cnt = int(df.status.isin(["ABANDONED", "NO_SHOW"]).sum()) if denom else 0
    service_rate  = (served_cnt    / denom) if denom else math.nan
    abandon_rate  = (abandoned_cnt / denom) if denom else math.nan
    busy_after = sum(max(0.0, min(e, T_max) - max(s, warmup))
                     for s, e in busy_segments)
    util_1 = busy_after / max(1e-9, N * (T_max - warmup)) if N > 0 else math.nan
    kpi = {"service_rate": service_rate, "abandon_rate": abandon_rate, "util_1": util_1}
    return df, kpi, system_size_history


def simulate_reservation_2queue_full(
    trace: List[PatientTrace],
    gamma: float,
    N_total: int,
    warmup: float,
    p_ns: float,
    seed_split: int,
    seed_run: int,
    T_max: float,
    NE_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, int]]]:
    """
    Two dedicated EDF queues on a shared trace, returning full patient-level
    DataFrame, kpi dict, and combined N(t) history.
    Matches run_two_queue_replication output format.

    NE_override: if provided, use this fixed integer NE (deterministic floor path
    for display metrics). If None, use stochastic rounding (search path).
    """
    gamma = float(min(max(gamma, 0.0), 1.0))
    if NE_override is not None:
        NE = int(min(max(NE_override, 0), N_total))
    else:
        x = gamma * N_total; k = int(math.floor(x)); f = x - k
        rng_s = np.random.default_rng(int(seed_split))
        NE = k + (1 if (rng_s.random() < f) else 0)
        NE = int(min(max(NE, 0), N_total))
    NL = int(N_total - NE)

    traceE = [tp for tp in trace if tp.cohort == "Early"]
    traceL = [tp for tp in trace if tp.cohort == "Late"]

    dfE, kpiE, histE = simulate_pooled_queue_full(traceE, "EDF", NE, warmup, p_ns, seed_run + 0, T_max)
    dfL, kpiL, histL = simulate_pooled_queue_full(traceL, "EDF", NL, warmup, p_ns, seed_run + 1, T_max)

    df    = pd.concat([dfE, dfL], ignore_index=True)
    denom = len(df)
    served_cnt    = int((df.status == "SERVED").sum())    if denom else 0
    abandoned_cnt = int(df.status.isin(["ABANDONED", "NO_SHOW"]).sum()) if denom else 0
    service_rate  = (served_cnt    / denom) if denom else math.nan
    abandon_rate  = (abandoned_cnt / denom) if denom else math.nan
    if N_total <= 0:
        util_1 = math.nan
    else:
        util_1 = 0.0; wsum = 0
        if NE > 0 and not math.isnan(kpiE.get("util_1", math.nan)):
            util_1 += kpiE["util_1"] * NE; wsum += NE
        if NL > 0 and not math.isnan(kpiL.get("util_1", math.nan)):
            util_1 += kpiL["util_1"] * NL; wsum += NL
        util_1 = (util_1 / wsum) if wsum > 0 else math.nan
    kpi  = {"service_rate": service_rate, "abandon_rate": abandon_rate, "util_1": util_1}
    hist = combine_histories(histE, histL)
    return df, kpi, hist


def run_stream2(
    pars: Dict[str, Any],
    n_runs: int,
    seed0: int,
    N: int,
    Delta: float,
    ratios: List[float],
    c_pairs: List[Tuple[float, float]],
    gamma_grid: np.ndarray,
) -> Dict[str, Any]:
    """
    Full p2 computation with CRN across all 3 policies.
    — verbatim from p2.py —
    """
    warmup = float(pars["warmup"])
    lamE   = float(pars["lamE"]); lamL = float(pars["lamL"])
    p_ns   = float(pars["p_ns"])
    G      = len(gamma_grid)

    rec = {"FCFS": {"pE": [], "pL": []}, "EDF": {"pE": [], "pL": []}}
    res_by_gamma = {float(g): {"pE": [], "pL": []} for g in gamma_grid}

    for i in range(n_runs):
        trace = generate_common_trace_v7_style(pars, seed=seed0 + i)

        pE_fcfs, pL_fcfs = simulate_pooled_queue_v7_style(trace, "FCFS", N=N, warmup=warmup, p_ns=p_ns, seed=seed0 + 100_000 + i)
        pE_edf,  pL_edf  = simulate_pooled_queue_v7_style(trace, "EDF",  N=N, warmup=warmup, p_ns=p_ns, seed=seed0 + 200_000 + i)

        rec["FCFS"]["pE"].append(pE_fcfs); rec["FCFS"]["pL"].append(pL_fcfs)
        rec["EDF"]["pE"].append(pE_edf);   rec["EDF"]["pL"].append(pL_edf)

        for j, g in enumerate(gamma_grid):
            g = float(g)
            pE_res, pL_res = simulate_reservation_2queue_v7_style(
                trace, gamma=g, N_total=N, warmup=warmup, p_ns=p_ns,
                seed_split=seed0 + 300_000 + i * G + j,
                seed_run=seed0   + 400_000 + i * G + j,
            )
            res_by_gamma[g]["pE"].append(pE_res)
            res_by_gamma[g]["pL"].append(pL_res)

    def _sd(vals):
        arr = np.array(vals, dtype=float)
        return float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else float("nan")

    gamma_stats = {}
    for g in gamma_grid:
        g = float(g)
        pE_m = float(np.nanmean(res_by_gamma[g]["pE"]))
        pL_m = float(np.nanmean(res_by_gamma[g]["pL"]))
        gamma_stats[g] = {"pE_m": pE_m, "pL_m": pL_m, "absPhi": abs(pE_m - pL_m)}

    gstars: Dict[str, float] = {}
    for r, (cE, cL) in zip(ratios, c_pairs):
        best_feas = None; best_infeas = None
        for g in gamma_grid:
            g = float(g); s_g = gamma_stats[g]
            C    = float(cE) * lamE * s_g["pE_m"] + float(cL) * lamL * s_g["pL_m"]
            ineq = s_g["absPhi"] - Delta
            if ineq <= 1e-12:
                if best_feas is None or C < best_feas[1]: best_feas = (g, C, ineq)
            else:
                if best_infeas is None or ineq < best_infeas[2] - 1e-12 or (abs(ineq - best_infeas[2]) <= 1e-12 and C < best_infeas[1]):
                    best_infeas = (g, C, ineq)
        gstars[str(r)] = float(best_feas[0] if best_feas is not None else best_infeas[0])

    report_rows = [
        {"policy": "FCFS (pooled)",  "ratio": "—", "gamma_star": "—",
         "p_ab_E (%)": mean_sd_str(rec["FCFS"]["pE"], scale=100.0),
         "p_ab_L (%)": mean_sd_str(rec["FCFS"]["pL"], scale=100.0)},
        {"policy": "EDF (pooled)",   "ratio": "—", "gamma_star": "—",
         "p_ab_E (%)": mean_sd_str(rec["EDF"]["pE"],  scale=100.0),
         "p_ab_L (%)": mean_sd_str(rec["EDF"]["pL"],  scale=100.0)},
    ]

    rows = []
    for r, (cE, cL) in zip(ratios, c_pairs):
        g_star  = float(gstars[str(r)])
        pE_list = res_by_gamma[g_star]["pE"]; pL_list = res_by_gamma[g_star]["pL"]
        pE_m    = float(np.nanmean(pE_list)); pL_m = float(np.nanmean(pL_list))

        report_rows.append({
            "policy": "Reservation (2-queue EDF)", "ratio": f"{float(r):g}",
            "gamma_star": f"{g_star:.3f}",
            "p_ab_E (%)": mean_sd_str(pE_list, scale=100.0),
            "p_ab_L (%)": mean_sd_str(pL_list, scale=100.0),
        })

        def _m(pol, k):
            return float(np.nanmean(rec[pol][k]))

        m_fcfs = compute_metrics(_m("FCFS","pE"), _m("FCFS","pL"), lamE, lamL, float(cE), float(cL), Delta)
        m_edf  = compute_metrics(_m("EDF","pE"),  _m("EDF","pL"),  lamE, lamL, float(cE), float(cL), Delta)
        m_res  = compute_metrics(pE_m, pL_m, lamE, lamL, float(cE), float(cL), Delta)

        C_fcfs_runs = [float(cE)*lamE*pE + float(cL)*lamL*pL for pE,pL in zip(rec["FCFS"]["pE"], rec["FCFS"]["pL"])]
        C_edf_runs  = [float(cE)*lamE*pE + float(cL)*lamL*pL for pE,pL in zip(rec["EDF"]["pE"],  rec["EDF"]["pL"])]
        C_res_runs  = [float(cE)*lamE*pE + float(cL)*lamL*pL for pE,pL in zip(pE_list, pL_list)]
        absPhi_fcfs_runs = [abs(pE-pL) for pE,pL in zip(rec["FCFS"]["pE"], rec["FCFS"]["pL"])]
        absPhi_edf_runs  = [abs(pE-pL) for pE,pL in zip(rec["EDF"]["pE"],  rec["EDF"]["pL"])]
        absPhi_res_runs  = [abs(pE-pL) for pE,pL in zip(pE_list, pL_list)]

        rows.append({
            "ratio": float(r), "cE": float(cE), "cL": float(cL), "gamma_star_sim": g_star,
            "pE_FCFS_mean": _m("FCFS","pE"), "pE_EDF_mean": _m("EDF","pE"), "pE_RES_mean": pE_m,
            "pL_FCFS_mean": _m("FCFS","pL"), "pL_EDF_mean": _m("EDF","pL"), "pL_RES_mean": pL_m,
            "C_FCFS": m_fcfs["C"],    "C_EDF": m_edf["C"],    "C_RES": m_res["C"],
            "absPhi_FCFS": m_fcfs["absPhi"], "absPhi_EDF": m_edf["absPhi"], "absPhi_RES": m_res["absPhi"],
            "ineq_FCFS": m_fcfs["ineq"],     "ineq_EDF": m_edf["ineq"],     "ineq_RES": m_res["ineq"],
            "rel_ineq_FCFS": m_fcfs["rel_ineq"], "rel_ineq_EDF": m_edf["rel_ineq"], "rel_ineq_RES": m_res["rel_ineq"],
            "C_FCFS_sd": _sd(C_fcfs_runs), "C_EDF_sd": _sd(C_edf_runs), "C_RES_sd": _sd(C_res_runs),
            "absPhi_FCFS_sd": _sd(absPhi_fcfs_runs), "absPhi_EDF_sd": _sd(absPhi_edf_runs), "absPhi_RES_sd": _sd(absPhi_res_runs),
        })

    out_df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    for col in ["pE_FCFS_mean","pE_EDF_mean","pE_RES_mean","pL_FCFS_mean","pL_EDF_mean","pL_RES_mean"]:
        if col in out_df.columns:
            out_df[col.replace("_mean","_pct")] = 100.0 * out_df[col]

    return {"out_df": out_df, "report_df": pd.DataFrame(report_rows)}


# =====================================================================
# Flexible CRN comparison — any subset of 4 policies (Tab 3)
# =====================================================================

def run_flexible_comparison(
    pars: Dict[str, Any],
    n_runs: int,
    seed0: int,
    N: int,
    Delta: float,
    ratios: List[float],
    c_pairs: List[Tuple[float, float]],
    gamma_grid: np.ndarray,
    policies: List[str],
    gamma_fixed: float = 0.72,
) -> Dict[str, Any]:
    """
    CRN policy comparison for any subset of:
        "FCFS", "EDF", "Optimal Hybrid-EDF", "Hybrid-EDF"

    policies     : list of policy names to include
    gamma_fixed  : γ used for "Hybrid-EDF" (non-optimal); ignored otherwise

    Seed offsets (backward-compatible with run_stream2 where applicable):
      trace:                seed0 + i
      FCFS:                 seed0 + 100_000 + i
      EDF:                  seed0 + 200_000 + i
      Optimal Hybrid-EDF:   seed_split = seed0 + 300_000 + i*G + j
                            seed_run   = seed0 + 400_000 + i*G + j
      Hybrid-EDF (fixed γ): seed_split = seed0 + 500_000 + i
                            seed_run   = seed0 + 600_000 + i
    """
    warmup = float(pars["warmup"])
    lamE   = float(pars["lamE"]); lamL = float(pars["lamL"])
    p_ns   = float(pars["p_ns"])
    G      = len(gamma_grid)

    run_fcfs = "FCFS"               in policies
    run_edf  = "EDF"                in policies
    run_opt  = "Optimal Hybrid-EDF" in policies
    run_hyb  = "Hybrid-EDF"         in policies

    rec_fcfs: Dict[str, list] = {"pE": [], "pL": []} if run_fcfs else {}
    rec_edf:  Dict[str, list] = {"pE": [], "pL": []} if run_edf  else {}
    res_by_gamma: Dict[float, Dict[str, list]] = (
        {float(g): {"pE": [], "pL": []} for g in gamma_grid} if run_opt else {}
    )
    rec_hyb: Dict[str, list] = {"pE": [], "pL": []} if run_hyb else {}

    for i in range(n_runs):
        trace = generate_common_trace_v7_style(pars, seed=seed0 + i)

        if run_fcfs:
            pE, pL = simulate_pooled_queue_v7_style(
                trace, "FCFS", N=N, warmup=warmup, p_ns=p_ns, seed=seed0 + 100_000 + i)
            rec_fcfs["pE"].append(pE); rec_fcfs["pL"].append(pL)

        if run_edf:
            pE, pL = simulate_pooled_queue_v7_style(
                trace, "EDF",  N=N, warmup=warmup, p_ns=p_ns, seed=seed0 + 200_000 + i)
            rec_edf["pE"].append(pE); rec_edf["pL"].append(pL)

        if run_opt:
            for j, g in enumerate(gamma_grid):
                g = float(g)
                pE, pL = simulate_reservation_2queue_v7_style(
                    trace, gamma=g, N_total=N, warmup=warmup, p_ns=p_ns,
                    seed_split=seed0 + 300_000 + i * G + j,
                    seed_run  =seed0 + 400_000 + i * G + j,
                )
                res_by_gamma[g]["pE"].append(pE)
                res_by_gamma[g]["pL"].append(pL)

        if run_hyb:
            pE, pL = simulate_reservation_2queue_v7_style(
                trace, gamma=gamma_fixed, N_total=N, warmup=warmup, p_ns=p_ns,
                seed_split=seed0 + 500_000 + i,
                seed_run  =seed0 + 600_000 + i,
            )
            rec_hyb["pE"].append(pE); rec_hyb["pL"].append(pL)

    # γ*(ratio) for Optimal Hybrid-EDF
    gamma_stats: Dict[float, Dict] = {}
    if run_opt:
        for g in gamma_grid:
            g = float(g)
            pE_m = float(np.nanmean(res_by_gamma[g]["pE"]))
            pL_m = float(np.nanmean(res_by_gamma[g]["pL"]))
            gamma_stats[g] = {"pE_m": pE_m, "pL_m": pL_m, "absPhi": abs(pE_m - pL_m)}

    gstars: Dict[str, float] = {}
    if run_opt:
        for r, (cE, cL) in zip(ratios, c_pairs):
            best_feas = None; best_infeas = None
            for g in gamma_grid:
                g = float(g); s = gamma_stats[g]
                C    = float(cE) * lamE * s["pE_m"] + float(cL) * lamL * s["pL_m"]
                ineq = s["absPhi"] - Delta
                if ineq <= 1e-12:
                    if best_feas is None or C < best_feas[1]: best_feas = (g, C, ineq)
                else:
                    if (best_infeas is None or ineq < best_infeas[2] - 1e-12 or
                            (abs(ineq - best_infeas[2]) <= 1e-12 and C < best_infeas[1])):
                        best_infeas = (g, C, ineq)
            gstars[str(r)] = float(best_feas[0] if best_feas is not None else best_infeas[0])

    # Build output rows
    rows = []
    for r, (cE, cL) in zip(ratios, c_pairs):
        row: Dict[str, Any] = {"ratio": float(r), "cE": float(cE), "cL": float(cL)}

        if run_fcfs:
            pE_m = float(np.nanmean(rec_fcfs["pE"]))
            pL_m = float(np.nanmean(rec_fcfs["pL"]))
            row["pE_FCFS"] = pE_m; row["pL_FCFS"] = pL_m
            for k, v in compute_metrics(pE_m, pL_m, lamE, lamL, float(cE), float(cL), Delta).items():
                row[f"{k}_FCFS"] = v

        if run_edf:
            pE_m = float(np.nanmean(rec_edf["pE"]))
            pL_m = float(np.nanmean(rec_edf["pL"]))
            row["pE_EDF"] = pE_m; row["pL_EDF"] = pL_m
            for k, v in compute_metrics(pE_m, pL_m, lamE, lamL, float(cE), float(cL), Delta).items():
                row[f"{k}_EDF"] = v

        if run_opt:
            g_star  = float(gstars[str(r)])
            pE_list = res_by_gamma[g_star]["pE"]
            pL_list = res_by_gamma[g_star]["pL"]
            pE_m    = float(np.nanmean(pE_list))
            pL_m    = float(np.nanmean(pL_list))
            row["pE_OPT"] = pE_m; row["pL_OPT"] = pL_m; row["gamma_star"] = g_star
            for k, v in compute_metrics(pE_m, pL_m, lamE, lamL, float(cE), float(cL), Delta).items():
                row[f"{k}_OPT"] = v

        if run_hyb:
            pE_m = float(np.nanmean(rec_hyb["pE"]))
            pL_m = float(np.nanmean(rec_hyb["pL"]))
            row["pE_HYB"] = pE_m; row["pL_HYB"] = pL_m; row["gamma_fixed"] = gamma_fixed
            for k, v in compute_metrics(pE_m, pL_m, lamE, lamL, float(cE), float(cL), Delta).items():
                row[f"{k}_HYB"] = v

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    return {"out_df": out_df, "policies": policies, "gamma_fixed": gamma_fixed, "gstars": gstars}


# =====================================================================
# Reporting helpers — verbatim from p2.py
# =====================================================================

def compute_metrics(
    pE: float, pL: float, lamE: float, lamL: float, cE: float, cL: float, Delta: float
) -> Dict[str, float]:
    AE = lamE * pE; AL = lamL * pL; C = cE * AE + cL * AL
    Phi = pE - pL; absPhi = abs(Phi); ineq = absPhi - Delta
    rel = (ineq / Delta) * 100.0 if Delta > 0 else float("nan")
    return {"C": C, "Phi": Phi, "absPhi": absPhi, "ineq": ineq, "rel_ineq": rel}


def mean_sd_str(vals: List[float], scale: float = 1.0, digits: int = 2) -> str:
    arr = np.array(vals, dtype=float)
    m  = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
    sd = float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else float("nan")
    if not np.isfinite(m):  return "—"
    if not np.isfinite(sd): return f"{scale*m:.{digits}f}"
    return f"{scale*m:.{digits}f} ± {scale*sd:.{digits}f}"


def parse_ratios(text: str) -> List[float]:
    raw  = [t.strip() for t in text.split(",") if t.strip() != ""]
    vals = []
    for t in raw:
        v = float(t)
        if v <= 0: raise ValueError("Cost ratios must be positive.")
        vals.append(v)
    vals = sorted(set([round(v, 10) for v in vals]))
    if len(vals) == 0: raise ValueError("Please provide at least one ratio.")
    return vals


# =====================================================================
# Per-rep statistics helper (used by both app.py and parallel workers)
# =====================================================================

def _run_stats_core(df: "pd.DataFrame", kpi: Dict) -> Dict:
    """Compute per-rep statistics dict from simulation output DataFrame + KPI dict."""
    se_e = df[(df.cohort == 'Early') & (df.status == 'SERVED')]
    ab_e = df[(df.cohort == 'Early') & (df.status.isin(['ABANDONED', 'NO_SHOW']))]
    se_l = df[(df.cohort == 'Late')  & (df.status == 'SERVED')]
    ab_l = df[(df.cohort == 'Late')  & (df.status.isin(['ABANDONED', 'NO_SHOW']))]
    early_df = df[df.cohort == 'Early']
    late_df  = df[df.cohort == 'Late']
    nan = float('nan')
    p_ae = len(ab_e) / len(early_df) if len(early_df) else nan
    p_al = len(ab_l) / len(late_df)  if len(late_df)  else nan
    return dict(
        service_rate=kpi['service_rate'], abandon_rate=kpi['abandon_rate'], util_1=kpi['util_1'],
        es_wait_active=se_e.wait_active.mean() if len(se_e) else nan,
        es_slack=se_e.slack.mean()             if len(se_e) else nan,
        es_wait_total=se_e.wait_total.mean()   if len(se_e) else nan,
        ea_wait_active=ab_e.wait_active.mean() if len(ab_e) else nan,
        ea_wait_total=ab_e.wait_total.mean()   if len(ab_e) else nan,
        ls_wait_active=se_l.wait_active.mean() if len(se_l) else nan,
        ls_slack=se_l.slack.mean()             if len(se_l) else nan,
        ls_wait_total=se_l.wait_total.mean()   if len(se_l) else nan,
        la_wait_active=ab_l.wait_active.mean() if len(ab_l) else nan,
        la_wait_total=ab_l.wait_total.mean()   if len(ab_l) else nan,
        e_wait_active=early_df.wait_active.mean(), e_slack=early_df.slack.mean(),
        e_wait_total=early_df.wait_total.mean(),
        l_wait_active=late_df.wait_active.mean(), l_slack=late_df.slack.mean(),
        l_wait_total=late_df.wait_total.mean(),
        prop_served_early=len(se_e) / len(early_df) if len(early_df) else nan,
        prop_aband_early=p_ae,
        prop_served_late=len(se_l)  / len(late_df)  if len(late_df)  else nan,
        prop_aband_late=p_al,
        aband_ratio_e_over_l=(
            p_ae / p_al if (not math.isnan(p_ae) and not math.isnan(p_al) and p_al > 0)
            else nan),
    )


# =====================================================================
# Parallel gamma-sweep worker  (must be module-level for pickling)
# =====================================================================

def _run_one_gamma_worker(args: tuple) -> tuple:
    """
    Worker: run all reps for ONE gamma value on pre-shifted traces.
    args = (shifted_traces, g, N, T0, pns, s0, T_max, cE, cL, Delta)
    shifted_traces: List[List[PatientTrace]]  (one list per rep, already shifted)
    Returns: (g, costs, phis, runs_dicts, dfs, hists)
    """
    shifted_traces, g, N, T0, pns, s0, T_max, cE, cL, Delta, v = args
    g = float(g)
    costs: List[float] = []
    phis:  List[float] = []
    runs:  List[Dict]  = []
    dfs:   list        = []
    hists: list        = []

    for i, trace in enumerate(shifted_traces):
        df_h, kpi_h, hist_h = simulate_reservation_2queue_full(
            trace, g, N, T0, pns,
            seed_split=s0 + 300_000 + i, seed_run=s0 + 400_000 + i, T_max=T_max)

        ab_e = df_h[(df_h.cohort == "Early") & (df_h.status.isin(["ABANDONED", "NO_SHOW"]))]
        ab_l = df_h[(df_h.cohort == "Late")  & (df_h.status.isin(["ABANDONED", "NO_SHOW"]))]
        n_e  = len(df_h[df_h.cohort == "Early"])
        n_l  = len(df_h[df_h.cohort == "Late"])
        pE_i = len(ab_e) / n_e if n_e else float("nan")
        pL_i = len(ab_l) / n_l if n_l else float("nan")
        cost_i = cE * len(ab_e) + cL * len(ab_l) - v * g * N * (T_max - T0)
        phi_i  = pE_i - pL_i
        if not (math.isnan(cost_i) or math.isnan(phi_i)):
            costs.append(cost_i)
            phis.append(phi_i)

        runs.append(_run_stats_core(df_h, kpi_h))
        dfs.append(df_h.assign(run=i))
        hists.append(hist_h)

    return g, costs, phis, runs, dfs, hists
