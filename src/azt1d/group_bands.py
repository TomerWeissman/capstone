"""Group-conditional (Mondrian) conformal bands and the calibration metrics used to compare them.

Works on frames from azt1d.uq_frames: calibrate on the validation frame, score on the test frame.
Every method returns symmetric half-widths around the point forecast (GARCH variants add the
GARCH mean shift mu to the centre).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .glimmer.uncertainty import SENSOR_MAX, SENSOR_MIN

Z = {0.8: 1.2815515655446004, 0.9: 1.6448536269514722, 0.5: 0.6744897501960817, 0.95: 1.959963984540054}


def conformal_q(scores, level: float) -> float:
    """Split-conformal quantile: the ceil((n+1) level)-th smallest score."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = min(math.ceil((n + 1) * level), n)
    return float(s[k - 1])


def group_q(v_scores, v_groups, t_groups, level: float, min_n: int = 100):
    """Per-group conformal quantile from validation, looked up for each test row.
    Groups with fewer than min_n validation rows fall back to the pooled quantile."""
    pooled = conformal_q(v_scores, level)
    q = {}
    vs = pd.Series(np.asarray(v_scores, float))
    for g, idx in pd.Series(np.asarray(v_groups)).groupby(np.asarray(v_groups)).groups.items():
        q[g] = conformal_q(vs.loc[idx].to_numpy(), level) if len(idx) >= min_n else pooled
    return np.array([q.get(g, pooled) for g in np.asarray(t_groups)])


def patient_scale(v: pd.DataFrame) -> pd.Series:
    """Each patient's validation error scale: mean absolute residual."""
    return v.groupby("subject_id").resid.apply(lambda r: float(np.mean(np.abs(r))))


# ---- methods: each returns (centre, half-width) for the test rows ----

def pooled_conformal(v, t, level):
    return t.pred.to_numpy(), np.full(len(t), conformal_q(np.abs(v.resid), level))


def per_patient_conformal(v, t, level):
    return t.pred.to_numpy(), group_q(np.abs(v.resid), v.subject_id, t.subject_id, level, min_n=1)


def situation_mondrian(v, t, level):
    return t.pred.to_numpy(), group_q(np.abs(v.resid), v.cell, t.cell, level)


def patient_scaled_situation(v, t, level):
    """|e| / patient scale, Mondrian by situation pooled across patients, times the patient scale."""
    s = patient_scale(v)
    sv, st = v.subject_id.map(s).to_numpy(), t.subject_id.map(s).to_numpy()
    q = group_q(np.abs(v.resid) / sv, v.cell, t.cell, level)
    return t.pred.to_numpy(), q * st


def patient_by_situation_raw(v, t, level):
    """Separate quantile in every patient-by-situation cell (falls back to pooled when small)."""
    gv = v.subject_id.astype(str) + "|" + v.cell.astype(str)
    gt = t.subject_id.astype(str) + "|" + t.cell.astype(str)
    return t.pred.to_numpy(), group_q(np.abs(v.resid), gv, gt, level, min_n=30)


def garch(v, t, level):
    return (t.pred + t.mu).to_numpy(), Z[level] * t.sigma.to_numpy()


def garch_situation(v, t, level):
    """Normalized conformal with GARCH scale, Mondrian by situation (the 5_10 hybrid)."""
    q = group_q((np.abs(v.resid - v.mu) / v.sigma).to_numpy(), v.cell, t.cell, level)
    return (t.pred + t.mu).to_numpy(), q * t.sigma.to_numpy()


METHODS = {
    "pooled conformal": pooled_conformal,
    "per-patient conformal": per_patient_conformal,
    "GARCH": garch,
    "situation (Mondrian)": situation_mondrian,
    "GARCH x situation": garch_situation,
    "patient-scaled situation": patient_scaled_situation,
    "patient x situation cells": patient_by_situation_raw,
}


# ---- metrics ----

def band(t, centre, half):
    lo = np.clip(centre - half, SENSOR_MIN, SENSOR_MAX)
    hi = np.clip(centre + half, SENSOR_MIN, SENSOR_MAX)
    y = t.actual.to_numpy()
    return lo, hi, (y >= lo) & (y <= hi)


def cov_err(inside, groups, level, min_n: int = 30) -> float:
    """Mean |coverage - level| over groups with at least min_n test rows."""
    d = pd.DataFrame({"g": np.asarray(groups), "i": np.asarray(inside, float)})
    c = d.groupby("g").i.agg(["mean", "size"])
    c = c[c["size"] >= min_n]
    return float((c["mean"] - level).abs().mean())


def dispersion(inside, groups, min_n: int = 30) -> float:
    """Mean |group coverage - the method's own overall coverage| over groups with at least min_n rows.
    Unlike cov_err, this ignores a shift in overall coverage (for example from validation-to-test drift),
    so it measures only how unevenly the band covers different groups."""
    ins = np.asarray(inside, float)
    d = pd.DataFrame({"g": np.asarray(groups), "i": ins})
    c = d.groupby("g").i.agg(["mean", "size"])
    c = c[c["size"] >= min_n]
    return float((c["mean"] - ins.mean()).abs().mean())


def interval_score(t, lo, hi, level) -> float:
    """Winkler interval score (lower is better)."""
    a = 1 - level
    y = t.actual.to_numpy()
    return float(np.mean((hi - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)))


def evaluate(v, t, level=0.8, methods=METHODS):
    rows, inside = [], {}
    pc = t.subject_id.astype(str) + "|" + t.cell.astype(str)
    for name, fn in methods.items():
        c, h = fn(v, t, level)
        lo, hi, ins = band(t, c, h)
        inside[name] = ins
        rows.append({"method": name, "coverage": ins.mean(), "width": float(np.mean(hi - lo)),
                     "interval_score": interval_score(t, lo, hi, level),
                     "err_situation": cov_err(ins, t.cell, level), "err_patient": cov_err(ins, t.subject_id, level),
                     "err_patient_x_situation": cov_err(ins, pc, level),
                     "disp_situation": dispersion(ins, t.cell), "disp_patient": dispersion(ins, t.subject_id),
                     "disp_patient_x_situation": dispersion(ins, pc)})
    return pd.DataFrame(rows), inside


def patient_bootstrap(t, stat, n_boot=1000, seed=0):
    """Cluster bootstrap over patients. stat(sub_frame_index) -> float. Returns (low, high) 90% interval."""
    rng = np.random.default_rng(seed)
    pats = t.subject_id.unique()
    idx_by = t.groupby("subject_id").indices
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(pats, len(pats), replace=True)
        idx = np.concatenate([idx_by[p] for p in pick])
        vals.append(stat(idx))
    return tuple(np.nanpercentile(vals, [5, 95]))


# ---- noise floors: how uneven would coverage look from sampling noise alone? ----

def dispersion_null_shift(inside, groups, subjects, n_rep: int = 50, min_shift: int = 288, seed: int = 0,
                          min_n: int = 30):
    """Dispersion expected if coverage were the same in every group.

    Within each patient, the group labels are circularly shifted by a random lag of at least
    min_shift windows (one day), so labels keep their own autocorrelation but lose their link to the
    inside/outside sequence. Groups may combine patient and situation (the patient part stays fixed).
    Rows must be in time order within each patient. Returns (mean, 95th percentile) over n_rep shifts."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    idx_by = pd.Series(np.arange(len(groups))).groupby(np.asarray(subjects)).indices
    vals = []
    for _ in range(n_rep):
        shifted = groups.copy()
        for idx in idx_by.values():
            n = len(idx)
            if n <= 2 * min_shift:
                continue
            shifted[idx] = np.roll(groups[idx], int(rng.integers(min_shift, n - min_shift)))
        vals.append(dispersion(inside, shifted, min_n=min_n))
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def dispersion_null_patient(inside, subjects, block: int = 288, n_boot: int = 200, seed: int = 0):
    """Expected per-patient dispersion from sampling noise alone: mean over patients of
    sqrt(2/pi) times the standard error of that patient's coverage, from a moving-block bootstrap
    (one-day blocks) of the patient's own inside/outside sequence."""
    rng = np.random.default_rng(seed)
    ins = np.asarray(inside, float)
    out = []
    for idx in pd.Series(np.arange(len(ins))).groupby(np.asarray(subjects)).indices.values():
        x = ins[idx]
        n = len(x)
        b = min(block, n)
        starts = np.arange(n - b + 1)
        k = int(np.ceil(n / b))
        means = [np.concatenate([x[s:s + b] for s in rng.choice(starts, k)])[:n].mean() for _ in range(n_boot)]
        out.append(np.sqrt(2 / np.pi) * np.std(means))
    return float(np.mean(out))


# ---- adaptive conformal inference (Gibbs and Candes 2021) on top of a group-conditional band ----

def aci_situation(v, t, level, gamma: float = 0.005, delay: int = 12, min_n: int = 100):
    """Situation (Mondrian) band whose miss-rate target is adapted online, per patient.

    At each test window the band uses the (1 - alpha_t) quantile of the validation scores of its
    situation. alpha is updated with the miss indicator of the window issued `delay` steps earlier
    (its outcome is only known after the 60-minute horizon). Rows must be in time order per patient."""
    scores = np.abs(v.resid.to_numpy())
    pooled = np.sort(scores)
    by_cell = {c: np.sort(scores[idx]) for c, idx in v.groupby("cell").indices.items() if len(idx) >= min_n}
    target = 1 - level
    half = np.zeros(len(t))
    y, pred, cells = t.actual.to_numpy(), t.pred.to_numpy(), t.cell.to_numpy()
    for idx in t.groupby("subject_id").indices.values():
        alpha = np.full(len(idx) + 1, target)
        miss = np.zeros(len(idx))
        for j, r in enumerate(idx):
            s = by_cell.get(cells[r], pooled)
            lvl = 1 - alpha[j]
            if lvl >= 1:
                h = SENSOR_MAX
            elif lvl <= 0:
                h = 0.0
            else:
                h = s[min(math.ceil(lvl * (len(s) + 1)), len(s)) - 1]
            half[r] = h
            miss[j] = abs(y[r] - pred[r]) > h
            fb = miss[j - delay] if j >= delay else target  # outcome known only after the horizon
            alpha[j + 1] = alpha[j] + gamma * (target - fb)
    return pred, half


def miss_run_length(inside, subjects) -> float:
    """Mean length of runs of consecutive misses, within patients (rows in time order)."""
    ins = np.asarray(inside, bool)
    runs = []
    for idx in pd.Series(np.arange(len(ins))).groupby(np.asarray(subjects)).indices.values():
        m = ~ins[idx]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        runs.extend(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return float(np.mean(runs)) if runs else 0.0


def scale_aci_situation(v, t, level, eta: float = 0.01, delay: int = 12, min_n: int = 100):
    """Situation (Mondrian) band times a per-patient multiplicative scale exp(theta), adapted online:
    theta += eta * (miss indicator from `delay` steps earlier - (1 - level)). Keeps the situation
    shape of the band and corrects only the patient's overall error scale."""
    centre, base = situation_mondrian(v, t, level)
    target = 1 - level
    y = t.actual.to_numpy()
    half = np.zeros(len(t))
    for idx in t.groupby("subject_id").indices.values():
        theta = 0.0
        miss = np.zeros(len(idx))
        for j, r in enumerate(idx):
            half[r] = base[r] * np.exp(theta)
            miss[j] = abs(y[r] - centre[r]) > half[r]
            fb = miss[j - delay] if j >= delay else target
            theta += eta * (fb - target)
    return centre, half


def level_scale_aci_situation(v, t, level, eta: float = 0.01, delay: int = 12, min_n: int = 100):
    """Situation band times a per-patient, per-glucose-level online scale. A miss (known `delay`
    steps later) updates only the scale of the level band the missed window was in, so frequent
    situations do not inflate the band in rare ones."""
    centre, base = situation_mondrian(v, t, level)
    target = 1 - level
    y, lv = t.actual.to_numpy(), (t.cell.to_numpy() // 5)
    half = np.zeros(len(t))
    for idx in t.groupby("subject_id").indices.values():
        theta = np.zeros(5)
        miss = np.zeros(len(idx))
        for j, r in enumerate(idx):
            half[r] = base[r] * np.exp(theta[lv[r]])
            miss[j] = abs(y[r] - centre[r]) > half[r]
            if j >= delay:
                r0 = idx[j - delay]
                theta[lv[r0]] += eta * (miss[j - delay] - target)
    return centre, half
