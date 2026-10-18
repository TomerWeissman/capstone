"""Validation and test forecast frames for uncertainty analysis, for any dataset with checkpoints.

One row per forecast window, with the residual, the GARCH error scale, the glucose situation, and
the context variables the variance analyses use. Built once per (dataset, run) and cached as
parquet under data/processed/frames/ (regenerable, not in git).

Datasets:
  "azt1d"      AZT1D raw files (loading.py)
  "hupa_ucm"   HUPA-UCM preprocessed files (hupa.py)
  "ohiot1dm"   OhioT1DM through MetaboNet
  "mb:<Src>"   any MetaboNet source, e.g. "mb:BrisT1D"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import hupa, loading, metabonet
from . import reference as ref
from .glimmer import checkpoint as ckpt
from .glimmer import uncertainty as unc

LEVEL_EDGES = [70, 120, 180, 250]  # 5 level bands
RATE_EDGES = [-30, -10, 10, 30]  # change over 30 minutes: 5 trend bands
TREND_STEPS = 6
LEVEL_NAMES = ["under 70", "70 to 120", "120 to 180", "180 to 250", "250 and over"]
RATE_NAMES = ["fast fall", "slow fall", "flat", "slow rise", "fast rise"]


def project_root() -> Path:
    return next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


def load_dataset(name: str, subject_ids=None) -> pd.DataFrame:
    root = project_root()
    if name == "azt1d":
        df = loading.load_real_dataset(root / "data" / "raw", root / "data" / "processed")
    elif name == "hupa_ucm":
        ids = subject_ids or hupa.list_subject_ids(root / hupa.DEFAULT_DIR)
        df = pd.concat([hupa.load_subject(s, root / hupa.DEFAULT_DIR) for s in ids], ignore_index=True)
    elif name == "ohiot1dm":
        df = metabonet.load_source("OhioT1DM")
    elif name.startswith("mb:"):
        df = metabonet.load_source(name[3:], subject_ids=subject_ids)
    else:
        raise ValueError(name)
    if subject_ids is not None:
        df = df[df.subject_id.isin(subject_ids)]
    return df


def situation(g_now, g_prev):
    level = np.digitize(g_now, LEVEL_EDGES)
    trend = np.digitize(np.asarray(g_now) - np.asarray(g_prev), RATE_EDGES)
    return level, trend, level * 5 + trend


def subject_frames(dfs: pd.DataFrame, result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frames for one patient from its checkpoint. GARCH is fitted on validation residuals only."""
    sid = dfs.subject_id.iloc[0]
    val, test = unc.forecast_frames(result, dfs)
    p, s_test = unc.garch_sigma(val, test)
    s_val = np.sqrt(unc.garch_h_step_variance(val.resid, p, float(np.var(val.resid - p.mu)), unc.LAG))
    times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
    g = dfs[ref.CGM].to_numpy(float)
    carbs = dfs[ref.CARB_SIZE].fillna(0).to_numpy(float)
    bolus = dfs[ref.TOTAL_BOLUS_INSULIN_DELIVERED].fillna(0).to_numpy(float)
    out = []
    for split, fr, sig in (("val", val, s_val), ("test", test, s_test)):
        i = np.searchsorted(times, fr.issue_time)
        lo2h = np.maximum(i - 23, 0)
        csum, bsum = np.concatenate([[0], np.cumsum(carbs)]), np.concatenate([[0], np.cumsum(bolus)])
        d = pd.DataFrame({
            "subject_id": sid, "split": split, "issue_time": pd.to_datetime(fr.issue_time),
            "actual": fr.actual, "pred": fr.pred, "resid": fr.resid, "mu": p.mu, "sigma": sig,
            "g_now": g[i], "g_prev": g[np.maximum(i - TREND_STEPS, 0)], "g_2h": g[lo2h],
            "carbs_2h": csum[i + 1] - csum[lo2h], "bolus_2h": bsum[i + 1] - bsum[lo2h],
            "hour": pd.to_datetime(fr.issue_time).hour + pd.to_datetime(fr.issue_time).minute / 60,
        })
        d["level"], d["trend"], d["cell"] = situation(d.g_now.to_numpy(), d.g_prev.to_numpy())
        out.append(d[(i >= TREND_STEPS) & np.isfinite(sig)].reset_index(drop=True))
    return out[0], out[1]


def frames(name: str, run: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(validation, test) frames for every patient with a checkpoint under checkpoints/<run>."""
    root = project_root()
    cache = root / "data" / "processed" / "frames" / f"{name.replace(':', '_')}__{run}.parquet"
    if cache.exists() and not refresh:
        f = pd.read_parquet(cache)
    else:
        run_dir = root / "data" / "processed" / "checkpoints" / run
        df = load_dataset(name)
        parts = []
        for sid in df.subject_id.unique():
            res = ckpt.load_result(run_dir, sid)
            if res is None:
                continue
            dfs = df[df.subject_id == sid].reset_index(drop=True)
            v, t = subject_frames(dfs, res)
            parts += [v, t]
        f = pd.concat(parts, ignore_index=True)
        f["subject_id"] = f.subject_id.astype(str)
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(cache, index=False)
    return f[f.split == "val"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)
