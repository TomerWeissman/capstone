"""H77, H78: does rate-of-change variability explain patient difficulty and the child effect?

Adds each patient's SD of 30-minute glucose changes (training period) to the H65-H67 model.

Run:    python work/12_10/scripts/round15_rate_variability.py
Writes: work/12_10/results/round15_patients.csv, verdict_h77_h78.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import reference as ref, uq_frames  # noqa: E402
from round1_patient_vs_situation import r2  # noqa: E402
from round6_panel import PANEL, TRAIN_SHARE, ols  # noqa: E402


def rate_sd():
    out = {}
    for name in ["azt1d", "ohiot1dm", "hupa_ucm", *[f"panel:{s}" for s in PANEL]]:
        df = uq_frames.load_dataset(name)
        for sid, d in df.groupby(df.subject_id.astype(str)):
            g = d[ref.CGM].to_numpy(float)[: int(len(d) * TRAIN_SHARE)]
            out[(name, sid)] = float(np.nanstd(g[6:] - g[:-6]))
    return out


def loso(X, y, src):
    pred = np.zeros(len(y))
    for s in np.unique(src):
        k = src == s
        b = ols(X[~k], y[~k])
        pred[k] = b[0] + X[k] @ b[1:]
    return r2(y, pred)


def main():
    P = pd.read_csv(WEEK / "results" / "round6_patients.csv", dtype={"subject_id": str})
    rs = rate_sd()
    P["rate_sd"] = [rs[(d, s)] for d, s in zip(P.dataset, P.subject_id)]
    P = P.dropna(subset=["rmse", "glucose_sd", "age", "rate_sd"]).reset_index(drop=True)
    P.to_csv(WEEK / "results" / "round15_patients.csv", index=False)
    y, src = np.log(P.rmse.to_numpy()), P.source.to_numpy()
    lsd, lrate = np.log(P.glucose_sd.to_numpy()), np.log(P.rate_sd.to_numpy())
    r2_sd = loso(lsd[:, None], y, src)
    r2_both = loso(np.column_stack([lsd, lrate]), y, src)
    r2_rate = loso(lrate[:, None], y, src)
    X = np.column_stack([lsd, lrate, P.aid, P.child])
    b = ols(X, y)
    rng = np.random.default_rng(0)
    srcs = P.source.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(srcs, len(srcs), replace=True)
        idx = np.concatenate([np.flatnonzero(P.source == s) for s in pick])
        boots.append(ols(X[idx], y[idx]))
    boots = np.array(boots)
    child = np.exp(b[4]) - 1
    ci = np.exp(np.percentile(boots[:, 4], [5, 95])) - 1
    aid = np.exp(b[3]) - 1
    ci_a = np.exp(np.percentile(boots[:, 3], [5, 95])) - 1
    h77 = abs(child) < 0.05 and ci[0] <= 0 <= ci[1]
    text = "\n".join([
        f"H77 (child effect gone after rate-of-change SD): {'SUPPORTED' if h77 else 'NOT SUPPORTED'}; child {child:+.1%} (90% {ci[0]:+.1%} to {ci[1]:+.1%}); AID {aid:+.1%} (90% {ci_a[0]:+.1%} to {ci_a[1]:+.1%})",
        f"H78 (glucose SD + rate SD, leave-one-source-out R^2 >= 0.60): {'SUPPORTED' if r2_both >= 0.6 else 'NOT SUPPORTED'}; R^2 {r2_both:.3f} (glucose SD alone {r2_sd:.3f}; rate SD alone {r2_rate:.3f})",
        f"coefficients (log-log): glucose SD {b[1]:.2f}, rate SD {b[2]:.2f}; corr(log glucose SD, log rate SD) {np.corrcoef(lsd, lrate)[0, 1]:.2f}",
        f"median rate SD children {P[P.child == 1].rate_sd.median():.1f} vs adults {P[P.child == 0].rate_sd.median():.1f} mg/dL per 30 min; median glucose SD {P[P.child == 1].glucose_sd.median():.1f} vs {P[P.child == 0].glucose_sd.median():.1f}",
        f"patients {len(P)}",
    ])
    (WEEK / "results" / "verdict_h77_h78.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
