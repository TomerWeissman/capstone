"""Correction check for H66/H77: MetaboNet leaves 12 of 13 BrisT1D panel patients without a delivery
modality, and round6 coded them as non-AID. Their devices (MetaboNet): 780G (AID) x3, 640G (not AID)
x2, Omnipod Eros -> Omnipod 5 (switched) x1, t:slim X2 (Control-IQ or Basal-IQ, unknown) x7.

Refits the closed-loop and child effects (with and without rate-of-change SD) two ways:
  A. BrisT1D excluded
  B. BrisT1D recoded: 780G and t:slim X2 as AID, 640G as not AID, the switcher excluded

Run:    python work/12_10/scripts/round15b_modality_labels.py
Writes: work/12_10/results/verdict_h66_h77_label_check.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from round6_panel import ols  # noqa: E402

BRIS = {"P12": 1, "P18": 1, "P16": 1, "P15": 0, "P11": 0, "P07": None,
        **{p: 1 for p in ["P04", "P22", "P02", "P03", "P10", "P19", "P24"]}}


def fit(P, with_rate):
    y = np.log(P.rmse.to_numpy())
    cols = [np.log(P.glucose_sd)] + ([np.log(P.rate_sd)] if with_rate else []) + [P.aid, P.child]
    X = np.column_stack(cols)
    b = ols(X, y)
    rng = np.random.default_rng(0)
    srcs = P.source.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(srcs, len(srcs), replace=True)
        idx = np.concatenate([np.flatnonzero(P.source == s) for s in pick])
        boots.append(ols(X[idx], y[idx]))
    boots = np.array(boots)
    out = {}
    for name, j in (("AID", -2), ("child", -1)):
        e = np.exp(b[j]) - 1
        lo, hi = np.exp(np.percentile(boots[:, j], [5, 95])) - 1
        out[name] = f"{e:+.1%} ({lo:+.1%} to {hi:+.1%})"
    return out


def main():
    P = pd.read_csv(WEEK / "results" / "round15_patients.csv", dtype={"subject_id": str})
    meta = pd.read_csv(ROOT / "data" / "processed" / "metabonet_panel" / "metadata.csv", dtype={"subject_id": str})
    P = P.merge(meta[["subject_id", "metabonet_id"]], on="subject_id", how="left")
    A = P[P.source != "BrisT1D"].reset_index(drop=True)
    B = P.copy()
    m = B.source == "BrisT1D"
    B.loc[m, "aid"] = B.loc[m, "metabonet_id"].map(BRIS)
    B = B.dropna(subset=["aid"]).reset_index(drop=True)
    lines = ["Label check for H66 (AID) and H77 (child), source-cluster bootstrap 90% intervals:"]
    for label, D in (("original coding (round6)", P), ("A: BrisT1D excluded", A), ("B: BrisT1D recoded", B)):
        for wr in (False, True):
            r = fit(D, wr)
            lines.append(f"  {label}, {'with' if wr else 'without'} rate SD (n={len(D)}, AID {int(D.aid.sum())}): AID {r['AID']}; child {r['child']}")
    text = "\n".join(lines)
    (WEEK / "results" / "verdict_h66_h77_label_check.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
