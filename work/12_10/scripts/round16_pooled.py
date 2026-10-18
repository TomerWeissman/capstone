"""H79: full cold start. Forecasts from one pooled model trained only on the MetaboNet panel, bands
calibrated (a) per dataset as usual and (b) cold, from other patients only, plus the online scale.

Run:    python work/12_10/scripts/round16_pooled.py   (after train_pooled.py)
Writes: work/12_10/results/round16_pooled.csv, verdict_h79.txt
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, uq_frames  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402
from round8_cold_start import cold  # noqa: E402

METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale": partial(gb.scale_aci_situation, eta=0.01),
           "cold situation": partial(cold, adapt=False),
           "cold situation x patient scale": partial(cold, adapt=True)}


def main(datasets):
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, "pooled_panel")
        _, t_own = uq_frames.frames(name, run_for(name))
        rmse_pooled = t.groupby("subject_id").resid.apply(lambda r: np.sqrt(np.mean(r ** 2))).median()
        rmse_own = t_own.groupby("subject_id").resid.apply(lambda r: np.sqrt(np.mean(r ** 2))).median()
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(), "rmse_pooled_median": rmse_pooled, "rmse_per_patient_median": rmse_own,
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round16_pooled.csv", index=False)
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    a = sum(g(d, "situation (Mondrian)").err_situation <= 0.8 * g(d, "GARCH").err_situation for d in datasets)
    c = "cold situation x patient scale"
    b = sum(g(d, c).excess_patient <= 0.01 and g(d, c).excess_situation <= 0.015 for d in datasets)
    cols = ["coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient"]
    text = f"H79 (pooled forecaster, full cold start): {'SUPPORTED' if a >= 2 and b >= 2 else 'NOT SUPPORTED'} (a {a} of {len(datasets)}, b {b} of {len(datasets)})\n"
    for d in datasets:
        r0 = g(d, "GARCH")
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "err_situation", "excess_situation", "excess_patient"]] *= 100
        text += f"{d} (median patient RMSE: pooled {r0.rmse_pooled_median:.1f}, per-patient models {r0.rmse_per_patient_median:.1f}):\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h79.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
