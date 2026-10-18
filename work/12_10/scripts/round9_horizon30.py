"""H71: do the situation and per-patient findings hold at a 30-minute horizon?

Run:    python work/12_10/scripts/round9_horizon30.py   (after train_horizon30.py)
Writes: work/12_10/results/round9_horizon30.csv, verdict_h71.txt
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
from round1_patient_vs_situation import error_predictors, r2  # noqa: E402

H = 6
RUNS = {"azt1d": "cnn_lstm_v0_h6", "ohiot1dm": "ohiot1dm_cnn_lstm_v0_h6"}
METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01, delay=H)}


def main():
    rows = []
    for name, run in RUNS.items():
        v, t = uq_frames.frames(name, run, horizon=H)
        y, preds = error_predictors(v, t)
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(), "rmse": float(np.sqrt(np.mean(t.resid ** 2))),
                         "r2_situation": r2(y, preds["situation"]), "r2_patient": r2(y, preds["patient"]),
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round9_horizon30.csv", index=False)
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    sp = "situation x patient scale (0.01)"
    a = all(g(d, "situation (Mondrian)").excess_situation <= 0.015 for d in RUNS)
    b = all((g(d, sp).excess_patient <= 0.5 * g(d, "situation (Mondrian)").excess_patient)
            and g(d, sp).interval_score <= 1.03 * g(d, "situation (Mondrian)").interval_score for d in RUNS)
    cols = ["coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient"]
    text = f"H71 (30-minute horizon): {'SUPPORTED' if a and b else 'NOT SUPPORTED'} (a {a}, b {b})\n"
    for d in RUNS:
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "err_situation", "excess_situation", "excess_patient"]] *= 100
        r0 = g(d, "GARCH")
        text += f"{d} (RMSE {r0.rmse:.1f}; R^2 situation {r0.r2_situation:.3f}, patient {r0.r2_patient:.3f}):\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h71.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
