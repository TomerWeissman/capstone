"""H69: do the noise-floor and per-patient-scale findings hold for a different forecaster
(CNN-Transformer, cnn_transformer_v0) on AZT1D and OhioT1DM?

Run:    python work/12_10/scripts/round7_other_forecaster.py
Writes: work/12_10/results/round7_transformer.csv, verdict_h69.txt
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import group_bands as gb, uq_frames  # noqa: E402

RUNS = {"azt1d": "cnn_transformer_v0", "ohiot1dm": "ohiot1dm_cnn_transformer_v0"}
METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01)}


def main():
    rows = []
    for name, run in RUNS.items():
        v, t = uq_frames.frames(name, run)
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(),
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round7_transformer.csv", index=False)
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    a = all(g(d, "situation (Mondrian)").excess_situation <= 0.015 for d in RUNS)
    sp = "situation x patient scale (0.01)"
    b = all((g(d, sp).excess_patient <= 0.5 * g(d, "situation (Mondrian)").excess_patient)
            and g(d, sp).interval_score <= 1.03 * g(d, "situation (Mondrian)").interval_score for d in RUNS)
    cols = ["coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient"]
    text = f"H69 (findings hold for the CNN-Transformer): {'SUPPORTED' if a and b else 'NOT SUPPORTED'} (a {a}, b {b})\n"
    for d in RUNS:
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "err_situation", "excess_situation", "excess_patient"]] *= 100
        text += f"{d}:\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h69.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
