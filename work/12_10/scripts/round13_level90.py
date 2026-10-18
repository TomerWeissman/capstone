"""H76: do the situation and per-patient findings hold for 90% bands?

Run:    python work/12_10/scripts/round13_level90.py
Writes: work/12_10/results/round13_level90.csv, verdict_h76.txt
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, uq_frames  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402

LEVEL = 0.9
METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01),
           "situation x patient-level scale (0.01)": partial(gb.level_scale_aci_situation, eta=0.01)}


def main(datasets):
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        res, inside = gb.evaluate(v, t, LEVEL, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(),
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round13_level90.csv", index=False)
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    a = sum(g(d, "situation (Mondrian)").err_situation <= 0.8 * g(d, "GARCH").err_situation for d in datasets)
    b = sum(g(d, "situation x patient scale (0.01)").excess_patient <= 0.5 * g(d, "situation (Mondrian)").excess_patient for d in datasets)
    cols = ["coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient"]
    text = f"H76 (90% bands): {'SUPPORTED' if a >= 2 and b >= 2 else 'NOT SUPPORTED'} (a {a} of {len(datasets)}, b {b} of {len(datasets)})\n"
    for d in datasets:
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "err_situation", "excess_situation", "excess_patient"]] *= 100
        text += f"{d}:\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h76.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
