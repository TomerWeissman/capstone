"""H73: per-patient, per-level online scale vs the single per-patient scale (H59).

Run:    python work/12_10/scripts/round11_level_scale.py [dataset ...]
Writes: work/12_10/results/round11_level_scale.csv, verdict_h73.txt
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
from paper_figures import daily_sd  # noqa: E402

METHODS = {"situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale": partial(gb.scale_aci_situation, eta=0.01),
           "situation x patient-level scale": partial(gb.level_scale_aci_situation, eta=0.01)}


def main(datasets):
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(), "daily_sd": daily_sd(t, ins),
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round11_level_scale.csv", index=False)
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    new, old, base = "situation x patient-level scale", "situation x patient scale", "situation (Mondrian)"
    crit = sum((g(d, new).excess_patient <= 0.01) and (g(d, new).excess_situation <= 0.015)
               and (g(d, new).interval_score <= 1.03 * g(d, base).interval_score) for d in datasets)
    better = sum(g(d, new).excess_situation < g(d, old).excess_situation for d in datasets)
    cols = ["coverage", "width", "interval_score", "excess_situation", "excess_patient", "daily_sd"]
    text = (f"H73 (per-patient per-level scale): {'SUPPORTED' if crit >= 2 and better >= 2 else 'NOT SUPPORTED'} "
            f"(criteria met {crit} of {len(datasets)}; situation excess below single scale {better} of {len(datasets)})\n")
    for d in datasets:
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "excess_situation", "excess_patient", "daily_sd"]] *= 100
        text += f"{d}:\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h73.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
