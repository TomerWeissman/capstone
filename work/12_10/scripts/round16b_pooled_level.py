"""H81: per-patient per-level online scale on cold situation bands with the pooled forecaster.

Run:    python work/12_10/scripts/round16b_pooled_level.py
Writes: work/12_10/results/verdict_h81.txt, round16b_pooled_level.csv
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
from paper_figures import daily_sd  # noqa: E402


def cold_level(v, t, level):
    centre, half = t.pred.to_numpy(), np.zeros(len(t))
    for sid, idx in t.groupby("subject_id").indices.items():
        _, h = gb.level_scale_aci_situation(v[v.subject_id != sid], t.iloc[idx], level, eta=0.01)
        half[idx] = h
    return centre, half


def main(datasets):
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, "pooled_panel")
        res, inside = gb.evaluate(v, t, methods={"cold situation x patient-level scale": cold_level})
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(), "daily_sd": daily_sd(t, ins),
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round16b_pooled_level.csv", index=False)
    ok = int(sum((R.excess_patient <= 0.01) & (R.excess_situation <= 0.015)))
    cols = ["dataset", "coverage", "width", "interval_score", "excess_situation", "excess_patient", "daily_sd"]
    S = R[cols].copy()
    S[["coverage", "excess_situation", "excess_patient", "daily_sd"]] *= 100
    text = f"H81 (pooled forecaster, cold per-level patient scale): {'SUPPORTED' if ok >= 2 else 'NOT SUPPORTED'} ({ok} of {len(datasets)})\n" + S.round(2).to_string(index=False) + "\n"
    (WEEK / "results" / "verdict_h81.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
