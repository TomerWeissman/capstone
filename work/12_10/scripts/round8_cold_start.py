"""H70: cold-start bands. Each patient's situation quantiles come only from the other patients'
validation data; the online per-patient scale then adapts to the newcomer during the test period.

Compared: own-data situation bands, cold situation bands, cold + patient scale, own-data + patient scale.

Run:    python work/12_10/scripts/round8_cold_start.py [dataset ...]
Writes: work/12_10/results/round8_cold_start.csv, verdict_h70.txt
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

LEVEL = 0.8


def cold(v, t, level, adapt: bool):
    """Leave-one-patient-out situation quantiles, optionally times the online per-patient scale."""
    centre, half = t.pred.to_numpy(), np.zeros(len(t))
    for sid, idx in t.groupby("subject_id").indices.items():
        vo = v[v.subject_id != sid]
        tp = t.iloc[idx]
        if adapt:
            _, h = gb.scale_aci_situation(vo, tp, level, eta=0.01)
        else:
            _, h = gb.situation_mondrian(vo, tp, level)
        half[idx] = h
    return centre, half


METHODS = {"own-data situation": gb.situation_mondrian,
           "own-data situation x patient scale": partial(gb.scale_aci_situation, eta=0.01),
           "cold situation": partial(cold, adapt=False),
           "cold situation x patient scale": partial(cold, adapt=True)}


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
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round8_cold_start.csv", index=False)
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    ok = sum((g(d, "cold situation x patient scale").excess_patient <= 0.5 * g(d, "cold situation").excess_patient)
             and (g(d, "cold situation x patient scale").excess_patient <= g(d, "own-data situation x patient scale").excess_patient + 0.01)
             for d in datasets)
    cols = ["coverage", "width", "interval_score", "err_patient", "excess_situation", "excess_patient"]
    text = f"H70 (cold start + patient scale halves per-patient excess and is within 1 point of own-data): {'SUPPORTED' if ok >= 2 else 'NOT SUPPORTED'} ({ok} of {len(datasets)})\n"
    for d in datasets:
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "err_patient", "excess_situation", "excess_patient"]] *= 100
        text += f"{d}:\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h70.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
