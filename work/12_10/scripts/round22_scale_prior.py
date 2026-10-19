"""H86: does a rate-of-change prior for the patient scale make the first two days more honest?

For each patient: cold situation quantiles (other patients of the same study). Prior: log learned
scale ~ log rate-of-change SD, fitted on the other patients of the same study (round21_scales.csv),
evaluated at this patient's training-period rate SD. Compared with the neutral start (scale 1).
Metric: mean over patients of |coverage - 80%| in the first 576 test windows.

Run:    python work/12_10/scripts/round22_scale_prior.py
Writes: work/12_10/results/verdict_h86.txt, round22_scale_prior.csv
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

from azt1d import group_bands as gb, uq_frames  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402
from round6_panel import PANEL  # noqa: E402

FIRST = 576


def main():
    S = pd.read_csv(WEEK / "results" / "round21_scales.csv", dtype={"subject_id": str})
    rows = []
    for name in ["azt1d", "ohiot1dm", "hupa_ucm", *[f"panel:{s}" for s in PANEL]]:
        v, t = uq_frames.frames(name, run_for(name))
        Sd = S[S.dataset == name]
        errs = {"neutral start": [], "rate prior": []}
        for sid, idx in t.groupby("subject_id").indices.items():
            others = Sd[Sd.subject_id != sid]
            b = np.polyfit(np.log(others.rate_sd), np.log(others.learned_scale), 1)
            me = Sd[Sd.subject_id == sid]
            if me.empty:
                continue
            th = float(b[0] * np.log(me.rate_sd.iloc[0]) + b[1])
            vo, tp = v[v.subject_id != sid], t.iloc[idx]
            for arm, theta0 in (("neutral start", None), ("rate prior", {sid: th})):
                c, h = gb.scale_aci_situation(vo, tp, 0.8, eta=0.01, theta0=theta0)
                ins = gb.band(tp, c, h)[2][:FIRST]
                errs[arm].append(abs(ins.mean() - 0.8))
        rows.append({"dataset": name, "neutral": np.mean(errs["neutral start"]), "prior": np.mean(errs["rate prior"]),
                     "n": len(errs["rate prior"])})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R["gain"] = 1 - R.prior / R.neutral
    R.to_csv(WEEK / "results" / "round22_scale_prior.csv", index=False)
    ok = int((R.gain >= 0.15).sum())
    text = (f"H86 (rate prior >= 15% better first-two-day per-patient calibration in >= 7 of 11 studies): "
            f"{'SUPPORTED' if ok >= 7 else 'NOT SUPPORTED'} ({ok} of {len(R)})\n"
            + R.assign(neutral=R.neutral * 100, prior=R.prior * 100).round(3).to_string(index=False)
            + f"\nmedian gain {R.gain.median():+.0%}\n")
    (WEEK / "results" / "verdict_h86.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
