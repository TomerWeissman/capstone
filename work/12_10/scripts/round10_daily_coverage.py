"""H72: do GARCH-scaled bands even out coverage from day to day?

Run:    python work/12_10/scripts/round10_daily_coverage.py
Writes: work/12_10/results/verdict_h72.txt
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

METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian, "GARCH x situation": gb.garch_situation,
           "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01)}


def main(datasets):
    lines, ok = [], 0
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        _, inside = gb.evaluate(v, t, methods=METHODS)
        day = t.issue_time.dt.floor("D")
        sds = {}
        for m, ins in inside.items():
            d = pd.DataFrame({"s": t.subject_id, "day": day, "i": ins.astype(float)})
            c = d.groupby(["s", "day"]).i.agg(["mean", "size"])
            sds[m] = float(c[c["size"] >= 200]["mean"].std())
        ok += sds["GARCH x situation"] <= 0.85 * sds["situation (Mondrian)"]
        lines.append(f"{name}: SD of daily coverage (points): " + ", ".join(f"{m} {s * 100:.1f}" for m, s in sds.items()))
    text = f"H72 (GARCH x situation daily-coverage SD >= 15% lower than situation bands): {'SUPPORTED' if ok >= 2 else 'NOT SUPPORTED'} ({ok} of {len(datasets)})\n" + "\n".join(lines)
    (WEEK / "results" / "verdict_h72.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
