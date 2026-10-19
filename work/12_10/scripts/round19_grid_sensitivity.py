"""H83: does the result depend on the 5x5 situation grid?

Situation bands are rebuilt on 3x3 and 7x7 grids; every band is scored on the standard 25 situations.

Run:    python work/12_10/scripts/round19_grid_sensitivity.py
Writes: work/12_10/results/verdict_h83.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, uq_frames  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402

GRIDS = {"3x3": ([70, 180], [-10, 10]), "5x5": ([70, 120, 180, 250], [-30, -10, 10, 30]),
         "7x7": ([70, 100, 140, 180, 250, 300], [-40, -20, -10, 10, 20, 40])}


def regrid(f, levels, rates):
    f = f.copy()
    f["cell"] = np.digitize(f.g_now, levels) * (len(rates) + 1) + np.digitize(f.g_now - f.g_prev, rates)
    return f


def main(datasets):
    lines, ok = [], 0
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        garch = gb.cov_err(gb.band(t, *gb.garch(v, t, 0.8))[2], t.cell, 0.8)
        errs = {}
        for g, (lv, rt) in GRIDS.items():
            c, h = gb.situation_mondrian(regrid(v, lv, rt), regrid(t, lv, rt), 0.8)
            errs[g] = gb.cov_err(gb.band(t, c, h)[2], t.cell, 0.8)  # scored on the standard 25 situations
        good = all(errs[g] <= 0.8 * garch for g in ("3x3", "7x7")) and errs["5x5"] <= min(errs.values()) + 0.01
        ok += good
        lines.append(f"{name}: GARCH {garch * 100:.2f}; " + ", ".join(f"{g} {e * 100:.2f}" for g, e in errs.items()) + " points")
    text = f"H83 (3x3 and 7x7 both >= 20% better than GARCH; 5x5 within 1 point of best): {'SUPPORTED' if ok >= 2 else 'NOT SUPPORTED'} ({ok} of {len(datasets)})\n" + "\n".join(lines)
    (WEEK / "results" / "verdict_h83.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
