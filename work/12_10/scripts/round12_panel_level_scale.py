"""H74: the per-patient per-level online scale (H73) on the external MetaboNet panel.

Run:    python work/12_10/scripts/round12_panel_level_scale.py
Writes: work/12_10/results/round12_panel_level_scale.csv, verdict_h74.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import round11_level_scale as r11  # noqa: E402
from round6_panel import PANEL  # noqa: E402


def main():
    datasets = [f"panel:{s}" for s in PANEL]
    r11.main(datasets)  # writes round11 files; rename for the panel run
    for a, b in (("round11_level_scale.csv", "round12_panel_level_scale.csv"), ("verdict_h73.txt", "verdict_h74.txt")):
        (WEEK / "results" / a).rename(WEEK / "results" / b)
    R = pd.read_csv(WEEK / "results" / "round12_panel_level_scale.csv")
    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    new, base = "situation x patient-level scale", "situation (Mondrian)"
    ok = sum((g(d, new).excess_patient <= 0.01) and (g(d, new).excess_situation <= 0.015)
             and (g(d, new).interval_score <= 1.03 * g(d, base).interval_score) for d in datasets)
    head = f"H74 (per-level scale meets H58 criteria on the panel): {'SUPPORTED' if ok >= 5 else 'NOT SUPPORTED'} ({ok} of 8)\n"
    v = WEEK / "results" / "verdict_h74.txt"
    v.write_text(head + v.read_text().split("\n", 1)[1])
    print(v.read_text())


if __name__ == "__main__":
    main()
