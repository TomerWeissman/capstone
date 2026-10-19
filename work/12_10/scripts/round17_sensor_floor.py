"""H82: how much of the 60-minute forecast error could be the sensor's own error at the target?

Assumptions (stated in the paper): CGM error relative to true glucose is independent of the forecast
error and proportional to glucose, |error| / glucose averaging MARD; with normal errors the SD is
MARD x glucose / 0.798. Sensitivity: MARD 8%, 10%, 14%; and an absolute floor of 10 mg/dL below
100 mg/dL (CGM error is larger in relative terms when glucose is low).

Run:    python work/12_10/scripts/round17_sensor_floor.py
Writes: work/12_10/results/round17_sensor_floor.csv, verdict_h82.txt, figures/h82_sensor_floor.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting, uq_frames  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402
from round6_panel import PANEL  # noqa: E402

DATASETS = ["azt1d", "ohiot1dm", "hupa_ucm", *[f"panel:{s}" for s in PANEL]]


def sensor_var(g, mard, floor=False):
    sd = mard * g / 0.798
    if floor:
        sd = np.where(g < 100, np.maximum(sd, 10.0), sd)
    return sd ** 2


def main():
    plotting.apply_style()
    T = pd.concat([uq_frames.frames(d, run_for(d))[1].assign(dataset=d) for d in DATASETS], ignore_index=True)
    rows = []
    for mard in (0.08, 0.10, 0.14):
        for floor in (False, True):
            share = sensor_var(T.actual.to_numpy(), mard, floor).mean() / np.mean(T.resid ** 2)
            rows.append({"mard": mard, "low_floor": floor, "share_pooled": share})
    S = pd.DataFrame(rows)
    by = T.assign(sv=sensor_var(T.actual.to_numpy(), 0.10), se=T.resid ** 2).groupby("cell").agg(
        n=("se", "size"), mse=("se", "mean"), sv=("sv", "mean"))
    by["share"] = by.sv / by.mse
    by = by[by.n >= 300]
    S.to_csv(WEEK / "results" / "round17_sensor_floor.csv", index=False)
    main_share = float(S[(S.mard == 0.10) & (~S.low_floor)].share_pooled.iloc[0])
    text = "\n".join([
        f"H82 (sensor error at the target >= 20% of forecast error variance at MARD 10%): {'SUPPORTED' if main_share >= 0.20 else 'NOT SUPPORTED'}; {main_share:.1%}",
        "sensitivity:\n  " + S.assign(share_pooled=S.share_pooled.map(lambda x: f"{x:.1%}")).to_string(index=False).replace("\n", "\n  "),
        f"forecast RMSE pooled {np.sqrt(np.mean(T.resid ** 2)):.1f} mg/dL; windows {len(T):,}; studies {len(DATASETS)}",
        "by situation (MARD 10%; share of forecast error variance):",
        "  " + ", ".join(f"{uq_frames.LEVEL_NAMES[c // 5]}/{uq_frames.RATE_NAMES[c % 5]} {r.share:.0%}" for c, r in by.iterrows()),
    ])
    (WEEK / "results" / "verdict_h82.txt").write_text(text + "\n")
    print(text)

    grid = np.full((5, 5), np.nan)
    for c, r in by.iterrows():
        grid[c // 5, c % 5] = r.share * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(grid, cmap="Purples", vmin=0, vmax=np.nanmax(grid), origin="lower")
    for i in range(5):
        for j in range(5):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if grid[i, j] > np.nanmax(grid) * 0.6 else "black")
    ax.set_xticks(range(5), uq_frames.RATE_NAMES, rotation=30, fontsize=8)
    ax.set_yticks(range(5), uq_frames.LEVEL_NAMES, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of forecast error variance")
    ax.set_title(f"H82: how much of the 60-minute error could be sensor error at the target?\n(MARD 10%, 11 studies; pooled {main_share:.0%})",
                 loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(WEEK / "figures" / "h82_sensor_floor.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
