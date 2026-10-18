"""Paper figure: what makes a patient hard to forecast (glucose SD vs rate-of-change SD), 215 patients.

Run:    python work/12_10/scripts/paper_fig_difficulty.py   (after round15_rate_variability.py)
Writes: work/12_10/figures/paper_fig_difficulty.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from azt1d import plotting  # noqa: E402

P = pd.read_csv(WEEK / "results" / "round15_patients.csv")
plotting.apply_style()
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
cmap = plt.get_cmap("tab20")
for ax, col, label, r2 in zip(axes, ["glucose_sd", "rate_sd"],
                              ["patient's glucose SD (mg/dL)", "patient's SD of 30-minute glucose change (mg/dL)"],
                              [0.444, 0.588]):
    for k, (s, d) in enumerate(P.groupby("source")):
        ax.scatter(d[col], d.rmse, s=22, alpha=0.85, color=cmap(2 * k % 20 + (k >= 10)), marker="os^D"[k % 4], label=s)
    b = np.polyfit(np.log(P[col]), np.log(P.rmse), 1)
    xs = np.linspace(P[col].min(), P[col].max(), 50)
    ax.plot(xs, np.exp(b[1]) * xs ** b[0], color="black", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel(label)
    ax.set_title(f"R^2 in new studies: {r2:.2f}", loc="left", fontsize=10)
axes[0].set_ylabel("patient's 60-minute forecast RMSE (mg/dL)")
axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
fig.suptitle("What makes a patient hard to forecast? How fast glucose moves, more than how widely it ranges (215 patients, 11 studies)",
             x=0.01, ha="left")
fig.tight_layout(); fig.savefig(WEEK / "figures" / "paper_fig_difficulty.png", dpi=150)
