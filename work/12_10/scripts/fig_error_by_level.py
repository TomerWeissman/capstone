"""Quick look: how large is the 60-minute forecast error at each glucose level, in mg/dL and as a
percentage of glucose? Pooled test windows from all 11 studies (plain per-patient CNN-LSTMs).

Binned by the actual glucose at the forecast target (20 mg/dL bins, at least 500 windows).
Percent error = |actual - forecast| / actual, the same definition as sensor MARD; the shaded band
is the published MARD range of current CGMs (about 9-14%), for comparison.

Run:    python work/12_10/scripts/fig_error_by_level.py
Writes: work/12_10/figures/error_by_glucose_level.png, work/12_10/results/error_by_glucose_level.csv
"""
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

T = pd.concat([uq_frames.frames(d, run_for(d))[1] for d in DATASETS], ignore_index=True)
T = T[T.actual >= 40]
bins = np.arange(40, 421, 20)
T["bin"] = pd.cut(T.actual, bins)
T["abs_err"] = T.resid.abs()
T["pct_err"] = T.abs_err / T.actual * 100
g = T.groupby("bin", observed=True).agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                                        rmse=("resid", lambda r: np.sqrt(np.mean(r ** 2))),
                                        mard=("pct_err", "mean"), p80=("pct_err", lambda x: np.percentile(x, 80)))
g = g[g.n >= 500]
g["mid"] = [b.mid for b in g.index]
g.to_csv(WEEK / "results" / "error_by_glucose_level.csv")
print(g.round(1).to_string())

plotting.apply_style()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(g.mid, g.mae, "o-", color="#4c78a8", label="mean absolute error")
axes[0].plot(g.mid, g.rmse, "o-", color="#f58518", label="RMSE")
axes[0].set_ylabel("60-minute forecast error (mg/dL)")
axes[0].set_title("In mg/dL: smallest around 120-150, larger at both ends", loc="left", fontsize=10)
axes[1].axhspan(9, 14, color="#dddddd", label="sensor's own error (MARD of current CGMs, about 9-14%)")
axes[1].plot(g.mid, g.mard, "o-", color="#4c78a8", label="mean percentage error (MARD-style)")
axes[1].plot(g.mid, g.p80, "o--", color="#54a24b", label="80% of errors are below this")
axes[1].set_ylabel("60-minute forecast error (% of glucose)")
axes[1].set_title("As a percentage of glucose: error is largest at low glucose", loc="left", fontsize=10)
for ax in axes:
    ax.set_xlabel("actual glucose at the forecast time (mg/dL)")
    for x in (70, 180):
        ax.axvline(x, color="black", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"How big is the 60-minute forecast error at each glucose level? (11 studies, {len(T):,} forecasts; dotted lines at 70 and 180)",
             x=0.01, ha="left")
fig.tight_layout(); fig.savefig(WEEK / "figures" / "error_by_glucose_level.png", dpi=150)
