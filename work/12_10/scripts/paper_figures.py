"""Paper figures: where interval miscalibration lives and what fixes it.

Figure 1: coverage of the 80% band in each of the 25 situations, pooled over datasets, for GARCH,
          situation (Mondrian) and situation x online patient scale.
Figure 2: three axes of honesty per dataset: per-situation excess above the noise floor,
          per-patient excess above the noise floor, and SD of daily coverage.

Run:    python work/12_10/scripts/paper_figures.py [dataset ...]
Writes: work/12_10/figures/paper_fig1_situation_coverage.png, paper_fig2_three_axes.png,
        work/12_10/results/paper_three_axes.csv
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, plotting, uq_frames  # noqa: E402
from round1_patient_vs_situation import LABEL as _LABEL, run_for  # noqa: E402


class _Labels(dict):
    def get(self, k, d=None):
        return _LABEL.get(k, k.replace("panel:", ""))


LABEL = _Labels()

METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale": partial(gb.scale_aci_situation, eta=0.01)}
COLORS = {"GARCH": "#4c78a8", "situation (Mondrian)": "#f58518", "situation x patient scale": "#54a24b"}


def daily_sd(t, ins):
    d = pd.DataFrame({"s": t.subject_id, "day": t.issue_time.dt.floor("D"), "i": ins.astype(float)})
    c = d.groupby(["s", "day"]).i.agg(["mean", "size"])
    return float(c[c["size"] >= 200]["mean"].std())


def main(datasets):
    plotting.apply_style()
    cells, rows = [], []
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for m, ins in inside.items():
            cells.append(pd.DataFrame({"dataset": name, "method": m, "cell": t.cell.to_numpy(), "inside": ins}))
            r = res.set_index("method").loc[m]
            rows.append({"dataset": name, "method": m, "interval_score": r.interval_score, "width": r.width,
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy()),
                         "daily_sd": daily_sd(t, ins)})
        print(name, "done", flush=True)
    C = pd.concat(cells, ignore_index=True)
    A = pd.DataFrame(rows)
    A.to_csv(WEEK / "results" / "paper_three_axes.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"wspace": 0.08})
    for k, (ax, m) in enumerate(zip(axes, METHODS)):
        g = C[C.method == m].groupby("cell").inside.agg(["mean", "size"])
        grid = np.full((5, 5), np.nan)
        for c, r in g.iterrows():
            if r["size"] >= 100:
                grid[c // 5, c % 5] = r["mean"] * 100
        im = ax.imshow(grid, cmap="RdBu", vmin=60, vmax=100, origin="lower")
        for i in range(5):
            for j in range(5):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(grid[i, j] - 80) > 12 else "black")
        ax.set_xticks(range(5), uq_frames.RATE_NAMES, rotation=30, fontsize=8)
        ax.set_yticks(range(5), uq_frames.LEVEL_NAMES if k == 0 else [], fontsize=8)
        ax.set_title(m, loc="left", fontsize=10)
        ax.grid(False)
    axes[0].set_ylabel("current glucose (mg/dL)")
    axes[1].set_xlabel("30-minute trend")
    fig.colorbar(im, ax=axes, shrink=0.8, label="coverage of the 80% band (%)")
    fig.suptitle(f"Where an 80% band is honest: coverage in each glycemic situation ({len(datasets)} studies pooled; 80 is right)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(WEEK / "figures" / "paper_fig1_situation_coverage.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(max(17, 1.6 * len(datasets) * 1.0 + 6), 4.8))
    x = np.arange(len(datasets))
    for k, m in enumerate(METHODS):
        d = A[A.method == m].set_index("dataset").loc[datasets]
        for ax, col, sc in zip(axes, ["excess_situation", "excess_patient", "daily_sd"], [100, 100, 100]):
            ax.bar(x + (k - 1) * 0.27, d[col] * sc, 0.26, color=COLORS[m], label=m)
    for ax, title, yl in zip(axes, ["By situation: unevenness above sampling noise", "By patient: unevenness above sampling noise",
                                    "By day: spread of daily coverage"],
                             ["points above the noise floor", "points above the noise floor", "SD of daily coverage (points)"]):
        ax.set_xticks(x, [LABEL.get(d, d) for d in datasets], rotation=30 if len(datasets) > 4 else 0, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10); ax.set_ylabel(yl); ax.axhline(0, color="black", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Three axes of honesty for an 80% glucose forecast band (lower is better)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(WEEK / "figures" / "paper_fig2_three_axes.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
