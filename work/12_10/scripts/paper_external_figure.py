"""Figure 2 of the capstone paper: average distance from 80% coverage across situations on the
eight held-out MetaboNet datasets, GARCH vs the situation-aware band only.

Reads the saved round 6 results, no retraining.

Run:    python work/12_10/scripts/paper_external_figure.py
Writes: capstone/paper/figures/external_situation_distance.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
PANEL = ["Loop", "DCLP3", "DCLP5", "PEDAP", "Flair", "IOBP2", "ReplaceBG", "BrisT1D"]
METHODS = [("GARCH", "GARCH band", "#4c78a8"), ("situation (Mondrian)", "Situation-aware band", "#f58518")]


def main():
    bands = pd.read_csv(WEEK / "results" / "round6_panel_bands.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(PANEL))
    for k, (method, label, color) in enumerate(METHODS):
        d = bands[bands.method == method].set_index("source").loc[PANEL]
        ax.bar(x + (k - 0.5) * 0.38, d.err_situation * 100, 0.36, color=color, label=label)
    ax.set_xticks(x, PANEL, rotation=30, fontsize=9)
    ax.set_ylabel("Average distance from 80% (points)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "capstone" / "paper" / "figures" / "external_situation_distance.png", dpi=200)


if __name__ == "__main__":
    main()
