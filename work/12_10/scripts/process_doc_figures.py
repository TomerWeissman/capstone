"""Figures for the process documentation (capstone/process_doc/figures/), built from existing results.

1. clarke_explainer.png: one patient's 60-minute forecasts on the Clarke error grid, zones A to E labeled.
2. situation_vs_garch.png: coverage of the 80% band in each glucose situation, GARCH vs situation-aware
   (11 studies pooled; same data as paper Figure 1, two panels for readability).
3. alarm_garch_patient.png: the GARCH panel cropped from the week 21_9 alarm figure.
4. garch_band_zones.png: the Sep 28 notebook figure (wider GARCH bands vs Clarke zones), copied.

Run:    python work/12_10/scripts/process_doc_figures.py
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, plotting, uq_frames  # noqa: E402
from azt1d.glimmer import clinical  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402
from round6_panel import PANEL  # noqa: E402

OUT = ROOT / "capstone" / "process_doc" / "figures"


def clarke_explainer():
    _, t = uq_frames.frames("azt1d", "cnn_lstm_v0")
    p = t[t.subject_id == "1"]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    clinical.draw_clarke_grid(ax, p.actual.to_numpy(), p.pred.to_numpy(), "#4c78a8")
    for label, (x, y) in {"A": (40, 20), "A ": (330, 375), "B": (350, 250), "B ": (260, 360), "C": (130, 330),
                          "C ": (160, 25), "D": (25, 120), "D ": (330, 120), "E": (25, 330), "E ": (330, 25)}.items():
        ax.text(x, y, label.strip(), fontsize=18, weight="bold", color="#333333", ha="center", va="center")
    ax.set_xlabel("actual glucose (mg/dL)")
    ax.set_ylabel("forecast made 60 minutes earlier (mg/dL)")
    ax.set_title("Clarke error grid: how harmful each forecast error would be", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "clarke_explainer.png", dpi=170); plt.close(fig)


def situation_vs_garch():
    methods = {"Standard band (GARCH)": gb.garch, "Situation-aware band": gb.situation_mondrian}
    cells = []
    for name in ["azt1d", "ohiot1dm", "hupa_ucm", *[f"panel:{s}" for s in PANEL]]:
        v, t = uq_frames.frames(name, run_for(name))
        _, inside = gb.evaluate(v, t, methods=methods)
        for m, ins in inside.items():
            cells.append(pd.DataFrame({"method": m, "cell": t.cell.to_numpy(), "inside": ins}))
    C = pd.concat(cells, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"wspace": 0.08})
    for k, (ax, m) in enumerate(zip(axes, methods)):
        g = C[C.method == m].groupby("cell").inside.agg(["mean", "size"])
        grid = np.full((5, 5), np.nan)
        for c, r in g.iterrows():
            if r["size"] >= 100:
                grid[c // 5, c % 5] = r["mean"] * 100
        im = ax.imshow(grid, cmap="RdBu", vmin=60, vmax=100, origin="lower")
        for i in range(5):
            for j in range(5):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=13,
                            color="white" if abs(grid[i, j] - 80) > 12 else "black")
        ax.set_xticks(range(5), uq_frames.RATE_NAMES, rotation=30, fontsize=11)
        ax.set_yticks(range(5), uq_frames.LEVEL_NAMES if k == 0 else [], fontsize=11)
        ax.set_title(m, loc="left", fontsize=14)
        ax.grid(False)
    axes[0].set_ylabel("current glucose (mg/dL)", fontsize=12)
    fig.supxlabel("direction of glucose over the last 30 minutes", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.8, label="% of real values inside the 80% band (80 is right)")
    fig.savefig(OUT / "situation_vs_garch.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def copies():
    im = Image.open(ROOT / "work" / "12_10" / "figures" / "alarms_three_methods_patient_a_21_9.png")
    w, h = im.size
    im.crop((0, int(h * 0.355), w, int(h * 0.665))).save(OUT / "alarm_garch_patient.png")
    nb = nbformat.read(ROOT / "work" / "21_9" / "notebooks" / "02_one_patient_view.ipynb", 4)
    o = next(o for o in nb.cells[11].outputs if "image/png" in o.get("data", {}))
    (OUT / "garch_band_zones.png").write_bytes(base64.b64decode(o["data"]["image/png"]))
    for f in ("methods_tradeoff_21_9.png",):
        (OUT / f).write_bytes((ROOT / "work" / "12_10" / "figures" / f).read_bytes())


if __name__ == "__main__":
    plotting.apply_style()
    clarke_explainer()
    situation_vs_garch()
    copies()
    print("done")
