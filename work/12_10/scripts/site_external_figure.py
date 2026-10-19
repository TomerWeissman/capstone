"""Portfolio-site figure: H54 external validation as a simple per-study dumbbell chart.

Reads the per-study table in verdict_h54_h56_h60.txt (GARCH vs situation bands, per-situation
distance from 80%) and draws one row per MetaboNet study. Plain-language labels for a lay reader.

Run:    python work/12_10/scripts/site_external_figure.py
Writes: work/12_10/figures/site_external_validation.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import plotting  # noqa: E402

GREY = "#8a8985"  # GARCH, matching the grey used in the earlier band figures
BLUE = plotting.CATEGORICAL[0]  # situation bands


def read_studies(path: Path) -> list[dict]:
    """One dict per study: name, patients, GARCH and situation per-situation error (column 5)."""
    studies, current = [], None
    for line in path.read_text().splitlines():
        head = re.match(r"^(\w+) \((\d+) patients;", line)
        if head:
            current = {"name": head.group(1), "patients": int(head.group(2))}
            studies.append(current)
            continue
        row = re.match(r"^\s+(GARCH|situation \(Mondrian\))\s+([\d.\-\s]+)$", line)
        if current and row:
            values = [float(v) for v in row.group(2).split()]
            current["garch" if row.group(1) == "GARCH" else "situation"] = values[3]  # err_situation
    return studies


def main() -> None:
    plotting.apply_style()
    studies = sorted(read_studies(WEEK / "results" / "verdict_h54_h56_h60.txt"), key=lambda s: s["garch"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, s in enumerate(studies):
        ax.plot([s["situation"], s["garch"]], [i, i], color=plotting.BASELINE, lw=2, zorder=1)
        ax.scatter(s["garch"], i, s=70, color=GREY, zorder=2, edgecolor=plotting.SURFACE, linewidth=2)
        ax.scatter(s["situation"], i, s=70, color=BLUE, zorder=2, edgecolor=plotting.SURFACE, linewidth=2)
    ax.set_yticks(range(len(studies)), [f"{s['name']} ({s['patients']} patients)" for s in studies])
    ax.set_xlim(0, 7.5)
    ax.set_xlabel("How far the 80% range's catch rate was from 80%, averaged over situations\n(percentage points; closer to 0 is better)")
    top = studies[-1]
    ax.annotate("standard range (GARCH)", (top["garch"], len(studies) - 1), xytext=(0, 12),
                textcoords="offset points", ha="center", color=plotting.INK_SECONDARY, fontsize=9)
    ax.annotate("sized by situation", (top["situation"], len(studies) - 1), xytext=(0, 12),
                textcoords="offset points", ha="center", color=plotting.INK_SECONDARY, fontsize=9)
    ax.set_ylim(-0.6, len(studies) - 0.2)
    ax.set_title("8 outside studies, never used to design the method: closer to 80% in every one",
                 loc="left", fontsize=11)
    fig.tight_layout()
    out = WEEK / "figures" / "site_external_validation.png"
    fig.savefig(out, dpi=200)
    print(out, [(s["name"], s["garch"], s["situation"]) for s in studies])


if __name__ == "__main__":
    main()
