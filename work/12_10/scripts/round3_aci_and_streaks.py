"""Round 3 (H57, H58): do GARCH-scaled bands spread misses out in time, and does adaptive conformal
inference per patient remove the real between-patient miscalibration?

Run:    python work/12_10/scripts/round3_aci_and_streaks.py [dataset ...]
Writes: work/12_10/results/round3_aci.csv, verdict_h57_h59.txt, figures/h57_h59_aci.png
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
from round1_patient_vs_situation import LABEL, run_for  # noqa: E402

OUT, FIGS = WEEK / "results", WEEK / "figures"
METHODS = {
    "GARCH": gb.garch,
    "situation (Mondrian)": gb.situation_mondrian,
    "GARCH x situation": gb.garch_situation,
    "situation + ACI (0.001)": partial(gb.aci_situation, gamma=0.001),
    "situation + ACI (0.005)": partial(gb.aci_situation, gamma=0.005),
    "situation + ACI (0.02)": partial(gb.aci_situation, gamma=0.02),
    "situation x patient scale (0.003)": partial(gb.scale_aci_situation, eta=0.003),
    "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01),
    "situation x patient scale (0.03)": partial(gb.scale_aci_situation, eta=0.03),
}


def main(datasets):
    plotting.apply_style()
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"dataset": name, **r.to_dict(),
                         "miss_run_length": gb.miss_run_length(ins, t.subject_id),
                         "floor_situation": gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "floor_patient": gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R["excess_situation"] = R.disp_situation - R.floor_situation
    R["excess_patient"] = R.disp_patient - R.floor_patient
    R.to_csv(OUT / "round3_aci.csv", index=False)

    g = lambda d, m: R[(R.dataset == d) & (R.method == m)].iloc[0]  # noqa: E731
    h57 = sum(g(d, "GARCH x situation").miss_run_length <= 0.85 * g(d, "situation (Mondrian)").miss_run_length for d in datasets)
    main_aci = "situation + ACI (0.005)"
    h58 = sum((g(d, main_aci).excess_patient <= 0.01) and (g(d, main_aci).excess_situation <= 0.015)
              and (g(d, main_aci).interval_score <= 1.03 * g(d, "situation (Mondrian)").interval_score) for d in datasets)
    main_s = "situation x patient scale (0.01)"
    h59 = sum((g(d, main_s).excess_patient <= 0.01) and (g(d, main_s).excess_situation <= 0.015)
              and (g(d, main_s).interval_score <= 1.03 * g(d, "situation (Mondrian)").interval_score) for d in datasets)
    lines = [f"H59 (scale-adaptive, eta 0.01, same criteria as H58): {'SUPPORTED' if h59 >= 2 else 'NOT SUPPORTED'} ({h59} of {len(datasets)})",
             f"H57 (GARCH x situation miss runs >= 15% shorter than situation bands): {'SUPPORTED' if h57 >= 2 else 'NOT SUPPORTED'} ({h57} of {len(datasets)})",
             f"H58 (ACI 0.005 per-patient excess <= 1, situation excess <= 1.5, interval score <= +3%): {'SUPPORTED' if h58 >= 2 else 'NOT SUPPORTED'} ({h58} of {len(datasets)})"]
    cols = ["coverage", "width", "interval_score", "miss_run_length", "excess_situation", "excess_patient"]
    for d in datasets:
        sub = R[R.dataset == d].set_index("method")[cols].copy()
        sub[["coverage", "excess_situation", "excess_patient"]] *= 100
        lines.append(f"{d}:\n" + sub.round(2).to_string())
    text = "\n".join(lines)
    (OUT / "verdict_h57_h59.txt").write_text(text + "\n")
    print(text)

    show = list(METHODS)
    colors = ["#4c78a8", "#f58518", "#b279a2", "#c7e9c0", "#54a24b", "#00441b", "#fdd0a2", "#e6550d", "#7f2704"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(datasets))
    for k, m in enumerate(show):
        d = R[R.method == m].set_index("dataset").loc[datasets]
        for ax, col, scale in zip(axes, ["miss_run_length", "excess_patient", "interval_score"], [1, 100, 1]):
            ax.bar(x + (k - 4) * 0.09, d[col] * scale, width=0.085, color=colors[k], label=m)
    for ax, title, yl in zip(axes, ["H57: how long do runs of misses last?", "H58: real per-patient miscalibration (above noise)",
                                    "Cost: interval score (lower is better)"],
                             ["mean consecutive misses (windows of 5 min)", "points above the noise floor", "interval score"]):
        ax.set_xticks(x, [LABEL.get(d, d) for d in datasets]); ax.set_title(title, loc="left", fontsize=10); ax.set_ylabel(yl)
    axes[1].axhline(1, color="black", lw=1, ls="--")
    axes[2].set_ylim(R.interval_score.min() * 0.95, R.interval_score.max() * 1.02)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "h57_h59_aci.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
