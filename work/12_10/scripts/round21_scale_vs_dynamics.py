"""H85: does the band's learned per-patient scale reflect the patient's glucose dynamics?

For each patient: cold situation quantiles (other patients of the same dataset), the online scale
(eta 0.01); learned scale = median of half-width / cold situation half-width over the last 576
test windows (two days). Compared with the rate-of-change SD from round15_patients.csv.

Run:    python work/12_10/scripts/round21_scale_vs_dynamics.py
Writes: work/12_10/results/round21_scales.csv, verdict_h85.txt, figures/h85_scale_vs_dynamics.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, plotting, uq_frames  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402
from round6_panel import PANEL  # noqa: E402


def learned_scales(name):
    v, t = uq_frames.frames(name, run_for(name))
    out = {}
    for sid, idx in t.groupby("subject_id").indices.items():
        vo, tp = v[v.subject_id != sid], t.iloc[idx]
        _, base = gb.situation_mondrian(vo, tp, 0.8)
        _, adapted = gb.scale_aci_situation(vo, tp, 0.8, eta=0.01)
        out[sid] = float(np.median((adapted / base)[-576:]))
    return out


def main():
    P = pd.read_csv(WEEK / "results" / "round15_patients.csv", dtype={"subject_id": str})
    rows = []
    for name in ["azt1d", "ohiot1dm", "hupa_ucm", *[f"panel:{s}" for s in PANEL]]:
        for sid, s in learned_scales(name).items():
            rows.append({"dataset": name, "subject_id": sid, "learned_scale": s})
        print(name, "done", flush=True)
    S = pd.DataFrame(rows).merge(P[["dataset", "subject_id", "source", "rate_sd", "glucose_sd", "rmse"]], on=["dataset", "subject_id"])
    S.to_csv(WEEK / "results" / "round21_scales.csv", index=False)
    rho_rate = spearmanr(S.learned_scale, S.rate_sd).statistic
    rho_sd = spearmanr(S.learned_scale, S.glucose_sd).statistic
    within = np.mean([spearmanr(d.learned_scale, d.rate_sd).statistic for _, d in S.groupby("source") if len(d) >= 10])
    text = "\n".join([
        f"H85 (learned scale vs rate-of-change SD, Spearman >= 0.4): {'SUPPORTED' if rho_rate >= 0.4 else 'NOT SUPPORTED'}; rho {rho_rate:.2f}",
        f"  vs glucose SD: rho {rho_sd:.2f}; mean within-study rho (rate SD) {within:.2f}; patients {len(S)}",
        f"  learned scale: median {S.learned_scale.median():.2f}, 10th-90th percentile {S.learned_scale.quantile(0.1):.2f} to {S.learned_scale.quantile(0.9):.2f}",
    ])
    (WEEK / "results" / "verdict_h85.txt").write_text(text + "\n")
    print(text)
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("tab20")
    for k, (s, d) in enumerate(S.groupby("source")):
        ax.scatter(d.rate_sd, d.learned_scale, s=22, alpha=0.85, color=cmap(2 * k % 20 + (k >= 10)), marker="os^D"[k % 4], label=s)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xlabel("patient's SD of 30-minute glucose change (mg/dL)")
    ax.set_ylabel("band scale learned online (1 = population width)")
    ax.set_title(f"H85: does the learned patient scale track glucose dynamics? (Spearman {rho_rate:.2f})", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(WEEK / "figures" / "h85_scale_vs_dynamics.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
