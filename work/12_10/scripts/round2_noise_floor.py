"""Round 2 (H51-H53): how much of the remaining calibration unevenness is just sampling noise?

For each band method, observed dispersion of coverage (around the method's own overall coverage)
by situation, by patient, and by patient-by-situation cell, next to the dispersion sampling noise
alone would produce:
  situation and patient-by-situation: situation labels circularly shifted within each patient
  patient: moving-block bootstrap standard errors of each patient's coverage (one-day blocks)

Run:    python work/12_10/scripts/round2_noise_floor.py [dataset ...]
Writes: work/12_10/results/round2_noise_floor.csv, verdict_h51_h53.txt, figures/h51_noise_floor.png
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

from azt1d import group_bands as gb, plotting, uq_frames  # noqa: E402
from round1_patient_vs_situation import LABEL, run_for  # noqa: E402

SHOW = ["pooled conformal", "GARCH", "situation (Mondrian)", "patient-scaled situation", "GARCH x situation"]
OUT, FIGS = WEEK / "results", WEEK / "figures"


def floors(t, ins):
    pc = (t.subject_id.astype(str) + "|" + t.cell.astype(str)).to_numpy()
    obs = {"situation": gb.dispersion(ins, t.cell), "patient": gb.dispersion(ins, t.subject_id),
           "patient x situation": gb.dispersion(ins, pc)}
    null = {"situation": gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy())[0],
            "patient": gb.dispersion_null_patient(ins, t.subject_id.to_numpy()),
            # shift only the situation part; the patient part of the label stays fixed
            "patient x situation": _null_pxs(t, ins)}
    return obs, null


def _null_pxs(t, ins, n_rep=50, min_shift=288, seed=0):
    rng = np.random.default_rng(seed)
    cell, sid = t.cell.to_numpy(), t.subject_id.astype(str).to_numpy()
    idx_by = t.groupby("subject_id").indices
    vals = []
    for _ in range(n_rep):
        sh = cell.copy()
        for idx in idx_by.values():
            n = len(idx)
            if n > 2 * min_shift:
                sh[idx] = np.roll(cell[idx], int(rng.integers(min_shift, n - min_shift)))
        vals.append(gb.dispersion(ins, np.char.add(np.char.add(sid, "|"), sh.astype(str))))
    return float(np.mean(vals))


def main(datasets):
    plotting.apply_style()
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        _, inside = gb.evaluate(v, t, methods={k: gb.METHODS[k] for k in SHOW})
        for m in SHOW:
            obs, null = floors(t, inside[m])
            for lvl in obs:
                rows.append({"dataset": name, "method": m, "level": lvl, "observed": obs[lvl], "floor": null[lvl],
                             "excess": obs[lvl] - null[lvl]})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "round2_noise_floor.csv", index=False)
    get = lambda d, m, l: R[(R.dataset == d) & (R.method == m) & (R.level == l)].iloc[0]  # noqa: E731
    h51 = sum(get(d, "situation (Mondrian)", "situation").excess <= 0.01 for d in datasets)
    h52 = sum(get(d, "GARCH x situation", "patient x situation").excess >= 0.03 for d in datasets)
    h53 = sum(get(d, "situation (Mondrian)", "patient").excess >= 0.02 for d in datasets)
    fmt = lambda r: f"{r.observed * 100:.1f} vs floor {r.floor * 100:.1f} (excess {r.excess * 100:+.1f})"  # noqa: E731
    lines = [f"H51 (situation bands' per-situation unevenness within 1 point of noise): {'SUPPORTED' if h51 >= 2 else 'NOT SUPPORTED'} ({h51} of {len(datasets)})",
             f"H52 (GARCH x situation exceeds patient-by-situation floor by >= 3 points): {'SUPPORTED' if h52 >= 2 else 'NOT SUPPORTED'} ({h52} of {len(datasets)})",
             f"H53 (situation bands' per-patient unevenness exceeds floor by >= 2 points): {'SUPPORTED' if h53 >= 2 else 'NOT SUPPORTED'} ({h53} of {len(datasets)})"]
    for d in datasets:
        for m in SHOW:
            lines.append(f"  {d} | {m}: " + "; ".join(f"{l} {fmt(get(d, m, l))}" for l in ["situation", "patient", "patient x situation"]))
    text = "\n".join(lines)
    (OUT / "verdict_h51_h53.txt").write_text(text + "\n")
    print(text)

    levels = ["situation", "patient", "patient x situation"]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 3.6 * len(datasets)), squeeze=False)
    for i, d in enumerate(datasets):
        for j, l in enumerate(levels):
            ax = axes[i, j]
            sub = R[(R.dataset == d) & (R.level == l)].set_index("method").loc[SHOW]
            x = np.arange(len(SHOW))
            ax.bar(x, sub.observed * 100, color="#f58518", width=0.6, label="observed unevenness")
            ax.scatter(x, sub.floor * 100, color="black", marker="_", s=600, lw=2.5, zorder=3, label="sampling-noise floor")
            ax.set_xticks(x, [m.replace(" (Mondrian)", "").replace("patient-scaled", "patient-\nscaled") for m in SHOW],
                          fontsize=7, rotation=20)
            ax.set_title(f"{LABEL.get(d, d)}: by {l}", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel("coverage unevenness (points)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("H51-H53: how much of the bands' unevenness is real, and how much is sampling noise?", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGS / "h51_noise_floor.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
