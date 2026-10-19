"""Round 1 (H49, H50): is forecast uncertainty a property of the patient or of the situation, and
does a band that scales by patient and groups by situation get both right?

H49: R^2 of the test-period absolute error from group means fitted on validation:
     constant, patient, situation (25 cells), patient x situation (multiplicative), time of day
     (8 three-hour bins), GARCH scale. Patient-minus-situation R^2 with a patient-bootstrap interval.
H50: band methods from azt1d.group_bands at 80%, calibration error by situation, by patient and by
     patient-by-situation cell; interval score; bootstrap interval of the patient-by-situation gain.

Run:    python work/12_10/scripts/round1_patient_vs_situation.py [dataset ...]
        default datasets: azt1d ohiot1dm hupa_ucm; panel sources as panel:<Source>
Writes: work/12_10/results/round1_<dataset>.csv, round1_r2.csv, verdict_h49_h50.txt,
        work/12_10/figures/h49_error_sources.png, h50_patient_scaled_bands.png
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

from azt1d import group_bands as gb, plotting, uq_frames  # noqa: E402

RUNS = {"azt1d": "cnn_lstm_v0", "ohiot1dm": "ohiot1dm_cnn_lstm_v0", "hupa_ucm": "hupa_ucm_cnn_lstm_v0"}
LABEL = {"azt1d": "AZT1D", "ohiot1dm": "OhioT1DM", "hupa_ucm": "HUPA-UCM"}
OUT, FIGS = WEEK / "results", WEEK / "figures"


def run_for(name):
    if name in RUNS:
        return RUNS[name]
    tag, src = name.split(":")
    return f"{tag}_{src}_cnn_lstm_v0"  # panel_<src>_... or panel60_<src>_...


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def error_predictors(v, t):
    """Predicted absolute test error from each grouping, fitted on validation only."""
    av, at = np.abs(v.resid), np.abs(t.resid)
    pooled = av.mean()
    pat = v.assign(a=av).groupby("subject_id").a.mean()
    cell = v.assign(a=av).groupby("cell").a.mean()
    ratio = v.assign(r=av / v.subject_id.map(pat)).groupby("cell").r.mean()
    tod_v, tod_t = (v.hour // 3).astype(int), (t.hour // 3).astype(int)
    tod = v.assign(a=av, b=tod_v).groupby("b").a.mean()
    k_garch = av.mean() / v.sigma.mean()  # GARCH scale rescaled on validation so its average matches
    g_ratio = v.assign(a=av).groupby("cell").apply(lambda d: d.a.sum() / d.sigma.sum())  # ratio of sums: stable when sigma is tiny
    return at.to_numpy(), {
        "constant": np.full(len(t), pooled),
        "time of day": tod_t.map(tod).fillna(pooled).to_numpy(),
        "GARCH scale": (t.sigma * k_garch).to_numpy(),
        "GARCH x situation": (t.sigma * t.cell.map(g_ratio).fillna(k_garch)).to_numpy(),
        "situation": t.cell.map(cell).fillna(pooled).to_numpy(),
        "patient": t.subject_id.map(pat).to_numpy(),
        "patient x situation": (t.subject_id.map(pat) * t.cell.map(ratio).fillna(1.0)).to_numpy(),
    }


def main(datasets):
    plotting.apply_style()
    r2_rows, band_rows, lines = [], [], []
    wins49, wins50 = 0, 0
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        y, preds = error_predictors(v, t)
        for k, p in preds.items():
            r2_rows.append({"dataset": name, "predictor": k, "r2": r2(y, p)})
        diff = r2(y, preds["patient"]) - r2(y, preds["situation"])
        ci49 = gb.patient_bootstrap(t, lambda i: r2(y[i], preds["patient"][i]) - r2(y[i], preds["situation"][i]))
        wins49 += diff > 0

        res, ins = gb.evaluate(v, t)
        res.insert(0, "dataset", name)
        res.to_csv(OUT / f"round1_{name.replace(':', '_')}.csv", index=False)
        band_rows.append(res)
        r = res.set_index("method")
        e_new, e_sit = r.loc["patient-scaled situation", "err_patient_x_situation"], r.loc["situation (Mondrian)", "err_patient_x_situation"]
        gain = 1 - e_new / e_sit
        is_ok = r.loc["patient-scaled situation", "interval_score"] <= 1.02 * r.loc["situation (Mondrian)", "interval_score"]
        pc = (t.subject_id.astype(str) + "|" + t.cell.astype(str)).to_numpy()
        ci50 = gb.patient_bootstrap(t, lambda i: gb.cov_err(ins["situation (Mondrian)"][i], pc[i], 0.8)
                                    - gb.cov_err(ins["patient-scaled situation"][i], pc[i], 0.8), n_boot=300)
        wins50 += (gain >= 0.20) and is_ok
        lines.append(f"{name}: R2 patient {r2(y, preds['patient']):.3f}, situation {r2(y, preds['situation']):.3f}, "
                     f"patient x situation {r2(y, preds['patient x situation']):.3f}; patient minus situation {diff:+.3f} "
                     f"(90% patient bootstrap {ci49[0]:+.3f} to {ci49[1]:+.3f})")
        lines.append(f"  bands: patient-by-situation error {e_sit * 100:.1f} -> {e_new * 100:.1f} points ({gain:+.0%}; "
                     f"bootstrap gain {ci50[0] * 100:+.1f} to {ci50[1] * 100:+.1f} points); interval score "
                     f"{r.loc['situation (Mondrian)', 'interval_score']:.1f} -> {r.loc['patient-scaled situation', 'interval_score']:.1f}")
        print(lines[-2]); print(lines[-1]); print(res.round(3).to_string(index=False), flush=True)

    R = pd.DataFrame(r2_rows)
    R.to_csv(OUT / "round1_r2.csv", index=False)
    B = pd.concat(band_rows, ignore_index=True)
    n = len(datasets)
    text = "\n".join([f"H49 (patient explains more of the error size than situation): {'SUPPORTED' if wins49 >= 2 else 'NOT SUPPORTED'} ({wins49} of {n})",
                      f"H50 (patient-scaled situation bands >= 20% better per patient-by-situation cell): {'SUPPORTED' if wins50 >= 2 else 'NOT SUPPORTED'} ({wins50} of {n})",
                      *lines])
    (OUT / "verdict_h49_h50.txt").write_text(text + "\n")
    print(text)

    preds_order = ["constant", "time of day", "GARCH scale", "situation", "GARCH x situation", "patient", "patient x situation"]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), datasets):
        d = R[R.dataset == name].set_index("predictor").loc[preds_order]
        ax.bar(range(len(d)), d.r2 * 100, color=["#bbbbbb", "#bbbbbb", "#4c78a8", "#f58518", "#b279a2", "#54a24b", "#2f7d32"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(d)), preds_order, rotation=35, ha="right", fontsize=8)
        ax.set_title(LABEL.get(name, name), loc="left")
    np.atleast_1d(axes)[0].set_ylabel("% of variation in |test error| explained")
    fig.suptitle("H49: what predicts how large the 60-minute forecast error will be? (fitted on validation, scored on test)",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGS / "h49_error_sources.png", dpi=150); plt.close(fig)

    show = ["pooled conformal", "per-patient conformal", "GARCH", "situation (Mondrian)", "patient-scaled situation", "GARCH x situation"]
    metrics = [("err_situation", "by situation"), ("err_patient", "by patient"), ("err_patient_x_situation", "by patient x situation")]
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6), sharey=True)
    colors = ["#bbbbbb", "#72b7b2", "#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for ax, name in zip(np.atleast_1d(axes), datasets):
        d = B[B.dataset == name].set_index("method").loc[show]
        x = np.arange(3)
        for k, m in enumerate(show):
            ax.bar(x + (k - 2.5) * 0.14, [d.loc[m, c] * 100 for c, _ in metrics], width=0.13, color=colors[k], label=m)
        ax.set_xticks(x, [l for _, l in metrics])
        ax.set_title(LABEL.get(name, name), loc="left")
    np.atleast_1d(axes)[0].set_ylabel("mean distance from 80% coverage (points)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("H50: which band is honest for every patient, every situation, and every patient in every situation?", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGS / "h50_patient_scaled_bands.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
