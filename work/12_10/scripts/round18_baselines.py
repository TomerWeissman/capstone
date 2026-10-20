"""H80: strong uncertainty baselines vs the situation x per-patient scale band.

Baselines (from train_baselines.py):
  - deep ensemble (5 CNN-LSTMs per patient): raw band mean +/- z x SD, and normalized conformal with
    the ensemble SD as the scale (pooled conformal quantile of |y - mean| / SD on validation)
  - conformalized quantile CNN-LSTM: 10%/90% heads; conformal correction from the second half of each
    patient's validation period (pooled over patients)
Ours: situation x per-patient scale, on the plain forecaster (v0) and on the ensemble mean.

Run:    python work/12_10/scripts/round18_baselines.py
Writes: work/12_10/results/round18_baselines.csv, verdict_h80.txt, figures/h80_baselines.png
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
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
from round1_patient_vs_situation import LABEL, run_for  # noqa: E402

PREFIX = {"azt1d": "", "ohiot1dm": "ohiot1dm_", "hupa_ucm": "hupa_ucm_"}
CK = ROOT / "data" / "processed" / "checkpoints"
LEVEL = 0.8


def ensemble_frames(name):
    """Frames whose forecast is the 5-member ensemble mean, plus the ensemble SD per window."""
    path = uq_frames.cache_path(name, "ensemble5")
    if path.exists():
        f = pd.read_parquet(path)
        return f[f.split == "val"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)
    df = uq_frames.load_dataset(name)
    parts = []
    for sid in df.subject_id.unique():
        dfs = df[df.subject_id == sid].reset_index(drop=True)
        members = [unc.forecast_frames(ckpt.load_result(CK / f"{PREFIX[name]}cnn_lstm_ens{k}", sid), dfs) for k in range(1, 6)]
        fr, sds = [], []
        for j in range(2):
            P = np.stack([m[j].pred for m in members])
            base = members[0][j]
            mean = P.mean(0)
            fr.append(unc.Frame(actual=base.actual, pred=mean, resid=base.actual - mean, X=None,
                                issue_time=base.issue_time, target_time=base.target_time))
            sds.append(pd.DataFrame({"issue_time": pd.to_datetime(base.issue_time), "ens_sd": P.std(0)}))
        v, t = uq_frames.subject_frames(dfs, None, forecasts=tuple(fr))
        parts += [v.merge(sds[0], on="issue_time", how="left"), t.merge(sds[1], on="issue_time", how="left")]
    f = pd.concat(parts, ignore_index=True)
    f["subject_id"] = f.subject_id.astype(str)
    f.to_parquet(path, index=False)
    return f[f.split == "val"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)


def ens_raw(v, t, level):
    return t.pred.to_numpy(), gb.Z[level] * t.ens_sd.to_numpy()


def ens_normalized(v, t, level):
    eps = 1e-3
    q = gb.conformal_q(np.abs(v.resid) / (v.ens_sd + eps), level)
    return t.pred.to_numpy(), q * (t.ens_sd.to_numpy() + eps)


def quantile_cqr(name):
    Q = pd.read_parquet(WEEK / "results" / "quantile_net_preds.parquet")
    Q = Q[Q.dataset == name].copy()
    Q["issue_time"] = pd.to_datetime(Q.issue_time)

    def method(v, t, level):
        qv = v.merge(Q[Q.split == "val"], on=["subject_id", "issue_time"], how="left")
        qt = t.merge(Q[Q.split == "test"], on=["subject_id", "issue_time"], how="left")
        second = (qv.groupby("subject_id").cumcount() >= qv.groupby("subject_id").subject_id.transform("size") / 2).to_numpy()
        b = qv[second].dropna(subset=["q10"])
        y = b.actual.to_numpy()
        q = gb.conformal_q(np.maximum(b.q10.to_numpy() - y, y - b.q90.to_numpy()), level)
        lo, hi = qt.q10.to_numpy() - q, qt.q90.to_numpy() + q
        return (lo + hi) / 2, (hi - lo) / 2
    return method


def evaluate(name, label, v, t, methods):
    res, inside = gb.evaluate(v, t, LEVEL, methods=methods)
    rows = []
    for _, r in res.iterrows():
        ins = inside[r.method]
        rows.append({"dataset": name, "forecaster": label, **r.to_dict(),
                     "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                     "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
    return rows


def main(names=None):
    plotting.apply_style()
    rows = []
    ours = partial(gb.scale_aci_situation, eta=0.01)
    global PREFIX
    if names:  # e.g. only the datasets whose baselines finished
        PREFIX = {k: v for k, v in PREFIX.items() if k in names}
    for name in PREFIX:
        v0, t0 = uq_frames.frames(name, run_for(name))
        rows += evaluate(name, "plain CNN-LSTM", v0, t0, {"situation x patient scale (ours)": ours, "quantile CNN-LSTM (CQR)": quantile_cqr(name)})
        ve, te = ensemble_frames(name)
        rows += evaluate(name, "ensemble mean", ve, te, {"ensemble, mean +/- z SD": ens_raw, "ensemble, normalized conformal": ens_normalized,
                                                         "situation x patient scale (ours)": ours})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round18_baselines.csv", index=False)
    g = lambda d, f, m: R[(R.dataset == d) & (R.forecaster == f) & (R.method == m)].iloc[0]  # noqa: E731
    ok = 0
    for d in PREFIX:
        o = g(d, "plain CNN-LSTM", "situation x patient scale (ours)")
        b1, b2 = g(d, "ensemble mean", "ensemble, normalized conformal"), g(d, "plain CNN-LSTM", "quantile CNN-LSTM (CQR)")
        better_is = min(b1.interval_score, b2.interval_score)
        ok += (o.excess_situation < min(b1.excess_situation, b2.excess_situation)
               and o.excess_patient < min(b1.excess_patient, b2.excess_patient)
               and o.interval_score <= 1.03 * better_is)
    cols = ["coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient"]
    n = len(PREFIX)
    text = f"H80 (ours beats ensemble-normalized conformal and conformalized quantile network on both excesses, interval score within 3%): {'SUPPORTED' if ok >= 2 else 'NOT SUPPORTED'} ({ok} of {n}; rule: at least 2 of 3 datasets)\n"
    for d in PREFIX:
        sub = R[R.dataset == d].set_index(["forecaster", "method"])[cols].copy()
        sub[["coverage", "err_situation", "excess_situation", "excess_patient"]] *= 100
        text += f"{d}:\n" + sub.round(2).to_string() + "\n"
    (WEEK / "results" / "verdict_h80.txt").write_text(text)
    print(text)

    show = [("plain CNN-LSTM", "situation x patient scale (ours)", "#54a24b"), ("ensemble mean", "situation x patient scale (ours)", "#2f7d32"),
            ("ensemble mean", "ensemble, normalized conformal", "#4c78a8"), ("plain CNN-LSTM", "quantile CNN-LSTM (CQR)", "#f58518"),
            ("ensemble mean", "ensemble, mean +/- z SD", "#bbbbbb")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    x = np.arange(len(PREFIX))
    for k, (fc, m, c) in enumerate(show):
        d_ = R[(R.forecaster == fc) & (R.method == m)].set_index("dataset").loc[list(PREFIX)]
        lab = m if "ours" not in m else f"ours ({fc})"
        for ax, col, sc in zip(axes, ["coverage", "excess_situation", "excess_patient", "interval_score"], [100, 100, 100, 1]):
            ax.bar(x + (k - 2) * 0.16, d_[col] * sc, 0.15, color=c, label=lab)
    for ax, t_ in zip(axes, ["Overall coverage (target 80)", "Situation unevenness above noise", "Patient unevenness above noise", "Interval score (lower is better)"]):
        ax.set_xticks(x, [LABEL[d] for d in PREFIX]); ax.set_title(t_, loc="left", fontsize=10)
    axes[0].axhline(80, color="black", lw=1)
    axes[3].set_ylim(R.interval_score.min() * 0.9, min(R.interval_score.max(), 250) * 1.02)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("H80: our band vs deep ensembles and a quantile network", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(WEEK / "figures" / "h80_baselines.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
