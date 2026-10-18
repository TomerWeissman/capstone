"""Round 4 (H61-H63): how much of the error size can machine learning predict from rich context,
which features matter, and does a learned band (CQR) beat the situation band with per-patient
scale adaptation?

Features per window (all known when the forecast is issued):
  level, 30-min trend, 2-h trend, carbs and bolus in the last 2 h, hour (sin, cos), forecast value,
  GARCH scale, the patient's validation error scale, the most recently observed error (window issued
  60 min earlier, signed and absolute), and the mean absolute error of the windows observed in the
  2 hours before that.
H61: GBM on |error| trained on validation, R^2 on test vs the patient x situation group means.
H62: permutation importance on test (drop in R^2).
H63: CQR (GBM quantiles 10% and 90% of the residual, fitted on the first half of each patient's
     validation period, conformalized on the second half) vs scale-adaptive situation bands.

Run:    python work/12_10/scripts/round4_ml_error_size.py [dataset ...]
Writes: work/12_10/results/round4_ml.csv, round4_importance.csv, verdict_h61_h64.txt,
        work/12_10/figures/h61_h62_ml_error_size.png
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, plotting, uq_frames  # noqa: E402
from round1_patient_vs_situation import LABEL, run_for  # noqa: E402

OUT, FIGS = WEEK / "results", WEEK / "figures"
LAG = 12  # an error becomes known 60 minutes after its forecast was issued
FEATURES = ["g_now", "trend_30m", "trend_2h", "carbs_2h", "bolus_2h", "hour_sin", "hour_cos", "pred", "sigma",
            "patient_scale", "last_err", "last_abs_err", "recent_abs_err"]
NICE = {"g_now": "glucose level", "trend_30m": "30-min trend", "trend_2h": "2-hour trend", "carbs_2h": "carbs (2 h)",
        "bolus_2h": "bolus (2 h)", "hour_sin": "time of day (sin)", "hour_cos": "time of day (cos)",
        "pred": "forecast value", "sigma": "GARCH scale", "patient_scale": "patient error scale",
        "last_err": "latest known error (signed)", "last_abs_err": "latest known error (size)",
        "recent_abs_err": "known errors, prior 2 h (size)"}


def add_features(v, t):
    f = pd.concat([v, t], ignore_index=True).sort_values(["subject_id", "issue_time"]).reset_index(drop=True)
    g = f.groupby("subject_id")
    f["trend_30m"] = f.g_now - f.g_prev
    f["trend_2h"] = f.g_now - f.g_2h
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * f.hour / 24), np.cos(2 * np.pi * f.hour / 24)
    f["last_err"] = g.resid.shift(LAG)
    f["last_abs_err"] = f.last_err.abs()
    f["recent_abs_err"] = g.resid.transform(lambda r: r.abs().shift(LAG + 1).rolling(24, min_periods=6).mean())
    scale = f[f.split == "val"].groupby("subject_id").resid.apply(lambda r: np.abs(r).mean())
    f["patient_scale"] = f.subject_id.map(scale)
    f = f.dropna(subset=["last_err", "recent_abs_err"])
    return f[f.split == "val"].reset_index(drop=True), f[f.split == "test"].reset_index(drop=True)


def gbm(**kw):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=100, random_state=0, **kw)


def ml_cqr(v, t, level):
    """CQR with the rich features: quantile models on the first half of each patient's validation
    period, conformal correction from the second half."""
    first = (v.groupby("subject_id").cumcount() < v.groupby("subject_id").subject_id.transform("size") / 2).to_numpy()
    a, b = v[first], v[~first]
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    m_lo = gbm(loss="quantile", quantile=lo_q).fit(a[FEATURES], a.resid)
    m_hi = gbm(loss="quantile", quantile=hi_q).fit(a[FEATURES], a.resid)
    score = np.maximum(m_lo.predict(b[FEATURES]) - b.resid, b.resid - m_hi.predict(b[FEATURES]))
    q = gb.conformal_q(score, level)
    lo, hi = m_lo.predict(t[FEATURES]) - q, m_hi.predict(t[FEATURES]) + q
    centre = t.pred.to_numpy() + (lo + hi) / 2
    return centre, (hi - lo) / 2


def mondrian_cqr_scaled(v, t, level, eta=0.01, delay=LAG, min_n=100):
    """H64: rich-feature CQR, conformal correction per situation, times a per-patient online scale."""
    first = (v.groupby("subject_id").cumcount() < v.groupby("subject_id").subject_id.transform("size") / 2).to_numpy()
    a, b = v[first], v[~first]
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    m_lo = gbm(loss="quantile", quantile=lo_q).fit(a[FEATURES], a.resid)
    m_hi = gbm(loss="quantile", quantile=hi_q).fit(a[FEATURES], a.resid)
    score = np.maximum(m_lo.predict(b[FEATURES]) - b.resid, b.resid - m_hi.predict(b[FEATURES]))
    q = gb.group_q(score, b.cell, t.cell, level, min_n=min_n)
    lo, hi = m_lo.predict(t[FEATURES]), m_hi.predict(t[FEATURES])
    centre = t.pred.to_numpy() + (lo + hi) / 2
    base = np.maximum((hi - lo) / 2 + q, 0)
    y, target, half = t.actual.to_numpy(), 1 - level, np.zeros(len(t))
    for idx in t.groupby("subject_id").indices.values():
        theta, miss = 0.0, np.zeros(len(idx))
        for j, r in enumerate(idx):
            half[r] = base[r] * np.exp(theta)
            miss[j] = abs(y[r] - centre[r]) > half[r]
            theta += eta * ((miss[j - delay] if j >= delay else target) - target)
    return centre, half


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main(datasets):
    plotting.apply_style()
    r2_prev = pd.read_csv(OUT / "round1_r2.csv")
    rows, imps, band_rows = [], [], []
    for name in datasets:
        v0, t0 = uq_frames.frames(name, run_for(name))
        v, t = add_features(v0, t0)
        model = gbm().fit(v[FEATURES], np.abs(v.resid))
        y = np.abs(t.resid).to_numpy()
        r2_ml = r2(y, model.predict(t[FEATURES]))
        base = float(r2_prev[(r2_prev.dataset == name) & (r2_prev.predictor == "patient x situation")].r2.iloc[0])
        rows.append({"dataset": name, "r2_ml": r2_ml, "r2_patient_x_situation": base})
        sub = t.sample(min(len(t), 20000), random_state=0)
        pi = permutation_importance(model, sub[FEATURES], np.abs(sub.resid), scoring="r2", n_repeats=3, random_state=0)
        imps.append(pd.DataFrame({"dataset": name, "feature": FEATURES, "importance": pi.importances_mean}))

        methods = {"situation (Mondrian)": gb.situation_mondrian,
                   "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01),
                   "CQR, rich features": ml_cqr,
                   "Mondrian CQR x patient scale": mondrian_cqr_scaled}
        res, inside = gb.evaluate(v, t, methods=methods)
        for _, r in res.iterrows():
            ins = inside[r.method]
            band_rows.append({"dataset": name, **r.to_dict(),
                              "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                              "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
        print(name, "done", flush=True)
    R, I, B = pd.DataFrame(rows), pd.concat(imps, ignore_index=True), pd.DataFrame(band_rows)
    R.to_csv(OUT / "round4_ml.csv", index=False); I.to_csv(OUT / "round4_importance.csv", index=False)
    B.to_csv(OUT / "round4_bands.csv", index=False)
    h61 = int(sum(R.r2_ml >= 2 * R.r2_patient_x_situation))
    top = I.loc[I.groupby("dataset").importance.idxmax()].set_index("dataset").feature
    h62 = int(sum(top.isin(["last_err", "last_abs_err"])))
    gB = lambda d, m: B[(B.dataset == d) & (B.method == m)].iloc[0]  # noqa: E731
    h63 = sum((gB(d, "CQR, rich features").interval_score <= 0.97 * gB(d, "situation x patient scale (0.01)").interval_score)
              and gB(d, "CQR, rich features").excess_situation <= 0.015 and gB(d, "CQR, rich features").excess_patient <= 0.015
              for d in datasets)
    n = len(datasets)
    h64 = sum((gB(d, "Mondrian CQR x patient scale").interval_score <= 0.98 * gB(d, "situation x patient scale (0.01)").interval_score)
              and gB(d, "Mondrian CQR x patient scale").excess_situation <= 0.015 and gB(d, "Mondrian CQR x patient scale").excess_patient <= 0.015
              for d in datasets)
    lines = [f"H64 (Mondrian CQR x patient scale: >= 2% better interval score than H59 band, excess <= 1.5): {'SUPPORTED' if h64 >= 2 else 'NOT SUPPORTED'} ({h64} of {n})",f"H61 (ML explains >= 2x the patient x situation R^2): {'SUPPORTED' if h61 >= 2 else 'NOT SUPPORTED'} ({h61} of {n})",
             "  " + R.round(3).to_string(index=False).replace("\n", "\n  "),
             f"H62 (latest known error is the top feature): {'SUPPORTED' if h62 >= 2 else 'NOT SUPPORTED'} ({h62} of {n}); top: " + ", ".join(f"{d} {f}" for d, f in top.items()),
             "  " + I.pivot(index="feature", columns="dataset", values="importance").round(4).sort_values(datasets[0], ascending=False).to_string().replace("\n", "\n  "),
             f"H63 (CQR rich features >= 3% better interval score, excess <= 1.5): {'SUPPORTED' if h63 >= 2 else 'NOT SUPPORTED'} ({h63} of {n})"]
    for d in datasets:
        sub = B[B.dataset == d].set_index("method")[["coverage", "width", "interval_score", "excess_situation", "excess_patient"]].copy()
        sub[["coverage", "excess_situation", "excess_patient"]] *= 100
        lines.append(f"  {d}:\n    " + sub.round(2).to_string().replace("\n", "\n    "))
    text = "\n".join(lines)
    (OUT / "verdict_h61_h64.txt").write_text(text + "\n")
    print(text)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    x = np.arange(n)
    axes[0].bar(x - 0.2, R.r2_patient_x_situation * 100, 0.4, color="#54a24b", label="patient x situation group means")
    axes[0].bar(x + 0.2, R.r2_ml * 100, 0.4, color="#4c78a8", label="gradient boosting, rich context")
    axes[0].set_xticks(x, [LABEL.get(d, d) for d in datasets])
    axes[0].set_ylabel("% of variation in |test error| explained")
    axes[0].set_title("H61: how predictable is the size of each forecast's error?", loc="left")
    axes[0].legend(frameon=False)
    P = I.pivot(index="feature", columns="dataset", values="importance")[datasets]
    P = P.loc[P.mean(1).sort_values().index]
    yy = np.arange(len(P))
    for k, d in enumerate(datasets):
        axes[1].barh(yy + (k - 1) * 0.27, P[d] * 100, height=0.26, color=["#4c78a8", "#f58518", "#72b7b2"][k], label=LABEL.get(d, d))
    axes[1].set_yticks(yy, [NICE[f] for f in P.index], fontsize=8)
    axes[1].set_xlabel("drop in R^2 when the feature is shuffled (points)")
    axes[1].set_title("H62: what drives the size of the error?", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "h61_h62_ml_error_size.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
