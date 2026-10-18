"""H46-H48: what the teacher-forcing encoder keeps beyond glucose level and trend.

Retrains the HUPA-UCM 7-input encoder-decoder (same settings and seed as
work/5_10/scripts/tsne_hupa_all_inputs.py) and encodes 400 test windows per patient. Then:

1. Level-and-trend part: predict each of the 8 summary numbers from current glucose and the
   30-minute trend (gradient boosting, cross-fitted with folds by patient). Leftover = summary
   minus that prediction.                                                         -> H46
2. PCA of the leftover (how many directions it has).
3. What the leftover tracks: cross-fitted R^2 of the leftover from each recorded variable alone.
   Patient is scored as the between-patient share (in-sample).                    -> H47
4. Does it matter?
   a. decoder error with the full summary vs only its level-and-trend part vs zero
   b. predicting the plain CNN-LSTM's absolute 60-minute error on the same test windows,
      level and trend vs level, trend and leftover (folds by patient)             -> H48

Run:    python work/12_10/scripts/encoder_leftover.py
Writes: work/12_10/results/{verdict_h46_h48.txt, encoder_leftover_variables.csv},
        work/12_10/figures/{h46_leftover_share.png, h47_leftover_variables.png, h48_leftover_error.png}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work" / "5_10" / "scripts"))  # encoder code lives in week 5_10

from azt1d import hupa, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
from teacher_forcing_autoencoder import (TFEncoderDecoder, LOOKBACK, HORIZON, TRAIN_SHARE, TEST_SHARE,  # noqa: E402
                                         TRAIN_STRIDE, TEST_PER_PATIENT, SEED)
from tsne_hupa_all_inputs import ALL_INPUTS, CAP, raw_patient, windows  # noqa: E402

CKPT = ROOT / "data" / "processed" / "checkpoints" / "hupa_ucm_cnn_lstm_v0"
OUT, FIGS = WEEK / "results", WEEK / "figures"
TREND, IOB_STEPS = 6, 48  # 30-minute trend; insulin on board over 4 hours, linear decay
FOLDS = 5


def train_encoder(raws):
    """Same training as tsne_hupa_all_inputs.run, but returns the model and test windows."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, te = ([], [], []), []
    for sid, raw in raws:
        X, Y, P = windows(raw, ALL_INPUTS)
        n = len(X)
        cut_tr, cut_te = int(n * TRAIN_SHARE), int(n * (1 - TEST_SHARE))
        sel = np.arange(0, cut_tr, TRAIN_STRIDE)
        if len(sel) > CAP:
            sel = np.sort(rng.choice(sel, CAP, replace=False))
        for lst, arr in zip(tr, (X, Y, P)):
            lst.append(arr[sel])
        te_idx = np.arange(cut_te, n)
        pick = np.sort(rng.choice(te_idx, min(TEST_PER_PATIENT, len(te_idx)), replace=False))
        te.append((sid, pick, X[pick], Y[pick], P[pick]))
    Xtr, Ytr, Ptr = (np.concatenate(l) for l in tr)
    xm, xs = Xtr.reshape(-1, len(ALL_INPUTS)).mean(0), Xtr.reshape(-1, len(ALL_INPUTS)).std(0) + 1e-6
    gm, gs = float(xm[0]), float(xs[0])
    sx = lambda a: torch.from_numpy(((a - xm) / xs).astype(np.float32))  # noqa: E731
    sg = lambda a: torch.from_numpy(((a - gm) / gs).astype(np.float32))  # noqa: E731
    model = TFEncoderDecoder(len(ALL_INPUTS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, Yt, Pt = sx(Xtr), sg(Ytr), sg(Ptr)
    for _ in range(8):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            pred, _ = model(Xt[i], Pt[i], Yt[i])
            loss = ((pred - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    X = np.concatenate([t[2] for t in te]); Y = np.concatenate([t[3] for t in te]); P = np.concatenate([t[4] for t in te])
    with torch.no_grad():
        Z = model.encode(sx(X)).numpy()
    return model, (sx(X), sg(Y), sg(P)), gs, Z, [(t[0], t[1]) for t in te]


def decoder_rmse(model, tensors, z, gs):
    """Teacher-forced next-hour error (mg/dL) with a supplied summary z (same steps as forward)."""
    x, y, p = tensors
    z = torch.from_numpy(z.astype(np.float32))
    with torch.no_grad():
        dec_in = torch.cat([p[:, None], y[:, :-1]], dim=1)[..., None]
        zz = z[:, None, :].expand(-1, HORIZON, -1)
        o, _ = model.dec(torch.cat([dec_in, zz], dim=-1), torch.tanh(model.init_h(z))[None])
        pred = model.out(o).squeeze(-1)
    return float(torch.sqrt(((pred - y) ** 2).mean())) * gs


def window_table(raws, picks):
    """Per encoded window: the variables the leftover is checked against."""
    by_sid, rows = dict(raws), []
    for sid, starts in picks:
        r = by_sid[sid]
        g, bol = r.glucose.to_numpy(float), r.bolus_volume_delivered.to_numpy(float)
        decay = np.maximum(0, 1 - np.arange(IOB_STEPS) / IOB_STEPS)
        for s in starts:
            e = s + LOOKBACK - 1  # last input row
            w = slice(s, e + 1)
            past = bol[max(0, e - IOB_STEPS + 1):e + 1][::-1]
            rows.append({
                "subject_id": sid, "issue_time": r.time.iloc[e],
                "glucose": g[e], "trend_30min": g[e] - g[e - TREND],
                "trend_2h": g[e] - g[s], "acceleration": (g[e] - g[e - TREND]) - (g[e - TREND] - g[e - 2 * TREND]),
                "bumpiness": np.diff(g[w]).std(),
                "carbs_2h": r.carb_input.iloc[w].sum(), "bolus_2h": bol[w].sum(), "insulin_on_board": past @ decay[:len(past)],
                "basal": r.basal_rate.iloc[w].mean(), "heart_rate": r.heart_rate.iloc[w].mean(),
                "steps_2h": r.steps.iloc[w].sum(), "calories_2h": r.calories.iloc[w].sum(),
                "hour_sin": np.sin(2 * np.pi * r.hour.iloc[e] / 24), "hour_cos": np.cos(2 * np.pi * r.hour.iloc[e] / 24)})
    return pd.DataFrame(rows)


def gbm(loss="squared_error"):
    return HistGradientBoostingRegressor(loss=loss, max_iter=150, learning_rate=0.08, random_state=SEED)


def cross_fit(F, T, groups, loss="squared_error"):
    """Out-of-fold predictions of each column of T from F, folds by patient."""
    T = T.reshape(len(T), -1)
    out = np.zeros_like(T, dtype=float)
    for tr, te in GroupKFold(FOLDS).split(F, groups=groups):
        for j in range(T.shape[1]):
            out[te, j] = gbm(loss).fit(F[tr], T[tr, j]).predict(F[te])
    return out


def r2(T, pred):
    return 1 - ((T - pred) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()


VARIABLES = {  # label: columns
    "carbs (2 h)": ["carbs_2h"], "bolus (2 h)": ["bolus_2h"], "insulin on board (4 h)": ["insulin_on_board"],
    "basal": ["basal"], "heart rate": ["heart_rate"], "steps (2 h)": ["steps_2h"], "calories (2 h)": ["calories_2h"],
    "time of day": ["hour_sin", "hour_cos"], "2-hour trend": ["trend_2h"], "acceleration": ["acceleration"],
    "bumpiness": ["bumpiness"],
}


def forecast_errors(sids):
    """Absolute error of the plain CNN-LSTM's test forecasts, keyed by (patient, issue time)."""
    parts = []
    for sid in sids:
        dfs = hupa.load_subject(sid, ROOT / hupa.DEFAULT_DIR)
        _, test = unc.forecast_frames(ckpt.load_result(CKPT, sid), dfs)
        parts.append(pd.DataFrame({"subject_id": sid, "issue_time": pd.to_datetime(test.issue_time),
                                   "abs_err": np.abs(test.resid)}))
    return pd.concat(parts, ignore_index=True)


def main():
    t0 = time.time()
    plotting.apply_style()
    raws = [(sid, raw_patient(sid)) for sid in hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR)]
    model, tensors, gs, Z, picks = train_encoder(raws)
    W = window_table(raws, picks)
    groups = W.subject_id.to_numpy()
    print(f"encoder trained, {len(Z):,} test windows ({time.time() - t0:.0f}s)", flush=True)

    # 1. level-and-trend part and leftover (H46)
    LT = W[["glucose", "trend_30min"]].to_numpy()
    Zhat = cross_fit(LT, Z, groups)
    L = Z - Zhat
    share_lt = 1 - L.var(0).sum() / Z.var(0).sum()
    h46 = "SUPPORTED" if share_lt < 0.70 else "NOT SUPPORTED"
    pca = PCA().fit(L)

    # 3. what the leftover tracks (H47)
    rows = []
    for label, cols in VARIABLES.items():
        rows.append({"variable": label, "r2_of_leftover": r2(L, cross_fit(W[cols].to_numpy(), L, groups))})
    allcols = sorted({c for cols in VARIABLES.values() for c in cols})
    rows.append({"variable": "all of the above together", "r2_of_leftover": r2(L, cross_fit(W[allcols].to_numpy(), L, groups))})
    pm_ = pd.DataFrame(L).groupby(groups).transform("mean").to_numpy()
    rows.append({"variable": "patient (between-patient share, in-sample)", "r2_of_leftover": r2(L, pm_)})
    var = pd.DataFrame(rows)
    var.to_csv(OUT / "encoder_leftover_variables.csv", index=False)
    single = var[~var.variable.str.startswith("all")]
    best = single.loc[single.r2_of_leftover.idxmax()]
    h47 = "SUPPORTED" if best.r2_of_leftover >= 0.20 else "NOT SUPPORTED"
    print(f"variables scored ({time.time() - t0:.0f}s)", flush=True)

    # 4a. decoder with full summary vs level-and-trend part vs zero
    dec = {"full summary": decoder_rmse(model, tensors, Z, gs),
           "level-and-trend part only": decoder_rmse(model, tensors, Zhat, gs),
           "summary zeroed": decoder_rmse(model, tensors, np.zeros_like(Z), gs)}

    # 4b. does the leftover predict the CNN-LSTM's error size? (H48)
    err = forecast_errors(sorted(set(groups)))
    M = W.reset_index().merge(err, on=["subject_id", "issue_time"], how="inner")
    idx, g_m, y = M["index"].to_numpy(), M.subject_id.to_numpy(), M.abs_err.to_numpy()
    feats = {"level + trend": LT[idx],
             "level + trend + leftover": np.column_stack([LT[idx], L[idx]]),
             "level + trend + recorded variables": np.column_stack([LT[idx], W[allcols].to_numpy()[idx]])}
    mae = {k: float(np.abs(y - cross_fit(F, y, g_m, "absolute_error").ravel()).mean()) for k, F in feats.items()}
    mae["always the median"] = float(np.abs(y - np.median(y)).mean())
    gain = 1 - mae["level + trend + leftover"] / mae["level + trend"]
    h48 = "SUPPORTED" if gain >= 0.10 else "NOT SUPPORTED"

    text = "\n".join([
        f"H46 (level and trend explain < 70% of the summary): {h46}; they explain {share_lt:.0%}",
        f"  leftover PCA, share of leftover variance per direction: " + ", ".join(f"{v:.0%}" for v in pca.explained_variance_ratio_),
        f"H47 (one variable explains >= 20% of the leftover): {h47}; best: {best.variable} {best.r2_of_leftover:.0%}",
        "  " + var.assign(r2_of_leftover=var.r2_of_leftover.map(lambda v: f"{v:.1%}")).to_string(index=False).replace("\n", "\n  "),
        "Decoder teacher-forced next-hour error (mg/dL): " + ", ".join(f"{k} {v:.2f}" for k, v in dec.items()),
        f"H48 (leftover improves prediction of CNN-LSTM error size by >= 10%): {h48}; gain {gain:+.1%}",
        f"  matched test windows {len(M):,} of {len(W):,}; mean absolute error of predicting |error| (mg/dL): "
        + ", ".join(f"{k} {v:.2f}" for k, v in mae.items()),
        f"runtime {time.time() - t0:.0f}s; one seed; folds by patient (5); preliminary",
    ])
    (OUT / "verdict_h46_h48.txt").write_text(text + "\n")
    print(text)

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(["level and trend", "leftover"], [share_lt * 100, (1 - share_lt) * 100], color=["#4c78a8", "#f58518"])
    axes[0].axhline(70, color="black", lw=1, ls="--")
    axes[0].text(0.5, 71, "rule: level and trend under 70%", ha="center", fontsize=8)
    axes[0].set_ylabel("% of the summary's variance")
    axes[0].set_title("H46: how much of the encoder summary is level and trend?", loc="left")
    axes[1].bar(np.arange(1, 9), pca.explained_variance_ratio_ * 100, color="#f58518")
    axes[1].set_xlabel("leftover direction (PCA)")
    axes[1].set_ylabel("% of the leftover")
    axes[1].set_title("Does the leftover have a few main directions, or many?", loc="left")
    fig.tight_layout(); fig.savefig(FIGS / "h46_leftover_share.png", dpi=150); plt.close(fig)

    v = var.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(v.variable, v.r2_of_leftover * 100,
            color=["#999999" if s.startswith(("all", "patient")) else "#f58518" for s in v.variable])
    ax.axvline(20, color="black", lw=1, ls="--", label="rule: 20%")
    ax.set_xlabel("% of the leftover explained (cross-fitted, new patients)")
    ax.set_title("H47: what does the leftover track?", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "h47_leftover_variables.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(list(dec), list(dec.values()), color=["#4c78a8", "#f58518", "#999999"])
    axes[0].set_ylabel("teacher-forced next-hour error (mg/dL)")
    axes[0].set_title("Does the decoder use the leftover?", loc="left")
    axes[0].tick_params(axis="x", labelsize=8)
    k = ["always the median", "level + trend", "level + trend + leftover", "level + trend + recorded variables"]
    axes[1].bar(range(4), [mae[x] for x in k], color=["#999999", "#4c78a8", "#f58518", "#72b7b2"])
    axes[1].set_xticks(range(4), [x.replace(" + ", "\n+ ") for x in k], fontsize=8)
    axes[1].set_ylim(min(mae.values()) * 0.9, max(mae.values()) * 1.03)
    axes[1].set_ylabel("error predicting |CNN-LSTM error| (mg/dL)")
    axes[1].set_title(f"H48: does the leftover predict forecast error size? ({gain:+.1%})", loc="left")
    fig.tight_layout(); fig.savefig(FIGS / "h48_leftover_error.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
