"""Round 6: external validation on the MetaboNet panel (H54-H56, H60) and what makes a patient hard
to forecast (H65-H67), using every patient with a forecaster.

Panel: 8 MetaboNet sources x 20 patients, 45 days each (build_metabonet_panel.py), never used to
design any method.

H54: situation bands >= 20% lower per-situation calibration error than GARCH, >= 6 of 8 sources.
H55: situation explains more of |test error| than patient, >= 5 of 8 sources.
H56: GARCH x situation lowest patient-by-situation dispersion of the 7 methods, >= 5 of 8 sources.
H60: situation x patient scale (eta 0.01) meets the H58 criteria, >= 5 of 8 sources.
H65: log RMSE ~ log glucose SD (training period), leave-one-source-out R^2 >= 0.60.
H66, H67: closed loop (AID) and children (< 18) effects on RMSE after SD; source-cluster bootstrap.

Run:    python work/12_10/scripts/round6_panel.py
Writes: work/12_10/results/round6_panel_bands.csv, round6_patients.csv, verdict_h54_h56_h60.txt,
        verdict_h65_h67.txt, figures/h54_panel_external.png, figures/h65_patient_difficulty.png
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

from azt1d import group_bands as gb, plotting, reference as ref, uq_frames  # noqa: E402
from round1_patient_vs_situation import error_predictors, r2, run_for  # noqa: E402

OUT, FIGS = WEEK / "results", WEEK / "figures"
PANEL = ["Loop", "DCLP3", "DCLP5", "PEDAP", "Flair", "IOBP2", "ReplaceBG", "BrisT1D"]
METHODS = {**gb.METHODS, "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01)}
TRAIN_SHARE = 0.64


def panel_bands():
    rows = []
    for src in PANEL:
        name = f"panel:{src}"
        v, t = uq_frames.frames(name, run_for(name))
        y, preds = error_predictors(v, t)
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            pc = (t.subject_id.astype(str) + "|" + t.cell.astype(str)).to_numpy()
            rows.append({"source": src, **r.to_dict(),
                         "floor_situation": gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "floor_patient": gb.dispersion_null_patient(ins, t.subject_id.to_numpy()),
                         "r2_situation": r2(y, preds["situation"]), "r2_patient": r2(y, preds["patient"]),
                         "r2_patient_x_situation": r2(y, preds["patient x situation"]),
                         "n_patients": t.subject_id.nunique()})
        print(src, "done", flush=True)
    B = pd.DataFrame(rows)
    B["excess_situation"] = B.disp_situation - B.floor_situation
    B["excess_patient"] = B.disp_patient - B.floor_patient
    return B


def patient_table():
    """One row per patient with a forecaster: test RMSE, training-period glucose stats, metadata."""
    meta = pd.read_csv(ROOT / "data" / "processed" / "metabonet_panel" / "metadata.csv")
    meta["subject_id"] = meta.subject_id.astype(str)
    sets = {"azt1d": ("AZT1D", "AID", 58), "ohiot1dm": ("OhioT1DM", "SAP", 40), "hupa_ucm": ("HUPA-UCM", "SAP/MDI", 34)}
    rows = []
    for name in [*sets, *[f"panel:{s}" for s in PANEL]]:
        df = uq_frames.load_dataset(name)
        df["subject_id"] = df.subject_id.astype(str)
        _, t = uq_frames.frames(name, run_for(name))
        rmse = t.groupby("subject_id").resid.apply(lambda r: float(np.sqrt(np.mean(r ** 2))))
        for sid, d in df.groupby("subject_id"):
            if sid not in rmse.index:
                continue
            g = d[ref.CGM].to_numpy(float)[: int(len(d) * TRAIN_SHARE)]
            row = {"dataset": name, "subject_id": sid, "rmse": rmse[sid], "glucose_sd": g.std(), "glucose_mean": g.mean(),
                   "glucose_cv": g.std() / g.mean(), "time_below_70": (g < 70).mean()}
            if name in sets:
                row.update(source=sets[name][0], modality=sets[name][1], age=sets[name][2])
            else:
                m = meta[meta.subject_id == sid].iloc[0]
                row.update(source=m.source, modality=m.modality, age=m.age)
            rows.append(row)
    P = pd.DataFrame(rows)
    P["aid"] = P.modality.astype(str).str.upper().eq("AID").astype(float)
    P["child"] = (P.age < 18).astype(float)
    return P


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def determinants(P):
    P = P.dropna(subset=["rmse", "glucose_sd", "age"]).reset_index(drop=True)
    y, x = np.log(P.rmse.to_numpy()), np.log(P.glucose_sd.to_numpy())
    pred = np.zeros(len(P))
    for s in P.source.unique():
        k = (P.source == s).to_numpy()
        b = ols(x[~k, None], y[~k])
        pred[k] = b[0] + b[1] * x[k]
    loso = r2(y, pred)
    b_sd = ols(x[:, None], y)
    X = np.column_stack([x, P.aid, P.child])
    b = ols(X, y)
    rng = np.random.default_rng(0)
    srcs = P.source.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(srcs, len(srcs), replace=True)
        idx = np.concatenate([np.flatnonzero(P.source == s) for s in pick])
        try:
            boots.append(ols(X[idx], y[idx]))
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots)
    ci = lambda j: np.nanpercentile(boots[:, j], [5, 95])  # noqa: E731
    return P, {"loso_r2": loso, "slope_sd": b_sd[1], "b": b, "ci_aid": ci(2), "ci_child": ci(3)}


def main():
    plotting.apply_style()
    B = panel_bands()
    B.to_csv(OUT / "round6_panel_bands.csv", index=False)
    g = lambda s, m: B[(B.source == s) & (B.method == m)].iloc[0]  # noqa: E731
    h54 = sum(g(s, "situation (Mondrian)").err_situation <= 0.8 * g(s, "GARCH").err_situation for s in PANEL)
    h55 = sum(g(s, "GARCH").r2_situation > g(s, "GARCH").r2_patient for s in PANEL)
    h56 = sum(B[B.source == s].set_index("method").disp_patient_x_situation.drop("situation x patient scale (0.01)").idxmin()
              == "GARCH x situation" for s in PANEL)
    sp = "situation x patient scale (0.01)"
    h60 = sum((g(s, sp).excess_patient <= 0.01) and (g(s, sp).excess_situation <= 0.015)
              and (g(s, sp).interval_score <= 1.03 * g(s, "situation (Mondrian)").interval_score) for s in PANEL)
    cols = ["coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient", "disp_patient_x_situation"]
    lines = [f"H54 (situation >= 20% lower per-situation error than GARCH): {'SUPPORTED' if h54 >= 6 else 'NOT SUPPORTED'} ({h54} of 8)",
             f"H55 (situation R^2 > patient R^2): {'SUPPORTED' if h55 >= 5 else 'NOT SUPPORTED'} ({h55} of 8)",
             f"H56 (GARCH x situation lowest patient-by-situation dispersion): {'SUPPORTED' if h56 >= 5 else 'NOT SUPPORTED'} ({h56} of 8)",
             f"H60 (patient-scale band meets H58 criteria): {'SUPPORTED' if h60 >= 5 else 'NOT SUPPORTED'} ({h60} of 8)"]
    for s in PANEL:
        sub = B[B.source == s].set_index("method")[cols].copy()
        sub[[c for c in cols if c not in ("width", "interval_score")]] *= 100
        r = g(s, "GARCH")
        lines.append(f"{s} ({int(r.n_patients)} patients; R^2 situation {r.r2_situation:.3f}, patient {r.r2_patient:.3f}, both {r.r2_patient_x_situation:.3f}):\n  "
                     + sub.round(2).to_string().replace("\n", "\n  "))
    text = "\n".join(lines)
    (OUT / "verdict_h54_h56_h60.txt").write_text(text + "\n")
    print(text, flush=True)

    P, d = determinants(patient_table())
    P.to_csv(OUT / "round6_patients.csv", index=False)
    aid_eff, child_eff = np.exp(d["b"][2]) - 1, np.exp(d["b"][3]) - 1
    ci_a, ci_c = np.exp(d["ci_aid"]) - 1, np.exp(d["ci_child"]) - 1
    h66 = abs(aid_eff) >= 0.05 and (ci_a[0] > 0 or ci_a[1] < 0)
    h67 = abs(child_eff) >= 0.05 and (ci_c[0] > 0 or ci_c[1] < 0)
    t2 = "\n".join([
        f"H65 (log RMSE ~ log glucose SD, leave-one-source-out R^2 >= 0.60): {'SUPPORTED' if d['loso_r2'] >= 0.6 else 'NOT SUPPORTED'}; R^2 {d['loso_r2']:.3f}; slope {d['slope_sd']:.2f}",
        f"H66 (AID effect after SD >= 5%, interval excludes 0): {'SUPPORTED' if h66 else 'NOT SUPPORTED'}; {aid_eff:+.1%} (90% source bootstrap {ci_a[0]:+.1%} to {ci_a[1]:+.1%})",
        f"H67 (child effect after SD >= 5%, interval excludes 0): {'SUPPORTED' if h67 else 'NOT SUPPORTED'}; {child_eff:+.1%} (90% source bootstrap {ci_c[0]:+.1%} to {ci_c[1]:+.1%})",
        f"patients {len(P)}, sources {P.source.nunique()}; AID {int(P.aid.sum())}, children {int(P.child.sum())}",
        "per source: " + P.groupby("source").agg(n=("rmse", "size"), rmse=("rmse", "median"), sd=("glucose_sd", "median")).round(1).to_string(),
    ])
    (OUT / "verdict_h65_h67.txt").write_text(t2 + "\n")
    print(t2)

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(PANEL))
    for k, (m, c) in enumerate([("GARCH", "#4c78a8"), ("situation (Mondrian)", "#f58518"), (sp, "#54a24b")]):
        d_ = B[B.method == m].set_index("source").loc[PANEL]
        axes[0].bar(x + (k - 1) * 0.27, d_.err_situation * 100, 0.26, color=c, label=m)
        axes[1].bar(x + (k - 1) * 0.27, d_.excess_patient * 100, 0.26, color=c, label=m)
        axes[2].bar(x + (k - 1) * 0.27, d_.interval_score, 0.26, color=c, label=m)
    for ax, t_, yl in zip(axes, ["H54: per-situation distance from 80%", "H60: per-patient unevenness above noise", "Interval score (lower is better)"],
                          ["points", "points above floor", "interval score"]):
        ax.set_xticks(x, PANEL, rotation=30, fontsize=8); ax.set_title(t_, loc="left", fontsize=10); ax.set_ylabel(yl)
    axes[2].set_ylim(B.interval_score.min() * 0.9, B.interval_score.max() * 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("External validation on 8 MetaboNet studies never used to design the methods (about 20 patients each)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGS / "h54_panel_external.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for s, dd in P.groupby("source"):
        ax.scatter(dd.glucose_sd, dd.rmse, s=22, label=s, alpha=0.8)
    xs = np.linspace(P.glucose_sd.min(), P.glucose_sd.max(), 50)
    bb = ols(np.log(P.glucose_sd.to_numpy())[:, None], np.log(P.rmse.to_numpy()))
    ax.plot(xs, np.exp(bb[0]) * xs ** bb[1], color="black", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("patient's glucose SD, training period (mg/dL)"); ax.set_ylabel("patient's 60-minute forecast RMSE (mg/dL)")
    ax.set_title(f"H65: how much of a patient's forecast difficulty is their glucose variability? (R^2 {d['loso_r2']:.2f}, new studies)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "h65_patient_difficulty.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
