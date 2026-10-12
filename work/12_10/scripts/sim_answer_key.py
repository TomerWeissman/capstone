"""H44/H45: test the situation bands against known truth on simulated patients.

Data: data/processed/simulated/sim_true.parquet (simulate_patients.py), noise-free blood glucose.
Two copies of every patient:
  clean: CGM = true glucose
  noisy: CGM = true glucose + planted noise, Gaussian, independent per reading, SD set by the
         true glucose level (NOISE_SD below), clipped to the sensor range 40-400
The plain CNN-LSTM (v0) is trained per patient on each copy (same code and settings as the real
datasets). GARCH and situation (Mondrian) bands are built the same way as in week 5_10.

H44: on noisy data, situation bands' per-situation coverage error is at least 20% lower than GARCH's.
H45: per level band, (mean squared 80% half-width, noisy minus clean) / 1.2816^2, divided by the
     mean planted noise variance at the forecast targets, is between 0.5 and 2 in at least 4 of 5
     level bands. GARCH is reported the same way, and so is the raw rise in squared error.

Also a realism table: simulated vs AZT1D vs HUPA-UCM.

Run:    python work/12_10/scripts/sim_answer_key.py
Writes: work/12_10/results/{sim_realism.csv, h45_noise_recovery.csv, verdict_h44_h45.txt},
        work/12_10/figures/{sim_realism.png, h44_sim_coverage.png, h45_noise_recovery.png}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import cleaning, hupa, loading, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
from azt1d.glimmer.train import run_with_checkpoints  # noqa: E402
import predictability_map as pm  # noqa: E402
import verification_checks as vc  # noqa: E402

CKPT = ROOT / "data" / "processed" / "checkpoints"
RUNS = {"clean": "sim_clean_cnn_lstm_v0", "noisy": "sim_noisy_cnn_lstm_v0"}
LEVELS = ["under 70", "70-120", "120-180", "180-250", "250+"]
NOISE_SD = np.array([15.0, 5.0, 5.0, 10.0, 20.0])  # planted SD per level band (pm.LEVEL_EDGES)
Z80 = 1.2815515655446004
OUT, FIGS = WEEK / "results", WEEK / "figures"


def planted_sd(bg):
    return NOISE_SD[np.digitize(bg, pm.LEVEL_EDGES)]


def canonical(sim, condition):
    """One canonical frame for all simulated patients, with planted noise for 'noisy'."""
    parts = []
    for sid, d in sim.groupby("subject_id"):
        cgm = d.BG.to_numpy()
        if condition == "noisy":
            rng = np.random.default_rng(2000 + sid)  # one noise seed per patient
            cgm = np.clip(cgm + rng.normal(0, planted_sd(cgm)), unc.SENSOR_MIN, unc.SENSOR_MAX)
        parts.append(pd.DataFrame({"subject_id": sid, ref.EVENT_DATETIME: d.Time.to_numpy(), ref.CGM: cgm,
                                   ref.BASAL: d.basal.to_numpy(), ref.TOTAL_BOLUS_INSULIN_DELIVERED: d.bolus.to_numpy(),
                                   ref.CARB_SIZE: d.carbs.to_numpy()}))
    df = pd.concat(parts, ignore_index=True)
    for col in (ref.DEVICE_MODE, ref.BOLUS_TYPE, ref.CORRECTION_DELIVERED, ref.FOOD_DELIVERED):
        df[col] = pd.NA
    return cleaning.clean_canonical_frame(df[["subject_id", *ref.CANONICAL_COLUMNS]])


def frames(df, run, sim):
    """Validation and test frames with GARCH sigma, situation cell, and the planted noise variance
    at each target (same construction as hybrid_bands.frames_with_sigma)."""
    vals, tests = [], []
    for sid in sorted(df.subject_id.unique()):
        dfs = df[df.subject_id == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(ckpt.load_result(CKPT / run, sid), dfs)
        eng = unc.BandEngine(val, test, analogs=False)
        p = eng.garch_params
        s_val = np.sqrt(unc.garch_h_step_variance(val.resid, p, float(np.var(val.resid - p.mu)), unc.LAG))
        times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
        g = dfs[ref.CGM].to_numpy()
        bg = sim[sim.subject_id == sid].set_index("Time").BG
        for fr, sig, bucket in ((val, s_val, vals), (test, eng.garch_sigma, tests)):
            i = np.searchsorted(times, fr.issue_time)
            d = pd.DataFrame({"subject_id": sid, "target_time": pd.to_datetime(fr.target_time),
                              "actual": fr.actual, "pred": fr.pred, "resid": fr.resid, "mu": p.mu, "sigma": sig,
                              "g_now": g[i], "g_prev": g[np.maximum(i - pm.TREND_STEPS, 0)]})
            d["bg_target"] = bg.reindex(d.target_time).to_numpy()
            bucket.append(d[(i >= pm.TREND_STEPS) & np.isfinite(sig)].reset_index(drop=True))
    v, t = pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)
    for f in (v, t):
        f["level"] = np.digitize(f.g_now.to_numpy(), pm.LEVEL_EDGES)
        f["cell"] = f.level * 5 + np.digitize((f.g_now - f.g_prev).to_numpy(), pm.RATE_EDGES)
        f["planted_var"] = planted_sd(f.bg_target.to_numpy()) ** 2
    return v, t


def realism(sim_dfs):
    """Summary statistics per dataset, plus each patient's plain CNN-LSTM test RMSE."""
    real = {"AZT1D": (loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed"), "cnn_lstm_v0"),
            "HUPA-UCM": (pd.concat([hupa.load_subject(s, ROOT / hupa.DEFAULT_DIR)
                                    for s in hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR)], ignore_index=True),
                         "hupa_ucm_cnn_lstm_v0")}
    sets = {**real, "sim clean": (sim_dfs["clean"], RUNS["clean"]), "sim noisy": (sim_dfs["noisy"], RUNS["noisy"])}
    rows, dists = [], {}
    for name, (df, run) in sets.items():
        g = df[ref.CGM].to_numpy(float)
        # 30-minute change, within patient
        dg = df.groupby("subject_id")[ref.CGM].transform(lambda s: s - s.shift(pm.TREND_STEPS)).dropna().to_numpy()
        lvl = np.bincount(np.digitize(g, pm.LEVEL_EDGES), minlength=5) / len(g)
        rmse = [float(ckpt.load_result(CKPT / run, int(s)).rmse) for s in df.subject_id.unique()
                if ckpt.load_result(CKPT / run, int(s)) is not None]
        rows.append({"dataset": name, "patients": df.subject_id.nunique(), "mean": g.mean(), "sd": g.std(),
                     "time_in_range_70_180": ((g >= 70) & (g <= 180)).mean(), "time_below_70": (g < 70).mean(),
                     "sd_30min_change": dg.std(), **{f"share_{n}": s for n, s in zip(LEVELS, lvl)},
                     "rmse_median": float(np.median(rmse)), "rmse_min": min(rmse), "rmse_max": max(rmse)})
        dists[name] = (g, dg, rmse)
    return pd.DataFrame(rows), dists


def recovery(fr):
    """Per level band: mean squared half-width (situation, GARCH) and mean squared error, test windows."""
    out = {}
    for cond, (v, t) in fr.items():
        lo, hi = vc.mondrian(v, t)
        q_sit = (hi - lo) / 2
        q_garch = Z80 * t.sigma.to_numpy()
        d = pd.DataFrame({"level": t.level, "sit": q_sit ** 2, "garch": q_garch ** 2,
                          "sq_err": t.resid.to_numpy() ** 2, "planted": t.planted_var})
        out[cond] = d.groupby("level").mean()
    c, n = out["clean"], out["noisy"]
    rec = pd.DataFrame({"level": LEVELS, "planted_sd_at_targets": np.sqrt(n.planted).to_numpy(),
                        "ratio_situation": ((n.sit - c.sit) / Z80 ** 2 / n.planted).to_numpy(),
                        "ratio_garch": ((n.garch - c.garch) / Z80 ** 2 / n.planted).to_numpy(),
                        "ratio_sq_error": ((n.sq_err - c.sq_err) / n.planted).to_numpy(),
                        "n_test_noisy": fr["noisy"][1].groupby("level").size().to_numpy()})
    return rec


def fig_realism(dists, table):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {"AZT1D": "#4c78a8", "HUPA-UCM": "#72b7b2", "sim clean": "#e45756", "sim noisy": "#f58518"}
    bins_g, bins_d = np.arange(40, 401, 10), np.arange(-120, 121, 5)
    for name, (g, dg, rmse) in dists.items():
        axes[0].hist(g, bins=bins_g, density=True, histtype="step", lw=2, color=colors[name], label=name)
        axes[1].hist(dg, bins=bins_d, density=True, histtype="step", lw=2, color=colors[name], label=name)
    for x, (name, (_, _, rmse)) in enumerate(dists.items()):
        axes[2].scatter(np.full(len(rmse), x) + np.random.default_rng(0).uniform(-0.15, 0.15, len(rmse)), rmse,
                        s=18, color=colors[name])
        axes[2].hlines(np.median(rmse), x - 0.3, x + 0.3, color="black", lw=2)
    axes[0].set_title("Does simulated glucose cover the same levels?", loc="left")
    axes[0].set_xlabel("glucose (mg/dL)")
    axes[1].set_title("Does it move as fast? (30-minute change)", loc="left")
    axes[1].set_xlabel("change over 30 minutes (mg/dL)")
    axes[1].set_yscale("log")
    axes[2].set_title("Is it as hard to forecast? (CNN-LSTM 60-min RMSE)", loc="left")
    axes[2].set_xticks(range(len(dists)), list(dists))
    axes[2].set_ylabel("test RMSE per patient (mg/dL); bar = median")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "sim_realism.png", dpi=150)
    plt.close(fig)


def fig_coverage(t, ins):
    d = pd.DataFrame({"cell": t.cell, **ins})
    g = d.groupby("cell").filter(lambda x: len(x) >= vc.MIN_TEST).groupby("cell").mean() * 100
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(g))
    ax.axhspan(75, 85, color="#eeeeee")
    ax.axhline(80, color="black", lw=1)
    ax.scatter(x, g["GARCH"], color="#4c78a8", s=40, label="GARCH")
    ax.scatter(x, g["situation"], color="#f58518", s=40, label="situation (Mondrian)")
    ax.set_xticks(x, [f"{pm.LEVEL_NAMES[c // 5]}, {pm.RATE_NAMES[c % 5]}" for c in g.index], rotation=70,
                  ha="right", fontsize=7)
    ax.set_ylabel("coverage of the 80% band (%)")
    ax.set_title("H44: on simulated data with planted noise, which band stays near 80% in each situation?", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "h44_sim_coverage.png", dpi=150)
    plt.close(fig)


def fig_recovery(rec):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(rec))
    ax.axhspan(0.5, 2, color="#eeeeee", label="rule: 0.5 to 2")
    ax.axhline(1, color="black", lw=1)
    for k, (col, c) in enumerate((("ratio_situation", "#f58518"), ("ratio_garch", "#4c78a8"),
                                  ("ratio_sq_error", "#999999"))):
        ax.bar(x + (k - 1) * 0.27, rec[col], width=0.25, color=c,
               label={"ratio_situation": "situation band", "ratio_garch": "GARCH band",
                      "ratio_sq_error": "actual squared error"}[col])
    ax.set_xticks(x, [f"{l}\n(planted SD {s:.0f})" for l, s in zip(rec.level, rec.planted_sd_at_targets)])
    ax.set_ylabel("rise from clean to noisy / planted noise variance")
    ax.set_title("H45: does the band widen by the noise we planted? (1 = exactly)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "h45_noise_recovery.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    plotting.apply_style()
    sim = pd.read_parquet(ROOT / "data" / "processed" / "simulated" / "sim_true.parquet")
    dfs = {c: canonical(sim, c) for c in RUNS}
    for c, run in RUNS.items():
        run_with_checkpoints(dfs[c], CKPT / run, epochs=30, region_weights=None)
    print(f"models ready ({time.time() - t0:.0f}s)", flush=True)

    fr = {c: frames(dfs[c], RUNS[c], sim) for c in RUNS}
    v, t = fr["noisy"]
    ins = {"GARCH": vc.inside(t, *vc.garch(t)), "situation": vc.inside(t, *vc.mondrian(v, t))}
    e_g, e_s = vc.cov_err(t, ins["GARCH"]), vc.cov_err(t, ins["situation"])
    cut = 1 - e_s / e_g
    per_pt = [(vc.cov_err(g, ins["situation"][g.index]) < vc.cov_err(g, ins["GARCH"][g.index]))
              for _, g in t.groupby("subject_id")]
    h44 = "SUPPORTED" if cut >= 0.20 else "NOT SUPPORTED"

    rec = recovery(fr)
    hits = int(rec.ratio_situation.between(0.5, 2).sum())
    h45 = "SUPPORTED" if hits >= 4 else "NOT SUPPORTED"
    rec.to_csv(OUT / "h45_noise_recovery.csv", index=False)

    table, dists = realism(dfs)
    table.to_csv(OUT / "sim_realism.csv", index=False)
    text = "\n".join([
        f"H44 (situation bands beat GARCH per situation on simulated noisy data): {h44}",
        f"  coverage error GARCH {e_g * 100:.1f}, situation {e_s * 100:.1f} points ({cut:+.0%}); "
        f"situation better for {sum(per_pt)} of {len(per_pt)} patients",
        f"  overall coverage GARCH {ins['GARCH'].mean():.1%}, situation {ins['situation'].mean():.1%}",
        "  rule: at least 20% lower coverage error",
        f"H45 (situation bands recover the planted noise): {h45} ({hits} of 5 level bands inside 0.5-2)",
        "  " + rec.round(2).to_string(index=False).replace("\n", "\n  "),
        "Realism:",
        "  " + table.round(3).to_string(index=False).replace("\n", "\n  "),
        f"runtime {time.time() - t0:.0f}s; one seed per patient; preliminary (8 simulated patients)",
    ])
    (OUT / "verdict_h44_h45.txt").write_text(text + "\n")
    print(text)
    fig_realism(dists, table)
    fig_coverage(t, ins)
    fig_recovery(rec)


if __name__ == "__main__":
    main()
