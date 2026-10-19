"""H84: replication on the expanded MetaboNet panel (up to 60 patients per study).

Run:    python work/12_10/scripts/round20_panel60.py   (after build_metabonet_panel.py 60)
Writes: work/12_10/results/round20_panel60_bands.csv, round20_patients.csv, verdict_h84.txt
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import group_bands as gb, reference as ref, uq_frames  # noqa: E402
from round1_patient_vs_situation import r2, run_for  # noqa: E402
from round6_panel import PANEL, TRAIN_SHARE, ols  # noqa: E402

METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian,
           "situation x patient scale": partial(gb.scale_aci_situation, eta=0.01),
           "situation x patient-level scale": partial(gb.level_scale_aci_situation, eta=0.01)}


def bands():
    rows = []
    for src in PANEL:
        name = f"panel60:{src}"
        v, t = uq_frames.frames(name, run_for(name))
        res, inside = gb.evaluate(v, t, methods=METHODS)
        for _, r in res.iterrows():
            ins = inside[r.method]
            rows.append({"source": src, "n_patients": t.subject_id.nunique(), **r.to_dict(),
                         "excess_situation": r.disp_situation - gb.dispersion_null_shift(ins, t.cell.to_numpy(), t.subject_id.to_numpy(), n_rep=30)[0],
                         "excess_patient": r.disp_patient - gb.dispersion_null_patient(ins, t.subject_id.to_numpy())})
        print(src, "done", flush=True)
    return pd.DataFrame(rows)


def patients():
    dev = pd.read_csv(WEEK / "results" / "round15_patients.csv", dtype={"subject_id": str})
    dev = dev[dev.dataset.isin(["azt1d", "ohiot1dm", "hupa_ucm"])]
    meta = pd.read_csv(ROOT / "data" / "processed" / "metabonet_panel60" / "metadata.csv", dtype={"subject_id": str})
    rows = []
    for src in PANEL:
        name = f"panel60:{src}"
        df = uq_frames.load_dataset(name)
        df["subject_id"] = df.subject_id.astype(str)
        _, t = uq_frames.frames(name, run_for(name))
        rmse = t.groupby("subject_id").resid.apply(lambda r: float(np.sqrt(np.mean(r ** 2))))
        for sid, d in df.groupby("subject_id"):
            if sid not in rmse.index:
                continue
            g = d[ref.CGM].to_numpy(float)[: int(len(d) * TRAIN_SHARE)]
            m = meta[meta.subject_id == sid].iloc[0]
            rows.append({"dataset": name, "subject_id": sid, "source": src, "rmse": rmse[sid], "glucose_sd": g.std(),
                         "rate_sd": float(np.nanstd(g[6:] - g[:-6])), "age": m.age})
    P = pd.concat([dev[["dataset", "subject_id", "source", "rmse", "glucose_sd", "rate_sd", "age"]], pd.DataFrame(rows)], ignore_index=True)
    P = P.dropna(subset=["rmse", "glucose_sd", "rate_sd", "age"]).reset_index(drop=True)
    P["child"] = (P.age < 18).astype(float)
    return P


def main():
    B = bands()
    B.to_csv(WEEK / "results" / "round20_panel60_bands.csv", index=False)
    g = lambda s, m: B[(B.source == s) & (B.method == m)].iloc[0]  # noqa: E731
    a = sum(g(s, "situation (Mondrian)").err_situation <= 0.8 * g(s, "GARCH").err_situation for s in PANEL)
    lv = "situation x patient-level scale"
    b = sum((g(s, lv).excess_patient <= 0.01) and (g(s, lv).excess_situation <= 0.015)
            and (g(s, lv).interval_score <= 1.03 * g(s, "situation (Mondrian)").interval_score) for s in PANEL)

    P = patients()
    P.to_csv(WEEK / "results" / "round20_patients.csv", index=False)
    y, src = np.log(P.rmse.to_numpy()), P.source.to_numpy()
    X = np.column_stack([np.log(P.glucose_sd), np.log(P.rate_sd)])
    pred = np.zeros(len(y))
    for s in np.unique(src):
        k = src == s
        bb = ols(X[~k], y[~k]); pred[k] = bb[0] + X[k] @ bb[1:]
    c_r2 = r2(y, pred)
    Xc = np.column_stack([X, P.child])
    bc = ols(Xc, y)
    rng = np.random.default_rng(0)
    srcs = P.source.unique()
    boots = np.array([ols(Xc[idx], y[idx]) for idx in
                      (np.concatenate([np.flatnonzero(P.source == s) for s in rng.choice(srcs, len(srcs))]) for _ in range(2000))])
    child = np.exp(bc[3]) - 1
    ci = np.exp(np.percentile(boots[:, 3], [5, 95])) - 1
    d = ci[0] <= 0 <= ci[1]
    ok = a >= 7 and b >= 5 and c_r2 >= 0.55 and d
    cols = ["n_patients", "coverage", "width", "interval_score", "err_situation", "excess_situation", "excess_patient"]
    lines = [f"H84 (replication at scale): {'SUPPORTED' if ok else 'NOT SUPPORTED'}",
             f"  (a) situation beats GARCH by >= 20%: {a} of 8",
             f"  (b) per-level patient scale meets H58 criteria: {b} of 8",
             f"  (c) leave-one-source-out R^2 (glucose SD + rate SD): {c_r2:.3f}",
             f"  (d) child effect after rate SD: {child:+.1%} (90% {ci[0]:+.1%} to {ci[1]:+.1%})",
             f"  patients in (c, d): {len(P)} ({int(P.child.sum())} children), sources {P.source.nunique()}"]
    for s in PANEL:
        sub = B[B.source == s].set_index("method")[cols].copy()
        sub[["coverage", "err_situation", "excess_situation", "excess_patient"]] *= 100
        lines.append(f"{s}:\n  " + sub.round(2).to_string().replace("\n", "\n  "))
    text = "\n".join(lines)
    (WEEK / "results" / "verdict_h84.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
