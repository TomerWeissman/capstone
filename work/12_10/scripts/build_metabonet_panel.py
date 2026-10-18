"""Build and train an external-validation panel from MetaboNet: 20 patients from each of 8 sources.

Sources span delivery modality (AID closed loop, pump with sensor), CGM devices, and ages from
young children to adults: Loop, DCLP3, DCLP5, PEDAP, Flair, IOBP2, ReplaceBG, BrisT1D.

Per patient (eligible: at least 45 days of CGM readings):
  - the 45-day stretch with the most CGM readings, on MetaboNet's 5-minute grid
  - gaps of 30 minutes or less filled by linear interpolation; longer gaps left out (rows dropped,
    as in the existing cleaning); patients whose best stretch has under 90% coverage are skipped
  - the same plain CNN-LSTM as the other datasets (30 epochs, early stopping), one seed per patient
Patient ids become integers (source index * 10000 + MetaboNet id order); the mapping and patient
metadata (age, sex, modality, CGM device) go to the panel's metadata file.

Run:    python work/12_10/scripts/build_metabonet_panel.py
Writes: data/processed/metabonet_panel/<source>.parquet, data/processed/metabonet_panel/metadata.csv,
        data/processed/checkpoints/panel_<source>_cnn_lstm_v0/
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

from azt1d import cleaning, metabonet, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402
from azt1d.glimmer.train import train_subject_model  # noqa: E402

SOURCES = ["Loop", "DCLP3", "DCLP5", "PEDAP", "Flair", "IOBP2", "ReplaceBG", "BrisT1D"]
PER_SOURCE, DAYS, MIN_COVER, MAX_GAP = 20, 45, 0.90, 6
STEPS = DAYS * 288
OUT = ROOT / "data" / "processed" / "metabonet_panel"
PARQUET = str(metabonet.DEFAULT_METABONET_PATH)


def eligible(source):
    q = f"""select id, count(CGM) n, any_value(age) age, any_value(gender) gender,
                   any_value(insulin_delivery_modality) modality, any_value(cgm_device) cgm_device
            from read_parquet('{PARQUET}') where source_file = '{source}' group by id having count(CGM) >= {STEPS}"""
    return duckdb.sql(q).df().sort_values("id").reset_index(drop=True)


def best_segment(d):
    """d: one patient's raw MetaboNet rows on the 5-minute grid. Returns the best 45-day stretch."""
    d = d.set_index(ref.EVENT_DATETIME).sort_index()
    d = d[~d.index.duplicated()]
    grid = pd.date_range(d.index[0].floor("5min"), d.index[-1].ceil("5min"), freq="5min")
    d = d.reindex(grid, method="nearest", tolerance=pd.Timedelta("2min"))
    have = d[ref.CGM].notna().to_numpy().astype(float)
    if len(have) < STEPS:
        return None, 0.0
    run = np.convolve(have, np.ones(STEPS), "valid")
    start = int(np.argmax(run))
    cover = run[start] / STEPS
    seg = d.iloc[start:start + STEPS].copy()
    seg[ref.CGM] = seg[ref.CGM].interpolate(limit=MAX_GAP, limit_area="inside")
    seg = seg.rename_axis(ref.EVENT_DATETIME).reset_index()
    return seg, cover


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for si, source in enumerate(SOURCES):
        el = eligible(source)
        rng = np.random.default_rng(100 + si)
        order = rng.permutation(len(el))  # random order; take the first PER_SOURCE that pass coverage
        raw = metabonet.load_source(source, subject_ids=[str(i) for i in el.id[order[:PER_SOURCE * 2]]], clean=False)
        raw["subject_id"] = raw.subject_id.astype(str)
        parts, kept = [], 0
        for k in order[:PER_SOURCE * 2]:
            if kept == PER_SOURCE:
                break
            row = el.iloc[k]
            seg, cover = best_segment(raw[raw.subject_id == str(row.id)].drop(columns="subject_id"))
            if seg is None or cover < MIN_COVER:
                continue
            sid = (si + 1) * 10000 + int(k)
            seg.insert(0, "subject_id", sid)
            parts.append(seg)
            kept += 1
            meta_rows.append({"subject_id": sid, "source": source, "metabonet_id": row.id, "age": row.age,
                              "gender": row.gender, "modality": row.modality, "cgm_device": row.cgm_device,
                              "cgm_coverage_45d": round(cover, 3)})
        df = cleaning.clean_canonical_frame(pd.concat(parts, ignore_index=True)[["subject_id", *ref.CANONICAL_COLUMNS]])
        df.to_parquet(OUT / f"{source}.parquet", index=False)
        print(f"{source}: {kept} patients from {len(el)} eligible ({time.time() - t0:.0f}s)", flush=True)

        run_dir = ROOT / "data" / "processed" / "checkpoints" / f"panel_{source}_cnn_lstm_v0"
        for sid in df.subject_id.unique():
            if ckpt.load_result(run_dir, sid) is not None:
                continue
            seed = zlib.crc32(f"{source}-{sid}".encode()) % 2**31  # one seed per patient
            res = train_subject_model(df[df.subject_id == sid].reset_index(drop=True), epochs=30, seed=seed)
            ckpt.save_result(run_dir, res)
            print(f"  {source} {sid}: RMSE {res.rmse:.1f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(meta_rows).to_csv(OUT / "metadata.csv", index=False)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
