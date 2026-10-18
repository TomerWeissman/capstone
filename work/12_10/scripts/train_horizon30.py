"""Train plain per-patient CNN-LSTMs with a 30-minute horizon (6 steps) on AZT1D and OhioT1DM,
same settings as the 60-minute models, one seed per patient. For H71.

Run:    python work/12_10/scripts/train_horizon30.py
Writes: data/processed/checkpoints/{cnn_lstm_v0_h6, ohiot1dm_cnn_lstm_v0_h6}/
"""
import sys
import time
import zlib
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

from azt1d import uq_frames  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402
from azt1d.glimmer.train import train_subject_model  # noqa: E402

RUNS = {"azt1d": "cnn_lstm_v0_h6", "ohiot1dm": "ohiot1dm_cnn_lstm_v0_h6"}

t0 = time.time()
for name, run in RUNS.items():
    df = uq_frames.load_dataset(name)
    run_dir = ROOT / "data" / "processed" / "checkpoints" / run
    for sid in df.subject_id.unique():
        if ckpt.load_result(run_dir, sid) is not None:
            continue
        seed = zlib.crc32(f"{name}-h6-{sid}".encode()) % 2**31
        res = train_subject_model(df[df.subject_id == sid].reset_index(drop=True), horizon=6, epochs=30, seed=seed)
        ckpt.save_result(run_dir, res)
        print(f"{name} {sid}: RMSE {res.rmse:.1f} ({time.time() - t0:.0f}s)", flush=True)
print(f"runtime {time.time() - t0:.0f}s")
