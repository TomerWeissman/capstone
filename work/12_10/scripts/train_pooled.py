"""H79: one pooled forecaster trained on the 8 MetaboNet panel studies, applied to AZT1D,
OhioT1DM and HUPA-UCM (patients it never saw).

Same CNN-LSTM design as the per-patient models, larger (conv 32-64-32, LSTM 64), one global
feature scaling and target z-score fitted on the panel training windows. Each panel patient's
first 64% of windows train it (every 2nd window), the next 16% are its early-stopping set.
For each development patient, the 64/16/20 split is the same as everywhere else; the pooled
model forecasts its validation and test windows, which become frames for the band analyses.

Run:    python work/12_10/scripts/train_pooled.py
Writes: data/processed/checkpoints/pooled_panel_cnn_lstm/model.pt,
        data/processed/frames/<dataset>__pooled_panel.parquet (read by uq_frames.frames(name, "pooled_panel")),
        work/12_10/results/pooled_training_log.txt (stdout)
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

from azt1d import reference as ref, uq_frames  # noqa: E402
from azt1d.glimmer import features as feat, sequences as seq, uncertainty as unc  # noqa: E402
from azt1d.glimmer.model import CNNLSTM  # noqa: E402
from azt1d.glimmer.train import get_device  # noqa: E402

PANEL = ["Loop", "DCLP3", "DCLP5", "PEDAP", "Flair", "IOBP2", "ReplaceBG", "BrisT1D"]
TARGETS = ["azt1d", "ohiot1dm", "hupa_ucm"]
CKPT = ROOT / "data" / "processed" / "checkpoints" / "pooled_panel_cnn_lstm"
SEED, BATCH, EPOCHS, PATIENCE, LR = 7, 512, 25, 4, 1e-3


def windows(dfs):
    enriched = feat.add_engineered_features(dfs)
    X, y = seq.make_windows(enriched, feat.FEATURE_COLUMNS, feat.TARGET_COLUMN, seq.LOOKBACK, seq.HORIZON)
    n = len(X)
    n_test = int(round(n * 0.2)); n_tv = n - n_test
    n_val = int(round(n_tv * 0.2)); n_train = n_tv - n_val
    return X, y, n_train, n_val


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = get_device()
    Xtr, ytr, Xva, yva = [], [], [], []
    for src in PANEL:
        df = uq_frames.load_dataset(f"panel:{src}")
        for sid, dfs in df.groupby("subject_id"):
            X, y, n_tr, n_va = windows(dfs.reset_index(drop=True))
            Xtr.append(X[:n_tr:2]); ytr.append(y[:n_tr:2])
            Xva.append(X[n_tr:n_tr + n_va:4]); yva.append(y[n_tr:n_tr + n_va:4])
    Xtr, ytr, Xva, yva = (np.concatenate(a) for a in (Xtr, ytr, Xva, yva))
    nf = Xtr.shape[-1]
    fm, fs = Xtr.reshape(-1, nf).mean(0), Xtr.reshape(-1, nf).std(0) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std())
    scale = lambda X: ((X - fm) / fs).astype(np.float32)  # noqa: E731
    Xtr, Xva = scale(Xtr), scale(Xva)
    ytr_z, yva_z = ((ytr - ym) / ys).astype(np.float32), ((yva - ym) / ys).astype(np.float32)
    print(f"training windows {len(Xtr):,}, validation {len(Xva):,} ({time.time() - t0:.0f}s)", flush=True)

    model = CNNLSTM(nf, conv_channels=(32, 64, 32), lstm_hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xva_t, yva_t = torch.from_numpy(Xva).to(device), torch.from_numpy(yva_z).to(device)
    best, best_state, stale = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), BATCH):
            i = perm[b:b + BATCH]
            opt.zero_grad()
            loss = torch.mean((model(torch.from_numpy(Xtr[i]).to(device)) - torch.from_numpy(ytr_z[i]).to(device)) ** 2)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(torch.sqrt(torch.mean((torch.cat([model(Xva_t[k:k + 8192]) for k in range(0, len(Xva_t), 8192)]) - yva_t) ** 2))) * ys
        print(f"epoch {ep + 1}: validation RMSE {v:.2f} mg/dL ({time.time() - t0:.0f}s)", flush=True)
        if v < best:
            best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in best_state.items()}, "fm": fm, "fs": fs, "ym": ym, "ys": ys,
                "conv": (32, 64, 32), "lstm": 64, "val_rmse": best}, CKPT / "model.pt")

    def predict(X):
        with torch.no_grad():
            out = [model(torch.from_numpy(scale(X[k:k + 8192])).to(device)).cpu().numpy() for k in range(0, len(X), 8192)]
        return np.concatenate(out) * ys + ym

    for name in TARGETS:
        df = uq_frames.load_dataset(name)
        parts = []
        for sid in df.subject_id.unique():
            dfs = df[df.subject_id == sid].reset_index(drop=True)
            X, y, n_tr, n_va = windows(dfs)
            times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
            fr = []
            for a, b in ((n_tr, n_tr + n_va), (n_tr + n_va, len(X))):
                idx = np.arange(a, b)
                p = predict(X[a:b])
                fr.append(unc.Frame(actual=y[a:b].astype(float), pred=p.astype(float), resid=y[a:b].astype(float) - p,
                                    X=None, issue_time=times[idx + seq.LOOKBACK - 1],
                                    target_time=times[idx + seq.LOOKBACK + seq.HORIZON - 1]))
            v, t = uq_frames.subject_frames(dfs, None, forecasts=tuple(fr))
            parts += [v, t]
        f = pd.concat(parts, ignore_index=True)
        f["subject_id"] = f.subject_id.astype(str)
        out = uq_frames.cache_path(name, "pooled_panel")
        out.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(out, index=False)
        t_ = f[f.split == "test"]
        print(f"{name}: pooled test RMSE {np.sqrt(np.mean(t_.resid ** 2)):.1f} mg/dL; median per patient "
              f"{t_.groupby('subject_id').resid.apply(lambda r: np.sqrt(np.mean(r ** 2))).median():.1f}", flush=True)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
