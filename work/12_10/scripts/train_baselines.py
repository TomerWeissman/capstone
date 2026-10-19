"""H80 baselines on AZT1D, OhioT1DM, HUPA-UCM:
  1. deep ensembles: 5 per-patient CNN-LSTMs (plain loss, same settings as v0), one new seed each
     (checkpoint runs <prefix>cnn_lstm_ens1..5)
  2. a quantile CNN-LSTM per patient: same architecture with 3 outputs trained with the pinball loss
     at 10%, 50%, 90% (z-scored target), early stopping on validation pinball loss

Writes: checkpoints for the ensemble members; work/12_10/results/quantile_net_preds.parquet with the
quantile network's validation and test predictions (issue time, q10, q50, q90 in mg/dL).

Run:    python work/12_10/scripts/train_baselines.py
"""
from __future__ import annotations

import copy
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import reference as ref, uq_frames  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, sequences as seq  # noqa: E402
from azt1d.glimmer.model import CNNLSTM  # noqa: E402
from azt1d.glimmer.train import get_device, prepare_subject_data, train_subject_model  # noqa: E402

PREFIX = {"azt1d": "", "ohiot1dm": "ohiot1dm_", "hupa_ucm": "hupa_ucm_"}
QS = torch.tensor([0.1, 0.5, 0.9])
N_MEMBERS = 5


def train_quantile(dfs, seed, device, epochs=30, patience=5, batch=64):
    data = prepare_subject_data(dfs)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ym, ys = data.y_mean, data.y_std
    m = CNNLSTM(data.n_features, n_outputs=3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    q = QS.to(device)

    def pinball(pred, y):
        e = y[:, None] - pred
        return torch.mean(torch.maximum(q * e, (q - 1) * e))

    Xv = torch.from_numpy(data.X_val).to(device)
    yv = torch.from_numpy(((data.y_val - ym) / ys).astype(np.float32)).to(device)
    ytr = ((data.y_train - ym) / ys).astype(np.float32)
    best, best_state, stale = np.inf, None, 0
    for _ in range(epochs):
        m.train()
        perm = rng.permutation(len(ytr))
        for b in range(0, len(perm), batch):
            i = perm[b:b + batch]
            opt.zero_grad()
            loss = pinball(m(torch.from_numpy(data.X_train[i]).to(device)), torch.from_numpy(ytr[i]).to(device))
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = pinball(m(Xv), yv).item()
        if v < best:
            best, best_state, stale = v, copy.deepcopy(m.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pv = m(Xv).cpu().numpy() * ys + ym
        pt = m(torch.from_numpy(data.X_test).to(device)).cpu().numpy() * ys + ym
    times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
    n_tr, n_va = len(data.X_train), len(data.X_val)
    rows = []
    for split, p, first in (("val", pv, n_tr), ("test", pt, n_tr + n_va)):
        idx = np.arange(first, first + len(p))
        rows.append(pd.DataFrame({"split": split, "issue_time": times[idx + seq.LOOKBACK - 1],
                                  "q10": p[:, 0], "q50": p[:, 1], "q90": p[:, 2]}))
    return pd.concat(rows, ignore_index=True)


def main(names=None):
    t0 = time.time()
    device = get_device()
    qparts = []
    out = WEEK / "results" / "quantile_net_preds.parquet"
    if out.exists():  # keep predictions from earlier (partial) runs
        qparts.append(pd.read_parquet(out))
    for name, prefix in PREFIX.items():
        if names and name not in names:
            continue
        df = uq_frames.load_dataset(name)
        for sid in df.subject_id.unique():
            dfs = df[df.subject_id == sid].reset_index(drop=True)
            for k in range(1, N_MEMBERS + 1):
                run_dir = ROOT / "data" / "processed" / "checkpoints" / f"{prefix}cnn_lstm_ens{k}"
                if ckpt.load_result(run_dir, sid) is None:
                    seed = zlib.crc32(f"{name}-ens{k}-{sid}".encode()) % 2**31
                    ckpt.save_result(run_dir, train_subject_model(dfs, epochs=30, seed=seed))
            done = any(((q.dataset == name) & (q.subject_id == str(sid))).any() for q in qparts)
            if not done:
                qp = train_quantile(dfs, zlib.crc32(f"{name}-quant-{sid}".encode()) % 2**31, device)
                qp.insert(0, "subject_id", str(sid)); qp.insert(0, "dataset", name)
                qparts.append(qp)
                pd.concat(qparts, ignore_index=True).to_parquet(out, index=False)  # save as we go
            print(f"{name} {sid} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
