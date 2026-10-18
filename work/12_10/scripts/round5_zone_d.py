"""H68: does the per-patient scale-adaptive situation band catch as many dangerous misses (Clarke
zone D points of the point forecast) as the situation band? GARCH bands for reference.

Run:    python work/12_10/scripts/round5_zone_d.py
Writes: work/12_10/results/round5_zone_d.csv, verdict_h68.txt
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

from azt1d import group_bands as gb, uq_frames  # noqa: E402
from azt1d.glimmer import uncertainty as unc  # noqa: E402
from round1_patient_vs_situation import run_for  # noqa: E402

METHODS = {"GARCH": gb.garch, "situation (Mondrian)": gb.situation_mondrian, "GARCH x situation": gb.garch_situation,
           "situation x patient scale (0.01)": partial(gb.scale_aci_situation, eta=0.01)}


def main(datasets):
    rows = []
    for name in datasets:
        v, t = uq_frames.frames(name, run_for(name))
        zones = np.asarray(unc.clarke_error_grid_zones(t.actual.to_numpy(), t.pred.to_numpy()))
        _, inside = gb.evaluate(v, t, methods=METHODS)
        for m, ins in inside.items():
            for z in ["A", "B", "C", "D", "E"]:
                k = zones == z
                if k.sum():
                    rows.append({"dataset": name, "method": m, "zone": z, "n": int(k.sum()), "caught": float(ins[k].mean())})
    R = pd.DataFrame(rows)
    R.to_csv(WEEK / "results" / "round5_zone_d.csv", index=False)
    D = R[R.zone == "D"].pivot(index="dataset", columns="method", values="caught")
    ok = int(sum(D["situation x patient scale (0.01)"] >= D["situation (Mondrian)"] - 0.02))
    text = (f"H68 (patient-scale band catches zone D no worse than situation bands by > 2 points): "
            f"{'SUPPORTED' if ok >= 2 else 'NOT SUPPORTED'} ({ok} of {len(datasets)})\n"
            f"zone D share caught (%):\n{(D * 100).round(1).to_string()}\n"
            f"zone D windows: {R[(R.zone == 'D')].groupby('dataset').n.first().to_dict()}")
    (WEEK / "results" / "verdict_h68.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["azt1d", "ohiot1dm", "hupa_ucm"])
