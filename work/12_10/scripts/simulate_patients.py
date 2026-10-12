"""Simulate virtual T1D patients with simglucose (UVA/Padova model) as an answer key for the bands.

8 virtual adults, 30 days each, 5-minute steps, random meals (simglucose's RandomScenario), and
a basal-bolus controller made messier than the textbook one so the data is less clean:
  - meal boluses are scaled by a random factor (lognormal, SD 0.25 on the log scale)
  - 10% of main-meal boluses are missed, and those carbs are not logged
  - snacks under 20 g get no bolus and are not logged
Each patient has its own seed (meals, sensor, bolus errors).

Saves the noise-free blood glucose (BG) with the logged inputs. Sensor noise is added later, by the
analysis script, so the noise is known exactly.

Run:    python work/12_10/scripts/simulate_patients.py
Writes: data/processed/simulated/sim_true.parquet (regenerable, not in git)
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OUT = ROOT / "data" / "processed" / "simulated"

PATIENTS = [f"adult#{i:03d}" for i in range(1, 9)]
DAYS = 30
START = datetime(2026, 1, 1)
SNACK_G, MISS_P, BOLUS_LOG_SD = 20, 0.10, 0.25


def simulate(args):
    idx, name = args
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.base import Action
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario_gen import RandomScenario
    from simglucose.simulation.sim_engine import SimObj

    seed = 1000 + idx  # one seed per patient
    rng = np.random.default_rng(seed)
    log = {}

    class MessyBolus(BBController):
        def policy(self, observation, reward, done, **kw):
            a = super().policy(observation, reward, done, **kw)
            st = kw.get("sample_time", 1)
            grams = kw.get("meal", 0) * st
            bolus, logged = a.bolus, grams
            if grams > 0:
                if grams < SNACK_G or rng.random() < MISS_P:
                    bolus, logged = 0.0, 0.0
                else:
                    bolus *= float(np.exp(rng.normal(0, BOLUS_LOG_SD)))
            log[kw["time"]] = (a.basal * 60, bolus * st, logged)  # U/h, U per step, g
            return Action(basal=a.basal, bolus=bolus)

    env = T1DSimEnv(T1DPatient.withName(name), CGMSensor.withName("GuardianRT", seed=seed),
                    InsulinPump.withName("Insulet"), RandomScenario(start_time=START, seed=seed))
    sim = SimObj(env, MessyBolus(), timedelta(days=DAYS), animate=False, path=None)
    sim.simulate()
    r = sim.results().reset_index()
    inp = pd.DataFrame([(t, *v) for t, v in log.items()], columns=["Time", "basal", "bolus", "carbs"])
    d = r[["Time", "BG"]].merge(inp, on="Time", how="inner")
    d.insert(0, "subject_id", idx + 1)
    return d


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        parts = pool.map(simulate, list(enumerate(PATIENTS)))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT / "sim_true.parquet", index=False)
    print(df.groupby("subject_id").BG.describe().round(1).to_string())
    print(f"{len(df):,} rows; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
