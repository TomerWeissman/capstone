# Findings ledger

The spine of the paper. One entry per finding, with its evidence tier (see `RESEARCH_PROTOCOL.md`). Updated whenever new evidence changes a tier. Negative findings stay.

Datasets: AZT1D (25 patients), OhioT1DM (12, via MetaboNet), HUPA-UCM (25), simulated (simglucose, 8). Forecaster: plain per-patient CNN-LSTM, 60-minute horizon, unless noted.

## Seeded Oct 17, 2026 from weeks 21_9 to 12_10

### F1. Group-conditional (Mondrian) bands fix per-situation miscalibration. Tier 2.
- Standard bands (GARCH, plain conformal) are right on average and wrong in specific situations: coverage of an 80% band ranges from 64% (70 to 120, fast rise) to 95% (under 70, fast fall) on AZT1D.
- Sizing the band separately for each of 25 situations (level by 30-minute trend) cuts the mean distance from 80% from 6.7 to 3.9 points on OhioT1DM, data not used to design the method, with narrower bands (87 vs 99 mg/dL). 9 of 12 patients closer.
- Holds against shuffled groupings (H33), across models (H34), and under patient bootstrap (H35; gain interval 0.3 to 3.6 points). Beats tuned CQR on AZT1D clearly; on OhioT1DM 19% better but the interval crosses zero (H36).
- Hypotheses: H27, H28, H32 to H36.
- Not yet: external validation on MetaboNet sources (tier 3).

### F2. Per-situation honesty is not overall efficiency. Tier 2.
- At equal width, all band methods catch nearly the same share of readings (H37). CQR is best on interval score, situation bands close second.
- Claim per-situation calibration, never overall efficiency.

### F3. Calibration by situation trades off dangerous-miss coverage. Tier 2.
- Situation bands catch fewer Clarke zone D points than GARCH (41% vs 62% OhioT1DM; 22% vs 39% AZT1D). A GARCH-normalized version recovers most of the gap (54%, 34%) but not all (H38).

### F4. Bands do not make better alarms than a tuned threshold. Tier 1.
- At matched false-trigger rates, conformal, GARCH and analog ensemble bands catch 74 to 76.5% of danger (H10). A tuned point threshold beats band alarms (H24).

### F5. More data of the same kind barely helps the forecaster. Tier 1.
- Learning curve asymptote is 98% of today's error (H30, AZT1D). The model is not data-limited.

### F6. Forecast error tracks each patient's glucose variability, not time. Tier 1.
- No drift on average; per-patient test/validation error ratios follow changes in variability (correlation 0.77, H29).

### F7. A predictive encoder compresses a 2-hour window mostly into level and trend. Tier 2.
- Teacher-forcing encoder summary: PC1 vs glucose 0.99 (AZT1D H39, HUPA-UCM H40); level and trend explain 87% (H46). The leftover tracks patient setup (basal 21%, H47) and does not help predict error size (H48).
- Learned groupings from the encoder tie the hand-drawn situation grid (H43).

### F8. Sensor noise costs more than its own variance. Tier 1 (simulation, post hoc).
- Planting level-dependent noise raised squared forecast error by 1.1 to 3.5 times the noise variance, because noise also corrupts the inputs (H45, post hoc). Simulated patients were easier to forecast than real ones.

### F9. For a new patient, error size is set more by who the patient is than by the moment. Tier 0 (post hoc, H48).
- On HUPA-UCM with folds by patient, level and trend predicted the size of the CNN-LSTM's error no better than a constant.
- Next: a direct variance decomposition (patient vs situation vs time) on several datasets.
