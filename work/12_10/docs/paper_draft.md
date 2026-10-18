# Where glucose forecast uncertainty lives: situation, patient, and noise in 60-minute CGM prediction intervals

Working draft, started Oct 17, 2026. Built from `work/FINDINGS.md`; every number traces to a result file named in `work/HYPOTHESES.md`. Places marked **[PENDING Hxx]** wait on the external MetaboNet panel.

## Abstract (draft)

Prediction intervals for continuous glucose monitoring (CGM) forecasts are usually judged by their average coverage. We ask where the uncertainty of a 60-minute forecast actually lives, and what an interval must adapt to in order to be honest everywhere it is used. Using per-patient CNN-LSTM forecasters on three public datasets (AZT1D, OhioT1DM, HUPA-UCM; 62 patients) and an external panel of 160 patients from eight MetaboNet studies, we find:
- **Situation.** Standard intervals (GARCH, split conformal) are right on average but miscalibrated in specific glycemic situations, defined by level and 30-minute trend. Group-conditional (Mondrian) conformal calibration by situation removes this, down to the sampling-noise limit **[PENDING H54]**.
- **A noise floor for calibration metrics.** We introduce a circular-shift noise floor for calibration metrics on autocorrelated forecast windows. It shows that most apparent patient-by-situation miscalibration is sampling noise, while real miscalibration remains between patients.
- **Patient.** An online per-patient scale correction, updated from errors observed 60 minutes later, removes that residual and sharpens the intervals. It holds for two forecaster architectures **[PENDING H60]**.
- **Unpredictable size.** The size of individual forecast errors is mostly unpredictable: at most 20% of its variation is explained, even by gradient boosting with rich context. Errors barely cluster at this horizon, which undercuts the volatility-clustering premise behind GARCH-style bands.
- **Patient difficulty.** **[PENDING H65-H67]**

## 1. Introduction

- CGM forecasting is mature for point accuracy, and uncertainty quantification is growing: quantile heads, evidential layers, MC dropout, and conformal methods.
- The common evaluation is marginal: does an 80% interval cover 80% of readings overall? Clinical use is conditional. A patient falling fast from 110 mg/dL needs an interval that is honest for that situation, and for that patient.
- Questions:
  1. How miscalibrated are standard intervals across glycemic situations and patients?
  2. How much of the apparent miscalibration is real, given that forecast windows are autocorrelated and the cells are small?
  3. What must an interval adapt to (situation, patient, recent volatility) to be honest at each level?
  4. How predictable is the size of the next error at all?
  5. What makes some patients harder to forecast?

## 2. Related work (to expand from work/12_10/docs/literature_log.md and work/5_10/docs/literature_review_uncertainty.md)

- Uncertainty for glucose forecasting: Gluformer; TFT with quantile outputs; evidential vs MC dropout on HUPA-UCM (Tan and McBeth 2026). Mostly marginal calibration.
- Conformal prediction: split conformal; Mondrian/group-conditional conformal (Vovk 2003; Vovk 2012 conditional validity); normalized conformal; CQR (Romano et al. 2019); adaptive conformal inference under shift (Gibbs and Candes 2021).
- Glucose-specific conformal work found so far: imputation (Matabuena et al.), conformal risk control with deferral (KTU). No study found applying group-conditional or online per-patient conformal calibration to CGM forecasting (searches logged).
- Large-scale data: MetaboNet (Khamesian et al. 2026) and MetaboNet-Bench (accuracy only).

## 3. Data

| Dataset | Patients | Therapy | Notes |
|---|---|---|---|
| AZT1D | 25 | AID | Development |
| OhioT1DM (via MetaboNet) | 12 | Pump with sensor (SAP) | Used for H28 as fresh data |
| HUPA-UCM | 25 | SAP/MDI | Most heterogeneous |
| MetaboNet panel | 160 (8 x 20) | AID and SAP; children to adults | Loop, DCLP3, DCLP5, PEDAP, Flair, IOBP2, ReplaceBG, BrisT1D. Never used to design any method. 45 days per patient, best-coverage stretch, gaps of 30 min or less interpolated. |
| Simulated (simglucose) | 8 | Basal-bolus with errors | Answer key for noise; too easy to forecast (Section 5.6) |

## 4. Methods

- **Forecaster.** Per-patient CNN-LSTM (GLIMMER baseline architecture), 2-hour lookback, 60-minute horizon. Chronological split: 64% train, 16% validation, 20% test. One seed per patient. CNN-Transformer used as a second architecture.
- **Situations.** 5 levels (<70, 70-120, 120-180, 180-250, >=250 mg/dL) x 5 trends (30-minute change with edges -30, -10, 10, 30 mg/dL).
- **Interval methods** (all calibrated on validation, scored on test; 80% nominal):
  - pooled split conformal
  - per-patient conformal
  - GARCH(1,1) on validation residuals with a 60-minute information lag
  - situation Mondrian conformal
  - GARCH-normalized Mondrian
  - patient-scaled Mondrian
  - CQR with rich features
  - Mondrian CQR
  - ACI (quantile-level adaptation)
  - situation x online per-patient scale: half-width = situation quantile x exp(theta), with theta updated by eta x (miss indicator 60 minutes earlier - 0.2), eta = 0.01
- **Calibration metrics.**
  - Distance from nominal coverage averaged over groups (situations, patients, patient-by-situation cells with at least 30 test windows).
  - Dispersion around each method's own overall coverage.
  - Interval score.
- **Noise floor (new).** The dispersion that sampling noise alone produces. Situation labels are circularly shifted within each patient by at least one day, preserving the autocorrelation of both series while breaking their link. Per patient, the floor comes from a moving-block bootstrap (one-day blocks) of each patient's coverage. "Excess" = observed minus floor.
- **Error-size predictability.**
  - Group means fitted on validation (R^2 of absolute test error).
  - Gradient boosting with rich context (level, trends, carbs, bolus, time, GARCH scale, patient scale, recently observed errors).
  - Permutation importance.
- **Statistics.** Patient as the unit, patient-cluster bootstrap intervals, source-cluster bootstrap for cross-study effects. Every hypothesis pre-registered with a threshold before running; the full register (H1-H69 so far, including nulls) is in the supplement.

## 5. Results

### 5.1 Standard intervals are miscalibrated by situation; situation calibration fixes it (F1)
- AZT1D: coverage of 80% conformal bands ranges from 64% (70-120, fast rise) to 95% (<70, fast fall).
- OhioT1DM (fresh data): mean distance from 80% across situations, GARCH 6.7 vs situation Mondrian 3.9 points; narrower bands (87 vs 99 mg/dL); 9 of 12 patients improved. Robust to shuffled groupings, to two other forecasters, and to the patient bootstrap.
- **[PENDING H54: 8 external studies]**

### 5.2 A noise floor shows what is real (F11)
- Situation Mondrian bands sit at the floor by situation (2.7 vs 2.9 AZT1D; 2.1 vs 4.0 HUPA-UCM; OhioT1DM 1.6 points above).
- Patient-by-situation unevenness is 80-95% sampling noise: floors of 6.5-15.3 points, real excess 0.8-1.8.
- Real miscalibration remains between patients: 3.5, 2.3, 9.1 points above the floor.
- GARCH-based bands look more even mainly because their noise floor is lower; the cause is not shorter miss streaks (H57). Mechanism open.
- Same pattern with the CNN-Transformer (H69).

### 5.3 Online per-patient scaling removes the between-patient residual (F12)
- Per-patient excess 3.5, 2.3, 9.1 to -0.9, -0.9, +0.4 points. Interval score improves: 112.0 vs 113.5; 130.2 vs 130.8; 107.5 vs 110.8.
- Cost: per-situation excess rises to 0.9, 2.5, 1.2. Quantile-level ACI fixes patients but costs more situation evenness (H58).
- Recovers part of the dangerous-miss gap: zone D caught 29%, 44%, 52% vs 22%, 41%, 46% for situation bands; GARCH 39%, 62%, 69% (H68).
- CNN-Transformer: same result (H69).
- **[PENDING H60: 8 external studies]**

### 5.4 Error size is mostly unpredictable (F9, F13)
- Group means: R^2 at most 15%. Gradient boosting with rich context: 9.8%, 3.7%, 19.9%.
- Top drivers: the forecast value and the patient's error scale. The latest observed error adds little, and GARCH's time-varying scale predicts error size no better than a constant.
- Situation vs patient: neither dominates (situation wins on AZT1D and OhioT1DM, patient on HUPA-UCM). **[PENDING H55]**

### 5.5 Sharpness and conditional calibration trade off (F14)
- Rich-feature CQR is sharpest but uneven by situation (excess 2.4-4.0) and by patient (2.3-6.2).
- Mondrian CQR with patient scaling: best interval score (108.0, 128.1, 99.6) and calibrated by patient, but about 2 points uneven by situation.

### 5.6 What makes a patient hard to forecast **[PENDING H65-H67]**

### 5.7 Sensor noise costs more than its own variance (F8; simulation, preliminary)
- Planted level-dependent noise raised squared 60-minute error by 1.1-3.5 times the noise variance, because noisy inputs mislead the forecaster too.
- The simulator was too easy (median RMSE 11-16 vs 31-39 mg/dL real), so this is a lead, not a result.

## 6. Discussion (notes)

- Honest intervals need two adaptations: a fixed one by glycemic situation and a slow one by patient. Recent-error volatility, the core of GARCH, matters little at this horizon.
- Calibration claims on small conditional cells need a noise floor; otherwise method rankings at fine granularity are noise.
- Clinical: per-situation honesty does not equal catching dangerous misses; the patient-scale band narrows but does not close the zone D gap.
- Limits:
  - per-patient forecasters only
  - one seed per patient
  - 60-minute horizon only
  - MetaboNet panel limited to 45-day stretches and 20 patients per study
  - modality labels as given by MetaboNet
  - simulated noise is independent from reading to reading

## 7. Reproducibility

Code: `src/azt1d/uq_frames.py`, `src/azt1d/group_bands.py`, `work/12_10/scripts/round*.py`. Register: `work/HYPOTHESES.md`. Protocol: `work/RESEARCH_PROTOCOL.md`.
