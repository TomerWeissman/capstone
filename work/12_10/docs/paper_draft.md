# Where glucose forecast uncertainty lives: situation, patient, and day in 60-minute CGM prediction intervals across 11 studies

Working draft, updated Oct 17, 2026. Built from `work/FINDINGS.md`; every number traces to a result file named in `work/HYPOTHESES.md` (H49-H76 for this paper; H1-H48 are earlier project history). Items marked **[PENDING]** are running.

## Abstract (draft)

**Background.** Prediction intervals for continuous glucose monitoring (CGM) forecasts are usually evaluated by average coverage. Clinical use is conditional: an interval must be honest for this glycemic situation, this patient, and this day.

**Methods.**
- Data: per-patient CNN-LSTM forecasters (60-minute horizon) for 215 people with type 1 diabetes from 11 public studies. Three datasets were used for development (AZT1D, OhioT1DM, HUPA-UCM; 62 patients). Eight MetaboNet studies were held out for external validation (153 patients; closed-loop and sensor-augmented pump; children to adults; three CGM families: Dexcom, Medtronic, Abbott).
- Interval methods: GARCH, split, Mondrian, normalized and quantile-regression conformal, adaptive conformal inference, and an online per-patient scale correction.
- Evaluation: a new circular-shift noise floor that separates real conditional miscalibration from the sampling noise of autocorrelated forecast windows.

**Results.**
- **Situation.** Standard intervals were miscalibrated across 25 glycemic situations (level by 30-minute trend). Pooled over all 11 studies, coverage of nominal 80% GARCH intervals ranged from 62% (70-120 mg/dL, rising fast) to 91% (250 mg/dL and over, rising fast). Situation-conditional conformal calibration reduced per-situation miscalibration in all 8 external studies (2.3-4.2 vs 5.5-6.8 points), down to the noise floor.
- **Patient.** Real miscalibration remained between patients. An online per-patient scale, updated from errors observed one horizon later, removed it in every development dataset and every external study. It also sharpened intervals, and it needed no calibration data from the new patient.
- **Day.** Adaptive intervals halved the day-to-day spread of coverage.
- **Error size.** It was largely unpredictable: at most 20% of its variation was explained, even by gradient boosting with rich context. Recent errors carried little information, contrary to the volatility-clustering premise of GARCH-type intervals.
- **Patient difficulty.** What made a patient hard to forecast was how fast their glucose moved. The spread of 30-minute changes explained 59-63% of between-patient forecast difficulty in new studies, against 44% for overall glucose variability. It fully accounted for children being 7.5% harder to forecast. Closed-loop therapy made no detectable difference.

**Conclusion.** Honest CGM prediction intervals need a fixed adaptation to glycemic situation and a slow online adaptation to the patient. Calibration claims on fine subgroups need a noise floor.

## 1. Introduction

- CGM forecasting is mature for point accuracy, and uncertainty quantification is growing: quantile heads, evidential layers, MC dropout, conformal methods. Evaluations report marginal calibration.
- Decisions are conditional. A 110 mg/dL reading falling fast, in a child, on a day of unusual control, is where an interval must be honest.
- Questions:
  1. Where are standard intervals miscalibrated: by situation, by patient, by day?
  2. How much apparent conditional miscalibration is sampling noise?
  3. What must an interval adapt to, and how fast?
  4. How predictable is the size of the next error?
  5. What makes a patient hard to forecast?

## 2. Related work

- **Glucose forecast uncertainty.**
  - Gluformer (arXiv:2209.04526).
  - Temporal Fusion Transformer with quantiles (PMC12349322).
  - Evidential vs MC dropout on HUPA-UCM (Tan and McBeth 2026, arXiv:2603.04955).
  - Calibration is reported marginally; accuracy is sometimes stratified by glycemic range (GlyRAG, arXiv:2601.05353), but interval coverage is not.
- **Conformal prediction.**
  - Mondrian / group-conditional conformal (Vovk 2003; Vovk 2012).
  - Conditional guarantees over subgroup classes (Gibbs, Cherian, Candes, arXiv:2305.12616).
  - Normalized conformal; CQR (Romano et al. 2019).
  - Time series: Xu and Xie 2021; online multi-step, arXiv:2410.13115.
  - Adaptive conformal inference under shift (Gibbs and Candes 2021).
- **Interval evaluation in econometrics.** Christoffersen-style tests of unconditional coverage and of the independence of misses. We found no method for a sampling-noise floor of group-level coverage dispersion under autocorrelation (searches logged in `work/12_10/docs/literature_log.md`).
- **CGM conformal work.** Imputation (Matabuena et al., arXiv:2403.18069) and conformal risk control with deferral (KTU). No CGM forecasting study found with group-conditional or online per-patient conformal calibration.
- **Large-scale data.** MetaboNet (arXiv:2601.11505); MetaboNet-Bench (arXiv:2606.18640, accuracy only).

## 3. Data

| Role | Dataset | Patients | Therapy | Notes |
|---|---|---|---|---|
| Development | AZT1D | 25 | AID | Dexcom G6 |
| Development | OhioT1DM (via MetaboNet) | 12 | SAP | Medtronic Enlite |
| Development | HUPA-UCM | 25 | SAP/MDI | FreeStyle Libre 2; short records (about 13 days) |
| External | Loop, DCLP3, DCLP5, PEDAP, Flair, IOBP2, ReplaceBG, BrisT1D (MetaboNet public subset) | 153 (20 per study; BrisT1D 13) | AID and SAP; 69 children overall | Best 45-day stretch per patient, at least 90% CGM coverage, gaps of 30 min or less interpolated; patients selected at random among the eligible |
| Answer key | simglucose (UVA/Padova) | 8 | Basal-bolus with dosing errors | Section 5.9 |

## 4. Methods

- **Forecasters.**
  - Per-patient CNN-LSTM, 2-hour lookback, 60-minute horizon, 30 epochs with early stopping.
  - Chronological split: 64% train, 16% validation, 20% test.
  - One seed per patient.
  - Robustness checks: a CNN-Transformer, a 30-minute horizon, and a second seed for every development forecaster.
- **Situations.** 5 levels (<70, 70-120, 120-180, 180-250, >=250 mg/dL) x 5 trends (30-minute change; edges -30, -10, 10, 30 mg/dL).
- **Intervals.** All are calibrated on validation and scored on test.
  - GARCH(1,1) with a one-horizon information lag
  - pooled and per-patient split conformal
  - situation-Mondrian conformal
  - GARCH-normalized Mondrian
  - patient-scaled Mondrian
  - rich-feature CQR and Mondrian CQR
  - adaptive conformal inference (ACI)
  - situation x online patient scale: half-width = q_situation x exp(theta_t), with theta_{t+1} = theta_t + eta (miss_{t-h} - alpha), eta = 0.01, and h equal to the horizon
  - situation x online patient-per-level scale: one theta per glucose level
- **Calibration metrics.**
  - Mean distance from nominal over groups with at least 30 test windows.
  - Dispersion around each method's own overall coverage.
  - Interval (Winkler) score.
  - SD of daily coverage.
- **Noise floor.**
  - Situation-type groups: labels circularly shifted within each patient by at least one day, 30-50 times. This keeps the autocorrelation of both series and breaks their link.
  - Per patient: moving-block bootstrap (one-day blocks) of each patient's coverage; expected dispersion = mean of sqrt(2/pi) x SE.
  - Excess = observed minus floor.
- **Error-size predictability.**
  - Group means fitted on validation (R^2 on absolute test error).
  - Gradient boosting with level, trends, carbs, bolus, time of day, forecast value, GARCH scale, patient scale, and recently observed errors.
  - Permutation importance.
- **Patient difficulty.** Log RMSE on log glucose SD (training period) with leave-one-source-out R^2; AID and child effects with a source-cluster bootstrap.
- **Pre-registration.** Every hypothesis had a threshold committed before running (git history). All 28 hypotheses of this paper are reported, including nulls (supplement table).

## 5. Results

### 5.1 Standard intervals are miscalibrated by situation; situation calibration fixes it (F1, tier 3)
- Pooled over all 11 studies, GARCH coverage of a nominal 80% band ranged from 62% (70-120, fast rise) to 91% (>=250, fast rise). Situation calibration held every situation between 72% and 82%, and 23 of 24 between 78% and 82% (Figure 1). (Development datasets only: GARCH 65-95%, situation 76-87%.)
- External: per-situation distance from 80% fell in all 8 held-out studies, from 5.5-6.8 to 2.3-4.2 points (H54).
- Also holds at 90% nominal (H76), at a 30-minute horizon (H71), and for a CNN-Transformer (H69).

### 5.2 A noise floor separates real from apparent miscalibration (F11)
- Situation bands sit at the floor by situation (2.7 vs 2.9 AZT1D; 2.1 vs 4.0 HUPA-UCM). OhioT1DM keeps a small real residual (1.6 above; 1.9 with a second seed, H75).
- Patient-by-situation unevenness is 80-95% sampling noise (floors 6.5-15.3 points; real excess 0.8-1.8). Method rankings at that granularity, without a floor, are misleading (H51-H52).
- Real miscalibration remains between patients: 3.5, 2.3, 9.1 points above the floor (H53), and 1.4-4.8 in the external studies.

### 5.3 An online per-patient scale removes the between-patient residual (F12, tier 3)
- Development: per-patient excess 3.5, 2.3, 9.1 fell to -0.9, -0.9, +0.4 points; interval score improved (H59).
- External: removed in all 8 studies (-0.55 to -0.84), interval score better or within 0.5% (H60).
- Trade-off: a single scale over-covers rare extreme situations (Figure 1) and costs about one point of situation evenness. One scale per glucose level keeps both and meets all criteria in 6 of 8 external studies (H74), but needs longer records (it fails on HUPA-UCM's 2.6-day test periods, H73).
- Standard ACI (adapting the quantile level) also fixes patients but costs more situation evenness (H58).
- Robust to a second architecture (H69), a 30-minute horizon (H71), 90% nominal (H76), and a second training seed (H75b).

### 5.4 The new patient needs no calibration data of their own (F15)
- With situation quantiles from other patients only, the online scale gives per-patient excess -0.87, -0.89, 0.49 vs -0.88, -0.91, 0.43 for own-data calibration (H70). The forecaster was still per patient.

### 5.4b Summary over all 11 studies (Figure 2; medians, points)
| Method | Situation excess above floor | Patient excess above floor | SD of daily coverage |
|---|---|---|---|
| GARCH | 3.8 | 1.7 | 6.0 |
| Situation (Mondrian) | 0.7 | 3.2 | 9.3 |
| Situation x patient scale | 1.8 | -0.7 | 5.5 |

Source: `work/12_10/results/paper_three_axes.csv`.

### 5.5 Honesty by day (F16)
- Static situation bands cover some days far less than others (SD of daily coverage 10.3, 9.3, 12.4 points). Adaptive bands halve this (patient scale 5.5, 5.9, 6.7; H72).
- This is why adaptive bands have lower noise floors. It is not explained by shorter miss streaks (H57).

### 5.6 Error size is mostly unpredictable, and situation matters more than patient (F9, F13, tier 3)
- Development: situation and patient group means explain at most 15%; gradient boosting with rich context explains 9.8%, 3.7%, 19.9% (H49, H61).
- Top drivers: the forecast value and the patient's error scale. The latest known error adds little, and GARCH's scale predicts error size no better than a constant (H62).
- External: situation explains more than patient in 7 of 8 studies; everything at most 8.8% (H55).

### 5.7 Sharpness and conditional calibration trade off (F14)
- Rich-feature CQR is sharpest but uneven by situation (excess 2.4-4.0) and by patient (2.3-6.2) (H63).
- Mondrian CQR with the patient scale has the best interval score (108.0, 128.1, 99.6) and is patient-calibrated, but is about 2 points uneven by situation (H64).

### 5.8 What makes a patient hard to forecast (F17, tier 3; 215 patients, 11 studies)
- Glucose variability (SD) explains 44% of between-patient difficulty in new studies (H65; log-log slope 0.64).
- After glucose SD, children are 7.5% harder (90% CI +3.3% to +12.4%, H67). Closed loop: +2.4% (-4.3% to +8.9%), not detectable (H66).
- How fast glucose moves explains much more: the SD of 30-minute changes gives leave-one-source-out R^2 0.59 alone and 0.63 with glucose SD. In the joint model the glucose-SD slope falls to 0.22 and the rate slope is 0.63 (H78).
- With rate variability in the model, the child effect vanishes (+0.4%, -4.5% to +5.0%; H77). Children's glucose moves faster (median 30.7 vs 25.4 mg/dL per 30 min).

### 5.9 Safety: dangerous misses (F3)
- Situation bands catch fewer Clarke zone D points than GARCH. The patient-scale band recovers part of the gap: 29%, 44%, 52% vs 22%, 41%, 46% (GARCH 39%, 62%, 69%; H68).

### 5.10 Sensor noise costs more than its own variance (F8; simulation, preliminary)
- Planted noise raised squared error by 1.1-3.5 times its own variance, because noisy inputs also mislead the forecaster. The simulator was easier than real data, so this is a lead.

## 6. Discussion (notes)

- Two adaptations are needed and they work at different speeds: a fixed one by glycemic situation, and a slow online one by patient. Recent-error volatility, the core of GARCH, matters little at 30-60 minutes.
- Fine-grained calibration claims need a noise floor.
- A deployable recipe:
  1. Calibrate situation quantiles on a population.
  2. Apply them to a new patient.
  3. Adapt one scale per patient (short records) or per patient and level (records of several weeks) from errors observed one horizon later.
- Clinical caveat: per-situation honesty does not equal catching dangerous misses.
- Limits:
  - per-patient forecasters (the cold start covers only the band)
  - one seed per patient in the main analysis (a second seed for the 37 development patients changes patient RMSE by a median 0.8-1.5 mg/dL and leaves the conclusions intact)
  - 45-day external records and 20 patients per study
  - modality labels as given by MetaboNet
  - simulator too easy for a noise ceiling

## 7. Figures

1. Coverage in each of 25 situations, 11 studies pooled: GARCH, situation, situation x patient scale (`figures/paper_fig1_situation_coverage.png`).
2. Three axes of honesty, by situation, by patient, by day, all 11 studies (`figures/paper_fig2_three_axes.png`).
3. Noise floor vs observed unevenness (`figures/h51_noise_floor.png`).
4. External validation (`figures/h54_panel_external.png`).
5. Error-size predictability and drivers (`figures/h61_h62_ml_error_size.png`).
6. Patient difficulty vs variability (`figures/h65_patient_difficulty.png`); to add: difficulty vs rate-of-change SD.

## 8. Reproducibility

Code: `src/azt1d/uq_frames.py`, `src/azt1d/group_bands.py`, `work/12_10/scripts/`. Register: `work/HYPOTHESES.md`. Protocol: `work/RESEARCH_PROTOCOL.md`. Data: public sources only (`datasets/README.md`).
