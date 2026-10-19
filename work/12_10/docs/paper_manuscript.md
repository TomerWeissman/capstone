# Honest uncertainty for CGM glucose forecasts: prediction intervals must adapt to the glycemic situation and to the patient

Manuscript draft (prose), started Oct 18, 2026. Companion to `paper_draft.md` (the structured notes). Results sections are filled from `work/HYPOTHESES.md`; items marked [PENDING] wait on running analyses.

## Abstract

**Background.** Forecasts from continuous glucose monitors (CGM) increasingly come with prediction intervals, which are evaluated almost always by their average coverage. Clinical decisions are made in specific situations and for specific people. An interval that is right on average can still be too narrow when glucose is rising fast from the normal range, or for one patient and not another.

**Methods.** We trained 60-minute glucose forecasters for 215 people with type 1 diabetes from 11 public studies:
- three development datasets (62 patients)
- eight studies from the MetaboNet consolidation, held out for external validation (153 patients; closed-loop and sensor-augmented pump therapy; ages 2 to 74; Dexcom, Medtronic and Abbott sensors)

We compared GARCH, split, group-conditional (Mondrian), normalized and quantile-regression conformal intervals, adaptive conformal inference, and a new online per-patient scale correction. Calibration was assessed by glycemic situation (25 combinations of glucose level and 30-minute trend), by patient, and by day. We introduce a noise floor that separates real conditional miscalibration from the sampling noise of autocorrelated forecast windows. All 38 hypotheses were pre-registered with decision rules before analysis, and all are reported.

**Results.**
- **Situation.** Pooled over 11 studies, nominal 80% GARCH intervals covered between 62% and 91% of outcomes depending on the glycemic situation. Situation-conditional conformal intervals held every situation between 72% and 82%, and reduced per-situation miscalibration in all 8 external studies.
- **Patient.** Beyond situation, the real remaining miscalibration was between patients. An online per-patient scale, updated from each forecast's error once its outcome is observed, removed it in every development dataset and every external study, at equal or better interval score. It needed no calibration data from the new patient and held with a forecaster that had never seen the patient.
- **Day.** Adaptive intervals halved the day-to-day spread of coverage.
- **Error size.** The size of individual errors was largely unpredictable (at most 20% of its variation explained).
- **Patient difficulty.** Between patients, forecast difficulty was explained mainly by how fast glucose moves. This accounted for the higher difficulty of children and of closed-loop users.
- **Sensor error.** Under published sensor accuracy, sensor error at the forecast target accounts for about 30% of the variance of 60-minute forecast error.

**Conclusions.** Honest CGM prediction intervals need two adaptations: a fixed one to the glycemic situation and a slow online one to the patient. Claims of conditional calibration on fine subgroups require a noise floor.

## 1. Introduction

Short-term glucose forecasting from CGM data is now accurate enough to support alarms, decision support and automated insulin delivery. A forecast is only as useful as the trust that can be placed in it. Prediction intervals, which give a range expected to contain the future value with a stated probability, are the standard way to express that trust. Recent work has added intervals to glucose forecasters through quantile outputs, mixture densities, Monte Carlo dropout, evidential layers and conformal methods.

These intervals are usually evaluated by marginal coverage: across all forecasts, does an 80% interval contain the outcome 80% of the time? Clinical use is conditional. The cost of a missed outcome depends on whether glucose is low and falling or high and steady. Clinicians and patients rely on an interval for one person on one day, not for an average patient. An interval can be marginally calibrated while systematically too narrow in the situations that matter most.

We ask five questions:
1. Where are standard intervals miscalibrated: by glycemic situation, by patient, by day?
2. How much apparent conditional miscalibration is real, given that consecutive forecast windows overlap and subgroups are small?
3. What must an interval adapt to, and how quickly, to be honest at each of these levels?
4. How predictable is the size of an individual forecast error?
5. What makes some patients harder to forecast than others?

We answer them on 215 patients from 11 studies, with 8 studies held out entirely, using a pre-registered testing protocol.

## 2. Related work

[To be written from `paper_draft.md` Section 2 and `literature_log.md`. Key contrasts:
- Range-stratified interval coverage has been reported once, in ICU glucose prediction (PMC8324237), without trend and without a correction.
- Volatility-aware horizon control (MAKE 8:145) adapts the horizon, not the interval.
- Personalized uncertainty via meta-learning (Zhu et al.) is fixed before deployment, not updated online.
- Mondrian and adaptive conformal methods exist in general (Vovk; Gibbs and Candes); our contribution is their combination for CGM, the evidence across 11 studies, and the noise floor.]

## 3. Data

We used three development datasets and eight external studies.
- **AZT1D:** 25 adults on automated insulin delivery with Dexcom G6.
- **OhioT1DM:** 12 adults on sensor-augmented pumps with Medtronic Enlite, accessed through MetaboNet.
- **HUPA-UCM:** 25 adults on pump or injection therapy with FreeStyle Libre 2; records of about 13 days.
- **External validation:** eight studies from the public subset of MetaboNet, which consolidates public type 1 diabetes datasets: Loop, DCLP3, DCLP5, PEDAP, Flair, IOBP2, ReplaceBG and BrisT1D.
  - From each study we drew patients at random among those with at least 45 days of CGM data, kept the 45-day stretch with the most readings, and required at least 90% coverage.
  - Gaps of 30 minutes or less were interpolated; longer gaps were left as gaps.
  - This gave 20 patients per study (13 for BrisT1D, where fewer qualified), 153 in total.
  - The MetaboNet public subset contains only the closed-loop arms of the randomized trials DCLP3, IOBP2 and Flair.
  - Delivery modality is missing for 12 of the 13 BrisT1D patients; analyses involving modality are shown with BrisT1D excluded and with modality recoded from the recorded device.

[PENDING H84: expanded panel of up to 60 patients per study.]

## 4. Methods

### 4.1 Forecasters
- **Per-patient models.** For each patient we trained a convolutional-recurrent network (CNN-LSTM) on a 2-hour window of CGM, insulin and carbohydrate data to forecast glucose 60 minutes ahead.
  - Records were split chronologically: the first 64% of forecast windows for training, the next 16% for validation (early stopping and interval calibration), and the last 20% for testing.
  - Each patient's model had its own random seed.
- **Robustness checks:**
  - a CNN-Transformer
  - a 30-minute horizon
  - a second seed for every development model
  - a single pooled forecaster with larger capacity, trained only on the external studies and applied to the development patients, whom it had never seen

### 4.2 Glycemic situations
Each forecast was assigned to one of 25 situations: five levels of current glucose (below 70, 70-120, 120-180, 180-250, 250 mg/dL and above) by five 30-minute trends (fast fall below -30, slow fall -30 to -10, flat, slow rise 10 to 30, fast rise above 30 mg/dL).

### 4.3 Prediction intervals
All intervals were calibrated on validation forecasts and scored on test forecasts at 80% nominal coverage (90% as a robustness check):
- **GARCH(1,1)** on validation residuals, with a one-horizon information lag (an error is known only once its target time has passed).
- **Split conformal,** pooled over patients and per patient.
- **Situation-conditional (Mondrian) conformal:** a separate conformal quantile of absolute residuals in each situation with at least 100 validation windows, the pooled quantile otherwise.
- **GARCH-normalized Mondrian conformal.**
- **Conformalized quantile regression** with gradient-boosted quantiles on rich context, with and without situation-conditional correction.
- **Adaptive conformal inference,** adapting the quantile level.
- **Situation x online patient scale.** The situation quantile is multiplied by exp(theta). For each patient, theta starts at 0 and is updated after each forecast's outcome becomes known (one horizon later): theta increases by 0.01 x 0.8 after a miss and decreases by 0.01 x 0.2 after a hit, so it settles where 20% of outcomes fall outside the interval. The same rule with one theta per glucose level gives the per-level variant.
- **Cold start.** Situation quantiles were computed from other patients only.

### 4.4 Calibration metrics and the noise floor
- **Per-group calibration error:** the mean absolute difference between observed and nominal coverage over groups with at least 30 test windows.
- **Dispersion:** the same around the method's own overall coverage, which removes a shift in overall coverage.
- **Overall quality:** the interval (Winkler) score.
- **Day-to-day honesty:** the standard deviation of daily coverage.
- **The noise floor.** Because consecutive windows overlap and misses cluster, coverage estimated in small groups varies a lot by chance. We estimate the dispersion that sampling noise alone produces:
  - For situation-type groups, labels are circularly shifted within each patient by at least one day. This preserves the autocorrelation of both the labels and the hit-or-miss sequence, and breaks their relationship.
  - For patients, we use a moving-block bootstrap (one-day blocks) of each patient's coverage.
  - Reported "excess" is observed dispersion minus the floor.

### 4.5 Error-size predictability and patient difficulty
- **Error size.** We predicted absolute test errors from group means fitted on validation, and from gradient boosting on rich context:
  - level, trends, carbohydrates, insulin, time of day
  - the forecast itself, the GARCH scale, the patient's error scale
  - recently observed errors

  Drivers were assessed by permutation importance.
- **Patient difficulty.** We regressed log patient RMSE on log glucose SD and log SD of 30-minute glucose change (both from the training period), with leave-one-study-out R², and estimated child and closed-loop effects with a study-cluster bootstrap.

### 4.6 Pre-registration and statistics
Every hypothesis was registered with a metric, a threshold and the data to be used before the analysis was run, and committed to version control. The design of a test was changed before running only for runtime, and such changes are recorded. The patient is the unit of analysis: intervals come from patient-cluster bootstrap, or study-cluster bootstrap for cross-study effects. When several designs were tried for the same goal, all are reported with their order.

## 5. Results

[To be written from `paper_draft.md` Section 5 once H80 and H84 are in.]
