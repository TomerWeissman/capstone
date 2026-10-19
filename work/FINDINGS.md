# Findings ledger

The spine of the paper. One entry per finding, with its evidence tier (see `RESEARCH_PROTOCOL.md`). Updated whenever new evidence changes a tier. Negative findings stay.

Datasets: AZT1D (25 patients), OhioT1DM (12, via MetaboNet), HUPA-UCM (25), simulated (simglucose, 8). Forecaster: plain per-patient CNN-LSTM, 60-minute horizon, unless noted.

## Seeded Oct 17, 2026 from weeks 21_9 to 12_10

### F1. Group-conditional (Mondrian) bands fix per-situation miscalibration. Tier 3.
- Standard bands (GARCH, plain conformal) are right on average and wrong in specific situations: coverage of an 80% band ranges from 64% (70 to 120, fast rise) to 95% (under 70, fast fall) on AZT1D.
- Sizing the band separately for each of 25 situations (level by 30-minute trend) cuts the mean distance from 80% from 6.7 to 3.9 points on OhioT1DM, data not used to design the method, with narrower bands (87 vs 99 mg/dL). 9 of 12 patients closer.
- Holds against shuffled groupings (H33), across models (H34), and under patient bootstrap (H35; gain interval 0.3 to 3.6 points). Beats tuned CQR on AZT1D clearly; on OhioT1DM 19% better but the interval crosses zero (H36).
- Hypotheses: H27, H28, H32 to H36.
- Grid choice matters (H83): a 3x3 grid barely beats GARCH; 7x7 is best on OhioT1DM but weaker on AZT1D and HUPA-UCM; 5x5 is the best balance on 2 of 3 (scored on 5x5 situations, which favors it). The claim is that a moderately fine level-by-trend grid is needed, not that 5x5 is uniquely right.
- External validation (H54): in all 8 MetaboNet studies never used for design (153 patients; AID and SAP; children to adults; Dexcom, Medtronic, Libre), per-situation distance from 80% is 2.25-4.19 points vs 5.48-6.76 for GARCH.

### F2. Per-situation honesty is not overall efficiency. Tier 2.
- At equal width, all band methods catch nearly the same share of readings (H37). CQR is best on interval score, situation bands close second.
- Claim per-situation calibration, never overall efficiency.

### F3. Calibration by situation trades off dangerous-miss coverage. Tier 2.
- Situation bands catch fewer Clarke zone D points than GARCH (41% vs 62% OhioT1DM; 22% vs 39% AZT1D). A GARCH-normalized version recovers most of the gap (54%, 34%) but not all (H38).
- The per-patient scale-adaptive band recovers part of it (H68): 29.4%, 44.2%, 51.7% vs 22.4%, 41.4%, 45.8% for situation bands (AZT1D, OhioT1DM, HUPA-UCM); GARCH 39.1%, 62.1%, 68.9%.

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

### F9. Error size is only weakly predictable; situation matters more than patient. Tier 3 (H49, H55).
- Group means fitted on validation explain at most 15% of the variation in absolute test error. Situation beats patient on AZT1D (5.8% vs 2.8%) and OhioT1DM (3.3% vs -0.8%); patient wins on HUPA-UCM (11.2% vs 8.0%), the most heterogeneous cohort. Combining both is best on AZT1D and HUPA-UCM.
- GARCH's moment-to-moment scale predicts error size no better than a constant once rescaled (R^2 -0.6%, -6.7%, -4.7%).
- Time of day explains 2 to 3%.
- The earlier claim (from H48) that the patient dominates holds only for HUPA-UCM.
- External (H55): situation explains more than patient in 7 of 8 MetaboNet studies; all groupings explain at most 8.8%. Same at a 30-minute horizon (H71).

### F11. Situation calibration is solved to the limit of what can be measured; the real remaining miscalibration is between patients. Tier 2 (H51-H53).
- A new noise floor for calibration metrics: circularly shift situation labels within each patient (keeps autocorrelation, breaks the link) to see how uneven coverage looks from sampling noise alone.
- Situation bands sit at the floor by situation (2.7 vs 2.9 AZT1D; 2.1 vs 4.0 HUPA-UCM; OhioT1DM 1.6 above, and 1.9 above with a second seed, H75: a small real residual there).
- Patient-by-situation unevenness is 80-95% sampling noise (floors 6.5 to 15.3 points). Comparing methods at that level without a floor is misleading.
- Per-patient unevenness is real: 3.5, 2.3, 9.1 points above the floor.
- Same pattern with the CNN-Transformer forecaster (H69).

### F10. GARCH x situation bands are the most even across patients and situations. Tier 1 (post hoc, Round 1); mostly explained by F11.
- Lowest patient-by-situation calibration error on all three datasets (8.5, 7.6, 12.3 points vs 9.7, 8.3, 16.7 for situation-only), and still lowest or tied after removing each method's overall coverage offset.
- Patient-scaled situation bands (H50) help only on HUPA-UCM.
- Puzzle: GARCH's scale does not predict error size (F9), yet normalizing by it evens out coverage. Round 2 answer: GARCH-based bands have lower sampling-noise floors (6.5-6.8 vs 8.5 AZT1D), so their raw unevenness is lower mostly because of noise, not calibration; by real excess, situation bands are equal or better.
- Mechanism (H57 no, H72 yes): not shorter miss streaks, but more even coverage from day to day.

### F12. An online per-patient scale correction removes the real between-patient miscalibration and sharpens the bands. Tier 3 (H59, H60, H69, H71, H74).
- Situation bands times exp(theta), theta adapted per patient from misses known 60 minutes later (eta 0.01): per-patient excess above the noise floor goes from 3.5, 2.3, 9.1 points to -0.9, -0.9, +0.4; interval score improves (112.0 vs 113.5; 130.2 vs 130.8; 107.5 vs 110.8).
- Cost: per-situation evenness erodes as adaptation speeds up (excess 0.9, 2.5, 1.2 at eta 0.01). Adapting the quantile level (standard ACI, H58) erodes it more.
- Interpretation: the remaining miscalibration is a slowly drifting patient-level error scale, consistent with F6 (error tracks each patient's variability over time).
- Holds for a second forecaster, the CNN-Transformer (H69): per-patient excess 3.5 to -0.8 (AZT1D) and 2.6 to -0.9 (OhioT1DM), interval score slightly better.
- Holds at a 30-minute horizon (H71b): per-patient excess 3.7 to -0.5 (AZT1D), 1.8 to -0.9 (OhioT1DM), interval score better. Situation bands sit at 1.56 and 1.25 points above the floor at 30 minutes (H71a missed its 1.5 rule narrowly).
- External (H60): the per-patient fix replicates in all 8 MetaboNet studies (per-patient excess -0.55 to -0.84; interval score better or within 0.5%), but the single scale costs per-situation evenness (excess over 1.5 in 6 of 8).
- Holds with a second training seed for every development forecaster (H75b): per-patient excess 4.5 to -0.8 (AZT1D), 2.9 to -1.0 (OhioT1DM).
- Holds at 90% bands (H76): per-patient excess 2.6 to -0.6, 1.2 to -0.5, 8.0 to 1.0; at 90% the single scale stays within 1.5 points of situation evenness on all three datasets.
- External (H74): one scale per patient per glucose level keeps the per-patient fix and meets all three criteria in 6 of 8 studies (per-situation excess 0.61-1.62), with the best interval score in 6 of 8. Recommended design for records of several weeks; the single scale for short records (H73 on HUPA-UCM).

### F13. The size of each forecast's error is mostly unpredictable, and errors barely cluster at the 60-minute horizon. Tier 2 (H49, H61, H62).
- Gradient boosting with rich context explains 9.8%, 3.7%, 19.9% of the variation in absolute test error, about 1.4 times the patient x situation group means.
- The top drivers are the forecast value itself and the patient's error scale. The most recently observed error adds little, so volatility clustering (the premise of GARCH) is weak at this horizon.

### F14. Sharpness and conditional calibration trade off. Tier 2 (H58, H59, H63, H64).
- Learned bands (rich-feature CQR) are sharpest but uneven across situations and patients.
- Group-conditional bands are even by situation; online per-patient scaling makes them even by patient; each added adaptation costs about one point of per-situation evenness.
- Best interval score with per-patient calibration: Mondrian CQR x patient scale (108.0, 128.1, 99.6), at a per-situation excess of about 2 points. Best balance of all three criteria: situation x patient scale (H59).
- Three designs were tried for the patient-level fix (H58, H59, H64); all are reported.
- A per-patient, per-level scale (H73, fourth design) restores situation evenness (excess 0.75, 1.15, 0.17) and meets all criteria on AZT1D and OhioT1DM, but learns too slowly on HUPA-UCM's short records (per-patient excess 4.7) and evens days less well. Choice depends on record length.
- Why adaptation costs situation evenness (paper Figure 1): the online scale is driven by misses in the common situations, so when it widens, the rare extreme situations (under 70, 250 and over) over-cover (84-90%).

### F15. Bands for a new patient need no calibration data of that patient's own. Tier 2 (H70, H79).
- Situation quantiles from other patients plus the online per-patient scale match own-data calibration: per-patient excess -0.87, -0.89, 0.49 vs -0.88, -0.91, 0.43 points; interval scores within 0.3.
- Without the online scale, cold bands are uneven across patients (3.6, 2.5, 11.5 points above the noise floor).
- The situation shape of uncertainty transfers between patients; the patient-specific part is a scale that can be learned while the band is in use.
- Caveat in H70: the forecaster itself was still trained per patient.
- Full cold start (H79): a pooled forecaster trained only on the 8 MetaboNet studies forecasts the development patients as well as their own models or better (median patient RMSE 31.5 vs 31.2, 32.7 vs 37.3, 35.8 vs 38.7). With it, situation bands still beat GARCH everywhere, and cold bands with the online patient scale fix per-patient calibration everywhere (excess -0.65, -0.76, 0.50). Situation evenness is weaker with this forecaster on OhioT1DM and HUPA-UCM (excess about 3 points); neither the single nor the per-level scale (H81) brings it under 1.5 there.

### F16. Honesty has a third axis: the day. Tier 2 (H72).
- A static band covers some days far less than others: the SD of daily coverage for situation bands is 10.3, 9.3, 12.4 points.
- Bands that adapt to recent errors even this out: GARCH x situation 7.2, 6.3, 8.9. The online per-patient scale band does best: 5.5, 5.9, 6.7.
- This is what lowers the noise floors of adaptive bands (F11).

### F17. What makes a patient hard to forecast is how fast their glucose moves. Tier 3 (H65-H67, H77, H78; 215 patients, 11 studies).
- Log RMSE vs log glucose SD (training period): leave-one-source-out R^2 0.44, slope 0.64.
- After glucose SD, children (under 18) have 7.5% higher RMSE (90% interval +3.3% to +12.4%).
- Closed loop (AID): +2.4% (-4.3% to +8.9%) as first coded, but 12 BrisT1D patients had missing modality labels coded as non-AID. Corrected (BrisT1D excluded or recoded from device): AID patients are 5.6-6.5% harder (intervals exclude zero).
- Median RMSE by study ranges from 31.2 (AZT1D) to 40.2 mg/dL (BrisT1D, DCLP5).
- The SD of 30-minute glucose changes explains it far better: leave-one-source-out R^2 0.59 alone, 0.63 with glucose SD (H78); with both in the model the glucose-SD slope falls to 0.22 and the rate slope is 0.63.
- Both the child effect and the closed-loop effect vanish once rate variability is included (child +0.4%, -4.5% to +5.0%, H77; AID -2.9% to -3.6%, intervals include zero, label check). Children and closed-loop users are harder to forecast because their glucose moves faster relative to its range (children: median 30.7 vs 25.4 mg/dL per 30 minutes).

### F18. A large share of the 60-minute error may be the sensor's own error, most of all when glucose is high and steady. Tier 1 (H82; estimate under stated assumptions).
- If CGM error is about 10% of glucose on average (published MARD for current sensors is roughly 9-14%) and independent of forecast error, the sensor's error at the target accounts for about 29% of forecast error variance across 11 studies (19% at MARD 8%, 57% at 14%).
- By situation: 39-45% when glucose is flat above 120 mg/dL; 11-15% below 70 or when glucose moves fast, where the dynamics dominate.
- Replaces the simulation lead (F8) as the paper's statement about irreducible error. Lag (about 9 minutes) and calibration bias are not separated here.
