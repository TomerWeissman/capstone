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

### F9. Error size is only weakly predictable, and neither patient nor situation dominates. Tier 2 (H49; replaces the Oct 17 post-hoc hint from H48).
- Group means fitted on validation explain at most 15% of the variation in absolute test error. Situation beats patient on AZT1D (5.8% vs 2.8%) and OhioT1DM (3.3% vs -0.8%); patient wins on HUPA-UCM (11.2% vs 8.0%), the most heterogeneous cohort. Combining both is best on AZT1D and HUPA-UCM.
- GARCH's moment-to-moment scale predicts error size no better than a constant once rescaled (R^2 -0.6%, -6.7%, -4.7%).
- Time of day explains 2 to 3%.
- The earlier claim (from H48) that the patient dominates holds only for HUPA-UCM.

### F11. Situation calibration is solved to the limit of what can be measured; the real remaining miscalibration is between patients. Tier 2 (H51-H53).
- A new noise floor for calibration metrics: circularly shift situation labels within each patient (keeps autocorrelation, breaks the link) to see how uneven coverage looks from sampling noise alone.
- Situation bands sit at the floor by situation (2.7 vs 2.9 AZT1D; 2.1 vs 4.0 HUPA-UCM; OhioT1DM 1.6 above).
- Patient-by-situation unevenness is 80-95% sampling noise (floors 6.5 to 15.3 points). Comparing methods at that level without a floor is misleading.
- Per-patient unevenness is real: 3.5, 2.3, 9.1 points above the floor.
- Same pattern with the CNN-Transformer forecaster (H69).

### F10. GARCH x situation bands are the most even across patients and situations. Tier 1 (post hoc, Round 1); mostly explained by F11.
- Lowest patient-by-situation calibration error on all three datasets (8.5, 7.6, 12.3 points vs 9.7, 8.3, 16.7 for situation-only), and still lowest or tied after removing each method's overall coverage offset.
- Patient-scaled situation bands (H50) help only on HUPA-UCM.
- Puzzle: GARCH's scale does not predict error size (F9), yet normalizing by it evens out coverage. Round 2 answer: GARCH-based bands have lower sampling-noise floors (6.5-6.8 vs 8.5 AZT1D), so their raw unevenness is lower mostly because of noise, not calibration; by real excess, situation bands are equal or better.
- Mechanism (H57 no, H72 yes): not shorter miss streaks, but more even coverage from day to day.

### F12. An online per-patient scale correction removes the real between-patient miscalibration and sharpens the bands. Tier 1 to 2 (H58, H59; second try, recorded as such).
- Situation bands times exp(theta), theta adapted per patient from misses known 60 minutes later (eta 0.01): per-patient excess above the noise floor goes from 3.5, 2.3, 9.1 points to -0.9, -0.9, +0.4; interval score improves (112.0 vs 113.5; 130.2 vs 130.8; 107.5 vs 110.8).
- Cost: per-situation evenness erodes as adaptation speeds up (excess 0.9, 2.5, 1.2 at eta 0.01). Adapting the quantile level (standard ACI, H58) erodes it more.
- Interpretation: the remaining miscalibration is a slowly drifting patient-level error scale, consistent with F6 (error tracks each patient's variability over time).
- Holds for a second forecaster, the CNN-Transformer (H69): per-patient excess 3.5 to -0.8 (AZT1D) and 2.6 to -0.9 (OhioT1DM), interval score slightly better.
- External test pending (H60).

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

### F15. Bands for a new patient need no calibration data of that patient's own. Tier 2 (H70).
- Situation quantiles from other patients plus the online per-patient scale match own-data calibration: per-patient excess -0.87, -0.89, 0.49 vs -0.88, -0.91, 0.43 points; interval scores within 0.3.
- Without the online scale, cold bands are uneven across patients (3.6, 2.5, 11.5 points above the noise floor).
- The situation shape of uncertainty transfers between patients; the patient-specific part is a scale that can be learned while the band is in use.
- Caveat: the forecaster itself was still trained per patient.

### F16. Honesty has a third axis: the day. Tier 2 (H72).
- A static band covers some days far less than others: the SD of daily coverage for situation bands is 10.3, 9.3, 12.4 points.
- Bands that adapt to recent errors even this out: GARCH x situation 7.2, 6.3, 8.9. The online per-patient scale band does best: 5.5, 5.9, 6.7.
- This is what lowers the noise floors of adaptive bands (F11).
