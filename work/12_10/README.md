# Week 12_10: Oct 12 to Oct 18, 2026

Started Oct 12. The user asked to start now and answer the week-start questions later (findings, overview corrections, Volkan's feedback, the second goal).

## Hypotheses this week
- **H44, not supported**: on simulated patients with planted, level-dependent sensor noise, situation bands were further from 80% per situation than GARCH (7.0 vs 4.6 points). Preliminary: the simulated patients were too easy to forecast.
- **H45, not supported**: the situation band's widening matched the planted noise variance (within a factor of 2) in 3 of 5 level bands, not 4. The rule was wrong-footed: noise raised the actual error by 1.1x to 3.5x its own variance.

- **H46, not supported**: level and trend explain 87% of the encoder summary (rule: under 70%).
- **H47, supported (narrowly)**: basal explains 21% of the leftover (rule 20%), likely standing in for patient identity.
- **H48, not supported**: the leftover improves prediction of the CNN-LSTM's error size by 1.2% (rule 10%).

- **H49, not supported**: patient explains more of the error size than situation on 1 of 3 datasets; error size is weakly predictable (at most 15%).
- **H50, not supported**: patient-scaled situation bands help only on HUPA-UCM (+28%); post hoc, GARCH x situation is most even everywhere.

- **H51, supported**: situation calibration is at the sampling-noise floor on 2 of 3 datasets.
- **H52, not supported**: patient-by-situation unevenness is mostly sampling noise (real excess 1 to 2 points).
- **H53, supported**: real miscalibration remains between patients (2.3 to 9.1 points above noise).

- **H57, not supported**: GARCH-scaled bands do not have much shorter miss streaks; the lower noise floor is unexplained.
- **H58, not supported**: standard adaptive conformal fixes per-patient calibration but erodes per-situation calibration.
- **H59, supported (2 of 3)**: a per-patient multiplicative scale, adapted online, fixes per-patient calibration, sharpens the band, and mostly keeps situation calibration.

- **H61, not supported**: machine learning explains 1.4x what groups explain of error size (up to 20%).
- **H62, not supported**: the latest known error is a weak predictor; forecast value and patient scale lead.
- **H63, not supported**: rich-feature CQR is sharpest but conditionally miscalibrated.
- **H64, not supported**: Mondrian CQR with patient scale is sharpest and patient-calibrated but about 2 points uneven by situation.

- **H68, supported**: the patient-scale band catches more dangerous misses than situation bands on all three datasets (still fewer than GARCH).

- **H69, supported**: the noise-floor and per-patient-scale findings hold for the CNN-Transformer forecaster.

- **H70, supported**: bands calibrated only on other patients plus the online patient scale are as honest as own-data bands.

- **H72, supported**: adaptive bands even out coverage from day to day; the patient-scale band is most even.

- **H73, supported (2 of 3)**: a per-patient, per-level online scale keeps situation evenness and fixes patients on long records; too slow on HUPA-UCM's short ones.

- **H71, not supported (narrowly)**: at 30 minutes the patient-scale result holds; situation bands are 1.56 points above the floor on AZT1D (rule 1.5).

- **H54, confirmed**: situation bands beat GARCH per situation in all 8 external MetaboNet studies.
- **H55, supported**: situation explains more error size than patient in 7 of 8 studies.
- **H56, supported**: GARCH x situation most even by patient-by-situation in 7 of 8 (mostly a lower noise floor).
- **H60, not supported**: the single patient scale fixes patients in all 8 studies but costs situation evenness in 6.
- **H74, supported**: the per-level patient scale meets all criteria in 6 of 8 studies.
- **H65, not supported**: glucose variability explains 44% of between-patient difficulty in new studies.
- **H66, not supported**: closed loop makes no detectable difference.
- **H67, supported**: children are 7.5% harder to forecast after variability.

- **H76, supported**: the findings hold for 90% bands.

- **H75, not supported (part a)**: with a second seed the patient fix holds; OhioT1DM's situation bands stay 1.9 points above the floor (a small real residual).

- **H77, supported**: the child effect is explained by faster-moving glucose.
- **H78, supported**: rate-of-change variability explains 59-63% of between-patient forecast difficulty in new studies.

## Findings
- **Research program started (Oct 17).** Protocol in [../RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md), findings ledger in [../FINDINGS.md](../FINDINGS.md). Shared code: `src/azt1d/uq_frames.py` (frames for any dataset, cached) and `src/azt1d/group_bands.py` (band methods and metrics); the frames reproduce the H28/H31 numbers exactly.
- **External validation (MetaboNet panel, 153 patients, 8 studies never used for design).** Situation calibration replicates everywhere (tier 3). The per-patient scale fixes between-patient miscalibration everywhere; with one scale per glucose level it also keeps situation calibration in 6 of 8 studies. Across 215 patients from 11 studies, glucose variability explains 44% of how hard a patient is to forecast; children are 7.5% harder; closed loop makes no difference.
- **Round 4: how predictable is error size, and can a learned band be both sharp and calibrated?** Error size is mostly unpredictable even with rich context (at most 20%), and recent errors barely help: errors do not cluster at the 60-minute horizon. Learned bands are sharper but uneven; adding situation calibration and patient scaling fixes patients but leaves about 2 points of situation unevenness. Sharpness and conditional calibration trade off.
- **Round 3: can the between-patient miscalibration be fixed?** Yes, online. Situation bands times a per-patient scale updated from misses known 60 minutes later remove the real per-patient miscalibration on all three datasets and lower the interval score. Faster adaptation erodes situation evenness; standard adaptive conformal (shifting the quantile level) erodes it more.
- **Round 2: how much of the bands' unevenness is real?** A sampling-noise floor (situation labels circularly shifted within each patient) shows situation calibration is solved to the limit of detection, patient-by-situation unevenness is mostly noise, and the real remaining miscalibration is between patients. GARCH-based bands look more even mainly because their noise floor is lower.
- **Round 1: what sets the size of the forecast error?** Neither patient nor situation dominates, and both together explain at most 15% (H49). The earlier "patient dominates" hint was HUPA-UCM only. Patient-scaled bands help only there (H50). GARCH x situation is the most even band across patient-by-situation cells on all three datasets.
- **The encoder summary is mostly level and trend (87%).** The leftover 13% has two main directions. Removing it barely changes the decoder (5.77 to 5.85 mg/dL).
- **The leftover is mostly "which patient, on which pump settings".** Basal explains 21% of it and heart rate 12%; meals, insulin, and curve shape explain nothing for new patients. Everything recorded together explains 48%, so about half is still unexplained.
- **The leftover does not predict how wrong the forecast will be** (1.2% better). More striking: for a patient the model has never seen, level and trend don't predict the size of the forecast error better than a constant guess. Error size is set mostly by the patient, not the moment. This fits H29 (errors follow each patient's variability) and says per-patient calibration matters at least as much as per-situation.
- **Noise costs more than its own size.** Planting sensor noise raised the 60-minute forecast's squared error by 1.1x to 3.5x the noise variance, depending on level. The noise hits twice: once in the reading being forecast, and again in the past readings the model uses as inputs. So an error-based noise estimate (like H23's) would overstate the sensor's noise.
- **The situation band follows the real change in error; GARCH does not.** From clean to noisy data, the situation band widened by 0.6x to 1.3x the actual rise in error in every level band. GARCH's width rose too much in 70-120 and fell above 180, where the noise was largest. Post hoc, not the pre-registered rule.
- **On simulated data, situation bands were less calibrated than GARCH** (7.0 vs 4.6 points), mostly under-covering (77.8% overall). GARCH adapts to recent errors. The real-data result (H28) does not carry over to these simulated patients.
- **The simulator is too easy.** Glucose levels match the real datasets well, but glucose moves more slowly and the forecaster's error is about half of real (median 11 clean, 16 noisy vs 31 AZT1D, 39 HUPA-UCM). Missing: day-to-day changes in insulin sensitivity, exercise, stress, sensor artifacts.

## Outcomes
- H44: coverage error GARCH 4.6, situation 7.0 points; situation better for 4 of 8 patients; overall coverage GARCH 83.1%, situation 77.8%.
- H45 ratios (situation band / GARCH / actual squared error), by level: under 70 0.66 / 1.43 / 1.10; 70-120 1.76 / 4.52 / 2.39; 120-180 3.20 / 3.37 / 3.49; 180-250 2.88 / -0.15 / 2.29; 250+ 1.29 / -0.56 / 1.57. Test windows per level 483, 5311, 6928, 838, 208.
- Realism: mean glucose 132 sim vs 146 AZT1D and 141 HUPA-UCM; SD 45 vs 48 and 57; time in range 86% vs 78% and 72%; SD of the 30-minute change 13 to 16 vs 24 and 21; CNN-LSTM RMSE median 11 (clean) and 16 (noisy) vs 31 and 39.
- Runtime: simulation 399 s; training 16 models and analysis 243 s.

## Work done
- Oct 17: research protocol, findings ledger, shared frames and band modules; Round 1 (H49, H50); MetaboNet panel training started.
- Oct 17: H46-H48, encoder leftover after level and trend (531 s).
- Oct 14: key-figures notebook (9 figures in three sections).
- Oct 14: notebook collecting all 21 uncertainty-band figures (5_10, 12_10).
- Oct 12: simglucose installed (needs setuptools<81); simulated 8 virtual adults x 30 days; ran H44/H45.
- Oct 12: week start. Carried forward the situation band scripts, the presentation figure, and the literature review (copies; originals stay in 5_10).

## Files
| File | What it is | Status |
|---|---|---|
| [scripts/situation_bands.py](scripts/situation_bands.py) | Carried from 5_10. Situation-specific band widths (conformal_halfwidth) | carried; original in 5_10 |
| [scripts/predictability_map.py](scripts/predictability_map.py) | Carried from 5_10. Situation grid constants and the predictability map | carried; original in 5_10 |
| [scripts/hybrid_bands.py](scripts/hybrid_bands.py) | Carried from 5_10. Frames with GARCH sigma; imported by verification_checks | carried; original in 5_10 |
| [scripts/verification_checks.py](scripts/verification_checks.py) | Carried from 5_10. Band methods (Mondrian, GARCH, hybrid, CQR) and coverage error | carried; original in 5_10 |
| [scripts/presentation_figure.py](scripts/presentation_figure.py) | Carried from 5_10. Presentation figure script | carried; original in 5_10 |
| [figures/presentation_situation_aware.png](figures/presentation_situation_aware.png) | Carried from 5_10. Presentation figure | carried; original in 5_10 |
| [docs/literature_review_uncertainty.md](docs/literature_review_uncertainty.md) | Carried from 5_10. Literature review, seven gaps | carried; original in 5_10 |
| [notebooks/01_uncertainty_band_figures.ipynb](notebooks/01_uncertainty_band_figures.ipynb) | All saved uncertainty-band figures from 5_10 and 12_10 in one place, images only | current |
| [notebooks/02_key_figures.ipynb](notebooks/02_key_figures.ipynb) | Short selection, in order: bands with false triggers marked, band methods before situations, situation grid, teacher-forcing encoder, distance from 80% (H28 on). Images only | current |
| figures/alarms_*_21_9.png (5 files) | Extracted from 21_9 notebooks 01 and 02: 24-hour band views with correct and false triggers marked | current |
| [figures/methods_tradeoff_21_9.png](figures/methods_tradeoff_21_9.png) | Extracted from 21_9 notebook 01: danger caught vs false triggers, three methods | current |
| [figures/methods_width_vs_error_28_9.png](figures/methods_width_vs_error_28_9.png) | Extracted from 28_9 notebook 04: band width vs forecast error, six methods | current |
| [scripts/round1_patient_vs_situation.py](scripts/round1_patient_vs_situation.py) | Round 1, H49/H50: error-size predictors and seven band methods, three calibration levels | current |
| [results/verdict_h49_h50.txt](results/verdict_h49_h50.txt) | H49/H50 verdicts with bootstrap intervals | current |
| results/round1_*.csv | Per-dataset band metrics; R^2 table | current |
| [figures/h49_error_sources.png](figures/h49_error_sources.png) | H49: what predicts the size of the forecast error | current |
| [figures/h50_patient_scaled_bands.png](figures/h50_patient_scaled_bands.png) | H50: calibration by situation, patient, patient x situation for six band methods | current |
| [scripts/round2_noise_floor.py](scripts/round2_noise_floor.py) | Round 2, H51-H53: observed unevenness vs sampling-noise floor, three levels, five methods | current |
| [results/verdict_h51_h53.txt](results/verdict_h51_h53.txt) | H51-H53 verdicts with every observed/floor pair | current |
| [results/round2_noise_floor.csv](results/round2_noise_floor.csv) | Observed, floor, excess per dataset, method, level | current |
| [figures/h51_noise_floor.png](figures/h51_noise_floor.png) | Observed unevenness vs noise floor, 3 datasets x 3 levels | current |
| [scripts/round3_aci_and_streaks.py](scripts/round3_aci_and_streaks.py) | Round 3, H57-H59: miss streaks, adaptive conformal, per-patient scale adaptation | current |
| [results/verdict_h57_h59.txt](results/verdict_h57_h59.txt) | H57-H59 verdicts and full tables | current |
| [results/round3_aci.csv](results/round3_aci.csv) | Per-dataset, per-method metrics incl. noise floors and miss runs | current |
| [figures/h57_h59_aci.png](figures/h57_h59_aci.png) | Miss runs, per-patient excess, interval score for nine methods | current |
| [scripts/round4_ml_error_size.py](scripts/round4_ml_error_size.py) | Round 4, H61-H64: gradient boosting of error size, permutation importance, CQR and Mondrian CQR bands | current |
| [results/verdict_h61_h64.txt](results/verdict_h61_h64.txt) | H61-H64 verdicts, importance table, band table | current |
| results/round4_*.csv | R^2, importance, band metrics | current |
| [figures/h61_h62_ml_error_size.png](figures/h61_h62_ml_error_size.png) | Error-size predictability and feature importance | current |
| [scripts/round5_zone_d.py](scripts/round5_zone_d.py) | H68: Clarke zone catch rates for four band methods | current |
| [results/verdict_h68.txt](results/verdict_h68.txt) | H68 verdict and zone D table | current |
| [results/round5_zone_d.csv](results/round5_zone_d.csv) | Catch rate per zone, method, dataset | current |
| [scripts/round7_other_forecaster.py](scripts/round7_other_forecaster.py) | H69: Rounds 2-3 findings with the CNN-Transformer forecaster | current |
| [results/verdict_h69.txt](results/verdict_h69.txt) | H69 verdict and tables | current |
| [scripts/round6_panel.py](scripts/round6_panel.py) | Round 6: external panel tests (H54-H56, H60) and patient determinants (H65-H67) | current |
| [scripts/round8_cold_start.py](scripts/round8_cold_start.py) | H70: cold-start bands (leave-one-patient-out calibration) with and without the online patient scale | current |
| [results/verdict_h70.txt](results/verdict_h70.txt) | H70 verdict and tables | current |
| [docs/paper_draft.md](docs/paper_draft.md) | Paper draft built from the findings ledger | current |
| [scripts/round10_daily_coverage.py](scripts/round10_daily_coverage.py) | H72: SD of daily coverage per method | current |
| [results/verdict_h72.txt](results/verdict_h72.txt) | H72 verdict | current |
| [scripts/train_horizon30.py](scripts/train_horizon30.py) | 30-minute-horizon CNN-LSTMs for H71 | current |
| [scripts/round9_horizon30.py](scripts/round9_horizon30.py) | H71: Rounds 2-3 findings at 30 minutes | current |
| [scripts/round11_level_scale.py](scripts/round11_level_scale.py) | H73: per-patient per-level online scale vs single scale | current |
| [results/verdict_h73.txt](results/verdict_h73.txt) | H73 verdict and tables | current |
| [scripts/paper_figures.py](scripts/paper_figures.py) | Paper Figures 1-2: situation coverage maps; three axes of honesty | current |
| [figures/paper_fig1_situation_coverage.png](figures/paper_fig1_situation_coverage.png) | Coverage in each of 25 situations for GARCH, situation, situation x patient scale | current |
| [figures/paper_fig2_three_axes.png](figures/paper_fig2_three_axes.png) | Situation excess, patient excess, daily-coverage SD per dataset | current |
| [results/verdict_h71.txt](results/verdict_h71.txt) | H71 verdict and tables (30-minute horizon) | current |
| [results/verdict_h54_h56_h60.txt](results/verdict_h54_h56_h60.txt) | External panel verdicts H54-H56, H60, full per-study tables | current |
| [results/verdict_h65_h67.txt](results/verdict_h65_h67.txt) | Patient-difficulty verdicts H65-H67 | current |
| [results/round6_panel_bands.csv](results/round6_panel_bands.csv) | Panel band metrics per study and method | current |
| [results/round6_patients.csv](results/round6_patients.csv) | 215 patients: RMSE, glucose stats, source, modality, age | current |
| [figures/h54_panel_external.png](figures/h54_panel_external.png) | External validation: per-situation error, per-patient excess, interval score | current |
| [figures/h65_patient_difficulty.png](figures/h65_patient_difficulty.png) | Patient RMSE vs glucose SD, 11 studies | current |
| [scripts/round12_panel_level_scale.py](scripts/round12_panel_level_scale.py) | H74: per-level patient scale on the panel | current |
| [results/verdict_h74.txt](results/verdict_h74.txt) | H74 verdict and per-study tables | current |
| results/metabonet_panel_log*.txt | Panel build and training logs | current |
| [scripts/round13_level90.py](scripts/round13_level90.py) | H76: 90% bands | current |
| [results/verdict_h76.txt](results/verdict_h76.txt) | H76 verdict and tables | current |
| [scripts/train_seed2.py](scripts/train_seed2.py) | Second-seed forecasters for H75 | current |
| [scripts/round14_seed2.py](scripts/round14_seed2.py) | H75: second-seed check | current |
| [results/verdict_h75.txt](results/verdict_h75.txt) | H75 verdict and tables | current |
| [scripts/round15_rate_variability.py](scripts/round15_rate_variability.py) | H77/H78: rate-of-change variability and patient difficulty | current |
| [results/verdict_h77_h78.txt](results/verdict_h77_h78.txt) | H77/H78 verdicts | current |
| [results/round15_patients.csv](results/round15_patients.csv) | 215 patients with rate-of-change SD | current |
| [scripts/build_metabonet_panel.py](scripts/build_metabonet_panel.py) | 160-patient external panel from 8 MetaboNet sources (45 days each, one seed per patient) | current |
| [docs/literature_log.md](docs/literature_log.md) | Literature searches for the research program | current |
| [scripts/encoder_leftover.py](scripts/encoder_leftover.py) | H46-H48: removes level and trend from the 7-input encoder summary and tests what the leftover tracks and whether it predicts forecast error | current |
| [results/verdict_h46_h48.txt](results/verdict_h46_h48.txt) | H46-H48 verdicts and numbers | current |
| [results/encoder_leftover_variables.csv](results/encoder_leftover_variables.csv) | Share of the leftover explained by each variable | current |
| [figures/h46_leftover_share.png](figures/h46_leftover_share.png) | H46: level-and-trend share of the summary; PCA of the leftover | current |
| [figures/h47_leftover_variables.png](figures/h47_leftover_variables.png) | H47: what each variable explains of the leftover | current |
| [figures/h48_leftover_error.png](figures/h48_leftover_error.png) | Decoder with and without the leftover; H48 error-size prediction | current |
| [scripts/simulate_patients.py](scripts/simulate_patients.py) | Simulates 8 virtual adults x 30 days with simglucose, messier boluses and unlogged snacks; saves noise-free glucose | current |
| [scripts/sim_answer_key.py](scripts/sim_answer_key.py) | H44/H45: plants level-dependent noise, trains the CNN-LSTM on clean and noisy copies, scores both bands, realism table | current |
| [results/verdict_h44_h45.txt](results/verdict_h44_h45.txt) | H44/H45 verdicts, recovery table, realism table | current |
| [results/h45_noise_recovery.csv](results/h45_noise_recovery.csv) | Per-level widening ratios | current |
| [results/sim_realism.csv](results/sim_realism.csv) | Simulated vs AZT1D vs HUPA-UCM summary statistics | current |
| [results/simulate_log.txt](results/simulate_log.txt) | Console log of the simulation run | current |
| [figures/sim_realism.png](figures/sim_realism.png) | Glucose levels, 30-minute change, and forecast error: simulated vs real | current |
| [figures/h44_sim_coverage.png](figures/h44_sim_coverage.png) | H44: coverage per situation, GARCH vs situation bands, simulated noisy data | current |
| [figures/h45_noise_recovery.png](figures/h45_noise_recovery.png) | H45: widening from clean to noisy, per level, vs planted noise | current |
| README.md | This page | current |

## Remember
- Simulated glucose is in data/processed/simulated/ (not in git; rerun simulate_patients.py). Simulated checkpoints are sim_clean_cnn_lstm_v0 and sim_noisy_cnn_lstm_v0.
- The planted noise is independent reading to reading. Real sensor noise is correlated over time.
- Simulated data is an answer key for testing methods, not training data for real patients, and not a source of band widths for real patients.
- The Gantt lists the Second Reader Meeting twice: Oct 8 on its own row, Oct 16 in this week's row. Check with the user.
- Week-start questions are still unanswered: corrections to the overview, Volkan's feedback, and the second goal.

## Open
- Goal: start the paper structure for the Oct 30 midterm draft.
- Goal: simulation answer-key test (H44, H45). From the Oct 12 journal entry.
- Second goal (the user picked "Other"): to be asked.
- Carried from 5_10: H18/H20 not run; H17 direction; workback plan to Dec 11; stale numbers in the paper draft.
