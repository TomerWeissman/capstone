# Week 12_10: Oct 12 to Oct 18, 2026

Started Oct 12. The user asked to start now and answer the week-start questions later (findings, overview corrections, Volkan's feedback, the second goal).

## Hypotheses this week
- **H44, not supported**: on simulated patients with planted, level-dependent sensor noise, situation bands were further from 80% per situation than GARCH (7.0 vs 4.6 points). Preliminary: the simulated patients were too easy to forecast.
- **H45, not supported**: the situation band's widening matched the planted noise variance (within a factor of 2) in 3 of 5 level bands, not 4. The rule was wrong-footed: noise raised the actual error by 1.1x to 3.5x its own variance.

- **H46, not supported**: level and trend explain 87% of the encoder summary (rule: under 70%).
- **H47, supported (narrowly)**: basal explains 21% of the leftover (rule 20%), likely standing in for patient identity.
- **H48, not supported**: the leftover improves prediction of the CNN-LSTM's error size by 1.2% (rule 10%).

## Findings
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
