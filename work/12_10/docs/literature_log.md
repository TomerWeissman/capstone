# Literature log, week 12_10

Searches run for the research program (see work/RESEARCH_PROTOCOL.md, step 2). Gaps are "not found in these searches," not "does not exist."

## Oct 17: patient vs situation in glucose forecast uncertainty

| Query | What came back | Gap status |
|---|---|---|
| inter-patient variability glucose forecasting error decomposition patient versus glycemic state prediction interval calibration | Inter-patient variability is widely noted as an obstacle (Bunescu et al. 2013, OhioT1DM; GLYFE benchmark, arXiv:2006.15946, "some patients are easier"). Accuracy reported by glycemic range (PMC12650137) and per-patient RMSE spread (Sci Rep 2026). Interval models under intra-patient variability for postprandial glucose (PubMed 22677264). | No paper found that decomposes forecast error (or interval width) into patient vs glycemic-state components, or evaluates interval calibration by patient and by state together. |
| conformal prediction continuous glucose monitoring forecasting per-patient coverage Mondrian group-conditional | General Mondrian conformal theory (Vovk 2003); Mondrian conformal bands for randomly timed biomarker trajectories stratified by covariate group (arXiv:2511.13911); conformal imputation for CGM (Matabuena et al., cited only). | No CGM forecasting paper found with group-conditional or per-patient conformal coverage. Consistent with the Oct 4 review. |
| MetaboNet public dataset license | MetaboNet (arXiv:2601.11505): public subset downloadable from metabo-net.org; the consolidated dataset is for non-commercial research and education; JAEB-hosted studies under JAEB terms of use, others CC BY 4.0. MetaboNet-Bench (arXiv:2606.18640): accuracy benchmark for multimodal forecasting; abstract mentions no intervals, calibration, or per-patient variance. | Using the public subset for this research is within its terms. Data stays out of git. MetaboNet-Bench is the closest large-scale benchmark and does not cover uncertainty. |

Sources: [Bunescu et al. 2013](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/pubs/ICMLA-2013.pdf), [GLYFE](https://arxiv.org/pdf/2006.15946), [PMC12650137](https://pmc.ncbi.nlm.nih.gov/articles/PMC12650137/), [PubMed 22677264](https://pubmed.ncbi.nlm.nih.gov/22677264/), [arXiv:2511.13911](https://arxiv.org/pdf/2511.13911), [MetaboNet](https://arxiv.org/pdf/2601.11505), [MetaboNet-Bench](https://arxiv.org/abs/2606.18640).

## Oct 17: adaptive conformal inference for glucose

| Query | What came back | Gap status |
|---|---|---|
| adaptive conformal inference glucose forecasting CGM online coverage distribution shift | ACI method (Gibbs and Candes, NeurIPS 2021, arXiv:2106.00170). CGM: neural CDE forecaster with conformal risk control and deferral (KTU repository); conformal imputation (arXiv:2403.18069). | No CGM forecasting paper found using ACI-style online updating or evaluating per-patient coverage under drift. |

Sources: [ACI](https://arxiv.org/pdf/2106.00170), [KTU](https://epubl.ktu.edu/object/elaba:279147543/index.html), [arXiv:2403.18069](https://arxiv.org/pdf/2403.18069).

## Oct 17: is the calibration noise floor new?

| Query | What came back | Gap status |
|---|---|---|
| conditional coverage evaluation sampling noise autocorrelated time series prediction intervals subgroup coverage variability null baseline | Conformal for dynamic time series with conditional-coverage bounds (Xu and Xie, ICML 2021); conditional guarantees over subgroup classes (Gibbs, Cherian, Candes, arXiv:2305.12616), which notes small-subgroup width problems; pintervals R package (arXiv:2601.03994) summarizes coverage uniformity across groups without autocorrelation or a null; online multi-step conformal preserving score dependence (arXiv:2410.13115); interval-length vs coverage correlation as a conditional diagnostic (arXiv:2406.14904); evaluation of interval forecasts in econometrics (Diebold's J. Time Ser. Anal. 2018 paper; Christoffersen-style unconditional coverage, independence, and conditional coverage tests). | No method found for a sampling-noise floor of group-level coverage dispersion under autocorrelation. Christoffersen's independence test (clustering of misses) is the closest relative and should be cited and contrasted: it tests serial independence of misses, not whether between-group coverage differences exceed what autocorrelated sampling produces. |

Sources: [Xu and Xie 2021](https://proceedings.mlr.press/v139/xu21h/xu21h.pdf), [Gibbs et al.](https://arxiv.org/pdf/2305.12616), [pintervals](https://arxiv.org/pdf/2601.03994), [arXiv:2410.13115](https://arxiv.org/html/2410.13115v2), [arXiv:2406.14904](https://arxiv.org/html/2406.14904), [Diebold](https://www.sas.upenn.edu/~fdiebold/papers2/Eval.pdf).

## Oct 18: systematic check for prior work that would overlap the paper

| Query | Closest prior work found | What it means for our claims |
|---|---|---|
| conformal prediction blood glucose forecasting prediction intervals 2025 2026 | In-progress GitHub project with conformal intervals for 30/60-min CGM forecasts (no method or results yet); a 2026 review of foundation models for T1D forecasting (Technologies 14:399) calls for calibrated intervals and coverage metrics in future benchmarks; older conformal intervals for 5-year HbA1c (arXiv:2012.06564). | No published conformal CGM-forecasting study found. The 2026 review supports the need. |
| glucose prediction uncertainty calibration stratified by glycemic range, rate of change, coverage, hypoglycemia | ICU glucose prediction (PMC8324237) reports 95% interval coverage by blood glucose range (conservative at 100-169 mg/dL). Uncertainty-aware horizon control (MAKE 8:145, 2026): aleatoric uncertainty undercovers at all levels; shortens the horizon under rapid variability. Tan and McBeth 2026 (arXiv:2603.04955): coverage at several nominal levels, Brier for hypo/hyper detection. TFT study (PMC12349322): bands too narrow before a hyperglycemic excursion. Normal reference for glucose rates of change (Richardson 2026, JDST). | Range-stratified coverage has a precedent (ICU, not ambulatory T1D, no trend, no correction). Joint level-by-trend coverage in T1D CGM forecasting and a method to correct it: not found. Must cite PMC8324237 and MAKE 8:145. |
| personalized uncertainty quantification CGM patient-specific calibration online adaptation | Gluformer (arXiv:2209.04526); Zhu et al. meta-learning + evidential personalization (cited in Sci Rep 2025); TFT personalized (PMC12349322); per-patient retraining (GSD study). | No online, streaming per-patient recalibration of intervals found. Zhu et al. is the closest personalization-of-uncertainty work and should be cited and contrasted (meta-learning before deployment vs online scale during use). |

Sources: [GitHub cgm-forecast-conformal](https://github.com/suhasreddybv/cgm-forecast-conformal), [Technologies 2026](https://doi.org/10.3390/technologies14070399), [PMC8324237](https://pmc.ncbi.nlm.nih.gov/articles/PMC8324237/), [MAKE 8:145](https://doi.org/10.3390/make8060145), [arXiv:2603.04955](https://arxiv.org/html/2603.04955), [PMC12349322](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349322/), [Richardson 2026](https://journals.sagepub.com/doi/10.1177/15209156251390822), [Gluformer](https://arxiv.org/html/2209.04526v2), [GSD study](https://www.nature.com/articles/s41598-025-97391-8).
