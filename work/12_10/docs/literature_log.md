# Literature log, week 12_10

Searches run for the research program (see work/RESEARCH_PROTOCOL.md, step 2). Gaps are "not found in these searches," not "does not exist."

## Oct 17: patient vs situation in glucose forecast uncertainty

| Query | What came back | Gap status |
|---|---|---|
| inter-patient variability glucose forecasting error decomposition patient versus glycemic state prediction interval calibration | Inter-patient variability is widely noted as an obstacle (Bunescu et al. 2013, OhioT1DM; GLYFE benchmark, arXiv:2006.15946, "some patients are easier"). Accuracy reported by glycemic range (PMC12650137) and per-patient RMSE spread (Sci Rep 2026). Interval models under intra-patient variability for postprandial glucose (PubMed 22677264). | No paper found that decomposes forecast error (or interval width) into patient vs glycemic-state components, or evaluates interval calibration by patient and by state together. |
| conformal prediction continuous glucose monitoring forecasting per-patient coverage Mondrian group-conditional | General Mondrian conformal theory (Vovk 2003); Mondrian conformal bands for randomly timed biomarker trajectories stratified by covariate group (arXiv:2511.13911); conformal imputation for CGM (Matabuena et al., cited only). | No CGM forecasting paper found with group-conditional or per-patient conformal coverage. Consistent with the Oct 4 review. |
| MetaboNet public dataset license | MetaboNet (arXiv:2601.11505): public subset downloadable from metabo-net.org; the consolidated dataset is for non-commercial research and education; JAEB-hosted studies under JAEB terms of use, others CC BY 4.0. MetaboNet-Bench (arXiv:2606.18640): accuracy benchmark for multimodal forecasting; abstract mentions no intervals, calibration, or per-patient variance. | Using the public subset for this research is within its terms. Data stays out of git. MetaboNet-Bench is the closest large-scale benchmark and does not cover uncertainty. |

Sources: [Bunescu et al. 2013](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/pubs/ICMLA-2013.pdf), [GLYFE](https://arxiv.org/pdf/2006.15946), [PMC12650137](https://pmc.ncbi.nlm.nih.gov/articles/PMC12650137/), [PubMed 22677264](https://pubmed.ncbi.nlm.nih.gov/22677264/), [arXiv:2511.13911](https://arxiv.org/pdf/2511.13911), [MetaboNet](https://arxiv.org/pdf/2601.11505), [MetaboNet-Bench](https://arxiv.org/abs/2606.18640).
