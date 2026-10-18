# Research protocol

How this project turns questions into findings that can go in a journal paper. Written Oct 17, 2026. Every unit of work follows it. When the protocol and a convenient shortcut disagree, the protocol wins.

## 1. The loop

Each question goes through the same eight steps.

1. **Ask a sharp question.** One sentence, about a quantity that can be measured. "Does patient identity explain more of the forecast error than glucose situation?" is sharp. "What drives uncertainty?" is not.
2. **Check what is already known.**
   - Our own record first: `work/HYPOTHESES.md`, `work/FINDINGS.md`, the week READMEs.
   - Then the literature. Log searches and what they found in `docs/literature_log.md` for the week. A gap is "not found in a stated search," never "does not exist."
3. **Pre-register.** Before running, add a row to `HYPOTHESES.md` with the claim, the metric, the threshold, and the data. Commit it. The threshold can't move after the run. If the design must change before the run (for example, runtime), edit the row and say why.
4. **Run the smallest test that can answer it.** Reuse checkpoints. Only train what the question needs.
5. **Read the result honestly.** Supported, not supported, or inconclusive, against the pre-registered rule. Look at every figure before writing about it. Post-hoc observations are labelled post hoc.
6. **Stress it.** Promising results move up the evidence tiers (section 2): more seeds, more datasets, patient bootstrap, alternative explanations.
7. **Record it.** Week README, `WORK_LOG.md`, `HYPOTHESES.md`, and, for results at tier 2 or higher, `FINDINGS.md`.
8. **Generate the next question.** Every result implies something. Write down what it implies, what could explain it instead, and what would falsify it. The best of those becomes the next question.

## 2. Evidence tiers

| Tier | Name | What it takes | Where it goes in a paper |
|---|---|---|---|
| 0 | Look | A picture or a number, no rule set beforehand | Nowhere, or a motivating figure |
| 1 | Preliminary | Pre-registered, one dataset or one seed | Discussion, as a lead |
| 2 | Replicated | Holds on at least two independent datasets, and a patient-level bootstrap interval excludes no effect | Results |
| 3 | Paper-grade | Tier 2, plus external validation on data never used to design the method (held-out MetaboNet sources), plus robustness to the main analysis choices | Headline results and abstract |

## 3. Statistical rules

- **The patient is the unit.** Windows from one patient are not independent. Confidence intervals come from resampling patients (cluster bootstrap). Cross-validation folds are split by patient whenever the claim is about new patients.
- **Test data stays out** of fitting, calibration, early stopping, and threshold choice. Calibration uses the validation period; scoring uses the test period.
- **Seeds.** Every model, ensemble member, and patient gets its own seed.
- **Pooled results can hide one patient.** Report per-patient spread next to pooled numbers. Check that no single patient drives a pooled result.
- **Compare at matched operating points.** Wider bands only add alarms. Compare band methods at matched coverage or matched width, or with a proper scoring rule (interval score).
- **Effect sizes, not just verdicts.** Report the size of every effect with its interval.
- **Count every test.** `HYPOTHESES.md` is the full denominator, nulls included. A paper reports how many hypotheses were tested.
- **Report only numbers seen in outputs or saved result files.**

## 4. Finding gaps and building on studies

- **Replicate before extending.** A new direction starts by reproducing the closest published result, or our own earlier result, on the current data.
- **Beat the strongest baseline, not the weakest.** For bands: plain conformal, per-patient conformal, CQR, GARCH. For forecasts: the plain CNN-LSTM, and a persistence or linear baseline.
- **Name methods by their standard names** (Mondrian conformal, normalized conformal, CQR). The novelty claim is about what we found, not about relabeling a method.
- **External validity.** MetaboNet's 14 sources include closed-loop and pump-with-sensor patients, several CGM devices, children and adults. A finding that holds across these is worth much more than one on 25 patients.

## 5. When to stop or pivot a line

- Stop a line after two pre-registered nulls in a row that give no new mechanism to test. Record the line as closed, with the reason.
- Pivot when a result changes what the most important question is. Record the pivot in the week README.
- Don't tune a method until it wins. If a method needs many tries to beat a baseline, report the number of tries.

## 6. Writing as we go

- `work/FINDINGS.md` is the paper's spine: one entry per finding, with its claim, tier, key numbers with intervals, the figure, and the hypotheses behind it.
- A finding's entry is updated when new evidence changes its tier. Negative findings are kept; they are part of the paper.
- Figures meant for the paper are titled by the question they answer, with plain-language axes.
