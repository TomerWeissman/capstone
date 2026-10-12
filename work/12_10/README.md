# Week 12_10: Oct 12 to Oct 18, 2026

Started Oct 12. The user asked to start now and answer the week-start questions later (findings, overview corrections, Volkan's feedback, the second goal).

## Hypotheses this week
- **H44, testing**: on simulated patients with planted, level-dependent sensor noise, situation bands are closer to 80% per situation than GARCH (rule: at least 20% lower coverage error).
- **H45, testing**: situation bands recover the planted noise. Per level band, the rise in squared 80% half-width from clean to noisy data, over 1.28 squared, is within a factor of 2 of the planted noise variance in at least 4 of 5 level bands.

## Findings

## Outcomes

## Work done
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
| README.md | This page | current |

## Remember
- Simulated data is an answer key for testing methods, not training data for real patients, and not a source of band widths for real patients.
- The Gantt lists the Second Reader Meeting twice: Oct 8 on its own row, Oct 16 in this week's row. Check with the user.
- Week-start questions are still unanswered: corrections to the overview, Volkan's feedback, and the second goal.

## Open
- Goal: start the paper structure for the Oct 30 midterm draft.
- Goal: simulation answer-key test (H44, H45). From the Oct 12 journal entry.
- Second goal (the user picked "Other"): to be asked.
- Carried from 5_10: H18/H20 not run; H17 direction; workback plan to Dec 11; stale numbers in the paper draft.
