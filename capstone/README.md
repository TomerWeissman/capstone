# Capstone: organization map

Everything produced for the capstone so far, and where it lives. Organized Oct 25, 2026. Nothing was moved; this page points to it.

Deliverables are built in this folder (`capstone/`). The weekly working files stay in `work/`.

## Requirements (sources)

| What | Where |
|---|---|
| Capstone Handbook 2026-27: the four parts of a capstone, assignment descriptions | Google Doc "Capstone Handbook 2026-2027" (mnrva.link/cp-handbook) |
| General Capstone Cover Sheet (required first page of the Midterm PDF) | mnrva.link/cp-general-cover-sheet; its fields are copied into [midterm/STRUCTURE.md](midterm/STRUCTURE.md) |
| Midterm Deliverables on Forum, CP193.010 (Watson) | Weight 25%. Due Sat Oct 31, 4:00 PM; closes Sun Nov 1, 3:00 PM. Graded on #navigation, #outcomeanalysis, #qualitydeliverables. No section-specific instructions beyond the handbook. |
| Full Draft | Due Dec 11 (37.5%) |
| HC and LO plans (my picks from the Project Brief) | Drive: "Capstone planning and project management templates" (tabs HCs, LOs, HC/LO tracker, Gantt) |
| HC definitions | my.minerva.edu, HC handbook |

## Feedback received so far

- Project Brief feedback and scores are on Forum (kept out of this public repo). Its research suggestions (approximate entropy, autoencoders, separating meal and insulin effects from shifts that aren't externally predictable) were tested since: sample entropy and compressibility (H41, H42), teacher-forcing autoencoders (H39, H40, H43, H46-H48), and error-size predictability and patient difficulty (H49, H61, H77, H78).
- Volkan (second reader), Oct 8 meeting: the path is good, and it is critical that the findings have real value. This matched the doubt in the Oct 6 slides ("this may not be a valuable avenue"), which the research program since then was built to answer.

## The work, in order

| Period | Phase | Main outcome | Where |
|---|---|---|---|
| Sep 7-13 | Exploring AZT1D; plain CNN-LSTM baseline | Baseline forecaster works | [work/7_9](../work/7_9/README.md) |
| Sep 14-20 | GLIMMER replication (AZT1D, OhioT1DM) | Replicated; the personalization gain was a seed bug | [work/14_9](../work/14_9/README.md) |
| Sep 21-27 | First uncertainty bands (GARCH, analog ensemble, conformal) | Bands raise too many false alarms; the methods look alike at matched alarms | [work/21_9](../work/21_9/README.md) |
| Sep 28-Oct 4 | Interval quality, more methods, fine-tuning | Plain model has the best intervals; the per-patient fine-tuning gain is unclear | [work/28_9](../work/28_9/README.md) |
| Oct 5-11 | Situation-aware bands; autoencoder and t-SNE; noise ceiling | Situation bands confirmed on fresh data (H28); the autoencoder line was closed | [work/5_10](../work/5_10/README.md) |
| Oct 12-21 | Research program: pre-registered tests, 11 studies, external panel | The paper's findings (H44-H86) | [work/12_10](../work/12_10/README.md) (also holds the Oct 19-21 work; no 19_10 folder was made) |

## Key artifacts

**The paper (work product)**
- Structured draft with every result: [work/12_10/docs/paper_draft.md](../work/12_10/docs/paper_draft.md)
- Prose so far (abstract, introduction, data, methods): [work/12_10/docs/paper_manuscript.md](../work/12_10/docs/paper_manuscript.md)
- Findings with evidence tiers: [work/FINDINGS.md](../work/FINDINGS.md)
- Figures:
  - `work/12_10/figures/paper_fig1_situation_coverage.png`
  - `paper_fig2_three_axes.png`
  - `h51_noise_floor.png`
  - `h54_panel_external.png`
  - `h61_h62_ml_error_size.png`
  - `paper_fig_difficulty.png`
  - `h80_baselines.png`
  - `h82_sensor_floor.png`
  - `error_by_glucose_level.png`
- Code: `src/azt1d/uq_frames.py`, `src/azt1d/group_bands.py`, `work/12_10/scripts/`
- Older Google Doc draft (GLIMMER framing, Oct 7): Drive "Tomer Weissman - Capstone Research Paper Draft"

**Process evidence**
- How work was done: [work/RESEARCH_PROTOCOL.md](../work/RESEARCH_PROTOCOL.md) (pre-registration, evidence tiers)
- Every hypothesis and verdict (H1-H86, including all nulls): [work/HYPOTHESES.md](../work/HYPOTHESES.md)
- Every unit of work, with commits: [work/WORK_LOG.md](../work/WORK_LOG.md)
- Literature searches: `work/12_10/docs/literature_log.md`, `work/5_10/docs/literature_review_uncertainty.md`
- Notion journal (dated entries, Sep 10 onward), Gantt ("The Big Doc of Planning"), and #Nav Notes
- Process documentation started in the Google Doc (journal, Oct 6)
- Slides for Volkan (Drive, Oct 6); Loom video for the Project Brief
- Git history on GitHub: TomerWeissman/capstone (public)

**AI use evidence** (needed in detail for the process documentation)
- The Claude Code session transcripts and the work log record what the AI did.
- The protocol records how its work was checked: pre-registered rules, no thresholds moved after a run, all designs reported.
- Corrections caught along the way show human-in-the-loop checking: the seed bug, the H24 data leak, the BrisT1D label error, the GARCH baseline fairness fix.

## Open items

- The MetaboNet data file is missing from Downloads. It blocks two optional checks (OhioT1DM baselines, the 480-patient replication); the paper doesn't depend on them.
- No week folder for Oct 19-25 (the work is in 12_10). Decide whether to start week 26_10 tomorrow.
