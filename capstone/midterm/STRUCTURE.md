# Midterm Deliverable: structure draft (v2, for your review)

Barebones skeleton only. Mark it up: cut, move, rename. Once you approve it, I write the actual draft.

**Due:** Sat Oct 31, 4:00 PM (Forum closes Sun Nov 1, 3:00 PM). **Weight:** 25%. **Graded on:** #qualitydeliverables, #navigation, #outcomeanalysis (#curation possible).

**What the handbook asks for:** one PDF, cover sheet first, then a substantial, focused piece of the work product, plus process documentation of how that piece was made. Not a rough draft of everything.

**Proposed piece:** the core of the journal manuscript, meaning the main finding with its evidence. Side findings (patient difficulty, the sensor floor, error-size predictability) are named as "planned for the full draft" instead of written up.

---

## Part A. General Capstone Cover Sheet (template fields, in order)

1. Name
2. Current assignment: Midterm Deliverables
3. Major(s), concentration(s), minor(s): *you fill in*
4. Capstone title: *proposed:* "Honest uncertainty for CGM glucose forecasts: prediction intervals that adapt to the glycemic situation and the patient"
5. Abstract: 150-200 words, academic (from the manuscript abstract)
6. Advisor name: Prof. Watson
7. Real note to advisor: *you write.* Possible topics:
   - the pivot from GLIMMER to uncertainty
   - which journal or venue to target
   - feedback wanted on the paper's framing
8. Confirmations (initials): handbook read; info sheet updated; group systems understood
9. Links to working files:
   - main folder (GitHub repo)
   - main writeup (manuscript)
   - data and analysis (`work/`, HYPOTHESES.md)
   - GitHub
   - HC & LO plans (Drive spreadsheet)
   - planning and progress (Notion)
10. Since the last assignment:
    - Summary of changes: the title and thesis changed, GLIMMER replication became uncertainty calibration
    - Brief reflection on progress and process
    - Feedback implemented: Watson's ApEn and autoencoder suggestions were tested; the results and how they shaped the direction
11. Interdisciplinary minor: *N/A? you confirm*
12. **AI statement (required; penalties if thin)**

## Part B. Work product: placeholder for now

Left mostly empty on purpose (Oct 25 decision): the paper will change, so it is written after the process documentation. For now:
- working title and a 2-3 sentence summary of the planned manuscript
- a link to the current structured draft (`work/12_10/docs/paper_draft.md`)

## Part C. Process documentation (the main focus now; written in the Google Doc, built on continuously)

1. **Overview:** what this deliverable is and how it fits the year
2. **Navigation: how this piece got made**
   - the path: GLIMMER replication, then uncertainty bands, then situation-aware bands, then the research program
   - pivots and dead ends: the autoencoder line, the simulation, standard ACI
   - setbacks and how they were handled: the seed bug, the data leak, the BrisT1D label error, the missing data file
   - a timeline: Gantt plan vs what happened, and how this month changed the plan to the Dec 11 full draft
3. **Outcome analysis: defining success before running**
   - the research protocol (pre-registration, evidence tiers)
   - success criteria for this deliverable, and a self-assessment against them
   - the full hypothesis register as the evidence (38 tests, nulls included)
4. **Modeling professional process:** how ML and health research is usually done (pre-registration, external validation, baselines) and where this project followed or departed from it
5. **Use of generative AI (detailed)**
   - what Claude did, and what I decided
   - how its work was checked
   - 2-3 annotated excerpts (for example: catching the BrisT1D label error; stopping tuning after five designs; rejecting an unfair GARCH baseline)
6. **HCs and LOs for this deliverable** (short; full appendix later)

## Part D. HCs and LOs to claim (proposed; fewer than the brief, as the handbook advises)

**Capstone LOs (graded):** #navigation, #outcomeanalysis, #qualitydeliverables (+ #curation)

**Major LOs** (from your sheet, all CS):
- #cs156-MLCode
- #cs156-MLExplanation
- #cs166-EmpiricalAnalysis
- #cs114-ModelSelection
- #cs110-CodeReadability

**HCs**
- *From your sheet:*
  - #hypothesisdevelopment
  - #gapanalysis
  - #studyreplication
  - #dataviz
  - #confidenceintervals
  - #thesis
  - #audience
  - #organization
  - #professionalism
- *Suggested additions:*
  - #testability (pre-registered rules)
  - #biasidentification (the label error, unfair baselines)
  - #sampling (the external panel)
  - #significance (bootstrap intervals)

## Decisions for you

1. Is the manuscript core the right piece, or would you rather submit something narrower (for example, only Results 6.1-6.3)?
2. Title and thesis wording.
3. Target journal or venue: formatting can follow it now (see the AAAI paper you mentioned) or later.
4. Word targets above: too long, too short?
5. Which HCs and LOs to claim; any custom LOs?
6. Volkan's feedback from Oct 8, to include under "Feedback implemented".
7. Format: write in Markdown here and convert to PDF, or draft directly in Google Docs?
