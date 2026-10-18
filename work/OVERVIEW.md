# Overview: where the capstone stands

Updated at the end of every week. Last updated: 2026-10-17 (week 12_10). Not yet confirmed by the user.

The capstone is now a paper about where the uncertainty of a 60-minute glucose forecast lives, tested on 215 patients from 11 public studies, 153 of them in 8 studies held out for validation. Standard error bands are honest on average but wrong in specific glycemic situations; sizing bands by situation (level by trend) fixes that in every external study. A new noise floor shows the real remaining error is between patients, and an online per-patient scale, learned from errors seen an hour later, removes it, sharpens the bands, and needs no calibration data from the new patient. How large any single error will be is mostly unpredictable, and what makes a patient hard to forecast is how fast their glucose moves. The draft is in `work/12_10/docs/paper_draft.md`; next is turning it into the Oct 30 midterm submission.
