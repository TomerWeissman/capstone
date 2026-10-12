# Overview: where the capstone stands

Updated at the end of every week. Last updated: 2026-10-12 (end of week 5_10). Draft: the user has not yet confirmed this paragraph.

We have moved from replicating GLIMMER to situation-aware uncertainty bands. Sizing bands separately for each glucose situation (level by 30-minute trend) makes them honest in every situation, and this held on a dataset never used for tuning, across models, and under resampling. It does not make bands more efficient overall, and it catches fewer dangerous misses than volatility-based bands. Attempts to explain why some patients are harder to predict found nothing beyond glucose variability, and more data would barely help the model. This week tests the bands against known truth in simulated data, ahead of the Oct 30 midterm draft.
