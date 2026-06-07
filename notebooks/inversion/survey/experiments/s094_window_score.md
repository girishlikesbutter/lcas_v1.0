---
title: "s094 — windowed (between-anchor) vs full-LC scoring of cross-cloud survivors [BUILT, NOT YET RUN]"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s094_window_score.py
related:
  - experiments/s093_fullc_score_survivors.md
  - experiments/s092_cross_cloud_116.md
created: 2026-05-22
updated: 2026-05-22
confidence: n/a (script built and reviewed; not executed — session ended on the conceptual probe below)
---

> **🚨 PARTLY REVERSED (2026-05-22, see s097).** A `times[0]==0` propagator-gauge bug corrupted the windowed RMSEs. **What's reversed:** the empirical claim that a *short* between-anchor window BEATS the full LC. On correct geometry the **full LC is the BEST window** (top-100 within 10°: W_full **54** > W_ACb 19 > W_AC 13 > W_AB 12) — windowing it down HURTS, and the "divergence-room vs phase-fragility tradeoff" I invoked to justify a short window was a bug artifact (full-LC RMSE never failed — see s093). **What STANDS (and is now confirmed):** the user's structural insight — the interior between anchors is free, unconstrained signal that discriminates by breaking twin-degeneracy, *and the more of it you use the better*. Even W_AB alone beats baseline (12/100 vs 2.5%); W_full is best. The insight was upstream of any run; only the "window it down" operationalization was wrong. See `experiments/s097_times0_gauge_bug_audit.md`.

# TL;DR
**Built, not run.** A re-score of the s092 survivors over several light-curve windows ([A→B], [A→C], [A→C+buffer], full) to test whether scoring only the *interval between anchors* discriminates better than the full LC (which failed in s093). The session ended on a conceptual probe that **corrected a wrong premise of mine** and is the load-bearing takeaway — captured below.

# The insight (the session's main deliverable)
**The connectability ω-solve is purely GEOMETRIC, not a light-curve fit.** `shoot` finds the ω whose torque-free trajectory carries q_a → q_b by minimising the *quaternion geodesic* residual at the endpoint. The light curve never enters the ω objective. Consequence:

- Only **two epochs** (A and B) match the observed brightness *by construction* — because the clouds were *built* brightness-consistent there. That is ~2 epochs out of the ~180 in an [A,C] window.
- The **entire interior is free, unconstrained signal** — nothing optimised it to fit. So scoring the between-anchor interval is scoring *genuine* discrimination, not something rigged. (My earlier "the interval is the worst place to look because the in-between is made to fit" was **wrong**, and the user corrected it.)
- **Why the interior discriminates — brightness-twin breaking:** cloud members q_a, q_b are *brightness-twins* of truth's orientations (same magnitude, possibly different orientation, since brightness is many-to-one — the isophote is a 2-surface). At the endpoints that degeneracy is invisible. At every intermediate epoch the sun/observer geometry differs, the twin-degeneracy breaks, and a junk pair's interior brightness diverges from observed. **This twin-breaking-at-new-geometry IS the discrimination, and it is free of phase accumulation near the anchors.**

The one genuine tension that survives (much milder than "made to fit"): **divergence-room vs phase-fragility.** Epochs far from the anchors discriminate *more* (a wrong ω has had more time to drive the orientation visibly wrong) but are also where a truth-near candidate's tiny ω error has accumulated into a phase shift that pointwise RMSE punishes. Epochs near the anchors discriminate less per epoch but are phase-safe. The [A,C] window is a sensible middle: free discriminating signal with phase accumulation capped.

# What / How
`experiments/s094_window_score.py` re-scores the 79,443 s092 survivors (pool indices from `results/s092/cross.json`): one full propagation each, RMSE sliced over W_AB / W_AC / W_ACb / W_full; reports top-K within-10° enrichment and the best rank of any "truth-like" survivor (q_a<7° AND ω-dir<7°). Pool(24), v2 surrogate. **Not executed.**

# Why this matters
- The windowed test is a *fair* test (scores free signal), so it is worth running — my pessimism was unfounded.
- It probes whether a shorter window (less phase-wrap → potentially wider coherent basin than s003's ~1° full-LC tube) can rank truth-near survivors that full-LC RMSE could not (s093).
- Caveat it cannot fix: at 30k pool the best survivor is 3.3°; if that is simply too far for *any* metric, windowing and pool-density are confounded — the clean disentangler is a denser-pool re-run.

# Out of scope / next
- Run s094; compare windows to the s093 full-LC baseline.
- Denser-pool cross+score (isolate coherent-tube factor from metric factor).
- Sharp-epoch / σ-weighted (RF25-style) objective as the alternative to flat RMSE.

# Cross-references
- `s093_fullc_score_survivors.md` — the full-LC negative this responds to.
- `s092_cross_cloud_116.md` — the survivors being re-scored.
- `s003_landscape_vs_omega.md` — the ~1° coherent-tube result.
