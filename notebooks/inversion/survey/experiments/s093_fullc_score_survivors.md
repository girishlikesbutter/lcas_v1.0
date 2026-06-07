---
title: "s093 — full-LC RMSE does NOT rank the s092 cross-cloud survivors toward truth"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s093_fullc_score_survivors.py
  - notebooks/inversion/survey/results/s093/score.json
related:
  - experiments/s092_cross_cloud_116.md
  - experiments/s003_landscape_vs_omega.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (clean negative; truth-LC RMSE floor confirms scoring infra is correct, so the failure is real, not a bug)
---

> **🚨 RETRACTED (2026-05-22, see s097).** This conclusion was a `times[0]==0` propagator-gauge bug: the candidate trajectory was mis-referenced by up to 108°. **Re-run on correct geometry REVERSES it** — most-truth-like survivor ranks **4/79,443** (not 31,515), top-100 is **54%** within 10° (not 0). **Full-LC RMSE is a STRONG discriminator.** Everything below is the original pre-fix text, kept for the historical chain. See `experiments/s097_times0_gauge_bug_audit.md`.

# TL;DR
Applied the "real discriminator" — full-LC v2-surrogate RMSE — to the 79,443 s092 survivors. It **fails to rank truth up**: top-100 has **0** survivors within 10° ω-dir of truth, top-500 enrichment (2.2%) is *at/below* the 2.46% baseline, the actual truth pair (ω-dir 3.27°) does **not make the top 500**, and the most-truth-like-by-ω survivor ranks **31,515 / 79,443** (source: `results/s093/score.json`). Truth's own LC fits to RMSE 0.0102 mag, but a junk state at ω-dir 112.9° tops the ranking at RMSE 0.544. **Architectural consequence: full-LC pointwise RMSE is not a working discriminator on prefilter survivors — they sit outside the coherent basin (s003), and pointwise RMSE is phase-fragile.**

# What
s092 left truth retained but buried in 79k survivors. The plan was that full-LC surrogate scoring (s081-validated agreement) would isolate it. This tests that directly.

# How
`experiments/s093_fullc_score_survivors.py`, Pool(24). For each survivor pair (pool indices from `results/s092/cross.json`): re-solve ω (finite-diff-init `shoot`), propagate (q_a, ω) over ALL epochs relative to anchor A, surrogate-predict the LC, RMSE vs observed `mag_hifi`. Rank by RMSE; measure enrichment of near-truth candidates at the top.

# Result
| metric | value |
|---|---|
| truth's own LC RMSE (floor) | 0.0102 mag |
| #1-ranked survivor | RMSE 0.544, ω-dir 112.9°, q_a 98.8° |
| top-100 within 10° ω-dir | 0 |
| top-500 enrichment | 2.2% (vs 2.46% baseline) |
| truth pair (ω-dir 3.27°) | worse than rank 500 |
| most-truth-like-by-ω (0.11°, q_a 18.6°) | RMSE rank 31,515 / 79,443 |

Full-LC scoring cost **~80 ms/candidate** (288s for 79k on Pool(24)) — ~8× the pre-run estimate of 10 ms.

# Why this matters
- **The full-LC RMSE ranking adds essentially zero discrimination over the prefilter.** Prefilter + full-LC-rank is NOT a working inversion on this seed.
- **Two diagnosed causes (category C, not a bug — truth floor 0.0102 confirms infra):** (a) survivors are outside the ~1° coherent ω-tube where the surrogate landscape funnels to truth (s003); even the best is 3.3°. (b) Pointwise RMSE is phase-fragile — a few-° ω error accumulates over ~1 polhode period, slides sharp features off-epoch, and the misaligned features blow RMSE to junk levels, so truth-near looks like junk.
- **Doubly motivates pool density** — needed not just for final accuracy but to get a candidate *inside* the coherent tube where any LC metric (and polish) functions.
- **Promotes a sharper score** — flat 500-point RMSE drowns the discriminating epochs and is phase-fragile; a sharp-epoch / σ-weighted (RF25-style) objective may discriminate where flat RMSE fails (→ s094 windowed test is the first probe).

# Numbers
- truth_rmse 0.0102; top1 rmse 0.5443 @ ω-dir 112.86° q_a 98.84°; truthlike_rank 31515/79443 (source: `results/s093/score.json`).
- top-{10,50,100,500} within-10° = {0,0,0,11}; baseline 2.46%.
- wall 288s Pool(24).

# Artefacts
- `results/s093/score.json` (top-10 table, truthlike rank, baseline).

# Out of scope
- Windowed scoring (between-anchor interval) — s094.
- Denser-pool re-run to isolate the coherent-tube factor from the metric factor.
- Sharp-epoch / σ-weighted objective.

# Cross-references
- `s092_cross_cloud_116.md` — the survivors scored here.
- `s003_landscape_vs_omega.md` — the ~1° coherent-tube result this corroborates.
- `s094_window_score.py` — windowed-score test (built, not yet run).
