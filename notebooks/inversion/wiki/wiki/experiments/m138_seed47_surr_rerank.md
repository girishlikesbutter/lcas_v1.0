---
title: "m138 seed 47 — surrogate-MSE re-rank bypasses H1 cost-shape pathology"
type: experiment
sources:
  - "notebooks/inversion/m138_rerank_harmdiv_surr.py"
  - "notebooks/inversion/m138_rerank_analyse.py"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/seed_047/h1_harmdiv/rerank_surr_result.json"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/seed_047/h1_harmdiv/rerank_surr_ckpt.npz"
related:
  - "[[m138_seed47_lombscargle_bracket]]"
  - "[[m138_isoshell_h1_pilot]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[alignment-cost]]"
  - "[[surrogate-rerank]]"
created: 2026-04-29
updated: 2026-04-29
confidence: high
---

# m138 seed 47 — surrogate-MSE re-rank closes the cost-shape blocker

#open — direct continuation of `[[m138_seed47_lombscargle_bracket]]` Finding B (cost-shape pathology). Hi-fi m115 verification still pending.

## Premise

`[[m138_seed47_lombscargle_bracket]]` (commit `91da4bf`) confirmed that the H1 epoch-density cost has a **cost-shape pathology**: at eps_cluster=10° on the harmonic-division grid, **12 joint truth-near candidates** (≤5° dir + ≤5% mag) exist in the 80k pool but the cost ranks the best one at position 22,231 / 80,000 (top 28%, far outside m115's top-30 hand-off). The cost rewards extreme-|ω| candidates at either end (slow-spin pile-up from cloud collapse, fast-spin pile-up from sweep) over truth's real 26-epoch convergence.

Round A here: substitute surrogate full-LC MSE for the H1 cost on the cached 80k pool. Mirrors `score_lofi_surrogate.py` exactly.

## Setup

- Pool: `seed_047/h1_harmdiv/isoshell_ckpt.npz` (80k candidates, omega_batch + q0_estimate + H1 cost).
- Cost: `surr_mse[i] = mean((surrogate.predict_magnitude(propagate(q0[i], ω[i])) - observed_lc)^2)` over the full 500-epoch window, masked to finite observations.
- No q0 polish (Round A). The saved q0_estimate is H1's densest-cluster centroid; for wrong-ω candidates this is at an accidentally-dense spot, but the comparison is fair across the pool.
- Pool(8) parallelisation, **single-thread BLAS per worker** (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`). Without single-thread BLAS the pool ran at 5 cand/s (4h ETA); with it, 85 cand/s (16 min wall).

## Verdict — Round A SUFFICES

| metric | H1 cost | Surrogate MSE re-rank |
|---|---:|---:|
| rank-1 ω-dir | 112.18° | **13.93°** |
| rank-1 ω-mag err | -82.2% | -3.08% |
| rank-1 q0_to_twin | (broken) | **26.4°** |
| best joint candidate rank | 22,231 / 80k | **4 / 80k** |
| n top-30 joint (5°+5%) | 0 | **1** |
| n top-30 within 5° dir | 0 | **1** |
| nearest 5°-dir candidate rank | 22,231 | 4 |

**Top-5 of the surrogate-MSE re-rank** (MSE / ω-dir / ω-mag / q0_to_twin):

| rank | MSE | ω-dir | ω-mag | q0_to_truth | q0_to_twin | H1 rank |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.703 | 13.9° | -3.1% | 175° | **26°** | 9874 |
| 2 | 0.732 | 160.9° | -3.1% | 60° | 167° | 17981 |
| 3 | 0.741 | 160.9° | -2.3% | 59° | 167° | 17982 |
| 4 | 0.829 | 100.7° | -17.9% | 140° | 136° | 27915 |
| 5 | **0.901** | **4.25°** | +4.6% | 161° | **30°** | 9496 |

Two viable candidates within top-5:
- **Rank 1**: twin-q0 (26.4° from twin), ω-dir off by 13.9° but ω-mag right. Borderline for m115's documented 5° dir / 5% mag joint envelope, but DE polish has shown bridging up to ~15° in past tests.
- **Rank 5**: joint truth-near (4.25° dir, +4.6% mag), twin-q0 (30°). Inside m115's 5°+5% envelope. Primary anchor.

Ranks 2-4 are anti-aligned (ω_dir 100-161°) accidentally low-MSE candidates — expected aliases.

## Cost-shape diagnostic

**Spearman ρ(H1 cost, surrogate MSE) = -0.2998** (p ≈ 0). Negative = anti-correlation. The H1 epoch-density cost ranks **anti-correlated with truth-fit quality** — high H1-cost candidates tend to have low surrogate MSE. This corroborates the cost-shape pathology mechanism: H1 rewards "high overlap" candidates that are *less* plausible LC-wise.

This rules in surrogate-MSE substitution as the right algorithmic fix and rules out cost-shape redesigns (volume-normalised, pile-up-tightness) as having less leverage than this single substitution.

## MSE distribution

Pool surrogate-MSE percentiles: min 0.70, p1 1.89, p5 2.42, p25 3.31, **median 4.10**, p75 5.22, max 48.65. Top-1 to median ratio is ~6×; not the orders-of-magnitude separation hoped for, but enough for top-5 to surface the joint truth-near candidate.

The pool floor (~0.7 mag²) is set by the noise distribution of "wrong-q0 with reasonable ω" candidates — these can accidentally fit the smooth/dim parts of the LC and miss the bright peaks. The **min** is meaningfully separated from the median by a factor of ~6×, which is what enables ranking.

## Why no q0 polish

The NEXT_SESSION_PROMPT flagged Round B (q0 polish, 30-iter L-BFGS-B over q0 with ω fixed) as a potential ~3× wall fallback if Round A failed. Result: not needed. The joint truth-near candidates have q0_estimate that is roughly random (q0_to_truth 47°-177°) but their ω is correct enough that surrogate MSE under the *wrong-q0-correct-ω* configuration is still substantially lower than *random-q0-random-ω*. The signal is there.

A `m138_rerank_harmdiv_surr_polish.py` template exists for Round B — kept on disk in case generalisation to other failure-cohort seeds requires it.

## Limitations

1. **m115 verification not run**. Top-5 contains two viable candidates but actual hi-fi MSE outcome (ρ < 1?) is the next step.
2. **Twin-only solutions in top-5**. Both viable candidates are q0_to_twin ≈ 30°, not q0_to_truth. For IS-901 this is fine (only ±X twin is valid), but worth noting.
3. **q0_estimate is unreliable for any rank**. Even joint truth-near candidates have q0_estimate scattered all over q0 space. m115 must re-optimise q0 from scratch — do not trust H1's saved q0 as a polish init.
4. **Single-seed result**. Pipeline must be tested on failure cohort {51, 79, 84, 89} (~16 min wall each) before generalisation can be claimed.
5. **Pool(8) BLAS oversubscription**: requires explicit single-thread BLAS env vars. Documented as a caveat.

## Next experiments

1. **m115 hi-fi DE polish on the top-30 of `rerank_surr_ckpt.npz`** (rank-1 + rank-5 are the primary anchors). Validates whether ρ < 1.
2. **H1 + harmdiv + surr-rerank on cohort {51, 79, 84, 89}** to test generalisation. Expect ~64 min total wall.
3. **(Conditional)** if either of the above shows q0_estimate-from-centroid is too unreliable to seed m115 cleanly, add Round B (q0 polish) to the standard pipeline.

## See also

- Concept: `[[surrogate-rerank]]` (the substitution pattern, applied here)
- Concept: `[[alignment-cost]]` (anti-correlation pattern documented; this is now a second instance — H1 cost is also anti-correlated with surrogate MSE)
- Branch: `[[surrogate-attitude-isoshell]]` — append to the 2026-04-28 disconnection note: surr-MSE substitution circumvents the cross-component-membership cost design entirely.
