---
title: "s063d — Jacobi polhode-identity filter retest on s055d pool (seed 14, ORACLE upper bound)"
type: experiment
sources:
  - lib/jacobi_propagator.py
  - results/s055d_cascade_pol_diam_filter/pool.npz
related:
  - project_jacobi_propagation_priority.md
  - project_polhode_prior.md
  - project_score_function_broken_const_omega.md
  - feedback_finite_diff_small_geodesic_noise.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (numerical filter result; the architectural conclusion follows directly)
---

# TL;DR

Replacing s055d's scalar pol_diam filter with the joint (|L|², 2T) Jacobi-exact filter is **not enough to rescue the cascade pool**. Best 2D enrichment is **2.13× at the 10% tube** (vs s055d's 1.24× / s063d's 1.11× for 1D pol_diam — both below the 3× operational gate). The reason is structural and visible in the data: the 30 truth-q_a candidates in the s049 cascade pool have ω spanning **|L|² ∈ [4.5e-2, 3.95×] of truth and 2T ∈ [4.7e-2, 3.62×] of truth** — about **two orders of magnitude in polhode-invariant space**. The cascade-pool ω-derivation (finite-diff on anchor-pair quaternions) produces ω vectors that share truth's *direction* (truth-q_a are q-close to truth_q_a) but with random *magnitude*. No polhode-identity filter can recover from this — **the underlying ω-derivation needs to be replaced, not its post-filter**. Validates the s061 reframe direction: ω must be an INPUT to candidate generation, not a finite-diff output.

# What

The s055d retest was originally framed as "v2 substrate may rescue the scalar pol_diam filter." Wrong framing — s055d already ran the pool with DOP853 (rtol=1e-8), so the pol_diam values themselves are not v1-noise-tainted. The right Jacobi-driven retest is:

1. For each of 39,591 surviving candidates in the s055d cached pool: compute the joint polhode invariants (|L|², 2T) analytically via Jacobi (zero propagation).
2. Compare 1D scalar pol_diam filter (replicates s055d ranking) to 2D (|L|², 2T) joint filter.
3. Use **truth-oracle** (filter against true (|L|², 2T)) — gives an upper bound for the polhode-identity-filter discrimination ratio. If oracle fails, LC-derived will fail.

# How

`experiments/s063d_polhode_identity_filter.py`. Pure linear algebra: `L = I·ω`, `|L|² = L·L`, `2T = ω·I·ω` — vectorized over the (39591, 3) survivor array. Wall <1 s. Filter sweeps: 1D pol_diam at {5, 10, 15, 25, 35, 50, 100}%; 2D (|L|², 2T) symmetric at {1, 2, 3, 5, 10, 15, 25, 50}%; 2D asymmetric at all (1, 2, 5)% × (1, 2, 5, 10)% combinations.

# Result

**Truth-q_a candidates' polhode-invariant spread (this is the key number):**

| Invariant | min | median | max | spread (max/min) |
|---|---:|---:|---:|---:|
| |L|² | 4.55% of truth | 0.55× truth | 3.95× truth | 87× |
| 2T | 4.74% of truth | 0.51× truth | 3.62× truth | 76× |

The 30 truth-q_a candidates do NOT cluster around truth's polhode in (|L|², 2T) — they span ~2 orders of magnitude.

**Filter enrichment comparison:**

| Filter | Best threshold | n_keep | n_truth | Enrichment |
|---|---|---:|---:|---:|
| 1D scalar pol_diam | 5% | 2373 | 2 | **1.11×** |
| 2D Jacobi (|L|², 2T) symmetric | 10% | 1863 | 3 | **2.13×** |
| 2D Jacobi asymmetric | (1%, 1%) | 15 | 0 | 0.00× (kills all truth) |

Tightening below 5% in 2D kills ALL truth-q_a candidates: their polhode invariants are typically >5% off from truth in at least one of (|L|², 2T). The truth-q_a candidates only show up in moderately-loose tubes, and at those tube sizes the random-q_a candidates also slip through in similar numbers.

Even the oracle upper bound stays **< 3× the operational gate**.

# Why this matters

This is a clean diagnostic outcome with two structural implications:

1. **s055d's 1.24× scalar pol_diam ceiling was not a sharpness problem.** The Jacobi reframe gives ~2× lift at best, still below operational. The bottleneck is **upstream** — the s049 cascade pool's ω-derivation. Each "truth-q_a" candidate is a q-tuple (q_a, q_b at +Δt) where q_a and q_b are both close to the corresponding truth quaternions, but the per-pair finite-diff ω that the cascade extracts has 1–2 orders of magnitude scatter even on those near-truth pairs. The earlier "17% magnitude error" estimate from s055d/memories is an average; the tail spans 100×.

2. **Validates the s061 + s062 reframe direction.** s061's design (constant-ω propagation per (q_a, ω) candidate where ω is an INPUT) eliminates this whole class of pathology — ω-noise can't propagate from finite-diff because there is no finite-diff. The Jacobi machinery makes this efficient. Once s062b lands closed-form q(t), polhode-basis sampling (s062d) at scale becomes the natural replacement for cascade-pool filtering, and the v1-substrate "revisit list" entries that depend on cascade-pool ω derivation can be dropped (s057f anchor-pair finite-diff, s060 multi-anchor Newton-shoot, the W-cascade idea — all share this failure mode).

3. **Negative result on a closed-form filter is informative.** This is the type of cheap experiment that closes a memory entry. The `project_v1_substrate_reframe.md` revisit list flagged s055d as v1-substrate-suspect; s063d removes it from the revisit list with a clean ORACLE-impossibility result. The cascade-pool architecture is *structurally* broken in a way no downstream filter can rescue. **Recommended memory update**: cross-out s055d retest in the revisit list; the v1-substrate caveat is real but doesn't apply to s055d's bottleneck.

# Numbers

(`results/s063d/summary.json`)

- N_survivors = 39,591; n_truth = 30; rate_before = 7.58e-4
- truth_qa_L2_relspread = 3.90; truth_qa_twoT_relspread = 3.57 (in units of truth invariant — see "max/min" in result table for the geometric spread)
- best_1d_enrichment = 1.112× at thr=5%
- best_2d_enrichment_symmetric = 2.125× at thr=10%
- best_2d_enrichment_asymmetric = 0 (tight asymmetric tubes kill all truth)
- truth |L|² = 4.36e+05, truth 2T = 12.5

# Out of scope

- **LC-derived (|L|², 2T)**. Doing this would require regressing two new LC features beyond pol_diam (s055a). Not run because the oracle upper bound (truth values) already fails the operational gate.
- **A non-cascade-pool test**. The actual win is in a pool built by polhode-basis sampling (s062d), not by retro-filtering the cascade. That's the architectural reframe, not an experiment for this session.
- **Cohort-scale retest**. Single-seed (seed 14) is enough to land the structural conclusion; the failure mode is generic to finite-diff anchor-pair ω-derivation, not seed-specific.

# Artefacts

- `experiments/s063d_polhode_identity_filter.py`
- `results/s063d/summary.json`
- `results/s063d/polhode_identity_filter.png`. **Saved: `notebooks/inversion/survey/results/s063d/polhode_identity_filter.png`**

# Cross-references

- `experiments/s055d_*.md` — original scalar pol_diam filter
- `feedback_finite_diff_small_geodesic_noise.md` — small-Δt anchor-pair geometric noise — s063d shows that even when the small-Δt geometric noise is excluded (s049 cascade uses fairly large Δt), the ω-magnitude spread is still 100× on truth-q_a candidates
- `project_v1_substrate_reframe.md` — revisit list; s063d removes s055d from it
- `experiments/s061{a,b,c}_*.md` — the reframe target (constant-ω per candidate eliminates finite-diff)
- `experiments/s062_jacobi.md` — the Jacobi machinery used here
