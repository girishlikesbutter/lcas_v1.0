---
title: "Surrogate full-LC MSE re-rank"
type: concept
sources:
  - "notebooks/inversion/score_lofi_surrogate.py"
  - "notebooks/inversion/m138_rerank_harmdiv_surr.py"
related:
  - "[[surrogate-model]]"
  - "[[alignment-cost]]"
  - "[[m133_rerank_findings]]"
  - "[[m134_pipeline_test_q0polish]]"
  - "[[m135_lofi_surrogate_audit]]"
  - "[[m138_seed47_surr_rerank]]"
created: 2026-04-29
updated: 2026-04-29
confidence: high
---

# Surrogate full-LC MSE re-rank

Substitute "surrogate full-LC MSE vs observed" for whatever the upstream stage's grid cost was, on the cached candidate pool. Mirrors the pattern in `score_lofi_surrogate.py`.

## Recipe

```
For each cached (q0, ω) candidate i:
    quats   = propagate_attitude(q0[i], ω[i], obs_times, "tumbling", I_tensor)
    R(t)    = quat_to_rotmat(quats)
    k1(t)   = R(t) @ sun_dir_inertial(t)
    k2(t)   = R(t) @ obs_dir_inertial(t)
    pred_lc = surrogate.predict_magnitude(k1, k2, panel=0, dish=15, obs_dist_km)
    surr_mse[i] = mean_over_finite_epochs( (pred_lc - observed_lc)^2 )
re-rank ascending by surr_mse
```

Critical: pair with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` when running under `multiprocessing.Pool` — without single-thread BLAS, the surrogate's matmuls oversubscribe cores and parallel speedup collapses (16-17× slowdown observed on the m138 80k pool).

## When it surfaces truth

Two preconditions must hold for the re-rank to land truth in top-K:

1. **The pool must contain a truth-near candidate.** Re-ranking does not enlarge the search space — if the upstream grid missed truth (wrong |ω|-base, too-coarse direction sphere), no re-rank fixes it. Verify pool coverage first via `grid_n_joint_5deg_5pct` against a truth oracle.

2. **The surrogate-MSE noise floor on wrong-(q0,ω) candidates must be separable from the truth-near MSE.** Empirically:
   - Wrong-q0 + wrong-ω: MSE ~3-5 mag² (set by LC variance).
   - Wrong-q0 + correct-ω: MSE ~1-2 mag² (smooth/dim parts fit, bright peaks miss).
   - Correct-q0 + correct-ω (or ±X twin): MSE ~0.001-0.01 mag² (surrogate offset only).

The 6× separation at the wrong-q0+correct-ω vs wrong-everywhere boundary is what enables ranking when q0 is unreliable. When q0 is also approximately right, the separation is orders of magnitude.

## Documented anti-correlation pattern

Re-ranking has revealed two upstream costs that are **anti-correlated** with surrogate-MSE — i.e., the upstream cost ranks high candidates that surrogate-MSE ranks as among the worst fits:

- **[[alignment-cost]]** (m103 grid+NM step): cost-at-truth >> cost-at-noise on failure seeds (1.85–17 orders of magnitude). Documented in [[alignment-cost-anti-truth]].
- **H1 epoch-density cost** (m138): Spearman ρ(H1 cost, surr_MSE) = -0.30, p≈0 on the 80k seed-47 pool. Documented in [[m138_seed47_surr_rerank]].

Both costs reward "high overlap" or "high coverage" of geometric structure that does not correspond to truth-fit quality. The recurring lesson: when an upstream cost is structurally noise-attracted (min-over-anchor with large anchor sets, density estimation in collapsed q-space, etc.), surrogate-MSE substitution is the most leverage per line of code among the available fixes.

## Variants

- **Round A (cheap, default)**: score the cached (q0, ω) directly. Use when q0 from the upstream stage is "approximately good enough" (m103's NM polish q0, or H1's centroid q0 for truth-near ω).
- **Round B (q0 polish)**: for each candidate ω, run a brief L-BFGS-B over q0 with ω fixed before scoring. ~3× wall. Use only if Round A fails to surface truth-near candidates AND the upstream q0 is suspect (e.g., H1's densest-cluster centroid for non-truth-ω candidates lands at random spots).
- **Round C (joint q0+ω polish)**: full m115-style DE per candidate. Effectively becomes the next pipeline stage rather than a re-rank.

## Compute envelope

- Per-candidate cost: ~25 ms wall under Pool(8) + BLAS=1 (m138 measurement). 80k candidates → 16 min wall. 300 candidates (m103 lofi pool) → ~5 sec.
- Memory: each Pool worker loads a fresh SurrogateModel ensemble (~MB). 8 workers → ~tens of MB total. Negligible.
- Fits comfortably inside a research-loop session even at the 80k scale.

## Caveats

- **Pool coverage matters more than re-rank quality.** A re-rank that lands the best-pool candidate at rank-1 still fails if pool coverage is ≥10° from truth. Always report `grid_n_joint_5deg_5pct` alongside re-rank ranks.
- **q0_estimate-from-upstream is unreliable.** Even joint truth-near candidates may have q0 scattered widely (m138 case: q0_to_truth 47°-177° across the 12 joint candidates). Downstream m115 must re-optimise q0 from scratch — do NOT use the upstream q0 as a polish init for non-rank-1 candidates.
- **Twin solutions surface naturally.** ±X twin candidates of IS-901 give comparable surrogate-MSE to truth (correctly, since they're observationally indistinguishable). Top-K typically contains both truth and twin partners, plus ω-sign-flipped aliases.
