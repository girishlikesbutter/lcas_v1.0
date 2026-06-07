---
title: "m133 — Rerank m103 geo_ckpt pool with surrogate-LC costs"
type: experiment
sources: ["notebooks/inversion/14_rerank_experiment/", "data/results/inversion_diagnostics/rerank_experiment/FINDINGS.md"]
related: ["[[m103_hybrid]]", "[[m115_surrogate_pipeline]]", "[[m132_solution_count_by_band]]", "[[surrogate-model]]", "[[upstream-redesign-6dof-surrogate-de]]"]
created: 2026-04-22
updated: 2026-04-22
confidence: high
related_followup: "[[m134_pipeline_test_q0polish]]"
---

# m133 — Rerank the m103 geo_ckpt pool

Offline rerank experiment on saved `m103_hybrid_m048/seed_XXX/geo_ckpt.npz` pools (26 candidates per seed, 22 seeds reachable). Goal: find a non-oracle cost that ranks the truth-closest ω candidate into the top-3 more reliably than `geo_cost` does.

Motivation in [[m048-random-cohort-baseline]]: 12 of 17 rescuable Band-D seeds are "m103 top-K ranking" failures where the full 26-candidate pool HAS a near-truth ω but `geo_cost`'s top-3 drops it.

## Setup

Per seed, for each of 26 `(q0_ref, w0_ref)` candidates, propagate with tumbling ODE, predict LC via surrogate (`~/surrogate_model/surrogate_model/`, panel=0°/dish=15°), score with a catalog of LC-based costs. Oracle `w0_ref_errs` used for evaluation ONLY — never inside the cost.

"Rescuable" = pool contains a candidate with `w0_ref_err < 20°`. m048 cohort: 17/22. m046 cross-validation: 13/13.

## Cost catalog

| cost | definition |
|------|------------|
| `surr_mse` | Plain LC MSE (full 500 epochs) |
| `surr_mae` | Mean absolute error |
| `surr_bright_mse` | MSE restricted to epochs with `observed_mag < 10` |
| `surr_peak_mse` | MSE at observed-peak epochs ±2 |
| `surr_peak_cnt` | `n_observed_peaks - n_matched` (m103 Step-2b cost on surrogate) |
| `surr_peak_time` | RMS peak-timing error (for each observed peak, distance to nearest predicted peak) |
| `surr_peak_xcorr` | `-max_lag cross-correlation of peak trains in [-30, 30]` |
| `surr_detrend_mse` | MSE after subtracting per-LC means |
| `surr_deriv_mse` | MSE of first derivatives |
| `surr_spectrum` | Normalized FFT power spectrum RMSE |
| `surr_envelope` | Segment-wise (min,max) envelope RMSE over 20 windows |
| `surr_autocorr` | Autocorrelation profile RMSE over lags 0..100 |
| `surr_peak_sig` | Sorted peak-brightness distribution RMSE |
| `surr_q0marg_mse` | Best MSE across 20 random q0 perturbations (σ=15°) + ±X twin |
| `surr_q0marg_bright_mse` | Same, restricted to bright epochs |
| `surr_q0polish_mse` | **Best MSE after 4-restart Nelder-Mead polish of q0** (original, ±X twin, 2 randomized antipodal bases) |

Plus `geo_cost` baseline and several rank-sum composites.

## Key m048 results (17 rescuable of 22)

| cost | top-1 | top-3 | top-5 | mean rank |
|------|:---:|:---:|:---:|:---:|
| **surr_q0polish_mse** | **9/17** | 10/17 | 10/17 | **7.94** |
| `surr_bright_mse` | 7/17 | 10/17 | 11/17 | 8.29 |
| `surr_autocorr` | 7/17 | 10/17 | 10/17 | 8.35 |
| `surr_peak_time` | 4/17 | 9/17 | 11/17 | 8.76 |
| `geo_cost` (baseline) | 5/17 | 9/17 | 10/17 | 9.82 |

Single best (q0_polish_mse) gets 1 more top-3 than `geo_cost`. The real leverage is in **union-of-top-K across multiple costs** because the costs are complementary — different seeds are best-ranked by different costs.

### Greedy set cover (K=3 per cost)

| step | top-3-union hit | avg union size |
|------|:---:|:---:|
| `surr_q0polish_mse` | 10/17 | 3 |
| + `surr_peak_time` | 13/17 | 5.6 |
| + `surr_autocorr` (best triple) | **15/17 (88%)** | 7.4 |
| + `surr_q0marg_bright_mse` (best quadruple) | **16/17 (94%)** | 8.8 |

### K=5 quintuple hits the oracle ceiling

`{surr_peak_time, surr_autocorr, surr_q0polish_mse, surr_peak_sig, surr_peak_mse}` → **17/17** (100% of rescuable seeds) with avg union 15.4 candidates.

### Why q0_polish specifically matters

Pre-q0_polish ceiling was 15/17. The 2 irreducible seeds (64, 99) had truth-closest candidates with `q0_err > 145°` paired with them — `geo_cost` glint-polished q0 for the brightness geometry, not for LC. Random q0 perturbations (σ=15°) couldn't bridge this gap; Nelder-Mead polish with a ±X-twin base restart could. With q0_polish the ceiling becomes 16/17 (and 17/17 with K=5 ensemble).

## m046 cross-validation (13 seeds with geo_ckpt, all rescuable)

m046 is kinder (13/13 rescuable vs m048's 17/22 because m046 phase-angle window is less extreme). Cross-val confirms the **direction** generalizes — every surrogate cost beats `geo_cost` — but the specific best cost changes:

| cost | m046 top-3 | m048 top-3 | robust? |
|------|:---:|:---:|:---:|
| `surr_autocorr` | 11/13 | 10/17 | **yes** |
| `surr_spectrum` | 11/13 | 9/17 | m046-strong |
| `surr_detrend_mse` | 11/13 | 9/17 | m046-strong |
| `surr_mse` | 11/13 | 9/17 | m046-strong |
| `surr_envelope` | 11/13 | 8/17 | m046-strong |
| `surr_bright_mse` | 9/13 | 10/17 | acceptable-both |
| `surr_peak_time` | 7/13 | 9/17 | **m048-overfit** |
| `geo_cost` (baseline) | 7/13 | 9/17 | tied-baseline |

### Cross-validated recommended triple

```
union(
    top-3 by surr_autocorr,
    top-3 by surr_q0polish_mse,
    top-3 by surr_spectrum,
)
```

- m048 expected: ≥15/17 (swapping peak_time→spectrum doesn't hurt m048 and gains 4 on m046)
- m046 expected: ≥12/13

## Pipeline intervention

Replace m103's "top-3 by `geo_cost`" handoff to m115 with the union above.

- Cost: compute 3 surrogate LC predictions per candidate × 26 candidates = ~3 s per seed, plus ~2 min/seed for q0_polish (4-restart NM × 26 cands, Pool(8)).
- Effect: m115 receives ~7 ω candidates instead of 3 → ~2.3× DE wall-clock.
- Expected rescue: **9/17 → ~15/17** on m048 random cohort (53% → 88%). Still doesn't help the 5 sampling-miss seeds (47, 51, 79, 84, 89) where the pool itself has no truth-close ω.

## Implications

1. **m103's candidate pool is much better than its ranking.** On m048, truth-close ω exists in the full 26-candidate pool for 12 of the 17 rescuable seeds but is buried by `geo_cost` ranking. Replacing the last-step ranker is the cheapest leverage in the pipeline.
2. **No single cost dominates.** Union of 3+ costs is needed for near-oracle performance. This motivates either (a) passing more candidates to m115, or (b) a learned/ensembled cost that consolidates the costs into one scalar.
3. **q0 polish per candidate is required to crack the hardest cases** where m103's q0 is glint-adversarial to LC fitting.
4. **Next frontier is m103 SAMPLING** — the irreducible 5 seeds whose pool simply doesn't contain a near-truth ω need a different candidate generator (denser grid, surrogate-guided sampling, or [[upstream-redesign-6dof-surrogate-de]]).

## Artifacts

- Scripts: `notebooks/inversion/14_rerank_experiment/`
- m048 results: `data/results/inversion_diagnostics/rerank_experiment/FINDINGS.md` + `rerank_results.json` + per-seed JSON/NPZ
- m046 results: `data/results/inversion_diagnostics/rerank_experiment_m046/`
- Dissection script: `notebooks/inversion/dissect_m048_25seed.py`, `notebooks/inversion/dissect_m048_25seed_by_band.py`
- 25-seed dissection report: `data/results/inversion_diagnostics/m048_25seed_dissection_2026_04_22.md`

## End-to-end validation (m134, 2026-04-22 afternoon)

First actual pipeline rerun of the finding. Seeds 59, 64, 67 picked via `pick_target_seeds.py` as the 3 Band-D seeds where `surr_q0polish_mse` *uniquely* introduces a truth-close ω (<20°) into the top-3 that `geo_cost` had missed. Patched m115 with `M115_SORT_BY=surr_q0polish_mse`, cleared baseline checkpoints, rerun through m115 + m126 + wrappedbest.

ρ-band results (ρ = √hifi/0.05):

| seed | baseline | new |
|:---:|:---:|:---:|
| 59 | ρ=27.63 D | **ρ=1.02 A** (q0=0.21°, w_dir=0.05°, w_mag=−0.00%) |
| 64 | ρ=30.15 D | ρ=26.92 D (w_dir=50.9°, w_mag=−64%; wrong-ω winner) |
| 67 | ρ=33.09 D | **ρ=3.56 B** (q0=179.6°, w_dir=0.3°, w_mag=−0.08%; twin-type) |

**Takeaway:** ranking fix is necessary but not sufficient. Seed 59 cleanly rescued (D→A). Seed 67 recovered ω perfectly but landed at q0_err=179.6° (likely geometric twin; LC still distinguishable at 3.6σ). Seed 64 had truth-close ω=14.63° placed correctly at slot 1, but m115's 10-start DE couldn't bridge that ω-error to locate truth-adjacent q0 (0/10 starts).

**Implied m115 ω-bridging radius:** ~3-5°, not 15°. The single-cost `surr_q0polish_mse` expected to gain +1 rescue over baseline on m048 — matches observation (1 clean Band A rescue). Fuller 15/17 requires K=3 triple, still untested. See [[m134_pipeline_test_q0polish]].

## Caveats

- ~~Scored on saved checkpoints only; no m115/m126 rerun done yet. Pipeline impact is predicted, not measured. End-to-end validation is the natural next step.~~ Superseded 2026-04-22: single-cost end-to-end tested on 3 seeds (above).
- m046 geo_ckpts are from 2026-04-17 (oracle-ω bug era at m115, but m103 itself was unaffected — only m115's sort consumed the wrong field). Confirmed via file-layout identical to m048. Cross-val result valid.
- All scoring uses fixed articulation (panel=0°, dish=15°) matching `setup_experiment` defaults. Tumbling-mode propagator with the m048 shared inertia tensor.
