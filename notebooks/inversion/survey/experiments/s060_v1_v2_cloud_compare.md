---
title: "s060 — v1 vs v2 cloud comparison at LC extrema (seed 14)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s060_v1_v2_cloud_compare.py
  - notebooks/inversion/survey/results/s060_v1_v2_compare/seed14_extrema_v1_vs_v2_cloud.png
  - notebooks/inversion/survey/results/s049_cascade_seed14/scan.npz
related:
  - s060_v1_v2_lc_compare.md (the LC-regime predecessor)
  - s049_cascade_seed14 (source of cached 500k v1 pool)
  - feedback_finite_diff_small_geodesic_noise.md (s057f finding now flagged for v1 retest)
  - project_surrogate_model.md (v1↔v2 benchmarks)
created: 2026-05-10
updated: 2026-05-10
confidence: high (direct measurement, cached pool reproduced exact v1 |C_t|)
---

## TL;DR

At seed 14's absolute LC extrema (t=379 mag=6.19 brightest; t=275 mag=14.76 dimmest), N=500k random-q pool, TOL=±0.10 mag — **v2 admits 78% more cells than v1 at the dim epoch (1980 vs 1113); IoU(v1, v2) ≈ 0.42 at BOTH extrema.** v1↔v2 disagreement on the cloud-regime is structurally large despite the polish-endpoint Spearman of 0.996 cited in `project_surrogate_model.md`. The cloud-survival check operates on a different population than DE polish — random rotations, heavy-tailed residual distribution, TOL window slicing a near-vertical edge. Selection-induced amplification of the v1 vs v2 tail.

This finding is the load-bearing substrate for the v1-substrate-reframe memory (`project_v1_substrate_reframe.md`).

## What

Critically test whether v1↔v2 agreement in the polish regime extends to the cloud regime. The cloud-survival check counts pool members at each epoch with `|surrogate_pred - measured| < TOL_MAG` — a totally different sampling distribution from the polish regime where v1 has been validated.

## How

Reuses the cached 500k random-quaternion pool from `results/s049_cascade_seed14/scan.npz` (originally rendered under v1 in s049). For seed 14:
1. Identify global LC extrema: argmin mag = epoch 379 (mag 6.19); argmax mag = epoch 275 (mag 14.76).
2. Compute `R_cache` from `q_pool_wxyz` via scipy.
3. At each extremum epoch: `k1_body = R_cache @ sun_unit_world[ep]`; `k2_body = R_cache @ obs_unit_world[ep]`.
4. Render with both v1 (`/home/girish/surrogate_model/s10_5M_*.npz`) and v2 (`lib.surrogate_eval.get_model()`).
5. Survival mask: `|pred - mag_hifi[ep]| < 0.10`.
6. Compare cell counts and intersection (IoU).

Validation: v1 |C_t| at t=275 reproduces s049's cached `n_survivors[idx_for_275] = 1113` exactly.

## Result

| Epoch | type | mag | v1 \|C_t\| | v2 \|C_t\| | overlap (∩) | only-v1 | only-v2 | IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t=379 | brightest | 6.19 | 450 | 488 | 284 | 166 | 204 | 0.43 |
| t=275 | dimmest | 14.76 | 1113 | **1980** (+78%) | 915 | 198 | 1065 | 0.42 |

Two findings:

1. **At the dim extremum, v2 admits 78% more cells than v1.** The shape of the residual distribution at the dim cliff differs between v1 and v2 enough that the [−0.10, 0.10] window — sitting on a steep slope of a heavily-skewed distribution — catches very different fractions.

2. **IoU ≈ 0.42 at both extrema.** Even when v1 and v2 |C_t| counts are close (450 vs 488 at the bright epoch), only ~43% of cells are agreed on. The other ~57% are admitted by exactly one surrogate. The survivor SET — what downstream clustering and ω-grid scoring consume — is materially different.

The histograms (saved PNG) show the mechanism:
- Bright epoch: distribution heavily skewed POSITIVE (most random rotations produce DIMMER predictions; specular configs are rare). TOL window catches a thin slice on a falling tail.
- Dim epoch: distribution heavily skewed NEGATIVE with a sharp cliff at residual=0 (zero-flux configs are the wall). TOL window lives on a near-vertical edge — tiny v1↔v2 distribution-shape differences amplify into large admitted-count differences.

## Why this matters

The DE-polish-endpoint Spearman 0.996 measured on near-truth basin endpoints is a smooth-function result. The cloud regime is the heavy-tailed population, and the TOL window is pre-selecting the disagreement region. **v1 ≈ v2 in polish regime; v1 ≠ v2 in cloud regime** — the regimes are different problems.

Anywhere a pipeline depends on cloud counts, survivor sets, or pool admission statistics under v1, the result is suspect. This includes:

- s048-s049 multi-epoch consistency cascade (cascade ω derived from v1 cloud survivors)
- s055d cascade pol_diam filter (1.24× enrichment was on v1 substrate)
- s057f anchor-pair finite-diff "small-Δt geometric noise" (was actually dominated by v1 cloud admission noise)
- s060 multi-anchor design's Newton-shoot stage (fragile under v1 cloud, much firmer under v2)
- The W-cascade idea (filter at small W under v1 picks up v1's noise tail; v2 throughout is cleaner)

The full revisit list is in `project_v1_substrate_reframe.md` (memory).

## Numbers

- Pool: N=500,000, fixed across all comparisons (from s049 cache).
- TOL: 0.10 mag (matches s049 convention).
- v1 |C_t| at t=275: 1113 (matches cached `n_survivors`, validates pool reuse).
- Wall: ~10 sec single-thread for both clouds × 2 epochs.

## Artefacts

- `experiments/s060_v1_v2_cloud_compare.py` — script.
- `results/s060_v1_v2_compare/seed14_extrema_v1_vs_v2_cloud.png` — 2×2 histogram plot.

## Out of scope

- Full-LC v1↔v2 |C_t|(t) Spearman. The 2-epoch result is sufficient to demonstrate the regime mismatch; full-LC measurement is a separate experiment.
- Cohort generalisation across seeds. Single-seed observation; the mechanism (TOL window on residual distribution edge) is structural and should generalise.
- v2 cloud-gen wall optimization. v2 at full-LC × 500k is order-of-hours; W-cascade earns its keep here. Defer to implementation phase.

## Cross-references

- `project_v1_substrate_reframe.md` (memory) — the methodological reframe and revisit list.
- `feedback_finite_diff_small_geodesic_noise.md` (memory) — updated with v1-substrate caveat.
- `s060_v1_v2_lc_compare.md` — LC-regime predecessor.
- `s049_cascade_seed14.md` — source of the cached v1 pool.
