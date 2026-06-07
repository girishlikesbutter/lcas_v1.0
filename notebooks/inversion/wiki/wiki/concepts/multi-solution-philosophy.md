---
title: "Multi-Solution Philosophy"
type: concept
sources: ["raw/inversion_diagnostics/m103_hybrid/", "raw/inversion_diagnostics/m113_de_attitude/"]
related: ["[[candidate-selection]]", "[[twin-degeneracy]]", "[[de-attitude-search]]", "[[hi-fi-scoring]]", "[[omega-sign-degeneracy]]", "[[m126_wrapped_pipeline]]", "[[m127_flipped_omega_search]]"]
created: 2026-04-13
updated: 2026-04-17
confidence: high
---

# Multi-Solution Philosophy

## Core Principle

Light curve inversion is ill-posed. Multiple (q0, omega) pairs produce light curves that are observationally indistinguishable. **The goal is NOT to find the single true state — it is to find ALL valid states that fit the observations below a hi-fi residual threshold.**

A "valid solution" is any (q0, omega) pair whose hi-fi light curve MSE is below a threshold (e.g., MSE < 1.0). The algorithm should return the full set of valid solutions, not a single winner.

## Evidence: "Failures" That Are Actually Valid

m103 data proves that many seeds classified as FAIL actually found excellent solutions:

| Seed | "Winner" q0_err | Hi-fi MSE | Truth q0_err | Truth hi-fi MSE | Verdict |
|------|----------------|-----------|-------------|-----------------|---------|
| 14 | 178.4° | **0.249** | 8.1° | 0.489 | Winner BEATS truth on hi-fi |
| 24 | 179.2° | **0.123** | — | — | Winner has best hi-fi of all seeds |
| 75 | 12.6° | 0.575 | — | — | Other valid solutions exist at 0.386 |

Seed 14 is the clearest case: the q0=178° solution has **better** hi-fi fit (0.249) than the q0=8° truth-adjacent solution (0.489). The observations genuinely cannot distinguish these attitudes. Calling 178° a "failure" is incorrect — it's a valid observational degeneracy.

## What This Changes

### Evaluation Metric
- **Old:** q0_err < 5° AND w_dir_err < 5° AND |w_mag_err| < 5% → "OK"
- **New:** Does the candidate set contain at least one solution with hi-fi MSE < T? How many valid solutions exist? Is truth (or its twin) among them?

### Algorithm Design  
- **Old:** Find the single best (q0, omega). Optimizers (NM, DE, CMA-ES) converge to one minimum.
- **New:** Find ALL valid (q0, omega) pairs. Use multi-start optimization, MCMC posterior sampling, or non-social PSO (Burton 2024) that deliberately finds many local minima rather than converging to one.

### Pipeline Output
- **Old:** One winner with classification OK/PARTIAL/FAIL.
- **New:** A ranked candidate set with hi-fi residuals. Report how many solutions are below threshold. Flag which ones are near truth, near twin, or novel degeneracies.

### Candidate Selection
- **Old:** Select the single candidate with lowest hi-fi MSE.
- **New:** Keep ALL candidates below threshold. The "selection" problem disappears — we don't need to pick one winner, we need to enumerate the valid set.

## Implications for DE Attitude Search (m113)

m113 used DE to find one q0 per omega candidate. Under the new philosophy:
- Run DE from multiple starting points (or use BIPOP-CMA-ES restarts) to find ALL basins
- For each basin, record the (q0, omega) and its hi-fi MSE
- The output is a set of valid solutions, not a single winner
- Seeds like 14 and 24 where DE found q0≈180° are potentially finding valid alternative solutions — check their hi-fi MSE before declaring failure

## Threshold Selection

The hi-fi MSE threshold T needs to account for:
- Measurement noise (σ=0.05 mag → MSE_noise ≈ 0.0025 for 500 epochs)
- Model error (lo-fi vs hi-fi, BRDF parameter uncertainty)
- A reasonable threshold might be MSE < 1.0 (generous) or MSE < 0.5 (strict)
- This should be calibrated empirically: what MSE does truth achieve across the 100-seed population?

## Connection to ±X Twin Degeneracy

The ±X twin ([[twin-degeneracy]]) is just ONE specific degeneracy (180° about +X). The multi-solution philosophy recognises that there may be OTHER degeneracies — approximate symmetries in the satellite geometry, shadow patterns that happen to match at different attitudes, or dynamical coincidences where different (q0, omega) pairs produce similar attitude trajectories over the observation window.

The twin is the one we understood first. It is not necessarily the only one.

## New observational degeneracy: flipped-ω (2026-04-16)

[[m126_wrapped_pipeline]] found seed 33's DE basins at `(q0 ≈ 98°–178°, ω_retrograde, |ω| ≈ 1.00 × |ω_true|)` produce a hi-fi MSE of 0.082 — **below the PARTIAL-class threshold 0.1**. This is a valid solution under the multi-solution criterion, distinct from any known q0 symmetry. See [[omega-sign-degeneracy]] for the mechanism. (Corrected 2026-04-16: an earlier version here claimed `|ω| ≈ 0.30 × |ω_true|`. The actual magnitude error is −0.70% — ω is essentially the same magnitude as truth, just pointing backwards. The compensating rotation appears to be ~98° about the body +Z axis, combined with ω sign flip.)

Practical implication: pipelines reporting "seed 33 failed" are wrong. The pipeline correctly found a valid solution; it just isn't truth-adjacent. Output labelling should read `{"truth_adjacent": false, "twin_related": false, "flipped_omega": true, "hi_fi_mse": 0.082}` rather than `{"classification": "FAIL"}`. This is exactly the labelling reform this concept page was arguing for.

### Second confirmed flipped-ω case: seed 12 ([[m127_flipped_omega_search]], 2026-04-16)

[[m127_flipped_omega_search]] ran a 60k SO(3) + L-BFGS-B polish with `ω = −ω_true` across 11 baseline seeds. Headline: 1/10 non-control seed produced a WIDE flipped-ω attractor — seed 12 at `(q0_err 140.36°, −ω_true)`, **hi-fi MSE 0.171** (PARTIAL class, <0.5). Not a twin-relative of any seed-12 m115 basin; a **genuinely independent attractor**. Body-frame compensation axis `[−0.848, −0.485, +0.215]` — closest to body +X, NOT body +Z like seed 33. This means the flipped-ω degeneracy's compensation geometry is seed-specific in BOTH rotation angle AND body axis; it is not a single shared structural symmetry of IS-901.

**Per this concept's output-labelling reform**, seed 12's `(q0_err=140°, −ω_true, hi-fi=0.171)` should be surfaced as a valid alternative solution with `{"truth_adjacent": false, "flipped_omega": true, "basin_width": "wide"}`, alongside any truth-adjacent basins (from [[m126_wrapped_pipeline]] seed 12 has hi-fi 0.327 best, so the flipped-ω attractor is actually the BEST solution for seed 12 under the ω=−ω_true slice).

**Underpowered-negative caveat:** m127's positive control for seed 33 FAILED (known narrow basin invisible to 3° grid). So the 9 non-PARTIAL seeds in m127 are inconclusive — narrow flipped-ω attractors may still exist there. m128 (warm-started L-BFGS from DE basins with negated ω) is queued to resolve.

## Cohort-level solution count (post-Option-A, 2026-04-17)

[[m132_solution_count_by_band]] aggregated all 33 polished basins across the 11-seed post-fix cohort:

| band | count | % of 33 |
|---|---:|---:|
| `< 0.005` (≤2× noise floor) | 4 | 12% |
| `0.005 – 0.01` (OK) | 1 | 3% |
| `0.01 – 0.05` | 7 | 21% |
| `0.05 – 0.1` (PARTIAL) | 4 | 12% |
| `0.1 – 0.3` | 8 | 24% |
| `≥ 0.3` (FAIL) | 9 | 27% |

**Honest multi-solution yield: 16/33 basins below hi-fi 0.1 (48%).** 5/33 below 0.01. Winner-class reporting (5 OK / 4 PARTIAL / 2 FAIL) conceals that 14/33 basins are above 0.3 — for 6/11 seeds, 2/3 of the basins FAIL. The wrapper's OK-class winners emerge because some basin per seed happens to land in the right place; basin enumeration quality (upstream DE) is the bottleneck, not polish.

This is the concept page's labelling reform operationalised: instead of `{"classification": "OK"}` for the seed, the pipeline should surface `{"n_basins_below_0.01": k, "n_basins_below_0.1": m, "truth_adjacent": [list], "twin_related": [list], "flipped_omega": [list]}`. The 16/33-at-valid metric generalises across seeds without forcing a winner pick.
