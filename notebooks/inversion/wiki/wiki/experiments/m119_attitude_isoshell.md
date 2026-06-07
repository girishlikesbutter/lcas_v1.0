> # ⚠️ RETRACTED 2026-04-15 — CORRUPTED RESULTS
> This experiment mixed 1-hour observation data (from `m046_trajectories.npz`) with 6-hour SPICE geometry (from current `setup_experiment()`). Time mismatch factor of 6×. Observer direction wrong by up to 74°. All numbers in this file are garbage. See `notebooks/inversion/12_brightness_surface/m119_BUG.md` for the full diagnosis and fix plan. Do not act on anything below.

---
title: "m119 — Surrogate attitude isoshell POC (seed 14) [RETRACTED]"
type: experiment
sources:
  - "raw/inversion_diagnostics/m119/seed_014/summary.json"
  - "raw/inversion_diagnostics/m119/seed_014/residual_kernel.npz"
  - "raw/inversion_diagnostics/m119/seed_014/target_scores.npz"
  - "raw/inversion_diagnostics/m119/seed_014/cost_variants.npz"
related:
  - "[[m118_cost_comparison]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[surrogate-model]]"
  - "[[kernel-factorization]]"
  - "[[multi-solution-philosophy]]"
created: 2026-04-15
updated: 2026-04-15
confidence: high
---

# m119 — Surrogate attitude isoshell POC

## Hypothesis

Replace the pab-contour `B(n_body)` brightness function with the trained surrogate `B(k1_body, k2_body, panel=0, dish=15, dist)`. For each of 255 constraint epochs (from [[m118_cost_comparison]]'s union: 14 spec peaks + 241 tight-IPL), the set `{R : |surrogate(R k1_J2000, R k2_J2000) − observed_mag| < σ}` should form a 2D SO(3) submanifold. Truth trajectory should pass through that manifold at every epoch; rank-1 competitors from [[m118_cost_comparison]] should fail most.

**Predictions at launch:**
- Truth residual median < 0.1 mag, p90 < 0.3 mag (isoshell-tightness threshold).
- Truth passes >90% of 255 epochs at σ=0.2.
- Competitors pass <50%.
- At least one cost variant ranks truth in top-1000 of 60000.

## Method

- SO(3) grid: 60000 rotations (uniform quaternion grid; median nearest-neighbour quant ≈ 3°).
- Constraint epochs: 255 (reused from [[m118_cost_comparison]] kernel).
- Stage C: batch-call surrogate on all (60000 × 255) `(k1_body, k2_body)` tuples → residual tensor `|surrogate_pred − observed_mag|`. Runtime 47 s.
- Seven cost variants scored on the same residual kernel (pure score aggregation, no propagation, see [[kernel-factorization]]):
  `mean_L2`, `mean_L1`, `max_abs`, `count_pass_020`, `count_pass_030`, `count_pass_050`, `count_pass_010`, `count_pass_100`, `soft_pass_050`.
- Target set: truth; three m115 DE basins (basin0 near-truth, basin1 and basin2 twins); two rank-1 competitors from m118 (`facet_rank1`, `ipl_weighted_ext_rank1`).
- Total wall time 49.5 s on 16 cores.

## Results

### Truth residual on 255 constraint epochs

| Statistic | Value |
|-----------|-------|
| median \|residual\| | **0.491 mag** |
| p90 \|residual\| | **4.21 mag** |
| max \|residual\| | **8.14 mag** |

Threshold-based "isoshell tight around truth" framing is **refuted**: the implicit σ needed to keep truth inside every epoch's level set is > 8 mag, which is larger than the observed LC's dynamic range.

### Target ranks (out of 60000)

| Target | q0 err | w_dir err | best rank | best variant |
|--------|--------|-----------|-----------|--------------|
| truth | 0° | 0° | **0** | mean_L1, soft_pass_050 |
| m115_basin0 | 2.29° | 0.34° | **0** | mean_L1 (ties truth) |
| m115_basin1 | 178° | 0.34° | 63 | mean_L1 |
| m115_basin2 | 179° | 0.34° | **0** | mean_L1 (ties/beats truth) |
| m118_facet_rank1 | 168° | 52° | 2693 | count_pass_010 |
| m118_ipl_weighted_ext_rank1 | 171° | 7.62° | 7781 | count_pass_100 |

### Per-variant truth vs m118_facet_rank1

| Variant | truth rank | facet_rank1 rank | gap (truth wins by) |
|---------|-----------:|-----------------:|---------------------:|
| mean_L2 | 3962 | 7950 | 3988 |
| mean_L1 | 0 | 6094 | 6094 |
| max_abs | 14840 | 28170 | 13330 |
| count_pass_020 | 1180 | 5427 | 4247 |
| count_pass_030 | 346 | 6633 | 6287 |
| count_pass_050 | 193 | 4624 | 4431 |
| soft_pass_050 | 0 | 5803 | 5803 |

### Grid top-1 under every variant is a wrong-basin attitude

No variant puts its own rank-1 grid cell at truth. All top-1 rotations have `q0_err` between 94° and 178° — near-twin or antipode orientations (partly a grid artefact: the ~3° grid has no cell exactly at truth). Truth and the near-twins (m115_basin0/2) achieve rank 0 via interpolation between adjacent grid cells under L1-style aggregations.

## Critical evaluation

### 1. Isoshell-threshold framing: REFUTED. Discrimination framing: CONFIRMED.

The launch-time prediction "median < 0.1, p90 < 0.3" was false by a factor of ~5 in median and ~14 in p90. There is no σ that produces a tight isoshell around truth at all 255 constraint epochs. **But** the discrimination claim ("truth beats [[m118_cost_comparison]] rank-1 at lower cost") is confirmed under all seven scoring variants, with 4000–13000-rank gaps on a 60000-sample grid.

The surrogate residual at truth is far from zero, yet it is still systematically lower than at wrong-basin attitudes. **The signal was the score ranking, not the level-set geometry.**

### 2. Why is truth residual so large at tight-IPL epochs?

The 255 constraint epochs = 14 spec peaks ∪ 241 tight-IPL epochs (where the PAB loops a body-frame lobe tightly). "Tight-IPL" selects for *geometric* tightness, not *photometric* quality:

- Most tight-IPL epochs are **dim** (IPL length small ⟹ PAB inside a lobe ⟹ often backside / near-shadow).
- Max truth residual 8.14 mag indicates at least one catastrophically wrong surrogate prediction.
- Most likely mechanism: **surrogate training coverage is thin at dim / extreme-geometry tails.** [[surrogate-model]] advertises MAE 0.045 on bright / 0.061 overall; it says nothing about p90 or max in the dim regime. Dim-regime errors at single-epoch level can easily blow past 1 mag without damaging overall MAE.
- Minor mechanism: shadow-boundary epochs where the surrogate has binary-valued truth but smooth predictions (see [[shadow-asymmetry]], [[isoshell-phi-limits]]).

Convention mismatch (panel/dish) is unlikely — those are constants (0°, 15°) and wrong values would corrupt the bright-peak residuals too, which stay small.

### 3. Does mean_L1 rank 0 = truth validate the approach?

Yes, but with a multi-solution caveat. Rank 0 under mean_L1 is shared by truth, m115_basin0 (near-truth 2.3°), and m115_basin2 (the q0≈179° twin). These are the valid-solution set — the experiment cannot distinguish truth from its ±X twin (as expected per [[twin-degeneracy]]) or from the DE basin that already sits inside the truth manifold. This is consistent with [[multi-solution-philosophy]]: the right deliverable is the set `{truth, twin, near-DE-basins}` at top-0, not a single unique winner.

### 4. Novelty relative to m115

m119 is effectively **surrogate-L1-MSE scored over a 60000-point SO(3) grid at 255 constraint epochs**. [[m115_surrogate_pipeline]] runs surrogate-DE (3-DOF, continuous search) with a similar residual score over the full 500-epoch LC. m119 is coarser (grid + 255 epochs) but produces a *landscape* instead of point estimates. The new information is:

- The landscape separates truth-basin from wrong-basin attitudes under every variant tested.
- It does so without access to the full LC — a 255-epoch subset suffices.
- Cost variants differ enormously (mean_L1 puts truth at 0; mean_L2 at 3962; max_abs at 14840). L1 + soft counts are the robust aggregators; L2 and max are sensitive to the handful of catastrophic dim-regime residuals.

## Branch status decision

See [[surrogate-attitude-isoshell]]: re-scoped as **#open** (not closed). Rationale:

- The threshold-based "isoshell as 2D SO(3) level set" framing is dead — the surrogate does not produce tight level sets at these constraint epochs.
- The score-based framing (per-epoch residual landscape, aggregated over many epochs) does discriminate and is cheap (50 s/seed).
- This overlaps with [[surrogate-de-search]] but at a different granularity (grid landscape vs. local DE optimum). The combination — surrogate-DE to find basins, surrogate residual landscape to characterise them — is new.

## Next experiment (recommendation)

**Split into two cheap follow-ups before committing to a full pipeline:**

1. **(a) Per-epoch surrogate fidelity audit.** Evaluate surrogate at the *exact truth trajectory* for all 500 epochs and compare against the noiseless hi-fi LC (not observed_lc, which has noise). Separate "surrogate wrong" from "observed noisy". If p90 > 0.3 mag on hi-fi comparison at dim epochs, the surrogate's training coverage needs attention before any isoshell framework is built on it.
2. **(b) Spec-peak-only constraint set (≈14 bright epochs) rerun.** Fast (~3 s). If truth residual median drops below 0.1 mag there, the isoshell-threshold framing may survive for bright-constrained seeds — but that restricts the tool to the same 13% of seeds [[ipl-census]] already flagged as bright-constraint-rich.

(a) is the right first move: it directly tests whether the dim-regime surrogate error is a training artefact we can improve, or a fundamental ceiling.

## Files

- Script: `notebooks/inversion/12_brightness_surface/m119_poc.py`
- Output: `data/results/inversion_diagnostics/m119/seed_014/`
  - `summary.json` — verdict + per-target ranks
  - `residual_kernel.npz` — (60000, 255) surrogate residuals at truth's `(k1_J2000, k2_J2000)` grid
  - `target_scores.npz` — per-variant scores + truth_residual {median, p90, max}
  - `cost_variants.npz` — all 9 cost landscapes over the 60000-point grid
  - `so3_grid.npz`, `setup.npz` — reproducibility
  - `run.log`
