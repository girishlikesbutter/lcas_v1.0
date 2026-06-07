---
title: "m129 — Densified SO(3) grid on seed 33 (flipped-ω density test)"
type: experiment
sources:
  - "raw/inversion_diagnostics/m129_densegrid/batch_summary.json"
  - "raw/inversion_diagnostics/m129_densegrid/seed_033/result.json"
  - "raw/inversion_diagnostics/m129_densegrid/seed_033/stage_a_grid.npz"
  - "raw/inversion_diagnostics/m129_densegrid/seed_033/stage_b_polish.npz"
  - "raw/inversion_diagnostics/m129_densegrid/seed_033/stage_c_hifi.npz"
related:
  - "[[m126_wrapped_pipeline]]"
  - "[[m127_flipped_omega_search]]"
  - "[[m128_warmstart_polish]]"
  - "[[omega-sign-degeneracy]]"
  - "[[basin-of-attraction]]"
  - "[[gradient-based-inversion]]"
created: 2026-04-16
updated: 2026-04-16
confidence: high
---

# m129 — Densified SO(3) grid on seed 33 (flipped-ω density test)

## Status: REFUTED — pure grid-density does NOT recover seed 33's narrow flipped-ω basin

## Hypothesis (falsifiable)

A 600k super-Fibonacci SO(3) grid (~1.4° median geodesic spacing, **10× denser** than [[m127_flipped_omega_search]]'s 60k ~3° grid) + top-50 L-BFGS-B polish (ω fixed at `−ω_true`) on seed 33 alone recovers its known flipped-ω basin (from [[m126_wrapped_pipeline]], hi-fi MSE ≈ 0.082, q0 compensation 98.5° about body +Z axis).

- **CONFIRMED iff** winner hi-fi < 0.15 → pure grid-density is the fix; [[m127_flipped_omega_search]]'s REFUTED verdict is retractable on wider grids; pivot to m130 DE for population enumeration.
- **REFUTED otherwise** → narrow basins are NOT grid-density-recoverable at 600k (and probably at any feasible grid density); forces pivot to population-based (DE) enumeration — m130 becomes required, not optional.

## Method (3-stage, seed-33-only)

Script: `notebooks/inversion/12_brightness_surface/m129_dense_grid_eval.py` (committed; modifies the [[m127_flipped_omega_search]] `m127_flipped_omega_search.py` template to N_SO3=600000, TOP_K=50, per-seed pool collapsed for the single-seed case).

1. **Load truth + setup.** `experiment_setup` at `2020-02-05T10:00:00 → 11:00:00`, 500 constraint epochs, `σ_obs = 0.05 mag`. Negate truth ω → `ω_search = −ω_truth`, magnitude preserved to machine precision (`ω_search = [+0.0150, −0.0176, +0.0103]` rad/s, truth ω flipped).
2. **Stage A — dense SO(3) grid.** 600 000 super-Fibonacci quaternions on S³ (median nearest-neighbour ~1.4° — 10× denser than m127's 60k at ~3°). Score each as `surr_MSE(q0, ω_search)` over all 500 epochs via surrogate forward. Keep top-50 by surrogate cost.
3. **Stage B — L-BFGS-B polish over q0.** From each top-50 seed, run L-BFGS-B with FD Jacobian on 3-parameter `q0_tangent` space (ω held at `ω_search`). `ftol=1e-7, gtol=1e-4, maxiter=200, maxfun=1000`. Cluster polished endpoints by q0 distance (`5°` threshold); keep up to `MAX_BASINS=5` unique basins.
4. **Stage C — hi-fi validation.** Propagate each basin through the true BRDF+shadow hi-fi forward model over the 500-epoch window; record hi-fi MSE vs observed CE curve. Hi-fi pool cap 2 for memory.
5. **Classify.** `FLIPPED_VALID` if winner hi-fi < 0.1, `FLIPPED_PARTIAL` if < 0.5, else `FLIPPED_FAIL`. `hypothesis_verdict = CONFIRMED` iff winner hi-fi < 0.15, else REFUTED.

Wall breakdown (from `timing` in result.json):

| stage | wall (s) |
|---|---|
| Stage A (600k grid) | 231.86 |
| Stage B (top-50 polish) | 28.96 |
| Stage C (top-5 hi-fi) | 176.23 |
| **Total** | **437.06 s (~7.3 min)** |

Within the pre-run 5–8 min estimate.

## Results (verified from `seed_033/result.json` and `batch_summary.json`)

### Headline

| quantity | value |
|---|---|
| traj_seed | 33 |
| classification | **FLIPPED_FAIL** |
| hypothesis_verdict | **REFUTED** (threshold hi-fi < 0.15) |
| winner hi-fi MSE | **2.6186** |
| winner basin_idx | 2 |
| winner q0_err_deg | 110.25° |
| winner surr_mse | 0.9995 |
| Stage A best surr (600k grid) | **1.1800** |
| Stage A top-50 worst surr | 1.1855 |
| n_basins_found | 5 |

### Per-basin detail (post-polish)

| basin | surr_mse | hifi_mse | q0_err_deg | n_members (cluster) |
|------:|---------:|---------:|-----------:|--------------------:|
| 0 | 0.98036 | 2.63157 | 159.19° | 11 |
| 1 | 0.99890 | 2.70046 |  91.55° | 1  |
| 2 | 0.99950 | **2.61858** | 110.25° | 1  |
| 3 | 0.99956 | 3.03776 | 140.72° | 3  |
| 4 | 1.00130 | 2.78363 | 145.04° | 1  |

All 5 basins cluster around surrogate cost ≈ 0.98–1.00 and hi-fi ≈ 2.6–3.0 — a shallow fan of local minima on the `−ω_true` slice, none of which is the known m126 basin.

### Comparison to [[m127_flipped_omega_search]] (60k grid, same seed 33, same `ω = −ω_true`)

| quantity | m127 (60k) | m129 (600k) | ratio / Δ |
|----------|:--------------:|:---------------:|:---------:|
| Stage A best grid surr | 1.1833 | 1.1800 | 0.997 (0.3% improvement) |
| Stage B best post-polish surr | 0.978 | 0.9804 | ≈ identical |
| Hi-fi winner MSE | 2.578 | 2.6186 | essentially same |
| Winner q0_err | 91.43° | 110.25° | different shallow basin |
| classification | FLIPPED_FAIL | FLIPPED_FAIL | no change |

**10× grid density bought 0.3% improvement in the best Stage A score.** The known basin (surr ~0.086, hi-fi 0.082) remained unreached.

### Known target (not reached)

Seed 33's known flipped-ω basin from [[m126_wrapped_pipeline]]:
- q0_err: 98.53° (about body +Z, 2.5° off per [[omega-sign-degeneracy]])
- surrogate cost: ~0.086
- hi-fi MSE: ~0.082
- q0-width: <0.001° (per [[m126_wrapped_pipeline]] polish observations `Δq0 ≤ 0.0004°`)

Closest m129 basin to the known basin by q0-geodesic geometry:
- Basin 1 at q0_err 91.55° is **~7° off** from the known 98.53° basin, yet its surrogate cost is 0.9989, not 0.086. The cost surface 7° away from the known basin is fully saturated at ~1.0 — no gradient signal pointing toward the 0.086 attractor.

## Verdict: REFUTED

Winner hi-fi 2.6186 > threshold 0.15. Script-emitted `hypothesis_verdict = REFUTED`.

### Mechanism of failure

1. **600k → 3° to 1.4° spacing; basin is <0.001° wide.** Grid density improved by 10×, median spacing from ~3° to ~1.4°. Seed 33's known basin is sub-0.001° wide (per [[m126_wrapped_pipeline]] polish `Δq0 ≤ 0.0004°`). The basin is still ~1000× narrower than the densest grid cell in m129 — the mismatch merely went from 10000× to 1000×. The improvement is arithmetically irrelevant: grid density would need to increase by a further 1000× (i.e. to ~600 million points, ~60 GB surrogate forward) to have a grid vertex inside the basin.

2. **0.3% improvement in best grid score confirms the cost surface is saturated away from the known basin.** Most of the `−ω_true` slice has surrogate cost ~1.0 regardless of `q0` — the [[dark-mag-saturation]] plateau dominates. 10× more grid points just samples more saturated plateau. The true basin is a pinprick hole in a saturated landscape; no amount of grid refinement short of an astronomically finer mesh finds it.

3. **L-BFGS-B polish cannot cross from basin 1 (q0_err 91.55°) into the known basin (q0_err 98.53°) — 7° away in q0 but on the other side of a saturation wall.** Both points sit on the saturated plateau at surr ~0.998 → the gradient from basin 1's neighbourhood points at its own local minimum (~0.9989), not across the 7° saturation gap toward the 0.086 basin. L-BFGS-B requires a continuous gradient descent path; no such path connects the two minima on this cost surface.

4. **The 50-polish cluster at surrogate ~1.0 and hi-fi ~2.6 describes the cost landscape on the `−ω_true` slice for seed 33:** a shallow plateau with many surrogate local minima near 1.0 and no single deep basin except the pinprick known one. This is structurally *different* from what polish plus grid works for: attractors found in [[m115_surrogate_pipeline]] sit in wide basins where nearby samples point toward them over degrees of q0. Seed 33's flipped basin is a saturation-bounded well.

### Does this generalize to other seeds' hypothesized narrow flipped-ω basins?

The mechanism is seed-agnostic: ANY narrow-basin attractor (q0-width << grid-spacing) is invisible to grid-then-polish regardless of density, because:
- Grid misses the basin (<0.001° vs ≥1.4° spacing).
- Plateau around the basin is saturated, so polish from anywhere nearby does not walk into it.

This is a property of the surrogate cost's plateau structure on the `−ω_true` slice, not a property of seed 33 specifically. Other seeds with narrow flipped-ω attractors (if they exist) would fail the same way. Seed 12's wide basin (from [[m127_flipped_omega_search]], width ≥ 3°) is the ONLY class reachable by this method.

## Implications

### For [[omega-sign-degeneracy]]

Confidence stays at **medium**. m129 neither confirms nor refutes the existence of narrow flipped-ω basins on the other 9 "FLIPPED_FAIL" seeds from [[m127_flipped_omega_search]] — it only proves that grid-then-polish cannot find such basins even at 10× density. The 2/11 confirmed cases stand:
- Seed 33: narrow basin (hi-fi 0.082, known from [[m126_wrapped_pipeline]]).
- Seed 12: wide basin (hi-fi 0.171 at q0_err 140°, from [[m127_flipped_omega_search]]).

Added to concept Open-Questions: m129 closes the grid-density question in the negative.

### For [[gradient-based-inversion]]

Third `NOTE:` added. The branch's "densify the grid 10–100×" option (option b) in the post-[[m127_flipped_omega_search]] decision-tree note) is now tested and REFUTED. The remaining option for finding narrow basins on the flipped-ω slice is DE (option c). This does NOT change the `#validated` status of the wrapped forward-ω pipeline — the refutation is specific to the narrow-basin enumerator question.

### For [[basin-of-attraction]]

The "flipped-ω basins span a wide range of widths" observation (from [[m127_flipped_omega_search]]) is now supplemented by this hard bound: narrow basins are not grid-density-recoverable at 600k on SO(3). Any grid-based enumeration strategy for narrow attractors must use a *non-uniform* sampler (adaptive refinement from hot regions, DE's mutation-selection, etc.) — uniform grid density is arithmetically ruled out.

## Next-experiment queue

1. **m130 (RECOMMENDED, ~30–45 min).** DE-over-q0 with ω fixed at `−ω_true`, population ~200, 10 restarts per seed, all 11 baseline seeds. DE's population-based mutation-selection is width-agnostic — it samples near previous best candidates and converges into whatever basin is locally deepest. If seed 33's narrow basin is reachable at all within a reasonable compute budget, DE is the only remaining tool. Hypothesis: DE recovers seed 33's known basin (hi-fi ~0.082) AND finds 0–3 additional narrow flipped-ω basins on the other 9 FAIL seeds from m127.

2. **Close the flipped-ω investigation** if m130 also lands negative with a clean "narrow flipped-ω basins cannot be enumerated by any search method tested" summary. The collective evidence is:
   - Coarse grid + polish ([[m127_flipped_omega_search]]): finds wide basins only (seed 12).
   - Dense grid + polish (m129): confirms density alone is not the fix.
   - Warm-start polish from m115 basins ([[m128_warmstart_polish]]): finds no flipped-ω basins (wrong cost manifold).
   - DE on `−ω_true` slice (m130, pending): only remaining enumerator.

3. **Deferred** — structural analysis of why the saturation plateau is so wide on the `−ω_true` slice for seed 33 specifically. Could motivate a retrained surrogate with wrong-attitude negatives to widen the basin.

## Plots + checkpoints

Per-seed outputs in `data/results/inversion_diagnostics/m129_densegrid/seed_033/`:
- `stage_a_grid.npz` — 600k surrogate scores, top-50 quaternions
- `stage_b_polish.npz` — up to 5 polished basins, iteration counts, polish metadata
- `stage_c_hifi.npz` — hi-fi MSE + 500-epoch hi-fi magnitudes per basin
- `result.json` — structured per-seed summary
- `run.log` — per-seed pipeline log

Batch-level:
- `data/results/inversion_diagnostics/m129_densegrid/batch_summary.json` — single-seed row with config snapshot, control classification, hypothesis verdict
