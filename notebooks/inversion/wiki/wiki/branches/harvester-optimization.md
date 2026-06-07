---
title: "Geo-ckpt Harvester + Grid-Step Optimisation"
type: branch
sources: ["raw/inversion_diagnostics/m103_hybrid/", "raw/inversion_diagnostics/m102_fullmse/"]
related: ["[[surrogate-omega-selection]]", "[[surrogate-de-search]]", "[[candidate-selection]]", "[[grid-search]]", "[[nm-refinement]]"]
created: 2026-04-15
updated: 2026-04-16
confidence: medium
---

# Branch: Geo-ckpt Harvester + Grid-Step Optimisation

## Status: #open

## Two related questions in one branch

1. **Immediate (engineering):** how do we cheaply harvest `geo_ckpt.npz`-shaped omega-vector pools for the 6 baseline seeds (6, 12, 33, 36, 74, 93) that lack them?
2. **Research (optimisation):** can the alignment-cost grid step itself be shrunk by 5×? Surrogate-DE now provides a cheap verifier that didn't exist when grid resolution was last tuned.

## Background

Today's [[surrogate-omega-selection]] inline test (4/4 confirmation) relies on per-seed `geo_ckpt.npz` files containing all 26 geo-refined omega candidates as vectors. These exist for ~13 seeds in `data/results/inversion_diagnostics/m103_hybrid/seed_NNN/geo_ckpt.npz`, but not for all 10 baseline seeds. The 6 missing baseline seeds (6, 12, 33, 36, 74, 93) were processed by older experiments (m095, 99, 102) that predate the "save all candidates per stage" rule and only stored the winner's q0/w0.

## Cost picture

- **Recurring iteration on existing geo_ckpts:** ~6 min/seed (proven 2026-04-15)
- **Naive serial harvest (legacy m102 grid+NM+geo per seed):** ~12 min × 6 = ~70 min one-time
- **Optimised harvest (this branch):** target **~15-20 min wall-clock for all 6 seeds**, then 6 min/seed forever

## Three optimisation levers

> **Status update 2026-04-15:** Lever 2 (skip geo) is **REFUTED** by [[m117_result_harvester]] sanity check on seed 14. NM-only top-26 pool's best ω was 48.81° vs 0.34° in the geo-refined pool; the truth-adjacent cluster was entirely absent, and NM-only minimum cost was ~80× worse than geo-refined. L-BFGS-B reaches narrow deep basins that Nelder-Mead cannot. **Geo refinement stays in the 6-seed harvester.** Levers 1 (coarser grid) and 3 (outer-parallel) remain untested and viable.

### Lever 1: Coarser grid
- Current: 2000 directions × 20 magnitudes = 40000 grid points. ~6-7 min for grid step.
- Proposed: 1000 × 20 = 20000 points. Predicted ~3-4 min.
- **Risk:** truth omega might not land in NM_TOP=300 pool with coarser angular sampling.
- **Memory rule constraint:** `feedback_magnitude_grid.md` says N_MAGS >= 20 (do NOT reduce magnitude axis).
- **Sanity check:** run on the 4 seeds with known truth omega (0, 14, 24, 27) and verify NM pool still contains the truth omega. If yes, safe to use for harvest.

### Lever 2: Skip geo refinement — REFUTED 2026-04-15 ([[m117_result_harvester]])
- Hypothesised: geo's L-BFGS-B polish just refines omega values; surrogate-DE re-optimises attitude per omega, so imprecise omegas should still rank.
- **Reality:** geo is not a polish, it is a *basin-class change*. Nelder-Mead (derivative-free simplex) finds wide shallow basins around wrong omegas; L-BFGS-B (quasi-Newton, gradient-based) reaches narrow deep basins around truth-adjacent omegas that NM structurally cannot access. Seed 14 sanity showed NM-only best ω = 48.81° (vs geo-refined 0.34°) and ~80× cost gap.
- **Decision:** keep the geo step. Add ~2-3 min/seed back into the harvester budget.

### Lever 3: Parallelise across seeds
- Current: each script uses `Pool(24)` on a 16-core machine (already 1.5× oversubscribed).
- Proposed: run 6 seeds concurrently with `Pool(2-3)` each. Total in-flight workers 12-18 ≤ 16 cores.
- **Why this works:** grid is embarrassingly parallel across directions; going from `Pool(24)` to `Pool(3)` doesn't 8× the per-seed time, more like 2-3×. Net wall-clock ~30-35 min for 6 seeds vs ~70 min serial.
- **Caveat:** verify `OPENBLAS_NUM_THREADS=1` per worker so BLAS doesn't fight for math libraries (already standard in surrogate scripts).

## Open research question (Lever 4)

**Can the alignment-cost grid resolution be reduced 5× (e.g. 400 dirs × 20 mags) without losing truth from the NM pool?**

Previously this was hard to verify because hi-fi MSE was the only ground-truth-quality scorer (~5 min/eval). Now surrogate-DE-MSE is a ±0.005 hi-fi proxy at ~14 s/eval. So the verifier is cheap.

**Experimental shape:** for each grid resolution {2000×20, 1000×20, 500×20, 250×20}, run grid+NM on a small cohort, then surrogate-DE-rank the resulting NM_TOP=300 pool. Count seeds where a valid omega (top-2 by surrogate MSE) is recovered. The point at which recovery rate drops marks the grid floor.

If 500×20 still works, the grid step drops from ~6-7 min to ~1.5 min per seed — a real production-pipeline win, not just a one-time harvester win.

## Next experiment (proposed)

**`harvester_v2.py`** (writer task, scope ≤ 200 lines):
- Load trajectory + SPICE setup (same as m102)
- Grid step (start with full 2000×20 baseline; Lever 1 coarser-grid sanity on seed 14 as follow-up)
- NM step (NM_TOP=300, same as m102)
- **Geo step retained** (L-BFGS-B polish — [[m117_result_harvester]] proved this is load-bearing)
- Save full candidate pool to `data/results/inversion_diagnostics/harvester/seed_NNN/geo_ckpt.npz` (schema matches `archive/inline_omega_selection_test.py`: `w0_refs`, `geo_costs`)
- Pool size configurable; default `Pool(3)` per seed, outer-parallel 6 seeds

**Lever 1 sanity (cheap next step):** re-run harvester on seed 14 with 1000×20 grid, then validate via surrogate-MSE that the truth-adjacent cluster still survives geo refinement. If yes → adopt coarser grid for harvest and production pipeline.

**Open question for strategist:** NM_TOP=300 produced only ~8 distinct basins in m117 (many duplicates). Worth diagnosing whether NM starts are re-converging because grid seeds are too clustered, or because the alignment-cost landscape has genuinely few shallow basins. If the former, spreading seed selection could improve pool diversity without more compute.

## Decision tree

- Lever 2 skip-geo sanity → **REFUTED** ([[m117_result_harvester]]); geo step retained.
- If Lever 1 coarser-grid sanity passes → harvest 6 seeds with coarser grid + geo → extend `archive/inline_omega_selection_test.py` to all 10 baseline seeds.
- If Lever 1 fails → keep 2000×20 grid, harvest with full cost, accept ~25-30 min wall-clock.

## Why this branch matters

The current pipeline has `~12 min × N seeds` baked into every fresh-cohort experiment. That's the dominant cost-of-iteration for any future research direction (100-seed population study, symmetry census, parameter sweeps). Shrinking it is leverage everywhere downstream.

## Update 2026-04-15 — deprioritised

[[m118_cost_comparison]] showed that the alignment-cost framework (which the harvester was built to optimise) has a fundamental ~25° noise floor caused by the pab-contour's zero-phase and lo-fi approximations — see [[pab-contour-phase-angle-limitation]]. Harvesting better geo_ckpts wouldn't fix this — the cost function itself is the bottleneck, not the harvest efficiency. The research priority has shifted to [[surrogate-attitude-isoshell]]. This branch stays `#open` because if that direction pans out and we still need cheap `geo_ckpts` under the old framework (e.g. reproducing historical results), the harvester remains the right tool.
