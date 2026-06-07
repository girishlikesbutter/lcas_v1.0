---
title: "Upstream Redesign — 6-DOF Surrogate DE replaces m103"
type: branch
sources:
  - "raw/inversion_diagnostics/phase_B_cohort_summary.json"
  - "raw/inversion_diagnostics/m126_wrapped/batch_summary.json"
related:
  - "[[m103_hybrid]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[surrogate-model]]"
  - "[[alignment-cost]]"
  - "[[constraint-poor-regime]]"
  - "[[phase_B_m048_cohort]]"
  - "[[phase-angle-operating-range]]"
  - "[[m048-migration]]"
  - "[[gradient-based-inversion]]"
  - "[[dark-mag-saturation]]"
created: 2026-04-17
updated: 2026-04-28
confidence: medium
---

# Branch: 6-DOF Surrogate DE replaces m103

## Status: #open-validated (single-seed)

Concrete instance built and pilot-tested on seed 91 in [[m135_alignment_cost_forensics_constrained_anchor]] — the constrained-anchor algorithm (`score_constrained_anchor.py`) implements: surrogate-driven anchor-epoch selection (most-constrained by q-count) → δq factorisation over ω-grid → surrogate full-LC MSE ranking. Single-shot rank-1 lands at 2.18° from truth on seed 91, well within m115's 3-5° bridging radius. Multi-seed validation on the 5 random-cohort failure seeds (47, 51, 79, 84, 89) is the next gate.

[[m135_alignment_cost_forensics_constrained_anchor]] also empirically demolishes alignment cost: it is anti-correlated with truth (4-22 orders of magnitude difference between cost(geo_best) and cost(truth) on failure seeds). This makes the case for replacement effectively unconditional — alignment cost is the wrong objective, not an imperfect one.

## The question

Can a single-stage 6-DOF differential evolution over `(q0, ω)` using the neural surrogate's LC MSE as cost replace all of [[m103_hybrid]]'s grid + lo-fi re-rank + NM + multi-phi expansion + geo refinement, and simultaneously fix the two failure modes that stage-based pipeline exposed?

## Motivation

[[m103_hybrid]] is a 5-stage patchwork accumulated across ~20 micro-experiments (micro64 → micro77 → micro90 → micro95 → micro99 → micro102 → m103). Each stage exists to work around a specific limitation of [[alignment-cost]]: narrow basins, magnitude-grid coarseness, anchor-normal degeneracy, wrong-phi convergence. It was designed in a world where hi-fi LC evaluation was too slow to use as the primary search cost.

[[surrogate-model]] changes that. The trained MLP produces predicted magnitudes at **~50,000× hi-fi speedup** (per `project_surrogate_model.md`). Surrogate LC MSE is already used as the primary search cost in [[m115_surrogate_pipeline]]'s inner 3-DOF DE over q0 (ω fixed). Extending it to 6-DOF `(q0, ω)` is a small step mechanically, a large step architecturally.

## Why it might fix both failure modes

### [[constraint-poor-regime]] (seed 23)

m103's alignment cost uses only spec peaks (~2 in seed 23). Surrogate LC MSE uses **all 500 epochs**. Even with few peaks, the whole light curve shape (plateau lengths, relative depths, timing) carries rich information that the surrogate scores against.

### High-phase flatness (seeds 28, 69)

m103's alignment cost is geometric (normal-vs-PAB dot products). At high phase geometry, the PAB sweeps through body-normal space in ways that produce many near-equivalent configurations. Surrogate LC MSE is **phenomenological** (matches the actual brightness measurements), not geometric, so it doesn't inherit this failure mode directly.

### Caveat: surrogate has its OWN high-phase failure mode

[[dark-mag-saturation]] documented this during m119/m120: the surrogate saturates at near-terminator geometries (more of the LC spent near the mag-23 noise floor → gradient vanishes for DE/polish). High-phase seeds are exactly this regime. Surrogate-DE might STILL fail at 87° phase — just differently from m103.

**Prediction for the validation gates below:** surrogate-DE likely wins on seed 23 (constraint-poor — surrogate's strength) but may only match or slightly beat m103 on seeds 28/69 (high-phase — surrogate's weakness).

## Method sketch

```
def surrogate_6dof_inversion(truth):
    def cost(params):
        axis_angle = params[:3]           # rotvec in R^3
        omega      = params[3:6]          # rad/s in R^3
        q0 = axis_angle_to_quaternion(axis_angle)
        pred_lc = surrogate_forward(q0, omega, truth.observation_times, ...)
        return mean_squared_error(pred_lc, truth.observed_lc)

    # Bounds informed by |omega|_est and prior on q0
    omega_bound = truth.omega_mag_est * 1.5
    bounds = [(-pi, pi)] * 3 + [(-omega_bound, omega_bound)] * 3

    # Multi-start DE
    basins = []
    for start in range(N_STARTS):
        res = differential_evolution(cost, bounds, seed=start, workers=-1,
                                     popsize=DE_POPSIZE, maxiter=DE_MAXITER,
                                     tol=1e-6, polish=False)
        basins.append(res.x)

    # Cluster winners in (q0 geodesic, ω angle) space
    clusters = basin_cluster(basins, q0_tol=5_deg, omega_tol=2_deg)

    # Return top-3 basins by surrogate MSE (or all basins with MSE below threshold)
    return sorted(clusters, key=lambda c: c.best_mse)[:3]
```

Then the existing m126 polish chain takes over: hi-fi validate each basin (before), L-BFGS-B polish on the surrogate for each basin, hi-fi validate each polished basin (after), keep the minimum.

**Compute estimate:** pop=300, maxiter=150, 5 restarts = 5 × 45,000 surrogate evals ≈ 5 × 225 s = ~20 min/seed at the wide DE scale. Could be faster with `workers=-1` actually wired through.

## Validation gates (before Phase 3 100-seed batch)

Three sequential gates. Any failure falls back to the quick path (patch m103 instead).

### Gate A: m046 11-seed cohort match

Run surrogate-6DOF-DE on seeds `{0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}`. Target: **matches [[m126_wrapped_pipeline]] on ≥ 4 of the 5 current OK seeds** (0, 6, 12, 74, 93), with hi-fi MSE within 2× of the m126 winner.

Rationale: if the new upstream can't reproduce known-good recoveries, it's a regression regardless of what it fixes.

### Gate B: m046 FAIL rescue

On the same 11 m046 seeds, target: **improves ≥ 1 current FAIL seed** (36, 46) — i.e. at least one of them drops from hi-fi > 0.3 into the PARTIAL (0.01-0.3) or OK (< 0.01) range.

Rationale: if the new upstream doesn't add value on seeds we already can't handle, the architectural change isn't justified.

### Gate C: Phase-B FAIL rescue

Run on seeds `{23, 28, 69}` from [[phase_B_m048_cohort]]. Critical target: **seed 23 (constraint-poor)** must move from hi-fi 1.60 into the PARTIAL or OK range. This is the best-case scenario for the surrogate-LC cost — seed 23's failure was specifically about alignment-cost degeneracy that the surrogate sidesteps by construction.

Secondary target (softer): seeds 28 and 69 improve by at least a factor of 2 in hi-fi. If they don't improve at all, the high-phase regime is genuinely intractable for both architectures and we'd revert to accepting those as FAIL in the cohort.

## Compute budget

- Gate A: 11 seeds × ~8 min/seed (surrogate DE is fast but still ctx setup + validation takes time) ≈ 90 min
- Gate B: re-scored from Gate A output (0 additional compute)
- Gate C: 3 seeds × ~8 min/seed ≈ 25 min
- **Total validation:** ~2 hours compute, 1-2 sessions of implementation work before that
- **If all gates pass:** Phase 3 100-seed m048 batch ≈ 12-15 hours with the new upstream

Compare to quick path: ~3 hours infrastructure work + ~24 hours batch compute with expected 30-40% upstream-FAILs.

## Related prior art

- [[m115_surrogate_pipeline]] — already uses surrogate DE, but only over q0 (3-DOF) with ω as input from m103
- [[m119]] / [[m120]] — used surrogate for attitude scoring, validated per-epoch fidelity (~0.03 mag MAE)
- [[m127_flipped_omega_search]] / [[m128_warmstart_polish]] / [[m129_dense_grid_eval]] — attempted variants of narrow-basin search on the surrogate; informed the [[dark-mag-saturation]] caveat
- External: Burton 2024 ([[project_burton_pso_literature|burton-pso-literature]]) uses two-stage PSO + NN surrogate for full 6-DOF inversion, best attitude error 0.8° — direct architectural analogue

## Open questions

1. DE population/iteration sizing: m115's inner DE uses `popsize=15, maxiter=50, N_STARTS=3` per ω candidate (≈ 2k evals × 3). 6-DOF needs more — Burton's papers suggest ~500k evals for 6-DOF is typical. Validation should determine the minimum useful budget.
2. ω magnitude bounds: peak-count estimate is ±13% median error (micro52), worst-case ~86% (seed 42). DE bounds of `|ω|_est × [0.5, 1.5]` would cover most cases. Adaptive widening for outliers?
3. Clustering criterion: q0 geodesic + ω angle jointly define basins. What tolerances? m126 uses `q0 < 5°, ω < 2°` — adopt those.
4. Does surrogate differentiability help? Gradient-based polish (L-BFGS-B via `autograd`) is strictly more efficient than DE near a basin — but DE is needed for GLOBAL search. Architecture: DE for discovery, gradient for refinement. Same as current m126, just moved upstream one stage.

## What this branch is NOT

- Not a research claim. No experimental evidence yet.
- Not a validated path forward. Three explicit gates must pass before Phase 3 compute commits.
- Not a rejection of [[m103_hybrid]] — m103 remains the fallback if any gate fails.

## Next action (proposed, awaiting user sign-off)

Implement the minimal 6-DOF surrogate DE experiment as a standalone script (e.g. `notebooks/inversion/12_brightness_surface/m133_surrogate_6dof.py`). Run Gate A on the 11 m046 seeds. Analyse, then decide.
