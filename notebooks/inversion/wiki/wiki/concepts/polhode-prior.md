---
title: "Polhode-Conditioned ω Prior"
type: concept
sources:
  - "notebooks/inversion/survey/experiments/s051_polhode_observation.md"
  - "notebooks/inversion/survey/s048c_viewer/"
related:
  - "[[l-conservation]]"
  - "[[omega-magnitude-estimation]]"
  - "[[brightness_surface_path_matching]]"
  - "[[surrogate-attitude-isoshell]]"
created: 2026-05-07
updated: 2026-05-07
confidence: medium-high (architectural; requires cohort-scale measurement to fully promote)
---

# Polhode-Conditioned ω Prior

> **TL;DR.** For a torque-free rigid body the angular velocity in body frame is confined to a closed 1-D curve — the **polhode** — determined by the inertia tensor `I` (known) and a single scalar parameter `E/L²` (unknown). Treating the ω prior as "a 3-vector with bounds" wastes most of the search space; treating it as `(|L|, polhode-label, polhode-phase)` encodes the dynamic invariant for free. The amplitude of `|ω|(t)` wobble *is* the polhode size — a direct geometric readout of how stable the rotation is.

## Definition

For a rigid body with inertia tensor `I` and zero applied torque, two scalars are conserved in the inertial frame:

- **Energy:** `T = (1/2) ω · I · ω = const`
- **Momentum-squared:** `|L|² = ω · I² · ω = const`

In body coordinates this places `ω` on the intersection of two ellipsoids — the **polhode**, a closed curve. The shape and size of each polhode are fully determined by `I` and the dimensionless ratio `2T·I_max / |L|²` (the "energy-to-momentum" parameter). For a given satellite (`I` fixed), the family of possible polhodes is a 1-parameter family that **fully tessellates the body-frame ω-space**.

Polhodes near a principal axis are small (quasi-stable rotation about that axis). Polhodes near the separatrix are large and explore most of the unit sphere ("tumbling"). The separatrix passes through the medium-axis principal direction.

## Why this is a prior, not a complication

Existing wiki coverage treats polhodes as a *nuisance*: [[omega-magnitude-estimation]] notes "glint recurrence is useless because polhode geometry breaks periodicity"; [[brightness_surface_path_matching]] mentions the "polhode complication" that adds 5–15° uncertainty to ω-direction derived from peak geometry. Both framings are reactive — polhodes mess up easy methods built on the implicit assumption of fixed-axis rotation.

This page reframes: **the polhode IS the structural prior on ω**. The inertia tensor `I` is a satellite property, computable a priori (`src.computation.inertia_calculator.compute_inertia_from_config` + cached `lib.hifi_render.COMPONENT_MASSES`). The polhode family is therefore knowable before any inversion runs. The unknown is which polhode, and the phase along it — three scalars total, exactly matching the dimensionality of raw ω, but each scalar is dynamics-informed.

## Visual signature: polhode size ≡ |ω|(t) wobble amplitude

For an asymmetric body, `|ω|²` is **not** conserved in body frame — only `T` and `|L|` are. Therefore:

- **Tiny polhode** (near principal axis): `|ω|(t)` is essentially flat; `ω` wobbles in a small loop near a fixed direction.
- **Large polhode** (near separatrix): `|ω|(t)` oscillates substantially; `ω` traces a large excursion across the body sphere.

The amplitude of `|ω|(t)` variation is therefore a **direct readout** of polhode size, observable from a single visualization of the body-frame ω-vector path. This is what the s048c+ viewer's body-frame ω-direction sphere panel renders.

**Empirical anchor (seed 89, m048 trajectory database):**
- `|ω|_mean = 0.2415 dps`, `|ω|_std/mean = 0.5%` → tiny polhode.
- Visual confirmation at frame 499: ω̂(t) traces a small closed loop near body −X (essentially principal-axis-locked rotation with small wobble).
- LC peak count: 14 across 1 hour → consistent with quasi-stable rotation.

Compare with the s042 cohort finding: ω-mag basin width inversely correlates with `|ω|`. **Hypothesis:** the mechanism is polhode size, not `|ω|` per se — high-|ω| seeds happen to be on near-separatrix polhodes (large amplitude), where any mis-specification of the polhode-label scalar moves you to a structurally-different ω̂(t) trajectory and breaks the LC fit. Verifiable by rendering polhodes for the high-|ω| / many-spike cohort tail (seeds 14, 17, 68, 81) and measuring polhode "size" (max angular deviation, area subtended on the unit sphere) against [[basin-of-attraction]] width.

## Relation to other concepts

- **[[l-conservation]]** — the polhode lives on the intersection of the energy ellipsoid AND the momentum ellipsoid; momentum conservation alone (1 ellipsoid) leaves a 2-D surface. The 2nd ellipsoid (energy) collapses to the polhode.
- **[[omega-magnitude-estimation]]** — `|ω|` regression from peak count gives one scalar (`|L|/I_eff`). Polhode-prior gives the same scalar plus the polhode label — two of the three needed.
- **[[brightness_surface_path_matching]]** — each peak constrains `ω` projection at one body-frame normal lobe; multiple peaks at the same lobe trace a polhode arc. The "polhode precession" mentioned there *is* this curve.
- **[[basin-of-attraction]]** — surrogate basin shapes (q0 ≳ 5°, ω-dir < 0.1°, ω-mag < 0.25%) likely reflect polhode tangent vs normal directions: ω perturbations *along* the polhode (changing phase) are partially absorbed by q0 changes (which slide the entire orbit); ω perturbations *off* the polhode (changing the label) are the structurally hard ones.
- **[[surrogate-attitude-isoshell]]** — the surrogate cost surface lives in body coordinates; restricting ω to its polhode is a structurally consistent restriction of the cost surface to a lower-dimensional submanifold without information loss.

## Architectural implication

**Replace the existing ω prior with polhode-conditioned sampling.** Three concrete forms:

1. **Polhode-aware grid.** Discretize `(|L|, polhode-label, polhode-phase)` instead of `(ω_x, ω_y, ω_z)`. The polhode-label dimension is naturally bounded by the inertia tensor's principal moments; polhode-phase is `[0, τ_polhode)` for a known polhode. Every grid sample is dynamics-admissible.

2. **Polhode-constrained LM polish.** When polishing `(q0, ω)` candidates, project ω updates onto the polhode tangent rather than free in `ℝ³`. Reduces the LM step from 3 ω-DOF to 1 (phase along polhode). Should tighten the ω-mag basin by a factor proportional to polhode-tangent-length / polhode-area on the unit sphere.

3. **Polhode-period frequency analysis.** The polhode period `τ_p` is computable from `(I, E, L²)`. LC peak spacings should reflect a beat between `τ_p` (body-frame precession) and the orbital phase-angle evolution (inertial). Lomb-Scargle on the LC should show both components for any tumbling seed; identifying which one is `τ_p` pins down the polhode label without touching ω.

The first form is the cheapest to test (no new physics, just a different sampler); the second is the highest-payoff (LM convergence is the cohort-scale bottleneck per [[gradient-based-inversion]]); the third is most diagnostic (gives the polhode label without inversion).

## What needs to be measured before promoting to #validated

- Polhode size (max angular extent + area subtended on unit sphere) for the m048 cohort's high-|ω| tail (seeds 14, 17, 68, 81). **Predicted:** large, near-separatrix.
- Correlation between polhode size and ω-mag basin width as measured in s042. **Predicted:** strong positive correlation (large polhode → narrow basin).
- Polhode period `τ_p` vs LC peak count: for cohort seeds, does peak count scale with `n × T_obs / τ_p` for some integer `n` (lobe count)? If yes, peak spacing carries `τ_p` and the polhode label is recoverable from the LC alone.
- Sample-efficiency comparison: polhode-conditioned grid vs uniform-ω grid at fixed seed-yield. **Predicted:** 10–100× reduction in samples-per-band-A-basin.

## Provenance

Insight derived from direct visual observation of the s048c+ viewer's body-frame ω-direction sphere panel on the 500-epoch seed-89 sweep. The closed-loop trail at frame 499 made the polhode topology immediate; the 0.5% `|ω|(t)` wobble simultaneously confirmed quasi-stable rotation. Verbal recognition that "ω̂ traces a polhode" + "polhodes are 1-D in body frame" + "we know I" yielded the architectural reframe in a single chain. Documented contemporaneously in `survey/experiments/s051_polhode_observation.md`.

The visual feedback loop was load-bearing: the same data (cached body-frame ω vectors per epoch in trajectory NPZs) was available for months without surfacing this restructuring; the reframe required *seeing* the closed loop on the unit sphere with the satellite present for orientation context.
