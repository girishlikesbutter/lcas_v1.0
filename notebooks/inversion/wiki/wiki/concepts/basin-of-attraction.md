---
title: "Basin of Attraction"
type: concept
sources:
  - "raw/inversion_diagnostics/"
  - "raw/inversion_diagnostics/m121/seed_014/summary.json"
  - "raw/inversion_diagnostics/m121/seed_027/summary.json"
  - "raw/inversion_diagnostics/m121/seed_046/summary.json"
  - "raw/inversion_diagnostics/m122/summary.json"
  - "raw/inversion_diagnostics/m123/summary.json"
related:
  - "[[grid-search]]"
  - "[[nm-refinement]]"
  - "[[candidate-selection]]"
  - "[[m121_basin_width_metric]]"
  - "[[m122_hessian_curvature]]"
  - "[[m123_lbfgs_polish]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[m127_flipped_omega_search]]"
  - "[[m129_dense_grid_eval]]"
  - "[[gradient-based-inversion]]"
  - "[[omega-sign-degeneracy]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[surrogate-truth-offset]]"
created: 2026-02-15
updated: 2026-04-16
confidence: high
---

> ## ✅ 2026-04-17 — m122 RE-RUN ON CORRECT 1-HOUR WINDOW; QUALITATIVE GEOMETRY STANDS; SPECIFIC NUMBERS UPDATED
>
> m122 re-run (commit `b906691`) gave new correct-window Hessian data for all 5 seeds. Key updates:
>
> - **Basin widths q0:** roughly 0.30°–2.82° range across seeds (was 0.3–3.56° on wrong window — similar magnitudes, still 10–100× tighter than [[m121_basin_width_metric]]'s finite-scale empirical widths).
> - **Basin widths ω-dir:** 0.014°–0.028° where non-inf (was 0.003–0.05° wrong-window — qualitatively unchanged, still < 0.1°).
> - **Basin widths ω-mag:** 0.020%–0.035% (consistent with wrong-window numbers).
> - **Cohort universality (m122 HYP3):** REFUTED on correct window (was CONFIRMED). ATT_FAIL and OK cohorts have basin-geometry eig-spreads within 1.20× — basin geometry does NOT explain ATT_FAIL. This **flips** the strategic read: the driver of ATT_FAIL is upstream ω-direction quality (see [[m126_wrapped_pipeline]]), not basin shape.
>
> The qualitative "narrow ω basin / wider q0 basin / saturation plateau" geometry is unaffected. Per-seed numbers in the tables below should be updated from the new `m122/seed_NNN/summary.json` before being cited. See `notebooks/inversion/DATA_INTEGRITY_BUG.md` (Bug 2).

# Basin of Attraction

The convergence basin defines how close an initial guess must be for an optimiser to converge to the true solution. The shape depends on the optimiser and the cost function.

## Basin widths — Nelder-Mead on alignment cost (classical pipeline)

| Parameter | Basin width | Notes |
|-----------|------------|-------|
| Attitude (q0) | ~5 deg | All 48 trials converge, 0.15 deg final error |
| Omega direction | ~2 deg | Collapses by 10 deg |
| Omega magnitude | +/-10% hi-fi, +/-5% lo-fi | |

Source: m073, m087-88, m098-99. Cost is alignment-based, evaluated over windows of the light curve.

## Basin widths — surrogate-residual cost (post-[[m121_basin_width_metric]])

Measured directly on 3 ATT_FAIL cohort seeds (14, 27, 46) by perturbing (q0, ω) along each axis at scales {0.1-20°} and {0.25-20%}, 250 samples per scale, full 500-epoch trajectory and surrogate scoring.

| Parameter | Basin width | Cost at 5× basin | Notes |
|-----------|------------|:-----------------:|-------|
| Attitude (q0) | ≳ 5° | +534% at 5°; smooth, monotonic | WIDEST axis. Cost bump: +1.4% @ 0.1°, +73% @ 1°, +2048% @ 20°. Gradient-bearing across 20°. |
| Omega direction | < 0.1° at 1σ noise floor | saturated by 0.1° (+1596% @ 0.1°) | NARROW. Saturated (cost ≈ 2.0) by 0.5°. |
| Omega magnitude | < 0.25% at 1σ noise floor | saturated by 0.5% (+3081% @ 0.5%) | NARROWEST axis. Saturated (cost ≈ 2.0) by 1%. |
| Joint | < 0.25° | Dominated by ω component. |  |

### ω-direction is ANISOTROPIC within SO(3)

At 0.5° ω-direction perturbation, the cost varies 15-25× depending on the rotation axis. The basin is a narrow SLAB, not a sphere. Preferred body-frame axis is seed-specific:

- Seed 14: preferred ≈ body +Z; worst ≈ body −Y. Ratio 25×.
- Seed 27: preferred ≈ body +X; worst ≈ body ~(−Y, −Z). Ratio 17×.
- Seed 46: preferred ≈ body −X with +Z; worst ≈ body ~(+Y, +Z). Ratio 15×.

Common across all seeds: worst axis has dominant body ±Y component. Body Y is IS-901's solar-panel spin axis; ω perturbations around ±Y maximally swap bright/dark facet geometry.

### Mechanism: q0 vs ω asymmetry

- q0 error = constant rotational offset → bounded LC perturbation at every epoch (not cumulative).
- ω error = constant angular-velocity offset → attitude divergence grows LINEARLY with time.

End-of-window drift (IS-901, 3600 s window, |ω_true| ≈ 0.0215 rad/s):
- 0.1° ω-direction error → 7.8° attitude drift → enough to offset peak timing by 1–2 peak widths in back half of window.
- 0.25% ω-magnitude error → 11° drift → same regime.
- 1° q0 error → 1° LC perturbation at every epoch, not cumulative → mean_L1 bump ≈ 0.04 mag (+73% of truth cost).

Cost saturation at ~2.0 mean_L1 is the [[dark-mag-saturation]] floor: wrong ω produces all-dark surrogate predictions (~23 mag) vs observed bright (~13 mag), residual saturates at ~10, mean_L1 saturates at ~2 because only fraction of epochs matter for BRDF.

### Local curvature from Hessian ([[m122_hessian_curvature]])

6-DOF finite-difference Hessian at `(q0_true, ω_true)` measured on 5 seeds (ATT_FAIL: 14, 27, 46; OK: 74, 93). Per-axis 50%-cost-rise half-widths extracted from eigenvalues as `w = sqrt(0.5 · truth_cost / λ)` and projected via eigenvector axis fractions:

| Parameter | Hessian-derived width (5 seeds) | m121 empirical width |
|-----------|:-------------------------------:|:------------------------:|
| q0 (per principal axis) | 0.3°–0.9° | ≳ 5° gradient-bearing, cost saturates by 20° |
| ω-direction | 0.003°–0.05° | < 0.1° (saturated by 0.5°) |
| ω-magnitude | 0.006%–0.008% | < 0.25% (saturated by 1%) |

Hessian widths are **10–100× tighter** than m121's empirical widths. This is expected, not a contradiction: the Hessian is the local quadratic approximation, whereas m121's widths are the finite perturbation scales at which the cost has already saturated at the [[dark-mag-saturation]] ceiling (~2.05 mean_L1). The two measurements describe the same landscape at different scales — the cost is quadratic for a very short distance (the Hessian width) then transitions abruptly to the saturated plateau. There is no intermediate "gradient-bearing but non-quadratic" regime.

**Cohort universality (hyp3 CONFIRMED).** Eigenvalue spreads: ATT_FAIL median 2.9e+05, OK median 1.38e+05, ratio 2.11× — OK and ATT_FAIL basins are qualitatively identical. Basin width does NOT explain the ATT_FAIL classification; the classical pipeline's difficulty with these seeds comes from elsewhere (finding, not local geometry).

**ω-direction anisotropy — scale-dependent.** m121's "preferred body-frame axis" at 0.5° perturbation does NOT match the body-frame projection of m122's stiffest Hessian eigenvector for 2 of 3 ATT_FAIL seeds (seed 14 dot 0.07, seed 46 dot 0.26; only seed 27 matches at dot 0.89). Interpretation: the seed-specific anisotropy reported in m121 is a finite-scale saturation-shape property, not a local-curvature property. The statement "each seed has a preferred ω-rotation direction" still holds at the scale m121 measured, but the direction is NOT the one recovered from local curvature. Both measurements are valid; they describe different geometric features.

**Saddle at truth on seed 46.** λ_min = −203 (200× above FD noise floor). Truth is NOT a local minimum of the surrogate cost for this seed. Combined with non-zero gradient at truth on all 5 seeds (‖grad‖ = 9.6–53.7), this introduces the [[surrogate-truth-offset]] concept: the surrogate's local optimum is displaced from truth by the MLP's modelling error. Relevant to gradient-based inversion — even with perfect init, converged point ≠ truth.

**CORRECTION 2026-04-16 ([[m123_lbfgs_polish]]):** the saddle and offset claims are retracted. L-BFGS from truth on all 5 seeds (including seed 46) moves q0 ≤5e-6° and ω-dir ≤2.3e-4° — truth IS the surrogate's local minimum in physical units. The m122 gradient magnitudes were in parameter-space units; translated via `Δ = |grad| / λ_stiff` they predict physical displacements at FD-noise-floor scale (matches the measured L-BFGS final errors). Seed 46's negative eigenvalue is an FD-cancellation sign artifact, not a physical saddle. See [[surrogate-truth-offset]] (CORRECTED) and [[m123_lbfgs_polish]].

## Gradient-descent behavior at DE basins ([[m123_lbfgs_polish]])

From DE basins at ω-dir error ~3° and ω-mag error 10–50% (well outside the basin-of-attraction as measured above), L-BFGS-B with FD Jacobian still polishes ω meaningfully:

| Quantity | Range across 12 basin starts |
|----------|:---------------------------:|
| q0 move | <1e-3° (essentially locked) |
| ω-dir move | 0.02°–0.21° |
| ω-mag move | 0.09%–0.32% |
| Cost reduction ratio | 1.14×–12.9× (mostly 4–7×) |

This shows the surrogate cost in the [[dark-mag-saturation]] regime has a **small-but-nonzero gradient in ω** despite the empirical basin being <0.1°. The "saturated plateau" is not truly flat — it's shallowly sloped in ω and flat in q0. Gradient descent from a DE basin cannot escape the q0 attractor but CAN refine ω within it. This is the mechanism behind the DE + GD hybrid architecture documented on [[gradient-based-inversion]].

### Flipped-ω basins span a wide range of widths ([[m127_flipped_omega_search]], 2026-04-16)

Basins are NOT all Hessian-tight. [[m127_flipped_omega_search]] demonstrated that the flipped-ω half of parameter space contains attractors with **dramatically different q0-widths**:

- **Narrow (seed 33, [[m126_wrapped_pipeline]]):** q0-width sub-0.001° (measured as `Δq0 ≤ 0.0004°` during L-BFGS-B polish). Invisible to a 60k super-Fibonacci SO(3) grid at 3° median spacing.
- **Wide (seed 12, [[m127_flipped_omega_search]]):** visible at 3° grid spacing (Stage A surrogate best 0.6147 already drops via polish to 0.2297 → hi-fi 0.171). The basin's q0-width is at least 3° and probably more (didn't measure directly).

Both basins have ω fixed at `−ω_true` and produce hi-fi MSE ≤ 0.2. Both are genuine observational degeneracies. The 600× difference in q0-width across these two confirmed cases means "basin-of-attraction" is not a single scalar per seed — the landscape contains a distribution of widths, and grid-based search methods are biased toward the wide end.

**Implication for search design:** pure grid-then-polish methods (like m127) find wide basins; DE on surrogate (like [[m115_surrogate_pipeline]]) finds both wide and narrow because DE's mutation-selection is width-agnostic at convergence. This is additional evidence that **DE is irreplaceable as an attractor-enumerator for this cost landscape**, not just a warm-start for gradient-based methods.

**Quantitative bound from [[m129_dense_grid_eval]] (2026-04-16):** a 10× SO(3) grid density increase (60k → 600k super-Fibonacci, median spacing 3° → 1.4°) delivered **0.3% improvement in Stage A best surrogate score on seed 33** (1.183 → 1.180) and did NOT recover its known 0.001°-wide basin. To place a grid vertex inside a 0.001°-wide basin via uniform SO(3) sampling would require ~600 million grid points. Uniform grid density is thus arithmetically ruled out for narrow-basin enumeration on this cost landscape; the landscape near narrow basins saturates at the [[dark-mag-saturation]] plateau cost ~1.0, preventing gradient-based polish from reaching the basin even from 7° away in q0 (seed 33 basin 1 in m129: q0_err 91.55°, only 7° from the known 98.53° target, yet surrogate cost 0.999 vs the basin's 0.086).

### Exception: gradient-bearing plateau edge (seed 6 basin 1, [[m126_wrapped_pipeline]])

The "q0 locked" claim has a rare exception. When the initial state sits at the **edge of the saturated plateau** — specifically q0_err ~5° (inside the q0 gradient-bearing band) AND ω-dir ~3° with a correctly-oriented (non-retrograde) ω — L-BFGS-B can find a joint q0+ω gradient direction that walks the state down into the basin proper. Seed 6 basin 1 in [[m126_wrapped_pipeline]]: q0 moved 5.10° → 1.64° (Δ=−3.46°), ω-dir 3.08° → 1.03° (Δ=−2.05°), ω-mag 15.2% → 0.6%. Hi-fi MSE 0.130 → 0.017 (OK-class).

This is the only known baseline-cohort case of polish escaping a q0 attractor. Condition: the "q0 gradient-bearing to ~20°" per m121 is real when the surrogate is in a bright-geometry trajectory (so the plateau is not saturated), AND the ω-dir direction slopes toward truth rather than away. Not a general rule — 10 of 11 baseline seeds still have polish lock q0.

## Key Interactions

- **Attitude error > 5 deg collapses the omega basin**: the parameters are coupled. A bad attitude estimate means even the correct omega will not produce a matching LC.
- **Window length affects omega basin**: 180s window gives ~10 deg omega basin; 3600s window narrows to ~2 deg (alignment cost) or ≲ 0.1 deg (surrogate cost). Longer observations constrain more tightly — time-integrated ω error grows.

## Design Targets

### For the classical pipeline (grid + NM + alignment cost)
- ~5 deg of true attitude
- ~2 deg of true omega direction

This is the rationale for NM_TOP=300 ([[m098_m099_nm_grid_pipeline]]): more candidates increases the probability that at least one falls within the basin.

### For gradient-based search on surrogate-residual cost
- < 0.1° of true omega direction
- < 0.25% of true omega magnitude
- < 5° of true q0

Classical pipeline currently delivers ω-direction error 0.3–3° ([[m102_fullmse]], "OK" seeds) and ω-magnitude 0.1–0.7%. This meets the ω-magnitude target marginally but misses the ω-direction target by **3–30×**. Gradient-based search cannot start from the classical pipeline's NM output unless the ω-direction is first tightened. See [[gradient-based-inversion]] decision tree.

## Implications

- Grid density (N_DIRS) must be fine enough that the nearest grid point is within ~2 deg of truth (classical). With 2000 directions, mean nearest-neighbour distance is ~4 deg — marginal. NM expansion compensates.
- The narrow omega basin is why [[lo-fi-mse]] cannot replace [[alignment-cost]] at grid level ([[m097_candidate_ranking]]): lo-fi false positives at 4 deg spacing flood the NM pool.
- For surrogate-based search, the narrow ω basin means DE (population-based, no gradient needed) remains a better fit than L-BFGS unless omega is pre-refined. This keeps [[surrogate-de-search]] as the primary attitude finder even after [[gradient-based-inversion]] infra lands.
