---
title: "s063b — Cost-surface curvature: polhode tangent vs polhode-normal at truth (3 seeds)"
type: experiment
sources:
  - lib/jacobi_propagator.py
  - lib/surrogate_eval.py
  - lib/forward.py
related:
  - project_jacobi_propagation_priority.md
  - project_polhode_prior.md
  - project_omega_mag_basin_scales_with_omega.md
created: 2026-05-12
updated: 2026-05-12
confidence: medium-high (3 seeds; consistent direction; deeper-perturbation saturation limits the curvature ratio precision)
---

# TL;DR

The polhode tangent IS the soft direction. At truth (q₀, ω₀) on seeds 89/28/14, 1D ω-perturbations along the Euler-equation tangent grow much more slowly than perturbations along the two off-polhode normals. **Off-polhode curvature is 4–17× sharper than along-polhode**, consistently across the three test seeds spanning |ω| ∈ [0.24, 1.44] dps. **LM polish should reparameterize ω onto the polhode basis** — the two off-polhode dimensions are where overshooting hurts; the on-polhode dimension is benign.

# What

For seeds 89 (slow, |ω| = 0.24 dps), 28 (fast, 1.44 dps), 14 (1.23 dps), at truth (q₀, ω₀):

1. Compute three orthonormal ω-perturbation directions in body frame:
   - **tangent**: Euler dynamics `dω/dt = I⁻¹((I·ω) × ω)` — the polhode tangent
   - **normal_E**: grad(2T) = 2 I·ω, Gram-Schmidt against tangent — breaks energy conservation
   - **normal_L**: grad(L²) = 2 I²·ω, Gram-Schmidt against (tangent, normal_E) — breaks momentum²
2. For each direction d, sweep ε ∈ [-5%, +5%] of |ω| (21 points). At each ε, perturb ω, propagate q via DOP853, render via surrogate, compute full-LC MSE vs cached `mag_hifi`.
3. Fit `MSE(ε) = c₀ + α·ε + β·ε²` and report β (curvature) in fractional-eps units.

Total wall ≤ 5 min Pool(1) for 3 × 63 = 189 forward evaluations.

# How

`experiments/s063b_polhode_curvature.py` + `experiments/s063b_plot.py`. Truth-ω evaluation gives baseline ρ_truth ∈ [0.29, 0.48] across the three seeds — all Band A — so the surrogate floor is well below the perturbation signal.

The off-polhode normals are obtained by Gram-Schmidt against the tangent rather than as principal-axis directions; this gives an honest "off-polhode" subspace decomposition where any perturbation in the (normal_E, normal_L) plane is guaranteed to move to a different polhode.

# Result

Curvature β (in fractional-ε units, dimensionless cross-seed):

| Seed | \|ω\| dps | ρ_truth | β_tangent | β_normal_E | β_normal_L | ratio_E | ratio_L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 89 | 0.240 | 0.40 | 1.82e+02 | 1.48e+03 | 7.63e+02 | **8.13×** | **4.18×** |
| 28 | 1.438 | 0.48 | 1.52e+02 | 1.60e+03 | 2.66e+03 | **10.53×** | **17.46×** |
| 14 | 1.229 | 0.29 | 1.57e+02 | 1.06e+03 | 6.50e+02 | **6.75×** | **4.15×** |

ρ at ±5% perturbation:

| Seed | ρ at ε=±5% tangent | ρ at ε=±5% normal_E | ρ at ε=±5% normal_L |
|---:|---:|---:|---:|
| 89 | 13.6 (Band D) | 40.0 (deep D) | 28.8 (deep D) |
| 28 | 15.4 | **70.6** | **77.4** |
| 14 | 15.6 | 48.0 | 48.7 |

The seed 28 and seed 14 normals SATURATE — ρ at ±2.5% is already as high as at ±5%, meaning the quadratic fit underestimates the true curvature ratio. The reported numbers are lower bounds. The cleanest case is seed 89 where the parabolas are well-behaved through ±5% on all three directions.

# Why this matters

Three concrete consequences:

1. **LM polish dimensional reduction.** Current `s058::lm_polish` updates ω as a free 3-vector. If the cost-surface is 4–17× sharper in 2 of 3 directions, the LM Jacobian's condition number is dominated by those tight directions. Reparameterizing ω as `(|L|, 2T, phase)` (Jacobi coords) aligns the polish axes with the cost-surface anisotropy. The hard-to-overshoot soft direction (phase) is decoupled from the easy-to-overshoot sharp directions (|L|, 2T).

2. **Sampling density allocation.** For any ω-grid or Sobol architecture (s059j, s059k), density along the polhode tangent need not match density across polhodes. A coarse-in-phase, fine-in-(|L|,2T) grid will hit the same basin recall at a fraction of the candidates of an isotropic Sobol.

3. **Polhode prior interpretation.** s053 already showed pol_diam predicts basin width cohort-wide. That's a measurement of the *across-polhode* basin extent. s063b confirms the basin in the *along-polhode* direction is much wider. So the s053 cohort prior is really an across-polhode prior; the along-polhode dimension needs its own handling (per-seed phase from anchor, or LM polish from a coarse phase grid).

# Numbers

Headline (results/s063b/summary.json):
- `seed 89`: β_tangent=182, β_normal_E=1480, β_normal_L=763 → ratios 8.1× / 4.2×
- `seed 28`: 152, 1600, 2660 → 10.5× / 17.5×
- `seed 14`: 157, 1060, 650 → 6.8× / 4.2×

ρ_truth (surrogate floor) per seed:
- seed 89: 0.40 — Band A
- seed 28: 0.48 — Band A
- seed 14: 0.29 — Band A

Across all 3 seeds: **both normal directions are at least 4× sharper than the tangent**, with no exception.

# Out of scope

- Validation on a cohort scale (3 seeds; expect curvature ratio to vary across the cohort but stay > 1 from the structural argument).
- A fully reparameterized LM polish implementation in Jacobi coords (queued behind s062b — without closed-form q(t), per-iteration cost stays DOP853-bound regardless of which ω basis we use).
- The phase direction's true basin width along the polhode (3 seeds × 21 points underestimates; finer sweep at ±1% and ±0.5% would be a follow-up).
- Coupling of ω perturbations with q₀ perturbations (this is fixed-q₀ analysis only).

# Artefacts

- `experiments/s063b_polhode_curvature.py` — main
- `experiments/s063b_plot.py` — figure
- `results/s063b/summary.json` — per-seed β, ρ, perturbation grids
- `results/s063b/polhode_curvature_slices.png` — 3-panel log-y plot of ρ(ε) per direction per seed. **Saved: `notebooks/inversion/survey/results/s063b/polhode_curvature_slices.png`**

# Cross-references

- `experiments/s063a_polhode_bijectivity.md` — verified the coords used here
- `experiments/s063c_polhode_census.md` — cohort distribution of polhode invariants
- `project_omega_mag_basin_scales_with_omega.md` — pol_diam = across-polhode basin predictor; s063b shows tangent direction is wider than s053 measured
