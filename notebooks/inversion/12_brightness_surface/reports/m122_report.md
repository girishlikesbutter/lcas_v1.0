# m122 — Hessian-at-truth basin geometry

## Hypothesis (3 falsifiable claims)

1. **Per-axis curvature matches m121 widths.** Finite-difference Hessian
   eigenvalues at truth yield at least one eigenvector with ω-direction width
   < 0.1° and at least one q0 eigenvector with width ~5° (≤10°).
2. **Stiff eigenvectors live in ω-direction subblock.** The two stiffest
   eigenvectors are predominantly ω-direction, and their body-frame projection
   aligns (|dot| > 0.7) with m121's empirical stiff axes
   (seed 14 ≈ +Z, seed 27 ≈ +X, seed 46 ≈ -X/+Z).
3. **Cohort basin geometry is similar.** OK-cohort seeds (74, 93) have
   eigenvalue spread within one order of magnitude of ATT_FAIL (14, 27, 46).
   If confirmed, basin geometry is NOT why ATT_FAIL seeds are hard.

## Method

6-DOF tangent-space FD: r ∈ ℝ³ (Rodrigues, LEFT-multiply), ξ ∈ ℝ² (tangent ⟂
ω̂_true via deterministic Gram-Schmidt), δ ∈ ℝ (|ω| fraction). Cost = mean_L1
residual of surrogate (identical to m121). Steps h = [0.01°, 0.01°, 0.01°,
0.005°, 0.005°, 0.01%]. Central differences → 73 surrogate evals per seed
(1 truth + 12 edge + 60 off-diagonal). Symmetrise H, eigendecompose.

## What to look at when results land

- `data/results/inversion_diagnostics/m122/summary.json` verdicts line.
- Per-seed `seed_NNN/summary.json` for eigenvalues (descending), axis
  classification, `basin_widths_physical`, and `anisotropy_axis_dot_micro121`.
- `seed_NNN/evals.npz` caches all 73 cost evals for future re-analysis
  (e.g. Richardson, multi-h, alternate cost variants).

## Refutation criteria

- **hyp1 REFUTED** if no eigenvector reaches ω_dir width < 0.1° or no q0
  width ≤ 10° for any ATT_FAIL seed (14/27/46).
- **hyp2 REFUTED** if seeds 14/27/46's stiffest ω-dir eigenvector body-axis
  has |dot| < 0.7 with the empirical m121 axis, or if neither of the top
  two eigenvectors is predominantly ω-direction.
- **hyp3 REFUTED** if OK/ATT_FAIL eigenvalue-spread medians differ by ≤10×
  (basin geometry is *not* the discriminator).

## Deferred (TODO v2)

Richardson step-size check, alt cost variants, eigenvector bar plots,
cohort comparison figure, ω-dir axis projection at multiple times, saved
hi-fi LCs at ±h perturbations.
