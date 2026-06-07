---
title: "Surrogate-truth offset (CORRECTED — truth IS the surrogate minimum)"
type: concept
sources:
  - "raw/inversion_diagnostics/m122/seed_014/summary.json"
  - "raw/inversion_diagnostics/m122/seed_027/summary.json"
  - "raw/inversion_diagnostics/m122/seed_046/summary.json"
  - "raw/inversion_diagnostics/m122/seed_074/summary.json"
  - "raw/inversion_diagnostics/m122/seed_093/summary.json"
  - "raw/inversion_diagnostics/m123/summary.json"
  - "raw/inversion_diagnostics/m124/summary.json"
related:
  - "[[surrogate-model]]"
  - "[[m122_hessian_curvature]]"
  - "[[m123_lbfgs_polish]]"
  - "[[m124_hifi_validate]]"
  - "[[gradient-based-inversion]]"
  - "[[basin-of-attraction]]"
created: 2026-04-16
updated: 2026-04-16
confidence: low
---

## CORRECTED 2026-04-16 — see [[m123_lbfgs_polish]]

This concept, as originally written, is **wrong**. [[m123_lbfgs_polish]] directly measured the displacement that this page predicted (L-BFGS from truth on all 5 seeds) and found it at machine precision — truth IS the surrogate minimum in physical units. The [[m122_hessian_curvature]] findings that motivated this page (nonzero gradient, seed-46 negative Hessian eigenvalue) are parameter-space / finite-difference artifacts, not physical offsets. The page is retained as a historical record of the reasoning; `confidence` is `low`.

# Surrogate-truth offset — historical claim and retraction

## Original claim (RETRACTED)

The minimum of the surrogate-residual cost was claimed to be offset from truth by the MLP's modelling error. Evidence: `‖grad‖ ∈ {9.6, 15.2, 38.5, 40.4, 53.7}` at truth across 5 seeds; `λ_min = −203` on seed 46. An order-of-magnitude estimate via `Δ ≈ H⁻¹·grad` predicted displacements of 0.06–4 rad.

## Why it was wrong

### Units mismatch

The m122 gradient is in parameter-space units: `rad` for the 3-component quaternion tangent and ω-direction tangent, `fractional` for the ω-magnitude perturbation. The Hessian eigenvalues likewise are mixed-unit. Translating back to physical displacement:

- **Seed 27** — `‖grad‖ = 40`, dominated by the stiff ω-direction eigendirection with `λ ≈ 1.11e7`. Predicted `Δ_param = 40 / 1.11e7 = 3.6e-6 rad = 2.06e-4°`. [[m123_lbfgs_polish]] measured ω-direction error from truth-start L-BFGS on seed 27 at `2.28e-4°`. Match within 10%.
- **Same calculation for the soft q0 eigenvalue (~14-169)** gives large apparent displacements (the original 0.06–4 rad estimate), but projects into the q0 subspace which [[m123_lbfgs_polish]] shows does not actually move: q0 error stays at ≤5e-6° after convergence. The soft-axis displacement prediction is an artifact of ignoring that the gradient's q0-component is itself near the parameter-space noise floor.

### FD noise floor on seed-46 negative eigenvalue

On `cost ≈ 0.05` with `h = 1e-4` and float64, the second-difference noise floor is `~cost_eps / h² ≈ 1e-10 / 1e-8 = 1e-2`. An eigenvalue of magnitude 203 is 20000× above that in magnitude, but the SIGN is determined by cancellation in the 4-corner stencil and is not robust at this scale. L-BFGS from truth on seed 46 converged in 2 iterations with zero q0 movement and 6e-5° ω-direction error — no saddle-like escape behaviour.

## What actually holds

- **Truth IS the surrogate minimum** in physical units. The gradient in parameter units is nonzero but maps to displacements at or below machine precision.
- **L-BFGS polishing from truth is a no-op** — useful as a sanity check but not as a research lever.
- **Gradient-based inversion does NOT inherit a bias from a surrogate-truth offset.** The remaining concerns about gradient-based inversion (narrow ω basin, saturated cost outside the basin) are still live — see [[gradient-based-inversion]] and [[dark-mag-saturation]].
- **HMC / Bayesian sampling does NOT inherit an offset** either. Posterior mode == truth (modulo the surrogate's overall 0.03 mag MAE, which is below the discriminating threshold).

## Confirmed safe by hi-fi (2026-04-16, [[m124_hifi_validate]])

[[m124_hifi_validate]] hi-fi-validated the truth-polish outputs from [[m123_lbfgs_polish]] on all 5 seeds. Polished-from-truth hi-fi MSE = {0.0024, 0.0024, 0.0025, 0.0024, 0.0024} for seeds {14, 27, 46, 74, 93} — all within 5% of the noise-floor reference σ² = 0.0025. Polishing near truth is unconditionally safe at the hi-fi level too, not just at the surrogate level. This strengthens the retraction above.

The m124 catastrophe (seed 27 DE-basin polish gives hi-fi 12–16× worse) is a SEPARATE phenomenon — it is about off-truth surrogate modelling error, not a truth-vs-surrogate-min offset. The two findings are compatible: the surrogate IS at its minimum at truth, AND the surrogate's off-truth landscape contains modelling-error-driven false minima that L-BFGS happily descends.

## Lesson

Dimensional consistency was the oversight. For any future FD-Hessian analysis on this cost: (a) record Hessian eigenvalues AND eigenvectors in mixed units, (b) project to the axis the researcher actually cares about (q0°, ω-dir°, ω-mag%) before interpreting magnitudes, (c) cross-check with a direct gradient-descent run before taking the Hessian's verdict seriously.
