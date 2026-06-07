---
title: "s063a — Polhode coordinates ↔ omega_a bijectivity (200-case round-trip)"
type: experiment
sources:
  - lib/jacobi_propagator.py
  - experiments/s062_jacobi.md
related:
  - project_jacobi_propagation_priority.md
  - project_polhode_prior.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (200/200 cases pass at machine precision)
---

# TL;DR

The Jacobi-coords parameterization of body-frame ω is a clean bijection. Round-trips machine-tight across 200 test cases (120 cohort seeds + 60 random + 12 corner + 8 sign-perm + near-separatrix synthetics): max recon err **3.23e-14**, max polhode-curve drift **1.33e-14** rel, round-trip via `omega_jacobi(t=0)` **3.23e-14**. The polhode parameterization is safe to use as the sampling basis for any future architecture that wants dynamics-admissibility by construction (s062d target). Wall: **0.02 s** for all 200 cases.

# What

Three gates on `(regime, k², a=(a1,a2,a3), τ_0, sgn)` extracted from `_build_omega_func`:

1. **Reconstruct.** Given the polhode-coords info dict, evaluate ω_PA(τ=τ_0) via the closed-form sn/cn/dn formulas. Must equal omega_pa input.
2. **Polhode-curve consistency.** Evaluate ω_PA at 200 phases τ ∈ [0, 4K(m)). All points must have constant 2T and L² (the polhode is the joint level set of energy + angular-momentum-squared).
3. **Round-trip.** `omega_jacobi(times=[0], omega0, I)[0]` must equal omega0 in body frame.

# How

`experiments/s063a_polhode_bijectivity.py`. Test population (200 cases):
- 120 cohort seeds (real m048 ω₀, |ω| ∈ [0.11, 1.48] dps)
- 60 random: log-uniform |ω| ∈ [1e-4, 5e-2] rad/s, random direction
- 12 corner: near-principal-axis perturbations
- 8 sign-permuted: same |ω|, varying ω-component signs
- 9 near-separatrix synthetics in PA frame: constructed by setting (a, c) with a²·I₁(I₂-I₁) ≈ c²·I₃(I₃-I₂) so `disc → 0`

For each: forward map via `_build_omega_func` → reconstruct → polhode-curve sweep → round-trip via `omega_jacobi`.

`reconstruct_omega_pa(info, tau)` exposed as a clean inverse helper for downstream s062d use — given polhode coords + phase, returns ω_PA without any time-mapping step.

# Result

| Gate | Max error | Threshold | Pass? |
|---|---|---|---|
| Reconstruct ω from coords | 3.23e-14 | 1e-10 | ✅ (3 orders of magnitude margin) |
| Polhode-curve 2T, L² constancy | 1.33e-14 rel | 1e-12 | ✅ (100× margin) |
| Round-trip via `omega_jacobi(t=0)` | 3.23e-14 | 1e-12 | ✅ |

All 200 cases pass. The reconstruct and round-trip errors are the same (3.23e-14) because they trace the same code path internally.

# Why this matters

The polhode parameterization `(|L|, 2T, regime, τ_0, sgn)` is operationally a clean substitute for raw `(ω₁, ω₂, ω₃)` — same 3 DOF, but every sample is dynamics-admissible by construction (lies on a real polhode). For sampling-heavy architectures (s062d polhode-basis sampler, the s061 constant-ω cloud reframe), this matters because:

1. Uniform sampling in raw ω-space wastes density on physically equivalent points (different (ω₁, ω₂, ω₃) but same polhode = same dynamics).
2. The natural priors are *per-polhode* (s053 pol_diam) and *along-polhode* (phase) — these decouple cleanly in polhode coords.
3. Inverting (LC → polhode coords) is a different problem class than (LC → raw ω); the s055a result (pol_diam recoverable at 25% MAPE) hints that some polhode coords are LC-informative even when raw ω components are not.

Bijectivity at machine precision was the precondition for any of that to work. Now confirmed.

# Numbers

- 200 cases / 200 passing all 3 gates
- Max reconstruct err: 3.23e-14 (vs 1e-10 gate)
- Max polhode-curve rel err in 2T, L²: 1.33e-14 (vs 1e-12 gate)
- Wall: 0.02 s total

# Artefacts

- `experiments/s063a_polhode_bijectivity.py` — script
- `results/s063a/summary.json` — headline gates
- `results/s063a/per_case.json` — per-case (regime, k², |ω|, recon_err, etc.)
- `results/s063a/demo_polhode_curve.npz` — clean polhode curve sample (Case A, k² ≈ 0.5) for any future viz

# Out of scope

- Inverting LC → polhode coords (s055a already measured pol_diam LC-recoverability; |L| and 2T LC-recoverability TBD).
- Near-separatrix conditioning (only synthetic cases tested; real cohort has 1 seed at k² > 0.99 — see s063c).

# Cross-references

- `lib/jacobi_propagator.py` — module under test
- `experiments/s062_jacobi.md` — the parent omega-validation
- `experiments/s063b_polhode_curvature.md` — uses the same parameterization for cost-surface analysis
- `experiments/s063c_polhode_census.md` — cohort distribution of polhode coords
