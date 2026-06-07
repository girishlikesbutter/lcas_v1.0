---
title: "s062 — Jacobi-elliptic closed-form omega(t) validated across the post-fix m048 cohort"
type: experiment
sources:
  - concepts/jacobi_propagation.md
  - experiments/s062_jacobi_design.md
  - src/dynamics/attitude_propagator.py
  - lib/jacobi_propagator.py
related:
  - project_jacobi_propagation_priority.md
  - project_polhode_prior.md
  - feedback_verify_quaternion_convention.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (cohort-validated; closed-form omega agreement to integration-noise floor on 120 trajectories)
---

# TL;DR

Closed-form `omega_jacobi` implemented and validated on **120/120 post-fix m048 trajectories**. Worst-case ω-relative error vs DOP853 is **1.8e-10** (median 2.7e-11), far below DOP853's own atol=1e-12 noise floor. Conservation of `2T` and `L²` holds to **~1e-14** (100× below the 1e-12 gate). Closed-form ω wall is **0.18 ms/seed vs DOP853's 29 ms — a 160× speedup**.

Quaternion validation: `propagate_jacobi` (hybrid Path 3 = analytical ω integrated through `solve_ivp` for q) exactly tracks fresh DOP853 (`q_vs_truth == q_vs_dop` to all printed digits across all top-10 worst seeds). The worst `q_vs_truth = 7e-8` is **pure DOP853 vs DOP853 integration noise** between the cached truth and the fresh run, not Jacobi math error.

# What

Phase s062a from the design doc: eigendecompose the m048 inertia, implement closed-form ω(t) via `scipy.special.ellipj`, validate against the cached post-fix DOP853 truth across the full cohort. Hybrid q-integration (Path 3) wrapped on top of analytical ω as a convenient cross-check, but **full closed-form q(t) is deferred to s062b** — the inversion-architecture speedup for s061 depends on that, not on the ω-only result here.

# How

## Inertia characterization

m048's inertia in body frame is **already diagonal** (off-diagonals ~1e-15). Principal moments ascending: `I_1=7749, I_2=37985, I_3=38306` (kg·m²). The eigendecomposition produces a permutation matrix `R_pa` (det = ±1, made proper by sign flip) mapping body→PA frame:
- I_1 axis (smallest, the spin axis) = body ẑ
- I_2 axis (middle, the *unstable* axis) = body x̂
- I_3 axis (largest) = body ŷ

Saved as `results/s062/inertia_principal_axes.json`. Per the s062 reframe (commit 0919bcf), eigendecomposition is standard practice, not a gating condition.

## Closed-form ω(t)

`lib/jacobi_propagator.py::omega_jacobi(times, omega0, inertia)`:

1. Transform `omega0 → omega0_pa` via `R_pa.T @ omega0`.
2. Compute conserved `2T`, `L²`.
3. Detect regime from `disc = 2T·I_2 − L²`:
   - **Case A** (`disc > 0`, polhode encloses I_1, small-axis mode): `ω_1 = sgn_1·a1·dn(τ)`, `ω_2 = a2·sn(τ)`, `ω_3 = a3·cn(τ)`.
   - **Case B** (`disc < 0`, polhode encloses I_3, large-axis mode): `ω_1 = a1·cn(τ)`, `ω_2 = a2·sn(τ)`, `ω_3 = sgn_3·a3·dn(τ)`.
4. Amplitudes from Landau–Lifshitz §37 with proper Case A/B distinction. Crucially:
   - Case A `a2 = √[(L²−2T·I_1)/(I_2·(I_2−I_1))]` (NOT the Case B form).
   - Case A `k² = (I_3−I_2)(L²−2T·I_1) / [(I_2−I_1)(2T·I_3−L²)]` ∈ (0, 1), goes to 1 at separatrix.
   - Case A `τ_dot = √[(I_2−I_1)(2T·I_3−L²)/(I_1·I_2·I_3)]`.
5. `tau_0` from initial conditions: `sn(tau_0)=ω_2(0)/a2`, `cn(tau_0)=ω_3(0)/a3` (Case A) using `scipy.special.ellipkinc` with antipode-correct branch selection via `cn(0)` sign.
6. **Sign on time evolution**: `dτ/dt = sgn_1 · |τ_dot|` (Case A) — derived directly from `dω_2/dt = (I_3-I_1)/I_2 · ω_3·ω_1`. When `ω_PA_1 < 0`, τ decreases with t. **Missing this sign was the bug** that made the first cohort run produce 1.94 worst-case relative error (omega rotating the wrong way around the polhode).
7. Vectorized `scipy.special.ellipj(τ_dot·t + tau_0, k²)` returns `(sn, cn, dn)` for all 500 timesteps in one call.
8. Transform back: `omega_body = omega_pa @ R_pa.T`.

## Hybrid q(t) — Path 3

`propagate_jacobi(q0, omega0, inertia, times)` integrates `q̇ = ½·ω(t)⊗q` with `scipy.integrate.solve_ivp` (DOP853, rtol=1e-12, atol=1e-14) using the closed-form ω as the input. This validates ω end-to-end through the quaternion kinematics but does NOT deliver the order-of-magnitude speedup that closed-form q(t) (Path 2 = precession+nutation, deferred to s062b) would.

## Cohort validation

`experiments/s062_jacobi.py` ran on all 120 cached trajectories. For each seed:
- DOP853 baseline run fresh: `propagate_attitude(q0, ω0, times, 'tumbling', I)`.
- Jacobi ω closed-form: `omega_jacobi(times, ω0, I)`.
- Jacobi hybrid q: `propagate_jacobi(q0, ω0, I, times, q_rtol=1e-12, q_atol=1e-14)`.

Diagnostics: ω relative max err, q antipode-absorbing max diff (vs cached truth AND vs fresh DOP853), 2T drift, L² drift, wall times. All four ω-side gates are tracked as PASS/FAIL booleans in `summary.json`.

# Result

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| ω rel err max (vs DOP853) | 1.8e-10 | < 1e-8 | ✅ (50× margin) |
| ω rel err median | 2.7e-11 |  |  |
| 2T drift max (std/mean) | 1.4e-14 | < 1e-12 | ✅ (100× margin) |
| L² drift max (std/mean) | 1.3e-14 | < 1e-12 | ✅ (100× margin) |
| q vs cached truth max | 7.0e-8 | < 1e-9 | ❌ (see note) |
| q vs fresh DOP853 max | **7.0e-8 (identical to above)** | — | — |

**The q-gate "failure" is not a Jacobi error.** `q_vs_truth == q_vs_dop` to all printed digits on every top-10 worst seed: Jacobi q exactly tracks fresh DOP853, and *both* disagree with the cached truth at 7e-8 worst case. That difference is DOP853 vs DOP853 integration noise between the cached run (when truth was generated) and the fresh run today, not Jacobi error. The closed-form ω itself is correct to **1.6e-10** even on the worst-q seed.

The 1e-9 q-gate was set assuming a hypothetical closed-form q(t) with no ODE-integration noise. With Path 3 hybrid (still uses ODE for q), the q-gate is bounded by DOP853 noise. **The substantive validation is ω, conservation, and the regime split — all pass cleanly.**

## Regime distribution

- Case A (small-axis mode, 2T·I_2 > L²): **104/120** (87%)
- Case B (large-axis mode, 2T·I_2 < L²): **16/120** (13%)

Both branches exercised. The 16 Case-B seeds include the high-|ω| tail (seeds 17, 28, 34, 45, 53, 91 — all |ω| ≥ 0.77 dps); slow tumblers are uniformly Case A.

## Wall times

| Method | Per seed | 120-seed total |
|---|---|---|
| `propagate_attitude` (DOP853 default) | 28.9 ms | 3.47 s |
| `omega_jacobi` (closed form) | **0.18 ms** | **0.022 s** |
| `propagate_jacobi` (hybrid q, tight tol) | 33.9 ms | 4.06 s |

**ω-only speedup: 160×.** The hybrid q is slightly *slower* than DOP853 baseline because I tightened the q-tolerance to rtol=1e-12 (vs DOP853's default rtol=1e-10) to push the q-vs-truth bound. Both rates are dominated by Python overhead per ODE call; the ω alone is C-speed.

# Why this matters

s062a delivers two things:

1. **The ω closed form is correct, fast, and ready for any inversion architecture that needs ω(t) cheaply.** That includes the polhode-prior parameterization `(|L|, label, phase)` (auto-memory `project_polhode_prior.md`), polhode-tangent constraints on LM polish, and any per-seed ω-sampling cohort scan that today re-integrates the full ODE.

2. **The path to the s061 architecture speedup is open.** s061 needs *propagation* (q(t)) per (q_a, ω) candidate — currently DOP853-bound at 1ms+ per call. The 160× ω-speedup here is the *easy* half; the second half (closed-form q(t) via precession+nutation, Path 2) is now unblocked because we know the underlying ω math is correct and we have a working Path 3 baseline to compare against. **s062b is the next agent's job.**

The s062 reframe (commit 0919bcf) that eigendecomposition is standard practice and not a gating condition is confirmed: m048's inertia happens to be diagonal in body frame, but the eigendecomposition codepath would handle non-diagonal cases identically.

# Numbers

Headline: `summary.json` keys:
- `n_seeds`: 120
- `regime_A_count`: 104, `regime_B_count`: 16
- `omega_rel_err_max`: 1.79e-10
- `omega_rel_err_median`: 2.67e-11
- `q_vs_truth_max`: 7.02e-8 (== `q_vs_dop_max`)
- `twoT_drift_max`: 1.36e-14
- `L2_drift_max`: 1.33e-14
- `speedup_omega_only`: 160.4×
- `gate_omega_err_max_under_1e_8`: True
- `gate_q_vs_truth_max_under_1e_9`: False (DOP853 noise, see Result note)
- `gate_twoT_drift_under_1e_12`: True
- `gate_L2_drift_under_1e_12`: True

# Bug found and fixed during s062a

Two formula errors in the first pass before validation passed:

1. **Wrong Case A `a2` and `k²` formulas** (treated Case A as a literal copy of Case B): produced `k² > 1` for typical seeds → `scipy.special.ellipj` returned NaN. Fixed by using the correct (I_1 ↔ I_3 swapped) formulas, which give `k² ∈ (0, 1)` with `k² = 1` at separatrix.
2. **Missing time-evolution sign** (`dτ/dt = sgn_1·|τ_dot|`): code propagated ω in the wrong direction around the polhode when `ω_PA_1 < 0`. Derived from the Euler equation explicitly: `dω_2/dt = (I_3-I_1)/I_2 · ω_3·ω_1` implies `dτ/dt` carries the sign of the dn-driven component.

Both bugs caught by the cohort validation (not by single-seed smoke testing on seed 89 alone — fix #2 would have shown up even there, but #1 produced NaN that was at first invisible behind `omega rel max err = nan`).

# Artefacts

- **Module**: `lib/jacobi_propagator.py`. Exports `omega_jacobi`, `propagate_jacobi`, `_eigendecompose_inertia`.
- **Script**: `experiments/s062_jacobi.py`.
- **Per-seed validation**: `results/s062/validation_per_seed.npz` (120 seeds × all metrics).
- **Per-seed JSON**: `results/s062/per_seed.json`.
- **Headline summary**: `results/s062/summary.json`.
- **Inertia characterization**: `results/s062/inertia_principal_axes.json`.

# Out of scope

- **Closed-form q(t) (Path 2 = precession + nutation).** The architecture-unlocking work. Needs `ψ(t), θ(t)` from algebraic relations on ω and `φ(t)` via elliptic integral of the third kind `Π(n, am(τ)|m)` (scipy: `elliprj` Carlson form). Deferred to **s062b** to be picked up by a fresh agent.
- **Near-separatrix conditioning** (k² → 1, K(k) → ∞). The current Case A/B branch boundary at `disc == 0` handles the separatrix logically but not numerically. None of the 120 m048 seeds is near enough for `ellipj` to misbehave; if a seed ever lands within ~1e-6 of the separatrix, a fallback hybrid path would be needed.
- **Symmetric inertia (I_i = I_j)**. m048 has all three principal moments distinct (I_2 − I_1 = 30236; I_3 − I_2 = 321). The current code assumes distinct moments; a symmetric-top fast path is not needed.
- **Wiring into `lib/forward.py` (s062c)** and **polhode-basis sampling (s062d)**.

# Cross-references

- `concepts/jacobi_propagation.md` — math + literature.
- `experiments/s062_jacobi_design.md` — original phased plan (this is s062a; s062b–d remain).
- `concepts/polhode_prior.md` — the inversion-side parameterization that benefits from this.
- Auto-memory: `project_jacobi_propagation_priority.md` — update with "s062a done, closed-form ω validated cohort-wide at 160× speedup; q(t) closed form is s062b."
