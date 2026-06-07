---
title: "s062b — closed-form Jacobi q(t) attempted via Path 2; convention puzzle resolved by s065; hybrid retained as production"
type: experiment
sources:
  - lib/jacobi_propagator.py
  - experiments/s062_jacobi_design.md
  - experiments/s065_propagator_convention_spice_test.md
  - concepts/jacobi_propagation.md
  - concepts/quaternion_convention.md
  - src/dynamics/attitude_propagator.py
related:
  - project_jacobi_propagation_priority.md
  - feedback_verify_quaternion_convention.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (hybrid validated; Path 2 convention puzzle root-caused by s065 — propagator has an omega-sign convention residual, not a quaternion bug)
---

> **UPDATE (2026-05-12, post-s065)** — the convention puzzle described below is RESOLVED.
> s065 ran the SPICE comparison and confirmed: the propagator's q-interpretation is
> correct (matches SPICE `pxform` to 3e-16 at t=0), but the kinematic ODE has the
> OPPOSITE sign of the textbook passive-J2000→body kinematic, so omega is implicitly
> treated as `-omega_physical`. This is a bookkeeping convention, not a fix-+-regenerate
> bug; option-1 (document, don't fix) was chosen on 2026-05-12. The original "hand-off
> recipe" at the bottom of this writeup is superseded — see `experiments/s065_propagator
> _convention_spice_test.md` and `concepts/quaternion_convention.md` "Omega-sign residual"
> section. Path 2 closed-form q(t) can be implemented by deriving under the codebase's
> sign convention (effectively: textbook 3-1-3 derivation with `omega -> -omega`); whether
> that's worth the work given the m048 cohort being invariant under sign flip is a
> separate architectural call.

# TL;DR

Attempted closed-form q(t) via Path 2 (precession + nutation 3-1-3 Euler decomposition) and **hit a convention puzzle that this session did NOT resolve**. The codebase's q ODE `dq/dt = +0.5 omega ⊗ q` (LEFT, Hamilton) does NOT correspond to the conventional passive-J2000→body kinematic equation `dR/dt = -[omega]_x R`, and `L_J2000 = R.T @ L_body` is NOT conserved by the propagator's output even on a toy asymmetric tumbler — even though physics demands it be. **Path 2 deferred for a fresh agent.**

Production `propagate_jacobi` reverted to the HYBRID path (closed-form omega + DOP853 on q-quaternion ODE) which is correct but does not deliver the ~10× architecture speedup that true closed-form q(t) would. The Path 2 attempt is preserved as `_propagate_jacobi_path2_incomplete` in `lib/jacobi_propagator.py` for handoff.

3-gate validation (hybrid path):
- **GATE 2 (2T, L² conservation): PASS strict 1e-12** on all 3 seeds.
- **GATE 1 (q vs DOP853, antipode-aware element-wise):** PASS 1e-9 on seed 89 (1.0e-10) and seed 14 (6.1e-10); **FAIL at 1.6e-8 on seed 28** — DOP853-vs-DOP853 integration-noise floor between the hybrid's decoupled 4D q-ODE and the propagator's coupled 7D system. Not Jacobi math error.
- **GATE 3 (k1_body vs cached truth):** numbers track ~2× GATE 1 (slow-tumbler: 2e-10, fast-tumbler: 3e-8). Misses the strict 1e-12 spec, which is unachievable by any DOP853-based path.

# What

s062b was scoped (per `experiments/s062_jacobi_design.md`) to implement true closed-form q(t) via either Path 1 (theta-function form) or Path 2 (precession + nutation Euler decomposition), gated by a non-negotiable 3-gate round-trip validation. The session chose Path 2 and built the math out cleanly. The validation gate exposed a convention mismatch between my standard right-handed Goldstein derivation and the codebase's q evolution. The mismatch was diagnosed in detail but not resolved this session.

# How — Path 2 attempt

Math sketch (in `_propagate_jacobi_path2_incomplete`):

1. Eigendecompose I = R_pa · diag(I_1, I_2, I_3) · R_pa^T (ascending, identical to s062a).
2. Closed-form omega_PA(t) via existing `_build_omega_func` (s062a, validated cohort-wide).
3. Algebraic theta(t), psi(t) from omega_PA(t):
   - cos θ(t) = (I_3 ω_3(t)) / |L|
   - ψ(t) = atan2(I_1 ω_1(t), I_2 ω_2(t))
   These follow from `L_PA = R_z(ψ) R_x(θ) R_z(φ) · (0, 0, |L|)` = (|L| sin θ sin ψ, |L| sin θ cos ψ, |L| cos θ) and the constitutive `L_PA = I_PA · ω_PA`.
4. φ(t) via 1D scalar ODE: `dφ/dt = |L| (2T - I_3 ω_3²) / (L² - I_3² ω_3²)` (Goldstein 3-1-3 result, integrated by DOP853 rtol=1e-13).
5. R_J2000_to_L computed once at t=0 from (θ_0, ψ_0, q_0) with φ(0) = 0 fixing the precession-axis-rotation freedom.
6. R_J2000_to_body(t) = R_pa · R_z(ψ(t)) · R_x(θ(t)) · R_z(φ(t)) · R_J2000_to_L → quaternion via Shepperd's method.

# Result — the convention puzzle

**At t = 0**, my Path 2 reconstruction matches the propagator's R exactly to 1e-16. ✓ Setup is sound.

**At t > 0**, the reconstruction drifts. Quantitative diagnostics on seed 89:
- Element-wise antipode-aware diff grows linearly from 0 at idx 1 (Δt=7.21s) at rate ~3e-2 per step initially.
- The drift rate matches φ̇(t) (precession rate). Two interpretations: (a) my φ has the wrong sign and contributes nothing usable, or (b) my φ-sign flip is correct but ALSO breaks θ/ψ.
- **Negating φ(t) (left-handed body rotation reading) reduces the diff 6×** at idx 1 (from 3e-2 → 5e-3). The remaining residual is a slow θ drift: algebraic θ_alg = arccos(I_3 ω_3 / |L|) and DOP-extracted θ_dop (from R_J2000_to_PA(t) · R_J2000_to_L^T) **diverge by 0.6° per step** even though ω_3 matches DOP to 1e-12.
- That residual is NOT a sign flip — it's a structural mismatch.

**The deeper issue** found while debugging: under torque-free dynamics, the conserved angular momentum L_J2000 must be a constant vector in J2000. Standard formulas:
- If R is passive J2000→body: L_J2000 = R.T @ L_body should be constant.
- If R is active body→J2000: L_J2000 = R @ L_body should be constant.

**Neither is constant** for this codebase's propagator output, on either seed 89 or a controlled toy asymmetric tumbler (I = diag(1, 2, 3), ω₀ = (0.5, 0.3, 0.7), q₀ = identity, dt = 5s). The magnitude |L_J2000| is conserved (orthogonal R + |L_body| conservation), but the direction precesses. This is physically inconsistent with torque-free dynamics under any standard interpretation.

The propagator's q ODE `dq/dt = +0.5 ω ⊗ q (LEFT, Hamilton)` gives `dR/dt = +[ω]_× · R` for R built via Hamilton's active-formula. The conventional passive-J2000→body kinematic equation is `dR/dt = -[ω]_× · R`. These differ by the sign of [ω]_× — so the codebase's q evolution is consistent with the body rotating by `-ω` (not `+ω`) under standard right-handed conventions, OR with a left-handed body frame, OR with some other convention I haven't pinned down.

The Euler equation `I·ω̇ = -ω × I·ω` IS standard right-handed (verified by hand-derivation: dL_body/dt = -ω × L_body at idx 0 matches the propagator's L_body(1) - L_body(0) at 1e-3). So omega itself is right-handed. The mismatch is somewhere in how q + omega are interpreted together.

Despite this, the renderer + propagator chain is internally self-consistent (cached k1_body matches `R @ sun_J2000` bit-exactly), so all downstream code works. My Path 2 closed-form, derived under standard right-handed conventions, doesn't match this internal consistency.

**Hand-off note for the fresh agent (SUPERSEDED — see s065)**: ~~the puzzle to crack
is whether the codebase's `dq/dt = +0.5 ω ⊗ q (LEFT)` should actually be `dq/dt = -0.5
ω ⊗ q (LEFT)` to satisfy passive-J2000→body conservation, or whether the codebase has
a different consistent convention (e.g., left-handed body frame) that Path 2 should
adopt. Suggested first move: write a 20-line standalone script that integrates
`dq/dt = -0.5 ω ⊗ q (LEFT)` on the same toy and checks whether `R.T @ L_body` is
constant. If yes, the propagator has a sign bug; if no, the convention is something
else entirely.~~

**Resolution (s065, 2026-05-12)**: The recipe above WAS tested against SPICE directly
(`experiments/s065_propagator_convention_spice_test.py`). Verdict: the propagator's
q-interpretation is correct (matches SPICE `pxform` to 3e-16 at t=0), but the kinematic
ODE has the opposite sign of textbook passive-J2000→body — equivalent to feeding the
propagator the negation of the standard right-hand body-frame omega. This is a sign
convention residual baked into the codebase, internally consistent across the m048
cohort + STL/BRDF/surrogate/inversion. NOT a fix-+-cohort-regeneration situation. The
user chose option 1 on 2026-05-12: document, don't fix. s063 numerical findings,
m048 cached LCs, polhode invariants etc. are all invariant under `omega -> -omega`
and unaffected.

# Why this matters

- **Architecture-unlock (s062b's primary value) is deferred.** True closed-form q(t) was supposed to give ~10² speedup on the s061 constant-ω-per-candidate reframe and unlock s062d polhode-basis enumeration. The hybrid path retains the ~160× ω speedup from s062a but the q piece is still DOP853-bound.
- **Downstream impact**: s064 (Jacobi-coord LM polish reparameterization, the s063b payoff) can still be implemented because polish needs ~50 forward evals per candidate, not millions — DOP853 cost is tolerable there. The bigger loss is s062d (polhode-basis enumerator for s061), which is propagator-bound at 100k–1M candidates × 500 epochs.
- **Honest cost**: the closed-form attempt cost real session time without producing a working artefact. The convention finding itself is the deliverable.

# Numbers

Hybrid 3-gate on seeds 89 / 28 / 14, single thread (Pool 1, OMP=1):

| Seed | GATE 1 (q vs DOP) | GATE 2 (2T drift) | GATE 2 (L² drift) | GATE 3 (k1 vs cached) | Jacobi wall |
|---:|---:|---:|---:|---:|---:|
| 89 | **1.02e-10**  PASS | 2.48e-16 | 2.83e-16 | 2.03e-10 | 10.1 ms |
| 28 | **1.61e-08**  FAIL | 3.28e-16 | 3.38e-16 | 3.17e-08 | 44.5 ms |
| 14 | **6.11e-10**  PASS | 1.20e-15 | 4.39e-16 | 1.00e-09 | 60.5 ms |

GATE 1 spec was < 1e-9, GATE 3 spec was < 1e-12. The 1e-12 GATE 3 spec is unachievable by any DOP853-based path (k1 diff ≈ 2× q diff); it was set assuming true closed-form q(t) with zero integration noise.

DOP853 vs cached truth on all 3 seeds: 0.0 (bit-identical fresh re-run). So the "DOP-vs-DOP integration noise" floor is the Jacobi-hybrid's decoupled 4D q-ODE differing from the propagator's coupled 7D system at the chosen tolerances — not propagator-vs-cached drift.

# Out of scope

- Resolving the convention puzzle. Hand-off to fresh agent.
- Path 1 (theta-function form). Path 2 was chosen per the design doc's recommendation; Path 1 has the same convention problem from the inside.
- Tightening DOP853 tolerance to make GATE 1 pass strict 1e-9 on seed 28. Verified `rtol ∈ {1e-12, 1e-13}` × `atol ∈ {1e-14, 1e-15}` all floor at 1.6e-8 for seed 28 (decoupled-4D-vs-coupled-7D ODE structural noise, not solver tolerance limited).
- Wiring `propagate_jacobi` into `lib/forward.py` as method= flag (s062c). Deferred until Path 2 is resolved or hybrid is confirmed adequate.

# Artefacts

- **Updated module**: `lib/jacobi_propagator.py` — production `propagate_jacobi` = hybrid; Path 2 attempt preserved as `_propagate_jacobi_path2_incomplete` with full docstring documenting the failure mode.
- **Validation script**: `experiments/s062b_jacobi_quaternion.py`.
- **Validation output**: `results/s062b/summary.json`.

# Cross-references

- `experiments/s062_jacobi.md` — s062a foundation (closed-form ω validated cohort-wide).
- `experiments/s062_jacobi_design.md` — phased plan; this session attempted Phase 2.
- `concepts/jacobi_propagation.md` — math + literature.
- `concepts/quaternion_convention.md` — the convention doc; the codebase's q ODE appears to violate the standard passive-J2000→body kinematic by a sign, which the convention doc does not flag.
- Auto-memory: `project_jacobi_propagation_priority.md` — update needed: s062b partial; closed-form q deferred pending convention resolution.
