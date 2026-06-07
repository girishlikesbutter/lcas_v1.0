---
title: "s067 — Post-fix propagator validation: SPICE pxform + L_J2000 + Jacobi cross-check, all 4 gates PASS"
type: experiment
sources:
  - src/dynamics/attitude_propagator.py (post-2026-05-12 sign fix)
  - lib/jacobi_propagator.py (post-2026-05-12 sign fix)
  - experiments/s065_propagator_convention_spice_test.md
  - experiments/s066_lj2000_nonconservation.md
related:
  - feedback_propagator_omega_sign_convention.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (4-gate regression test, all PASS)
---

# TL;DR

Two-line sign fix in `propagate_euler` and `propagate_principal_axis` (one matching change in `lib/jacobi_propagator.py`'s hybrid q-ODE) closes the s066 L_J2000 non-conservation finding. Bundled regression test passes 4/4 gates:

| Gate | Pre-fix | Post-fix | Criterion | Result |
|------|---------|----------|-----------|--------|
| A1 SPICE pxform @ dt=1s, +ω_textbook | 2.04e-4 | **9.73e-11** | < 1e-9 | PASS |
| A2 L_J2000 toy drift over 50s | 4.28 (191% rel) | **7.83e-12** | < 1e-10 | PASS |
| A3 L_J2000 m048 cohort, max drift | 97-903 (36-137% rel) | **1e-9 to 3e-7** (~0% rel) | < 1e-6 | PASS |
| A4 Jacobi ↔ propagate_euler parity | n/a | **2e-10 to 2e-8** | < 1e-7 | PASS |

The fix: change `+0.5` → `-0.5` in the q-kinematic ODE so that the kinematic matches the q-interpretation `as_matrix(q) == passive J2000→body`. Both production propagators (`propagate_euler` line 318, `propagate_principal_axis` lines 177-184) plus the Jacobi hybrid (`lib/jacobi_propagator.py` line 432) updated.

# What

s066 (2026-05-12 earlier today) showed `L_J2000 = R(t).T @ I @ ω(t)` drifts by 36–137% of |L_body| over the m048 cohort's 60-min LCs. Real torque-free physics requires exact conservation. Mechanism (already in s065 + s066): `dq/dt = +0.5 ω ⊗ q (LEFT)` is the opposite sign from the textbook conv-(a) kinematic `dq/dt = -0.5 ω ⊗ q (LEFT)`.

This experiment applies the one-character fix to both production propagators + the Jacobi hybrid, then runs a 4-gate regression test against SPICE and the textbook-control derivations from s065 and s066.

# How

The fix:
- `src/dynamics/attitude_propagator.py:318` — `q_dot = +0.5 * _quaternion_multiply(omega_quat, q)` → `q_dot = -0.5 * ...`
- `src/dynamics/attitude_propagator.py:177-184` — `q_rot = exp(+0.5 ω t)` → `q_rot = exp(-0.5 ω t)` (negate sin components in the closed form)
- `lib/jacobi_propagator.py:432` — `return 0.5 * _quat_multiply(omega_quat, q)` → `return -0.5 * ...`
- All three module docstrings updated.

The validation script `s067_postfix_propagator_validation.py` bundles four gates as a single pass/fail test (~10 s wall):

- **A1 SPICE pxform** (duplicate of s065's setup): finite-diff body-frame ω from `pxform('J2000', 'IS901_BUS_FRAME', et)`, propagate with `propagate_euler(+ω_textbook)`, compare to `pxform` at 1s/10s/60s. Identity inertia isolates the q-kinematic.
- **A2 L_J2000 toy** (duplicate of s066's setup with rtol=1e-12, atol=1e-14): I=diag(1,2,3), ω₀=(0.5,0.3,0.7), q₀=identity, dt=50s. Drift = `max ||L_J(t) - L_J(0)||`.
- **A3 L_J2000 m048 cohort**: seeds 89 (slow), 28 (fast), 14 (near-separatrix), full LC, default tolerances. Drift sampled at 6 epochs across the LC.
- **A4 Jacobi parity**: same 3 seeds, propagate via both `propagate_attitude(mode="tumbling")` and `propagate_jacobi` (closed-form ω + DOP853 on q), compare antipode-aware. Both share the same q ODE so should match to integration noise.

# Result

```
[Gate A1] SPICE direction-aware comparison ...
  err@1s = 9.726e-11  (err@1s < 1e-9)  ->  PASS
    dt=  0.0s  err=3.141e-16
    dt=  1.0s  err=9.726e-11
    dt= 10.0s  err=9.726e-10
    dt= 60.0s  err=5.836e-09

[Gate A2] L_J2000 conservation on toy ...
  drift_abs_max = 7.833e-12  (0.0000% of |L_body|)  ->  PASS

[Gate A3] L_J2000 conservation on m048 cohort (3 seeds) ...
  seed 089: drift 6.673e-09 (0.0000%)  ->  PASS
  seed 028: drift 1.263e-07 (0.0000%)  ->  PASS
  seed 014: drift 2.799e-07 (0.0000%)  ->  PASS

[Gate A4] Jacobi ↔ propagate_euler parity (3 seeds) ...
  seed 089: q_diff_max = 1.589e-10  ->  PASS
  seed 028: q_diff_max = 1.643e-08  ->  PASS
  seed 014: q_diff_max = 1.796e-09  ->  PASS

OVERALL: PASS
```

# Why this matters

**Forward model is now physical.** L_J2000 is conserved at the DOP853 noise floor (1e-12 with tight tolerances; 1e-7 with default tolerances over 60 min). The propagator output corresponds to a real torque-free rigid body. Real telescope data intersection is no longer architecturally blocked.

**SPICE direction-aware compatibility.** Feeding `propagate_euler` the textbook body-frame ω (no negation) now matches `pxform` at 1e-10 per second — six orders of magnitude tighter than pre-fix. The s065 "feed `-omega` to match SPICE" workaround is no longer needed.

**Jacobi propagator continues to work.** Path 3 hybrid (closed-form ω + DOP853 on q) still matches `propagate_euler` to integration noise after the joint sign fix. s062a's closed-form ω(t) is unchanged (Euler is sign-agnostic at the RHS); only the hybrid q-path needed the matching update.

**s062b Path 2 (closed-form q(t)) is now unblocked.** With L_J2000 constant, the precession+nutation Euler decomposition has its required anchor. Re-attempting Path 2 is queued as future work.

**Cohort regen needed.** Cached `(q_hist, omega_hist)` in `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed*.npz` were generated with the buggy propagator. Same RNG seeds → same (q0, ω0) inputs → new physical trajectories. Surrogate is bridge-independent (`(k1_body, k2_body) → mag`); no retraining needed.

# Numbers

| Metric | Pre-fix | Post-fix | Improvement |
|--------|---------|----------|-------------|
| SPICE err @ 1 s, +ω_textbook | 2.04e-4 | 9.73e-11 | 2.1e+6× |
| L_J2000 toy drift, 50 s | 4.28 (191%) | 7.83e-12 | 5.5e+11× |
| L_J2000 m048 seed 89, 60 min | 97.2 (62.5%) | 6.67e-9 | 1.5e+10× |
| L_J2000 m048 seed 28, 60 min | 341.8 (36.4%) | 1.26e-7 | 2.7e+9× |
| L_J2000 m048 seed 14, 60 min | 902.7 (137%) | 2.80e-7 | 3.2e+9× |

# Artefacts

- `experiments/s067_postfix_propagator_validation.py`
- `results/s067_postfix_validation/summary.json`

# Out of scope

- Cohort regen (next step in the cleanup plan).
- Replication of pre-fix numerical findings (s068+ in the cleanup plan).
- s062b Path 2 closed-form q(t) re-attempt (future work).

# Cross-references

- `experiments/s065_propagator_convention_spice_test.{py,md}` — t=0 / 1s SPICE comparison; the `-omega` workaround was a precursor to this fix.
- `experiments/s066_lj2000_nonconservation.{py,md}` — L_J2000 finding that triggered the fix.
- `src/dynamics/attitude_propagator.py` — fixed.
- `lib/jacobi_propagator.py` — fixed.
- `concepts/quaternion_convention.md` — to be updated to reflect post-fix state.
- Auto-memory: `feedback_propagator_omega_sign_convention.md` — to be overturned.
