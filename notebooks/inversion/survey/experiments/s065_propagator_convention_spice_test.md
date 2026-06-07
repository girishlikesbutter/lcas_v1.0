---
title: "s065 — Propagator-vs-SPICE convention test (resolves s062b puzzle)"
type: experiment
sources:
  - src/dynamics/attitude_propagator.py
  - src/spice/spice_handler.py
  - data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm
related:
  - experiments/s062b_jacobi_quaternion.md
  - concepts/quaternion_convention.md
  - feedback_verify_quaternion_convention.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (machine-precision SPICE comparison, reproducible script)
---

# TL;DR

The s062b "convention puzzle" (`L_J2000` not conserved by the propagator) is resolved.
The propagator's quaternion **convention is correct** — `scipy.from_quat(q[xyzw]).as_matrix()`
matches SPICE's `pxform('J2000', body, et)` to 3e-16 at t=0 on a real IS-901 epoch.
**But the kinematic ODE has the opposite sign of the textbook passive-J2000→body convention**,
so feeding the propagator the textbook body-frame ω produces a rotation that drifts at
~2e-4 per second from SPICE. **Negating ω closes that to 9.7e-11 per second** — machine
precision over a 60 s window.

Practical meaning: the symbol `ω` inside `propagate_euler` gets a hidden minus sign
relative to the physical right-hand body-frame angular velocity. Cached
`q0_wxyz` is the physical initial attitude; cached `omega0_rad` is the **negation** of
the physical body-frame angular velocity. m048 LCs are valid physical light curves of
satellites spinning at `−omega0_rad`. STL geometry, BRDF, surrogate, and inversion
comparisons are all internally consistent under this convention.

This is a bookkeeping choice baked into the codebase, not a geometric fidelity problem.

# What

The 2026-04-30 convention bug fix made the propagator and renderer agree on a common
q-interpretation. m139 audit validated `propagate_euler` against SPICE-driven IS-901
runs "at machine precision." s062b found a puzzle: `L_J2000 = R(q).T @ L_body` is not
conserved by the propagator output on either a real seed or a controlled toy, even
though torque-free physics demands conservation. The s062b hand-off recipe proposed
testing the flipped-sign kinematic `dq/dt = −0.5 ω ⊗ q (LEFT)`. s065 ran that test
against SPICE directly and resolves what the actual situation is.

# How

For a single epoch `t0 = 2020-02-05T10:00:00 UTC` in the IS-901 kernel coverage:

1. `R_SPICE(t0) = pxform('J2000', 'IS901_BUS_FRAME', t0)` — the SPICE-definitional
   passive J2000→body matrix. Cross-checked: `det = 1` to 3e-16.
2. Seed the propagator with the q that matches the codebase consumer pattern:
   `q0_wxyz` such that `scipy.from_quat(q[xyzw]).as_matrix() == R_SPICE(t0)`. This
   is the q an honest `propagate_euler` would produce at t=0 for this attitude.
3. Compute textbook body-frame ω via finite difference of SPICE: the kinematic
   `dR/dt = −[ω]_× R` (R passive J2000→body) gives `[ω]_× = −(dR/dt) R^T`.
4. Propagate via `propagate_euler` with `+ω_textbook` and `−ω_textbook` to
   `dt ∈ {0, 1, 10, 60}` s under identity inertia (isolates the q-kinematic
   from Euler dynamics — over 60 s with `|ω| ≈ 7e-5 rad/s` and any reasonable
   inertia, ω evolution is negligible).
5. At each `dt`, compute the propagator's R via the codebase consumer pattern and
   compare to `pxform('J2000', body, t0 + dt)` directly.

Script: `experiments/s065_propagator_convention_spice_test.py`.

# Result

```
t0 (ET):                 634168869.185
q0 from R0 (wxyz):       [ 0.92219803 -0.00391478 -0.00951702  0.38658103]
omega_textbook (rad/s):  [-1.37e-06  1.66e-06  7.21e-05]   |w|=7.22e-05

--- feeding propagator +omega_textbook (current convention) ---
     dt      ||R_prop - R_SPICE||
     0.0 s       3.141e-16
     1.0 s       2.041e-04
    10.0 s       2.041e-03
    60.0 s       1.225e-02

--- feeding propagator -omega_textbook (matches SPICE) ---
     dt      ||R_prop - R_SPICE||
     0.0 s       3.141e-16
     1.0 s       9.726e-11
    10.0 s       9.726e-10
    60.0 s       5.836e-09
```

At t=0 the q-interpretation is correct (3e-16). With `+ω` (current calling
convention) the propagator's R drifts at ~2e-4 per second; with `−ω` it drifts
at ~1e-10 per second — six orders of magnitude tighter, and consistent with
DOP853 integration noise over the 60 s window.

# Why this matters

The s062b memo speculated "if flipped sign conserves L, propagator has a long-standing
sign bug requiring fix + cohort regeneration." s065 confirms the flipped-sign relationship
empirically against SPICE — but the correct interpretation is the **less invasive** one:

- **q is correct.** Every cached `q0_wxyz` in the m048 cohort IS the physical initial
  attitude (in the SPICE convention). Same convention the STL/BRDF/articulation/SPICE
  geometry pipeline expects.
- **ω is sign-flipped.** The propagator's kinematic `dq/dt = +0.5 ω ⊗ q (LEFT)` is
  inconsistent with the textbook passive-J2000→body kinematic by a sign on ω. For the
  propagator's q-output to match SPICE, the input ω must be negated.
- **All downstream pipeline is consistent.** Because the m048 cohort generator
  (`m048_generate_trajectories_v2.py`) calls `propagate_attitude` with the same
  cached `omega0_rad` that's later stored in the NPZ, both the cached LC and the
  cached ω label are under the same convention. Inversion comparing recovered
  ω against cached ω will always be self-consistent.
- **Implication for real-satellite physics.** To predict the LC of a satellite
  spinning physically at `ω_phys` in standard right-hand body-frame ω, you must
  call `propagate_attitude(..., omega0=-ω_phys, ...)`. Equivalently: an `omega0_rad`
  value in any m048 NPZ corresponds to a physical body-frame angular velocity of
  `−omega0_rad`. q is unchanged.
- **No regeneration needed.** The cached LCs are real, physical light curves of
  real, physical satellites — under the relabeling `ω_cache = −ω_phys`. STL geometry
  is rendered against a physically correct body frame (because q is correct).
  The polhode invariants, |L|, 2T, basin widths, polhode-diameter scaling rules,
  etc. are all invariant under `ω → −ω`. Every numerical finding in s001…s063
  stands.

# Why m139 didn't catch it

The audit smoke test in `concepts/quaternion_convention.md` only verifies that a
body-fixed vector's J2000 trace under `R.T @ v_body` is non-zero ("`> 0.99 * dt`").
It checks magnitude of motion, not direction. The sign-flipped kinematic still
gives motion of the right magnitude — it just goes the wrong way. A test that
checks the cross-product direction `v_J1 − v_J0 ≈ (ω × v_J0) dt`, or any test
against SPICE `pxform`, catches it immediately.

# Numbers

See "Result" above. Per-row numerical interpretation:

- `||R_prop − R_SPICE|| = 3.141e-16` at dt=0 is the float64 round-off in
  `from_matrix → as_quat → from_quat → as_matrix` round trip. q-interpretation is
  bit-exact.
- `||R_prop − R_SPICE|| = 2.041e-04` at dt=1 s with `+ω` is `|2ω·dt|` for
  `|ω| ≈ 7.22e-5 rad/s` — the kinematic sign error grows linearly with `2ω·dt`
  (each "ω in disagreement" perturbs R by approximately `|ω·dt|` per second; the
  cumulative discrepancy across 1 s of "wrong-direction" motion is `2|ω|·dt`).
- `||R_prop − R_SPICE|| = 9.726e-11` at dt=1 s with `−ω` is DOP853 integration
  noise. Six orders of magnitude tighter.

# Out of scope

- Fixing the propagator. User chose to leave the code as-is and document the
  convention (option 1, decided 2026-05-12).
- Re-running m139 audit with a direction-aware smoke test. Same direction-aware
  test now exists at `experiments/s065_propagator_convention_spice_test.py`
  and is the canonical reference going forward.
- Resolving why `dq/dt = +0.5 ω ⊗ q (LEFT)` was used instead of the textbook
  `−0.5 ω ⊗ q (LEFT)`. Historical — the kinematic was in place before the 2026-04-30
  bug fix and the fix preserved it.

# Artefacts

- Script: `experiments/s065_propagator_convention_spice_test.py` — runs in ~3 seconds
  with SPICE kernels loaded; reproduces the table above.

# Cross-references

- `experiments/s062b_jacobi_quaternion.md` — supersedes the "hand-off recipe" there.
- `concepts/quaternion_convention.md` — updated with the s065 finding.
- `src/dynamics/attitude_propagator.py` — docstring updated with the ω-sign note.
- Auto-memory: `feedback_propagator_omega_sign_convention.md` (new).
