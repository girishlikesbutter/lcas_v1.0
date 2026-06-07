---
title: "Quaternion convention (a vs b) at the propagator/renderer boundary"
type: concept
sources:
  - "src/dynamics/attitude_propagator.py"
  - "src/computation/observation_geometry.py"
  - "src/interpolation/attitude_interpolator.py"
  - "notebooks/inversion/lib/lc_compare.py"
  - "notebooks/inversion/09_glint_analysis/m048_generate_trajectories_v2.py"
related:
  - "[[m139_convention_bug_fix]]"
  - "[[surrogate-model]]"
  - "[[twin-degeneracy]]"
  - "[[omega-sign-degeneracy]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# Quaternion convention (a vs b) at the propagator/renderer boundary

> **The bug fixed in [[m139_convention_bug_fix]] was a mismatch between
> these two conventions at the propagator/renderer boundary. Read this
> page before touching anything that converts a quaternion to a rotation
> matrix or integrates the kinematic equation.**

## Two conventions, one matrix

Both `scipy.spatial.transform.Rotation.from_quat(q).as_matrix()` and
`quaternion.as_rotation_matrix(q)` return the same matrix for the same
`q` (modulo `xyzw` vs `wxyz` layout) — the *active rotation matrix*
associated with `q`. Whether that matrix is interpreted as a passive
J2000→body transformation or a passive body→J2000 transformation is a
**convention choice** at the call site. Both interpretations are
mathematically valid; mixing them in a single code path is the bug.

### Convention (a) — what this codebase uses

`as_rotation_matrix(q) = passive J2000 → body matrix`.

Equivalently:

- `q` is the **active inertial → body rotation**.
- `R(q) @ v_J2000 = v_body_components`. (Renderer formula
  `k1 = att_matrix @ sun_J2000` is correct under this convention.)
- Kinematic with body-frame ω: `dq/dt = (1/2) ω̄_body ⊗ q` — **LEFT
  multiply, Hamilton**.
- Closed-form principal-axis solution: `q(t) = q_rot ⊗ q0` with
  `q_rot = exp(0.5 * ω_body * t)` — LEFT multiply.
- This matches what SPICE's `pxform("J2000", body_frame, et)` returns
  directly as a matrix; SPICE never goes through the quaternion path
  here.

### Convention (b) — what the propagator USED to produce (pre-fix)

`as_rotation_matrix(q) = passive body → J2000 matrix`.

Equivalently:

- `q` is the **active body → inertial rotation**.
- `R(q) @ v_body = v_J2000_components`.
- Kinematic with body-frame ω: `dq/dt = (1/2) q ⊗ ω̄_body` — RIGHT
  multiply, Hamilton.
- Closed-form principal-axis solution: `q(t) = q0 ⊗ q_rot` — RIGHT
  multiply.

### The two are inverses

For the same physical attitude, `q_a = q_b*` (quaternion conjugate),
and `R(q_a) = R(q_b)^T` (matrix transpose). They are not the same
quaternion; they represent the same rotation under two conventions.

## Where each convention shows up in this codebase

| Code site | Convention | Notes |
|---|---|---|
| `src/dynamics/attitude_propagator.py` | (a) post-fix; (b) pre-fix | The bug. Fixed in m139 by swapping multiplication order in `propagate_euler` (line 287→298) and `propagate_principal_axis` (line 167→173). |
| `src/spice/spice_handler.py` (`get_target_orientation`, `pxform`) | (a) | Returns the matrix directly; never builds a quaternion. |
| `src/computation/observation_geometry.py` | (a) | `att_matrix @ sun_vector_j2000`. Path consistent with both SPICE matrix and conv-(a) quaternion. |
| `src/interpolation/attitude_interpolator.py` | (a) | `quaternion.as_rotation_matrix(q)` for SLERPed quaternion → directly used as J2000→body in `compute_observation_geometry`. Convention-(a) is required of the input quaternion. |
| `notebooks/inversion/lib/lc_compare.py` (`generate_hifi_lc`) | (a) | `R = scipy.from_quat(q[xyzw]).as_matrix(); k1 = R @ sun_J2000`. Convention-(a) required of input. |
| `notebooks/inversion/09_glint_analysis/m048_generate_trajectories_v2.py` | (a) | Same renderer pattern. Convention-(a) required of input. |
| `notebooks/lcas_stl_pipeline-is901-3.py` (custom-quat path) | (a) | Same downstream. |

The propagator is the **only** place a convention choice was being made
silently; everything else assumes the input quaternion is in convention
(a) and uses it directly in the renderer formula.

## How to verify a convention-consistent boundary (recipe)

When in doubt, run this 60-second smoke test on any new code that
produces or consumes quaternions:

1. Pick a non-trivial `q0` and `ω` — e.g., 90° about body-X with
   `ω = (0, 0, 1)` rad/s.
2. Propagate for a small `dt = 1e-4 s` with symmetric inertia (so ω is
   conserved).
3. Compute `R(q(dt))` via `scipy.from_quat(...).as_matrix()`.
4. Construct the analytical `R_i→b(dt)` and `R_b→i(dt)` predictions
   (both derivable in 5 lines from the body-axes-in-inertial picture
   and the rotation about the inertial-frame ω vector).
5. Confirm `R(q(dt))` matches `R_i→b(dt)` to machine precision (under
   convention (a)) and differs from `R_b→i(dt)` by O(dt).

The reusable script lives at
`data/results/inversion_diagnostics/convention_bug_audit/step1_verify_convention.py`.

## Recommended runtime gates (proposed, not yet implemented)

To make this class of bug structurally impossible, the propagator and
the renderer should each carry an assertion at boundary entry:

- **Propagator output gate.** After the first integration step, render
  the SPICE-equivalent attitude (start with `q0 = quaternion.from_rotation_matrix(spice_M_J2000_to_body)`,
  `ω = 0`) and confirm that propagating with `ω = 0` for `t = 1 s` produces
  `R(q(1)) ≈ R(q0) ≈ spice_M_J2000_to_body` (rotation-free). If the
  multiplication order is wrong, this would still be identity (since ω=0)
  — so the more discriminating check is propagating with a nonzero ω
  for which `R_i→b(t)` and `R_b→i(t)` differ visibly, then asserting
  match against the expected branch.
- **Renderer input gate.** When `compute_observation_geometry` receives
  `attitude_keyframes`, optionally accept a sentinel epoch where SPICE
  attitude is also available, and assert
  `as_rotation_matrix(q_keyframe) ≈ pxform("J2000", body, et)` to
  machine precision. Loud failure if not.

These are not in place today; m139 was an audit + fix, not an
infrastructure-hardening pass. Adding gates is part of the m141+ scope.

## Pitfalls when reading the literature

Quaternion-rotation conventions in textbooks vary widely:

- **Kuipers** uses convention-(a)-like notation but with
  q_b/i for body-relative-to-inertial; the kinematic in some derivations
  has the opposite sign of what you'd expect because of the way
  body-frame ω is defined (active vs passive interpretation of ω
  relative to the rotating body).
- **Markley & Crassidis** (and most spacecraft GNC textbooks) use a
  non-Hamilton multiplication convention (sometimes called the JPL or
  Shuster convention). Their equations look like ours up to a sign that
  is absorbed into the multiplication operator.
- **Wikipedia's "Quaternions and spatial rotation"** uses Hamilton with
  `v' = q v q*` for active rotation, and explicitly distinguishes
  active/passive.

The verification recipe above sidesteps all of this — it is a numerical
finite-difference check, not a textbook lookup.

## Cross-references

- [[m139_convention_bug_fix]] — the audit + fix experiment.
- [[surrogate-model]] — trained under the bug; retraining question is
  open.
- [[twin-degeneracy]] / [[omega-sign-degeneracy]] — pre-existing
  degeneracies; whether the buggy forward model spuriously preserved
  some degeneracies that aren't real, or hid some that are, is an
  empirical follow-up.
