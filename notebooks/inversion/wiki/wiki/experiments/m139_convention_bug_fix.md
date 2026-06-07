---
title: "m139 — Quaternion convention bug in the inversion-side propagator: identify, fix, validate"
type: experiment
sources:
  - "src/dynamics/attitude_propagator.py"
  - "data/results/inversion_diagnostics/convention_bug_audit/step1_verify_convention.py"
  - "data/results/inversion_diagnostics/convention_bug_audit/step3_validate_fix.py"
  - "data/results/inversion_diagnostics/convention_bug_audit/step3_seed006_lc_compare.npz"
related:
  - "[[quaternion-convention]]"
  - "[[m048-migration]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
  - "[[surrogate-model]]"
  - "[[surrogate-rerank]]"
  - "[[twin-degeneracy]]"
  - "[[omega-sign-degeneracy]]"
  - "[[multi-solution-philosophy]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m139 — Quaternion convention bug: identify, fix, validate

#validated  #critical-bug

## TL;DR

The inversion-side attitude propagator (`src/dynamics/attitude_propagator.py`)
was integrating the kinematic equation
`dq/dt = (1/2) q ⊗ ω̄_body` (RIGHT multiply, Hamilton). That kinematic
yields `q` in **convention (b)** — `as_rotation_matrix(q) = R_b→i`
(passive body→J2000 matrix; equivalently `q` is the active body→inertial
rotation). The renderer everywhere downstream — both the inversion-side
`lc_compare.py:generate_hifi_lc` and the canonical main pipeline
(`compute_observation_geometry` with `attitude_keyframes='quaternion'`) —
expects **convention (a)**: `as_rotation_matrix(q) = R_i→b` (passive
J2000→body), with the formula `k1_body = R(q) @ sun_J2000`. Conventions
(a) and (b) are inverses: `q_a = q_b*`. Feeding `q_b` to a renderer that
expects `q_a` produces a physically meaningless `k1`/`k2`, and therefore a
physically meaningless light curve. **All inversion-side hi-fi LCs ever
produced — including every truth NPZ in `m046_trajectories.npz` and
`m048_trajectories/per_trajectory/traj_seed*.npz`, every `pred_lc.npy`
cache, every `mag_hifi` reported anywhere — share the same bug.**

The fix is two single-line edits to flip the multiplication order in the
kinematic and the principal-axis closed form. After the fix, the
inversion-side renderer agrees with the main pipeline at machine precision
(RMS 6.26e-15 mag, max 7.1e-14 mag on m048 seed 6).

This page documents the audit, the fix, and the validation. **Downstream
consequences (truth regeneration, cohort re-runs, inversion-code review)
are deliberately out of scope here and are scheduled for follow-on work.**

## What

A user-driven audit of the inversion-side forward model uncovered a
convention mismatch at the boundary between the propagator and the
renderer. The audit was carried out in three explicitly numbered steps:

1. **Step 1 — independently verify the bug exists** (without making any
   code changes), by checking three orthogonal claims:
   - scipy's `Rotation.from_quat(...).as_matrix()` and numpy-quaternion's
     `quaternion.as_rotation_matrix(...)` return the same active rotation
     matrix for the same `q`.
   - The propagator's RIGHT-multiply kinematic produces a trajectory that
     matches the convention-(b) prediction, not the convention-(a)
     prediction, when applied to a non-trivial `(q0, ω)` over a small
     finite-difference Euler step.
   - SPICE's `pxform("J2000", body_frame, et)` returns the passive
     J2000→body matrix (i.e., `M @ v_J2000 = v_body`).
2. **Step 2 — apply the fix** to `src/dynamics/attitude_propagator.py`
   (and only that file).
3. **Step 3 — validate the fix** by rendering the LC for one m048 seed
   two ways: Way A (the `lc_compare.py` style inversion-side renderer)
   and Way B (the canonical main-pipeline path
   `compute_observation_geometry(attitude_keyframes={'format':'quaternion',
   ...})`). Both should agree to machine precision because they share the
   same `R(q)` math.

## Step 1 — independent verification (results)

Script: `data/results/inversion_diagnostics/convention_bug_audit/step1_verify_convention.py`.

**T1 — scipy ↔ numpy-quaternion.** Drew 50 random unit quaternions and
converted each to a rotation matrix by both routes. Max element-wise
difference: `7.77e-16`. **PASS.**

**T2 — propagator RIGHT-mult is convention (b).** Set
`q0 = (cos45°, sin45°, 0, 0)` (90° about body-X), `ω_body = (0, 0, 1)
rad/s`, symmetric inertia (so `ω` is constant under Euler), `dt = 1e-4 s`.
Ran the unmodified propagator and compared `R(q_propagator)` element-wise
to:

- `R_b→i(dt)` under conv-(b) interpretation (body z = `(0, −1, 0)` in
  inertial; ω_inertial = `(0, −1, 0)`; rotate about inertial −y by `dt`).
- `R_i→b(dt)` under conv-(a) interpretation (body z = `(0, +1, 0)` in
  inertial; ω_inertial = `(0, +1, 0)`; rotate about inertial +y, take
  transpose).

Results:

| comparison | max abs error |
|---|---|
| `R(q_prop) − R_b→i (conv b)` | **2.22e-16** |
| `R(q_prop) − R_i→b (conv a)` | **1.00e-04** = O(dt) |

Finite-difference Euler-step quaternions confirm the same conclusion:

```
finite-diff q_pred RIGHT mult: [0.7071, 0.7071, -3.535e-5, +3.535e-5]
finite-diff q_pred LEFT  mult: [0.7071, 0.7071, +3.535e-5, +3.535e-5]
propagator q at dt           : [0.7071, 0.7071, -3.535e-5, +3.535e-5]
```

**PASS.** RIGHT-mult propagator output ↔ convention (b), not (a).

**T3 — SPICE `pxform('J2000', body, et)` is the J2000→body passive
matrix.** At `et = utc_to_et('2020-02-05T11:00:00')`:

| check | value |
|---|---|
| `\|M_J2b @ M_b2J − I\|` (transposes) | 2.22e-16 |
| `M_J2b @ sun_J2000` vs `sun_body` (independent SPICE query), max abs err | 7.45e-9 km |
| same, relative | 5.32e-17 |

**PASS.** SPICE direction is exactly J2000→body passive.

## Step 2 — the fix

`src/dynamics/attitude_propagator.py`. Two single-line edits + docstring
updates. No other source file changes.

| Function | Line | Before | After |
|---|---|---|---|
| `propagate_principal_axis` | 167 → 173 | `_quaternion_multiply(q0, q_rot)` | `_quaternion_multiply(q_rot, q0)` |
| `propagate_euler` | 287 → 298 | `0.5 * _quaternion_multiply(q, omega_quat)` | `0.5 * _quaternion_multiply(omega_quat, q)` |

Docstrings rewritten to state convention (a) explicitly:
`as_rotation_matrix(q) = passive J2000→body`; kinematic
`dq/dt = (1/2) ω̄_body ⊗ q` (LEFT mult, Hamilton).
`_quaternion_multiply` itself is unchanged (Hamilton is correct as
defined). Euler's equation `dω/dt = -I^-1 (ω × Iω)` is unchanged
(body-frame Euler is convention-independent).

## Step 3 — validation (results)

Script: `data/results/inversion_diagnostics/convention_bug_audit/step3_validate_fix.py`.
Subject: m048 seed 6.

After the fix, propagated `q(t)` once with the fixed propagator, then
rendered the LC two ways:

- **Way A** — inversion-side `lc_compare.py:generate_hifi_lc` style:
  `R = scipy.from_quat(q[xyzw]).as_matrix()`,
  `k1 = R @ (sun_J2000 − sat_J2000)/|·|`,
  `k2 = R @ (obs_J2000 − sat_J2000)/|·|`,
  → `compute_shadows` + `generate_lightcurves` with
  `art_matrices = compute_rotation_matrices_from_angles({SP=0, AD=15})`
  (the m048 truth-generation convention).
- **Way B** — main pipeline `compute_observation_geometry` with
  `attitude_keyframes={'times': observation_epochs, 'time_format': 'et',
  'attitudes': q(t), 'format': 'quaternion'}`.

Identical art_matrices and SPICE state in both paths.

| metric | value |
|---|---|
| RMS (Way A − Way B) | **6.26e-15 mag** |
| max abs (Way A − Way B) | **7.11e-14 mag** |
| max `\|k1_A − k1_B\|` | 4.44e-16 |
| max `\|k2_A − k2_B\|` | 6.11e-16 |
| max `\|obs_dist_A − obs_dist_B\|` | 0.0 |

LC range was `[6.5169, 16.4320]` mag. Residual is at the double-precision
floor for BRDF/shadow summation. **Fix is validated.**

## Why this matters

Both Way A and Way B share the same `R(q)` computation downstream of the
propagator, so this validation is a *self-consistency* check on the
inversion-side and main-pipeline renderers. It does not, by itself,
re-prove that the propagator is now in convention (a) — that part is
established by Step 1 (the RIGHT-mult kinematic is conv-(b), and the
LEFT-mult swap is the only change made). Together: Step 1 nails the
convention; Step 2 applies the prescribed swap; Step 3 confirms the
inversion and main-pipeline renderers behave identically given the same
attitude history. The next experiment in this thread (m140 in plan) is to
re-render *one* canonical seed end-to-end with the fixed forward model
and compare against the corresponding `truth NPZ.mag_hifi` — that
comparison is where the buggy-vs-correct LC difference becomes visible.

## What this implies for prior work

Every inversion-side hi-fi LC produced before this fix used
`R(q_b) @ sun_J2000`, where `q_b = q_a*`. The matrix `R(q_b)` is the
*transpose* of the matrix the renderer formula expects. The k1/k2 that
land in shadow ray-tracing and BRDF integration are therefore some
inertial-frame vectors that bear no clean physical interpretation — they
are NOT the sun/observer direction in the body frame.

Concretely, this means:

- **Truth NPZs (`m046_trajectories.npz`, `m048_trajectories/per_trajectory/*.npz`)**
  store `mag_hifi` LCs that were rendered under the bug. `q0_wxyz` and
  `omega0_rad` are still well-defined initial conditions; the
  `quaternions[t]` arrays are conv-(b) trajectories. The LC field
  `mag_hifi` is internally consistent (same bug everywhere) but is NOT
  the LC a real telescope would observe for the stored `(q0, ω0, I)`.
- **`pred_lc.npy` caches** under `data/results/inversion_diagnostics/*/`
  inherit the bug.
- **All inversion verdicts (m070 → m138)** classified candidates against
  these buggy truth LCs. The classifications are internally consistent —
  if a candidate `(q0, ω0)` reproduces the buggy LC, that's a real
  attractor of the buggy forward model — but they are not classifications
  against physical truth.
- **The surrogate model** was trained on data generated under the buggy
  convention (per [[surrogate-model]] origin notes). The surrogate
  *learned* the buggy forward model's mapping `(q0, ω, t) → mag`. Its
  internal consistency with the inversion-side renderer is not in
  question; its physical correctness is.
- **All α/β-pipeline successes, all ρ-band yield numbers, all per-seed
  basin classifications** are properties of the buggy forward model.
  Whether they survive under the fixed forward model is an empirical
  question (m140+).

The convention error is a sign-equivalent error: q and q* differ in the
imaginary part. Operationally this is the SAME class of error as
applying R^T instead of R when transforming a vector — exactly the kind
of bug that produces internally-consistent but physically-wrong results.

## What this does NOT change

- `_quaternion_multiply` is correct (standard Hamilton).
- Euler's equation `dω/dt = -I^-1 (ω × Iω)` is correct (body frame,
  convention-independent).
- The shadow ray-tracing engine and Ashikhmin-Shirley BRDF code are
  correct.
- The renderer formula `k1 = R @ sun_J2000` (in both
  `compute_observation_geometry` and `lc_compare.py:generate_hifi_lc`)
  is correct *given* `R = R_i→b`. The bug was upstream — the propagator
  fed an `R_b→i` matrix into a slot that expected `R_i→b`.
- SPICE-driven attitude paths in the **main pipeline** were always
  correct (they use `pxform("J2000", body, et)` directly, which IS the
  J2000→body passive matrix, no quaternion intermediary). The bug was
  confined to the custom-quaternion path on the inversion side.

## Artefacts

- `src/dynamics/attitude_propagator.py` — the fix (commit referenced in
  log).
- `data/results/inversion_diagnostics/convention_bug_audit/`:
  - `step1_verify_convention.py` — Step 1 script (T1/T2/T3).
  - `step3_validate_fix.py` — Step 3 validation on m048 seed 6.
  - `step3_seed006_lc_compare.npz` — quats, mag_way_a, mag_way_b, RMS,
    max_abs.

## Out of scope here (planned follow-ons)

- **m140** — render one canonical seed (m048 seed 6) end-to-end with the
  fixed forward model and compare against the existing truth NPZ
  `mag_hifi` to *quantify* the buggy-vs-correct LC delta.
- **Truth-NPZ regeneration** — re-run `m048_generate_trajectories_v2.py`
  (and m046 equivalent) with the fix to produce physically correct truth
  databases.
- **Inversion-code review** — every place that calls
  `propagate_attitude` and consumes its `q` output, audit for any code
  that relied (knowingly or accidentally) on conv-(b) semantics. Likely
  candidates: any code computing `R(q).T @ sun_J2000` (which would have
  been the *correct* conv-(b)-aware way to get `sun_body`), any
  hand-rolled SLERP, anything that converts q ↔ axis-angle ↔ rotation
  matrix and assumes a specific convention.
- **Old-ideas re-explore** — branches that were marked dead-end
  (`crossing-geometry-scoring`, `ipl-candidate-generation`,
  `pa-mode-bridge`, `multi-epoch-winding`, etc.) may have failed because
  the cost surface they computed was on the buggy forward model. Worth
  spot-checking after the truth NPZs are regenerated.
- **Surrogate retraining** — the surrogate model was trained on buggy
  forward-model outputs. Whether retraining on correct truth LCs changes
  its accuracy is an open question.
- **Convention gates in the code** — add runtime assertions at the
  propagator/renderer boundary that flag a convention mismatch (e.g.,
  feed a known `(q0=identity, ω=0)` and assert `R(q(t)) ≈ I` for all t,
  or compare a SPICE-driven attitude history against a propagated one
  for a known-equivalent dynamics setup, and assert agreement).

## Cross-references

- [[quaternion-convention]] — the concept page that names and pins down
  the two conventions, the inverse relationship, and which one this
  codebase uses where.
- [[m048-migration]] — the cohort whose truth NPZs are now known to be
  buggy (m048 generator code path is in scope).
- [[surrogate-model]] — trained on buggy-forward-model data; retraining
  is a separate question.
- [[upstream-redesign-6dof-surrogate-de]] — the proposed Play-3 path; if
  pursued, the upstream stage must consume convention-(a) quaternions
  end-to-end.
