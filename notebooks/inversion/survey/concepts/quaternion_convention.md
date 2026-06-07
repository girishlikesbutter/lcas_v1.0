---
title: "Quaternion convention — two sign bugs (2026-04-30 + 2026-05-12), both fixed; what every renderer expects"
type: concept
created: 2026-04-30
updated: 2026-05-12
confidence: high
---

> **STATUS (2026-05-12) — FIXED.** The propagator has had two sign bugs in its
> history. Both are fixed; m048 cohort has been regenerated under the post-fix
> propagator. SPICE-driven runs (`lcas_stl_pipeline-is901-3.py` and friends)
> have always been correct because they bypass `propagate_euler` via `pxform`.
>
> 1. **Pre-2026-04-30 — q-conjugation bug.** `dq/dt = (1/2) q ⊗ ω̄_body` (conv-(b))
>    while the renderer used `R(q) = passive J2000→body` (conv-(a)). Per-epoch
>    every body-frame sun/observer direction was the conjugate of the correct one.
>    Fixed in commit f7fabbe by reordering `_quaternion_multiply` calls.
> 2. **Pre-2026-05-12 — q-kinematic-sign bug.** Post-2026-04-30 the kinematic
>    became `dq/dt = +0.5 ω ⊗ q (LEFT)` — opposite sign of the textbook conv-(a)
>    kinematic `dq/dt = -0.5 ω ⊗ q (LEFT)`. q-interpretation matched SPICE at t=0
>    (3e-16) and the local 1s window matched SPICE under the ω-sign substitution
>    (s065). But over an LC the L_J2000 drifted 36–137% on the m048 cohort (s066).
>    Fixed in commit d5705ff by `+0.5 → -0.5` in both production propagators
>    + the Jacobi hybrid. Cohort regenerated. s067 4/4 gates PASS.
>
> **Operational meaning post-2026-05-12 fix:**
> - Forward model is now PHYSICAL — propagator output corresponds to a real
>   torque-free rigid body (L_J2000 conserved to DOP853 noise floor over 60 min).
> - Real telescope data intersection is no longer architecturally blocked.
> - Pre-fix cached results (s001..s066) need numerical replication; methodology
>   and qualitative findings survive forward-model-agnostically.
> - Path 2 closed-form q(t) (s062b redux) is now UNBLOCKED — L_J2000 has its
>   anchor for the precession+nutation Euler decomposition.

# Quaternion convention

## The two conventions

Let `R(q)` be the rotation matrix derived from a unit quaternion `q`. Two conventions exist for the relationship between `q`'s integration ODE and the resulting `R(q)`:

- **Convention (a)** — `R(q) = R_{i→b}` (J2000 → body, **active interpretation**: takes a J2000-frame vector and returns its body-frame components). The kinematic ODE is `dq/dt = (1/2) ω̄_body ⊗ q` (LEFT-multiplication of `ω̄`).
- **Convention (b)** — `R(q) = R_{b→i}` (body → J2000). The kinematic ODE is `dq/dt = (1/2) q ⊗ ω̄_body` (RIGHT-multiplication of `ω̄`).

The two are exact inverses: `R_a(q) = R_b(q)^T = R_b(q*)`. Equivalently `q_a = q_b*` (conjugation).

## What the LCAS renderer expects

Every downstream renderer in this codebase does:

```python
R = quaternion_to_matrix(q)
k1_body = R @ sun_J2000          # body-frame sun direction
k2_body = R @ obs_J2000          # body-frame observer direction
```

This is correct **only under convention (a)**. With conv-(b) it produces `R_{b→i} @ sun_J2000`, which is geometric nonsense.

## The bug (pre-2026-04-30)

`src/dynamics/attitude_propagator.py` integrated `dq/dt = (1/2) q ⊗ ω̄_body` — convention (b) kinematic — for both `propagate_principal_axis` and `propagate_euler`. Renderer was always applying conv-(a) formula. **Per-epoch every body-frame sun/observer direction was the conjugate of the correct one.**

Fix is two single-line edits inside `_quaternion_multiply` ordering: line 167→173 (`q0 ⊗ q_rot` → `q_rot ⊗ q0`) and line 287→298 (`q ⊗ ω̄` → `ω̄ ⊗ q`). Validated end-to-end at machine precision (m139 audit). Commit `f7fabbe`.

SPICE-driven main-pipeline runs were always correct — they take `pxform('J2000', body, et)` directly and never touch quaternions.

## What the bug DID across a trajectory

Per m146 cohort comparison: quaternion geodesic p99 ≈ 178.7° on 100/100 seeds — the bug is approximately `q → q*` per epoch. Bug-vs-postfix LC delta is Band D on every seed (ρ median 48), with bright-peak displacement median 19 epochs. Phase-angle modulated: high-PA seeds had larger LC deltas (specular geometry sensitive to attitude scrambling); low-PA seeds had smaller deltas (Lambertian dominates).

## Verifying any new q→matrix code (DIRECTION-AWARE smoke test, post-s065)

Before trusting any code that converts `q` to `R` or computes `R @ v` or `R.T @ v`,
run a direction-aware smoke test. The original smoke test (preserved below) only
checked magnitude of motion (`> 0.99 * dt`) and missed the omega-sign residual that
s065 surfaced. The direction-aware version compares against SPICE `pxform`, which
is the canonical reference.

```python
import numpy as np
import spiceypy as spice
from pathlib import Path
from scipy.spatial.transform import Rotation
from src.spice.spice_handler import SpiceHandler
from src.dynamics.attitude_propagator import propagate_euler

# Load IS-901 SPICE kernels
sh = SpiceHandler()
sh.load_metakernel_programmatically(
    str(Path("data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm")))
t0 = sh.utc_to_et("2020-02-05T10:00:00")

# SPICE reference R(t0): passive J2000->body
R0 = np.asarray(spice.pxform("J2000", "IS901_BUS_FRAME", t0))

# Seed q from R0 using the codebase convention (scipy.as_matrix(q) == R0)
q_xyzw = Rotation.from_matrix(R0).as_quat()
q0_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

# Textbook body-frame omega from finite-diff of SPICE: dR/dt = -[omega]_x R
R0p = np.asarray(spice.pxform("J2000", "IS901_BUS_FRAME", t0 + 0.01))
om_skew = -((R0p - R0) / 0.01) @ R0.T
omega = np.array([om_skew[2, 1], om_skew[0, 2], om_skew[1, 0]])

# Propagate with +omega and check against SPICE at t0 + 1 s
times = np.array([0.0, 1.0])
q_hist, _ = propagate_euler(q0_wxyz, +omega, np.eye(3) * 1000.0, times,
                            rtol=1e-12, atol=1e-14)
R_prop = Rotation.from_quat([q_hist[1, 1], q_hist[1, 2], q_hist[1, 3], q_hist[1, 0]]).as_matrix()
R_ref  = np.asarray(spice.pxform("J2000", "IS901_BUS_FRAME", t0 + 1.0))

err = np.linalg.norm(R_prop - R_ref)
print(f"||R_prop - R_SPICE|| = {err:.3e}")
# Currently this prints ~2e-4 (omega-sign residual, s065).
# After feeding -omega instead, it prints ~1e-10 (machine precision).
# A propagator with textbook-standard kinematic + textbook omega would
# also print ~1e-10.
```

Reproducible script: `experiments/s065_propagator_convention_spice_test.py`.

### Original (magnitude-only) smoke test — kept for reference

This is the original smoke test from the 2026-04-30 bug-fix doc. It catches the
q-conjugation bug (the original 2026-04-30 issue) but is insufficient to catch
the omega-sign residual surfaced by s065.

```python
import numpy as np
from scipy.spatial.transform import Rotation
from src.dynamics.attitude_propagator import propagate_attitude

dt = 1e-4
q0 = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z) = identity
omega = np.array([0.0, 0.0, 1.0])    # 1 rad/s about body Z
quats, _ = propagate_attitude(q0, omega, np.array([0.0, dt]), "tumbling", np.eye(3))

R0 = Rotation.from_quat(quats[0, [1, 2, 3, 0]]).as_matrix()  # xyzw input
R1 = Rotation.from_quat(quats[1, [1, 2, 3, 0]]).as_matrix()
v_body = np.array([1.0, 0.0, 0.0])
v_J0 = R0.T @ v_body
v_J1 = R1.T @ v_body
assert np.linalg.norm(v_J1 - v_J0) > 0.99 * dt  # MAGNITUDE only
```

If the code instead applies `R @ v_body` for body→J2000, it's silently using conv-(b), and pre-fix it happened to work because everything was wrong consistently.

## Historical: q-kinematic-sign bug (s065 → s066 → s067), fixed 2026-05-12

The 2026-04-30 fix established that `scipy.from_quat(q[xyzw]).as_matrix()`
correctly represents the passive J2000→body matrix at any single instant — the
q-interpretation is right. What the audit did NOT check is whether the
propagator's q **evolves** under the textbook kinematic for that
q-interpretation. s065 found it does not.

The propagator integrated `dq/dt = +0.5 omega_body ⊗ q (LEFT)`. For the
q-convention above (`as_matrix(q) == passive J2000→body`), the textbook
kinematic is `dq/dt = -0.5 omega_body ⊗ q (LEFT)`. The signs differed.

### s065 (initial framing, superseded)

Empirically (IS-901, dt=1 s): feeding `propagate_euler` the textbook body-frame
omega gave R drifting from SPICE at ~2e-4 per second; feeding `-omega` gave
R matching SPICE at ~1e-10 per second. s065 concluded: "the kinematic sign
issue is absorbed by feeding `-omega`; document, don't fix."

### s066 (load-bearing measurement)

`L_J2000 = R(t).T @ I @ omega(t)` over the full 60-min m048 LC, real torque-free
physics requires exact conservation. Cohort drift:

| Seed | \|ω₀\| dps | LC duration | L_J2000 drift |
|------|-----:|-----:|-----:|
| 89 | 0.240 | 60 min | 62.5% |
| 28 | 1.438 | 60 min | 36.4% |
| 14 | 1.229 | 60 min | 136.7% |

Control (independent textbook integrator on toy): conserves to 1e-12. Codebase
on same toy: 191% drift over 50 s. Mechanism: codebase paired textbook Euler
(correct, RHS even in ω) with opposite-sign Hamilton q-kinematic. ω → -ω
makes the kinematic textbook *locally* but Euler is sign-asymmetric in time
(starting from ±ω₀ shares initial dω/dt but diverges thereafter). No global
sign substitution closes both at once. Decision flipped: fix + regen.

### s067 (fix + validation, 2026-05-12)

Fix: `+0.5 → -0.5` in `propagate_euler` (line 318), `propagate_principal_axis`
(lines 177-184, negate sin in q_rot), and `lib/jacobi_propagator.py:432`
(matching update in Jacobi hybrid q-ODE). Bundled regression test
(`experiments/s067_postfix_propagator_validation.py`) — 4 gates, all PASS:

| Gate | Pre-fix | Post-fix |
|------|---------|----------|
| A1 SPICE pxform @ dt=1s, +ω_textbook | 2.04e-4 | 9.73e-11 |
| A2 L_J2000 toy drift over 50s | 4.28 (191% rel) | 7.83e-12 |
| A3 L_J2000 m048 cohort drift | 36-137% rel | 1e-9 to 3e-7 (~0% rel) |
| A4 Jacobi ↔ propagate_euler parity | n/a | 1.6e-10 to 1.6e-8 |

Cohort regenerated under post-fix propagator (commit TBD). Pre-fix files
preserved at `data/results/inversion_diagnostics/m048_trajectories/_s066buggy_archive/`.
Pre-fix `survey/results/` archived to `survey/results_prefix/` for traceability.

References: commits f7fabbe (2026-04-30 conjugation fix) and d5705ff
(2026-05-12 sign fix). `experiments/s065_propagator_convention_spice_test.{py,md}`,
`experiments/s066_lj2000_nonconservation.{py,md}`,
`experiments/s067_postfix_propagator_validation.{py,md}`.

## What's silently safe vs silently wrong

- **Silently safe (the "self-healing" pattern):** `R = as_matrix(q); v_body = R @ v_J2000`. Correct under conv-(a). The codebase overwhelmingly uses this — it was wrong pre-fix (because q was conv-(b)), correct post-fix.
- **Silently wrong post-fix:** explicit `.T` patterns like `R.T @ v_J2000` to extract body-frame, or `R @ v_body` to compute J2000 vectors. These were bug-compensating pre-fix and are now broken. m140 audit found one production case (`lib/attitude_viz.py:361,363`); fixed in same session.

## Cross-references

- Bug-fix commit: `f7fabbe`
- Cohort regen: `ac1fdf4`
- m139 audit: `notebooks/inversion/wiki/wiki/experiments/m139_convention_bug_fix.md` (frozen reference)
- m146 population delta: `data/results/inversion_diagnostics/m048_buggy_vs_postfix_compare_2026_04_30/`
- Auto-memory entry: `feedback_verify_quaternion_convention.md`
