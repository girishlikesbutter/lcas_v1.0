---
title: "s066 — L_J2000 non-conservation: m048 codebase trajectories are non-physical"
type: experiment
sources:
  - src/dynamics/attitude_propagator.py
  - experiments/s065_propagator_convention_spice_test.md
  - experiments/s062b_jacobi_quaternion.md
  - concepts/quaternion_convention.md
related:
  - feedback_propagator_omega_sign_convention.md
  - project_convention_bug_2026_04_30.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (empirical, reproduced on a controlled toy AND the m048 cohort, with an independent textbook integrator as control)
---

# TL;DR

The codebase's coupled q-omega ODE does NOT conserve L_J2000 (the J2000-frame angular momentum). On the m048 cohort it drifts by **36–137%** of |L_body| over a 60-min LC. Real torque-free physics demands `dL_J2000/dt = 0`. The cohort trajectories therefore do NOT correspond to any real tumbling rigid body. The s065 framing ("omega-sign convention residual; m048 LCs are real physical LCs of satellites at `-omega0_rad`") is **understated** — it holds only at t=0 and approximately over short ~1 s windows, not over the LC duration.

The forward model is still self-consistent (deterministic, reproducible, invertible), so the inversion benchmark is well-posed and methodology / qualitative findings carry over. But: for any future use against real telescope data, the propagator must be fixed and the cohort regenerated.

# What

s065 (2026-05-12) resolved the s062b convention puzzle as an "omega-sign residual" — a bookkeeping convention internally consistent across the codebase, validated by a SPICE comparison that matched `pxform('J2000', body, et)` at 3e-16 at t=0 and at 1e-10/s over a 1-second window. The decision was: document, don't fix.

This experiment (s066) tested **a different invariant** — L_J2000 conservation over the full LC duration — on:
- A controlled asymmetric toy (I = diag(1, 2, 3), ω₀ = (0.5, 0.3, 0.7), q₀ = identity, dt = 50 s) with an independent textbook integrator as control.
- The m048 cohort (seeds 89, 28, 14: slow / fast / near-separatrix) over 60-min LC durations.

# How

`L_J2000(t) := R(t).T @ I @ ω(t)` where R(t) = scipy.from_quat(q[xyzw]).as_matrix(). For a real torque-free rigid body, L_J2000 is exactly constant (no external torque ⇒ no change in inertial angular momentum). The script measures `max_t || L_J2000(t) - L_J2000(0) ||` and reports it as an absolute and as a fraction of |L_body|.

**Control:** an independent textbook integrator using `dR/dt = -[ω]_x R` (textbook passive-J2000→body kinematic) on the same toy. This MUST conserve L_J2000 to machine precision if my setup is correct.

# Result

**Textbook control on toy:** drift = **1.13e-12** over 50 s (machine precision). Conservation holds. Control passes.

**Codebase on the same toy:** drift = **191%** of |L_body| over 50 s. L_J2000 precesses ~2x its own magnitude.

**Codebase on m048 cohort:**

| Seed | \|ω₀\| dps | LC duration | abs drift | rel drift (% of \|L_body\|) |
|------|-----:|-----:|-----:|-----:|
| 89 | 0.240 | 60 min | 97.2 | **62.5%** |
| 28 | 1.438 | 60 min | 341.8 | **36.4%** |
| 14 | 1.229 | 60 min | 902.7 | **136.7%** |

In contrast, |L_body| (the body-frame angular momentum magnitude) is conserved by the codebase to ~1e-14 across all seeds — the Euler equation is correctly textbook. Only the q-evolution is wrong.

# Why this matters

**It supersedes s065.** s065 concluded "omega is the negation of physical omega; q0 is physical; document and move on." That framing was correct at t=0 but doesn't extend to the LC duration:

- At t=0: codebase's R(0) matches SPICE at 3e-16. ✓ Both interpretations agree.
- At t=dt (small): codebase's R(dt) matches textbook R(dt) for ω = -ω_input. ✓ (s065 tested this.)
- At t=T_LC (60 min): codebase's R(T) does NOT match any textbook R(T) for any input sign. ✗

The mechanism: the codebase pairs textbook Euler (correct) with the wrong-signed kinematic. ω → -ω makes the kinematic textbook *locally*, but the Euler equation is sign-asymmetric in time (its RHS is even in ω; starting from ±ω₀ gives the same initial dω/dt but trajectories diverge thereafter). So no global sign substitution fixes things.

**It overturns the "physical interpretation" claim** in `feedback_propagator_omega_sign_convention.md` and the codebase's `attitude_propagator.py` docstring. The m048 cached LCs are NOT light curves of real tumbling satellites at any (q0, ω). They are deterministic outputs of a self-consistent but non-physical forward model.

**It explains the s062b Path 2 failure.** The textbook closed-form q(t) decomposes the motion as precession of the body around the (constant) L_J2000 direction. In the codebase's trajectories L_J2000 isn't constant — there is no fixed axis to precess around. So the precession + nutation decomposition has no anchor. The "fix it under codebase sign convention" recipe (s062b redux as planned) does NOT work because the codebase's trajectory is not a textbook torque-free trajectory under any sign substitution.

**Inversion benchmark is still well-posed.** The codebase's forward model is deterministic and invertible, so recovering `(q0, ω0)` such that `propagate_attitude` reproduces the LC is meaningful. All s001…s065 methodology (surrogate-first, multi-mag-start, polhode-basis LM, ρ-band convention, multi-solution acceptance, no oracle injection) is forward-model-agnostic and carries over. Qualitative findings (multi-solution structure, polhode tangent is the soft direction, basin scaling with |ω|) follow from BRDF + STL + geometry and persist.

**Architectural decision: fix-and-regen is now strongly indicated.** Previously the decision was "document, don't fix" because the cohort was thought to be physical. With L_J2000 non-conservation, the decision should be revisited as: "accept a non-physical forward model only as long as the project doesn't intersect real telescope data; otherwise fix soon, because regen cost grows with cached results." This is a separate decision from this writeup; flagged in `PROGRESS.md` as `READ FIRST`.

# Numbers

Toy (I = diag(1,2,3), ω₀=(0.5, 0.3, 0.7), q₀=identity, dt=50 s):
- Textbook control drift:   1.13e-12  (machine precision)
- Codebase drift:           4.28  (190.9% of |L_body|=2.241)
- |L_body| conserved to:    1e-14 (codebase Euler is correct)

m048 cohort (60-min LC, common inertia diag(37985, 38306, 7749) kg m²):
- seed 89 (slow):           drift 97.2 / |L|=155.4 → 62.5%
- seed 28 (fast):           drift 341.8 / |L|=939.1 → 36.4%
- seed 14 (near-separatrix): drift 902.7 / |L|=660.4 → 136.7%

# Artefacts

- `experiments/s066_lj2000_nonconservation.py`
- `results/s066_lj2000/summary.json`

# Out of scope

- Fixing the propagator. That's a project-level decision (cohort regen, surrogate retrain or revalidate, all cached findings need re-measurement) and a separate conversation.
- Deriving closed-form q(t) for the codebase's non-physical ODE. The naive "textbook with ω → -ω" recipe doesn't work; a fundamentally different derivation would be needed. Not pursued.
- Quantifying which prior numerical findings change after regen. Most methodology survives; specific numerical values (q0_err, ρ, polhode_diameter for specific seeds, etc.) will shift. Out of scope here.

# Cross-references

- `experiments/s065_propagator_convention_spice_test.md` — s065's t=0 SPICE comparison, superseded by this finding.
- `experiments/s062b_jacobi_quaternion.md` — original Path 2 attempt; the "convention puzzle" findings reframed by s066.
- `src/dynamics/attitude_propagator.py` — propagator source; docstring updated by s066.
- `concepts/quaternion_convention.md` — convention doc; "Omega-sign residual" section reframed by s066.
- Auto-memory: `feedback_propagator_omega_sign_convention.md` — claim corrected by s066.
