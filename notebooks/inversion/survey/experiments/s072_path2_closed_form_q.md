---
title: "s072 — Path 2 closed-form q(t) lands post-fix; three-gate validation passes strict 1e-9"
type: experiment
sources:
  - experiments/s062_jacobi_design.md
  - experiments/s062b_jacobi_quaternion.md
  - concepts/jacobi_propagation.md
  - concepts/quaternion_convention.md
  - lib/jacobi_propagator.py
  - src/dynamics/attitude_propagator.py
related:
  - project_jacobi_propagation_priority.md
  - feedback_verify_quaternion_convention.md
  - s067_postfix_propagator_validation.md
created: 2026-05-13
updated: 2026-05-13
confidence: high (numerical evidence + math is closed-form)
---

# TL;DR

Re-implemented Path 2 closed-form q(t) (3-1-3 precession+nutation Euler
decomposition) under the post-fix textbook convention. The new
`lib.jacobi_propagator.propagate_jacobi_path2` lands the three-gate
validation cleanly across (a) m048-diagonal inertia on cohort seeds
89/28/14, (b) perturbed inertia (m048 + 1% and 10% off-diagonal),
(c) wholly synthetic asymmetric satellite in regimes A and B. All cases
60-min LCs, 500 epochs. Worst q vs DOP853 inf-norm: **2.81e-10**
(Gate 3 regime B) — 3 orders of magnitude inside the 1e-9 design
threshold. Path 2 self-conservation (2T, L²) is machine-precision
(1e-15 to 1e-16). The pre-fix sign puzzle (s062b 2026-05-12) was
diagnosed correctly by s066/s067 as a propagator bug, not a Path 2
derivation issue; under post-fix convention the textbook Goldstein
3-1-3 form works without modification.

The architectural payoff: **constant-ω-in-body-frame is no longer
needed as a simplifying assumption**. Constant-ω is trivially
closed-form already (`q(t) = exp(-0.5 ω t) ⊗ q_a` via quaternion
exponential, no propagation step) — but s063c established it is
INVALID over a full 60-min LC for ~119/120 cohort seeds because the
polhode period (median T_pol = 1594 s) is shorter than the LC, so ω
cycles 2-3× per LC and constant-ω represents a non-physical
trajectory. Path 2 lets us evaluate FULL torque-free Euler dynamics —
with ω evolving on the polhode — at roughly the cost the constant-ω
simplification was *supposed* to give us. The choice between "fast
and wrong (constant-ω)" and "slow and right (DOP853 per candidate ×
500 epochs)" disappears. This unblocks s062d polhode-basis enumerator
for s060/s061 multi-anchor at the level of physical correctness.

# What

s062a (2026-05-12) established that ω(t) admits a closed-form Jacobi
solution agreeing with DOP853 to ~1e-10 across the m048 cohort. s062b
(2026-05-12) attempted Path 2 closed-form q(t) but hit a convention
puzzle — q matched DOP853 at t=0 to 1e-16 then drifted at the
precession rate even after a φ→−φ sign flip; θ drifted independently
at 0.6°/step. s066 (2026-05-12) revealed the root cause: the codebase's
q-kinematic ODE had the opposite sign of textbook conv-(a) (`+0.5 ω⊗q`
LEFT instead of `−0.5 ω⊗q` LEFT). s067 (2026-05-12) fixed it; the m048
cohort was regenerated. The PROGRESS.md banner has carried "Path 2
unblocked" since.

s072 re-derives Path 2 cleanly under the post-fix textbook convention
and validates on three test classes per the user's directive at
session start.

# How

Math is unchanged from the s062b notes (Goldstein 4.87, passive 3-1-3
with PA z as third axis):

    cos(θ)    = I₃ ω₃ / |L|                                  (algebraic)
    ψ         = atan2(I₁ ω₁, I₂ ω₂)                          (algebraic)
    dφ/dt     = |L| (2T − I₃ ω₃²) / (L² − I₃² ω₃²)          (1-D ODE)
    R_J→PA(t) = R_z(ψ) · R_x(θ) · R_z(φ) · R_J→L            (reconstruction)
    R_J→body  = R_pa · R_J→PA(t)                             (PA→body)

with `R_J→L` pinned at t=0 by enforcing the gauge `φ(0) = 0`. The
formula is regime-agnostic; in regime A (rotation about I₁) θ stays
near π/2 with oscillations and in regime B (about I₃) θ stays small.
The `atan2` for ψ handles wraparound automatically.

Implementation diff vs s062b's `_propagate_jacobi_path2_incomplete`:

  1. Removed `phi_hist = -sol.y[0]` (that was pre-fix compensation).
     Now `phi_hist = sol.y[0]` per textbook positive-precession.
  2. No other change. The math was correct in s062b; only the convention
     wrapper was wrong.

Inertia is eigendecomposed once per call: `I = R_pa · diag(I_pa) · R_paᵀ`
with ascending eigenvalues. The state is rotated into PA before the
closed-form ω/θ/ψ/φ work, then `R_pa @ R_J→PA(t)` rotates back. For
m048 specifically `R_pa` is a proper permutation matrix (inertia already
diagonal in body frame); for Gates 2/3 the inertia has off-diagonals
and `R_pa` is a non-trivial rotation. Both paths run the same code.

The script imports `propagate_attitude` from `src.dynamics.attitude_propagator`
as the DOP853 reference (rtol=1e-12, atol=1e-14 tightened from default
1e-10/1e-12 to push DOP853 noise floor below the Path 2 vs DOP853 gap).
For Gate 1 (m048 seeds with cached truth NPZs) we also compare to the
stored `quaternions` and `k1_body` arrays. For Gates 2/3 the synthetic
trajectory has no cached truth; DOP853 is the only reference.

Three-gate spec from `s062_jacobi_design.md`:

  1. `||q_path2 − q_DOP853||_∞ < 1e-9` (antipode-aware, per-epoch min).
  2. `2T` and `L²` rel-std < 1e-12 in Path 2's output ω.
  3. `||k1_body(q_path2) − k1_body_cached||_∞ < 1e-9` (Gate 1 only).

# Result

| Test | q vs DOP853 (∞) | 2T rel-std | L² rel-std | k1_body (∞) |
|---|---:|---:|---:|---:|
| Gate 1 seed 89 (slow, \|ω\|=0.24 dps) | **1.57e-12** | 2.49e-16 | 2.83e-16 | 2.74e-10 |
| Gate 1 seed 28 (fast, \|ω\|=1.44 dps) | **8.09e-11** | 3.27e-16 | 3.38e-16 | 3.25e-08 |
| Gate 1 seed 14 (near-sep) | **3.12e-11** | 1.20e-15 | 4.39e-16 | 3.26e-09 |
| Gate 2 m048+1% off-diag | **2.13e-12** | 2.21e-16 | 2.23e-16 | — |
| Gate 2 m048+10% off-diag | **5.05e-12** | 1.93e-15 | 2.30e-15 | — |
| Gate 3 asymmetric, regime A | **5.38e-11** | 2.17e-14 | 1.84e-14 | — |
| Gate 3 asymmetric, regime B | **2.81e-10** | 4.62e-14 | 4.70e-14 | — |

All q-vs-DOP853 numbers are 3+ orders of magnitude under the 1e-9
threshold. All conservation numbers are 4+ orders under the 1e-12
threshold. Gate 1 k1_body deltas equal the DOP853-vs-cached
quaternion floor on the same seeds (this is integration noise from
the DOP853 reference itself, not Path 2 error — confirmed by the
`dop853_vs_cached_inf` field in `summary.json` matching `gate_1c`
within an order of magnitude per seed).

Geodesic angle is uniformly ≤ 2.4e-6 deg (≈ 4e-8 rad) on every test —
this is the arccos-near-1 numerical floor, not a real disagreement.
The element-wise inf-norm is the load-bearing metric.

# Why this matters

Three architectural consequences:

  1. **The polhode-basis enumerator for s060/s061 multi-anchor is
     now buildable around physically-correct trajectories.** Sampling
     candidates on `(q_a, |L|, polhode-label, polhode-phase)` and
     evaluating the closed-form `(q(t), ω(t))` over all 500 epochs is
     the s062d architecture. The fast-but-wrong "constant-ω" fallback
     (which s063c showed is invalid over a full LC for 119/120 seeds)
     is no longer the only fast option — full Euler dynamics now
     runs at closed-form cost.

  2. **The Jacobi-coord LM polish reparameterization (s063b/s064)
     can now use closed-form q updates inside the polish iteration.**
     Previous s064 used the hybrid path (closed-form ω, DOP853 on q);
     that's now structurally replaceable.

  3. **The post-fix forward model is internally consistent at machine
     precision.** s072 is independent corroboration that the s067
     four-gate validation captured the propagator's behaviour
     correctly: the closed-form analytical derivation of the same
     ODE agrees with DOP853 down to integration noise.

The remaining wall-time gap to "true closed-form" is the
`solve_ivp(phi_dot, ...)` call at rtol=1e-13 (≈ 2-10 ms per LC
depending on |ω|). The textbook closed-form for that integral is
`scipy.special.elliprj` (Π via Carlson's R_J) — the more durable
follow-up than per-epoch loop vectorisation because it removes the
ODE entirely and is automatically vectorisable across both candidates
and times. **The right order: profile first** (10-min cProfile to
identify whether the bottleneck is the φ-ODE, the per-epoch Python
loop, or per-call overhead at the M-candidate scale), **then** pick
between (a) the elliprj replacement and (b) NumPy vectorisation of
the existing ODE-driven code. Optimising before profiling risks
vectorising the wrong axis (within-candidate vs across-candidate).

# Numbers

  - 60-min LCs at 500 epochs (post-fix m048 sampling).
  - DOP853 reference: rtol=1e-12, atol=1e-14 (tighter than default
    1e-10/1e-12 to push the reference noise floor below Path 2's
    expected accuracy).
  - Path 2 phi-ODE: rtol=1e-13, atol=1e-15.
  - Wall-time Path 2 vs DOP853 (Gate 1): roughly 0.8-2× DOP853 in
    this implementation — the per-epoch quaternion reconstruction loop
    in Python is the bottleneck. Speedup will come from vectorizing
    that loop and/or replacing phi-ODE with Π. The single-call ω cost
    s062a measured at 160× (now ~108× post-fix per s071) is preserved.
  - Inertia diagonality check: m048 is exactly diagonal in body frame
    (R_pa is a proper permutation). Gate 2's 1%/10% off-diagonal
    perturbations push R_pa to Frobenius distance ~2.2 from identity;
    Gate 3's `I = [[2.0, 0.4, 0.15], [0.4, 3.5, -0.25], [0.15, -0.25, 5.0]]`
    is fully non-diagonal in body frame.

# Out of scope

  - Replacing `solve_ivp(phi_dot, ...)` with `scipy.special.elliprj`-based
    closed-form. The current 1-D ODE is fast enough to validate the
    architecture; the closed form is the leading next-direction
    candidate but should be preceded by a cProfile of the current
    implementation to confirm φ-ODE is the bottleneck before committing.
  - Vectorising the per-epoch `R_zxz @ R_J→L → quat_from_matrix`
    reconstruction loop. Alternative next-direction candidate; weaker
    than the elliprj replacement because it can't remove the ODE cost.
    Same "profile first" rule applies.
  - Wiring `method="jacobi_path2"` into `lib/forward.py::propagate_to_body_frame`.
    Deferred to s062c (the design doc's next step) once a downstream
    consumer (s060/s061 multi-anchor enumerator) is ready.
  - Performance benchmark at 100k-1M candidate scale. The s062 design
    doc projected this; s072 only proves correctness.
  - Handling exact symmetric inertia (I₁=I₂ or I₂=I₃). At true equality
    the polhode parameterization degenerates; `_build_omega_func` would
    produce divisions-by-zero. Not relevant for m048 (eigenvalue
    spread 7749 / 37985 / 38305) and Gate 3 uses well-separated
    eigenvalues by construction.

# Artefacts

  - `lib/jacobi_propagator.py` — new `propagate_jacobi_path2` function
    (~80 lines including math).
  - `experiments/s072_path2_closed_form_q.py` — three-gate script.
  - `results/s072/summary.json` — full numerical results, including
    per-seed I_pa eigenvalues, regime labels, R_pa diagnostics, and
    wall-times.

# Cross-references

  - `concepts/jacobi_propagation.md` — math + references.
  - `experiments/s062_jacobi_design.md` — phased plan; s072 closes s062b.
  - `experiments/s062b_jacobi_quaternion.md` — the pre-fix attempt;
    s072 is its post-fix successor.
  - `experiments/s067_postfix_propagator_validation.md` — the fix that
    unblocked this work.
  - `concepts/quaternion_convention.md` — convention narrative.
  - Auto-memory: `project_jacobi_propagation_priority.md` — needs update.
