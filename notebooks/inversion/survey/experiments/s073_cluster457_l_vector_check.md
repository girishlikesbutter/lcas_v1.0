---
title: "s073 — cluster_457 L-vector check: L-conservation cross-anchor discriminates the seed-89 Band A multi-sol"
type: experiment
sources:
  - experiments/s069_replicate_s059k.md
  - experiments/s066_lj2000_nonconservation.py
  - lib/jacobi_propagator.py
  - lib/traj_load.py
related:
  - project_jacobi_propagation_priority.md
  - feedback_propagator_omega_sign_convention.md
created: 2026-05-14
updated: 2026-05-14
confidence: high (one number, computed two ways agreeing)
---

# TL;DR

For seed 89 post-fix, **L_J2000(cluster_457) vs L_J2000(truth)** at t=0:
**|L|** magnitudes match to **0.55%** but **direction differs by 128.6°**.
Total `|ΔL_J2000| / |L_truth| = 181%`, dominated by the direction
mismatch. Same body-frame polhode (same |L|, same 2T), completely
different inertial orientation of the angular momentum vector.

**L-conservation cross-anchor matching is a real discriminator against
this Band A multi-sol class** — propagation-free 3-DOF gate on the
12-DOF pair-space `(q_A, ω_A, q_B, ω_B)`, orthogonal to brightness
matching. Worth designing as the s062d consumer; slots into s060
multi-anchor design without depending on Path 2.

# What

s069 (2026-05-12, post-fix s059k replication) surfaced cluster id=457
as the **single Band A multi-solution attractor** on seed 89:
`q0_err=59.58°, ω_dir_err=25.05°, |ω|_err=+0.57%, hi-fi ρ=0.904`.
Truth ranks 1987/2013 by the W=10 local-window score and never enters
the top-50 polish window. So on this specific LC, the inversion's
single Band A is a multi-sol, not truth.

The 2026-05-13 discussion captured an architectural question: is
L_J2000 a discriminator against this kind of multi-sol? Real
torque-free dynamics conserves L_J2000 exactly (post-s067 fix), so if
two candidates have different L_J2000(0), their L_J2000(t) trajectories
also stay different at all later times — a single-shot comparison
gates the entire architecture.

s073 answers that with a one-epoch computation.

# How

```python
L_J2000 = R_J2000_to_body(q0).T @ (I_body @ ω0)
```

`_quat_to_matrix(q)` in `lib/jacobi_propagator.py` returns the post-fix
conv-(a) passive J2000→body matrix; transposing it gives body→J2000.
Formula matches `experiments/s066_lj2000_nonconservation.py::measure_drift`.

Truth state from `lib.traj_load.load_truth(89)`. Cluster 457 state from
`results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json` —
the `mag_pct_offset=0.0` row of the canonical polished endpoint.
Inertia from `lib.hifi_render.build_context(89)["inertia_tensor"]`
(m048 diagonal: I = diag(37985.16, 38305.71, 7749.01) kg·m²; off-diag
max 1.13e-14, machine precision).

# Result

| Quantity | Truth | cluster_457 | Δ |
|---|---:|---:|---:|
| q0_err vs truth (°) | 0 (by def) | 59.58 | — |
| ω_dir_err (°) | 0 | 25.05 | — |
| |ω| (rad/s) | 4.95e-03 | 4.98e-03 | +0.57% |
| **|L_J2000| (kg m²/s)** | **1.554e+02** | **1.563e+02** | **+0.55%** |
| **L_J2000 dir angle (°)** | — | — | **128.56** |
| **|ΔL_J2000| / |L_truth|** | — | — | **180.68%** |
| hi-fi ρ vs truth LC | 0 (by def) | 0.904 (Band A) | — |

The magnitude match to 0.55% reflects that cluster_457 has nearly
identical body-frame angular-momentum magnitude and rotational energy
to truth — physically, the two trajectories live on essentially the
same polhode (same energy ellipsoid, same momentum ellipsoid). The
128.6° direction difference says they're tracing that same polhode
**oriented differently in inertial space**. Brightness ρ=0.904 makes
sense: the LC is a function of body-frame `(k1_body, k2_body)`, which
depends on `R(q(t))` not on the absolute J2000 attitude of L.

# Why this matters

**Architectural gate result: positive.** L-conservation cross-anchor
matching is a real discriminator against this Band A multi-sol class.
Design as follows:

1. At two anchors A, B (separated in time across the LC), generate
   independent C_t-style q-clouds + ω-grid candidate pairs.
2. For each candidate, compute `L_J2000(t_A) = R(q_A).T @ I @ ω_A` and
   `L_J2000(t_B) = R(q_B).T @ I @ ω_B`.
3. Reject pairs whose `|L_J2000(t_A) − L_J2000(t_B)|` exceeds a noise
   threshold. The L_J2000 vector is a 3-DOF gate on the 12-DOF
   pair-space `(q_A, ω_A, q_B, ω_B)`.

The filter is **propagation-free** (no Path 2 dependence; no DOP853
dependence). It's orthogonal to brightness matching (which only
constrains body-frame geometry). It does not depend on knowing
absolute |L| a priori — `|L|` cancels out of the cross-anchor
comparison; the filter only requires that the two anchors produce
the *same* L_J2000.

Cluster_457 specifically would be rejected by this filter against
truth — its L_J2000 sits 128.6° off truth's L_J2000 direction, and at
the truth anchor pair the truth |L_J2000| is conserved exactly.
Whether L-matching rejects multi-sols where the |L| also matches but
the direction is off is exactly what this experiment confirms:
**directional disagreement is the load-bearing axis**, not magnitude.

This generalises the diagnostic: classes of multi-sols that share the
truth's polhode but differ in inertial orientation are caught by
L_J2000 matching. Classes that have the same `R_J→L` (same |L| AND
same direction of L in J2000) but differ in q-rotation within the
L-pinned 3-1-3 phi-precession are NOT caught — those would need a
separate discriminator (likely phi-phase consistency across anchors).

# Numbers

- Inertia diag (kg m²): `[37985.156, 38305.706, 7749.015]`
  (off-diag max 1.13e-14, machine precision; m048 is exactly diagonal
  in body frame).
- |L_J2000| at truth: `155.41 kg m²/s`.
- |L_J2000| at cluster_457: `156.27 kg m²/s`.
- |ΔL_J2000|: `280.80 kg m²/s` (1.81× |L_truth|).
- |ΔL_J2000| coming entirely from direction: `2 |L| sin(128.56°/2) ≈ 1.80 |L|`,
  matching the 1.81× exactly to expected.
- ω_truth (rad/s): `[1.46e-04, -3.92e-03, -3.03e-03]`,
  |ω|_truth = 4.95e-03 rad/s = 0.284 dps.
- ω_c457 (rad/s): `[4.07e-03, -5.74e-04, -9.15e-04]`.
  ω points along different body axes between truth and cluster_457 —
  confirms that cluster_457 lives on a different point of the same
  polhode (or that the body axis itself is rotated by R_pa permutation
  ambiguity; m048's I_1 = body-z is the smallest eigenvalue here).

# Out of scope

- Designing the L-conservation filter implementation (s062d candidate).
  This experiment only gates whether the architecture is worth designing.
- Cohort-wide audit of whether the L-direction discriminator survives
  for every Band A multi-sol class. Cluster_457 is the only post-fix
  Band A multi-sol surfaced so far; pre-fix evidence (different
  trajectories, archived in `_s066buggy_archive/`) does not transfer.
- Magnitude-vs-direction sensitivity sweep on the filter threshold.
  Operational threshold (what relative |ΔL| is "noise") depends on
  the surrogate accuracy at the anchor and on the q-cloud TOL setting.
- Comparison with phi-phase consistency as an alternative discriminator
  for the L-matched residual class.

# Artefacts

- `experiments/s073_cluster457_l_vector_check.py` — script (one-shot, ~5s wall).
- `results/s073/summary.json` — all numbers.
- `results/s073/l_vectors.npz` — L_truth, L_c457, ΔL, q0/ω0 inputs, inertia.

# Cross-references

- `experiments/s066_lj2000_nonconservation.md` — where L_J2000 = R.T @ I @ ω comes from.
- `experiments/s067_postfix_propagator_validation.md` — confirms conservation post-fix.
- `experiments/s069_replicate_s059k.md` — where cluster_457 was first surfaced.
- `concepts/jacobi_propagation.md` — relates L_J2000 to the Jacobi closed-form's R_J→L.
