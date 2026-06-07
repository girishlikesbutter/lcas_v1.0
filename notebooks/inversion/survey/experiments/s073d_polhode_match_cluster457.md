---
title: "s073d — Body-frame polhode match check for cluster_457 vs truth (seed 89): nearby-not-identical polhodes; cat-4 same-polhode leg not confirmed at the 1% scalar gate"
type: experiment
sources:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073c_literature_novelty_audit.md
  - lib/jacobi_propagator.py
related:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073b_path2_cprofile.md
  - experiments/s073c_literature_novelty_audit.md
  - experiments/s063b_polhode_tangent_softness.md
created: 2026-05-14
updated: 2026-05-14
confidence: high on the quantitative numbers; medium on the interpretation (single pair, polish-residual ambiguity)
---

# TL;DR

Cheapest decisive check in the s073c cat-4 hypothesis-conversion sequence (`s073c §Next #1`). Computed body-frame ω(t) for truth and the s059k-polished `cluster_457` on seed 89 via closed-form `omega_jacobi`. Both states are **Case-A regime** with k² agreeing to **0.36%** and polhode periods agreeing to **0.82%**, *but* the two scalar invariants that fully determine the polhode curve — `2T` and `|L|²` — differ by **1.118%** and **1.112%** respectively, and the body-frame ω(t) traces are separated by **0.67% × |ω₀_truth|** in max nearest-neighbour set-distance (both directions, well above the 1e-15 propagator noise floor). Visually the two polhodes overlap; quantitatively they are nearby but distinct closed curves. **The strict "same body-frame polhode" leg of the s073 cat-4 framing is not confirmed at the 1% scalar gate on this single pair.** The result does *not* fully kill cat-4: the s059k LM polish minimised LC residual rather than polhode distance, so a ~1% polhode shift could be polish-residual rather than a genuine geometric distinction. But the s073 assertion that |L| at 0.55% and 2T-implied-by-|L|-match jointly verify same-polhode was not literally checked — and when we check it now, 2T's offset is comparable in size to |L|'s. Reframing direction: the cat-4 hypothesis as originally stated (same body-frame polhode, free L_J2000 direction) is weakened; the empirically observed LC ambiguity on seed 89 looks more like "nearby polhodes also fit at Band A", which is a different theoretical statement and overlaps with cat-3 (within-noise) more than s073 admitted.

# What

s073 observed that cluster_457 (the lone Band A multi-sol attractor on post-fix seed 89; `q0_err = 59.58°`, hi-fi ρ = 0.904) matches truth's `|L|` to 0.55% while its `L_J2000` direction is 128.6° off truth. The wind-down banner asserted this as "same body-frame polhode, different inertial L" — the empirical core of a hypothesised fourth ambiguity category (s073c). The s073c audit flagged that "same polhode" was *asserted* but not directly verified. s073d does the verification.

The polhode is the closed curve in body-frame ω-space traced by ω(t) under torque-free motion. It is the simultaneous level set of two scalar constants of motion:

- energy:  `ω · I · ω = 2T`
- momentum (body-frame magnitude):  `|I · ω|² = |L|²`

Two trajectories share the same polhode IFF both `(2T, |L|²)` match — the curve is fully determined by `(I, 2T, |L|²)`. The orbit on that curve is unique up to phase (where on the curve ω₀ sits) and traversal direction.

# How

For both truth and cluster_457 on seed 89:

1. **Scalar invariants** — compute `2T = ω₀ᵀ I ω₀` and `|L|² = |I ω₀|²` directly.
2. **Jacobi parameterisation cross-check** — call `lib.jacobi_propagator.omega_jacobi`; record regime label (A or B), elliptic modulus `m ≡ k²`, polhode period `T_pol = 4·K(m)/|τ̇|`. Same `(2T, |L|²)` ⇒ same `(regime, k²)` and `T_pol`.
3. **Closed-form trace** — propagate ω_body(t) on a 1001-point grid spanning the larger of LC duration and one full polhode period (so the set-match test covers the whole closed curve).
4. **Conservation sanity** — along each trace, drift in `2T` and `|L|²` must be at the `omega_jacobi` floor (≤ 1e-12 per s062a validation; expected ~1e-15).
5. **Set-match** — for each row of `omega_c457(t)`, distance to the nearest row of `omega_truth(t)` (and vice versa). Normalise by `|ω₀_truth|`. Compare to a 1% gate.
6. **Visual** — 3D scatter of both traces + three 2D projections to confirm the geometric picture.

Gate for "same polhode": `max(|Δ 2T|/2T, |Δ |L|²|/|L|²) < 1%` AND same regime AND `max set-dist < 1% × |ω₀_truth|`.

Cluster_457's `(q0, ω0)` is loaded from the same s059k summary that s073 used (`results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json` at `mag_pct_offset=0.0`). Truth state from `data/trajectories/traj_seed089.npz`. Inertia from `build_context(seed)`.

# Result

| Quantity | truth | cluster_457 | relative diff |
|---|---|---|---|
| 2T (kg m² · rad² / s²) | 6.407934e-01 | 6.479558e-01 | **1.118%** |
| \|L\|² (kg² m⁴ / s²)   | 2.415203e+04 | 2.442056e+04 | **1.112%** |
| regime (A/B)            | A            | A            | match       |
| k² ≡ m                  | 5.162490e-01 | 5.143704e-01 | 0.36%       |
| T_pol (s)               | 7269.49      | 7209.94      | 0.82%       |

Conservation along the closed-form trace (`omega_jacobi` floor sanity):

- truth: 2T drift = 1.213e-15, |L|² drift = 1.506e-15
- c457:  2T drift = 1.028e-15, |L|² drift = 1.490e-15

Both at machine precision. The propagation step itself is sound; the 1.1% gap is real.

Set-match (nearest-neighbour distance, normalised by |ω₀_truth| = 4.1857e-03 rad/s):

- c457 → truth: max = **6.699e-03**, median = 6.313e-03
- truth → c457: max = **6.699e-03**, median = 6.313e-03

Both directions agree to numerical precision (each polhode is uniformly displaced from the other in body-frame ω-space — not just one outlier point).

**Gate verdict (script-level, transparent):** FAIL on `max(2T, |L|²)` gate; FAIL on set-distance gate. Same regime, same case A.

## Visual

`results/s073d/polhode_match.png` shows 3D + 2D projections. At the scale of the curve (~|ω₀_truth|), the two polhodes overlap visually. The truth ω₀ and cluster_457 ω₀ sit at *very different phases* on these nearly-coincident curves (truth ω₀ is near ω_y ≈ +|ω|; cluster_457 ω₀ is near ω_x ≈ +|ω|, ~90° around the polhode). The 1% geometric separation is below visual resolution at this scale but quantitatively unambiguous.

# Why this matters

Two readings of the result, both load-bearing on next steps:

**Reading 1 (strict).** The polhodes are different. Cluster_457 is *not* on truth's polhode. The s073 framing — "same body-frame polhode, free inertial L direction" — does not apply to this pair on this seed. Reframe required.

**Reading 2 (charitable).** The s059k LM polish minimised LC residual, not polhode distance. The polished endpoint can sit on a nearby polhode that produces a similarly good LC. The "true" basin attractor may still lie on truth's polhode, and what we are seeing is polish-residual rather than a geometric distinction.

We cannot distinguish between (1) and (2) from this single pair. To distinguish would require: (a) a tighter polish targeted at minimising polhode distance, or (b) testing whether perturbing cluster_457's `(q0, ω0)` onto truth's exact `(2T, |L|²)` while keeping the L_J2000 direction fixed produces an LC at Band A — i.e., checking whether truth's polhode admits a multi-sol at cluster_457's inertial L direction.

The neutral characterisation that survives both readings:

> On post-fix seed 89, the Band A multi-sol cluster_457 is on a polhode that is **near** truth's polhode (1% in scalar invariants, 0.67% × |ω₀| in body-frame trace distance) but is **not identical** to it at the resolution of the s059k polish. Cluster_457's inertial L direction is 128.6° from truth's at t=0.

This wording neither overclaims same-polhode invariance (correcting the s073 banner) nor overclaims that the cat-4 framing is dead (the polish-residual ambiguity is real).

**Operational consequence for the cat-4 hypothesis-conversion sequence (s073c §Next):**

- §Next #1 (this experiment) returns *ambiguous* rather than *clean pass* or *clean fail*. The framing in s073c assumed a binary answer; the data says "nearby but not identical, polish-residual cannot be ruled out".
- §Next #2 (continuous-family local test on cluster_457) is the natural follow-up — if perturbing cluster_457's `(q0, ω0)` parallel to L_J2000 stays at Band A, while perturbing along (2T, |L|²) leaves Band A quickly, that supports a polhode-orthogonal-to-LC-tolerance picture. If both directions leave Band A at similar rates, cluster_457 looks more like an isolated point and cat-4 weakens further.
- §Next #3 (cohort sweep) becomes more interesting, not less — even if cluster_457 is "near-polhode" rather than "exact same polhode", a repeating pattern of *near-polhode* multi-sols across seeds is its own structural finding. If cluster_457 is unique, it's an anecdote.

**Methodology lesson reinforced (already captured in `feedback_dont_overclaim_from_one_data_point.md` from s073c).** The s073 banner stated "|L| matches, same body-frame polhode" — but `2T` was not measured. The "same polhode" half required `(2T, |L|²)` both matching. Computing only `|L|` and asserting `2T` matches "by implication" was the overclaim. The CLAUDE.md "cite the file for every number" rule applies to scalar invariants in dynamical claims too: if the hypothesis hinges on two quantities, measure both.

# Numbers

Source: `results/s073d/summary.json`.

| Field | Value |
|---|---|
| seed | 89 |
| cluster_id | 457 |
| cluster_457 ρ_hifi (from s073/s069) | 0.904 |
| cluster_457 q0_err (from s073) | 59.58° |
| inertia diag (kg m²) | (37985.156, 38305.706, 7749.015) |
| 2T_truth | 6.407934e-01 |
| 2T_c457 | 6.479558e-01 |
| 2T rel diff | 1.118e-02 |
| |L|²_truth | 2.415203e+04 |
| |L|²_c457 | 2.442056e+04 |
| |L|² rel diff | 1.112e-02 |
| regime truth / c457 | A / A |
| k² truth | 5.162490e-01 |
| k² c457 | 5.143704e-01 |
| T_pol truth (s) | 7269.49 |
| T_pol c457 (s) | 7209.94 |
| LC duration (s) | 3600.00 |
| 2T drift along truth trace | 1.213e-15 |
| |L|² drift along truth trace | 1.506e-15 |
| 2T drift along c457 trace | 1.028e-15 |
| |L|² drift along c457 trace | 1.490e-15 |
| |ω₀_truth| (rad/s) | 4.1857e-03 |
| max NN dist c457→truth (rel) | 6.699e-03 |
| max NN dist truth→c457 (rel) | 6.699e-03 |
| median NN dist (rel, both) | 6.313e-03 |

# Artefacts

- This writeup: `notebooks/inversion/survey/experiments/s073d_polhode_match_cluster457.md`
- Script: `notebooks/inversion/survey/experiments/s073d_polhode_match_cluster457.py`
- Summary: `notebooks/inversion/survey/results/s073d/summary.json`
- Traces NPZ: `notebooks/inversion/survey/results/s073d/omega_traces.npz`
- Figure: `notebooks/inversion/survey/results/s073d/polhode_match.png`

# Out of scope

- A tighter LM polish targeting polhode distance directly (to resolve the polish-residual interpretation).
- The continuous-family local test (s073c §Next #2): take cluster_457's `(q0, ω0)`, apply parameterised inertial-L rotations + small `(2T, |L|²)` perturbations, render LCs, map the residual surface. Distinguishes "isolated point" from "manifold near cluster_457".
- The cohort sweep (s073c §Next #3): same analysis on Band A∪B multi-sols on other seeds (10, 14, 28, 91, …).
- An operational L-conservation cross-anchor filter test (s073c §Next #4): would need >1 anchor time and a candidate pool, not just one pair.
- Robinson & Frueh 2025 *J. Astronaut. Sci.* paper (Springer paywall; still unread; closest residual literature threat).

# Cross-references

- `experiments/s073_cluster457_l_vector_check.md` — the parent measurement that asserted same-polhode.
- `experiments/s073c_literature_novelty_audit.md` — the scope-correction audit and experiment plan (this is §Next #1).
- `experiments/s063b_polhode_tangent_softness.md` — polhode tangent vs cost-curvature analysis on truth; orthogonal in scope but uses the same `omega_jacobi` substrate.
- `lib/jacobi_propagator.py::omega_jacobi` — closed-form ω(t) used here.
- `concepts/known_pathologies_to_revalidate.md` — broader audit list.
- Memory: `feedback_dont_overclaim_from_one_data_point.md` (the recurring N=1 trap, reinforced here).
