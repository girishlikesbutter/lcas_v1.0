---
title: "Jacobi-elliptic closed-form propagation for torque-free attitude — cash on the table"
type: concept
created: 2026-05-12
updated: 2026-05-12
confidence: high (math is 175 years old and textbook; engineering work is just careful implementation + convention validation)
---

# Jacobi-elliptic closed-form propagation

## TL;DR

`src/dynamics/attitude_propagator.py::propagate_euler` integrates the torque-free Euler equations numerically with DOP853 (rtol=1e-10, atol=1e-12). Those equations have had a **closed-form solution in Jacobi elliptic functions since Jacobi himself (1849)**. The closed form gives ω(t) in O(1) time via `scipy.special.ellipj`, and q(t) in O(1) time via either theta-function ratios (van Zon & Schofield 2007) or a precession + nutation decomposition. **Two orders of magnitude faster, drift-free, and parameterized exactly the way the polhode prior wants** — (|L|, polhode-label, polhode-phase) ARE the Jacobi parameters. The s061 cloud-threading reframe (constant-ω propagation per (q_a, ω) candidate) is the forcing function: 100k candidates × 500 epochs is 14 h via DOP853 and ~8 min via Jacobi.

**Not implemented yet — this is a priority queued for a fresh agent to take slowly and carefully.** See `experiments/s062_jacobi_design.md` for the phased implementation plan.

## Why now

Three things converged in 2026-05:

1. **The polhode-prior reframe** (`project_polhode_prior.md`, `concepts/polhode_prior.md`, s051–s055): the natural inversion parameterization is `(|L|, polhode-label, polhode-phase)` — three dynamics-admissible scalars in place of raw (ω_x, ω_y, ω_z). The Jacobi solution **is** the analytical form of that parameterization. Sampling on (|L|, label, phase) and evaluating the closed-form ω(t) and q(t) directly replaces "sample ω uniformly, propagate numerically, project onto polhode."

2. **The s061 reframe** (`PROGRESS.md` top section, `experiments/s061c_seed28_thread.md`): the architectural fix for the broken lineage tracking is **constant-ω propagation per (q_a, ω) candidate**, brightness-filtering at every epoch. Each candidate trajectory is then a coherent (q_a, ω) hypothesis by construction. This shifts the propagator from "called once per LM polish step" (where DOP853 cost is fine) to "called once per (q_a, ω) candidate per epoch" (where the cost dominates).

3. **The forward-model substrate is sound and the convention bug is fixed.** Post-2026-04-30, we know the propagator + renderer pair round-trips cached truth at machine precision. We have a clean baseline to validate any new propagation method against.

## The math (compact reference)

Torque-free Euler equations in principal-axis body frame (I_1 ≤ I_2 ≤ I_3):

    I_1 ω̇_1 = (I_2 − I_3) ω_2 ω_3
    I_2 ω̇_2 = (I_3 − I_1) ω_3 ω_1
    I_3 ω̇_3 = (I_1 − I_2) ω_1 ω_2

Conserved quantities:

    2T  = I_1 ω_1² + I_2 ω_2² + I_3 ω_3²
    L²  = I_1² ω_1² + I_2² ω_2² + I_3² ω_3²

The trajectory in ω-space lies on the intersection of two ellipsoids — the **polhode**. The regime depends on `2T · I_2` vs `L²`:

- **Regime A: 2T · I_2 > L²** — rotation predominantly about I_1; polhode encloses the I_1 axis.
- **Regime B: 2T · I_2 < L²** — rotation predominantly about I_3; polhode encloses the I_3 axis.
- **Separatrix: 2T · I_2 = L²** — heteroclinic orbit (period → ∞).

### Closed-form ω(t)

For Regime A (Landau & Lifshitz, *Mechanics* §37, eqs 37.10–37.12):

    ω_1(t) = √[(2T·I_3 − L²) / (I_1·(I_3 − I_1))] · dn(τ, k)
    ω_2(t) = √[(2T·I_3 − L²) / (I_2·(I_3 − I_2))] · sn(τ, k)
    ω_3(t) = √[(L² − 2T·I_1) / (I_3·(I_3 − I_1))] · cn(τ, k)

with

    k² = [(I_2 − I_1)·(L² − 2T·I_1)] / [(I_3 − I_2)·(2T·I_3 − L²)]
    τ  = (t − t_0) · √[(I_3 − I_2)·(L² − 2T·I_1) / (I_1·I_2·I_3)]

Regime B is symmetric with the roles of (I_1, ω_1) ↔ (I_3, ω_3) swapped and `dn ↔ cn`. `t_0` is fixed by the initial condition.

`scipy.special.ellipj(τ, k²)` returns `(sn, cn, dn, ph)` vectorized. The full ω(t) for an entire trajectory is one vectorized call.

### Closed-form q(t)

Three formulations in the literature, in roughly decreasing implementation difficulty:

1. **Theta-function form (van Zon & Schofield 2007, "Numerical implementation of the exact dynamics of free rigid bodies").** Most algorithmic-friendly explicit closed form. Uses ratios of Jacobi theta functions θ_1, θ_2, θ_3, θ_4. Stable except near the separatrix; handles all generic cases.

2. **Precession + nutation decomposition.** Use the fact that angular momentum L is fixed in the inertial frame. Decompose q(t) = q_precess_around_L_inertial(t) ⊗ q_nutation_in_body(t). The precession is uniform-rate around the fixed L_J2000 vector with rate Ω = L / (some elliptic-function-mean of I); the nutation is the body-frame wobble periodic with the same period as ω. Each piece is closed-form. Easier to derive from first principles; numerical work split into two pieces.

3. **Hybrid: analytical ω(t), numerical q(t) on a coarse grid.** Integrate q̇ = ½ ω(t) ⊗ q (Hamilton convention, see `quaternion_convention.md`) using Magnus / Lie-group integrator on a step that exploits the smoothness of analytical ω(t). Sacrifices most of the speedup but inherits zero ω-derivative error from the analytical ω. Use as a sanity intermediate, not as production.

**Recommendation: pick (1) or (2), validate against `propagate_attitude` on cached truth at machine precision, then keep one.**

## The polhode prior → Jacobi parameter mapping

`concepts/polhode_prior.md` parameterizes ω as `(|L|, polhode-label, polhode-phase)`. These map directly onto the Jacobi solution's parameters:

| Polhode-prior | Jacobi |
|---|---|
| `|L|` (overall scale) | sets the magnitudes inside the square roots above; equivalently, scales `dn / sn / cn` amplitudes. |
| polhode-label (regime + sign) | selects regime A vs B, and which sign of dominant axis. 4 cases (+I_1, −I_1, +I_3, −I_3). Determines which permutation of the ω formula above to use. |
| polhode-phase | τ_0 ∈ [0, 4K(k)), where K(k) is the complete elliptic integral of the first kind (the period of the polhode in τ). |
| inertia I | fixed (m048-known, see "Inertia tensor assumption" below). |

The state space (q_a ∈ S³ / ±1, |L|, label, phase) has the same dimension (3 + 1 + 1 + 1 = 6) as (q_a, ω_a) (3 + 3 = 6). Same DOF, different basis. The Jacobi basis is **dynamics-admissible by construction** — uniform sampling over (|L|, phase) traces uniform sampling over the polhode at uniform-in-time density.

## The s061 forcing function

Today's `propagate_attitude` cost per call (Euler mode, 500 epochs):
- DOP853, rtol=1e-10, atol=1e-12: **~1 ms** per call.

s061 reframe (constant-ω per (q_a, ω) candidate, brightness-filter every epoch):
- Candidate budget ~100k–1M (anchor × tube product space).
- Epochs per candidate = N_obs ≈ 500.
- Total propagator calls = 50M–500M.
- DOP853 cost: 14 h – 6 d (Pool(8) gives ~5–18× scaling at best).

Jacobi cost (rough order):
- `scipy.special.ellipj` is a C-speed batched call. ~10 μs per (q_a, ω, t) tuple including the q(t) work.
- 50M–500M evaluations: **8 min – 1.4 h** (Pool(8) or pure vectorization).

**This is the inversion-architecture-unblocking number.** The s061 reframe is currently bounded by propagator cost in a way it doesn't have to be.

Even without s061, the existing `s059k` inversion does ~50 polishes × hundreds of LM iterations × propagate_attitude per iteration ≈ 50k–500k propagator calls per seed. ~50–500 sec → 5–50 sec at Jacobi speed. Modest win for the polish path, large win for the search path.

## Edge cases that the implementation MUST handle

1. **Separatrix (2T · I_2 → L²).** k → 1, K(k) → ∞. The sn / cn / dn degenerate to tanh / sech. `scipy.special.ellipj` handles k² close to 1 but may lose precision; check.
2. **Symmetric inertia (I_i = I_j).** Two principal moments equal → Jacobi degenerates to elementary trig (one axis is dynamically free; precession is uniform). Detect upstream and dispatch to `propagate_principal_axis`.
3. **Pure principal-axis rotation (ω parallel to a principal axis).** Already handled by `propagate_principal_axis`. Detect this case (e.g., when two of ω_1, ω_2, ω_3 are below a threshold relative to |ω|) and dispatch.
4. **Numerical conditioning of the regime discriminant `2T · I_2 − L²`.** Near zero on near-separatrix trajectories. Use the absolute regime label cautiously and add a fallback path (e.g., switch to the hybrid form near the separatrix).
5. **Time periodicity / branch cuts.** τ is unbounded; modulo by 4K(k) for the periodic part. Carry over the integer-period count to avoid catastrophic cancellation on long trajectories.

## Working in principal-axis coordinates

The Jacobi-elliptic formulae above are written in terms of the principal moments (I_1, I_2, I_3) and require ω to be expressed in the principal-axis (PA) frame. **Inertia tensors are symmetric positive-definite and always diagonalizable: I = R_pa · diag(I_1, I_2, I_3) · R_paᵀ, where R_pa's columns are the principal axes.** So the algorithm for ANY rigid body — symmetric body frame or not — is the same:

1. Eigendecompose I once at startup → (I_1, I_2, I_3, R_pa).
2. Convert initial state into PA frame: ω_a_pa = R_paᵀ @ ω_a_body; q_a_pa = q_pa_to_body* ⊗ q_a_body (composing the constant frame rotation).
3. Run the closed-form Jacobi solution in PA coordinates: (q_pa(t), ω_pa(t)).
4. Convert back: ω_body(t) = R_pa @ ω_pa(t); q_body(t) = q_pa_to_body ⊗ q_pa(t).

**R_pa is a constant rotation matrix that depends only on the satellite, not on the trajectory.** Bake it once into the propagator helper at startup. The per-propagation cost is two constant matrix multiplies — negligible against the closed-form Jacobi evaluation.

**What s062a should check** is just whether R_pa is already the identity matrix in the propagator's body frame, in which case steps 2 and 4 simplify away. For the m048 satellite specifically, the inertia tensor comes from `lib.hifi_render`. Check whether it's already diagonal; report (I_1, I_2, I_3) and R_pa. **Either way, the algorithm proceeds — diagonalization is a standard part of the Jacobi method, not a special case or a gating condition.**

## Validation gate (non-negotiable)

The convention bug fix on 2026-04-30 (`concepts/quaternion_convention.md`) is recent. Any new propagator must round-trip through the renderer to the same machine-precision tolerance as `propagate_attitude` on cached truth before being trusted:

1. Load cached truth NPZ for several seeds spanning the polhode regime distribution (e.g., seeds 14, 28, 89 — near-separatrix, narrow-basin, slow tumbler).
2. Extract `(q0_truth, ω_truth, inertia, observation_times)`.
3. Call `propagate_jacobi(q0_truth, ω_truth, inertia, observation_times)` and `propagate_attitude(q0_truth, ω_truth, observation_times, "tumbling", inertia)`.
4. Verify: `||q_jacobi[i] − q_DOP853[i]|| < 1e-9` for all i (or `||q_jacobi[i] + q_DOP853[i]|| < 1e-9` to absorb the antipode sign).
5. Verify: conserved quantities `2T_jacobi(t)` and `L²_jacobi(t)` are constant to machine precision (this is the Jacobi-internal sanity).
6. Verify: pipe `q_jacobi` through `survey/lib/forward.py::propagate_to_body_frame` (substituting `quats` with the Jacobi output) and confirm `k1_body` matches the cached `k1_body` to `< 1e-12`.

**If step 4 fails by a sign-flip or factor-of-2 pattern, that's a convention bug — fix it in the Jacobi q(t) derivation, not in a sign-fudge wrapper.**

## References

- **Landau & Lifshitz**, *Mechanics* (Vol. 1 of the Course of Theoretical Physics), §37 "Motion of asymmetrical top". The textbook closed-form ω(t).
- **van Zon & Schofield (2007)**, "Numerical implementation of the exact dynamics of free rigid bodies," *J. Comput. Phys.* 225, 145–164. Algorithmic q(t) via theta-function ratios; explicit pseudocode.
- **Celledoni, Sundnes, Sætran (2012)**, "Analytical solution of attitude propagation," *Communications in Nonlinear Science and Numerical Simulation*. Modern presentation; alternative parameterization.
- **Romano (2008)**, "Exact analytic solution for the rotation of a rigid body having spherical ellipsoid of inertia and subjected to a constant torque," *Celestial Mechanics and Dynamical Astronomy*. Related but for constant torque (not our case); useful for sign-convention cross-checking.
- **Hurtado**, various papers on quaternion-based attitude formulations. Useful for verifying q(t) form under Hamilton convention.

## Cross-references

- `concepts/polhode_prior.md` — the inversion-side parameterization motivating this.
- `concepts/quaternion_convention.md` — the convention that the Jacobi q(t) must round-trip through.
- `src/dynamics/attitude_propagator.py` — the numerical baseline the closed form must agree with.
- `experiments/s062_jacobi_design.md` — phased implementation plan.
- `experiments/s061c_seed28_thread.md` — the architectural forcing function (constant-ω propagation per candidate).
- `PROGRESS.md` "Track J (Jacobi)" — live status.
- Auto-memory: `project_jacobi_propagation_priority.md`.
