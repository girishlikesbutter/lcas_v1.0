---
title: "s062 — design doc for Jacobi-elliptic closed-form propagation"
type: design
sources:
  - concepts/jacobi_propagation.md
  - concepts/polhode_prior.md
  - concepts/quaternion_convention.md
  - src/dynamics/attitude_propagator.py
  - experiments/s061c_seed28_thread.md
related:
  - project_jacobi_propagation_priority.md
  - project_polhode_prior.md
  - feedback_verify_quaternion_convention.md
  - feedback_use_existing_lib_forward.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (math + literature are settled; risk is in convention bookkeeping during the q(t) implementation — gated by a non-negotiable round-trip test)
---

# TL;DR

`propagate_euler` is integrating the torque-free Euler equations numerically with DOP853. The closed-form Jacobi-elliptic solution gives ω(t) and q(t) in O(1) per epoch. ~10²× speedup matters for the s061 constant-ω-per-candidate reframe; the natural parameterization aligns with the polhode prior. **This design lays out a phased implementation by a fresh agent. Do NOT skip the validation gate.**

See `concepts/jacobi_propagation.md` for the math, polhode-parameter mapping, and references. This doc is the *implementation* plan.

# Phasing

## s062a — characterize the m048 inertia + Jacobi ω(t) (~1–2 h)

**Goal.** Set up the principal-axis (PA) frame infrastructure and validate the closed-form ω(t):

1. **Eigendecompose the m048 inertia tensor and record (I_1, I_2, I_3, R_pa).** Load `inertia_tensor` via the canonical path used by `lib.hifi_render` / `propagate_attitude`. Check first whether `np.allclose(I, np.diag(np.diag(I)), atol=1e-12)` — if yes, R_pa = I (identity), and the per-propagation frame rotation drops out. If no, eigendecompose `I = R_pa · diag(I_1, I_2, I_3) · R_paᵀ` and bake R_pa into the propagator helper as a constant matrix to be composed around the Jacobi core (preprocessing: rotate state into PA frame; postprocessing: rotate state back). **Either case proceeds; eigendecomposition is a standard step in the Jacobi algorithm, not an exceptional fallback.** Report (I_1, I_2, I_3) and R_pa in `results/s062a/inertia_principal_axes.json` either way.

2. **Analytical ω(t) matches `propagate_euler` to rtol < 1e-8 across the polhode regime distribution.**
   - Implement `omega_jacobi(t, omega_a, inertia)` using `scipy.special.ellipj`. Handle Regime A vs B via the `2T·I_2 − L²` sign.
   - On 5–10 seeds spanning polhode regimes (small polhode / large polhode / near-separatrix — pick from the s053 cohort scan results: e.g., seeds 89 small, 28 large, 14/17/68/81 near-separatrix), call both `propagate_euler` and `omega_jacobi` and compare ω component-wise.
   - Also verify 2T and L² are conserved to machine precision in the Jacobi output (this catches algebra errors in the formula itself).
   - On the same seeds, verify the polhode-label inferred from `2T·I_2 − L²` matches what the s048c+ viewer or s053 cohort labels would say (sanity on the regime detection).

**Exit criterion.** Inertia characterized (whether already diagonal or eigendecomposed with R_pa baked in). ω(t) agrees with DOP853 to rtol < 1e-8 on all test seeds. Conserved quantities exact.

**Artefacts.**
- `experiments/s062a_inertia_and_omega.py` (script).
- `experiments/s062a_inertia_and_omega.md` (writeup).
- `results/s062a/inertia_principal_axes.json` (I matrix, (I_1, I_2, I_3), R_pa, whether already diagonal).
- `results/s062a/omega_vs_dop853_seedXXX.npz` per test seed.

## s062b — Jacobi q(t) + the convention validation gate (~3–4 h)

**Goal.** Implement `propagate_jacobi(q0, omega_a, inertia, times) → (q_history, omega_history)` that round-trips cached truth through the renderer at machine precision.

**Implementation choice — pick ONE and stay disciplined.** Two paths:

- **Path 1: theta-function form (van Zon & Schofield 2007).** Most algorithmic; follow their pseudocode. Estimated 100–150 lines including theta-function evaluation (`mpmath.jtheta` for reference; for production, custom-implement the relevant theta ratios since `mpmath` is slow). Risk: numerical conditioning near the separatrix; theta functions can be tricky.
- **Path 2: precession + nutation decomposition.** Compute L_J2000 (constant in inertial), then q(t) = q_precess(t) ⊗ q_nutation(t). Precession is uniform around L_J2000 at rate set by an elliptic-integral mean; nutation is the body-frame wobble closed-form via cn/sn/dn. ~150–200 lines. Risk: rederiving the precession rate formula correctly.

**Recommendation: start with Path 2.** It's easier to debug because the two pieces are physically interpretable (you can plot the precession trajectory of body-Z in J2000 vs L_J2000 and visually confirm it's wobbling around L). Path 1 is more compact but harder to localize bugs in.

**Validation gate (do not skip).**

```python
# pseudocode — fill in with your chosen path
for seed in (14, 28, 89):  # near-separatrix, fast tumbler, slow tumbler
    npz = np.load(f"data/trajectories/traj_seed{seed:03d}.npz")
    q0, omega0, I = npz["q0_wxyz"], npz["omega0_rad"], npz["inertia_tensor"]
    times = npz["observation_times"]
    cached_k1_body = npz["k1_body"]  # truth k1 from the trajectory generator

    q_dop, _ = propagate_attitude(q0, omega0, times, "tumbling", I)
    q_jac, _ = propagate_jacobi(q0, omega0, I, times)

    # GATE 1: q matches DOP853 to machine precision (absorb antipode sign)
    d_q = min(
        np.max(np.linalg.norm(q_jac - q_dop, axis=1)),
        np.max(np.linalg.norm(q_jac + q_dop, axis=1)),
    )
    assert d_q < 1e-9, f"seed {seed}: q drift {d_q:.2e}"

    # GATE 2: conserved 2T and L^2 constant to machine precision in Jacobi
    twoT = np.einsum("ni,ij,nj->n", omega_jac, I, omega_jac)
    L2 = np.einsum("ni,ni->n", omega_jac @ I, omega_jac @ I)
    assert np.std(twoT) / np.mean(twoT) < 1e-12
    assert np.std(L2) / np.mean(L2) < 1e-12

    # GATE 3: k1_body via the renderer matches cached truth
    k1_jac, _, _ = propagate_to_body_frame_with_quats(q_jac, ...)
    assert np.max(np.linalg.norm(k1_jac - cached_k1_body, axis=1)) < 1e-12
```

**If GATE 1 fails by a sign-flip or factor-of-2 pattern**, the bug is in the convention of the Jacobi q(t) derivation, NOT a "small numerical issue" to fix with a wrapper. Reread `concepts/quaternion_convention.md` and trace it. The temptation will be to slap a `.conjugate()` somewhere — resist. Find the actual issue.

**Exit criterion.** All three gates pass on all three seeds. Save the round-trip evidence in `results/s062b/round_trip_seedXXX.json`.

**Artefacts.**
- `experiments/s062b_jacobi_quaternion.py`.
- `experiments/s062b_jacobi_quaternion.md`.
- `results/s062b/round_trip_seedXXX.json`.
- `lib/jacobi_propagator.py` (the production module, exporting `propagate_jacobi`).

## s062c — wire into lib/forward.py + profile (~1 h)

**Goal.** Make `propagate_jacobi` a drop-in replacement for `propagate_attitude` in `lib/forward.py::propagate_to_body_frame`, gated by a `method="jacobi"` flag with `method="dop853"` as the (unchanged) default. Confirm the speedup on a realistic batch.

- Add the parameter to `propagate_to_body_frame`. Default unchanged.
- Profile both methods on a 1000-candidate × 500-epoch synthetic batch. Report wall, throughput, and any precision delta in `k1_body` / `mag` outputs.
- If the precision delta is non-zero at the surrogate-eval level (i.e., changes which (q_a, ω) wins by surrogate-MSE on a real seed's score grid), investigate before flipping the default.

**Exit criterion.** Drop-in wired, both paths verified equivalent on a realistic batch, speedup measured.

**Artefacts.**
- Modified `lib/forward.py`.
- `experiments/s062c_profile.md` with the wall numbers.

## s062d — sample on (q_a, |L|, label, phase) for the s061 reframe (~2–3 h)

**Goal.** Make the constant-ω cloud-threading reframe (`s061` next iteration) sample candidates directly on the polhode-parameter basis instead of (q_a, ω_a uniform).

This is the inversion-architecture win — it's why s062a–c are worth doing. The work:
- Implement `sample_polhode_candidates(q_a_pool, |L|_range, n_per_qa, inertia)` returning candidates parameterized as (q_a, |L|, polhode-label, polhode-phase).
- For each candidate, evaluate the closed-form `(q(t), ω(t))` over the full LC at all 500 epochs in one vectorized call.
- Brightness-filter at every epoch (per s061 architecture). The surviving (q_a, |L|, label, phase) tuples are coherent candidate trajectories — no lineage drift, no random-walk decoupling.

**Exit criterion.** Sample-time of 100k candidates × 500 epochs in < 15 min Pool(8). Truth-(q_a, |L|, label, phase) survives all-epochs filter on a slow seed (89) and a fast seed (28).

**Artefacts.**
- `experiments/s062d_polhode_sample.py`.
- `experiments/s062d_polhode_sample.md`.
- `lib/polhode_sample.py`.

# Risks and how to manage them

| Risk | Mitigation |
|---|---|
| Convention bug introduced in q(t) | The 3-gate validation in s062b is non-negotiable. No "I'll fix the sign later" — find the issue. |
| Inertia is not in PA frame | Standard preprocessing step, not a special case: eigendecompose once in s062a, bake R_pa as a constant matrix in the propagator helper. Algorithm is unchanged. |
| Near-separatrix numerical conditioning | Test on seeds 14/17/68/81 explicitly in s062a; if Jacobi loses precision near k → 1, fall back to the hybrid (analytical ω, Magnus q) in that regime only. |
| `scipy.special.ellipj` precision | scipy uses AMS-55 algorithm; check against `mpmath.ellipfun` on a few cases. |
| Convention drift between Jacobi q(t) and renderer | GATE 3 catches this. If only GATE 3 fails (not GATE 1), the renderer expectation is inconsistent with the Jacobi convention — fix the conversion at the boundary, not in the math. |
| Implementation expands scope (e.g., refactoring `propagate_attitude`) | DON'T. `propagate_jacobi` is additive. The DOP853 path remains the default until s062c profile justifies switching. |

# Out of scope for s062

- Polishing with Jacobi-only candidates (i.e., LM polish path). The polish path runs ~50–500 propagator calls per seed; the speedup is modest there. Leave for s063+ after the cloud-threading reframe lands.
- General asymmetric-rigid-body work outside torque-free. We have torque-free explicitly.
- Refactoring `propagate_attitude` itself. It stays untouched.

# Cross-references

- `concepts/jacobi_propagation.md` — the math + literature.
- `concepts/polhode_prior.md` — what the parameterization is for.
- `concepts/quaternion_convention.md` — the convention the q(t) must respect.
- `experiments/s061c_seed28_thread.md` — the architectural forcing function.
- `experiments/s059j_design.md` — design-doc template / structure.
- `PROGRESS.md` "Track J (Jacobi)" — live status of this thread.
- Auto-memory: `project_jacobi_propagation_priority.md`.
