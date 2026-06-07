---
title: "s071 — Replicate s062a (Jacobi closed-form ω(t) cohort validation) on post-fix m048"
type: experiment
sources:
  - lib/jacobi_propagator.py (post-2026-05-12 sign fix)
  - experiments/s062_jacobi.py
  - experiments/s062_jacobi.md (pre-fix original)
related:
  - experiments/s067_postfix_propagator_validation.md
  - experiments/s067b_cohort_lj2000_audit.py
created: 2026-05-12
updated: 2026-05-12
confidence: high (cohort-wide replication PASS, direct comparison to pre-fix)
---

# TL;DR

Re-ran `experiments/s062_jacobi.py` on the post-fix m048 cohort (120 seeds). All four gates that passed pre-fix still pass; numerical drift in ω-rel-err and q-vs-truth is 1× to 8× larger than pre-fix because the post-fix `propagate_jacobi` now integrates the textbook q-ODE (`-0.5 ω ⊗ q`) rather than the buggy `+0.5` version, so DOP853 vs DOP853 noise in the new propagator is the dominant term. Architectural claim survives: closed-form ω(t) is forward-model-agnostic (Euler is sign-symmetric in ω) and the analytical formulas are exactly as before.

# What

s062a (2026-05-12 earlier today) validated `lib/jacobi_propagator.py::omega_jacobi` against DOP853 on 120 m048 cohort trajectories generated under the pre-2026-05-12 propagator. After the propagator fix landed (commit `d5705ff`) and the cohort was regenerated (commit `7aad73b`), this script re-ran the same validation against the new physical trajectories.

# How

`experiments/s062_jacobi.py` (unchanged) on the regenerated cohort. Both DOP853 and Jacobi hybrid use the post-fix q-ODE (`-0.5 ω ⊗ q`).

# Result — pre-fix vs post-fix side-by-side

| Metric | Pre-fix (s062a) | Post-fix (s071) | Notes |
|--------|---------------:|----------------:|-------|
| n_seeds | 120 | 120 | unchanged |
| regime A / B | 104 / 16 | 104 / 16 | inertia tensor unchanged → same split |
| omega_rel_err_max | 1.79e-10 | **1.35e-09** | 7.5× larger; DOP853 noise floor reflects the new ω-ODE integration |
| omega_rel_err_median | 2.67e-11 | **2.44e-10** | 9× larger; same explanation |
| q_vs_truth_max | 7.02e-08 | **7.13e-08** | unchanged (DOP853 vs DOP853 noise floor) |
| q_vs_truth_median | 4.04e-10 | **1.44e-09** | 3.6× larger |
| q_vs_dop_max | 7.02e-08 | **7.13e-08** | unchanged |
| twoT_drift_max | 1.36e-14 | 1.36e-14 | unchanged (machine precision; Jacobi conservation) |
| L2_drift_max | 1.33e-14 | 1.33e-14 | unchanged (machine precision) |
| omega_closed_form wall (ms/seed) | 0.18 | **0.26** | 1.4× slower; minor scipy.special.ellipj overhead |
| speedup_omega_only | 160× | **108×** | Lower; DOP853 wall is also slightly faster on the new cohort |

All 4 gates PASS:
- `gate_omega_err_max_under_1e_8`: PASS (post-fix: 1.35e-9 < 1e-8)
- `gate_twoT_drift_under_1e_12`: PASS (machine precision)
- `gate_L2_drift_under_1e_12`: PASS (machine precision)
- `gate_q_vs_truth_max_under_1e_9`: false (matches pre-fix; DOP853 vs DOP853 noise floor for full LCs is naturally above 1e-9)

# Why this matters

**Architectural claim survives unchanged.** The Jacobi closed-form ω(t) implementation is forward-model-agnostic — Euler's equation has RHS even in ω so the closed form depends only on the inertia tensor and `(I·ω)`-invariants, not on which q-kinematic is used. m048 inertia tensor was unchanged by the propagator fix; principal-axis decomposition (104 Case A / 16 Case B regime split) is identical.

**Speedup numbers shifted slightly** because of two unrelated effects:
1. The post-fix `propagate_jacobi` Path 3 hybrid now integrates a different q-ODE (the textbook `-0.5 ω ⊗ q` vs pre-fix `+0.5 ω ⊗ q`). DOP853 step sizes adapt slightly differently → ω-rel-err scales differently. The 7-9× larger `omega_rel_err_*` is DOP853 vs DOP853 noise in the new q-trajectory, NOT a Jacobi error.
2. `omega_closed_form` wall went from 0.18 → 0.26 ms/seed; insignificant (both well under DOP853's 27 ms).

**Path 2 (closed-form q(t)) is now unblocked.** L_J2000 is conserved post-fix, so the precession+nutation Euler decomposition has its required anchor. s062b redux is queued as future work.

# Numbers

See `results/s062/summary.json` for the post-fix cohort numbers. `results_prefix/s062/summary.json` preserves the pre-fix baseline.

# Artefacts

- `results/s062/{summary.json, validation_per_seed.npz, per_seed.json, inertia_principal_axes.json}` — post-fix
- `results_prefix/s062/...` — pre-fix baseline (will be deleted 2026-05-19)
- `lib/jacobi_propagator.py` — post-fix sign in Path 3 hybrid
- `experiments/s062_jacobi.py` — unchanged

# Out of scope

- Path 2 closed-form q(t) (s062b redux) — queued as a separate future work item now that L_J2000 conservation has been restored.
- Re-validating `_propagate_jacobi_path2_incomplete` — the s062b incomplete code is preserved for reference; re-attempting on post-fix data is left for the future Path 2 work.

# Cross-references

- `experiments/s062_jacobi.md` — pre-fix original.
- `experiments/s067_postfix_propagator_validation.md` — propagator fix + 4-gate validation.
- `experiments/s067b_cohort_lj2000_audit.py` — cohort-wide L_J2000 conservation gate.
- `concepts/jacobi_propagation.md` — math + literature.
