---
title: "s002 — surrogate landscape probe (PA-stratified seeds, Sobol-SO(3))"
type: experiment
sources:
  - data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed*.npz (post-fix; commit ac1fdf4)
  - data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz (inertia tensor)
  - ~/surrogate_model (v2 residual ensemble; bridge-independent)
  - lib/forward.py (post-fix propagator + scipy Rotation; round-trips cached k1/k2 at machine precision)
related:
  - experiments/s001_cost_at_truth_cohort.md (Q1 — cost-at-truth)
  - concepts/known_pathologies_to_revalidate.md (m145 deceptive q0=135° claim on seed 91)
  - concepts/surrogate_model.md (bridge-independence)
  - concepts/twin_degeneracy.md (q_180x · q0 antipode)
  - concepts/quaternion_convention.md (post-fix propagator + xyzw scipy reshuffle)
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s002 — surrogate landscape probe

## TL;DR

On 8 PA-stratified m048 seeds (anchors 6, 10, 91 + PA picks 21, 41, 48, 60, 84) under post-fix truth, with ω **fixed at truth-ω**, the surrogate full-LC MSE landscape was probed at 2046 Sobol-Shoemake quaternions plus the truth-q0 and its body-x twin (q_180x · q0):

- **Argmin sits at truth-q0 on 8/8 seeds.** Including seed 91 — the m145 "deceptive q0=135° attractor" claim is **REFUTED** under correct truth.
- **The basin around truth is narrow but very deep.** The best sobol candidate (closest beating attempt) has MSE 52× to 1975× larger than truth-MSE on every seed. Closest sobol-to-truth geodesic is 4°–12° depending on grid placement — yet even those nearby points are well above truth.
- **The body-x twin is NOT a competing minimum at fixed truth-ω.** twin_mse is between 1.1 and 10.2 mag² across the 8 seeds (vs truth_mse between 1.3e-4 and 3.6e-3). Twin degeneracy applies at the LC level only when ω co-rotates appropriately — it does not survive ω being held at truth-ω.
- **No deceptive attractor anywhere on the sobol probe.** No seed has a sobol candidate beating truth (`n_sobol_below_truth = 0` for all 8 seeds).

Decision: **surrogate-MSE is structurally honest under correct truth, when ω is at truth.** Solver failure under m115's per-ω DE on seed 91 (m145) was not driven by a deceptive landscape — it was either solver-side (DE escape, ω misspecification) or conditional on ω being away from truth (Q3 territory). The buggy-era priors saying "the surrogate landscape lies on seed 91" are no longer load-bearing; they were artifacts of conv-(b) propagation feeding a conv-(a) renderer.

## What

Sample SO(3) at low discrepancy via 2046 Sobol-Shoemake quaternions, hold ω = ω_truth, propagate (q0_c, ω_truth) under the post-fix tumbling propagator using each seed's cached observation_times + sun/obs/sat positions, project sun/observer into the resulting body frames, evaluate the surrogate, and score full-LC MSE vs cached `mag_hifi`. Insert truth-q0 at index 0 and twin (q_180x · q0_truth) at index 1 so both are exactly evaluated rather than approximated by nearest-grid-neighbour. Score one seed at a time using `Pool(8)` workers with BLAS=1 in each worker.

## How

- **Seed picks (8):** anchors 6 (m141 elevated surrogate-floor), 10 (s001 max surr_full_mse=3.6e-3), 91 (m145 deceptive-q0 claim) + PA-stratified 21, 41, 48, 60, 84. PA span 9.9°–88.1° from `s001/per_seed.csv`.
- **Grid:** `scipy.stats.qmc.Sobol(d=3, scramble=True, seed=42)`, 2046 points, mapped through Shoemake's uniform-S^3 formula (in `experiments/s002_*.py:shoemake_to_quat`). Plus truth + twin at indices 0 and 1.
- **Propagation:** `lib/forward.py:propagate_to_body_frame` — wraps `src.dynamics.attitude_propagator.propagate_attitude(mode="tumbling", inertia_tensor=...)` plus per-epoch `Rotation.from_quat([qx,qy,qz,qw]).as_matrix()` projection of sun/obs into body frame. Inertia tensor pulled from m048 master NPZ (3.799e4, 3.831e4, 7.749e3 kg·m² diagonal).
- **Round-trip validation:** `propagate_to_body_frame(q0_truth, ω_truth, ...)` reproduces cached `k1_body / k2_body / quaternions` at machine precision (max|Δ|=0) on seeds 6, 10, 91. So the propagation pathway used to score the grid is bit-identical to the one that generated the cached truth.
- **Scoring:** `surrogate_eval.full_lc_mse(predicted, mag_hifi)` — full LC, no bright-mask; `bright_mse` also recorded.
- **Geodesic metric:** `quat_geodesic_deg(q1, q2) = 2·arccos(|q1·q2|)`, antipode-aware so q and −q both give 0°. Reported as min(geo_to_truth, geo_to_twin) for landscape plotting.
- **Wall:** ~25 s/seed × 8 = 199 s total (Pool(8), 2048 candidates/seed, 500 obs each, BLAS=1 in workers).

## Result

Per-seed table (truth_mse, best non-truth sobol candidate, ratio, sobol-grid spacing near truth):

| seed | truth_mse | best_sobol_mse | best_sobol/truth | best_sobol_geo to truth/twin | min_sobol_geo to truth/twin | argmin |
|------|-----------|----------------|------------------|-------------------------------|------------------------------|--------|
|    6 | 2.92e-03  | 5.08e-01       | **174×**         | 154.9°                        | 9.1°                         | truth  |
|   10 | 3.59e-03  | 1.90e-01       | **53×**          | 75.7°                         | 10.9°                        | truth  |
|   21 | 5.77e-04  | 1.14e+00       | **1975×**        | 7.6°                          | 7.6°                         | truth  |
|   41 | 1.33e-04  | 4.09e-02       | **307×**         | 108.5°                        | 4.7°                         | truth  |
|   48 | 3.08e-04  | 1.79e-01       | **580×**         | 138.1°                        | 12.1°                        | truth  |
|   60 | 1.56e-03  | 6.46e-01       | **416×**         | 176.8°                        | 9.8°                         | truth  |
|   84 | 2.56e-04  | 2.01e-01       | **786×**         | 137.5°                        | 4.2°                         | truth  |
|   91 | 2.03e-04  | 9.64e-02       | **475×**         | 35.0°                         | 12.4°                        | truth  |

`argmin = truth` on every seed; `n_sobol_below_truth = 0` on every seed. Twin MSE is between 1.1 and 10.2 mag² — well above the sobol cloud — confirming that under fixed truth-ω the body-x twin is NOT a degenerate basin.

## Why this matters

1. **m145's "deceptive q0=135° attractor on seed 91" was a buggy-era artifact.** Under post-fix truth, with ω at truth-ω, the surrogate landscape on seed 91 has no competing basin within sobol resolution (~5°). Truth dominates by 475× MSE over the best sobol candidate. The m145 "REFUTED" verdict on m115's patch chain was real (the patches didn't bridge q0), but the diagnosis ("surrogate landscape lies on this seed") was not — solver failure was not landscape-driven.
2. **Surrogate-MSE is a structurally honest cost when ω is correct.** The basin-at-truth is unambiguous and dominant. Any inversion that knows ω can polish q0 from a local basin via local search; the landscape doesn't need to be re-engineered.
3. **The buggy-era "argmin elsewhere" claim from m103 alignment-cost work is NOT addressed by this experiment.** That claim is about the m103 alignment surface (different cost), not the surrogate-MSE surface. s002 establishes that surrogate-MSE is honest; whether m103 alignment cost is also honest is Q3.
4. **The basin is narrow.** The closest sobol point to truth (4°–12° geodesic depending on seed) is already 50×–2000× above truth. This means a sobol-style global search would need ≳ 10⁵ candidates to land *inside* the basin; it would still detect that "everything is far above truth", but not find truth itself by random sampling. Implication for inversion: global search on this surface is necessary AND insufficient — must be paired with local polish from a near-truth basin entry.
5. **Twin degeneracy at fixed-ω is broken.** This is a useful negative result: if you fix ω at truth-ω and ask "is q_180x · q0 also a minimum?", the answer is **no** — twin_mse is huge. So the twin degeneracy that the inversion literature has worried about only operates when ω is also free to flip — which is the realistic case but the survey hasn't yet addressed.

## Numbers

- N seeds: **8**
- N candidates per seed: **2048** = 1 truth + 1 twin + 2046 Sobol-Shoemake
- N seeds with argmin at truth: **8/8**
- N seeds with `n_sobol_below_truth > 0`: **0/8**
- truth_mse range: **1.33e-4 to 3.59e-3 mag²** (ρ-equiv 0.23 to 1.20; all Band A)
- twin_mse range: **1.10 to 10.19 mag²** (massive — not competing basins)
- best_sobol_mse / truth_mse ratio range: **53× to 1975×** (median ≈ 416×)
- closest sobol-to-truth/twin geodesic: **4.2° to 12.4°**
- wall: **199 s** for 8 seeds × 2048 candidates with Pool(8), BLAS=1

## Artefacts

- `experiments/s002_surrogate_landscape_probe.py` — the script (Pool(8), Sobol-Shoemake, save NPZ + JSON + PNG).
- `experiments/s002_surrogate_landscape_probe.md` — this writeup.
- `lib/forward.py` — new helper, `propagate_to_body_frame` and `quat_geodesic_deg(_batch)`.
- `results/s002/per_seed_landscape.npz` — q_grid + kind + full_mse + bright_mse + geo arrays per seed.
- `results/s002/summary.json` — argmin / truth-rank / best-sobol stats per seed.
- `results/s002/landscape_grid.png` — 2×4 panel: MSE vs min-geo-to-(truth,twin) per seed; truth (red star), twin (orange diamond), argmin (black X), sobol cloud (blue).

## Out of scope

- **Landscape under ω-misspecification.** s002 fixes ω at truth-ω. The realistic inversion problem has ω uncertain. m115's per-ω DE was failing on seed 91 *during ω search*, not after ω was nailed — so the s002 result doesn't fully exonerate the surrogate landscape from an inversion-failure perspective. This is **Q3 territory**: vary ω over a coarse grid, re-probe q0 landscape, look for ω values where deceptive q0 attractors emerge.
- **Sub-sobol-resolution basins.** Sobol spacing is ~5°. A deceptive basin narrower than that could be missed. s002 doesn't bound this rigorously, only argues by the depth of truth-MSE that any plausible *competing* basin would need to be detectable at sobol resolution to fool a global solver — and none is.
- **m103 alignment-cost surface.** Different cost; not probed here. The "argmin << truth" rank claim from m135 lives on the alignment surface and is still on the Q3 docket.
- **Other satellites / inertia tensors.** All seeds use the m048 IS-901 inertia tensor.

## Cross-references

- **Q1 (s001):** established surrogate-MSE-at-truth is universally Band-A and is the only cost surface defined cohort-wide. s002 extends this to: surrogate-MSE-at-truth is also the **argmin** on the local sobol probe, on every PA-stratified seed.
- **m145 (frozen reference):** `notebooks/inversion/wiki/wiki/experiments/m145_*.md` — claimed seed-91 surrogate MSE has deceptive q0=135° basin under buggy truth. **REFUTED at sobol resolution under post-fix truth.** Memory entry `feedback_q_from_w_solver_breaks_post_fix.md` is now in tension with this finding for the landscape clause; the per-ω DE escape claim still stands as a separate (untested-here) claim.
- **m141 (frozen reference):** seed-6 elevated surrogate MSE noted (~3e-3). s002 reproduces that floor (2.9e-3) and shows it's still the basin minimum despite the elevated absolute value.
- **Quaternion convention (concepts/quaternion_convention.md):** the round-trip Δ=0 sanity in `lib/forward.py` is the per-experiment realisation of the convention smoke test.
- **Twin degeneracy (concepts/twin_degeneracy.md):** s002 finds twin (q_180x · q0) is NOT a fixed-ω degenerate minimum. Twin degeneracy operates over (q0, ω) jointly, not q0 alone — concept page is consistent but worth highlighting this fixed-ω corollary.
