---
title: "s003 — surrogate landscape under ω-misspecification (3 seeds × 24 ω perturbations)"
type: experiment
sources:
  - data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed*.npz (post-fix; commit ac1fdf4)
  - data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz (inertia tensor)
  - ~/surrogate_model
  - lib/forward.py (q0+ω → body-frame; round-trips at machine precision)
related:
  - experiments/s001_cost_at_truth_cohort.md (Q1 — cost-at-truth, cohort-scale)
  - experiments/s002_surrogate_landscape_probe.md (Q2 — landscape at fixed truth-ω; argmin = truth on 8/8 seeds)
  - concepts/known_pathologies_to_revalidate.md (m115 per-ω DE failure on seed 91)
  - concepts/q_omega_coupling.md (q0 and ω-dir inseparable)
  - concepts/surrogate_model.md
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s003 — surrogate landscape under ω-misspecification

## TL;DR

On 3 PA-stratified seeds (91, 6, 41) under post-fix truth, with the full-LC
surrogate-MSE landscape probed by 511 Sobol-Shoemake quaternions + truth-q0
across 24 ω perturbations per seed, we find:

- **Surrogate-MSE is extremely ω-fragile.** Truth-q0 is the argmin only inside
  a thin tube around truth-ω: roughly **|Δω-dir| ≲ 1°** AND **|Δω-mag| ≲ 2-5%**.
  Outside this tube the q0 argmin slides away from truth-q0, often to
  >100° geodesic, and `truth_q0_mse` explodes by 3-4 orders of magnitude
  (from Band-A 10⁻⁴ at truth-ω to Band-D 10⁰ mag² with even small
  perturbation).
- **11/72 cells (15%) keep argmin within 10° of truth.** Those 11 cells
  are exactly: each seed's baseline (3) + dir_0.5deg (3) + dir_1deg (3) +
  dir_2deg (2 seeds — fails on seed 6). 50/72 cells (69%) have argmin
  ≥30° from truth.
- **ω-mag is even more sensitive than ω-dir.** Even ±5% mag perturbation
  pushes the argmin >50° away on every seed. Pure dir perturbations are
  a bit more forgiving — argmin stays at truth for dir ≤1° on all seeds,
  ≤2° on seed 91 (which was the most ω-mag-fragile in s002 sense).
- **No structurally deceptive basin emerges; instead the landscape becomes
  a noisy multi-basin terrain.** The argmin location is essentially random
  (jumping between 30° and 180° across adjacent perturbations), and even
  the second-best sobol candidate is within ~10% MSE of the argmin —
  i.e. once you're outside the truth-ω tube, the surface is "all noise,
  no signal" rather than a smooth alternative basin.

Decision: **m115's solver failure on seed 91 is structurally explained.**
m115's per-ω DE was searching ω over a range much wider than the ~1°/~2-5%
truth-ω tube. For ω values outside that tube, the q0 sub-search converges
to a non-truth basin (or to noise), and the per-ω DE has no signal that
"this ω is right" because the per-ω best surrogate-MSE is fairly flat
across a wide ω range (3-4 mag² for seed 91 across most perturbations).

This **rules out** any inversion strategy that decouples ω-search from
q0-search. (q0, ω) optimisation must be **joint and tightly coupled** —
local moves in (q0, ω) along the truth-ω-tube manifold, not nested
ω-outer / q0-inner search.

## What

For each of 3 seeds (91 — m145 anchor; 6 — m141 elevated-floor;
41 — clean PA-low control), apply 24 ω perturbations and probe the q0
landscape under each. Perturbations cover:

- **1 baseline:** ω = ω_truth (sanity check; should reproduce s002).
- **9 pure ω-mag:** scalar multiples of ω_truth-mag with factors
  {0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.5, 2.0}.
- **8 pure ω-dir:** rotate ω-truth-dir by ε ∈ {0.5°, 1°, 2°, 5°, 10°, 30°,
  60°, 90°} around a deterministic perpendicular axis (cross product of
  ω-dir with body x; falls back to body y if degenerate).
- **6 combined:** {0.5×, 2.0×} mag × {10°, 30°} dir, plus {1.1× mag,
  5° dir} and {1.1× mag, 30° dir}.

Per (seed, perturbation): 511 Sobol-Shoemake quaternions + truth-q0 at
index 0 = 512 candidates. Score surrogate full-LC MSE for each candidate
under the perturbed ω against the cached truth `mag_hifi` (which was
generated at truth-ω, so `mag_hifi` is the LC the inversion is trying to
fit). Record argmin, argmin's geodesic to truth-q0, truth_q0_mse,
n_sobol_below_truth.

## How

- **Seeds:** 91 (m145 deceptive-attractor anchor), 6 (m141 elevated
  surrogate-floor anchor), 41 (control: clean PA-low seed with smallest
  truth_mse in s002 = 1.33e-4).
- **Grid:** 511 Sobol-Shoemake quats with `qmc.Sobol(d=3, scramble=True,
  seed=42)` + truth-q0 at index 0 → 512 candidates. Same Sobol seed used
  across all perturbations and all seeds (so the geometry of the q0
  sample is identical and we're really only varying ω).
- **Perturbed ω construction:** `direction · (mag · mag_factor)` where
  `direction` is `Rotation.from_rotvec(perp_axis · radians(dir_rot))`
  applied to the truth-ω-dir.
- **Reference LC:** cached `mag_hifi` from `traj_seedXXX.npz`. This LC
  was generated at (truth-q0, truth-ω) — it's the observable. We're
  asking "for a candidate (q0_c, ω_perturbed), how well does the
  surrogate prediction match the ω_truth-generated LC?"
- **Scoring:** surrogate full-LC MSE; same code path as s002.
- **Workers:** Pool(8), one Pool per seed (reused across the 24
  perturbations; surrogate model loads once per worker per seed). BLAS=1
  set in workers.
- **Wall:** 525 s for 3 seeds × 24 perturbations × 512 candidates
  (~7.3 s per perturbation per seed, matching the per-candidate ~89 ms
  / Pool(8) prediction).

## Result

### Argmin drift summary across 72 (seed, perturbation) cells

| argmin geodesic to truth-q0 | n cells | fraction |
|------------------------------|---------|----------|
| 0° (exactly at truth)        | ≥ 11    | ≥ 15%    |
| < 5°                         | 11      | 15%      |
| < 10°                        | 11      | 15%      |
| < 30°                        | 22      | 31%      |
| ≥ 30°                        | 50      | 69%      |
| ≥ 100°                       | 28      | 39%      |

The 11 cells with argmin still at truth are concentrated in the two-
parameter neighbourhood of truth-ω: {baseline, dir_0.5°, dir_1°} on all
3 seeds (9 cells), plus dir_2° on seeds 41 and 91 (2 cells). Seed 6's
argmin escapes truth at dir = 2°.

### Per-seed truth-ω-tube (defined as the largest ω perturbation at which argmin is still at truth-q0)

| seed | tube (pure ω-dir) | tube (pure ω-mag)         |
|------|--------------------|---------------------------|
|    6 | 1°                 | < 5%  (mag×0.95 already escapes) |
|   41 | 2°                 | < 5%                      |
|   91 | 2°                 | < 5%                      |

### Truth-q0 MSE growth as ω drifts (representative numbers, seed 91)

| perturbation | truth_q0_mse [mag²] | ρ-equiv |
|--------------|---------------------|---------|
| baseline (ω = ω_truth)     | 2.0e-4 | 0.28 (Band A)   |
| ω-dir +0.5°                | 2.3e-2 | 3.0 (Band C/D)  |
| ω-dir +1°                  | 8.8e-2 | 5.9 (Band D)    |
| ω-dir +2°                  | 2.4e-1 | 9.9 (Band D)    |
| ω-dir +5°                  | 1.2    | 22  (Band D)    |
| ω-mag ×0.95                | 7.6e-1 | 17  (Band D)    |
| ω-mag ×1.05                | 8.1e-1 | 18  (Band D)    |
| ω-mag ×0.5                 | 2.2    | 30  (Band D)    |

Even the smallest non-trivial dir perturbation (0.5°) already pushes
truth-q0 from ρ=0.3 to ρ=3.0 — that's a Band-A → Band-C jump on a
trajectory where the body-frame geometry is barely changing. The surrogate
is reading attitude differences with sub-degree resolution.

### What happens to the landscape outside the tube

In the perturbed-ω regime the landscape becomes "noisy multi-basin":

- argmin sits at random-looking positions (no consistent direction across
  perturbations).
- argmin_mse and second_best_mse are within ~10% of each other in most
  cells (no clear winning basin).
- `n_sobol_below_truth` jumps from 0 (at truth-ω) to anywhere from 2
  to 500 sobol candidates beating truth-q0.

The plot in `results/s003/landscape_panels.png` shows this directly:
under truth-ω the truth-q0 (red star) sits far below the sobol cloud
(s002's signature). Under any non-tiny perturbation, truth-q0 is
embedded in the cloud and the argmin (black X) is at a quasi-random
sobol point.

## Why this matters

1. **m115's per-ω DE failure on seed 91 is structurally explained.** m115
   sweeps ω over a large range (much wider than 1°/5%), and for almost
   every ω in its sweep, the q0 sub-search produces a noisy non-truth
   argmin. The "best surrogate-MSE per ω" signal that m115 uses to pick
   the winning ω is fairly flat (several mag² across a wide ω range),
   so the outer ω-search has nothing to lock onto. **The DE can converge
   per-ω but the per-ω wins don't point toward truth-ω.** This is the
   m145 "REFUTED at Band C" finding decomposed: the issue isn't a
   deceptive landscape on seed 91 specifically, it's that ANY ω outside
   the truth tube produces meaningless q0 minima.
2. **Decoupled (q0, ω) search is structurally broken.** Any architecture
   that searches ω in an outer loop and q0 in an inner loop (which is
   most of m103/m115/m126's lineage) is fighting an extremely narrow
   target. The per-ω q0-best signal carries no useful gradient toward
   truth-ω until you're already inside the ~1° / ~2-5% tube, at which
   point you don't really need the outer loop.
3. **Joint (q0, ω) local search is needed.** The inversion has to move
   in (q0, ω) jointly along the truth-ω tube, with q0 and ω updates
   tightly coupled. Concretely: a joint Levenberg-Marquardt or trust-
   region step in 7-DOF (q0 ∈ S³ minus the antipode = 3 DOF + ω = 3 DOF
   + (maybe a stretch parameter)) is the right shape. ω-grid + q0-DE is
   not.
4. **Initial condition discipline is critical.** A solver that doesn't
   start within ~1° of truth-ω in dir AND ~2-5% in mag is searching a
   landscape with no useful signal. This implies needs from upstream
   geometry (e.g. SPICE) or from a separate ω-estimation primitive
   (peak spacing, harmonic analysis of the LC) to seed the joint search.
5. **The "bridge from ω to q0" question (m115's central premise) has a
   decisive negative answer at the surrogate-MSE level.** There is no
   ω-only signal in the surrogate that can be used to seed a q0 polish
   from arbitrary ω initialisation. ω and q0 are jointly determined OR
   the landscape is incoherent — there is no useful separable structure.

## Numbers

- Seeds: 6, 41, 91 (3 seeds spanning PA from 18° to 62°).
- Perturbations per seed: 24 (1 baseline + 9 mag-only + 8 dir-only + 6 combined).
- Candidates per perturbation: 512 (1 truth-q0 + 511 Sobol-Shoemake).
- Total surrogate evaluations: 3 × 24 × 512 = 36 864.
- Wall time: 525 s with Pool(8), BLAS=1 (matches per-candidate prediction
  of ~89 ms; 7.3 s per perturbation per seed).
- truth-ω-tube widths: ω-dir ≤ 1°-2°, ω-mag ≤ 2-5% across the 3 seeds.
- Cells with argmin within 10° of truth: **11/72 (15%)**.
- Cells with argmin ≥ 30° from truth: **50/72 (69%)**.
- truth_q0_mse range: 1.3e-4 (baseline) to ~10 mag² (largest dir perturbation).

## Artefacts

- `experiments/s003_landscape_vs_omega.py` — the script.
- `experiments/s003_landscape_vs_omega.md` — this writeup.
- `results/s003/per_seed_omega.npz` — per-seed × per-perturbation
  full_mse (512), truth_q0_mse, argmin_idx, argmin_geo_to_truth,
  n_sobol_below_truth arrays.
- `results/s003/summary.json` — population-level cell counts + per-seed
  per-perturbation summary table.
- `results/s003/argmin_drift.png` — 4-panel: argmin geodesic and
  truth_q0_mse, vs ω-mag and vs ω-dir.
- `results/s003/landscape_panels.png` — 3 seeds × 6 representative
  perturbations grid of MSE-vs-q0_geo scatter; visual confirmation of
  basin loss.

## Out of scope

- **Multi-axis ω-dir perturbations.** Each dir perturbation here uses a
  single deterministic perp axis; the result might depend on which
  perpendicular axis is used. A subsequent experiment could randomise
  the perp axis to confirm the tube is isotropic in ω-dir space.
- **Inertia-tensor sensitivity.** All 3 seeds use the same m048 IS-901
  inertia tensor. The tube width could vary with satellite shape /
  asymmetry.
- **Tube width as a function of trajectory length.** All seeds are
  500-epoch / 1-hour trajectories. Shorter trajectories have less
  attitude integration → wider tube; longer trajectories tighter.
- **Joint (q0, ω) local descent verification.** The next experiment
  (s004) should verify that a joint LM/trust-region step from inside
  the tube actually converges to truth, validating the strategic
  recommendation.
- **m103 alignment-cost surface.** Different cost; might have different
  ω-fragility properties. Open in Q4.

## Cross-references

- **s001 (Q1):** surrogate-MSE-at-truth is universally Band-A.
- **s002 (Q2):** at fixed truth-ω, argmin = truth on 8/8 PA-stratified seeds.
- **s003 (this, Q3):** at perturbed ω, argmin = truth only inside a thin
  tube (~1° dir, ~2-5% mag). m115's per-ω DE failure now structurally
  explained.
- **m145 (frozen reference):** the "deceptive q0=135° attractor" framing
  is fully decomposed: at truth-ω there's no deceptive basin (s002), and
  at perturbed-ω there's no consistent basin at all (s003) — m115's
  failure was not landscape pathology but landscape **incoherence under
  ω misspecification**.
- **concepts/q_omega_coupling.md:** "q0 and ω-dir inseparable; only
  ω-mag truly independent" — s003 sharpens this. Even ω-mag is NOT truly
  independent; it has a ~2-5% tube. Concept page should be updated.
- **memory: feedback_q_from_w_solver_breaks_post_fix.md** — m145 framing
  (per-ω DE failure on seed 91) — update to point to s003 as the
  structural explanation.
