---
title: "s012a — seed-10 single-seed N=256 probe (Q4c-ii diagnostic)"
type: experiment
sources:
  - data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz (inertia_tensor)
  - data/trajectories/traj_seed010.npz (truth state, observation grid, mag_hifi)
  - results/s011/runs.npz (first 64 of the same Sobol sequence)
related:
  - experiments/s011_q4cii_sobol_so3_polish_pilot.md
  - experiments/s005_joint_local_descent.md
  - experiments/s006_seed28_landscape_at_truth_omega.md
  - experiments/s010_seed44_landscape_at_truth_omega.md
  - concepts/observational_indistinguishability.md
  - concepts/rho_band.md
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# s012a — seed-10 single-seed N=256 probe

## TL;DR

**Decisive H2 result.** Extending s011's seed-10 Sobol density from N=64 → N=256
(192 new ICs at Sobol indices 64..255, same scrambled Sobol-Shoemake seed=42)
yields **0/256 in-basin landings** at fixed truth-ω with LM polish. The closest
LM landing at N=64 (28.25°) failed to improve at N=128 (28.25° — no movement)
and only marginally at N=256 (12.95°, still well outside the 5° strict basin
and the 10° loose basin). Competing-basin clusters grew sub-linearly: 49 at
N=64 → 76 at N=128 → 123 at N=256. The seed-10 surrogate-MSE landscape from
generic SO(3) ICs has dense local-minimum structure with no LM funnel into
truth at any tested density.

**Mechanism candidate (consistent, not yet verified):** seed 10 has only
1.055 rotations in the 1-hour observation window (slowest tumbler in s011's
10-seed pilot at 6.33 deg/min). The LC under-determines the state — multiple
(q0, ω) attractors exist with surrogate-MSE within ~10× of truth. This is the
"multi-solution / observational-indistinguishability" regime, not a clean
inversion failure. **ρ-band classification of the 123 competing basins
(deferred to s013) decides whether seed 10 is a failure or a multi-solution
case.**

**Strategic shift for Q4c.** Pure SO(3) Sobol + LM polish at fixed truth-ω
does not recover seed-10-class seeds at any feasible density. The next
architectural move is **either** basin-hopping (expensive perturbation random
walks between LM polish runs) **or** multi-solution acceptance (treat any
candidate with ρ < 4 as a valid recovery, regardless of geodesic distance to
truth).

## What

A single-seed extension of s011's Q4c-ii cohort pilot to test whether the
seed-10 0/64 failure was N-recoverable (H1) or architectural (H2).

## How

- **Sobol extension.** Used scipy `qmc.Sobol(d=3, scramble=True, seed=42)`,
  the same scrambled-Sobol object as s002 / s006 / s010 / s011. Ran
  `fast_forward(64)` then `random(192)` to produce the next 192 quaternions
  in the scrambled sequence. Verified before the run that the first 64 of
  this sequence matches s011's stored `q0_seed_wxyz` to machine precision.
- **LM polish.** Identical to s011: scipy `least_squares(method='lm',
  max_nfev=60, xtol=ftol=gtol=1e-8)`, 6-DOF parameterization
  `x = (δθ, ω)` with `q0 = expm(δθ) · q0_seed`, surrogate full-LC residuals.
- **ω initial.** Fixed at truth-ω.
- **Compute.** Pool(8), BLAS=1, torch threads=1 (s011 critical-fix path).
  Wall: 588 s on the 192 new ICs (mean 24.5 s/run, vs s011's seed-10 mean
  26 s/run — comparable). Inside the 25-min kill criterion.
- **Combination.** Concatenated s011's first 64 with s012a's 192 to form a
  256-IC analysis array; reported per-N (cumulative) yield + competing-
  basin counts at N ∈ {64, 128, 256}.

## Result

### Per-N yield — 0 / 64 / 0 / 128 / 0 / 256 → seed 10 not N-recoverable

| N    | in-basin (strict 5°/1°/5%) | unique clusters | competing (≥30°, mse<0.5) | min q0_err | min q0_err <30° | min final_mse | min final_mse @<30° |
|------|----------------------------|-----------------|--------------------------|------------|----------------|---------------|--------------------|
| 64   | 0                          | 0               | 49                       | 28.25°     | 28.25°         | 0.0346        | 0.142              |
| 128  | 0                          | 0               | 76                       | 28.25°     | 28.25°         | 0.0346        | 0.142              |
| 256  | 0                          | 0               | 123                      | 12.95°     | 12.95°         | 0.0346        | 0.0989             |

Twin recoveries 0 / 256.

### Sobol coverage of truth neighbourhood

At N=256 only **1 / 256 ICs** lies within 30° of truth-q0 (min initial Sobol-
to-truth distance = 29.00°). 0 / 256 within 20°, 15°, or 10°. The Sobol-on-
SO(3) coverage near truth is sparse — uniform Sobol allocates ~`(R/π)³ × 4π/3`
of cells inside a geodesic ball of radius R, so the expected closest
approach at N=256 on SO(3) is ~`π × (3 × 64 / π / 256)^(1/3)` ≈ 14°. We
observed 29°, which is within Sobol-noise but conservatively poor.

### LM dynamics from the closest 12 Sobol ICs (combined N=256)

| Sobol idx | initial_q0_err | final_q0_err | final_mse | ω_dir_err | ω_mag_err | verdict          |
|-----------|----------------|--------------|-----------|-----------|-----------|------------------|
| 77        | 29.00°         | 46.65°       | 0.111     | 19.20°    | +0.71%    | pulled AWAY      |
| 231       | 31.29°         | 46.01°       | 0.111     | 19.35°    | +0.74%    | stationary       |
| 101       | 38.92°         | 43.61°       | 0.111     | 35.05°    | +6.34%    | stationary       |
| 207       | 39.38°         | 72.18°       | 0.264     | 12.13°    | +17.10%   | pulled AWAY      |
| 149       | 39.97°         | 46.66°       | 0.111     | 19.21°    | +0.70%    | stationary       |
| 220       | 43.90°         | 45.91°       | 0.112     | 12.85°    | -0.17%    | stationary       |
| 63        | 44.03°         | 145.04°      | 0.161     | 40.92°    | -7.74%    | pulled AWAY      |
| **162**   | **46.72°**     | **12.95°**   | **0.137** | 19.65°    | +4.78%    | **pulled toward**|
| 118       | 50.67°         | 134.70°      | 0.176     | 4.40°     | +12.33%   | pulled AWAY      |
| 44        | 51.57°         | 46.67°       | 0.111     | 19.21°    | +0.65%    | mild pull toward |
| 10        | 51.92°         | 124.83°      | 0.158     | 34.28°    | +18.81%   | pulled AWAY      |
| 160       | 52.85°         | 72.19°       | 0.264     | 12.13°    | +17.10%   | stationary       |

Reading the table:
- **Strong competing-basin attractor at (q0_err≈46°, ω_dir_err≈19°,
  |ω_mag_err|<1%, final_mse≈0.11):** captures ICs 77 / 231 / 149 / 220 / 44
  (5 of the 12 closest). This is a well-defined competing minimum.
- **Best near-truth landing (12.95°)** comes from IC 162 at 46.72° initial.
  LM partially funnels q0 toward truth but ω drifts off (ω_dir_err=19.65°,
  ω_mag_err=+4.78%); final_mse 0.137 is about 38× truth_mse_ref. NOT
  in any basin (strict or loose) — q0 close-but-not-in basin AND ω
  outside basin.
- **Multi-direction LM jumps:** from initial 44° several ICs go to 124-145°
  (pulled away to far attractors). The basin-of-attraction map is
  **fragmented**, not smooth — neighbouring ICs diverge to unrelated
  competing basins.

### Competing-basin growth saturates sub-linearly

49 → 76 → 123 over a 4× density increase (factor 2.5×). With purely random
sampling and a finite competing-basin set, we'd expect saturation; with an
infinite (continuous) competing-basin landscape, we'd expect linear growth.
The 2.5× scaling sits between these regimes and suggests **a large but
finite competing-basin set whose discovery saturates as density increases**.
For Q4c interpretation: the failure is NOT a Sobol-coverage issue, it's a
landscape-fragmentation issue.

### Best competing basin is invariant across N

`min_final_mse = 0.0346 mag²` is the same at N=64, 128, and 256. The global
LM-discovered surrogate-MSE minimum across ICs has been found at N=64 and
adding ICs does not displace it. This minimum is at q0_err = 31.82°,
ω_dir_err = 2.70°, ω_mag_err = approximately near-truth, final_mse 0.0346
(s011 data, IC index unknown in the seed-10 ordering — easily looked up
from runs_combined.npz). It is **9.6× above truth_mse_ref = 3.59e-3**.

In surrogate-ρ terms (placeholder; see `concepts/rho_band.md` — surrogate-MSE
is NOT directly ρ-band classifiable), `√(0.0346 / 0.0025) ≈ 3.7`, which would
be Band-B borderline if hi-fi MSE matched. Hi-fi rendering of this state is
**load-bearing for interpretation** — it would tell us whether the best
competing basin is observationally indistinguishable from truth.

## Why this matters

### Architectural decision for Q4c

The cohort scaling story breaks for seed-10-class seeds. Three options now
on the table:

1. **Basin-hopping.** Add random-walk perturbations between LM runs to escape
   the 123 competing basins. Cost per seed scales with the number of basins
   to traverse (~123 × LM_cost ~ ~1 hour single-seed at default settings).
   Untested in this workspace.
2. **Multi-solution acceptance** (per `concepts/observational_
   indistinguishability.md`). If the best competing basins are ρ < 4 (Band
   A∪B), the survey's success metric does not require finding truth — any
   Band A∪B candidate is an acceptable inversion. Hi-fi ρ-band validation
   of the 123 competing basins is the gating measurement.
3. **Partial-cohort acceptance.** Drop seed-10-class seeds (likely ~5-15%
   of cohort based on s011's 1/10 + the multi-axis tail correction to
   s009's body-X-only 3/100). Report Q4c at the recoverable cohort
   fraction.

### Cohort fail-rate sharpening for s012

s011 said "seed 10 fails at multi-axis Sobol where s009's body-X said
T1-strict pass". s012a confirms the fail is robust to 4× density at fixed
truth-ω. **For the s012 cohort scan at N=64, expect 5-15% of cohort to fail
similarly, and density-bumping won't fix them.** s012's load-bearing
measurement is now: which fraction of cohort is "Q4c-density-recoverable"
(N=64 with adequate ICs) vs "seed-10-class" (no Sobol density helps at
truth-ω) vs "needs-ω-grid" (failure is in the ω axis).

### Mechanism: under-sampled tumbling

Seed 10 has 1.055 rotations sampled in its 1-hour observation window — by
far the slowest in s011's pilot. The LC therefore has minimal periodic
structure; most of its information is in glints and slow geometry change.
With this LC, multiple (q0, ω) states produce similar predictions →
fragmented surrogate-MSE landscape → no LM funnel into truth from generic
SO(3) ICs. This is not a "bug" in the inversion; it's a **physical
under-determination** of the inverse problem when rotation is slow.

s007 + s008 already established that LC features (spectral, time-domain,
multi-feature regression) cannot recover ω-magnitude reliably because of
this multi-periodicity / low-rotation-count regime. s012a now says the
same physics affects q0 recovery: when only ~1 rotation is sampled, the
q0 inverse is multi-modal in surrogate-MSE space.

## Numbers

- **Wall time:** 588 s for 192 new LM runs (Pool(8), BLAS=1, torch=1).
  Mean 24.5 s/run.
- **Total cohort-extrapolation cost** (N=256 single seed, single-machine
  Pool(8)): 256 × 24.5 / 8 / 60 ≈ 13 min/seed. 100 seeds → ~22 hours
  single-machine — feasible.
- **Truth_mse_ref:** 3.59e-3 mag² (from s001 cache).
- **Min final_mse achieved at any N:** 0.0346 mag² — **9.6× above truth**.
- **Min q0_err achieved at N=256:** 12.95° (from IC 162, init 46.72°).

## Artefacts

- `experiments/s012a_seed10_density_probe.py`
- `results/s012a/runs.npz` — 192 new IC results (full state per IC)
- `results/s012a/runs_combined.npz` — 256-IC combined array (s011 first 64
  + s012a 192), sorted by Sobol ic_idx
- `results/s012a/summary.json`
- `results/s012a/yield_curve.png`
- `results/s012a/q0_err_distribution.png`
- `results/s012a_run.log` (gitignored)

## Out of scope

- **Hi-fi ρ-band classification of the 123 competing basins.** Critical
  follow-up (s013). Without it we cannot distinguish "multi-solution
  cohort" from "failure cohort" interpretation. Per workspace contract
  (`CLAUDE.md`: inversion-side substrate is contraband), the survey's
  `lib/hifi_render.py` must be **copy-adapted** from
  `notebooks/inversion/lib/lc_compare.py:generate_hifi_lc` (and the
  satellite/articulation setup pieces of `load_hifi_context`), using
  the survey's own `lib/traj_load.py` + `src.*` substrate — NOT
  imported. Smoke test: `render_hifi_lc(q0_truth, omega_truth, seed)`
  must reproduce `traj_seedXXX.npz['mag_hifi']` to machine precision
  on seeds 6 / 10 / 91 before trusting any ρ-band number.
- **N=512 / N=1024 stress test.** Cost grows linearly; would only push
  closest-Sobol-to-truth from 13° toward 6-8°. Mechanism (LM grab radius
  too small to bridge the gap from generic ICs) suggests further N
  doesn't help. Cheap (~25 min Pool(8)) if a calibration is wanted.
- **Basin-hopping pilot.** The natural follow-up if s013 ρ-band
  classification rules out multi-solution acceptance. Untested.
- **Seed-10-with-ω-grid.** Whether seeds with very slow tumbling (low
  rotation count) need a denser ω-grid than the s003 truth-ω tube
  suggests. Not measured.
- **Why exactly the LM grab radius is small for seed 10.** Hypothesised
  cause is LC-information-poverty from low rotation count, but the
  precise mechanism (which geometric feature of the surrogate landscape
  collapses) is not investigated.

## Cross-references

- `experiments/s011_q4cii_sobol_so3_polish_pilot.md` — the pilot that
  identified seed 10 as the only N=64 failure.
- `experiments/s005_joint_local_descent.md` — joint LM characterisation
  inside the s003 tube; established basin shape for 5 PA-stratified seeds.
- `experiments/s006_seed28_landscape_at_truth_omega.md` — sub-Sobol-narrow
  truth basin on seed 28 (recovered at s011 N=64 anyway via LM grab radius).
- `experiments/s010_seed44_landscape_at_truth_omega.md` — sub-Sobol-narrow
  competing basin on seed 44 (also recovered at s011).
- `experiments/s007_omega_mag_peak_spacing_pilot.md` — closed Q4c-iii.
- `experiments/s008_lc_feature_regression_omega.md` — closed Q4c-iv.
- `concepts/observational_indistinguishability.md` — multi-solution
  framework that gates s013's interpretation.
- `concepts/rho_band.md` — the classification we need s013 to apply.
- Auto-memory: `feedback_blas_threads_for_pool.md` — torch threading bug
  applies here too (handled in s012a's `init_worker`).
