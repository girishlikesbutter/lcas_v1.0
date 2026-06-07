---
title: "s013 — hi-fi ρ-band validation of s011 / s012a candidates"
type: experiment
sources:
  - results/s011/runs.npz (cohort A: 36 in-basin landings)
  - results/s012a/runs_combined.npz (cohort B: 256 seed-10 final states)
  - data/trajectories/traj_seed{XXX}.npz (per-seed truth + cached SPICE state)
  - lib/hifi_render.py (smoke-tested round-trip on seeds 6/10/91)
related:
  - experiments/s011_q4cii_sobol_so3_polish_pilot.md
  - experiments/s012a_seed10_density_probe.md
  - experiments/s005_joint_local_descent.md
  - concepts/observational_indistinguishability.md
  - concepts/rho_band.md
  - concepts/known_pathologies_to_revalidate.md
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

# s013 — hi-fi ρ-band validation

## TL;DR

Hi-fi ρ-band classification of LM-converged candidates from
`s011` (multi-axis Sobol(q0)+LM polish at truth-ω, 10 PA-stratified
seeds × 64 ICs) and `s012a` (seed-10 density extension to N=256). Two
cohorts:

- **Cohort A:** the 36 LM-converged "in-basin" landings (s011's
  `truth_basin_strict=True`, distributed across 9 recoverable seeds).
  Sanity check that surrogate-MSE-converged candidates land in **Band A**
  (ρ < 2) under hi-fi rendering.
- **Cohort B:** all 256 seed-10 final states from s012a (which includes
  s011's first 64 to bit-identical precision after sorting by IC). Decides
  whether seed-10's competing basins are observationally indistinguishable
  from truth (any ρ < 4 → **MULTI-SOLUTION**) or visibly worse (all ρ ≥ 4 →
  **GENUINE FAILURE**).

**Headline result.**
- Cohort A: **A=36, B=0, C=0, D=0** (n=36). ρ ∈ [0.040, 0.230]; median 0.078.
  All 36 in-basin LM-converged states render solidly Band A (ρ ≪ 1)
  in hi-fi terms.
- Cohort B (seed 10): **A=0, B=2, C=132, D=122** (n=256). ρ ∈ [3.90, 33.05];
  median 7.82. Best-ρ landing IC#6 with ρ=3.90 (Band B), hi-fi
  MSE = 0.0381 mag², surrogate MSE = 0.0346 mag², q0_err = 31.82°,
  ω_dir_err = 2.70°, |ω_mag_err| = +2.24%. **Surrogate-best == hi-fi-best
  (same IC#6).** Spearman rank correlation surrogate ↔ hi-fi ρ across all
  256 landings: **0.999**. **DECISION: MULTI_SOLUTION** (2/256 = 0.78%
  candidates with ρ < 4).
- The two Band B candidates (IC#6 and IC#86) cluster into a **single
  basin** (geodesic distance 0.013° between them). Greedy 5° clustering
  on the 256 final states yields **1 Band B basin + 75 Band C basins +
  82 Band D basins = 158 distinct (q0, ω) attractors total**.

## What

Per-candidate hi-fi forward render and ρ-band tag. For each LM-converged
state `(q0_final, ω_final)` from `s011/runs.npz` (cohort A) and
`s012a/runs_combined.npz` (cohort B):

1. Build per-seed `ctx` from cached `traj_seedXXX.npz` (sun_pos, obs_pos,
   sat_pos, obs_dist, observation_times) plus globally-cached satellite +
   BRDF + inertia from `intelsat_901_config.yaml` (matches m048 generator
   exactly).
2. Propagate `(q0_final, ω_final)` via post-fix
   `src.dynamics.attitude_propagator.propagate_attitude` in `tumbling`
   mode.
3. Project to body frame, compute shadows, run BRDF lightcurve generator —
   yields `pred_hifi[500]`.
4. Compute `hifi_MSE = mean((pred - mag_hifi_truth)²)` and `ρ =
   √(hifi_MSE) / 0.05`. Tag Band A/B/C/D per `concepts/rho_band.md`.

`render_hifi` was smoke-tested before this run: on seeds 6/10/91 the
round-trip `(q0_truth, ω0_truth)` → `mag_hifi` reproduces the cached
NPZ to bit-identical (max|Δ| = 0.0).

## How

- **Loader.** Uses `lib.hifi_render.build_context(seed)` (per-seed,
  caches the satellite + inertia at module level) and `render_hifi(q0, ω,
  ctx)` (the full forward chain).
- **Pool.** `multiprocessing.Pool(N=8)` with `init_worker` setting
  `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` and
  `torch.set_num_threads(1)` / `set_num_interop_threads(1)` per the s011
  fix (`memory/feedback_blas_threads_for_pool.md`).
- **Per-worker ctx cache.** First render per seed in each worker pays one
  STL+BRDF load (~1 s); subsequent renders for the same seed reuse it. With
  N_renders=292 and Pool(8), wall ≈ 12 s/render × 292 / 8 ≈ 7 min.
- **Cohort A** (36 items): rows of `s011/runs.npz` where
  `truth_basin_strict=True`. Per-seed counts: 6=8, 21=3, 28=7, 41=2, 44=5,
  48=1, 60=4, 84=3, 91=3 (seed 10 = 0; seed 10 is cohort B exclusively).
- **Cohort B** (256 items): every row of `s012a/runs_combined.npz`. All
  on seed 10. None in basin (`truth_basin_strict=False` for all 256).

## Result

### Cohort A — in-basin sanity check

All 36 in-basin LM-converged landings render Band A (ρ ≪ 1) in hi-fi
terms — exactly the expected sanity-check outcome. The strict
(`q0_err<5°, ω_dir_err<1°, |ω_mag_err|<5%`) basin definition from s005 /
s009 / s011 agrees with hi-fi ρ < 1 cleanly.

| seed | n | ρ-min | ρ-median | ρ-max | A | B | C | D |
|------|---|-------|----------|-------|---|---|---|---|
| 6    | 8 | 0.230 | 0.230    | 0.230 | 8 | 0 | 0 | 0 |
| 21   | 3 | 0.048 | 0.048    | 0.048 | 3 | 0 | 0 | 0 |
| 28   | 7 | 0.053 | 0.053    | 0.053 | 7 | 0 | 0 | 0 |
| 41   | 2 | 0.093 | 0.093    | 0.093 | 2 | 0 | 0 | 0 |
| 44   | 5 | 0.078 | 0.078    | 0.078 | 5 | 0 | 0 | 0 |
| 48   | 1 | 0.123 | 0.123    | 0.123 | 1 | 0 | 0 | 0 |
| 60   | 4 | 0.087 | 0.089    | 0.092 | 4 | 0 | 0 | 0 |
| 84   | 3 | 0.077 | 0.077    | 0.077 | 3 | 0 | 0 | 0 |
| 91   | 3 | 0.040 | 0.040    | 0.040 | 3 | 0 | 0 | 0 |
| **all** | **36** | **0.040** | **0.078** | **0.230** | **36** | **0** | **0** | **0** |

The per-seed ρ-spread is essentially zero — all in-basin landings on
each seed converge to within numerical-precision of the same near-truth
state, which renders the same hi-fi LC. ρ varies seed-to-seed (0.04
to 0.23) because the LM-converged state itself is slightly offset
from truth (q0_err typically 0.03°-1°, contributing a small but
consistent LC residual). Even seed 6's 0.23 (the highest) is well
inside Band A.

### Cohort B — seed-10 N=256

| n | ρ-min | ρ-median | ρ-max | A | B | C | D | n_band_A∪B | distinct basins (5° cluster) |
|---|-------|----------|-------|---|---|---|---|------------|------------------------------|
| 256 | 3.902 | 7.821 | 33.052 | 0 | **2** | 132 | 122 | **2** | 1 (B) + 75 (C) + 82 (D) = 158 |

**Best-ρ landing (and surrogate-best landing — same IC #6):**
- ρ = **3.902** (Band B, just inside the multi-solution acceptance bar)
- Hi-fi MSE = 0.0381 mag², surrogate MSE = 0.0346 mag² (ratio = 1.10)
- q0_err = **31.82°** (geodesic to truth-q0)
- ω_dir_err = **2.70°**, |ω_mag_err| = **+2.24%**
- IC #86 lands in the same basin (geodesic 0.013° to IC#6) at
  ρ=3.902, q0_err=31.83°.

**Surrogate ↔ hi-fi rank consistency.** Across all 256 landings:
- Spearman ρ-correlation(surrogate_MSE, hi-fi MSE) = **0.999**
- hi-fi MSE / surrogate MSE: median **1.003**, p10=0.987, p90=1.041
The surrogate is essentially bias-free against hi-fi MSE on seed 10;
its global argmin coincides with the hi-fi global argmin to numerical
precision.

**Top-12 lowest-ρ landings (covers all candidates ρ ≤ 4.62):**

| rank | IC# | ρ | hi-fi MSE | surr MSE | q0_err | ω_dir_err | ω_mag_err | band |
|------|-----|---|-----------|----------|--------|-----------|-----------|------|
|   1  |   6 | 3.90 | 0.0381 | 0.0346 |  31.82° |  2.70° |  +2.24% | B |
|   2  |  86 | 3.90 | 0.0381 | 0.0346 |  31.83° |  2.70° |  +2.23% | B |
|   3  | 195 | 4.16 | 0.0433 | 0.0412 | 178.92° |  2.17° |  −2.47% | C |
|   4  | 218 | 4.28 | 0.0458 | 0.0437 | 172.21° |  1.02° |  −2.60% | C |
|   5  |  53 | 4.33 | 0.0468 | 0.0437 | 172.79° |  0.98° |  −2.71% | C |
|   6  | 255 | 4.35 | 0.0474 | 0.0440 | 171.61° |  1.19° |  −2.57% | C |
|   7  |  74 | 4.36 | 0.0476 | 0.0440 | 172.60° |  1.02° |  −2.96% | C |
|   8  | 141 | 4.38 | 0.0479 | 0.0486 | 170.74° | 13.05° |  +2.30% | C |
|   9  |  19 | 4.38 | 0.0479 | 0.0486 | 170.74° | 13.05° |  +2.30% | C |
|  10  |  31 | 4.38 | 0.0479 | 0.0486 | 170.71° | 13.05° |  +2.29% | C |
|  11  |   0 | 4.38 | 0.0479 | 0.0486 | 170.70° | 13.06° |  +2.29% | C |
|  12  | 193 | 4.62 | 0.0533 | 0.0540 | 110.72° | 34.88° | −22.51% | C |

**Cluster structure (greedy 5° geodesic on q0_final, antipode-aware,
included where ρ ≤ 8 = Band B+C):**
- 1 Band B basin: cluster 4 (q0=31.83°, ω_dir=2.70°), 2 ICs (#6, #86).
  This is the single multi-solution candidate.
- 75 Band C basins; the most populated is cluster 7 (q0=165.6°,
  ω_dir=14.96°, ρ-min=5.57, n_ICs=11) — large near-rotation-pole
  attractor.
- 82 Band D basins (after 5° clustering on the 122 Band-D landings).
- Several near-180° clusters (clusters 25, 27, 53, 65, 66 — q0 ∈
  [156°, 179°], ω_dir < 11°). These have ω matching truth tightly but
  q0 close to a 180° flip — likely body-Y or body-Z 180° degenerates,
  not the body-X twin (which s011/s012a both confirmed at 0/256
  twin_basin_strict).

### Surrogate-best vs hi-fi-best ranking

The Spearman correlation 0.999 + hi-fi/surrogate ratio 1.003 (median)
together imply: on seed 10, the surrogate's global ranking IS the
hi-fi ranking. **A "find lowest surrogate-MSE candidate per seed"
architecture would have produced the IC #6 = Band B candidate as the
solver's output.** The surrogate's local-minimum structure is
faithful, not deceptive — the seed-10 failure is not a surrogate
artefact, it is an LC information-content limitation.

## Why this matters

For **Cohort A:** sanity check. If ANY in-basin landings render Band B+
under hi-fi, that means the s005-validated "basin-strict =
q0_err<5°/ω_dir<1°/|ω_mag|<5%" is too generous — the hi-fi LC at the
basin edge does not agree with truth at noise level. This would tighten
the "basin radius" definition future experiments rely on.

For **Cohort B (decisive):** seed 10 is the only seed in the s011 pilot
that fails Q4c-ii at any feasible Sobol density (s012a closed N=64/128/256
all 0 in-basin). Two interpretations:

1. **MULTI-SOLUTION.** If at least one of seed-10's 256 final states has
   ρ < 4 (Band A∪B), the LC physically does not pin down `(q0, ω)`
   uniquely on this seed. Per
   `concepts/observational_indistinguishability.md`, any Band A∪B
   candidate is a valid recovery — the survey's success criterion is met
   even though geometric truth is not recovered. Seed 10's "failure" is
   actually a data limitation, not a solver limitation.
2. **GENUINE FAILURE.** If all 256 final states have ρ ≥ 4, the surrogate
   has discovered local minima in (q0, ω)-space that produce visibly
   worse hi-fi LCs than truth. The surrogate's local-minimum structure is
   misleading; basin-hopping or a stronger global search is needed to
   escape.

The decision affects **what survey-final architecture looks like for the
seed-10-class subset** (~5-15% of cohort by s012a's projection):
- If MULTI-SOLUTION → cohort architecture: "find any Band A∪B candidate per
  seed." Seed 10 is solved at ρ ~3.7 from the surrogate-best landing
  (predicted, given surr_mse = 0.0346 ≈ 9.6× truth and surrogate-vs-hifi
  MSE coupling).
- If GENUINE FAILURE → cohort architecture: "find truth basin or accept
  partial-cohort failure." Seed-10-class needs basin-hopping or a method
  beyond LM polish.

**Actual outcome: MULTI_SOLUTION at the boundary.** ρ-min on seed 10 is
**3.90**, just inside the survey's ρ < 4 acceptance bar. The seed-10
multi-solution interpretation is real but narrow — only 0.78% (2/256)
of LM landings achieve Band B; the remaining 99.2% are Band C∪D. If the
survey's acceptance bar were tightened from ρ < 4 to ρ < 3.5, seed 10
would flip to GENUINE_FAILURE. This boundary fragility is itself a
finding: **seed-10-class is at the LC information-content limit**, not
comfortably inside a multi-solution regime. The mechanism (consistent
with s007/s008/s012a): seed 10 has only 1.06 rotations sampled in its
1-hour window; multi-periodic LCs from low-rotation tumblers
under-determine (q0, ω). The Band B candidate's ω is close to truth
(ω_dir 2.7°, |ω_mag| 2.2%) but its q0 is 31.8° off — so the surrogate
has found a different attitude with similar dynamics that produces an
LC matching the truth LC at the boundary of observational
indistinguishability.

**Strong corollary: surrogate ≈ hi-fi rank.** Spearman 0.999 + median
ratio 1.003 means a "lowest surrogate-MSE per seed" cohort architecture
would have output the Band B candidate on seed 10. No hi-fi
re-ranking step is needed *for this seed*. Whether this property holds
across the cohort is a follow-up question (cheap: extend s013 to all
640 s011 candidates ≈ 6 min Pool(8)).

## Numbers

- 292 hi-fi renders × ~9.2 s/render amortised (Pool=8) = **wall 2680 s
  (44.7 min)**. Per-worker render wall ~73 s; the per-worker slowdown
  (vs lc_compare's reported ~12 s/render single-process) is consistent
  with shadow-engine and trimesh's internal threading not being capped
  by `OMP_NUM_THREADS` — workers contend for shared C-extension threads.
  Note for future Pool-based hi-fi runs: investigate whether trimesh /
  shadow_engine have a thread-cap setting analogous to torch's.
- 36 in-basin landings, all Band A (ρ-max 0.230 on seed 6).
- 256 seed-10 landings, ρ-min = 3.90, ρ-median = 7.82, ρ-max = 33.05;
  Band counts A/B/C/D = 0/2/132/122.
- Distinct seed-10 basins at 5° geodesic clustering: **158 total**
  (1 B + 75 C + 82 D). The single Band B basin captures 2/256 ICs
  (#6 + #86, geodesic 0.013° between them — same physical basin).
- Spearman correlation surrogate ↔ hi-fi MSE on seed 10 = **0.999**.

(full per-candidate tables in `rho_in_basin.npz` and `rho_seed10.npz`;
full cluster decomposition in `clusters.json`)

## Artefacts

- `experiments/s013_rho_band_validation.py`
- `experiments/s013_rho_band_validation.md` (this file)
- `lib/hifi_render.py` (built and smoke-tested as part of this round —
  round-trip on seeds 6/10/91 reproduces cached `mag_hifi` to bit-
  identical precision)
- `results/s013/rho_in_basin.npz` (cohort A: 36 rows)
- `results/s013/rho_seed10.npz` (cohort B: 256 rows)
- `results/s013/summary.json`
- `results/s013/clusters.json` (cohort B greedy-5° decomposition)
- `results/s013/rho_distribution.png` (log-x histograms, both cohorts,
  with A/B/C/D threshold lines)
- `results/s013_run.log` (gitignored)

## Out of scope

- **Twin-attractor analysis on cohort B.** s011/s012a both confirm 0/256
  in twin basin on seed 10; this experiment doesn't re-classify.
- **Cohort-scale ρ-band classification of all 640 s011 runs.** Cohort A
  is in-basin only; cohort B is seed-10 only. Future extension:
  the 640 - 36 - 64 = 540 not-in-basin not-seed-10 runs are also
  ρ-classifiable from cached s011 states (a 540-render extension, ~6 min
  Pool(8)). Useful for characterising the cohort-wide LM-stall vs
  competing-basin ρ-band distribution. **Note:** s011's 64 seed-10 finals
  are bit-identical to s012a's first 64 after sorting by ic_idx (verified
  before s013 run); cohort B subsumes them.
- **N=512+ density on seed 10.** Outside the bounds of the existing
  cached state. Listed under PROGRESS.md as cheap insurance (~25 min
  Pool(8)) but not load-bearing — s012a's 49→76→123 sub-linear competing-
  basin growth makes recovery at higher N very unlikely.
- **Basin-hopping pilot.** Listed in PROGRESS.md as conditional on s013
  ruling out multi-solution. If s013 says MULTI-SOLUTION, basin-hopping
  on seed 10 is unnecessary.
- **Hi-fi ρ-band on s005 / s010 LM-landings.** Now infrastructure-cheap;
  could bundle as a follow-up. Not done in s013 to keep the cohorts
  decisive about the seed-10-class question.
- **Cohort-scale s012 (full 100 seeds × 64 IC).** s013 informs the
  interpretation of s012's eventual result; doesn't replace s012.

## Cross-references

- Methodology: `feedback_rho_band_convention.md` (auto-memory),
  `feedback_save_results.md`, `feedback_save_hifi_lcs.md`.
- Concepts: `rho_band.md`, `observational_indistinguishability.md`,
  `q_omega_coupling.md`. Auto-memory `feedback_multi_solution.md` covers
  the multi-solution philosophy (no survey-local concept page yet).
- Cohort context: `s005`, `s011`, `s012a` for the inputs; `s006`, `s010`
  for the s002 argmin-survives-on-seed-X precedent.
