---
title: "s014b — n_rotations / PA correlation with ρ-min and multi-solution clusters"
type: experiment
sources:
  - results/s011/runs.npz (s011 LM landings, 640 rows on 10 pilot seeds)
  - results/s013/rho_in_basin.npz (s013 in-basin landings, 36 rows)
  - results/s013/rho_seed10.npz (s013 seed-10 N=256 hi-fi rendered)
  - results/s014/rho_s011_nb.npz (s014 non-basin landings, 540 rows)
  - results/s014/rho_s005.npz (s014 cohort B, 50 rows)
  - results/s014/analysis_summary.json (s014 post-analysis multi-sol dump)
  - data/trajectories/traj_seed{XXX}.npz (cohort metadata: n_rotations, mean PA)
related:
  - experiments/s014_cohort_rho_band.md
  - experiments/s013_rho_band_validation.md
  - experiments/s012a_seed10_density_probe.md
  - experiments/s011_q4cii_sobol_so3_polish_pilot.md
  - experiments/s007_omega_mag_peak_spacing_pilot.md
  - experiments/s008_lc_feature_regression_omega.md
  - concepts/observational_indistinguishability.md
  - concepts/rho_band.md
created: 2026-05-01
updated: 2026-05-01
confidence: medium-high (10-seed pilot; load-bearing claim explicitly
limited to "consistent with low-rotation under-determination", not
"proves causality")
---

# s014b — n_rotations / PA correlation with ρ-min and multi-solution clusters

## TL;DR

Cheap cached-data analysis (~25 s wall, no new renders) closing two
follow-on questions PROGRESS flagged after s014:

1. **n_rotations vs ρ-min on the 10-seed pilot.** Spearman = **−0.648**
   (n=10), Pearson on log-log = **−0.826**. The trend is dominated by
   seed 10 (1.06 rot, ρ=3.90) but **persists when seed 10 is dropped**:
   Spearman = −0.517, Pearson(log,log) = −0.535 on the 9 recoverable
   seeds (all ρ < 0.25). Consistent with the s007/s008/s012a hypothesis
   that low rotational coverage under-determines (q0, ω) jointly. **The
   cohort tail predicted by this — n_rotations < 2 — has 10 seeds: 3,
   10, 13, 16, 31, 42, 43, 72, 78, 79.**

2. **Multi-solution attractor structural classification.** 19 ρ<4
   candidates outside in-basin (s014's 17 + s013 seed10's 2) → greedy
   5° geodesic clustering on q0_final yields **9 distinct attractors
   across 5 seeds**: 1 class_1 (seed 28, near-truth-q0 + ω just outside
   strict basin), **4 class_2 (q0 ∈ [30°, 150°] + near-truth ω: seeds
   10, 41, 48 plus seed 41 cross-cohort)**, and **5 class_3 (q0 ≥ 170°,
   near-180° flip, near-truth ω: seeds 48 ×2, 84 ×3)**. Class_3 is the
   surprising one: 5 distinct near-180° attractors are NOT pure
   body-twins (twin recoveries 0/640+50 + 0/256), so the m048 geometry
   carries a near-flip symmetry only resolvable under longer
   observation windows.

PA shows weaker but coherent secondary correlation
(Spearman(PA, ms_count) = −0.540): low-PA seeds (48 at 10°, 41 at 18°)
are multi-solution-rich, high-PA seeds (60 at 88°, 21 at 75°) are
clean.

## What

Two cached-data analyses on the 10-seed s011 pilot, no new compute on
the forward model:

**Question A — does n_rotations predict ρ-min?** s007/s008/s012a/s013
hypothesised that low rotational coverage under-determines (q0, ω).
With s013/s014 ρ-min data on the full 9-seed pilot + s013's N=256
seed-10 cohort, this is now directly testable.

**Question B — what is the structural composition of the 19 ρ<4
multi-solution candidates?** s014 enumerated 17 outside-basin
candidates across 4 seeds (28, 41, 48, 84) plus 1 cross-cohort
confirmation in s005. Including seed 10's 2 cohort-B candidates from
s013 gives 19 total. Greedy geodesic clustering on q0_final → distinct
attractor count per seed; classification into the three structural
classes flagged in s014's wind-down (near-truth-q0/off-truth-ω,
q0=30-150°/near-truth-ω, near-180°/near-truth-ω).

## How

- **Per-seed metadata for all 100 cohort seeds.** Load truth NPZ via
  `lib.traj_load.load_truth(seed)`; n_rotations = ω_mag_dps × duration_s
  / 360 (duration_s = 3600 across cohort); mean / min / max PA from
  cached `phase_angle_3d`.

- **Per-seed ρ-min on the 10 pilot seeds.** For 9 recoverable seeds
  (6/21/28/41/44/48/60/84/91): merge s013 in-basin (4-8 ICs each) with
  s014 non-basin (56-60 ICs each) → full-64 coverage; min ρ over finite
  values. For seed 10: from s013's N=256 cohort B (256 ρ values, all
  finite). 11 hi-fi-`inf` candidates on seeds 28/60 (extreme-attitude
  zero-flux LCs, properly Band D) excluded from finite-min.

- **Multi-solution candidate collection.** All ρ < 4 candidates
  outside strict basin (`~truth_basin_strict` where the flag is
  available) from cohort A non-basin + cohort B + s013 seed10. 19
  total. Each candidate carries `q0_final_wxyz`, allowing geodesic
  clustering.

- **Greedy 5° geodesic clustering.** Per seed, iterate candidates in
  order; assign to existing cluster if geodesic distance to any
  existing centroid < 5°; else open new cluster. Same idiom as s013's
  q0-final clustering on seed 10's 256 finals (158 distinct basins
  there, 9 here on the multi-solution subset).

- **Structural classification per cluster** (s014 convention,
  refined for strict-basin compatibility):
  - **class_1** — q0_err < 10° AND ω outside STRICT basin (|ω_dir|≥1°
    OR |ω_mag|≥5%). Reads as "near-truth-q0, ω just outside strict
    basin"; seed 28's q0=2.1°/ω_dir=2.0° is the canonical case.
  - **class_2** — q0 ∈ [30°, 150°] AND ω near truth (|ω_dir|<5°,
    |ω_mag|<5%). LC under-determination class. Seeds 10, 41, 48.
  - **class_3** — q0 ≥ 170° AND ω near truth (|ω_dir|<5°, |ω_mag|<5%).
    Near-180°-flip class. Seeds 48, 84.

- **Correlations.** Spearman + Pearson(log,log) for n_rotations vs
  ρ-min and ms_count (n=10), repeated on n=9 with seed 10 dropped.
  Also Spearman for PA vs both targets.

## Result

### Cohort metadata distribution (100 seeds)

n_rotations: median **7.82**, p10 **2.00**, p90 **13.82**, range
[1.06, 14.76]. **Cohort rotation bands: low (<2): 10 seeds, mid [2-5):
24, high (≥5): 66.**

Mean PA: median 38.5°, p10 13.8°, p90 79.4°.

**Predicted seed-10-class cohort (n_rotations < 2):**

| seed | n_rot | ω_mag (dps) | mean PA |
|------|-------|-------------|---------|
| 10   | 1.06  | 0.1055      | 40.2°   |
| 79   | 1.21  | 0.1207      | 14.1°   |
| 42   | 1.29  | 0.1288      | 85.9°   |
| 43   | 1.41  | 0.1406      | 62.3°   |
| 3    | 1.42  | 0.1418      | 34.8°   |
| 13   | 1.49  | 0.1491      | 19.8°   |
| 31   | 1.62  | 0.1615      | 12.0°   |
| 78   | 1.70  | 0.1704      | 37.8°   |
| 72   | 1.93  | 0.1934      | 39.1°   |
| 16   | 1.98  | 0.1985      | 83.0°   |

Only seed 10 has been hi-fi-rendered at depth (s012a/s013); the other
9 are predicted-failure candidates by the n_rotations hypothesis. s012
or s015 should explicitly include 2-3 of these seeds as calibration
targets.

### Pilot table (10 seeds with full hi-fi rendering)

| seed | n_rot | mean_PA | ρ-min  | ρ-p10  | ms_count | ms_classes        |
|------|-------|---------|--------|--------|----------|-------------------|
| 6    | 7.13  | 62.5°   | 0.230  | 0.230  | 0        | —                 |
| 10   | 1.06  | 40.2°   | **3.902** | 5.321 | 1     | class_2           |
| 21   | 7.94  | 75.3°   | 0.048  | 17.71  | 0        | —                 |
| 28   | 14.38 | 87.3°   | 0.053  | 0.053  | 1        | class_1           |
| 41   | 5.65  | 18.5°   | 0.093  | 18.35  | 1        | class_2           |
| 44   | 14.45 | 34.8°   | 0.078  | 4.866  | 0        | —                 |
| 48   | 2.57  | 10.2°   | 0.123  | 13.55  | **3**    | class_2 + 2× class_3 |
| 60   | 5.72  | 88.1°   | 0.087  | 7.728  | 0        | —                 |
| 84   | 5.07  | 38.6°   | 0.077  | 2.855  | **3**    | 3× class_3        |
| 91   | 14.26 | 51.7°   | 0.040  | 6.191  | 0        | —                 |

ρ-p10 includes the in-basin landings, so for seeds with ≥2 in-basin
landings the p10 sits inside Band A; seeds with only 1-3 in-basin
landings show p10 mid-cohort (the 64-IC bulk is non-basin).

### Correlations on the pilot

| correlation                              | n=10  | n=9 (drop seed 10) |
|------------------------------------------|-------|--------------------|
| Spearman(n_rot, ρ_min)                   | **−0.648** | **−0.517**     |
| Pearson(log n_rot, log ρ_min)            | **−0.826** | −0.535         |
| Spearman(n_rot, ms_count)                | −0.645 | −0.645             |
| Spearman(mean PA, ρ_min)                 | −0.333 | −0.350             |
| Spearman(mean PA, ms_count)              | −0.540 | −0.507             |
| Spearman(ω_mag_dps, ρ_min)               | −0.648 | —                  |

**The n_rot ↔ ρ_min correlation persists with seed 10 removed**; the
trend is not a single-outlier artefact. **The n_rot ↔ ms_count
correlation is identical n=10 vs n=9** — preserved by rank stability.

PA correlation is weaker (~−0.35 to −0.54) but coherently negative:
low-PA seeds tend to be multi-solution-rich. PA is a secondary
predictor; n_rotations dominates.

### Multi-solution attractor breakdown

19 ρ < 4 candidates outside in-basin → **9 distinct attractors across 5
seeds** after 5°-greedy clustering on q0_final:

| seed | n_rot | mean_PA | n_clusters | classes (per-cluster ρ_min) |
|------|-------|---------|------------|------------------------------|
| 10   | 1.06  | 40.2°   | 1          | class_2 (3.90)               |
| 28   | 14.38 | 87.3°   | 1          | class_1 (1.58)               |
| 41   | 5.65  | 18.5°   | 1          | class_2 (1.94)               |
| 48   | 2.57  | 10.2°   | 3          | class_2 (1.06), class_3 (1.28), class_3 (1.31) |
| 84   | 5.07  | 38.6°   | 3          | class_3 (2.49), class_3 (2.86), class_3 (3.88) |

Class totals across the cohort: **class_1: 1, class_2: 4, class_3: 5**.

**Class_2 (q0 ∈ [30°, 150°] + near-truth ω) seeds — 10, 41, 48 — span
n_rot {1.06, 5.65, 2.57}.** Median 2.57, all ≤ 5.65. Consistent with
LC under-determination: the LM polish lands at a different attitude
whose multi-periodic LC matches truth's at the multi-solution
boundary.

**Class_3 (near-180° flip + near-truth ω) seeds — 48, 84.** 5 distinct
attractors at q0_err ≈ 171-179.6° (NOT exactly 180°, and NOT pure
body-twins — twin recoveries 0/640 in s011, 0/50 in s005, 0/256 in
s013, all at strict body-X 180° pole). q0_final values within these
clusters span 2-7° from each other (resolved by 5° clustering); they
correspond to different near-flip attitudes around different axes.
**The m048 geometry has approximate near-180°-flip symmetry that
becomes detectable when LC information content is moderate.** Seeds 48
(low-rot) and 84 (mid-rot) hit it; high-rot seeds (28/91/44/21) do not.

**Class_1 (seed 28, q0=2.1°/ω_dir=2.0°/|ω_mag|=0.0%) is a basin-
definition artefact, not a structural multi-solution.** ρ=1.58 (Band
A), and the geometric position is essentially at truth-q0 with ω just
outside strict basin (ω_dir 2° vs strict 1°). Future work that adopts
loose basin (ω_dir<2°) or hi-fi-ρ<2 directly will reclassify this
landing as in-basin.

## Why this matters

**Three implications for the s015 design and beyond:**

1. **n_rotations is a useful pre-filter for cohort-tail prediction.**
   The 10 cohort seeds with n_rot < 2 (3, 10, 13, 16, 31, 42, 43, 72,
   78, 79) are the predicted seed-10-class. **s015 / s012 should
   include 2-3 of these (e.g., seeds 13 + 42 + 79) as calibration
   targets** to test whether the joint q0×ω architecture recovers them
   into Band B (multi-solution acceptance) or fails like seed 10 did
   at fixed truth-ω. If the trend holds, the cohort's expected
   long-tail failure rate at fixed truth-ω is ~10/100 ≈ 10% — close to
   the s011 pilot's 1/10 = 10% — independently corroborating the
   pilot's representativeness.

2. **Multi-solution is structural, not a few-seed quirk.** The 5
   class_3 near-flip attractors on 2 seeds (48, 84) point to a
   geometric symmetry of the m048 satellite: under low-information LCs
   (low n_rot or low PA), a near-180° flip around some axis produces a
   nearly identical LC. **This means the s015 joint pilot must
   accept that even with truth-ω given, multi-solution candidates will
   appear as second-best; the cohort architecture's surrogate-argmin
   selector continues to return truth-basin (s014 confirmed) but
   reporting Band A∪B yield-per-seed will count these as additional
   valid landings.** Survey acceptance bar (ρ<4) admits them; tighter
   bar (ρ<3.5) flips seed 10 from MULTI_SOLUTION to GENUINE_FAILURE
   and may flip seed 84's q0=171.5° cluster (ρ=2.86, would survive)
   and q0=179.6° cluster (ρ=3.88, would FLIP to FAILURE). Decision
   must precede s015 if the bar is being tightened.

3. **PA is a secondary predictor — does NOT replace n_rotations.**
   Spearman(PA, ρ_min) = −0.333 is much weaker than Spearman(n_rot,
   ρ_min) = −0.648. PA captures geometry-dependent BRDF specularity
   sensitivity (low PA = strong glints dominate, less off-axis info)
   but doesn't determine the LC's information capacity directly. Both
   matter; both are cohort-distributed; n_rot is the dominant signal.
   **For s015 IC stratification, prioritise PA-stratified seed picks
   only AFTER n_rot stratification — pick from the low/mid/high n_rot
   bands, then ensure PA spread within each.**

The mechanism story across s007/s008/s012a/s014b is now coherent:

| experiment | finding                                                  |
|------------|----------------------------------------------------------|
| s007       | Peak-spacing ω-mag oracle 7.4% — multi-periodic LCs scatter |
| s008       | 28-feature LOO ω-mag MAPE 16.4% — structural ceiling      |
| s012a      | Seed 10 (1.06 rot) 0/256 in-basin — q0 also under-determined |
| s013       | Seed 10 multi-solution at boundary (ρ=3.90)              |
| **s014b**  | **n_rotations correlates with ρ_min on cohort pilot**    |

Mechanism not directly verified (would require constructing two
distinct (q0, ω) tuples with identical multi-periodic LC at low
rotation count) but consistent across 5 independent measurements.

## Numbers

- Wall: ~25 s (data load + clustering + plotting). 100% cached data.
- Cohort n_rotations: median **7.82**, p10 **2.00**, range [1.06,
  14.76]. **10 seeds < 2 rot, 24 in [2, 5), 66 ≥ 5.**
- Pilot Spearman(n_rot, ρ_min) = **−0.648 (n=10)**, **−0.517 (n=9
  drop seed 10)**.
- Pilot Pearson(log n_rot, log ρ_min) = **−0.826 (n=10)**, **−0.535
  (n=9)**.
- Pilot Spearman(n_rot, ms_count) = **−0.645 (rank-stable n=10/n=9)**.
- Multi-solution attractors after 5° geodesic clustering: **9 unique**
  (class_1: 1, class_2: 4, class_3: 5) across 5 seeds (10, 28, 41,
  48, 84).
- Predicted seed-10-class candidates (n_rot < 2): seeds **3, 10, 13,
  16, 31, 42, 43, 72, 78, 79** (n=10 in cohort).

## Artefacts

- `experiments/s014b_n_rotations_analysis.py`
- `experiments/s014b_n_rotations_analysis.md` (this file)
- `results/s014b/seed_metadata.npz` (n_rot, PA, ω_mag for all 100
  seeds + pilot ρ_min + ms_count)
- `results/s014b/clusters.json` (per-seed multi-solution clusters +
  structural classes + members)
- `results/s014b/summary.json` (cohort stats, pilot table,
  correlations, ms breakdown)
- `results/s014b/n_rotations_vs_rho.png`
- `results/s014b/pa_vs_rho.png`
- `results/s014b/multi_solution_breakdown.png`
- `results/s014b_run.log` (gitignored)

## Out of scope

- **Mechanism verification — direct LC degeneracy.** s014b shows
  n_rotations ↔ ρ-min correlation; the underlying claim "low-rotation
  multi-periodic LCs admit distinct (q0, ω) producing identical LC"
  is not directly tested. Would require constructing two specific
  (q0, ω) tuples with matching LC at low rotation and verifying their
  attitude divergence — a targeted forward-model probe, not a survey
  measurement.
- **Cohort-scale extension.** Pilot is 10 seeds; the n_rot vs ρ_min
  correlation is measured on this subset only. Full cohort needs
  s012's 6400 LM runs + ρ-band classification (~7 h Pool(8)). The
  predicted-failure 10 seeds (n_rot < 2) are the natural calibration
  targets.
- **PA-only structural classification.** Did not partition multi-
  solution candidates by PA band (only by n_rot band) because PA
  correlation is weaker. A full 2D (n_rot × PA) stratification needs
  more samples than 9 multi-solution attractors.
- **Class_3 mechanism.** The 5 near-180°-flip attractors are a
  surprising structural finding. Mechanism could be (a) a
  near-symmetry of the m048 STL geometry resolvable as a
  surrogate-MSE local minimum, (b) a numerical artefact of LM polish
  at near-flip ICs, or (c) an LC-similarity coincidence at near-truth
  ω. Distinguishing (a) vs (b) vs (c) needs a targeted experiment —
  e.g., render hi-fi LCs from each class_3 attractor + compare
  pointwise to truth LC + visualise which facets reflect at each
  attitude.

## Cross-references

- s014 (closest precedent — enumerated the 17 multi-solution candidates
  and the 3 structural classes)
- s013 (provided seed-10 cohort B data; first to flag n_rotations
  hypothesis on a single seed)
- s012a (single-seed N=256 stress test; failed at any density)
- s011 (cohort source; pilot 10 seeds and 64 ICs each)
- s007 (closed peak-spacing ω-mag prior)
- s008 (closed LC-feature ω-mag regression — first cohort-scale
  evidence of LC under-determination at scale)
- `concepts/observational_indistinguishability.md`
- `concepts/rho_band.md`
- `concepts/q_omega_coupling.md`
- Methodology: `feedback_rho_band_yield_metric.md`,
  `feedback_report_all_three_errors.md`,
  `feedback_test_full_population.md`
