---
title: s059i_density_scan — closest-to-truth in survival cloud at T_A vs SO(3) pool size
type: validator
sources:
  - experiments/s059i_density_scan.py
  - experiments/s059i_validator_perturbed.md
  - experiments/s059_thread.md
related:
  - project_omega_grid_architecture.md
created: 2026-05-08
updated: 2026-05-08
confidence: high (single seed, 3 RNG trials per density; surfaces a structural variance issue not captured by raw sample count)
---

# TL;DR

Densifying the SO(3) pool from 100k → 400k drops the median closest-to-truth
member of the T_A=25 survival cloud on seed 28 from **10.96°** (bimodal-cliff
regime per s059i_validator_perturbed) to **2.32°** (deep in the robust
regime). But variance is high: at N=400k the closest-survive ranges from
1.41° to 8.33° across 3 random seeds, and the bad RNG case persists
unchanged at N=800k — adding samples can't beat a "donut hole" in the
survival shell. **Implication for s059j**: N=400k with one good RNG seed
(e.g. 42 or 43 here) puts seed 28 in the robust regime; the cheaper fix
to handle unlucky RNG is multi-RNG union or true low-discrepancy
(Sobol-Shoemake) sampling instead of `Rotation.random`. The current
sample_so3_pool's name ("Sobol") is a misnomer — it returns Haar-uniform
random samples, which is the source of the variance.

# What

For each pool size N ∈ {100k, 200k, 400k, 800k} and each RNG seed
∈ {42, 43, 44}: sample N Haar-uniform quaternions on SO(3); project at
T_A=25 on seed 28; survive at TOL_MAG=0.10 against measured magnitude;
report (a) closest geodesic distance from truth_q_a to any survivor and
(b) closest geodesic distance from truth_q_a to any pool member
(no survival filter, geometric-only baseline).

# How

`experiments/s059i_density_scan.py`. Reuses
`lib.c_t_pipeline.sample_so3_pool` + `project_directions` + `survive_at_epoch`
+ `nearest_in_pool_to_truth` (the same primitives s059_pilot uses for cloud
generation). Surrogate v1 with the s10_5M weights matches s059_pilot's
filter exactly. No multiprocessing — vectorised numpy is fast enough at
800k samples (~5s per trial).

# Result

| N        | trials | \|C_a\| median | closest-survive median (range)  | closest-pool median (range) |
|----------|-------:|---------------:|---------------------------------|----------------------------:|
| 100,000  | 3      |   383          | 10.96° ( 9.41°–17.08°)          |  3.37° ( 3.06°–3.47°)       |
| 200,000  | 3      |   748          |  8.33° ( 5.50°– 9.41°)          |  3.06° ( 2.49°–3.47°)       |
| 400,000  | 3      |  1473          |  2.32° ( 1.41°– 8.33°)          |  2.20° ( 1.41°–2.49°)       |
| 800,000  | 3      |  2965          |  2.32° ( 1.41°– 8.33°)          |  1.41° ( 0.97°–1.45°)       |

Three readings:

1. **Median improves with N.** Median closest-survive drops 10.96° → 2.32°
   from 100k → 400k. By 400k, the median is well inside the s059i robust
   regime (≤ 7.5°).
2. **Variance is high and structural.** At 400k, RNG seed 44 sits at
   8.33°, and at 800k it stays at 8.33° — adding 400k more samples failed
   to plug the gap. The issue is that the closer-to-truth pool members
   added at higher N happen to fall *outside* the survival shell at
   T_A=25 (predicted-mag mismatch > TOL_MAG), while the survivors near
   truth at lower N stay roughly where they were.
3. **Geometric closest-to-pool decreases monotonically with N.** All-pool
   median drops 3.37° → 1.41° from 100k → 800k, scaling roughly as
   N^(-1/3) as expected for uniform 3-D sampling. This says the sampler
   is doing its job; the variance in *survival-cloud* closest is a
   property of the survival shell's geometry, not the sampler.

# Why this matters

s059i_validator_perturbed established that the ω-grid cost surface is
sharply discriminative at q_a noise ≤ 7.5° and bimodal at ~10°. The 100k
pool's median 10.96° was right at the cliff — that's why s059j needed a
mitigation.

This scan says **densifying to N=400k is sufficient on seed 28 with the
right RNG seed**: closest-survive 1.41-2.32° puts the architecture deep
in the robust regime. The cost is ~2.5s per epoch survival eval at 400k
(scales linearly), so a full 500-epoch cloud at 400k is roughly 4× the
current 5-min cold cloud-gen ≈ 20 min wall — still well within a session
budget.

But the persistent 8.33° outlier at RNG seed 44 even at 800k says raw
densification with one RNG seed is **not bulletproof** — there is
RNG-dependent risk of landing on an unlucky shell sampling. Two
mitigations are cheaper than going to 1.6M or 3.2M:

- **Multi-RNG union.** Generate 2-3 different RNG-seed pools at
  N=400k each, union the survivors, take the closest. The N=400k pools
  with seeds 42, 43, 44 give closest-survive 2.32°, 1.41°, 8.33°
  respectively; union closest = 1.41° (deep robust). Cost: 3× a
  400k cloud-gen ≈ 60 min wall, still session-tractable.
- **True low-discrepancy sampling.** `sample_so3_pool` is named "Sobol"
  but uses `Rotation.random` (Haar-uniform i.i.d.). A real
  Sobol-Shoemake or quaternion-Halton sequence has lower discrepancy
  and tighter near-neighbour distance bounds for the same N. This is
  a one-time refactor in `lib/c_t_pipeline.py:sample_so3_pool` that
  could turn 400k into a reliable robust regime without a multi-seed
  union. Worth doing before scaling further.

# Numbers worth remembering

- 100k Sobol on seed 28 T_A=25: |C_a|=383, closest-survive=10.96° (the
  s059h baseline). Bimodal regime.
- 400k Sobol seed=42: |C_a|=1473, closest-survive=2.32°. Deep robust.
- 400k Sobol seed=44: |C_a|=1500, closest-survive=8.33°. Borderline.
- All-pool closest-to-truth scales roughly as N^(-1/3): 3.37° → 1.41°
  from 100k → 800k.
- Wall: ~5 sec per density per RNG at 800k.
- Cohort cost projection: full 500-epoch cloud at 400k ≈ 20 min wall;
  multi-RNG (3×) ≈ 60 min.

# Artefacts

- `experiments/s059i_density_scan.py`
- `results/s059i_density_scan/seed028_T025/{run.log, summary.json,
  density_scan.png}`

# Out of scope

- Cohort generality. Seed 28 may not be representative; some seeds may
  reach the robust regime at 100k, others may need 800k+. A single-pass
  cohort scan (~5-10 representative seeds × 4 densities × 3 RNGs) is the
  next decisive measurement.
- True Sobol-Shoemake sampler. The library is currently misnamed; a
  refactor to use a low-discrepancy sequence is independent of the
  s059j build but reduces the multi-RNG cost.
- s059i_validator at the actual closest-survive q_a (not random
  perturbation). This is the most direct test that the architecture
  works end-to-end — `run s059i_validator with --qa-perturb-deg
  <closest-survive-deg>` would do it, or a custom variant that uses
  the actual cloud member's quaternion.

# Cross-references

- `experiments/s059i_validator_perturbed.md` — the regime boundaries
  (≤7.5° robust, ~10° bimodal, ≥15° fail).
- `experiments/s059_thread.md` — s059h's 10.96° baseline measurement.
- `lib/c_t_pipeline.py:sample_so3_pool` — the misnamed sampler.
