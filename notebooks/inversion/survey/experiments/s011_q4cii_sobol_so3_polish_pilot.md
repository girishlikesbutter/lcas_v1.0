---
title: "s011 — Q4c-ii cohort pilot: multi-axis Sobol(q0) on SO(3) × LM polish at truth-ω"
type: experiment
sources:
  - "results/s011/runs.npz"
  - "results/s011/summary.json"
  - "results/s011_run.log"
related:
  - "[[s002_surrogate_landscape_probe]]"
  - "[[s005_joint_local_descent]]"
  - "[[s006_seed28_landscape_at_truth_omega]]"
  - "[[s009_cohort_basin_radius_probe]]"
  - "[[s010_seed44_landscape_at_truth_omega]]"
created: 2026-04-30
updated: 2026-05-01
confidence: high
---

## TL;DR

**Decisive positive at low density: 9/10 pilot seeds get ≥1 in-basin landing
at N=64.** Multi-axis Sobol-Shoemake on SO(3) + LM polish at fixed truth-ω
on 10 seeds × 64 ICs = 640 LM runs (Pool(8), max_nfev=60, wall 39 min).

**Surprise of the round: LM has a basin-grab radius FAR larger than the
basin itself.** s006 measured seed-28 truth-basin radius at sub-Sobol-2046
resolution (sub-10° q0); s010 same for seed 44. Yet at N=64 (cell radius
~28° on SO(3)), seed 28 gets 7/64 in-basin landings and seed 44 gets 5/64.
LM-polished-Sobol does NOT need Sobol density inside the basin — only
inside the LM grab radius (empirically ~50-100° on these seeds).

**Failure mode (seed 10): NEW tight-tail seed.** s009 said seed 10's
body-X-axis T1 passed strict basin recovery; multi-axis Sobol on SO(3)
exposes 0/64 in-basin landings. 55/64 ICs LM-polish to final_mse < 0.5
mag² but at q0_err 28°-180° — many low-MSE competing attractors, none
near truth. Confirms s009's tight-tail census (3/100 along body-X)
under-counts the multi-axis cohort tail.

**Decision:** Q4c works at low density. Architecture = uniform Sobol-
Shoemake(q0) on SO(3) at N=64 + LM polish per candidate covers 90% of
the pilot cohort (and 97% if we extrapolate from s009's body-X 97%
recovery — the actual cohort coverage at N=64 across 100 seeds is the
natural follow-up).

## What

Q4c — global-search → local-polish handoff at cohort scale. With s005
having settled joint LM as the inversion architecture inside the s003
truth-ω tube, s006/s010 having confirmed sub-Sobol-resolution narrow
basins exist on at least 2 of 100 cohort seeds, and s009 having measured
the cohort body-X-axis basin distribution (45% wide / 52% mid / 3% tight),
this round answers:

> At what Sobol(q0) density on SO(3) (uniform Shoemake) does LM polish
> from each candidate land ≥1 IC inside the per-seed truth basin for
> ≥90% of pilot seeds, AT FIXED TRUTH-ω?

ω is held at truth in this round to isolate the q0 axis. The next round
(s012) will add the ω-grid axis once we know how the Sobol(q0) density
scales with in-basin yield.

## How

10 pilot seeds: 8 PA-stratified s002 anchors + 2 cohort-tail seeds
(28 sub-Sobol-narrow truth basin; 44 sub-Sobol-narrow competing basin).

Per seed:
- 64 Sobol-Shoemake quaternions on SO(3), SOBOL_SEED=42 (matches
  s002 / s006 / s010 — N=64 is a prefix of N=2046 modulo Sobol-Sobol's
  draw-order semantics).
- ω fixed at truth-ω (per seed, from `traj_load.load_truth(seed)`).
- LM polish per IC: scipy `least_squares(method='lm')`, 6-DOF
  `x = (δθ, ω)`, residuals = surrogate full-LC residuals.
- `max_nfev = 60` (calibrated against s005 in-basin nfev p90=47, max=66
  + smoke test on seed 41 random Sobol ICs).
- xtol/ftol/gtol = 1e-8.

Pool(8), BLAS=1.

**Critical implementation note: `init_worker` must call `torch.set_num_threads(1)`
and `torch.set_num_interop_threads(1)`.** Surrogate is PyTorch-backed; torch
ignores `OMP_NUM_THREADS` env var and defaults to `cpu_count // 2` (16
threads here). Pool(8) workers each spinning 16 threads = 128 threads on
8 cores → ~8× per-worker slowdown.  Updated
`memory/feedback_blas_threads_for_pool.md` and bumped to "feedback —
critical methodology" status.

In-basin definitions (match s005, s009):
- Strict: q0_err < 5° AND ω_dir_err < 1° AND |ω_mag_err| < 5%
- Loose: q0_err < 10° AND ω_dir_err < 2° AND |ω_mag_err| < 10%
- Twin: same as strict but on twin = q_180x · q0_truth

Per-cell decision metrics:
- `n_truth_basin_strict`: count of ICs landing in strict basin
- `n_unique_in_basin_clusters`: greedy 1°-geodesic clustering of in-basin
  landings (multi-cluster basins would show up here)
- `n_competing_basins_below_mse_0_5`: clusters of LM landings ≥30° from
  truth with final_mse < 0.5 mag² (the s009/s010 competing-basin
  signature)
- `min_q0_err`, `min_final_mse`: best landing per seed

Pseudocode:
```
for seed in [6, 10, 21, 28, 41, 44, 48, 60, 84, 91]:
    q0_truth, ω_truth ← traj_load.load_truth(seed)
    q0_set ← shoemake(scrambled-Sobol(N=64, seed=42))
    in parallel (Pool(8)):
        for q0_seed in q0_set:
            x_final ← scipy.least_squares(
                fun=residuals(x, q0_seed, ω at truth),
                x0=(0,0,0, ω_truth), method='lm', max_nfev=60,
            )
            classify in-basin / twin / competing
    aggregate per seed
cohort_yield = #seeds with n_in_basin ≥ 1
```

## Result

```
=== s011 cohort yield by density ===
  N=  64: 9/10 seeds with ≥1 in-basin landing  (90.0%)

=== per-(seed, density) yield ===
  seed     N  in-basin  unique  competing  min_q0_err   min_mse
     6    64         8       1          5     0.287°    2.788e-03
    10    64         0       0         49    28.245°    3.460e-02   ← FAIL
    21    64         3       1          1     0.016°    5.718e-04
    28    64         7       1          1     0.034°    5.776e-04
    41    64         2       1          2     1.045°    1.211e-04
    44    64         5       1          4     0.033°    3.208e-04
    48    64         1       1         13     0.188°    2.813e-04
    60    64         4       1          3     0.056°    1.539e-03
    84    64         3       1          5     0.178°    2.447e-04
    91    64         3       1          3     0.073°    2.005e-04

  → DECISION: Sobol-q0 cheap (N=64 achieves 9/10); Q4c works at low density.

Total wall: 2361.2 s (39.4 min)
```

Twin recoveries: 0/640 (consistent with s005, s009, s010 — twin not an
attractor at near-truth ω).

Per-seed in-basin fractions:
| seed | in-basin / N | min_final_mse | basin radius (s005/s009) |
|------|------|------|------|
| 6    | 8/64 = 12.5% | 2.79e-3 | wide (~15°) |
| 10   | 0/64         | 3.46e-2 | NEW failure |
| 21   | 3/64 = 4.7%  | 5.72e-4 | not measured |
| 28   | 7/64 = 10.9% | 5.78e-4 | tight (~2°) |
| 41   | 2/64 = 3.1%  | 1.21e-4 | mid (~8°) |
| 44   | 5/64 = 7.8%  | 3.21e-4 | tight competing-basin |
| 48   | 1/64 = 1.6%  | 2.81e-4 | not measured |
| 60   | 4/64 = 6.3%  | 1.54e-3 | not measured |
| 84   | 3/64 = 4.7%  | 2.45e-4 | not measured |
| 91   | 3/64 = 4.7%  | 2.01e-4 | mid (~5°) |

Seed 10 failure decomposition (the only round-1 failure):
- All 64 ICs landed at q0_err 28°-180° (median 137°, p10 86°).
- 55/64 final_mse < 0.5 mag² (Band-B-equivalent surrogate fits).
- 5/64 final_mse < 0.05 mag² — same order of magnitude as
  truth_mse_ref (3.59e-3 mag² from s001 cache).
- 49 competing-basin clusters detected (≥30° from truth, mse < 0.5).
- s009's body-X-axis T1 passed strict for seed 10. The body-X-axis IC
  is a special slice; multi-axis Sobol exposes a much wider competing-
  basin landscape.

## Why this matters

1. **LM has a much larger basin-grab radius than the basin itself.**
   - s006: seed 28 truth basin invisible at Sobol-2046 (closest 10°);
     s005: seed 28 truth basin radius ~2°.
   - Yet at N=64 (cell radius ~28°), seed 28 gets 7/64 in-basin landings.
   - Same for seed 44: sub-Sobol-narrow per s010, but 5/64 in-basin at N=64.
   - Mechanism: LM is finite-difference-Jacobian Levenberg-Marquardt,
     which can take very large initial steps when far from minima.
     Empirically the polish funnels into truth from up to ~75-100° away
     in q0_geodesic when ω is near truth (smoke test: 75° IC → basin in
     11 nfev).
   - **This invalidates the s006-driven worry** that sub-Sobol-narrow
     basins force impractically high Sobol density. The Sobol density
     just needs to seed inside the LM grab radius, which is much larger
     than the basin.

2. **Q4c-cohort architecture is settled at the q0 axis.** Uniform
   Sobol-Shoemake(q0) on SO(3), N=64 candidates, LM polish per candidate.
   Wall 39 min for 10 seeds at fixed ω; the 100-seed cohort extrapolation
   is ~6.5 hours wall (~13 hours / 8 workers single-machine), or ~80 min
   if parallelised across 100 / 8 = 13 hosts. The full 5×5 ω-grid
   extension multiplies this 25× — feasible but heavy.

3. **Seed 10 reveals a new tight-tail seed not seen by s009.** Body-X-axis
   ICs for seed 10 happened to LM-polish into truth (s009 T1 strict pass);
   generic SO(3) ICs LM-polish into 49 distinct competing basins, none
   near truth at fixed truth-ω. **This is the multi-axis effect s009 said
   was needed but couldn't measure** with body-X-only ICs. New cohort
   estimate (extrapolating): the true tight-tail at multi-axis Sobol on
   100 seeds is likely ≥3% (s009's body-X count) but plausibly 5-10%.

4. **Competing basins are widespread, not unique to s009/s010 finding.**
   5 of 10 seeds have ≥3 LM-discovered competing-basin clusters
   (final_mse < 0.5 mag², ≥30° from truth). Seeds 10, 48 in particular
   have 49 / 13 clusters — many distinct low-MSE attractors. This means
   the surrogate-MSE landscape at fixed truth-ω has rich multi-attractor
   structure on most seeds; the s002 finding "argmin = truth at fixed
   truth-ω on 8/8" is preserved (truth is still the global argmin per
   s002/s006/s010), but the local-minimum structure is dense. LM polish
   discovers these; bare Sobol-2046 doesn't.

5. **In-basin landings are single-cluster everywhere.** All 9 seeds with
   in-basin landings show n_unique_clusters = 1 — every IC that
   converges to truth basin converges to the same point. No multi-
   cluster truth basins. Consistent with s005 / s009 / s010 single-
   basin findings.

## Numbers (cached)

- Wall: 2361.2 s (39.4 min) on 640 LM runs, Pool(8), BLAS=1, max_nfev=60.
- Per-LM mean: 29.5 s effective (vs Pool(1) diagnostic 22 s/run; small
  Pool overhead).
- Cohort yield at N=64: **9/10** strict in-basin / **9/10** loose in-basin.
- Twin recoveries: 0/640.
- min_q0_err over the 9 successful seeds: 0.016° (seed 21) — sub-degree
  recovery from random Sobol IC, confirming LM's grab radius is ample.
- Seed 41 was the only successful seed with min_q0_err > 1° (1.045°);
  still inside strict basin. May indicate seed 41's basin floor is
  slightly noisy / surrogate-intrinsic.

## Artefacts

- `experiments/s011_q4cii_sobol_so3_polish_pilot.py` (script)
- `results/s011/runs.npz` (640 LM runs × full state)
- `results/s011/summary.json` (per-cell + cohort-yield + decision)
- `results/s011/yield_vs_density.png` (per-seed bars + cohort yield)
- `results/s011/q0_err_distribution.png` (CDF of final q0_err)
- `results/s011_run.log` (gitignored)

## Out of scope

- **ω-grid extension (s012).** This round holds ω at truth. The natural
  follow-up adds 5×5 = 25 ω-cells (5 dirs × 5 mags) per (seed, q0-Sobol)
  cell. Total: 100 seeds × 25 ω-cells × 64 Sobol = 160k LM runs;
  100-seed wall ~80 hours / 8 workers single-machine. Heavy; needs
  cluster or budget pruning.

- **N=128/256 follow-up on seed 10.** Not run this round. Decision-bar
  was 90% at N=64; achieved. The interesting question is whether seed 10
  is recoverable at N=128/256, or genuinely requires architectural
  changes (basin-hopping or ω-grid). 30-second probe on seed 10 at
  N=256 would test this.

- **Hi-fi ρ-band classification.** Surrogate-MSE alone does not give a
  ρ-band per `concepts/rho_band.md` (ρ requires hi-fi MSE). Building
  `lib/hifi_render.py` and re-rendering the 1-cluster-per-seed in-basin
  landing + the 100-ish competing-basin landings would enable
  observational-indistinguishability classification. Likely most
  in-basin landings are Band A (truth_mse_ref class); competing-basin
  states need actual hi-fi to classify.

- **Cohort-scale 100-seed run.** This pilot is 10 seeds; the 100-seed
  cohort run at N=64 is the natural follow-up that converts the pilot
  yield into a cohort yield (likely 90-97%). Wall extrapolation: ~6.5
  hours / 8 workers.

- **Sobol-density × yield curve.** This round is N=64 only. N=32 / N=128
  / N=256 would map out the diminishing-returns curve. Out of scope
  this round; the headline decision is reached at N=64.

## Cross-references

- `s005` — established the LM polish in 6 DOF used here. Inside-tube ICs
  on 5 PA-stratified seeds gave per-seed basin radii. s011 is the
  cohort-density extension.
- `s006` / `s010` — measured sub-Sobol-narrow basins on seeds 28 / 44 at
  2046 Sobol density. Predicted (incorrectly) those seeds would fail at
  N=64. Both succeed at N=64 because LM's grab radius >> basin radius.
- `s009` — cohort body-X-axis basin probe; identified seeds 7/44/76 as
  tight tail. s011's seed-10 failure shows that body-X axis under-
  represents the multi-axis tail; the true tail is broader.
- `concepts/q_omega_coupling.md` — informs the 6-DOF parameterization.
- `memory/feedback_blas_threads_for_pool.md` — updated this session
  with the torch-set_num_threads requirement that was missed in the
  original 2026-04-29 m138 lesson.
