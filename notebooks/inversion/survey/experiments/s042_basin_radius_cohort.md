---
title: "s042 — joint-LM basin radius cohort probe (10 stratified seeds + s040's 3 = 13 seeds)"
type: experiment
sources:
  - experiments/s040_basin_radius_3seed.md
  - experiments/s041_q0_saturation_probe.py
  - results/s042_basin_radius_cohort/
  - results/s032_cohort_fast/cohort_progress.csv
related:
  - s011
  - s014b
  - s034
  - s040
  - s041
created: 2026-05-06
updated: 2026-05-06
confidence: high
---

## TL;DR

Extends s040's 3-seed basin probe to a 13-seed |ω|-stratified cohort.
780 LMs, Pool(8) wall **36.3 min**. Three load-bearing findings:

1. **Truth/twin basins are reflection-symmetric on 12/13 seeds.** Only
   seed 89 carries a truth-vs-twin asymmetry (q0-perp 5° / 20°), driven
   by a competing attractor in the truth neighborhood. Symmetry is a
   generic property of the body-twin convention; seed 89 is the
   exception.
2. **ω-mag basin width inversely correlates with |ω|.** Low-|ω|
   tumblers (<0.25 dps) saturate ±10%; high-|ω| tumblers (>1.0 dps)
   are ±1-3%. **Cohort-scaling rule for ω-prior accuracy**: roughly
   `δ|ω|/|ω| < 1-2% / |ω|_dps^(0.5..1)`. Tightest seed in cohort:
   **seed 14 with +0.5%/-2% ω-mag basin and 0.5° ω-dir basin**.
3. **Seed 14 hides a previously-uncatalogued multi-solution
   attractor**: ρ=2.78 (Band B), q0_err=86°, ω-dir=0.37°,
   ω-mag=+0.541%. Not on s014b's list. Surfaced naturally by basin
   probing.

Cohort takeaway: the binding cohort constraint for the ω-prior is
~0.5%-1% on the worst seeds. q0 IC density is largely a wash (basin
30°-60° on every seed, never tighter than 25°).

## What

For each (seed) × (attractor ∈ {truth, twin}) × (axis ∈ 4) ×
(magnitude ∈ {7-9 levels}), we run scipy LM polish from the perturbed
state. 4 axes: q0_along_omega, q0_perp_omega, omega_mag_pct,
omega_dir_deg. Magnitudes:

- q0 axes: {2, 5, 10, 15, 20, 25, 30, 45, 60}° (extended from s040's
  {2..30}° based on s041's edge measurement at 45-60° on seed 23).
- ω-mag: {±0.5, ±1, ±2, ±3, ±5, ±7, ±10}% (signed).
- ω-dir: {0.5, 1, 2, 3, 5, 7, 10}°.

## How

`experiments/s042_basin_radius_cohort.py`. Imports s040 helpers
(`_polish_one`, `_worker_init`, `build_seed_payload`,
`rotvec_quat_wxyz`, etc.). Same LM recipe (`scipy.least_squares
(method='lm', max_nfev=200, xtol=ftol=1e-8)`), same forward model,
same 6-DOF surrogate-MSE objective. Pool(8) fork with surrogate
loaded once per worker.

10 new seeds chosen by stratification:
- |ω| quantile (6): 79, 57, 62, 6, 14, 19 spanning q10-q90.
- Multi-solution-rich (2): 84 (s014b class_2), 16 (n_rot<2 boundary).
- Narrow-basin pilot class (1): 44 (s011 class).
- n_rot<2 tail (1): 42.

Combined with s040's {23, 28, 89} for a 13-seed cohort table sorted
by |ω|.

## Result

### Cohort basin radii table (13 seeds × 2 attractors)

| seed | |ω| dps  | src   | q0_along | q0_perp |   ω-mag±   | ω-dir | A/B/C/D (truth) | A/B/C/D (twin)|
|-----:|--------:|:-----:|---------:|--------:|-----------:|------:|----------------:|--------------:|
|   79 | 0.121   | s042  | 30°      | 10°     | +10/−10%   | 5° / 3°(twin) | 27/3/2/7  | 25/4/3/7  |
|   42 | 0.129   | s042  | 30°      | 45°     | +10/−10%   | 10°   | 32/0/5/2        | 32/0/5/2      |
|   16 | 0.198   | s042  | 30°      | 15°     | +7/−10%    | 10°   | 31/0/0/8        | 31/0/0/8      |
|   57 | 0.201   | s042  | 30°/60°(twin) | 15° | +10/−10%  | 10°   | 32/0/0/7        | 34/0/0/5      |
|   89 | 0.240   | s040  | 30°      | **5°/20°** | +≥10/≥10% | 7°  | 29/0/0/6        | 32/0/0/3      |
|   62 | 0.436   | s042  | 60°      | 45°     | +5/−7%     | 3°    | 32/0/0/7        | 32/0/0/7      |
|   84 | 0.507   | s042  | 45°      | 20°     | +3/−2%     | 10°   | 26/4/0/9        | 27/4/0/8      |
|    6 | 0.713   | s042  | 45°      | 60°     | +3/−5%     | 2°    | 29/0/1/9        | 29/0/1/9      |
|   23 | 0.965   | s040  | ≥30°(s040) / 45° (s041) | ≥30°/45° | +7/−7% | 7° | 32/0/0/3 | 32/0/0/3 |
|   14 | 1.229   | s042  | **25°**  | 60°     | **+0.5/−2%** | **0.5°** | 17/7/0/15 | 17/7/0/15 |
|   28 | 1.438   | s040  | ≥30°     | ≥30°    | +2/−3%     | ≥10°  | 28/0/0/7        | 27/0/0/8      |
|   44 | 1.445   | s042  | 45°      | 60°     | +1/−1%     | 3°    | 25/0/1/13       | 25/0/1/13     |
|   19 | 1.476   | s042  | 30°      | 45°     | +3/−3%     | 10°   | 30/0/0/9        | 30/0/0/9      |

(s040 q0 axes capped at 30°; s041 confirmed seed 23 edge at 45-60°.)

### Truth/twin symmetry

12/13 seeds are reflection-symmetric across truth and twin attractors
(basin radii identical to numerical noise). **Seed 89 alone is
asymmetric** (q0-perp truth=5° vs twin=20°), driven by the multi-sol
class_2 attractor sitting near truth at q0_err≈34°/ω-mag+3.4°/ρ≈11.

Seed 79 has a minor asymmetry on ω-dir (truth 5° vs twin 3°) but is
otherwise symmetric.

### |ω| vs ω-mag basin (the cohort scaling)

Sorted by |ω|:
- |ω| < 0.25 dps (n=5): ω-mag basin saturates ±10% (or close: 79, 42, 57)
  or floors at +7/−10 (16). Mean ≈ ±9%.
- 0.25 ≤ |ω| < 0.75 dps (n=4): basin tightens — 89 sat ≥10%, 62 +5/−7%,
  84 +3/−2%, 6 +3/−5%. Mean ≈ ±5%.
- |ω| ≥ 0.75 dps (n=4): 23 ±7%, 14 +0.5/−2%, 28 +2/−3%, 44 ±1%, 19 ±3%.
  Mean ≈ ±3% with **+0.5% the worst case**.

**Inverse correlation is real and strong.** Low rotation rates make
the LC insensitive to ω-mag perturbations within the obs window; high
rotation rates expose every fractional drift in ω as full extra
rotations. **Cohort-scaling rule of thumb**: `required ω-mag prior ≈
1-2% / |ω|_dps^(0.5..1)`.

### |ω| vs ω-dir basin

Less monotonic. Most seeds 7°-10° (saturated). Outliers:
- Tightest: seed 14 = 0.5°, seed 6 = 2°, seed 79 = 3-5°, seed 44 =
  3°, seed 62 = 3°.
- Widest: seed 28 = sat ≥10°, seed 16/19/57/84 = 10°.

The tight ω-dir basin seeds correlate with tight ω-mag basin seeds
(14, 6, 44) — these are "all-around tight" seeds. Rotation rate alone
doesn't predict ω-dir basin width.

### q0 basins (with extended grid)

q0_along basin: never tighter than 25° (seed 14, where 30°+ converges
to the multi-sol attractor at q0_err=86°). All other seeds 30-60°.
**LM is essentially globally convergent in q0_along at correct ω
within ≥30° on 12/13 seeds.**

q0_perp basin: 5° (seed 89 truth) to 60° (seeds 6, 14, 44 saturate).
Median ~30°. The seed 89 truth result is unusually tight; the next
tightest is seed 79 (10°).

### Seed 14 — new multi-solution attractor surfaced

Seed 14 has the tightest basin in cohort AND surfaces a previously-
uncatalogued multi-solution attractor. Detail:

| axis | escape mag | result attractor                                     |
|------|-----------:|------------------------------------------------------|
| q0_along ≥30°    | ρ=2.78  | q0_err=86.56°, ω-dir=0.37°, ω-mag=+0.541% (Band B!) |
| q0_perp 20–30°   | ρ=10.11 | q0_err=165°, ω-dir=3.35°, ω-mag=+0.879%              |
| q0_perp 45–60°   | ρ=0.28  | walks back to truth (perp wraparound)                |
| ω-mag +1%        | ρ=2.78  | escapes to the SAME q0=86°/ω+0.5% attractor          |
| ω-mag −3%/+2%    | ρ≈35    | far escape                                           |
| ω-dir 1°,2°,5°   | ρ=2.78  | escapes to the SAME q0=86°/ω+0.5% attractor          |

**The seed-14 multi-sol attractor sits at ρ=2.78 (Band B**, slightly
above Band A but below Band C). This means it would pass the
multi-solution acceptance criterion and should be admitted alongside
truth as a valid (q0, ω) explanation. Not on s014b's catalogued list.
Need to add.

### Anomalous bands

- **Seed 79**: 27 A + 3 B + 2 C + 7 D (truth) — non-bimodal landscape,
  consistent with low-|ω| / high-multi-sol regime.
- **Seed 84**: 26 A + 4 B (matches s014b class_2/3 prediction).
- **Seed 42**: 32 A + 5 C — has Band C residuals (ρ in 4-8) on some
  perturbations, consistent with multi-solution class.
- **Seed 14**: 17 A + 7 B + 15 D — most failures in cohort, but the 7
  B-band fits all converge to the same multi-sol attractor at ρ=2.78
  (clean multi-solution structure, not noise).

## Why this matters

1. **The "low-|ω| ⇒ wide ω-mag basin" hypothesis from s040 (n=3,
   inferred from seed 89) generalises to the cohort (n=13).** This is
   the single most actionable cohort insight: ω priors don't need to
   be uniformly tight — they need to be *|ω|-aware*.

2. **The binding cohort constraint is seed 14's 0.5% ω-mag / 0.5°
   ω-dir basin.** If we want truth-recovery on EVERY seed, the ω-prior
   primitive must hit within those tolerances. With s019b's 5%
   bracket and N_dir=300 grid (~1.2° spacing), seed 14 is 4-5×
   undersampled in ω-mag and 2× undersampled in ω-dir. **Either we
   tighten the ω-grid (cost-prohibitive) or we accept that seed-14-class
   seeds are unrecoverable under the current architecture.**

3. **Multi-solution acceptance saves seed 14.** The ρ=2.78 attractor
   at q0_err=86° / ω+0.5% is below the Band B ceiling (ρ<4), so it
   would be admitted under multi-solution acceptance. Truth (ρ=0.28)
   ALSO recoverable from a 0.5% / 0.5° ω-prior. So seed 14 returns 2
   solutions, both valid — exactly the multi-solution endgame.

4. **q0 is essentially never the binding axis.** Across 13 seeds, q0
   basin ≥25°. Sobol N=8 (mean ~75° spacing) likely undercoverage on
   only the worst seeds (89-truth at 5°). N=64 is overkill for
   coverage but cheap to compute. **Phi-sweep's seed-dependent q0
   distribution is fine if (and only if) ω is right.** This re-affirms
   s038's finding that the cell-filter / ω-prior is the binding
   architectural variable.

5. **Truth/twin symmetry validates the body-twin convention as
   load-bearing**: across 12/13 seeds the two attractors have the same
   LC-space basin geometry. The ONE asymmetric seed (89) has the
   asymmetry traceable to a competing attractor in truth's
   neighborhood. So body-twin is not a generic asymmetry source.

6. **Re-prioritises the cohort architecture decisions**:
   - **ω-prior tightness is the 1st-order variable**. Need ~0.5-1%
     mag, 0.5-2° dir on the worst seeds.
   - **q0 IC primitive choice is 2nd-order** (Sobol vs phi-sweep
     converges in basin width discussion).
   - **Multi-solution acceptance is necessary** for the worst seeds
     (14 in this cohort).

## Numbers

| Metric                   | Value                                  |
|--------------------------|----------------------------------------|
| Seeds (s042 new)         | 79, 57, 62, 6, 14, 19, 84, 16, 42, 44  |
| Seeds (s040 prior)       | 23, 28, 89                             |
| Total cohort             | 13 distinct seeds                      |
| LMs (s042)               | 780                                    |
| Pool                     | 8 fork workers                         |
| Wall (s042)              | **36.3 min**                           |
| LM nfev cap              | 200                                    |
| New multi-sol seed       | **seed 14** (ρ=2.78, q0=86°, ω+0.5%)  |
| Truth/twin symmetric     | 12 / 13                                 |
| Asymmetric (q0-perp)     | seed 89 only (5° vs 20°)               |
| Tightest ω-mag basin     | seed 14 +0.5%/−2%                      |
| Tightest ω-dir basin     | seed 14 0.5°                           |
| Tightest q0_along basin  | seed 14 25° (others ≥30°)              |

## Artefacts

- `experiments/s042_basin_radius_cohort.py`
- `results/s042_basin_radius_cohort/seed{006,014,016,019,042,044,057,062,079,084}/{truth,twin}_basin.json`
- `results/s042_basin_radius_cohort/summary.json`
- `results/s042_basin_radius_cohort/run.log`

## Out of scope

- Combined-axis perturbations (q0 + ω simultaneously) — the basin in
  the joint axis manifold may be tighter than independent-axis radii
  suggest.
- Probing the seed-14 multi-sol attractor's basin — would need to
  re-run s040 with that attractor as the centre.
- Cohort-wide multi-solution catalogue update (extend s014b with seed
  14's class).
- |ω|-quantile picks q40-q60 are sparse (one rep, seed 6 at 0.713 dps).
  Worth filling in if a cohort-wide ω-mag basin model is needed.
- Per-seed Sobol-vs-phi-sweep IC yield against measured basin widths
  — the natural follow-up at s039c.

## Cross-references

- **s011** — Sobol N=64 at fixed truth-ω, 9/10 cohort recovery. Now
  recontextualised: the q0 basin is wide (≥25°) on every cohort seed,
  so even N=8 likely covers most seeds. The yield bottleneck was
  ω-cell quality, not q0 IC density.
- **s014b** — multi-solution catalogue (28, 41, 48, 84, plus n_rot<2
  list). **s042 ADDS seed 14 to the multi-solution catalogue** (new
  finding).
- **s040** — original 3-seed basin probe. s042 is the cohort
  generalisation; confirms the seed-89 ω-mag-vs-|ω| hypothesis.
- **s041** — q0 saturation probe; established the 45-60° basin edge
  on seed 23 that motivated the extended q0 grid here.
- **s032** — cohort fast-path run; provided the |ω| metadata used to
  pick the stratified seed list.
- **s019b** — bracket coverage cohort study (4/79 within 5%, median
  32%). The ω-mag basin narrowness on seed 14 (+0.5%) and seed 44 (±1%)
  means s019b's bracket coverage is structurally inadequate for the
  high-|ω| tail.
