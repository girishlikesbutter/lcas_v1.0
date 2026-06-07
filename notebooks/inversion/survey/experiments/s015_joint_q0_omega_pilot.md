---
title: "s015 — joint (q0, ω) inversion pilot: random-ω stitched-Sobol + 6-DOF LM"
type: experiment
sources:
  - "results/s015_smoke/runs.npz"  # seed 6, N=512 pre-flight
  - "results/s015_diagnostic_n64/runs.npz"  # 8 seeds × N=64 cohort diagnostic
  - "results/s015_diagnostic_n64/summary.json"
related:
  - "[[s003_landscape_vs_omega]]"
  - "[[s005_joint_local_descent]]"
  - "[[s011_q4cii_sobol_so3_polish_pilot]]"
  - "[[s014_cohort_rho_band]]"
  - "[[s014b_n_rotations_analysis]]"
  - "[[s016_omega_mag_frozen_polish]]"  # follow-on
  - "[[s016c_prime_fresh_sobol_fixed_omag]]"  # follow-on
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## TL;DR

**Decisive negative: 0/512 in-basin on seed 6 (smoke), 0/8 cohort yield at
N=64.** First joint (q0, ω) inversion pilot under random-ω. Pre-registered
prediction (lower bound 0-2/512 on seed 6 from tube-volume argument)
confirmed exactly. The cohort architecture from s011 (lowest-surrogate-MSE
selector returning Band A truth on 9/9 seeds at fixed truth-ω) is
**structurally broken under joint search** — argmin-MSE landings are at
q0=80-180° from truth on every seed, not near-truth.

The planned 8-seed N=512 full pilot was cancelled mid-design after the
seed-6 smoke confirmed the prediction; replaced with the cheaper N=64
8-seed cohort diagnostic. Diagnostic settled the cohort failure mode:
uniform across n_rotations strata, no clean basin / tail split that
existed at fixed truth-ω. Architectural pivot required (s016 chain).

## What

Q4c at the joint search level — does our s011 architecture (uniform Sobol
+ LM polish, lowest surrogate-MSE per seed) still work when ω has to be
discovered alongside q0?

The proximal precedent is s003: surrogate-MSE has a thin truth-ω tube
(~1° dir / ~2-5% mag). Outside the tube, the per-ω q0-best landscape is
incoherent and gives no useful gradient toward truth-ω. So s003 ruled out
**decoupled** outer-ω / inner-q0 search. s005 then showed joint LM
recovers from inside the tube. s015 asks: does Sobol density on the joint
6-DOF surface seed enough ICs into the tube for LM to do its work?

## How

### Sampling — three independent Sobol streams

| stream  | dim | seed | mapping            | bounds                    |
|---------|-----|------|--------------------|---------------------------|
| q0      | 3   | 42   | Shoemake → S^3     | uniform on SO(3)          |
| ω-dir   | 2   | 43   | inverse-CDF on S²  | uniform on S²             |
| ω-mag   | 1   | 44   | linear             | uniform [0.1, 1.5] dps    |

Three streams (NOT a 6-D Sobol) are stitched per-IC: ω = ω_dir · ω_mag,
then assembled with q0_seed_wxyz into the IC dict. Qualitative argument
for separate streams: q0 / ω-dir / ω-mag drive different LC features
(orientation / glint axis sweep / temporal compression); independent
streams preserve marginal coverage of each.

The q0 stream's first N=64 IDs are bit-identical to s011's N=64 set
(SOBOL_SEED=42 unchanged) — when paired with truth-ω vs random ω, this
gives a direct cross-experiment comparison.

ω-mag bounds [0.1, 1.5] dps cover the cohort exactly (measured: min
0.106, max 1.476 dps). Linear (not log) — only 1.5 OOM range.

### Search

scipy `least_squares(method='lm', max_nfev=120)`, 6-DOF parameterization
identical to s005 / s011: x = (δθ, ω), δθ in tangent space around
q0_seed, ω absolute 3-vec. residuals = surrogate full-LC residuals.
max_nfev=120 (vs s011's 60): joint 6-DOF from random ω needs more
iteration than 3-DOF from truth-ω.

### Seeds (8 total)

| seed | role                    | n_rot | ω_truth dps | truth surr_MSE | s011 N=64 result        |
|------|-------------------------|-------|-------------|----------------|-------------------------|
| 6    | anchor                  | 7.13  | 0.713       | 2.92e-3        | 8/64 in-basin (Band A)  |
| 28   | anchor_narrow_basin     | 14.38 | 1.438       | 5.85e-4        | 7/64 in-basin (s006-narrow) |
| 41   | anchor_multi_solution   | 5.65  | 0.565       | 1.33e-4        | 2/64 in-basin + class-2 attractor |
| 91   | anchor_m115_failure     | 14.26 | 1.426       | 2.03e-4        | 3/64 in-basin (Band A)  |
| 13   | n_rot_lt_2_tail         | 1.49  | 0.149       | 1.90e-4        | not in s011 — predicted seed-10-class |
| 42   | n_rot_lt_2_tail         | 1.29  | 0.129       | 8.28e-4        | not in s011 — predicted seed-10-class |
| 79   | n_rot_lt_2_tail         | 1.21  | 0.121       | 4.40e-4        | not in s011 — predicted seed-10-class |
| 49   | fresh_average           | 11.10 | 1.110       | 5.53e-4        | not in s011 — clean median pick     |

## Result

### Prediction (pre-result, pre-registered 2026-05-01)

**Quantitative tube-volume argument.** s003 measured the truth-ω tube on
seeds 6/41/91 as ~1° dir-radius / ~2-5% mag-radius. The fractional volume
of this tube on the s015 ω sample space (S² × [0.1, 1.5] dps):

```
S² fraction:  cap of half-angle 1° = 2π(1−cos(1°))/4π ≈ 1.5e-4 / 4π ≈ 1.2e-5
mag fraction: ω_truth × ±5% over a 1.4-dps range ≈ 0.05·ω_truth/1.4
              = 5% for ω_truth=1.4 dps; 0.4% for ω_truth=0.1 dps
joint tube:   ~1e-6 (high-ω seeds) to ~5e-8 (low-ω seeds)
```

Expected number of N=512 random ω samples landing inside the tube: ~5e-4
to ~3e-5 per seed. **Essentially zero.** If LM's ω basin-grab radius does
not extend BEYOND the s003 tube, s015 in-basin yield will be ~0 on every
seed.

The s011-style picture (LM grab radius ≫ basin radius) might NOT
generalise to ω: s003 explicitly said outside the tube the landscape is
noisy-multi-basin, with no q0-coherent gradient to pull LM toward
truth-ω. So my honest prediction:

| seed | n_rot | predicted s015 N=512 in-basin yield |
|------|-------|--------------------------------------|
| 6    | 7.13  | 0-2/512                              |
| 28   | 14.38 | 0-1/512  (narrow joint basin)        |
| 41   | 5.65  | 0-2/512                              |
| 91   | 14.26 | 0-2/512                              |
| 13   | 1.49  | 0/512   (n_rot<2 tail)               |
| 42   | 1.29  | 0/512   (n_rot<2 tail)               |
| 79   | 1.21  | 0/512   (n_rot<2 tail)               |
| 49   | 11.10 | 0-2/512                              |

**Cohort yield prediction: 1-3/8 seeds get ≥1 in-basin landing.** If the
result is much higher, LM's ω-grab radius is far wider than s003 implies
— good news, joint search is easier than expected. If the result matches
prediction (≤3/8), the next move is **ω-grid stratification**: lay a
fixed grid of ω cells, run s011-style Sobol-q0 + LM polish within each
cell, accept the cell that lands inside the s003 tube.

### Pre-flight (seed 6, N=512, smoke) — 2724 s wall

Result matches lower-bound prediction exactly.

| metric | value |
|---|---|
| in-basin (strict) | 0/512 |
| in-basin (loose) | 0/512 |
| min q0_err | 11.45° (one-axis only; ω at this IC was 142° / 60% off) |
| min ω_dir_err | 7.74° (different IC) |
| min surrogate-MSE | 1.61 mag² (500× higher than truth-MSE 3e-3) |
| argmin-MSE landing | q0=147°, ω_dir=107°, |ω_mag|=31% (cohort selector returns garbage) |
| MSE p0-p100 | 1.61 → 6.73 (whole 512-IC distribution sits in narrow flat plateau) |
| per-IC wall | median 36s, p90 72s (max_nfev=120 binding for >10% of ICs) |

### N=64 cohort diagnostic (8 seeds, 3208 s wall) — replaced the planned N=512×8 pilot

Cohort yield 0/8 in-basin (per prediction). Per-seed signature:

| seed | role | min q0_err | min ω_dir_err | min surr_MSE | competing basins |
|---|---|---|---|---|---|
| 6 | anchor | 46° | 8° | 1.62 | 0 |
| 13 | n_rot<2 | 22° | 4° | 0.86 | 0 |
| 28 | narrow | 51° | 13° | 0.43 | 1 |
| 41 | multi | 61° | 16° | 0.42 | 3 |
| 42 | n_rot<2 | 36° | 10° | 3.55 | 0 |
| 49 | fresh-avg | 29° | 15° | 3.16 | 0 |
| 79 | n_rot<2 | 49° | 6° | 2.04 | 0 |
| 91 | m115 | **14°** | 5° | **0.14** | 2 |

Five takeaways:
1. **No clean cohort split.** s014b's n_rotations correlation (predictive
   at truth-ω) does NOT predict failure mode at random-ω.
2. **ω-mag is dramatically more searchable than ω-dir.** 41/512 ICs (8%)
   landed within 10% of truth-ω-mag; only 6/512 within 5° of truth-ω-dir.
3. **n_rot<2 seeds collapse ω-mag toward zero** (medians 409% / 567% / 622%).
4. **Cohort selector returns garbage on 8/8 seeds** at random-ω.
5. **Seed 91 outperforms** with min_q0_err=14° / min_mse=0.14 (5-10× better
   than other anchors).

Cost-of-ω-search vs s011 truth-ω: ∞ on every comparable seed (s011 had
in-basin counts; s015 has zero).

### Full pilot (8 seeds, N=512) — CANCELLED

The planned ~6-hr full pilot was cancelled after the smoke confirmed the
tube-volume prediction. Per workspace cost-benefit gate, repeating the
seed-6 negative on 7 more seeds at higher density burns compute to confirm
a structural finding we already understood. Replaced with the N=64
diagnostic (~53 min) which characterises the cohort failure mode.

## Why this matters

s015 is the **first measurement of the actual inversion problem.** Up to
s014, every "cohort yield" claim was conditional on knowing ω — a luxury
the production inversion does not have. If s015 yield drops sharply from
s011's 9/10 (truth-ω cheat), we'll have a quantitative measure of how
much the ω-search costs us. If it stays high, the architecture is
production-ready and the next move is the cohort scan + ρ-band sweep.

## Numbers

Smoke (seed 6, N=512): see Result section above. Diagnostic (8 seeds, N=64):
see per-seed table. Full numerics in `results/s015_diagnostic_n64/summary.json`.

## Artefacts

- `experiments/s015_joint_q0_omega_pilot.py` — script
- `results/s015_smoke/{runs.npz, summary.json, *.png, run_n512.log}` — pre-flight
- `results/s015_diagnostic_n64/{runs.npz, summary.json, *.png, run.log}` — cohort diagnostic
- `results/s015/` — empty (full pilot cancelled)

## Out of scope

- ρ-band hi-fi validation of the s015 winners (deferred to s016 once we
  know which seeds produce winners).
- Cohort-scale 100-seed scan (deferred until N is calibrated).
- Architecture optimisation (early termination, basin-hopping, hierarchical
  search) — s015 measures the un-optimised baseline so we can quantify
  what those optimisations need to deliver.

## Cross-references

- `concepts/known_pathologies_to_revalidate.md` — s015 confronts the
  "joint search structurally infeasible" claim from buggy-era m115.
- `PROGRESS.md` — current question section.
- `s003_landscape_vs_omega.md` — the truth-ω tube measurement that
  motivates the joint architecture.
- `s011_q4cii_sobol_so3_polish_pilot.md` — same q0 Sobol stream, fixed
  truth-ω instead of random.
- `s014b_n_rotations_analysis.md` — the n_rot<2 tail prediction tested
  here on 13/42/79.
