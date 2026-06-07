---
title: "s016-C' — fresh Sobol(q0 × ω-dir) at fixed harvested ω-mag (proper C test)"
type: experiment
sources:
  - "results/s016c_prime/runs.npz"
  - "results/s016c_prime/summary.json"
  - "results/s016c_prime/run.log"
related:
  - "[[s003_landscape_vs_omega]]"
  - "[[s011_q4cii_sobol_so3_polish_pilot]]"
  - "[[s014_cohort_rho_band]]"
  - "[[s015_joint_q0_omega_pilot]]"
  - "[[s016_omega_mag_frozen_polish]]"
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## TL;DR

**ω-mag prior is structurally insufficient. 0/7 seeds reach loose basin
even with fresh Sobol starts at fixed-correct ω-mag — including seed 28
where harvested ω-mag was at 0.01% error from truth (effectively
perfect).** S016 chain (Option C) is now closed: ω-mag accuracy alone
cannot bridge the q0 + ω-dir search gap.

**Critical reframe:** the cohort selector ("argmin surrogate-MSE per
seed") returns candidates with **MSE 0.005-0.011 mag²** on seeds 13 / 41
/ 79 — well into Band-A territory by surrogate proxy (ρ ≈ 0.3-0.5 if
s014's surrogate≈hi-fi non-basin holds). These are **multi-solution
attractors** (q0 80-173° from truth + ω-dir 73-122° from truth + LC
matches truth). Hi-fi ρ-band-validation of these candidates is
**deferred** — would flip the per-seed verdict from "geometric truth not
found" to "valid multi-solution candidate found" on possibly 4-5/7 seeds.

960 LM runs / 76.6 min wall.

## What

S016-C tested the C hypothesis incorrectly (re-polishing already-converged
ICs). S016-C' is the proper test: take the top-3-by-s015-final_mse ICs
per seed with |ω_mag_err|<10%, use each ω-mag as a fixed value, run
**fresh** Sobol-Shoemake(q0) on SO(3) × Sobol(ω-dir) on S² at N=64 ICs
per harvest value + 5-DOF LM polish at fixed ω-mag.

Asks: at fixed correct (or near-correct) ω-mag, can fresh random sampling
on (q0, ω-dir) at N=64 land in the truth basin? If yes, the production
architecture is two-stage: (random search) → (ω-mag harvest) → (fresh
fixed-ω-mag search).

## How

Harvest: top-3-by-s015-final_mse ICs per seed where
`|omega_mag_err_pct| < 10`. Per-seed: 6=1, 13=1, 28=3, 41=3, 42=0, 49=3,
79=1, 91=3 = 15 harvest ω-mag values total.

Per harvest value: build 64 fresh ICs via:
- Sobol-Shoemake(q0) on SO(3), SOBOL_SEED=50
- Sobol-S²(ω-dir) (inverse-CDF), SOBOL_SEED=51
- ω-magnitude held at the harvested value

5-DOF LM (same as s016): x = (δθ_3, theta, phi). max_nfev=120.
Pool(8), BLAS=1 + torch_threads=1. Total: 960 LM runs.

## Result

**0/7 seeds reach loose basin (q0<10° AND ω_dir<2° AND |ω_mag|<10%).
76.6 min wall.**

Per-seed best landings (geometric):

| seed | harvest ω-mag err | min q0 | min ω_dir | min surr_MSE |
|---|---|---|---|---|
| 6 | -2.68% | **7.1°** | 10.4° | 1.25 |
| 13 | -2.25% | 23.3° | 23.5° | **0.0045** |
| 28 | **-0.01%** (perfect) | 21.6° | 8.0° | 5.39 |
| 41 | -1.4% / +2.9% / -9.7% | 27.6° | 7.3° | **0.0059** |
| 49 | -7.7% / -8.2% / -9.1% | 13.4° | **2.7°** | 3.15 |
| 79 | -4.7% | 43.4° | **0.21°** | 0.0107 |
| 91 | +0.2% / +1.0% / -6.4% | 32.9° | 6.0° | 0.083 |

**Joint axis-condition counts** (out of total starts per seed):

| seed | q0<10° | ω_dir<2° | q0<10°+ω_dir<2°+\|ω_mag\|<10% |
|---|---|---|---|
| 6 | 1/64 | 0/64 | 0 |
| 13 | 0/64 | 0/64 | 0 |
| 28 | 0/192 | 0/192 | 0 |
| 41 | 0/192 | 0/192 | 0 |
| 49 | 0/192 | 0/192 | 0 |
| 79 | 0/64 | 1/64 | 0 |
| 91 | 0/192 | 0/192 | 0 |

**Best q0 and best ω_dir landings are at OPPOSITE ICs** — seed 6's
q0=7° IC has ω_dir=10°; no IC got tight on both axes simultaneously.

Compared to s015 N=64 diagnostic per-seed min_q0_err: seed 6 went 46° →
7° (huge), seed 28 went 51° → 22° (improvement), seed 41 went 61° → 28°
(improvement), seed 49 went 29° → 13° (improvement). Fixing ω-mag at
correct value DID improve cohort-wide min_q0_err — the bridge gets
**closer** but never reaches the strict basin.

**Multi-solution candidates at low surrogate-MSE (Band-A territory if
s014 Spearman 0.9952 holds):**

| seed | mse | q0_err | ω_dir_err | ω_mag_err |
|---|---|---|---|---|
| 13 | 0.0045 | 153° | 96° | -2.3% |
| 41 | 0.0059 | 173° | 122° | -1.4% |
| 41 | 0.0099 | 126° | 125° | +2.9% |
| 79 | 0.0107 | 83° | 79° | -4.7% |
| 91 | 0.083 | 127° | 73° | +0.2% |

These are surrogate-Band-A candidates at q0 80-173° from truth +
ω-dir 73-125° from truth. Multi-solution attractors structurally similar
to s014b's class_2/class_3 clusters (low-rotation LC under-determination
of (q0, ω)) — but found here with **high-rotation seeds 41/91/79 too**.
Hi-fi ρ-band-validation deferred.

## Why this matters

Two distinct conclusions:

1. **Closes the C-class hypothesis.** ω-mag prior — even with PERFECT
   value (seed 28: 0.01% error) — does NOT bridge the q0 + ω-dir search
   gap at N=64 fresh Sobol density. The truth-ω-dir tube (~1° per s003)
   is too small to be hit by random Sobol on S². Either (a) ω-dir grid
   stratification, or (b) higher density per harvest value (N=512+),
   or (c) different architecture entirely.

2. **Reframes the "failure" via multi-solution acceptance.** The cohort
   selector returns candidates with surrogate MSE 0.005-0.083 on seeds
   13 / 41 / 79 / 91 — looks like Band A. If hi-fi ρ-band confirms,
   these are **valid answers under the survey's multi-solution
   acceptance philosophy**, just not geometric truth. The user's earlier
   confirmation ("rho<4 is still B band correct? If yes, then yes I am
   very happy with rho<4") makes this reframe load-bearing.

## Numbers

960 LM runs, 76.6 min wall, 4.79 s per IC mean (faster than s015 because
5-DOF LM finishes faster — fewer iter to exhaust). Top harvested ω-mag
values + per-seed yield in `results/s016c_prime/summary.json`.

## Artefacts

- `experiments/s016c_prime_fresh_sobol_fixed_omag.py` — 5-DOF + harvest script
- `results/s016c_prime/{runs.npz, summary.json, run.log}`

## Out of scope

- Hi-fi ρ-band-validation of the 5 multi-solution candidates above.
  **Critical pending step before declaring C' fully refuted.** Estimated
  ~7-10 hi-fi renders × 75 s = ~10 min wall (single-process) or ~3 min
  Pool(4).
- Higher density per harvest (N=512+). Would test whether ω-dir tube is
  hittable at higher Sobol density. ~5x compute of s016c_prime per
  density step.
- Production cohort scan (seeds beyond the 8 pilot).

## Cross-references

- `s015_joint_q0_omega_pilot.md` — source of harvested ω-mag values.
- `s016_omega_mag_frozen_polish.md` — flawed first-pass C test.
- `s014b_n_rotations_analysis.md` — original multi-solution attractor
  classification.
- `concepts/known_pathologies_to_revalidate.md` — multi-solution at
  cohort scale, now broadened beyond seed 10.
- `concepts/observational_indistinguishability.md` — the philosophical
  basis for accepting multi-solution candidates.
