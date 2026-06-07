---
title: "s016 — ω-mag-frozen 5-DOF polish from harvested s015 ICs (Option C, first pass)"
type: experiment
sources:
  - "results/s016c/runs.npz"
  - "results/s016c/summary.json"
  - "results/s016c/run.log"
related:
  - "[[s003_landscape_vs_omega]]"
  - "[[s015_joint_q0_omega_pilot]]"
  - "[[s016c_prime_fresh_sobol_fixed_omag]]"  # follow-on
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## TL;DR

**Refuted, but for an unexpected reason.** S016-C tested the hypothesis
"ω-mag-correct ICs are close enough to truth-ω tube for 5-DOF LM (q0 +
ω-dir) to bridge." 0/7 seeds with starts reached loose basin. **But LM
didn't move at all from most starts** — 30/41 ICs had q0/ω-dir changes
< 1°. The s015 starting points were *already converged local minima*;
restarting LM with frozen ω-mag (a 5-DOF reparameterization of the same
residual) just re-lands at the same minimum. Gradient signal hadn't
suddenly appeared.

**The C hypothesis was tested incorrectly here.** The actual hypothesis
("ω-mag is sufficient prior") requires fresh ICs at fixed ω-mag, not
re-optimization from already-converged starts. Re-tested correctly in
s016c_prime, which falsifies the hypothesis structurally.

## What

S015 cohort diagnostic surfaced 41 ICs (across 7/8 seeds) where
LM had landed within 10% of truth ω-magnitude — even though q0 and
ω-direction were wrong. S016-C asked: take those 41 ICs as starting
points, freeze ω-mag at its current value, polish q0 + ω-dir with 5-DOF
LM. If "ω-mag-right" is sufficient bridge, some ICs reach truth basin.

## How

5-DOF parameterization: x = (δθ_3, theta, phi). δθ rotates around
q0_seed; (theta, phi) are spherical coords of ω-direction; ω-magnitude
is held at the harvested value.

41 starts (per-seed: 6=1, 13=1, 28=9, 41=7, 42=0, 49=12, 79=1, 91=10).
Source: `results/s015_diagnostic_n64/runs.npz`, mask
`|omega_mag_err_pct| < 10`.

Polish: scipy `least_squares(method='lm', max_nfev=200)`, residuals =
surrogate full-LC residuals.

Pool(8), BLAS=1 + torch_threads=1.

## Result

**0/7 seeds in-basin (loose). 128 s wall.**

**The interesting finding is in the deltas.** Initial vs final state:

| pattern | n ICs |
|---|---|
| q0 / ω-dir changes < 1° (LM didn't move) | 30/41 |
| Small movement (1-20° to other local minima) | 11/41 |
| Movement TOWARD truth | 0/41 |

The two largest moves (seed 41 IC#26: -11° q0, -14° ω_dir; seed 41 IC#42:
+20° q0) were LM finding marginally-better local minima, not approaching
truth.

**Why LM didn't move:** the s015 starting points were already converged
local minima of the surrogate-MSE landscape. Restarting LM with a tighter
parameterization (frozen ω-mag) just re-lands at the same minimum. The
gradient signal hasn't appeared by removing one DOF.

## Why this matters

Two distinct findings:

1. **Methodological lesson:** "harvest converged ICs from a previous run
   and re-polish them with a constrained variant" is a vacuous test if
   the prior run's LM was at true minima (not stalls). For the C
   hypothesis to be tested fairly, fresh starts at fixed-ω-mag are
   required (s016c_prime).

2. **Positive content from the negative:** the s015 LM in the diagnostic
   was finding *genuine* local minima, not stall points. The joint
   surrogate-MSE landscape outside the truth-ω tube is **densely
   populated with deep wrong-answer basins** — consistent with s003's
   "outside the tube the landscape is noisy-multi-basin."

## Numbers

128 s wall. Per-seed deltas in `results/s016c/runs.npz`. No new physics
— just confirms LM convergence from s015.

## Artefacts

- `experiments/s016_omega_mag_frozen_polish.py` — 5-DOF script
- `results/s016c/{runs.npz, summary.json, run.log}`

## Out of scope

- Higher max_nfev (would not help: the gradient is genuinely zero at
  these landings, not just slow to follow).
- Multi-restart from perturbations of the harvested ICs (functionally
  equivalent to s016c_prime fresh-Sobol).

## Cross-references

- `s015_joint_q0_omega_pilot.md` — source of the harvested ICs.
- `s016c_prime_fresh_sobol_fixed_omag.md` — proper C-hypothesis test.
- `s003_landscape_vs_omega.md` — original truth-ω tube measurement.
