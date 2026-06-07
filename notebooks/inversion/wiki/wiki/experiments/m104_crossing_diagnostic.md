---
title: "m104 — Peak Crossing Geometry Diagnostic"
type: experiment
sources:
  - "data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz"
related:
  - "[[brightness_surface_path_matching]]"
  - "[[crossing-geometry-scoring]]"
  - "[[alignment-cost]]"
  - "[[candidate-selection]]"
  - "[[m096_exp1_oracle_grid]]"
  - "[[omega-magnitude-estimation]]"
created: 2026-04-12
updated: 2026-04-16
confidence: high
---

# m104 — Peak Crossing Geometry Diagnostic

Validation of the brightness surface path matching concept: can body-frame PAB crossing velocity at LC peaks extract omega components?

## Hypothesis

Body-frame PAB crossing velocity at LC peaks can extract omega_body components with <=30 deg direction error and <=30% speed error, enabling 10-100x search space reduction for omega direction.

## Method

Read m046_trajectories.npz (100 seeds x 500 epochs, 7.2s sampling), compute crossing velocities at all 2969 peaks, compare extracted Omega_L_perp to true Omega_L = R(t) * omega_body(t). Also measure peak FWHM from observed LC to estimate crossing speed from data.

## Critical Kinematic Correction

The concept page ([[brightness_surface_path_matching]]) had a fundamental error. It claimed:

```
dp_B/dt = omega_body x p_B
```

The correct equation is:

```
dp_B/dt = Omega_L x p_B    where Omega_L = R(t) * omega_body(t)
```

The concept confused omega_body (body-frame angular velocity, from Euler's equations) with Omega_L (the "left" angular velocity, Rdot * R^T). Since Omega_L depends on both the attitude R(t) AND angular velocity omega_body(t), you CANNOT extract omega_0 from peak crossing geometry alone -- you would also need to know the attitude, which is the other unknown.

## Results

### Phase A: Kinematic extraction (known attitude)

Using the true attitude R(t) to compute Omega_L, then extracting the perpendicular component at peaks:

- 2969 peaks across 100 seeds
- Direction error: median=0.13 deg, P25=0.06 deg, P75=0.20 deg
- Magnitude error: median=0.48%, P25=0.29%, P75=0.69%
- 99% of peaks below 1 deg direction error, 100% below 5 deg

**Conclusion:** Kinematic extraction is essentially exact. NOT a limiting factor.

### Phase B: Peak shape measurement (from observed LC)

Estimating crossing speed from peak FWHM in the discrete light curve:

- 636 bright peaks (mag < 9) with measurable FWHM
- FWHM range: [7.3, 123.2] seconds
- FWHM x crossing_speed product: mean=11.7 deg, std=5.8 deg, CV=50%
- Crossing speed estimated from FWHM: 27% median error
- Only 36% of peaks within 20% error

**Conclusion:** Discrete 7.2s sampling with 3-5 points per peak is too coarse. Peak width measurement is the bottleneck.

### Phase C: Scoring value

Testing whether FWHM x speed consistency can discriminate correct candidates:

- 79/100 seeds have >=3 measurable peaks
- Product CV per seed: median=25%, P25=19%, P75=42%

**Conclusion:** The FWHM x v consistency metric is too noisy to reliably discriminate candidates.

## Verdict

**Hypothesis REFUTED.**

1. The kinematic extraction is perfect when attitude is known (Phase A), but the observable (peak FWHM from discrete LC) is too noisy (Phase B, C).
2. The concept also had a fundamental kinematic error (Omega_L vs omega_body) which means direct omega_0 extraction requires knowing the attitude -- the other unknown.
3. The claimed 10-100x search space reduction is not supported.

**Recommendation:** Do NOT pursue this approach for grid search replacement or augmentation. The selection failure problem is better addressed by improving existing metrics (see [[candidate-selection]], [[hybrid-selection]]).
