---
title: "s055b — L̂ (inertial angular momentum direction) regression from LC features"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s055b_lhat_lc_regression.py
  - notebooks/inversion/survey/lib/lc_features.py
  - notebooks/inversion/survey/lib/hifi_render.py (cached inertia tensor)
related:
  - s055a — pol_diam regression (COARSE_USEFUL @ 25% holdout MAPE)
  - s053 — cohort polhode survey (pol_diam ρ=−0.94 vs basin width)
  - s048c+ — cloud-evolution viewer (visual surfacing of polhode framework)
  - s008 — LC-feature regression for ω-mag (closed LC-only |ω| priors)
  - feedback_holdout_validation.md
created: 2026-05-07
updated: 2026-05-07
confidence: high (negative result)
status: WEAK — L̂ direction NOT recoverable from s008+PAB feature set; 74° holdout median, beats random by only 15°. Operationally not useful for ω-direction grid pruning at this feature granularity.
---

## TL;DR

**L̂ direction in J2000 is NOT operationally recoverable from the s008 28-feature LC set augmented by 6 simple geometric features (PAB at brightest peak + mean PAB).** Best holdout median angular error = 74.3° (linear); random-on-sphere baseline = 89.2°; the regression beats random by ~15° but only 0% of holdout seeds land within 20° and 15% within 45°. Decision: **WEAK**. Implication: ω-direction grid pruning by L̂-cone priors is not unlocked at this feature granularity. The user's question — "can we predict ω-direction components?" — gets a clean negative answer for this feature set. Possible escalations (PAB at top-K peaks, peak-time × PAB triangulation) remain untested but require non-trivial feature engineering.

## What

The polhode framework reparameterises ω as `(|L|, polhode_label, polhode_phase)` instead of free `(ω_x, ω_y, ω_z)`. **L̂ is a single 3-vector per seed**, conserved under torque-free dynamics, and if recoverable from LC it would collapse the inertial-frame ω-direction grid from full unit sphere (4π sr) to a small precession disc — at ±20° L̂ uncertainty, ω-direction grid shrinks ~1/40; at ±10° ~1/160.

This experiment tests whether L̂ falls out of the LC features at all.

## How

**Truth**: L̂_J2000 = `R(q0)ᵀ · I · ω0_body / |...|`, where `R(q0)` is the passive J2000→body rotation per `src/dynamics/attitude_propagator.py:103` convention and `I` is the cached post-fix inertia tensor from `lib.hifi_render._build_model()` (eigvals 7749, 37985, 38306 kg·m²).

**Features (34 total)**:
- 28 s008 LC features from `lib.lc_features` (rotation-invariant scalar properties of mag time series)
- 6 geometric: `pab_brightest[3]` — PAB unit vector at LC's brightest epoch (argmin mag); `pab_mean[3]` — mean PAB direction over the window. Both in J2000.

**Pipeline**: 4 models (linear, ridge, RF (300×8), GBR(300×4) wrapped in MultiOutputRegressor) × 3-output regression × {LOO on cohort 0..99, fit-cohort + predict holdout 100..119}. Predictions are renormalised onto the unit sphere at inference.

**Metric**: angular error in degrees between predicted unit L̂ and truth unit L̂. Compared against a 1000-trial random-on-sphere baseline.

## Result

| Model  | Cohort LOO median | Cohort <20° | Holdout median | Holdout <20° | Holdout <45° |
|--------|-------------------|-------------|----------------|--------------|--------------|
| linear | 89.7°             | 2%          | **74.3°**      | 0%           | 15%          |
| ridge  | 91.2°             | 4%          | 103.2°         | 5%           | 30%          |
| rf     | 95.9°             | 5%          | 77.4°          | 0%           | 25%          |
| gbr    | 72.9°             | 5%          | 77.8°          | 0%           | 25%          |

**Random baseline**: cohort med 90.1°, holdout med 89.2°, frac<20° ≈ 3%.

The best model (linear) on holdout beats random by 15° on the median — **statistically meaningful but operationally far short** of the 20° gate. Fewer than 30% of holdout seeds fall within 45° even in the best case.

The truth L̂_J2000 distribution across cohort spans the full unit sphere (each seed has random q0 in J2000 → essentially uniform L̂). The s008 features are inertial-rotation-invariant (LC magnitude time series doesn't directly encode J2000 direction); the 6 PAB features encode the seed's observation geometry but don't, on their own, triangulate L̂.

The ~15° beat over random plausibly comes from a weak correlation: cohorts with similar LC modulation depth have similar |cos(L̂, PAB)| (since modulation depth scales ~ sin(L̂-PAB angle)), and the regressor picks up some of that. But |cos(L̂, PAB)| only fixes a cone — it doesn't pick out direction within the cone.

## Why this matters

**ω-direction grid pruning by an L̂-cone prior is NOT unlocked at this feature granularity.** Combined with s055c's parallel finding (|cos(L̂, PAB)| is also not LC-recoverable above the cohort-mean baseline), the conclusion is that the s008+PAB-mean feature set extracts ω-MAGNITUDE info (pol_diam ✓ at 25%, |ω| ✓ at 16% LOO MAPE on s008) but essentially no ω-DIRECTION info.

The polhode framework remains real and useful — `pol_diam` adaptive bracket is operational per s055a — but it speaks to the |ω| basin width, not to where in ω-direction phase space the search should focus. **Operational architecture should NOT count on L̂ priors from LC features at this granularity.**

This result narrows the path forward:

- **Don't pursue more L̂ direction regressions with the same feature set.** Closed.
- **Possible escalations (untested)**: (a) PAB direction at top-K LC peaks (5–10 peaks, 15–30 features), encoding peak-time triangulation; (b) explicit body-frame inversion via single bright peak (s018b territory) → infer R(q(t_peak)) → propagate to L̂. Both are non-trivial feature engineering.
- **Or accept the limitation**: ω-direction grid stays uniform (or polhode-tangent-constrained per s053 mechanism), with adaptive bracket on |ω| via pol_diam.

## Numbers

- n_cohort = 100, n_holdout = 20
- n_features = 34 (28 LC + 6 geometric)
- I_body eigvals = 7749 / 37985 / 38306 kg·m² (I_a, I_b, I_c)
- Wall: 67s
- Decision: **WEAK**. Best holdout median 74.3°, vs 89.2° random baseline (15° beat); 0% within 20°, 15% within 45°.
- Holdout < 20° gate (operational): NOT MET.
- Holdout < 45° gate (coarse-useful): NOT MET (best ~30% with ridge).

## Artefacts

- `notebooks/inversion/survey/experiments/s055b_lhat_lc_regression.py`
- `notebooks/inversion/survey/results/s055b_lhat_lc_regression/`:
  - `features.npz` — `X[120,34]`, `y_lhat[120,3]`, seeds, is_cohort_mask
  - `regression.npz` — per-model LOO + holdout predictions and errors
  - `summary.json` — decision-grade scalars + random baseline
  - `angular_error_hist.png` — 4-panel histograms with random reference
  - `predicted_vs_truth_lhat.png` — 3D scatter of predicted vs truth on unit sphere

## Out of scope

- PAB at top-K LC peaks (richer geometric features). Worth testing only if the cohort-scale yield bottleneck is later traced specifically to ω-direction grid density.
- Per-peak body-frame inversion + propagation (s018b ↔ peak-times-to-L̂ chain). Major feature engineering — not justified by the operational benefit until pol_diam-only architecture saturates.
- Predicting L̂ in PAB-frame instead of J2000. Won't help: a uniform-on-sphere distribution rotated to PAB-frame is still uniform on sphere; the issue is the feature set, not the target frame.
- Polhode period τ_p regression. Could be tested but s055a's feature importance already includes spectral features; the modest pol_diam MAPE suggests τ_p isn't strongly distinguishable beyond what pol_diam captures.

## Cross-references

- s055a — `s055a_pol_diam_lc_regression.{py,md}`: pol_diam recoverable @ 25% holdout MAPE; basis for adaptive bracket on |ω|.
- s055c — `s055c_aux_priors.{py,md}`: |cos(L̂, PAB)| and polhode-binary-class also closed weak.
- s053 — `s053_cohort_polhode_survey.{py,md}`: pol_diam is the load-bearing basin-width predictor.
- s008 — `s008_lc_feature_regression_omega.{py,md}`: closed LC-only |ω| priors at LOO MAPE 16.4%.
- `feedback_lc_spectral_omega_prior_dead.md`: do not re-propose LC-only ω-mag priors.
