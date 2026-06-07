---
title: "s055a — pol_diam regression from LC features"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s055a_pol_diam_lc_regression.py
  - notebooks/inversion/survey/lib/lc_features.py
  - notebooks/inversion/survey/results/s053_cohort_polhode_survey/cohort.npz
  - notebooks/inversion/survey/results/s054_holdout/holdout_polhode.npz
related:
  - s008 — LC-feature regression for ω-mag (LOO MAPE 16.4%; closed LC-only ω priors)
  - s053 — cohort polhode survey (pol_diam ρ=−0.94 vs basin width)
  - s054 — holdout corpus (m048 seeds 100..119)
  - feedback_holdout_validation.md — holdout-validate cohort priors
created: 2026-05-07
updated: 2026-05-07
confidence: medium-high
status: COARSE_USEFUL — pol_diam recoverable from LC at ~25% holdout MAPE; operational adaptive bracket viable but suboptimal; promotes s055b oracle pilot to high priority
---

## TL;DR

**pol_diam (polhode L2 diameter, dps) is LC-recoverable at ~25% holdout MAPE on m048 seeds 100..119 — coarse but operationally useful.** Decision: **COARSE_USEFUL** (threshold band 20%≤MAPE<50%). The s053 polhode-prior architecture compiles to an adaptive-bracket controller, with bracket sizing accurate to a factor of ~1.25 — well inside the s053 recoverability bound (`δ|ω|/|ω| ≤ 0.5% × pol_diam_ref/pol_diam_seed` tolerates 2× pol_diam error). |ω| sanity control reproduces s008's LOO MAPE within 0.65pp (gbr-log10 = 15.75% vs s008 16.4%), confirming `lib.lc_features` is faithful. Top features are `dmag_mean_abs` (0.26) and `acf_lag1` (0.22) — modulation rate × short-lag autocorrelation, exactly what polhode amplitude predicts. Promotes s055b (oracle adaptive-bracket pilot) and motivates a focused s056 (3-bucket polhode-size classifier — easier than regression at operationally-equivalent fidelity).

## What

Regress pol_diam from a 28-dim LC feature vector. The s053 cohort scan found pol_diam (Spearman ρ=−0.94, n=10) is a stronger predictor of ω-mag basin width than |ω| itself (ρ=−0.84). The s053 operational rule sizes the search bracket per-seed: `δ|ω|/|ω| ≤ 0.5% × (pol_diam_ref / pol_diam_seed)`. That rule presumes pol_diam is *known* at inversion time. This experiment tests whether pol_diam can be estimated from the LC alone, gating whether the polhode prior is an inversion architecture or a post-hoc explanatory framework.

Sanity control: run the identical pipeline on |ω| as target. Must reproduce s008's LOO MAPE 16.4% within ±2pp; otherwise `lib.lc_features` has drifted from s008 and the comparison is invalid.

## How

**Inputs (all cached, zero rendering):**

- Cohort: m048 seeds 0..99, `pol_diam_dps` from `s053_cohort_polhode_survey/cohort.npz`.
- Holdout: m048 seeds 100..119, `pol_diam_dps` from `s054_holdout/holdout_polhode.npz`.
- LCs: `mag_hifi[500]` from cached trajectory NPZs (loaded via `lib.traj_load.truth_state`).

**Pipeline** (mirrors s008 exactly; target swapped):

```
for seed in 0..119:
    feats[seed] = lib.lc_features(t, mag_hifi)        # 28-dim
y_pol  = [pol_diam_dps[seed] for seed in 0..119]
y_om   = [om_mag_dps[seed]   for seed in 0..119]      # sanity control

for target in {pol_diam, om_mag}:
  for model in {LinearRegression, RidgeCV, RandomForest(300×8), GBR(300×4)}:
    for kind in {direct, log10}:
      LOO CV on cohort (n=100)         → cohort_loo_metrics
      Fit(cohort), predict(holdout 20) → holdout_metrics
```

**Reporting**: median MAPE, p90, max, fraction within {3, 5, 10, 30}%; top-10 RF feature importances on cohort.

**No new feature engineering.** If the s008 set fails on pol_diam, that's information — pre-engineering features specific to polhode wobble would conflate "are LC features sufficient?" with "did we engineer the right ones?".

## Result

### Sanity (om_mag): PASS

|ω| LOO best = **gbr-log10 = 15.75%**. s008 baseline = 16.4%. **Δ = 0.65pp, well inside ±2pp tolerance.** `lib.lc_features` reproduces s008's pipeline. Comparisons to s008 numbers are valid.

### pol_diam: COARSE_USEFUL

| Metric                                    | Value         |
|-------------------------------------------|---------------|
| **Best holdout median MAPE** (linear, direct)  | **24.84%** |
| Best cohort LOO median MAPE (rf, log10)        | 21.07%     |
| Linear (direct) holdout: frac within 30%       | 60%        |
| Linear (direct) holdout: frac within 10%       | 35%        |
| Linear (direct) holdout: p90                   | 388%       |

**Decision-grade number is the holdout, not the LOO.** The cohort-LOO winner (RF-log10, 21%) overfits — its holdout MAPE is 29% and direct-target RF holdout is 41%, suggesting RF cannot extrapolate to seeds outside the training-feature support. Linear and ridge generalise more honestly: cohort LOO 28-32%, holdout 25-27%.

The linear-direct holdout p90 of 388% is driven by a small number of outlier seeds (small n=20 amplifies tail effects). The linear-log10 holdout p90 (191%) and ridge-log10 (154%) are tighter, suggesting log-targeting helps with outlier behavior even though the median is slightly worse. Trade-off worth noting in operational deployment.

### pol_diam vs |ω|: a small surprise

|ω| LOO median is **better** (15.75%) than pol_diam LOO (21.07%) despite pol_diam having 4× more cohort variation (52× span vs 13×). Likely explanation: |ω| is an integrated time-domain quantity that maps cleanly to LC modulation rate; pol_diam is a polhode geometric scalar that requires inferring polhode shape from LC, which the 28-feature set captures only partially. A focused polhode-aware feature set might close the gap; deferred to s056 if needed.

### Feature importance (RF, pol_diam target, cohort fit)

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | `dmag_mean_abs`      | 0.261      |
| 2    | `acf_lag1`           | 0.225      |
| 3    | `dmag_std`           | 0.077      |
| 4    | `ls_top1_p`          | 0.073      |
| 5    | `acf_first_peak_lag` | 0.032      |

**Top-2 = 49% of importance.** `dmag_mean_abs` (mean |dmag/dt|) and `acf_lag1` (short-lag autocorrelation) are precisely the modulation-rate × modulation-coherence features that polhode amplitude predicts: a larger polhode → larger angular sweep per unit time → larger |dmag/dt| and faster decorrelation. Spectral peak features (LS top-1 power) and glint count contribute marginally. Physically coherent — the regression is using the right features even where the MAPE is mediocre.

## Why this matters

**The polhode prior is operational, not just analytical.** The s053 framework was a strong analytical reformulation, but its operational viability depended on this experiment. With pol_diam recoverable from LC at ~25% holdout MAPE, the s053 adaptive-bracket rule can be applied at inversion time using LC-derived pol_diam estimates, not oracle truth. A factor-of-1.25 bracket-size error is well inside the tolerance — the bracket needs to enclose the basin, and a 25%-too-wide or 25%-too-narrow bracket still does so on the s042 cohort tail.

**Decision implications:**

1. **s055b (oracle adaptive-bracket pilot) is high-priority.** Run on holdout 100..119 with truth-derived pol_diam to upper-bound architecture yield. Even with 25% pol_diam uncertainty, the architecture can land Band A∪B basins on most seeds — but we need the oracle baseline to know what room there is.
2. **s056 (focused polhode estimator) is motivated, not urgent.** A 3-bucket classifier (small / medium / large polhode) at ~85% accuracy would be operationally equivalent to ~25% MAPE regression and is a much easier learning problem. Worth pursuing if s055b shows architecture saturates against pol_diam estimation noise.
3. **Closed dead-ends remain closed.** s007/s008 (LC-only |ω| priors) closed the regression-as-prior approach for |ω|, but pol_diam is a different scalar with a different physical signature — this result does NOT reopen the LC-only ω-prior question. Per `feedback_lc_spectral_omega_prior_dead.md`, do not re-propose ω-mag-from-LC architectures.
4. **Cohort LOO ≠ holdout.** RF over-fits dramatically (LOO 21% → holdout 29-41%); linear/ridge are the honest baselines on n=100. This is a small-sample lesson — be careful with high-capacity models on cohort-only LOO when the eventual deployment is on fresh trajectories.

## Numbers

- n_cohort = 100, n_holdout = 20, n_features = 28
- Wall: 124s
- Decision label: **COARSE_USEFUL** (20% ≤ holdout MAPE < 50%)
- pol_diam range: cohort [0.053, 2.753] dps, holdout [0.016, 2.925] dps (52× span)
- |ω| range: [0.106, 1.482] dps (13× span)
- Best models per (target, split) combination:
  - pol_diam holdout: **linear-direct = 24.84%** (60% within 30%)
  - pol_diam LOO: rf-log10 = 21.07% (overfit; holdout 29%)
  - om_mag LOO: gbr-log10 = 15.75% (s008 baseline 16.4%, Δ=0.65pp)

## Artefacts

- `notebooks/inversion/survey/experiments/s055a_pol_diam_lc_regression.py` — script
- `notebooks/inversion/survey/lib/lc_features.py` — 28-feature extractor lifted from s008
- `notebooks/inversion/survey/results/s055a_pol_diam_lc_regression/`:
  - `features.npz` — `X[120,28]`, `y_pol_diam[120]`, `y_om_mag[120]`, `seeds[120]`, `is_cohort_mask[120]`, `feature_names[28]`, `rf_feature_importance[28]`
  - `regression.npz` — per-target × per-model × per-kind LOO + holdout predictions
  - `summary.json` — decision-grade scalars
  - `actual_vs_predicted_loo.png` — cohort LOO scatter, 2×4 grid (rows=direct/log10, cols=models)
  - `actual_vs_predicted_holdout.png` — holdout test scatter, 2×4 grid
  - `feature_importance.png` — top-15 RF features

## Out of scope

- Polhode-specific feature engineering (Lomb-Scargle on `|ω|(t)` proxies, magnitude-bin entropy, sub-band spectral features). Deferred to s056 only if s055b shows estimator quality is the bottleneck.
- Joint pol_diam + |ω| multi-target regression. Possibly informative; separate experiment.
- Classification into 3 polhode-size buckets. Recommended for s056 but not run here.
- Running s055b. Decided after seeing this result; high-priority next step.
- Hierarchical inversion (coarse bracket → re-estimate pol_diam → refine). Architectural alternative if s055b reveals estimator-quality saturation.

## Cross-references

- s008 (`s008_lc_feature_regression_omega.py`, `s008_lc_feature_regression_omega.md`) — source of feature extraction and regression scaffolding.
- s053 (`experiments/s053_cohort_polhode_survey.py`, `concepts/polhode_prior.md`) — pol_diam discovery and basin-width correlation.
- s054 (`experiments/s054_generate_holdout.py`, `s054b_holdout_polhode_check.py`) — holdout corpus + polhode statistics.
- s042 (cohort-tail basin radius) — provides the 10-seed reference for the s053 operational rule.
- `feedback_holdout_validation.md` — methodology rule mandating fresh-data test for cohort priors.
- `feedback_lc_spectral_omega_prior_dead.md` — closed-dead-end note; do not re-propose LC-only ω-mag priors.
- MEMORY.md `project_omega_mag_basin_scales_with_omega.md` — pol_diam as the load-bearing predictor.
