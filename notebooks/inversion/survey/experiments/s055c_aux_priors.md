---
title: "s055c — Auxiliary scalar priors from LC: |cos(L̂,PAB)| and polhode-topology class"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s055c_aux_priors.py
  - notebooks/inversion/survey/lib/lc_features.py
  - notebooks/inversion/survey/lib/hifi_render.py
related:
  - s055a — pol_diam regression (COARSE_USEFUL @ 25% holdout MAPE)
  - s055b — L̂ direction regression (WEAK @ 74° holdout median)
  - s053 — cohort polhode survey (pol_diam ρ=−0.94, D NOT correlated with basin width)
created: 2026-05-07
updated: 2026-05-07
confidence: high (negative result)
status: BOTH WEAK — |cos(L̂,PAB)| is worse than cohort-mean baseline; polhode-binary-class matches majority baseline (acc) and is only modestly above-chance in balanced accuracy. LC features carry only ω-MAGNITUDE info (pol_diam ✓), no usable ω-DIRECTION info at this granularity.
---

## TL;DR

After s055b found L̂ direction operationally unrecoverable, this experiment tested two scalar auxiliary priors: `|cos(L̂, PAB_mean)|` (regression in [0,1]) and `polhode_class ∈ {I_a-encl D<1, I_c-encl D≥1}` (binary classification). **Both are essentially uninformative**: |cos| holdout RMSE 0.299 vs cohort-mean baseline 0.285 — the regressor is *worse* than predicting the constant mean. Polhode-binary best holdout accuracy 0.80 (ridge) vs majority-class baseline 0.85 — also worse on raw accuracy; balanced accuracy 0.745 catches some minority-class seeds but at high false-positive cost. **Decision: WEAK on both signals.** With s055a (pol_diam ✓) + s055b (L̂ ✗) + s055c (cos ✗, class ✗), the LC feature set is now characterised: it carries ω-MAGNITUDE info (pol_diam, |ω|) but essentially no usable ω-DIRECTION info at this feature granularity. The user's hypothesis — that LC has rich direction-related signal we're not using — receives a clean negative answer for the s008-style features.

## What

After s055b found L̂ direction in J2000 was operationally unrecoverable (74° holdout median), two simpler scalar reductions of L̂'s direction info were proposed as easier alternatives:

1. **|cos(L̂, PAB_mean)|** — a scalar in [0,1] capturing the L̂-PAB angle. Modulation depth in the LC scales monotonically with sin(L̂-PAB angle), so this scalar should — physically — fall out of LC modulation features. Sign degeneracy (PAB and -PAB give same modulation) is removed by taking absolute value.

2. **polhode_class binary** — `D = 2T·I_b/|L|²` with cut at D=1: 0 (I_a-enclosing) vs 1 (I_c-enclosing). m048 cohort distribution under this rule is ~13/87 (matches MEMORY.md `s053`), holdout 3/17. Different topology classes have structurally different ω̂(t) precession patterns.

Both targets are computed analytically from cached truth (q0, ω0, I) — no quaternion-trajectory derivation needed.

## How

- **Train cohort**: m048 seeds 0..99 (n=100). **Test holdout**: m048 seeds 100..119 (n=20).
- **Features**: 28 s008 features (LC-only). PAB direction not needed since both targets are rotation-invariant scalars.
- **|cos| regression**: 4 models (LinearRegression, RidgeCV, RandomForest 300×8, GradientBoosting 300×4). Direct target. Predictions clipped to [0,1]. Metrics: RMSE, MAE, p90 abs err, frac within {0.1, 0.2, 0.3}.
- **polhode-binary classification**: 4 classifiers (LogisticRegression class-balanced, RidgeClassifier class-balanced, RandomForest class-balanced 300×8, HistGradientBoostingClassifier class-balanced). Metrics: raw accuracy, balanced accuracy, confusion matrix.
- **Baselines**: |cos| → predict cohort mean (0.490). polhode-binary → predict majority class (I_c, 87% cohort).

## Result

### |cos(L̂, PAB)| regression — WEAK (worse than baseline)

| Model  | Cohort LOO RMSE | Holdout RMSE | Holdout <0.1 | Holdout <0.2 |
|--------|------------------|---------------|--------------|--------------|
| linear | 0.343            | 0.371         | 20%          | 35%          |
| ridge  | 0.291            | **0.299**     | 25%          | 50%          |
| rf     | 0.268            | 0.323         | 20%          | 45%          |
| gbr    | 0.271            | 0.369         | 10%          | 35%          |
| **cohort-mean baseline (0.490)** | **0.280** | **0.285** | — | — |

**The cohort-mean baseline is BETTER than every model on the holdout RMSE.** Ridge is only 0.014 above baseline RMSE, with the variance-explained signal indistinguishable from noise. The scatter plot (cos_scatter.png) confirms predictions cluster near the cohort mean regardless of truth — zero diagonal structure.

### Polhode-class (binary) classification — WEAK / MARGINAL

Cohort split: 13 I_a / 87 I_c. Holdout split: 3 I_a / 17 I_c.

| Model    | Cohort LOO acc | Cohort LOO bacc | Holdout acc | Holdout bacc |
|----------|----------------|-----------------|-------------|--------------|
| logreg   | 0.46           | 0.43            | 0.75        | 0.72         |
| ridge    | 0.45           | 0.39            | **0.80**    | **0.745**    |
| rf       | 0.87           | 0.50            | 0.85        | 0.50         |
| hgb      | 0.81           | 0.47            | 0.80        | 0.47         |
| **majority baseline (predict I_c)** | **0.87** | 0.50 | **0.85** | 0.50 |

**Raw accuracy: every model is ≤ majority baseline on holdout.** The class-balanced LogReg/Ridge variants achieve balanced accuracy 0.72–0.745 — meaningfully above 0.50 chance — by trading raw accuracy for minority-class detection (catching ~2/3 of I_a seeds while misclassifying ~3/17 I_c seeds). This is potentially useful for flagging probable-I_a seeds at the cost of false positives, but **at 13% I_a base rate the operational gain over a uniform-ω-direction grid is marginal**: prune-by-classifier-on-an-I_c-flagged-seed only removes ~13% of grid (the I_a-leading subspace), and on the rare I_a-flagged seed has a 1/3 risk of throwing out the truth.

RF and HGB default to "predict I_c always" (matches majority baseline) — no signal beyond the class imbalance.

## Why this matters

**This result, combined with s055a + s055b, completes the characterisation of what the s008-style LC feature set can and cannot extract:**

| LC-derivable target  | Result            | Operational? |
|----------------------|-------------------|--------------|
| `|ω|` (s008)         | LOO 16.4%         | Closed dead-end as direct prior; useful as bracket center |
| `pol_diam` (s055a)   | Holdout 25%       | YES — adaptive bracket on |ω| via s053 rule |
| `L̂` direction (s055b)| Holdout 74° median| NO — only 15° better than random |
| `|cos(L̂, PAB)|` (s055c) | Holdout RMSE 0.299 vs 0.285 baseline | NO |
| Polhode-binary (s055c) | Holdout acc ≤ 0.85 baseline | NO (marginal balanced-acc only) |

**The LC + simple-PAB feature set captures ω-MAGNITUDE info (rotation-invariant scalars) but essentially no ω-DIRECTION info.** The user's directional question is answered: at this feature granularity, ω-direction is not LC-recoverable.

**Implications for architecture:**

1. **ω-direction grid pruning via LC-derived priors is not unlocked here.** s055b's full L̂ failed; s055c's scalar reductions also fail. Operational ω-direction handling stays uniform (or polhode-tangent-constrained per the s053 dynamics-admissible reframe).
2. **The polhode-prior architecture is operational only on |ω|, not on direction.** s055a + s053 give adaptive bracket density per seed; s055b/c rules out a complementary direction prior at this feature granularity.
3. **Possible escalations (untested) for ω-direction**:
   - Richer geometric features: PAB at top-K LC peaks (peak-time triangulation), peak times themselves, dimming-pattern-between-peaks features. Non-trivial engineering.
   - Body-frame inversion via single bright peak (s018b territory) → R(q(t_peak)) → propagate to L̂. Hard.
   - Cascade-pool pol_diam filtering (the original s055c-as-cascade-filter proposal) — uses the polhode geometry on cascade survivors directly, doesn't require LC-direction recovery. Still viable as a post-cascade prune.
4. **Closed-now-clearly**: predicting L̂ direction or |cos(L̂, PAB)| from s008+PAB features. Don't propose this class again without new feature engineering.

## Numbers

- n_cohort = 100, n_holdout = 20, n_features = 28
- Wall: 50s
- |cos| range: cohort [0.012, 0.999]; mean 0.490
- D range: cohort + holdout [0.993, 4.791]; cohort 13 I_a / 87 I_c, holdout 3 I_a / 17 I_c
- |cos| best holdout: ridge RMSE 0.299 (baseline 0.285) → **WEAK, decision below baseline**
- polhode-binary best holdout: ridge balanced-accuracy 0.745 (acc 0.80, vs majority acc 0.85) → **MARGINAL on bacc only**

## Artefacts

- `notebooks/inversion/survey/experiments/s055c_aux_priors.py`
- `notebooks/inversion/survey/results/s055c_aux_priors/`:
  - `features.npz` — `X[120,28]`, `y_abs_cos[120]`, `y_polclass[120]`, `y_D[120]`, seeds, is_cohort_mask
  - `cos_regression.npz` — per-model LOO + holdout predictions for |cos|
  - `class_regression.npz` — per-model LOO + holdout predictions for polhode binary
  - `summary.json` — decision-grade scalars
  - `cos_scatter.png` — actual vs predicted |cos|, 2×4 grid (LOO/holdout × 4 models)
  - `class_confusion.png` — confusion matrices (4 models, holdout)

## Out of scope

- 3-class polhode topology with separatrix carve-out — first attempt produced 0 I_a seeds under stricter `D<0.95` rule (m048 cohort minimum D≈0.99). Binary cut at D=1 cleaner.
- Polhode period τ_p regression — separate experiment if motivated. Likely confounded with |ω| features already in s008.
- Recovering ω-direction in PAB-frame instead of J2000 — direction-on-sphere is direction-on-sphere; rotating the frame doesn't change recoverability.
- Bringing in body-frame ω̂(t) trajectory features — those are derivable from cached `quaternions[N,4]` but require careful finite-diff handling and don't address the fundamental issue (LC features are inertial-rotation-blind).

## Cross-references

- s055a — pol_diam (✓ COARSE_USEFUL).
- s055b — L̂ direction (✗ WEAK).
- s053 — cohort polhode survey: D not correlated with basin width but pol_diam is.
- s008 — closed LC-only |ω| priors.
- `feedback_lc_spectral_omega_prior_dead.md` — closed dead-end note.
- `feedback_holdout_validation.md` — every claim here is holdout-tested per the rule.
