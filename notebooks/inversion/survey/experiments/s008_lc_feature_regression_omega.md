---
title: "s008 — LC-feature regression for ω-magnitude prior (Q4c-iv)"
type: experiment
sources:
  - "results/s008/features.npz"
  - "results/s008/predictions.npz"
  - "results/s008/summary.json"
  - "results/s008/scatter.png"
  - "results/s008/feature_importance.png"
related:
  - "[[s007_omega_mag_peak_spacing_pilot]]"
  - "[[s003_landscape_vs_omega]]"
  - "[[s005_joint_local_descent]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s008 — LC-feature regression for ω-magnitude prior (Q4c-iv)

## TL;DR

**Decisive negative.** A 28-feature LOO-cross-validated regression model
(linear / ridge / random-forest / gradient-boosting × {direct ω, log ω}
targets) on 100 m048 truth LCs cannot predict ω-magnitude within
s005's 3-5% basin radius. Best model (gradient-boosting on log target)
achieves median LOO MAPE = **16.4%**, worse than s007's truth-aware
oracle ceiling (7.4%) and far above the 3% bar.

Even the well-sampled subcohort (≥5 rotations / 60-min LC, n=66)
gives median MAPE 13.5%. The slow-rotator subcohort (<2 rotations,
n=10) gives 22.8% with p90 91.5%. ω-magnitude is structurally
under-determined by LC features for tumbling-rigid-body trajectories.

**Implication for Q4c:** the model-aware ω prior path is closed.
Combined with s007 (cheap-spectral-prior closure) and s003 (no useful
ω-gradient outside the truth tube), **all LC-only ω-conditioning
strategies are dead**. Q4c at cohort scale must either (a) accept a
brute-force ω-mag grid + ω-dir grid + Sobol(q0) + LM polish at a
density that costs >>1 day per seed, (b) allow a fraction of the cohort
(seed-28-class tight basins) to fail, or (c) move to a hierarchical /
adaptive search architecture (basin-hopping, Bayesian optimisation, or
similar) where the structure is replaced by exploration.

---

## What

For each of 100 m048 seeds, extract a 28-dimensional feature vector
from the truth hi-fi LC (no use of truth-ω anywhere in the feature
extraction):

- **Spectral (Lomb-Scargle, 4000-pt freq grid):** top-5 peak frequencies
  and powers.
- **Spectral energy ratio:** integrated power above vs below 0.01 Hz.
- **Time-domain stats:** mag mean / std / min / max / skew / kurtosis,
  |dmag/dt| mean and std.
- **ACF:** lag-1, lag-5, lag-20 autocorrelation, plus first non-zero
  ACF peak lag.
- **Glint stats:** count of bright local minima below the 10th-percentile
  brightness threshold, and mean / std of inter-glint spacing.

Train four regression models (linear, ridge with log-spaced α-CV,
random-forest with 300 trees / depth 8, gradient-boosting with 300
trees / depth 4) on both `ω_mag_dps` directly and `log(ω_mag_dps)`.
Evaluate via leave-one-out cross-validation across all 100 seeds.

## How

- Feature extraction: ~50 ms per seed (Lomb-Scargle dominates).
- LOO eval: 4 models × 2 targets × 100 folds = 800 fits. Wall ~60 s
  total (BLAS=1).
- Decision bar: best-model LOO median MAPE < 3% → Q4c-iv passes
  (Q4c becomes tractable). 3-7% → marginal (improves on s007 7.4%,
  worth keeping as coarse prior). ≥7% → LC-only priors are dead.

## Result

### Per-model LOO MAPE (median / p90 / max, percent)

| model  | direct median | direct p90 | direct max | log median | log p90 | log max |
|--------|----------------|-------------|-------------|-------------|----------|----------|
| linear | 24.0           | 87.0        | 213         | 25.3        | 79.4     | 207      |
| ridge  | 24.3           | 76.6        | 192         | 21.9        | 73.0     | 197      |
| rf     | 20.0           | 65.6        | 138         | 18.5        | 62.5     | 137      |
| gbr    | 18.6           | 64.6        | 144         | **16.4**    | 65.6     | 148      |

Best: **gbr (log target) — median 16.4%**.

### Per-subcohort error (gbr-log)

| subcohort           | n  | median MAPE | p90  | max  | <3%   | <5%   | <10%  |
|---------------------|----|-------------|------|------|-------|-------|-------|
| <2 rotations        | 10 | 22.8%       | 91.5 | 105  | 10%   | 10%   | 20%   |
| 2-3 rotations       | 11 | 37.8%       | 104  | 148  | 0%    | 0%    | 0%    |
| 3-5 rotations       | 13 | 20.0%       | 61.8 | 82   | 15%   | 15%   | 23%   |
| ≥5 rotations        | 66 | 13.5%       | 53.3 | 85   | 11%   | 21%   | 42%   |

Even the most-favourable subcohort (≥5 rotations) gives only 21% of
seeds inside 5%. The fraction inside 3% is 11% across the cohort —
indistinguishable from random luck.

### Worst-10 seeds (gbr-log)

| seed | truth ω (dps) | pred  | err%   | rot/lc |
|------|----------------|-------|--------|--------|
| 5    | 0.213          | 0.528 | 147.6  | 2.13   |
| 10   | 0.106          | 0.217 | 105.3  | 1.06   |
| 61   | 0.297          | 0.606 | 103.9  | 2.97   |
| 42   | 0.129          | 0.245 | 90.0   | 1.29   |
| 55   | 0.593          | 1.096 | 84.7   | 5.93   |
| 9    | 0.218          | 0.399 | 83.5   | 2.18   |
| 47   | 0.458          | 0.837 | 82.5   | 4.58   |
| 57   | 0.201          | 0.359 | 79.1   | 2.01   |
| 91   | 1.426          | 0.301 | **78.9** | **14.26** |
| 66   | 0.222          | 0.380 | 71.2   | 2.22   |

**Seed 91** is striking — 14.3 rotations sampled (best in cohort), yet
predicted as 0.30 dps (close to truth/4). Confirms the s007 finding
that seed 91's LC is dominated by precession-driven envelope at
~1/4 the rotation rate; both spectral and time-domain features
inherit this bias and the regressor mis-learns ω from them.

### Top-10 RF feature importances (full-data fit, ω-mag target)

| rank | feature              | importance |
|------|----------------------|------------|
| 1    | acf_first_peak_lag   | 0.281      |
| 2    | acf_lag1             | 0.102      |
| 3    | acf_lag5             | 0.101      |
| 4    | glint_count          | 0.100      |
| 5    | ls_top0_f            | 0.044      |
| 6    | ls_top1_f            | 0.042      |
| 7    | glint_spacing_mean   | 0.040      |
| 8    | pwr_ratio            | 0.024      |
| 9    | dmag_mean_abs        | 0.024      |
| 10   | ls_top1_p            | 0.023      |

The ACF-first-peak-lag dominates. This is exactly the s007 ACF
estimator that gave median 22.6% on 10 seeds (worse than LS oracle).
The model is essentially leaning on a single dominant feature that
reflects the LC's main periodicity — and since that periodicity is
multi-component for tumbling bodies, the prediction errs whenever
precession or beats dominate over spin.

## Why this matters

The progression of negative results is now structural:

1. **s003** — at non-truth ω the surrogate landscape is incoherent
   (multi-basin, no useful ω-gradient).
2. **s007** — simple peak-spacing ω-mag prior fails (oracle 7.4%
   median, blind 15.7%).
3. **s008** — model-aware multi-feature ω-mag regression also fails
   (best 16.4% median LOO MAPE; even best subcohort 13.5%).

These three measurements close the LC-feature-prior class
**structurally**, not as a model-tuning failure. The LC's periodic
content reflects body-frame trajectories of (k1, k2), which depend
on q0 + ω + inertia ratios. Without satellite-model knowledge of the
inertia ratios, the LC features are jointly under-determined for ω.

The remaining tractable Q4c paths are:

- **Cohort basin-radius probe (Q4b extension).** Measure how seed-28-
  like the cohort tail is. If <10% of seeds have <3° basins, Q4c can
  target the bulk and accept a partial-cohort recovery rate. Wall ~15
  min Pool(8). Independent of the prior question.
- **Hierarchical / adaptive ω search.** Coarse-to-fine ω-grid, with
  the q0-Sobol density refined where surrogate-MSE drops. Or
  basin-hopping LM with stochastic ω-step. New architecture; no
  prior survey experiment for it. Cost: deferred design.
- **Accept architecture limit.** Document Q4c as "tractable for the
  4/5 wide-basin majority; tight-basin tail (seed 28) is open" and
  pivot to hi-fi ρ-band validation and downstream-result robustness.
  Pragmatic shortcut.

The s008 negative confirms there is no cheap shortcut. Q4c either
costs days per seed (brute force) or needs an architectural shift.

## Numbers

- Wall: 60.1 s for the full pilot (100 seeds × 4 models × 2 targets
  × 100 LOO folds = 800 fits + feature extraction). BLAS=1.
- 28 features × 100 seeds dataset. Output sizes: features.npz 30 KB,
  predictions.npz 30 KB, summary.json 4 KB, scatter.png ~140 KB,
  feature_importance.png ~60 KB.

## Artefacts

- `experiments/s008_lc_feature_regression_omega.py` — pilot script.
- `experiments/s008_lc_feature_regression_omega.md` — this writeup.
- `results/s008/features.npz` — 100 × 28 feature matrix, log-y, names.
- `results/s008/predictions.npz` — per-model LOO predictions and errors.
- `results/s008/summary.json` — decision-grade scalars.
- `results/s008/scatter.png` — 4×2 grid of truth-vs-prediction plots.
- `results/s008/feature_importance.png` — top-15 RF importances.
- `results/s008_run.log` — console output.

## Out of scope

- Larger / deeper neural-net regressors. Could be slightly better but
  unlikely to bridge a 2-3× gap (16% → 5%); fundamental information
  ceiling is set by tumbling-body LC structure, not model capacity.
- Including PA / `q0` as input features (cheating — these are
  inversion targets).
- Joint (ω-mag, ω-dir) regression — ω-dir prediction is even harder
  given the same multi-periodicity; not pursued.
- Inertia-ratio-aware features (would require knowing the satellite
  inertia matrix, i.e. model-specific information — closes the gap
  but isn't a black-box LC prior anymore).
- Direct training of an "ω-recovery" objective via differentiable
  surrogate inversion (out of scope; that is the inversion task itself).

## Cross-references

- **[[s007_omega_mag_peak_spacing_pilot]]** — single-feature
  spectral-prior closure that motivated this experiment.
- **[[s003_landscape_vs_omega]]** — established the truth-ω tube
  structural property that any ω prior needs to land inside.
- **[[s005_joint_local_descent]]** — established the 3-5% ω-mag
  basin radius that this pilot's 3% threshold targets.
- **[[concepts/known_pathologies_to_revalidate]]** — none of the
  buggy-era findings revisited here.
