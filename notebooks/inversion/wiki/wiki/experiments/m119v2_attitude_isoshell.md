---
title: "m119v2 — attitude-isoshell POC with fixed geometry (seed 14)"
type: experiment
sources: ["raw/inversion_diagnostics/m119v2/seed_014/summary.json"]
related: ["[[m119_attitude_isoshell]]", "[[m118_cost_comparison]]", "[[surrogate-attitude-isoshell]]", "[[kernel-factorization]]", "[[surrogate-model]]"]
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

# m119v2 — attitude-isoshell POC with fixed geometry (seed 14)

## Hypothesis

Re-run the retracted [[m119_attitude_isoshell]] POC under **internally-consistent geometry** (all per-epoch arrays from `setup_experiment()` at today's 6-hour window, no mixing with `m046_trajectories.npz`). Report the honest rank of the truth trajectory against the 60,000-candidate super-Fibonacci SO(3) grid under 13 cost variants.

## Context — why this exists

The previous session's `m119_attitude_isoshell.py` mixed two data sources with incompatible time grids:
- `m046_trajectories.npz`: stale 1-hour window, 7.21 s/epoch (500 epochs)
- `setup_experiment()`: current config, 6-hour window, 43.29 s/epoch (500 epochs)

Epoch index `i` meant different wall-clock times in each. Observer direction `k2` was off by up to 74° at epoch 499. Full diagnosis: `notebooks/inversion/12_brightness_surface/m119_BUG.md`.

## Fix applied

1. **Lib patch** in `notebooks/inversion/lib/experiment_setup.py`: added optional kwargs `true_q0_wxyz` and `true_omega0_rad` so `setup_experiment()` can be given a seed-specific truth. Backwards-compatible (defaults preserve pre-patch behaviour).
2. **New script** `notebooks/inversion/12_brightness_surface/m119v2_attitude_isoshell.py`. Stage A rewritten to source all per-epoch arrays from `setup_experiment` — m046 read ONLY for `q0s[seed]` and `omega0s[seed]` scalars (seed parameters, not time-dependent).
3. Constraint epochs = all 500 (dropped the pab-contour-based selection that was inherited from `m118_kernel_computation.py` at master's time grid).
4. Competitor target list reduced to `truth` only (m115 basins dropped — that pipeline is also affected by the same time-mismatch bug, to be addressed separately).

## Method

- SO(3) grid: 60,000 super-Fibonacci rotations (median ~3° geodesic resolution)
- Constraint epochs: all 500 observation epochs at today's 6-hour grid
- Residual kernel: `residual[60000, 500]` float32 (240 MB), ~79 s on Pool(8)
- Cost variants: 13 (`mean_L2`, `mean_L1`, `max_abs`, `count_pass_σ` for σ ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00}, `soft_pass_σ` for σ ∈ {0.10, 0.20, 0.50})
- Target: truth quaternion trajectory (propagated on today's 6-hr `obs_times` from the seed's q0/omega0)

## Results (seed 14)

- **Total runtime:** 123 s (Stage A 42 s, Stage C 79 s, rest negligible)
- **Truth residual:** median = 0.043 mag, p90 = 0.112 mag, max = 0.549 mag
- **Truth rank = 0 / 60,000 under ALL 13 cost variants**
- **Grid top-1 q0_err to truth:** 54° – 178° depending on variant (grid best is far from truth)
- **POC verdict:** PASS under all four checks (median < 0.1, p90 < 0.3, top-1000 under any variant, top-percentile < 0.1%)

## Honest interpretation

**The bug fix is correct** — geometry is now internally consistent, `obs_times` spans 21,600 s with uniform dt = 43.29 s, the assertion in Stage A guards against future regression.

**But the POC test design is still weak.** Every grid candidate is a **static** rotation — it represents a ω=0 trajectory that holds one attitude across all 500 epochs. Truth is a **tumbling** trajectory with 500 distinct quaternions. Truth beats 60,000 static rotations under every reasonable cost function near-tautologically: truth has 500 degrees of freedom (one per epoch) vs the grid's 1.

**What this experiment DOES establish (robustly):**
- The geometry-fix is correct (dt assertion, setup_experiment consistency).
- The surrogate model evaluates accurately at truth on today's 6-hr grid: median |residual| = 0.043 mag, p90 = 0.112 mag, max = 0.549 mag.
- The kernel-factorization infrastructure (Stage C builds, Stage D cheaply scores 13 variants) survives the rewrite.
- At seed 14's particular truth, brightness residual is concentrated in a few outlier epochs (max 0.549 vs median 0.043) — worth investigating, but not pathological.

**What this experiment does NOT establish:**
- Whether surrogate residual cost can discriminate truth from *nearby wrong tumbling* trajectories — the real inversion question.
- Anything about attitude-isoshell behaviour across other seeds.
- Whether the "isoshell" framing (threshold level-set) or the "score-ranking" framing is a useful discriminator for the tumbling-candidate set.

## Comparison with retracted v1

| Metric | m119 (v1, retracted) | m119v2 (fixed) |
|---|---|---|
| Constraint epochs | 255 (spec_peaks ∪ tight-IPL) | 500 (all) |
| Truth residual median | 0.49 mag | **0.043 mag** |
| Truth residual p90 | 4.21 mag | **0.112 mag** |
| Truth rank under mean_L1 | 0 / 60,000 | 0 / 60,000 |
| Competitors (m118 rank-1, m115 basins) | ranks 160–10,600 | not evaluated (pipelines affected by same bug) |
| Geometry | CORRUPTED | CONSISTENT |

The headline "rank 0 / 60000" number survives the fix, **but the v1 residual statistics were inflated by mixed geometry** (the truth propagated on master times evaluated against setup_experiment sun/obs directions accumulated ~74° k2 error by late epochs, yielding the bogus median 0.49 / p90 4.21 residuals). The v2 residuals (median 0.04, p90 0.11) are the honest surrogate-at-truth numbers.

## Outcome

- Bug fixed. Infrastructure preserved. Honest numbers landed.
- [[surrogate-attitude-isoshell]] branch cannot claim validation from a static-rotation POC.
- Next meaningful test requires a **tumbling-competitor set** (not static rotations) — e.g., the m118 rank-1 candidates re-propagated on today's obs_times, or a random sample of (q0, ω) from the 6-DOF parameter space. Deferred pending m115/inline_omega audit resolution.

## Wider audit finding

Two other scripts have the same time-mismatch bug pattern (read `obs_times` from m046 but `sun_pos/obs_pos/obs_dist` from `setup_experiment`):
- `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` — this means the "10/10 surrogate multi-start DE breakthrough" is in doubt.
- `notebooks/inversion/12_brightness_surface/archive/inline_omega_selection_test.py` — "4/4 surrogate omega-selection" is in doubt.

Two scripts that use m046 ARE internally consistent (pure m046, no setup_experiment mixing):
- `notebooks/inversion/12_brightness_surface/m117_result_harvester.py` (setup_experiment imported as `noqa F401`, never called)
- `notebooks/inversion/12_brightness_surface/m118_kernel_computation.py`

The fix pattern demonstrated here (lib-level override + all-setup_experiment script) applies to both affected scripts. Deferred to the next session.

## Files

- `data/results/inversion_diagnostics/m119v2/seed_014/summary.json`
- `data/results/inversion_diagnostics/m119v2/seed_014/residual_kernel.npz`
- `data/results/inversion_diagnostics/m119v2/seed_014/cost_variants.npz`
- `data/results/inversion_diagnostics/m119v2/seed_014/target_scores.npz`
- `data/results/inversion_diagnostics/m119v2/seed_014/plots/{residual_distribution,target_pass_curves,topK_overlap,truth_rank}.png`
- Script: `notebooks/inversion/12_brightness_surface/m119v2_attitude_isoshell.py`
- Report: `notebooks/inversion/12_brightness_surface/m119v2_REPORT.md`
