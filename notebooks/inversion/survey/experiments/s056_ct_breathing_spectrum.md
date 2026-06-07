---
title: "s056 — |C_t|(t) breathing-spectrum sanity look on seed 89"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s056_ct_breathing_spectrum.py
  - notebooks/inversion/survey/results/s048c_cloud_viewer/seed089/8bb9b81f1602/spread.npz
  - notebooks/inversion/survey/results/s053_cohort_polhode_survey/cohort.npz
  - notebooks/inversion/survey/data/trajectories/traj_seed089.npz
related:
  - s048c — cloud viewer (substrate)
  - s055a — pol_diam LC-recoverable at ~25% MAPE (the OTHER cloud-derived prior)
  - s055b/c/d — closed ω-direction priors from LC features alone
  - s056b — residual spectrum after regressing out mag (companion)
created: 2026-05-07
updated: 2026-05-07
confidence: high (single-seed result; methodology robust)
status: WEAK direction (2). |C_t|(t) breathing carries 300× dynamic range and a dominant 20.5-min spectral peak, but is dominated by mag_hifi (Pearson ρ=+0.78); essentially decoupled from |ω|(t) (ρ=+0.06). Not a new signal channel beyond what mag_hifi gives.
---

## TL;DR

The cloud-viewer's per-epoch survivor count `|C_t|(t)` varies dramatically (300×, from ~50 at deepest constrictions to ~17000 at peaks) and has a sharply-peaked Lomb-Scargle spectrum at P=20.5 min — *not* at the rotation period (25.0 min for seed 89). But the breathing is **strongly correlated with the LC magnitude itself** (Pearson ρ=+0.78 on log scale), and **decoupled from |ω|(t)** (ρ=+0.06). Counterintuitive: |C_t| at LC PEAKS is *larger*, not smaller (ratio 1.81×). Bright peaks have *more* attitudes consistent with the magnitude within ±0.1 mag tolerance because they're stationary points in mag(t). Direction (2) breathing-spectrum is mostly a geometric reflection of the LC magnitude time series, not an independent signal channel.

## What

User asked to mine the s048c+ viewer's per-epoch quaternion clouds for ω-direction priors. Three sketched directions: (1) cross-epoch ω-aggregation, (2) |C_t|(t) breathing-spectrum, (3) cloud-centroid SO(3) drift. This experiment pilots (2) on seed 89's dense viewer run (100k Sobol pool × 500 epochs, fully cached as `survive_all` boolean mask + `pred_all` surrogate magnitudes).

## How

1. Load `n_survivors`, `mag_hifi`, `hifi_peak_epochs`, `closest_deg_per_epoch` from cached spread.npz (no recomputation).
2. Compute |C_t|(t) at LC peaks vs off-peaks: median ratio.
3. Detect |C_t| local minima via `scipy.signal.find_peaks(-log_n, prominence=0.3)` (~factor-2 dips).
4. Lomb-Scargle on (|C_t|, mag_hifi, |dmag/dt|⁻¹, |ω|_body(t)) over [f_min=1/(3T), f_max=0.4/dt], 4000 freq grid.
5. ω_body(t) reconstructed by quaternion finite-diff of cached `traj["quaternions"]` (passive convention).
6. Pearson correlations: log|C_t| vs (mag_hifi, log|dmag/dt|⁻¹, |ω|(t), mag).

Wall: <2s. No rendering, no Pool.

## Result

- **|C_t|(t) range**: [44, 17085], median 4844 — 300× dynamic range
- **At LC peaks vs off-peaks**: median |C_t| 8772 vs 4841, ratio **1.81×** — peaks have MORE survivors
- **Top |C_t| spectral peak**: P = 20.52 min, PSD = 0.481 (vs runner-up at P=8.4 min, PSD=0.073 — ~7× separation)
- **Rotation period**: P_rot = 25.02 min; the |C_t| dominant period is shorter than f_rot
- **Pearson correlations** (log|C_t| vs):
  - mag_hifi: ρ = +0.78 (strong)
  - log|dmag/dt|⁻¹: ρ = +0.58 (moderate)
  - |ω|_body(t): ρ = +0.06 (essentially zero)
  - mag_hifi: ρ = +0.78 (raw scale)
- **|ω|(t) sanity**: median 0.2413 dps via finite-diff (vs cohort 0.2398 dps, Δ=0.6%); std/mean = 0.54% (matches s053 cohort exactly)
- **7 |C_t| minima detected**; median distance from each minimum to nearest LC peak = 50.5s

## Why this matters

The cloud data is a richer substrate than its summary statistics suggest, but the breathing dimension is geometric reflection of the LC magnitude — same dominant period, same rotation-harmonic content (after residual analysis in s056b). For ω-direction priors specifically, |C_t|(t) carries no decoupled information from |ω|(t) on this seed. Direction (2) is largely closed (with caveat for non-separatrix seeds where polhode period is finite — see Out of scope).

## Numbers

| Quantity | Value |
|----------|-------|
| n_epochs | 500 |
| dt | 7.21 s |
| duration | 3600 s |
| |ω| (cohort) | 0.2398 dps |
| pol_diam (s053) | 0.468 dps |
| D (s053, near-separatrix) | 1.008 |
| |C_t| dynamic range | 300× |
| |C_t| peak/off-peak ratio | 1.81× |
| Top spectral peak (|C_t|) | P=20.52 min, PSD=0.481 |
| log|C_t| ρ vs mag_hifi | +0.78 |
| log|C_t| ρ vs |ω|_body(t) | +0.06 |

## Artefacts

- `experiments/s056_ct_breathing_spectrum.py`
- `results/s056_ct_breathing_spectrum/ct_breathing_overview.png` (4-panel: LC + peaks; |C_t|(t) log; spectra overlay; |ω|_body(t))
- `results/s056_ct_breathing_spectrum/summary.json`
- `results/s056_ct_breathing_spectrum/breathing_data.npz`

## Out of scope (deferred)

- **Test on a non-separatrix seed (D far from 1)**: seed 89 has D=1.008 where polhode period diverges, so |ω|(t) is essentially flat. On a non-separatrix seed (e.g. seed 14, D=1.089, |ω|=1.23 dps, pol_diam=1.97 dps), |ω|(t) modulation is finite — direction (2) might survive there even if it fails here.
- **Peak detection algorithm sensitivity**: tested only one prominence threshold for |C_t| minima.

## Cross-references

- `experiments/s056b_ct_residual_spectrum.md` — residual after regressing log|C_t| on (mag, slope, mag²): 32% variance unexplained, peaks at 2× and 3× rotation harmonics
- `experiments/s055a_pol_diam_lc_regression.md` — companion: pol_diam IS LC-recoverable (~25% MAPE)
- `concepts/polhode_prior.md` — broader polhode-prior architecture
- `MEMORY/feedback_lc_spectral_omega_prior_dead.md` — closes the LC-only ω-prior class; s056 supports
