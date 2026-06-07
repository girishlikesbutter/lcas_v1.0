---
title: "s056b — log|C_t|(t) residual spectrum after regressing out mag_hifi"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s056b_ct_residual_spectrum.py
  - notebooks/inversion/survey/results/s056_ct_breathing_spectrum/breathing_data.npz
related:
  - s056 — primary breathing-spectrum analysis (mag-correlation finding)
created: 2026-05-07
updated: 2026-05-07
confidence: high (single-seed; methodology robust)
status: WEAK on seed 89. After regressing out mag_hifi (R²=0.61) + slope (+0.04) + mag² (+0.03), 32% residual variance remains in log|C_t|(t). Residual spectrum peaks at 2× and 3× rotation harmonics — recoverable signals, but already extractable from LC peak-spacing (s007/s008 closed dead-end).
---

## TL;DR

After regressing log|C_t|(t) on `[mag_hifi, log|dmag/dt|⁻¹, mag²]`, the linear fit captures **R²=0.681** of the variance. The 32% residual has spectral structure: top peaks at **P=12.99 min (2× f_rot)** and **P=8.26 min (3× f_rot)** — these are body-rotation-symmetry harmonics. They're real signal, but already encoded in LC peak-spacing — not a NEW signal channel beyond what s007/s008 LC features extract.

## What

s056 found log|C_t|(t) ↔ mag_hifi(t) at Pearson ρ=+0.78 — much of breathing is mag-derived. This experiment asks: in the 32% residual variance, is there structure at f_rot, polhode period, or other physically-meaningful frequencies that mag_hifi can't capture?

## How

1. Load s056's `breathing_data.npz` (t, log|C_t|, mag_hifi, slope_proxy, |ω|(t), closest_deg).
2. OLS regress log|C_t| on three predictor sets:
   - `[mag_hifi]` only
   - `[mag_hifi, log|dmag/dt|⁻¹]`
   - `[mag_hifi, log|dmag/dt|⁻¹, mag²]`
3. Lomb-Scargle on residuals + reference series (|ω|(t), closest_deg(t)).
4. Identify residual spectrum top peaks via `find_peaks(pgram, height=0.02)`.

Wall: ~1s.

## Result

| Predictor set | R² | residual std (dex) | residual factor on |C_t| |
|---|---|---|---|
| `mag_hifi` | 0.612 | 0.335 | 2.16× |
| `mag_hifi + log|dmag/dt|⁻¹` | 0.649 | 0.319 | 2.08× |
| `mag_hifi + slope + mag²` | 0.681 | – | – |

Residual top spectral peaks (after regressing out mag_hifi):
- P = 12.99 min (f = 0.077/min) — **2× f_rot**, PSD = 0.187
- P = 8.26 min (f = 0.121/min) — **3× f_rot**, PSD = 0.165
- P = 41.97 min (f = 0.024/min) — long-period, PSD = 0.093
- P = 6.59 min (f = 0.152/min) — **4× f_rot**, PSD = 0.078

Reference: |ω|_body(t) PSD concentrated at very low frequencies (single broad dome — std/mean=0.54%, essentially constant). closest_deg(t) PSD nearly flat / noise-like.

## Why this matters

- The residual structure IS rotation-harmonic, **not** polhode-period. For seed 89 at D=1.008 (separatrix), polhode period diverges — so polhode is not expected in the spectrum.
- Rotation harmonics 2× f_rot and 3× f_rot are extractable from LC peak-spacing analysis (the s007 closed dead-end). |C_t|-residual offers no new channel beyond LC peak-spacing.
- Power split: residual carries 68% of power below f=0.2/min (P>5 min), vs full log|C_t| 86%. Residual is shifted toward higher frequencies (the rotation harmonics).

## Numbers

| Quantity | Value |
|----------|-------|
| R² (mag only) | 0.612 |
| R² (mag + slope + mag²) | 0.681 |
| Residual std (mag only) | 0.335 dex |
| Residual factor on |C_t| | 2.16× |
| Top residual peak | P=12.99 min ≈ 2× f_rot |
| 2nd residual peak | P=8.26 min ≈ 3× f_rot |
| f_rot | 0.040/min (P=25 min) |

## Artefacts

- `experiments/s056b_ct_residual_spectrum.py`
- `results/s056b_ct_residual_spectrum/ct_residual_spectrum.png` (3-panel: log|C_t| + OLS fit; residuals; PSD overlay)
- `results/s056b_ct_residual_spectrum/summary.json`
- `results/s056b_ct_residual_spectrum/residual_data.npz`

## Out of scope

- Test on non-separatrix seed where polhode period is finite — would directly test if breathing-spectrum carries polhode signal as opposed to just rotation harmonics.

## Cross-references

- `experiments/s056_ct_breathing_spectrum.md` — primary
- `MEMORY/feedback_lc_spectral_omega_prior_dead.md` — s007/s008 already closed LC-spectrum-derived |ω| priors
