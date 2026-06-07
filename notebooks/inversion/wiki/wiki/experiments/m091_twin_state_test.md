---
title: "m091 — Hi-fi Twin Test"
type: experiment
sources:
  - "raw/inversion_diagnostics/m091_twin_test/twin_lcs.npz"
related:
  - "[[twin-degeneracy]]"
  - "[[m090_robust_peak_selection]]"
created: 2026-04-04
updated: 2026-04-16
confidence: high
---

# m091 — Hi-fi Twin Test

Definitive test of which rotation axes produce optically identical light curves for IS-901.

## Setup

Generated hi-fi light curves for the true state rotated 180deg about each of six axes: +X, -X, +Y, -Y, +Z, -Z, plus the omega-direction (WD) and Earth-direction (ED) axes. Compared via RMS residual.

## Results

| Axis | RMS residual |
|------|-------------|
| +X | 0.000000 |
| -X | 0.000000 |
| +Y | 0.542 |
| -Y | 0.542 |
| +Z | 0.542 |
| -Z | 0.542 |
| +WD | ~1.6 |
| +ED | ~1.6 |

## Key Takeaway

Only 180deg rotation about the +X body axis produces an exactly identical light curve (RMS = 0). All other axes break the symmetry because the shadow geometry (solar panel self-occlusion) is X-axis symmetric for IS-901 but not Y/Z symmetric. This confirms [[twin-degeneracy]] is strictly +X only, which constrains how the pipeline should handle twin solutions (see [[m092_twin_axis_visualization]]).
