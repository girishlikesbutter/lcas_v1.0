---
title: "s060 — v1 vs v2 surrogate LC residuals on seed 28"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s060_v1_v2_render.py
  - notebooks/inversion/survey/results/s060_v1_v2_compare/seed28_v1_vs_truth.png
  - notebooks/inversion/survey/results/s060_v1_v2_compare/seed28_v2_vs_truth.png
related:
  - project_surrogate_model.md (v1↔v2 benchmarks)
  - s060_v1_v2_cloud_compare.md (the cloud-regime sequel that flipped the conclusion)
created: 2026-05-10
updated: 2026-05-10
confidence: high (direct render of cached k1/k2)
---

## TL;DR

Rendered seed 28's truth trajectory under both v1 and v2 surrogates using the cached `k1_body / k2_body` from `data/trajectories/traj_seed028.npz`. **v1 RMS = 0.104 mag, ρ = 2.09; v2 RMS = 0.024 mag, ρ = 0.48.** v1 has a localised ~1.0 mag spike near epoch 110; v2's worst spikes are ~0.2 mag at scattered epochs (240, 320, 370). Critically, v1 and v2 error spikes are at DIFFERENT epochs — not co-located. This was the seed observation that motivated the cloud-regime test (s060_v1_v2_cloud_compare).

## What

The user asked to see how v1 and v2 differ on a real LC. Seed 28 picked for dynamism: |ω|=1.44 dps, 500 epochs, mag range 5.01–20.62, 30+ specular peaks.

## How

`s060_v1_v2_render.py`: load `traj_seed028.npz`, predict mag with v1 (`/home/girish/surrogate_model/{s10_5M_weights,s10_5M_normalization}.npz`) and v2 (`lib.surrogate_eval`) at SP=0°, AD=15°. Save NPYs and plot via `python -m lib.lc_compare`.

## Result

| Surrogate | RMS (mag) | ρ | Band | Worst spike |
|---|---:|---:|---|---|
| v1 | 0.104 | 2.09 | B | ~1.0 mag at t≈110 |
| v2 | 0.024 | 0.48 | A | ~0.2 mag at t≈240, 320, 370 |

Visually: v1 is "mostly tight ±0.1 with localised spikes." v2 is "noise within ±0.2 across the LC." The spikes are NOT at the same epochs.

## Why this matters

This was the seed observation. v1's worst error is ~10× v2's. At cloud TOL=0.10 mag, v1's spikes shift cells across the threshold where v2 wouldn't, producing systematic bias in survivor sets. The follow-up (s060_v1_v2_cloud_compare) measured the cloud-regime IoU directly and found it ≈ 0.42 — see that writeup.

## Numbers

- Seed 28: |ω|=1.44 dps, mag range 5.01..20.62, 500 epochs.
- v1 RMS=0.104, ρ=2.09 (Band B).
- v2 RMS=0.024, ρ=0.48 (Band A).

## Artefacts

- `experiments/s060_v1_v2_render.py` — render script.
- `results/s060_v1_v2_compare/v1_lc.npy`, `v2_lc.npy` — cached LCs.
- `results/s060_v1_v2_compare/seed28_v1_vs_truth.png` — v1 vs hi-fi truth.
- `results/s060_v1_v2_compare/seed28_v2_vs_truth.png` — v2 vs hi-fi truth.

## Out of scope

- Cohort-wide v1↔v2 RMS distribution. Single-seed observation; the cloud-regime experiment generalises better.
- Spike-localisation analysis (which (k1, k2) configurations does v1 misbehave on). Defer.

## Cross-references

- `s060_v1_v2_cloud_compare.md` — the load-bearing follow-up (cloud-regime IoU=0.42, |C_t| differ by 78% at dim).
- `project_v1_substrate_reframe.md` (memory) — the methodological reframe this enabled.
