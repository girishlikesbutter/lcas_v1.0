---
title: "s060 — full-LC |C_t|(t) sharpness diagnostic"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s060_sharpness_map.py
  - notebooks/inversion/survey/results/s060_sharpness_map/seed{028,089,010}/
related:
  - s059j_design — single-anchor argmin|C_t|(t) restricted to early window [3, 30)
  - s048c+ cloud viewer — frame-by-frame visualization of survivor cloud
  - project_sharpness_not_brightness.md — sharpness ≠ brightness
created: 2026-05-09
updated: 2026-05-09
confidence: high (direct measurement on 3 seeds; full-LC scan)
---

## TL;DR

Measured |C_t|(t) — count of attitude candidates surviving the brightness gate at each epoch — across the full LC for seeds 28, 89, 10 with N=25k Sobol pool and TOL=0.10 mag. **|C_t| varies 20-300× across the LC**: seeds have many sharp moments scattered between wide diffuse-moderate bands. Sharp anchors are NOT bright-peak-only — on seed 28 the single sharpest moment (|C_t|=5) is the dimmest point in the LC. Cluster-independent K=4 sharp anchors exist on seed 28 (six available) and seed 89 (four available); seed 10 has only three. This diagnostic substrates the s060 multi-anchor architecture design.

## What

The s059j architecture picks ONE sharp anchor `T_A = argmin |C_t|(t)` restricted to early window [3, 30) — chosen for backward-propagation sensitivity bounds. This restriction misses the fact that many seeds have sharper |C_t| moments LATER in the LC, and that sharpness is not synonymous with brightness. This diagnostic measures |C_t| at every epoch (full LC) to see how many sharp anchors actually exist and where they sit.

## How

For each epoch t ∈ [0, 500):
- Project the Sobol pool to body frame at t.
- Survive: `|surrogate_pred(pool@t) - measured_mag(t)| < 0.10 mag`.
- |C_t|(t) = sum of survivors.

Pool size N=25k. Single-threaded, BLAS=1, ~17 min per seed. Top-K identified by `np.argsort(Ct)[:K]`. Cluster-independent K extracted by greedy: argmin Ct, mask ±30 epochs, repeat until |C_t| > 500 or K reached.

## Result

| seed | \|ω\| dps | min |C_t| | median | max | <100 | <500 | <25 (floor) | K cluster-indep |
|---|---|---|---|---|---|---|---|---|
| 28 | 1.43 | **5** at t=312 | 719 | 1880 | 32 | 135 | 5 | **6** |
| 89 | 0.24 | 12 at t=413 | 1219 | 4291 | 52 | 101 | 8 | 4 |
| 10 | 0.11 | 44 at t=147 | 1161 | 4068 | 20 | 90 | 0 | 3 |

**Top-K sharp anchors are spread between bright and dim extremes, never in mid-mag.** On seed 28, top-10 sharp anchors are 4 in dim regime (mag > 18) + 6 in bright regime (mag < 7), 0 in mid-band. The single sharpest moment is at the dimmest LC point (t=312, mag=20.62, |C_t|=5).

**Cluster-independent K=4 anchor sets per seed:**

- Seed 28 (Δt≥30, |C_t|<500): {312, 224, 492, 390} or {123, 312, 390, 492} — span ~270 epochs ≈ 1+ polhode period at |ω|=1.43 dps.
- Seed 89 (only 4 anchors meet threshold): {413, 208, 240, 91} — span ~320 epochs.
- Seed 10 (only 3 anchors meet threshold): {147, 70, 32} — ALL within t=32-147 window, all in same dim regime. Limited temporal spread.

## Why this matters

1. **Multi-anchor architecture (s060) is viable on seeds 28 and 89** — K=4 cluster-independent sharp anchors exist on both. Seed 10 has only K=3 anchors, all in one dim regime; the architecture's joint-constraint power is weaker on slow tumblers.
2. **Restricting anchor selection to early window [3, 30)** as s059j does misses the cohort's sharpest moments. On seed 28 the early-window argmin gives |C_a|=383 at T_A=25; the full-LC argmin is |C_t|=5 at t=312 — 76× sharper.
3. **Sharp anchors are not bright-peak-only**, against typical glint-detector intuition. The dimmest LC moments can be the sharpest. Architecture must scan full LC, not bright-peak-detect.
4. **Cohort generalization measured separately** in s060_cohort_anchor_topology — see that writeup for the 11-seed sample.

## Numbers

- Pool: N=25,000 Sobol on SO(3), tolerance 0.10 mag.
- Wall: ~17.5 min per seed single-threaded (12k surrogate predictions/sec, 12.5M total).
- Three seeds run in parallel (single-threaded each, separate processes).
- Saved per-seed: sharpness_map.npz (Ct, mag, observation_times, top-K indices), summary.json, sharpness_map.png.

## Artefacts

- `notebooks/inversion/survey/experiments/s060_sharpness_map.py` — diagnostic script.
- `notebooks/inversion/survey/results/s060_sharpness_map/seed{028,089,010}/`:
  - `sharpness_map.npz`
  - `summary.json`
  - `sharpness_map.png` — LC + |C_t|(t) overlay with top-K marked.

## Out of scope

- Dense-pool measurement (N=400k) at top-K anchors. Diagnostic at N=25k is sufficient for top-K identification; the multi-anchor architecture's Stage 2 will resample at top-K with N_DENSE.
- Per-anchor cluster structure measurement. Done separately in s060_anchor_topology.
- Cohort generalization across full m048. Done separately in s060_cohort_anchor_topology on 11 PA-stratified seeds.
- Full-LC sweep with parallel Pool(N) workers — would cut wall to ~2 min per seed but adds setup complexity; deferred unless the diagnostic gets re-run.

## Cross-references

- `s059j_design.md` line 66-70 — early-window single-anchor `argmin|C_t|`; this diagnostic is the full-LC superset.
- `project_sharpness_not_brightness.md` — saved memory note that anchor selection should NOT be brightness-filtered.
- `s060_anchor_topology.md` — per-anchor cluster structure.
- `s060_multi_anchor_design.md` — architectural design that this diagnostic substrates.
