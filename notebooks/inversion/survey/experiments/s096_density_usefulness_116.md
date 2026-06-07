---
title: "s096 — density IS the lever: at δ<1° the full-LC RMSE discriminator + connectability both work (seed 116)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s096_density_usefulness_116.py
  - notebooks/inversion/survey/results/s096/basin.json
related:
  - experiments/s097_times0_gauge_bug_audit.md
  - experiments/s092_cross_cloud_116.md
  - experiments/s087_omega_dir_threshold.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (clean monotone basin; infra floor 0.0077 mag validates the scoring; N=1 seed 116)
---

# TL;DR
A forward-model basin probe on seed 116: place candidates at controlled geodesic distance δ from truth at the two solve anchors, `shoot` for ω, score the W_ACb window vs the **real** s092 junk band. Result is decisive — **at δ<1° the truth-near windowed RMSE is ~0.01-0.03 mag, 100% below the junk floor (junk p1 = 0.109), and 100% of candidates have ω-dir < 1.25°** (the s087 recovery threshold). Both levers degrade through δ≈2-3° — **exactly the 3-6° gap a 30k pool gives** (s092) — which is *why* s092/s093 were marginal. **Densifying the anchor pool to ≲1-1.5° is the lever; the discriminator and connectability then both work.** This run also caught the `times[0]==0` gauge bug (see s097) via its exact-truth infra floor.

# What
The densification plan rested on one untested assumption: that IF a denser pool put a candidate at <1°, the connectability ω-dir error would drop below s087's 1.25° gate AND its windowed RMSE would separate from junk. s092/s093/s094 all ran at 30k (nearest-truth 3.2-5.7°), confounding "metric is weak" with "pool too sparse." This measures the basin profile directly to disentangle them — before building any adaptive densifier.

# How
`experiments/s096_density_usefulness_116.py`, Pool(24), v2 surrogate. Anchors/window from `results/s092/cross.json` (A=112, B=244, C=295; W_ACb=[112,355]). For δ ∈ {0.25…7}° × M=120 random-axis perturbations of truth at A and B: `shoot` → ω-dir error vs truth + windowed RMSE (propagating from the anchor with `times[0]==0`, the s097 fix). Junk reference = 2500 **real** s092 survivors re-shot + scored over the same window (no oracle). Floor = exact (q_truth, ω_truth).

# Result

| δ (deg) | ω-dir p50 (deg) | windowed RMSE p50 (mag) | % < junk_p1 (0.109) | % ω-dir < 1.25° |
|---|---|---|---|---|
| 0.25 | 0.12 | 0.0115 | 100 | 100 |
| 0.50 | 0.23 | 0.0183 | 100 | 100 |
| **1.00** | **0.46** | **0.0332** | **100** | **100** |
| 1.50 | 0.70 | 0.0480 | 100 | 96 |
| 2.00 | 0.88 | 0.0658 | 95 | 71 |
| 3.00 | 1.43 | 0.0972 | 63 | 35 |
| 5.00 | 2.23 | 0.1650 | 12 | 22 |
| 7.00 | 3.41 | 0.2324 | 4 | 12 |

- exact-truth floor (infra): **0.0077 mag**; reproduction 1.71e-6° (buggy way 107.9° — see s097).
- junk band: p1 = 0.1093, p50 = 0.8313 mag (real s092 survivors, source: `results/s096/basin.json`).
- required uniform pool for nearest-truth 1.0° ≈ 3.75M, for 0.5° ≈ 30M (N^(−1/3) from 30k→5°; arithmetic, not measured).

# Why this matters
- **Confirms density is the lever, quantified:** truth-near must be within ~1-1.5° for clean (100%) separation on both connectability and the LC discriminator. The 30k pool's 3-6° gap sits on the degradation knee → the s092/s093 marginality is a pool-density artifact, not a metric failure.
- **Uniform densification is infeasible (~3.75M pool → cross is cloud², ~days)** → the adaptive coarse→fine resample (densify only inside the surviving coarse cloud) is the required architecture.
- **Caveat:** N=1 seed 116 (deliberately easiest). The aliased half of the cohort (s095 `near%`=0) can't form a truth-near-ω candidate at this anchor baseline regardless of density at the cloud — orthogonal problem.

# Numbers
- δ=1°: RMSE p50 0.033, 100% < junk_p1, 100% ω-dir<1.25° (source: `results/s096/basin.json` `per_delta`).
- δ=2°: 95% < junk_p1, 71% ω-dir<1.25°; δ=3°: 63% / 35%.
- floor 0.0077 mag; junk p1/p50 0.1093/0.8313 (source: same).

# Artefacts
- `results/s096/basin.json` (per-δ stats, floor, junk band, pool-size estimates).
- `results/s096/basin.npz` (per-δ ω-dir & RMSE arrays, junk arrays).
- `results/s096/basin.png` (two-panel: ω-dir vs δ with 1.25° line; RMSE vs δ with junk band + floor).

# Out of scope
- Polish-top-K from the now-working full-LC ranking (the loop-closing test).
- The adaptive coarse→fine densifier itself (this only sizes the target density).
- Fast / aliased seeds; cohort generalisation.

# Cross-references
- `s097_times0_gauge_bug_audit.md` — the bug this run caught + the s093/s094 reversals.
- `s092_cross_cloud_116.md` — the survivors used as the junk reference.
- `s087_omega_dir_threshold.md` — the 1.25° ω-dir recovery threshold.
