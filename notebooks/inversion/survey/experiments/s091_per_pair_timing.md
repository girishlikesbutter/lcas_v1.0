---
title: "s091 — per-pair cost of the cross-cloud spin-solve"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s091_per_pair_timing.py
related:
  - experiments/s088_bvp_shoot_conditioning.md
  - experiments/s092_cross_cloud_116.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (direct single-process timings; Pool(24) throughputs assume ideal scaling — treat as upper bounds)
---

# TL;DR
Measured the atomic costs of processing one (q_a, q_b) cross-cloud pair on seed 119. A single 2-anchor BVP solve is **1.72 ms** (~14k/s on Pool(24)); the cloud-free 3rd-anchor brightness check is **0.84 ms**. Per-pair cost is dominated by winding *enumeration*: **~3.8 s/pair** brute (2200-init multi-start) vs **~0.13 s/pair** targeted (40-init |ω|-line). **Architectural consequence: the connectability screen (one shoot) is cheap enough to run on all A×B pairs; the bottleneck is cloud_size² (number of pairs), not the solve.**

# What
Locks the budget for the cross-cloud architecture before scaling: how long per pair, and where does the cost sit?

# How
`experiments/s091_per_pair_timing.py`, seed 119, single-process timings (per-call clarity) + Pool(24) throughput projections.

# Result
| operation | cost |
|---|---:|
| finite-diff init | 0.004 ms |
| one 2-anchor BVP solve | 1.72 ms |
| one 3-anchor joint solve | 2.71 ms |
| brightness check (propagate to C + 1 surrogate) | 0.84 ms |
| brute 2200-init enumeration | ~3.8 s/pair |
| targeted 40-init |ω|-line enumeration | ~0.13 s/pair |

# Why this matters
- **Connectability screen = 1 shoot/pair (1.7 ms, ~14k/s Pool(24)).** Cheap gate for all A×B pairs.
- **Disambiguation is nearly free** (0.84 ms × ~15 windings ≈ 13 ms/pair).
- **The driver is cloud_size².** 1k clouds (1M pairs) → ~70s screen; 5k clouds (25M) → ~30 min — over the 15-min budget on the screen alone. So cloud size trades against budget quadratically (a counter-pressure on "denser sampling fixes accuracy"). The targeted |ω|-line enumeration (plausible from s089's same-direction ladder, unvalidated) is what keeps the survivor stage affordable.

# Numbers
- 2-anchor shoot 1.718 ms; 3-anchor 2.710 ms; brightness check 0.837 ms; FD init 0.0041 ms (source: `experiments/s091_per_pair_timing.py` run).
- brute ~3.78 s/pair (≈6 pairs/s Pool24); targeted ~0.13 s/pair (≈180 pairs/s); single shoot ~14k/s Pool24.

# Out of scope
- Validating that the 40-init |ω|-line catches the same in-prior windings as the 2200-init brute.
- A cheaper pre-prune (|L|-magnitude matmul) before the 1.7 ms shoot for very large pair counts.

# Cross-references
- `s092_cross_cloud_116.md` — the cross where this cost was realised (1.87M pairs, 325s).
- `s088` — the BVP primitive timed here.
