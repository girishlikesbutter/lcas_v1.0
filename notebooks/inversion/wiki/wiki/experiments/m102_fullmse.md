---
title: "m102 — Full-Window MSE Selection (CURRENT BEST)"
type: experiment
sources:
  - "raw/inversion_diagnostics/m102_fullmse/"
related:
  - "[[candidate-selection]]"
  - "[[nm-refinement]]"
  - "[[m100_m101_batch_multi_phi]]"
  - "[[lo-fi-mse]]"
created: 2026-04-09
updated: 2026-04-16
confidence: high
---

# m102 — Full-Window MSE Selection (CURRENT BEST)

Conservative consolidation: NM_TOP=300 + full-window MSE selection without multi-phi.

## Setup

- **NM_TOP=300** (from [[m098_m099_nm_grid_pipeline]])
- **Full-window MSE selection** (from [[m100_m101_batch_multi_phi]] finding #1)
- **No multi-phi** (conservative: avoids wrong-phi risk and 4x geo cost)
- ~12 min/seed runtime

## Results (all 10 seeds)

| Seed | q0 err | w_dir err | w_mag err | Status |
|------|--------|-----------|-----------|--------|
| 0 | 8.70deg | 3.12deg | +0.22% | OK |
| 6 | 176.85deg | 3.08deg | -0.15% | PARTIAL (+X twin, good omega) |
| 12 | 169.60deg | 8.01deg | +0.21% | PARTIAL (+X twin, marginal omega) |
| 14 | 90.72deg | 56.27deg | +0.71% | FAIL (off-basin omega) |
| 24 | 152.29deg | 34.91deg | +0.32% | FAIL (wrong att/omega) |
| 27 | 175.39deg | 38.38deg | +0.15% | FAIL (~180deg but high w_dir) |
| 33 | 134.66deg | 18.46deg | -0.70% | FAIL (off-basin) |
| 36 | 177.32deg | 10.73deg | +0.27% | PARTIAL (+X twin, moderate omega) |
| 74 | 6.70deg | 4.88deg | -0.09% | OK (PARTIAL to OK improvement) |
| 93 | 178.43deg | 0.49deg | -0.11% | OK (+X twin, excellent omega) |

**Summary**: 3 OK + 3 PARTIAL + 4 FAIL

## Comparison to m090 baseline

- **Seed 74**: PARTIAL to OK (net improvement)
- **Seed 14**: OK to FAIL (regression — full-MSE selects wrong candidate on this seed)
- **Seed 24**: OK to FAIL (regression — same issue)
- Net: traded 2 OK seeds for 1 OK + different failure pattern

## Key Takeaway

Full-window MSE selection is better in principle (universal, doesn't need glint constraints) but can regress seeds that alignment cost happened to get right. The 4 FAIL seeds remain the core challenge — they need either multi-phi ([[m100_m101_batch_multi_phi]]) or a fundamentally different approach to attitude-basin selection.
