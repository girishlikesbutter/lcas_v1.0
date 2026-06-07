---
title: "m100-101 — Multi-phi + Selection"
type: experiment
sources:
  - "raw/inversion_diagnostics/m100/"
  - "raw/inversion_diagnostics/m101/"
related:
  - "[[candidate-selection]]"
  - "[[multi-phi]]"
  - "[[m098_m099_nm_grid_pipeline]]"
  - "[[m102_fullmse]]"
created: 2026-04-08
updated: 2026-04-16
confidence: high
---

# m100-101 — Multi-phi + Selection

Two experiments testing multi-phi attitude search and candidate selection strategies.

## Setup

- **Multi-phi**: Test multiple attitude angles (phi values) per omega candidate instead of just the alignment-derived phi
- **Selection strategies**: Compare full-window MSE vs multi-window vote for final candidate selection

## Results

- **Multi-phi fixes 14/24** previously failing (seed, phi) combinations (FAIL to OK)
- **Full-MSE > multi-window vote** for final selection
- **Geo step hangs** on flat landscapes when processing 80 candidates (4x cost from multi-phi)
- Multi-phi risks **wrong-phi selection** — more phi options means more chances to pick the wrong one
- **LC MSE doesn't perfectly correlate with actual errors** — lowest MSE candidate isn't always closest to truth

## Three Key Findings

1. **Full-MSE beats vote**: Scoring the entire light curve window is more reliable than voting across sub-windows
2. **Multi-phi fixes attitude basins** but introduces wrong-phi risk and 4x geometric refinement cost
3. **LC MSE is imperfect**: A candidate can have lower LC residual than the true solution due to noise fitting

## Key Takeaway

Multi-phi is powerful but expensive and risky. The conservative path — keep single-phi but use full-MSE selection — captures most of the benefit without the cost. This motivated [[m102_fullmse]].
