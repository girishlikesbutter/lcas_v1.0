---
title: "m109 — IPL Phi Discrimination Diagnostic"
type: experiment
sources: ["raw/inversion_diagnostics/m109_ipl_phi/diagnostic.npz"]
related: ["[[isoshell-phi-limits]]", "[[pab-contour-isoshell]]", "[[att-fail-diagnosis]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m109 — IPL Phi Discrimination Diagnostic

## Hypothesis

IPL centroid metrics can discriminate the correct phi for ATT_FAIL seeds where lo-fi MSE fails.

## Method

8 metrics tested on 6 seeds (3 ATT_FAIL + 2 borderline + 1 OK), 36 phi values, truth omega:
1. Centroid proximity (mean angular distance to nearest centroid)
2. Weighted centroid proximity
3. Loop membership count
4. Stability (fraction of epochs in same loop)
5. Derivative matching (dL/dt predicted vs observed)
6. Multi-anchor constraint satisfaction
7. Zero-phase lo-fi MSE
8. Combined scores

## Results

| Metric | Win rate | Mean disc ratio | Verdict |
|--------|----------|----------------|---------|
| Centroid proximity | 2/5 | 0.997 | FAIL |
| Weighted proximity | 2/5 | 0.997 | FAIL |
| Loop membership | 2/5 | - | FAIL |
| Stability | 0/5 | - | FAIL |
| Derivative matching | 5/5 | 1.021 | Marginal |
| Multi-anchor | varies | 0.970 | FAIL |

## What We Learned

ALL IPL centroid metrics reduce to alignment cost or lo-fi MSE. The zero-phase brightness surface is too symmetric about lobe normals. IPL centroids ≈ standard normals at tight epochs. The isoshell framework cannot discriminate phi — only shadows break the lobe symmetry.
