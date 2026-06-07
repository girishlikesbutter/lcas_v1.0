---
title: "m107-108 — IPL centroid grid cost (negative result)"
type: experiment
sources: []
related: ["[[ipl-census]]", "[[pab-contour-isoshell]]", "[[ipl-candidate-generation]]", "[[alignment-cost]]", "[[candidate-selection]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m107-108 — IPL Centroid Grid Cost (Negative Result)

## m107: Centroid Alignment Cost

Used 8 tight IPL minima as grid constraints with centroid alignment cost. Seed 27 truth ranked 754/2000. Root cause: delta-q amplification at distant epochs (0.67° grid error -> 30° PAB error at 40 min).

## m108: Central-Anchor Strategy

Central-anchor strategy to limit delta-q. Two selection strategies:
- **Strategy A:** Minimize max |dt|
- **Strategy B:** Minimize max(centroid_dist + PAB_error)

Strategy B gives better clusters but truth ranked 1470/2000. Root cause: epochs with 8-12 loops have centroids covering the sphere -- max alignment cost is non-discriminating.

## Population Analysis

All 81 viable seeds have estimated grid cost < 0.05 with strategy B, but actual runs fail because many-centroid epochs create false positives.

## Key Lesson

IPL centroids as a DROP-IN grid cost replacement doesn't work. The precision (1-3° centroids) is real but the discrimination mechanism (max-over-centroids) breaks with >4 centroids.
