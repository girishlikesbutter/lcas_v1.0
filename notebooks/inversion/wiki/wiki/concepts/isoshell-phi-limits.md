---
title: "Isoshell Framework — Limits for Phi Discrimination"
type: concept
sources: []
related: ["[[pab-contour-isoshell]]", "[[att-fail-diagnosis]]", "[[hi-fi-scoring]]", "[[alignment-cost]]"]
created: 2026-04-13
updated: 2026-04-13
confidence: high
---

# Isoshell Framework — Limits for Phi Discrimination

## Key Finding

The isoshell/IPL framework **cannot discriminate phi (attitude twist) beyond what lo-fi MSE already provides** for ATT_FAIL seeds. This was tested comprehensively in three approaches, all with the truth omega and 36-phi sweeps.

## What Was Tested

### 1. IPL Centroid Proximity (mean angular distance to nearest centroid)
- **Seed 27** (pure ATT_FAIL): truth score 29.91° vs best other 29.81° → disc ratio 0.997
- At dim epochs (96% of observations): centroid distance barely changes with phi (30.8° → 31.1° across 180° phi range). Loops too wide.
- At bright epochs (1.4%): marginal discrimination (3.84° vs 3.87°) — but these are the same epochs alignment cost already uses.

### 2. Brightness Derivative Matching (dL/dt predicted vs observed)
- **Seed 27**: truth disc ratio 1.021 (barely better than noise)
- The derivative captures crossing direction but the zero-phase brightness surface gradient is too smooth to discriminate.

### 3. Multi-Anchor IPL Constraint Satisfaction
- At 2-3 tight IPL epochs within 70-epoch window, check centroid alignment
- **Seed 27**: No valid secondary anchors (nearby peaks have centroid_dist > 30°)
- **Seed 46**: Truth rank 161/360, disc ratio 0.970
- **Seed 75**: Truth rank 246/360, disc ratio 0.942
- Even OK control (seed 93): Truth rank 148/360

## Why It Fails

The isoshell framework is the **zero-phase brightness surface** (k1=k2=PAB, no shadows). Any metric derived from it reduces to some form of lo-fi evaluation:
- Centroid proximity ≈ alignment cost (centroids = lobe normals at tight epochs)
- Loop membership ≈ lo-fi magnitude matching (point is on loop iff brightness matches)
- Derivative matching ≈ lo-fi derivative (just a different view of the same smooth function)

For ATT_FAIL seeds, the zero-phase brightness surface is **too symmetric about lobe normals.** Rotating phi about the anchor normal keeps brightness nearly constant (specular BRDF depends on cos(angle_from_normal), which varies slowly near peaks).

## What DOES Break the Symmetry

**Shadows** are the only mechanism:
- Shadow effects at ±Y lobes reach 4.6 mag (solar panels occlude bus)
- 40-44% of epochs have |shadow effect| > 0.1 mag
- Shadows are lobe-specific: different phis produce different lobe-visiting schedules → different shadow patterns
- This is captured by **hi-fi evaluation** (which includes shadow ray tracing) but NOT by the isoshell framework

## Where the Isoshell IS Valuable

1. **Understanding failure modes** — revealed the discrimination gap at off-normal PAB directions
2. **Epoch quality metric** — IPL set length and loop count measure constraint tightness
3. **Shadow-critical epoch identification** — large lo-fi/hi-fi gaps reveal where shadows act
4. **Visualization** — interactive viewers show PAB-brightness-surface relationship
