---
title: "m111 — Anchor Alignment Error Discovery"
type: experiment
sources: []
related: ["[[anchor-alignment-error]]", "[[isoshell-phi-limits]]", "[[shadow-asymmetry]]", "[[att-fail-diagnosis]]", "[[m112_bestanchor_selection]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m111 — Anchor Alignment Error Discovery

## Hypothesis (evolved through session)

Initial: shadows are the only phi discriminator → shadows at specific epochs fix ATT_FAIL.
Revised: the anchor alignment error in `anchor_q_from_phi` is the real phi bottleneck.

## Method

Systematic inline analysis of phi sensitivity components:
1. Shadow contribution quantification (lo-fi vs hi-fi std across phis)
2. Brightest-peak phi sensitivity (cos^250 flatness)
3. Anchor alignment error vs cos^250 amplification
4. Multi-normal anchor comparison at specific peaks

## Key Results (all [inline])

### Shadow contribution is small (<7%)
- Lo-fi and hi-fi have nearly identical phi sensitivity (std across phis)
- BRDF geometry dominates phi variation (std 0.1-2.4 mag)

### Brightest peak has ZERO phi sensitivity
- cos^250 is flat at its maximum → no phi signal at the brightest epoch
- Phi signal lives in dimmer peaks, overwhelmed by parameterization MSE floor

### Anchor alignment error is the bottleneck

| Param error | cos^250 effect | MSE/epoch | vs noise² |
|------------|----------------|-----------|-----------|
| 0.3° | 0.98 | 0.0004 | 0.2× |
| 1.5° | 0.83 | 0.03 | 12× |
| 2.0° | 0.71 | 0.09 | 36× |
| 3.0° | 0.47 | 0.28 | 112× |

### Better anchor fixes ATT_FAIL seeds

| Seed | ±X err | ±X rank | Best alt err | Alt rank |
|------|--------|---------|-------------|----------|
| 27 | 1.46° | 37/72 | 0.29° (+Z) | 7/72 |
| 58 | 2.13° | 35/72 | 1.36° (-Z) | 4/72 |

### Negative results
- Shadow-corrected isoshell: FAILS (k1≠PAB decorrelation)
- Hi-fi at shadow-rich epochs: FAILS (binary shadow transitions)
- Full-curve lo-fi: FAILS (shadow mismatch MSE floor)

## What We Learned

1. The anchor alignment error IS the phi bottleneck, not shadows or epoch selection
2. The isoshell framework's IPL centroid distance = optimal anchor quality metric
3. Better anchor selection should rescue ATT_FAIL seeds → m112
4. The ISP framework's real value: epoch quality metrics and anchor selection, not phi discrimination
