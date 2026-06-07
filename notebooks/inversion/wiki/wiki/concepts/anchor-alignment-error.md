---
title: "Anchor Alignment Error — The Real Phi Bottleneck"
type: concept
sources: []
related: ["[[att-fail-diagnosis]]", "[[phi-sweep]]", "[[pab-contour-isoshell]]", "[[isoshell-phi-limits]]", "[[de-attitude-search]]", "[[m111_shadow_isoshell]]", "[[m112_bestanchor_selection]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# Anchor Alignment Error — The Real Phi Bottleneck

## Key Finding

The dominant failure mode in phi selection is NOT shadows, NOT epoch selection, NOT lo-fi vs hi-fi evaluation. It is the **anchor alignment error** — the angular distance between the true body-frame PAB at the anchor epoch and the nearest standard normal used in the phi parameterization.

## The Mechanism

The `anchor_q_from_phi` function creates attitudes by:
1. Aligning a body-frame normal `n_body` (e.g., [1,0,0]) to the J2000 PAB
2. Twisting by angle phi about `n_body`

This forces PAB_body = n_body exactly. But the TRUE body-frame PAB is typically 1-5° from the nearest standard normal. This angular offset, amplified by cos^250 at ±X lobes, creates 0.2-1.2 mag² MSE — 100-1200× above the noise floor.

| Parameterization error | cos^250 effect | MSE per epoch | vs noise² |
|----------------------|----------------|---------------|-----------|
| 0.3° | 0.98 | 0.0004 | 0.2× |
| 1.0° | 0.92 | 0.007 | 3× |
| 1.5° | 0.83 | 0.03 | 12× |
| 2.0° | 0.71 | 0.09 | 36× |
| 3.0° | 0.47 | 0.28 | 112× |

At >1.5° error, the parameterization MSE floor overwhelms the phi-discriminating signal.

## Evidence (m111 session, all [inline])

Tested ±X anchor vs ±Z anchor at brightest peaks of each lobe:

| Seed | ±X err | ±X rank | Best alt err | Alt rank | Alt lobe |
|------|--------|---------|--------------|----------|----------|
| 27 | 1.46° | 37/72 | **0.29°** | **7/72** | +Z |
| 58 | 2.13° | 35/72 | **1.36°** | **4/72** | -Z |
| 46 | 2.23° | 31/72 | 2.23° (same) | 31/72 | +X |

Seed 58 goes from FAIL (rank 35) to near-OK (rank 4, disc=0.996) by switching to a ±Z anchor with 1.36° error.

## Why Shadows Are Not the Bottleneck

Comprehensive analysis showed:
- Shadow contribution to phi variation: **<7%** of total (lo-fi ≈ hi-fi std across phis)
- BRDF geometry dominates phi variation (std 0.1-2.4 mag across phis)
- Brightest peak (±X) has **zero phi sensitivity** (cos^250 flat at maximum)
- The phi signal lives in DIMMER peaks, where it's overwhelmed by the parameterization MSE floor

## Implications for the Pipeline

1. **Anchor selection should minimize alignment error**, not maximize brightness
2. The isoshell framework's IPL centroid distance IS the alignment error metric
3. For each omega candidate, search all peaks for the anchor with minimum centroid distance
4. ±Z/±Y anchors can be better than ±X despite being dimmer
5. Seeds where no peak has < 1.5° alignment error (like seed 46) remain fundamentally hard

## Connection to Isoshell Framework

The IPL centroid at each peak epoch gives the body-frame PAB direction. The distance from this centroid to the nearest standard normal is exactly the anchor alignment error. The isoshell framework provides this metric for free from precomputed data (IPL census).

**The isoshell's real value for phi:** not in discrimination (the zero-phase surface is too symmetric for that), but in **selecting the optimal anchor epoch** by finding the tightest IPL where the centroid is closest to a standard normal.
