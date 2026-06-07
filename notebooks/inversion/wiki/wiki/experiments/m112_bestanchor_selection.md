---
title: "m112 — Post-NM best-anchor phi re-sweep"
type: experiment
sources: ["raw/inversion_diagnostics/m112_bestanchor/seed_027/result.json"]
related: ["[[anchor-alignment-error]]", "[[best-anchor-selection]]", "[[att-fail-diagnosis]]", "[[m102_fullmse]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m112 — Post-NM Best-Anchor Phi Re-sweep

## Hypothesis

After NM refinement, searching all specular peaks for the body-frame-normal-aligned anchor with minimum alignment error, then re-sweeping phi at that anchor, should fix ATT_FAIL seeds.

## Method

m102 + Step 3.5 inserted between NM dedup and geo refinement:
1. For each top-20 NM candidate, propagate full trajectory
2. At each specular peak (excluding original anchor): compute body-frame PAB, find closest standard normal
3. Select peak/normal with minimum alignment error
4. Fine phi sweep (360 values) at best anchor with lo-fi MSE
5. Feed both original and re-swept candidates to geo and hi-fi (40 total)

## Results (seed 27)

**Classification: FAIL** (q0=175.4°, w_dir=38.4°, same as m102)

### Step 3.5 diagnostics

| w# | Orig w_err | Best alt epoch | Alt normal | Est. align err | dt(s) | Resweep w_err |
|----|-----------|---------------|-----------|---------------|-------|--------------|
| 6 | 4.7° | ep 135 | -Z | 1.32° | -1118 | **65.1°** |
| 7 | 54.1° | ep 279 | -Z | 4.56° | -79 | 33.7° |
| 10 | 36.1° | ep 196 | +Y | 1.01° | -678 | 55.2° |

ALL 20 re-swept candidates degraded vs originals. No re-swept candidate was selected by any hi-fi window.

### Root cause

NM-refined omega has ~3-5° direction error. Over 500-1500s propagation to distant peaks, body-frame PABs shift 20-150° from truth. The "best anchor" found by the search is not actually well-aligned — the estimated alignment error is unreliable at distant epochs.

At the original anchor epoch (ep 290), body-frame PAB alignment is trivially 0° (by construction of anchor_q_from_phi). After fix to exclude original anchor, the search finds alternative peaks 80-1700s away — all too distant for reliable PAB estimation.

### Timing

| Step | Time |
|------|------|
| Grid | 82s |
| Lo-fi | 9s |
| NM | 104s |
| **Step 3.5** | **126s** |
| Geo (40 cands) | **288s** |
| Hi-fi (40 cands) | **940s** |
| **Total** | **1549s (25.8 min)** |

2× m102 runtime due to 40 candidates through geo+hifi. Future: cap hi-fi candidates to top 20.

## What We Learned

1. **Post-NM anchor search does NOT work** — omega error at distant epochs corrupts the search
2. **The anchor alignment ceiling exists** (97/100 seeds < 0.5° at truth omega) but is inaccessible with estimated omega
3. **The approach needs to be earlier in the pipeline** (grid-level or parameterization change), not post-hoc
4. **±Y phi range bug confirmed**: only ±X has 180° twin; ±Y should use [0, 2π)
5. **IS-901 inertia is truly diagonal** even with antennas at 15° (AD cancellation)
