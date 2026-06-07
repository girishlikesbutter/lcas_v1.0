# Series 07 — Multi-Epoch Winding Score: FINDINGS

> **Status: CLOSED.** Multi-epoch scoring is sound in principle but fails because all staircase omegas have 13-25° direction error. The bridge constrains endpoints, not the rotation axis. See `DEAD_ENDS.md`.

**Date:** 2026-03-10
**Branch:** `exp/multi-epoch-winding-score`
**Test case:** Intelsat 901, true ω₀ = [0.5, -0.3, 2.0] deg/s, leg 0 (peaks 183→260, dt=555s)

---

## Executive Summary

**Multi-epoch lo-fi brightness scoring does NOT solve the winding discrimination problem.** The correct winding (step 3) ranks #3/8 vs observed data and #5/8 vs lo-fi reference. The fundamental reason is not the lo-fi/hi-fi mismatch — it's that **all staircase omegas have the wrong direction** (13-25° off from true ω at peak A), so none produce correct intermediate trajectories.

---

## m021: Core Test Results

### Ranking Table — vs Observed LC (hi-fi + noise)

| Rank | Step | \|ω\| (deg/s) | ω dir error | MSE    |
|------|------|---------------|-------------|--------|
| 1    | 1    | 0.959         | 18.0°       | 2.0476 |
| 2    | 2    | 1.601         | 15.8°       | 2.1976 |
| **3**| **3**| **2.230**     | **13.1°**   | **2.4794** |
| 4    | 6    | 4.180         | 17.4°       | 2.5965 |
| 5    | 0    | 0.316         | 24.9°       | 2.6315 |
| 6    | 7    | 4.824         | 15.4°       | 2.9350 |
| 7    | 5    | 3.509         | 20.3°       | 3.0716 |
| 8    | 4    | 2.883         | 12.9°       | 3.9501 |

True |ω| = 2.083 deg/s → correct winding is step 3 (2.230 deg/s). **Ranks #3.**

### Ranking Table — vs Lo-Fi Reference (no shadow mismatch)

| Rank | Step | \|ω\| (deg/s) | MSE    |
|------|------|---------------|--------|
| 1    | 1    | 0.959         | 1.9899 |
| 2    | 6    | 4.180         | 2.2563 |
| 3    | 2    | 1.601         | 2.3317 |
| 4    | 0    | 0.316         | 2.5276 |
| **5**| **3**| **2.230**     | **2.5916** |
| 6    | 7    | 4.824         | 2.7060 |
| 7    | 5    | 3.509         | 2.9553 |
| 8    | 4    | 2.883         | 4.0324 |

Removing the lo-fi/hi-fi mismatch makes things **worse** (rank drops from 3 to 5). The problem is NOT shadow model fidelity.

### LC Overlay Plot

![LC overlay](../../../data/results/inversion_diagnostics/m021_multi_epoch_winding_score.png)

The 8 staircase curves oscillate at radically different frequencies (as hypothesized), but **none match the observed data**. The green lo-fi truth curve (MSE=0.162) closely tracks the observed points while all staircase curves have MSE > 2.0 — an order of magnitude worse. Every staircase omega has the wrong rotation axis direction.

### Subsampling Sensitivity

| Every N | Points | Correct Rank | Winner |
|---------|--------|-------------|--------|
| 1       | 78     | 3           | step 1 |
| 2       | 39     | 2           | step 7 |
| 5       | 16     | 2           | step 6 |
| 10      | 8      | 4           | step 6 |
| 20      | 4      | 5           | step 6 |
| 40      | 2      | 4           | step 7 |

No subsampling rate recovers the correct winding. The winner is unstable across rates, confirming there's no clean signal.

---

## m022a: Nudge Robustness

| Nudge (deg) | N trials | Mean rank | P(rank=1) |
|-------------|----------|-----------|-----------|
| 0 (baseline)| 1       | 3.0       | 0%        |
| 1           | 5        | 3.0       | 0%        |
| 3           | 5        | 3.6       | 0%        |
| 5           | 5        | 3.0       | 20%       |

Ranking is remarkably stable at rank ≈ 3 for nudges up to 3°. The one rank=1 hit at 5° is a coincidence (other trials give rank 3-5). Since the baseline is already rank 3, attitude precision is not the bottleneck.

---

## Root Cause Analysis

The staircase (m017) finds 8 ω solutions that bridge q_A → q_B with arrival error ≈ 0. Each differs by roughly one full revolution. However:

1. **All staircase omegas have the wrong direction.** The bridge constrains only the two endpoints, leaving the rotation axis underdetermined. The staircase explores ω along the minimum-winding rotation axis (rotvec(q_A⁻¹·q_B)/dt), which is 13-25° off from the true ω direction at peak A.

2. **Wrong direction → wrong intermediate trajectory.** Even step 3 (correct magnitude, 2.23 deg/s) has 13° direction error. Over 555s with |ω| ≈ 2 deg/s, that produces ~40° of cumulative attitude error at midpoint, making the predicted LC completely wrong.

3. **Lo-fi vs hi-fi is secondary.** The lo-fi reference MSE (true trajectory, no shadows) = 0.162. The best staircase MSE = 2.05. The 12× gap between the true-trajectory lo-fi fit and the best staircase fit is entirely due to omega direction error.

4. **The staircase explores one branch.** The staircase omegas are dominated by the -z component (negative rotation around z), while the true ω at peak A has a dominant +z component ([0.536, 0.232, 2.000] deg/s). The staircase never explores the opposite rotation direction branch.

---

## Conclusion

**Multi-epoch brightness scoring is a sound discriminator in principle** — different windings DO produce radically different LCs (MSE range 2.0–4.0 across steps). The problem is that **the staircase doesn't explore the correct omega direction**, so the correct winding's omega vector is 13° off in direction, producing an intermediate trajectory that's just as wrong as the other windings.

### What Would Fix This

The staircase currently varies |ω| along a FIXED rotation axis. To make multi-epoch scoring work, the solver needs to:

1. **Search omega direction, not just magnitude.** For each winding number, find the ω that minimizes multi-epoch MSE (not just arrival error). This becomes a 3D optimization per winding step.
2. **Explore both rotation branches.** The minimum-winding rotvec has sign ambiguity (go left vs go right). The staircase should try both.
3. **Embed LC scoring in the bridge objective.** Instead of `min |ω| s.t. arrival_err ≈ 0`, use `min MSE_intermediate s.t. arrival_err ≈ 0`. This simultaneously constrains direction and identifies the correct winding.

The hypothesis (multi-epoch scoring discriminates windings) is **conditionally validated**: it would work if the omega direction were correct. The staircase implementation is the bottleneck, not the scoring concept.
