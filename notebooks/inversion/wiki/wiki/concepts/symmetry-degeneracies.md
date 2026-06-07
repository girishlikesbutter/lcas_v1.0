---
title: "Symmetry Degeneracies"
type: concept
sources: []
related:
  - "[[twin-degeneracy]]"
  - "[[multi-solution-philosophy]]"
  - "[[de-attitude-search]]"
  - "[[surrogate-model]]"
created: 2026-04-13
updated: 2026-04-16
confidence: low
---

# Symmetry Degeneracies

## Status: OPEN — full analysis needed

## Core Question

What symmetries does IS-901 have (exact and approximate), and what degeneracies does each imply for light curve inversion?

For any rotation R_sym that leaves the brightness function invariant — `brightness(R_sym·k1, R_sym·k2) = brightness(k1, k2)` — the state `(q_sym * q0_true, omega_true)` produces an identical LC (where q_sym is the LEFT-multiply quaternion corresponding to R_sym). Each such symmetry doubles the number of valid solutions.

## Known: ±X Twin (Exact)

R_180x = diag(1, -1, -1). LC residual = 0. See [[twin-degeneracy]].

Reason: IS-901 bus is symmetric about X. Solar panels on ±Y are mirror images. Antenna dishes are approximately symmetric.

## To Investigate: Other Exact Symmetries

### 180° about Y
LC residual = 0.542 mag RMS (from m091). BROKEN — solar panels have different illumination profiles on front vs back faces. The sun-tracking face has high specular, the back face is matte.

### 180° about Z
LC residual = 0.542 mag RMS (from m091). BROKEN — same reason (panel asymmetry when viewed from ±Z).

### 90°, 120°, etc. about X
Unlikely exact — the bus cross-section (Y-Z plane) is rectangular (3.5m × 2.8m), not square. 90° about X would map Y→Z which changes geometry.

### Combined rotations
Any composition of broken symmetries is also broken. Only the ±X twin and identity survive as exact symmetries. The symmetry group is Z_2 = {I, R_180x}.

## To Investigate: Near-Symmetries (Broken by Small Effects)

These are rotations where the LC residual is small but nonzero. They may create shallow local minima in the MSE landscape that trap optimizers.

### Near-symmetries to characterize:

1. **±Y near-symmetry:** 0.542 mag RMS is the AVERAGE. At what epochs is the residual large vs small? If shadow-rich epochs (±Y lobes) dominate the residual, then during shadow-free observation windows the ±Y twin might be nearly valid. Quantify: what fraction of epochs have residual < 0.01 mag?

2. **±Z near-symmetry:** Same question as ±Y.

3. **Small-angle rotations near ±X:** The exact twin is at exactly 180° about X. What's the MSE landscape NEAR the twin? Is it a sharp minimum or a broad basin? If broad, solutions at 170-175° might have low MSE for geometric reasons (not just omega-error artifacts).

4. **Continuous symmetries at specific epochs:** At bright peaks (PAB near ±X lobe normal), the brightness is dominated by one face. A single-face BRDF has continuous rotational symmetry about its normal. This means at peak epochs, MANY attitudes produce similar brightness. The peak-epoch near-symmetry could explain why the phi sweep fails — it's trying to discriminate attitudes using epochs where the brightness function is nearly rotationally symmetric.

5. **Shadow-broken symmetries:** Quantify for each candidate symmetry: how much of the residual comes from shadow effects vs BRDF asymmetry? If shadows are the dominant symmetry-breaker, the lo-fi model (no shadows) has a LARGER symmetry group than the hi-fi model, explaining why lo-fi can't discriminate attitudes that hi-fi can.

## How to Leverage Near-Symmetries

1. **Solution enumeration:** If we know the symmetry group (including approximate symmetries), we can generate candidate solutions from any single solution by applying all symmetry operations. Instead of multi-start DE searching blindly, apply known symmetries to found solutions.

2. **Search space reduction:** If the symmetry group is Z_2 (just ±X), we can restrict the q0 search to half the rotation space (one fundamental domain). This halves the DE search space.

3. **Expected solution count:** For n symmetries, expect n solutions (or n families of solutions). This sets expectations for multi-start: if we only find 1 solution, we should find n-1 more.

4. **Near-symmetry basins:** Map the MSE landscape near each approximate symmetry. If a near-symmetry creates a basin with MSE < threshold, it's a valid solution under [[multi-solution-philosophy]]. The number of "valid" solutions may be larger than the exact symmetry group suggests.

## Proposed Analysis (for next session)

1. **Symmetry census:** For R in {R_180x, R_180y, R_180z, R_90x, ...}, compute surrogate MSE of (R*q0_true, omega_true) vs truth LC. Use surrogate for speed, hi-fi for the promising ones. Do this for 5-10 seeds.

2. **Near-twin basin shape:** Sweep q0 along the geodesic from truth (0°) to twin (180°) about X, at 1° steps. Plot MSE vs angle. How broad is the twin basin?

3. **Epoch-resolved residuals:** For the ±Y and ±Z rotations, plot the per-epoch residual. Identify which epochs break the symmetry. Are they shadow-dominated?

4. **Lo-fi vs hi-fi symmetry group:** Repeat the symmetry census with lo-fi (no shadows). Is the lo-fi symmetry group larger? This would explain ATT_FAIL: the lo-fi optimizer can't distinguish attitudes that hi-fi can, because lo-fi has more symmetries.

5. **Surrogate MSE landscape near found DE solutions:** For each DE solution at q0_err ≈ 170°, check: is it near a known approximate symmetry? Or is it an omega-error artifact with no geometric explanation?
