---
title: "ATT_FAIL — Attitude Failure Diagnosis"
type: concept
sources: ["data/results/inversion_diagnostics/m103_hybrid/"]
related: ["[[candidate-selection]]", "[[phi-sweep]]", "[[m107_m108_ipl_cost]]", "[[pab-contour-isoshell]]", "[[anchor-alignment-error]]", "[[de-attitude-search]]", "[[surrogate-de-search]]", "[[m115_surrogate_pipeline]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# ATT_FAIL — Attitude Failure Diagnosis

## Failure Mode Breakdown

Of 15 m103 seeds: 3 GRID fail, 3 TWIN, 6 ATT_FAIL, 1 OK, 2 PARTIAL.

ATT_FAIL = omega correct (<5°) but attitude wrong (47-165°). Seeds: 0, 11, 27, 46, 58, 75.

## Anchor Alignment Error IS the Bottleneck (updated from m111)

**CORRECTED (2026-04-13):** Earlier analysis concluded anchor quality was NOT the bottleneck. m111 refuted this: the cos^250 specular BRDF amplifies even 1-2° anchor error to 100-1200× the noise floor. Switching anchors dramatically improves ranking (seed 58: rank 35→4, seed 27: rank 37→7). See [[anchor-alignment-error]].

However, fixing the anchor within the 1-DOF phi sweep framework proved impossible (m112: omega error corrupts distant-epoch PABs). The solution is to **abandon the 1-DOF parameterization entirely** and search the full 3-DOF attitude space with DE. See [[de-attitude-search]].

## The Problem Is Phi Selection (within the 1-DOF framework)

The twist angle around the anchor normal is wrong. Lo-fi phi discrimination tests:
- At peaks: 3.7x range
- At plateaus: 125x range
- Truth phi consistently ranks worse than false positives

For OK seeds (93, 73): lo-fi finds exact truth phi with 52x discrimination. For ATT_FAIL seeds: lo-fi brightness surface is too symmetric at their lobes -- multiple phis produce similar brightness profiles.

## Shadow Analysis

Some peaks have 0.1-0.5 mag shadow asymmetry on flanks, but some zero-shadow peaks also fail -- shadows alone don't explain ATT_FAIL.

## Isoshell Framework Analysis (m109, 2026-04-13)

Comprehensive testing of 3 isoshell-based phi discrimination approaches:
1. **IPL centroid proximity**: disc ratio 0.997 (seed 27) — no discrimination
2. **Brightness derivative matching**: disc ratio 1.021 — barely above noise
3. **Multi-anchor constraint satisfaction**: truth rank 161/360 (seed 46) — fails

Root cause: the isoshell IS the zero-phase brightness surface, which is too symmetric about lobe normals. All IPL-derived metrics reduce to alignment cost or lo-fi MSE. See [[isoshell-phi-limits]].

**Shadows** (up to 4.6 mag at ±Y lobes) are the ONLY mechanism that breaks the lo-fi symmetry. See [[shadow-asymmetry]]. However, sparse hi-fi phi sweep also failed (m110), and the anchor alignment error (m111) is the more fundamental bottleneck. The current approach: bypass the 1-DOF phi sweep entirely with 3-DOF surrogate DE attitude search ([[de-attitude-search]], [[surrogate-de-search]]).

## Resolution: Surrogate-DE (m113→m115, 2026-04-13)

**ATT_FAIL is SOLVED.** m115 showed surrogate multi-start 3-DOF DE finds valid solutions for all 10 baseline seeds (10/10 hi-fi MSE < 1.0). The ATT_FAIL failure mode no longer exists in the new pipeline — it was an artifact of the 1-DOF phi parameterization, not an intrinsic limitation.

Remaining bottleneck: **omega selection** (some seeds have correct omega in pool but m102 selects the wrong one). Addressed (4/4 seeds, inline 2026-04-15) via surrogate-DE MSE ranking — see [[surrogate-omega-selection]]. Extension to all 10 baseline seeds pending the [[harvester-optimization]] branch (m117).
