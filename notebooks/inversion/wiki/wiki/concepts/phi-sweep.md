---
title: "Phi Sweep"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[alignment-cost]]"
  - "[[glint-physics]]"
  - "[[multi-phi]]"
  - "[[m100_m101_batch_multi_phi]]"
  - "[[m102_fullmse]]"
  - "[[de-attitude-search]]"
  - "[[surrogate-de-search]]"
  - "[[m113_de_attitude_search]]"
created: 2026-03-18
updated: 2026-04-16
confidence: high
---

# Phi Sweep

At a specular [[glint-physics|glint]], the attitude is constrained to a **1-DOF circle** (rotation about the PAB). The phi sweep enumerates rotations around this circle.

## Mechanism

1. Identify glint epoch and the facet normal responsible
2. Compute the PAB (phase angle bisector) at that epoch
3. Enumerate 360 rotation angles (phi) in 1-degree increments around the PAB axis
4. Each phi value defines a candidate initial attitude q0
5. Score each candidate via [[alignment-cost]] or [[brdf-cost]]

## Two-Phase Approach

- **Coarse sweep**: 1-degree bins, 360 evaluations
- **Local refinement**: fine-tune best phi values

With oracle omega, this achieves 1.77deg attitude error and 1.15deg omega error (m042b).

## Important Constraints

- MUST score against +/-X normals only (bug fix from m068 (pre-wiki; fix carried into [[m070_full_pipeline]]))
- Single best phi was used in pre-m115 pipeline ([[m102_fullmse]]); SUPERSEDED below
- [[multi-phi]] (testing multiple phi values) was explored in [[m100_m101_batch_multi_phi]] but abandoned due to cost and wrong-phi selection noise

## Role in Pipeline

Phi sweep produces the initial attitude estimate q0. This q0, combined with each omega candidate from [[grid-search]], defines the full state for LC evaluation. The [[grid-search|delta-q factorization]] means one propagation per (dir, mag) pair serves all phi values.

## Anchor Alignment Error (2026-04-13)

**Critical limitation discovered:** The `anchor_q_from_phi` function forces the body-frame PAB to equal a standard normal (e.g., [1,0,0]). The TRUE body-frame PAB is typically 1-3° from the nearest standard normal at the brightest peak. This **anchor alignment error** is amplified by cos^250 at ±X lobes to 100-1200× the noise floor MSE. See [[anchor-alignment-error]].

The error makes phi selection fail for ATT_FAIL seeds (40% of failures). Multiple fixes attempted within the 1-DOF framework (best-anchor, sparse hi-fi, IPL centroids) — all failed. See [[m112_bestanchor_selection]], [[m110_hifi_phi_study]], [[m107_m108_ipl_cost]].

## SUPERSEDED by 3-DOF Surrogate DE (2026-04-13)

**The phi sweep is replaced in the new pipeline.** m113 showed 3-DOF DE attitude search achieves 0.6° error with truth omega (vs 165° with phi sweep on the same seed). m115 validated the surrogate-powered version on 10 seeds: 10/10 valid solutions.

The phi sweep's fundamental limitation — forcing the body-frame PAB to a standard normal — cannot be fixed within the 1-DOF parameterization. The 3-DOF DE searches all of SO(3) directly, bypassing the anchor alignment error entirely. See [[de-attitude-search]] and [[surrogate-de-search]].
