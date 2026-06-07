---
title: "Alignment Cost (PAB)"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[brdf-cost]]"
  - "[[m093_expected_dot_nelder_mead]]"
  - "[[m096_exp1_oracle_grid]]"
  - "[[phi-sweep]]"
  - "[[glint-physics]]"
  - "[[m104_crossing_diagnostic]]"
  - "[[constraint-poor-regime]]"
  - "[[phase_B_m048_cohort]]"
  - "[[phase-angle-operating-range]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
created: 2026-03-20
updated: 2026-04-17
confidence: high
---

# Alignment Cost (PAB)

The PAB alignment cost measures how well a candidate attitude aligns a facet normal with the **phase angle bisector** (PAB = normalized sum of sun and observer directions) at [[glint-physics|glint epochs]].

## Mechanism

At a specular glint, the facet normal must align with the PAB. Given a candidate attitude q, the cost computes:

- Rotate each candidate facet normal into the inertial frame
- Measure angular deviation from the PAB at each glint epoch
- Aggregate (typically min over normals, sum over epochs)

This provides **1 constraint per epoch** (the normal-PAB alignment angle).

## Limitations

- Effective for seeds with bright +/-X glints (only ~13% of population per [[m096_exp1_oracle_grid]] census)
- Basins are **too narrow** for the 87% of seeds that lack bright +/-X constraints
- Uses only 1 of the 3 available geometric constraints (see [[brdf-cost]])
- Replaced by [[brdf-cost]] as primary cost for grid scoring in [[m094_brdf_cost_function]]

## High-phase flatness (added 2026-04-17)

At high sun–observer phase angles (≥ ~65°), the alignment cost surface over `(q0, ω)` space **flattens dramatically**. Multiple pipeline seeds have now confirmed this independently of constraint count:

- [[phase_B_m048_cohort]] seed 028 (87.3° phase, 10 spec peaks): NM top-20 best ω_err = 30.8° at rank #19, `geo_cost` values spread across 3.05e-02 to 6.67e-02 but with no clear descent gradient visible to L-BFGS-B. Pool(24) Step 4 hung twice (workers burned unbounded function evals per iteration, never converged).
- [[phase_B_m048_cohort]] seed 069 (67.7° phase, 11 spec peaks): NM top-20 best ω_err = 28.8° at rank #8, similar `geo_cost` distribution. Killed pre-geo on the grounds that no candidate was within polish range of truth.

**Both seeds had plenty of constraints (≥10 spec peaks).** This rules out [[constraint-poor-regime]] as the mechanism — the failure is geometric: at high phase, the PAB sweep across body normals during an observation window produces a cost landscape with many near-equivalent minima. Grid density becomes insufficient and NM's basin of attraction is narrower than the spacing between false minima.

Seed 28's Pool(24) geo hang is a **downstream symptom** of this flatness, not a separate bug. L-BFGS-B's line search on a near-flat surface spawns huge numbers of function evaluations before the `maxfun` cap triggers, and if a worker hits a NaN or failed-pickle in that region, the pool deadlocks on the missing result.

## Two failure regimes

| regime | trigger | `geo_cost` of NM top-20 | truth in pool? | fix |
|---|---|---|---|---|
| [[constraint-poor-regime]] | ≤2 spec peaks | 1e-23 to 1e-20 (pathologically small) | yes, but unselectable | selection metric change (lo-fi tiebreaker) or full-LC cost |
| high-phase flatness | phase ≥ ~65° | 1e-2 to 1e-1 (normal scale but undifferentiated) | no (basin too narrow for grid to sample) | grid densification won't help; need a different cost function |

Both regimes point at the same architectural limitation: **alignment cost uses only 1 of 3 available geometric constraints per epoch** (see [[brdf-cost]]) and **is blind to the full LC** — it only scores spec peaks. [[upstream-redesign-6dof-surrogate-de]] is the proposed replacement.

## Anti-correlation with truth (added 2026-04-28 in [[m135_alignment_cost_forensics_constrained_anchor]])

Empirically — not just imperfectly correlated but **anti-correlated**: cost-at-truth probe across the 5 sampling-failure seeds (47, 51, 79, 84, 89) from the random m048 cohort showed `cost(geo_best) << cost(truth)` by **4 to 22 orders of magnitude**, with geo_best at 8°–82° from truth direction. Even on Band-A seed 91 the alignment cost prefers a non-truth basin (factor 2.4× better cost at geo_best than at truth, with geo_best 14.6° from truth). On the seed 91 lofi-300 pool, alignment-cost rank-1 is at **174.83°** from truth (anti-truth ±X twin); surrogate full-LC MSE rank-1 on the SAME 300 candidates is at **2.76°** from truth.

Interpretation: alignment cost rewards a 1-D PAB-normal-alignment property that is satisfied (often more strongly) by spurious geometries unrelated to truth. The two prior failure regimes are surface manifestations of this single root cause. **Denser sampling categorically cannot recover truth** for these seeds — even if a grid point landed exactly on truth, the L-BFGS-B geo refinement would walk away from it toward a deeper spurious basin.

## History

## History

- Core cost function from early pipeline through [[m093_expected_dot_nelder_mead]]
- [[m096_exp1_oracle_grid]] experiments 2-3 showed alignment basins collapse for majority of seeds
- Still used in [[phi-sweep]] for initial anchor identification
- 2026-04-17: Phase-B m048 cohort exposed high-phase flatness + constraint-poor regime as distinct failure modes (see above)
