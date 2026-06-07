---
title: "s038 handoff: s036/s037 diagnosis was surface-level — what next agent must investigate"
type: handoff
sources:
  - experiments/s036_multi_seed_pilot.md
  - experiments/s037_sobol_lm_pilot.md
  - results/s036_multi_seed_pilot/seed023/
  - lib/filter_costs.py
related:
  - s032
  - s033
  - s036
  - s037
created: 2026-05-06
updated: 2026-05-06
confidence: low (this is a problem statement, not a result)
---

## TL;DR

The s036 → s037 narrative ("phi-sweep is the bottleneck, replace with Sobol") was a premature jump. The s036 failure on seed 23 was diagnosed as "phi-sweep ICs too far from truth (ρ=50)" and the response was to throw away the entire geometric IC + filter pipeline. **That was wrong.** The filters were doing valuable work (99.9%+ rejection per s027) and were not the cause of failure. The actual chain of reasoning was never tested end-to-end; multiple unverified assumptions were stacked. This page lists the gaps the next agent must investigate before any further architecture changes.

## What was actually verified vs. asserted

### Verified (data-backed)
- s036 seed 23 produced 160 survivors at default density, all at WRONG ω-cells (ω-mag +254% to +424% off truth, q0 > 61° from truth). At the near-truth ω-cell (5.64% off), 1,056,000 candidates were generated and **0 passed the filter**.
- s036 seed 23 phi-sweep q_target_pool minimum distance to truth-q0 = 26.90° (face 5, phi 9). The "best face" by truth PAB alignment is face 0 (5.8° from PAB at epoch 20).
- Face 0 is missing from the q_target_pool because the tier classification puts the brightest peaks (mag 6.2-6.4) in T2 (faces [2,3,4,5]), while T1 faces [0,1] require mag < 6.0.
- Adding face 0 to the pool would not help: face-0 phi-sweep ICs are 37.47° from truth at N_PHI=12 and 37.05° at N_PHI=60. The phi axis is not the dominant error component.
- s037b confirmed Sobol N=64 + LM converges from ~90° on all three seed classes (8-9/64 Band A).

### Asserted without testing (gaps)
1. **"5.8° face misalignment propagates to 37° total attitude error"** — guessed, not measured. The actual mechanism producing the 37° off-circle floor is not understood.
2. **"The filter rejected the 37° candidates"** — never measured. We know 0 survivors at the near-truth ω-cell, but the actual geo and alignment scores of the closest phi-sweep ICs (at 26.9°, 37°, etc.) were never computed. The rejection mechanism could be filter-too-strict, OR the ICs simply don't produce matching bright peaks even at the correct ω-cell.
3. **"LM couldn't bridge 37°"** — never tested directly. s034 showed LM bridges from ρ=10 (q0~8°). s037b showed Sobol+LM bridges from ~90° in q0. The 37° regime is in between and was not explicitly tested with phi-sweep ICs at the correct ω-cell.
4. **"Phi-sweep is fundamentally limited"** — overgeneralised from one failed seed. The off-circle floor is seed-dependent (7.86° on seed 89, 26.9° on seed 23) but the population distribution is unknown. We never measured the off-circle floor across the cohort.

## What the next agent must investigate

### 1. Measure the actual filter rejection mechanism (highest priority)

Take the seed 23 phi-sweep IC pool. At the **near-truth ω-cell** (cell at 5.64% off, ω-direction matched to truth), compute for every candidate:
- geo cost (per spec event AND aggregate)
- alignment cost (per bright peak AND aggregate)
- q0 distance to truth
- surrogate MSE → predicted ρ

Sort by q0 distance to truth. **Plot geo and align score vs q0 distance.** This will answer:
- Is the filter rejecting candidates at q0_err < 30° (the LM convergence basin)?
- Is there a cliff in the score function near truth, or a smooth fall-off?
- Are the rejected near-truth candidates failing on geo, alignment, or both?

If the filter is rejecting candidates within the LM convergence basin, the filter is too strict. If it's accepting them but they're at wrong ω-cells, the issue is the ω-direction grid (Fibonacci sphere alignment). If neither, the phi-sweep ICs genuinely don't match truth's bright-peak structure even when q0 is close.

### 2. Build a per-event scale-aware geo filter

Current geo cost uses uniform 5° tolerance regardless of which spec event. But at each spec event, truth's face-PAB angle is different (0.5° to 7.2° on seed 23). A candidate at 4° from face-0/PAB at epoch 20 is INSIDE truth's natural error (5.8°) but OUTSIDE the filter's 5° threshold.

Implementation:
- Per spec event, measure truth's `min_ang_dist(face_normals[best_face], pab_at_event)` from cached `spec_geometry.npz`
- Set per-event tolerance = max(5°, truth_angle + margin), where margin is e.g. 2-3°
- Re-run filter on s036 seed 23 IC pool with the new tolerance — count survivors at near-truth ω-cell

### 3. Test LM convergence from phi-sweep ICs at the correct ω-cell

Bypass the filter entirely. Take the 528 phi-sweep ICs from seed 23, pair each with the near-truth ω (truth direction × nearest bracket cell), run joint LM. Count how many converge to Band A. This isolates the IC-quality question from the filter question.

If LM converges from 26-37° phi-sweep ICs: filters were the problem, fix them.
If LM doesn't converge: the phi-sweep ICs are genuinely too far for LM, even with the correct ω. Then we need either Sobol replacement OR phi-sweep with larger N.

### 4. Decompose the 37° off-circle floor

For face 0 at epoch 20: truth has face_0 at 5.8° from PAB. The phi-sweep places it at 0° from PAB. What's the closest possible attitude that has face_0 within some tolerance of PAB AND matches the phi rotation? Express the 37° as the geodesic between truth-q0 and the nearest point on the phi-circle, decomposed into:
- Component along face_0 axis (the "face misalignment" — should be ~5.8°)
- Component perpendicular (the "off-circle" component — what actually accounts for 37°)

This will reveal whether the 37° is intrinsic to the phi-sweep approach or fixable with a 2-axis sweep (phi × offset-direction).

### 5. Cohort-wide off-circle floor distribution

For each of the 78 OK seeds (per s032), compute the minimum phi-sweep IC distance to truth-q0 at the correct ω-cell. This tells us:
- What fraction of the cohort has off-circle floor < LM convergence radius (~30°)?
- Is seed 23 a tail case or representative?

If most seeds have off-circle floor < 30°, fix the filter (and tier classification) and stay with phi-sweep. If most have > 30°, Sobol becomes necessary.

## What NOT to do

- **Do not assume Sobol+LM is the answer.** s037b showed it works at one ω-cell. The compute cost at full grid (40k cells × 64 ICs × 43s) is infeasible. The hybrid (Sobol+filters) requires the filters to actually work on Sobol ICs, which was never verified.
- **Do not trust the s037 "architecture validated" framing.** What was validated was: at the correct ω-cell with correct ω-direction, Sobol+LM finds truth. The full cohort-scale pipeline (including ω-search) was not validated.
- **Do not treat the s018a/s018b tier table as definitive.** It was calibrated against population statistics, not seed-specific. Seeds where the brightest face peaks fall at tier boundaries (mag ~6.0, ~7.0, ~8.0, ~9.0) will mis-classify. Seed 23 epoch 20 (mag 6.37) is one such case.

## Concrete artifacts to produce

1. `s038_filter_diagnostics_seed023.py` — runs items 1, 3, 4 above on seed 23. Produces per-IC table: q0_err, geo_score, align_score, surrogate_rho, lm_outcome.
2. `s038_scale_aware_filter.py` — implements per-event geo tolerance, re-runs filter on cached IC pool.
3. `s038_off_circle_decomposition.py` — produces the geometric breakdown of the 37° floor.
4. `s038_cohort_off_circle_floor.py` — sweeps the 78 OK seeds, reports the distribution.

## Trust-but-verify

The s037b "9/64 Band A on seed 23 L1" is real, but the meaning was overstated. It proves Sobol+LM can solve seed 23 at one cell. It does not prove Sobol must replace phi-sweep. The phi-sweep + filter pipeline may still be the right primary architecture with the fixes above; Sobol may be a fallback for seeds with off-circle floor > 30°.

The agent's previous diagnosis ("phi-sweep is the bottleneck, replace it") was a leap from one failed seed to a full architecture change without testing the intermediate hypotheses. The next agent should resist that pattern.
