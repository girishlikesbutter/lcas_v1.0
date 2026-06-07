---
title: "Constraint-Poor Regime"
type: concept
sources:
  - "raw/inversion_diagnostics/m103_hybrid_m048/seed_023/pipeline.log"
  - "raw/inversion_diagnostics/invert_m048_seed023/result.json"
related:
  - "[[alignment-cost]]"
  - "[[phase_B_m048_cohort]]"
  - "[[m096_exp1_oracle_grid]]"
  - "[[phase-angle-operating-range]]"
  - "[[candidate-selection]]"
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

# Constraint-Poor Regime

## Definition

A seed is **constraint-poor** when the observed light curve contains so few specular peaks that m103's [[alignment-cost]] falls below ~3 non-anchor constraints. In practice: ≤2 spec peaks → 1 constraint after anchor → **the alignment cost becomes trivially satisfiable**.

## Mechanism

m103's alignment cost is:

```
cost(q0, ω) = Σ_constraint_epochs weight · (1 − max_allowed_normals(n · PAB_body(t_ep)))²
```

With a single constraint epoch, this reduces to a scalar "does ω propagate q0 such that SOME allowed body normal points close to PAB at THIS one epoch". A 2-DOF subspace of (q0, ω) satisfies this trivially — you can rotate q0 freely within a 1-parameter family while adjusting ω to compensate, or vary ω's direction and tweak q0.

Empirically, the NM top-20 for such a seed produces alignment costs in the **1e-23 to 1e-20 range** (effectively zero, indistinguishable from numerical noise). Rank #1 by geo_cost is then determined by floating-point roundoff, not physical meaning.

## Empirical evidence (seed 23 @ 30° phase)

m103 pipeline log for seed 23:
```
Peaks: 25 total, 2 spec
|omega| est: 1.034 dps (true: 0.965)
Anchor: ep 235, mag=7.64
Constraints: 1
```

NM top-20 after Step 3 dedup:
```
w#1  gcost=1.75e-23 | q0=172.1° w=41.5°
w#2  gcost=1.07e-22 | q0=130.4° w=80.4°
w#3  gcost=1.09e-22 | q0=141.8° w=58.3°
...
w#8  gcost=1.06e-21 | q0=172.8° w=5.1°   <-- TRUTH (within IS-901 ±X twin)
...
w#20 gcost=3.24e-20 | q0=167.9° w=70.9°
```

Truth is present at **rank #8** with excellent ω-direction error (5.1°) and a likely ±X twin q0. But geo_cost ranking puts rank #1 (w=41.5°) first. Downstream m115/m126 work from that bad ω candidate and produce the FAIL result: hi-fi 1.60, ω dir flipped 175°, +35% ω mag.

**Truth is findable. Truth is unselectable.** This is a selection failure, not a search failure.

## Distinction from high-phase failure

[[alignment-cost]]'s high-phase flatness mechanism looks similar on the surface (NM top-20 ranking useless) but is physically different:
- **High-phase (≥65°):** geo_cost is in a normal range (1e-2 to 1e-1) but the surface is flat enough that L-BFGS-B burns maxfun without moving; multiple wrong candidates end up with nearly-identical cost.
- **Constraint-poor:** geo_cost is pathologically small because there's literally not enough data to penalize wrong answers; any near-satisfying (q0, ω) scores near zero.

Both surface as "rank #1 ≠ truth", but the fix is different in each case.

## Frequency across population

From [[m096_exp1_oracle_grid]] Stage 1 census (100 m046 seeds):
- 10/100 have < 2 specular peaks (pipeline can't run at all)
- 13/100 have ≥ 2 bright (±X-only) constraints
- 87/100 have 0 or 1 bright constraints (dominated by dim medium-band peaks)

Pipeline was tuned on seed 93 (18 spec peaks, 5 bright — rich in constraints). **The "87% minority" is the reality of the population.**

From `scan_m048_constraints_vs_phase.py` (all 100 m048 seeds, 2026-04-17):
| bucket | phase range | median spec peaks |
|---|---|---|
| very_low (< 20°) | 22 seeds | 5 |
| low (20-35°) | 22 seeds | 5 |
| mid_low (35-50°) | 17 seeds | 4 |
| mid_high (50-60°) | 8 seeds | **10** (sweet spot) |
| high (60-75°) | 16 seeds | 6 |
| very_high (75-100°) | 15 seeds | 6 |

Low phase is NOT systematically constraint-poor — seed 17 at 11° has 12 peaks, seed 35 at 12° has 10. But seed 23 at 30° got unlucky with 2. **Low-phase seeds have high variance in constraint count.**

## How to fix

### Within the current pipeline (patch)

A tiebreaker when geo_cost < threshold (e.g. 1e-10): use lo-fi MSE over the full LC to re-rank, since lo-fi sees the whole curve not just alignment constraints. [[m097_candidate_ranking]] showed lo-fi is a good GLOBAL discriminator (truth vs random, 100/100 in micro96 Exp 4) but a poor LOCAL one (near-truth vs nearby grid ω). Constraint-poor is exactly the global-discrimination regime where lo-fi's strength applies.

### Architectural fix

Replace alignment cost with a full-LC-based cost that doesn't collapse on constraint shortage. The surrogate LC MSE (already used in [[m115_surrogate_pipeline]] and motivating [[upstream-redesign-6dof-surrogate-de]]) fits this description — it scores over all 500 epochs regardless of how many peaks exist.

## Related

- [[alignment-cost]] — the cost function this regime exposes as brittle
- [[m096_exp1_oracle_grid]] — 100-seed constraint census
- [[phase_B_m048_cohort]] — empirical demonstration (seed 23)
- [[upstream-redesign-6dof-surrogate-de]] — proposed architectural fix
- [[candidate-selection]] — related problem of selecting truth from a pool when multiple candidates score similarly
