---
title: "Pairwise Peak Alignment — Constraint Satisfaction"
type: branch
sources:
  - "notebooks/inversion/wiki/wiki/concepts/brightness_surface_path_matching.md"
related:
  - "[[brightness_surface_path_matching]]"
  - "[[m104_crossing_diagnostic]]"
  - "[[m106_pairwise_vec_ipl]]"
  - "[[alignment-cost]]"
  - "[[grid-search]]"
  - "[[candidate-selection]]"
created: 2026-04-12
updated: 2026-04-16
confidence: medium
---

# Pairwise Peak Alignment — Constraint Satisfaction

## Status: #dead-end

## Question

Can per-peak candidate generation + dynamics-constrained matching find the truth omega where the grid search fails?

## Background

[[m104_crossing_diagnostic]] tested direct omega extraction from peak crossing geometry and correctly identified that the kinematic equation uses Omega_L = R * omega_body (not omega_body), making single-peak extraction impossible. That branch was closed as dead-end.

However, the USER identified a different formulation: per-peak CONSTRAINT SATISFACTION. Each peak constrains a manifold of (q0, omega0) via alignment, and dynamics links the constraints across peaks. Two peaks with known lobe assignments give an exactly-determined system (3 remaining unknowns, 3 constraints from peak 2).

## Key insight

The approach does NOT extract omega0 from peaks. Instead:
1. Each peak constrains R(t_k) to a 1-DOF family (twist psi about the lobe normal)
2. For a given omega0 candidate, delta-q factorization determines q0 from psi
3. At a second peak, the propagated attitude must also align — this is a HARD constraint
4. Multiple peak pairs overdetermine and prune false positives

## POC results [inline] (2026-04-12)

Tested on grid-failure seeds 28 and 44 with 2500 candidates (500 dirs x 5 mags) via inline strategist bash testing. These numbers were NOT produced by `m105_pairwise_ipl.py` (that script only implements Stage 1; Stage 2 full-observation scoring was done inline):

| Seed | 3-pair intersection | Truth: peaks aligned | Best FP: peaks aligned |
|------|---------------------|---------------------|----------------------|
| 28   | 2501 -> 18 (139x)   | 11/14               | 6/14                 |
| 44   | 2501 -> 20 (125x)   | 11/11               | 6/11                 |

Truth has near-perfect alignment at all bright peaks; no false positive exceeds 6. The approach finds truth where the grid search fails.

**Crossing speed was NOT used.** Pure alignment at peak pairs + dynamics was sufficient.

## Limitation: requires cross-family bright peaks

Pre-analysis of the 5 grid-failure seeds from m103 revealed:

| Seed | Bright peaks | Families | Cross-family pairs | Viable? |
|------|-------------|----------|-------------------|---------|
| 1    | 2           | X only   | 0                 | **No** — slow tumbler, too few peaks |
| 11   | 11          | X, Y, Z  | 33                | Yes |
| 28   | 14          | ED, WD, Y, Z | 60           | Yes (POC confirmed) |
| 44   | 13          | ED, WD, X, Y, Z | 45        | Yes (POC confirmed) |
| 46   | 8           | X, Y, Z  | 16                | Yes — fewer peaks, weaker discrimination |

Seed 1 (0.39 deg/s) produces only 2 bright peaks, both from the same lobe family. The pairwise approach fundamentally cannot help here — it requires ≥2 bright peaks from different lobe families.

## m106 — Negative result (2026-04-12)

m106 ran vectorized pairwise alignment on 4 seeds (11, 28, 44, 46) with 10K candidates (2000 dirs × 5 mags). **Truth FAILED oracle Stage 1 for ALL seeds** — the nearest grid candidate (1° from truth) produced alignment distances of 12-29° at peak pairs, far exceeding the 10° threshold.

**Key diagnostic (seed 28):**
- Exact truth omega → 14/14 peaks aligned (perfect)
- Nearest grid candidate (1.16° dir error, 0% mag error) → 6/14 peaks aligned

**Root cause:** Attitude quaternion error accumulates over the observation timespan (~3600s). A 1° omega direction error produces ~60-90° quaternion deviation at late peaks. The pairwise constraint is too sensitive for grid-level omega precision.

**The POC inline results (139x reduction, truth 11/14) were obtained with exact or near-exact truth omega injected.** This confirms the math is correct but the practical applicability at grid resolution is nil.

## Why this is a dead end

The pairwise approach requires omega precision of ~0.01-0.1° to discriminate truth from false positives at all peak epochs. The grid has ~1-3° spacing. Even NM refinement (which achieves ~0.1-0.5° precision) may not be precise enough for late peaks. And after NM, the bottleneck is **selection** (choosing the right candidate from the pool), not **finding** — the pairwise filter adds no value at the stage where it could work.

## Lessons

1. **Delta-q sensitivity scales with observation timespan.** Constraints at early peaks (within ~100 epochs) are robust; constraints at late peaks (300+ epochs) are extremely sensitive to omega errors.
2. **Always test with grid candidates, not truth.** The POC's implicit use of truth omega made the approach appear to work when it fundamentally doesn't at practical precision.
3. This is a textbook example of the provenance gap the research-loop skill now guards against.
