---
title: "s038: Filter diagnostics on seed 23 — why phi-sweep + LM lost the basin"
type: experiment
sources:
  - experiments/s038_handoff_diagnosis_gaps.md
  - experiments/s036_multi_seed_pilot.md
  - experiments/s037_sobol_lm_pilot.md
  - results/s036_multi_seed_pilot/seed023/
  - results/s037b_sobol_lm_pilot/seed023_L1_result.json
related:
  - s020
  - s034
  - s036
  - s037b
created: 2026-05-06
updated: 2026-05-06
confidence: definitive
---

## TL;DR

The s036 → s037 narrative ("phi-sweep is the bottleneck, replace with Sobol") was
a premature jump but the underlying empirical instinct was right: **at the bracket
ω-cell on seed 23, LM polish from phi-sweep ICs converges to 0/107 Band A — even
from the IC at q0_err = 2.61° to body-twin**. Sobol N=64 + LM at the SAME cell
yields 9/64 Band A (s037b L1). The phi-sweep IC pool's q0-distribution clusters on
phi-circles around bright peaks; on seed 23, those circles do not pass close to
truth-q0, and **all 107 LM polishes get stuck in non-truth local minima** (clusters
at q0_err ~21°, ω-dir ~28°, ρ_final ~32-36).

The two findings together:
1. **The strict 1.0/1.0 filter is structurally incompatible with phi-sweep IC
   architecture** (geo-score capped at 0.333 — IC architecture provides ≤1/3 spec
   matches; truth provides 3/3 by physics).
2. **Even bypassing the filter, phi-sweep ICs cannot bridge to truth on seed 23.**
   Phi-sweep IC quality is **seed-dependent**: on seed 89 it lands ICs in the LM
   basin (s034 24/50 Band A); on seed 23 it doesn't. Sobol N=64 + LM is robust
   across seed classes (8-9/64 Band A on 23/28/89).

The right framing for the user's "candidates → cost filters → NM refinement" pipeline:
**phi-sweep + cost filters are a CELL filter (cheap, geometry-aware), not an IC
primitive.** Use them to identify ω-cells inside the LM basin; use Sobol N=64 as
the q0-IC primitive at surviving cells. Both layers retained, but with their roles
clarified.

## What

Two-stage diagnostic on the cached s036 seed-23 IC pool, at the near-truth ω-cell.

**Stage A** (sub-minute, no LM): score every one of the 528 phi-sweep ICs at the
near-truth ω-cell on geo + align + pre-LM surrogate-MSE. Tabulate vs q0_err to
truth and to body-twin.

**Stage B** (Pool(8), ~12 min): bypass the filter; run LM polish on 107 ICs
(top-60 by q0_err_min ∪ top-60 by pre-LM ρ; 13 overlap). Tabulate Band A∪B yield.
Compare directly against s037b L1 (Sobol N=64 + LM at the same ω-cell, 9/64 Band A).

## How

`experiments/s038_filter_diagnostics_seed023.py`. Single script with `--no-lm` flag
for Stage A only. Reuses cached `candidates_meta.npz` / `q_target_pool.npz` /
`omega_grid.npz` from `results/s036_multi_seed_pilot/seed023/`. Forward function
matches s034 (post-fix `Rotation.from_quat().as_matrix()` returns R_i2b directly,
no transpose). LM uses `scipy.optimize.least_squares(method='lm', max_nfev=200)` —
matches s037b for direct comparison.

## Result

### Stage A — IC pool structure at the near-truth ω-cell

| Metric                              | Value                          |
|-------------------------------------|--------------------------------|
| ω-cell index (mag × dir)            | 21804 (mag bin 10 × dir 1804)  |
| ω-mag offset from truth             | 5.64%                          |
| ω-dir offset from truth             | 2.26°                          |
| 528 ICs: q0_err_truth               | min 10.95°, median 132.77°     |
| 528 ICs: q0_err_twin                | **min 2.61°**, median 131.96°  |
| 528 ICs: q0_err_min (truth or twin) | **min 2.61°**                  |
| Geo-score histogram                 | **328 at 0.000, 200 at 0.333, 0 at any higher value** |
| Align-score                         | NaN (s020 short-circuits align eval on geo-rejects) |
| Pre-LM ρ                            | min 42.40, median 47.25        |
| Truth ρ (surrogate noise floor)     | 0.445                          |
| ICs within q0_err_min < 30°         | 10/528 — 7 with geo=0, 3 with geo=0.333 |

**The filter caps geo-score at 0.333 = 1/3 spec events matched** because each
phi-sweep IC anchors on exactly ONE bright peak (seed 23 has 28 classifiable peaks,
only 3 of which are spec events). Truth gets 1.0 because the actual physical state
aligns with all 3 spec events naturally. **The 1.0/1.0 strict threshold is a
truth-calibrated score, but the IC primitive can never reach it** — for any phi-sweep
IC, geo is ≤ (n_anchored_spec_events / n_total_spec_events).

### Closest-to-truth/twin ICs (all rejected by filter)

| ic | peak | face | phi | tier | q0_t°  | q0_w° | min° | geo  | ρ_pre |
|---:|-----:|-----:|----:|-----:|-------:|------:|-----:|-----:|------:|
|  60|   84 |    7 |   0 |    3 | 177.76 |  2.61 | 2.61 | 0.000| 42.45 |
|  49|   84 |    6 |   1 |    3 |  10.95 |177.76 |10.95 | 0.000| 43.41 |
|  50|   84 |    6 |   2 |    3 |  22.26 |178.02 |22.26 | 0.000| 43.17 |
| 255|  262 |    3 |   3 |    2 |  24.85 |155.61 |24.85 | 0.333| 44.38 |
| 230|  215 |    9 |   2 |    2 |  25.46 |163.67 |25.46 | 0.000| 52.16 |
| 241|  262 |    2 |   1 |    2 | 155.79 | 27.22 |27.22 | 0.333| 45.15 |

The closest IC (q0_err = 2.61° to twin) is anchored on peak 84, a non-spec bright
peak. The closest spec-anchored IC is at q0_err = 24.85° (twin).

### Why even spec-anchored ICs are ~25° off truth at this cell

A second test (denser N_phi=60 phi-sweep on the 3 spec events only, 960 ICs) showed
q0_err_min floor = 14.99° (to twin). At this ω-cell, even infinitely dense phi-sweep
on spec events cannot place ICs closer than ~15° to twin, because the bracket cell
is 2.26° / 5.64% off truth-ω — that perturbation alone shifts the trajectory enough
that spec-event-anchored ICs no longer pass through truth-q0.

### Stage B — LM polish, filter bypassed (107 ICs, Pool(8) wall 12.1 min)

**0/107 Band A. All 107 in Band D. Best ρ_final = 32.31.**

| Stat                          | Value             |
|-------------------------------|-------------------|
| Panel size                    | 107 (top-60 q0 ∪ top-60 ρ_pre, 13 overlap) |
| Band A (ρ < 2)                | **0**             |
| Band B (2 ≤ ρ < 4)            | 0                 |
| Band C (4 ≤ ρ < 8)            | 0                 |
| Band D (ρ ≥ 8)                | 107               |
| ρ_final: min                  | 32.31             |
| ρ_final: median               | 39.64             |

LM converges to several non-truth attractors:
- Cluster A (n≈4): q0_fin → 21° to twin, ω-dir 27.8°, ω-mag +1.6%, ρ ≈ 32-36
- Cluster B (n≈4): q0_fin → 115° to truth, ω-dir 26°, ω-mag −10%, ρ ≈ 35
- Cluster C (n≈4): q0_fin → 44° to twin, ω-dir 28°, ω-mag +1.3%, ρ ≈ 35
- Cluster D (n≈4): q0_fin → 63° to truth, ω-dir 30°, ω-mag +3.4%, ρ ≈ 35
- Multiple LMs from IC #60 (closest to twin) end at q0_err 15-16° with ρ ≈ 41 (basically static — the LM didn't move from the twin neighborhood, it just couldn't reduce residual)

**No LM polished to truth-q0 (q0_err < 1°) or twin-q0 (q0_err < 1°) basins from
any phi-sweep IC at this cell**, despite 6 ICs starting within 30° of truth/twin.

### Direct comparison with Sobol N=64 + LM at the same ω-cell (s037b L1)

| Architecture                       | n   | Band A | Band B | ρ_min | ω → truth |
|------------------------------------|----:|-------:|-------:|------:|----------:|
| Phi-sweep + LM (this s038, panel)  | 107 |   **0** |   0    | 32.31 | NO        |
| Sobol N=64 + LM (s037b L1)         |  64 |   **9** |   0    |  0.44 | YES (+0.0003%) |

s037b succeeded on seed 23 from random q0 + bracket-ω. **The IC primitive — not LM
quality, not the bracket cell — is what differs.** Phi-sweep clusters q0 on
phi-circles anchored on bright peaks; on seed 23 those circles don't pass close to
truth, and LM gradients from phi-sweep ICs systematically point into non-truth
local minima.

### Cohort context (s037b across 23/28/89, plus s032 cohort distribution)

- Sobol+LM is **robust across seed classes**: seed 23 → 9/64 Band A, seed 28 →
  6/64 Band A (narrow basin recoverable), seed 89 → 1/64 Band A at L1 (5.6% off).
- Phi-sweep+LM **on seed 89 at 1.91% off** (s034) → 24/50 Band A from a
  surrogate-MSE-selected top-50 of 4170 IC pool. Phi-sweep is BETTER than Sobol on
  seed 89 specifically, presumably because seed 89 has many ICs whose anchored
  phi-circles happen to pass close to truth-q0.
- s032 cohort distribution: only **4/79 seeds** have nearest bracket cell ≤ 5%
  off truth-ω-mag (seed 23 is one of them). Median nearest_cell_pct = 32%. So the
  bracket coverage problem is independent of the IC primitive question.

## Why this matters

1. **Phi-sweep IC primitive is seed-dependent.** It works when truth-q0 lies near
   a phi-circle anchored on a bright peak (seed 89). It fails when it doesn't
   (seed 23). The s011 9/10 Sobol pilot at truth-ω validated Sobol's robustness;
   the s034 24/50 phi-sweep success on seed 89 validates phi-sweep's potential
   when the geometry happens to align. Neither is a universal solution.

2. **The strict 1.0/1.0 filter is structurally incompatible with phi-sweep ICs.**
   Even relaxing geo to 1/3 (200/528 ICs admitted on seed 23) doesn't help here,
   because the LM polish itself fails — relaxation isn't the binding fix.

3. **The right architectural framing**: phi-sweep + filter pipeline is a
   **CELL FILTER**, not an IC primitive. It identifies ω-cells where geometry
   admits truth-like ICs (cheap, ~75 s/seed per s032). At surviving cells, use a
   robust IC primitive (Sobol N=64) for LM polish. The two layers compose:
   - **Cell filter (s020 architecture)**: bracket × Fibonacci × phi-sweep × geo + align.
     Cells that admit ≥1 candidate scoring positive on either filter are "interesting".
   - **Q0 search (s037 architecture)**: at each interesting cell, Sobol N=64 + LM polish.
   - **ρ-band gate (s035 hi-fi rerank)**: hi-fi confirm Band A∪B.

4. **The previous agent's "switch to Sobol" was right empirically but wrong on
   the diagnosis.** They claimed phi-sweep ICs were "too far from truth" — but
   IC #60 at 2.61° to twin was IN the IC pool. They claimed "LM cannot bridge 37°"
   — but the actual failure is LM gets stuck in non-truth local minima even from
   2.61°. The fix is not "throw away the filter pipeline"; the fix is "use Sobol
   for the IC primitive, keep the cell filter."

5. **The compute-cost picture for the corrected hybrid**:
   - Cell filter (s020 fast-path, geo-only, MEASURE_GEO_FAIL_ALIGN=False): 75 s/seed cohort.
   - At surviving cells (varies seed-to-seed; typical p90 ~2k cells): Sobol+LM
     at Pool(8) ~6 s/IC × 64 ICs = 6 min/cell. 2k cells × 6 min/cell = 200 hr/seed.
     **Still infeasible.** Need a cell-level filter SHARPER than current geo-fast-path.
   - Possible refinement: rank cells by survivor count + by best ρ_pre (cheap surrogate
     eval per cell) — only Sobol+LM at top-K cells. K ~10-50 per seed.

## Numbers

| Stage | Wall              | Output                          |
|------:|-------------------|---------------------------------|
| A     | 0.38 min          | `stage_a_ic_scores.npz` (528 ICs × 16 fields) |
| B     | 12.1 min Pool(8)  | `stage_b_lm_panel.json` (107 LM polishes) |

## Artefacts

- `experiments/s038_filter_diagnostics_seed023.py`
- `results/s038_filter_diagnostics_seed023/stage_a_ic_scores.npz`
- `results/s038_filter_diagnostics_seed023/stage_b_lm_panel.json`
- `results/s038_filter_diagnostics_seed023/run.log`

## Out of scope

- Cohort-wide phi-sweep-vs-Sobol convergence comparison at near-truth ω-cells
  (s039 candidate). Would need s032 cohort, near-truth cell selection, paired
  Stage-B-style LMs.
- Optimal Sobol-N for cohort recovery (8 vs 16 vs 32 vs 64).
- Cell pre-filter design: surrogate-MSE-rank cells before launching Sobol+LM.
- Random-panel filter validation under the relaxed (1/3) geo threshold.

## Cross-references

- s011: Sobol-Shoemake N=64 at truth-ω → 9/10 cohort recovery; the original
  evidence that Sobol works as an IC primitive.
- s020: original filter pipeline (1.0/1.0 strict).
- s021/s023: filter validation against random panels (truth scores 1.0, 88/100
  zero random survivors). Validation was correct at 1.0/1.0 but doesn't bound the
  IC-architecture-vs-filter compatibility re-discovered here.
- s034: LM polish recipe; works from ρ < ~15 → Band A on seed 89.
- s036: failed multi-seed pilot — diagnosis was incomplete; this is the corrected
  diagnostic.
- s037b: Sobol N=64 + LM at the same near-truth ω-cell, 9/64 Band A. Direct
  comparator for the Stage B numbers.
- s003: ω-mag tube width — explains why even spec-anchored ICs floor at 15° q0_err
  at the L1 (5.6% off) cell.
