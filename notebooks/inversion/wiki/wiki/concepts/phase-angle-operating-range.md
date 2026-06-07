---
title: "Phase-Angle Operating Range"
type: concept
sources:
  - "raw/inversion_diagnostics/phase_B_cohort_summary.json"
  - "raw/inversion_diagnostics/phase_B_gap_seeds.json"
  - "raw/inversion_diagnostics/phase_B_descending_seeds.json"
  - "raw/inversion_diagnostics/m126_wrapped/batch_summary.json"
related:
  - "[[phase_B_m048_cohort]]"
  - "[[observation-geometry-sources]]"
  - "[[alignment-cost]]"
  - "[[constraint-poor-regime]]"
  - "[[m048-migration]]"
  - "[[dark-mag-saturation]]"
created: 2026-04-17
updated: 2026-04-17
confidence: medium
---

# Phase-Angle Operating Range

## Definition

The range of sun–satellite–observer phase angles over which the current inversion pipeline (m103 → m115 → m126) reliably recovers ground truth. Derived by combining the m046 11-seed cohort (all at 30–60° phase on a shared observation window) with the m048 Phase-B 8-seed cohort spanning 15°–87° phase on per-seed windows.

## Headline

**Reliable band: ~38–55° phase with ≥ 4 spec peaks.** Outside this band the pipeline has two distinct failure modes:
- Below ~35° phase, **constraint-poor regime** triggers on seeds with few specular peaks (no phase-angle threshold — it's an individual-seed geometry property).
- Above ~65° phase, **alignment-cost flatness** kicks in regardless of peak count.

## Data

### m046 cohort (reference baseline)

11 seeds at 30–60° phase (all on the 2020-02-05 10:00–11:00 UTC shared window):

| cls | count | seeds |
|---|---:|---|
| OK | 5 | 0, 6, 12, 74, 93 |
| PARTIAL | 4 | 14, 24, 27, 33 |
| FAIL | 2 | 36, 46 |

Source: `m126_wrapped/batch_summary.json` (post-Option-A 2026-04-17). Every seed in this cohort is within the 30–60° phase band, and all failures are downstream selection issues, not upstream alignment-cost failures.

### m048 Phase-B cohort (this analysis)

8 seeds at 15°–87° phase on per-seed windows. See [[phase_B_m048_cohort]] for the full result table.

| phase band | N | classes | notes |
|---|---:|---|---|
| 15.3° (seed 024) | 1 | PARTIAL (hifi 0.034) | borderline — inside constraint-rich sub-regime |
| 30.4° (seed 023) | 1 | **FAIL** (hifi 1.60) | constraint-poor (2 spec peaks) |
| 38.9° (seed 049) | 1 | OK | reliable |
| 45.6° (seed 081) | 1 | OK (±X twin) | reliable |
| 50.8° (seed 090) | 1 | OK | reliable |
| 51.7° (seed 091) | 1 | OK | reliable |
| 67.7° (seed 069) | 1 | **upstream-FAIL** | NM best ω_err = 28.8°, 11 spec peaks |
| 87.3° (seed 028) | 1 | **upstream-FAIL** | Pool(24) geo hang, 10 spec peaks |

Per-seed phase ranges (min/max, not just median) are documented in [[phase_B_m048_cohort]]. Seed 091's max phase of 58.8° is the **highest confirmed OK** so far.

## Constraint count × phase angle cross-tabulation

From `scan_m048_constraints_vs_phase.py` (all 100 m048 seeds, 2026-04-17). Spec peaks counted from `mag_hifi` using scipy `find_peaks(-mag, distance=5, prominence=0.3)` then filtered to mag < 9.0:

| bucket | phase range | N seeds | spec peaks (min/q25/med/q75/max) |
|---|---|---:|---|
| very_low | [0, 20°) | 22 | 1 / 3.0 / 5.0 / 7.0 / 12 |
| low | [20, 35°) | 22 | 0 / 3.0 / 5.0 / 7.8 / 16 |
| mid_low | [35, 50°) | 17 | 0 / 2.0 / 4.0 / 9.0 / 14 |
| **mid_high** | **[50, 60°)** | **8** | **5 / 7.2 / 10.0 / 11.2 / 13** |
| high | [60, 75°) | 16 | 2 / 3.0 / 6.0 / 9.0 / 11 |
| very_high | [75, 100°) | 15 | 0 / 2.5 / 6.0 / 7.5 / 11 |

**Finding:** the mid_high band (50-60° phase) is uniquely constraint-rich with median 10 spec peaks. All other bands median 4-6. The m046 cohort's successes are concentrated in or near this sweet spot.

**Finding:** constraint count is NOT strongly correlated with phase band outside the mid_high sweet spot. High and very_high bands still contain seeds with 7-11 spec peaks (seed 58 at 86° has 11; seed 7 at 69° has 10). This enables **natural experiments** that disentangle phase-angle effects from constraint-count effects (see [[upstream-redesign-6dof-surrogate-de]] validation plan).

## Why m046 hid this

m046 uses a single fixed observation window (10:00–11:00 UTC, 2020-02-05) shared across all 100 seeds. Every seed therefore sees the same sun/observer geometry — the same phase angle — only q0 and ω vary seed-to-seed. The phase angle variation across the window is the same ~30° range for every seed.

This means:
1. **Phase-angle edge effects never triggered.** No m046 seed ever saw 70° or 15° phase.
2. **Constraint count varied** seed-to-seed (because different q0, ω rotate different normals into view), but always within a window where the sweet-spot sub-band (50-60°) dominated the LC.
3. **Cohort-level claims** therefore measured pipeline performance inside a sub-regime, not across the full feasible observation space.

[[observation-geometry-sources]] documents this m046 vs m048 distinction. [[m048-migration]] is the response — run the pipeline on per-seed randomly-sampled observation windows spanning the realistic geometry distribution (9–95° phase in m048).

## Implications

### For the pipeline's honest capability claim

Without further work: the pipeline reliably inverts m048 seeds in the **~38–55° phase band with ≥ 4 spec peaks**. That's roughly 25-30% of the 100-seed m048 population (by the cross-tabulation above — mid_low + mid_high sum to 25, minus constraint-poor outliers).

Outside this band:
- Low phase + few peaks → FAIL (constraint-poor)
- High phase → upstream-FAIL (alignment-cost flatness)

### For Phase 3 design

Two paths exist (see [[upstream-redesign-6dof-surrogate-de]]):
1. **Patch m103's symptoms** (geo timeout, enriched checkpoints) and accept the 70%+ upstream-FAIL rate outside the reliable band.
2. **Replace m103's alignment cost** with a full-LC-based cost (surrogate or hi-fi) that doesn't have the flatness + degeneracy pathologies.

The operating range finding is the strongest argument for path 2: patching symptoms would confirm the FAIL rate, not reduce it.

## Open questions

- Does the operating range extend if spec-peak threshold is loosened from 9.0 to 10 or 11? (Would include more dim peaks, changing the constraint count distribution.)
- Does the high-phase failure mode have a sharp boundary (reliable at 55°, failing at 65°) or a gradual degradation? Phase-B only has 2 seeds in the 55-75° range (69 at 67.7° failed, nothing at 60°). Worth one more pilot seed at ~60° to tighten the edge.
- Is there a constraint-count threshold above which high-phase seeds recover? Seeds 28 and 69 had ≥10 spec peaks and still failed upstream, suggesting phase geometry alone is sufficient to break alignment cost even with rich constraints. But N=2 at high phase is small.

## Related

- [[phase_B_m048_cohort]] — the empirical basis for this concept
- [[observation-geometry-sources]] — m046 vs m048 architecture, why phase-angle range wasn't previously exposed
- [[alignment-cost]] — the cost function whose failure modes define the operating range edges
- [[constraint-poor-regime]] — one of the two failure modes
- [[dark-mag-saturation]] — related high-phase concept (surrogate saturates at near-terminator geometries)
- [[upstream-redesign-6dof-surrogate-de]] — proposed architectural fix
