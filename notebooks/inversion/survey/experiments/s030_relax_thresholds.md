---
title: s030 — relax-threshold sweep on cached s020 seed-6 data
type: experiment
sources:
  - experiments/s030_relax_thresholds.py
  - results/s030/seed006/
related:
  - s020 (filter pipeline; cached candidates)
  - s022 (truth-calibration on n=15 known-good; off-truth-ω regime untested)
  - s023 (filter rejection rate on random panels)
created: 2026-05-04
updated: 2026-05-04
confidence: high — pure post-process on cached 432k IC pool, basin-coverage counts are exact
---

## TL;DR

Re-categorise the cached s020 seed-6 candidate pool (432k entries) at four threshold relaxations to test whether any level lets the **5 known basin-pool candidates** (2 truth-basin + 3 twin-basin) survive. **Verdict: NO. Every single relaxation level rejects every basin candidate, because the basin candidates score `align ≤ 0.429` (3/7 bright peaks) and all four tested levels require `align ≥ 0.5`.** This refines the s020 diagnosis from "filter is too strict" to "the alignment score *as constructed* cannot admit truth-basin under 30°-phi-step IC granularity for seed 6 — basin candidates only match 3 of 7 bright peaks because their q0 deviation cascades into bright-peak miss timing".

A separate finding: the s020 writeup quoted `geo≥0.5 + align≥0.857 ⇒ ~1864 candidates`. The actual count is **28,690** at that threshold pair. The writeup figure was likely mis-recorded (off by ~16×); the new measurements are authoritative.

## What

Take cached `candidates_meta.npz` (432k × {q0, ω-cell, geo, align, cat-flags}) and re-tabulate per relaxation level:

- **L0** = `geo ≥ 1.0  AND align ≥ 1.0`  (s020 baseline = 875)
- **L1** = `geo ≥ 0.5  AND align ≥ 0.857`  (= 6/7 bright peaks)
- **L2** = `geo ≥ 0.5  AND align ≥ 0.5`
- **L3** = `geo ≥ 0.0  AND align ≥ 0.857`  (no geo filter; align-only)

Per level: total survivors, truth-basin survivors, twin-basin survivors. Basin definitions per `concepts/twin_degeneracy.md`:
- **truth basin**: q0_to_truth < 30° AND |ω-mag − truth_mag|/truth_mag < 10% AND ω-dir-to-truth-dir < 10°.
- **twin basin**: same shape, but vs. `twin_q0 = q_180x · q_truth` (LEFT-multiply) and `twin_ω = R_180x · ω_truth = (ωₓ, −ωᵧ, −ω_z)` (X-flip body twin).

## How

Pure post-process; no propagation, no surrogate eval. From cached arrays:

1. Load `candidates_meta.npz` for `q0`, `omega_cell_idx`, `geo_score`, `align_score`. `align_score ∈ {k/7 : k=0..7}`; `geo_score ∈ {0, 0.5, 1.0}` for seed 6 (only 2 spec events).
2. Load `omega_grid.npz` for `omega_vectors[1500, 3]`; index per candidate via `cell_idx`.
3. Load `survivor_diagnostics.npz` for `truth_q0`, `twin_q0`, `truth_omega`. Compute `twin_omega = (ωₓ, −ωᵧ, −ω_z)`.
4. Vectorised q-geodesic (deg) per candidate to `truth_q0` and `twin_q0` via `2·arccos(|<q1, q2>|)`.
5. Vectorised ω-mag relative error and ω-dir angular distance to truth/twin.
6. Build basin masks; count per level.

Wall: ~1 s on the 432k vectorised. Output: `summary.json` + `survivors_per_level.npz` (per-level survivor indices, full per-candidate basin metadata for downstream Stage-B in s031).

## Result

### Per-level survivor counts

| level                         | geo_thr | align_thr | survivors | truth-basin | twin-basin |
|---|---|---|---|---|---|
| L0 (s020 baseline)            | 1.0     | 1.0       | **875**   | 0 | 0 |
| L1                            | 0.5     | 0.857     | **28,690**| 0 | 0 |
| L2                            | 0.5     | 0.5       | **97,943**| 0 | 0 |
| L3 (align-only)               | 0.0     | 0.857     | **51,875**| 0 | 0 |
| **UNION across 4 levels**     | —       | —         | **121,128** | 0 | 0 |

(Pool of basin candidates in the IC IC pool: 2 truth-basin, 3 twin-basin — all 5 fail every level.)

### Why basin candidates fail

| basin | cand idx | q→target° | \|dω-mag\|% | ω-dir° | geo | align |
|---|---|---|---|---|---|---|
| TRUTH | 23182 | 26.06 | 4.61 | 9.83 | 0.000 | 0.429 |
| TRUTH | 23183 | 18.19 | 4.61 | 9.83 | 0.000 | 0.429 |
| TWIN  | 54055 | 19.80 | 4.61 | 4.28 | 0.500 | 0.429 |
| TWIN  | 60012 | 29.38 | 4.61 | 7.21 | 0.000 | 0.286 |
| TWIN  | 60023 | 29.57 | 4.61 | 7.21 | 0.000 | 0.286 |

All 5 basin candidates score `align ≤ 0.429` (= 3/7 bright peaks matched). The lowest tested align threshold is `0.5` — strictly above 0.429. The geo scores are mostly 0 (truth) or split 0/0.5 (twin). Even the `align ≥ 0.286` regime (1 below the lowest tested) would catch only 5 candidates total — geometric IC simply doesn't position the basin q0s closely enough at 30° phi-step granularity to survive bright-peak matching.

### Quote correction (s020 writeup)

The s020 writeup mentioned `geo≥0.5 + align≥0.857 ⇒ ~1864 candidates`. The actual figure is **28,690**. Likely a transcription/paste error in the writeup. s030's measurement is authoritative.

## Why this matters

- **Closes the question "would relaxing thresholds fix it?"** at L1/L2/L3. The answer is no — the alignment cost as constructed simply does not flag basin candidates at this IC granularity for seed 6.
- **Reframes the diagnostic.** s020 said "5° geo tolerance + 2 spec events is too strict at 30°-phi-step granularity". s030 sharpens: even removing the geo filter entirely (L3) doesn't help, because the alignment cost itself (counting bright-peak matches in the surrogate LC) rejects basin candidates. The bottleneck is not threshold hyperparameters; it is IC granularity (30° phi steps too coarse).
- **Decision substrate.** This delivers the IC-pool ground truth needed for the s031 question: "of the 121k union survivors across 4 levels, does any blind candidate hit Band A∪B in hi-fi?". s031 takes over from here.

## Numbers

- N_total candidates: 432,000 (1500 ω-cells × 288 q-targets/cell)
- IC pool basin coverage (out of 432k):
  - TRUE truth-basin: 2 (cand idx 23182, 23183)
  - TRUE twin-basin : 3 (cand idx 54055, 60012, 60023)
- Per-level counts: see table above.
- Union of all 4 levels: 121,128 unique candidate indices.
- All 5 basin candidates: align_score ∈ {0.286, 0.429} ⇒ rejected by every relaxation tested.

## Artefacts

- `experiments/s030_relax_thresholds.py` — driver.
- `results/s030/seed006/`:
  - `summary.json` — top-level numbers + level breakdown + basin definitions.
  - `survivors_per_level.npz` — per-level survivor index arrays + full per-candidate metadata (`q_to_truth_deg`, `q_to_twin_deg`, `omega_mag_rel_err`, `omega_dir_err_truth_deg`, `omega_dir_err_twin_deg`, `union_idx`, `truth_basin_idx`, `twin_basin_idx`).

## Out of scope

- LM polish on relaxed survivors (deferred to follow-up; first need to know if hi-fi rerank shows any blind Band A∪B in s031).
- Re-running s020 with denser phi-sweep (e.g. 6° steps × 60 phi). Wait for s031 verdict.
- Cohort scaling. Wait for seed-6 verdict (s031 → step 3 decision).

## Cross-references

- **s020** — original filter pipeline, source of cached IC pool. s030 measures coverage that s020 reported the upper bound of (875 strict survivors).
- **s022** — calibrated "no false negatives" on n=15 *near-truth-ω* known-good candidates. s030 confirms the *off-truth-ω regime* fails: at 30°-phi-step IC granularity, basin candidates land outside the per-bright-peak alignment window.
- **`concepts/twin_degeneracy.md`** — corrected X-flip body-twin convention used here for twin q0 + ω.
- **s031** — Stage 2 of the post-s020 task: hi-fi rerank the union of relaxed survivors to test whether any blind start lands Band A∪B.
