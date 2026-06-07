---
title: "Phase-B m048 Cohort — 8-seed pilot on per-seed observation geometries"
type: experiment
sources:
  - "raw/inversion_diagnostics/invert_m048_seed024/result.json"
  - "raw/inversion_diagnostics/invert_m048_seed091/result.json"
  - "raw/inversion_diagnostics/invert_m048_seed081/result.json"
  - "raw/inversion_diagnostics/invert_m048_seed090/result.json"
  - "raw/inversion_diagnostics/invert_m048_seed049/result.json"
  - "raw/inversion_diagnostics/invert_m048_seed023/result.json"
  - "raw/inversion_diagnostics/m103_hybrid_m048/seed_028/pipeline.log"
  - "raw/inversion_diagnostics/m103_hybrid_m048/seed_069/pipeline.log"
  - "raw/inversion_diagnostics/phase_B_cohort_summary.json"
related:
  - "[[m048-migration]]"
  - "[[observation-geometry-sources]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[m103_hybrid]]"
  - "[[alignment-cost]]"
  - "[[constraint-poor-regime]]"
  - "[[phase-angle-operating-range]]"
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

# Phase-B m048 Cohort — 8-seed pilot on per-seed observation geometries

## Purpose

First pipeline runs against [[observation-geometry-sources|m048 trajectories]] (per-seed `start_et`, 9-95° phase angle range), using the [[m048-migration|Phase-1 plumbing]] that landed 2026-04-17 (`invert.py` single-entry driver chaining m103 → m115 → m126 → wrappedbest → lc_compare). Eight seeds total across three sub-runs.

## Method

Each seed run via `python3 notebooks/inversion/invert.py --seed N --traj-source m048`. Pipeline:

1. **m103** (SKIP_HIFI mode): grid+NM+multi-phi+geo → 26 (q0, ω) candidates saved as `geo_ckpt.npz`
2. **m115**: 3-DOF surrogate-DE over q0 for top-3 ω candidates from m103, then hi-fi validation of 3 best basins
3. **m126**: wrapped polish (3-basin DE → L-BFGS-B polish → hi-fi before/after → keep-min)
4. **wrappedbest refresh + lc_compare**: per-seed winner write-out and truth-vs-estimate LC plot

Per-seed artifacts live at `data/results/inversion_diagnostics/`:
- `m103_hybrid_m048/seed_NNN/` — m103 harvest (multi_phi_ckpt, geo_ckpt, pipeline.log)
- `m115_surrogate_pipeline_m048/seed_NNN/` — DE solutions + hi-fi validation
- `m126_wrapped_m048/seed_NNN/` — polish_ckpt + hifi_ckpt
- `wrappedbest_m048_seedNNN/` + `wrappedbest_m048_seedNNN_lc_compare.png`
- `invert_m048_seedNNN/result.json` — top-level winner summary

Seed selection criteria (see `data/results/inversion_diagnostics/phase_B_*.json` for picks and rationale):
- NOT in m046 baseline `{0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93}` (except seed 24 — same seed index, different obs window on m048)
- `valid_frac > 10%` (LC not saturated over most of window)
- `omega_mag ∈ [0.5, 1.5]` dps ("mainstream" tumble rate)

## Results

All numbers from on-disk `invert_m048_seedNNN/result.json` (authoritative, not transcript-quoted).

| seed | phase_med | phase_range | classification | hifi MSE | q0_err | w_dir_err | w_mag_err | wall (s) | winner source |
|---:|---:|:---:|:---:|---:|---:|---:|:---:|---:|:---:|
| 024 | 15.30° | 10.4–21.6° | **PARTIAL** | 0.0338 | 5.03° | 0.99° | -0.16% | 792 | m126 basin 0 after |
| 091 | 51.65° | 44.5–58.8° | **OK** | 0.0025 | 0.05° | 0.01° | -0.00% | 952 | m126 basin 1 after |
| 081 | 45.55° | 38.5–52.7° | **OK** (±X twin) | 0.0026 | 179.93° | 0.03° | -0.01% | 999 | m126 basin 0 after |
| 090 | 50.80° | 43.7–57.9° | **OK** | 0.0024 | 0.15° | 0.06° | +0.00% | 818 | m126 basin 0 after |
| 049 | 38.89° | 31.9–46.0° | **OK** | 0.0070 | 1.15° | 0.89° | +0.03% | 836 | m126 basin 1 after |
| 023 | 30.39° | 23.6–37.4° | **FAIL** | 1.5973 | 68.29° | 175.29° | +35.37% | 820 | m126 basin 0 after |
| 069 | 67.69° | 60.5–74.8° | **killed mid-run** | — | — | — | — | — | (see diagnosis) |
| 028 | 87.31° | 80.2–94.4° | **upstream FAIL** | — | — | — | — | — | (see diagnosis) |

**Tally:** 4 OK + 1 PARTIAL + 1 FAIL (downstream) + 2 upstream-FAILs on 8 seeds. Note seed 081 is a textbook ±X twin (q0_err ≈ 180°, but ω tight and hi-fi matches truth), valid IS-901 degeneracy per [[twin-degeneracy]].

### Seeds 28 and 69 — upstream failure mechanism

Neither seed reached `invert.py`'s RESULT stage:

- **Seed 028** (87.3° phase): m103 Step 4 geo stage hung in Pool(24). Two sequential launches both hung after ~5 min — all 24 workers dropped from 80% CPU to idle `futex_do_wait` with no `Geo done` emission. Serial rescue (`retry_geo_serial.py` with per-candidate L-BFGS-B) completed 3 candidates in 87 s with all having q0_err > 130°, ω_err > 38° — confirming truth was never in the grid→NM pool to begin with. Killed after 26 CPU-min without completing. See `[[m103_seed028_geo_hang]]`.
- **Seed 069** (67.7° phase): m103 Step 4 about to start Pool(24) geo. Killed pre-geo based on NM top-20 dump showing best ω_err = 28.8° (rank #8) — too far outside m126's ~1-5° polish basin to be recoverable. Geo might not have hung at 68° (cost surface less flat than at 87°), but truth wasn't findable regardless.

Both seeds have **plenty of constraints** (spec peaks: 28→10, 69→11). This is not a constraint-shortage failure — it's an alignment-cost-surface problem specific to high phase angles. See `[[alignment-cost]]#high-phase-flatness`.

### Seed 23 — distinct failure mechanism (constraint-poor)

Seed 23 at 30° phase completed the full pipeline but produced a spectacularly wrong answer: q0_err 68°, **ω direction flipped 175°**, ω magnitude +35% (outside the ±30% grid range). Hi-fi MSE 1.60 — FAIL.

Diagnostic from m103 pipeline.log for seed 23:

```
Peaks: 25 total, 2 spec
|omega| est: 1.034 dps (true: 0.965)
Anchor: ep 235, mag=7.64
Constraints: 1
```

**Only 2 specular peaks.** After picking the brightest as anchor, only 1 alignment constraint remains. With a single constraint, the alignment cost `Σ_ep (1 − max(n·PAB))²` is trivially satisfiable by infinitely many (q0, ω) pairs. The NM top-20 dump confirms: all 20 candidates have `geo_cost` in the 1e-23 to 1e-20 range (indistinguishable from zero), with truth actually present at **rank #8** (q0=172.8°, ω_err=5.1°) but not selected by geo_cost ranking.

This is a different failure mode from seeds 28/69: **truth is findable, but unselectable** via the current ranking metric when constraints are too few to discriminate. See `[[constraint-poor-regime]]`.

## Distilled ceiling picture

Combining the 8 m048 seeds with the 11-seed m046 cohort (shared 30-60° phase window):

| phase band | failure mode | seeds in band | observations |
|---|---|---|---|
| very_low (< 20°) | mostly OK but some PARTIAL when constraints are tight | 024 (PARTIAL) | hi-fi borderline PARTIAL |
| low (20-35°) | **Constraint-poor FAIL possible** when spec peaks ≤ 2 | 023 (FAIL, 2 peaks) | truth in pool, rank #8, geo_cost degenerate |
| mid_low (35-50°) | OK | 049 (OK), 081 (OK ±X twin) | reliable |
| mid_high (50-60°) | OK (constraint-rich sweet spot, median 10 peaks) | 090, 091 (both OK) | reliable |
| high (60-75°) | **Upstream FAIL** — alignment cost surface flattens | 069 (killed, NM best ω_err 28.8°) | grid+NM can't localise truth |
| very_high (> 75°) | Upstream FAIL + Pool(24) geo hangs | 028 (hung twice) | deepest failure regime |

**Reliable operating band: ~38-55° phase with ≥ 4 spec peaks.** This is tighter than the m046 cohort implicitly assumed — m046 was always in the 30-60° range AND had the same obs geometry for every seed, hiding the constraint-count variability.

See `[[phase-angle-operating-range]]` for the synthesis concept.

## What we learned

1. **[[m048-migration|Phase 2 (pilot)]] is effectively complete** across 8 seeds spanning 15° to 87° phase. The Phase-1 plumbing works end-to-end for seeds where the upstream pipeline doesn't collapse.
2. **The m046 cohort's 5 OK / 4 PARTIAL / 2 FAIL was NOT a general performance estimate** — it was a measurement inside the 30-60° band, which happens to include the sweet-spot sub-band (50-60° is constraint-rich). Extending to wider geometries exposes two distinct upstream failure modes that m046 never triggered.
3. **The two failure modes are disentangled, not confounded**. A cross-tabulation (`notebooks/inversion/scan_m048_constraints_vs_phase.py`) of spec-peak count vs phase angle across all 100 m048 seeds shows mid_high band has median 10 peaks vs 4-6 elsewhere, but high and very_high bands still contain individual seeds with 7-11 spec peaks. Seeds 28 and 69 proved the point empirically — both had ≥10 spec peaks and still failed upstream.
4. **The geo-hang symptom is downstream of a deeper problem**. Patching the hang (SERIAL_GEO fallback, per-candidate timeouts) prevents the hang but does NOT rescue the seed — the NM pool doesn't contain truth to begin with, because alignment cost is flat at that phase geometry.

## Decision point exposed by this cohort

The cohort forces a choice before running a 100-seed m048 batch (Phase 3):

- **Quick path**: patch m103 geo timeout + enrich checkpoints. ~3 hr work, then ~24 hr batch. Expect 30-40% of seeds to upstream-FAIL (per the cohort's failure rate).
- **Patient path**: design a surrogate-based upstream that replaces m103 (one-stage 6-DOF DE over (q0, ω) with surrogate-LC cost, basin clustering). Validate on the m046 11-seed cohort + the 3 phase-B FAIL seeds (23, 28, 69) before the batch. See `[[upstream-redesign-6dof-surrogate-de]]`.

The patient path is the current recommendation, gated on cross-cohort validation. Not yet started.

## Script provenance and handoff notes

- Pipeline scripts are the committed `invert.py` chain — NO inline deviations produced any reported hi-fi/q0/ω numbers.
- Seed-selection scripts (`notebooks/inversion/select_phase_seeds_v2.py`, `notebooks/inversion/select_phase_descending.py`) and the sequential runner (`notebooks/inversion/run_phase_B_descending.sh`) are uncommitted at the time of this page's creation (will be committed with session handoff).
- The constraint-vs-phase cross-tabulation (`notebooks/inversion/scan_m048_constraints_vs_phase.py`) is uncommitted at the time of this page's creation.
- Seeds 024, 091, 028 were run in a prior session (2026-04-17 late evening); seeds 081, 069 (killed), 090, 049, 023 in the current session. The numbers in the results table all come from on-disk `result.json` files (seeds that produced them) and on-disk `m103_hybrid_m048/seed_NNN/pipeline.log` files (seeds that didn't).
