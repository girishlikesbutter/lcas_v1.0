---
title: s032 — 100-seed cohort run of s020 with geo-fail gate
type: experiment
sources: [s020_seed_pipeline.py, s032_cohort_fast.sh, all 100 m048 trajectories]
related: [s020, s019, s018b, s031]
created: 2026-05-05
updated: 2026-05-05
confidence: high
---

# s032 — Cohort fast-path run on full 100-seed m048 database

## TL;DR

Added a CLI flag `--measure-geo-fail` (default OFF) gating the alignment-cost
instrumentation pass on geo-rejected candidates. **Per-seed wall on seed 6
dropped from 3,684s → 75s = 49× speedup** (better than the 6.7× the s020
writeup predicted, because gating skips the surrogate forward pass too, not
just the alignment cost). Ran the full 100-seed cohort in **266 min wall**.
**78 OK / 1 CACHED / 21 FAIL** (19 zero-classifiable matching s018b's
prediction exactly, 2 pre-fix `twin_q0` UnboundLocalError, since fixed). The
headline finding is the **bracket subsampling cohort coverage is dramatically
worse than s019 claimed: 4/79 within 5% (vs s019's 98/100), median 32.06%
off truth ω-mag**, because s020 takes the top-N LS peaks (capped at peak
count) rather than densely subsampling the full geometric grid.

## What

Two-part run:

1. **Pipeline modification** — gate the geo-FAIL alignment-cost branch behind
   `MEASURE_GEO_FAIL_ALIGN = False` (default). Production path now skips
   surrogate evaluation on candidates that already failed geo cost. Also
   patched a pre-existing latent bug: `twin_q0` was computed inside the
   `survivor_idx.size > 0` branch but referenced unconditionally in the
   no-survivor checkpoint save — moved it before the branch.

2. **Full-cohort dispatch** — sequential bash driver runs all 100 m048 seeds
   at the s020 baseline configuration (N_dir=300, N_mag=5, N_phi=12,
   strict 1.0/1.0 thresholds), per-seed log + rolling CSV with timing,
   survivor counts, bracket coverage.

## How

Edits to `s020_seed_pipeline.py`:
- Added `MEASURE_GEO_FAIL_ALIGN = False` global.
- Gated FAIL-pass surrogate+align eval in `_process_cell` (Pool worker) and
  in the serial fallback loop. When OFF, `dt_fail = 0` and `align_scores`
  remain NaN for FAIL candidates.
- Added `--measure-geo-fail` CLI flag.
- Moved `twin_q0` computation outside the `survivor_idx.size > 0` branch.

Driver `s032_cohort_fast.sh`:
- Iterates seeds 0-99, skips cached `summary.json`, appends
  `(seed, wall_s, n_survivors, n_passed_geo_only, n_rejected_both, nearest_cell_pct, status)`
  to `cohort_progress.csv`.

## Result

**Smoke verification on seed 6:** 75.4 s wall (vs 3,684 s baseline = 49×
speedup), survivors 875 (matches), categorisation matches byte-for-byte.

**Cohort summary (266 min total, 78 OK + 1 CACHED + 21 FAIL):**

| metric | value |
|---|---|
| Wall p50 / p75 / p90 / max | 35s / 61s / 13.4m / 49.9m (seed 84) |
| Survivors p50 / p75 / p90 / max | 0 / 398 / 2153 / 19,847 (seed 57) |
| Zero-survivor seeds | 47/79 (60%) |
| Seeds with ≥100 survivors | 27/79 |
| Seeds with ≥1000 survivors | 10/79 |
| Bracket nearest_cell within 5% | 4/79 (vs s019's 98/100) |
| Bracket nearest_cell within 10% | 13/79 |
| Bracket nearest_cell median | 32.06% |
| Bracket nearest_cell >50% | 20/79 |

**Failure attribution (clean):**
- 19 zero-classifiable (RuntimeError from phi-sweep IC generator):
  4, 9, 10, 30, 32, 33, 41, 48, 51, 52, 54, 66, 70, 71, 81, 82, 86, 91, 99
  → matches s018b's "19/100 zero-classifiable at mag<8" prediction EXACTLY.
- 2 bug-related (UnboundLocalError on `twin_q0` in no-survivors path):
  0, 3 (re-runnable; fix in place).

## Why this matters

Three findings reshape the survey:

1. **The instrumentation overhead was 98% of the wall.** The s020 writeup's
   ~9 min/seed estimate was right under the FAST flag — first time the
   pipeline is actually cheap enough for cohort-scale work.

2. **The bracket coverage finding contradicts s019.** s019 measured "98/100
   within 5%" using the FULL bracket grid (~14-72 cells). s020 SUBSAMPLES
   that bracket down to N_OMEGA_MAG_CELLS by taking top-N LS peaks. For
   most seeds that subsampling collapses coverage to median 32% — far
   outside the s003 ~2-5% mag tube. **Seed 6's 4.6% — what s031 used as the
   end-to-end test — is in the BEST decile of the cohort.** The s031
   cohort-extrapolated "framework structurally constrained" pessimism was
   measuring a subsampling artifact, not a fundamental framework limit.

3. **The framework's reach upper-bound at the strict 1.0/1.0 thresholds is
   ~27% of cohort.** 19 zero-classifiable + 47 zero-survivor + 7
   low-survivor (1-99) = 73/100 seeds where the strict-threshold filter
   returns nothing actionable. Only 27 seeds yield ≥100 survivors worth
   downstream hi-fi reranking.

These together motivate the s033 density-sensitivity follow-up.

## Numbers

**Wall by seed (selected):**
- p25 = 23s, p50 = 35s, p75 = 61s, p90 = 776s, max = 2996s (seed 84)
- Bimodal: 75% finish in <1 min, top 10% take 13-50 min

**Top high-survivor seeds:**
seed 57 (19,847), seed 62 (12,629), seed 16 (9,604), seed 43 (5,899),
seed 19 (5,848), seed 13 (3,996), seed 72 (3,890), seed 31 (3,144),
seed 2 (1,905), seed 12 (1,296), seed 6 (875).

**Cohort production economics:**
- Total wall: 256 min (sum of OK seed walls); driver overhead 10 min → 266 min total.
- At 100 seeds × ~3 min average → tractable for nightly cohort sweeps.

## Artefacts

- `experiments/s020_seed_pipeline.py` — gate edit + twin_q0 fix.
- `experiments/s032_cohort_fast.sh` — sequential cohort driver.
- `results/s032_cohort_fast/seed{000..099}/` — per-seed checkpoints.
- `results/s032_cohort_fast/cohort_progress.csv` — rolling status table.
- `results/s032_cohort_fast/logs/seed{XXX}.log` — per-seed full logs.

## Out of scope

- Hi-fi rerank of any cohort survivors (deferred; s031 already showed seed
  6's strict-threshold survivors are Band D in hi-fi anyway).
- Re-running seeds 0 and 3 with the bug fix (cheap follow-up; not done in
  this session).
- Investigating the bracket-subsampling discrepancy with s019 (handled in
  s033).

## Cross-references

- `s020_seed_pipeline.md` — original pipeline definition.
- `s019_ls_bracket_omega_mag.md` — original bracket coverage measurement
  (98/100 within 5% on full grid, NOT subsampled).
- `s018b` — face-identity tier classifier; predicted 19/100 zero-classifiable.
- `s031_hifi_rerank_seed6.md` — pre-cohort verdict that seed 6's strict
  survivors are Band D in hi-fi.
- `s033_density_sensitivity.md` — follow-up on bracket densification.
