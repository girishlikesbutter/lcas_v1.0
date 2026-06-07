# Play 1 — consolidate the random m048 cohort under ρ-band yield

This directory implements the analytical scaffolding for Play 1 of the 2026-04-29 strategic reframe (see `notebooks/inversion/CURRENT_STATE.md`). Goal: re-rank m103's existing 26-candidate ω-pool by the m133 3-cost union, run m115 on the union top-K=7, report cohort yield as **% seeds with ≥1 (A∪B) basin**.

## Scripts

### `score_union_3cost.py` — Phase 1a (analytical, ~minutes)

Reads cached surrogate-cost scores from `data/results/inversion_diagnostics/rerank_experiment/seed_NNN_{scores,q0polish}.json` and produces, for each of the 22 random m048 seeds (random_25 minus upstream errors 35/42/69), the union of top-3 candidates from each of `{surr_q0polish_mse, surr_autocorr, surr_spectrum}`.

Outputs:
- `data/results/inversion_diagnostics/play1_random_cohort_yield/seed_NNN_union3_topK.json` — per-seed union top-K ω indices
- `data/results/inversion_diagnostics/play1_random_cohort_yield/union3_diagnostic.json` — cohort diagnostics
- `data/results/inversion_diagnostics/play1_random_cohort_yield/union3_summary.md` — human summary

Key diagnostic from a clean run:
- 8/22 seeds have ANY ω in their pool within 5° of truth (m115's bridging radius). This is the Play 1 ranking ceiling.
- The other 14 seeds (47, 51, 79, 84, 89, etc.) have pool min ω-error >5°. They CANNOT be rescued by Play 1 — they need Play 2 (constrained-anchor on sampling-failure seeds) or Play 3 (6-DOF surrogate DE upstream).

### `play1_yield_report.py` — Phase 1d (analytical, ~minutes)

Aggregates m115/m126 outputs into a ρ-band yield report. Works on whatever data exists at the time it's run.

Usage:
```
python3 notebooks/inversion/15_play1_consolidate/play1_yield_report.py --tag baseline
# After Phase 1c batch with union-3cost ranking:
python3 notebooks/inversion/15_play1_consolidate/play1_yield_report.py --tag union3_K7
```

Outputs to `data/results/inversion_diagnostics/play1_random_cohort_yield/REPORT_<tag>.md` and `classification_<tag>.json`.

## What this session shipped

Phase 1a + Phase 1d analytical scripts. They run on existing data with no new compute.

**Critical baseline finding (run 2026-04-29):** Under candidate-set yield (counting all m126 basins, not just the wrappedbest winner), acceptance is already **6/25 (24%)** — vs 4/25 (16%) under single-winner classification (the `batch_m048_v1` baseline). The lift comes from seeds 59 and 67, whose Band-A and Band-B basins are NOT the wrappedbest winner. This is the multi-solution philosophy showing real work without any pipeline changes — the metric change alone surfaces +1A, +1B.

## Phase 1b — m115 patch (NEXT SESSION)

`m115_surrogate_pipeline.py::load_omega_candidates` already supports `M115_SORT_BY=surr_q0polish_mse` reading from `rerank_experiment/seed_NNN_q0polish.json`. The patch needed:

1. Add `'union_3cost'` to the `_VALID_SORTS` tuple.
2. When `sort_by == 'union_3cost'`: read `play1_random_cohort_yield/seed_NNN_union3_topK.json`, take its `union_idx` field, pass through. Order in the union is already the right order (q0polish-first, autocorr-second, spectrum-third — matches m133's robustness ordering on m048).
3. Add `M115_NUM_OMEGA_CANDIDATES` env var. When set, override `n_top` in `run_seed`'s call to `load_omega_candidates`.

Regression test before launching the batch: pick one already-Band-A seed (e.g. seed 91) and verify `M115_SORT_BY=union_3cost M115_NUM_OMEGA_CANDIDATES=7` produces the same Band-A winner as `M115_SORT_BY=geo_cost M115_NUM_OMEGA_CANDIDATES=3`. If it doesn't, the patch is wrong.

## Phase 1c — batch run (NEXT SESSION)

```
M115_SORT_BY=union_3cost M115_NUM_OMEGA_CANDIDATES=7 TRAJ_SOURCE=m048 \
  python3 notebooks/inversion/run_m048_batch.py --seeds 6,7,8,11,16,17,34,45,47,48,51,57,59,64,67,71,78,79,84,89,91,99 \
  --pool-size 3
```

Estimated wall: ~5 hr on Pool(3). Use the existing `run_m048_batch.py` driver (resume markers, batch logging).

## Phase 1d — yield report (NEXT SESSION, after batch lands)

```
python3 notebooks/inversion/15_play1_consolidate/play1_yield_report.py --tag union3_K7
```

Compare `REPORT_union3_K7.md` headline numbers against `REPORT_baseline.md`. The key delta is the ★ line (% with ≥1 (A∪B) basin).

## Acceptance metric (load-bearing)

Per `feedback_rho_band_yield_metric.md`: ρ < 4 (Band A or B) is acceptance. Band C is publishable as an LC fit but flag the partial-state caveat. ρ = √hifi_MSE / 0.05 mag. Per-basin classification, not per-seed-winner — see the 2026-04-17 Roberto report Table 2 for the format.
