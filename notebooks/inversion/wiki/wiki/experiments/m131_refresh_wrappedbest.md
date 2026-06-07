---
title: "m131 — Refresh wrappedbest dirs post-Option-A (plumbing)"
type: experiment
sources: ["raw/inversion_diagnostics/wrappedbest_seed000/result.json", "raw/inversion_diagnostics/wrappedbest_seed006/result.json", "raw/inversion_diagnostics/wrappedbest_seed012/result.json", "raw/inversion_diagnostics/wrappedbest_seed014/result.json", "raw/inversion_diagnostics/wrappedbest_seed024/result.json", "raw/inversion_diagnostics/wrappedbest_seed027/result.json", "raw/inversion_diagnostics/wrappedbest_seed033/result.json", "raw/inversion_diagnostics/wrappedbest_seed036/result.json", "raw/inversion_diagnostics/wrappedbest_seed046/result.json", "raw/inversion_diagnostics/wrappedbest_seed074/result.json", "raw/inversion_diagnostics/wrappedbest_seed093/result.json"]
related: ["[[m126_wrapped_pipeline]]", "[[m125_keep_better_inline]]", "[[m124_hifi_validate]]", "[[m132_solution_count_by_band]]"]
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

# m131 — Refresh wrappedbest dirs post-Option-A

Plumbing-only experiment. No hi-fi regeneration. Rebuilds the 11 `wrappedbest_seed{NNN}/` dirs that lc_compare consumes, so their winners and cached predicted LCs reflect the post-Option-A data.

## Why this was needed

The pre-existing `m126_wrapped/wrappedbest_seed*/` dirs were generated 2026-04-16, BEFORE Option A (Bug 2 fix, commit `b906691`, 2026-04-17). Their winners were stale for 7/11 seeds — notably seed 12 pointed at `m126_basin0_before` (hi-fi 0.327) when the post-fix winner is `m126_basin0_after` at 0.00265 (124× better). Their cached `pred_lc.npy` files would also have been computed on whichever observation window the pre-fix scripts used.

Any lc_compare plot generated from those dirs would silently misrepresent the wrapped pipeline's current performance.

## Method

For each of 11 seeds:

1. Determine the wrapper winner = basin with min `hifi_wrapped` across the 3 polished basins.
   - Seeds 0, 6, 12, 24, 33, 36: read basins from `m126_wrapped/seed_NNN/result.json` (all 3 basins `polish_helped=true`, so winner state is post-polish `q0_after`, `omega_after`).
   - Seeds 14, 27, 46, 74, 93: read basin labels from `m125_keep_better/summary.json` (min `hifi_best_wrapped`), then look up `final_q0_wxyz` and `final_omega_rad` in `m124/summary.json` `per_candidate` entries.
2. Retrieve the corresponding cached hi-fi LC — zero regeneration:
   - m126 seeds: `seed_NNN/hifi_ckpt.npz['hifi_mags_after'][basin_idx]` (500 points, correct-window).
   - m124 seeds: `m124/hifi_results.npz['hifi_mags'][idx]` where `idx` is selected by matching `(seeds=seed, kinds=b'polished', tag=f'_seed_{seed:03d}_{label}')`.
3. Write `data/results/inversion_diagnostics/wrappedbest_seed{NNN}/{result.json, pred_lc.npy}` in the schema `lib/lc_compare.py` expects (`winner.q0_wxyz`, `winner.w0_rad`, `winner.w0_dps`, `winner.q0_err`, `winner.w0_err`, `winner.w_mag_err_pct`, `winner.hifi`).

Error metrics computed in-script (quaternion inner-product for q0_err, acos of ω dot for signed w_dir_err, relative magnitude diff for w_mag_err_pct). ±X twin detection via `q0_err > 170° AND w_dir_err < 90°`; flipped-ω detection via `w_dir_err > 90°`.

## Results — 11-seed winner table (post-fix)

| seed | source | q0_err | w_dir | w_mag% | hi-fi | class | note |
|-----:|---|-------:|------:|-------:|------:|:-----:|------|
| 0  | m126 basin_1 after | 0.54°   | 0.02° | +0.01% | 0.00367 | OK | truth-adjacent |
| 6  | m126 basin_0 after | 179.59° | 0.20° | +0.01% | 0.00291 | OK | ±X twin (twin + truth within 3e-4) |
| 12 | m126 basin_0 after | 0.14°   | 0.03° | -0.01% | 0.00265 | OK | truth-adjacent |
| 14 | m124 basin_1 polished | 178.30° | 0.29° | -0.03% | 0.01565 | PARTIAL | ±X twin |
| 24 | m126 basin_0 after | 179.85° | 1.35° | -0.00% | 0.01363 | PARTIAL | ±X twin |
| 27 | m124 basin_1 polished | 1.39° | 1.12° | +0.11% | 0.02221 | PARTIAL | truth-adjacent |
| 33 | m126 basin_0 after | 98.53° | 161.54° | **-67%** | 0.07965 | PARTIAL | [[omega-sign-degeneracy]] |
| 36 | m126 basin_0 after | 169.59° | 8.96° | +0.12% | 0.39371 | FAIL | upstream ω-dir 10° off |
| 46 | m124 basin_2 polished | 179.54° | 0.26° | -0.02% | 0.14160 | FAIL | borderline FAIL (±X-like) |
| 74 | m124 basin_0 polished | 1.34° | 0.25° | +0.03% | 0.01091 | PARTIAL | truth-adjacent (borderline OK) |
| 93 | m124 basin_1 polished | 179.66° | 0.35° | -0.01% | 0.00730 | OK | ±X twin |

Three of the 11 winners are ±X twins (6, 14, 93), two are truth-adjacent OK (0, 12), one is flipped-ω (33). Under [[multi-solution-philosophy]] all of these are valid — the wrapper picks whichever basin has the lowest hi-fi MSE, and twin vs truth gives physically identical LCs. Seed 6 twin-vs-truth difference is 3e-4 hi-fi: essentially coin-flip.

## Classifications vs prior claim

Prior (EXPERIMENTS.md): 5 OK / 4 PARTIAL / 2 FAIL.

Verified from this script: 4 OK outright (0, 6, 12, 93 — all below 0.01) + 1 OK-borderline (74 at 0.01091) + 5 PARTIAL + 2 FAIL. The "5 OK" claim is slightly generous; 74 technically crosses into PARTIAL.

## Correction of prior table

The previous session's text said seed 6 winner was basin_1 at q0_err 0.40° (truth-adjacent). The actual winner is basin_0 at q0_err 179.59° (±X twin) — basin_0's hi-fi of 0.00291 beats basin_1's 0.00300 by 9e-5. Twin and truth are physically equivalent so this doesn't change the success assessment, but the printed q0_err was wrong.

## Artifacts

- Script: `notebooks/inversion/12_brightness_surface/m131_refresh_wrappedbest.py`
- Rebuilt: `data/results/inversion_diagnostics/wrappedbest_seed{NNN}/{result.json, pred_lc.npy}` × 11
- Plots: `data/results/inversion_diagnostics/wrappedbest_seed{NNN}_lc_compare.png` × 11 (generated by `lib/lc_compare.py` with truth rendered as blue dots, markersize 3)

## What this enables

Downstream analyses (solution-count ([[m132_solution_count_by_band]]), per-seed LC inspection, future comparison plots) can now trust the `wrappedbest_*` dirs as the canonical "what is the wrapped pipeline's best basin for seed N" reference. The old pre-fix dirs under `m126_wrapped/wrappedbest_seed*/` are untouched and still on disk as pre-fix historical artefacts.
