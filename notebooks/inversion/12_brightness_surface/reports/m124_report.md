# m124 -- Hi-fi validation of m123 polished candidates

## Hypothesis
Hi-fi MSE reduction factor from L-BFGS polish tracks surrogate MSE reduction
factor within ±30% across the 12 DE-basin polished points
(|log(surr_ratio) − log(hifi_ratio)| < 0.3).

- CONFIRMED ⇔ ≥ 9/12 (75%) basins within threshold → polish produces real
  hi-fi improvement → gradient branch → `#validated`.
- REFUTED ⇔ < 3/12 (25%) basins within threshold → surrogate-internal
  cosmetic polish only.
- PARTIAL otherwise.

Sub-check (5 truth-start polished points): truth_hifi_mse ≈ noise² = 0.0025.

## What was implemented
- `m124_hifi_validate.py` — three-stage pipeline:
  - **A** build candidate list by loading `m123/seed_NNN/polish.npz`
    `records_json` (truth + basin_k starts) and pulling `hifi_mse_before`
    from `m115/seed_NNN/result.json` `de_results.basins[k].hifi_mse`.
  - **B** build one `ExperimentContext` per seed via `setup_experiment` with
    `true_q0_wxyz` / `true_omega0_rad` overrides (same pattern as m119v2).
    Fork `Pool(POOL_SIZE=8)` so workers inherit the 5 ctx objects via COW;
    dispatch 22 jobs (5 truth-reference + 17 polished-point evals) to
    `imap_unordered`; reuse `m115.hifi_validate` inside each worker.
  - **C** compute per-candidate surr/hifi ratios, log-ratio agreement, final
    verdict; save `summary.json` with all per-candidate state + aggregate.

- Skip-if-exists at each stage (`candidates.json`, `hifi_results.npz`,
  `summary.json`) with `MICRO124_FORCE=1` override. Console follows spec.

## TODOs punted
- Per-candidate hi-fi LC overlay plots (raw mags already saved in
  `hifi_results.npz`).
- Cross-link to m123 attractor clusters.

## Files
- `notebooks/inversion/12_brightness_surface/m124_hifi_validate.py`
- `notebooks/inversion/12_brightness_surface/m124_REPORT.md`

## Expected outputs under `data/results/inversion_diagnostics/m124/`
`candidates.json`, `hifi_results.npz`, `summary.json`, `run.log`.
