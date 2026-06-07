# m129 -- Densified SO(3) Grid on Seed 33 (Flipped-omega Density Test)

## Hypothesis

A 600k super-Fibonacci SO(3) grid (~1.4 deg median geodesic spacing, 10x
denser than [[m127_flipped_omega_search]]'s 60k ~3 deg grid) + top-50 L-BFGS-B polish
(omega fixed at `-omega_true`) recovers seed 33's known flipped-omega basin
(from [[m126_wrapped_pipeline]], hi-fi MSE ~= 0.082, q0 compensation 98.5 deg about body
+Z axis). Success = winner hi-fi < 0.15.

## Method

- Positive control: seed 33 (single seed).
- Stage A: super-Fibonacci SO(3) grid, N=600000, omega_search = `-omega_true`,
  mean-L1 residual vs observed LC. Pool(8) workers.
- Stage B: L-BFGS-B polish of top-50 grid quaternions in 3-DOF rotvec
  (omega held fixed). Basins clustered by quaternion geodesic < 5 deg
  (cap 5 basins).
- Stage C: hi-fi validation of each basin via vendored `hifi_validate`,
  Pool(2).
- Classification: FLIPPED_VALID (<0.1), FLIPPED_PARTIAL (<0.5), FLIPPED_FAIL.
- Per-basin records: `q0_err_deg` only. `w_dir_err_deg` = 180 (signed) and
  `w_mag_err_pct` = 0 by construction; flagged once at top level via
  `omega_search_is_negated_truth: True`.

## Confirm vs Refute

- **CONFIRMED (winner hi-fi < 0.15):** pure grid-density is the fix for
  narrow flipped-omega basin recovery. [[m127_flipped_omega_search]]'s REFUTED is retractable
  on wider grids. Triggers m130 (DE-over-q0, omega=-omega_true, all 11
  seeds) for population enumeration.
- **REFUTED (winner hi-fi >= 0.15):** narrow basins are NOT
  grid-density-recoverable at 600k. Pivots to population-based (DE)
  enumeration in m130 next session.

## Compute budget

- Stage A: 600k surrogate evals x 500 epochs -> ~90 s on Pool(8)
  (10x m127 at ~10 s).
- Stage B: 50 L-BFGS-B polishes x ~0.5 s serial = ~25 s.
- Stage C: 5 basins x ~65 s hi-fi / Pool(2) = ~165 s wall.
- Setup: ~50 s SPICE + STL + surrogate load.
- Total expected: ~5-10 min wall.

## Kill criteria

- **Per-seed hard cap: 15 min.** If hit, investigate and abort.
- **Stage A watchdog: >5 min wall on grid scoring** strongly suggests worker
  crash -- abort.
- **OOM during Stage A:** drop `MICRO129_POOL` from 8 to 4 and retry.

## Env vars

- `MICRO129_POOL` -- Stage A worker pool size (default 8; drop to 4 on OOM).
- `MICRO129_FORCE` -- set to `1` to ignore cached stage NPZs and recompute.
- `MICRO129_SEEDS` -- comma-separated seed list override; default is
  `[33]` (positive control only).

## Output

`data/results/inversion_diagnostics/m129_densegrid/`
  - `seed_033/stage_a_grid.npz`
  - `seed_033/stage_b_polish.npz`
  - `seed_033/stage_c_hifi.npz`
  - `seed_033/result.json`
  - `seed_033/run.log`
  - `batch_summary.json`

## Status

Script written. Not yet executed.
