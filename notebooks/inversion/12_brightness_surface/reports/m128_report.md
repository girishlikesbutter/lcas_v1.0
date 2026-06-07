# m128 -- Warm-start flipped-omega polish from m115 DE basins

## Title

**m128** -- Warm-start flipped-omega polish from m115 DE basins.

## Hypothesis (falsifiable)

At least 2 of seeds `{0, 6, 14, 24, 27, 36, 46, 74, 93}` yield polished
`(q0', -omega_basin)` with hi-fi MSE < 0.5.

**Positive control (seed 33):** MUST reproduce hi-fi ~= 0.082 (matching its
m126-validated flipped-omega basin within factor 2). If seed 33 fails to
hit hi-fi < 0.15, the verdict is **INVALID** (polish mechanics broken), NOT
REFUTED.

**Sanity control (seed 12):** has a wide flipped-omega attractor known from
m127; expect hi-fi ~= 0.17 warm-start result roughly matching
(< 0.3 for sanity).

### Verdict rules

| Condition                                                  | Verdict     |
|------------------------------------------------------------|-------------|
| seed 33 best_hifi >= 0.15 (control fails)                  | INVALID     |
| >= 2 non-control seeds hit hifi < 0.5                      | CONFIRMED   |
| exactly 1 non-control seed hits hifi < 0.5                 | MIXED       |
| 0 non-control seeds hit hifi < 0.5                         | REFUTED     |

A CONFIRMED verdict would be strong evidence that narrow flipped-omega basins
exist at the same q0 attractors that m115's DE enumerated for forward
omega, undetectable by m127's 3-deg SO(3) grid.

## Method summary

For each seed in `BASELINE_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93]`:

- **Stage A (L-BFGS-B polish, serial per basin).** Load the 3 m115 DE
  basins from `seed_NNN/step2_hifi.npz:hifi_json`. For each basin: warm-start
  `q0_init = basin['q0_wxyz']`, fix `omega_flipped = -basin['omega_rad']` (note:
  per-basin omega, NOT truth omega). Polish the 3-DOF rotvec at q0 using the
  surrogate mean-|residual| cost (same as m127).
- **Stage B (hi-fi validate, Pool(4)).** Dispatch 6 jobs/seed (3 basins x
  {before, after}) to a forked worker pool. Each job runs
  `hifi_validate(q0, omega_flipped, ...)` using the full shadow-raytracing
  pipeline at all 500 epochs. Backfill hi-fi fields into `stage_a_polish.npz`.
- **Per-seed assembly.** For each basin, `best_hifi = min(before, after)`;
  seed-level `best_hifi_mse_seed = min over basins`. Classify
  `FLIPPED_VALID<0.1, FLIPPED_PARTIAL<0.5, FLIPPED_FAIL` otherwise.

## Key distinctions from m127

1. **No SO(3) grid.** m127 scanned 60k quaternions with omega fixed at
   `-truth_omega`; m128 warm-starts directly at the 3 m115 basin q0s
   and polishes.
2. **Per-basin omega negation.** `omega_flipped = -basin['omega_rad']`, not
   `-truth_omega`. Basin omega can differ from truth by up to ~5 deg
   (e.g. seed 74), so this is a meaningful distinction. Recorded as
   `omega_flipped_is_negated_per_basin_not_truth: true` in result.json.
3. **Known-attractor test.** We're testing whether each m115 forward-omega
   attractor has a nearby flipped-omega valid point, NOT discovering new
   attractors via a grid.

## Compute budget

| Stage        | Per-seed cost                       | Total (11 seeds) |
|--------------|-------------------------------------|------------------|
| setup_experiment | ~30 s (skip_true_lc=False)      | ~330 s           |
| Stage A (polish) | 3 basins x 1-3 s = ~10 s        | ~110 s           |
| Stage B (hi-fi)  | 6 jobs on Pool(4), ~50 s/job, ~75 s wall | ~825 s    |
| Per-seed total   | ~2 min                          | **~22 min**      |

Pool size `HIFI_POOL=4` bounds peak RAM (each worker forks the full shadow
engine). Polish is serial — sub-second per basin, no benefit from pooling.

## Kill criteria

- **Per-seed hard cap: 5 min wall.** If any seed exceeds 5 min (2.5x the
  expected 2 min), investigate immediately (likely a hi-fi worker crash or
  shadow engine hang). Do not sleep-wait; abort and diagnose.
- **Batch hard cap: 30 min wall.** The full 11-seed run should finish in ~22
  min; exceeding 30 min means a stage is mis-scaled and the whole batch should
  be reviewed.
- **Pool watch:** if Stage B emits no progress messages for > 2 min on any
  seed, suspect a forked-worker deadlock (OpenBLAS, etc.).

## What the script does NOT do

- **No SO(3) grid search.** No Stage A a la m127. We never enumerate q0
  from a sampling of rotation space.
- **No q0 attractor discovery.** We assume m115's DE correctly enumerated
  the 3 forward-omega basins per seed and test only warm-starts at those
  points.
- **No omega search / optimisation.** Omega is fixed at `-basin['omega_rad']`
  throughout the polish; we search only the 3-DOF rotvec at q0.
- **No clustering / basin merging.** There is no post-polish clustering:
  each m115 basin contributes exactly one warm-start and produces exactly
  one polished (q0, omega_flipped) point.
- **No truth-q0 warm-start.** The key test is "does a flipped-omega valid
  point exist near each DE basin's q0?" — not "does `(q0_truth, -omega_truth)`
  pass hi-fi?".
- **No library modification.** Script vendors all helpers (`hifi_validate`,
  `make_polish_cost`, quaternion helpers, Tee, atomic_json_save) verbatim
  from m127; does not touch `src/` or `lib/`.

## Output layout

```
data/results/inversion_diagnostics/m128_warmstart_polish/
├── run.log                 # batch-level teed stdout
├── batch_summary.json      # per-seed rows + counts + verdict
└── seed_NNN/
    ├── run.log             # per-seed teed stdout
    ├── stage_a_polish.npz  # init, polished, timings, hi-fi backfilled
    ├── stage_b_hifi.npz    # hi-fi MSE/mags before & after, observed_lc
    └── result.json         # per-seed structured summary
```

## Reproducibility / provenance

- Truth `(q0, omega)` read from
  `data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz`.
- `observed_lc` generated by `setup_experiment(..., skip_true_lc=False,
  random_seed=42, noise_sigma=0.05, end_time_utc='2020-02-05T11:00:00')`
  (identical to m126).
- m115 basins loaded from `step2_hifi.npz:hifi_json` (single source of
  truth for q0_wxyz, omega_rad, and m115 errors used in result.json).
- Surrogate weights/normalization at `/home/girish/surrogate_model/s10_5M_*.npz`.
- L-BFGS-B options `{ftol:1e-7, gtol:1e-4, maxiter:200, maxfun:1000}`
  (identical to m127).
- Multiprocessing `fork` start method; worker state inherited via
  module-global dict (same pattern as m124/126/127).
