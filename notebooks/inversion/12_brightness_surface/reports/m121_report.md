# m121 — Basin-width characterization of surrogate-residual cost

## Hypothesis

The surrogate-residual cost basin-of-attraction is much tighter than the 20°
upper bound suggested by m120. Expected behaviour:

- at <1° perturbation, cost ≈ surrogate noise floor (~0.055 mag mean_L1)
- at 5-10° it discriminates clearly
- cost is likely **elongated**: wider along the ω-magnitude axis than along
  ω-direction or q0-rotation axes (a 1% magnitude error accumulates less
  trajectory drift over 500 epochs than a 1° direction error).

**Falsifiable:** if cost at 20° is indistinguishable from cost at 0.25° → no
local basin structure → search integration infeasible with this cost.

## Relationship to m120 / scope narrowing

This is a direct cousin of `m120_tumbling_competitors.py`. The
surrogate-evaluation code path is **identical** so cost values are directly
comparable. Two deliberate deviations:

1. **Pool composition** — m120's {uniform, close-ω, close-q0, near-truth}
   buckets are replaced with axis-separated perturbations (q0-only,
   ω-direction-only, ω-magnitude-only, joint). This isolates the basin along
   each DOF.
2. **Cost variants** — restricted to `{mean_L1, mean_L2}` instead of
   m120's 13 variants. Basin shape is about the rank statistic, not the
   soft/count-pass thresholding; the other 11 would cost ~1.5× compute for
   diminishing information. Recorded here so it's not mistaken for an
   oversight.

## Pool composition (~6751 candidates)

| Bucket | Scales | Samples/scale | Total |
|---|---|---|---|
| truth | — | 1 | 1 |
| q0-only (deg) | 0.1, 0.25, 0.5, 1, 2, 5, 10, 20 | 250 | 2000 |
| ω-direction-only (deg) | 0.1, 0.25, 0.5, 1, 2, 5, 10, 20 | 250 | 2000 |
| ω-magnitude-only (pct) | 0.25, 0.5, 1, 2, 5, 10, 20 | 250 | 1750 |
| joint (deg) | 0.25, 0.5, 1, 2 | 250 | 1000 |

- q0-only: rotate `q0_true` by exactly `scale`° about a uniform-S² random
  axis. ω unchanged.
- ω-direction-only: rotate `ω_true` by `scale`° about a **perpendicular**
  random axis. `|ω|` preserved exactly.
- ω-magnitude-only: scale `|ω|` by `(1 + scale_pct/100 × sign_random)`.
  Direction preserved.
- joint: q0 rotation + ω-dir perpendicular rotation + `|ω|` scaling
  all at the same nominal `scale` (magnitude applied as percent). Most directly
  comparable to m120's "near_truth" bucket.

## Target seeds

`14`, `27`, `46` — all have `data/results/inversion_diagnostics/m119v2/seed_NNN/setup.npz`
already generated. The script reads these; it does not re-run the setup.

## Compute

- `Pool(8)` matching m120.
- Expected runtime ~3 min/seed × 3 seeds ≈ 10 min.
- **Kill criteria:** any single seed >15 min wall-clock.

## How to run

```bash
# From PROJECT_ROOT
MICRO121_SEED=14 python3 notebooks/inversion/12_brightness_surface/m121_basin_width_metric.py
MICRO121_SEED=27 python3 notebooks/inversion/12_brightness_surface/m121_basin_width_metric.py
MICRO121_SEED=46 python3 notebooks/inversion/12_brightness_surface/m121_basin_width_metric.py

# Env vars:
#   MICRO121_SEED   (default 14) must be one of {14, 27, 46}
#   MICRO121_POOL   (default 8)  pool size
#   MICRO121_FORCE  (default 0)  set to 1 to ignore cached stages
```

## Outputs

`data/results/inversion_diagnostics/m121/seed_NNN/`:

| File | Schema |
|---|---|
| `candidates.npz` | `q0s[N,4] f64`, `omegas[N,3] f64`, `bucket[N] i32`, `axis[N] S12`, `scale[N] f64`, `labels[N] S32` |
| `residuals.npz` | `residual[N,500] f32`, `observed_mag[500] f32` |
| `cost_variants.npz` | `score_mean_L1[N] f32`, `score_mean_L2[N] f32`, `argsort_mean_L1[N] i32`, `argsort_mean_L2[N] i32`, `variant_names` |
| `summary.json` | see schema below |
| `run.log` | combined stdout/stderr |
| `plots/cost_vs_scale_per_axis.png` | 4-panel (q0 / ω-dir / ω-mag / joint), X=scale log, Y=median + p10-p90 band, truth cost as dashed line |
| `plots/basin_boundary.png` | bar chart: per-axis smallest scale where median_cost > truth_cost + k·σ for k∈{1,2,3} |

### `summary.json` schema

```json
{
  "seed": 14,
  "N_cand": 6751,
  "truth_idx": 0,
  "variant_list": ["mean_L1", "mean_L2"],
  "scales": {
    "q0_deg":        [0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
    "omega_dir_deg": [0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
    "omega_mag_pct": [0.25, 0.5, 1, 2, 5, 10, 20],
    "joint_deg":     [0.25, 0.5, 1, 2]
  },
  "n_per_scale": 250,
  "headline": {
    "mean_L1": {"truth_cost": 0.055, "truth_rank": 0, "N": 6751},
    "mean_L2": {"truth_cost": 0.009, "truth_rank": 0, "N": 6751}
  },
  "per_axis": {
    "mean_L1": {
      "q0":        [{"axis":"q0", "scale":0.1, "n_samples":250,
                     "median_cost":..., "p10_cost":..., "p90_cost":...,
                     "min_cost":..., "max_cost":..., "truth_beats_all":bool},
                    ...],
      "omega_dir": [...],
      "omega_mag": [...],
      "joint":     [...]
    },
    "mean_L2": {...}
  },
  "basin_boundary": {
    "mean_L1": {
      "q0":        {"k1":0.5, "k2":1.0, "k3":2.0, "truth_cost":0.055, "sigma":0.001},
      "omega_dir": {"k1":0.25, ...},
      "omega_mag": {"k1":5.0, ...},   // expected widest — hypothesis
      "joint":     {"k1":0.5, ...}
    },
    "mean_L2": {...}
  },
  "noise_floor_ref": {
    "variant":"mean_L1","bucket":"q0","scale_deg":0.1,"sigma":0.001
  },
  "plots": {"cost_vs_scale_per_axis": "...", "basin_boundary": "..."}
}
```

## Script conventions (all non-negotiable ones met)

- `#!/usr/bin/env python3` + full docstring
- `mp.set_start_method('fork', force=True)` before Pool
- `Pool(8)`, `__main__` guard
- `MICRO121_SEED`, `MICRO121_FORCE`, `MICRO121_POOL` env vars
- Tee class for dual stdout + `run.log`
- `time.time()` around every stage with elapsed print
- Atomic JSON saves via `.tmp` + rename
- Skeleton `np.savez_compressed(...)` blocks written before computation; each
  stage checkpoints so later stages can be resumed via `FORCE=0`
- Final headline print matches m120 style:
  `truth rank X/N under mean_L1 | min competitor cost Y at scale Z° (axis)`
- **No library code modified.** `src/` and `notebooks/inversion/lib/` are untouched.

## Expected interpretation

- If hypothesis holds: basin boundary (`k1`) column should show ω-magnitude
  wider than q0 or ω-direction (e.g. ω_mag crosses at ~5-10%, others at
  ~0.5-1°). Medians at <1° should be close to σ; medians at 20° should be
  clearly separated.
- If falsified (flat median across all scales for any axis): cost has no
  local basin along that DOF → local search not viable; would pivot to a
  different cost or to grid-only search.
