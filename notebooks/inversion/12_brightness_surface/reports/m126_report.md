# m126 — Wrapped pipeline on 6 untested baseline seeds

**Script:** `notebooks/inversion/12_brightness_surface/m126_wrapped_pipeline.py`
**Outputs under:** `data/results/inversion_diagnostics/m126_wrapped/`
**Seeds:** `[0, 6, 12, 24, 33, 36]`
**Status:** written (not yet run)

## Hypothesis

On the 6 m115 baseline seeds never tested with the keep_better wrapper
(`0, 6, 12, 24, 33, 36`), the wrapped pipeline improves seed-level best
hi-fi MSE by ≥10% vs plain m115 on ≥3 of 6 seeds.

Decision rule:
- **CONFIRMED** iff `n_seeds_improved_≥10%` ≥ 3 → promotes `#gradient-based-inversion` to `#validated`.
- **REFUTED** otherwise.

This is the direct replication of the m125 conclusion (3/4 of the seeds
m124 probed were helped by the wrapper) on a disjoint slice of the
population, explicitly held out because m124 had only sampled 5 seeds.

## Method

No new DE compute. Per seed:

1. **Load** `m115_surrogate_pipeline/seed_NNN/result.json`. Take each
   clean (non-NaN) basin in `de_results.basins`; keep its `hifi_mse` verbatim
   as `hifi_before`.
2. **Build context** via `setup_experiment(n=500, sigma=0.05, seed=42,
   true_q0_wxyz=..., true_omega0_rad=...)` using `q0s[seed]` / `omega0s[seed]`
   from `m046_trajectories.npz`. Yields BOTH the polish-cost ctx
   (`k1_j2000_ce`, `k2_j2000_ce`, `obs_dist_ce`, `observed_mag_ce`, 500 all-epoch
   constraints) AND the hi-fi `ExperimentContext`.
3. **L-BFGS polish** each basin (serial) using tangent-at-start 6-DOF
   parameterisation (`r ∈ ℝ³` Rodrigues LEFT-multiplying `q0_start`; `ξ ∈ ℝ²`
   tangent-plane perturbation to `ω̂`; `δ` scaling `|ω|`). Cost = mean |predicted_mag −
   observed_mag| with surrogate at panel=0°, dish=15°. Options:
   `ftol=1e-6, gtol=1e-3, maxiter=100, maxfun=500`, 2-sided FD jac.
   — Copied verbatim from m123.
4. **Hi-fi validate** the polished `(q0_after, ω_after)` per basin using
   `m115.hifi_validate` (imported, not reimplemented), parallelised with
   `mp.Pool(8)` under `fork` start method. Workers read a module-global
   `_WORKER_HIFI_STATE` populated in the parent and inherited via fork — same
   idiom as m124.
5. **Wrapper:** `hifi_wrapped = min(hifi_before, hifi_after)` per basin.
6. **Seed-level best:** `min(hifi_wrapped across basins)`. Compare against the
   source `best_hifi_mse` from m115.

### Checkpoints (per seed)

| File                  | Contents                                                                                                                                                                                          |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `polish_ckpt.npz`     | Per-basin `q0_before/after`, `omega_before/after`, `surr_mse_before/after`, `q0_err_before/after`, `w_dir_err_before/after`, `w_mag_err_pct_before/after`, `n_iter`, `n_fev`, records JSON blob. |
| `hifi_ckpt.npz`       | Per-basin `hifi_before`, `hifi_after`, `hifi_wrapped`, `polish_helped`, `hifi_mags_after` (500 predicted magnitudes), `hifi_wall_s`.                                                               |
| `result.json`         | Per-seed summary: `best_hifi_wrapped`, `best_hifi_m115`, `improvement_pct`, classification, full per-basin detail, timing breakdown.                                                               |

**Batch-level:** `batch_summary.json` with per-seed rows + population
aggregates (`n_seeds_improved_ge10pct`, `n_basins_helped`, `n_basins_hurt`,
classification counts, verdict, full timing).

### Classification thresholds
- **OK**: `hifi_wrapped < 0.01`
- **PARTIAL**: `0.01 ≤ hifi_wrapped < 0.1`
- **FAIL**: `hifi_wrapped ≥ 0.1`

(matches m124 / m125.)

## Compute budget

Seed population: 6 seeds × 3 basins (per m115 result.json inspection) = **18 polished basins**.

| Stage               | Est. time                       | Notes                                                                                   |
|---------------------|----------------------------------|-----------------------------------------------------------------------------------------|
| Stage A (ctx build) | ~5–6 min (serial; 50–60 s × 6)   | `setup_experiment(skip_true_lc=False)` — truth LC is computed (reproduces observed_lc). |
| Stage B (polish)    | ~9 min (serial; 18 × ~30 s)      | Surrogate forward pass inside FD L-BFGS. Spec says don't fuse with hi-fi.               |
| Stage C (hi-fi)     | ~2 min (Pool(8); 18 × ~50 s)     | `hifi_validate` ≈ 50 s per call (m115 per-basin `hifi_time_s` median).              |
| Stage D/E (assemble)| <5 s                             | JSON/NPZ save only.                                                                     |
| **Total (wall)**    | **~16–18 min**                  | Well within 25-min target; kill at 60 min.                                              |

Parallelisation gain on hi-fi: 18 evals on 8 cores → ~3 waves → ~150 s vs
900 s serial. Polish is serial by spec. Stage A is not parallelised because
`setup_experiment` uses SPICE + trimesh which don't fork cleanly across many
simultaneous contexts; running them serially is safer and cheap.

## Expected outcomes

1. **Seeds with small `q0_err` basins** (seeds 24, 33 per m115 best_hifi
   = 0.044 / 0.082) are the most likely improvers: they already live near a
   gradient-bearing region of the surrogate and GD polish should descend.
2. **Seeds with saturated q0_err ~180°** (seeds 0, 6, 36 all have
   best_hifi > 0.1) may break-even, mirroring the seed 27 pattern in
   m125 where polish moved nowhere and the wrapper's safety-floor kicked
   in.
3. **Seed 12** (best_hifi ≈ 0.33) is the most uncertain — large enough MSE
   that some basin might still have exploitable gradient signal.

Prior (from m125 on the other 5 seeds): 3/4 with basins improved,
roughly 75% basin-level help rate. If that transfers, we'd expect ~4.5/6
seeds improved → clearly CONFIRMED.

## Results

### Per-seed

| seed | m115_best_hifi | wrapped_best | improvement (%) | classification | basins_helped / total |
|-----:|---------------:|-------------:|----------------:|:---------------|----------------------:|
|    0 |               |              |                 |                |                       |
|    6 |               |              |                 |                |                       |
|   12 |               |              |                 |                |                       |
|   24 |               |              |                 |                |                       |
|   33 |               |              |                 |                |                       |
|   36 |               |              |                 |                |                       |

### Aggregates

| metric                       | value |
|------------------------------|------:|
| n_seeds_improved_≥10%        |       |
| n_seeds_improved_any         |       |
| n_basins_helped / total      |       |
| n_basins_hurt / total        |       |
| OK count                     |       |
| PARTIAL count                |       |
| FAIL count                   |       |

### Verdict
(fill after run)

## Files to update post-run

- `notebooks/inversion/EXPERIMENTS.md` — add m126 row.
- `notebooks/inversion/wiki/wiki/branches/gradient-based-inversion.md` —
  update branch status (#validated if CONFIRMED, remain #open-active if REFUTED).
- `notebooks/inversion/wiki/wiki/log.md` — add dated entry.
