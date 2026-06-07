# m119 — Attitude Isoshell POC (seed 14)

## Purpose

Kernel-factored proof-of-concept: test whether replacing the m118
pab-contour brightness surrogate `B(n_body)` with the trained MLP surrogate
`B(k1_body, k2_body, panel=0, dish=15, dist)` eliminates the ~25° contour
noise floor observed in m118 (`data/results/inversion_diagnostics/
m118/seed_014/`).

## Hypothesis

Per-epoch attitude isoshells

    { R ∈ SO(3) : | surrogate(R k1_J2000, R k2_J2000, 0°, 15°, d) − obs_mag | < σ }

are 2-D submanifolds. The truth trajectory should pass through nearly all
255 constraint epochs' isoshells at tight σ, while the m118 rank-1
competitor fails most.

## Kernel-factoring principle

One expensive step (Stage C) builds a single float32 residual tensor
`residual[N_SO3=60000, N_ep=255]`. Multiple cost variants and σ thresholds
are then re-scored cheaply (Stage D). No σ or cost form is baked into the
expensive stage.

## Pipeline pseudocode

```
Stage A — Setup (fast, once)
  Load m118/seed_014/kernel.npz
    → constraint_epochs[255], truth_q0, truth_omega0,
      observed_lc[500], obs_times[500]
  Load m046_trajectories.npz → obs_dist[500], q0s[14] (cross-check)
  ctx = setup_experiment(500, skip_true_lc=True)
    → sun_pos, obs_pos, sat_pos (J2000 km), inertia_tensor
  k1_j2000 = unit(sun_pos - sat_pos); k2_j2000 = unit(obs_pos - sat_pos)
  Subset to constraint_epochs → k1_j2000_ce[255,3], k2_j2000_ce[255,3],
                                 obs_dist_ce[255], observed_mag_ce[255]
  Save setup.npz

Stage B — SO(3) grid
  quats_xyzw = super_fibonacci(N_SO3=60000)    # Alexa 2022
  R_matrices = Rotation.from_quat(quats_xyzw).as_matrix()
  Save so3_grid.npz

Stage C — Residual kernel (expensive, parallel, skip-if-cached)
  Pool(POOL_SIZE) with per-worker SurrogateModel init
  For each epoch block:
    For each epoch ep:
      k1_body = einsum('nij,j->ni', R_matrices, k1_j2000_ce[ep])
      k2_body = einsum('nij,j->ni', R_matrices, k2_j2000_ce[ep])
      normalise k1_body, k2_body
      mag_pred[:, ep] = model.predict_magnitude(
          k1_body, k2_body, zeros(N), full(N,15), full(N,dist_ce[ep]))
  residual = mag_pred - observed_mag_ce[None, :]           # float32
  Save residual_kernel.npz (~60 MB)

Stage D — Cost variants (cheap re-score on saved residual)
  For variant in {mean_L2, mean_L1, max_abs,
                  count_pass_{5,10,15,20,30,50,100 × 0.01},
                  soft_pass_{10,20,50 × 0.01}}:
    scores_full[N_SO3] = score_variant(residual, name)
    topK = argsort(scores)[:1000]
    Save both scores and topK indices
  Save cost_variants.npz

Stage E — Target scoring (exact trajectories, not grid-nearest)
  Targets: truth, m118_facet_rank1, m118_ipl_weighted_ext_rank1,
           m115 basins (6 for seed 14).
  For each target (q0, omega0):
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I)
    q_ce = quats[constraint_epochs]
    R_ce = from_quat(xyzw).as_matrix()
    k1_body_tgt = einsum('nij,nj->ni', R_ce, k1_j2000_ce)
    k2_body_tgt = einsum('nij,nj->ni', R_ce, k2_j2000_ce)
    mag_pred_tgt = surrogate(k1_body_tgt, k2_body_tgt, 0, 15, dist_ce)
    residual_tgt[255] = mag_pred_tgt - observed_mag_ce
    For each variant: score_tgt = score_variant(residual_tgt, name)
                      rank = sum(grid_scores better than score_tgt)
    Nearest grid: per-epoch argmax |dot(q_ce[ep], grid_q)|
  Save target_scores.npz

Stage F — Plots + printed summary
  residual_distribution.png   : |r| histogram + per-epoch percentile curves
  target_pass_curves.png      : frac_pass vs σ (log x), per target
  topK_overlap.png            : |topK_i ∩ topK_j|/1000 heatmap across variants
  truth_vs_competitors_rank.png : bar chart truth vs m118 rank-1

POC classification
  PASS iff ALL:
    truth median |r| < 0.1
    truth p90 |r|    < 0.3
    truth rank < 1000 under ≥ 1 variant
    truth rank < competitor rank under ≥ 1 variant
```

## File manifest

- Script:
  `notebooks/inversion/12_brightness_surface/m119_attitude_isoshell.py`
- This report:
  `notebooks/inversion/12_brightness_surface/m119_REPORT.md`

Output (after running):
- `data/results/inversion_diagnostics/m119/seed_014/`
  - `setup.npz` — J2000 geometry, truth, inertia
  - `so3_grid.npz` — super-Fibonacci quaternions + rotation matrices
  - `residual_kernel.npz` — `residual[60000,255]` float32 (~60 MB), `mag_pred`
  - `cost_variants.npz` — per-variant scores + top-1000 indices
  - `target_scores.npz` — per-target residuals, scores, ranks, grid-quant
  - `summary.json` — timings, truth stats, POC verdict, file paths
  - `run.log` — tee'd stdout
  - `plots/*.png` — four diagnostic plots

## Usage

```bash
# Full run (first time)
python3 notebooks/inversion/12_brightness_surface/m119_attitude_isoshell.py

# Force re-run of Stage C (expensive residual kernel)
MICRO119_FORCE=1 python3 notebooks/inversion/12_brightness_surface/m119_attitude_isoshell.py

# Smoke test with smaller grid
MICRO119_N_SO3=5000 MICRO119_POOL=4 \
  python3 notebooks/inversion/12_brightness_surface/m119_attitude_isoshell.py

# Custom pool size
MICRO119_POOL=16 python3 notebooks/inversion/12_brightness_surface/m119_attitude_isoshell.py
```

Environment variables:
- `MICRO119_SEED` (default 14) — only seed 14 has complete upstream artifacts
- `MICRO119_POOL` (default 8) — worker count for Stage C
- `MICRO119_FORCE` (default 0) — set to `1` to rebuild expensive stages
- `MICRO119_N_SO3` (default 60000) — SO(3) grid size

BLAS threading is pinned to 1 via env-var defaults set before `import numpy`
so the `multiprocessing.Pool` does not contend with inner BLAS threads.

## Expected runtime

- Stage A: ~30 s (setup_experiment at N=500)
- Stage B: ~1 s
- Stage C: dominant cost. With `N_SO3=60000`, 255 epochs, and `POOL_SIZE=8`:
  rough estimate ~5–15 min depending on MLP throughput at batch-60000.
  Cached afterwards.
- Stage D: ~10 s
- Stage E: ~30 s (8 target propagations + grid rank scoring)
- Stage F: ~10 s

Total first run: ~10–20 min. Subsequent runs (residual cached): ~1 min.

Peak disk use: `residual_kernel.npz` is ~60 MB. `so3_grid.npz` is ~6 MB
(R_matrices float64). `target_scores.npz` is ~a few MB.

## Scope discipline

- No modifications to `src/` or `notebooks/inversion/lib/`.
- Helper functions (`attitude_error_deg`, `omega_dir_err`, `omega_mag_err_pct`)
  are copied locally from `m115_surrogate_pipeline.py` per convention.
- No git commits, no pipeline runs beyond this POC.
- All checkpoints written before computation so any stage can be restarted
  from its predecessor's NPZ.
