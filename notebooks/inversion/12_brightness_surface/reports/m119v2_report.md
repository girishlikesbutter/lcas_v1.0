# m119v2 - attitude isoshell POC (time-mismatch fix)

## What the bug was

The original `m119_attitude_isoshell.py` (now retracted) mixed two
inconsistent data sources per epoch. `data/results/inversion_diagnostics/
m046_trajectories/m046_trajectories.npz` holds a stale pre-computed
dataset at a 1-hour window / 7.21 s per epoch sampling; `lib/experiment_setup.
setup_experiment()` builds today's config at a 6-hour window / 43.29 s per
epoch sampling. Both are 500 epochs, so epoch index `i` means two different
wall times (off by 6x). v1's Stage A combined `observed_lc` + `obs_dist` +
`constraint_epochs` from m046 with `k1_j2000` + `k2_j2000` freshly
regenerated via `setup_experiment`; observer direction `k2` in body frame was
off by up to 74 deg at epoch 499. The "truth at rank 0 / 60000 under
mean_L1" result is an artifact, not signal. See
`notebooks/inversion/12_brightness_surface/m119_BUG.md` for the full
diagnosis.

## What the fix changes

**Lib patch (`notebooks/inversion/lib/experiment_setup.py`):** two new optional
kwargs on `setup_experiment`, `true_q0_wxyz: Optional[NDArray] = None` and
`true_omega0_rad: Optional[NDArray] = None`. When `None` (default) the function
preserves the pre-patch axis=(0.6, 0.3, 0.8) angle=45 deg / `true_omega_deg`
behaviour verbatim. When provided, the arrays are assigned directly to
`ctx.true_q0` / `ctx.true_omega0`. Backwards-compatible: every existing caller
keeps the same code path.

**Script architecture delta from v1:**
- Stage A rewritten. Pulls only `q0s[SEED]` and `omega0s[SEED]` scalars out
  of m046 (safe: not time-dependent), forwards them to `setup_experiment(
  n_observations=500, skip_true_lc=False, true_q0_wxyz=..., true_omega0_rad=...)`,
  and derives `k1`, `k2`, `obs_dist`, `obs_times`, `observed_mag_ce`,
  `inertia_tensor` all from the returned `ctx`. No m046 per-epoch arrays.
  Constraint epochs = `np.arange(500)` (all epochs). Asserts
  `obs_times[-1] > 20000 s` and uniform `dt` to catch any future config drift.
- Stage B unchanged.
- Stage C unchanged; residual shape grows from `[N_SO3, 255]` to
  `[N_SO3, 500]` (kernel ~120 MB instead of ~61 MB).
- Stage D unchanged.
- Stage E simplified. Only `('truth', truth_q0, truth_omega0)` is scored.
  m115 basins and m118 rank-1 competitors were dropped: propagating
  their `(q0, omega)` on today's 6-hr `obs_times` gives the wrong attitudes
  for them, and they are not the comparison baseline of this honest POC.
- Stage F unchanged except the rank-comparison plot is replaced by a
  single-series `truth_rank.png` with top-0.1% and mid-pack reference lines.
- POC classification adds a third check
  `truth_rank_top_percentile`: `min_rank / N_SO3 < 0.001` (top-0.1%). PASS
  requires all four checks.
- Output dir is `data/results/inversion_diagnostics/m119v2/seed_014/`
  (the `v1` tree is left intact as provenance of the retracted claim).

## How to run

```
python3 notebooks/inversion/12_brightness_surface/m119v2_attitude_isoshell.py
```

Override env vars (all optional, same semantics as v1):

```
MICRO119_SEED=14 MICRO119_POOL=8 MICRO119_FORCE=0 \
MICRO119_N_SO3=60000 \
python3 notebooks/inversion/12_brightness_surface/m119v2_attitude_isoshell.py
```

Expected runtime dominated by Stage C (~N_SO3 x 500 surrogate evals,
pooled across 8 workers). Stage A now includes a hi-fi LC regeneration
(~50 s) because `skip_true_lc=False` is required to obtain a time-consistent
`observed_lc`.

## What to look at when it finishes

Primary:

- `data/results/inversion_diagnostics/m119v2/seed_014/summary.json`
  - `truth_ranks` (dict variant -> rank out of 60000)
  - `truth_min_rank`, `truth_min_rank_percentile`
  - `poc_verdict` ("POC PASS" or "POC FAIL")
  - `poc_checks` (per-check PASS/FAIL)

Decision rule (from spec):

- Any `truth_ranks[variant] <= 50` under a mean-based variant
  (`mean_L1`, `mean_L2`, `max_abs`) => signal is robust; the v1 claim
  survives the fix.
- `truth_ranks[variant] > 1000` under every variant => v1 claim refuted.
- In between => partial signal.

Plots:

- `plots/truth_rank.png` - bar chart of truth rank per variant (symlog)
- `plots/residual_distribution.png` - grid-wide |residual| histogram + per-epoch stats
- `plots/target_pass_curves.png` - truth pass fraction vs sigma
- `plots/topK_overlap.png` - top-1000 Jaccard matrix across variants

Log:

- `run.log` carries the printed per-stage diagnostics, including the Stage A
  sanity prints (`obs_times: dt=43.29 s, span=21602.0 s`) that prove the
  geometry is no longer mixed.
