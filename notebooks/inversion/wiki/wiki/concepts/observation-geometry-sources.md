---
title: "Observation geometry sources — m046 vs m048"
type: concept
sources:
  - "raw/inversion_diagnostics/m046_trajectories/m046_trajectories.npz"
  - "raw/inversion_diagnostics/m048_trajectories/m048_trajectories.npz"
related:
  - "[[m048-migration]]"
  - "[[surrogate-model]]"
  - "[[multi-solution-philosophy]]"
  - "[[DATA_INTEGRITY_BUG]]"
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

# Concept: Observation geometry sources

## The two datasets

Two trajectory databases are used by the pipeline. They differ in **observation-window diversity**, not in satellite physics (same IS-901 model, same inertia tensor, same noise model).

| field | **m046** | **m048** |
|-------|----------|----------|
| file | `data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz` | `data/results/inversion_diagnostics/m048_trajectories/{m048_trajectories.npz, per_trajectory/traj_seed{NNN}.npz}` |
| n_trajectories | 100 | 100 |
| n_obs per traj | 500 | 500 |
| duration per traj | 3600 s | 3600 s |
| observation window | **fixed** 2020-02-05 10:00 → 11:00 UTC for ALL seeds | **per-seed random** `start_et ∈ [08:35, 15:35 UTC]` |
| phase-angle range | 30°–60° | 9.2°–95.2° |
| inertia_tensor | identical 3×3 (same rigid body) | identical 3×3 (same rigid body) |
| per-seed truth | `q0s[seed], omega0s[seed], mag_hifi[seed]` in the master npz | `q0_wxyz, omega0_rad, mag_hifi` in each `traj_seed{NNN}.npz` |
| observation_times | shared `(500,)` array | per-seed `(500,)`, but equal modulo `start_et` offset (all are `linspace(0, 3600, 500)`) |
| provides `start_et` | ❌ | ✅ (key field for [[m048-migration]]) |

## Why two datasets exist

[[DATA_INTEGRITY_BUG]] (discovered 2026-04-16) flagged that every pipeline ran on m046 only — a *single observation window* — so all "100-seed cohort" claims are really "100 tumble-states observed from one vantage point." m048 was generated to fix this ("Bug 1"), but as of Phase 1 of [[m048-migration]] the scripts that actually drive the pipeline still default to m046.

The surrogate is **geometry-agnostic** — it's trained on uniform random `(k1, k2, articulation)` per epoch, not on any particular observation window — so switching datasets is plumbing, not retraining.

## Loader abstraction (`lib.traj_source`)

All pipeline scripts now route through `load_truth(seed, source)` instead of opening the npz directly:

```python
from lib.traj_source import load_truth  # source ∈ {'m046', 'm048'}

truth = load_truth(seed=42, source='m048')
# → dict with keys:
#   q0_wxyz, omega0_rad, mag_hifi, observation_times,
#   inertia_tensor, start_et, end_time_utc, duration_s,
#   source, seed

# Thread directly into setup_experiment:
ctx = setup_experiment(
    n_observations=500,
    start_et=truth['start_et'],       # None for m046, float for m048
    end_time_utc=truth['end_time_utc'],# '...T11:00:00' for m046, None for m048
    duration_s=truth['duration_s'],    # 3600.0 for both
    skip_true_lc=True)
```

## Why it matters for inversion mechanics

- **ctx build cost.** m046 → one `setup_experiment` shared across seeds (~50 s once). m048 → per-seed ctx build (~50 s × N seeds). Cost is fixed across the 4 Stage B pipeline stages for the same seed.
- **Narrow-basin risk.** [[m121_basin_width_metric]] measured ω-dir basin width <0.1° on m046. At higher phase angles (m048's tail), shadowing is stronger and basins may be narrower — the 50k DE population from m115 could miss them.
- **Dark-mag-saturation exposure.** [[dark-mag-saturation]] predicts more of the LC spent at the surrogate's 23-mag ceiling when phase angle is extreme. Expect more FAIL seeds in m048's high-phase tail.
- **Upstream data debt.** m103_hybrid geo_ckpts and m102_fullmse result.npz files were harvested on m046 only. `m115.load_omega_candidates` has no m048 fallback — it will return `None` for any m048 seed without an upstream harvest. Phase 2 of [[m048-migration]] must fix this before the pilot can run.

## Invariant (DATA_INVARIANTS.md)

Every call to `setup_experiment` must result in an `observation_times` array that matches the truth file's `observation_times` (m046) or the per-seed `observation_times` (m048). Before Phase 1 landed, a defaults-only call gave a 6-hour window (config default end = 16:00) when the m046 truth was 1-hour — the root cause of [[DATA_INTEGRITY_BUG]]'s Bug 2. Phase 1 does NOT change this invariant, but it makes the `m048` path work by threading `start_et` through as a kwarg.
