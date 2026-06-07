# DATA_INVARIANTS

> Load-bearing parameters that every inversion pipeline MUST match. Writer prompts MUST cite this file by name. Reviewer checklist MUST verify each item below against the script being reviewed.

**Last updated:** 2026-04-16
**Status:** Current invariants reflect the committed pipeline as of [DATA_INTEGRITY_BUG.md](notebooks/inversion/DATA_INTEGRITY_BUG.md). The migration decision (A=fix Bug 2 only / B=migrate to micro48 / C=institutional) is pending. When that decision is made, this file is the first thing to update.

---

## 1. Trajectory dataset

| Field | Value | Source |
|---|---|---|
| Path | `data/results/inversion_diagnostics/shared/m046_trajectories/m046_trajectories.npz` (post-rename) | Originally `micro46_trajectories.npz` — see RENAMES.md |
| `observation_times` shape | `(500,)` | shared across all 100 seeds |
| Window | `2020-02-05T10:00:00` → `2020-02-05T11:00:00` UTC | fixed, single geometry |
| `mag_hifi` shape | `(100, 500)` | one hi-fi LC per seed on the shared window |

⚠ **Alternative dataset `m048_trajectories.npz` exists** with per-seed random start times in `[08:35, 15:35 UTC]` and 1-hour duration — the design the user originally intended. **Zero current pipelines use it.** Do NOT migrate a pipeline to m048 without an explicit migration plan (Option B in the bug doc). Mixing m046 + m048 silently yields wrong-window scoring.

## 2. Observation window — MUST pass `end_time_utc` explicitly

```python
from lib.experiment_setup import setup_experiment

ctx = setup_experiment(
    n_observations=500,                           # MUST match trajectory dataset
    end_time_utc='2020-02-05T11:00:00',           # MUST pass; default is 16:00 (6-hr) — WRONG
    noise_sigma=0.05,
    random_seed=42,
    ...
)
```

**Why:** `setup_experiment` falls back to the config default `end_time: '2020-02-05T16:00:00'` (a 6-hour window) if `end_time_utc` is omitted. Scoring against the observed LC on the wrong window silently invalidates every hi-fi MSE result. This caused the Bug 2 corruption of m122/m123/m124/m125/m126 (committed 8dd4aed).

## 3. Sampling & noise

| Field | Value |
|---|---|
| `n_observations` | `500` |
| `noise_sigma` | `0.05` mag |
| `random_seed` (observation-noise seed) | `42` (fixed for reproducibility across micro* pipelines) |

## 4. Magnitude grid

| Field | Value |
|---|---|
| `N_MAGS` | **≥ 20** |

Reducing below 20 produces numerical failures in grid search. Load-bearing per memory `feedback_magnitude_grid.md`.

## 5. Satellite configuration

| Field | Value |
|---|---|
| Config | `data/models/intelsat_901/intelsat_901_config.yaml` |
| SPICE satellite ID | `-126824` |
| Body frame | `IS901_BUS_FRAME` |
| Metakernel | `data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm` |
| Articulation (fixed in all inversion experiments) | SP_North/SP_South: 0°; AD_East/AD_West: 15° |

## 6. Parallel-compute rules

| Field | Value |
|---|---|
| CPU cores available | 16 |
| Default `Pool(N)` for grid + NM pipelines | `24` (matches m102 onward — oversubscribes fork model intentionally) |
| Default `Pool(N)` for hi-fi validation | `2` (OOM safety; ray tracing is memory-heavy) |
| Concurrency rule | **Never stack CPU-saturating jobs.** Run batches sequentially. |

## 7. Reviewer checklist (data-plumbing)

Every new pipeline script MUST pass ALL of these before merge:

- [ ] Cites `DATA_INVARIANTS.md` by name in its docstring.
- [ ] Calls `setup_experiment(...)` with **explicit** `end_time_utc='2020-02-05T11:00:00'`.
- [ ] Calls `setup_experiment(...)` with **explicit** `n_observations=500` (matching trajectory dataset).
- [ ] Loads trajectory data from the canonical `m046_trajectories.npz` path. Does NOT silently switch to m048.
- [ ] Uses `N_MAGS >= 20` wherever a magnitude grid is constructed.
- [ ] `Pool(N)` size matches § 6 for its pipeline stage.
- [ ] Dry-run + LC-compare sanity step included in the writer's test plan — reviewer verifies the script includes a one-seed dry-run before full batch.

## 8. When you change an invariant

1. Update this file FIRST with the new value + rationale + effective date.
2. Grep the project for the old value and update every consumer in the same commit.
3. Record a migration entry in `notebooks/inversion/wiki/wiki/log.md` tagged `## [YYYY-MM-DD] invariant | {field} {old → new}`.
4. Any pipeline script relying on the old value must be either (a) retired (move to `archive/`) or (b) updated.

## 9. What this file is NOT

- Not a design doc — design lives in EXPERIMENTS.md and the wiki.
- Not a place to describe new proposed architectures — those live in the branch pages.
- Not backward-compatibility infrastructure — when an invariant changes, the old pipelines migrate or retire.
