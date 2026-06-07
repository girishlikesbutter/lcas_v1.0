# m072 — Minimal Viable Pipeline (Implementation Spec)

## Overview

Rewrite m070_full_pipeline.py as a cleaner, faster version. Same algorithm,
but with redundancies removed and inner loops vectorized.

## Source of truth

Read `m070_full_pipeline.py` in this same directory for the current implementation.
The new script should be `m072_mvp_pipeline.py` in the same directory.

## Changes from m070

### 1. Setup: use skip_true_lc=True

```python
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)
```

Load observed LC from the trajectory NPZ (m046_trajectories.npz) with noise:
```python
rng = np.random.default_rng(42)
observed_lc = master['mag_hifi'][TRAJ_SEED] + rng.normal(0, 0.05, 500)
```

### 2. Add bright constraints to grid search and NM refinement

Currently steps 2-3 only score ±X alignment at specular epochs (mag < 6.0).
Add bright epoch (mag 6.0-9.0) alignment to the cost, matching step 6's formulation:

```
cost = spec_weight * sum((1 - max(dot(±X, pb)))^2)  [specular epochs]
     + bright_weight * sum((1 - max(dot(any_normal, pb)))^2)  [bright epochs]
```

Use spec_weight=10.0, bright_weight=5.0 (same as step 6).

The bright constraints require propagating delta-qs to bright epoch times too
(not just specular constraints). So dt_constraints should include BOTH specular
AND bright epochs, with a flag for which type each is.

### 3. Merge steps 3+4: record best phi during NM

The NM refinement already sweeps all 360 phis at each evaluation. At the final
evaluation, record which phi gave the best cost. This eliminates step 4 entirely.

The NM `refine_one_nm` should return: (idx, cost, omega, best_phi, nfev)

After NM, extract TOP_N_OMEGA candidates directly as (omega, phi) pairs.
Back-propagate each to t=0 to get (q0, w0).

### 4. Vectorize the phi×constraint inner loop

Replace the pure-Python loop:
```python
for qa in q_anchors:        # 360 iterations
    cost = 0.0
    for ci in range(n_constraints):
        qg = quat_multiply(qa, delta_qs[ci])
        R = Rotation.from_quat(...)
        pb = R @ pab[ci]
        bd = max(dot(n_pX, pb), dot(n_mX, pb))
        cost += (1-bd)**2
```

With vectorized numpy. Key approach:
- Pre-compute q_anchors as Rotation objects: `R_anchors = Rotation.from_quat(q_anchors_xyzw)`
- For each constraint ci, compute `R_delta_ci = Rotation.from_quat(delta_q_ci_xyzw)`
- Combined: `R_all = R_anchors * R_delta_ci` (broadcasts: 360 × 1 = 360 rotations)
- `pbs = R_all.apply(pab[ci])` → shape (360, 3)
- Specular: `bds = np.maximum(pbs @ n_pX, pbs @ n_mX)` → shape (360,)
- Bright: `bds = (pbs @ unique_normals.T).max(axis=1)` → shape (360,)
- `costs += weight * (1 - bds)**2`

### 5. Coarse-to-fine phi in grid search

Grid search (step 2): use 36 phi bins (10° spacing) — enough for ranking.
NM refinement (step 3): use 360 phi bins (1° spacing) — for precise phi extraction.

### 6. Reorder: geometric refinement BEFORE hi-fi

New flow:
- Step 5 (was step 6): Geometric refinement of ALL K candidates (parallelized on 8 cores)
- Step 6 (was step 5): Hi-fi LC of top 2-3 by geometric cost (parallelized)

This reduces the number of expensive hi-fi evaluations from 10 to 2-3.

### 7. Reduce NM pool

NM_TOP: 50 → 20 (bright constraints improve discrimination)
TOP_N_OMEGA: 5 → 5 (keep same, extract both +X and -X phis per omega)

For each of the 5 best omegas, extract best phi from +X anchor AND best phi from
-X anchor → 10 candidates total (same count as before, but handles ±X disambiguation).

## Pipeline structure

```
Step 1: Peak count → |ω| estimate + anchor/constraint selection
Step 2: Grid search (2000 dirs × 20 mags, 36 phis, vectorized, spec+bright cost)
        [CHECKPOINTED → grid.npz]
Step 3: NM refinement of top 20 (360 phis, record best phi, spec+bright cost)
        → extract top 5 omegas × 2 phis (±X) = 10 candidates
        [CHECKPOINTED → refined.npz]
Step 4: Geometric refinement of all 10 candidates (parallelized)
        [CHECKPOINTED → geo_refined.npz]
Step 5: Hi-fi LC of top 3 by geometric cost (parallelized)
        → winner by hi-fi residual
        [CHECKPOINTED → result.npz + result.json]
```

## Logging

- Use `tee`-style logging: all print output also goes to a log file.
- Log file: `{CKPT_DIR}/pipeline.log`
- Implementation: redirect stdout through a Tee class at the top of the script.

```python
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
```

## Printout style

Each step should print:
- Step header: `--- Step N: description ---`
- Key parameters on one line
- Progress indicator for parallel work (just "launching N jobs on M cores")
- Result summary: 2-3 lines with key numbers
- Oracle comparison (true errors) where available
- Step timing

At the end, print a summary table of all steps with timing.

## Environment variables

```
MICRO72_SEED=93     (default)
MICRO72_ANCHOR=0    (default, 0=+X, but now both ±X are handled internally)
```

## Constants

```python
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36    # grid search
N_PHI_FINE = 360     # NM refinement
NM_TOP = 20
TOP_N_OMEGA = 5
GRID_WORKERS = 16
NM_WORKERS = 16
GEO_WORKERS = 8
HIFI_WORKERS = 8
TOP_HIFI = 3         # number of candidates for final hi-fi
SPEC_WEIGHT = 10.0
BRIGHT_WEIGHT = 5.0
```

## Key imports (same as m070)

```python
from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle
```

## Critical details to preserve from m070

1. Delta-q factorization: propagate identity quaternion, compose with anchor q later
2. Forward/backward propagation for constraints before/after anchor
3. Back-propagation from anchor to t=0: `propagate(q_anchor, -omega, [0, anchor_time])`
4. Quaternion convention: wxyz (scalar-first) throughout, convert to xyzw for scipy Rotation
5. The geometric cost at step 4 must propagate over ALL 500 obs_times (not just constraints)
6. Hi-fi evaluation uses ObjectiveFunction.evaluate() with axis-angle parameterization
