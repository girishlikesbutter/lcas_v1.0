# Experiment Progress Tracker

**Last updated**: 2026-02-23 (Phase 1 complete)
**Branch**: `ralph/mixed-fidelity-inversion`
**Deadline**: Roberto update by 2:30 PM NZST

---

## HOW TO USE THIS FILE

After a context clear, tell Claude:
> "Read notebooks/inversion/progress_tracker.md and continue from where we left off"

Claude should: read this file, check the current phase, read any referenced result files, and pick up the next incomplete task.

---

## STATUS OVERVIEW

| Phase | Status | Key Result |
|-------|--------|------------|
| Phase 0: Setup infrastructure | DONE | roadmap.md, progress_tracker.md, MEMORY.md (commit 0e606c2) |
| Phase 1: Exp 00 + 01 (timing + sanity) | DONE | lo-fi 221ms, hi-fi 60s, 272x ratio; all sanity pass |
| Phase 2: Exp 02 + 03 (basin characterization) | NOT STARTED | |
| Phase 3: Exp 04 + 05 (joint basin + window) | NOT STARTED | |
| Phase 4: Exp 06-09 (progressive + filtering) | NOT STARTED | |
| Phase 5: Compile Roberto update | NOT STARTED | |

---

## PHASE 1: Timing + Sanity (Exp 00, Exp 01)

### Instructions for Claude
Write TWO scripts: `exp00_timing.py` and `exp01_sanity.py` in `notebooks/inversion/`.
Then run them both. Commit with message like "feat: exp00+01 timing benchmark and sanity checks".
Update this tracker with results.

### Exp 00 — Timing Benchmark
**File**: `notebooks/inversion/exp00_timing.py`
**Expected runtime**: ~2 min
**What to measure** (10 reps each, report median):
1. `propagate_attitude(true_q0, true_omega0, obs_times, "tumbling", I)` — ODE only
2. `ObjectiveFunction.evaluate(true_params)` — lo-fi full LC (500 epochs)
3. `ObjectiveFunction.evaluate(true_params)` — hi-fi full LC (with shadows)
4. `brightness_single_epoch(true_q0, 0, ctx, use_shadows=False)` — single epoch lo-fi
5. `brightness_single_epoch(true_q0, 0, ctx, use_shadows=True)` — single epoch hi-fi
6. One L-BFGS-B run from truth+tiny perturbation, 6 params, lo-fi, maxiter=10 — time per iteration

**Key constants from experiment_setup.py**:
- `true_omega_deg=(0.5, -0.3, 2.0)`, `end_time_utc='2020-02-05T11:00:00'`
- `n_observations=500`, `noise_sigma=0.05`, `random_seed=42`
- Component masses: Bus=1532, SP=170x2, AD=50x2

**Result**: Save to `data/results/inversion_diagnostics/exp00_timing.json`

### Exp 01 — Forward Model Sanity
**File**: `notebooks/inversion/exp01_sanity.py`
**Expected runtime**: ~1 min
**Checks**:
1. Lo-fi residual at truth (should be small — just noise)
2. Hi-fi residual at truth (should be ~ noise^2 * N_obs since LC was generated with hi-fi + noise)
3. Conservation: T and L at epochs [0, 50, 100, 250, 499] — relative spread < 1e-6
4. L <-> omega roundtrip: ||omega - I^{-1} R(q) L|| < 1e-12
5. L-param objective vs omega-param objective at truth: same value?

**Key imports** (all from `lib/experiment_setup.py`):
```python
from lib.experiment_setup import (
    ExperimentContext, setup_experiment, brightness_single_epoch,
    attitude_error_deg, save_results,
)
from src.inversion.objective_function import ObjectiveFunction, _quaternion_to_rotation_matrix
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle
from src.dynamics.attitude_propagator import propagate_attitude
```

**L-param helpers needed** (copy into each script or import from a shared file):
```python
def omega_to_L(q_wxyz, omega_body, inertia_tensor):
    """R(q) is body->inertial (propagator uses q_dot = 0.5*q*omega_quat)."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    L_body = inertia_tensor @ omega_body
    return R @ L_body   # body -> inertial

def L_to_omega(q_wxyz, L_inertial, inertia_tensor):
    R = _quaternion_to_rotation_matrix(q_wxyz)
    L_body = R.T @ L_inertial   # inertial -> body
    return np.linalg.solve(inertia_tensor, L_body)
```
**CONVENTION NOTE**: `_quaternion_to_rotation_matrix(q)` returns the **body→inertial** rotation
matrix (NOT inertial→body as the comment in objective_function.py says). Verified by conservation
check: `L_inertial = R @ (I @ omega_body)` is conserved; `R.T @ (I @ omega_body)` is NOT.

**Result**: Save to `data/results/inversion_diagnostics/exp01_sanity.json`

### Phase 1 Results
```
Exp 00 timing (median of 10 reps):
  propagate_attitude (500 epochs, 3600s): 76.9 ms
  lo-fi full LC eval:                     220.8 ms
  hi-fi full LC eval:                     60,136 ms (~60s)
  single-epoch lo-fi:                     13.2 ms
  single-epoch hi-fi:                     46.7 ms
  L-BFGS-B iteration (6 params, lo-fi):   4,222 ms (~4.2s)
    → 10 iters, 133 evals = ~13 evals/iter (6 params × 2 finite diff + 1)
  hi-fi / lo-fi ratio:                    272x

Exp 01 sanity (ALL PASS):
  Lo-fi residual at truth:   0.148054  (high because observed LC is hi-fi)
  Hi-fi residual at truth:   0.002402  (≈ noise σ²=0.0025, PASS)
  Conservation spread (T):   2.46e-10  (< 1e-6, PASS)
  Conservation spread (|L|): 2.81e-10  (< 1e-6, PASS)
  L direction conserved:     0.00e+00  (< 1e-6, PASS)
  L<->omega roundtrip error: 1.94e-18  (< 1e-12, PASS)
  L-param vs omega-param:    4.44e-15  (< 1e-10, PASS)

Key insight: R(q) convention is body→inertial, so L_inertial = R @ (I @ omega).
The comment in objective_function.py claiming R is "J2000 to body" is WRONG
(but the code works because R is used consistently throughout).
```

---

## PHASE 2: Basin Characterization (Exp 02, Exp 03)

### Prerequisites
- Phase 1 complete, timing numbers known
- Sanity checks pass

### Instructions for Claude
Write TWO scripts: `exp02_attitude_basin.py` and `exp03_omega_basin.py`.
Run them. Commit. Update tracker.

### Exp 02 — Attitude-Only Basin
**File**: `notebooks/inversion/exp02_attitude_basin.py`
**Expected runtime**: ~5 min (parallelised)
**Method**:
- TRULY fix omega at truth by wrapping objective:
  ```python
  def f_att_only(axis_angle):
      params = np.concatenate([axis_angle, ctx.true_omega0])
      return obj.evaluate(params)
  ```
- 3 parameters, L-BFGS-B, maxiter=100
- Perturbation levels: [1, 2, 5, 10, 15, 20] degrees
- 8 trials per level, parallelize with Pool(8)
- Convergence criterion: attitude error < 1.0 deg

### Exp 03 — Omega-Only Basin: Omega-Space vs L-Space (KEY EXPERIMENT)
**File**: `notebooks/inversion/exp03_omega_basin.py`
**Expected runtime**: ~10 min (parallelised)
**Method**:

Sub-A (omega-space):
- TRULY fix attitude: `f_omega_only(omega) = obj.evaluate([true_aa, omega])`
- 3 params, maxiter=100
- Levels: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] deg/s
- 8 trials per level, parallelised
- Convergence: omega error < 0.01 deg/s

Sub-B (L-space):
- TRULY fix attitude: `f_L_only(L) = ...` (convert L→omega via true q0, evaluate)
- Same levels converted to equivalent L perturbation: `delta_L = R0.T @ (I @ delta_omega)`
- Same 8 trials, maxiter=100
- Convergence: recovered omega error < 0.01 deg/s

**Output**: Side-by-side comparison table + bar chart

### Phase 2 Results
_(fill in after running)_
```
Exp 02 attitude-only basin:
  1 deg:  ?/8 converged
  2 deg:  ?/8
  5 deg:  ?/8
  10 deg: ?/8
  15 deg: ?/8
  20 deg: ?/8

Exp 03 omega-only basin (HEADLINE RESULT):
  Level(dps)  Omega-param  L-param  Ratio
  0.005       ?/8          ?/8      ?x
  0.01        ?/8          ?/8      ?x
  0.02        ?/8          ?/8      ?x
  0.05        ?/8          ?/8      ?x
  0.1         ?/8          ?/8      ?x
  0.2         ?/8          ?/8      ?x
  0.5         ?/8          ?/8      ?x
  1.0         ?/8          ?/8      ?x

VERDICT: L-param basin is ???x wider / same / narrower
DECISION: Proceed to Phase 3 with [omega-param / L-param]
```

---

## PHASE 3: Joint Basin + Window Effects (Exp 04, Exp 05)

### Prerequisites
- Phase 2 complete
- Decision on parameterization made

### Instructions for Claude
Read Phase 2 results first. If L-param won, use it for all subsequent experiments.
Write `exp04_joint_basin.py` and `exp05_window_basin.py`. Run, commit, update.

### Exp 04 — Joint Basin
**File**: `notebooks/inversion/exp04_joint_basin.py`
**Expected runtime**: ~15 min
**Only run if Exp 03 shows L-param advantage** (otherwise skip to Exp 05)
- Full 6-param L-BFGS-B
- Joint levels: [(1,0.01), (3,0.05), (5,0.1), (5,0.5), (10,1.0)] in (deg, deg/s)
- 8 trials per level, parallelised, maxiter=100
- Both omega-param and L-param
- Convergence: att < 5 deg AND omega < 0.1 deg/s

### Exp 05 — Basin Width vs Window Length
**File**: `notebooks/inversion/exp05_window_basin.py`
**Expected runtime**: ~10 min
- Window lengths: [50, 100, 200, 500, 1000, 3600] seconds
- Use best parameterization from Phase 2
- For EACH window: joint perturbation at (5 deg, 0.5 deg/s), 8 trials
- Measure success rate at each window length
- Plot: success rate vs window length

### Phase 3 Results
_(fill in after running)_

---

## PHASE 4: Filtering + Progressive Window (Exp 06-09)

### Prerequisites
- Phase 3 complete

### Instructions for Claude
Based on Phase 3 results, write the relevant experiments.
Exp 06 (progressive window) is high priority if Exp 05 shows short windows help.
Exp 07-09 (candidate generation + conservation filtering) are independent.

### Exp 06 — Progressive Window Refinement
### Exp 07 — Iso-Brightness Candidate Stats
### Exp 08 — Two-Epoch Pairing Stats
### Exp 09 — Conservation Filter Culling

_(Details in experiment_roadmap.md)_

---

## PHASE 5: Compile Roberto Update

### Instructions for Claude
Read all results from Phases 1-4. Write a concise summary suitable for emailing to Roberto.
Save to `notebooks/inversion/roberto_update_feb23.md`.

Key points to cover:
1. Timing benchmarks
2. L-param vs omega-param verdict
3. Window length effect
4. Conservation filtering (if we got there)
5. Next steps / what's needed

---

## MICRO-EXPERIMENT SERIES (Peak Anchoring)

### Micro-13: Roberto's Full Graph Pipeline (2026-02-27)
**File**: `notebooks/inversion/micro13_graph_pipeline.py`
**Commit**: 83314f5
**Runtime**: 2712s (~45 min)

**Design**: 3-peak graph pipeline (50 candidates × 3 peaks, L-BFGS-B bridge optimisation
with full Euler dynamics, intermediate lo-fi brightness scoring, shortest-path search).

**Results**:
```
Stage 1 — Candidate generation (1M SO(3) per peak):
  Peak 1 (ep 183): 1,447 hits → 50 cands, nearest = 3.89°
  Peak 2 (ep 260): 1,400 hits → 50 cands, nearest = 7.13°
  Peak 3 (ep 360): 2,987 hits → 50 cands, nearest = 0.56°

Stage 2 — Bridge optimisation (L-BFGS-B, Euler dynamics):
  Leg 0 (dt=555s): 2462/2500 feasible (<5° mismatch) = 98.5%
  Leg 1 (dt=721s): 2464/2500 feasible = 98.6%

Stage 3 — Intermediate brightness scoring:
  Leg 0: RMS median=2.44, min=1.13 (truth bridge RMS=1.53)
  Leg 1: RMS median=2.27, min=1.32 (truth bridge RMS=2.25)

Stage 4 — Graph search:
  Valid paths: 121,326 / 125,000 (97%)
  Truth path rank: #11,979 / 121,326  ← NO DISCRIMINATION
  Top-10 paths: all have 20-179° attitude errors
```

**Key Findings**:
1. Bridge optimisation connects ~98.5% of arbitrary pairs — 3 free ω DOF trivially
   connect any two attitudes (confirms micro-04)
2. Intermediate lo-fi brightness scoring provides ZERO discrimination —
   the ~0.15 mag/epoch lo-fi vs hi-fi systematic offset dominates the cost
3. Truth path is ranked ~10th percentile, indistinguishable from random

**Implication**: The graph pipeline needs either:
- Hi-fi intermediate scoring (expensive but eliminates systematic offset)
- Shape-based / relative scoring (ignore absolute brightness offset)
- A different discriminating signal (e.g., conservation constraints)

---

## GIT COMMIT LOG

| Time | Commit | Files |
|------|--------|-------|
| Phase 1 | 56dbc4f | exp00_timing.py, exp01_sanity.py, progress_tracker.md, exp00_timing.json, exp01_sanity.json |
| Micro-13 | 83314f5 | micro13_graph_pipeline.py, micro13_*.npz, micro13_graph_pipeline.json |

---

## BACKGROUND: Mega-Script Status

The original `exp_conservation_and_L_param.py` has been running since Friday afternoon.
- Part A.1 finished in 37.7 hours
- Part A.2 was at 6/7 levels after 33 hours (as of 10:30 AM Feb 23)
- Bug: "attitude held at truth" isn't actually held (all 6 params free)
- Results are still useful for L-vs-omega comparison but basin widths are pessimistic
- Let it finish — compare with our clean isolated results later
