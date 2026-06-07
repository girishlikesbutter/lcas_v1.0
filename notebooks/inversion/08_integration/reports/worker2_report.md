# Micro-27 Worker Report

## What was attempted

**Goal:** m027 — Single-bridge screening for candidate pair pruning. Test whether
a single cheap bridge solve per pair (L-BFGS-B from zero initial guess, minimising
arrival error) provides a screening signal to prune N² pairs before expensive full
band-sweep.

### Script written

`notebooks/inversion/08_integration/m027_bridge_screening.py` — 170 lines.

- Part A: 50 oracle-based candidates per peak (oracle + 1-3 deg random nudge),
  2500 pairs per leg, 2 legs.
- Part B: 50 m013 iso-brightness candidates per peak, same structure.
- Uses `multiprocessing.Pool(8)` with `set_start_method('fork')` (Python 3.14 fix).
- Plots 4-panel histogram (min-|omega| and residual distributions, true pair marked).

### Experiments run

1. **First run attempt** — killed after ~30 min, stuck on Part A leg 0. The 2500
   bridge solves were far too slow.

2. **Benchmarking** — systematic timing of bridge solve variants:

   | Approach | Per-solve time | Est. 2500/8 workers | Notes |
   |----------|---------------|---------------------|-------|
   | Full L-BFGS-B (rtol=1e-10) | **11s** | **3,419s** | 148 evals × ~74ms. Found |w|=4.18 deg/s |
   | Custom DOP853 (rtol=1e-6) | ~287s | 89,569s | Low accuracy → noisy gradients → L-BFGS-B diverges |
   | RK23 (rtol=1e-4) | ~471s | 147,159s | Same issue, worse |
   | Principal axis (closed form) | **5ms** | **2s** | Found |w|=57 deg/s — wrong local minimum, not physical |
   | Bounded L-BFGS-B (±5 deg/s) | 275ms | 86s | residual=0.77 — didn't converge, hit bounds after 1 iter |
   | Geodesic metric (analytical) | **0.001ms** | instant | |w|=0.314 deg/s — shortest-path only, no Euler dynamics |
   | Geodesic + Euler forward prop | **1.9ms** | **0.6s** | Arrival error doesn't discriminate (true pair err=0.116, random=0.097) |

## What results exist

- **Script:** `notebooks/inversion/08_integration/m027_bridge_screening.py` (uncommitted)
- **No results files** — the experiment never completed a full run.

## What went wrong

### 1. Bridge solves are ~70x slower than estimated

The task spec assumed ~160ms/solve. Actual cost with IS-901 inertia tensor and
tumbling-mode Euler dynamics over dt=555s is **~11s/solve** (full precision) or
**~2s/solve** (relaxed tolerances but then L-BFGS-B fails to converge).

Root cause: `propagate_attitude(..., "tumbling", I)` uses DOP853 with rtol=1e-10,
atol=1e-12. Each call is ~3.5ms at low omega but **~74ms** at the omega magnitudes
L-BFGS-B explores during optimization. With ~150 function evaluations per solve,
total is ~11s.

### 2. Cheaper alternatives don't provide screening signal

- **Principal axis:** Closed-form and fast (5ms) but L-BFGS-B finds arbitrary
  high-frequency local minima (57 deg/s) rather than the minimum-|omega| solution.
  The PA objective has many local minima (every time the rotation wraps around 2pi).

- **Geodesic metric:** Instant but gives ~0.31 deg/s for ALL pairs (true and random
  alike). The geodesic measures quaternion distance / dt, which is the same for any
  pair of candidates that are close on SO(3). Doesn't discriminate.

- **Geodesic + Euler propagation:** Propagate the geodesic omega through Euler
  dynamics and check arrival error. The arrival error is poor (~0.1) for all pairs
  because the geodesic omega (0.3 deg/s) is far from the true omega (2.1 deg/s).
  True pair and random pairs are indistinguishable.

### 3. Fundamental issue: the bridge objective is highly non-convex

For Euler dynamics over ~500s, the mapping omega → q_end has many oscillations.
L-BFGS-B from a single initial guess finds an arbitrary local minimum. The "minimum
|omega|" solution may not even be reachable from zero. This makes "single bridge
solve" an unreliable screening metric for this problem.

## Current state

- **Branch:** `exp/bridge-screening` (created fresh from `exp/integration-oracle`)
- **Running:** Nothing.
- **Uncommitted:** `m027_bridge_screening.py` script (functional but infeasible
  at N=50 due to solve cost), `WORKER_REPORT.md` (from prior m028 work).
- **No results committed.**

## Recommendations for next attempt

1. **Reduce N to ~15 candidates/peak** (225 pairs/leg). At 11s/solve / 8 workers,
   that's ~309s/leg, ~1236s total (~20 min). Borderline feasible.

2. **Use bounded L-BFGS-B with tighter range and geodesic initial guess** to keep
   omega small and propagations cheap. Needs testing to verify convergence.

3. **Consider alternative screening metrics** that don't require optimization:
   - Angular momentum consistency: compute L = R(q) @ (I @ omega_true) at each peak
     and check if L is consistent across peaks for a given pair.
   - Multi-epoch brightness consistency: evaluate a few intermediate epochs and
     check light curve fit.

4. **Accept that single-bridge screening may not work** for this problem. The
   non-convexity of the Euler bridge objective over ~500s time spans means a single
   L-BFGS-B solve doesn't reliably find the minimum-|omega| solution. The "two-phase
   screening" idea in EXPERIMENTS.md may need a different cheap metric.
