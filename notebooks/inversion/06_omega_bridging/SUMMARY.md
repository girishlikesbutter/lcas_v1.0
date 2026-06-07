# Omega Bridging Benchmarks

Given two quaternion endpoints (q1, q2) separated by a known time gap dt,
recover the angular velocity omega that connects them under torque-free
tumbling dynamics. These experiments benchmark the feasibility, cost, and
noise sensitivity of this "omega bridging" operation.

All experiments use the Intelsat 901 inertia tensor and the standard fast
tumbler: omega_true = [0.5, -0.3, 2.0] deg/s, dt = 50s.

## Experiments

### archive/bench_omega_bvp.py — Single-pair recovery

Propagate q1 forward by omega_true to get q2. Treat omega as unknown and
recover it by minimising `1 - |dot(q_arrived, q2)|` (quaternion geodesic
distance). Initial guess: principal-axis approximation (axis-angle of
q1_inv * q2, divided by dt).

**Results (3 optimizers):**

| Optimizer   | Wall time | Evals | Omega error (deg/s) |
|-------------|-----------|-------|---------------------|
| L-BFGS-B    | 162 ms    | 44    | 0.000001            |
| Nelder-Mead | 480 ms    | 297   | ~0                  |
| Powell      | 1663 ms   | 327   | ~0                  |

**Conclusion:** L-BFGS-B is the clear winner. 44 evaluations, ~162 ms,
sub-microdegree/s accuracy. The principal-axis guess starts within ~0.43
deg/s of truth, which is close enough for reliable convergence.

### archive/bench_omega_bvp_parallel.py — Parallel scaling (n_c^2 pairings)

Solve n_c^2 independent omega BVPs in parallel using multiprocessing.Pool(8).
Tests the combinatorial use case: n_c candidate quaternions at epoch 1 crossed
with n_c candidates at epoch 2.

**Results (n_c = 50, 2500 pairs, random quaternion candidates):**

| Metric              | Value         |
|---------------------|---------------|
| Total wall time     | 177 s         |
| Per-pair cost       | 70.9 ms       |
| Mean function evals | 85            |
| Max omega error     | converged     |
| Speedup vs seq.     | ~4.9x on 8 cores |

Per-pair cost is higher than trajectory-consecutive pairs (70.9 vs 28.6 ms)
because random quaternion pairings produce larger omega magnitudes and require
more optimizer iterations.

**Scaling estimates (8 cores, random pairings):**

| n_c  | Pairs    | Estimated time |
|------|----------|----------------|
| 10   | 100      | ~7 s           |
| 20   | 400      | ~28 s          |
| 50   | 2,500    | ~3 min         |
| 100  | 10,000   | ~12 min        |

### archive/bench_omega_bvp_noisy.py — Noise sensitivity

Ground truth q1 and q2 from exact propagation. Before recovery, perturb both
endpoints by a controlled angular error (0.5 to 10 degrees). Measures how
quaternion noise maps to omega recovery error.

**Results (10 trials averaged per noise level):**

| q error (deg) | omega error (deg/s) | Time (ms) |
|----------------|---------------------|-----------|
| 0.5            | 0.013               | 205       |
| 1.0            | 0.025               | 198       |
| 1.5            | 0.033               | 196       |
| 2.0            | 0.060               | 230       |
| 2.5            | 0.062               | 199       |
| 3.0            | 0.070               | 192       |
| 3.5            | 0.074               | 187       |
| 4.0            | 0.089               | 255       |
| 4.5            | 0.119               | 260       |
| 5.0            | 0.139               | 208       |
| 5.5            | 0.153               | 195       |
| 6.0            | 0.162               | 200       |
| 6.5            | 0.179               | 207       |
| 7.0            | 0.188               | 222       |
| 7.5            | 0.212               | 172       |
| 8.0            | 0.177               | 217       |
| 8.5            | 0.186               | 204       |
| 9.0            | 0.211               | 218       |
| 9.5            | 0.261               | 209       |
| 10.0           | 0.254               | 220       |

**Conclusions:**
- Timing is flat (~200 ms) regardless of noise. The optimizer always converges
  to machine-precision cost; it perfectly fits the noisy endpoints.
- Omega error scales roughly linearly: ~0.025 deg/s per degree of q error.
- The error is entirely from input noise, not convergence failure.

## Key Insights

1. **Omega bridging is cheap.** A single solve costs ~160-200 ms with L-BFGS-B
   (44-85 function evaluations, each a 50s tumbling propagation at ~3 ms).

2. **Parallelism works.** 2500 pairs on 8 cores in ~3 minutes. Scaling is
   roughly linear in pair count with ~5x speedup on 8 cores.

3. **Principal-axis guess is reliable.** The axis-angle/dt approximation
   always lands in the convergence basin, even for fast tumblers and random
   quaternion pairings.

4. **Noise propagation is predictable.** 1 degree of quaternion endpoint error
   produces ~0.025 deg/s omega error over a 50s gap. This is a geometric
   property of the problem, not an optimizer limitation.

## Propagator Benchmarks (supplementary)

From earlier timing runs (not in scripts above):

| Integration window | Time   | Cost per sim-second |
|--------------------|--------|---------------------|
| 100s               | 3.0 ms | 0.030 ms/s          |
| 500s               | 11 ms  | 0.023 ms/s          |
| 3600s              | 61 ms  | 0.017 ms/s          |

Principal-axis mode (closed-form): 2.0 ms for 3600s — 30x faster than
tumbling but only valid for rotation about a principal axis.
