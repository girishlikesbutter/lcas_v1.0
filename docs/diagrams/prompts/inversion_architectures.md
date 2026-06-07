# Functional Block Diagram Prompts: 7 Inversion Architectures

These are 7 structurally distinct pipeline architectures that have been tried to solve the **satellite attitude inversion problem**: recovering the initial attitude (quaternion q0) and angular velocity (omega0) of a tumbling satellite from its observed brightness time-series (light curve).

All 7 share a common **forward model** (described separately at the end) but differ in their search strategy, parameterization, and filtering logic.

**Common test case across all architectures:**
- Satellite: Intelsat 901 (~10,000 triangular facets)
- True attitude: q0 from axis=[0.6, 0.3, 0.8]/norm, angle=45 deg
- True angular velocity: omega0 = [0.5, -0.3, 2.0] deg/s ("fast tumbler")
- Observation window: 3600 seconds (1 hour), 500 epochs, dt ~7.2s
- Observed light curve: hi-fi (ray-traced shadows) + Gaussian noise (sigma=0.05 mag)
- Parameter space: 6D — axis-angle representation of q0 (3 params) + omega0 (3 params)

---
---

# Architecture 1: Dense Grid Search

Create a functional block diagram for a brute-force grid enumeration approach to satellite attitude inversion.

## Inputs
- **Observed light curve**: brightness vs. time (500 epochs), generated with hi-fi model + noise
- **Satellite model**: 3D mesh with BRDF material properties (loaded, ready to evaluate)
- **Forward model**: given (q0, omega0) → propagate attitude → generate synthetic light curve → return residual vs. observed

## Block 1: Grid Construction

**Label:** `Construct 6D Parameter Grid`

| | |
|---|---|
| **Inputs** | Attitude bounds: [-pi, pi] per axis-angle component (3 axes); Omega bounds: [-0.15, +0.15] deg/s per axis (3 axes) |
| **Processing** | Create uniform meshgrid: N_ATT points per attitude axis, N_OMG points per omega axis |
| **Outputs** | Grid of 6D parameter vectors. Typical size: 8^3 x 5^3 = 64,000 points |

## Block 2: Lo-Fi Objective Evaluation (Batch)

**Label:** `Evaluate Lo-Fi Residual at Every Grid Point`

| | |
|---|---|
| **Inputs** | 64,000 parameter vectors (each: [axis_angle(3), omega(3)]) |
| **Processing** | For each point: propagate attitude via Euler's equations (DOP853 ODE solver) → compute synthetic light curve WITHOUT ray-traced shadows (lo-fi) → compute MSE residual vs. observed |
| **Outputs** | 64,000 (parameter, residual) pairs |
| **Timing** | ~0.2s per eval x 64,000 = ~3.5 hours sequential; parallelized across 8 cores |

## Block 3: Top-K Selection

**Label:** `Select Top-K Candidates by Residual`

| | |
|---|---|
| **Inputs** | 64,000 (parameter, residual) pairs |
| **Processing** | Sort by residual ascending; keep top K=20 |
| **Outputs** | 20 candidate parameter vectors |

## Block 4: Lo-Fi Local Refinement

**Label:** `L-BFGS-B Polish (Lo-Fi)`

| | |
|---|---|
| **Inputs** | 20 candidate starting points |
| **Processing** | For each: gradient-based local optimizer (L-BFGS-B) on 6D lo-fi objective, maxiter=100, maxfun=500, bounded |
| **Outputs** | 20 refined parameter vectors with lo-fi residuals |

## Block 5: Hi-Fi Verification

**Label:** `L-BFGS-B Polish (Hi-Fi, with Shadows)`

| | |
|---|---|
| **Inputs** | 20 lo-fi-refined candidates |
| **Processing** | For each: L-BFGS-B on hi-fi objective (ray-traced shadows enabled), maxiter=30, maxfun=60 |
| **Outputs** | 20 final candidates with hi-fi residuals and attitude/omega errors vs. truth |
| **Success criterion** | Attitude error < 5 deg AND omega error < 0.1 deg/s |

## Block 6: Best Solution Selection

**Label:** `Rank & Select Best`

| | |
|---|---|
| **Inputs** | 20 hi-fi-verified candidates with error metrics |
| **Processing** | Rank by composite score: att_err + 10 x omega_err |
| **Outputs** | Single best (q0, omega0) estimate |

## Data Flow
```
[Grid Construction] ──64,000 points──→ [Lo-Fi Eval] ──sorted──→ [Top-20] ──→ [Lo-Fi L-BFGS-B] ──→ [Hi-Fi L-BFGS-B] ──→ [Best Solution]
```

## Variant: Decoupled Grid (show as alternative path from Block 1)
Instead of evaluating all 6D combinations:
1. Grid only over attitude (12^3 = 1,728 points)
2. For each attitude grid point: run a small 3-parameter L-BFGS-B to find best omega (maxfun=30)
3. This decouples the narrow omega basin from the attitude search
4. Then proceed to Blocks 4-6 as before

## Visual Notes
- Block 2 is the bottleneck (most wall-clock time)
- The transition from lo-fi (Block 2-4) to hi-fi (Block 5) is the "fidelity handoff" — show as a color change
- The decoupled variant replaces Block 1+2 with a nested loop (show as an inset or alternative branch)

---
---

# Architecture 2: Global Derivative-Free Optimizer

Create a functional block diagram for a global optimization approach (dual annealing / simulated annealing) applied directly to the 6D inversion problem.

## Inputs
- **Observed light curve**: 500 epochs
- **Forward model**: (q0, omega0) → lo-fi residual
- **FFT period estimate**: used to bound omega search range

## Block 1: FFT Period Analysis

**Label:** `Estimate Dominant Period via FFT`

| | |
|---|---|
| **Inputs** | Observed light curve (500 values), sampling interval dt |
| **Processing** | Lomb-Scargle periodogram with 10,000 frequency points; find dominant frequency f_dom |
| **Outputs** | Omega upper bound: omega_bound = 5.0 x 2*pi x f_dom (rad/s) — generous safety margin |

## Block 2: Dual Annealing (Global Search)

**Label:** `Dual Annealing on 6D Lo-Fi Objective`

| | |
|---|---|
| **Inputs** | Lo-fi objective function, 6D bounds: attitude in [-pi, pi]^3, omega in [-omega_bound, +omega_bound]^3 |
| **Processing** | scipy.optimize.dual_annealing(): generalized simulated annealing with embedded local L-BFGS-B. Custom annealing schedule (initial_temp=5230, visit=2.62). Hard timeout: 8 minutes. |
| **Outputs** | Single best 6D parameter vector found within budget (~500-2000 evaluations) |
| **Key property** | Can escape local minima via stochastic jumps; no grid needed |

## Block 3: Hi-Fi Handoff

**Label:** `L-BFGS-B Polish (Hi-Fi)`

| | |
|---|---|
| **Inputs** | Best point from dual annealing |
| **Processing** | L-BFGS-B on hi-fi objective (shadows enabled), maxiter=30, maxfun=60 |
| **Outputs** | Final (q0, omega0) estimate with hi-fi residual |

## Data Flow
```
[Observed LC] ──→ [FFT Period] ──omega bounds──→ [Dual Annealing, 8 min] ──best point──→ [Hi-Fi L-BFGS-B] ──→ [Solution]
```

## Visual Notes
- Block 2 is a single monolithic optimizer call — show it as a large block with internal stochastic + local search loops
- The FFT block provides critical bounds that make the search tractable
- Very compact pipeline (only 3 blocks) compared to other architectures
- Show the 8-minute timeout as a clock/timer annotation on Block 2

---
---

# Architecture 3: Multi-Start Local Optimization

Create a functional block diagram for a parallel multi-start approach with L-BFGS-B local optimization from many random initial points.

## Inputs
- **Observed light curve**: 500 epochs
- **Forward model**: lo-fi and hi-fi objective functions
- **Omega bounds**: multiple levels tested (0.1, 0.05, 0.02 deg/s)

## Block 1: Random Initialization

**Label:** `Generate Random Starting Points`

| | |
|---|---|
| **Inputs** | N_STARTS=500 per omega bound level; 3 levels |
| **Processing** | For each start: sample attitude uniformly from [-pi, pi]^3, omega uniformly from [-omega_bound, +omega_bound]^3 |
| **Outputs** | 1,500 random 6D starting points (500 per level) |

## Block 2: Parallel Lo-Fi L-BFGS-B

**Label:** `L-BFGS-B from Each Start (Lo-Fi)`

| | |
|---|---|
| **Inputs** | 1,500 starting points, lo-fi objective |
| **Processing** | For each: L-BFGS-B with bounds, maxiter=30, maxfun=100. Parallelized across 8 cores. |
| **Outputs** | 1,500 converged points with lo-fi residuals |
| **Timing** | ~100 lo-fi evals per start x 0.2s = ~20s per start; 1,500 / 8 cores ~ 1 hour |

## Block 3: Per-Level Best Selection

**Label:** `Select Best per Omega Bound Level`

| | |
|---|---|
| **Inputs** | 1,500 results grouped into 3 levels of 500 |
| **Processing** | Within each level: rank by residual, select best |
| **Outputs** | 3 best candidates (one per omega bound level) + overall best |

## Block 4: Hi-Fi Handoff

**Label:** `L-BFGS-B Polish (Hi-Fi)`

| | |
|---|---|
| **Inputs** | 3 best candidates + overall best |
| **Processing** | L-BFGS-B on hi-fi objective, tight omega bounds, maxiter=50, maxfun=100 |
| **Outputs** | Final verified candidates with hi-fi residuals and error metrics |

## Data Flow
```
                ┌─── Level 1 (±0.10 °/s): 500 starts ──→ [Lo-Fi L-BFGS-B] ──→ best_1
[Random Init] ──┼─── Level 2 (±0.05 °/s): 500 starts ──→ [Lo-Fi L-BFGS-B] ──→ best_2  ──→ [Hi-Fi L-BFGS-B] ──→ [Solution]
                └─── Level 3 (±0.02 °/s): 500 starts ──→ [Lo-Fi L-BFGS-B] ──→ best_3
```

## Variant: Alternating Sub-Problem Optimization (show as alternative Block 2)
Instead of joint 6D L-BFGS-B from each start:
1. Fix attitude, optimize omega only (3 params, maxfun=40) — exploits wider omega basin when attitude is held
2. Fix omega, optimize attitude only (3 params, maxfun=40)
3. Repeat 5 times (alternating)
4. Then joint 6D polish from the alternated result
- This costs ~400 evals per start vs. 100, but explores the landscape better

## Visual Notes
- Show the 3 omega levels as parallel lanes
- Block 2 is the dominant compute cost — annotate with parallelism (8 cores)
- The alternating variant replaces Block 2's internals — show as a sub-diagram with a loop arrow

---
---

# Architecture 4: Basin Characterization (Convergence Radius Mapping)

Create a functional block diagram for a diagnostic pipeline that maps the convergence basin of the inversion problem — measuring how far from truth you can start and still converge.

**Note:** This architecture does NOT solve the inversion problem. It characterizes its difficulty by answering: "Given a starting point at distance delta from truth, does L-BFGS-B converge to truth?"

## Inputs
- **True parameters**: known q0 and omega0
- **Forward model**: lo-fi or hi-fi objective function
- **Perturbation schedule**: list of angular distances to test

## Block 1: Controlled Perturbation Generation

**Label:** `Generate Perturbed Starting Points`

| | |
|---|---|
| **Inputs** | True (q0, omega0); perturbation levels for attitude (1, 2, 5, 10, 15, 20 deg) and/or omega (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 deg/s) |
| **Processing** | For each level: generate N_TRIALS=8 random perturbation directions at exactly that geodesic distance from truth |
| **Outputs** | Structured set of perturbed starting points, each tagged with (delta_att, delta_omega) |

## Block 2: Optimization from Each Perturbed Start

**Label:** `L-BFGS-B from Perturbed Start`

| | |
|---|---|
| **Inputs** | Perturbed starting points; objective function (lo-fi or hi-fi) |
| **Processing** | For each: L-BFGS-B with bounds, maxiter=100. Can test 3 sub-problems: attitude-only (freeze omega at truth), omega-only (freeze attitude at truth), or joint 6D. Parallelized via Pool(8). |
| **Outputs** | Per trial: converged parameters, final attitude error (deg), final omega error (deg/s) |

## Block 3: Convergence Classification

**Label:** `Classify Convergence`

| | |
|---|---|
| **Inputs** | Per-trial final errors |
| **Processing** | Converged if: attitude error < 1 deg (or omega error < threshold). Compute convergence rate per perturbation level. |
| **Outputs** | Convergence rate table: perturbation_level vs. fraction_converged |

## Block 4: Basin Boundary Identification

**Label:** `Identify Basin Radius`

| | |
|---|---|
| **Inputs** | Convergence rate table |
| **Processing** | Find largest perturbation level with >50% convergence rate |
| **Outputs** | Basin radius estimates: attitude-only ~5-6 deg, omega-only ~0.02 deg/s, joint 6D: <1 deg + <0.001 deg/s |

## Data Flow
```
[True Params] ──→ [Perturb at delta] ──→ [L-BFGS-B] ──→ [Converged?] ──→ [Basin Radius Table]
                       │                       │
                       └── repeat for ─────────┘
                           each (delta, trial)
```

## Sub-Problem Variants (show as 3 parallel lanes in Block 2)
- **Lane A: Attitude-only** — freeze omega at truth, optimize 3 attitude params → measures attitude basin width
- **Lane B: Omega-only** — freeze attitude at truth, optimize 3 omega params → measures omega basin width
- **Lane C: Joint 6D** — all 6 params free → measures the catastrophically narrow joint basin

## Key Results (annotate on diagram)
- Attitude-only basin: ~5-6 deg (workable)
- Omega-only basin: ~0.02 deg/s (very narrow)
- Joint 6D basin: <1 deg attitude AND <0.001 deg/s omega simultaneously (essentially unsolvable by local search alone)

## Visual Notes
- This is a diagnostic/scientific pipeline, not a solver — use a different visual style (e.g., dashed borders or a "measurement" icon)
- The 3 sub-problem lanes are the key insight — the joint basin is dramatically narrower than either sub-basin alone
- Show the output as a heatmap or convergence-vs-distance plot rather than a single answer

---
---

# Architecture 5: Isobrightness Filtering Cascade

Create a functional block diagram for a multi-epoch brightness-matching pipeline that exploits the many-to-one relationship between attitude and brightness.

**Key insight:** At any single epoch, many different attitudes produce the same observed brightness. This "isobrightness set" is a 2D manifold on SO(3). By matching brightness at multiple epochs and applying physical constraints, the candidate set can be progressively narrowed.

## Inputs
- **Observed light curve**: 500 epochs
- **Forward model**: single-epoch brightness function (lo-fi and hi-fi)
- **FFT omega bound**: maximum plausible angular velocity

## Block 1: Epoch Selection

**Label:** `Select Well-Separated Screening Epochs`

| | |
|---|---|
| **Inputs** | Observed light curve, number of screening epochs (2-4) |
| **Processing** | Choose epochs spread across the observation window to maximize independent information (e.g., indices [0, 125, 250, 375]) |
| **Outputs** | 2-4 epoch indices with their observed brightness values |

## Block 2: Isobrightness Candidate Generation (per epoch)

**Label:** `Find Attitudes Matching Observed Brightness`

| | |
|---|---|
| **Inputs** | Target brightness at epoch i; N_SEEDS random SO(3) starting attitudes (200-10,000) |
| **Processing** | For each seed: L-BFGS-B minimization of (predicted_brightness - target_brightness)^2 over 3 attitude parameters (axis-angle). Lo-fi evaluation. maxiter=50. Parallelized across 8 cores. |
| **Outputs** | Candidate set per epoch: list of attitudes matching brightness within tolerance (typically 50-500 converged candidates per epoch) |
| **Key property** | Single-epoch brightness has massive degeneracy: 5-20% of random SO(3) attitudes match within 5% brightness tolerance |

**Show this block repeated for each of the 2-4 selected epochs (parallel lanes).**

## Block 3: Cross-Epoch Rotation-Angle Culling

**Label:** `Filter by Physical Rotation Constraint`

| | |
|---|---|
| **Inputs** | Candidate sets from 2 epochs; time gap dt between them; omega_bound from FFT |
| **Processing** | For each pair (q_a from epoch i, q_b from epoch j): compute rotation angle between them. max_allowed_angle = omega_bound x dt. Keep only pairs where rotation_angle < max_allowed_angle. Vectorized quaternion algebra. |
| **Outputs** | Feasible pairs: list of (q_a, q_b, implied_omega) tuples. Typically 99%+ of pairs are culled. |

## Block 4: Multi-Epoch Brightness Validation

**Label:** `Validate at Additional Epochs`

| | |
|---|---|
| **Inputs** | Feasible pairs with derived omega; validation epoch indices (e.g., [10, 25, 40, 60, 75, 90]) |
| **Processing** | For each pair: propagate q_a forward by omega x dt to each validation epoch. Evaluate predicted brightness. Check |predicted - observed| < 3 x noise_sigma. Progressive culling at each validation epoch. |
| **Outputs** | Surviving pairs after multi-epoch validation (typically 10-100) |

## Block 5: Full Light Curve Ranking

**Label:** `Evaluate Full Lo-Fi Light Curve`

| | |
|---|---|
| **Inputs** | Surviving pairs (q0, omega0) |
| **Processing** | For each: propagate attitude over full 500-epoch window. Generate complete lo-fi light curve. Compute RMS residual vs. observed. |
| **Outputs** | Ranked list of (q0, omega0) candidates by RMS residual |

## Block 6: Hi-Fi Refinement (Optional)

**Label:** `Hi-Fi L-BFGS-B Polish`

| | |
|---|---|
| **Inputs** | Top 5 candidates from lo-fi ranking |
| **Processing** | L-BFGS-B on hi-fi objective (shadows enabled), maxiter=200 |
| **Outputs** | Final refined (q0, omega0) estimates |

## Data Flow
```
[Observed LC] ──→ [Pick 4 Epochs] ──→ ┌─ Epoch 0: 10k seeds → 200 candidates ─┐
                                       ├─ Epoch 125: 10k seeds → 300 candidates ├──→ [Rotation-Angle Culling]
                                       ├─ Epoch 250: 10k seeds → 250 candidates ─┤      (pairwise filtering)
                                       └─ Epoch 375: 10k seeds → 180 candidates ─┘           │
                                                                                         500 pairs
                                                                                              │
                                                                                    [Validate at 6 epochs]
                                                                                              │
                                                                                          50 pairs
                                                                                              │
                                                                                    [Full LC Ranking]
                                                                                              │
                                                                                    [Hi-Fi Polish top 5]
                                                                                              │
                                                                                         [Solution]
```

## Variant: Mixed-Fidelity Hierarchical Search (alternative Block 2)
Instead of optimizer-based isobrightness search:
1. Sample 100,000 random SO(3) attitudes (no optimization)
2. Evaluate lo-fi brightness at target epoch for all 100k (batch evaluation, ~30s)
3. Select top 200 closest to target brightness
4. Refine these 200 with hi-fi L-BFGS-B (parallel, 8 workers)
- Faster than optimizer-per-seed but requires larger sample size

## Visual Notes
- Block 2 is repeated per epoch — show as parallel vertical lanes converging into Block 3
- The culling funnel (10,000 → 200 → 50 → 5) should be visually prominent — perhaps as a narrowing funnel shape
- The rotation-angle constraint (Block 3) is the key physical filter — highlight it
- Show candidate counts at each transition point

---
---

# Architecture 6: Epoch Chaining / Forward Propagation

Create a functional block diagram for a forward-propagation approach that chains attitude candidates across consecutive epochs using Euler's equations and omega-consistency filtering.

**Key insight:** If you know the attitude at epoch t and the angular velocity omega, Euler's equations uniquely determine the attitude at epoch t+dt. This lets you "chain" candidates forward, culling those whose predicted brightness doesn't match observation.

## Inputs
- **Observed light curve**: 500 epochs
- **Forward model**: single-epoch brightness function
- **Euler dynamics propagator**: given (q, omega, dt, inertia_tensor) → q_next
- **Conservation laws**: kinetic energy T and angular momentum L are constant for torque-free rotation

## Block 1: Identify Chain Epochs

**Label:** `Select Chain of Consecutive Epochs`

| | |
|---|---|
| **Inputs** | Observed LC, desired chain length (3-10 epochs), sampling dt (~7.2s) |
| **Processing** | Choose N_CHAIN consecutive epochs with good brightness variation (max brightness range). Or choose brightness-separated epochs with large magnitude differences (>0.5 mag). |
| **Outputs** | Chain epoch indices (e.g., 3 epochs: [i1, i2, i3]) with their observed brightness values |

## Block 2: Candidate Generation at Each Chain Epoch

**Label:** `Isobrightness Sampling at Each Epoch`

| | |
|---|---|
| **Inputs** | Target brightness per epoch; N_SEEDS=5,000-10,000 random SO(3) seeds |
| **Processing** | Same as Architecture 5 Block 2: L-BFGS-B on brightness-matching objective per seed, lo-fi. Parallelized. |
| **Outputs** | Per epoch: 50-500 candidate attitudes matching observed brightness within tolerance |

## Block 3: Pairwise Rotation-Angle Filtering (Forward Pass)

**Label:** `Forward Chain: Rotation-Angle Culling`

| | |
|---|---|
| **Inputs** | Candidate sets at epochs j and j+1; omega_bound x dt = max_angle |
| **Processing** | For each pair (q_j, q_{j+1}): compute quaternion rotation angle. Cull if angle > max_angle. Vectorized: compute dot product matrix of all q_j vs all q_{j+1}, threshold on arccos. |
| **Outputs** | Surviving candidate indices at each epoch. Typical culling rate: 90-99% per link. |
| **Key property** | After K links: survival rate ~ (1-cull_rate)^K — exponential narrowing |

**Show this block as a loop over chain links (j=0 to N_CHAIN-2).**

## Block 4: Backward Pass (Bidirectional Filtering)

**Label:** `Backward Chain: Ensure Full-Path Connectivity`

| | |
|---|---|
| **Inputs** | Forward-pass survivor sets per epoch |
| **Processing** | Propagate survival constraints backward: a candidate at epoch j survives only if it has at least one valid partner at epoch j+1 that itself survived the forward pass. |
| **Outputs** | Bidirectionally-filtered candidate sets — each survivor has a valid path through all chain epochs |

## Block 5: Omega Derivation & Consistency Check

**Label:** `Derive Omega from Attitude Pairs`

| | |
|---|---|
| **Inputs** | Surviving candidate pairs across consecutive epochs |
| **Processing** | For each pair (q_j, q_{j+1}): omega = rotation_vector(q_{j+1} * q_j^{-1}) / dt. For triplets: check omega_{12} ~ omega_{23} within tolerance (0.05 deg/s). |
| **Outputs** | Consistent triplets/paths with derived omega estimates |
| **Key filter** | Omega consistency: ||omega_{12} - omega_{23}|| < threshold. Uses KD-tree for efficient spatial matching in 3D omega space. |

## Block 6: Conservation Law Filtering (Optional)

**Label:** `Filter by Energy & Momentum Conservation`

| | |
|---|---|
| **Inputs** | Candidate triplets with derived omega; inertia tensor I |
| **Processing** | For each candidate: compute T = 0.5 * omega^T * I * omega and L = R(q) * (I * omega) in inertial frame. Check: T_epoch1 ~ T_epoch2 (within 1%). Check: L_direction_epoch1 ~ L_direction_epoch2 (within 1 deg). |
| **Outputs** | Physically consistent candidates — those satisfying conservation laws |

## Block 7: Full LC Validation & Ranking

**Label:** `Propagate & Score Full Light Curve`

| | |
|---|---|
| **Inputs** | Surviving (q0, omega0) pairs |
| **Processing** | Propagate attitude over full 500-epoch window. Generate lo-fi or hi-fi light curve. RMS residual vs. observed. |
| **Outputs** | Ranked solution candidates |

## Data Flow
```
[Select 7-10 Epochs] ──→ [Iso-brightness per epoch: 5k seeds → 100 cands each]
                                          │
                                [Forward Pass: rotation-angle culling]
                                    epoch 0 → 1: 99% culled
                                    epoch 1 → 2: 99% culled
                                    epoch 2 → 3: 99% culled
                                           ...
                                    epoch 8 → 9: 99% culled
                                          │
                                [Backward Pass: connectivity check]
                                          │
                                [Omega Consistency: triplet matching]
                                          │
                                [Conservation: T & L filtering]
                                          │
                                     0-5 survivors
                                          │
                                [Full LC Ranking] ──→ [Solution]
```

## Visual Notes
- The forward/backward chain is the defining visual element — show as a left-to-right chain of epoch nodes with bidirectional arrows
- The exponential culling is the key strength — annotate each link with the survival rate
- Conservation law filtering (Block 6) is a physics-based post-filter — show as a distinct colored block
- The chain length vs. culling tradeoff: longer chains = more culling power but more computational cost

---
---

# Architecture 7: Peak-Anchored Graph Pipeline

Create a functional block diagram for the most sophisticated inversion architecture: a graph-based pipeline that anchors candidate attitudes at light curve peaks, connects them via dynamically-optimized bridges, scores intermediate brightness, and finds the lowest-cost path through a connectivity graph.

**Key insight:** At light curve peaks, the brightness derivative dB/dt ~ 0, which means the brightness gradient g dotted with omega is near zero: g . omega ~ 0. Peaks are natural anchor points because: (1) they are easy to detect in the observed LC, (2) the zero-derivative constraint provides information about omega's direction relative to the brightness gradient.

## Inputs
- **Observed light curve**: 500 epochs with identified peak locations
- **Forward model**: single-epoch brightness function (lo-fi and hi-fi)
- **Euler dynamics propagator**: (q, omega, dt, I) → q(t+dt)
- **Brightness gradient**: dB/d(rotation) at any attitude — 3D vector

## Block 1: Peak Detection

**Label:** `Identify Light Curve Peaks`

| | |
|---|---|
| **Inputs** | Observed light curve (500 values) |
| **Processing** | scipy.signal.find_peaks with prominence threshold. Optionally: interpolate to find exact sub-sample peak time where dB/dt = 0. |
| **Outputs** | 3-5 peak epoch indices and their observed brightness values. Typical: peaks at indices [183, 260, 360] |

## Block 2: Dense SO(3) Sampling at Each Peak

**Label:** `Isobrightness Candidate Generation (1M samples)`

| | |
|---|---|
| **Inputs** | Target brightness at each peak; N_SAMPLES = 1,000,000 random SO(3) attitudes per peak |
| **Processing** | Batch evaluation: sample 100k at a time, evaluate lo-fi brightness, keep those within 1% of target brightness. Repeat 10 batches. Select top N_CAND=50 by residual from all matches. Inject truth at index 0 for tracking. Compute brightness gradient g at each candidate. |
| **Outputs** | Per peak: 50 candidate attitudes (quaternions) + 50 gradient vectors. Typical: 1,400-3,000 matches per 1M samples, top 50 kept. |

**Show this block repeated for each of the 3 peaks (parallel lanes).**

## Block 3: Bridge Optimization (L-BFGS-B on Omega)

**Label:** `Optimize Omega to Connect Peak Pairs`

| | |
|---|---|
| **Inputs** | Candidate pairs between consecutive peaks (50 x 50 = 2,500 per leg, 2 legs); time gaps dt between peaks |
| **Processing** | For each pair (q_start at peak i, q_end at peak i+1): L-BFGS-B minimizes over omega (3 params): objective = ||R(q_start, omega, dt) - q_end||^2 + lambda_grad * (g . omega)^2. Initial guess: omega_0 = rotation_vector(q_end * q_start^{-1}) / dt. The gradient penalty term encourages omega perpendicular to brightness gradient (consistent with being at a peak). Parallelized across 8 cores. |
| **Outputs** | Per leg: 50x50 matrix of (optimized_omega, arrival_error_deg). Prune bridges with arrival error > 5 deg. Typically 98-99% of bridges are feasible (3 DOF omega easily connects arbitrary pairs). |

## Block 4: Intermediate Brightness Scoring

**Label:** `Score Bridges at Intermediate Epochs`

| | |
|---|---|
| **Inputs** | Feasible bridges with optimized omega; N_INTER=10 intermediate time points per bridge |
| **Processing** | For each bridge: propagate q_start by omega through 10 intermediate times between peaks. At each intermediate epoch: evaluate brightness (lo-fi or hi-fi). Compute RMS residual of predicted vs. observed at intermediate epochs. Add omega magnitude penalty: cost = RMS + lambda_rate * max(0, |omega| - rate_prior)^2. |
| **Outputs** | Per bridge: intermediate RMS score and total cost. The intermediate scoring is where lo-fi vs. hi-fi matters most. |

**This is the critical discriminating step.** Show it prominently.

## Block 5: Graph Construction & Shortest Path

**Label:** `Build Path Graph & Find Shortest Path`

| | |
|---|---|
| **Inputs** | Bridge costs for leg 0 (peak 1 → peak 2): 50x50 matrix; bridge costs for leg 1 (peak 2 → peak 3): 50x50 matrix |
| **Processing** | Total path cost for any 3-node path (i, j, k): cost[i,j,k] = bridge_cost[0][i,j] + bridge_cost[1][j,k]. Enumerate all 50x50x50 = 125,000 possible paths. Sort by total cost. Find truth path rank. |
| **Outputs** | Ranked list of all valid paths. Each path specifies: (q0 at peak 1, omega_leg1, q at peak 2, omega_leg2, q at peak 3). Best path = lowest total cost. |

## Block 6: Hi-Fi Rescore (Optional Refinement)

**Label:** `Rescore Bridges with Hi-Fi Model`

| | |
|---|---|
| **Inputs** | All feasible bridges from Block 3 (loaded from checkpoint); hi-fi brightness function |
| **Processing** | Repeat Block 4 but using hi-fi evaluation (ray-traced shadows) instead of lo-fi at intermediate epochs. ~60x slower but captures shadow effects. Recompute bridge costs. Rebuild graph and re-rank paths. |
| **Outputs** | Re-ranked paths using hi-fi intermediate scoring |
| **Key finding** | Truth rank improved from #11,979/121,326 (lo-fi) to a better position (hi-fi), but lo-fi still fails to discriminate — the systematic 0.15 mag lo-fi/hi-fi offset dominates |

## Data Flow
```
[Observed LC] ──→ [Find 3 Peaks]
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Peak 1:       Peak 2:      Peak 3:
   1M SO(3) →    1M SO(3) →   1M SO(3) →
    50 cands      50 cands     50 cands
          │            │            │
          └────┬───────┴──────┬─────┘
               │              │
          Leg 0: 2500      Leg 1: 2500
          bridge opts      bridge opts
               │              │
          [Intermediate    [Intermediate
           Scoring]         Scoring]
               │              │
               └──────┬───────┘
                      │
              [Graph: 125k paths]
                      │
              [Sort by cost]
                      │
                 [Best Path]
                      │
           (optional: hi-fi rescore)
                      │
                 [Solution]
```

## Visual Notes
- The 3 peaks should be shown as anchor nodes on a timeline (x-axis = time)
- The bridges between peaks should look like arcs connecting anchor nodes
- The graph construction (Block 5) is the mathematical core — show it as a grid/matrix
- The lo-fi vs. hi-fi scoring split is the key open question — highlight it with contrasting colors
- Show the path enumeration as flowing through the graph
- Annotate with the key finding: "lo-fi scoring fails to discriminate — truth at rank #11,979"
- The gradient penalty (g . omega ~ 0 at peaks) is a subtle but important physics constraint — show as a callout on Block 3

---
---

# Shared Component: The Forward Model

All 7 architectures share this forward model as their inner evaluation function. Include it as a reference sub-diagram.

## Forward Model Block Diagram

**Purpose:** Given a candidate (q0, omega0), produce a predicted light curve and compare to observed.

### Sub-Block A: Attitude Propagation

| | |
|---|---|
| **Inputs** | Initial quaternion q0 (4,), angular velocity omega0 (3,) in body frame, epoch times (N,), inertia tensor I (3x3) |
| **Processing** | Integrate Euler's equations of torque-free rigid body rotation using DOP853 (8th-order Runge-Kutta). State = [q(4), omega(3)]. Torque-free: d(omega)/dt = I^{-1} x (I*omega x omega). dq/dt = 0.5 * q * omega_quat. |
| **Outputs** | Quaternion time series: q(t) for each epoch (N, 4). Angular velocity time series: omega(t) (N, 3). |
| **Timing** | ~77 ms for 500 epochs |

### Sub-Block B: Body-Frame Vector Computation

| | |
|---|---|
| **Inputs** | q(t) quaternions (N, 4); Sun and observer positions in J2000 (from SPICE, precomputed) |
| **Processing** | For each epoch: R = rotation_matrix(q(t)). k1 = R^T * sun_direction_J2000 (sun in body frame). k2 = R^T * obs_direction_J2000 (observer in body frame). Normalize both. |
| **Outputs** | k1_vectors (N, 3), k2_vectors (N, 3) |

### Sub-Block C: Shadow Computation (Hi-Fi Only)

| | |
|---|---|
| **Inputs** | Satellite mesh, k1_vectors (N, 3), articulation matrices |
| **Processing** | For each epoch: cast rays from every facet center toward sun. Check if ray intersects any other mesh face (trimesh ray-mesh intersection). Back-face culling: facets with n.k1 < 0 are automatically shadowed. |
| **Outputs** | lit_status: per-facet boolean mask (N, n_facets) — True if sunlit |
| **Timing** | ~60s for 500 epochs (hi-fi); skipped entirely for lo-fi (all facets marked lit) |

### Sub-Block D: BRDF Flux Calculation

| | |
|---|---|
| **Inputs** | k1, k2 vectors; facet normals, areas, BRDF params (r_d, r_s, n_phong); lit_status mask |
| **Processing** | For each epoch, for each facet: visibility check (n.k1 > 0 AND n.k2 > 0 AND lit). Compute halfway vector h = normalize(k1 + k2). Evaluate Ashikhmin-Shirley BRDF. Per-facet flux = rho * area * (n.k1) * (n.k2). Sum over all active facets → total_flux. |
| **Outputs** | total_flux (N,) |

### Sub-Block E: Magnitude Conversion

| | |
|---|---|
| **Inputs** | total_flux (N,), observer_distances (N,) in km |
| **Processing** | m = m_sun + 5*log10(distance_meters) - 2.5*log10(flux), where m_sun = -26.74 |
| **Outputs** | magnitudes (N,) — predicted light curve |

### Sub-Block F: Residual Computation

| | |
|---|---|
| **Inputs** | Predicted magnitudes (N,), observed magnitudes (N,) |
| **Processing** | MSE = mean((predicted - observed)^2) |
| **Outputs** | Scalar residual value (the objective function being minimized) |

### Forward Model Data Flow
```
(q0, omega0) ──→ [Propagate] ──→ q(t) ──→ [Body Vectors] ──→ k1, k2
                                                                  │
                                              ┌───────────────────┤
                                              ▼                   ▼
                                    [Shadows (hi-fi)]    [BRDF Calculation]
                                         │                       │
                                    lit_status ──────────────────→│
                                                                  │
                                                            total_flux
                                                                  │
                                                        [Mag Conversion]
                                                                  │
                                                         predicted LC
                                                                  │
                                                  [MSE vs Observed] ──→ residual (scalar)
```

### Timing Summary (annotate on diagram)
| Mode | Per full LC eval | Per single epoch | Ratio |
|------|-----------------|-----------------|-------|
| Lo-fi (no shadows) | 221 ms | 13 ms | 1x |
| Hi-fi (ray-traced) | 60,136 ms | 47 ms | 272x |

---
---

# Visual Style Guide (for all 7 diagrams)

- **Rounded rectangles** for processing blocks
- **Parallelograms** for data inputs/outputs
- **Diamond** for decision points (lo-fi vs. hi-fi switch, convergence check)
- **Funnel shape** for filtering/culling steps (wide top → narrow bottom with candidate counts)
- **Color coding by domain:**
  - **Blue**: initialization, configuration, data loading
  - **Green**: orbital mechanics, SPICE, attitude propagation
  - **Orange**: optimization (L-BFGS-B, dual annealing, CMA-ES)
  - **Red**: ray tracing, shadow computation
  - **Purple**: BRDF, flux, brightness evaluation
  - **Gold/Yellow**: final outputs, solutions
  - **Gray**: diagnostic/characterization (Architecture 4)
- **Annotations to include on every diagram:**
  - Candidate counts at each filtering stage
  - Timing estimates for each block
  - Fidelity mode (lo-fi vs. hi-fi) for each evaluation block
  - Parameter dimensionality at each stage (6D, 3D, scalar)
- **Common header** for all 7: show the architecture number, name, and one-sentence summary
- **Common footer**: link to the shared Forward Model sub-diagram
