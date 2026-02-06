# PRD: Lightcurve Inversion Diagnostic Study

## Introduction

This is a **scientific investigation** to understand why the current lightcurve inversion approach (differential evolution) is slow and produces poor solutions. The study will systematically diagnose the objective function landscape, evaluate alternative optimization strategies, and quantify the conditions under which reliable inversion is achievable.

**Key question:** Is the problem with the optimizer, the problem formulation, or fundamental identifiability limitations?

## Goals

- Visualize the objective function landscape to understand its structure (multimodal? flat? well-conditioned?)
- Quantify the basin of attraction around the true solution
- Compare optimizer performance under fixed function evaluation budget
- Identify the minimum observation quality/quantity needed for reliable inversion
- Provide actionable recommendations for operational inversion settings

## Study Design

The study consists of **four focused notebooks**, executed in sequence. Each notebook produces specific outputs that inform the next. This incremental approach allows early termination if fundamental issues are discovered.

```
04_landscape_analysis.py      → Understand the problem structure
         ↓
05_basin_size_study.py        → Quantify constraint requirements
         ↓
06_optimizer_comparison.py    → Fair comparison of approaches
         ↓
07_robustness_analysis.py     → Sensitivity to noise/observations
```

## Success Criteria

An inversion is considered **successful** if BOTH:
- Angular velocity error < 0.1 deg/s (direction and magnitude)
- RMS residual < 2× noise sigma (fit quality)

---

## Notebook 1: Objective Landscape Analysis

**File:** `notebooks/inversion/04_landscape_analysis.py`

**Purpose:** Visualize the objective function to understand why optimization is difficult.

**Flow:**
1. Setup (reuse from 03) → Load satellite, compute geometry, generate synthetic lightcurve
2. Create ObjectiveFunction instance at true parameters
3. Compute 1D slices through objective (vary each of 6 parameters individually)
4. Compute 2D slices for key parameter pairs (attitude-attitude, omega-omega, attitude-omega)
5. Visualize and interpret results
6. Summary of findings

### US-001: Setup and synthetic data generation
**Description:** As a scientist, I need the same test case as notebook 03 so results are comparable.

**Acceptance Criteria:**
- [ ] Reuse Intelsat 901 config, geometry computation, and forward model from 03
- [ ] Extract into minimal setup (no redundant code)
- [ ] Store `true_params = np.concatenate([true_axis_angle, true_omega0])` for slicing
- [ ] Verify synthetic lightcurve matches 03 output

### US-002: 1D objective slices
**Description:** As a scientist, I want to see how the objective varies when changing one parameter at a time, to identify flat/steep directions.

**Acceptance Criteria:**
- [ ] Function `compute_1d_slice(objective_fn, true_params, param_index, delta_range, n_points)`
- [ ] Compute slice for each of 6 parameters: axis_angle (x,y,z), omega (x,y,z)
- [ ] Delta range: ±0.5 rad for axis-angle, ±0.01 rad/s for omega (adjustable)
- [ ] Plot all 6 slices in 2x3 subplot figure
- [ ] Mark true parameter location on each plot
- [ ] Log y-axis option to see structure near minimum

### US-003: 2D objective slices
**Description:** As a scientist, I want 2D heatmaps to see parameter correlations and identify ridges/valleys.

**Acceptance Criteria:**
- [ ] Function `compute_2d_slice(objective_fn, true_params, param_i, param_j, delta_i, delta_j, n_points)`
- [ ] Compute slices for pairs: (ax_x, ax_y), (ax_x, omega_z), (omega_x, omega_y), (omega_x, omega_z)
- [ ] Plot as heatmaps with contour overlay
- [ ] Mark true parameter location
- [ ] Identify and annotate any visible ridges, multiple minima, or flat regions

### US-004: Gradient analysis at true solution
**Description:** As a scientist, I want to know the condition number and gradient magnitudes to assess optimization difficulty.

**Acceptance Criteria:**
- [ ] Compute numerical gradient at true parameters using finite differences
- [ ] Compute numerical Hessian at true parameters
- [ ] Report eigenvalues of Hessian (condition number = max/min eigenvalue)
- [ ] Interpret: condition number > 1000 suggests ill-conditioning

### US-005: Landscape summary and findings
**Description:** As a scientist, I need a clear summary of what the landscape analysis reveals.

**Acceptance Criteria:**
- [ ] Markdown cell summarizing: Is there a unique minimum? How sharp? Any flat directions?
- [ ] Identify which parameters are well-constrained vs poorly constrained
- [ ] Preliminary assessment: Is global search necessary, or would local suffice?

---

## Notebook 2: Basin of Attraction Study

**File:** `notebooks/inversion/05_basin_size_study.py`

**Purpose:** Quantify how close initialization must be to guarantee convergence to true solution.

**Flow:**
1. Setup (copy from 04, minimal)
2. Define local optimizer wrapper (L-BFGS-B)
3. Systematic study: vary constraint radius, measure success rate
4. Identify critical radius threshold
5. Visualize and interpret

### US-006: Local optimizer wrapper
**Description:** As a scientist, I need a simple local optimization function to test convergence from different starting points.

**Acceptance Criteria:**
- [ ] Function `run_local_optimization(objective_fn, x0, bounds)` returning (x_opt, f_opt, success)
- [ ] Use scipy L-BFGS-B with sensible tolerances (ftol=1e-8, gtol=1e-6)
- [ ] Track number of function evaluations
- [ ] Return success based on our criteria (omega error < 0.1 deg/s AND rms < 2×sigma)

### US-007: Constrained random initialization
**Description:** As a scientist, I need to generate random starting points within a specified radius of the true solution.

**Acceptance Criteria:**
- [ ] Function `sample_in_ball(true_params, radius, n_samples, param_scales)`
- [ ] `param_scales` handles different units (radians vs rad/s)
- [ ] Uniform sampling within hypersphere (not just per-axis box)
- [ ] Verify samples are actually within specified radius

### US-008: Basin size sweep
**Description:** As a scientist, I want to systematically measure success rate vs constraint radius.

**Acceptance Criteria:**
- [ ] Test radii: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] degrees (converted appropriately)
- [ ] For each radius: run 20 random initializations, record success/failure
- [ ] Track: success rate, mean final error, mean function evaluations
- [ ] Total budget: ~200 local optimizations (feasible in reasonable time)

### US-009: Basin size visualization and threshold
**Description:** As a scientist, I want to identify the critical radius where success rate drops.

**Acceptance Criteria:**
- [ ] Plot success rate vs radius (with error bars from binomial uncertainty)
- [ ] Plot mean error vs radius
- [ ] Identify critical radius (e.g., radius where success rate drops below 80%)
- [ ] Report: "Local optimization succeeds with >80% probability when initialized within X degrees"

---

## Notebook 3: Optimizer Comparison

**File:** `notebooks/inversion/06_optimizer_comparison.py`

**Purpose:** Fair comparison of optimization strategies under fixed evaluation budget.

**Flow:**
1. Setup (copy from 04)
2. Define evaluation budget (e.g., 5000 function evaluations)
3. Run each optimizer strategy with same budget
4. Compare: best solution found, success rate, convergence behavior
5. Statistical summary across multiple trials

### US-010: Evaluation-counted objective wrapper
**Description:** As a scientist, I need to track and limit function evaluations for fair comparison.

**Acceptance Criteria:**
- [ ] Wrapper class `CountedObjective` that counts calls and can enforce a limit
- [ ] Raises exception or returns penalty when budget exhausted
- [ ] Reset method for multiple trials

### US-011: Multi-start local strategy
**Description:** As a scientist, I want to implement the supervisor's suggestion: random starts + local refinement.

**Acceptance Criteria:**
- [ ] Function `multistart_local(objective, bounds, n_starts, max_evals_per_start)`
- [ ] Random initialization using Latin Hypercube Sampling (scipy.stats.qmc.LatinHypercube)
- [ ] Run L-BFGS-B from each start, return best result
- [ ] Track evaluations to stay within budget

### US-012: Differential evolution baseline
**Description:** As a scientist, I need the current DE approach configured to use the same evaluation budget.

**Acceptance Criteria:**
- [ ] Configure DE with `maxiter` set to achieve target evaluation count
- [ ] Use same bounds and settings as current 03 notebook
- [ ] Disable or limit polish step to stay within budget

### US-013: Basin-hopping strategy (optional comparison)
**Description:** As a scientist, I want to test basin-hopping as a middle-ground approach.

**Acceptance Criteria:**
- [ ] Configure scipy.optimize.basinhopping with evaluation budget
- [ ] Use L-BFGS-B as local minimizer
- [ ] Tune stepsize and temperature based on landscape analysis findings

### US-014: Comparison experiment
**Description:** As a scientist, I want statistically meaningful comparison across strategies.

**Acceptance Criteria:**
- [ ] Fixed budget: 5000 function evaluations (adjustable based on 04 findings)
- [ ] Run each strategy 10 times with different random seeds
- [ ] Record: best objective value, parameter error, success (yes/no), wall time
- [ ] Report mean ± std for each metric

### US-015: Comparison visualization and conclusions
**Description:** As a scientist, I want clear visualization of which strategy performs best.

**Acceptance Criteria:**
- [ ] Box plot of final objective values by strategy
- [ ] Box plot of parameter errors by strategy
- [ ] Bar chart of success rates with confidence intervals
- [ ] Table summarizing: Strategy | Success Rate | Mean Error | Mean Time
- [ ] Clear conclusion: "Strategy X achieves Y% success rate, Z× faster than baseline"

---

## Notebook 4: Robustness Analysis

**File:** `notebooks/inversion/07_robustness_analysis.py`

**Purpose:** Test sensitivity to noise level and observation density.

**Flow:**
1. Setup (copy from 04)
2. Define test grid: noise levels × observation counts
3. For each configuration: generate synthetic data, run best optimizer, measure success
4. Create robustness heatmap
5. Identify operational requirements

### US-016: Parameterized synthetic data generation
**Description:** As a scientist, I need to generate test cases with varying noise and observation density.

**Acceptance Criteria:**
- [ ] Function `generate_test_case(n_observations, noise_sigma, seed)` returning lightcurve + geometry
- [ ] Reuse forward model from 03/04
- [ ] Same true parameters across all test cases (for comparability)

### US-017: Robustness sweep
**Description:** As a scientist, I want to map success rate across noise/observation space.

**Acceptance Criteria:**
- [ ] Noise levels: [0.01, 0.02, 0.05, 0.1, 0.2] mag
- [ ] Observation counts: [20, 35, 50, 75, 100]
- [ ] For each combination: run 5 trials with best optimizer from notebook 06
- [ ] Record success rate for each cell

### US-018: Robustness heatmap and requirements
**Description:** As a scientist, I want to identify minimum data quality for reliable inversion.

**Acceptance Criteria:**
- [ ] Heatmap: x=n_observations, y=noise_sigma, color=success_rate
- [ ] Annotate cells with success rate values
- [ ] Draw contour line at 80% success rate
- [ ] Report: "Reliable inversion requires at least N observations with noise < X mag"

### US-019: Final recommendations
**Description:** As a scientist, I want actionable conclusions from the entire study.

**Acceptance Criteria:**
- [ ] Summary markdown cell with key findings from all 4 notebooks
- [ ] Recommended optimizer settings for operational use
- [ ] Minimum data requirements for reliable inversion
- [ ] Known limitations and failure modes

---

## Functional Requirements

- FR-1: All notebooks must use the existing `ObjectiveFunction` class from `src/inversion/`
- FR-2: All notebooks must reuse `invert_lightcurve` components where applicable, not reimplement
- FR-3: Synthetic data generation must match 03 notebook exactly for comparability
- FR-4: Function evaluation counting must be accurate (no hidden evaluations in line searches)
- FR-5: Random seeds must be set for reproducibility
- FR-6: Each notebook must run independently (copy minimal setup, don't rely on notebook 03 state)
- FR-7: Plots must be saved to `data/results/inversion_diagnostics/` with descriptive names
- FR-8: Wall-clock times should be reported for practical guidance

## Non-Goals

- No changes to the core `invert_lightcurve` function in this study (diagnostic only)
- No new optimization algorithms beyond scipy (no custom implementations)
- No GPU acceleration or performance optimization
- No real observational data (synthetic only for ground truth comparison)
- No articulation angle estimation (fixed angles as in 03)
- No MCMC uncertainty analysis (focus is on point estimation)

## Technical Considerations

### Existing Code to Reuse
- `src/inversion/objective.py`: `ObjectiveFunction` class
- `src/inversion/core.py`: `invert_lightcurve` (for reference, not direct use)
- `src/dynamics/propagate.py`: `propagate_attitude`
- `notebooks/inversion/03_lightcurve_inversion.py`: Setup code, forward model

### New Code Required
- 1D/2D slice computation utilities (simple, ~50 lines)
- Evaluation-counted wrapper (simple, ~30 lines)
- Latin Hypercube initialization wrapper (scipy built-in, just configuration)

### Dependencies
- scipy.stats.qmc (for Latin Hypercube) - already available in scipy
- No new package installations required

## Success Metrics

| Metric | Target |
|--------|--------|
| Landscape analysis complete | Clear characterization of objective structure |
| Basin size quantified | Critical radius identified with confidence bounds |
| Optimizer comparison | Statistical significance (p < 0.05) if differences exist |
| Robustness map | Full grid computed, requirements identified |
| Actionable recommendations | Specific settings that achieve >80% success rate |

## Estimated Effort

| Notebook | Estimated Time | Compute Time |
|----------|----------------|--------------|
| 04_landscape_analysis | 2-3 hours | ~30 min (slices) |
| 05_basin_size_study | 1-2 hours | ~1-2 hours (200 optimizations) |
| 06_optimizer_comparison | 2-3 hours | ~2-3 hours (30 trials × 5000 evals) |
| 07_robustness_analysis | 1-2 hours | ~3-4 hours (125 configurations × 5 trials) |

**Total:** 6-10 hours implementation + 6-10 hours compute

## Open Questions

1. Should shadows be enabled for all experiments, or test with/without?
   - *Recommendation: Start without shadows (faster), verify findings hold with shadows for one configuration*

2. What if landscape analysis reveals multiple equivalent minima?
   - *This is a finding, not a failure. Document and adjust success criteria to "finds any minimum with rms < 2×sigma"*

3. Should we test alternative parameterizations (e.g., quaternion vs axis-angle)?
   - *Out of scope for initial study. Note as future work if axis-angle proves problematic*

## Incremental Development Strategy

**After Notebook 04:** If landscape reveals single smooth basin, skip to 06 (optimizer comparison may be moot). If highly multimodal, 05 becomes critical.

**After Notebook 05:** If basin is tiny (<1°), focus 06 on strategies that sample densely. If basin is large (>10°), simple multi-start likely sufficient.

**After Notebook 06:** If one strategy clearly dominates, use only that for 07. If similar, pick fastest.

This adaptive approach avoids wasted effort if early findings resolve the question.
