# PRD: Mixed-Fidelity Hierarchical Inversion for Lightcurve-Based Attitude Estimation

## Introduction

This study extends the inversion diagnostic series (notebooks 04-07) with a **mixed-fidelity optimization strategy** designed to dramatically reduce the wall-clock cost of lightcurve inversion while preserving solution accuracy.

The core insight: the LCAS forward model's ray-traced shadow computation (`compute_shadows`) dominates evaluation cost (~5-10s per call), but shadows have a relatively small effect on the objective landscape's global structure. By performing global search with shadows disabled (fast, approximate) and only enabling shadows for local refinement (slow, exact), we can potentially achieve the solution quality of full-fidelity Differential Evolution at a fraction of the cost.

**Deliverable:** `notebooks/inversion/08_mixed_fidelity_inversion.py` (Jupytext percent format)

**Predecessor findings that motivate this work:**
- Notebook 04: Condition number is 9.6e9 — severely ill-conditioned problem requiring robust global search
- Notebook 05: Basin of attraction is finite but well-defined near the true solution
- Notebook 06: DE achieves highest success rate but requires ~5000 evaluations at ~5-10s each
- Notebook 07: Noise/observation trade-offs are well characterized

## Goals

- Quantify the speedup factor of shadow-free forward model evaluation vs full-fidelity evaluation
- Determine whether the shadow-free objective landscape preserves basin locations (i.e., does the global minimum shift when shadows are disabled?)
- Implement and validate a two-stage mixed-fidelity pipeline: low-fidelity global search → high-fidelity local refinement
- Compare mixed-fidelity against the three notebook 06 baselines (Multi-start, DE, Basin-hopping) under both fixed wall-clock and fixed evaluation-count budgets
- Identify failure regimes where the low-fidelity approximation breaks down (specific phase angle ranges, observation geometries)
- Provide a clear recommendation on whether mixed-fidelity is viable for the research paper

## Study Design

The notebook is organized into seven sequential experiments, each building on the previous:

```
Experiment 1: Fidelity Benchmarking
    → Quantify cost ratio and lightcurve discrepancy between fidelity levels
         ↓
Experiment 2: Basin Shift Analysis
    → Determine whether shadow approximation moves the global minimum
         ↓
Experiment 3: Mixed-Fidelity Pipeline Implementation
    → Build the two-stage pipeline with configurable handoff
         ↓
Experiment 4: Baseline Comparison (Evaluation-Count Matched)
    → Fair comparison at same number of forward model calls
         ↓
Experiment 5: Baseline Comparison (Wall-Clock Matched)
    → Fair comparison at same total computation time
         ↓
Experiment 6: Handoff Parameter Ablation
    → Sweep top-N candidates passed from Stage 1 → Stage 2
         ↓
Experiment 7: Failure Regime Identification
    → Phase angle sweep to find where low-fidelity approximation fails
```

## Success Criteria

An inversion is considered **successful** if BOTH (same as notebook 06):
- Angular velocity error < 0.1 deg/s
- RMS residual < 2× noise sigma

The mixed-fidelity approach is considered **viable** if:
- Achieves ≥ 3× wall-clock speedup over full-fidelity DE
- Success rate within 5 percentage points of full-fidelity DE
- Parameter accuracy (omega error) within 5% of full-fidelity DE median

---

## Experiment 1: Fidelity Benchmarking

**Purpose:** Establish the quantitative cost difference between shadow-enabled and shadow-disabled forward model evaluation, and measure how much the predicted lightcurves diverge.

**Setup:**
- Reuse the Intelsat 901 test case from notebooks 04-07 (50 observations, 0.05 mag noise, tumbling mode)
- Same `true_params`, geometry data, articulation matrices, and inertia tensor

**Procedure:**
1. Create two `ObjectiveFunction` instances from the same observation data:
   - `obj_hifi`: `compute_shadows_flag=True` (high-fidelity)
   - `obj_lofi`: `compute_shadows_flag=False` (low-fidelity)
2. Time N=20 evaluations of each at the true parameters using `time.perf_counter()`, reporting mean and std
3. Compute the speedup factor: `t_hifi / t_lofi`
4. Generate predicted lightcurves at true parameters for both fidelity levels
5. Compute and plot the magnitude residual between fidelity levels: `mag_hifi - mag_lofi`
6. Evaluate both objectives on a grid of 100 random parameter samples; scatter-plot `f_lofi` vs `f_hifi` to assess rank correlation (Spearman's ρ)

**Key Outputs:**
- Print: mean evaluation time for each fidelity level, speedup factor
- Print: Spearman rank correlation between fidelity levels
- Figure: `mixed_fidelity_benchmark.png` — 3-panel figure:
  - (a) Lightcurve overlay at true params (both fidelity levels + observed)
  - (b) Magnitude residual (hifi - lofi) vs time
  - (c) Scatter plot of objective values (lofi vs hifi) with correlation annotation

**Expected range:** 5-20× speedup factor. Spearman ρ > 0.9 would indicate the low-fidelity landscape is a good proxy.

---

## Experiment 2: Basin Shift Analysis

**Purpose:** Determine whether the global minimum location shifts when shadows are disabled. This is the critical feasibility question — if the basin moves significantly, the two-stage approach is fundamentally flawed.

**Procedure:**
1. Run L-BFGS-B from the true parameters on both `obj_hifi` and `obj_lofi`
2. Compare the converged solutions: compute parameter-space Euclidean distance between the two minima, and separate this into axis-angle displacement (degrees) and omega displacement (deg/s)
3. Run L-BFGS-B from 20 perturbed starting points (within known basin radius from notebook 05) on both fidelity levels
4. For each start: record converged solution for both fidelity levels, compute pairwise displacement
5. Report displacement statistics: mean, std, max across the 20 starts
6. Generate 2D contour overlay: pick the two most sensitive parameter dimensions (from notebook 04's condition number analysis), plot objective landscape contours for both fidelity levels overlaid, mark minima locations

**Key Outputs:**
- Print: basin center displacement statistics (mean ± std for axis-angle and omega components)
- Print: qualitative assessment — "negligible shift" (< 1% of basin width), "moderate shift" (1-10%), or "significant shift" (> 10%)
- Figure: `basin_shift_contours.png` — 2D contour overlay with both minima marked
- Figure: `basin_shift_scatter.png` — scatter plot of converged hifi vs lofi parameter values across 20 starts

---

## Experiment 3: Mixed-Fidelity Pipeline Implementation

**Purpose:** Implement the core two-stage pipeline and validate it on the standard test case.

**Pipeline Design:**

```
Stage 1: Global Search (Low-Fidelity)
├── Create ObjectiveFunction with compute_shadows_flag=False
├── Run Differential Evolution (same hyperparameters as notebook 06)
│   - Strategy: 'best1bin', mutation: (0.5, 1.0), recombination: 0.7
│   - Population size: 15
│   - Budget: configurable (default: 5000 evals)
├── Extract top N candidates from DE's final population
│   - Sort by objective value, take top N (default N=3)
│   - N is a configurable parameter
└── Output: list of N candidate parameter vectors + their lofi objective values

Stage 2: Local Refinement (High-Fidelity)
├── Create ObjectiveFunction with compute_shadows_flag=True
├── For each of the N candidates:
│   ├── Run L-BFGS-B local optimization
│   │   - maxiter: 200, maxfun: 500
│   │   - Same parameter bounds as notebook 06
│   └── Record converged solution and hifi objective value
├── Select the overall best solution (lowest hifi objective)
└── Output: best parameters, best objective value, total evaluations (lofi + hifi)
```

**Implementation Notes:**
- The pipeline should be a standalone function `run_mixed_fidelity()` that returns a results dict compatible with notebook 06's `run_trial()` format (keys: `x_best`, `f_best`, `n_evals`, plus `n_evals_lofi`, `n_evals_hifi`, `stage1_time`, `stage2_time`)
- Budget tracking: count lofi and hifi evaluations separately; report both and combined
- Access DE's final population via scipy's `differential_evolution` return value (`.population` and `.population_energies` attributes when using `polish=False`)
- Use `polish=False` in DE to prevent automatic L-BFGS-B polishing (we do our own in Stage 2)

**Validation:**
1. Run the pipeline once on the standard test case with N=3
2. Print: Stage 1 time, Stage 2 time, total time, number of candidates refined
3. Print: final parameter error (omega error in deg/s) and RMS residual
4. Compare against true parameters — verify the result meets success criteria

**Key Outputs:**
- Print: detailed timing breakdown (stage 1 time, stage 2 time per candidate, total)
- Print: evaluation counts (lofi evals, hifi evals, total)
- Print: success/fail with parameter errors

---

## Experiment 4: Baseline Comparison — Evaluation-Count Matched

**Purpose:** Compare mixed-fidelity against notebook 06 baselines where all strategies get the same total number of forward model evaluations (5000). This isolates algorithmic benefit from cost savings.

**Procedure:**
1. Run mixed-fidelity with total budget = 5000 evals (split: e.g., 4500 lofi in Stage 1 + up to 500 hifi across N candidates in Stage 2)
2. Run the three notebook 06 strategies (Multi-start, DE, Basin-hopping) with 5000 eval budget and `compute_shadows_flag=True` — reuse the notebook 06 implementation directly
3. Each strategy: 10 trials, seeds = `BASE_SEED + trial * 100` (same as notebook 06)
4. Record per-trial: strategy name, seed, best_objective, omega_error, rms_residual, success, n_evals, wall_time

**Budget Allocation for Mixed-Fidelity:**
- Stage 1 (lofi DE): `total_budget - hifi_budget` evaluations
- Stage 2 (hifi L-BFGS-B): `hifi_budget` evaluations split across N candidates
- Default: `hifi_budget = N * 200` (200 evals per candidate for L-BFGS-B)

**Key Outputs:**
- Print: comparison table (strategy, success rate, mean objective, mean omega error, mean wall time)
- Figure: `evalcount_comparison_objectives.png` — box plots of final objective values per strategy (4 strategies: Multi-start, DE, Basin-Hopping, Mixed-Fidelity)
- Figure: `evalcount_comparison_success.png` — bar chart of success rates with 95% CI (Wilson interval)

---

## Experiment 5: Baseline Comparison — Wall-Clock Matched

**Purpose:** Compare mixed-fidelity against full-fidelity DE under the same wall-clock budget. This is the primary practical comparison — it answers "what can I achieve in the same amount of time?"

**Procedure:**
1. Run full-fidelity DE (10 trials) to establish mean wall-clock time per trial → `T_ref`
2. Run mixed-fidelity with a lofi evaluation budget chosen so that total wall time ≈ `T_ref`:
   - Estimate: `lofi_budget ≈ T_ref / t_lofi` (from Experiment 1 timing)
   - Keep hifi budget fixed at `N * 200`
3. 10 trials per strategy, same seeds

**Key Outputs:**
- Print: comparison table with wall-clock column showing near-equal times
- Print: "effective evaluations" — how many lofi evals fit in the time budget vs hifi evals
- Figure: `wallclock_comparison.png` — grouped bar chart showing success rate and mean omega error side by side for Full-Fidelity DE vs Mixed-Fidelity, with wall time annotations
- Figure: `wallclock_pareto.png` — scatter plot of (wall_time, omega_error) for all trials across both strategies, showing Pareto front

---

## Experiment 6: Handoff Parameter Ablation

**Purpose:** Determine the optimal number of candidates (N) to pass from Stage 1 to Stage 2.

**Procedure:**
1. Sweep N ∈ {1, 3, 5, 10}
2. For each N: run 10 trials of mixed-fidelity with fixed lofi budget (5000 evals)
3. hifi budget scales with N: `N * 200` evals
4. Record success rate, mean omega error, mean total time, mean Stage 2 time

**Trade-off:** Larger N → more robust (less dependent on lofi ranking accuracy) but more expensive Stage 2. Smaller N → cheaper but risky if lofi ranking is wrong.

**Key Outputs:**
- Print: table of N vs (success rate, mean omega error, mean total time, stage2 time fraction)
- Figure: `handoff_ablation.png` — 2-panel figure:
  - (a) Success rate vs N (with 95% CI error bars)
  - (b) Mean total wall time vs N (with breakdown: Stage 1 time + Stage 2 time stacked)

---

## Experiment 7: Failure Regime Identification

**Purpose:** Identify the observation geometries where the low-fidelity approximation fails, specifically by sweeping solar phase angle. At certain phase angles, shadows have a dominant effect on the lightcurve and cannot be neglected.

**Background:** Solar phase angle (angle between sun-satellite-observer) determines how much of the illuminated satellite is visible. At small phase angles (near opposition), shadows are minimal. At large phase angles (near quadrature/crescent), self-shadowing becomes a primary lightcurve feature.

**Procedure:**
1. Generate synthetic test cases at phase angles spanning [10°, 170°] in 10° steps (17 test cases)
   - Modify the observation epoch/geometry to achieve each target phase angle
   - Keep all other parameters fixed (same satellite, attitude, noise)
2. For each phase angle:
   - Compute the lightcurve discrepancy metric: `RMS(mag_hifi - mag_lofi)` at true parameters
   - Run mixed-fidelity pipeline (5 trials) and full-fidelity DE (5 trials)
   - Record success rate and omega error for both
3. Identify the crossover phase angle where mixed-fidelity success rate drops below full-fidelity DE

**Key Outputs:**
- Print: table of phase angle vs (lightcurve discrepancy, mixed-fidelity success rate, full-fidelity success rate)
- Figure: `phase_angle_failure.png` — 3-panel figure:
  - (a) Lightcurve fidelity discrepancy (RMS mag difference) vs phase angle
  - (b) Success rate vs phase angle for both strategies (with error bars)
  - (c) Mean omega error vs phase angle for both strategies
- Print: recommended phase angle range where mixed-fidelity is viable

---

## Functional Requirements

- FR-1: The notebook must reuse the identical Intelsat 901 test case setup from notebooks 04-07 (same config, geometry, articulation angles, inertia tensor, noise level, true parameters)
- FR-2: Two `ObjectiveFunction` instances must be created from the same data, differing only in `compute_shadows_flag`
- FR-3: Timing measurements must use `time.perf_counter()` (not `time.time()`) for sub-second accuracy
- FR-4: The mixed-fidelity pipeline function must return a results dict with keys compatible with notebook 06's trial format, plus additional keys: `n_evals_lofi`, `n_evals_hifi`, `stage1_time`, `stage2_time`
- FR-5: DE in Stage 1 must use `polish=False` and access the `.population` / `.population_energies` attributes to extract top N candidates
- FR-6: All comparison experiments must use the same random seeds as notebook 06 (`BASE_SEED + trial * 100`) for reproducibility
- FR-7: Success evaluation must use the same `evaluate_success()` function and thresholds as notebook 06
- FR-8: All figures must be saved to `data/results/inversion_diagnostics/` at 150 DPI with `bbox_inches='tight'`
- FR-9: A text summary (`mixed_fidelity_summary.txt`) must be saved with all key numerical results
- FR-10: The notebook must follow Jupytext percent format with the same header as notebooks 04-07
- FR-11: The wall-clock comparison must normalize for actual measured times (not assumed), re-measuring baseline times if needed
- FR-12: The handoff ablation must keep the Stage 1 lofi budget fixed while varying only N and the resulting Stage 2 hifi budget
- FR-13: Each experiment section must begin with a markdown cell stating purpose, and end with a markdown cell summarizing findings

## Non-Goals (Out of Scope)

- **Multi-fidelity surrogate models** (e.g., Gaussian process with multi-fidelity kernel) — this study tests the simplest two-stage approach first
- **Adaptive fidelity switching** during optimization (e.g., progressively enabling shadows as DE converges) — future work
- **Parallelized evaluation** — all evaluations are sequential for fair timing comparisons
- **Different satellite models** — Intelsat 901 only, for comparability with the existing diagnostic series
- **BRDF parameter estimation** — only attitude/angular velocity parameters are optimized
- **Shadow model improvements** (e.g., penumbra, partial shadowing) — use existing binary shadow engine as-is
- **Automated phase angle control** — Experiment 7 may require manual selection of epochs that achieve target phase angles; building a general-purpose phase angle scheduler is out of scope

## Technical Considerations

### Existing Infrastructure
- `ObjectiveFunction` (in `src/inversion/objective_function.py`) already has the `compute_shadows_flag` parameter — no source modifications needed
- `create_no_shadow_lit_status()` (in `src/computation/shadow_engine.py`) provides the fast shadow-free path
- `CountedObjective` wrapper from notebook 06 can be reused for budget enforcement
- `evaluate_success()` from notebook 06 can be reused for success metrics
- There is a skeleton `invert_lightcurve_multifidelity()` in `src/inversion/results.py` that may be referenced but is not required — the notebook should implement its own pipeline function for transparency

### scipy.optimize.differential_evolution Population Access
- Must pass `polish=False` to prevent automatic L-BFGS-B polishing after DE
- The result object has `.population` (shape: `(popsize, ndim)`) and `.population_energies` (shape: `(popsize,)`) when using scipy ≥ 1.7
- Sort by `.population_energies` to extract top N candidates

### Phase Angle Control (Experiment 7)
- Phase angle = angle between (sun→satellite) and (observer→satellite) vectors
- Can be computed from SPICE geometry: `phase = arccos(dot(sun_vec, obs_vec) / (|sun_vec| * |obs_vec|))`
- To achieve target phase angles, either: (a) search through the existing epoch range for epochs near target angles, or (b) rotate the observer position analytically
- Approach (a) is preferred for physical realism

### Runtime Estimates
- Experiment 1 (benchmark): ~5-10 min (40 evaluations total)
- Experiment 2 (basin shift): ~30-60 min (40 hifi evaluations + 40 lofi)
- Experiment 3 (pipeline validation): ~5-10 min (single run)
- Experiment 4 (eval-count comparison): ~3-6 hours (30 trials × ~5000 hifi evals + 10 mixed trials)
- Experiment 5 (wall-clock comparison): ~1-2 hours (matched time budget)
- Experiment 6 (handoff ablation): ~2-4 hours (40 mixed trials across 4 N values)
- Experiment 7 (phase angle sweep): ~4-8 hours (170 trials across 17 angles)
- **Total estimated runtime: 10-30 hours** (run overnight or across multiple sessions)

### Output Files
All saved to `data/results/inversion_diagnostics/`:
- `mixed_fidelity_benchmark.png`
- `basin_shift_contours.png`
- `basin_shift_scatter.png`
- `evalcount_comparison_objectives.png`
- `evalcount_comparison_success.png`
- `wallclock_comparison.png`
- `wallclock_pareto.png`
- `handoff_ablation.png`
- `phase_angle_failure.png`
- `mixed_fidelity_summary.txt`

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Wall-clock speedup | ≥ 3× (stretch: 10×) | Mixed-fidelity total time vs full-fidelity DE time at same success rate |
| Success rate preservation | Within 5 percentage points of full-fidelity DE | 10-trial comparison under wall-clock-matched budget |
| Parameter accuracy | Within 5% of full-fidelity DE median omega error | Median omega error across 10 trials |
| Basin shift | < 10% of basin width | Displacement metric from Experiment 2 |
| Rank correlation | Spearman ρ > 0.85 | Lofi vs hifi objective values from Experiment 1 |
| Viable phase angle range | Identified and documented | Phase angle range where mixed-fidelity success ≥ 80% of full-fidelity |

## Open Questions

1. **Population access in scipy DE:** Does the installed scipy version (need to verify) expose `.population` and `.population_energies` on the DE result? If not, an alternative extraction method is needed (e.g., callback-based tracking).
2. **Phase angle realism:** Can we find epochs within the existing SPICE kernel time range that span [10°, 170°] phase angles? If not, Experiment 7 may need synthetic geometry or a reduced angle range.
3. **Budget allocation optimization:** The split between Stage 1 (lofi) and Stage 2 (hifi) budgets is currently heuristic. Should we add an experiment to sweep the split ratio, or is the handoff N ablation sufficient?
4. **Reproducibility of notebook 06 baselines:** Should we re-run notebook 06 baselines within this notebook (for identical hardware/timing conditions), or import saved results? Re-running is more rigorous but adds hours of compute.
5. **Statistical significance:** With 10 trials, can we detect a 5 percentage-point difference in success rate with confidence? A power analysis may indicate whether more trials are needed for certain experiments.
