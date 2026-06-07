# Product Requirements Document: Lightcurve Inversion Tool

## 1. Introduction/Overview

The **Lightcurve Inversion Tool** is a new module for LCAS that solves the inverse problem: given an observed lightcurve and known orbital geometry, determine the initial attitude (orientation) and angular velocity of a spacecraft or debris object.

This tool complements the existing forward model in LCAS, which generates synthetic lightcurves from known initial conditions. The inversion tool uses optimization techniques to find the initial state parameters that produce a lightcurve best matching the observed data.

**Problem Statement**: Lightcurve observations contain information about an object's rotational state, but extracting this information requires solving a nonlinear inverse problem with potential degeneracies (multiple attitude/velocity combinations can produce similar lightcurves).

### Operational Scenarios

The tool supports two primary scenarios based on whether the RSO (Resident Space Object) model is known:

| Scenario | Model Known? | Inertia Tensor | What's Estimated |
|----------|--------------|----------------|------------------|
| **A: Known Model** | Yes (STL + masses provided) | Computed from geometry | Initial attitude (q₀) + angular velocity (ω₀) |
| **B: Unknown Model** | No | Estimated (ratios) | q₀, ω₀, inertia ratios, BRDF parameters |

**Scenario A** is the primary scope of this PRD. **Scenario B** is documented as future work.

### Attitude Definition Methods in LCAS

This PRD introduces a **third method** for defining attitude history, complementing the existing two:

| Method | Description | Use Case |
|--------|-------------|----------|
| **1. SPICE Kernels** | Load attitude from pre-computed CK kernels | When flight/simulated attitude data exists |
| **2. Keyframe Interpolation** | SLERP between user-provided quaternion keyframes | When key attitudes are known at specific times |
| **3. Initial Condition Propagation** *(NEW)* | Propagate from q₀, ω₀, and optionally inertia tensor | For dynamics-based simulation and inversion |

All three methods produce the same output format (quaternion history at specified epochs) and integrate seamlessly with the existing LCAS forward model.

## 2. Goals

1. **Estimate initial rotational state**: Determine the 6 degrees of freedom (initial quaternion attitude + initial angular velocity vector) from lightcurve observations
2. **Provide attitude propagation utility**: Create a standalone tool to propagate attitude from initial conditions, usable independently of inversion
3. **Compute inertia tensor from geometry**: Provide a utility to calculate the inertia tensor from STL models given user-provided component masses
4. **Support multiple dynamics modes**: Handle both principal-axis rotation (stable satellites) and tumbling motion (defunct satellites, debris)
5. **Quantify uncertainty**: Provide confidence bounds on estimated parameters, not just point estimates
6. **Integrate with existing workflow**: Deliver as Jupytext notebooks consistent with existing LCAS notebooks, with the propagator as a third attitude definition option
7. **Leverage existing forward model**: Use the LCAS lightcurve generator as the forward model within the optimization loop

## 3. User Stories

### US-1: Space Analyst Characterizing Unknown Object
> As a space situational awareness analyst, I want to estimate the rotational state of an observed satellite from its lightcurve so that I can characterize its operational status and predict future brightness variations.

### US-2: Researcher Validating Forward Model
> As a researcher, I want to perform round-trip validation (forward → inverse → forward) to verify that the inversion tool correctly recovers known states from synthetic lightcurves.

### US-3: Debris Analyst Studying Tumbling Object
> As a debris analyst, I want to estimate the spin state of tumbling debris using full Euler dynamics, accounting for the fact that angular velocity evolves over time even without external torques.

### US-4: Mission Operator Verifying Attitude
> As a mission operator with ground truth data, I want to compare inversion results against known attitude telemetry to validate the tool's accuracy and understand its limitations.

### US-5: Engineer Computing Satellite Inertia
> As an engineer with a satellite CAD model (STL files), I want to compute the inertia tensor by providing component masses, so I can use accurate dynamics in the inversion.

### US-6: Researcher Simulating Tumbling Dynamics
> As a researcher, I want to generate synthetic lightcurves for a tumbling satellite by specifying initial attitude and angular velocity (instead of keyframes), so I can study how tumbling affects observed brightness variations.

### US-7: Analyst Comparing Attitude Methods
> As an analyst, I want to choose between SPICE kernels, keyframe interpolation, or initial-condition propagation when setting up my simulation, using whichever method best fits my available data.

## 4. Functional Requirements

### 4.1 Inertia Tensor Calculation Utility

| ID | Requirement |
|----|-------------|
| FR-1 | The system shall provide a utility to compute the inertia tensor of a satellite model from its STL components |
| FR-2 | The utility shall accept user-provided mass (in kg) for each unique STL file in the model |
| FR-3 | The utility shall calculate the volume of each STL mesh using the divergence theorem (signed tetrahedra method) |
| FR-4 | The utility shall compute each component's inertia tensor about its center of mass, assuming homogeneous mass distribution |
| FR-5 | The utility shall use the parallel axis theorem to translate each component's inertia to the satellite body frame origin |
| FR-6 | When the same STL file is used for multiple components (e.g., two solar panels), each instance shall contribute separately to the total inertia based on its position in the body frame |
| FR-7 | The utility shall output: total mass, center of mass location, full 3x3 inertia tensor, principal moments of inertia, and principal axes orientation |
| FR-8 | The utility shall validate that STL meshes are closed (watertight) and warn the user if volume calculation may be inaccurate |

### 4.2 Attitude Propagation Utility

This is a **standalone general-purpose utility** that serves as the third method for defining attitude history in LCAS, alongside SPICE kernels and keyframe interpolation.

| ID | Requirement |
|----|-------------|
| FR-9 | The system shall provide an attitude propagation utility that generates quaternion history from initial conditions |
| FR-10 | The utility shall accept: initial attitude (q₀), initial angular velocity (ω₀), array of output times, and optionally an inertia tensor |
| FR-11 | The utility shall support **Principal Axis Rotation** mode: angular velocity ω remains constant in the body frame; attitude propagated via `dq/dt = 0.5 * q ⊗ [0, ω]` |
| FR-12 | The utility shall support **Tumbling (Euler Dynamics)** mode: angular velocity ω evolves in the body frame according to Euler's equations `I·ω̇ = -ω × (I·ω)`; attitude propagated via quaternion kinematics |
| FR-13 | In Tumbling mode, the utility shall require the inertia tensor as input (computed via FR-1 through FR-8 or user-provided) |
| FR-14 | In Tumbling mode, the utility shall integrate the coupled ODE system (Euler equations + quaternion kinematics) using a suitable numerical integrator (e.g., `scipy.integrate.solve_ivp` with RK45 or DOP853) |
| FR-15 | The utility shall output quaternion history in the same format as the existing keyframe interpolator, for seamless integration with the LCAS forward model |
| FR-16 | The utility shall optionally output the angular velocity history ω(t) in addition to the quaternion history |
| FR-17 | The utility shall be usable independently of the inversion tool (i.e., for forward simulation without inversion) |
| FR-18 | The utility shall provide a convenience function that matches the interface of the existing `attitude_keyframes` dict format, allowing drop-in replacement in existing notebooks |

### 4.3 Core Inversion Engine

| ID | Requirement |
|----|-------------|
| FR-19 | The system shall accept an observed lightcurve (time series of magnitudes or fluxes) as input |
| FR-20 | The system shall accept observation geometry information (observer position, sun position, or SPICE configuration to compute these) |
| FR-21 | The system shall estimate 6 parameters: initial attitude quaternion (q₀, 4 components with unit norm constraint = 3 DOF) and initial angular velocity vector (ω₀, 3 components) |
| FR-22 | The system shall use the attitude propagation utility (FR-9 through FR-18) internally to generate attitude histories during optimization |
| FR-23 | The system shall use the existing LCAS forward model (`generate_lightcurves`) to compute predicted lightcurves during optimization |
| FR-24 | The system shall compute a residual metric (chi-squared or mean squared error) between observed and predicted lightcurves |
| FR-25 | The system shall support optional per-observation uncertainty weights; if provided, use weighted chi-squared: `χ² = Σ((obs - pred) / σ)²`; if not provided, assume uniform weights |
| FR-26 | The system shall exclude observations where the satellite is not visible (infinite magnitude) from the residual calculation |

### 4.4 Optimization

| ID | Requirement |
|----|-------------|
| FR-27 | The system shall implement a **global optimization** stage to explore the parameter space and avoid local minima. Recommended algorithm: Particle Swarm Optimization (PSO) or Differential Evolution (DE) |
| FR-28 | The system shall implement a **local refinement** stage to converge to the precise optimum. Recommended algorithm: Levenberg-Marquardt or L-BFGS-B |
| FR-29 | The system shall support **multi-start optimization** (multiple random initializations) to improve robustness against multi-modality |
| FR-30 | The system shall enforce the quaternion unit norm constraint during optimization (either via parameterization or projection) |

### 4.5 Constraint Modes

| ID | Requirement |
|----|-------------|
| FR-31 | The system shall support a **"free exploration" mode** for debris/unknown objects with minimal constraints (only physical bounds on angular velocity magnitude) |
| FR-32 | The system shall support a **"physics-informed" mode** for functional satellites with constraints such as: principal axis rotation, stable spin states (rotation about max or min inertia axis), and bounded angular velocity |
| FR-33 | The user shall be able to select between constraint modes via a configuration parameter |
| FR-34 | The system shall allow user-specified bounds on angular velocity magnitude; default bounds shall be 0 to 30 deg/s |

### 4.6 Uncertainty Quantification

| ID | Requirement |
|----|-------------|
| FR-35 | The system shall provide uncertainty estimates for all estimated parameters |
| FR-36 | The system shall implement **"quick mode"** (default) using Fisher Information Matrix for approximate covariance estimation |
| FR-37 | The system shall implement **"full mode"** using MCMC sampling (via `emcee`) for complete posterior characterization; `pymc` support may be added in future versions |
| FR-38 | The system shall report parameter correlations (covariance matrix or correlation matrix) |
| FR-39 | The system shall optionally identify and report if multiple distinct solutions (modes) exist |

### 4.7 Input/Output

| ID | Requirement |
|----|-------------|
| FR-40 | Input lightcurve format: NumPy arrays or Pandas DataFrame with columns for time (UTC or ET), magnitude/flux, and optionally uncertainty |
| FR-41 | Output shall include: best-fit parameters (q₀, ω₀), goodness-of-fit metric (chi-squared, RMS residual), uncertainty estimates (standard deviations or confidence intervals), and the predicted lightcurve from the best-fit solution |
| FR-42 | The system shall provide visualization of: observed vs. predicted lightcurve comparison, parameter uncertainty distributions (histograms or corner plots), residual plot |
| FR-43 | For Tumbling mode, output shall also include the time evolution of ω(t) over the observation window |

### 4.8 Interface

| ID | Requirement |
|----|-------------|
| FR-44 | The tools shall be delivered as Jupytext-linked Python scripts (percent format) in the `notebooks/` directory, consistent with existing LCAS notebooks |
| FR-45 | The notebooks shall be structured with clear markdown sections guiding the user through each step |
| FR-46 | Core logic shall be implemented as importable modules in `src/` for reuse |
| FR-47 | The attitude propagation utility shall be located in a new `src/dynamics/` module to clearly distinguish dynamics-based propagation from interpolation |

## 5. Non-Goals (Out of Scope)

| ID | Exclusion | Notes |
|----|-----------|-------|
| NG-1 | **Unknown model joint estimation** | Joint estimation of inertia ratios + BRDF + attitude + velocity when model is unknown is **future work** (see Section 11) |
| NG-2 | **Shape estimation** | Object geometry is assumed known (STL model provided); shape inversion is out of scope |
| NG-3 | **Real-time/streaming inversion** | This is a batch processing tool; real-time attitude tracking is out of scope |
| NG-4 | **GUI/dashboard** | Only notebook and Python API interfaces are in scope |
| NG-5 | **Articulation estimation** | Solar panel angles and other articulation parameters are assumed known or fixed |
| NG-6 | **External torques** | Only torque-free motion is modeled; gravity gradient, magnetic, or aerodynamic torques are out of scope |
| NG-7 | **Non-homogeneous density** | Inertia calculator assumes homogeneous mass distribution; shell/hollow models may be added in future versions |

## 6. Design Considerations

### 6.1 Architecture

```
src/
├── computation/
│   └── inertia_calculator.py       # NEW: Inertia tensor from STL + masses
│
├── dynamics/                        # NEW: Dynamics module
│   ├── __init__.py
│   └── attitude_propagator.py      # Propagation from initial conditions
│
├── interpolation/                   # Existing module (unchanged)
│   └── attitude_interpolator.py    # Existing: SLERP keyframe interpolation
│
└── inversion/                       # NEW: Inversion module
    ├── __init__.py
    ├── objective_function.py        # Wraps forward model, computes residuals
    ├── optimizers.py                # PSO, DE, local refinement implementations
    ├── constraints.py               # Constraint modes (free vs physics-informed)
    ├── uncertainty.py               # MCMC (emcee), Fisher matrix methods
    ├── quaternion_utils.py          # Quaternion kinematics, normalization
    └── results.py                   # Result container class with visualization

notebooks/
├── inversion/
│   ├── 01_inertia_calculation.py       # Jupytext: compute inertia from STL
│   ├── 02_attitude_propagation.py      # Jupytext: demo propagator as standalone tool
│   └── 03_lightcurve_inversion.py      # Jupytext: main inversion workflow
│
└── lcas_stl_pipeline.py                # Existing: can now use propagator as 3rd option
```

### 6.2 Attitude Propagation Utility

The propagator provides a third method for generating attitude history, with an interface consistent with existing methods:

```python
# =============================================================================
# THREE WAYS TO DEFINE ATTITUDE HISTORY IN LCAS
# =============================================================================

# Method 1: SPICE Kernels (existing)
# Attitude loaded automatically from CK kernels via SpiceHandler

# Method 2: Keyframe Interpolation (existing)
from src.interpolation.attitude_interpolator import interpolate_attitudes

attitude_keyframes = {
    'times': ['2020-02-05T10:00:00', '2020-02-05T16:00:00'],
    'time_format': 'utc',
    'attitudes': [q_start, q_end],
    'format': 'quaternion'
}
attitudes = interpolate_attitudes(attitude_keyframes, epochs)

# Method 3: Initial Condition Propagation (NEW)
from src.dynamics.attitude_propagator import propagate_attitude

attitudes, omega_history = propagate_attitude(
    q0=initial_quaternion,           # Initial attitude
    omega0=initial_angular_velocity, # Initial angular velocity (rad/s, body frame)
    times=epochs,                    # Output times (ET seconds)
    mode='principal_axis',           # or 'tumbling'
    inertia_tensor=I                 # Required only for 'tumbling' mode
)

# All three methods produce attitudes in the same format,
# ready for use with compute_observation_geometry()
```

**Principal Axis Mode:**
```python
def propagate_principal_axis(q0, omega, times):
    """
    ω constant in body frame.
    Closed-form solution: q(t) = q0 * quaternion_exp(0.5 * ω * Δt)

    Returns: quaternions array of shape (N, 4)
    """
```

**Tumbling (Euler) Mode:**
```python
def propagate_euler(q0, omega0, inertia_tensor, times):
    """
    Coupled ODE system:
    - Euler: I·ω̇ = -ω × (I·ω)
    - Quaternion: dq/dt = 0.5 * q ⊗ [0, ω]

    State vector: [q_w, q_x, q_y, q_z, ω_x, ω_y, ω_z]

    Returns: (quaternions, omega_history) arrays
    """
    # Use scipy.integrate.solve_ivp with RK45 or DOP853
```

### 6.3 Inertia Calculation Algorithm

```python
def compute_inertia_from_stl(stl_components, masses):
    """
    For each STL component:
    1. Compute volume using signed tetrahedra method
    2. Compute density = mass / volume
    3. Compute inertia tensor about component's center of mass
    4. Translate to body frame origin using parallel axis theorem:
       I_body = I_cm + m * (d²·Identity - d⊗d)
    5. Sum all components

    Returns: I_total, principal_moments, principal_axes, center_of_mass
    """
```

### 6.4 Quaternion Handling

- Use scalar-first convention (w, x, y, z) consistent with existing LCAS code
- Enforce unit norm via: (a) 3-parameter axis-angle representation during optimization, or (b) normalization after each update
- Consider using `scipy.spatial.transform.Rotation` or the existing `numpy-quaternion` library

### 6.5 Libraries

| Purpose | Library | Notes |
|---------|---------|-------|
| **Optimization** | `scipy.optimize` | `differential_evolution`, `minimize` (L-BFGS-B) |
| **ODE Integration** | `scipy.integrate` | `solve_ivp` with RK45 or DOP853 |
| **MCMC** | `emcee` | Affine-invariant ensemble sampler; `pymc` support may be added later |
| **Visualization** | `matplotlib`, `corner` | Corner plots for posterior distributions |
| **STL Processing** | `numpy-stl` | Already used in LCAS |

## 7. Technical Considerations

### 7.1 Performance

- Single forward model evaluation: ~100-500ms (25-100 epochs)
- Attitude propagation (principal axis): <1ms (closed-form solution)
- Attitude propagation (tumbling): ~10-50ms (ODE integration)
- Global optimization may require 1,000-10,000 function evaluations
- Expected total inversion time: seconds to minutes depending on configuration
- Consider parallelizing function evaluations if using population-based optimizers

### 7.2 Dependencies

- Existing LCAS modules: `generate_lightcurves`, `compute_observation_geometry`, `compute_shadows`, `STLLoader`, `attitude_interpolator`
- New dependencies (to be added to requirements): `emcee`, `corner`
- Existing dependencies to leverage: `scipy`, `numpy-stl`, `numpy-quaternion`

### 7.3 Integration Points

- Attitude propagator must produce output in same format as existing `interpolate_attitudes()` function
- Must work with existing SPICE handler and configuration system
- Must accept attitude in same quaternion format as forward model
- Inertia calculator must work with existing STL loading infrastructure (`STLLoader`)
- Should reuse existing satellite configuration (component positions in body frame)

### 7.4 Known Challenges

1. **Degeneracy**: Multiple attitude solutions may produce similar lightcurves, especially with limited observation geometry
2. **Local minima**: Nonlinear objective function has many local minima; global optimization essential
3. **Computational cost**: MCMC for full posterior requires many forward evaluations; quick mode (Fisher matrix) provided as default
4. **Tumbling complexity**: Euler dynamics is more sensitive to initial conditions; may require finer optimization resolution
5. **STL mesh quality**: Inertia calculation requires watertight meshes; need validation and user warnings
6. **Quaternion discontinuities**: Long propagation times may benefit from quaternion renormalization at intervals

## 8. Success Metrics

| ID | Metric | Target |
|----|--------|--------|
| SM-1 | **Recovery accuracy on synthetic data (principal axis)**: Given a synthetic lightcurve from known initial conditions, recover attitude within 5 degrees and angular velocity within 10% |
| SM-2 | **Recovery accuracy on synthetic data (tumbling)**: Given a synthetic lightcurve with tumbling dynamics, recover initial attitude within 10 degrees and initial angular velocity within 15% |
| SM-3 | **Uncertainty calibration**: Reported 1-sigma uncertainties should contain the true value ~68% of the time on synthetic tests |
| SM-4 | **Convergence rate**: Global optimization should converge to within 10% of optimal solution in >90% of multi-start runs |
| SM-5 | **Runtime**: Complete inversion (global + local + uncertainty) should complete in <5 minutes for typical cases (50-100 epoch lightcurve) |
| SM-6 | **Inertia accuracy**: Computed inertia tensor should match analytical results for simple shapes (sphere, cylinder, box) within 1% |
| SM-7 | **Propagator accuracy**: Propagated attitudes should match analytical solutions (where available) to within numerical precision (~1e-10) |
| SM-8 | **API consistency**: Propagator output should be directly usable by existing `compute_observation_geometry()` without modification |

## 9. Design Decisions

The following decisions were made during PRD development:

| ID | Decision | Rationale |
|----|----------|-----------|
| DD-1 | **MCMC library**: Start with `emcee`, add `pymc` support later if needed | `emcee` is lightweight, well-documented, and standard in astronomy/lightcurve inversion literature |
| DD-2 | **Weighted observations**: Support optional per-observation uncertainties, default to uniform weights | Weighted chi-squared is standard practice; keeping it optional maintains simplicity for basic use |
| DD-3 | **Non-visible observations**: Exclude from fit | Simplest approach; modeling non-detections correctly is complex and rarely needed |
| DD-4 | **Quick mode as default**: Fisher Information Matrix for fast uncertainty estimates, full MCMC optional | Users need fast iteration during exploration; full MCMC for final publication-quality results |
| DD-5 | **Angular velocity bounds**: Default 0-30 deg/s, user-configurable | Covers most debris and satellite cases; allows override for special cases |
| DD-6 | **Inertia calculator**: Homogeneous density only for v1.0 | Keeps implementation simple; shell/hollow models can be added later |
| DD-7 | **Propagator location**: New `src/dynamics/` module | Semantic clarity: propagation is dynamics, not interpolation |

## 10. References

### Key Papers

1. **PSO for Quaternion Optimization**: "Particle Swarm Optimization on the Space of Quaternions" (2025) - Multiplicative PSO operating directly on quaternion space
2. **Bayesian Lightcurve Inversion**: Muinonen et al. (2020) "Asteroid lightcurve inversion with Bayesian inference" - Virtual-observation MCMC method
3. **Global Optimization**: Chng et al. (2022) "Globally optimal shape and spin pole determination with light-curve inversion" - Branch-and-Bound approach
4. **Classic Method**: Kaasalainen & Torppa (2001) "Optimization Methods for Asteroid Lightcurve Inversion I & II" - Foundational KTM method

### Existing Tools

- **DAMIT**: Database and software for asteroid model inversion (convexinv, conjgradinv)
- **LCAS forward model**: Existing pipeline in this repository

### Inertia Calculation References

- Mirtich (1996) "Fast and Accurate Computation of Polyhedral Mass Properties" - Standard algorithm for inertia from triangulated meshes

## 11. Future Work: Unknown Model Scenario

When the RSO model is unknown, a more complex joint estimation problem arises. This section documents the approach for future implementation.

### 11.1 Parameters to Estimate

| Parameter | DOF | Notes |
|-----------|-----|-------|
| Initial attitude (q₀) | 3 | Quaternion with unit norm |
| Initial angular velocity (ω₀) | 3 | Body frame |
| Inertia tensor ratios | 2 | I₂/I₁, I₃/I₁ (only ratios are observable) |
| BRDF parameters | 2-3 | r_d, r_s, (n_phong) per material class |
| **Total** | ~10-12 | High-dimensional optimization |

### 11.2 Approach

1. **Simplified shape model**: Assume triaxial ellipsoid or convex hull approximation
2. **Regularization**: Strong priors on physically reasonable parameter ranges
3. **Multi-stage optimization**: First estimate BRDF from overall brightness, then estimate dynamics
4. **Data requirements**: Will likely require multiple observation geometries (different phase angles) to break degeneracies

### 11.3 Key Challenges

- Much larger parameter space increases optimization difficulty
- Strong degeneracies between BRDF and shape
- May require complementary data (radar, occultation) for reliable solutions

---

*Document generated: 2026-01-21*
*Feature: lightcurve-inversion*
*Status: Final*
