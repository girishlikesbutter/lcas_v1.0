"""
Result container classes for lightcurve inversion.

This module provides the InversionResult dataclass for storing all
outputs from the inversion pipeline, including:
- Estimated initial attitude and angular velocity
- Fit quality metrics
- Uncertainty estimates
- Predicted vs observed lightcurve data for plotting
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING
import time
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..io.stl_loader import Satellite

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False


@dataclass
class InversionResult:
    """
    Container for all lightcurve inversion outputs.

    Attributes
    ----------
    q0 : ndarray, shape (4,)
        Estimated initial quaternion (w, x, y, z).
    omega0 : ndarray, shape (3,)
        Estimated initial angular velocity (rad/s).
    chi_squared : float
        Chi-squared statistic of the fit.
    rms_residual : float
        Root mean square residual in magnitudes.
    uncertainties : dict, optional
        Parameter uncertainties (from Fisher or MCMC).
    predicted_lightcurve : ndarray
        Predicted magnitudes from best-fit parameters.
    observed_lightcurve : ndarray
        Observed magnitudes for comparison.
    observation_times : ndarray
        Times of observations.
    omega_history : ndarray, optional
        Angular velocity evolution for tumbling mode.
    mcmc_samples : ndarray, optional
        MCMC posterior samples if full uncertainty was computed.
    """

    q0: NDArray[np.floating]
    omega0: NDArray[np.floating]
    chi_squared: float
    rms_residual: float
    predicted_lightcurve: NDArray[np.floating]
    observed_lightcurve: NDArray[np.floating]
    observation_times: NDArray[np.floating]
    uncertainties: Optional[Dict[str, NDArray[np.floating]]] = None
    omega_history: Optional[NDArray[np.floating]] = None
    mcmc_samples: Optional[NDArray[np.floating]] = None

    def plot_lightcurve_comparison(self, ax: Optional[Axes] = None) -> Axes:
        """
        Plot observed vs predicted lightcurve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Create mask for valid observations (finite magnitude)
        valid_obs = np.isfinite(self.observed_lightcurve)
        valid_pred = np.isfinite(self.predicted_lightcurve)
        valid_mask = valid_obs & valid_pred

        times = self.observation_times[valid_mask]
        obs = self.observed_lightcurve[valid_mask]
        pred = self.predicted_lightcurve[valid_mask]

        # Plot observed data with markers
        ax.scatter(times, obs, c='blue', s=20, alpha=0.7, label='Observed', zorder=2)

        # Plot predicted lightcurve as line
        ax.plot(times, pred, 'r-', linewidth=1.5, label='Predicted', zorder=1)

        ax.set_xlabel('Time')
        ax.set_ylabel('Magnitude')
        ax.set_title('Observed vs Predicted Lightcurve')
        ax.legend()

        # Invert y-axis (magnitudes are brighter when smaller)
        ax.invert_yaxis()

        ax.grid(True, alpha=0.3)

        return ax

    def plot_residuals(self, ax: Optional[Axes] = None) -> Axes:
        """
        Plot residuals (observed - predicted).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        # Create mask for valid observations (finite magnitude)
        valid_obs = np.isfinite(self.observed_lightcurve)
        valid_pred = np.isfinite(self.predicted_lightcurve)
        valid_mask = valid_obs & valid_pred

        times = self.observation_times[valid_mask]
        residuals = self.observed_lightcurve[valid_mask] - self.predicted_lightcurve[valid_mask]

        # Plot residuals
        ax.scatter(times, residuals, c='blue', s=20, alpha=0.7)

        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # Add +/- RMS lines if we have valid residuals
        if len(residuals) > 0:
            rms = self.rms_residual
            ax.axhline(y=rms, color='orange', linestyle=':', linewidth=1, label=f'+RMS ({rms:.3f})')
            ax.axhline(y=-rms, color='orange', linestyle=':', linewidth=1, label=f'-RMS ({rms:.3f})')

        ax.set_xlabel('Time')
        ax.set_ylabel('Residual (mag)')
        ax.set_title(f'Residuals (Observed - Predicted), RMS = {self.rms_residual:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_corner(self) -> Figure:
        """
        Plot corner plot of posterior samples.

        Requires MCMC samples to have been computed.

        Returns
        -------
        matplotlib.figure.Figure
            The corner plot figure.

        Raises
        ------
        ValueError
            If MCMC samples are not available.
        ImportError
            If the corner package is not installed.
        """
        if self.mcmc_samples is None:
            raise ValueError(
                "MCMC samples not available. Run inversion with uncertainty_mode='full' "
                "to generate posterior samples for corner plot."
            )

        if not CORNER_AVAILABLE:
            raise ImportError(
                "The 'corner' package is required for corner plots. "
                "Install it with: pip install corner"
            )

        # Parameter labels for the 6-parameter model
        labels = [
            r"$\phi_1$", r"$\phi_2$", r"$\phi_3$",  # axis-angle components
            r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"  # angular velocity components
        ]

        # Create corner plot
        fig = corner.corner(
            self.mcmc_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],  # 1-sigma intervals
            show_titles=True,
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 12},
        )

        fig.suptitle("Posterior Distribution of Inversion Parameters", y=1.02, fontsize=14)

        return fig


def invert_lightcurve(
    observed_lightcurve: NDArray[np.floating],
    satellite: Any,
    observation_times: NDArray[np.floating],
    sun_positions_j2000: NDArray[np.floating],
    observer_positions_j2000: NDArray[np.floating],
    satellite_positions_j2000: NDArray[np.floating],
    observer_distances: NDArray[np.floating],
    mode: Literal["principal_axis", "tumbling"] = "principal_axis",
    uncertainty_mode: Literal["quick", "full"] = "quick",
    inertia_tensor: Optional[NDArray[np.floating]] = None,
    measurement_uncertainties: Optional[NDArray[np.floating]] = None,
    compute_shadows: bool = True,
    n_starts: int = 3,
    constraint_mode: Optional[Any] = None,
    omega_max_deg_per_s: float = 30.0,
    seed: Optional[int] = None,
    mcmc_n_samples: int = 1000,
    mcmc_burn_in: int = 100,
    articulation_matrices: Optional[Dict[str, NDArray[np.floating]]] = None,
    workers: int = 1,
) -> InversionResult:
    """
    Run the full lightcurve inversion pipeline.

    This is the main API function that runs global optimization, local
    refinement, and uncertainty estimation to recover initial attitude
    and angular velocity from an observed lightcurve.

    Parameters
    ----------
    observed_lightcurve : ndarray
        Array of observed magnitudes.
    satellite : Satellite
        Satellite model with STL components and BRDF properties.
    observation_times : ndarray
        Array of observation times (seconds from reference epoch).
    sun_positions_j2000 : ndarray, shape (N, 3)
        Sun positions in J2000 frame in km.
    observer_positions_j2000 : ndarray, shape (N, 3)
        Observer positions in J2000 frame in km.
    satellite_positions_j2000 : ndarray, shape (N, 3)
        Satellite positions in J2000 frame in km.
    observer_distances : ndarray, shape (N,)
        Observer-to-satellite distances in km.
    mode : {'principal_axis', 'tumbling'}, optional
        Attitude propagation mode. Default is 'principal_axis'.
    uncertainty_mode : {'quick', 'full'}, optional
        Uncertainty estimation mode:
        - 'quick': Fisher Information approximation (fast)
        - 'full': Full MCMC posterior sampling (thorough)
        Default is 'quick'.
    inertia_tensor : ndarray, shape (3, 3), optional
        Inertia tensor in body frame (kg*m^2). Required for tumbling mode.
    measurement_uncertainties : ndarray, optional
        Measurement uncertainties for weighted chi-squared fitting.
    compute_shadows : bool, optional
        Whether to compute shadows via ray tracing. Default is True.
        Set to False for faster (but less accurate) estimation.
    n_starts : int, optional
        Number of multi-start optimizations. Default is 3.
    constraint_mode : ConstraintMode, optional
        Constraint mode for bounds. If None, uses default bounds.
    omega_max_deg_per_s : float, optional
        Maximum angular velocity in degrees per second. Default is 30.
    seed : int, optional
        Random seed for reproducibility.
    mcmc_n_samples : int, optional
        Number of MCMC samples for full uncertainty mode. Default is 1000.
    mcmc_burn_in : int, optional
        Number of MCMC burn-in steps. Default is 100.
    articulation_matrices : dict, optional
        Pre-computed rotation matrices for articulated components.
        Dict mapping component names to (N, 4, 4) transformation matrices.
        Use this for fixed articulation angles (e.g., solar panels at 0°,
        antenna dishes at 15°). Generate with:
        ``compute_rotation_matrices_from_angles(angles_dict, satellite)``
    workers : int, optional
        Number of parallel workers for optimization. Default is 1 (serial).
        Set to -1 to use all available CPU cores. Can significantly speed
        up optimization on multi-core systems.

    Returns
    -------
    InversionResult
        Complete inversion results including:
        - q0: Estimated initial quaternion
        - omega0: Estimated initial angular velocity
        - chi_squared: Fit quality metric
        - rms_residual: RMS of residuals
        - uncertainties: Parameter uncertainties
        - predicted_lightcurve: Model prediction at best-fit
        - observed_lightcurve: Input observed data
        - observation_times: Input times
        - omega_history: Angular velocity evolution (tumbling mode)
        - mcmc_samples: MCMC samples (if uncertainty_mode='full')

    Raises
    ------
    ValueError
        If tumbling mode is selected without providing inertia_tensor.

    Examples
    --------
    >>> result = invert_lightcurve(
    ...     observed_lightcurve=obs_mags,
    ...     satellite=satellite,
    ...     observation_times=times,
    ...     sun_positions_j2000=sun_pos,
    ...     observer_positions_j2000=obs_pos,
    ...     satellite_positions_j2000=sat_pos,
    ...     observer_distances=distances,
    ...     mode='principal_axis',
    ...     uncertainty_mode='quick',
    ... )
    >>> print(f"Estimated omega: {result.omega0}")
    >>> result.plot_lightcurve_comparison()
    """
    # Import dependencies
    from .objective_function import ObjectiveFunction
    from .optimizers import multi_start_optimize, get_default_bounds
    from .constraints import ConstraintMode, get_bounds
    from .uncertainty import compute_fisher_uncertainty, compute_mcmc_uncertainty
    from .quaternion_utils import axis_angle_to_quaternion, normalize_quaternion
    from ..dynamics.attitude_propagator import propagate_attitude

    # Start timing
    start_time = time.time()

    # Print header
    print("=" * 80, flush=True)
    print("LIGHTCURVE INVERSION", flush=True)
    print("=" * 80, flush=True)
    n_obs = len(observation_times)
    shadows_str = "enabled" if compute_shadows else "disabled"
    workers_str = "all cores" if workers == -1 else str(workers)
    print(f"Mode: {mode} | Uncertainty: {uncertainty_mode} | Shadows: {shadows_str}", flush=True)
    print(f"Observations: {n_obs} | Parameters: 6 | Multi-starts: {n_starts} | Workers: {workers_str}", flush=True)
    print("-" * 80, flush=True)

    # Validate inputs
    if mode == "tumbling" and inertia_tensor is None:
        raise ValueError(
            "inertia_tensor is required for tumbling mode. "
            "Provide a 3x3 inertia tensor in body frame."
        )

    if mode not in ("principal_axis", "tumbling"):
        raise ValueError(
            f"mode must be 'principal_axis' or 'tumbling', got '{mode}'"
        )

    if uncertainty_mode not in ("quick", "full"):
        raise ValueError(
            f"uncertainty_mode must be 'quick' or 'full', got '{uncertainty_mode}'"
        )

    # Create objective function
    objective = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=observed_lightcurve,
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=observer_positions_j2000,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=observer_distances,
        uncertainties=measurement_uncertainties,
        compute_shadows_flag=compute_shadows,
        articulation_matrices=articulation_matrices,
    )

    # Get parameter bounds
    if constraint_mode is not None:
        omega_max_rad_per_s = np.deg2rad(omega_max_deg_per_s)
        bounds = get_bounds(constraint_mode, omega_max=omega_max_rad_per_s)
    else:
        bounds = get_default_bounds(omega_max_deg_per_s=omega_max_deg_per_s)

    # Run multi-start optimization
    results = multi_start_optimize(
        objective=objective,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        use_local_refinement=True,
        workers=workers,
    )

    # Best result is first (sorted by cost)
    best_result = results[0]
    optimal_params = best_result.params

    # Extract optimal parameters
    axis_angle = optimal_params[:3]
    omega0 = optimal_params[3:6]

    # Convert to quaternion
    q0 = axis_angle_to_quaternion(axis_angle)
    q0 = normalize_quaternion(q0)

    # Propagate attitude to get predicted lightcurve and omega history
    quaternions, omega_history = propagate_attitude(
        q0=q0,
        omega0=omega0,
        times=observation_times,
        mode=mode,
        inertia_tensor=inertia_tensor,
    )

    # Generate predicted lightcurve at optimal parameters
    # We need to re-evaluate with the optimal params to get the predicted lightcurve
    k1_vectors, k2_vectors = objective._compute_body_frame_vectors(quaternions)
    predicted_lightcurve = objective._generate_predicted_lightcurve(k1_vectors, k2_vectors)

    # Compute residuals
    valid_obs = np.isfinite(observed_lightcurve)
    valid_pred = np.isfinite(predicted_lightcurve)
    valid_mask = valid_obs & valid_pred
    n_valid = np.sum(valid_mask)

    if n_valid > 0:
        residuals = observed_lightcurve[valid_mask] - predicted_lightcurve[valid_mask]
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
    else:
        rms_residual = float('inf')

    chi_squared = best_result.cost

    # Compute uncertainties
    uncertainties_dict: Optional[Dict[str, NDArray[np.floating]]] = None
    mcmc_samples: Optional[NDArray[np.floating]] = None

    if uncertainty_mode == "quick":
        # Fisher Information Matrix approximation
        covariance, std_devs = compute_fisher_uncertainty(
            objective=objective,
            optimal_params=optimal_params,
        )
        uncertainties_dict = {
            "covariance": covariance,
            "std_devs": std_devs,
        }
    else:  # uncertainty_mode == "full"
        # Full MCMC posterior sampling
        mcmc_result = compute_mcmc_uncertainty(
            objective=objective,
            optimal_params=optimal_params,
            n_samples=mcmc_n_samples,
            burn_in=mcmc_burn_in,
            bounds=bounds,
        )
        uncertainties_dict = {
            "covariance": mcmc_result["covariance"],
            "std_devs": mcmc_result["std_devs"],
            "acceptance_fraction": mcmc_result["acceptance_fraction"],
            "autocorr_time": mcmc_result["autocorr_time"],
            "converged": mcmc_result["converged"],
        }
        mcmc_samples = mcmc_result["samples"]

    # Include omega_history only for tumbling mode (it's constant for principal_axis)
    omega_history_output: Optional[NDArray[np.floating]] = None
    if mode == "tumbling":
        omega_history_output = omega_history

    # Compute total time and print summary
    total_time = time.time() - start_time
    total_evaluations = objective.n_evaluations

    print("\n" + "=" * 80, flush=True)
    print("INVERSION COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"  Chi-squared: {chi_squared:.6f}", flush=True)
    print(f"  RMS residual: {rms_residual:.4f} mag", flush=True)
    print(f"  Total time: {total_time:.1f} seconds", flush=True)
    print(f"  Total evaluations: {total_evaluations:,}", flush=True)
    print("=" * 80, flush=True)

    return InversionResult(
        q0=q0,
        omega0=omega0,
        chi_squared=chi_squared,
        rms_residual=rms_residual,
        predicted_lightcurve=predicted_lightcurve,
        observed_lightcurve=np.asarray(observed_lightcurve),
        observation_times=np.asarray(observation_times),
        uncertainties=uncertainties_dict,
        omega_history=omega_history_output,
        mcmc_samples=mcmc_samples,
    )


def invert_lightcurve_multifidelity(
    observed_lightcurve: NDArray[np.floating],
    satellite: Any,
    observation_times: NDArray[np.floating],
    sun_positions_j2000: NDArray[np.floating],
    observer_positions_j2000: NDArray[np.floating],
    satellite_positions_j2000: NDArray[np.floating],
    observer_distances: NDArray[np.floating],
    mode: Literal["principal_axis", "tumbling"] = "principal_axis",
    uncertainty_mode: Literal["quick", "full"] = "quick",
    inertia_tensor: Optional[NDArray[np.floating]] = None,
    measurement_uncertainties: Optional[NDArray[np.floating]] = None,
    n_starts_coarse: int = 5,
    n_starts_fine: int = 2,
    n_seeds_to_refine: int = 3,
    constraint_mode: Optional[Any] = None,
    omega_max_deg_per_s: float = 30.0,
    seed: Optional[int] = None,
    mcmc_n_samples: int = 1000,
    mcmc_burn_in: int = 100,
    articulation_matrices: Optional[Dict[str, NDArray[np.floating]]] = None,
    workers: int = 1,
) -> InversionResult:
    """
    Run multi-fidelity lightcurve inversion for faster convergence.

    This function uses a two-stage optimization approach:
    1. **Coarse stage**: Fast exploration with shadows disabled (no ray tracing)
    2. **Fine stage**: Accurate refinement with shadows enabled (full physics)

    The coarse stage quickly identifies promising regions in parameter space,
    then the fine stage refines these using the full physics model. This
    approach is typically 2-5x faster than single-fidelity optimization
    for problems with many observations.

    Parameters
    ----------
    observed_lightcurve : ndarray
        Array of observed magnitudes.
    satellite : Satellite
        Satellite model with STL components and BRDF properties.
    observation_times : ndarray
        Array of observation times (seconds from reference epoch).
    sun_positions_j2000 : ndarray, shape (N, 3)
        Sun positions in J2000 frame in km.
    observer_positions_j2000 : ndarray, shape (N, 3)
        Observer positions in J2000 frame in km.
    satellite_positions_j2000 : ndarray, shape (N, 3)
        Satellite positions in J2000 frame in km.
    observer_distances : ndarray, shape (N,)
        Observer-to-satellite distances in km.
    mode : {'principal_axis', 'tumbling'}, optional
        Attitude propagation mode. Default is 'principal_axis'.
    uncertainty_mode : {'quick', 'full'}, optional
        Uncertainty estimation mode:
        - 'quick': Fisher Information approximation (fast)
        - 'full': Full MCMC posterior sampling (thorough)
        Default is 'quick'.
    inertia_tensor : ndarray, shape (3, 3), optional
        Inertia tensor in body frame (kg*m^2). Required for tumbling mode.
    measurement_uncertainties : ndarray, optional
        Measurement uncertainties for weighted chi-squared fitting.
    n_starts_coarse : int, optional
        Number of multi-starts for the coarse (no shadow) stage. Default is 5.
    n_starts_fine : int, optional
        Number of multi-starts for each fine (shadow) refinement. Default is 2.
    n_seeds_to_refine : int, optional
        Number of top coarse results to refine in the fine stage. Default is 3.
    constraint_mode : ConstraintMode, optional
        Constraint mode for bounds. If None, uses default bounds.
    omega_max_deg_per_s : float, optional
        Maximum angular velocity in degrees per second. Default is 30.
    seed : int, optional
        Random seed for reproducibility.
    mcmc_n_samples : int, optional
        Number of MCMC samples for full uncertainty mode. Default is 1000.
    mcmc_burn_in : int, optional
        Number of MCMC burn-in steps. Default is 100.
    articulation_matrices : dict, optional
        Pre-computed rotation matrices for articulated components.
    workers : int, optional
        Number of parallel workers for optimization. Default is 1 (serial).
        Set to -1 to use all available CPU cores.

    Returns
    -------
    InversionResult
        Complete inversion results including:
        - q0: Estimated initial quaternion
        - omega0: Estimated initial angular velocity
        - chi_squared: Fit quality metric (from fine stage)
        - rms_residual: RMS of residuals
        - uncertainties: Parameter uncertainties
        - predicted_lightcurve: Model prediction at best-fit
        - observed_lightcurve: Input observed data
        - observation_times: Input times
        - omega_history: Angular velocity evolution (tumbling mode)
        - mcmc_samples: MCMC samples (if uncertainty_mode='full')

    Notes
    -----
    The multi-fidelity approach works because:
    - Without shadows, all facets are assumed lit (faster computation)
    - The coarse model captures the main attitude-dependent brightness variations
    - Good coarse solutions are usually good starting points for fine optimization
    - The fine stage corrects for shadow effects (self-occlusion)

    For simple geometries with minimal self-shadowing, the speedup is modest.
    For complex geometries with significant shadows, the speedup can be 3-5x.

    Examples
    --------
    >>> result = invert_lightcurve_multifidelity(
    ...     observed_lightcurve=obs_mags,
    ...     satellite=satellite,
    ...     observation_times=times,
    ...     sun_positions_j2000=sun_pos,
    ...     observer_positions_j2000=obs_pos,
    ...     satellite_positions_j2000=sat_pos,
    ...     observer_distances=distances,
    ...     n_starts_coarse=5,   # Broad exploration
    ...     n_starts_fine=2,     # Focused refinement
    ...     n_seeds_to_refine=3, # Top-3 coarse results
    ...     workers=-1,          # Use all CPU cores
    ... )
    """
    # Import dependencies
    from .objective_function import ObjectiveFunction
    from .optimizers import multi_start_optimize, global_optimize, local_refine, get_default_bounds
    from .constraints import ConstraintMode, get_bounds
    from .uncertainty import compute_fisher_uncertainty, compute_mcmc_uncertainty
    from .quaternion_utils import axis_angle_to_quaternion, normalize_quaternion
    from ..dynamics.attitude_propagator import propagate_attitude

    # Start timing
    start_time = time.time()

    # Print header
    print("=" * 80, flush=True)
    print("MULTI-FIDELITY LIGHTCURVE INVERSION", flush=True)
    print("=" * 80, flush=True)
    n_obs = len(observation_times)
    workers_str = "all cores" if workers == -1 else str(workers)
    print(f"Mode: {mode} | Uncertainty: {uncertainty_mode}", flush=True)
    print(f"Observations: {n_obs} | Parameters: 6 | Workers: {workers_str}", flush=True)
    print(f"Coarse: {n_starts_coarse} starts (no shadows) -> Top {n_seeds_to_refine} -> Fine: {n_starts_fine} starts/seed (with shadows)", flush=True)
    print("-" * 80, flush=True)

    # Validate inputs
    if mode == "tumbling" and inertia_tensor is None:
        raise ValueError(
            "inertia_tensor is required for tumbling mode. "
            "Provide a 3x3 inertia tensor in body frame."
        )

    if mode not in ("principal_axis", "tumbling"):
        raise ValueError(
            f"mode must be 'principal_axis' or 'tumbling', got '{mode}'"
        )

    if uncertainty_mode not in ("quick", "full"):
        raise ValueError(
            f"uncertainty_mode must be 'quick' or 'full', got '{uncertainty_mode}'"
        )

    # Get parameter bounds
    if constraint_mode is not None:
        omega_max_rad_per_s = np.deg2rad(omega_max_deg_per_s)
        bounds = get_bounds(constraint_mode, omega_max=omega_max_rad_per_s)
    else:
        bounds = get_default_bounds(omega_max_deg_per_s=omega_max_deg_per_s)

    # =========================================================================
    # STAGE 1: COARSE OPTIMIZATION (no shadows)
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("STAGE 1: COARSE OPTIMIZATION (shadows disabled)", flush=True)
    print("=" * 60, flush=True)
    coarse_start = time.time()

    # Create coarse objective function (no shadows)
    objective_coarse = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=observed_lightcurve,
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=observer_positions_j2000,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=observer_distances,
        uncertainties=measurement_uncertainties,
        compute_shadows_flag=False,  # Fast: no ray tracing
        articulation_matrices=articulation_matrices,
    )

    # Run multi-start optimization on coarse model
    coarse_results = multi_start_optimize(
        objective=objective_coarse,
        bounds=bounds,
        n_starts=n_starts_coarse,
        seed=seed,
        use_local_refinement=True,
        workers=workers,
    )

    coarse_time = time.time() - coarse_start
    coarse_evals = objective_coarse.n_evaluations

    print(f"\nCoarse stage complete: {coarse_time:.1f}s, {coarse_evals:,} evaluations", flush=True)
    print(f"Top {n_seeds_to_refine} coarse results:", flush=True)
    for i, result in enumerate(coarse_results[:n_seeds_to_refine]):
        print(f"  {i+1}. cost = {result.cost:.6f}", flush=True)

    # =========================================================================
    # STAGE 2: FINE OPTIMIZATION (with shadows)
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("STAGE 2: FINE OPTIMIZATION (shadows enabled)", flush=True)
    print("=" * 60, flush=True)
    fine_start = time.time()

    # Create fine objective function (with shadows)
    objective_fine = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=observed_lightcurve,
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=observer_positions_j2000,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=observer_distances,
        uncertainties=measurement_uncertainties,
        compute_shadows_flag=True,  # Full physics: ray tracing
        articulation_matrices=articulation_matrices,
    )

    # Refine top-N coarse results with fine model
    fine_results = []
    seeds_to_use = min(n_seeds_to_refine, len(coarse_results))

    for i in range(seeds_to_use):
        seed_params = coarse_results[i].params
        seed_i = None if seed is None else seed + 1000 + i  # Different seed for fine stage

        print(f"\n  Refining seed {i+1}/{seeds_to_use} (coarse cost = {coarse_results[i].cost:.6f}):", flush=True)

        # Run global optimization seeded with coarse result
        for j in range(n_starts_fine):
            start_seed = None if seed_i is None else seed_i + j

            print(f"    Start {j+1}/{n_starts_fine}:", flush=True)
            print(f"      Global optimization (seeded)...", flush=True)

            # Use x0 to seed the differential evolution
            global_result = global_optimize(
                objective=objective_fine,
                bounds=bounds,
                seed=start_seed,
                x0=seed_params,  # Seed with coarse result
                maxiter=500,
                polish=False,
                show_progress=True,
                workers=workers,
            )

            print(f"      Local refinement (L-BFGS-B)...", flush=True)

            # Refine with local optimizer
            refined_result = local_refine(
                objective=objective_fine,
                x0=global_result.params,
                bounds=bounds,
                maxiter=500,
                show_progress=True,
                initial_cost=global_result.cost,
            )

            fine_results.append(refined_result)
            print(f"    Start {j+1} complete: cost = {refined_result.cost:.6f}", flush=True)

    # Sort fine results by cost
    fine_results.sort(key=lambda r: r.cost)

    fine_time = time.time() - fine_start
    fine_evals = objective_fine.n_evaluations

    print(f"\nFine stage complete: {fine_time:.1f}s, {fine_evals:,} evaluations", flush=True)
    print(f"Best fine result: cost = {fine_results[0].cost:.6f}", flush=True)

    # =========================================================================
    # EXTRACT BEST RESULT AND COMPUTE OUTPUTS
    # =========================================================================
    best_result = fine_results[0]
    optimal_params = best_result.params

    # Extract optimal parameters
    axis_angle = optimal_params[:3]
    omega0 = optimal_params[3:6]

    # Convert to quaternion
    q0 = axis_angle_to_quaternion(axis_angle)
    q0 = normalize_quaternion(q0)

    # Propagate attitude to get predicted lightcurve and omega history
    quaternions, omega_history = propagate_attitude(
        q0=q0,
        omega0=omega0,
        times=observation_times,
        mode=mode,
        inertia_tensor=inertia_tensor,
    )

    # Generate predicted lightcurve at optimal parameters
    k1_vectors, k2_vectors = objective_fine._compute_body_frame_vectors(quaternions)
    predicted_lightcurve = objective_fine._generate_predicted_lightcurve(k1_vectors, k2_vectors)

    # Compute residuals
    valid_obs = np.isfinite(observed_lightcurve)
    valid_pred = np.isfinite(predicted_lightcurve)
    valid_mask = valid_obs & valid_pred
    n_valid = np.sum(valid_mask)

    if n_valid > 0:
        residuals = observed_lightcurve[valid_mask] - predicted_lightcurve[valid_mask]
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
    else:
        rms_residual = float('inf')

    chi_squared = best_result.cost

    # Compute uncertainties
    uncertainties_dict: Optional[Dict[str, NDArray[np.floating]]] = None
    mcmc_samples: Optional[NDArray[np.floating]] = None

    if uncertainty_mode == "quick":
        # Fisher Information Matrix approximation
        covariance, std_devs = compute_fisher_uncertainty(
            objective=objective_fine,
            optimal_params=optimal_params,
        )
        uncertainties_dict = {
            "covariance": covariance,
            "std_devs": std_devs,
        }
    else:  # uncertainty_mode == "full"
        # Full MCMC posterior sampling
        mcmc_result = compute_mcmc_uncertainty(
            objective=objective_fine,
            optimal_params=optimal_params,
            n_samples=mcmc_n_samples,
            burn_in=mcmc_burn_in,
            bounds=bounds,
        )
        uncertainties_dict = {
            "covariance": mcmc_result["covariance"],
            "std_devs": mcmc_result["std_devs"],
            "acceptance_fraction": mcmc_result["acceptance_fraction"],
            "autocorr_time": mcmc_result["autocorr_time"],
            "converged": mcmc_result["converged"],
        }
        mcmc_samples = mcmc_result["samples"]

    # Include omega_history only for tumbling mode
    omega_history_output: Optional[NDArray[np.floating]] = None
    if mode == "tumbling":
        omega_history_output = omega_history

    # Compute total time and print summary
    total_time = time.time() - start_time
    total_evaluations = coarse_evals + fine_evals

    print("\n" + "=" * 80, flush=True)
    print("MULTI-FIDELITY INVERSION COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"  Chi-squared: {chi_squared:.6f}", flush=True)
    print(f"  RMS residual: {rms_residual:.4f} mag", flush=True)
    print(f"  Total time: {total_time:.1f} seconds", flush=True)
    print(f"    Coarse stage: {coarse_time:.1f}s ({coarse_evals:,} evals)", flush=True)
    print(f"    Fine stage: {fine_time:.1f}s ({fine_evals:,} evals)", flush=True)
    print(f"  Total evaluations: {total_evaluations:,}", flush=True)
    print("=" * 80, flush=True)

    return InversionResult(
        q0=q0,
        omega0=omega0,
        chi_squared=chi_squared,
        rms_residual=rms_residual,
        predicted_lightcurve=predicted_lightcurve,
        observed_lightcurve=np.asarray(observed_lightcurve),
        observation_times=np.asarray(observation_times),
        uncertainties=uncertainties_dict,
        omega_history=omega_history_output,
        mcmc_samples=mcmc_samples,
    )
