"""
Optimization algorithms for lightcurve inversion.

This module provides global and local optimization algorithms for
finding the initial attitude and angular velocity that best explain
an observed lightcurve.

Includes:
- Differential Evolution for global optimization
- L-BFGS-B for local refinement
- Multi-start wrapper for robustness
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize

from .objective_function import ObjectiveFunction
from .quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle, normalize_quaternion


@dataclass
class OptimizationResult:
    """
    Container for optimization results.

    Attributes
    ----------
    params : ndarray, shape (6,)
        Optimal parameters [axis_angle(3), omega(3)].
    cost : float
        Final objective function value.
    n_evaluations : int
        Number of function evaluations.
    success : bool
        Whether optimization converged successfully.
    message : str
        Description of termination condition.
    """

    params: NDArray[np.floating]
    cost: float
    n_evaluations: int
    success: bool
    message: str


def get_default_bounds(omega_max_deg_per_s: float = 30.0) -> List[Tuple[float, float]]:
    """
    Get default parameter bounds for optimization.

    Parameters
    ----------
    omega_max_deg_per_s : float, optional
        Maximum angular velocity magnitude in degrees per second.
        Default is 30 deg/s.

    Returns
    -------
    list of (min, max) tuples
        Bounds for 6 parameters: [axis_angle(3), omega(3)].
        - axis_angle bounds: [-pi, pi] for each component (covers all rotations)
        - omega bounds: [-omega_max, omega_max] for each component in rad/s
    """
    omega_max_rad_per_s = np.deg2rad(omega_max_deg_per_s)

    # axis_angle parameters: [-pi, pi] covers all possible rotations
    # (axis_angle magnitude <= pi is sufficient for any rotation)
    axis_angle_bounds = [(-np.pi, np.pi)] * 3

    # omega parameters: each component bounded by max magnitude
    omega_bounds = [(-omega_max_rad_per_s, omega_max_rad_per_s)] * 3

    return axis_angle_bounds + omega_bounds


def global_optimize(
    objective: ObjectiveFunction,
    bounds: Optional[List[Tuple[float, float]]] = None,
    seed: Optional[int] = None,
    maxiter: int = 1000,
    tol: float = 0.01,
    workers: int = 1,
    polish: bool = False,
) -> OptimizationResult:
    """
    Run global optimization using Differential Evolution.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function to minimize.
    bounds : list of (min, max) tuples, optional
        Parameter bounds for each of the 6 parameters.
        If None, uses default bounds with omega max of 30 deg/s.
    seed : int, optional
        Random seed for reproducibility.
    maxiter : int, optional
        Maximum number of generations. Default is 1000.
    tol : float, optional
        Relative tolerance for convergence. Default is 0.01.
    workers : int, optional
        Number of parallel workers. Default is 1 (serial).
    polish : bool, optional
        Whether to polish the result with L-BFGS-B. Default is False
        (use local_refine separately for more control).

    Returns
    -------
    OptimizationResult
        Optimization result with best parameters and cost.
    """
    if bounds is None:
        bounds = get_default_bounds()

    # Wrap the objective function to track evaluations
    n_evals = [0]  # Use list for mutable closure

    def wrapped_objective(params: NDArray[np.floating]) -> float:
        n_evals[0] += 1
        return objective.evaluate(params)

    # Run differential evolution
    result = differential_evolution(
        wrapped_objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=tol,
        workers=workers,
        polish=polish,
        strategy="best1bin",  # Good balance of exploration and exploitation
        mutation=(0.5, 1.0),  # Dithered mutation for robustness
        recombination=0.7,    # Standard crossover probability
        updating="deferred",  # Required for parallel execution
    )

    return OptimizationResult(
        params=result.x,
        cost=result.fun,
        n_evaluations=n_evals[0],
        success=result.success,
        message=result.message,
    )


def local_refine(
    objective: ObjectiveFunction,
    x0: NDArray[np.floating],
    bounds: Optional[List[Tuple[float, float]]] = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
) -> OptimizationResult:
    """
    Refine solution using local optimization (L-BFGS-B).

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function to minimize.
    x0 : ndarray, shape (6,)
        Initial guess from global optimization.
    bounds : list of (min, max) tuples, optional
        Parameter bounds. If None, uses default bounds.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    ftol : float, optional
        Function tolerance for convergence. Default is 1e-8.
    gtol : float, optional
        Gradient tolerance for convergence. Default is 1e-5.

    Returns
    -------
    OptimizationResult
        Refined optimization result.

    Notes
    -----
    The quaternion normalization constraint is enforced by converting the
    axis-angle representation back to a normalized quaternion and then
    back to axis-angle after each iteration. This ensures the optimization
    stays on the unit quaternion manifold without adding explicit constraints.
    """
    if bounds is None:
        bounds = get_default_bounds()

    x0 = np.asarray(x0, dtype=np.float64)

    # Normalize the initial axis-angle to ensure valid starting point
    axis_angle = x0[:3]
    omega = x0[3:]

    # Convert to quaternion, normalize, convert back to axis-angle
    q = axis_angle_to_quaternion(axis_angle)
    q = normalize_quaternion(q)
    axis_angle_normalized = quaternion_to_axis_angle(q)

    x0_normalized = np.concatenate([axis_angle_normalized, omega])

    # Wrap the objective function to:
    # 1. Track evaluations
    # 2. Normalize the axis-angle representation at each evaluation
    n_evals = [0]  # Use list for mutable closure

    def wrapped_objective(params: NDArray[np.floating]) -> float:
        n_evals[0] += 1

        # Extract axis-angle and omega
        axis_angle_current = params[:3]
        omega_current = params[3:]

        # Normalize via quaternion round-trip (enforces unit quaternion constraint)
        q_current = axis_angle_to_quaternion(axis_angle_current)
        q_normalized = normalize_quaternion(q_current)
        axis_angle_norm = quaternion_to_axis_angle(q_normalized)

        # Create normalized parameter vector
        params_normalized = np.concatenate([axis_angle_norm, omega_current])

        return objective.evaluate(params_normalized)

    # Run L-BFGS-B optimization
    result = minimize(
        wrapped_objective,
        x0_normalized,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": ftol,
            "gtol": gtol,
            "disp": False,
        },
    )

    # Normalize final result
    final_axis_angle = result.x[:3]
    final_omega = result.x[3:]

    q_final = axis_angle_to_quaternion(final_axis_angle)
    q_final_normalized = normalize_quaternion(q_final)
    final_axis_angle_normalized = quaternion_to_axis_angle(q_final_normalized)

    final_params = np.concatenate([final_axis_angle_normalized, final_omega])

    return OptimizationResult(
        params=final_params,
        cost=result.fun,
        n_evaluations=n_evals[0],
        success=result.success,
        message=result.message,
    )


def multi_start_optimize(
    objective: ObjectiveFunction,
    bounds: Optional[List[Tuple[float, float]]] = None,
    n_starts: int = 5,
    seed: Optional[int] = None,
    global_maxiter: int = 500,
    local_maxiter: int = 500,
    use_local_refinement: bool = True,
) -> List[OptimizationResult]:
    """
    Run optimization with multiple random starting points.

    This function runs global optimization (Differential Evolution) from
    multiple random initial populations to improve robustness against
    local minima. Optionally refines each result with local optimization.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function to minimize.
    bounds : list of (min, max) tuples, optional
        Parameter bounds. If None, uses default bounds with omega max of 30 deg/s.
    n_starts : int, optional
        Number of random starting points. Default is 5.
    seed : int, optional
        Base random seed for reproducibility. Each start uses seed+i.
    global_maxiter : int, optional
        Maximum iterations for global optimizer. Default is 500.
    local_maxiter : int, optional
        Maximum iterations for local refinement. Default is 500.
    use_local_refinement : bool, optional
        Whether to refine global result with L-BFGS-B. Default is True.

    Returns
    -------
    list of OptimizationResult
        Results sorted by cost (best first). The first element is the
        best result found across all starts.

    Notes
    -----
    Each start uses a different random seed (seed + start_index) to ensure
    diverse initial populations. The results are sorted by final cost so
    the best solution is always first in the returned list.
    """
    if bounds is None:
        bounds = get_default_bounds()

    results: List[OptimizationResult] = []

    for i in range(n_starts):
        # Use different seed for each start
        start_seed = None if seed is None else seed + i

        # Run global optimization
        global_result = global_optimize(
            objective=objective,
            bounds=bounds,
            seed=start_seed,
            maxiter=global_maxiter,
            polish=False,  # We'll do local refinement separately
        )

        if use_local_refinement:
            # Refine with local optimizer
            refined_result = local_refine(
                objective=objective,
                x0=global_result.params,
                bounds=bounds,
                maxiter=local_maxiter,
            )

            # Combine evaluation counts
            total_evals = global_result.n_evaluations + refined_result.n_evaluations
            final_result = OptimizationResult(
                params=refined_result.params,
                cost=refined_result.cost,
                n_evaluations=total_evals,
                success=refined_result.success,
                message=f"Global + local: {refined_result.message}",
            )
        else:
            final_result = global_result

        results.append(final_result)

    # Sort by cost (best first)
    results.sort(key=lambda r: r.cost)

    return results
