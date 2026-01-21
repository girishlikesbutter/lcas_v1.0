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
from scipy.optimize import differential_evolution

from .objective_function import ObjectiveFunction


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
    bounds: List[Tuple[float, float]],
) -> OptimizationResult:
    """
    Refine solution using local optimization (L-BFGS-B).

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function to minimize.
    x0 : ndarray, shape (6,)
        Initial guess from global optimization.
    bounds : list of (min, max) tuples
        Parameter bounds.

    Returns
    -------
    OptimizationResult
        Refined optimization result.
    """
    raise NotImplementedError("local_refine not yet implemented")


def multi_start_optimize(
    objective: ObjectiveFunction,
    bounds: List[Tuple[float, float]],
    n_starts: int = 5,
) -> List[OptimizationResult]:
    """
    Run optimization with multiple random starting points.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function to minimize.
    bounds : list of (min, max) tuples
        Parameter bounds.
    n_starts : int, optional
        Number of random starting points. Default is 5.

    Returns
    -------
    list of OptimizationResult
        Results sorted by cost (best first).
    """
    raise NotImplementedError("multi_start_optimize not yet implemented")
