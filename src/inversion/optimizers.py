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
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

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


def global_optimize(
    objective: ObjectiveFunction,
    bounds: List[Tuple[float, float]],
) -> OptimizationResult:
    """
    Run global optimization using Differential Evolution.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function to minimize.
    bounds : list of (min, max) tuples
        Parameter bounds for each of the 6 parameters.
        Default omega bounds are 0 to ~30 deg/s magnitude.

    Returns
    -------
    OptimizationResult
        Optimization result with best parameters and cost.
    """
    raise NotImplementedError("global_optimize not yet implemented")


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
