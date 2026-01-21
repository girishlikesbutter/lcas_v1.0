"""
Inversion module for LCAS.

This module provides lightcurve inversion capabilities for estimating
satellite initial attitude and angular velocity from observed lightcurves
using optimization techniques.

Submodules:
    quaternion_utils: Quaternion manipulation for optimization.
    objective_function: Wrapper for the forward model as optimization objective.
    optimizers: Global and local optimization algorithms.
    constraints: Constraint modes for optimization bounds.
    uncertainty: Uncertainty estimation methods (Fisher, MCMC).
    results: Container classes for inversion results.

Main functions:
    invert_lightcurve: Run the full inversion pipeline.
"""

from .quaternion_utils import (
    normalize_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_multiply,
)

from .objective_function import ObjectiveFunction

from .optimizers import (
    OptimizationResult,
    get_default_bounds,
    global_optimize,
    local_refine,
    multi_start_optimize,
)

from .constraints import (
    ConstraintMode,
    get_bounds,
)

from .uncertainty import (
    compute_fisher_uncertainty,
    compute_mcmc_uncertainty,
)

from .results import (
    InversionResult,
    invert_lightcurve,
)

__all__ = [
    # Quaternion utilities
    "normalize_quaternion",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "quaternion_multiply",
    # Objective function
    "ObjectiveFunction",
    # Optimizers
    "OptimizationResult",
    "get_default_bounds",
    "global_optimize",
    "local_refine",
    "multi_start_optimize",
    # Constraints
    "ConstraintMode",
    "get_bounds",
    # Uncertainty
    "compute_fisher_uncertainty",
    "compute_mcmc_uncertainty",
    # Results
    "InversionResult",
    "invert_lightcurve",
]
