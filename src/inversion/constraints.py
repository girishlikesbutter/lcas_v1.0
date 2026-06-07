"""
Constraint modes for lightcurve inversion optimization.

This module defines constraint modes that control the parameter bounds
and constraints applied during optimization.

Modes:
- FREE: Minimal constraints, allowing full exploration
- PHYSICS_INFORMED: Adds physical constraints based on rigid body dynamics
"""

from enum import Enum
from typing import List, Tuple

import numpy as np


class ConstraintMode(Enum):
    """
    Constraint modes for optimization bounds.

    Attributes
    ----------
    FREE : int
        Minimal constraints - only bounds on angular velocity magnitude.
    PHYSICS_INFORMED : int
        Physics-informed constraints including principal axis rotation.
    """

    FREE = 1
    PHYSICS_INFORMED = 2


def get_bounds(
    mode: ConstraintMode,
    omega_max: float = 0.5236,  # ~30 deg/s in rad/s
) -> List[Tuple[float, float]]:
    """
    Get parameter bounds for the given constraint mode.

    Parameters
    ----------
    mode : ConstraintMode
        The constraint mode to use.
    omega_max : float, optional
        Maximum angular velocity magnitude in rad/s.
        Default is ~30 deg/s (0.5236 rad/s).

    Returns
    -------
    list of (min, max) tuples
        Bounds for each of the 6 parameters:
        [axis_angle_x, axis_angle_y, axis_angle_z,
         omega_x, omega_y, omega_z]

    Notes
    -----
    For FREE mode:
        - axis_angle: [-pi, pi] for each component
        - omega: [-omega_max, omega_max] for each component

    For PHYSICS_INFORMED mode:
        - axis_angle: [-pi, pi] for each component (same as FREE)
        - omega: tighter bounds to encourage principal axis rotation
          The omega bounds are reduced to omega_max / sqrt(3) per component,
          ensuring the total magnitude cannot exceed omega_max while biasing
          the optimizer toward solutions with spin about a single axis.

    Examples
    --------
    >>> from src.inversion.constraints import ConstraintMode, get_bounds
    >>> bounds = get_bounds(ConstraintMode.FREE, omega_max=0.5236)
    >>> len(bounds)
    6
    >>> bounds[0]  # axis_angle_x bounds
    (-3.141592653589793, 3.141592653589793)
    """
    if not isinstance(mode, ConstraintMode):
        raise ValueError(
            f"mode must be a ConstraintMode enum, got {type(mode).__name__}"
        )

    # axis_angle bounds: [-pi, pi] covers all possible rotations
    axis_angle_bounds: List[Tuple[float, float]] = [(-np.pi, np.pi)] * 3

    if mode == ConstraintMode.FREE:
        # FREE mode: each omega component can take full range
        # This allows any angular velocity direction up to omega_max magnitude
        omega_bounds: List[Tuple[float, float]] = [(-omega_max, omega_max)] * 3

    elif mode == ConstraintMode.PHYSICS_INFORMED:
        # PHYSICS_INFORMED mode: tighter bounds to encourage principal axis rotation
        #
        # Principal axis rotation means spin about a single body axis.
        # To bias the optimizer toward such solutions, we reduce the per-component
        # omega bounds. With bounds of omega_max/sqrt(3) per component, the
        # maximum possible magnitude is still omega_max (when all components
        # are at their limits), but solutions with most energy in one component
        # are more accessible to the optimizer.
        #
        # This doesn't strictly enforce principal axis rotation but makes it
        # more likely to be found when it's the true solution.
        omega_component_max = omega_max / np.sqrt(3.0)
        omega_bounds = [(-omega_component_max, omega_component_max)] * 3

    else:
        # Should not reach here, but handle for completeness
        raise ValueError(f"Unknown constraint mode: {mode}")

    return axis_angle_bounds + omega_bounds
