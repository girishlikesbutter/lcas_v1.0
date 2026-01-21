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
        - Additional constraints based on principal axis rotation assumption
    """
    raise NotImplementedError("get_bounds not yet implemented")
