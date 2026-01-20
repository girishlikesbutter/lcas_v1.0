"""
Dynamics module for LCAS.

This module provides attitude propagation capabilities for satellite dynamics,
supporting both principal axis rotation (constant angular velocity) and
tumbling motion with evolving angular velocity using Euler equations.

Submodules:
    attitude_propagator: Functions for propagating satellite attitude over time.

Main functions:
    propagate_attitude: Unified interface for attitude propagation.
    propagate_principal_axis: Propagate with constant angular velocity.
    propagate_euler: Propagate with Euler dynamics for tumbling motion.
"""

from .attitude_propagator import (
    propagate_attitude,
    propagate_principal_axis,
    propagate_euler,
)

__all__ = [
    "propagate_attitude",
    "propagate_principal_axis",
    "propagate_euler",
]
