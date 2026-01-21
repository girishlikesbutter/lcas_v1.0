"""
Objective function wrapper for lightcurve inversion.

This module provides the ObjectiveFunction class that wraps the LCAS
forward model (generate_lightcurves) for use in optimization algorithms.

The objective function computes the chi-squared residual between predicted
and observed lightcurves given a set of initial attitude parameters.
"""

from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray


class ObjectiveFunction:
    """
    Objective function wrapper for lightcurve inversion optimization.

    This class wraps the LCAS generate_lightcurves forward model to compute
    residuals between predicted and observed lightcurves for a given set
    of initial attitude parameters.

    Parameters
    ----------
    satellite_config : Any
        Satellite configuration for the forward model.
    observer_config : Any
        Observer configuration for the forward model.
    observation_times : ndarray
        Array of observation times.
    observed_lightcurve : ndarray
        Array of observed magnitudes.
    uncertainties : ndarray, optional
        Array of measurement uncertainties for weighted chi-squared.

    Attributes
    ----------
    n_evaluations : int
        Number of times evaluate() has been called.
    """

    def __init__(
        self,
        satellite_config: Any,
        observer_config: Any,
        observation_times: NDArray[np.floating],
        observed_lightcurve: NDArray[np.floating],
        uncertainties: Optional[NDArray[np.floating]] = None,
    ) -> None:
        """Initialize the objective function with observation data."""
        raise NotImplementedError("ObjectiveFunction.__init__ not yet implemented")

    def evaluate(
        self,
        params: NDArray[np.floating],
    ) -> float:
        """
        Evaluate the objective function for given parameters.

        Parameters
        ----------
        params : ndarray, shape (6,)
            Parameter array: [axis_angle(3), omega(3)]
            - axis_angle: Initial orientation as axis-angle (3 params)
            - omega: Initial angular velocity in body frame (3 params)

        Returns
        -------
        float
            Chi-squared residual between predicted and observed lightcurve.
            Lower values indicate better fit.
        """
        raise NotImplementedError("ObjectiveFunction.evaluate not yet implemented")
