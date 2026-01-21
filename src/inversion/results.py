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
from typing import Any, Dict, Literal, Optional
import numpy as np
from numpy.typing import NDArray


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

    def plot_lightcurve_comparison(self, ax: Optional[Any] = None) -> Any:
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
        raise NotImplementedError("plot_lightcurve_comparison not yet implemented")

    def plot_residuals(self, ax: Optional[Any] = None) -> Any:
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
        raise NotImplementedError("plot_residuals not yet implemented")

    def plot_corner(self) -> Any:
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
        """
        raise NotImplementedError("plot_corner not yet implemented")


def invert_lightcurve(
    observed_lightcurve: NDArray[np.floating],
    satellite_config: Any,
    observer_config: Any,
    observation_times: NDArray[np.floating],
    mode: Literal["principal_axis", "tumbling"] = "principal_axis",
    uncertainty_mode: Literal["quick", "full"] = "quick",
    inertia_tensor: Optional[NDArray[np.floating]] = None,
    uncertainties: Optional[NDArray[np.floating]] = None,
) -> InversionResult:
    """
    Run the full lightcurve inversion pipeline.

    Parameters
    ----------
    observed_lightcurve : ndarray
        Array of observed magnitudes.
    satellite_config : Any
        Satellite configuration for the forward model.
    observer_config : Any
        Observer configuration for the forward model.
    observation_times : ndarray
        Array of observation times.
    mode : {'principal_axis', 'tumbling'}, optional
        Attitude propagation mode. Default is 'principal_axis'.
    uncertainty_mode : {'quick', 'full'}, optional
        Uncertainty estimation mode:
        - 'quick': Fisher Information approximation
        - 'full': Full MCMC posterior sampling
        Default is 'quick'.
    inertia_tensor : ndarray, shape (3, 3), optional
        Inertia tensor, required for tumbling mode.
    uncertainties : ndarray, optional
        Measurement uncertainties for weighted fitting.

    Returns
    -------
    InversionResult
        Complete inversion results including estimates, uncertainties,
        and data for visualization.

    Raises
    ------
    ValueError
        If tumbling mode is selected without providing inertia_tensor.
    """
    raise NotImplementedError("invert_lightcurve not yet implemented")
