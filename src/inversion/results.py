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
