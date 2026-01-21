"""
Uncertainty estimation methods for lightcurve inversion.

This module provides methods for estimating parameter uncertainties
after optimization:

- Fisher Information Matrix: Quick approximation using Hessian
- MCMC sampling: Full posterior characterization using emcee
"""

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .objective_function import ObjectiveFunction


def compute_fisher_uncertainty(
    objective: ObjectiveFunction,
    optimal_params: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Estimate parameter uncertainties using Fisher Information Matrix.

    Uses numerical approximation of the Hessian at the optimum to
    compute the covariance matrix.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function (must be at the optimum).
    optimal_params : ndarray, shape (6,)
        Optimal parameters from optimization.

    Returns
    -------
    covariance : ndarray, shape (6, 6)
        Parameter covariance matrix (inverse of Fisher Information).
    std_devs : ndarray, shape (6,)
        Standard deviations for each parameter.
    """
    raise NotImplementedError("compute_fisher_uncertainty not yet implemented")


def compute_mcmc_uncertainty(
    objective: ObjectiveFunction,
    optimal_params: NDArray[np.floating],
    n_samples: int = 1000,
    n_walkers: Optional[int] = None,
    burn_in: int = 100,
) -> Dict[str, NDArray[np.floating]]:
    """
    Estimate parameter uncertainties using MCMC sampling with emcee.

    Performs full posterior characterization using affine-invariant
    ensemble sampling.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function.
    optimal_params : ndarray, shape (6,)
        Optimal parameters to initialize walkers around.
    n_samples : int, optional
        Number of samples per walker after burn-in. Default is 1000.
    n_walkers : int, optional
        Number of walkers. Default is 2 * n_params = 12.
    burn_in : int, optional
        Number of burn-in steps. Default is 100.

    Returns
    -------
    dict
        Dictionary containing:
        - 'samples': ndarray, shape (n_samples * n_walkers, 6)
            Posterior samples.
        - 'covariance': ndarray, shape (6, 6)
            Parameter covariance matrix from samples.
        - 'std_devs': ndarray, shape (6,)
            Standard deviations for each parameter.
        - 'acceptance_fraction': float
            Mean acceptance fraction across walkers.
        - 'autocorr_time': ndarray, shape (6,)
            Integrated autocorrelation time for each parameter.
    """
    raise NotImplementedError("compute_mcmc_uncertainty not yet implemented")
