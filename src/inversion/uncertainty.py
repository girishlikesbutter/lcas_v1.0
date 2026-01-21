"""
Uncertainty estimation methods for lightcurve inversion.

This module provides methods for estimating parameter uncertainties
after optimization:

- Fisher Information Matrix: Quick approximation using Hessian
- MCMC sampling: Full posterior characterization using emcee
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .objective_function import ObjectiveFunction

logger = logging.getLogger(__name__)


def _compute_hessian(
    func: ObjectiveFunction,
    x: NDArray[np.floating],
    step_size: float = 1e-5,
) -> NDArray[np.floating]:
    """
    Compute the Hessian matrix using central finite differences.

    Parameters
    ----------
    func : ObjectiveFunction
        Objective function with evaluate() method.
    x : ndarray, shape (n,)
        Point at which to compute the Hessian.
    step_size : float, optional
        Step size for finite differences. Default is 1e-5.

    Returns
    -------
    hessian : ndarray, shape (n, n)
        Symmetric Hessian matrix.
    """
    n = len(x)
    hessian = np.zeros((n, n), dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Compute diagonal elements using central difference formula:
    # d^2f/dx_i^2 = (f(x+h_i) - 2*f(x) + f(x-h_i)) / h^2
    f_x = func.evaluate(x)

    for i in range(n):
        h_i = np.zeros(n)
        h_i[i] = step_size

        f_plus = func.evaluate(x + h_i)
        f_minus = func.evaluate(x - h_i)

        hessian[i, i] = (f_plus - 2 * f_x + f_minus) / (step_size**2)

    # Compute off-diagonal elements using central difference:
    # d^2f/dx_i*dx_j = (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4*h^2)
    for i in range(n):
        for j in range(i + 1, n):
            h_i = np.zeros(n)
            h_i[i] = step_size
            h_j = np.zeros(n)
            h_j[j] = step_size

            f_pp = func.evaluate(x + h_i + h_j)
            f_pm = func.evaluate(x + h_i - h_j)
            f_mp = func.evaluate(x - h_i + h_j)
            f_mm = func.evaluate(x - h_i - h_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * step_size**2)
            hessian[j, i] = hessian[i, j]  # Symmetric

    return hessian


def compute_fisher_uncertainty(
    objective: ObjectiveFunction,
    optimal_params: NDArray[np.floating],
    step_size: float = 1e-5,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Estimate parameter uncertainties using Fisher Information Matrix.

    Uses numerical approximation of the Hessian at the optimum to
    compute the covariance matrix. The Fisher Information Matrix is
    approximated by the Hessian of the chi-squared objective function
    at the minimum.

    Parameters
    ----------
    objective : ObjectiveFunction
        The objective function (must be at the optimum).
    optimal_params : ndarray, shape (6,)
        Optimal parameters from optimization.
    step_size : float, optional
        Step size for numerical Hessian computation. Default is 1e-5.

    Returns
    -------
    covariance : ndarray, shape (6, 6)
        Parameter covariance matrix (inverse of Fisher Information).
    std_devs : ndarray, shape (6,)
        Standard deviations for each parameter.

    Notes
    -----
    For chi-squared minimization, the covariance matrix is given by:
        C = 2 * H^{-1}
    where H is the Hessian of the chi-squared function at the minimum.

    The factor of 2 arises because for chi-squared = sum((obs-pred)^2 / sigma^2),
    the second derivative gives 2 * Fisher Information.

    If the Hessian is not positive definite (indicating the optimum may not
    be well-defined), a warning is logged and a pseudo-inverse is used.
    """
    optimal_params = np.asarray(optimal_params, dtype=np.float64)
    n_params = len(optimal_params)

    logger.debug(f"Computing Hessian at optimum with step_size={step_size}")

    # Compute Hessian at optimum
    hessian = _compute_hessian(objective, optimal_params, step_size)

    # Fisher Information Matrix is 0.5 * Hessian for chi-squared
    # So covariance = 2 * Hessian^{-1}
    try:
        # Check if Hessian is positive definite
        eigenvalues = np.linalg.eigvalsh(hessian)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue <= 0:
            logger.warning(
                f"Hessian is not positive definite (min eigenvalue: {min_eigenvalue:.2e}). "
                "Uncertainties may be unreliable. Using pseudo-inverse."
            )
            # Add regularization to make it invertible
            reg = abs(min_eigenvalue) + 1e-10
            hessian_reg = hessian + reg * np.eye(n_params)
            hessian_inv = np.linalg.inv(hessian_reg)
        else:
            hessian_inv = np.linalg.inv(hessian)

        # Covariance = 2 * Hessian^{-1} for chi-squared
        covariance = 2.0 * hessian_inv

        # Ensure covariance is symmetric (numerical precision)
        covariance = 0.5 * (covariance + covariance.T)

        # Standard deviations from diagonal
        variances = np.diag(covariance)

        # Handle negative variances (should not happen for positive definite)
        if np.any(variances < 0):
            logger.warning(
                "Negative variances encountered. Setting to zero for std_dev calculation."
            )
            variances = np.maximum(variances, 0.0)

        std_devs = np.sqrt(variances)

        logger.debug(
            f"Fisher uncertainty estimation complete. "
            f"Parameter std_devs: {std_devs}"
        )

        return covariance, std_devs

    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to invert Hessian: {e}")
        # Return infinite uncertainties
        covariance = np.full((n_params, n_params), np.inf)
        std_devs = np.full(n_params, np.inf)
        return covariance, std_devs


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
