"""
Uncertainty estimation methods for lightcurve inversion.

This module provides methods for estimating parameter uncertainties
after optimization:

- Fisher Information Matrix: Quick approximation using Hessian
- MCMC sampling: Full posterior characterization using emcee
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

try:
    import emcee
except ImportError:
    emcee = None  # type: ignore

from .objective_function import ObjectiveFunction

logger = logging.getLogger(__name__)


def _compute_hessian(
    func: ObjectiveFunction,
    x: NDArray[np.floating],
    step_size: float = 1e-5,
    show_progress: bool = True,
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
    show_progress : bool, optional
        Whether to print progress during computation. Default is True.

    Returns
    -------
    hessian : ndarray, shape (n, n)
        Symmetric Hessian matrix.
    """
    n = len(x)
    hessian = np.zeros((n, n), dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Total number of Hessian elements to compute
    # n diagonal + n*(n-1)/2 off-diagonal = n*(n+1)/2
    total_elements = n * (n + 1) // 2
    computed_elements = 0

    # Compute diagonal elements using central difference formula:
    # d^2f/dx_i^2 = (f(x+h_i) - 2*f(x) + f(x-h_i)) / h^2
    f_x = func.evaluate(x)

    for i in range(n):
        h_i = np.zeros(n)
        h_i[i] = step_size

        f_plus = func.evaluate(x + h_i)
        f_minus = func.evaluate(x - h_i)

        hessian[i, i] = (f_plus - 2 * f_x + f_minus) / (step_size**2)

        computed_elements += 1
        if show_progress:
            print(f"    Computing Hessian elements: {computed_elements}/{total_elements}", end='\r', flush=True)

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

            computed_elements += 1
            if show_progress:
                print(f"    Computing Hessian elements: {computed_elements}/{total_elements}", end='\r', flush=True)

    if show_progress:
        print(f"    Computing Hessian elements: {total_elements}/{total_elements}", flush=True)

    return hessian


def compute_fisher_uncertainty(
    objective: ObjectiveFunction,
    optimal_params: NDArray[np.floating],
    step_size: float = 1e-5,
    show_progress: bool = True,
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
    show_progress : bool, optional
        Whether to print progress during computation. Default is True.

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

    if show_progress:
        print(f"\nComputing Fisher uncertainty (Hessian)...", flush=True)

    logger.debug(f"Computing Hessian at optimum with step_size={step_size}")

    # Compute Hessian at optimum
    hessian = _compute_hessian(objective, optimal_params, step_size, show_progress=show_progress)

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

        if show_progress:
            print(f"Fisher uncertainty complete.", flush=True)

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
    bounds: Optional[List[Tuple[float, float]]] = None,
    initial_scatter: float = 0.01,
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
    bounds : list of (min, max) tuples, optional
        Parameter bounds for prior. If None, uses defaults:
        axis_angle: [-pi, pi], omega: [-0.5236, 0.5236] (~30 deg/s).
    initial_scatter : float, optional
        Fractional scatter for initializing walkers around optimal_params.
        Default is 0.01 (1% scatter).

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
            Returns NaN if autocorrelation estimation fails.
        - 'converged': bool
            True if chain appears converged (n_samples > 50 * autocorr_time).

    Notes
    -----
    Uses a log-posterior formulation with flat priors within bounds.
    The log-likelihood is -0.5 * chi_squared from the objective function.

    Convergence is assessed using the integrated autocorrelation time.
    A chain is considered converged if it has at least 50 times the
    autocorrelation time of samples.
    """
    if emcee is None:
        raise ImportError(
            "emcee is required for MCMC uncertainty estimation. "
            "Install it with: pip install emcee"
        )

    optimal_params = np.asarray(optimal_params, dtype=np.float64)
    n_params = len(optimal_params)

    # Default number of walkers (emcee requires at least 2 * n_params)
    if n_walkers is None:
        n_walkers = 2 * n_params

    if n_walkers < 2 * n_params:
        logger.warning(
            f"n_walkers ({n_walkers}) should be >= 2 * n_params ({2 * n_params}). "
            "Increasing to minimum required."
        )
        n_walkers = 2 * n_params

    # Default bounds
    if bounds is None:
        # axis_angle: [-pi, pi], omega: [-30 deg/s, 30 deg/s] in rad/s
        omega_max = 0.5236  # ~30 deg/s
        bounds = [(-np.pi, np.pi)] * 3 + [(-omega_max, omega_max)] * 3

    bounds_array = np.array(bounds)
    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]

    # Log-prior: flat within bounds, -inf outside
    def log_prior(params: NDArray[np.floating]) -> float:
        if np.any(params < lower_bounds) or np.any(params > upper_bounds):
            return -np.inf
        return 0.0

    # Log-likelihood: -0.5 * chi_squared
    def log_likelihood(params: NDArray[np.floating]) -> float:
        chi_sq = objective.evaluate(params)
        if not np.isfinite(chi_sq):
            return -np.inf
        return -0.5 * chi_sq

    # Log-posterior: prior + likelihood
    def log_posterior(params: NDArray[np.floating]) -> float:
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)

    # Initialize walkers around optimal parameters with small scatter
    # Ensure initial positions are within bounds
    rng = np.random.default_rng(seed=42)
    initial_positions = np.empty((n_walkers, n_params))

    for i in range(n_walkers):
        for j in range(n_params):
            # Scale scatter based on parameter range
            param_range = upper_bounds[j] - lower_bounds[j]
            scatter = initial_scatter * param_range

            # Generate position with scatter, clip to bounds
            pos = optimal_params[j] + rng.uniform(-scatter, scatter)
            pos = np.clip(pos, lower_bounds[j], upper_bounds[j])
            initial_positions[i, j] = pos

    logger.info(
        f"Starting MCMC with {n_walkers} walkers, "
        f"{burn_in} burn-in steps, {n_samples} samples"
    )

    # Create sampler and run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior)

    # Run burn-in
    logger.debug("Running burn-in...")
    state = sampler.run_mcmc(initial_positions, burn_in, progress=False)
    sampler.reset()

    # Run production chain
    logger.debug("Running production chain...")
    sampler.run_mcmc(state, n_samples, progress=False)

    # Get flattened samples
    samples = sampler.get_chain(flat=True)

    # Compute statistics from samples
    covariance = np.cov(samples, rowvar=False)
    std_devs = np.std(samples, axis=0)

    # Mean acceptance fraction
    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))

    # Compute autocorrelation time for convergence diagnostic
    try:
        autocorr_time = sampler.get_autocorr_time(quiet=True)
        # Check convergence: need at least 50 * tau samples
        max_tau = np.max(autocorr_time)
        converged = n_samples > 50 * max_tau
    except emcee.autocorr.AutocorrError:
        logger.warning(
            "Could not estimate autocorrelation time. "
            "Chain may not be converged. Consider increasing n_samples."
        )
        autocorr_time = np.full(n_params, np.nan)
        converged = False

    logger.info(
        f"MCMC complete. Acceptance fraction: {acceptance_fraction:.3f}, "
        f"Converged: {converged}"
    )

    return {
        "samples": samples,
        "covariance": covariance,
        "std_devs": std_devs,
        "acceptance_fraction": np.array(acceptance_fraction),
        "autocorr_time": autocorr_time,
        "converged": np.array(converged),
    }
