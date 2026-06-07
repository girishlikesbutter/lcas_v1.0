"""Surrogate-model evaluation for the survey.

The v2 residual-ensemble surrogate maps body-frame (k1, k2) → magnitude. It
is bridge-independent — no propagator in its training pipeline — so the
convention bug fix did NOT change its accuracy on body-frame inputs. See
`concepts/surrogate_model.md`.

API mirrors the existing parent-project usage:
    model = SurrogateModel.load_default()
    pred  = model.predict_magnitude(k1, k2, sp_angle_deg, ad_angle_deg, obs_dist_km)

Where sp_angle_deg / ad_angle_deg are the solar-panel and antenna-dish
articulation angles. Survey baseline matches the IS-901 articulation
defaults: SP=0°, AD=15°.
"""

import sys
from pathlib import Path
import numpy as np

SURROGATE_PATH = Path("/home/girish/surrogate_model")
if str(SURROGATE_PATH) not in sys.path:
    sys.path.insert(0, str(SURROGATE_PATH))

from surrogate_model.surrogate import SurrogateModel  # noqa: E402

DEFAULT_SP_ANGLE_DEG = 0.0
DEFAULT_AD_ANGLE_DEG = 15.0
DEFAULT_BRIGHT_THRESHOLD = 11.0  # mag; "bright" subset for bright-MSE

_MODEL = None


def get_model():
    """Cache and return the default surrogate model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SurrogateModel.load_default()
    return _MODEL


def predict(
    k1_body: np.ndarray,
    k2_body: np.ndarray,
    obs_dist_km: np.ndarray,
    sp_angle_deg: float = DEFAULT_SP_ANGLE_DEG,
    ad_angle_deg: float = DEFAULT_AD_ANGLE_DEG,
) -> np.ndarray:
    """Predict per-epoch apparent magnitude.

    Args:
      k1_body: (N, 3) body-frame sun direction, unit vectors.
      k2_body: (N, 3) body-frame observer direction, unit vectors.
      obs_dist_km: (N,) observer distance per epoch.
      sp_angle_deg: solar-panel articulation in degrees.
      ad_angle_deg: antenna-dish articulation in degrees.

    Returns:
      (N,) predicted magnitude.
    """
    return get_model().predict_magnitude(
        k1_body, k2_body, sp_angle_deg, ad_angle_deg, obs_dist_km
    )


def full_lc_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over all valid epochs (mag^2)."""
    mask = np.isfinite(predicted) & np.isfinite(target)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((predicted[mask] - target[mask]) ** 2))


def bright_mse(
    predicted: np.ndarray,
    target: np.ndarray,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
) -> float:
    """MSE over the bright subset of the target LC (target_mag < threshold)."""
    mask = np.isfinite(predicted) & np.isfinite(target) & (target < bright_threshold)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((predicted[mask] - target[mask]) ** 2))


def rho(predicted: np.ndarray, target: np.ndarray) -> float:
    """ρ = √(MSE / 0.05²). See concepts/rho_band.md."""
    mse = full_lc_mse(predicted, target)
    return float(np.sqrt(mse) / 0.05)


# ---------------------------------------------------------------------------
# RF25-inspired NLL residual (opt-in; s078). NOT the production default.
#
# Transplant of Robinson & Frueh 2025 Eq. 3 — a negative-log-likelihood loss
# on flux with per-timestep σ_k weighting + ‖S‖/‖Ŝ‖ signal rescaling. RF25
# argue magnitude-space ℓ2 overfits bright specular glints; the rescaling
# absorbs systematic albedo/area scale errors ("right shape, wrong scale").
#
# Scope note: the m048 cohort uses a fixed apparent-magnitude noise floor
# (0.05 mag), not a per-timestep noise model. Propagated to flux that floor
# becomes a flux-proportional σ_k, which to first order makes the σ-weighted
# χ² ≈ plain magnitude MSE — so the genuine lever here is the ‖S‖/‖Ŝ‖
# rescaling, not the σ weighting. The `rescale` flag lets a caller isolate
# the two effects. A true per-timestep σ_k (RF25 §3.5.1) would need
# lib/noise_rf.py and is deliberately out of scope (would invalidate the
# cohort) — see experiments/s073e_robinson_frueh_2025_full_audit.md.
# ---------------------------------------------------------------------------

_FLUX_LN10_OVER_2P5 = np.log(10.0) / 2.5  # d(flux)/d(mag) factor: |dS/dm| = S · this


def _mag_to_rel_flux(mag: np.ndarray) -> np.ndarray:
    """Relative flux from apparent magnitude (arbitrary zero point; ratios only)."""
    return np.power(10.0, -0.4 * np.asarray(mag, dtype=np.float64))


def nll_residual(
    predicted: np.ndarray,
    target: np.ndarray,
    sigma_mag: float = 0.05,
    rescale: bool = True,
) -> np.ndarray:
    """RF25 Eq. 3 least-squares residual vector (flux space).

    r_k = sqrt(1/2) · (S_k − Ŝ_k) / σ_k   over valid epochs, 0 elsewhere.

    Σ r_k² equals the χ² (data-fit) part of the NLL. The ln σ_k term of the
    full NLL depends only on `target`, so it is constant across candidates
    sharing a truth LC and correctly drops out of a least_squares residual;
    use `nll_cost` for a scalar that includes it.

    Args:
      predicted: (N,) predicted magnitude.
      target:    (N,) measured/truth magnitude.
      sigma_mag: fixed magnitude noise floor (m048 default 0.05).
      rescale:   if True, scale Ŝ to ‖Ŝ‖ = ‖S‖ before the residual (RF25
                 signal rescaling — absorbs global flux-scale error).

    Returns:
      (N,) residual vector; non-finite / masked epochs are 0.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(predicted) & np.isfinite(target)
    out = np.zeros(target.shape, dtype=np.float64)
    if mask.sum() == 0:
        return out

    S = _mag_to_rel_flux(target[mask])
    S_hat = _mag_to_rel_flux(predicted[mask])
    if rescale:
        norm_hat = np.linalg.norm(S_hat)
        if norm_hat == 0.0:
            out[mask] = 1e3
            return out
        S_hat = S_hat * (np.linalg.norm(S) / norm_hat)

    # σ_k from the fixed mag floor propagated to flux (data-dependent only).
    sigma_k = _FLUX_LN10_OVER_2P5 * S * sigma_mag
    sigma_k = np.where(sigma_k > 0.0, sigma_k, np.finfo(np.float64).tiny)

    out[mask] = np.sqrt(0.5) * (S - S_hat) / sigma_k
    return out


def nll_cost(
    predicted: np.ndarray,
    target: np.ndarray,
    sigma_mag: float = 0.05,
    rescale: bool = True,
) -> float:
    """Scalar mean NLL (RF25 Eq. 3) for ranking candidates.

    NLL = (1/m) Σ [ ln σ_k + ½((S_k − Ŝ_k)/σ_k)² ].
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(predicted) & np.isfinite(target)
    if mask.sum() == 0:
        return float("nan")

    S = _mag_to_rel_flux(target[mask])
    S_hat = _mag_to_rel_flux(predicted[mask])
    if rescale:
        norm_hat = np.linalg.norm(S_hat)
        if norm_hat == 0.0:
            return float("inf")
        S_hat = S_hat * (np.linalg.norm(S) / norm_hat)

    sigma_k = _FLUX_LN10_OVER_2P5 * S * sigma_mag
    sigma_k = np.where(sigma_k > 0.0, sigma_k, np.finfo(np.float64).tiny)

    chi2 = 0.5 * ((S - S_hat) / sigma_k) ** 2
    return float(np.mean(np.log(sigma_k) + chi2))
