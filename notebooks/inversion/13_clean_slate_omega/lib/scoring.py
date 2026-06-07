"""Shared residual/error metrics and classification rubric for 13_clean_slate_omega.

The "multi-solution" philosophy (memory `feedback_multi_solution.md`) dictates
that every candidate under a residual threshold is kept. Thresholds here are
the single place that choice is encoded.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# v2 surrogate MAE on hi-fi truth is ~0.01 mag overall (memory). Squared that
# is ~1e-4 mag^2. A residual threshold of 3x the noise floor gives us the
# "acceptable candidates" gate. Keep these constants centralised so all three
# sub-experiments use the same gate.
V2_TRUTH_MSE_FLOOR = 0.0024     # mag^2, per project_surrogate_model.md m130 findings
RESIDUAL_MSE_GATE  = 0.02       # mag^2, ~3x noise floor — "pass" if surrogate-vs-truth MSE below this
RESIDUAL_MSE_TIGHT = 0.005      # mag^2, tight gate — "highly likely correct"


def lc_mse(pred: np.ndarray, obs: np.ndarray) -> float:
    """MSE in magnitude space between predicted and observed light curves."""
    return float(np.mean((pred - obs) ** 2))


def omega_errors(omega_est: np.ndarray, omega_true: np.ndarray) -> Dict[str, float]:
    """Direction error (deg) and magnitude relative error (%) for ω.

    Direction error is the angle between unit vectors. Magnitude error is
    `(|ω_est| - |ω_true|) / |ω_true| * 100`.
    """
    m_est = float(np.linalg.norm(omega_est))
    m_true = float(np.linalg.norm(omega_true))
    if m_est < 1e-12 or m_true < 1e-12:
        return {"dir_deg": np.nan, "mag_pct": np.nan, "mag_dps_est": np.degrees(m_est), "mag_dps_true": np.degrees(m_true)}
    cos_a = float(np.clip(np.dot(omega_est / m_est, omega_true / m_true), -1.0, 1.0))
    dir_deg = float(np.degrees(np.arccos(cos_a)))
    mag_pct = (m_est - m_true) / m_true * 100.0
    return {
        "dir_deg": dir_deg,
        "mag_pct": mag_pct,
        "mag_dps_est": np.degrees(m_est),
        "mag_dps_true": np.degrees(m_true),
    }


def quat_geodesic_deg(q_est: np.ndarray, q_true: np.ndarray) -> float:
    """Geodesic angle between two scalar-first quaternions (deg).

    Invariant under global sign flip q ↔ −q.
    """
    dp = float(np.abs(np.dot(q_est / (np.linalg.norm(q_est) + 1e-30),
                             q_true / (np.linalg.norm(q_true) + 1e-30))))
    dp = min(1.0, dp)
    return float(np.degrees(2.0 * np.arccos(dp)))


def classify(mse: float) -> str:
    """OK / PARTIAL / FAIL label based on the shared rubric.

    Aligned with the existing invert.py / wrappedbest rubric used across m046/m048.
    """
    if mse <= RESIDUAL_MSE_TIGHT:
        return "OK"
    if mse <= RESIDUAL_MSE_GATE:
        return "PARTIAL"
    return "FAIL"
