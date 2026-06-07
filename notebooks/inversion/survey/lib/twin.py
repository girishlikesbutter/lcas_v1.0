"""Body-twin canonicalisation for the IS-901 X-axis symmetry.

The body-twin map ``(q0, ω) → (q_180x ⊗ q0, R_180x · ω)`` is a bit-exact
LC equivalence under the post-fix forward model (s043: max ``|Δ|=2.29e-08``
mag across seeds {23, 28, 89}, six orders of magnitude below the 0.05 mag
photometric noise floor). The ``(q0, ω)`` search space is therefore
exactly 2-to-1 under this map; every search stage that enumerates
ω-direction cells should drop the non-canonical hemisphere for a free
~2× speedup.

Canonical hemisphere convention (this module):

    keep iff  ω_y > 0
         or   (ω_y == 0 and ω_z > 0)
         or   (ω_y == 0 and ω_z == 0 and q_x < 0)

Rationale: the twin map flips ``ω_y, ω_z`` and leaves ``ω_x`` invariant.
Thresholding ω_y first picks one hemisphere of the unit ω-direction
sphere; ω_z is the secondary tie-break for the ω_y == 0 great circle;
the q_x branch handles the measure-zero edge case where ω lies along
body ±X (``twin`` fixes ω there, so the LC distinction is in q0 alone).

Quaternion sign: ``q`` and ``-q`` represent the same rotation, and
``twin(twin(q, ω)) = (-q, ω)`` because ``q_180x ⊗ q_180x = -1``. To
make ``canonical`` idempotent at the array level (not just at the
rotation level), the output quaternion is sign-normalised so that
``q[0] ≥ 0``. After this, ``canonical(canonical(x)) is x``.

References:
- ``concepts/twin_degeneracy.md`` for the algebra.
- ``experiments/s043_twin_hifi_verify.md`` for the hi-fi verification.
- ``experiments/s044_canonical_validation.md`` for the validation of
  this module against scipy and against the surrogate LC.
"""

from __future__ import annotations

import numpy as np

# Body-X 180° rotation as a unit quaternion (w, x, y, z) and as a 3×3 matrix.
Q_180X = np.array([0.0, 1.0, 0.0, 0.0])
R_180X = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def twin(q0: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the body-X 180° twin map: ``(q0, ω) → (q_180x ⊗ q0, R_180x · ω)``."""
    q0 = np.asarray(q0, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    return quat_mul(Q_180X, q0), R_180X @ omega


def is_canonical(q0: np.ndarray, omega: np.ndarray) -> bool:
    """True iff ``(q0, ω)`` lies in the canonical hemisphere."""
    q0 = np.asarray(q0)
    omega = np.asarray(omega)
    if omega[1] > 0.0:
        return True
    if omega[1] < 0.0:
        return False
    if omega[2] > 0.0:
        return True
    if omega[2] < 0.0:
        return False
    return bool(q0[1] < 0.0)


def _sign_normalise(q: np.ndarray) -> np.ndarray:
    """Flip sign of q so that the scalar component is non-negative."""
    return q if q[0] >= 0.0 else -q


def canonical(
    q0: np.ndarray, omega: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical-hemisphere representative.

    For any ``(q0, ω)``, returns the unique element of
    ``{(q0, ω), twin(q0, ω)}`` that lies in the canonical hemisphere,
    with the output quaternion sign-normalised to ``q[0] ≥ 0``.
    Idempotent at the array level: ``canonical(canonical(x))`` returns
    bit-equal arrays.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if is_canonical(q0, omega):
        return _sign_normalise(q0), omega
    q0_t, omega_t = twin(q0, omega)
    return _sign_normalise(q0_t), omega_t


def is_canonical_batch(q0_arr: np.ndarray, omega_arr: np.ndarray) -> np.ndarray:
    """Vectorised canonical-membership test. Returns a boolean array of shape (N,)."""
    q0_arr = np.asarray(q0_arr, dtype=np.float64).reshape(-1, 4)
    omega_arr = np.asarray(omega_arr, dtype=np.float64).reshape(-1, 3)
    return (
        (omega_arr[:, 1] > 0.0)
        | ((omega_arr[:, 1] == 0.0) & (omega_arr[:, 2] > 0.0))
        | (
            (omega_arr[:, 1] == 0.0)
            & (omega_arr[:, 2] == 0.0)
            & (q0_arr[:, 1] < 0.0)
        )
    )


def canonical_batch(
    q0_arr: np.ndarray, omega_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised canonical mapping over (N, 4) and (N, 3) arrays."""
    q0_arr = np.asarray(q0_arr, dtype=np.float64).reshape(-1, 4)
    omega_arr = np.asarray(omega_arr, dtype=np.float64).reshape(-1, 3)
    keep = is_canonical_batch(q0_arr, omega_arr)
    # q_180x ⊗ q for q = (w, x, y, z) gives (-x, w, -z, y) — derived directly
    # from the Hamilton product expansion.
    q0_twin = np.column_stack(
        [-q0_arr[:, 1], q0_arr[:, 0], -q0_arr[:, 3], q0_arr[:, 2]]
    )
    omega_twin = omega_arr * np.array([1.0, -1.0, -1.0])
    q0_canon = np.where(keep[:, None], q0_arr, q0_twin)
    omega_canon = np.where(keep[:, None], omega_arr, omega_twin)
    # Sign-normalise quaternions so q[0] >= 0 — makes canonical idempotent at
    # the array level (not just rotation level).
    flip = q0_canon[:, 0] < 0.0
    q0_canon = np.where(flip[:, None], -q0_canon, q0_canon)
    return q0_canon, omega_canon
