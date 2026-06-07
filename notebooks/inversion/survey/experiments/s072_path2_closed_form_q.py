"""s072 — Path 2 closed-form q(t) validation under post-fix textbook convention.

Three gating tests over 60-min LCs:

  Gate 1: m048-diagonal inertia (seeds 89/28/14)
      - q_path2 vs q_DOP853 (fresh propagate_attitude reference)
      - q_path2 vs cached truth NPZ quaternions
      - 2T and L^2 conservation under Path 2's algebraic theta/psi + ODE phi
      - k1_body(q_path2) vs cached k1_body

  Gate 2: perturbed inertia (m048 + off-diagonal terms; same seed 89 IC)
      - Tests eigendecomposition precision: R_pa is no longer a permutation.
      - q_path2 vs q_DOP853 (no cached truth — synthetic reference)
      - 2T and L^2 conservation under Path 2.

  Gate 3: asymmetric satellite (deliberately non-diagonal I, two regimes)
      - I = symm 3x3 with off-diagonals; synthetic q0, omega0.
      - One case in regime A (rotation about smallest moment),
        one in regime B (rotation about largest moment).
      - q_path2 vs q_DOP853; conservation under Path 2.

All comparisons are antipode-aware: ||q_a - q_b||_inf or ||q_a + q_b||_inf,
whichever is smaller (q and -q represent the same rotation).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Single-thread BLAS — single-process script, no Pool, but cheap insurance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SURVEY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from lib.jacobi_propagator import propagate_jacobi_path2  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402


RESULTS_DIR = SURVEY_ROOT / "results" / "s072"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _antipode_inf(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """max_i min(||q_a[i] - q_b[i]||, ||q_a[i] + q_b[i]||).

    The antipode flip can occur at different epochs, so take the per-epoch
    minimum first, then the max — NOT max(min_a, min_b) over the whole array.
    """
    diff_pos = np.linalg.norm(q_a - q_b, axis=1)
    diff_neg = np.linalg.norm(q_a + q_b, axis=1)
    return float(np.minimum(diff_pos, diff_neg).max())


def _geodesic_deg_max(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Antipode-aware geodesic angle (deg), max over epochs."""
    dots = np.clip(np.abs(np.einsum("ni,ni->n", q_a, q_b)), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dots)).max())


def _conservation_2T_L2(omega_hist: np.ndarray, I: np.ndarray) -> tuple[float, float]:
    """Relative std of 2T and L^2 across the trajectory.

    2T = omega · (I omega) [scalar]
    L^2 = |I omega|^2     [scalar]
    """
    Iom = omega_hist @ I.T  # body-frame; I is symmetric so I.T == I
    twoT = np.einsum("ni,ni->n", omega_hist, Iom)
    L2 = np.einsum("ni,ni->n", Iom, Iom)
    rel_2T = float(np.std(twoT) / np.abs(np.mean(twoT)))
    rel_L2 = float(np.std(L2) / np.abs(np.mean(L2)))
    return rel_2T, rel_L2


def _k1_body_from_quats(
    q_hist: np.ndarray, sun_pos: np.ndarray, sat_pos: np.ndarray
) -> np.ndarray:
    """Mirror lib/forward.py: R(q) = scipy.from_quat([qx,qy,qz,qw]).as_matrix();
    k1_body = R @ (sun_pos - sat_pos)/||...||.
    """
    sun_vec = sun_pos - sat_pos
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    N = q_hist.shape[0]
    k1_body = np.empty((N, 3))
    for i in range(N):
        q = q_hist[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        k1_body[i] = R @ sun_unit[i]
    return k1_body


# ---------------------------------------------------------------------------
# Gate 1 — m048-diagonal inertia, three cohort seeds
# ---------------------------------------------------------------------------


def gate_1_m048_diagonal() -> dict:
    """Three seeds spanning the polhode regime distribution.

    Seed 89 — slow tumbler (|ω| = 0.24 dps).
    Seed 28 — fast tumbler (|ω| = 1.44 dps).
    Seed 14 — near-separatrix.
    """
    out = {"description": "Gate 1 — m048-diagonal inertia (cached truth)", "seeds": {}}
    for seed in (89, 28, 14):
        ctx = build_context(seed)
        I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)
        truth = load_truth(seed)
        q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
        omega0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
        times = np.asarray(truth["observation_times"], dtype=np.float64)
        q_cached = np.asarray(truth["quaternions"], dtype=np.float64)
        k1_cached = np.asarray(truth["k1_body"], dtype=np.float64)
        sun_pos = np.asarray(truth["sun_pos"], dtype=np.float64)
        sat_pos = np.asarray(truth["sat_pos"], dtype=np.float64)

        t0 = time.perf_counter()
        q_dop, om_dop = propagate_attitude(q0, omega0, times, "tumbling", I,
                                           rtol=1e-12, atol=1e-14)
        wall_dop = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_p2, om_p2 = propagate_jacobi_path2(q0, omega0, I, times)
        wall_p2 = time.perf_counter() - t0

        gate_1a = _antipode_inf(q_p2, q_dop)
        gate_1a_geo = _geodesic_deg_max(q_p2, q_dop)
        gate_1a_cache = _antipode_inf(q_p2, q_cached)
        rel_2T, rel_L2 = _conservation_2T_L2(om_p2, I)

        k1_p2 = _k1_body_from_quats(q_p2, sun_pos, sat_pos)
        gate_1c = float(np.linalg.norm(k1_p2 - k1_cached, axis=1).max())

        # Also: inertia diagonality + R_pa info
        from lib.jacobi_propagator import _eigendecompose_inertia
        I_pa, R_pa = _eigendecompose_inertia(I)
        is_diag = bool(
            np.allclose(I, np.diag(np.diag(I)), atol=1e-12)
        )
        R_pa_is_perm = bool(
            np.allclose(np.abs(R_pa), (np.abs(R_pa) > 0.5).astype(float), atol=1e-10)
        )

        # Tag DOP853 self-noise: fresh vs cached on the same propagator. If the
        # cached NPZ was generated under the same DOP853 settings this should
        # be the integration noise floor itself.
        dop_vs_cache = _antipode_inf(q_dop, q_cached)

        out["seeds"][f"seed_{seed:03d}"] = {
            "omega_mag_dps": float(truth["omega_mag_dps"]),
            "I_diag": [float(I[0, 0]), float(I[1, 1]), float(I[2, 2])],
            "is_diagonal": is_diag,
            "I_pa_ascending": [float(x) for x in I_pa],
            "R_pa_is_permutation": R_pa_is_perm,
            "wall_path2_s": wall_p2,
            "wall_dop853_s": wall_dop,
            "speedup": wall_dop / max(wall_p2, 1e-12),
            "gate_1a_q_vs_dop853_inf": gate_1a,
            "gate_1a_q_vs_dop853_geo_deg_max": gate_1a_geo,
            "gate_1a_q_vs_cached_inf": gate_1a_cache,
            "dop853_vs_cached_inf": dop_vs_cache,
            "gate_1b_2T_relstd": rel_2T,
            "gate_1b_L2_relstd": rel_L2,
            "gate_1c_k1_body_inf": gate_1c,
        }
        print(f"  seed {seed}: gate_1a (vs DOP853) = {gate_1a:.2e}  "
              f"gate_1a_geo_max = {gate_1a_geo:.2e} deg  "
              f"2T_relstd = {rel_2T:.2e}  L2_relstd = {rel_L2:.2e}  "
              f"k1_body_inf = {gate_1c:.2e}  "
              f"speedup ≈ {wall_dop / max(wall_p2, 1e-12):.1f}x")
    return out


# ---------------------------------------------------------------------------
# Gate 2 — perturbed inertia (seed 89 IC + off-diagonal perturbation)
# ---------------------------------------------------------------------------


def gate_2_perturbed_inertia() -> dict:
    """m048 inertia + symmetric off-diagonal perturbation. Tests R_pa non-permutation.

    Two perturbation magnitudes: small (1% of trace) and moderate (10% of trace).
    Uses seed 89's q0, omega0, observation_times.
    """
    truth = load_truth(89)
    q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    omega0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
    times = np.asarray(truth["observation_times"], dtype=np.float64)
    ctx = build_context(89)
    I_diag = np.asarray(ctx["inertia_tensor"], dtype=np.float64)

    out = {"description": "Gate 2 — perturbed m048 inertia (synthetic ref via DOP853)",
           "cases": {}}
    trace = float(np.trace(I_diag))
    for label, eps_frac in (("perturb_1pct", 0.01), ("perturb_10pct", 0.10)):
        # Symmetric off-diagonal perturbation in body frame.
        # Choose three distinct off-diagonals so eigenvectors aren't trivial.
        off = trace * eps_frac
        delta = np.array([
            [0.0, 0.6 * off, 0.2 * off],
            [0.6 * off, 0.0, 0.4 * off],
            [0.2 * off, 0.4 * off, 0.0],
        ])
        I_pert = I_diag + delta
        # Must remain positive-definite.
        I_eig = np.linalg.eigvalsh(I_pert)
        assert (I_eig > 0).all(), f"{label}: I lost positive definiteness {I_eig}"

        t0 = time.perf_counter()
        q_dop, om_dop = propagate_attitude(q0, omega0, times, "tumbling", I_pert,
                                           rtol=1e-12, atol=1e-14)
        wall_dop = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_p2, om_p2 = propagate_jacobi_path2(q0, omega0, I_pert, times)
        wall_p2 = time.perf_counter() - t0

        gate_a = _antipode_inf(q_p2, q_dop)
        gate_a_geo = _geodesic_deg_max(q_p2, q_dop)
        rel_2T, rel_L2 = _conservation_2T_L2(om_p2, I_pert)

        from lib.jacobi_propagator import _eigendecompose_inertia
        I_pa, R_pa = _eigendecompose_inertia(I_pert)
        # Distance of R_pa from identity (Frobenius)
        R_pa_offax = float(np.linalg.norm(R_pa - np.eye(3)))

        out["cases"][label] = {
            "eps_frac_of_trace": eps_frac,
            "I_eigenvalues": [float(x) for x in I_eig],
            "I_pa_ascending": [float(x) for x in I_pa],
            "R_pa_from_identity_frob": R_pa_offax,
            "wall_path2_s": wall_p2,
            "wall_dop853_s": wall_dop,
            "gate_a_q_vs_dop853_inf": gate_a,
            "gate_a_q_vs_dop853_geo_deg_max": gate_a_geo,
            "gate_b_2T_relstd": rel_2T,
            "gate_b_L2_relstd": rel_L2,
        }
        print(f"  {label}: gate_a = {gate_a:.2e}  "
              f"geo_max = {gate_a_geo:.2e} deg  "
              f"2T_relstd = {rel_2T:.2e}  L2_relstd = {rel_L2:.2e}  "
              f"R_pa_off-id = {R_pa_offax:.3e}")
    return out


# ---------------------------------------------------------------------------
# Gate 3 — asymmetric satellite (synthetic I and (q0, omega0))
# ---------------------------------------------------------------------------


def gate_3_asymmetric_satellite() -> dict:
    """Wholly synthetic inertia + initial state. Two configs covering A and B."""
    # Synthetic asymmetric inertia. Eigenvalues well-separated; non-diagonal.
    I_async = np.array([
        [2.0, 0.4, 0.15],
        [0.4, 3.5, -0.25],
        [0.15, -0.25, 5.0],
    ])
    assert np.allclose(I_async, I_async.T), "inertia must be symmetric"
    I_eig = np.linalg.eigvalsh(I_async)
    assert (I_eig > 0).all(), "I must be positive definite"

    # Same 60-min LC as the m048 cohort: 500 epochs uniform over [0, 3600].
    times = np.linspace(0.0, 3600.0, 500)

    # Two regimes. Use PA-frame omega and rotate to body via R_pa^T to make
    # the regime classification deterministic.
    from lib.jacobi_propagator import _eigendecompose_inertia
    I_pa, R_pa = _eigendecompose_inertia(I_async)
    I_1, I_2, I_3 = I_pa  # ascending

    out = {"description": "Gate 3 — asymmetric satellite (synthetic q0, omega)",
           "I_async": I_async.tolist(),
           "I_pa_ascending": [float(x) for x in I_pa],
           "cases": {}}

    # Initial quaternion: 45 deg about [1,2,3]/||...||.
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(22.5)
    q0 = np.array([np.cos(half), np.sin(half) * axis[0],
                   np.sin(half) * axis[1], np.sin(half) * axis[2]])

    # Regime A: rotation about smallest moment I_1. Pick omega_pa such that
    # 2T·I_2 > L^2: dominant component on I_1.
    omega_pa_A = np.array([0.05, 0.012, 0.008])  # rad/s. Dominant I_1.
    twoT_A = (I_pa * omega_pa_A**2).sum()
    L2_A = ((I_pa * omega_pa_A)**2).sum()
    assert twoT_A * I_2 - L2_A > 0, "regime A predicate failed"
    omega_body_A = R_pa @ omega_pa_A

    # Regime B: dominant on I_3.
    omega_pa_B = np.array([0.008, 0.012, 0.05])  # rad/s. Dominant I_3.
    twoT_B = (I_pa * omega_pa_B**2).sum()
    L2_B = ((I_pa * omega_pa_B)**2).sum()
    assert twoT_B * I_2 - L2_B < 0, "regime B predicate failed"
    omega_body_B = R_pa @ omega_pa_B

    for label, omega0, regime_predicted in (
        ("regime_A_dominant_I1", omega_body_A, "A"),
        ("regime_B_dominant_I3", omega_body_B, "B"),
    ):
        t0 = time.perf_counter()
        q_dop, om_dop = propagate_attitude(q0, omega0, times, "tumbling", I_async,
                                           rtol=1e-12, atol=1e-14)
        wall_dop = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_p2, om_p2 = propagate_jacobi_path2(q0, omega0, I_async, times)
        wall_p2 = time.perf_counter() - t0

        gate_a = _antipode_inf(q_p2, q_dop)
        gate_a_geo = _geodesic_deg_max(q_p2, q_dop)
        rel_2T, rel_L2 = _conservation_2T_L2(om_p2, I_async)

        # Regime detection in Path 2.
        from lib.jacobi_propagator import _build_omega_func
        omega0_pa = R_pa.T @ omega0
        _, info = _build_omega_func(I_pa, omega0_pa)
        regime_actual = info["regime"]
        assert regime_actual == regime_predicted, \
            f"{label}: regime mismatch predicted={regime_predicted} actual={regime_actual}"

        out["cases"][label] = {
            "regime": regime_actual,
            "omega_mag_dps": float(np.degrees(np.linalg.norm(omega0))),
            "wall_path2_s": wall_p2,
            "wall_dop853_s": wall_dop,
            "gate_a_q_vs_dop853_inf": gate_a,
            "gate_a_q_vs_dop853_geo_deg_max": gate_a_geo,
            "gate_b_2T_relstd": rel_2T,
            "gate_b_L2_relstd": rel_L2,
        }
        print(f"  {label} ({regime_actual}): gate_a = {gate_a:.2e}  "
              f"geo_max = {gate_a_geo:.2e} deg  "
              f"2T_relstd = {rel_2T:.2e}  L2_relstd = {rel_L2:.2e}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== s072 Gate 1: m048-diagonal inertia ===")
    g1 = gate_1_m048_diagonal()
    print()
    print("=== s072 Gate 2: perturbed inertia ===")
    g2 = gate_2_perturbed_inertia()
    print()
    print("=== s072 Gate 3: asymmetric satellite ===")
    g3 = gate_3_asymmetric_satellite()

    summary = {
        "experiment": "s072",
        "title": "Path 2 closed-form q(t) under post-fix convention — 3-gate validation",
        "gates": {
            "gate_1_m048_diagonal": g1,
            "gate_2_perturbed_inertia": g2,
            "gate_3_asymmetric_satellite": g3,
        },
        "thresholds": {
            "q_vs_dop853_inf": 1e-9,
            "q_vs_dop853_inf_relaxed": 1e-7,
            "conservation_relstd": 1e-12,
            "k1_body_inf": 1e-9,
        },
    }
    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
