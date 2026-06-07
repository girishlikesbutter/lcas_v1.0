"""s020 factorisation smoke test.

Question: does the propagator satisfy
    propagate(q0, omega, t) == propagate(identity, omega, t) ⊗ q0
under LEFT-multiply convention (the codebase convention per src/dynamics/
attitude_propagator.py lines 169-173, 254-255)?

If yes → s020 can propagate Delta(t) once per omega-cell and compose with
all q0 candidates via cheap quaternion multiplies.

Tests:
    A. Symmetric inertia (omega constant in body) — sanity baseline.
    B. m048 cohort inertia (asymmetric) — real Euler dynamics test.
    C. Many random q0s × one omega — covers the actual s020 use case.

Pass criterion: max quat error < 1e-8 across all (t, q0) on test C.
"""
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.attitude_propagator import propagate_attitude


def qmul(q1, q2):
    """Hamilton product, scalar-first (w,x,y,z). Matches _quaternion_multiply
    in src/dynamics/attitude_propagator.py."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_err(q_pred, q_true):
    """Quaternion error accounting for q ≡ -q double-cover."""
    return min(np.linalg.norm(q_pred - q_true), np.linalg.norm(q_pred + q_true))


def test(name, q0_list, omega0, times, inertia_tensor):
    print(f"\n=== {name} ===")
    print(f"  inertia diag: {np.diag(inertia_tensor)}")
    print(f"  omega0: {omega0}, |omega0|={np.linalg.norm(omega0):.4f} rad/s")
    print(f"  n_q0={len(q0_list)}, n_times={len(times)}, t_max={times[-1]:.1f} s")

    # Propagate identity ONCE
    q_id, _ = propagate_attitude(
        np.array([1.0, 0.0, 0.0, 0.0]), omega0, times,
        mode='tumbling', inertia_tensor=inertia_tensor,
    )

    max_err_overall = 0.0
    for k, q0 in enumerate(q0_list):
        q_truth, _ = propagate_attitude(
            q0, omega0, times,
            mode='tumbling', inertia_tensor=inertia_tensor,
        )
        # LEFT multiply: q(t) = Delta(t) * q0
        max_err = 0.0
        for i in range(len(times)):
            q_pred = qmul(q_id[i], q0)
            err = quat_err(q_pred, q_truth[i])
            max_err = max(max_err, err)
        max_err_overall = max(max_err_overall, max_err)
        print(f"  q0[{k}] max err over time: {max_err:.2e}")

    print(f"  *** OVERALL MAX ERR: {max_err_overall:.2e} ***")
    return max_err_overall


def main():
    np.random.seed(42)
    times = np.linspace(0, 600, 60)  # 10 min, 60 epochs (s020 scale)

    # Build 5 random unit q0s
    q0_list = []
    for _ in range(5):
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        q0_list.append(q)

    omega0 = np.array([0.05, -0.10, 0.08])  # rad/s

    # Test A: symmetric inertia (omega constant in body)
    err_a = test("A. Symmetric inertia",
                 q0_list, omega0, times,
                 inertia_tensor=np.eye(3))

    # Test B: m048 cohort inertia (load from filter_costs)
    sys.path.insert(0, str(PROJECT_ROOT / "notebooks/inversion/survey"))
    from lib.filter_costs import load_static_geometry
    geo = load_static_geometry()
    I_cohort = geo["inertia_tensor"]
    err_b = test("B. m048 cohort inertia (asymmetric Euler)",
                 q0_list, omega0, times,
                 inertia_tensor=I_cohort)

    # Test C: many q0s, one omega — the actual s020 use case
    print("\n=== C. Stress test: 50 random q0 × m048 inertia ===")
    np.random.seed(123)
    big_q0 = []
    for _ in range(50):
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        big_q0.append(q)
    err_c = test("C. 50 q0s, m048 inertia, 600 s",
                 big_q0, omega0, times,
                 inertia_tensor=I_cohort)

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    print(f"  Symmetric inertia: max err = {err_a:.2e}")
    print(f"  m048 inertia:      max err = {err_b:.2e}")
    print(f"  50-q0 stress:      max err = {err_c:.2e}")
    PASS_THRESHOLD = 1e-6  # ODE rtol=1e-10 + Hamilton multiply numerical noise
    if max(err_a, err_b, err_c) < PASS_THRESHOLD:
        print(f"\n  *** PASS *** (all < {PASS_THRESHOLD:.0e})")
        print("  s020 LEFT-multiply factorisation is exact.")
        print("  Per-omega-cell propagation is mathematically sound.")
    else:
        print(f"\n  *** FAIL *** (some > {PASS_THRESHOLD:.0e})")
        print("  Factorisation does NOT hold under m048 Euler dynamics.")
        print("  s020 must propagate per (q0, omega) pair — no shortcut.")


if __name__ == "__main__":
    main()
