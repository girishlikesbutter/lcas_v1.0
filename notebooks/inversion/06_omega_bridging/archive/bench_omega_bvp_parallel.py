"""
Benchmark: recover omega from n_c^2 candidate pairings.

n_c candidate quaternions at epoch 1, n_c at epoch 2.
All n_c^2 combinations solved in parallel with Pool(8).
"""
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader


# ── Helpers ──────────────────────────────────────────────────────────
def quat_dot(q1, q2):
    return abs(np.dot(q1, q2))


def quat_geodesic(q1, q2):
    return 1.0 - quat_dot(q1, q2)


def quat_angle_deg(q1, q2):
    d = np.clip(quat_dot(q1, q2), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def principal_axis_guess(q1, q2, dt):
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    dr = r1.inv() * r2
    return dr.as_rotvec() / dt


def random_quaternions(n, rng):
    """Generate n uniformly random unit quaternions (w,x,y,z)."""
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Ensure w > 0 for canonical form
    q[q[:, 0] < 0] *= -1
    return q


def solve_one_pair(args):
    """Solve a single omega BVP."""
    q1, q2, dt, I = args
    times = np.array([0.0, dt])

    def objective(omega_vec):
        quats_trial, _ = propagate_attitude(
            q1, omega_vec, times, mode='tumbling', inertia_tensor=I)
        return quat_geodesic(quats_trial[1], q2)

    omega_guess = principal_axis_guess(q1, q2, dt)
    result = minimize(objective, omega_guess, method='L-BFGS-B',
                      options={'gtol': 1e-15, 'ftol': 1e-15})

    q_final, _ = propagate_attitude(q1, result.x, times, mode='tumbling', inertia_tensor=I)
    angle_err = quat_angle_deg(q_final[1], q2)
    return {
        'omega': result.x,
        'angle_err_deg': angle_err,
        'nfev': result.nfev,
        'success': result.success,
        'cost': result.fun,
    }


# ── Main ─────────────────────────────────────────────────────────────
def main():
    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
    component_masses = {
        'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0,
        'AD_East': 50.0, 'AD_West': 50.0,
    }
    inertia_result = compute_inertia_from_config(
        config=config, config_manager=config_manager,
        masses=component_masses,
        articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
    I = inertia_result.inertia_tensor

    dt = 50.0
    N_C = 50  # candidates per epoch → 2500 pairs
    rng = np.random.default_rng(42)

    # Generate n_c random candidate quaternions at each epoch
    candidates_e1 = random_quaternions(N_C, rng)
    candidates_e2 = random_quaternions(N_C, rng)

    # All n_c^2 pairings
    pair_args = []
    for i in range(N_C):
        for j in range(N_C):
            pair_args.append((candidates_e1[i], candidates_e2[j], dt, I))

    n_pairs = len(pair_args)
    print(f"n_c = {N_C} candidates per epoch")
    print(f"Total pairs: {N_C}^2 = {n_pairs}")
    print(f"dt = {dt}s")
    print()

    # ── Parallel (8 workers) ─────────────────────────────────────────
    t0 = time.perf_counter()
    with Pool(8) as pool:
        results = pool.map(solve_one_pair, pair_args)
    wall_par = (time.perf_counter() - t0) * 1000

    # ── Report ───────────────────────────────────────────────────────
    angle_errs = [r['angle_err_deg'] for r in results]
    nfevs = [r['nfev'] for r in results]
    costs = [r['cost'] for r in results]
    all_ok = all(r['success'] for r in results)

    print(f"All converged: {all_ok}")
    print(f"q2 arrival error (deg): max={max(angle_errs):.2e}  mean={np.mean(angle_errs):.2e}")
    print(f"Function evals:         max={max(nfevs)}  mean={np.mean(nfevs):.0f}  total={sum(nfevs)}")
    print(f"Final cost:             max={max(costs):.2e}  mean={np.mean(costs):.2e}")
    print(f"\nParallel (8 cores): {wall_par/1000:.1f} s  ({wall_par/n_pairs:.1f} ms/pair)")


if __name__ == '__main__':
    main()
