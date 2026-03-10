"""
Benchmark: omega recovery with noisy quaternion endpoints.

Ground truth: propagate q1 by omega over 50s to get q2.
Then perturb both q1 and q2 by a known angular error before
attempting to recover omega. Sweep error from 0.5 to 10 degrees.
"""
import sys
import time
import numpy as np
from pathlib import Path
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


def perturb_quaternion(q_wxyz, angle_deg, rng):
    """Apply a random rotation of exactly angle_deg to a quaternion."""
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    dq = np.array([np.cos(half), *(np.sin(half) * axis)])
    # Hamilton product
    w1, x1, y1, z1 = q_wxyz
    w2, x2, y2, z2 = dq
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return result / np.linalg.norm(result)


# ── Setup ────────────────────────────────────────────────────────────
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager,
    masses={'Bus': 1532, 'SP_North': 170, 'SP_South': 170,
            'AD_East': 50, 'AD_West': 50},
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
I = inertia_result.inertia_tensor

# Ground truth
q1_true = np.array([0.9238795, 0.2209424, 0.1104712, 0.2946566])
q1_true /= np.linalg.norm(q1_true)
omega_true = np.deg2rad(np.array([0.5, -0.3, 2.0]))
dt = 50.0
times = np.array([0.0, dt])

quats_true, _ = propagate_attitude(q1_true, omega_true, times,
                                   mode='tumbling', inertia_tensor=I)
q2_true = quats_true[1]

# ── Sweep ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
errors_deg = np.arange(0.5, 10.5, 0.5)
N_TRIALS = 10  # average over random perturbation directions

print(f"{'Err (deg)':>9}  {'Time (ms)':>9}  {'nfev':>5}  "
      f"{'omega err (deg/s)':>18}  {'cost':>10}")
print("-" * 60)

for err_deg in errors_deg:
    trial_times = []
    trial_nfevs = []
    trial_omega_errs = []
    trial_costs = []

    for _ in range(N_TRIALS):
        q1_noisy = perturb_quaternion(q1_true, err_deg, rng)
        q2_noisy = perturb_quaternion(q2_true, err_deg, rng)

        guess = principal_axis_guess(q1_noisy, q2_noisy, dt)

        def objective(omega_vec, _q1=q1_noisy, _q2=q2_noisy):
            q, _ = propagate_attitude(_q1, omega_vec, times,
                                      mode='tumbling', inertia_tensor=I)
            return quat_geodesic(q[1], _q2)

        t0 = time.perf_counter()
        result = minimize(objective, guess, method='L-BFGS-B',
                          options={'gtol': 1e-15, 'ftol': 1e-15})
        trial_times.append((time.perf_counter() - t0) * 1000)
        trial_nfevs.append(result.nfev)
        trial_omega_errs.append(
            np.degrees(np.linalg.norm(result.x - omega_true)))
        trial_costs.append(result.fun)

    print(f"{err_deg:>9.1f}  {np.mean(trial_times):>9.0f}  "
          f"{np.mean(trial_nfevs):>5.0f}  "
          f"{np.mean(trial_omega_errs):>18.4f}  "
          f"{np.mean(trial_costs):>10.2e}")
