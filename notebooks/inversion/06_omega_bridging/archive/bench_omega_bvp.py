"""
Benchmark: recover omega from two quaternion endpoints.

Given q1 at t=0 and q2 at t=dt (from known tumbling propagation),
treat omega as unknown and minimize arrival error to recover it.

Measures: wall time, function evaluations, angular error, omega error.
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

# ── Load inertia tensor ──────────────────────────────────────────────
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
print(f"Inertia tensor diagonal: [{I[0,0]:.0f}, {I[1,1]:.0f}, {I[2,2]:.0f}]")


# ── Helpers ──────────────────────────────────────────────────────────
def quat_dot(q1, q2):
    """Absolute dot product (handles double cover)."""
    return abs(np.dot(q1, q2))


def quat_geodesic(q1, q2):
    """Geodesic distance: 1 - |q1 · q2|.  Zero means identical."""
    return 1.0 - quat_dot(q1, q2)


def quat_angle_deg(q1, q2):
    """Angular separation in degrees."""
    d = np.clip(quat_dot(q1, q2), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def principal_axis_guess(q1, q2, dt):
    """Initial omega guess from axis-angle of dq = q1* x q2, divided by dt."""
    # q1_inv * q2 gives the delta rotation
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])  # scipy uses xyzw
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    dr = r1.inv() * r2
    rotvec = dr.as_rotvec()  # axis * angle in radians
    return rotvec / dt


# ── Ground truth ─────────────────────────────────────────────────────
q1 = np.array([0.9238795, 0.2209424, 0.1104712, 0.2946566])
q1 /= np.linalg.norm(q1)
omega_true = np.deg2rad(np.array([0.5, -0.3, 2.0]))  # rad/s
dt = 50.0  # seconds

# Propagate to get q2
times = np.array([0.0, dt])
quats, _ = propagate_attitude(q1, omega_true, times, mode='tumbling', inertia_tensor=I)
q2 = quats[1]

print(f"\nTrue omega (deg/s): [{np.degrees(omega_true[0]):.1f}, {np.degrees(omega_true[1]):.1f}, {np.degrees(omega_true[2]):.1f}]")
print(f"q1: {q1}")
print(f"q2: {q2}")
angle_traversed = quat_angle_deg(q1, q2)
print(f"Angular separation q1->q2: {angle_traversed:.1f} deg")


# ── Objective function ───────────────────────────────────────────────
eval_count = 0

def objective(omega_vec):
    """Cost = 1 - |dot(propagated_q, q2_target)|"""
    global eval_count
    eval_count += 1
    quats_trial, _ = propagate_attitude(q1, omega_vec, times, mode='tumbling', inertia_tensor=I)
    q_arrived = quats_trial[1]
    return quat_geodesic(q_arrived, q2)


# ── Run optimizer ────────────────────────────────────────────────────
omega_guess = principal_axis_guess(q1, q2, dt)
guess_error = np.degrees(np.linalg.norm(omega_guess - omega_true))
print(f"\nPrincipal-axis initial guess (deg/s): [{np.degrees(omega_guess[0]):.3f}, {np.degrees(omega_guess[1]):.3f}, {np.degrees(omega_guess[2]):.3f}]")
print(f"Initial guess error: {guess_error:.4f} deg/s")

# Try multiple optimizers
methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell']

for method in methods:
    eval_count = 0
    t0 = time.perf_counter()
    result = minimize(objective, omega_guess, method=method,
                      options={'xatol': 1e-14, 'fatol': 1e-15, 'gtol': 1e-15, 'ftol': 1e-15})
    wall_ms = (time.perf_counter() - t0) * 1000

    omega_found = result.x
    omega_err_deg = np.degrees(np.linalg.norm(omega_found - omega_true))
    q_final, _ = propagate_attitude(q1, omega_found, times, mode='tumbling', inertia_tensor=I)
    angle_err = quat_angle_deg(q_final[1], q2)

    print(f"\n── {method} ──")
    print(f"  Wall time:    {wall_ms:.0f} ms")
    print(f"  Evaluations:  {eval_count}")
    print(f"  Omega error:  {omega_err_deg:.6f} deg/s")
    print(f"  q2 angle err: {angle_err:.2e} deg")
    print(f"  Converged:    {result.success}")
