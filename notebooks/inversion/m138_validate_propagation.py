"""Quick validation: does my RK4 propagate_identity_batch match
src/dynamics/attitude_propagator's propagate_attitude?

Test: compute q_truth(t) two ways and compare.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from src.dynamics.attitude_propagator import propagate_attitude
from m138_isoshell_h1 import (
    propagate_identity_batch, quat_mul, quat_canonicalise, geodesic_deg,
)
from lib.traj_source import load_truth

seed = 91
truth = load_truth(seed, "m048")
q0_truth = truth["q0_wxyz"]
w0_truth = truth["omega0_rad"]
I_tensor = truth["inertia_tensor"]
I_diag = np.diag(I_tensor) if I_tensor.shape == (3, 3) else np.asarray(I_tensor)

# Pick 10 representative times in seed 91's window
obs_times = truth["observation_times"]
t_eval = obs_times[::50]  # 10 samples
print("t_eval:", t_eval)

# Method A: project's propagate_attitude with q0=true_q0, ω=true_omega
quats_A, _ = propagate_attitude(q0_truth, w0_truth, t_eval, "tumbling", I_tensor)
print("\nMethod A (project's propagate_attitude with q0=truth):")
print(quats_A[:3])

# Method B: my batched RK4 from q0=I with ω=true_omega, then left-mul by q0_truth
q_world_B = propagate_identity_batch(w0_truth[None, :], I_diag, t_eval, dt=2.0)[0]
q_truth_B_left = quat_mul(np.broadcast_to(q0_truth, q_world_B.shape).copy(), q_world_B)
q_truth_B_right = quat_mul(q_world_B, np.broadcast_to(q0_truth, q_world_B.shape).copy())
print("\nMethod B (my RK4 from identity, then LEFT-multiply by q0_truth):")
print(q_truth_B_left[:3])
print("\nMethod B (my RK4 from identity, then RIGHT-multiply by q0_truth):")
print(q_truth_B_right[:3])

# Compare A vs B
print(f"\nA vs B-left geodesic deg: {geodesic_deg(quats_A, q_truth_B_left)}")
print(f"A vs B-right geodesic deg: {geodesic_deg(quats_A, q_truth_B_right)}")

# Also: check identity-only ω-truth propagation against propagate_attitude(q0=I)
quats_C, _ = propagate_attitude(np.array([1.0, 0.0, 0.0, 0.0]), w0_truth,
                                  t_eval, "tumbling", I_tensor)
print(f"\nC (project, q0=I) vs B-bare (my RK4, q0=I) geodesic deg:")
print(geodesic_deg(quats_C, q_world_B))
