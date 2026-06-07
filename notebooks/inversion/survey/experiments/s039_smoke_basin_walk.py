"""s039 smoke — sanity-check surrogate-MSE at truth/twin/IC#60 + walk straight
lines toward each attractor.

Rules out: (a) bug in residual; (b) IC #60 actually being at q0_twin numerically;
(c) twin not being a surrogate-MSE minimum.

If MSE walks DOWN from IC #60 toward (q0_twin, ω_twin) but LM walks UP, there
is a bug in the LM call. If MSE walks UP from IC #60 toward both attractors,
the surrogate landscape genuinely has a barrier.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.filter_costs import load_static_geometry

S036 = SURVEY_DIR / "results" / "s036_multi_seed_pilot" / "seed023"
TRAJ = SURVEY_DIR / "data" / "trajectories" / "traj_seed023.npz"
OUT = SURVEY_DIR / "results" / "s039_smoke_basin_walk"
OUT.mkdir(parents=True, exist_ok=True)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_R_i2b_batch(q_arr):
    qxyzw = q_arr[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def angular_dist(q1, q2):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(np.dot(q1, q2))))))


def main():
    # --- Load truth + IC #60 ---
    truth = np.load(TRAJ)
    obs_times = truth["observation_times"].astype(np.float64)
    sun_pos = truth["sun_pos"]
    obs_pos = truth["obs_pos"]
    sat_pos = truth["sat_pos"]
    obs_dist = truth["obs_dist"].astype(np.float64)
    mag_truth = truth["mag_hifi"].astype(np.float64)
    valid = np.isfinite(mag_truth)
    sun_unit = ((sun_pos - sat_pos) / np.linalg.norm(sun_pos - sat_pos, axis=1, keepdims=True)).astype(np.float64)
    obs_unit = ((obs_pos - sat_pos) / np.linalg.norm(obs_pos - sat_pos, axis=1, keepdims=True)).astype(np.float64)
    inertia = load_static_geometry()["inertia_tensor"].astype(np.float64)

    q0_truth = truth["q0_wxyz"].astype(np.float64)
    omega_truth = truth["omega0_rad"].astype(np.float64)
    twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), q0_truth)
    R_180x = np.diag([1.0, -1.0, -1.0])
    twin_omega = R_180x @ omega_truth

    # IC #60 — pull from cached candidates_meta + bracket cell
    cands = np.load(S036 / "candidates_meta.npz", mmap_mode="r")
    omega_grid = np.load(S036 / "omega_grid.npz")
    bracket = np.load(S036 / "bracket.npz")
    mag_idx = int(np.argmin(bracket["bracket_dist_to_truth_pct"]))
    nearest_dir = int(omega_grid["nearest_dir_idx"])
    cell_idx = mag_idx * 2000 + nearest_dir
    M_q = 528
    ic_60_q0 = np.array(cands["q0"][cell_idx * M_q + 60]).astype(np.float64)
    omega_bracket = omega_grid["omega_vectors"][cell_idx].astype(np.float64)

    print(f"q0_truth         : {q0_truth}")
    print(f"twin_q0          : {twin_q0}")
    print(f"IC #60 q0        : {ic_60_q0}")
    print(f"  d(IC60, truth) : {angular_dist(ic_60_q0, q0_truth):.4f}°")
    print(f"  d(IC60, twin)  : {angular_dist(ic_60_q0, twin_q0):.4f}°")
    print()
    print(f"ω_truth     : {omega_truth} (|·|={np.linalg.norm(omega_truth):.6f})")
    print(f"ω_twin      : {twin_omega} (|·|={np.linalg.norm(twin_omega):.6f})")
    print(f"ω_bracket   : {omega_bracket} (|·|={np.linalg.norm(omega_bracket):.6f})")
    truth_dir = omega_truth / np.linalg.norm(omega_truth)
    twin_dir = twin_omega / np.linalg.norm(twin_omega)
    bracket_dir = omega_bracket / np.linalg.norm(omega_bracket)
    print(f"  ang(bracket, truth_dir): {np.degrees(np.arccos(np.clip(bracket_dir @ truth_dir, -1, 1))):.3f}°")
    print(f"  ang(bracket, twin_dir):  {np.degrees(np.arccos(np.clip(bracket_dir @ twin_dir, -1, 1))):.3f}°")
    print(f"  mag offset bracket→truth: {(np.linalg.norm(omega_bracket)-np.linalg.norm(omega_truth))/np.linalg.norm(omega_truth)*100:+.3f}%")

    # --- Forward eval helper ---
    from lib.surrogate_eval import get_model
    surrogate = get_model()

    def surrogate_mse(q0, omega):
        q_traj, _ = propagate_attitude(q0, omega, obs_times, mode="tumbling",
                                        inertia_tensor=inertia)
        R = quat_to_R_i2b_batch(q_traj)
        k1 = np.einsum('eij,ej->ei', R, sun_unit)
        k2 = np.einsum('eij,ej->ei', R, obs_unit)
        mp = surrogate.predict_magnitude(k1, k2, 0.0, 15.0, obs_dist)
        return float(np.mean((mp[valid] - mag_truth[valid]) ** 2))

    print("\n=== Spot checks ===")
    pts = [
        ("truth (q0_t, ω_t)",   q0_truth, omega_truth),
        ("twin  (q0_w, ω_w)",   twin_q0,  twin_omega),
        ("twin  (q0_w, ω_t)",   twin_q0,  omega_truth),  # mixed: twin q0 + truth ω
        ("twin  (q0_w, ω_brkt)", twin_q0, omega_bracket),
        ("IC60  (q0_60, ω_brkt)", ic_60_q0, omega_bracket),
        ("IC60  (q0_60, ω_t)",   ic_60_q0, omega_truth),
    ]
    for name, q, w in pts:
        mse = surrogate_mse(q, w)
        print(f"  {name:>30}: MSE={mse:.6e}  ρ={np.sqrt(mse)/0.05:.3f}")

    # --- Straight-line walk: IC #60 → (q0_truth, ω_truth) ---
    print("\n=== Walk from IC#60 toward TRUTH (q0 SLERP, ω linear), 21 steps ===")
    rot_a = Rotation.from_quat(ic_60_q0[[1, 2, 3, 0]])
    rot_b = Rotation.from_quat(q0_truth[[1, 2, 3, 0]])
    key_rots = Rotation.concatenate([rot_a, rot_b])
    slerp = Slerp([0.0, 1.0], key_rots)
    ts = np.linspace(0, 1, 21)
    walks_truth = []
    for t in ts:
        qx = slerp(t).as_quat()
        qw = np.array([qx[3], qx[0], qx[1], qx[2]])
        wv = (1 - t) * omega_bracket + t * omega_truth
        mse = surrogate_mse(qw, wv)
        walks_truth.append((t, mse, np.sqrt(mse)/0.05))
        print(f"  t={t:.2f}  MSE={mse:.4e}  ρ={np.sqrt(mse)/0.05:7.3f}")

    print("\n=== Walk from IC#60 toward TWIN (q0 SLERP, ω linear), 21 steps ===")
    rot_b = Rotation.from_quat(twin_q0[[1, 2, 3, 0]])
    key_rots = Rotation.concatenate([rot_a, rot_b])
    slerp = Slerp([0.0, 1.0], key_rots)
    walks_twin = []
    for t in ts:
        qx = slerp(t).as_quat()
        qw = np.array([qx[3], qx[0], qx[1], qx[2]])
        wv = (1 - t) * omega_bracket + t * twin_omega
        mse = surrogate_mse(qw, wv)
        walks_twin.append((t, mse, np.sqrt(mse)/0.05))
        print(f"  t={t:.2f}  MSE={mse:.4e}  ρ={np.sqrt(mse)/0.05:7.3f}")

    np.savez_compressed(OUT / "walks.npz",
                        ts=ts,
                        ic_60_q0=ic_60_q0, omega_bracket=omega_bracket,
                        q0_truth=q0_truth, omega_truth=omega_truth,
                        twin_q0=twin_q0, twin_omega=twin_omega,
                        walks_truth=np.array(walks_truth),
                        walks_twin=np.array(walks_twin))


if __name__ == "__main__":
    main()
