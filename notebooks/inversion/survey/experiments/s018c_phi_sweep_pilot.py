"""s018c — phi-sweep IC generator + joint LM polish, 5-seed pilot.

Architecture:
  1. For each pilot seed, identify "bright" peaks at mag_abs < 8 (s018a
     calibration) and classify each into tier T1/T2/T3/T4 (s018b table).
  2. For each peak, compute Phi(t_peak; omega) for each omega in a coarse
     grid (omega-direction Fibonacci sphere × omega-mag log-spaced bins).
  3. For each (peak, candidate_face_in_tier_shortlist, phi):
       q_target_at_peak = rot_to_quat(R_phi(pab) @ R_align(n_g, pab))
       q0_ic            = Phi^-1 ⊗ q_target  (LEFT-mult — see s018c_smoke.py)
  4. Each (q0_ic, omega_grid) → joint IC. Surrogate-evaluate full LC for
     all (q0, omega) — this is the "score phase".
  5. Top-K by surrogate full-LC MSE → joint 6-DOF LM polish (max_nfev=60,
     pattern from s011/s015/s016c_prime).
  6. Cohort selector = lowest LM-polished surrogate full-LC MSE per seed.
  7. Top-3 per seed flagged for hi-fi rerank (handled by a follow-up
     script that calls lib.hifi_render).

Pre-registered question (decisive):
  Does s018c hit ≥ 4/5 Band A∪B (rho < 4) on the 5-seed pilot, where
  s017's joint-Sobol architecture hit 3/7? Yes → cohort scan.
  No → S016-A coarse adaptive omega grid is the principled fallback.

Pilot seeds (revised 2026-05-02 after pre-launch tier audit):
  6   — control (s011-recoverable, s017-D — architecture test)
  28  — sub-Sobol-narrow basin (s006 ~2°)
  79  — s017 boundary multi-solution
  44  — best-case (s010 sub-Sobol-narrow + 4 tiers, 15 classifiable peaks)
  84  — s014 dish-heavy multi-solution (T3+T4 dishes, 9 candidates)

Originally planned 41 and 91 dropped: both zero-classifiable under the
s018a mag_abs<8 threshold (no peaks bright enough to anchor a face); they
fall into the s018b "19/100 zero-classifiable, Sobol fallback" cohort.

Architecture parameters:
  N_PHI   = 6                # phi values per (peak, candidate face)
  N_OMEGA_DIR = 12           # Fibonacci sphere directions
  N_OMEGA_MAG = 3            # log-spaced mag bins over [0.1, 1.5] dps
  TOP_K_LM = 128             # rank by surrogate full-LC MSE, polish top-K

Pool(8), BLAS=1, torch.set_num_threads(1).

Wall budget per seed (estimate):
  Score phase:   ~10s/IC * N_q0 / 8 workers
                 typical N_q0 ~ 100-500 -> ~5-30 s
  Score across 36 omega cells: x36 -> ~3-20 min
  LM phase:      128 LMs * 20 s avg / 8 workers ~ 5 min
  Total: ~10-30 min/seed wall, 5 seeds: 50-150 min wall.

Outputs:
  results/s018c/
    summary.json                      cohort-level summary + pre-registered prediction
    seed{XXX}/{ic_pool, scored, lm_polished, top3_for_hifi}.npz
    s018c_run.log                     gitignored
"""

# ─── Threading guards (must precede any numpy/torch import) ─────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"

sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
S018B_NPZ = SURVEY_DIR / "results" / "s018b" / "face_tiers.npz"
OUT_DIR = SURVEY_DIR / "results" / "s018c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ─────────────────────────────────────────────────────
PILOT_SEEDS = [6, 28, 79, 44, 84]

PILOT_ROLES = {
    6:  "control_s011_recoverable_s017_D",
    28: "sub_sobol_narrow_basin_s006",
    79: "s017_boundary_multi_solution",
    44: "best_case_4_tiers_15_peaks",
    84: "dish_heavy_multi_solution_s014",
}

# IC-generation knobs — keep modest at first; can be bumped after smoke.
N_PHI = 6                  # phi values per (peak, candidate face)
N_OMEGA_DIR = 12           # Fibonacci sphere directions
N_OMEGA_MAG = 3            # log-spaced mag bins over [0.1, 1.5] dps
OMEGA_MAG_BINS_DPS = np.geomspace(0.1, 1.5, N_OMEGA_MAG)

# Score-phase + LM
TOP_K_LM = 128
MAX_NFEV = 60

# IS-901 face-group body-frame normals (10 groups), STL-derived geometry.
# Order matches s018b GROUP_NAMES.
IS901_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z',
               '+WD', '-WD', '+ED', '-ED']
IS901_NORMALS = np.array([
    [ 1.0000,  0.0000, 0.0],   # +X
    [-1.0000,  0.0000, 0.0],   # -X
    [ 0.0000,  1.0000, 0.0],   # +Y
    [ 0.0000, -1.0000, 0.0],   # -Y
    [ 0.0000,  0.0000, 1.0],   # +Z
    [ 0.0000,  0.0000, -1.0],  # -Z
    [ 0.9659, -0.2588, 0.0],   # +WD
    [-0.9659,  0.2588, 0.0],   # -WD
    [ 0.9659,  0.2588, 0.0],   # +ED
    [-0.9659, -0.2588, 0.0],   # -ED
])

# Tier shortlists — copied from s018b summary.json (purity 97-100%).
# Each row lists the IS901_NORMALS row indices in the candidate set.
TIER_BANDS = [(0.0, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0)]
TIER_LABELS = ['T1_X', 'T2_YZ', 'T3_any', 'T4_D']
TIER_SHORTLISTS = [
    [0, 1],                              # T1: ±X
    [2, 3, 4, 5],                        # T2: ±Y, ±Z
    [2, 3, 4, 5, 6, 7, 8, 9],            # T3: any non-±X
    [6, 7, 8, 9],                        # T4: dishes ±WD, ±ED
]

D_REF_KM = 38649.2
BRIGHT_THRESHOLD_MAG_ABS = 8.0   # s018a-calibrated

# ρ-band thresholds for downstream classification (not used here directly,
# left for the hi-fi rerank script).
N_WORKERS = 8


# ─── Quaternion / rotation helpers ─────────────────────────────────────
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation about body-X


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_rotvec_wxyz(rotvec):
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0 - 0.125 * angle * angle,
                         0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), s * rotvec[0], s * rotvec[1], s * rotvec[2]])


def rot_to_quat_wxyz(R_b2i):
    R_i2b = R_b2i.T
    q_xyzw = Rotation.from_matrix(R_i2b).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def rotation_aligning(a, b):
    """Rotation matrix R such that R @ a = b for unit vectors a, b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    s = float(np.linalg.norm(v))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def rotation_about_axis(axis_unit, angle_rad):
    return Rotation.from_rotvec(angle_rad * axis_unit).as_matrix()


def fibonacci_sphere(n):
    """Roughly-uniform unit vectors on S² (n points)."""
    out = np.empty((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        out[i] = [r * np.cos(theta), y, r * np.sin(theta)]
    return out


# ─── IC generation ─────────────────────────────────────────────────────
def classify_tier(mag_abs):
    """Return tier index 0..3 or -1 if outside all tiers."""
    for ti, (lo, hi) in enumerate(TIER_BANDS):
        if lo <= mag_abs < hi:
            return ti
    return -1


def build_omega_grid():
    dirs = fibonacci_sphere(N_OMEGA_DIR)        # (D, 3)
    mags = OMEGA_MAG_BINS_DPS                   # (M,) dps
    mags_rad = np.radians(mags)
    out = []
    for d in dirs:
        for m in mags_rad:
            out.append(d * m)
    return np.array(out)                        # (D*M, 3)  rad/s


def build_phi_sweep_ics_for_seed(seed, traj_data, omega_grid):
    """Return list of dicts, each {q0_wxyz, omega_rad, peak_idx, face_idx, phi_rad}.

    For each (omega) in the grid, propagate (identity, omega) once to peak
    epochs to cache Phi(t_peak; omega), then enumerate (peak × face × phi).
    """
    times       = traj_data['observation_times']
    pab_j2000   = traj_data['pab_j2000']
    mag_hifi    = traj_data['mag_hifi']
    obs_dist    = traj_data['obs_dist']
    peak_eps    = traj_data['hifi_peak_epochs'].astype(int)
    inertia     = traj_data['inertia_tensor']

    # Bright peak filter (s018a)
    mags_at_peaks = mag_hifi[peak_eps]
    obs_dist_peaks = obs_dist[peak_eps]
    mag_abs_at_peaks = mags_at_peaks - 5.0 * np.log10(obs_dist_peaks / D_REF_KM)
    bright_mask = mag_abs_at_peaks < BRIGHT_THRESHOLD_MAG_ABS
    bright_peak_eps = peak_eps[bright_mask]
    bright_mag_abs = mag_abs_at_peaks[bright_mask]

    if len(bright_peak_eps) == 0:
        return [], {
            'n_bright_peaks': 0, 'tier_counts': [0, 0, 0, 0],
            'n_total_ics': 0,
        }

    # Per-peak tier classification
    tiers_for_peaks = np.array([classify_tier(m) for m in bright_mag_abs])
    valid_peaks = tiers_for_peaks >= 0
    bright_peak_eps = bright_peak_eps[valid_peaks]
    tiers_for_peaks = tiers_for_peaks[valid_peaks]
    bright_mag_abs = bright_mag_abs[valid_peaks]

    if len(bright_peak_eps) == 0:
        return [], {
            'n_bright_peaks': 0, 'tier_counts': [0, 0, 0, 0],
            'n_total_ics': 0,
        }

    # Phi values uniformly spaced in [0, 2pi)
    phi_values = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)

    # Pre-compute R_align(n_g, pab(t_peak)) for each (peak, face_in_tier).
    pab_at_peaks = pab_j2000[bright_peak_eps]   # (P, 3)
    n_peaks = len(bright_peak_eps)

    # Precompute for each peak: which faces × R_align matrices.
    align_cache = []
    for pi in range(n_peaks):
        tier = tiers_for_peaks[pi]
        face_set = TIER_SHORTLISTS[tier]
        pab = pab_at_peaks[pi]
        per_face = []
        for fi in face_set:
            n_g = IS901_NORMALS[fi]
            R_align = rotation_aligning(n_g, pab)
            per_face.append((fi, n_g, R_align, pab))
        align_cache.append(per_face)

    # Enumerate ICs. For each ω in grid: forward-propagate (identity, ω) once,
    # extract Phi at each peak epoch, then build q0 candidates.
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    times_for_phi = np.concatenate([[times[0]], times[bright_peak_eps]])

    ics = []
    tier_counts_total = [0, 0, 0, 0]
    for omega in omega_grid:
        try:
            quats_phi, _ = propagate_attitude(
                q0=q_id, omega0=omega,
                times=times_for_phi, mode="tumbling", inertia_tensor=inertia,
            )
        except Exception:
            continue
        # quats_phi shape (1+P, 4); index 0 is at times[0], indices 1..P at peaks.
        for pi in range(n_peaks):
            Phi = quats_phi[1 + pi]
            Phi_inv = quat_conjugate_wxyz(Phi)
            tier = int(tiers_for_peaks[pi])
            for (fi, n_g, R_align, pab) in align_cache[pi]:
                for phi in phi_values:
                    R_phi = rotation_about_axis(pab, float(phi))
                    R_b2i = R_phi @ R_align
                    q_target = rot_to_quat_wxyz(R_b2i)
                    q0_ic = quat_multiply_wxyz(Phi_inv, q_target)
                    q0_ic /= np.linalg.norm(q0_ic)
                    ics.append({
                        'q0_wxyz': q0_ic.copy(),
                        'omega_rad': omega.copy(),
                        'peak_idx': int(pi),
                        'peak_epoch': int(bright_peak_eps[pi]),
                        'face_idx': int(fi),
                        'phi_rad': float(phi),
                        'tier': tier,
                    })
                    tier_counts_total[tier] += 1

    return ics, {
        'n_bright_peaks': int(n_peaks),
        'tier_counts': [int(c) for c in tier_counts_total],
        'n_total_ics': len(ics),
    }


# ─── Worker globals + functions for Pool ─────────────────────────────
_W_TRAJ_BY_SEED = {}     # per-seed trajectory dicts
_W_INERTIA = None


def init_worker(traj_data_by_seed, inertia):
    """Pool worker initializer. Imports torch *inside* so we can cap threads.

    Critical: torch ignores OMP_NUM_THREADS by default (cpu_count()//2). With
    Pool(N), un-capped torch yields ~N×16 threads → severe contention.
    """
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    global _W_INERTIA
    _W_INERTIA = inertia
    for s, d in traj_data_by_seed.items():
        _W_TRAJ_BY_SEED[s] = d
    surrogate_eval.get_model()  # warm cache


def score_ic_worker(args):
    """Surrogate full-LC MSE for a single (seed, q0, omega) tuple."""
    seed, q0_wxyz, omega_rad = args
    d = _W_TRAJ_BY_SEED[seed]
    times    = d['observation_times']
    sun_pos  = d['sun_pos']
    obs_pos  = d['obs_pos']
    sat_pos  = d['sat_pos']
    obs_dist = d['obs_dist']
    mag_truth = d['mag_hifi']
    inertia = _W_INERTIA
    finite = np.isfinite(mag_truth)
    try:
        k1b, k2b, _ = propagate_to_body_frame(
            q0_wxyz, omega_rad, times, sun_pos, obs_pos, sat_pos, inertia,
        )
        pred = surrogate_eval.predict(k1b, k2b, obs_dist)
    except Exception:
        return float("inf")
    r = pred - mag_truth
    bad = ~(finite & np.isfinite(r))
    r = np.where(bad, 0.0, r)
    return float(np.mean(r ** 2))


def lm_polish_worker(args):
    """6-DOF joint LM: x = [δθ_x, δθ_y, δθ_z, ω_x, ω_y, ω_z], δθ wrt q0_seed."""
    seed, q0_seed_wxyz, omega_seed_rad, ic_meta = args
    d = _W_TRAJ_BY_SEED[seed]
    times    = d['observation_times']
    sun_pos  = d['sun_pos']
    obs_pos  = d['obs_pos']
    sat_pos  = d['sat_pos']
    obs_dist = d['obs_dist']
    mag_truth = d['mag_hifi']
    q0_truth = d['q0_wxyz']
    omega_truth = d['omega0_rad']
    inertia = _W_INERTIA
    finite = np.isfinite(mag_truth)

    def residuals(x):
        delta_theta = x[:3]
        omega = x[3:6]
        delta_q = quat_from_rotvec_wxyz(delta_theta)
        q0 = quat_multiply_wxyz(delta_q, q0_seed_wxyz)
        q0 = q0 / np.linalg.norm(q0)
        try:
            k1b, k2b, _ = propagate_to_body_frame(
                q0, omega, times, sun_pos, obs_pos, sat_pos, inertia,
            )
            pred = surrogate_eval.predict(k1b, k2b, obs_dist)
        except Exception:
            return np.full_like(mag_truth, 1e3)
        r = pred - mag_truth
        bad = ~(finite & np.isfinite(r))
        r = np.where(bad, 0.0, r)
        return r

    x0 = np.concatenate([np.zeros(3), omega_seed_rad])
    initial_residual = residuals(x0)
    initial_mse = float(np.mean(initial_residual ** 2))

    t0 = time.time()
    try:
        result = least_squares(
            residuals, x0, method="lm",
            max_nfev=MAX_NFEV, xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )
        success = bool(result.success)
        status = int(result.status)
        n_fev = int(result.nfev)
        x_final = result.x.copy()
    except Exception:
        success = False
        status = -99
        n_fev = 0
        x_final = x0.copy()
    wall = time.time() - t0

    delta_theta_final = x_final[:3]
    omega_final = x_final[3:6]
    delta_q_final = quat_from_rotvec_wxyz(delta_theta_final)
    q0_final = quat_multiply_wxyz(delta_q_final, q0_seed_wxyz)
    q0_final /= np.linalg.norm(q0_final)
    final_residual = residuals(x_final)
    final_mse = float(np.mean(final_residual ** 2))

    # Truth-side diagnostics
    q0_err = quat_geodesic_deg(q0_final, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    twin_err = quat_geodesic_deg(q0_final, q_twin)
    omega_truth_mag = float(np.linalg.norm(omega_truth))
    omega_truth_dir = omega_truth / max(omega_truth_mag, 1e-12)
    omega_final_mag = float(np.linalg.norm(omega_final))
    if omega_final_mag < 1e-12:
        omega_dir_err = 180.0
    else:
        omega_dir_err = float(np.degrees(np.arccos(
            float(np.clip(np.dot(omega_final / omega_final_mag, omega_truth_dir),
                          -1.0, 1.0))
        )))
    omega_mag_err_pct = 100.0 * (omega_final_mag - omega_truth_mag) / omega_truth_mag

    return {
        'seed': seed,
        'ic_meta': ic_meta,
        'q0_seed_wxyz': q0_seed_wxyz,
        'omega_seed_rad': omega_seed_rad,
        'q0_final_wxyz': q0_final,
        'omega_final_rad': omega_final,
        'initial_mse': initial_mse,
        'final_mse': final_mse,
        'n_fev': n_fev,
        'success': success,
        'status': status,
        'wall_s': wall,
        'q0_err_deg': q0_err,
        'twin_err_deg': twin_err,
        'omega_dir_err_deg': omega_dir_err,
        'omega_mag_err_pct': omega_mag_err_pct,
    }


# ─── Pilot driver ──────────────────────────────────────────────────────
def main():
    print(f"=== s018c phi-sweep IC pilot ===", flush=True)
    print(f"Pilot seeds: {PILOT_SEEDS}", flush=True)
    print(f"N_PHI={N_PHI}, omega grid: {N_OMEGA_DIR}×{N_OMEGA_MAG} = "
          f"{N_OMEGA_DIR*N_OMEGA_MAG}", flush=True)
    print(f"Top-K LM polish: {TOP_K_LM}", flush=True)
    print(f"Pool({N_WORKERS}), BLAS=1, max_nfev={MAX_NFEV}", flush=True)

    # Load master + per-seed traj data
    master = np.load(M048_MASTER, allow_pickle=True)
    inertia = np.asarray(master['inertia_tensor'], dtype=float)

    seed_data = {}
    for seed in PILOT_SEEDS:
        traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
        d_npz = np.load(traj_path)
        d = {
            'observation_times': d_npz['observation_times'].astype(float),
            'sun_pos':           d_npz['sun_pos'].astype(float),
            'obs_pos':           d_npz['obs_pos'].astype(float),
            'sat_pos':           d_npz['sat_pos'].astype(float),
            'obs_dist':          d_npz['obs_dist'].astype(float),
            'mag_hifi':          d_npz['mag_hifi'].astype(float),
            'pab_j2000':         d_npz['pab_j2000'].astype(float),
            'hifi_peak_epochs':  d_npz['hifi_peak_epochs'],
            'q0_wxyz':           d_npz['q0_wxyz'].astype(float),
            'omega0_rad':        d_npz['omega0_rad'].astype(float),
            'inertia_tensor':    inertia,
        }
        seed_data[seed] = d

    # Build omega grid (shared across seeds)
    omega_grid = build_omega_grid()
    np.savez(OUT_DIR / "omega_grid.npz",
             omega_rad=omega_grid,
             dirs=fibonacci_sphere(N_OMEGA_DIR),
             mags_dps=OMEGA_MAG_BINS_DPS)
    print(f"Saved: {OUT_DIR/'omega_grid.npz'}", flush=True)

    # Prepare per-seed IC pool (single-process — fast, no surrogate calls)
    print(f"\n--- Phase 1: IC generation ---", flush=True)
    ic_pools_by_seed = {}
    ic_meta_by_seed = {}
    t_phase1 = time.time()
    for seed in PILOT_SEEDS:
        t0 = time.time()
        ics, meta = build_phi_sweep_ics_for_seed(seed, seed_data[seed], omega_grid)
        wall = time.time() - t0
        ic_pools_by_seed[seed] = ics
        ic_meta_by_seed[seed] = meta
        print(f"  seed {seed:3d}: bright_peaks={meta['n_bright_peaks']:2d}, "
              f"tiers={meta['tier_counts']}, "
              f"n_ics={meta['n_total_ics']:6d}, "
              f"wall={wall:.1f}s",
              flush=True)
    print(f"Phase 1 total wall: {time.time()-t_phase1:.1f}s", flush=True)

    # Phase 2: surrogate full-LC MSE scoring on entire IC pool
    print(f"\n--- Phase 2: surrogate scoring ---", flush=True)
    print(f"Total ICs across pilot: "
          f"{sum(len(p) for p in ic_pools_by_seed.values())}", flush=True)
    t_phase2 = time.time()
    scored_args = []
    for seed in PILOT_SEEDS:
        for ic in ic_pools_by_seed[seed]:
            scored_args.append((seed, ic['q0_wxyz'], ic['omega_rad']))

    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia)) as pool:
        scored_mses = pool.map(score_ic_worker, scored_args, chunksize=64)
    print(f"Phase 2 total wall: {time.time()-t_phase2:.1f}s", flush=True)

    # Bundle scoring results back per-seed
    score_idx = 0
    scored_by_seed = {}
    for seed in PILOT_SEEDS:
        N = len(ic_pools_by_seed[seed])
        mses = np.array(scored_mses[score_idx:score_idx + N], dtype=float)
        score_idx += N
        # rank by MSE asc; take top-K_LM (lowest MSE)
        finite_idx = np.where(np.isfinite(mses))[0]
        order = finite_idx[np.argsort(mses[finite_idx])]
        topk = order[:TOP_K_LM]
        scored_by_seed[seed] = {
            'mses': mses,
            'top_k_indices': topk,
        }
        print(f"  seed {seed:3d}: scored {N} ICs, "
              f"min_mse={mses[finite_idx].min():.4e} "
              f"(top-{min(TOP_K_LM, len(topk))} for LM)",
              flush=True)
        # Save
        seed_dir = OUT_DIR / f"seed{seed:03d}"
        seed_dir.mkdir(exist_ok=True)
        np.savez(seed_dir / "ic_pool_scored.npz",
                 q0_wxyz=np.array([ic['q0_wxyz'] for ic in ic_pools_by_seed[seed]]),
                 omega_rad=np.array([ic['omega_rad'] for ic in ic_pools_by_seed[seed]]),
                 peak_idx=np.array([ic['peak_idx'] for ic in ic_pools_by_seed[seed]]),
                 peak_epoch=np.array([ic['peak_epoch'] for ic in ic_pools_by_seed[seed]]),
                 face_idx=np.array([ic['face_idx'] for ic in ic_pools_by_seed[seed]]),
                 phi_rad=np.array([ic['phi_rad'] for ic in ic_pools_by_seed[seed]]),
                 tier=np.array([ic['tier'] for ic in ic_pools_by_seed[seed]]),
                 surrogate_mse=mses)

    # Phase 3: top-K LM polish per seed
    print(f"\n--- Phase 3: LM polish (top-{TOP_K_LM} per seed) ---", flush=True)
    t_phase3 = time.time()
    lm_args = []
    for seed in PILOT_SEEDS:
        topk = scored_by_seed[seed]['top_k_indices']
        for idx in topk:
            ic = ic_pools_by_seed[seed][idx]
            lm_args.append((seed, ic['q0_wxyz'], ic['omega_rad'],
                            {'pool_idx': int(idx),
                             'peak_idx': ic['peak_idx'],
                             'peak_epoch': ic['peak_epoch'],
                             'face_idx': ic['face_idx'],
                             'phi_rad': ic['phi_rad'],
                             'tier': ic['tier']}))
    print(f"Total LM runs: {len(lm_args)}", flush=True)

    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia)) as pool:
        lm_results = pool.map(lm_polish_worker, lm_args, chunksize=8)

    print(f"Phase 3 total wall: {time.time()-t_phase3:.1f}s", flush=True)

    # Cohort selector + per-seed writeup
    cohort_summary = {
        'pilot_seeds': PILOT_SEEDS,
        'pilot_roles': PILOT_ROLES,
        'config': {
            'N_PHI': N_PHI,
            'N_OMEGA_DIR': N_OMEGA_DIR,
            'N_OMEGA_MAG': N_OMEGA_MAG,
            'OMEGA_MAG_BINS_DPS': OMEGA_MAG_BINS_DPS.tolist(),
            'TOP_K_LM': TOP_K_LM,
            'MAX_NFEV': MAX_NFEV,
            'BRIGHT_THRESHOLD_MAG_ABS': BRIGHT_THRESHOLD_MAG_ABS,
            'D_REF_KM': D_REF_KM,
        },
        'ic_meta': ic_meta_by_seed,
        'pre_registered_prediction': '4/5 Band A∪B (uncertain on seeds 28 and 84)',
        'pre_registered_bar': '≥4/5 Band A∪B → cohort scan; <4/5 → S016-A fallback',
        'pre_registered_predictions_per_seed': {
            6:  'Band A (control; classifier covers both T2 + T3 events)',
            28: 'Band A or B (uncertain — sub-Sobol-narrow ~2°; T1 anchors should help)',
            79: 'Band B+ (boundary — s017 already B; phi-sweep should keep ≥B)',
            44: 'Band A (best case — 4 tiers, 15 classifiable peaks)',
            84: 'Band A∪B (uncertain — dish-tier purity 97% but normals close to ±Y)',
        },
        'per_seed': {},
    }

    print(f"\n--- Phase 4: cohort selector ---", flush=True)
    print(f"  seed | n_lm | best_surr_mse | best_q0_err | best_ωd_err | best_ωm_err_pct",
          flush=True)
    print(f"  -----+------+---------------+-------------+-------------+----------------",
          flush=True)

    seed_lm_results = {s: [] for s in PILOT_SEEDS}
    for r in lm_results:
        seed_lm_results[r['seed']].append(r)

    for seed in PILOT_SEEDS:
        results = seed_lm_results[seed]
        if not results:
            print(f"  {seed:4d} |    0 | (no ICs) ", flush=True)
            cohort_summary['per_seed'][seed] = {
                'n_lm': 0,
                'best_final_mse': None,
                'best_q0_err_deg': None,
                'note': 'zero classifiable peaks; no ICs generated',
            }
            continue
        finals = np.array([r['final_mse'] for r in results])
        order = np.argsort(finals)
        best = results[order[0]]
        # top-3 records for hi-fi rerank
        top3 = [results[i] for i in order[:3]]
        seed_dir = OUT_DIR / f"seed{seed:03d}"
        seed_dir.mkdir(exist_ok=True)

        # save full LM results + top3 metadata
        np.savez(
            seed_dir / "lm_polished.npz",
            q0_seed_wxyz=np.array([r['q0_seed_wxyz'] for r in results]),
            omega_seed_rad=np.array([r['omega_seed_rad'] for r in results]),
            q0_final_wxyz=np.array([r['q0_final_wxyz'] for r in results]),
            omega_final_rad=np.array([r['omega_final_rad'] for r in results]),
            initial_mse=np.array([r['initial_mse'] for r in results]),
            final_mse=np.array([r['final_mse'] for r in results]),
            n_fev=np.array([r['n_fev'] for r in results]),
            success=np.array([r['success'] for r in results]),
            status=np.array([r['status'] for r in results]),
            wall_s=np.array([r['wall_s'] for r in results]),
            q0_err_deg=np.array([r['q0_err_deg'] for r in results]),
            twin_err_deg=np.array([r['twin_err_deg'] for r in results]),
            omega_dir_err_deg=np.array([r['omega_dir_err_deg'] for r in results]),
            omega_mag_err_pct=np.array([r['omega_mag_err_pct'] for r in results]),
            ic_pool_idx=np.array([r['ic_meta']['pool_idx'] for r in results]),
            tier=np.array([r['ic_meta']['tier'] for r in results]),
            face_idx=np.array([r['ic_meta']['face_idx'] for r in results]),
            phi_rad=np.array([r['ic_meta']['phi_rad'] for r in results]),
            peak_epoch=np.array([r['ic_meta']['peak_epoch'] for r in results]),
        )
        # top-3 for hi-fi rerank
        np.savez(
            seed_dir / "top3_for_hifi.npz",
            q0_final_wxyz=np.array([r['q0_final_wxyz'] for r in top3]),
            omega_final_rad=np.array([r['omega_final_rad'] for r in top3]),
            final_surr_mse=np.array([r['final_mse'] for r in top3]),
            q0_err_deg=np.array([r['q0_err_deg'] for r in top3]),
            twin_err_deg=np.array([r['twin_err_deg'] for r in top3]),
            omega_dir_err_deg=np.array([r['omega_dir_err_deg'] for r in top3]),
            omega_mag_err_pct=np.array([r['omega_mag_err_pct'] for r in top3]),
        )

        print(f"  {seed:4d} | {len(results):4d} | "
              f"{best['final_mse']:13.4e} | "
              f"{best['q0_err_deg']:11.4f} | "
              f"{best['omega_dir_err_deg']:11.4f} | "
              f"{best['omega_mag_err_pct']:+15.3f}",
              flush=True)
        cohort_summary['per_seed'][seed] = {
            'n_lm': len(results),
            'best_final_mse': float(best['final_mse']),
            'best_q0_err_deg': float(best['q0_err_deg']),
            'best_twin_err_deg': float(best['twin_err_deg']),
            'best_omega_dir_err_deg': float(best['omega_dir_err_deg']),
            'best_omega_mag_err_pct': float(best['omega_mag_err_pct']),
        }

    # Save cohort summary
    out_summary = OUT_DIR / "summary.json"
    with open(out_summary, 'w') as f:
        json.dump(cohort_summary, f, indent=2)
    print(f"\nSaved: {out_summary}", flush=True)
    print(f"Saved: {OUT_DIR/'seed{XXX}/lm_polished.npz, top3_for_hifi.npz'}",
          flush=True)


if __name__ == "__main__":
    main()
