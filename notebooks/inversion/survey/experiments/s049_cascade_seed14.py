"""s049 — multi-epoch consistency cascade on seed 14 (high-|ω| retest of s048).

s048 ran the cascade on seed 89 (|ω|=0.24 dps) and Tier 1 rejected truth because
finite-diff ω-noise (|sample_res|/Δt ≈ 100%) was comparable to truth |ω|. The
s048 author flagged: "At |ω|=1.0 dps the per-Δt rotation is ~7°, dwarfing the
sampling noise, ω-noise drops to ~20%. Re-test on a higher-|ω| seed before
declaring the cascade dead."

Seed 14 (|ω|=1.229 dps) IS that retest. Per-Δt rotation = 8.8°, sample-res =
1.5° → finite-diff ω noise ~17% relative. The cascade is a SEED GENERATOR for
LM, not a basin-finder; ω noise of 17% is inside LM's q-basin tolerance even if
outside the strict ω-mag basin.

Architecture:
  Stage 0  Anchor scan: v1 surrogate, 30 dimmest epochs, 500k random q.
           Pick t_0 = smallest |C_t|; t_1 = nearest tight neighbour in
           [t_0-WINDOW, t_0+WINDOW] (close enough that constant-ω is exact).
  Stage 1  Q_0 = C_{t_0}, Q_1 = C_{t_1}.
  Stage 2  For each (q_a in Q_0, q_b in Q_1): ω_ab = omega_from_pair(q_a, q_b, Δt).
           ~|Q_0| * |Q_1| candidates (~9e4 if both ~300).
  Stage 3  Multi-epoch consistency. Pick K validation epochs (next K tight from
           the scan, excluding t_0, t_1). For each (q_a, ω_ab):
             for k in 1..K:
               q_pred_k = constant_ω_propagate(q_a, ω_ab, t_k - t_0)
               mag_pred = surrogate(q_pred_k at t_k)
               require |mag_pred - mag_measured| < tol_mag
             survive iff all K pass.
           Sweep tol ∈ {0.10, 0.15, 0.20, 0.30} mag and K ∈ {2, 3, 5}.
  Stage 4  Truth check:
             truth_q_a = nearest in Q_0 to quats_truth[t_0]
             truth_q_b = nearest in Q_1 to quats_truth[t_1]
             truth_om_cascade = omega_from_pair(truth_q_a, truth_q_b, Δt)
             question 1: |truth_om_cascade - omegas_full[t_0]| / |ω_truth| (the
                         cascade noise on a known-truth-adjacent pair)
             question 2: does (truth_q_a, truth_om_cascade) survive at each
                         (tol, K) parameter combo?

Outputs: summary.json with all parameter-combo results, scan_npz with the
30-epoch scan data, cascade_npz with the survivors per param combo.

Convention: scalar-first (w, x, y, z) quaternions; LEFT-multiply per
src/dynamics/attitude_propagator. Cohort SP=0°, AD=15°. Tolerance 0.10 mag.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
SURROGATE_PATH = Path("/home/girish/surrogate_model")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURROGATE_PATH))

from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1
from src.dynamics.attitude_propagator import propagate_attitude
from lib.filter_costs import load_static_geometry

# ---- config ----
SEED = 14
N_SAMPLES = 500_000
N_SCAN_EPOCHS = 30                  # scan dimmest 30 for anchor selection
N_VALIDATION_EPOCHS_MAX = 5         # use up to 5 tightest non-anchor epochs
ANCHOR_PAIR_WINDOW = 5              # search [t0-W, t0+W] for t_1
TOLERANCE_FILTER_MAG = 0.10         # mag tol for C_t membership at scan
OMEGA_MAG_BOUNDS_DPS = (0.05, 2.0)  # cohort range
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
RNG_SEED = 42
SCAN_SELECTION = "dimmest"

# Sweep grid for cascade validation
SWEEP_TOL_MAGS = (0.10, 0.15, 0.20, 0.30)
SWEEP_K_VALIDATION = (2, 3, 5)

OUT_DIR = SURVEY_DIR / "results" / "s049_cascade_seed14"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- quaternion utilities (vectorized, wxyz) ----
def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batched Hamilton product (wxyz). Broadcasts (1, 4) ↔ (N, 4)."""
    if q1.ndim == 1:
        q1 = q1[None, :]
    if q2.ndim == 1:
        q2 = q2[None, :]
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = np.broadcast_to(q1, q2.shape).copy()
    if q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = np.broadcast_to(q2, q1.shape).copy()
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_exp_batch(omega: np.ndarray, dt: float) -> np.ndarray:
    """Closed-form rotation quat from ω·dt: q_rot = [cos(θ/2), sin(θ/2)·axis]."""
    omega = np.asarray(omega).reshape(-1, 3)
    om_mag = np.linalg.norm(omega, axis=1)
    theta = om_mag * dt
    half = 0.5 * theta
    safe = om_mag > 1e-12
    axis = np.zeros_like(omega)
    axis[safe] = omega[safe] / om_mag[safe, None]
    sin_h = np.sin(half)
    return np.column_stack([np.cos(half), sin_h * axis[:, 0],
                             sin_h * axis[:, 1], sin_h * axis[:, 2]])


def constant_omega_propagate(q0_wxyz: np.ndarray, omega: np.ndarray,
                              dt: float) -> np.ndarray:
    """q(dt) = quat_exp(ω·dt) ⊗ q0 (LEFT mul, conv (a))."""
    q_rot = quat_exp_batch(omega, dt)
    if q0_wxyz.ndim == 1:
        q0_wxyz = q0_wxyz[None, :]
    if q0_wxyz.shape[0] == 1 and q_rot.shape[0] > 1:
        q0_wxyz = np.broadcast_to(q0_wxyz, q_rot.shape).copy()
    return quat_mul_batch(q_rot, q0_wxyz)


def omega_from_pair(qA: np.ndarray, qB: np.ndarray, dt: float) -> np.ndarray:
    """Vectorized: qA, qB shape (N, 4) wxyz → ω shape (N, 3) rad/s.

    q_rot = qB ⊗ qA^{-1}; ω = (axis · angle) / dt.
    """
    qA_inv = qA.copy()
    qA_inv[:, 1:] *= -1.0
    qr = quat_mul_batch(qB, qA_inv)
    flip = qr[:, 0] < 0
    qr[flip] *= -1
    w = np.clip(qr[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sin(angle / 2.0)
    axis = np.zeros((qr.shape[0], 3))
    ok = s > 1e-9
    axis[ok] = qr[ok, 1:] / s[ok, None]
    return (angle / dt)[:, None] * axis


def quat_to_R_i2b_batch(q_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz → (N, 3, 3) inertial→body rotation matrices."""
    qxyzw = q_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def angular_dist_deg_batch(q_arr: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Geodesic angle on SO(3), q_ref single, q_arr (N,4)."""
    dots = np.abs(q_arr @ q_ref)
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def predict_at_epoch(model: SurrogateV1, q_wxyz: np.ndarray,
                      sun_unit: np.ndarray, obs_unit: np.ndarray,
                      obs_dist_km: float) -> np.ndarray:
    """Predict mag for N candidate q's at ONE epoch (shared sun/obs/dist)."""
    R = quat_to_R_i2b_batch(q_wxyz)
    k1 = R @ sun_unit
    k2 = R @ obs_unit
    N = q_wxyz.shape[0]
    obs_dist_arr = np.full(N, obs_dist_km)
    return model.predict_magnitude(k1, k2, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist_arr)


def main():
    t_start = time.time()

    # ---- load v1 surrogate ----
    print("[load] surrogate v1 ...")
    v1_weights = SURROGATE_PATH / "surrogate_model" / "s10_5M_weights.npz"
    v1_norm = SURROGATE_PATH / "surrogate_model" / "s10_5M_normalization.npz"
    model = SurrogateV1(str(v1_weights), str(v1_norm))
    print(f"[load] {model}")

    # ---- load truth ----
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{SEED:03d}.npz"
    print(f"[load] {traj_path}")
    d = np.load(traj_path)
    obs_times = np.asarray(d["observation_times"], float)
    sun_pos = np.asarray(d["sun_pos"], float)
    obs_pos = np.asarray(d["obs_pos"], float)
    sat_pos = np.asarray(d["sat_pos"], float)
    obs_dist_all = np.asarray(d["obs_dist"], float)
    mag_hifi = np.asarray(d["mag_hifi"], float)
    quats_truth = np.asarray(d["quaternions"], float)
    q0_truth = np.asarray(d["q0_wxyz"], float)
    omega0_truth = np.asarray(d["omega0_rad"], float)
    omega_mag_dps = float(d["omega_mag_dps"])
    dt_sampling = float(d["dt_sampling"])

    sun_vec_j2000 = sun_pos - sat_pos
    obs_vec_j2000 = obs_pos - sat_pos
    sun_unit_all = sun_vec_j2000 / np.linalg.norm(sun_vec_j2000, axis=1, keepdims=True)
    obs_unit_all = obs_vec_j2000 / np.linalg.norm(obs_vec_j2000, axis=1, keepdims=True)
    geo = load_static_geometry()
    inertia = np.asarray(geo["inertia_tensor"], float)

    print(f"[seed {SEED}] |ω|={omega_mag_dps:.3f} dps, "
          f"truth |ω|·Δt = {omega_mag_dps*dt_sampling:.2f}°/Δt, dt={dt_sampling:.2f}s")

    summary = {
        "seed": SEED,
        "omega_mag_dps": omega_mag_dps,
        "dt_sampling": dt_sampling,
        "n_samples": N_SAMPLES,
        "tolerance_filter_mag": TOLERANCE_FILTER_MAG,
        "scan_selection": SCAN_SELECTION,
        "n_scan_epochs": N_SCAN_EPOCHS,
    }

    # ============================================================
    # Stage 0: anchor scan over N_SCAN_EPOCHS dimmest epochs
    # ============================================================
    print("\n=== Stage 0 — anchor scan (v1 surrogate) ===")
    t0 = time.time()

    N_OBS = len(obs_times)
    finite_mask = np.isfinite(mag_hifi)
    finite_idx = np.where(finite_mask)[0]
    order = finite_idx[np.argsort(-mag_hifi[finite_idx])]
    scan_epochs = np.sort(order[:N_SCAN_EPOCHS])
    print(f"[scan] {len(scan_epochs)} dimmest epochs; mag range "
          f"[{mag_hifi[scan_epochs].min():.2f}, {mag_hifi[scan_epochs].max():.2f}]")

    # ---- sample 500k random q ONCE ----
    print(f"[scan] sampling {N_SAMPLES} random q on SO(3)...")
    rng = np.random.default_rng(RNG_SEED)
    R_random = Rotation.random(N_SAMPLES, random_state=rng)
    q_xyzw = R_random.as_quat()
    q_pool = q_xyzw[:, [3, 0, 1, 2]]                  # (N, 4) wxyz
    R_cache = R_random.as_matrix()                     # (N, 3, 3)
    print(f"[scan] R cache built {time.time()-t0:.1f}s")

    # ---- per-epoch C_t and survivor mask, plus nearest-truth distance ----
    n_survivors_scan = np.zeros(len(scan_epochs), dtype=int)
    nearest_truth_deg_scan = np.zeros(len(scan_epochs))
    survive_mask_scan = np.zeros((len(scan_epochs), N_SAMPLES), dtype=bool)
    measured_at_scan = np.zeros(len(scan_epochs))

    for j, ep in enumerate(scan_epochs):
        ep = int(ep)
        k1_body = np.einsum('nij,j->ni', R_cache, sun_unit_all[ep])
        k2_body = np.einsum('nij,j->ni', R_cache, obs_unit_all[ep])
        obs_dist = np.full(N_SAMPLES, obs_dist_all[ep])
        pred = model.predict_magnitude(k1_body, k2_body, SP_ANGLE_DEG,
                                          AD_ANGLE_DEG, obs_dist)
        measured = mag_hifi[ep]
        keep = np.abs(pred - measured) < TOLERANCE_FILTER_MAG
        survive_mask_scan[j] = keep
        n_survivors_scan[j] = int(keep.sum())
        measured_at_scan[j] = measured
        if keep.any():
            d_truth = angular_dist_deg_batch(q_pool[keep], quats_truth[ep])
            nearest_truth_deg_scan[j] = float(d_truth.min())
        else:
            nearest_truth_deg_scan[j] = float("inf")
        print(f"[scan] ep {ep:3d} mag={measured:6.3f}  "
              f"|C_t|={n_survivors_scan[j]:6d}  nearest_truth={nearest_truth_deg_scan[j]:6.2f}°")

    print(f"[scan] wall: {time.time()-t0:.1f}s")
    np.savez(OUT_DIR / "scan.npz",
             q_pool_wxyz=q_pool,
             scan_epochs=scan_epochs,
             survive_mask=survive_mask_scan,
             n_survivors=n_survivors_scan,
             nearest_truth_deg=nearest_truth_deg_scan,
             measured_at=measured_at_scan,
             obs_times_at_scan=obs_times[scan_epochs],
             quats_truth_at_scan=quats_truth[scan_epochs])
    print(f"[saved] {OUT_DIR / 'scan.npz'}")

    summary["stage0_scan"] = {
        "wall_s": time.time() - t0,
        "n_scan_epochs": len(scan_epochs),
        "epoch_indices": scan_epochs.tolist(),
        "n_survivors_per_epoch": n_survivors_scan.tolist(),
        "nearest_truth_deg_per_epoch": nearest_truth_deg_scan.tolist(),
        "tightest_epoch_idx": int(scan_epochs[np.argmin(n_survivors_scan)]),
        "tightest_n_survivors": int(n_survivors_scan.min()),
    }

    # ============================================================
    # Stage 1: pick anchor pair (t_0, t_1)
    # ============================================================
    print("\n=== Stage 1 — anchor pair selection ===")
    t1 = time.time()

    # Filter valid scan epochs (truth still survives — should be all of them)
    valid = (n_survivors_scan > 0) & np.isfinite(nearest_truth_deg_scan)
    valid_idx_in_scan = np.where(valid)[0]
    if len(valid_idx_in_scan) < 2:
        print(f"[stage1] not enough valid scan epochs ({len(valid_idx_in_scan)}); abort.")
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return

    # t_0 = scan epoch with smallest |C_t|.
    sort_order = valid_idx_in_scan[np.argsort(n_survivors_scan[valid_idx_in_scan])]
    j0 = int(sort_order[0])
    t0_ep = int(scan_epochs[j0])

    # t_1 candidates: any obs_time within ANCHOR_PAIR_WINDOW samples of t_0_ep.
    # We pick the tightest |C_t| epoch within the time window using fresh C_t computation.
    candidate_eps = np.arange(max(0, t0_ep - ANCHOR_PAIR_WINDOW),
                                 min(N_OBS, t0_ep + ANCHOR_PAIR_WINDOW + 1))
    candidate_eps = candidate_eps[candidate_eps != t0_ep]
    candidate_eps = candidate_eps[np.isfinite(mag_hifi[candidate_eps])]
    print(f"[stage1] anchor t_0 = ep {t0_ep} (mag={mag_hifi[t0_ep]:.2f}, "
          f"|C_t0|={n_survivors_scan[j0]}, nearest_truth={nearest_truth_deg_scan[j0]:.2f}°)")
    print(f"[stage1] candidate t_1 epochs (within ±{ANCHOR_PAIR_WINDOW} samples): "
          f"{candidate_eps.tolist()}")

    # Compute |C_t| for each candidate (cheap, ~5 sec each at v1)
    cand_n = np.zeros(len(candidate_eps), dtype=int)
    cand_survive = np.zeros((len(candidate_eps), N_SAMPLES), dtype=bool)
    cand_nearest = np.zeros(len(candidate_eps))
    for k, ep in enumerate(candidate_eps):
        ep = int(ep)
        k1_body = np.einsum('nij,j->ni', R_cache, sun_unit_all[ep])
        k2_body = np.einsum('nij,j->ni', R_cache, obs_unit_all[ep])
        obs_dist = np.full(N_SAMPLES, obs_dist_all[ep])
        pred = model.predict_magnitude(k1_body, k2_body, SP_ANGLE_DEG,
                                          AD_ANGLE_DEG, obs_dist)
        keep = np.abs(pred - mag_hifi[ep]) < TOLERANCE_FILTER_MAG
        cand_survive[k] = keep
        cand_n[k] = int(keep.sum())
        if keep.any():
            d_truth = angular_dist_deg_batch(q_pool[keep], quats_truth[ep])
            cand_nearest[k] = float(d_truth.min())
        else:
            cand_nearest[k] = float("inf")
        print(f"[stage1] cand ep {ep:3d}  |C_t|={cand_n[k]:6d}  "
              f"nearest_truth={cand_nearest[k]:6.2f}°")

    # Pick the tightest valid candidate (must have truth nearby)
    valid_cands = (cand_n > 0) & np.isfinite(cand_nearest) & (cand_nearest < 30.0)
    if not valid_cands.any():
        print("[stage1] no valid t_1 candidate; abort.")
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return

    valid_idx = np.where(valid_cands)[0]
    k_best = valid_idx[np.argmin(cand_n[valid_idx])]
    t1_ep = int(candidate_eps[k_best])
    print(f"[stage1] anchor t_1 = ep {t1_ep} (mag={mag_hifi[t1_ep]:.2f}, "
          f"|C_t1|={cand_n[k_best]}, nearest_truth={cand_nearest[k_best]:.2f}°)")

    Q_0 = q_pool[survive_mask_scan[j0]]
    Q_1 = q_pool[cand_survive[k_best]]
    nA, nB = len(Q_0), len(Q_1)
    delta_t = obs_times[t1_ep] - obs_times[t0_ep]
    print(f"[stage1] |Q_0|={nA}, |Q_1|={nB}, Δt = {delta_t:.2f}s")
    print(f"[stage1] truth rotation over Δt: "
          f"{omega_mag_dps * abs(delta_t):.2f}° (sample-res ~1.5°)")

    summary["stage1_anchors"] = {
        "t0_epoch": t0_ep,
        "t1_epoch": t1_ep,
        "delta_t_s": float(delta_t),
        "n_Q0": nA,
        "n_Q1": nB,
        "nearest_truth_Q0_deg": float(nearest_truth_deg_scan[j0]),
        "nearest_truth_Q1_deg": float(cand_nearest[k_best]),
        "wall_s": time.time() - t1,
    }

    # ============================================================
    # Stage 2: derive ω hypotheses from Q_0 × Q_1
    # ============================================================
    print("\n=== Stage 2 — derive ω hypotheses ===")
    t2 = time.time()

    qA_grid = np.repeat(Q_0, nB, axis=0)         # (nA*nB, 4)
    qB_grid = np.tile(Q_1, (nA, 1))               # (nA*nB, 4)
    om_all = omega_from_pair(qA_grid, qB_grid, delta_t)  # (nA*nB, 3)
    om_mag = np.linalg.norm(om_all, axis=1)
    om_lo = np.deg2rad(OMEGA_MAG_BOUNDS_DPS[0])
    om_hi = np.deg2rad(OMEGA_MAG_BOUNDS_DPS[1])
    in_bounds = (om_mag > om_lo) & (om_mag < om_hi)
    qA_kept = qA_grid[in_bounds]
    om_kept = om_all[in_bounds]
    n_hyp = len(qA_kept)
    print(f"[stage2] hypotheses generated: {nA*nB}, in ω bounds: {n_hyp}")
    print(f"[stage2] ω-mag range of hypotheses (dps): "
          f"[{np.degrees(om_mag[in_bounds].min()):.3f}, "
          f"{np.degrees(om_mag[in_bounds].max()):.3f}], "
          f"truth |ω|={omega_mag_dps:.3f}")
    print(f"[stage2] wall: {time.time()-t2:.2f}s")

    summary["stage2_hypotheses"] = {
        "n_total_pairs": int(nA * nB),
        "n_in_omega_bounds": int(n_hyp),
        "wall_s": time.time() - t2,
    }

    if n_hyp == 0:
        print("[stage2] zero hypotheses in ω bounds; abort.")
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return

    # ============================================================
    # Stage 3: multi-epoch consistency filter
    # ============================================================
    print("\n=== Stage 3 — multi-epoch consistency filter ===")
    t3 = time.time()

    # Pick K_max validation epochs:
    # We want both (a) tight |C_t| (strong filter) and (b) close to t_0 in time
    # (small propagation drift). Strategy: rank scan epochs by |C_t| ascending
    # but ONLY among those within VALIDATION_DT_WINDOW samples of t_0; if not
    # enough close-to-anchor scan eps exist, also compute |C_t| on demand at
    # ±k samples from t_0 for small k.
    VALIDATION_DT_WINDOW = 10                 # max |k Δt| from anchor for validation
    excluded = {t0_ep, t1_ep}
    val_candidates = []                        # list of (ep, n_survivors, source)
    for j_scan, ep in enumerate(scan_epochs):
        ep = int(ep)
        if ep in excluded:
            continue
        if abs(ep - t0_ep) > VALIDATION_DT_WINDOW:
            continue
        if n_survivors_scan[j_scan] > 0:
            val_candidates.append((ep, int(n_survivors_scan[j_scan]), "scan"))

    # Augment with adjacent epochs not in scan (compute on-demand)
    needed_eps = []
    for k in range(-VALIDATION_DT_WINDOW, VALIDATION_DT_WINDOW + 1):
        if k == 0:
            continue
        ep = t0_ep + k
        if ep < 0 or ep >= N_OBS:
            continue
        if ep in excluded:
            continue
        if not np.isfinite(mag_hifi[ep]):
            continue
        if any(vc[0] == ep for vc in val_candidates):
            continue
        needed_eps.append(ep)

    if needed_eps:
        print(f"[stage3] computing |C_t| for {len(needed_eps)} adjacent epochs not in scan...")
        for ep in needed_eps:
            k1_body = np.einsum('nij,j->ni', R_cache, sun_unit_all[ep])
            k2_body = np.einsum('nij,j->ni', R_cache, obs_unit_all[ep])
            obs_dist = np.full(N_SAMPLES, obs_dist_all[ep])
            pred = model.predict_magnitude(k1_body, k2_body, SP_ANGLE_DEG,
                                              AD_ANGLE_DEG, obs_dist)
            keep = np.abs(pred - mag_hifi[ep]) < TOLERANCE_FILTER_MAG
            n_keep = int(keep.sum())
            if n_keep > 0:
                val_candidates.append((ep, n_keep, "adjacent"))

    # Sort by smallest |C_t| (tightest first)
    val_candidates.sort(key=lambda x: x[1])
    val_eps = [vc[0] for vc in val_candidates[:N_VALIDATION_EPOCHS_MAX]]
    val_n_survivors = [vc[1] for vc in val_candidates[:N_VALIDATION_EPOCHS_MAX]]
    val_source = [vc[2] for vc in val_candidates[:N_VALIDATION_EPOCHS_MAX]]
    print(f"[stage3] {len(val_eps)} validation epochs (within ±{VALIDATION_DT_WINDOW} Δt, "
          f"tightest first): {val_eps}")
    print(f"[stage3]   |C_t|: {val_n_survivors}")
    print(f"[stage3]   |Δk Δt|: {[ep - t0_ep for ep in val_eps]}")
    print(f"[stage3]   source: {val_source}")

    # For each validation epoch, propagate all hypotheses and predict mag.
    # Then compute |pred - measured| → store as (n_hyp, K) array of |Δmag|.
    n_K = len(val_eps)
    delta_mag = np.zeros((n_hyp, n_K), dtype=np.float32)
    for kk, ep in enumerate(val_eps):
        dt_k = obs_times[ep] - obs_times[t0_ep]
        # propagate all n_hyp candidates from t_0 to t_k
        q_pred = constant_omega_propagate(qA_kept, om_kept, dt_k)
        # predict mag at t_k
        R_pred = quat_to_R_i2b_batch(q_pred)
        k1 = R_pred @ sun_unit_all[ep]
        k2 = R_pred @ obs_unit_all[ep]
        obs_dist_arr = np.full(n_hyp, obs_dist_all[ep])
        pred_mag = model.predict_magnitude(k1, k2, SP_ANGLE_DEG, AD_ANGLE_DEG,
                                              obs_dist_arr)
        delta_mag[:, kk] = np.abs(pred_mag - mag_hifi[ep]).astype(np.float32)
        print(f"[stage3] val ep {ep:3d}  Δt={dt_k:+.1f}s  "
              f"Δmag p10/p50/p90 = "
              f"{np.percentile(delta_mag[:, kk], 10):.3f}/"
              f"{np.percentile(delta_mag[:, kk], 50):.3f}/"
              f"{np.percentile(delta_mag[:, kk], 90):.3f}")
    print(f"[stage3] wall: {time.time()-t3:.2f}s")

    # ---- parameter sweep ----
    sweep_results = []
    for tol_mag in SWEEP_TOL_MAGS:
        pass_per_epoch = delta_mag < tol_mag                  # (n_hyp, n_K)
        for K_required in SWEEP_K_VALIDATION:
            if K_required > n_K:
                continue
            # require all FIRST K_required validation epochs to pass
            survive_mask = pass_per_epoch[:, :K_required].all(axis=1)
            n_survivors = int(survive_mask.sum())
            sweep_results.append({
                "tol_mag": tol_mag,
                "K_required": K_required,
                "n_survivors": n_survivors,
                "fraction_in_pct": float(100.0 * n_survivors / n_hyp),
            })
            print(f"[sweep] tol={tol_mag:.2f}  K={K_required}  "
                  f"survivors={n_survivors:6d} ({100.0*n_survivors/n_hyp:.3f}%)")

    summary["stage3_filter"] = {
        "validation_epochs": val_eps,
        "n_hypotheses": int(n_hyp),
        "wall_s": time.time() - t3,
        "sweep_results": sweep_results,
    }

    # ============================================================
    # Stage 4: ground-truth check
    # ============================================================
    print("\n=== Stage 4 — ground truth ===")
    t4 = time.time()

    # truth_q nearest in Q_0 / Q_1 (separate lookups in the C_t survivor pool)
    d_truth_Q0 = angular_dist_deg_batch(Q_0, quats_truth[t0_ep])
    d_truth_Q1 = angular_dist_deg_batch(Q_1, quats_truth[t1_ep])
    iA_truth = int(np.argmin(d_truth_Q0))
    iB_truth = int(np.argmin(d_truth_Q1))
    truth_q_a = Q_0[iA_truth]
    truth_q_b = Q_1[iB_truth]
    truth_q_a_dist = float(d_truth_Q0[iA_truth])
    truth_q_b_dist = float(d_truth_Q1[iB_truth])

    # Cascade-derived ω from these truth-nearest q's
    om_truth_cascade = omega_from_pair(truth_q_a[None, :], truth_q_b[None, :],
                                          delta_t)[0]
    # True ω at t_0 (need full propagation to get body-frame ω at the right time)
    quats_full, omegas_full = propagate_attitude(
        q0_truth, omega0_truth, obs_times,
        mode="tumbling", inertia_tensor=inertia,
    )
    om_truth_at_t0 = omegas_full[t0_ep]
    om_truth_mag = float(np.linalg.norm(om_truth_at_t0))
    om_cascade_mag = float(np.linalg.norm(om_truth_cascade))
    om_diff = float(np.linalg.norm(om_truth_cascade - om_truth_at_t0))
    om_diff_rel = om_diff / max(om_truth_mag, 1e-12)
    om_dir_diff = float(np.degrees(np.arccos(np.clip(
        np.dot(om_truth_cascade, om_truth_at_t0) / (om_cascade_mag * om_truth_mag + 1e-12),
        -1.0, 1.0))))
    om_mag_rel = (om_cascade_mag - om_truth_mag) / om_truth_mag
    print(f"[stage4] truth_q_a dist in Q_0: {truth_q_a_dist:.3f}°")
    print(f"[stage4] truth_q_b dist in Q_1: {truth_q_b_dist:.3f}°")
    print(f"[stage4] cascade-derived ω from truth pair: |ω|={np.degrees(om_cascade_mag):.4f} dps")
    print(f"[stage4] truth ω at t_0:                    |ω|={np.degrees(om_truth_mag):.4f} dps")
    print(f"[stage4]                  Δ direction: {om_dir_diff:.3f}°")
    print(f"[stage4]                  Δ |ω| rel:   {om_mag_rel*100:+.3f}%")
    print(f"[stage4]                  Δ vec rel:   {om_diff_rel*100:.3f}%")

    # Question: is (truth_q_a, om_truth_cascade) in the kept hypothesis set?
    # The kept set was filtered by ω-bounds. Check membership against (qA_kept, om_kept).
    # Look for the hypothesis closest to (truth_q_a, om_truth_cascade).
    # First find rows in qA_kept that match truth_q_a (this is exact since qA_kept comes
    # from Q_0 and truth_q_a is one row of Q_0).
    matches_qa = np.all(qA_kept == truth_q_a[None, :], axis=1)
    n_matches_qa = int(matches_qa.sum())
    print(f"[stage4] hypotheses with q_a == truth_q_a: {n_matches_qa}")

    if n_matches_qa > 0:
        # among those, find the one with smallest distance to om_truth_cascade
        om_subset = om_kept[matches_qa]
        diffs = np.linalg.norm(om_subset - om_truth_cascade[None, :], axis=1)
        i_best_in_subset = int(np.argmin(diffs))
        i_best_global = int(np.where(matches_qa)[0][i_best_in_subset])
        nearest_dist = float(np.degrees(np.arccos(np.clip(
            np.dot(om_subset[i_best_in_subset], om_truth_cascade) /
            (np.linalg.norm(om_subset[i_best_in_subset]) * om_cascade_mag + 1e-12),
            -1.0, 1.0))))
        print(f"[stage4] nearest hypothesis ω-direction error vs cascade-truth: "
              f"{nearest_dist:.5f}°  (sanity: should be ≈0 if truth_q_b is in Q_1)")

        # And: does this best hypothesis (truth_q_a, ω_cascade) survive at each (tol, K)?
        cascade_truth_survives = {}
        for tol_mag in SWEEP_TOL_MAGS:
            for K_required in SWEEP_K_VALIDATION:
                if K_required > n_K:
                    continue
                pass_K = delta_mag[i_best_global, :K_required] < tol_mag
                cascade_truth_survives[f"tol{tol_mag:.2f}_K{K_required}"] = bool(pass_K.all())
        print(f"[stage4] cascade-truth survival per (tol, K):")
        for k, v in cascade_truth_survives.items():
            print(f"          {k}: {v}")
    else:
        print("[stage4] truth_q_a not in qA_kept (probably eliminated by ω-bounds filter); "
              "no cascade-truth survivor lookup possible.")
        cascade_truth_survives = {}
        i_best_global = -1
        nearest_dist = float("nan")

    # Best survivor in the OVERALL hypothesis set vs truth (regardless of which q_a)
    # (using the loosest sweep params: tol=0.30, K=2)
    best_overall = {}
    for tol_mag in SWEEP_TOL_MAGS:
        for K_required in SWEEP_K_VALIDATION:
            if K_required > n_K:
                continue
            pass_per_epoch = delta_mag < tol_mag
            survive_mask = pass_per_epoch[:, :K_required].all(axis=1)
            n_surv = int(survive_mask.sum())
            if n_surv == 0:
                continue
            # find the survivor closest to (q_truth_a, ω_truth)
            qA_surv = qA_kept[survive_mask]
            om_surv = om_kept[survive_mask]
            d_q = angular_dist_deg_batch(qA_surv, truth_q_a)
            d_om = np.linalg.norm(om_surv - om_truth_at_t0, axis=1) / max(om_truth_mag, 1e-12)
            # best by combined: q_dist + 30 * om_rel (heuristic)
            i_best = int(np.argmin(d_q + 30 * d_om))
            best_overall[f"tol{tol_mag:.2f}_K{K_required}"] = {
                "n_survivors": n_surv,
                "best_q_dist_truth_deg": float(d_q[i_best]),
                "best_om_rel_truth": float(d_om[i_best]),
            }
    print(f"[stage4] best survivor per (tol, K) by min(q_dist + 30·om_rel):")
    for k, v in best_overall.items():
        print(f"          {k}: n={v['n_survivors']:6d}  q_dist={v['best_q_dist_truth_deg']:.2f}°  "
              f"|Δω|/|ω|={v['best_om_rel_truth']*100:.2f}%")

    summary["stage4_truth_check"] = {
        "truth_q_a_dist_deg": truth_q_a_dist,
        "truth_q_b_dist_deg": truth_q_b_dist,
        "om_truth_at_t0_dps": float(np.degrees(om_truth_mag)),
        "om_cascade_dps": float(np.degrees(om_cascade_mag)),
        "om_dir_error_deg": om_dir_diff,
        "om_mag_error_rel": om_mag_rel,
        "om_vec_error_rel": om_diff_rel,
        "n_hypotheses_at_truth_qa": n_matches_qa,
        "cascade_truth_survives": cascade_truth_survives,
        "best_survivor_per_param": best_overall,
        "wall_s": time.time() - t4,
    }

    # ---- save cascade NPZ ----
    np.savez(OUT_DIR / "cascade.npz",
             qA_kept=qA_kept, om_kept=om_kept,
             delta_mag=delta_mag,
             val_eps=np.array(val_eps),
             t0_ep=t0_ep, t1_ep=t1_ep, delta_t=delta_t,
             truth_q_a=truth_q_a, truth_q_b=truth_q_b,
             om_truth_cascade=om_truth_cascade,
             om_truth_at_t0=om_truth_at_t0,
             best_idx_truth_q_a=i_best_global)
    print(f"[saved] {OUT_DIR / 'cascade.npz'}")

    summary["wall_total_s"] = time.time() - t_start
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"\n[done] total wall: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
