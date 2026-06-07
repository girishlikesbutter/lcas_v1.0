"""s048 — peak-cascade smoke test on seed 89 (single-seed run).

User idea: at each LC local minimum (peak in brightness), the (q, ω) state
is constrained because the brightness function has a critical point there.
- Tier 0: per-epoch tightness sanity (one peak, 100k random q, single-mag
  filter at peak epoch).
- Tier 1: joint (q_peak, ω) sampling + 3-epoch consistency filter under
  CONSTANT-ω over the short ±dt window.
- Tier 2: ONE chain hop — propagate each surviving (q_peak, ω) under true
  IS-901 Euler dynamics to the next peak, filter against that peak's
  3-epoch signature.
- Tier 3: chain forward across all peaks, plot |S_p| vs hop.

Pass/fail thresholds (written before running):
- Tier 0: |C_peak|/100k < 0.1                    (single-epoch tight enough)
- Tier 1: per-peak selectivity < 1e-3 AND truth survives
- Tier 2: decimation < 0.5 AND truth survives
- Tier 3: |S_final| ≤ 10 AND truth ∈ survivors AND ≥1 survivor lands Band A∪B

Conventions: scalar-first quaternions (w,x,y,z); LEFT-multiply for q-update
under conv (a) per src/dynamics/attitude_propagator.py.
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from lib.surrogate_eval import predict as surrogate_predict, get_model
from lib.twin import canonical_batch
from lib.filter_costs import load_static_geometry
from src.dynamics.attitude_propagator import propagate_attitude

# ---- config ----
SEED = 89
N_SAMPLES = 500_000                 # per epoch
TOLERANCE_MAG = 0.10                # 2× noise floor — accommodates surrogate error
PROMINENCE_MIN = 0.10               # filter noise-floor peaks
OMEGA_AGREE_REL = 0.05              # constant-ω consistency: |ω_ap − ω_pb|/|ω_ap| < this
OMEGA_MAG_BOUNDS_DPS = (0.05, 2.0)  # cohort range +slack
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
RNG_SEED = 42

OUT_DIR = SURVEY_DIR / "results" / "s048_peak_cascade_smoke" / f"seed{SEED:03d}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def quat_to_R_i2b_batch(q_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz → (N, 3, 3) inertial→body rotation matrices."""
    qxyzw = q_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batched Hamilton product (wxyz)."""
    if q1.ndim == 1:
        q1 = np.tile(q1, (q2.shape[0], 1))
    if q2.ndim == 1:
        q2 = np.tile(q2, (q1.shape[0], 1))
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_exp_batch(omega: np.ndarray, dt: float) -> np.ndarray:
    """Closed-form principal-axis rotation: q_rot = [cos(θ/2), sin(θ/2)*axis]
    where θ = |ω|·dt. Vectorized over (N, 3) ω."""
    omega = np.asarray(omega).reshape(-1, 3)
    om_mag = np.linalg.norm(omega, axis=1)
    theta = om_mag * dt
    half = 0.5 * theta
    safe = om_mag > 1e-12
    axis = np.zeros_like(omega)
    axis[safe] = omega[safe] / om_mag[safe, None]
    sin_h = np.sin(half)
    return np.column_stack([np.cos(half), sin_h * axis[:, 0], sin_h * axis[:, 1], sin_h * axis[:, 2]])


def constant_omega_propagate(q0_wxyz: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """q(dt) = quat_exp(omega·dt) ⊗ q0  (LEFT mul, conv (a))."""
    q_rot = quat_exp_batch(omega, dt)
    return quat_mul_batch(q_rot, q0_wxyz)


def predict_mags_at_epoch(q_wxyz: np.ndarray, sun_vec: np.ndarray, obs_vec: np.ndarray,
                          obs_dist: float) -> np.ndarray:
    """For N candidate q's at ONE epoch (single sun_vec, obs_vec, obs_dist),
    return N predicted mags."""
    R = quat_to_R_i2b_batch(q_wxyz)
    sun_unit = sun_vec / np.linalg.norm(sun_vec)
    obs_unit = obs_vec / np.linalg.norm(obs_vec)
    k1 = R @ sun_unit
    k2 = R @ obs_unit
    N = q_wxyz.shape[0]
    return surrogate_predict(k1, k2, np.full(N, obs_dist),
                              SP_ANGLE_DEG, AD_ANGLE_DEG)


def predict_mags_per_candidate_per_epoch(q_traj_per_cand_per_epoch: np.ndarray,
                                          sun_vec_per_epoch: np.ndarray,
                                          obs_vec_per_epoch: np.ndarray,
                                          obs_dist_per_epoch: np.ndarray) -> np.ndarray:
    """q_traj: (N_cand, N_ep, 4). returns (N_cand, N_ep) mags.
    Each candidate has its OWN q at each epoch; sun/obs/dist are shared across cands."""
    N_cand, N_ep, _ = q_traj_per_cand_per_epoch.shape
    flat = q_traj_per_cand_per_epoch.reshape(-1, 4)
    R = quat_to_R_i2b_batch(flat).reshape(N_cand, N_ep, 3, 3)
    sun_unit = sun_vec_per_epoch / np.linalg.norm(sun_vec_per_epoch, axis=1, keepdims=True)
    obs_unit = obs_vec_per_epoch / np.linalg.norm(obs_vec_per_epoch, axis=1, keepdims=True)
    # k1[i,j,:] = R[i,j,:,:] @ sun_unit[j,:]
    k1 = np.einsum('ijab,jb->ija', R, sun_unit).reshape(-1, 3)
    k2 = np.einsum('ijab,jb->ija', R, obs_unit).reshape(-1, 3)
    obs_dist_flat = np.tile(obs_dist_per_epoch, N_cand)
    mags = surrogate_predict(k1, k2, obs_dist_flat, SP_ANGLE_DEG, AD_ANGLE_DEG)
    return mags.reshape(N_cand, N_ep)


def main():
    t_start = time.time()

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
    pe = np.asarray(d["hifi_peak_epochs"], int)
    pp = np.asarray(d["hifi_peak_prominences"], float)
    dt = float(d["dt_sampling"])

    sun_vec_j2000 = sun_pos - sat_pos                  # (N_obs, 3) J2000
    obs_vec_j2000 = obs_pos - sat_pos
    geo = load_static_geometry()
    inertia = np.asarray(geo["inertia_tensor"], float)

    # filter peaks by prominence + must have at least 1-epoch buffer on each side
    valid = (pp >= PROMINENCE_MIN) & (pe >= 1) & (pe <= len(obs_times) - 2)
    pe = pe[valid]
    pp = pp[valid]
    order = np.argsort(-pp)
    pe = pe[order]; pp = pp[order]
    print(f"[peaks] kept {len(pe)} peaks (prominence ≥ {PROMINENCE_MIN}); ω={omega_mag_dps:.3f} dps")
    print(f"[peaks] top-5 (epoch, prom, mag): " + ", ".join(
        f"({int(e)}, {p:.2f}, {mag_hifi[int(e)]:.2f})" for e, p in zip(pe[:5], pp[:5])))

    summary = {
        "seed": SEED,
        "omega_mag_dps": omega_mag_dps,
        "n_peaks_kept": int(len(pe)),
        "tolerance_mag": TOLERANCE_MAG,
        "n_samples": N_SAMPLES,
        "tiers": {},
    }

    # ============================================================
    # Tier 0 — single-epoch tightness sanity
    # ============================================================
    print("\n=== Tier 0 — single-epoch tightness ===")
    t0 = time.time()
    peak_idx = int(pe[0])
    print(f"[tier0] peak epoch idx={peak_idx} (t={obs_times[peak_idx]:.1f}s, "
          f"mag={mag_hifi[peak_idx]:.3f}, prom={pp[0]:.2f})")

    rng = np.random.default_rng(RNG_SEED)
    R_random = Rotation.random(N_SAMPLES, random_state=rng).as_quat()  # xyzw
    q_random = R_random[:, [3, 0, 1, 2]]                                # → wxyz

    pred_t0 = predict_mags_at_epoch(q_random, sun_vec_j2000[peak_idx],
                                      obs_vec_j2000[peak_idx], obs_dist_all[peak_idx])
    measured_t0 = mag_hifi[peak_idx]
    survive_t0 = np.abs(pred_t0 - measured_t0) < TOLERANCE_MAG
    n_t0 = int(survive_t0.sum())
    frac_t0 = n_t0 / N_SAMPLES
    pass_t0 = frac_t0 < 0.1
    print(f"[tier0] survivors: {n_t0}/{N_SAMPLES} = {frac_t0*100:.2f}%  (PASS<10%: {pass_t0})")
    print(f"[tier0] wall: {time.time()-t0:.2f}s")
    summary["tiers"]["tier0"] = {
        "peak_epoch": peak_idx, "peak_prom": float(pp[0]), "measured_mag": float(measured_t0),
        "survivors": n_t0, "fraction": frac_t0, "pass": bool(pass_t0),
        "wall_s": time.time() - t0,
    }
    # save
    np.savez(OUT_DIR / "tier0.npz", q_random=q_random, pred=pred_t0,
             survive=survive_t0, peak_epoch=peak_idx)
    print(f"[saved] {OUT_DIR / 'tier0.npz'}")

    # ============================================================
    # Tier 1 — per-epoch C_t filter + triple constant-ω consistency
    # ============================================================
    print("\n=== Tier 1 — per-epoch C_t × constant-ω triple ===")
    t1 = time.time()
    # 3-epoch window centered at the peak
    eps_3 = np.array([peak_idx - 1, peak_idx, peak_idx + 1], dtype=int)
    measured_3 = mag_hifi[eps_3]                                  # (3,)
    sun_3 = sun_vec_j2000[eps_3]                                   # (3, 3)
    obs_3 = obs_vec_j2000[eps_3]
    dist_3 = obs_dist_all[eps_3]
    print(f"[tier1] 3-epoch mags (a, peak, b) = {measured_3}")

    # Sample N_SAMPLES random q on SO(3); filter at each of 3 epochs independently.
    rng = np.random.default_rng(RNG_SEED + 1)
    R_random = Rotation.random(N_SAMPLES, random_state=rng).as_quat()
    q_pool = R_random[:, [3, 0, 1, 2]]                             # wxyz
    print(f"[tier1] sampled {N_SAMPLES} q on SO(3); filtering per epoch...")

    C_q = []  # list of 3 arrays of surviving q's (wxyz)
    for j, ep in enumerate(eps_3):
        pred = predict_mags_at_epoch(q_pool, sun_vec_j2000[ep], obs_vec_j2000[ep],
                                      obs_dist_all[ep])
        keep = np.abs(pred - measured_3[j]) < TOLERANCE_MAG
        C_q.append(q_pool[keep])
        print(f"[tier1]   C_{['a','peak','b'][j]} (ep={ep}, mag={measured_3[j]:.2f}): "
              f"{int(keep.sum())} survivors")

    nA, nP, nB = len(C_q[0]), len(C_q[1]), len(C_q[2])
    if nA == 0 or nP == 0 or nB == 0:
        print("[tier1] empty C at one epoch; cascade cannot proceed.")
        summary["tiers"]["tier1"] = {"abort": "empty_C_t", "nA": nA, "nP": nP, "nB": nB,
                                       "wall_s": time.time() - t1}
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return

    # Truth (q, ω) at peak epoch from cached propagation
    q_peak_truth = quats_truth[peak_idx]
    quats_full, omegas_full = propagate_attitude(
        q0_truth, omega0_truth, obs_times,
        mode="tumbling", inertia_tensor=inertia,
    )
    omega_peak_truth = omegas_full[peak_idx]
    om_truth_mag = float(np.linalg.norm(omega_peak_truth))

    # Truth survival in each C_t — i.e., does the discrete q-sample include something near truth?
    def min_q_dist(arr, ref):
        if len(arr) == 0: return float('inf')
        d = np.abs(arr @ ref); d = np.clip(d, 0.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(d.max())))
    print(f"[tier1] truth-q distance to nearest C member: "
          f"a={min_q_dist(C_q[0], quats_truth[eps_3[0]]):.2f}°, "
          f"peak={min_q_dist(C_q[1], q_peak_truth):.2f}°, "
          f"b={min_q_dist(C_q[2], quats_truth[eps_3[2]]):.2f}°")

    # ---- Triple consistency: derive ω_ap and ω_pb, check agreement ----
    # ω from quaternion difference: q_b ≈ quat_exp(ω·dt) ⊗ q_a  (LEFT mul, conv (a))
    #   ⇒ q_rot = q_b ⊗ q_a^{-1}
    #   ⇒ ω = (2/dt) · log(q_rot) where log returns the rotation vector.
    def omega_from_pair(qA, qB, dt):
        """Vectorized: qA, qB shape (N, 4) wxyz → ω shape (N, 3) rad/s."""
        # q_a inverse: scalar same, vector negated (unit quat)
        qA_inv = qA.copy()
        qA_inv[:, 1:] *= -1.0
        # q_rot = qB ⊗ qA_inv
        w1, x1, y1, z1 = qB[:, 0], qB[:, 1], qB[:, 2], qB[:, 3]
        w2, x2, y2, z2 = qA_inv[:, 0], qA_inv[:, 1], qA_inv[:, 2], qA_inv[:, 3]
        qr = np.column_stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        # antipode normalize so w >= 0  (otherwise log gives flipped axis & angle reflex)
        flip = qr[:, 0] < 0
        qr[flip] *= -1
        w = np.clip(qr[:, 0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        s = np.sin(angle / 2.0)
        axis = np.zeros((qr.shape[0], 3))
        ok = s > 1e-9
        axis[ok] = qr[ok, 1:] / s[ok, None]
        return (angle / dt)[:, None] * axis

    # For each (q_a, q_peak) pair: ω_ap. For each (q_peak, q_b) pair: ω_pb.
    # Cross-tabulate: for each q_peak, find pairs (q_a, q_b) with matching ω.
    # Compute all pairs: nA × nP and nP × nB ω vectors. Then for each q_peak, intersect.
    print(f"[tier1] computing ω_ap ({nA*nP} pairs) and ω_pb ({nP*nB} pairs)...")
    qA_grid = np.repeat(C_q[0], nP, axis=0)        # (nA*nP, 4)
    qP_grid = np.tile(C_q[1], (nA, 1))
    om_ap_all = omega_from_pair(qA_grid, qP_grid, dt).reshape(nA, nP, 3)

    qP2_grid = np.repeat(C_q[1], nB, axis=0)
    qB_grid = np.tile(C_q[2], (nP, 1))
    om_pb_all = omega_from_pair(qP2_grid, qB_grid, dt).reshape(nP, nB, 3)

    # For each peak-q, look at all (ω_ap, ω_pb) outer pairs; accept if |Δω|/|ω| < threshold
    # AND |ω| in cohort bounds.
    om_lo = np.deg2rad(OMEGA_MAG_BOUNDS_DPS[0])
    om_hi = np.deg2rad(OMEGA_MAG_BOUNDS_DPS[1])

    accepted = []   # list of (qA_idx, qP_idx, qB_idx, omega_avg)
    for ip in range(nP):
        om_ap = om_ap_all[:, ip, :]                  # (nA, 3)
        om_pb = om_pb_all[ip, :, :]                  # (nB, 3)
        # Filter by ω-mag bounds
        mag_ap = np.linalg.norm(om_ap, axis=1)
        mag_pb = np.linalg.norm(om_pb, axis=1)
        ok_ap = (mag_ap > om_lo) & (mag_ap < om_hi)
        ok_pb = (mag_pb > om_lo) & (mag_pb < om_hi)
        if ok_ap.sum() == 0 or ok_pb.sum() == 0:
            continue
        # Outer difference: (nA, nB, 3)
        diff = om_ap[ok_ap, None, :] - om_pb[None, ok_pb, :]
        diff_mag = np.linalg.norm(diff, axis=2)
        ref_mag = mag_ap[ok_ap, None]
        rel = diff_mag / np.maximum(ref_mag, 1e-12)
        match = rel < OMEGA_AGREE_REL
        if match.any():
            ia_local, ib_local = np.where(match)
            ia = np.where(ok_ap)[0][ia_local]
            ib = np.where(ok_pb)[0][ib_local]
            for k in range(len(ia)):
                om_avg = 0.5 * (om_ap[ia[k]] + om_pb[ib[k]])
                accepted.append((int(ia[k]), int(ip), int(ib[k]), om_avg))

    n_t1 = len(accepted)
    pass_t1_sel = (n_t1 < 1000) and (n_t1 > 0)
    print(f"[tier1] peaking_trips_p: {n_t1} accepted triples")

    truth_q_min = float('nan'); truth_om_rel_min = float('nan'); truth_survives = False
    if n_t1 > 0:
        s_qpeak = np.array([C_q[1][a[1]] for a in accepted])
        s_om = np.array([a[3] for a in accepted])
        # truth survival
        dots = np.abs(s_qpeak @ q_peak_truth); dots = np.clip(dots, 0.0, 1.0)
        q_dist = np.degrees(2.0 * np.arccos(dots))
        om_diff = np.linalg.norm(s_om - omega_peak_truth, axis=1)
        om_rel = om_diff / max(om_truth_mag, 1e-9)
        truth_q_min = float(q_dist.min())
        truth_om_rel_min = float(om_rel.min())
        truth_survives = (truth_q_min < 10.0) and (truth_om_rel_min < 0.10)
        print(f"[tier1] nearest survivor to truth: q={truth_q_min:.2f}°, "
              f"|Δω|/|ω|={truth_om_rel_min*100:.2f}%   truth_survives={truth_survives}")
    else:
        s_qpeak = np.empty((0, 4)); s_om = np.empty((0, 3))

    print(f"[tier1] wall: {time.time()-t1:.2f}s")
    summary["tiers"]["tier1"] = {
        "nA": nA, "nP": nP, "nB": nB,
        "n_triples": n_t1, "pass": bool(pass_t1_sel),
        "truth_q_min_deg": truth_q_min, "truth_om_rel_min": truth_om_rel_min,
        "truth_survives": bool(truth_survives),
        "wall_s": time.time() - t1,
    }
    np.savez(OUT_DIR / "tier1.npz",
             C_q_a=C_q[0], C_q_peak=C_q[1], C_q_b=C_q[2],
             s_qpeak=s_qpeak, s_om=s_om,
             eps_3=eps_3, measured_3=measured_3,
             q_peak_truth=q_peak_truth, omega_peak_truth=omega_peak_truth)
    print(f"[saved] {OUT_DIR / 'tier1.npz'}")

    if n_t1 == 0:
        print("\nTier 1 zero triples. Aborting before Tier 2.")
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"[saved] {OUT_DIR / 'summary.json'}")
        return

    # Map for Tier 2: survivors are (s_qpeak, s_om) pairs
    q_canon = s_qpeak; om_canon = s_om
    survive_t1 = np.ones(len(s_qpeak), dtype=bool)

    # ============================================================
    # Tier 2 — one chain hop via true Euler propagation
    # ============================================================
    print("\n=== Tier 2 — one chain hop (Euler propagation) ===")
    t2 = time.time()

    # Cap the survivor set (in case Tier 1 was loose)
    MAX_TIER2 = 5000
    surv_idx = np.where(survive_t1)[0]
    if len(surv_idx) > MAX_TIER2:
        rng = np.random.default_rng(RNG_SEED + 2)
        surv_idx = rng.choice(surv_idx, MAX_TIER2, replace=False)
        print(f"[tier2] subsampled {MAX_TIER2}/{n_t1} survivors")
    s_q = q_canon[surv_idx]
    s_om = om_canon[surv_idx]
    n_surv = len(surv_idx)

    # Pick next peak
    if len(pe) < 2:
        print("[tier2] only one peak; skipping Tier 2.")
        return
    peak1_idx = int(pe[1])
    eps_3_p1 = np.array([peak1_idx - 1, peak1_idx, peak1_idx + 1], dtype=int)
    measured_3_p1 = mag_hifi[eps_3_p1]
    sun_3_p1 = sun_vec_j2000[eps_3_p1]
    obs_3_p1 = obs_vec_j2000[eps_3_p1]
    dist_3_p1 = obs_dist_all[eps_3_p1]
    print(f"[tier2] peak0 idx={peak_idx} (t={obs_times[peak_idx]:.1f}s) → "
          f"peak1 idx={peak1_idx} (t={obs_times[peak1_idx]:.1f}s); "
          f"Δt={obs_times[peak1_idx]-obs_times[peak_idx]:.1f}s")

    # For each survivor: propagate (q_peak, ω) under Euler from t_peak0 to t_peak1
    # Using src.dynamics.attitude_propagator.propagate_attitude (one solve per cand).
    # Times: relative-to-q0 — we provide times shifted so t=0 is the survivor's epoch.
    t_target = obs_times[peak1_idx] - obs_times[peak_idx]
    # Build per-survivor times array — must include t=0 to integrate from there.
    times_int = np.array([0.0, t_target])

    s_q_at_p1 = np.zeros_like(s_q)
    s_om_at_p1 = np.zeros((n_surv, 3))
    print(f"[tier2] running {n_surv} Euler propagations...")
    for i in range(n_surv):
        qs, oms = propagate_attitude(
            s_q[i], s_om[i], times_int, mode="tumbling", inertia_tensor=inertia,
        )
        s_q_at_p1[i] = qs[-1]
        s_om_at_p1[i] = oms[-1]
        if (i + 1) % 1000 == 0:
            print(f"[tier2]   {i+1}/{n_surv}  wall {time.time()-t2:.1f}s")

    # Now apply 3-epoch filter at peak1: constant-ω over ±dt around peak1 using
    # propagated ω at peak1
    q_p1_minus = constant_omega_propagate(s_q_at_p1, -s_om_at_p1, dt)
    q_p1_plus = constant_omega_propagate(s_q_at_p1, s_om_at_p1, dt)
    q_traj_p1 = np.stack([q_p1_minus, s_q_at_p1, q_p1_plus], axis=1)
    mags_p1 = predict_mags_per_candidate_per_epoch(q_traj_p1, sun_3_p1, obs_3_p1, dist_3_p1)
    diff_p1 = np.abs(mags_p1 - measured_3_p1[None, :])
    survive_t2 = np.all(diff_p1 < TOLERANCE_MAG, axis=1)
    n_t2 = int(survive_t2.sum())
    decimation = n_t2 / max(n_surv, 1)
    pass_t2_dec = decimation < 0.5
    print(f"[tier2] survivors: {n_t2}/{n_surv} = {decimation*100:.2f}% (decimation factor)")
    print(f"[tier2] PASS decimation<0.5: {pass_t2_dec}")

    # Truth survival at peak1
    q_peak1_truth = quats_truth[peak1_idx]
    omega_peak1_truth = omegas_full[peak1_idx]
    if n_t2 > 0:
        s2_q = s_q_at_p1[survive_t2]
        s2_om = s_om_at_p1[survive_t2]
        dots = np.abs(s2_q @ q_peak1_truth)
        dots = np.clip(dots, 0.0, 1.0)
        q_dist2 = np.degrees(2.0 * np.arccos(dots))
        om_diff2 = np.linalg.norm(s2_om - omega_peak1_truth, axis=1)
        om_truth_mag1 = np.linalg.norm(omega_peak1_truth)
        om_rel2 = om_diff2 / max(om_truth_mag1, 1e-9)
        truth_q_min2 = float(q_dist2.min())
        truth_om_rel_min2 = float(om_rel2.min())
        truth_survives2 = (truth_q_min2 < 10.0) and (truth_om_rel_min2 < 0.10)
        print(f"[tier2] nearest survivor to truth at peak1: q-dist={truth_q_min2:.2f}°, "
              f"|Δω|/|ω|={truth_om_rel_min2*100:.2f}%   truth_survives={truth_survives2}")
    else:
        truth_survives2 = False
        truth_q_min2 = float("nan"); truth_om_rel_min2 = float("nan")

    print(f"[tier2] wall: {time.time()-t2:.2f}s")
    summary["tiers"]["tier2"] = {
        "n_survivors_in": n_surv, "n_survivors_out": n_t2, "decimation": decimation,
        "pass_decimation": bool(pass_t2_dec),
        "truth_q_min_deg": truth_q_min2, "truth_om_rel_min": truth_om_rel_min2,
        "truth_survives": bool(truth_survives2),
        "wall_s": time.time() - t2,
    }
    np.savez(OUT_DIR / "tier2.npz",
             surv_idx=surv_idx, s_q_at_p1=s_q_at_p1, s_om_at_p1=s_om_at_p1,
             survive=survive_t2, peak1_idx=peak1_idx,
             q_peak1_truth=q_peak1_truth, omega_peak1_truth=omega_peak1_truth)
    print(f"[saved] {OUT_DIR / 'tier2.npz'}")

    summary["wall_total_s"] = time.time() - t_start
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"\n[done] total wall: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
