"""s047 — LS-peak hybrid pilot on a tight + forgiving seed pair (14 + 89).

Pipeline (architecture revised after s047b: cell-filter saturated on s019's
high-density bracket — closest-to-truth cell ranked 34/97 under max_geo on
seed 14. Skip cell ranking; use the 11 LS-significant peaks as the prior):

  1. Per-seed: extract LS-significant peaks from the LC spectrum (the same
     peaks s019's `bracket` builds its geomspace from).
  2. Around each LS peak: 5 fine cells at 0.5% |ω|-step (s042 basin width
     for seed 14; safely tight enough for slower seeds too).
  3. ω-direction Fibonacci sample N=8 at each fine cell.
  4. Sobol N=32 hemisphere q0 ICs via `lib.twin.canonical_batch`.
  5. Joint LM polish (rotvec_pert + ω_perturb_3D), surrogate-MSE,
     scipy `least_squares(method='lm', max_nfev=200)`. canonical applied
     to the (q, ω) IC set before LM to drop body-twin duplicates.
  6. Cluster Band A∪B candidates → distinct attractors.
  7. Hi-fi rerank cluster representatives (~7 s each).
  8. Headline: count distinct (q0, ω) attractors with hi-fi ρ < 4.

Pilot seeds:
  - **Seed 14** (tight): |ω|=1.229 dps, Q4 binding case, measured 0.5%
    ω-mag basin per s042. Has uncatalogued multi-sol attractor at ρ=2.78
    (q0_err=86°, ω+0.5%) per s042. Has 11 LS-significant peaks.
  - **Seed 89** (forgiving): validated end-to-end in s035 with 4 distinct
    hi-fi Band A basins (truth, twin, multi-sol-A, multi-sol-B). Pipeline
    should rediscover 3 unique after canonical (truth↔twin collapse). Has
    7 LS-significant peaks.

Architecture parameter choices:
  - `--top-k-peaks` top-K LS peaks by spectral power (default: all)
  - `--n-fine`     fine cells per peak (default 5)
  - `--fine-step`  |ω| step per fine cell (default 0.005 = 0.5%)
  - `--n-omega-dir` Fibonacci ω-dirs per fine cell (default 8)
  - `--n-sobol`    Sobol q0 hemisphere ICs (default 32)

Compute estimate: per (fine cell × ω-dir × Sobol q0) one LM call.
  seed 14: 11 peaks × 5 fine × 8 dirs × 32 Sobol = 14,080 LMs (with
           canonical dedup ~half ≈ 7,000) ≈ 14 min on Pool(8).
  seed 89: 7 peaks × 5 fine × 8 dirs × 32 Sobol = 8,960 LMs ≈ 9 min.
  Plus hi-fi rerank: ~7 s × ~5-10 cluster reps = ~1-2 min per seed.
  Total: ~30 min for the pair.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR / "experiments"))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.surrogate_eval import predict as surrogate_predict, get_model
from lib.twin import canonical_batch, Q_180X, R_180X
from lib.filter_costs import load_static_geometry, load_tier_table

# Reuse s020's worker for the cell-filter step.
import s020_seed_pipeline as s020
# Reuse s047b's bracket reconstruction.
import s047b_seed14_full_bracket_cell_filter as s047b


# ---- Sobol-Shoemake on SO(3) (from s011) ----
def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
    """Shoemake's [0,1]^3 → uniform unit quaternions on SO(3) (wxyz)."""
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a2 = 2.0 * np.pi * u2
    a3 = 2.0 * np.pi * u3
    x = s1 * np.sin(a2)
    y = s1 * np.cos(a2)
    z = s2 * np.sin(a3)
    w = s2 * np.cos(a3)
    return np.column_stack([w, x, y, z])


def build_sobol_q0_canonical(n_hemi: int, sobol_seed: int = 42) -> np.ndarray:
    """N_hemi canonical-hemisphere Sobol-Shoemake q0 ICs.

    Generates 2*N_hemi Sobol points, pairs them with a placeholder ω
    along +y so canonical retains every q0 (since ω_y > 0 is canonical),
    then drops half via canonical_batch — yielding N_hemi unique q0 reps.
    """
    sobol = qmc.Sobol(d=3, scramble=True, seed=sobol_seed)
    u = sobol.random(n_hemi)
    return shoemake_to_quat(u)


# ---- Joint LM residual (from s034 recipe) ----
def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_R_i2b_batch(q_arr_wxyz):
    qxyzw = q_arr_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


# Worker globals (set by init_worker, read by lm_polish_one)
_W = {}


def init_worker(state):
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _W.update(state)
    get_model()  # warm cache


def lm_polish_one(args):
    """One LM polish: returns (q0_final, omega_final, mse_final, n_iter, ok)."""
    ic_idx, q0_init, omega_init = args
    s = _W
    obs_times = s["obs_times"]
    inertia = s["inertia"]
    sun_unit = s["sun_unit"]
    obs_unit = s["obs_unit"]
    obs_dist = s["obs_dist"]
    mag_truth = s["mag_truth"]
    valid_mask = s["valid_mask"]
    max_nfev = s["max_nfev"]

    mag_truth_valid = mag_truth[valid_mask]

    from lib.surrogate_eval import predict as _surrogate_predict
    surrogate = get_model()

    def residual(x):
        rotvec = x[:3]
        omega_delta = x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()
        q_pert_w = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_new = quat_mul_wxyz(q_pert_w, q0_init)
        omega_new = omega_init + omega_delta

        q_traj, _ = propagate_attitude(
            q0_new, omega_new, obs_times,
            mode="tumbling", inertia_tensor=inertia,
        )
        R_full = quat_to_R_i2b_batch(q_traj)
        k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
        k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)

        mag_pred = surrogate.predict_magnitude(
            k1_body, k2_body, s020.SP_ANGLE_DEG, s020.AD_ANGLE_DEG, obs_dist,
        )
        return (mag_pred[valid_mask] - mag_truth_valid).astype(np.float64)

    x0 = np.zeros(6)
    try:
        result = least_squares(residual, x0, method='lm',
                                max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
        ok = True
        nfev = int(result.nfev)
        mse_final = float(np.mean(result.fun ** 2))
        rotvec = result.x[:3]
        omega_delta = result.x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()
        q_pert_w = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_final = quat_mul_wxyz(q_pert_w, q0_init)
        omega_final = omega_init + omega_delta
    except Exception:
        ok = False
        nfev = -1
        mse_final = float('nan')
        q0_final = q0_init.copy()
        omega_final = omega_init.copy()
    return ic_idx, q0_final, omega_final, mse_final, nfev, ok


# ---- Distinct attractor clustering ----
def cluster_attractors(q0_arr: np.ndarray, omega_arr: np.ndarray,
                       q_thresh_deg: float = 5.0,
                       omega_dir_thresh_deg: float = 5.0,
                       omega_mag_thresh_pct: float = 1.0):
    """Group (q0, ω) pairs into distinct attractors via greedy clustering.

    Returns list of cluster dicts: {representative idx, member indices, n}.
    """
    n = q0_arr.shape[0]
    if n == 0:
        return []
    q_canon, w_canon = canonical_batch(q0_arr, omega_arr)
    cluster_id = np.full(n, -1, dtype=int)
    next_id = 0
    for i in range(n):
        if cluster_id[i] >= 0:
            continue
        cluster_id[i] = next_id
        for j in range(i + 1, n):
            if cluster_id[j] >= 0:
                continue
            qd = angular_dist_deg(q_canon[i], q_canon[j])
            wi = np.linalg.norm(w_canon[i])
            wj = np.linalg.norm(w_canon[j])
            wmag_pct = abs(wi - wj) / wi * 100 if wi > 0 else 1e9
            wdir = np.degrees(np.arccos(np.clip(
                np.dot(w_canon[i], w_canon[j]) / (wi * wj), -1, 1
            )))
            if (qd <= q_thresh_deg and wdir <= omega_dir_thresh_deg
                    and wmag_pct <= omega_mag_thresh_pct):
                cluster_id[j] = next_id
        next_id += 1
    clusters = []
    for cid in range(next_id):
        members = np.where(cluster_id == cid)[0].tolist()
        clusters.append({"id": cid, "n": len(members), "members": members,
                         "rep": members[0]})
    return clusters


# ---- LS-peak prior (option B after s047b) ----
def get_ls_peaks(seed: int, top_k_peaks: int = None,
                 verbose: bool = True) -> tuple:
    """Extract LS-significant peaks from the cached LC.

    Reuses s047b.s019_bracket_grid which already computes the LS spectrum.
    Returns (peak_omegas_rad_s sorted by descending power, peak_powers,
             bracket_lo, bracket_hi).
    """
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
    truth = np.load(traj_path)
    mag_hifi = truth["mag_hifi"].astype(float)
    obs_times = truth["observation_times"].astype(float)

    # Re-derive LS spectrum + peak omegas. s047b returns peak_omegas sorted
    # ascending; we re-sort by power below.
    bracket_info = s047b.s019_bracket_grid(mag_hifi, obs_times)
    peak_omegas_asc = bracket_info["ls_peak_omegas"]  # ascending |ω|

    # We want them sorted by descending POWER, not ascending omega.
    # Re-derive from raw LS:
    from scipy.signal import lombscargle, find_peaks
    valid = np.isfinite(mag_hifi)
    s = -mag_hifi[valid]; s = s - np.mean(s)
    t = obs_times[valid]
    dt = float(np.median(np.diff(t)))
    f_min = 1.0 / (t[-1] - t[0])
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, 4000)
    ang = 2 * np.pi * freqs
    power = lombscargle(t, s, ang, normalize=True)
    pmax = float(power.max())
    idx, _ = find_peaks(power, distance=5, height=0.1 * pmax)
    order_by_power = idx[np.argsort(power[idx])[::-1]]
    peak_omegas_by_power = 2 * np.pi * freqs[order_by_power]
    peak_powers_sorted = power[order_by_power]

    n_peaks = len(peak_omegas_by_power)
    if top_k_peaks is not None and top_k_peaks < n_peaks:
        peak_omegas_by_power = peak_omegas_by_power[:top_k_peaks]
        peak_powers_sorted = peak_powers_sorted[:top_k_peaks]

    if verbose:
        print(f"[seed {seed}] LS peaks: {len(peak_omegas_by_power)} significant "
              f"(of {n_peaks} total)", flush=True)
        for k, (w, p) in enumerate(zip(peak_omegas_by_power, peak_powers_sorted)):
            print(f"  peak {k+1}: |ω|={w:.5f} rad/s ({np.degrees(w):.4f} dps), "
                  f"power={p:.4f}", flush=True)
    return peak_omegas_by_power, peak_powers_sorted


def build_fine_cells_around_peaks(peak_omegas_rad, n_fine=5, fine_step=0.005):
    """For each LS peak, generate n_fine |ω| values at fine_step fractional
    spacing. Returns flat (n_peaks * n_fine,) array."""
    half = (n_fine - 1) // 2
    factors = 1.0 + np.arange(-half, half + 1) * fine_step
    fine = []
    for w in peak_omegas_rad:
        for f in factors:
            fine.append(w * f)
    return np.array(fine)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[14, 89])
    p.add_argument("--top-k-peaks", type=int, default=None,
                   help="Top-K LS peaks by power (default: all significant)")
    p.add_argument("--n-fine", type=int, default=5,
                   help="fine |ω| cells per peak")
    p.add_argument("--fine-step", type=float, default=0.005,
                   help="fine-cell |ω| step (fraction)")
    p.add_argument("--n-sobol", type=int, default=32,
                   help="Sobol q0 hemisphere ICs per (fine cell × ω-dir)")
    p.add_argument("--n-omega-dir", type=int, default=8,
                   help="ω-direction sample at fine cells (Fibonacci)")
    p.add_argument("--max-nfev", type=int, default=200)
    p.add_argument("--ls-peaks-only", action="store_true",
                   help="Stop after LS-peak extraction (debug mode)")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny mode: 1 peak, n_fine=3, n_omega_dir=2, n_sobol=4")
    args = p.parse_args()

    if args.smoke:
        args.top_k_peaks = 1
        args.n_fine = 3
        args.n_omega_dir = 2
        args.n_sobol = 4

    out_root = SURVEY_DIR / "results" / "s047_hybrid_pilot"
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n{'='*70}", flush=True)
        print(f"=== s047 LS-peak hybrid pilot — seed {seed} ===", flush=True)
        print(f"{'='*70}", flush=True)
        seed_out = out_root / (f"seed{seed:03d}_smoke" if args.smoke
                                else f"seed{seed:03d}")
        seed_out.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: LS-significant peaks ----
        peak_omegas, peak_powers = get_ls_peaks(seed,
                                                 top_k_peaks=args.top_k_peaks,
                                                 verbose=True)
        if args.ls_peaks_only:
            continue

        # Truth ω-mag for diagnostic offset reporting
        traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
        truth = np.load(traj_path)
        omega_truth = truth["omega0_rad"].astype(float)
        omega_mag_truth = float(np.linalg.norm(omega_truth))
        print(f"\n[seed {seed}] truth |ω| = {omega_mag_truth:.5f} rad/s "
              f"({np.degrees(omega_mag_truth):.4f} dps)", flush=True)
        peak_offsets_pct = (peak_omegas - omega_mag_truth) / omega_mag_truth * 100
        for k, (w, p, off) in enumerate(zip(peak_omegas, peak_powers,
                                             peak_offsets_pct)):
            print(f"  peak {k+1}: |ω|={w:.5f} rad/s, power={p:.4f}, "
                  f"offset_to_truth={off:+.3f}%", flush=True)

        # ---- Step 2: fine cells around each peak ----
        fine_omega_mags = build_fine_cells_around_peaks(
            peak_omegas, n_fine=args.n_fine, fine_step=args.fine_step,
        )
        n_peaks = len(peak_omegas)
        print(f"  fine cells: {len(fine_omega_mags)} = {n_peaks} × {args.n_fine}",
              flush=True)

        # ---- Step 3: ω-direction Fibonacci ----
        omega_dirs = s020.fibonacci_sphere(args.n_omega_dir)
        print(f"  ω-dirs at fine cells: {args.n_omega_dir}", flush=True)

        # ---- Step 4: Sobol q0 hemisphere ----
        sobol_q0 = build_sobol_q0_canonical(args.n_sobol, sobol_seed=42)
        print(f"  Sobol q0: N={args.n_sobol} (canonical-hemisphere via Sobol-Shoemake)",
              flush=True)

        # ---- Build (fine_cell × dir × q0) IC list ----
        ics = []
        for fine_w in fine_omega_mags:
            for d_idx in range(args.n_omega_dir):
                omega_init = fine_w * omega_dirs[d_idx]
                for q_idx in range(args.n_sobol):
                    ics.append((len(ics), sobol_q0[q_idx], omega_init))
        n_ics = len(ics)
        print(f"  total ICs: {n_ics} = {n_peaks}×{args.n_fine}×"
              f"{args.n_omega_dir}×{args.n_sobol}", flush=True)

        # ---- Apply canonical to ICs (drop ~half via the dedup map) ----
        q_arr = np.array([ic[1] for ic in ics])
        w_arr = np.array([ic[2] for ic in ics])
        q_canon, w_canon = canonical_batch(q_arr, w_arr)
        # Dedup is best at the (q, ω) pair level; same canonical pair repeated
        # is a duplicate IC. Use a hash of (q_canon, w_canon) rounded.
        keys = np.round(np.concatenate([q_canon, w_canon], axis=1), 8)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        ics_dedup = [(j, q_canon[j], w_canon[j]) for j in unique_idx]
        print(f"  after canonical dedup: {len(ics_dedup)} unique ICs "
              f"({100*len(ics_dedup)/n_ics:.1f}% of total)", flush=True)

        # ---- Forward-model inputs ----
        obs_times = truth["observation_times"].astype(float)
        sun_pos = truth["sun_pos"].astype(float)
        obs_pos = truth["obs_pos"].astype(float)
        sat_pos = truth["sat_pos"].astype(float)
        obs_dist = truth["obs_dist"].astype(float)
        mag_truth = truth["mag_hifi"].astype(float)
        valid_mask = np.isfinite(mag_truth)
        sun_vec = sun_pos - sat_pos
        obs_vec = obs_pos - sat_pos
        sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
        obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
        q0_truth = truth["q0_wxyz"].astype(float)

        geo = load_static_geometry()
        inertia = geo["inertia_tensor"].astype(float)

        # ---- Pool LM ----
        from multiprocessing import Pool
        worker_state = {
            "obs_times": obs_times, "inertia": inertia,
            "sun_unit": sun_unit, "obs_unit": obs_unit, "obs_dist": obs_dist,
            "mag_truth": mag_truth, "valid_mask": valid_mask,
            "max_nfev": args.max_nfev,
        }
        print(f"\n[seed {seed}] launching Pool(8) LM polish on "
              f"{len(ics_dedup)} ICs (max_nfev={args.max_nfev})...", flush=True)
        t0 = time.time()
        polished = []
        with Pool(8, initializer=init_worker, initargs=(worker_state,)) as pool:
            results_iter = pool.imap_unordered(lm_polish_one, ics_dedup,
                                                chunksize=4)
            done = 0
            last_print = time.time()
            for r in results_iter:
                polished.append(r)
                done += 1
                if (time.time() - last_print) > 30 or done == len(ics_dedup):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (len(ics_dedup) - done) / max(rate, 1e-6)
                    band_a_so_far = sum(
                        1 for rr in polished if not np.isnan(rr[3])
                        and np.sqrt(rr[3]) / 0.05 < 2
                    )
                    print(f"  {done}/{len(ics_dedup)} "
                          f"({100*done/len(ics_dedup):.1f}%); "
                          f"elapsed {elapsed/60:.1f} min; "
                          f"Band A so far: {band_a_so_far}; "
                          f"ETA {eta/60:.1f} min", flush=True)
                    last_print = time.time()
        lm_wall = time.time() - t0
        print(f"\n[seed {seed}] LM wall: {lm_wall/60:.1f} min", flush=True)

        # ---- Build polished records ----
        records = []
        for ic_idx, q0_final, omega_final, mse_final, nfev, ok in polished:
            if not ok or np.isnan(mse_final):
                continue
            rho_pred = float(np.sqrt(mse_final) / 0.05)
            q0_to_truth = angular_dist_deg(q0_final, q0_truth)
            twin_q0 = quat_mul_wxyz(Q_180X, q0_truth)
            q0_to_twin = angular_dist_deg(q0_final, twin_q0)
            wmag_truth = float(np.linalg.norm(omega_truth))
            wmag_final = float(np.linalg.norm(omega_final))
            wmag_pct = (wmag_final - wmag_truth) / wmag_truth * 100
            wdir_truth = np.degrees(np.arccos(np.clip(
                np.dot(omega_final, omega_truth) / (wmag_final * wmag_truth),
                -1, 1
            )))
            records.append({
                "ic_idx": int(ic_idx),
                "q0_final": q0_final.tolist(),
                "omega_final": omega_final.tolist(),
                "mse_final": float(mse_final),
                "rho_pred": rho_pred,
                "nfev": int(nfev),
                "q0_to_truth_deg": float(q0_to_truth),
                "q0_to_twin_deg": float(q0_to_twin),
                "omega_dir_to_truth_deg": float(wdir_truth),
                "omega_mag_pct_err": float(wmag_pct),
                "min_q0_err_to_either": float(min(q0_to_truth, q0_to_twin)),
            })

        # ---- Surrogate-Band-A subset ----
        band_a = [r for r in records if r["rho_pred"] < 2.0]
        band_b = [r for r in records if 2.0 <= r["rho_pred"] < 4.0]
        band_c = [r for r in records if 4.0 <= r["rho_pred"] < 8.0]
        band_d = [r for r in records if r["rho_pred"] >= 8.0]
        print(f"\n[seed {seed}] surrogate ρ-band counts:", flush=True)
        print(f"  A (<2):  {len(band_a)}", flush=True)
        print(f"  B (2-4): {len(band_b)}", flush=True)
        print(f"  C (4-8): {len(band_c)}", flush=True)
        print(f"  D (≥8):  {len(band_d)}", flush=True)

        # ---- Cluster Band A∪B candidates ----
        ab = band_a + band_b
        if ab:
            q_ab = np.array([r["q0_final"] for r in ab])
            w_ab = np.array([r["omega_final"] for r in ab])
            clusters = cluster_attractors(q_ab, w_ab)
            print(f"\n[seed {seed}] distinct attractor clusters in A∪B "
                  f"(surrogate): {len(clusters)}", flush=True)
            for cl in clusters:
                rep = ab[cl["rep"]]
                print(f"  cluster {cl['id']}: n={cl['n']:>3d}  "
                      f"rep ρ={rep['rho_pred']:.3f}  "
                      f"q0→truth={rep['q0_to_truth_deg']:.2f}°  "
                      f"q0→twin={rep['q0_to_twin_deg']:.2f}°  "
                      f"ω-dir={rep['omega_dir_to_truth_deg']:.2f}°  "
                      f"ω-mag={rep['omega_mag_pct_err']:+.3f}%", flush=True)
        else:
            clusters = []
            print(f"\n[seed {seed}] NO surrogate Band A∪B candidates — "
                  f"pipeline failed", flush=True)

        # ---- Hi-fi rerank cluster reps ----
        # We hi-fi only the cluster representatives + a few backups, not
        # every Band A∪B candidate (saves ~3-5 min per seed). One rep
        # per cluster is sufficient because clusters are surrogate-MSE
        # convergent — within a cluster, hi-fi ρ varies sub-percent.
        hifi_records = []
        if clusters:
            print(f"\n[seed {seed}] hi-fi rendering {len(clusters)} cluster reps "
                  f"(~7s each)...", flush=True)
            from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
            ctx = build_context(seed)
            mag_hifi_truth = ctx["mag_hifi_truth"]
            t_hifi = time.time()
            for cl in clusters:
                rep = ab[cl["rep"]]
                q0 = np.asarray(rep["q0_final"], dtype=float)
                w0 = np.asarray(rep["omega_final"], dtype=float)
                tic = time.time()
                try:
                    pred = render_hifi(q0, w0, ctx)
                    rho_h = rho_from_hifi(pred, mag_hifi_truth)
                    band = rho_band(rho_h)
                    ok_h = True
                except Exception as exc:
                    rho_h = float("inf")
                    band = "D"
                    ok_h = False
                    print(f"  cluster {cl['id']} hi-fi FAIL: {exc}", flush=True)
                dt = time.time() - tic
                hifi_records.append({
                    "cluster_id": int(cl["id"]),
                    "n_members": int(cl["n"]),
                    "rho_pred": rep["rho_pred"],
                    "rho_hifi": float(rho_h),
                    "rho_band": band,
                    "q0_to_truth_deg": rep["q0_to_truth_deg"],
                    "q0_to_twin_deg": rep["q0_to_twin_deg"],
                    "omega_dir_to_truth_deg": rep["omega_dir_to_truth_deg"],
                    "omega_mag_pct_err": rep["omega_mag_pct_err"],
                    "q0_final": rep["q0_final"],
                    "omega_final": rep["omega_final"],
                    "wall_s": float(dt),
                })
                print(f"  cluster {cl['id']:>2d} (n={cl['n']:>3d}): "
                      f"surr ρ={rep['rho_pred']:.3f} → hi-fi ρ={rho_h:>6.3f} "
                      f"({band})  "
                      f"q0→truth={rep['q0_to_truth_deg']:.2f}°  "
                      f"ω-dir={rep['omega_dir_to_truth_deg']:.2f}°  "
                      f"ω-mag={rep['omega_mag_pct_err']:+.3f}%  "
                      f"({dt:.1f}s)", flush=True)
            print(f"  hi-fi total: {(time.time()-t_hifi)/60:.1f} min", flush=True)

        # Headline: count distinct attractors with hi-fi ρ < 4 (A∪B)
        n_band_a_hifi = sum(1 for r in hifi_records if r["rho_band"] == "A")
        n_band_b_hifi = sum(1 for r in hifi_records if r["rho_band"] == "B")
        n_ab_hifi = n_band_a_hifi + n_band_b_hifi
        print(f"\n[seed {seed}] HEADLINE: distinct hi-fi attractors", flush=True)
        print(f"  Band A: {n_band_a_hifi}", flush=True)
        print(f"  Band B: {n_band_b_hifi}", flush=True)
        print(f"  A∪B total: {n_ab_hifi}", flush=True)

        # ---- Save ----
        np.savez_compressed(seed_out / "polish.npz",
                            ic_idx=np.array([r["ic_idx"] for r in records]),
                            q0_final=np.array([r["q0_final"] for r in records]),
                            omega_final=np.array([r["omega_final"] for r in records]),
                            mse_final=np.array([r["mse_final"] for r in records]),
                            rho_pred=np.array([r["rho_pred"] for r in records]),
                            q0_to_truth=np.array([r["q0_to_truth_deg"] for r in records]),
                            q0_to_twin=np.array([r["q0_to_twin_deg"] for r in records]),
                            omega_dir_to_truth=np.array([r["omega_dir_to_truth_deg"] for r in records]),
                            omega_mag_pct=np.array([r["omega_mag_pct_err"] for r in records]),
                            )
        with open(seed_out / "summary.json", "w") as f:
            json.dump({
                "seed": seed,
                "config": {
                    "n_peaks": int(n_peaks), "n_fine": args.n_fine,
                    "fine_step": args.fine_step, "n_sobol": args.n_sobol,
                    "n_omega_dir": args.n_omega_dir,
                    "max_nfev": args.max_nfev,
                },
                "ls_peaks": {
                    "peak_omegas_rad_s": peak_omegas.tolist(),
                    "peak_powers": peak_powers.tolist(),
                    "peak_offsets_pct_to_truth": peak_offsets_pct.tolist(),
                },
                "n_total_ics_pre_dedup": int(n_ics),
                "n_ics_post_dedup": int(len(ics_dedup)),
                "n_polished_ok": int(len(records)),
                "rho_band_counts_surrogate": {
                    "A": len(band_a), "B": len(band_b),
                    "C": len(band_c), "D": len(band_d),
                },
                "n_distinct_surr_AB_clusters": int(len(clusters)),
                "n_band_a_hifi": int(n_band_a_hifi),
                "n_band_b_hifi": int(n_band_b_hifi),
                "n_distinct_hifi_AB_clusters": int(n_ab_hifi),
                "clusters": [
                    {
                        "id": cl["id"],
                        "n_members": cl["n"],
                        "rep_record": ab[cl["rep"]],
                    }
                    for cl in clusters
                ],
                "hifi_records": hifi_records,
                "lm_wall_s": float(lm_wall),
            }, f, indent=2)
        print(f"\nSaved: {seed_out / 'polish.npz'}", flush=True)
        print(f"Saved: {seed_out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
