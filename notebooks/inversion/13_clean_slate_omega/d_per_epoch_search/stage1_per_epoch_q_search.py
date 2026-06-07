"""stage1_per_epoch_q_search.py — 3-DOF attitude search per epoch (Pool(16)).

For each epoch t on a given seed:
  1. Known inertial geometry (k1_inertial(t), k2_inertial(t), phase(t)).
  2. Sweep (PAB_body on Fibonacci S², azimuth of k1 around PAB_body).
  3. Build (k1_body, k2_body) pairs on the cone of half-angle phase/2.
  4. Query v2 surrogate batched, filter by |mag_pred - mag_obs| < tol.
  5. For every passing sample compute the attitude quaternion.
  6. Record |C_t| and truth-survival metrics.

Parallelism: multiprocessing.Pool(16) with OMP=1 per worker. Each worker
loads v2 once in its initializer, processes (t, bundle_arrays, truth_q[t])
tasks, and saves per-epoch candidate NPZs to disk directly — avoids pickling
large arrays back to main.

Outputs under data/.../13_clean_slate_omega/d_per_epoch_search/seed{NNN}/:
  stage1_summary.npz      — per-epoch scalars: |C_t|, phi_deg, truth_dist,
                             min_residual, mag_obs.
  epochs/epoch_{t:04d}.npz — per-epoch candidate attitudes (q, residual);
                             only written when n_candidates > 0.
  stage1_topK.npz         — concatenated full candidates for the top-K
                             most-constrained epochs (for stage 3).
  stage1_result.json      — scalars summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# BLAS caps BEFORE numpy import (Pool workers inherit via fork).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path.home() / "surrogate_model" / "surrogate_model"))

from lib.data import load_seed  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
            / "13_clean_slate_omega" / "d_per_epoch_search")

PANEL_DEG = 0.0
DISH_DEG = 15.0
MAX_CANDIDATES_PER_EPOCH = 5000
TOP_K_FULL_CONCAT = 30


# ----------------------------------------------------------------------------
# Grid primitives + frame math
# ----------------------------------------------------------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def per_pab_perp_basis(pab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = pab.shape[0]
    idx_min = np.argmin(np.abs(pab), axis=-1)
    ref = np.zeros_like(pab)
    ref[np.arange(N), idx_min] = 1.0
    e_a_raw = np.cross(pab, ref)
    e_a = e_a_raw / np.linalg.norm(e_a_raw, axis=-1, keepdims=True)
    e_b = np.cross(pab, e_a)
    return e_a, e_b


def build_body_samples(pab_verts: np.ndarray, az_angles: np.ndarray,
                       phi_rad: float) -> tuple[np.ndarray, np.ndarray]:
    e_a, e_b = per_pab_perp_basis(pab_verts)
    cos_az = np.cos(az_angles)[None, :, None]
    sin_az = np.sin(az_angles)[None, :, None]
    cos_h = np.cos(phi_rad / 2.0)
    sin_h = np.sin(phi_rad / 2.0)
    e_hat = cos_az * e_a[:, None, :] + sin_az * e_b[:, None, :]
    pab_exp = pab_verts[:, None, :]
    k1_b = cos_h * pab_exp + sin_h * e_hat
    k2_b = cos_h * pab_exp - sin_h * e_hat
    return k1_b.reshape(-1, 3).copy(), k2_b.reshape(-1, 3).copy()


def construct_frame(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    k1 = np.atleast_2d(k1); k2 = np.atleast_2d(k2)
    e1 = k1
    dot = (k2 * k1).sum(axis=-1, keepdims=True)
    e2_raw = k2 - dot * k1
    e2 = e2_raw / np.clip(np.linalg.norm(e2_raw, axis=-1, keepdims=True), 1e-12, None)
    e3 = np.cross(e1, e2)
    F = np.stack([e1, e2, e3], axis=-1)
    return F[0] if F.shape[0] == 1 else F


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = r11 + r22 + r33
    w2 = np.maximum(0.0, 1.0 + tr) / 4.0
    x2 = np.maximum(0.0, 1.0 + r11 - r22 - r33) / 4.0
    y2 = np.maximum(0.0, 1.0 - r11 + r22 - r33) / 4.0
    z2 = np.maximum(0.0, 1.0 - r11 - r22 + r33) / 4.0
    case = np.argmax(np.stack([w2, x2, y2, z2], axis=-1), axis=-1)
    q = np.zeros(R.shape[:-2] + (4,), dtype=R.dtype)
    m0 = case == 0
    if np.any(m0):
        w = np.sqrt(w2[m0]); q[m0, 0] = w; c = 1.0 / (4.0 * np.clip(w, 1e-12, None))
        q[m0, 1] = (r32[m0] - r23[m0]) * c
        q[m0, 2] = (r13[m0] - r31[m0]) * c
        q[m0, 3] = (r21[m0] - r12[m0]) * c
    m1 = case == 1
    if np.any(m1):
        x = np.sqrt(x2[m1]); q[m1, 1] = x; c = 1.0 / (4.0 * np.clip(x, 1e-12, None))
        q[m1, 0] = (r32[m1] - r23[m1]) * c
        q[m1, 2] = (r12[m1] + r21[m1]) * c
        q[m1, 3] = (r13[m1] + r31[m1]) * c
    m2 = case == 2
    if np.any(m2):
        y = np.sqrt(y2[m2]); q[m2, 2] = y; c = 1.0 / (4.0 * np.clip(y, 1e-12, None))
        q[m2, 0] = (r13[m2] - r31[m2]) * c
        q[m2, 1] = (r12[m2] + r21[m2]) * c
        q[m2, 3] = (r23[m2] + r32[m2]) * c
    m3 = case == 3
    if np.any(m3):
        z = np.sqrt(z2[m3]); q[m3, 3] = z; c = 1.0 / (4.0 * np.clip(z, 1e-12, None))
        q[m3, 0] = (r21[m3] - r12[m3]) * c
        q[m3, 1] = (r13[m3] + r31[m3]) * c
        q[m3, 2] = (r23[m3] + r32[m3]) * c
    flip = q[..., 0] < 0
    q[flip] *= -1.0
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dp = np.abs((q1 * q2).sum(axis=-1))
    dp = np.clip(dp, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dp))


def truth_q_per_epoch(bundle: dict) -> np.ndarray:
    sun = bundle["sun_j2k"] - bundle["sat_j2k"]
    obs = bundle["obs_j2k"] - bundle["sat_j2k"]
    k1_i = sun / np.linalg.norm(sun, axis=-1, keepdims=True)
    k2_i = obs / np.linalg.norm(obs, axis=-1, keepdims=True)
    F_i = construct_frame(k1_i, k2_i)
    F_b = construct_frame(bundle["k1_body_true"], bundle["k2_body_true"])
    R = np.einsum("nij,nkj->nik", F_b, F_i)
    return rotmat_to_quat(R)


# ----------------------------------------------------------------------------
# Pool worker
# ----------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(pab_verts: np.ndarray, az_angles: np.ndarray,
                 out_epochs_dir: str, tol: float,
                 max_candidates: int):
    from surrogate import SurrogateModel
    _WORKER["model"] = SurrogateModel.load_default()
    _WORKER["pab_verts"] = pab_verts
    _WORKER["az_angles"] = az_angles
    _WORKER["out_dir"] = Path(out_epochs_dir)
    _WORKER["tol"] = float(tol)
    _WORKER["max_candidates"] = int(max_candidates)


def _task(payload: dict) -> dict:
    t = payload["t"]
    k1_i = payload["k1_i"]
    k2_i = payload["k2_i"]
    mag_obs = payload["mag_obs"]
    obs_dist = payload["obs_dist"]
    truth_q = payload["truth_q"]

    phi = float(np.arccos(np.clip(k1_i @ k2_i, -1.0, 1.0)))
    pab_verts = _WORKER["pab_verts"]
    az_angles = _WORKER["az_angles"]
    k1_b, k2_b = build_body_samples(pab_verts, az_angles, phi)
    N = k1_b.shape[0]

    dist_arr = np.full(N, obs_dist, dtype=np.float64)
    mag_pred = np.asarray(
        _WORKER["model"].predict_magnitude(k1_b, k2_b, PANEL_DEG, DISH_DEG, dist_arr),
        dtype=np.float64,
    )
    resid = np.abs(mag_pred - mag_obs)
    mask = resid < _WORKER["tol"]
    n_pass = int(mask.sum())

    if n_pass == 0:
        return {
            "t": int(t), "n_candidates": 0, "phi_deg": float(np.degrees(phi)),
            "mag_obs": float(mag_obs),
            "best_truth_dist_deg": float("nan"),
            "truth_in_set": False, "min_residual": float(resid.min()),
        }

    k1_p = k1_b[mask]
    k2_p = k2_b[mask]
    resid_p = resid[mask]
    if n_pass > _WORKER["max_candidates"]:
        order = np.argsort(resid_p)[:_WORKER["max_candidates"]]
        k1_p = k1_p[order]; k2_p = k2_p[order]; resid_p = resid_p[order]

    F_i = construct_frame(k1_i, k2_i)
    F_b = construct_frame(k1_p, k2_p)
    R = np.einsum("mij,kj->mik", F_b, F_i)
    q = rotmat_to_quat(R)
    dists = quat_geodesic_deg(q, truth_q[None, :])
    best = int(np.argmin(dists))

    # Save candidates to per-epoch NPZ (avoids pickling back to main)
    out = _WORKER["out_dir"] / f"epoch_{t:04d}.npz"
    np.savez_compressed(
        out,
        q=q.astype(np.float32),
        residual=resid_p.astype(np.float32),
        k1_body=k1_p.astype(np.float32),
        k2_body=k2_p.astype(np.float32),
    )
    return {
        "t": int(t),
        "n_candidates": n_pass,
        "phi_deg": float(np.degrees(phi)),
        "mag_obs": float(mag_obs),
        "best_truth_dist_deg": float(dists[best]),
        "truth_in_set": bool(dists[best] < 5.0),
        "min_residual": float(resid.min()),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pab", type=int, default=1500)
    ap.add_argument("--n-az", type=int, default=72)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, nargs="+", default=None)
    args = ap.parse_args()

    seed = args.seed
    bundle = load_seed(seed)
    N_epoch = bundle["observation_times"].shape[0]
    print(f"seed={seed}  N_epoch={N_epoch}  |ω|={bundle['omega_mag_dps']:.3f} dps")
    print(f"Grid: {args.n_pab} PAB × {args.n_az} az = {args.n_pab * args.n_az}/epoch")
    print(f"Tolerance: {args.tol} mag   Workers: {args.workers}")

    pab_verts = fibonacci_sphere(args.n_pab)
    az_angles = np.linspace(0.0, 2.0 * np.pi, args.n_az, endpoint=False)
    truth_q = truth_q_per_epoch(bundle)

    out_dir = OUT_ROOT / f"seed{seed:03d}"
    epochs_dir = out_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    # Clean out stale per-epoch NPZs from previous runs
    for p in epochs_dir.glob("epoch_*.npz"):
        p.unlink()

    # Precompute per-epoch payloads (small — just unit vectors and scalars)
    sun = bundle["sun_j2k"] - bundle["sat_j2k"]
    obs = bundle["obs_j2k"] - bundle["sat_j2k"]
    k1_i_all = sun / np.linalg.norm(sun, axis=-1, keepdims=True)
    k2_i_all = obs / np.linalg.norm(obs, axis=-1, keepdims=True)

    epochs = args.epochs if args.epochs is not None else list(range(N_epoch))
    payloads = []
    for t in epochs:
        payloads.append({
            "t": int(t),
            "k1_i": k1_i_all[t],
            "k2_i": k2_i_all[t],
            "mag_obs": float(bundle["mag_hifi"][t]),
            "obs_dist": float(bundle["obs_dist"][t]),
            "truth_q": truth_q[t],
        })

    t0 = time.perf_counter()
    results = []
    n_done = 0
    log_interval = max(1, len(payloads) // 20)
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(pab_verts, az_angles, str(epochs_dir), args.tol,
                           MAX_CANDIDATES_PER_EPOCH)) as pool:
        for r in pool.imap_unordered(_task, payloads, chunksize=4):
            results.append(r)
            n_done += 1
            if n_done % log_interval == 0 or n_done == len(payloads):
                el = time.perf_counter() - t0
                rate = n_done / max(el, 1e-6)
                eta = (len(payloads) - n_done) / max(rate, 1e-6)
                print(f"  [{n_done}/{len(payloads)}] t={r['t']}  "
                      f"|C|={r['n_candidates']:5d}  "
                      f"truth_dist={r['best_truth_dist_deg']:.2f}°  "
                      f"phi={r['phi_deg']:.1f}°  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")
    total_elapsed = time.perf_counter() - t0

    results.sort(key=lambda r: r["t"])
    t_arr = np.asarray([r["t"] for r in results], dtype=np.int32)
    n_cand = np.asarray([r["n_candidates"] for r in results], dtype=np.int32)
    phi_arr = np.asarray([r["phi_deg"] for r in results], dtype=np.float32)
    mag_obs_arr = np.asarray([r["mag_obs"] for r in results], dtype=np.float32)
    truth_dist = np.asarray([r["best_truth_dist_deg"] for r in results], dtype=np.float32)
    truth_in = np.asarray([r["truth_in_set"] for r in results], dtype=bool)
    min_res = np.asarray([r["min_residual"] for r in results], dtype=np.float32)

    summary_path = out_dir / "stage1_summary.npz"
    np.savez_compressed(
        summary_path,
        t=t_arr, n_candidates=n_cand, phi_deg=phi_arr,
        mag_obs=mag_obs_arr, best_truth_dist_deg=truth_dist,
        truth_in_set=truth_in, min_residual=min_res,
        grid_n_pab=args.n_pab, grid_n_az=args.n_az, tol=args.tol,
    )
    print(f"Saved: {summary_path}")

    # Top-K concatenated
    nonzero = np.where(n_cand > 0)[0]
    if len(nonzero) > 0:
        order_tight = nonzero[np.argsort(n_cand[nonzero])]
        keep_idx = order_tight[:TOP_K_FULL_CONCAT]
        kept_t = t_arr[keep_idx]
        qs, resids, k1s, k2s = [], [], [], []
        for t in kept_t:
            p = epochs_dir / f"epoch_{int(t):04d}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            qs.append(d["q"])
            resids.append(d["residual"])
            k1s.append(d["k1_body"])
            k2s.append(d["k2_body"])
        lengths = np.array([len(q) for q in qs], dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
        all_q = np.concatenate(qs, axis=0) if qs else np.zeros((0, 4))
        all_res = np.concatenate(resids, axis=0) if resids else np.zeros(0)
        all_k1 = np.concatenate(k1s, axis=0) if k1s else np.zeros((0, 3))
        all_k2 = np.concatenate(k2s, axis=0) if k2s else np.zeros((0, 3))
        topk_path = out_dir / "stage1_topK.npz"
        np.savez_compressed(
            topk_path,
            kept_t=kept_t.astype(np.int32),
            offsets=offsets, q=all_q, residual=all_res,
            k1_body=all_k1, k2_body=all_k2,
        )
        print(f"Saved: {topk_path}")

    report = {
        "seed": seed,
        "n_epoch": int(N_epoch),
        "grid": {"n_pab": args.n_pab, "n_az": args.n_az,
                 "samples_per_epoch": int(args.n_pab * args.n_az)},
        "tol": float(args.tol),
        "workers": int(args.workers),
        "total_elapsed_s": float(total_elapsed),
        "candidates_per_epoch": {
            "min": int(n_cand.min()) if n_cand.size else 0,
            "median": float(np.median(n_cand)),
            "mean": float(n_cand.mean()),
            "max": int(n_cand.max()) if n_cand.size else 0,
        },
        "truth_in_set_fraction": float(truth_in.mean()),
        "truth_dist_best_overall_deg": float(np.nanmin(truth_dist)),
        "truth_dist_median_deg": float(np.nanmedian(truth_dist)),
        "n_epochs_zero_candidates": int((n_cand == 0).sum()),
    }
    report_path = out_dir / "stage1_result.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")
    print(f"\nTotal wall time: {total_elapsed:.1f} s")
    print(f"|C_t|: min={report['candidates_per_epoch']['min']}  "
          f"median={report['candidates_per_epoch']['median']:.0f}  "
          f"max={report['candidates_per_epoch']['max']}")
    print(f"Truth in set: {truth_in.sum()}/{len(truth_in)} epochs "
          f"(median truth_dist={report['truth_dist_median_deg']:.2f}°)")


if __name__ == "__main__":
    main()
