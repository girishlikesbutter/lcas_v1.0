"""stage2_omega_factorization.py — recover (q0, ω) from stage 1 per-epoch candidates.

Given per-epoch candidate sets C_a, C_b at two tight anchors (t_a, t_b), for
every pair (q_a ∈ C_a, q_b ∈ C_b):

    q_rel = q_a^-1 ⊗ q_b
    axis n̂, principal angle θ_p ∈ [0, π] from q_rel
    for wrap k in {0, 1, 2} × sign s in {+1, -1}:
        |ω| = |θ_p + 2π k| / Δt
        if |ω| ∈ m048 domain [0.00175, 0.0262] rad/s:
            ω_at_ta = s · |ω| · n̂       # body-frame at t_a
            q0 = q_a ⊗ exp(-½ ω_at_ta · t_a)   # constant-ω back-prop init
            mag_pred = predict_lc(q0, ω_at_ta, mode='tumbling')
            mse = mean((mag_pred - mag_hifi)^2)
            save (q0, ω_at_ta, mse, meta)

Stage 2 is pairwise INITIALIZATION; the full-LC MSE via v2 (tumbling) is the judge.
Downstream NM polish on top-K candidates handles the constant-ω bias.

Outputs to data/.../13_clean_slate_omega/d_per_epoch_search/seed{NNN}/:
    stage2_candidates.npz — (N,4) q0, (N,3) omega_at_ta, (N,) mse, meta fields
    stage2_result.json    — summary scalars + truth recovery check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# BLAS caps BEFORE numpy — Pool workers inherit.
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

# m048 |ω| domain (rad/s): 0.1..1.5 dps.
OMEGA_MIN = np.radians(0.1)   # 0.001745 rad/s
OMEGA_MAX = np.radians(1.5)   # 0.02618 rad/s

PANEL_DEG = 0.0
DISH_DEG = 15.0


# ----------------------------------------------------------------------------
# Quaternion helpers (scalar-first w,x,y,z)
# ----------------------------------------------------------------------------


def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1
    return out


def qnormalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-30)


def qlog_axis_angle(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract principal angle θ_p ∈ [0, π] and axis n̂ from unit quaternion.

    Returns (theta_p, n) with shape (...,) and (..., 3). Always returns the
    non-negative principal angle so the sign is explicit at the call site.
    """
    q = qnormalize(q)
    # Handle sign ambiguity q ≡ -q: force w ≥ 0 so θ/2 ∈ [0, π/2].
    sgn = np.sign(q[..., 0:1])
    sgn = np.where(sgn == 0, 1.0, sgn)
    q = q * sgn
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    vnorm = np.linalg.norm(v, axis=-1)
    theta_p = 2.0 * np.arctan2(vnorm, w)  # ∈ [0, π] after sgn fix
    n = np.where(vnorm[..., None] > 1e-12, v / (vnorm[..., None] + 1e-30),
                 np.array([1.0, 0.0, 0.0]))
    return theta_p, n


def qexp_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Unit quaternion from axis n̂ and full rotation angle θ."""
    half = angle / 2.0
    c = np.cos(half)
    s = np.sin(half)
    return np.stack([c, s * axis[..., 0], s * axis[..., 1], s * axis[..., 2]],
                    axis=-1)


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


_W = {}


def _worker_init(seed: int):
    # With spawn context, workers don't inherit sys.path. Re-add.
    import sys as _sys
    from pathlib import Path as _Path
    _HERE = _Path(__file__).resolve().parent
    _sys.path.insert(0, str(_HERE.parent))
    _sys.path.insert(0, str(_Path.home() / "surrogate_model" / "surrogate_model"))
    _sys.path.insert(0, str(_HERE.parents[3]))  # project root for `src.*`
    from surrogate import SurrogateModel
    from src.dynamics.attitude_propagator import propagate_attitude
    from lib.data import load_seed as _load
    bundle = _load(seed)
    _W["bundle"] = bundle
    _W["model"] = SurrogateModel.load_default()
    _W["prop"] = propagate_attitude
    _W["dt"] = bundle["dt_sampling"]
    _W["I"] = bundle["inertia_tensor"]
    _W["mag_hifi"] = bundle["mag_hifi"]
    _W["obs_times_rel"] = bundle["observation_times"] - bundle["observation_times"][0]
    _W["sun_j2k"] = bundle["sun_j2k"]
    _W["obs_j2k"] = bundle["obs_j2k"]
    _W["sat_j2k"] = bundle["sat_j2k"]
    _W["obs_dist"] = bundle["obs_dist"]


def _score_candidate(task):
    """One task = one (q_a, q_b, k, sign) → (q0, ω_at_ta, mse, meta).

    We do the q0 back-prop via closed-form constant-ω (fast) — the full-LC
    forward uses tumbling mode which handles the real dynamics.
    """
    (q_a, q_b, t_a, q_a_idx, q_b_idx, k, sign, t_b) = task
    dt_sample = _W["dt"]

    # q_rel = q_a^-1 ⊗ q_b
    q_rel = qmul(qconj(q_a), q_b)
    theta_p, n_hat = qlog_axis_angle(q_rel)

    # Full accumulated rotation angle in body frame = θ_p + 2π·k.
    total_theta = theta_p + 2.0 * np.pi * k
    Dt = (t_b - t_a) * dt_sample
    omega_mag = total_theta / Dt  # rad/s
    if not (OMEGA_MIN <= omega_mag <= OMEGA_MAX):
        return None

    omega_at_ta = sign * omega_mag * n_hat  # body-frame at t_a

    # Back-prop q_a to t=0 via constant-ω principal-axis closed form.
    # q(0) = q_a ⊗ exp(-½ ω t_a)
    t_a_sec = t_a * dt_sample
    q_back = qexp_axis_angle(n_hat, -sign * omega_mag * t_a_sec)
    q0 = qmul(q_a, q_back)
    q0 = qnormalize(q0)

    # Now tumbling forward with ω_body at t=0 (needed). The back-prop via
    # constant-ω gave us q0 but doesn't tell us ω_body(0) under tumbling.
    # Approximation: use ω at t_a as ω at t=0 (OK for near-symmetric top where
    # ω_direction precesses but |ω| is constant). The tumbling forward then
    # handles actual precession from this init.
    try:
        quats, _ = _W["prop"](
            q0=q0.astype(np.float64),
            omega0=omega_at_ta.astype(np.float64),
            times=_W["obs_times_rel"].astype(np.float64),
            mode="tumbling",
            inertia_tensor=_W["I"].astype(np.float64),
        )
    except Exception as e:
        return None

    # Rotate inertial sun/obs vectors to body frame.
    n_epoch = len(quats)
    k1_body = np.empty((n_epoch, 3))
    k2_body = np.empty((n_epoch, 3))
    sun_vec = _W["sun_j2k"] - _W["sat_j2k"]
    obs_vec = _W["obs_j2k"] - _W["sat_j2k"]
    for i in range(n_epoch):
        w, x, y, z = quats[i]
        R_ib = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        k1 = R_ib @ sun_vec[i]
        k2 = R_ib @ obs_vec[i]
        k1_body[i] = k1 / (np.linalg.norm(k1) + 1e-30)
        k2_body[i] = k2 / (np.linalg.norm(k2) + 1e-30)

    mag_pred = np.asarray(_W["model"].predict_magnitude(
        k1_body, k2_body, PANEL_DEG, DISH_DEG, _W["obs_dist"]
    ), dtype=np.float64)

    mse = float(np.mean((mag_pred - _W["mag_hifi"]) ** 2))

    return (
        q0.astype(np.float32),
        omega_at_ta.astype(np.float32),
        mse,
        int(q_a_idx),
        int(q_b_idx),
        int(k),
        int(sign),
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-a", type=int, required=True,
                    help="First anchor epoch index (e.g., 313 for seed 0).")
    ap.add_argument("--anchor-b", type=int, required=True,
                    help="Second anchor epoch index (e.g., 400 for seed 0).")
    ap.add_argument("--top-n", type=int, default=50,
                    help="Keep top-N by residual in each anchor for pilot.")
    ap.add_argument("--wrap-k-max", type=int, default=2,
                    help="Max wrap count to enumerate (k ∈ {0, 1, ..., K}).")
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--label", type=str, default="pilot",
                    help="Tag for output files (stage2_{label}_*).")
    args = ap.parse_args()

    seed = args.seed
    t_a = args.anchor_a
    t_b = args.anchor_b

    print(f"seed={seed}  anchors: t_a={t_a}, t_b={t_b}")
    bundle = load_seed(seed)
    dt = bundle["dt_sampling"]
    Dt = (t_b - t_a) * dt
    print(f"Δt = {Dt:.2f} s  ({t_b - t_a} epochs)")
    print(f"truth |ω| = {bundle['omega_mag_dps']:.3f} dps "
          f"(|ω|·Δt = {np.radians(bundle['omega_mag_dps']) * Dt:.2f} rad = "
          f"{np.radians(bundle['omega_mag_dps']) * Dt / np.pi:.2f} π)")

    # Load stage 1 candidates
    cand_dir = (OUT_ROOT / f"seed{seed:03d}" / "anchor_scan_candidates")
    ca = np.load(cand_dir / f"epoch_{t_a:04d}.npz")
    cb = np.load(cand_dir / f"epoch_{t_b:04d}.npz")
    qa_all = ca["q"].astype(np.float64)
    qb_all = cb["q"].astype(np.float64)
    ra = ca["residual"]
    rb = cb["residual"]
    da = ca["truth_dist_deg"]
    db = cb["truth_dist_deg"]
    print(f"C_a: |C|={len(qa_all)}  min truth_dist={da.min():.2f}°")
    print(f"C_b: |C|={len(qb_all)}  min truth_dist={db.min():.2f}°")

    # Sort by residual, take top-N
    order_a = np.argsort(ra)[:args.top_n]
    order_b = np.argsort(rb)[:args.top_n]
    qa = qa_all[order_a]
    qb = qb_all[order_b]
    print(f"Selected top-{len(qa)} (by residual) of C_a and top-{len(qb)} of C_b")
    print(f"  truth_dist in top-{len(qa)} of C_a: "
          f"min={da[order_a].min():.2f}°  median={np.median(da[order_a]):.2f}°")
    print(f"  truth_dist in top-{len(qb)} of C_b: "
          f"min={db[order_b].min():.2f}°  median={np.median(db[order_b]):.2f}°")

    # Build task list
    tasks = []
    for ia in range(len(qa)):
        for ib in range(len(qb)):
            qa_vec = qa[ia]
            qb_vec = qb[ib]
            for k in range(args.wrap_k_max + 1):
                for sign in (+1, -1):
                    tasks.append((
                        qa_vec, qb_vec, t_a,
                        int(order_a[ia]), int(order_b[ib]),
                        k, sign,
                        t_b,   # extra for Dt compute
                    ))
    print(f"\nEnumerated {len(tasks)} (q_a, q_b, k, sign) tasks")
    print(f"  = {len(qa)} × {len(qb)} × {args.wrap_k_max + 1} × 2")

    # Parallel scoring
    print(f"\nScoring with Pool({args.n_workers})...")
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(seed,),
    ) as pool:
        results = pool.map(_score_candidate, tasks, chunksize=8)
    elapsed = time.perf_counter() - t0
    n_ok = sum(1 for r in results if r is not None)
    print(f"Scoring done: {elapsed:.1f} s ({elapsed / max(n_ok, 1) * 1000:.1f} ms/ok-cand)")
    print(f"  {n_ok}/{len(tasks)} passed |ω| domain filter")

    # Pack into arrays
    q0_arr = np.stack([r[0] for r in results if r is not None], axis=0)
    omega_arr = np.stack([r[1] for r in results if r is not None], axis=0)
    mse_arr = np.array([r[2] for r in results if r is not None], dtype=np.float64)
    qa_idx_arr = np.array([r[3] for r in results if r is not None], dtype=np.int32)
    qb_idx_arr = np.array([r[4] for r in results if r is not None], dtype=np.int32)
    k_arr = np.array([r[5] for r in results if r is not None], dtype=np.int8)
    sign_arr = np.array([r[6] for r in results if r is not None], dtype=np.int8)

    # Truth recovery metrics
    q0_true = bundle["q0_true"]
    omega_true = bundle["omega0_true"]

    def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        dot = np.abs(np.sum(q1 * q2, axis=-1))
        return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    q0_err = quat_geodesic_deg(q0_arr, q0_true[None, :])
    # ω at t=0 — for tumbling, our ω_at_ta propagated back is approximate.
    # For truth check, compare |ω| and direction at t=0 to truth via propagation
    # of (q0_arr, omega_arr) with tumbling from t=0 to... wait — we stored ω_at_ta.
    # Convert stored ω (at t_a) to init ω for propagation from q0 — actually we
    # used ω_at_ta as omega0 input to propagate_attitude starting at q0. That's
    # consistent with treating ω_at_ta as ω at t=0 too (constant-ω approximation).
    # So ω_err vs truth at t=0 is reasonable.
    omega_norm = np.linalg.norm(omega_arr, axis=-1)
    omega_true_norm = float(np.linalg.norm(omega_true))
    omega_dir_err = np.degrees(np.arccos(np.clip(
        np.sum(omega_arr * omega_true, axis=-1) /
        (omega_norm * omega_true_norm + 1e-30), -1.0, 1.0)))
    omega_mag_err = np.abs(omega_norm - omega_true_norm) / omega_true_norm

    # Sort by MSE, report top 20
    order = np.argsort(mse_arr)
    print(f"\n=== TOP 20 by MSE ===")
    print(f"{'rank':>4} {'mse':>10} {'q0_err°':>9} {'ω_dir°':>8} "
          f"{'ω_mag_err':>10} {'k':>2} {'sign':>5} {'qa_i':>5} {'qb_i':>5}")
    for rank, i in enumerate(order[:20]):
        print(f"{rank:>4} {mse_arr[i]:>10.4f} {q0_err[i]:>9.2f} "
              f"{omega_dir_err[i]:>8.2f} {omega_mag_err[i]:>10.3f} "
              f"{k_arr[i]:>2} {sign_arr[i]:>5} {qa_idx_arr[i]:>5} {qb_idx_arr[i]:>5}")

    print(f"\n=== TRUTH-CLOSEST by q0_err ===")
    order_q = np.argsort(q0_err)
    print(f"{'rank':>4} {'mse':>10} {'q0_err°':>9} {'ω_dir°':>8} "
          f"{'ω_mag_err':>10} {'k':>2} {'sign':>5}")
    for rank, i in enumerate(order_q[:10]):
        print(f"{rank:>4} {mse_arr[i]:>10.4f} {q0_err[i]:>9.2f} "
              f"{omega_dir_err[i]:>8.2f} {omega_mag_err[i]:>10.3f} "
              f"{k_arr[i]:>2} {sign_arr[i]:>5}")

    # Save all
    out_dir = OUT_ROOT / f"seed{seed:03d}"
    out_npz = out_dir / f"stage2_{args.label}_candidates.npz"
    np.savez_compressed(
        out_npz,
        q0=q0_arr,
        omega_at_ta=omega_arr,
        mse=mse_arr,
        qa_idx=qa_idx_arr,
        qb_idx=qb_idx_arr,
        wrap_k=k_arr,
        sign=sign_arr,
        q0_err_deg=q0_err.astype(np.float32),
        omega_dir_err_deg=omega_dir_err.astype(np.float32),
        omega_mag_rel_err=omega_mag_err.astype(np.float32),
        t_a=t_a, t_b=t_b, seed=seed,
        truth_q0=q0_true.astype(np.float32),
        truth_omega=omega_true.astype(np.float32),
    )
    print(f"\nSaved: {out_npz}")

    # Summary JSON
    best_i = int(order[0])
    truth_i = int(order_q[0])
    summary = {
        "seed": seed,
        "t_a": t_a, "t_b": t_b, "delta_t_s": Dt,
        "top_n": int(args.top_n),
        "wrap_k_max": int(args.wrap_k_max),
        "n_tasks": len(tasks),
        "n_scored": int(n_ok),
        "elapsed_s": round(elapsed, 2),
        "truth": {
            "q0_wxyz": [float(x) for x in q0_true],
            "omega_rad": [float(x) for x in omega_true],
            "omega_dps": round(bundle["omega_mag_dps"], 4),
        },
        "best_by_mse": {
            "rank": 0,
            "mse": float(mse_arr[best_i]),
            "q0_err_deg": float(q0_err[best_i]),
            "omega_dir_err_deg": float(omega_dir_err[best_i]),
            "omega_mag_rel_err": float(omega_mag_err[best_i]),
            "k": int(k_arr[best_i]), "sign": int(sign_arr[best_i]),
        },
        "truth_closest_q0": {
            "rank_by_mse": int(np.where(order == truth_i)[0][0]),
            "mse": float(mse_arr[truth_i]),
            "q0_err_deg": float(q0_err[truth_i]),
            "omega_dir_err_deg": float(omega_dir_err[truth_i]),
            "omega_mag_rel_err": float(omega_mag_err[truth_i]),
            "k": int(k_arr[truth_i]), "sign": int(sign_arr[truth_i]),
        },
    }
    out_json = out_dir / f"stage2_{args.label}_result.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
