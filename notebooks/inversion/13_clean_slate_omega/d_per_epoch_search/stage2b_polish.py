"""stage2b_polish.py — NM polish top-K stage 2 candidates on 7-DOF (q0, ω).

Loads stage2_{label}_candidates.npz, takes top-K by MSE, runs a bounded-ish
Nelder-Mead refinement in (q0_xyz, ω_xyz) parameterization (w reconstructed
from unit-norm constraint using sgn·sqrt(1 - |xyz|²)) minimising v2-LC MSE
under tumbling dynamics.

Each polish is a single NM call; for Pool parallelism we run K polishes across
workers. Cost per polish: ~40 NM iters × 1 LC eval ≈ 6 s at 150 ms/LC.
For K=100 on Pool(16): ~40 s.

Outputs: stage2_{label}_polished.npz with (q0, omega, mse, q0_err, w_err).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path.home() / "surrogate_model" / "surrogate_model"))
sys.path.insert(0, str(HERE.parents[3]))

from lib.data import load_seed  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
            / "13_clean_slate_omega" / "d_per_epoch_search")

PANEL_DEG = 0.0
DISH_DEG = 15.0
OMEGA_MAX = np.radians(1.5) * 1.5  # loose cap for NM — allows it to stray


_W = {}


def _init(seed: int):
    import sys as _sys
    from pathlib import Path as _Path
    _HERE = _Path(__file__).resolve().parent
    _sys.path.insert(0, str(_HERE.parent))
    _sys.path.insert(0, str(_Path.home() / "surrogate_model" / "surrogate_model"))
    _sys.path.insert(0, str(_HERE.parents[3]))
    from surrogate import SurrogateModel
    from src.dynamics.attitude_propagator import propagate_attitude
    from lib.data import load_seed as _load
    bundle = _load(seed)
    _W["bundle"] = bundle
    _W["model"] = SurrogateModel.load_default()
    _W["prop"] = propagate_attitude
    _W["obs_times_rel"] = bundle["observation_times"] - bundle["observation_times"][0]
    _W["sun_vec"] = bundle["sun_j2k"] - bundle["sat_j2k"]
    _W["obs_vec"] = bundle["obs_j2k"] - bundle["sat_j2k"]
    _W["obs_dist"] = bundle["obs_dist"]
    _W["I"] = bundle["inertia_tensor"]
    _W["mag_hifi"] = bundle["mag_hifi"]


def _q_from_xyz(xyz: np.ndarray, sgn: float) -> np.ndarray:
    s = float(np.linalg.norm(xyz))
    if s >= 1.0:
        xyz = xyz / s * 0.999999
    w = sgn * np.sqrt(max(0.0, 1.0 - np.dot(xyz, xyz)))
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)


def _lc_mse(q0: np.ndarray, omega: np.ndarray) -> float:
    try:
        quats, _ = _W["prop"](
            q0=q0, omega0=omega, times=_W["obs_times_rel"],
            mode="tumbling", inertia_tensor=_W["I"],
        )
    except Exception:
        return 1e6
    n = len(quats)
    k1 = np.empty((n, 3))
    k2 = np.empty((n, 3))
    for i in range(n):
        w, x, y, z = quats[i]
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        a = R @ _W["sun_vec"][i]
        b = R @ _W["obs_vec"][i]
        k1[i] = a / (np.linalg.norm(a) + 1e-30)
        k2[i] = b / (np.linalg.norm(b) + 1e-30)
    mag = np.asarray(_W["model"].predict_magnitude(
        k1, k2, PANEL_DEG, DISH_DEG, _W["obs_dist"]), dtype=np.float64)
    return float(np.mean((mag - _W["mag_hifi"]) ** 2))


def _polish_one(task):
    q0_init, omega_init, cand_idx = task
    sgn = np.sign(q0_init[0]) or 1.0
    x0 = np.concatenate([q0_init[1:], omega_init])

    def obj(x):
        xyz = x[:3]
        om = x[3:]
        if np.linalg.norm(om) > OMEGA_MAX:
            return 1e6
        q0 = _q_from_xyz(xyz, sgn)
        return _lc_mse(q0, om)

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 120})
    x = res.x
    q0 = _q_from_xyz(x[:3], sgn)
    om = x[3:]
    mse = float(res.fun)
    return (cand_idx, q0.astype(np.float32), om.astype(np.float32), mse, int(res.nit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", type=str, default="pilot")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--n-workers", type=int, default=16)
    args = ap.parse_args()

    seed = args.seed
    in_npz = OUT_ROOT / f"seed{seed:03d}" / f"stage2_{args.label}_candidates.npz"
    z = np.load(in_npz)
    q0_all = z["q0"]
    omega_all = z["omega_at_ta"]
    mse_all = z["mse"]

    order = np.argsort(mse_all)[:args.top_k]
    print(f"Loaded {len(mse_all)} stage 2 candidates; polishing top-{args.top_k}")
    print(f"  MSE range pre-polish: {mse_all[order[0]]:.4f} .. {mse_all[order[-1]]:.4f}")

    tasks = [(q0_all[i].astype(np.float64), omega_all[i].astype(np.float64), int(i))
             for i in order]

    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(
        processes=args.n_workers, initializer=_init, initargs=(seed,)
    ) as pool:
        results = pool.map(_polish_one, tasks, chunksize=2)
    print(f"Polished {len(results)} candidates in {time.perf_counter() - t0:.1f} s")

    bundle = load_seed(seed)
    q0_true = bundle["q0_true"]
    omega_true = bundle["omega0_true"]
    omega_true_norm = float(np.linalg.norm(omega_true))

    cand_idx = np.array([r[0] for r in results], dtype=np.int32)
    q0_pol = np.stack([r[1] for r in results])
    om_pol = np.stack([r[2] for r in results])
    mse_pol = np.array([r[3] for r in results])
    niter = np.array([r[4] for r in results], dtype=np.int32)

    def qg(a, b):
        dot = np.abs(np.sum(a * b, axis=-1))
        return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    q0_err = qg(q0_pol.astype(np.float64), q0_true[None, :])
    on = np.linalg.norm(om_pol, axis=-1)
    w_dir = np.degrees(np.arccos(np.clip(
        np.sum(om_pol * omega_true, axis=-1) / (on * omega_true_norm + 1e-30),
        -1.0, 1.0)))
    w_mag = np.abs(on - omega_true_norm) / omega_true_norm

    order2 = np.argsort(mse_pol)
    print(f"\n=== TOP 20 POLISHED by MSE ===")
    print(f"{'rank':>4} {'mse_pre':>9} {'mse_pol':>9} {'q0_err°':>9} "
          f"{'ω_dir°':>8} {'ω_mag_err':>10} {'niter':>6}")
    for rank, i in enumerate(order2[:20]):
        pre_mse = mse_all[cand_idx[i]]
        print(f"{rank:>4} {pre_mse:>9.4f} {mse_pol[i]:>9.4f} "
              f"{q0_err[i]:>9.2f} {w_dir[i]:>8.2f} {w_mag[i]:>10.3f} "
              f"{niter[i]:>6}")

    out_npz = OUT_ROOT / f"seed{seed:03d}" / f"stage2_{args.label}_polished.npz"
    np.savez_compressed(
        out_npz,
        q0=q0_pol, omega=om_pol, mse=mse_pol,
        cand_idx=cand_idx, niter=niter,
        q0_err_deg=q0_err.astype(np.float32),
        omega_dir_err_deg=w_dir.astype(np.float32),
        omega_mag_rel_err=w_mag.astype(np.float32),
    )
    print(f"\nSaved: {out_npz}")


if __name__ == "__main__":
    main()
