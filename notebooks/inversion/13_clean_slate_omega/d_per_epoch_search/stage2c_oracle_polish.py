"""stage2c_oracle_polish.py — DIAGNOSTIC only.

Runs NM polish starting from the TRUTH-CLOSEST stage 2 candidates (by q0_err).
Tells us the upper bound: if the pipeline's best possible init can't reach
MSE<0.01, the framework is dead. If it CAN, the problem is discriminating
truth-close from wrong-basin candidates in the pool.

This is NOT the real algorithm (which must rank without truth). It's a
diagnostic to distinguish "basin is reachable" from "basin is unreachable".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
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

# Reuse polish primitives from stage2b
from stage2b_polish import _init, _polish_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", type=str, default="pilot")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of truth-closest candidates to polish.")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()

    seed = args.seed
    in_npz = OUT_ROOT / f"seed{seed:03d}" / f"stage2_{args.label}_candidates.npz"
    z = np.load(in_npz)
    q0_all = z["q0"]
    omega_all = z["omega_at_ta"]
    mse_all = z["mse"]
    q0_err_all = z["q0_err_deg"]

    order_truth = np.argsort(q0_err_all)[:args.top_k]
    print(f"ORACLE DIAGNOSTIC — polish top-{args.top_k} by q0_err (truth-selected)")
    print(f"Stage2 q0_err range: {q0_err_all[order_truth[0]]:.2f}° .. "
          f"{q0_err_all[order_truth[-1]]:.2f}°")
    print(f"Their MSE range:     {mse_all[order_truth].min():.4f} .. "
          f"{mse_all[order_truth].max():.4f}")

    tasks = [(q0_all[i].astype(np.float64), omega_all[i].astype(np.float64), int(i))
             for i in order_truth]

    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(
        processes=args.n_workers, initializer=_init, initargs=(seed,)
    ) as pool:
        results = pool.map(_polish_one, tasks, chunksize=1)
    print(f"\nPolished {len(results)} in {time.perf_counter() - t0:.1f} s")

    bundle = load_seed(seed)
    q0_true = bundle["q0_true"]
    omega_true = bundle["omega0_true"]
    omega_true_norm = float(np.linalg.norm(omega_true))

    q0_pol = np.stack([r[1] for r in results])
    om_pol = np.stack([r[2] for r in results])
    mse_pol = np.array([r[3] for r in results])
    niter = np.array([r[4] for r in results])
    cand_idx = np.array([r[0] for r in results])

    def qg(a, b):
        dot = np.abs(np.sum(a * b, axis=-1))
        return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    q0_err = qg(q0_pol.astype(np.float64), q0_true[None, :])
    on = np.linalg.norm(om_pol, axis=-1)
    w_dir = np.degrees(np.arccos(np.clip(
        np.sum(om_pol * omega_true, axis=-1) / (on * omega_true_norm + 1e-30),
        -1.0, 1.0)))
    w_mag = np.abs(on - omega_true_norm) / omega_true_norm

    print(f"\n=== POLISHED (oracle-selected) ===")
    print(f"{'orig_q0°':>9} {'pol_q0°':>9} {'mse_pre':>9} {'mse_pol':>9} "
          f"{'ω_dir°':>8} {'ω_mag_err':>10} {'niter':>6}")
    for i in range(len(results)):
        pre_q0 = q0_err_all[cand_idx[i]]
        pre_mse = mse_all[cand_idx[i]]
        print(f"{pre_q0:>9.2f} {q0_err[i]:>9.2f} {pre_mse:>9.4f} {mse_pol[i]:>9.4f} "
              f"{w_dir[i]:>8.2f} {w_mag[i]:>10.3f} {niter[i]:>6}")

    best = int(np.argmin(mse_pol))
    print(f"\nBEST polished: q0_err={q0_err[best]:.3f}° "
          f"ω_dir={w_dir[best]:.2f}° ω_mag_err={w_mag[best]:.4f} "
          f"MSE={mse_pol[best]:.6f}")
    if q0_err[best] < 5.0 and mse_pol[best] < 0.01:
        print("→ ORACLE OK: truth recoverable under NM polish from truth-close init.")
    elif q0_err[best] < 5.0:
        print(f"→ ORACLE PARTIAL: q0 near truth but MSE = {mse_pol[best]:.4f} > 0.01.")
    else:
        print(f"→ ORACLE FAIL: q0_err = {q0_err[best]:.2f}° even from truth-close init.")

    out_npz = OUT_ROOT / f"seed{seed:03d}" / f"stage2_{args.label}_oracle_polished.npz"
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
