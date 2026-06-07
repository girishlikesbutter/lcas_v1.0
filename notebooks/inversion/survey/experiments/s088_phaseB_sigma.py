"""s088 Phase B sigma-sweep — does the BVP ω-dir error clear the s087 1.25° gate,
and how does it scale with anchor q-accuracy?

At each seed's conditioning sweet-spot Δt (fast 119: 60 ep; slow 116: 120 ep),
sweep the per-anchor q-cloud noise sigma and report BVP truth-branch ω-dir error.
Confirms the expected ω-dir error ∝ sigma_q scaling and locates the gate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

import lib.traj_load as tl
from lib.shoot import m048_inertia, omega_dir_err_deg, polhode_period
from lib.jacobi_propagator import propagate_jacobi_path2
from experiments.s088_phaseB_conditioning import _trial

OUT = Path(__file__).resolve().parent.parent / "results" / "s088"
INERTIA = m048_inertia()
I_A = 100
SWEET_DT = {119: 60, 116: 120}
SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.3]
N_NOISE = 300
GATE = 1.25


def main():
    out = []
    with Pool(24) as pool:
        for seed, de in SWEET_DT.items():
            d = tl.load_truth(seed)
            t0 = d["observation_times"].astype(np.float64); t0 = t0 - t0[0]
            q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
            q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, t0)
            w_true = w_hist[I_A]
            q_a, q_b = q_hist[I_A], q_hist[I_A + de]
            dt = float(t0[I_A + de] - t0[I_A])
            omdt = np.degrees(np.linalg.norm(w_true) * dt)
            print(f"\nseed {seed}  Δt={de}ep (|w|Δt={omdt:.0f}°):")
            rows = []
            for sig in SIGMAS:
                args = [(7000 + int(sig * 100) * 1000 + k, q_a, q_b, dt, w_true, sig)
                        for k in range(N_NOISE)]
                res = np.array(pool.starmap(_trial, args))
                bvp_dir = res[:, 0]
                med, p90 = float(np.median(bvp_dir)), float(np.percentile(bvp_dir, 90))
                rows.append(dict(sigma_q_deg=sig, bvp_dir_med=med, bvp_dir_p90=p90))
                print(f"  sigma_q={sig:.1f}°  BVP ω-dir med={med:5.2f}° p90={p90:5.2f}°"
                      f"  {'PASS' if med < GATE else '----'}")
            out.append(dict(seed=seed, dt_epochs=de, omega_dt_deg=omdt, rows=rows))
    with open(OUT / "phaseB_sigma.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'phaseB_sigma.json'}")


if __name__ == "__main__":
    main()
