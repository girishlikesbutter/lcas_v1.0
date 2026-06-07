"""s088 Gate A2 — multiplicity of the attitude BVP vs Δt (multi-start enumeration).

Gate A1 showed a single finite-diff-init shoot recovers truth only at small
|w|Δt, then aliases. A2 asks the structural question: how many DISTINCT omega
connect q_a -> q_b at each Δt, and is truth always among them (recoverable by
multi-start)?

Method: shoot from a grid of inits = Fibonacci(N_dir) directions x geomspace
magnitudes spanning the cohort |w| range. Keep connected roots (geo < 1e-4 deg),
dedup in omega-space (dir < 2 deg AND |w| within 1%). Report n_distinct, whether
truth is recovered, and the |w| spread of the alias family.

Pure closed-form propagation; Pool(24), BLAS threads pinned to 1.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import numpy as np

import lib.traj_load as tl
from lib.shoot import (
    m048_inertia, geodesic_angle, omega_dir_err_deg, omega_mag_err_frac,
    polhode_period, shoot,
)
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s088"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [119, 116]
I_A = 100
DT_EPOCHS = [15, 30, 60, 120, 250]
N_DIR = 200
MAG_DPS = np.geomspace(0.04, 2.2, 11)         # cohort |w| bracket (dps)
INERTIA = m048_inertia()
DEG = np.pi / 180.0

CONNECT_TOL_DEG = 1e-4                          # a root "connects"
DEDUP_DIR_DEG = 2.0                            # two roots identical if ...
DEDUP_MAG_FRAC = 0.01


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def _worker(w_init, q_a, q_b, dt):
    s = shoot(q_a, q_b, dt, INERTIA, w_init)
    return s["omega"], s["geo_err_deg"]


def dedup(roots: list[np.ndarray]) -> list[np.ndarray]:
    """Greedy cluster connected roots in omega-space."""
    uniq = []
    for w in roots:
        new = True
        for u in uniq:
            if (omega_dir_err_deg(w, u) < DEDUP_DIR_DEG and
                    abs(omega_mag_err_frac(w, u)) < DEDUP_MAG_FRAC):
                new = False
                break
        if new:
            uniq.append(w)
    return uniq


def run_seed(seed: int, pool: Pool) -> dict:
    d = tl.load_truth(seed)
    times0 = d["observation_times"].astype(np.float64)
    times0 = times0 - times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[I_A]
    T_pol = polhode_period(w0, INERTIA)

    dirs = fib_sphere(N_DIR)
    inits = np.vstack([(m * DEG) * dirs for m in MAG_DPS])   # (N_DIR*N_MAG, 3) rad/s

    print(f"\n===== seed {seed}  |w|={np.degrees(np.linalg.norm(w0)):.4f} dps "
          f"T_pol={T_pol:.0f}s ({T_pol/7.2144:.0f} ep)  inits={len(inits)} =====")
    rows = []
    for de in DT_EPOCHS:
        i_b = I_A + de
        if i_b >= len(times0):
            continue
        q_a, q_b = q_hist[I_A], q_hist[i_b]
        dt = float(times0[i_b] - times0[I_A])
        omdt = np.degrees(np.linalg.norm(w_true) * dt)

        res = pool.map(partial(_worker, q_a=q_a, q_b=q_b, dt=dt), inits, chunksize=64)
        connected = [w for (w, geo) in res if geo < CONNECT_TOL_DEG]
        uniq = dedup(connected)

        # per-root (dir_err_to_truth, |w|/|w|_true)
        per_root = sorted(
            ((omega_dir_err_deg(u, w_true),
              float(np.linalg.norm(u) / np.linalg.norm(w_true))) for u in uniq),
            key=lambda x: x[1])
        truth_found = min((de_ for de_, _ in per_root), default=999.0) < 1.0

        # operational counts after a per-seed |w| prior (centered on truth |w|;
        # in production this is centered on the s019/s055a |w| estimate).
        bands = {}
        for tag, lo, hi in [("p10", 0.9, 1.1), ("p30", 0.7, 1.3), ("p50", 0.5, 1.5)]:
            in_band = [(de_, mr) for de_, mr in per_root if lo <= mr <= hi]
            near = [de_ for de_, _ in in_band if de_ < 5.0]
            bands[tag] = dict(n=len(in_band),
                              n_near_truth_dir=len(near),
                              truth_unique=bool(len(in_band) == 1 and truth_found))

        rows.append(dict(
            dt_epochs=de, dt_s=dt, omega_dt_deg=float(omdt),
            n_connected=len(connected), n_distinct=len(uniq),
            truth_recovered=bool(truth_found),
            min_dir_err_to_truth_deg=float(per_root[0][0]) if per_root else None,
            per_root=[[float(a), float(b)] for a, b in per_root],
            bands=bands,
        ))
        b30 = bands["p30"]
        print(f"  Δt={de:>3}ep (|w|Δt={omdt:6.0f}°): distinct={len(uniq):3d} "
              f"truth={'Y' if truth_found else 'N'} | within |w|±30%: "
              f"{b30['n']:2d} roots ({b30['n_near_truth_dir']} near-truth-dir)"
              f"{'  <-- truth UNIQUE' if b30['truth_unique'] else ''}")
    return dict(seed=seed, omega_mag_dps=float(np.degrees(np.linalg.norm(w0))),
                T_pol_s=T_pol, rows=rows)


def main():
    with Pool(24) as pool:
        results = [run_seed(s, pool) for s in SEEDS]
    out = OUT / "gateA2.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print("\n===== Gate A2 summary (unconstrained vs |w|-prior root counts) =====")
    for r in results:
        for row in r["rows"]:
            b = row["bands"]
            print(f"  seed {r['seed']} Δt={row['dt_epochs']:>3}ep |w|Δt={row['omega_dt_deg']:6.0f}°"
                  f" -> distinct={row['n_distinct']:3d} | within |w| ±10%:{b['p10']['n']:2d}"
                  f" ±30%:{b['p30']['n']:2d} ±50%:{b['p50']['n']:2d}"
                  f" | truth {'recovered' if row['truth_recovered'] else 'MISSED'}")


if __name__ == "__main__":
    main()
