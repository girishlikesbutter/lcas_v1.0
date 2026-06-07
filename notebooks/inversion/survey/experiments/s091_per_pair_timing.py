"""s091 — per-pair cost of the cross-cloud spin-solve (cloud-free disambiguation).

Decomposes the wall cost of processing ONE (q_a, q_b) candidate pair, single
process (so the numbers are per-call, not Pool-amortised):

  1. finite_diff_omega init           -- the LM seed (q_b ⊗ q_a^-1).
  2. one 2-anchor shoot (1 root)       -- the atomic BVP solve.
  3. one 3-anchor joint shoot          -- (only used in the geometric-target form).
  4. brightness check per winding      -- propagate (q_a, w) to a 3rd epoch + 1
                                          surrogate eval (the cloud-free selector).
  5. winding enumeration per pair       -- two strategies:
       (a) brute multi-start 2200 inits (Fibonacci-200 x geomspace-11), as s088/89;
       (b) targeted 1-D |w|-line scan along the FD direction (windings are a
           same-direction ladder, s089) -- N_MAG inits.

Per-pair cost = enumeration + (n_windings x brightness-check). Reports both the
brute and targeted enumeration so the budget is bracketed honestly.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
import lib.surrogate_eval as se
from lib.shoot import (
    m048_inertia, finite_diff_omega, shoot, shoot_multianchor, polhode_period,
)
from lib.jacobi_propagator import propagate_jacobi_path2

I_A, SEED = 100, 119
INERTIA = m048_inertia()
SP_DEG, AD_DEG = 0.0, 15.0


def timeit(fn, n):
    fn()  # warm
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return 1e3 * (time.perf_counter() - t) / n   # ms/call


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def main():
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[I_A]
    T_pol = polhode_period(w0, INERTIA)
    span = min(T_pol, times0[-1] - times0[I_A])
    i_b = int(np.argmin(np.abs(times0 - (times0[I_A] + 0.30 * span))))
    i_c = int(np.argmin(np.abs(times0 - (times0[I_A] + 0.45 * span))))
    q_a, q_b, q_c = q_hist[I_A], q_hist[i_b], q_hist[i_c]
    dt_ab = float(times0[i_b] - times0[I_A])
    dt_ac = float(times0[i_c] - times0[I_A])
    anchors = [(q_b, dt_ab), (q_c, dt_ac)]

    sun = d["sun_pos"][i_c] - d["sat_pos"][i_c]; sun /= np.linalg.norm(sun)
    obs = d["obs_pos"][i_c] - d["sat_pos"][i_c]; obs /= np.linalg.norm(obs)
    obs_dist = float(d["obs_dist"][i_c])
    se.get_model()

    print(f"===== s091 per-pair timing | seed {SEED} =====")
    print(f"|w|={np.degrees(np.linalg.norm(w0)):.3f}dps T_pol={T_pol:.0f}s "
          f"ab={i_b-I_A}ep dt_ab={dt_ab:.0f}s ac={i_c-I_A}ep dt_ac={dt_ac:.0f}s\n")

    t_fd = timeit(lambda: finite_diff_omega(q_a, q_b, dt_ab), 2000)
    w_fd = finite_diff_omega(q_a, q_b, dt_ab)
    t_shoot = timeit(lambda: shoot(q_a, q_b, dt_ab, INERTIA, w_fd), 200)
    t_joint = timeit(lambda: shoot_multianchor(q_a, anchors, INERTIA, w_fd), 200)

    def bright_check(w):
        qh, _ = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, dt_ac]))
        R = Rotation.from_quat([qh[-1][1], qh[-1][2], qh[-1][3], qh[-1][0]]).as_matrix()
        return float(se.predict((R @ sun)[None, :], (R @ obs)[None, :],
                                np.array([obs_dist]), SP_DEG, AD_DEG)[0])
    t_bright = timeit(lambda: bright_check(w_fd), 500)

    print(f"  finite_diff init        : {t_fd:8.4f} ms")
    print(f"  1x 2-anchor shoot       : {t_shoot:8.3f} ms")
    print(f"  1x 3-anchor joint shoot : {t_joint:8.3f} ms")
    print(f"  brightness check (1 winding, propagate+surrogate): {t_bright:8.3f} ms")

    # enumeration strategies (single process)
    N_MAG = 40
    mags = np.geomspace(0.04, 2.2, N_MAG) * np.pi / 180.0
    fd_dir = w_fd / np.linalg.norm(w_fd)
    line_inits = mags[:, None] * fd_dir[None, :]
    t0 = time.perf_counter()
    for w_init in line_inits:
        shoot(q_a, q_b, dt_ab, INERTIA, w_init)
    t_line = 1e3 * (time.perf_counter() - t0)

    brute_inits = np.vstack([m * fib_sphere(200) for m in np.geomspace(0.04, 2.2, 11) * np.pi / 180.0])
    t_brute_est = t_shoot * len(brute_inits)   # estimate from atomic (2200 solves)

    print(f"\n  enumeration per pair (single process):")
    print(f"    (a) brute 2200-init multi-start : ~{t_brute_est/1000:6.2f} s  (= {t_shoot:.2f}ms x 2200)")
    print(f"    (b) targeted |w|-line {N_MAG} inits  : {t_line/1000:6.3f} s  ({t_line:.0f} ms)")

    # representative per-pair total assuming ~15 windings to brightness-check
    n_wind = 15
    pp_brute = t_brute_est + n_wind * t_bright
    pp_line = t_line + n_wind * t_bright
    print(f"\n  per-pair total (+ {n_wind} winding brightness-checks):")
    print(f"    brute   : {pp_brute/1000:7.2f} s/pair  -> {1000/pp_brute*24:8.1f} pairs/s on Pool(24)")
    print(f"    targeted: {pp_line/1000:7.3f} s/pair  -> {1000/pp_line*24:8.1f} pairs/s on Pool(24)")
    print(f"\n  single 2-anchor shoot throughput Pool(24): ~{1000/t_shoot*24:.0f} solves/s")


if __name__ == "__main__":
    main()
