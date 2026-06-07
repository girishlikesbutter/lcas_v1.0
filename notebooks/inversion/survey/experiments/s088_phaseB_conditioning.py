"""s088 Phase B — endpoint-noise conditioning of the truth-branch BVP root vs Δt.

The premise of the cross-cloud bridging idea: a BVP solve over LARGE Δt is
BETTER conditioned against endpoint (q-cloud) noise than a small-Δt finite-diff,
because (a) the solve removes the constant-omega/polhode-drift approximation
error, and (b) endpoint noise matters less when the trajectory spans more
rotation (s057c: omega error ~ q_err / (|w|Δt)).

This script perturbs the truth pair (q_a, q_b) by realistic q-cloud noise
(sigma matching s085 anchor accuracy ~1.65-3.30 deg), then:
  - BVP truth-branch: shoot initialised AT true omega (isolates the conditioning
    of the truth root; aliasing/basin is Gate A2's separate concern).
  - finite-diff baseline: finite_diff_omega on the same noisy endpoints
    (reproduces the s057c/d "old" method on identical pairs).
measures omega-DIRECTION and |omega| error distributions vs Δt, and checks the
s087 gate (fast-seed truth recovery needs omega-dir <~ 1.25 deg).

Pool(24), BLAS pinned. Pure closed-form propagation + cached truth.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib.traj_load as tl
from lib.shoot import (
    m048_inertia, finite_diff_omega, omega_dir_err_deg, omega_mag_err_frac,
    polhode_period, shoot,
)
from lib.jacobi_propagator import propagate_jacobi_path2, _quat_multiply

OUT = Path(__file__).resolve().parent.parent / "results" / "s088"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [119, 116]
I_A = 100
DT_EPOCHS = [7, 15, 30, 60, 120]
SIGMA_Q_DEG = 2.5            # per-anchor q-cloud noise (s085: 1.65-3.30 deg)
N_NOISE = 200
S087_GATE_DEG = 1.25
INERTIA = m048_inertia()


def perturb_q(q, sigma_deg, rng):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    a = np.radians(sigma_deg)
    qp = np.array([np.cos(a / 2), *(np.sin(a / 2) * ax)])
    out = _quat_multiply(qp, q)
    return out / np.linalg.norm(out)


def _trial(seed_arg, q_a, q_b, dt, w_true, sigma):
    rng = np.random.default_rng(seed_arg)
    qa_n = perturb_q(q_a, sigma, rng)
    qb_n = perturb_q(q_b, sigma, rng)
    # BVP truth-branch (init at true omega -> converges to the truth root).
    s = shoot(qa_n, qb_n, dt, INERTIA, w_true)
    bvp_dir = omega_dir_err_deg(s["omega"], w_true)
    bvp_mag = omega_mag_err_frac(s["omega"], w_true)
    # finite-diff baseline on identical noisy endpoints.
    w_fd = finite_diff_omega(qa_n, qb_n, dt)
    fd_dir = omega_dir_err_deg(w_fd, w_true)
    fd_mag = omega_mag_err_frac(w_fd, w_true)
    return bvp_dir, abs(bvp_mag), fd_dir, abs(fd_mag), s["geo_err_deg"]


def run_seed(seed, pool):
    d = tl.load_truth(seed)
    times0 = d["observation_times"].astype(np.float64)
    times0 = times0 - times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[I_A]
    T_pol = polhode_period(w0, INERTIA)
    print(f"\n===== seed {seed}  |w|={np.degrees(np.linalg.norm(w0)):.4f} dps "
          f"T_pol={T_pol:.0f}s  sigma_q={SIGMA_Q_DEG}deg  N={N_NOISE} =====")
    print(f"  {'Δt':>4} {'|w|Δt':>7} | {'BVP dir(med/p90)':>18} {'FD dir(med/p90)':>18}"
          f" | {'BVP mag%':>9} {'FD mag%':>9} | gate")
    rows = []
    for de in DT_EPOCHS:
        i_b = I_A + de
        if i_b >= len(times0):
            continue
        q_a, q_b = q_hist[I_A], q_hist[i_b]
        dt = float(times0[i_b] - times0[I_A])
        omdt = np.degrees(np.linalg.norm(w_true) * dt)
        args = [(1000 * de + k, q_a, q_b, dt, w_true, SIGMA_Q_DEG) for k in range(N_NOISE)]
        res = np.array(pool.starmap(_trial, args))   # (N,5)
        bvp_dir, bvp_mag, fd_dir, fd_mag, geo = res.T
        med_bvp, p90_bvp = np.median(bvp_dir), np.percentile(bvp_dir, 90)
        med_fd, p90_fd = np.median(fd_dir), np.percentile(fd_dir, 90)
        gate = "PASS" if med_bvp < S087_GATE_DEG else "----"
        rows.append(dict(
            dt_epochs=de, dt_s=dt, omega_dt_deg=float(omdt),
            bvp_dir_med=float(med_bvp), bvp_dir_p90=float(p90_bvp),
            fd_dir_med=float(med_fd), fd_dir_p90=float(p90_fd),
            bvp_mag_med_pct=float(100 * np.median(bvp_mag)),
            fd_mag_med_pct=float(100 * np.median(fd_mag)),
            connect_resid_med_deg=float(np.median(geo)),
        ))
        print(f"  {de:>4} {omdt:7.0f} | {med_bvp:7.2f}/{p90_bvp:7.2f}     "
              f"{med_fd:7.2f}/{p90_fd:7.2f}     | {100*np.median(bvp_mag):8.2f} "
              f"{100*np.median(fd_mag):8.2f} | {gate}")
    return dict(seed=seed, omega_mag_dps=float(np.degrees(np.linalg.norm(w0))),
                T_pol_s=T_pol, sigma_q_deg=SIGMA_Q_DEG, n_noise=N_NOISE, rows=rows)


def make_plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, r in zip(axes, results):
        om = [row["omega_dt_deg"] for row in r["rows"]]
        ax.plot(om, [row["bvp_dir_med"] for row in r["rows"]], "o-", label="BVP shoot (med)")
        ax.fill_between(om, [row["bvp_dir_med"] for row in r["rows"]],
                        [row["bvp_dir_p90"] for row in r["rows"]], alpha=0.15)
        ax.plot(om, [row["fd_dir_med"] for row in r["rows"]], "s--", label="finite-diff (med)")
        ax.axhline(S087_GATE_DEG, color="r", ls=":", label=f"s087 gate {S087_GATE_DEG}°")
        ax.set_title(f"seed {r['seed']}  |w|={r['omega_mag_dps']:.2f} dps")
        ax.set_xlabel("net rotation |w|·Δt  (deg)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("omega-direction error (deg)")
    fig.suptitle(f"s088 Phase B: BVP-shoot vs finite-diff omega-dir error "
                 f"(sigma_q={SIGMA_Q_DEG}°, N={N_NOISE})")
    fig.tight_layout()
    p = OUT / "phaseB_conditioning.png"
    fig.savefig(p, dpi=120)
    print(f"\nSaved: {p}")


def main():
    with Pool(24) as pool:
        results = [run_seed(s, pool) for s in SEEDS]
    with open(OUT / "phaseB.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {OUT / 'phaseB.json'}")
    make_plot(results)


if __name__ == "__main__":
    main()
