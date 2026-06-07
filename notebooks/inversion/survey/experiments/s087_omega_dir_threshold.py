"""s087 — how tight must the ω-direction estimate be for the fast-seed polish to converge?

s086 isolated the binding constraint on fast tumblers: the L1 failure on seed 119
is driven by the ω-DIRECTION error (1.5° → ~13° q0_seed_err through 738° of
back-propagation), NOT the 0.5% |ω|-magnitude error (which a 1-D refine fixes but
which buys only ~0.2°). So the lever is a tighter ω-direction grid. This experiment
measures the threshold: at what ω-direction error does the polish stop recovering
truth on the fast-seed class?

DESIGN (isolate direction):
  * |ω| held at TRUTH (mag 0.0%) — justified by s086 (magnitude adds ~0.2°).
  * q perturbed by a fixed 2.0° (≈ the s085 800k-pool anchor delivery: 119→1.56°,
    108→2.20°, 100→1.65°) — a realistic anchor, held constant so direction is the
    only swept variable.
  * sweep ω-direction error over {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}°,
    N_AXES random (q-axis, ω-dir-axis) pairs per level.
  * back_propagate (s057i) → lm_polish_jacobi (s064) → band from ρ_surr
    (s081: 145/145 surrogate↔hi-fi agreement); hi-fi spot-check the best converged
    per seed to confirm the band proxy.

METRICS per (seed, dir_level):
  * truth_recovery_rate : q0_err < 5° AND ρ_surr < 4   (lands the truth basin)
  * band_AB_rate        : ρ_surr < 4                   (good LC fit incl. twins/multi-sol)
  THRESHOLD = largest dir_level where truth_recovery_rate stays high.

This is a basin/grab-radius probe (controlled perturbation about truth, the
s005/s064-gate2 method), NOT a blind search-yield claim. Output sets the required
ω-direction grid density (N_dir) for the cohort run.

Usage:
    python experiments/s087_omega_dir_threshold.py            # 119/108/100
    python experiments/s087_omega_dir_threshold.py --smoke    # 119, 3 dir levels, 2 axes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.hifi_render import build_context, rho_band  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from experiments.s059_pilot import back_propagate  # noqa: E402
from experiments.s064_jacobi_polish import (  # noqa: E402
    lm_polish_jacobi, hifi_classify, perturb_q, perturb_omega, quat_geodesic_deg,
)

SEEDS = [119, 108, 100]                       # fast LAM (s085 |ω| 1.35-1.48 dps)
DIR_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
Q_PERT = 2.0                                  # fixed realistic anchor (s085 800k delivery)
MAG_PCT = 0.0                                 # |ω| perfect — isolate direction
N_AXES = 5
TRUTH_Q_ERR_DEG = 5.0                         # s005 truth-basin threshold
BAND_AB_RHO = 4.0


def _rng_seed(seed, dir_deg, ax):
    return 9000 + seed * 100 + int(round(dir_deg * 100)) + ax


# ----------------------- worker -----------------------

_CTX = None
_TARGET = None
_ANCHOR = None   # seed -> (q_a_truth, om_a_truth, t_a_seconds, T_A)


def _winit(ctx_by_seed, target_by_seed, anchor_by_seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _CTX, _TARGET, _ANCHOR
    _CTX, _TARGET, _ANCHOR = ctx_by_seed, target_by_seed, anchor_by_seed


def _worker(job):
    seed, dir_deg, ax = job
    ctx = _CTX[seed]; target = _TARGET[seed]
    inertia = ctx["inertia_tensor"]
    q0_truth = np.asarray(ctx["q0_truth"], float)
    om0_truth = np.asarray(ctx["omega0_truth_rad"], float)
    om_truth_mag = float(np.linalg.norm(om0_truth))
    q_a_truth, om_a_truth, t_a_seconds, _ = _ANCHOR[seed]

    rng = np.random.default_rng(_rng_seed(seed, dir_deg, ax))
    qa_p = perturb_q(q_a_truth, Q_PERT, rng)
    om_p = (perturb_omega(om_a_truth, dir_deg, MAG_PCT, rng)
            if dir_deg else om_a_truth)
    q0_seed, om0_seed = back_propagate(qa_p, om_p, t_a_seconds, inertia)
    q0_seed_err = quat_geodesic_deg(np.asarray(q0_seed), q0_truth)

    res = lm_polish_jacobi(q0_seed, om0_seed, ctx, target,
                           label=f"s{seed}_dir{dir_deg}_ax{ax}")
    q0p = np.asarray(res["q0_pol_wxyz"]); om0p = np.asarray(res["om0_pol_rad"])
    q0_err = quat_geodesic_deg(q0p, q0_truth)
    om_mag_err = float((np.linalg.norm(om0p) - om_truth_mag) / om_truth_mag * 100)
    om_dir_err = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)), om0_truth / om_truth_mag)), 0, 1))))
    rho_surr = float(res["surrogate_rho_polished"])
    band = rho_band(rho_surr)
    return {
        "seed": int(seed), "dir_deg": float(dir_deg), "axis": int(ax),
        "q0_seed_err_deg": float(q0_seed_err),
        "rho_pol_surr": rho_surr, "band_surr": band,
        "q0_err_deg": float(q0_err), "om_mag_err_pct": om_mag_err, "om_dir_err_deg": om_dir_err,
        "truth_recovered": bool(q0_err < TRUTH_Q_ERR_DEG and rho_surr < BAND_AB_RHO),
        "band_AB": bool(rho_surr < BAND_AB_RHO),
        "n_eval": int(res["n_eval"]),
        "q0_pol_wxyz": res["q0_pol_wxyz"], "om0_pol_rad": res["om0_pol_rad"],
    }


# ----------------------- aggregate + plot -----------------------

def aggregate(rows, seeds, dir_levels):
    agg = {}
    for seed in seeds:
        per_level = []
        for dl in dir_levels:
            grp = [r for r in rows if r["seed"] == seed and r["dir_deg"] == dl]
            n = len(grp)
            per_level.append({
                "dir_deg": dl, "n": n,
                "truth_recovery_rate": sum(r["truth_recovered"] for r in grp) / n,
                "band_AB_rate": sum(r["band_AB"] for r in grp) / n,
                "q0_seed_err_med": float(np.median([r["q0_seed_err_deg"] for r in grp])),
                "q0_err_med": float(np.median([r["q0_err_deg"] for r in grp])),
                "rho_surr_med": float(np.median([r["rho_pol_surr"] for r in grp])),
            })
        # threshold = largest dir level with truth_recovery_rate == 1.0 (then >=0.8)
        thr_full = max([p["dir_deg"] for p in per_level if p["truth_recovery_rate"] >= 1.0],
                       default=None)
        thr_80 = max([p["dir_deg"] for p in per_level if p["truth_recovery_rate"] >= 0.8],
                     default=None)
        agg[seed] = {"per_level": per_level,
                     "dir_threshold_full_recovery_deg": thr_full,
                     "dir_threshold_80pct_deg": thr_80}
    return agg


def plot(agg, seeds, out_dir):
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4.2), squeeze=False)
    for j, seed in enumerate(seeds):
        ax = axes[0, j]
        pl = agg[seed]["per_level"]
        x = [p["dir_deg"] for p in pl]
        ax.plot(x, [p["truth_recovery_rate"] for p in pl], "o-", color="tab:blue",
                label="truth recovery (q0<5°, ρ<4)")
        ax.plot(x, [p["band_AB_rate"] for p in pl], "s--", color="tab:green",
                label="Band A∪B (ρ<4)")
        thr = agg[seed]["dir_threshold_full_recovery_deg"]
        if thr is not None:
            ax.axvline(thr, color="red", ls=":", lw=1,
                       label=f"full-recovery thr {thr:.2f}°")
        ax.set_xlabel("ω-direction error (deg)")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(f"seed {seed}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("s087: fast-seed polish convergence vs ω-direction error "
                 f"(q_pert={Q_PERT}°, |ω| exact, N_axes={N_AXES})")
    fig.tight_layout()
    p = out_dir / "dir_threshold.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-workers", type=int, default=24)
    args = ap.parse_args()

    out_dir = SURVEY / "results" / "s087"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[:1] if args.smoke else SEEDS
    dir_levels = [0.0, 1.0, 2.0] if args.smoke else DIR_LEVELS
    n_axes = 2 if args.smoke else N_AXES

    s085 = json.load(open(SURVEY / "results" / "s085" / "summary.json"))
    T_A_by_seed = {s["seed"]: s["T_A"] for s in s085["seeds"]}

    ctx_by_seed, target_by_seed, anchor_by_seed = {}, {}, {}
    for seed in seeds:
        ctx = build_context(seed=seed)
        T_A = T_A_by_seed[seed]
        target = ctx["mag_hifi_truth"]
        inertia = ctx["inertia_tensor"]
        t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
        quats_t, omegas_t = propagate_attitude(
            q0=np.asarray(ctx["q0_truth"], float),
            omega0=np.asarray(ctx["omega0_truth_rad"], float),
            times=ctx["observation_times"], mode="tumbling", inertia_tensor=inertia)
        ctx_by_seed[seed] = ctx
        target_by_seed[seed] = target
        anchor_by_seed[seed] = (np.asarray(quats_t[T_A], float),
                                np.asarray(omegas_t[T_A], float), t_a_seconds, T_A)
        wdps = float(np.degrees(np.linalg.norm(ctx["omega0_truth_rad"])))
        print(f"seed {seed}: T_A={T_A}  |ω|={wdps:.3f} dps  "
              f"{wdps*t_a_seconds:.0f}° rotation to anchor")

    jobs = [(seed, dl, ax) for seed in seeds for dl in dir_levels for ax in range(n_axes)]
    print(f"\n{len(jobs)} polish jobs (Pool {args.n_workers}) ...", flush=True)
    t0 = time.time()
    rows = []
    if args.n_workers <= 1:
        _winit(ctx_by_seed, target_by_seed, anchor_by_seed)
        for j in jobs:
            rows.append(_worker(j))
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(args.n_workers, initializer=_winit,
                           initargs=(ctx_by_seed, target_by_seed, anchor_by_seed)) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                rows.append(r)
    polish_wall = time.time() - t0
    print(f"  polish wall {polish_wall:.1f}s", flush=True)

    agg = aggregate(rows, seeds, dir_levels)

    # hi-fi spot-check: best converged (lowest ρ_surr, truth-recovered) per seed
    print("\nhi-fi spot-checks (best truth-recovered per seed) ...", flush=True)
    t1 = time.time()
    hifi_checks = []
    for seed in seeds:
        cand = [r for r in rows if r["seed"] == seed and r["truth_recovered"]]
        if not cand:
            print(f"  seed {seed}: no truth-recovered candidate to spot-check")
            continue
        best = min(cand, key=lambda r: r["rho_pol_surr"])
        rho_h, band_h = hifi_classify(best["q0_pol_wxyz"], best["om0_pol_rad"],
                                      ctx_by_seed[seed], target_by_seed[seed])
        chk = {"seed": seed, "dir_deg": best["dir_deg"], "rho_surr": best["rho_pol_surr"],
               "band_surr": best["band_surr"], "rho_hifi": float(rho_h), "band_hifi": band_h,
               "q0_err_deg": best["q0_err_deg"]}
        hifi_checks.append(chk)
        print(f"  seed {seed} (dir {best['dir_deg']}°): ρ_surr={best['rho_pol_surr']:.3f}"
              f"({best['band_surr']}) → ρ_hifi={rho_h:.3f}({band_h})  q0_err={best['q0_err_deg']:.2f}°")
    hifi_wall = time.time() - t1

    png = plot(agg, seeds, out_dir)

    print(f"\n{'='*70}")
    for seed in seeds:
        a = agg[seed]
        print(f"\n=== seed {seed} === full-recovery thr={a['dir_threshold_full_recovery_deg']}° "
              f"| 80%-recovery thr={a['dir_threshold_80pct_deg']}°")
        print(f"  {'dir°':>5} {'truth_rec':>9} {'A∪B':>6} {'q0seed_err':>10} {'q0_err':>8} {'ρ_med':>7}")
        for p in a["per_level"]:
            print(f"  {p['dir_deg']:>5.2f} {p['truth_recovery_rate']:>9.2f} "
                  f"{p['band_AB_rate']:>6.2f} {p['q0_seed_err_med']:>9.1f}° "
                  f"{p['q0_err_med']:>7.1f}° {p['rho_surr_med']:>7.2f}")
    print(f"\nSaved: {png}")

    summary = {
        "experiment": "s087_omega_dir_threshold",
        "seeds": seeds, "dir_levels": dir_levels, "n_axes": n_axes,
        "q_pert_deg": Q_PERT, "mag_pct": MAG_PCT,
        "truth_q_err_deg": TRUTH_Q_ERR_DEG, "band_AB_rho": BAND_AB_RHO,
        "aggregate": {str(s): agg[s] for s in seeds},
        "hifi_checks": hifi_checks,
        "rows": rows,
        "polish_wall_s": polish_wall, "hifi_wall_s": hifi_wall,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"TOTAL wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
