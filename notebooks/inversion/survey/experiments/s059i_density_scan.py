"""s059i_density_scan — closest-to-truth survivor at T_A vs SO(3) sample density.

Question: at what SO(3) pool density does the closest-to-truth member of
the survival cloud at T_A=25 on seed 28 fall below 5° geodesic — the regime
where s059i_validator_perturbed showed truth-ω ranks top-16 every trial?

For each density N ∈ {100k, 200k, 400k, 800k}:
  - sample N Haar-uniform quaternions (3 RNG seeds for variance)
  - project at T_A=25, survive at TOL_MAG=0.10 (s059_pilot default)
  - report |C_a| and closest_to_truth_deg among survivors

Per-density wall is ~1 sec at 800k single-thread. Total ~1 min.

Usage:
    python experiments/s059i_density_scan.py --seed 28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions, survive_at_epoch,
    nearest_in_pool_to_truth,
)
from lib.hifi_render import build_context  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

TOL_MAG = 0.10
SP_DEG = 0.0
AD_DEG = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--T-A", type=int, default=25)
    ap.add_argument("--densities", type=str,
                    default="100000,200000,400000,800000")
    ap.add_argument("--rng-seeds", type=str, default="42,43,44")
    ap.add_argument("--out-root",
                    default=str(SURVEY / "results" / "s059i_density_scan"))
    args = ap.parse_args()

    densities = [int(x) for x in args.densities.split(",")]
    rng_seeds = [int(x) for x in args.rng_seeds.split(",")]

    out_dir = Path(args.out_root) / f"seed{args.seed:03d}_T{args.T_A:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059i_density_scan: seed {args.seed}, T_A={args.T_A} ===")
    log(f"densities: {densities}")
    log(f"rng seeds: {rng_seeds}")

    # Context + truth at T_A
    log("\n[1/3] context + truth state at anchor")
    ctx = build_context(args.seed)
    quats_truth, _ = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=ctx["observation_times"], mode="tumbling",
        inertia_tensor=ctx["inertia_tensor"],
    )
    q_a_truth = np.asarray(quats_truth[args.T_A], dtype=np.float64)
    sun_unit, obs_unit = compute_j2000_units(
        ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    measured_mag = float(ctx["mag_hifi_truth"][args.T_A])
    obs_dist_ep = float(ctx["obs_dist"][args.T_A])
    log(f"  truth q_a (wxyz) at T_A={args.T_A}: {q_a_truth}")
    log(f"  measured mag at T_A: {measured_mag:.4f}")
    log(f"  obs_dist at T_A: {obs_dist_ep:.1f} km")

    # Surrogate (v1) — same paths as s059_pilot.py uses
    from surrogate_model.surrogate_v1 import SurrogateModel as _V1
    V1_DIR = Path("/home/girish/surrogate_model/surrogate_model")
    surrogate_v1 = _V1(str(V1_DIR / "s10_5M_weights.npz"),
                       str(V1_DIR / "s10_5M_normalization.npz"))

    # 2. Sweep
    log(f"\n[2/3] density sweep × rng seeds")
    rows = []
    for N in densities:
        for rng_seed in rng_seeds:
            t_iter = time.time()
            pool = sample_so3_pool(N, sample_seed=rng_seed)
            k1, k2 = project_directions(
                pool["R_cache"], sun_unit[args.T_A], obs_unit[args.T_A])
            pred, keep = survive_at_epoch(
                surrogate_v1, k1, k2, obs_dist_ep, SP_DEG, AD_DEG,
                measured_mag, TOL_MAG)
            n_survive = int(keep.sum())
            if n_survive == 0:
                row = {
                    "N": N, "rng_seed": rng_seed,
                    "n_survive": 0,
                    "closest_to_truth_deg": float("nan"),
                    "closest_pool_idx": -1,
                    "wall_s": time.time() - t_iter,
                }
                rows.append(row)
                log(f"  N={N:>7}  seed={rng_seed}  |C_a|=0 — no survivors")
                continue
            q_survive = pool["q_pool_wxyz"][keep]
            closest_deg, closest_idx = nearest_in_pool_to_truth(
                q_survive, q_a_truth)
            wall = time.time() - t_iter
            # Also: closest among ALL pool members (no survival filter), for
            # comparison — this is the "geometric" closest-to-truth.
            all_closest_deg, _ = nearest_in_pool_to_truth(
                pool["q_pool_wxyz"], q_a_truth)
            row = {
                "N": N, "rng_seed": rng_seed,
                "n_survive": n_survive,
                "survival_rate": n_survive / N,
                "closest_to_truth_deg": float(closest_deg),
                "closest_to_truth_deg_all_pool": float(all_closest_deg),
                "wall_s": wall,
            }
            rows.append(row)
            log(f"  N={N:>7}  seed={rng_seed}  |C_a|={n_survive:>5}  "
                f"closest_to_truth(survive)={closest_deg:5.2f}°  "
                f"closest_to_truth(pool)={all_closest_deg:5.2f}°  "
                f"wall={wall:.1f}s")
            del pool, k1, k2, pred, keep

    # 3. Aggregate
    log(f"\n[3/3] summary")
    log(f"  N         | trials | |C_a| median | closest_survive median (range)  | closest_pool median (range)")
    log(f"  --------- | ------ | ------------ | ------------------------------- | ---------------------------")
    summary_per_N = {}
    for N in densities:
        rs = [r for r in rows if r["N"] == N and r["n_survive"] > 0]
        if not rs:
            log(f"  {N:>9}  | NO SURVIVORS in any trial")
            continue
        n_survive_med = int(np.median([r["n_survive"] for r in rs]))
        cs = [r["closest_to_truth_deg"] for r in rs]
        cs_pool = [r["closest_to_truth_deg_all_pool"] for r in rs]
        summary_per_N[N] = {
            "n_trials": len(rs),
            "n_survive_median": n_survive_med,
            "closest_survive_median_deg": float(np.median(cs)),
            "closest_survive_min_deg": float(min(cs)),
            "closest_survive_max_deg": float(max(cs)),
            "closest_pool_median_deg": float(np.median(cs_pool)),
        }
        log(f"  {N:>9}  | {len(rs):>4d}   | {n_survive_med:>11d}  | "
            f"{np.median(cs):>5.2f}° ({min(cs):.2f}°–{max(cs):.2f}°)         | "
            f"{np.median(cs_pool):>5.2f}° ({min(cs_pool):.2f}°–{max(cs_pool):.2f}°)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = np.array([r["N"] for r in rows])
    cs = np.array([r["closest_to_truth_deg"] for r in rows])
    cs_pool = np.array([r["closest_to_truth_deg_all_pool"] for r in rows])
    ax.scatter(Ns, cs, alpha=0.6, label="closest in survival cloud (trials)")
    ax.scatter(Ns, cs_pool, alpha=0.6, marker="x", color="orange",
               label="closest in entire pool (trials)")
    medians_survive = np.array([
        summary_per_N[N]["closest_survive_median_deg"]
        for N in densities if N in summary_per_N])
    medians_pool = np.array([
        summary_per_N[N]["closest_pool_median_deg"]
        for N in densities if N in summary_per_N])
    valid_Ns = [N for N in densities if N in summary_per_N]
    ax.plot(valid_Ns, medians_survive, "k-", lw=2, label="survival median")
    ax.plot(valid_Ns, medians_pool, "k--", lw=1.5, alpha=0.6,
            label="all-pool median")
    ax.axhline(7.5, color="green", lw=1, ls=":",
               label="s059i robust regime (≤7.5°)")
    ax.axhline(10, color="orange", lw=1, ls=":",
               label="s059i bimodal cliff (~10°)")
    ax.axhline(15, color="red", lw=1, ls=":",
               label="s059i failing regime (≥15°)")
    ax.set_xscale("log")
    ax.set_xlabel("Sobol pool size N")
    ax.set_ylabel("closest-to-truth quaternion (deg, geodesic)")
    ax.set_title(f"seed {args.seed} T_A={args.T_A}: closest-to-truth vs pool density")
    ax.legend()
    plt.tight_layout()
    plot_path = out_dir / "density_scan.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log(f"\nSaved: {plot_path}")

    summary = {
        "seed": int(args.seed),
        "T_A": int(args.T_A),
        "TOL_MAG": float(TOL_MAG),
        "densities": densities,
        "rng_seeds": rng_seeds,
        "rows": rows,
        "summary_per_N": {str(k): v for k, v in summary_per_N.items()},
        "wall_total_s": float(time.time() - t_overall),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Saved: {summary_path}")
    log(f"\nWall total: {time.time() - t_overall:.1f}s")


if __name__ == "__main__":
    main()
