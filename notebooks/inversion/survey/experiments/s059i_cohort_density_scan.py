"""s059i_cohort_density_scan — closest-to-truth survivor at T_A=25 vs SO(3)
pool density, across a representative cohort of m048 seeds.

Question: does N=400k Sobol put MOST seeds (not just seed 28) into the s059i
robust regime (≤ 7.5° closest-survive)? If yes, A1 (build s059j on densified
cloud) is the architecturally-honest path. If a meaningful tail still sits
at 8°+ regardless of density, A3 (low-discrepancy sampler) or B (joint LM
polish) is needed.

Cohort: 8 seeds chosen to span trajectory-class structure
  6   — s011 PA-stratified pilot, density-recoverable
  10  — s013 multi-solution-boundary
  14  — s042 high-|ω| polhode-binding
  23  — s011 density-recoverable
  28  — s014 multi-solution; this validator's reference seed
  44  — s010 narrow-basin
  84  — s014 multi-solution-rich
  89  — s011 density-recoverable; s058's reference

Optimisation: pool sampling is seed-independent. Sample once per (N, rng);
project + survive + nearest per seed inside that loop. Wall ~1 min total.

Usage:
    python experiments/s059i_cohort_density_scan.py
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


def per_seed_anchor_state(seed, T_A):
    """Build context + truth q at T_A + per-epoch SPICE units + measured mag."""
    ctx = build_context(seed)
    quats_truth, _ = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=ctx["observation_times"], mode="tumbling",
        inertia_tensor=ctx["inertia_tensor"],
    )
    q_a_truth = np.asarray(quats_truth[T_A], dtype=np.float64)
    sun_unit, obs_unit = compute_j2000_units(
        ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    return {
        "ctx": ctx,
        "q_a_truth": q_a_truth,
        "sun_unit_T": sun_unit[T_A],
        "obs_unit_T": obs_unit[T_A],
        "obs_dist_T": float(ctx["obs_dist"][T_A]),
        "mag_T": float(ctx["mag_hifi_truth"][T_A]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="6,10,14,23,28,44,84,89")
    ap.add_argument("--T-A", type=int, default=25)
    ap.add_argument("--densities", type=str,
                    default="100000,200000,400000")
    ap.add_argument("--rng-seeds", type=str, default="42,43,44")
    ap.add_argument("--out-root",
                    default=str(SURVEY / "results" / "s059i_density_scan"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    densities = [int(x) for x in args.densities.split(",")]
    rng_seeds = [int(x) for x in args.rng_seeds.split(",")]

    out_dir = Path(args.out_root) / f"cohort_T{args.T_A:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059i_cohort_density_scan: T_A={args.T_A} ===")
    log(f"seeds: {seeds}")
    log(f"densities: {densities}")
    log(f"rng seeds: {rng_seeds}")

    # 1. Build per-seed anchor states once
    log(f"\n[1/3] build per-seed anchor states ({len(seeds)} seeds)")
    states = {}
    for seed in seeds:
        t = time.time()
        states[seed] = per_seed_anchor_state(seed, args.T_A)
        log(f"  seed {seed:3d}: q_a built in {time.time()-t:.1f}s")

    # 2. Surrogate v1
    from surrogate_model.surrogate_v1 import SurrogateModel as _V1
    V1_DIR = Path("/home/girish/surrogate_model/surrogate_model")
    surrogate_v1 = _V1(str(V1_DIR / "s10_5M_weights.npz"),
                       str(V1_DIR / "s10_5M_normalization.npz"))

    # 3. Pool sampling outer loop (shared across seeds), per-seed inner loop
    log(f"\n[2/3] cohort sweep")
    rows = []
    for N in densities:
        for rng_seed in rng_seeds:
            t_pool = time.time()
            pool = sample_so3_pool(N, sample_seed=rng_seed)
            wall_pool = time.time() - t_pool
            log(f"  N={N:>7} rng={rng_seed}: pool sampled in {wall_pool:.1f}s")
            for seed in seeds:
                t_iter = time.time()
                st = states[seed]
                k1, k2 = project_directions(
                    pool["R_cache"], st["sun_unit_T"], st["obs_unit_T"])
                pred, keep = survive_at_epoch(
                    surrogate_v1, k1, k2, st["obs_dist_T"], SP_DEG, AD_DEG,
                    st["mag_T"], TOL_MAG)
                n_survive = int(keep.sum())
                if n_survive == 0:
                    rows.append({"seed": seed, "N": N, "rng_seed": rng_seed,
                                 "n_survive": 0,
                                 "closest_to_truth_deg": float("nan"),
                                 "wall_s": time.time() - t_iter})
                    log(f"    seed {seed:3d}: |C_a|=0 — no survivors")
                    continue
                q_survive = pool["q_pool_wxyz"][keep]
                closest_deg, _ = nearest_in_pool_to_truth(
                    q_survive, st["q_a_truth"])
                wall = time.time() - t_iter
                rows.append({
                    "seed": seed, "N": N, "rng_seed": rng_seed,
                    "n_survive": n_survive,
                    "survival_rate": n_survive / N,
                    "closest_to_truth_deg": float(closest_deg),
                    "wall_s": wall,
                })
                log(f"    seed {seed:3d}: |C_a|={n_survive:>5} "
                    f"closest_survive={closest_deg:5.2f}°  wall={wall:.1f}s")
            del pool, k1, k2, pred, keep

    # 4. Aggregate
    log(f"\n[3/3] cohort summary")

    # Per-seed at N=400k (median across RNGs)
    log(f"\n  Per-seed closest-survive at N=400k (median across {len(rng_seeds)} RNGs)")
    log(f"  seed | |C_a| med | closest median (range)        | regime")
    log(f"  ---- | --------- | ----------------------------- | ------")
    summary_400k_per_seed = {}
    for seed in seeds:
        rs = [r for r in rows if r["seed"] == seed and r["N"] == 400_000
              and r["n_survive"] > 0]
        if not rs:
            log(f"  {seed:3d}  | NO SURVIVORS at 400k")
            continue
        Cas = [r["n_survive"] for r in rs]
        css = [r["closest_to_truth_deg"] for r in rs]
        med_cs = float(np.median(css))
        regime = ("ROBUST" if med_cs <= 7.5
                  else "BIMODAL" if med_cs <= 12
                  else "FAIL")
        summary_400k_per_seed[seed] = {
            "Ca_median": int(np.median(Cas)),
            "closest_median_deg": med_cs,
            "closest_min_deg": float(min(css)),
            "closest_max_deg": float(max(css)),
            "regime": regime,
        }
        log(f"  {seed:3d}  | {int(np.median(Cas)):>8d}  | "
            f"{med_cs:>5.2f}° ({min(css):.2f}°–{max(css):.2f}°)         | "
            f"{regime}")

    # Cohort regime counts at each density
    log(f"\n  Regime distribution per density (using median across RNGs)")
    log(f"  N        | seeds in ROBUST | BIMODAL | FAIL | NO SURVIVORS")
    log(f"  -------- | --------------- | ------- | ---- | ------------")
    cohort_regimes_per_N = {}
    for N in densities:
        counts = {"ROBUST": 0, "BIMODAL": 0, "FAIL": 0, "NO_SURVIVORS": 0}
        for seed in seeds:
            rs = [r for r in rows if r["seed"] == seed and r["N"] == N
                  and r["n_survive"] > 0]
            if not rs:
                counts["NO_SURVIVORS"] += 1
                continue
            css = [r["closest_to_truth_deg"] for r in rs]
            med_cs = float(np.median(css))
            if med_cs <= 7.5:
                counts["ROBUST"] += 1
            elif med_cs <= 12:
                counts["BIMODAL"] += 1
            else:
                counts["FAIL"] += 1
        cohort_regimes_per_N[N] = counts
        log(f"  {N:>7}  |       {counts['ROBUST']:2d}        |   {counts['BIMODAL']:2d}    |  "
            f"{counts['FAIL']:2d}  |     {counts['NO_SURVIVORS']:2d}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-seed dot plot (color by density)
    cmap = plt.get_cmap("viridis")
    for j, N in enumerate(densities):
        for seed in seeds:
            rs = [r for r in rows if r["seed"] == seed and r["N"] == N
                  and r["n_survive"] > 0]
            css = [r["closest_to_truth_deg"] for r in rs]
            xs = [seed] * len(css)
            axes[0].scatter(xs, css, alpha=0.6,
                            color=cmap(j / max(1, len(densities)-1)),
                            label=f"N={N}" if seed == seeds[0] else None)
    axes[0].axhline(7.5, color="green", lw=1, ls=":", label="robust (≤7.5°)")
    axes[0].axhline(10, color="orange", lw=1, ls=":", label="bimodal (~10°)")
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("closest-survive (deg)")
    axes[0].set_yscale("log")
    axes[0].set_title(f"closest-survive at T_A={args.T_A} per seed × density")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_xticks(seeds)

    # Right: cohort regime bar chart per density
    Ns = list(densities)
    robust = [cohort_regimes_per_N[N]["ROBUST"] for N in Ns]
    bimodal = [cohort_regimes_per_N[N]["BIMODAL"] for N in Ns]
    fail = [cohort_regimes_per_N[N]["FAIL"] for N in Ns]
    no_surv = [cohort_regimes_per_N[N]["NO_SURVIVORS"] for N in Ns]
    x = np.arange(len(Ns))
    bot = np.zeros(len(Ns))
    axes[1].bar(x, robust, bottom=bot, label="robust (≤7.5°)", color="green")
    bot = bot + robust
    axes[1].bar(x, bimodal, bottom=bot, label="bimodal", color="orange")
    bot = bot + bimodal
    axes[1].bar(x, fail, bottom=bot, label="fail", color="red")
    bot = bot + fail
    axes[1].bar(x, no_surv, bottom=bot, label="no survivors", color="grey")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{N//1000}k" for N in Ns])
    axes[1].set_xlabel("pool density N")
    axes[1].set_ylabel(f"# seeds (out of {len(seeds)})")
    axes[1].set_title("cohort regime distribution vs density")
    axes[1].legend()

    plt.tight_layout()
    plot_path = out_dir / "cohort_density_scan.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log(f"\nSaved: {plot_path}")

    summary = {
        "T_A": int(args.T_A),
        "seeds": seeds,
        "densities": densities,
        "rng_seeds": rng_seeds,
        "TOL_MAG": float(TOL_MAG),
        "rows": rows,
        "summary_400k_per_seed": {str(k): v for k, v in summary_400k_per_seed.items()},
        "cohort_regimes_per_N": {str(k): v for k, v in cohort_regimes_per_N.items()},
        "wall_total_s": float(time.time() - t_overall),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Saved: {summary_path}")
    log(f"\nWall total: {time.time() - t_overall:.1f}s")


if __name__ == "__main__":
    main()
