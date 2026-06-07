"""s059i_validator_perturbed — sensitivity of the local-window cost surface
to q_a noise.

Sweeps q_a perturbation magnitude ∈ {0, 2.5, 5, 7.5, 10, 15}° (geodesic) and
reports where truth-ω ranks in the same Fibonacci ω-grid as s059i_validator.
3 trials per magnitude (different deterministic RNG axes) for variance.

Question: in the realistic search, q_a comes from the SO(3) survival cloud
(7-10° quantization on seed 28). Does truth-ω still rank top-K when q_a is
that far off? If yes → s059j worth building. If no → q_a noise alone
defeats the architecture.

Usage:
    python experiments/s059i_validator_perturbed.py --seed 28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.hifi_render import build_context  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from experiments.s059i_validator import (  # noqa: E402
    fibonacci_sphere,
    score_local_window,
    angle_between_unit_vecs_deg,
)


def perturb_quaternion(q_wxyz, angle_deg, rng):
    """Rotate q_wxyz by EXACTLY angle_deg geodesic distance about a random axis.

    Uses the relation: a quaternion rotation by θ about axis n contributes
    a rotation of angle θ; geodesic distance on SO(3) equals that θ.
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    delta = Rotation.from_rotvec(np.radians(angle_deg) * axis)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    q_new_xyzw = (delta * Rotation.from_quat(q_xyzw)).as_quat()
    return np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])


def quat_geodesic_deg(q1_wxyz, q2_wxyz):
    dot = float(np.clip(abs(np.dot(q1_wxyz, q2_wxyz)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--T-A", type=int, default=25)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--n-dirs", type=int, default=200)
    ap.add_argument("--n-mags", type=int, default=6)
    ap.add_argument("--mag-bracket", type=float, default=0.30)
    ap.add_argument("--perturbations-deg", type=str,
                    default="0,2.5,5,7.5,10,15",
                    help="comma-separated geodesic angles (degrees)")
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--rng-base-seed", type=int, default=42)
    ap.add_argument("--out-root", default=str(SURVEY / "results" / "s059i_validator"))
    args = ap.parse_args()

    perturbations = [float(x) for x in args.perturbations_deg.split(",")]

    out_dir = Path(args.out_root) / f"seed{args.seed:03d}_T{args.T_A:03d}_W{args.W:02d}_perturbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059i_validator_perturbed: seed {args.seed}, T_A={args.T_A}, W={args.W} ===")
    log(f"perturbations (deg): {perturbations}")
    log(f"trials per perturbation: {args.n_trials}")

    # Context + truth state
    log("\n[1/4] context + truth state at anchor")
    ctx = build_context(args.seed)
    quats_truth, omegas_truth = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=ctx["observation_times"], mode="tumbling",
        inertia_tensor=ctx["inertia_tensor"],
    )
    q_a_truth = np.asarray(quats_truth[args.T_A], dtype=np.float64)
    om_a_truth = np.asarray(omegas_truth[args.T_A], dtype=np.float64)
    om_a_truth_mag = float(np.linalg.norm(om_a_truth))
    om_a_truth_dir = om_a_truth / om_a_truth_mag
    log(f"  truth ω_body = {om_a_truth} rad/s, |ω| = {om_a_truth_mag:.6f} rad/s")

    # ω grid (same as s059i_validator)
    log(f"\n[2/4] build ω grid: {args.n_dirs} dirs × {args.n_mags} mags")
    fib_dirs = fibonacci_sphere(args.n_dirs)
    all_dirs = np.vstack([om_a_truth_dir[None, :], fib_dirs])
    factor_lo = 1.0 - args.mag_bracket
    factor_hi = 1.0 + args.mag_bracket
    mag_factors = np.concatenate([
        [1.0],
        np.linspace(factor_lo, factor_hi, args.n_mags),
    ])
    all_mags = mag_factors * om_a_truth_mag
    n_total = len(all_dirs) * len(all_mags)
    omegas = (all_dirs[:, None, :] * all_mags[None, :, None]).reshape(-1, 3)
    truth_idx = 0
    log(f"  total grid points: {n_total} (truth at idx {truth_idx})")

    # Sweep
    log(f"\n[3/4] sweep over q_a perturbations × trials (sequential)")
    rows = []
    for p_deg in perturbations:
        for trial in range(args.n_trials):
            t_p = time.time()
            rng = np.random.default_rng(args.rng_base_seed + 1000 * trial
                                          + int(round(p_deg * 10)))
            q_a_used = (q_a_truth if p_deg == 0.0
                        else perturb_quaternion(q_a_truth, p_deg, rng))
            actual_geodesic = quat_geodesic_deg(q_a_used, q_a_truth)
            scores = np.zeros(n_total, dtype=np.float64)
            for i in range(n_total):
                scores[i], _, _ = score_local_window(
                    q_a_used, omegas[i], args.T_A, args.W, ctx)
            truth_score = float(scores[truth_idx])
            truth_rank = int((scores < truth_score).sum() + 1)
            top_idx = int(np.argmin(scores))
            top_score = float(scores[top_idx])
            top_om = omegas[top_idx]
            top_om_mag = float(np.linalg.norm(top_om))
            top_om_dir_err = float(angle_between_unit_vecs_deg(
                top_om[None, :] / max(1e-12, top_om_mag),
                om_a_truth_dir)[0])
            top_om_mag_err_pct = (top_om_mag - om_a_truth_mag) / om_a_truth_mag * 100
            wall = time.time() - t_p
            row = {
                "perturb_deg": p_deg,
                "actual_geodesic_deg": actual_geodesic,
                "trial": trial,
                "truth_rank": truth_rank,
                "truth_rho_local": float(np.sqrt(truth_score) / 0.05),
                "truth_score": truth_score,
                "top1_rho_local": float(np.sqrt(top_score) / 0.05),
                "top1_om_dir_err_deg": top_om_dir_err,
                "top1_om_mag_err_pct": top_om_mag_err_pct,
                "top1_is_truth": bool(top_idx == truth_idx),
                "wall_s": wall,
            }
            rows.append(row)
            log(f"  perturb={p_deg:5.2f}° (actual {actual_geodesic:5.2f}°)  trial {trial}  "
                f"truth_rank={truth_rank:>5}/{n_total}  truth_ρ={row['truth_rho_local']:6.2f}  "
                f"top1_ρ={row['top1_rho_local']:6.2f}  top1_dir_err={top_om_dir_err:5.1f}°  "
                f"top1_|ω|Δ={top_om_mag_err_pct:+.1f}%  wall={wall:.1f}s")

    # Summary
    log(f"\n[4/4] summary")
    log(f"  perturb_deg | trials | truth_rank median (range) | truth_ρ median | top1_ρ median")
    log(f"  ----------- | ------ | ------------------------- | -------------- | -------------")
    summary_per_p = {}
    for p_deg in perturbations:
        rs = [r for r in rows if r["perturb_deg"] == p_deg]
        ranks = [r["truth_rank"] for r in rs]
        truth_rhos = [r["truth_rho_local"] for r in rs]
        top1_rhos = [r["top1_rho_local"] for r in rs]
        summary_per_p[p_deg] = {
            "n_trials": len(rs),
            "truth_rank_median": int(np.median(ranks)),
            "truth_rank_min": int(min(ranks)),
            "truth_rank_max": int(max(ranks)),
            "truth_rho_median": float(np.median(truth_rhos)),
            "top1_rho_median": float(np.median(top1_rhos)),
            "top1_is_truth_count": int(sum(r["top1_is_truth"] for r in rs)),
        }
        log(f"    {p_deg:5.2f}°    |   {len(rs):2d}   |  {int(np.median(ranks)):>4d}  "
            f"({min(ranks):>4d}–{max(ranks):>4d})  |   {np.median(truth_rhos):6.2f}    |   "
            f"{np.median(top1_rhos):6.2f}   "
            f"{'(top1=truth ' + str(sum(r['top1_is_truth'] for r in rs)) + '/' + str(len(rs)) + ')' if sum(r['top1_is_truth'] for r in rs) else ''}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    p_arr = np.array([r["perturb_deg"] for r in rows])
    rank_arr = np.array([r["truth_rank"] for r in rows])
    rho_arr = np.array([r["truth_rho_local"] for r in rows])
    top1_rho_arr = np.array([r["top1_rho_local"] for r in rows])

    axes[0].scatter(p_arr, rank_arr, alpha=0.6, label="trials")
    medians_rank = np.array([summary_per_p[p]["truth_rank_median"] for p in perturbations])
    axes[0].plot(perturbations, medians_rank, "k-", lw=2, label="median")
    axes[0].axhline(1, color="red", lw=0.5, ls="--", label="rank 1 (best)")
    axes[0].axhline(n_total, color="grey", lw=0.5, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("q_a perturbation (deg, geodesic)")
    axes[0].set_ylabel(f"truth-ω rank (out of {n_total})")
    axes[0].set_title(f"seed {args.seed} T_A={args.T_A} W={args.W}: rank vs q_a noise")
    axes[0].legend()

    axes[1].scatter(p_arr, rho_arr, alpha=0.6, label="truth-ω ρ_local (trials)")
    axes[1].scatter(p_arr, top1_rho_arr, alpha=0.6, marker="x",
                    color="orange", label="top-1 ρ_local (trials)")
    axes[1].set_xlabel("q_a perturbation (deg, geodesic)")
    axes[1].set_ylabel("ρ_local")
    axes[1].set_title("ρ_local at truth-ω vs at top-1 grid point")
    axes[1].axhline(2, color="grey", lw=0.5, ls=":", label="Band A/B boundary")
    axes[1].axhline(4, color="grey", lw=0.5, ls=":", label="Band B/C boundary")
    axes[1].axhline(8, color="grey", lw=0.5, ls=":", label="Band C/D boundary")
    axes[1].legend()
    axes[1].set_yscale("log")

    plt.tight_layout()
    plot_path = out_dir / "rank_vs_perturbation.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log(f"\nSaved: {plot_path}")

    summary = {
        "seed": int(args.seed),
        "T_A": int(args.T_A),
        "W": int(args.W),
        "n_grid": int(n_total),
        "perturbations_deg": perturbations,
        "n_trials": int(args.n_trials),
        "rows": rows,
        "summary_per_perturb": {
            f"{p:.2f}": v for p, v in summary_per_p.items()
        },
        "wall_total_s": float(time.time() - t_overall),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Saved: {summary_path}")
    log(f"\nWall total: {time.time() - t_overall:.1f}s")


if __name__ == "__main__":
    main()
