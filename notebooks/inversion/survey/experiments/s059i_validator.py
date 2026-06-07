"""s059i_validator — local-window cost surface at truth q_a, ω-grid only.

Tests whether surrogate-MSE on a local window discriminates ω when q_a is
fixed at truth. If truth-ω ranks at the bottom (top 1% or rank 1), the
ω-grid premise of s059i holds and the full search is worth building. If
truth-ω ranks in the middle of the pack, the local window underconstrains
ω even with optimal q_a, and a pure-ω-grid architecture cannot resolve it.

Pipeline:
  1. Build ctx for the seed; re-propagate truth (q0, ω0) to read truth-q
     and truth-ω_body at T_A.
  2. Build ω-grid: Fibonacci on the unit sphere × |ω| factors, with the
     true (dir, mag) inserted as point 0 so it is exactly representable.
  3. For each ω in the grid: back-propagate (truth_q_a, ω) from T_A to
     T_A-W, forward-propagate to T_A+W via real dynamics, surrogate-eval
     k1/k2, mean-square residual vs cached truth LC on the window.
  4. Report rank of truth-ω + top-K listing + summary JSON + score plot.

Usage:
    python experiments/s059i_validator.py --seed 28
    python experiments/s059i_validator.py --seed 28 --T-A 25 --W 10 \
        --n-dirs 200 --n-mags 6 --mag-bracket 0.30
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

from lib.forward import propagate_to_body_frame  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from lib.surrogate_eval import predict as surrogate_predict  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from experiments.s059_pilot import back_propagate  # noqa: E402


# ----- helpers -----------------------------------------------------------------


def fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic n-point Fibonacci lattice on the unit sphere.

    Returns (n, 3) unit vectors with near-uniform angular spacing.
    """
    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * (i / phi)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def angle_between_unit_vecs_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between rows of `a` and a single `b`. Both unit."""
    cos_a = np.clip(a @ b, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


# ----- core score -------------------------------------------------------------


def score_local_window(q_a_wxyz, om_a_rad, T_A, W, ctx):
    """Score (q_a, ω_a) at T_A by surrogate-MSE on epochs [T_A-W, T_A+W].

    Strategy mirrors s059e.propagate_local_window: back-prop to lo, then
    forward-prop the entire window in a single propagate_to_body_frame call.
    Returns (mse, rho, n_window).
    """
    obs_times = ctx["observation_times"]
    n = len(obs_times)
    lo = max(0, T_A - W)
    hi = min(n, T_A + W + 1)
    target = ctx["mag_hifi_truth"][lo:hi]

    dt_back = float(obs_times[T_A] - obs_times[lo])
    if dt_back > 0:
        q_lo, om_lo = back_propagate(q_a_wxyz, om_a_rad, dt_back,
                                      ctx["inertia_tensor"])
    else:
        q_lo, om_lo = q_a_wxyz, om_a_rad

    k1, k2, _ = propagate_to_body_frame(
        q0_wxyz=q_lo, omega0_rad=om_lo,
        observation_times=obs_times[lo:hi],
        sun_pos=ctx["sun_pos"][lo:hi],
        obs_pos=ctx["obs_pos"][lo:hi],
        sat_pos=ctx["sat_pos"][lo:hi],
        inertia_tensor=ctx["inertia_tensor"],
        mode="tumbling",
    )
    pred = surrogate_predict(k1, k2, ctx["obs_dist"][lo:hi])
    mse = float(np.mean((pred - target) ** 2))
    return mse, float(np.sqrt(mse) / 0.05), int(hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--T-A", type=int, default=25)
    ap.add_argument("--W", type=int, default=10,
                    help="local window radius (epochs)")
    ap.add_argument("--n-dirs", type=int, default=200)
    ap.add_argument("--n-mags", type=int, default=6)
    ap.add_argument("--mag-bracket", type=float, default=0.30,
                    help="|ω| grid spans truth × [1-b, 1+b]")
    ap.add_argument("--out-root", default=str(SURVEY / "results" / "s059i_validator"))
    args = ap.parse_args()

    out_dir = Path(args.out_root) / f"seed{args.seed:03d}_T{args.T_A:03d}_W{args.W:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059i_validator: seed {args.seed}, T_A={args.T_A}, W={args.W} ===")

    # 1. Context + truth (q_a, ω_a) at T_A
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
    log(f"  truth q_a (wxyz) = [{q_a_truth[0]:+.6f}, {q_a_truth[1]:+.6f}, "
        f"{q_a_truth[2]:+.6f}, {q_a_truth[3]:+.6f}]")
    log(f"  truth ω_body at T_A = {om_a_truth} rad/s")
    log(f"  |ω_body|_truth = {om_a_truth_mag:.6f} rad/s "
        f"({np.degrees(om_a_truth_mag):.4f} dps)")

    # Smoke: score at truth-ω should give the surrogate noise floor
    mse_truth, rho_truth, nW = score_local_window(
        q_a_truth, om_a_truth, args.T_A, args.W, ctx)
    log(f"  SMOKE: score at (truth_q_a, truth_ω) → MSE={mse_truth:.6e}, "
        f"ρ={rho_truth:.3f} on window of {nW} epochs")

    # 2. ω grid: Fibonacci dirs × magnitudes. Truth (dir, mag) inserted at index 0.
    log(f"\n[2/4] build ω grid: {args.n_dirs} dirs × {args.n_mags} mags")
    fib_dirs = fibonacci_sphere(args.n_dirs)
    all_dirs = np.vstack([om_a_truth_dir[None, :], fib_dirs])  # (n_dirs+1, 3)

    # Magnitudes: truth-mag at idx 0, then linspace across the bracket
    factor_lo = 1.0 - args.mag_bracket
    factor_hi = 1.0 + args.mag_bracket
    mag_factors = np.concatenate([
        [1.0],
        np.linspace(factor_lo, factor_hi, args.n_mags),
    ])
    all_mags = mag_factors * om_a_truth_mag
    log(f"  |ω| grid factors: {mag_factors}")

    # Cartesian product → (n_total, 3)
    n_dir_total = len(all_dirs)
    n_mag_total = len(all_mags)
    n_total = n_dir_total * n_mag_total
    omegas = (all_dirs[:, None, :] * all_mags[None, :, None]).reshape(-1, 3)
    truth_idx = 0  # dir 0 (truth) × mag 0 (truth) → flat idx 0
    assert np.allclose(omegas[truth_idx], om_a_truth), \
        f"truth not at idx 0: {omegas[truth_idx]} vs {om_a_truth}"
    log(f"  total grid points: {n_total} (truth at flat idx {truth_idx})")

    # 3. Score every ω
    log(f"\n[3/4] score grid (single-thread, sequential)")
    t_score = time.time()
    scores = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        scores[i], _, _ = score_local_window(
            q_a_truth, omegas[i], args.T_A, args.W, ctx)
        if (i + 1) % max(1, n_total // 10) == 0:
            log(f"  {i+1}/{n_total} done, elapsed {time.time()-t_score:.1f}s")
    wall_score = time.time() - t_score
    log(f"  scoring wall: {wall_score:.1f}s "
        f"({n_total/wall_score:.0f} grid/s)")

    # Sanity: truth grid score should match smoke
    assert abs(scores[truth_idx] - mse_truth) < 1e-12, \
        f"grid truth score {scores[truth_idx]} != smoke {mse_truth}"

    # 4. Report
    log(f"\n[4/4] results")
    truth_score = float(scores[truth_idx])
    truth_rank = int((scores < truth_score).sum() + 1)
    pct = 100.0 * truth_rank / n_total
    log(f"  truth-ω: MSE={truth_score:.6e}, ρ={np.sqrt(truth_score)/0.05:.4f}, "
        f"rank {truth_rank}/{n_total} ({pct:.2f}-th percentile)")

    # Top-20 list with diagnostics
    sort_idx = np.argsort(scores)
    log(f"\n  top-20 by MSE:")
    log(f"  {'rank':>4} {'flat_idx':>8} {'MSE':>12} {'ρ':>8} "
        f"{'ω_dir°':>8} {'|ω|Δ%':>8}  truth?")
    for j in range(min(20, n_total)):
        i = sort_idx[j]
        om = omegas[i]
        mag = float(np.linalg.norm(om))
        if mag > 1e-12:
            cos_d = float(np.clip(om @ om_a_truth / (mag * om_a_truth_mag), -1, 1))
            ang_d = float(np.degrees(np.arccos(cos_d)))
        else:
            ang_d = float("nan")
        mag_pct = (mag - om_a_truth_mag) / om_a_truth_mag * 100
        marker = "TRUTH" if i == truth_idx else ""
        log(f"  {j+1:>4} {i:>8} {scores[i]:>12.6e} "
            f"{np.sqrt(scores[i])/0.05:>8.3f} "
            f"{ang_d:>8.2f} {mag_pct:>+8.2f}  {marker}")

    # Plot: MSE as a heatmap over (dir_idx, mag_idx); rank histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rho_grid = np.sqrt(scores) / 0.05
    rho_grid_2d = rho_grid.reshape(n_dir_total, n_mag_total)
    im = axes[0].imshow(np.log10(np.maximum(rho_grid_2d, 1e-3)),
                        aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xlabel("|ω| factor index (0=truth)")
    axes[0].set_ylabel("direction index (0=truth)")
    axes[0].set_title(
        f"log10(ρ_local) — seed {args.seed}, T_A={args.T_A}, W={args.W}\n"
        f"truth at (0,0); rank {truth_rank}/{n_total}")
    fig.colorbar(im, ax=axes[0])
    axes[0].plot(0, 0, "rx", markersize=12)

    axes[1].hist(np.log10(np.maximum(rho_grid, 1e-3)), bins=50,
                 alpha=0.7, label="all grid points")
    axes[1].axvline(np.log10(np.maximum(rho_grid[truth_idx], 1e-3)),
                    color="red", lw=2, label=f"truth ρ={rho_grid[truth_idx]:.3f}")
    axes[1].set_xlabel("log10(ρ_local)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Score distribution")
    axes[1].legend()

    plt.tight_layout()
    plot_path = out_dir / "score_grid.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log(f"\nSaved: {plot_path}")

    # Summary JSON
    summary = {
        "seed": int(args.seed),
        "T_A": int(args.T_A),
        "W": int(args.W),
        "n_dirs_total": int(n_dir_total),
        "n_mags_total": int(n_mag_total),
        "n_grid": int(n_total),
        "mag_bracket": float(args.mag_bracket),
        "truth_q_a_wxyz": q_a_truth.tolist(),
        "truth_omega_a_body_rad": om_a_truth.tolist(),
        "truth_omega_a_mag_dps": float(np.degrees(om_a_truth_mag)),
        "truth_score_mse": truth_score,
        "truth_rho_local": float(np.sqrt(truth_score) / 0.05),
        "truth_rank": int(truth_rank),
        "truth_percentile": float(pct),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_median": float(np.median(scores)),
        "wall_score_s": float(wall_score),
        "wall_total_s": float(time.time() - t_overall),
        "top20": [
            {
                "rank": j + 1,
                "flat_idx": int(sort_idx[j]),
                "mse": float(scores[sort_idx[j]]),
                "rho": float(np.sqrt(scores[sort_idx[j]]) / 0.05),
                "omega_dir_err_deg": float(angle_between_unit_vecs_deg(
                    omegas[sort_idx[j]:sort_idx[j]+1] /
                    max(1e-12, np.linalg.norm(omegas[sort_idx[j]])),
                    om_a_truth_dir)[0]),
                "omega_mag_err_pct": float(
                    (np.linalg.norm(omegas[sort_idx[j]]) - om_a_truth_mag)
                    / om_a_truth_mag * 100),
                "is_truth": bool(sort_idx[j] == truth_idx),
            }
            for j in range(min(20, n_total))
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Saved: {summary_path}")

    # NPZ with full grid
    npz_path = out_dir / "score_grid.npz"
    np.savez(
        npz_path,
        omegas=omegas, scores=scores,
        q_a_truth=q_a_truth, om_a_truth=om_a_truth,
        T_A=args.T_A, W=args.W,
    )
    log(f"Saved: {npz_path}")

    log(f"\nWall total: {time.time() - t_overall:.1f}s")


if __name__ == "__main__":
    main()
