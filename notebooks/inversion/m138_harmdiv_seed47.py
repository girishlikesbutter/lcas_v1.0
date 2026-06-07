"""m138 seed 47 — H1 with harmonic-division |ω|-grid.

Replaces `estimate_omega_mag_grid` (single peak-count base) with:
    bases = LS_top1 × {1, 1/2, 1/3, 1/4}
    grid  = union of [span_low, span_high] × base × n_mags_per_basis  (log-spaced)

Reuses cached levelset_ckpt.npz + components_ckpt.npz from the original H1
run; only re-executes Stage 3 (cost-grid scoring). Saves to
`seed_047/h1_harmdiv/`.

Config (per the documented harmonic-division strategy):
  - 4 bases (k=1..4)
  - n_mags_per_basis = 20
  - span = [0.3, 3.0]
  - eps_cluster_deg = 10.0  (matches original H1 config; cost SNR strongest there
                              when grid is correct)
  - n_omega_dirs = 1000     (matches original)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.signal import lombscargle, find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.traj_source import load_truth, CANONICAL_NOISE_SIGMA
from lib.experiment_setup import setup_experiment
from m138_isoshell_h1 import (
    fibonacci_sphere, score_isoshell_omega_grid, geodesic_deg, quat_mul,
)

SEED = 47
SOURCE = "m048"
OUT = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "h1_harmdiv"
OUT.mkdir(parents=True, exist_ok=True)
LEVELSET_CKPT = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "levelset_ckpt.npz"

EPS_CLUSTER_DEG = 10.0
N_OMEGA_DIRS = 1000
N_MAGS_PER_BASIS = 20
SPAN_LOW = 0.3
SPAN_HIGH = 3.0
LS_POWER_THR = 0.0  # ignore — we just take the top peak via find_peaks


def harmonic_division_grid(obs_lc, obs_times, n_mags_per_basis=20,
                            span_low=0.3, span_high=3.0, n_freqs=4000):
    """LS top-1 frequency divided by {1, 2, 3, 4} as 4 candidate bases.

    Each base spans [span_low, span_high] × base with n_mags_per_basis log-spaced.
    Final grid = sorted unique union (some overlap collapses naturally at
    geomspace boundaries).
    """
    valid = np.isfinite(obs_lc)
    t = obs_times[valid]
    s = -obs_lc[valid]
    s = s - np.mean(s)
    dt = np.median(np.diff(t))
    f_min = 1.0 / (t[-1] - t[0])
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, n_freqs)
    power = lombscargle(t, s, 2 * np.pi * freqs, normalize=True)

    idx, _ = find_peaks(power, distance=5)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(power))])
    top_idx = idx[np.argmax(power[idx])]
    f_top = float(freqs[top_idx])
    omega_top = 2 * np.pi * f_top

    bases = [omega_top / k for k in (1, 2, 3, 4)]
    sub_grids = [np.geomspace(span_low * b, span_high * b, n_mags_per_basis) for b in bases]
    grid = np.unique(np.round(np.concatenate(sub_grids), 12))
    return grid, {
        "ls_top_freq_hz": f_top,
        "ls_top_omega": omega_top,
        "bases": [float(b) for b in bases],
    }


def main():
    t_total = time.time()

    # 1. Load truth + setup ctx (matches m138's m048 path)
    truth = load_truth(SEED, SOURCE)
    obs_lc = truth["observed_lc"]
    true_q0 = truth["q0_wxyz"]
    true_omega = truth["omega0_rad"]
    I_tensor = truth["inertia_tensor"]
    I_diag = np.diag(I_tensor) if I_tensor.shape == (3, 3) else np.asarray(I_tensor)
    truth_mag = float(np.linalg.norm(true_omega))
    truth_dir = true_omega / max(truth_mag, 1e-12)

    print(f"=== m138 seed 47 harmdiv rerun ===")
    print(f"  truth |ω| = {truth_mag:.5f} rad/s")

    ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                           random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                           start_et=truth["start_et"], skip_true_lc=True)

    # 2. Load cached levelset (Stage 1) + components (Stage 2)
    if not LEVELSET_CKPT.exists():
        raise SystemExit(f"missing {LEVELSET_CKPT} — run original H1 first to cache stages 1-2")
    print(f"  loading cached levelset: {LEVELSET_CKPT}")
    z = np.load(LEVELSET_CKPT, allow_pickle=True)
    kept_per_epoch_obj = z["kept_per_epoch"]
    levelset = {
        "q_grid": z["q_grid"],
        "constraint_idx": z["constraint_idx"],
        "kept_per_epoch": [np.asarray(k, dtype=int) for k in kept_per_epoch_obj],
    }
    print(f"  constraint epochs: {len(levelset['constraint_idx'])}, "
          f"q_grid: {levelset['q_grid'].shape}")

    components = None  # not used by score_isoshell_omega_grid but kept for signature

    # 3. Build harmonic-division grid
    omega_mags, ls_meta = harmonic_division_grid(obs_lc, ctx.observation_times,
                                                  n_mags_per_basis=N_MAGS_PER_BASIS,
                                                  span_low=SPAN_LOW, span_high=SPAN_HIGH)
    print(f"  LS top-1 freq: {ls_meta['ls_top_freq_hz']:.6f} Hz "
          f"(|ω| {ls_meta['ls_top_omega']:.5f})")
    print(f"  bases (k=1..4): {[f'{b:.5f}' for b in ls_meta['bases']]}")
    print(f"  |ω|-grid size after union: {len(omega_mags)}")
    print(f"    grid range: [{omega_mags.min():.5f}, {omega_mags.max():.5f}]")
    nearest_truth = omega_mags[np.argmin(np.abs(omega_mags - truth_mag))]
    nearest_pct = (nearest_truth / truth_mag - 1) * 100
    print(f"    nearest grid pt to truth |ω|: {nearest_truth:.5f}  ({nearest_pct:+.3f}%)")

    # 4. Stage 3 score
    print(f"\n  Stage 3: scoring {len(omega_mags)} mags × {N_OMEGA_DIRS} dirs = "
          f"{len(omega_mags)*N_OMEGA_DIRS} candidates  (eps_cluster={EPS_CLUSTER_DEG}°)...",
          flush=True)
    omega_dirs = fibonacci_sphere(N_OMEGA_DIRS)
    score = score_isoshell_omega_grid(levelset, components, ctx, omega_dirs,
                                       omega_mags, I_diag,
                                       eps_cluster_deg=EPS_CLUSTER_DEG, verbose=True)

    omega_batch = score["omega_batch"]
    cost = score["cost"]
    q0_est = score["q0_estimate"]

    # 5. Audit top-30
    order = np.argsort(cost)
    top_k_n = 30
    top_k_idx = order[:top_k_n]
    top_k_omegas = omega_batch[top_k_idx]
    top_k_costs = cost[top_k_idx]

    norms = np.linalg.norm(top_k_omegas, axis=1)
    dirs = top_k_omegas / norms[:, None].clip(1e-12)
    dot = np.clip(dirs @ truth_dir, -1.0, 1.0)
    dir_errs = np.degrees(np.arccos(dot))
    mag_errs_pct = (norms - truth_mag) / truth_mag * 100

    # joint criterion: 5° dir AND 5% mag
    joint_mask = (dir_errs <= 5.0) & (np.abs(mag_errs_pct) <= 5.0)
    joint_count = int(joint_mask.sum())
    near_dir_mask = dir_errs <= 5.0

    # grid-level oracle diagnostics
    all_norms = np.linalg.norm(omega_batch, axis=1)
    all_dirs = omega_batch / all_norms[:, None].clip(1e-12)
    all_dot = np.clip(all_dirs @ truth_dir, -1.0, 1.0)
    all_dir_errs = np.degrees(np.arccos(all_dot))
    all_mag_pct = (all_norms - truth_mag) / truth_mag * 100
    grid_joint_mask = (all_dir_errs <= 5.0) & (np.abs(all_mag_pct) <= 5.0)
    grid_n_joint = int(grid_joint_mask.sum())
    grid_min_dir = float(all_dir_errs.min())
    grid_min_idx = int(np.argmin(all_dir_errs))
    grid_min_cost_rank = int((cost <= cost[grid_min_idx]).sum() - 1)

    if grid_n_joint > 0:
        joint_idx_in_grid = np.where(grid_joint_mask)[0]
        # rank of best (lowest-cost) joint candidate across full grid
        best_joint_in_grid = joint_idx_in_grid[np.argmin(cost[joint_idx_in_grid])]
        best_joint_cost_rank = int((cost <= cost[best_joint_in_grid]).sum() - 1)
        best_joint_dir = float(all_dir_errs[best_joint_in_grid])
        best_joint_mag = float(all_mag_pct[best_joint_in_grid])
    else:
        best_joint_cost_rank = -1
        best_joint_dir = best_joint_mag = None

    # q0 errors top-30
    twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
    q0_to_truth = [float(geodesic_deg(q0_est[i], true_q0)) for i in top_k_idx]
    q0_to_twin = [float(geodesic_deg(q0_est[i], twin_q0)) for i in top_k_idx]

    summary = {
        "seed": SEED,
        "source": SOURCE,
        "experiment": "m138_h1_harmdiv_seed47",
        "config": {
            "eps_cluster_deg": EPS_CLUSTER_DEG,
            "n_omega_dirs": N_OMEGA_DIRS,
            "n_mags_per_basis": N_MAGS_PER_BASIS,
            "span_low": SPAN_LOW,
            "span_high": SPAN_HIGH,
            "n_omega_mags_total": len(omega_mags),
            "ls_meta": ls_meta,
            "n_constraint_epochs": len(levelset["constraint_idx"]),
        },
        "rank1_omega_dir_err_deg": float(dir_errs[0]),
        "rank1_omega_mag_err_pct": float(mag_errs_pct[0]),
        "rank1_cost": float(top_k_costs[0]),
        "top_k_omega_dir_errs": dir_errs.tolist(),
        "top_k_omega_mag_errs_pct": mag_errs_pct.tolist(),
        "top_k_costs": top_k_costs.tolist(),
        "top_k_q0_to_truth_deg": q0_to_truth,
        "top_k_q0_to_twin_deg": q0_to_twin,
        "pool_min_omega_dir_err_deg_in_topK": float(dir_errs.min()),
        "n_top30_within_5deg": int(near_dir_mask.sum()),
        "n_top30_joint_5deg_5pct": joint_count,
        "grid_min_omega_dir_err_deg": grid_min_dir,
        "grid_min_cost_rank": grid_min_cost_rank,
        "grid_n_joint_5deg_5pct": grid_n_joint,
        "best_joint_in_grid_cost_rank": best_joint_cost_rank,
        "best_joint_in_grid_dir_err": best_joint_dir,
        "best_joint_in_grid_mag_pct": best_joint_mag,
        "truth_omega_mag": truth_mag,
        "truth_omega_mag_deg_per_s": float(np.degrees(truth_mag)),
        "stage3_wall_s": float(score["elapsed_s"]),
        "total_wall_s": float(time.time() - t_total),
    }

    # Save isoshell ckpt for re-rank experiments
    np.savez_compressed(OUT / "isoshell_ckpt.npz",
                        omega_batch=omega_batch, cost=cost,
                        q0_estimate=q0_est,
                        top_k_idx=top_k_idx,
                        omega_mags_grid=omega_mags)

    with (OUT / "result.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Verdict
    print(f"\n=== verdict ===")
    print(f"  rank-1 ω-dir / mag: {dir_errs[0]:.2f}° / {mag_errs_pct[0]:+.1f}%")
    print(f"  pool_min ω-dir in top-30:        {dir_errs.min():.2f}°")
    print(f"  n top-30 within 5° dir:          {near_dir_mask.sum()}")
    print(f"  n top-30 joint (5° dir + 5% mag): {joint_count}    "
          f"<-- this is the m115-bridgeable count")
    print(f"  grid_min ω-dir:                  {grid_min_dir:.2f}° (cost rank {grid_min_cost_rank})")
    print(f"  grid_n joint (5° dir + 5% mag):  {grid_n_joint}")
    if grid_n_joint > 0:
        print(f"    best joint in grid: dir={best_joint_dir:.2f}°, mag={best_joint_mag:+.2f}%, "
              f"cost rank {best_joint_cost_rank}")
    print(f"  total wall: {time.time() - t_total:.1f}s")
    print(f"  saved: {OUT}/result.json")
    return summary


if __name__ == "__main__":
    main()
