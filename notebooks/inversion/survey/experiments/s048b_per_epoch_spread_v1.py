"""s048b — per-epoch C_t spread sweep on seed 89, surrogate v1, 15 evenly-spaced epochs.

For each of 15 epochs across the LC, filter 500k random q's against the measured
mag at that epoch and characterise:
  - count of survivors |C_t|
  - distance from truth-q to nearest survivor (truth survival)
  - mag value at the epoch (so we can correlate)

Output: NPZ checkpoint + 15-panel scatter plot showing q-spread, kills vs survivors,
and truth location per epoch.

Surrogate v1 (4.5 μs/sample) loaded directly from /home/girish/surrogate_model.
R(q) cached once across all 15 epochs (the smart way).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
SURROGATE_PATH = Path("/home/girish/surrogate_model")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURROGATE_PATH))

from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1
from src.dynamics.attitude_propagator import propagate_attitude
from lib.filter_costs import load_static_geometry

SEED = 89
N_SAMPLES = 500_000
N_EPOCHS_TO_SWEEP = 30
EPOCH_SELECTION = "dimmest"            # 'evenly_spaced' | 'dimmest' | 'brightest'
TOLERANCE_MAG = 0.10
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
RNG_SEED = 42

OUT_DIR = SURVEY_DIR / "results" / "s048_peak_cascade_smoke" / f"seed{SEED:03d}_v1_spread_n{N_EPOCHS_TO_SWEEP}_{EPOCH_SELECTION}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()

    # ---- load v1 surrogate ----
    print("[load] surrogate v1 ...")
    v1_weights = SURROGATE_PATH / "surrogate_model" / "s10_5M_weights.npz"
    v1_norm = SURROGATE_PATH / "surrogate_model" / "s10_5M_normalization.npz"
    model = SurrogateV1(str(v1_weights), str(v1_norm))
    print(f"[load] {model}")

    # ---- load trajectory ----
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{SEED:03d}.npz"
    print(f"[load] {traj_path}")
    d = np.load(traj_path)
    obs_times = np.asarray(d["observation_times"], float)
    sun_pos = np.asarray(d["sun_pos"], float)
    obs_pos = np.asarray(d["obs_pos"], float)
    sat_pos = np.asarray(d["sat_pos"], float)
    obs_dist_all = np.asarray(d["obs_dist"], float)
    mag_hifi = np.asarray(d["mag_hifi"], float)
    quats_truth = np.asarray(d["quaternions"], float)
    q0_truth = np.asarray(d["q0_wxyz"], float)
    omega0_truth = np.asarray(d["omega0_rad"], float)
    omega_mag_dps = float(d["omega_mag_dps"])
    pe = np.asarray(d["hifi_peak_epochs"], int)
    pp = np.asarray(d["hifi_peak_prominences"], float)

    sun_vec_j2000 = sun_pos - sat_pos
    obs_vec_j2000 = obs_pos - sat_pos
    sun_unit = sun_vec_j2000 / np.linalg.norm(sun_vec_j2000, axis=1, keepdims=True)
    obs_unit = obs_vec_j2000 / np.linalg.norm(obs_vec_j2000, axis=1, keepdims=True)

    N_OBS = len(obs_times)
    finite_mask = np.isfinite(mag_hifi)
    if EPOCH_SELECTION == "evenly_spaced":
        epoch_indices = np.linspace(0, N_OBS - 1, N_EPOCHS_TO_SWEEP).astype(int)
    elif EPOCH_SELECTION == "dimmest":
        # highest mag = dimmest. exclude any non-finite epochs.
        finite_idx = np.where(finite_mask)[0]
        order = finite_idx[np.argsort(-mag_hifi[finite_idx])]
        epoch_indices = np.sort(order[:N_EPOCHS_TO_SWEEP])
    elif EPOCH_SELECTION == "brightest":
        finite_idx = np.where(finite_mask)[0]
        order = finite_idx[np.argsort(mag_hifi[finite_idx])]
        epoch_indices = np.sort(order[:N_EPOCHS_TO_SWEEP])
    else:
        raise ValueError(f"Unknown EPOCH_SELECTION: {EPOCH_SELECTION}")
    print(f"[sweep] {N_EPOCHS_TO_SWEEP} epochs ({EPOCH_SELECTION}) @ idx: "
          f"{epoch_indices.tolist()}")
    print(f"[sweep] mag range across selected: "
          f"[{mag_hifi[epoch_indices].min():.2f}, {mag_hifi[epoch_indices].max():.2f}]")

    # ---- generate 500k random q ONCE ----
    print(f"[setup] sampling {N_SAMPLES} random q on SO(3)...")
    t1 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    R_random = Rotation.random(N_SAMPLES, random_state=rng)
    q_xyzw = R_random.as_quat()
    q_pool_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    R_cache = R_random.as_matrix()                     # (N, 3, 3)  — body-frame i→b
    print(f"[setup] R cache built in {time.time()-t1:.1f}s")

    # also rotvec for plotting (axis * angle, rad)
    rotvec_pool = R_random.as_rotvec()                 # (N, 3)

    # ---- truth q at each swept epoch ----
    q_truth_at = quats_truth[epoch_indices]            # (15, 4)
    rotvec_truth_at = Rotation.from_quat(
        q_truth_at[:, [1, 2, 3, 0]]).as_rotvec()       # (15, 3)

    # ---- per-epoch loop ----
    pred_all = np.zeros((N_EPOCHS_TO_SWEEP, N_SAMPLES), dtype=np.float32)
    survive_all = np.zeros((N_EPOCHS_TO_SWEEP, N_SAMPLES), dtype=bool)
    measured_at = np.zeros(N_EPOCHS_TO_SWEEP)
    n_survivors = np.zeros(N_EPOCHS_TO_SWEEP, dtype=int)
    nearest_truth_deg = np.zeros(N_EPOCHS_TO_SWEEP)

    t_loop = time.time()
    for j, ep in enumerate(epoch_indices):
        t_ep = time.time()
        # k1, k2 in body frame for ALL 500k candidates at this epoch
        k1_body = np.einsum('nij,j->ni', R_cache, sun_unit[ep])
        k2_body = np.einsum('nij,j->ni', R_cache, obs_unit[ep])
        obs_dist = np.full(N_SAMPLES, obs_dist_all[ep])
        pred = model.predict_magnitude(k1_body, k2_body, SP_ANGLE_DEG,
                                         AD_ANGLE_DEG, obs_dist)
        measured = mag_hifi[ep]
        keep = np.abs(pred - measured) < TOLERANCE_MAG

        # nearest truth-q distance
        dots = np.abs(q_pool_wxyz @ q_truth_at[j])
        dots_keep = dots[keep] if keep.any() else np.array([0.0])
        d_keep = np.degrees(2.0 * np.arccos(np.clip(dots_keep, 0.0, 1.0)))
        nearest_truth_deg[j] = float(d_keep.min()) if keep.any() else float("inf")

        pred_all[j] = pred.astype(np.float32)
        survive_all[j] = keep
        measured_at[j] = measured
        n_survivors[j] = int(keep.sum())
        print(f"[epoch {ep:3d} ({j+1:2d}/{N_EPOCHS_TO_SWEEP})] mag={measured:6.3f}  "
              f"survivors={int(keep.sum()):6d} ({100.0*keep.mean():.4f}%)  "
              f"nearest_truth={nearest_truth_deg[j]:6.2f}°  "
              f"wall={time.time()-t_ep:.1f}s")

    print(f"[loop] total wall: {time.time()-t_loop:.1f}s")

    # ---- save NPZ ----
    out_npz = OUT_DIR / "spread.npz"
    np.savez(out_npz,
             q_pool_wxyz=q_pool_wxyz, rotvec_pool=rotvec_pool,
             epoch_indices=epoch_indices, obs_times=obs_times[epoch_indices],
             measured_at=measured_at, pred_all=pred_all, survive_all=survive_all,
             n_survivors=n_survivors, nearest_truth_deg=nearest_truth_deg,
             q_truth_at=q_truth_at, rotvec_truth_at=rotvec_truth_at,
             tolerance_mag=TOLERANCE_MAG, omega_mag_dps=omega_mag_dps,
             mag_hifi=mag_hifi, hifi_peak_epochs=pe, hifi_peak_prominences=pp)
    print(f"[saved] {out_npz}")

    # ---- summary JSON ----
    summary = {
        "seed": SEED, "omega_mag_dps": omega_mag_dps,
        "n_samples": N_SAMPLES, "tolerance_mag": TOLERANCE_MAG,
        "n_epochs_swept": N_EPOCHS_TO_SWEEP,
        "epoch_indices": epoch_indices.tolist(),
        "obs_times_at_epochs": obs_times[epoch_indices].tolist(),
        "measured_mag_at_epochs": measured_at.tolist(),
        "n_survivors_per_epoch": n_survivors.tolist(),
        "nearest_truth_deg_per_epoch": nearest_truth_deg.tolist(),
        "wall_total_s": time.time() - t0,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")

    # ---- 15-panel plot ----
    print(f"[plot] generating {N_EPOCHS_TO_SWEEP}-panel spread plot...")
    n_cols = 6
    n_rows = (N_EPOCHS_TO_SWEEP + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    # subsample for plotting (500k → 8k visible kills + all survivors)
    n_plot_kill = 8000
    plot_rng = np.random.default_rng(0)

    for j, ax in enumerate(axes.flat):
        ep = int(epoch_indices[j])
        keep = survive_all[j]
        kill = ~keep
        n_keep = int(keep.sum())

        # subsample kills
        kill_idx = np.where(kill)[0]
        if len(kill_idx) > n_plot_kill:
            kill_idx = plot_rng.choice(kill_idx, n_plot_kill, replace=False)

        ax.scatter(rotvec_pool[kill_idx, 0], rotvec_pool[kill_idx, 1],
                   s=0.5, c="lightgrey", alpha=0.4, label="killed")
        if n_keep > 0:
            sx = rotvec_pool[keep, 0]; sy = rotvec_pool[keep, 1]
            ax.scatter(sx, sy, s=8, c="C0", alpha=0.8,
                       label=f"survived ({n_keep})")
        # truth
        ax.scatter(rotvec_truth_at[j, 0], rotvec_truth_at[j, 1],
                   s=120, c="red", marker="*", edgecolor="black", linewidth=0.8,
                   label="truth-q", zorder=5)

        ax.set_title(f"ep={ep} t={obs_times[ep]:.0f}s mag={measured_at[j]:.2f}\n"
                     f"|C_t|={n_keep}  nearest={nearest_truth_deg[j]:.2f}°",
                     fontsize=9)
        ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"Per-epoch C_t spread (seed {SEED}, ω={omega_mag_dps:.2f} dps, "
                 f"v1 surrogate, tol={TOLERANCE_MAG} mag, N={N_SAMPLES}, "
                 f"selection={EPOCH_SELECTION})  "
                 f"— rotvec(x,y) projection",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = OUT_DIR / "spread_15panel.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {plot_path}")

    print(f"\n[done] total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
