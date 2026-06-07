"""m147 — overlay buggy vs post-fix LC for 10 random m048 seeds.

Reads `traj_seed{XXX}.npz` (post-fix) and `traj_seed{XXX}_buggy.npz` (buggy
sibling) from m048_trajectories/per_trajectory/ for 10 RNG-selected seeds in
[0, 99], plots `mag_hifi` for each seed in a 5x2 grid (buggy red, post-fix
green), saves PNG.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
TRAJ_DIR = PROJECT_ROOT / "data/results/inversion_diagnostics/m048_trajectories/per_trajectory"
OUT_DIR = PROJECT_ROOT / "data/results/inversion_diagnostics/m147_lc_overlay_10seeds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260430
N_SEEDS = 10


def main():
    rng = np.random.default_rng(RNG_SEED)
    seeds = sorted(rng.choice(100, size=N_SEEDS, replace=False).tolist())
    print(f"Selected seeds: {seeds}")

    fig, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=False)
    axes = axes.flatten()

    summary_rows = []
    for ax, seed in zip(axes, seeds):
        post_path = TRAJ_DIR / f"traj_seed{seed:03d}.npz"
        buggy_path = TRAJ_DIR / f"traj_seed{seed:03d}_buggy.npz"
        post = np.load(post_path)
        buggy = np.load(buggy_path)

        t_post = post["observation_times"]
        t_buggy = buggy["observation_times"]
        t0 = t_post[0]
        t_post_rel = t_post - t0
        t_buggy_rel = t_buggy - t0

        mag_post = post["mag_hifi"]
        mag_buggy = buggy["mag_hifi"]

        ax.plot(t_buggy_rel, mag_buggy, color="red", lw=1.0, alpha=0.85, label="buggy")
        ax.plot(t_post_rel, mag_post, color="green", lw=1.0, alpha=0.85, label="post-fix")
        ax.invert_yaxis()
        ax.set_title(
            f"seed {seed:03d}  |  ω={post['omega_mag_dps'].item():.2f} dps  |  PA_med={np.median(post['phase_angle_3d']):.1f}°",
            fontsize=10,
        )
        ax.set_xlabel("t since first epoch (s)")
        ax.set_ylabel("apparent mag")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        diff = mag_post - mag_buggy
        rms = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        rho = float(rms / 0.05 * np.sqrt(1.0))
        rho_proper = float(np.sqrt(np.mean(diff ** 2) / 0.05 ** 2)) if False else float(np.sqrt(np.mean(diff ** 2) / (0.05 ** 2)))
        summary_rows.append((seed, rms, max_abs, rho_proper))

    fig.suptitle(
        "m147 — buggy (red) vs post-fix (green) hi-fi LCs, 10 random m048 seeds",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = OUT_DIR / "lc_overlay_10seeds.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved: {out_png}")

    print("\nseed  rms_mag  max|Δ|  ρ")
    for seed, rms, max_abs, rho in summary_rows:
        print(f"{seed:4d}  {rms:7.3f}  {max_abs:6.2f}  {rho:6.2f}")


if __name__ == "__main__":
    main()
