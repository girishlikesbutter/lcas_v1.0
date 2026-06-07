"""Read s018c omega-grab summary.json and recommend a coarse ω-grid for
the production pilot. Also produces a small plot.

Outputs to results/s018c_omega_grab_diag/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_DIR = Path("/home/girish/projects/lcas_v1.0/notebooks/inversion/survey")
DIAG_DIR = SURVEY_DIR / "results" / "s018c_omega_grab_diag"


def main():
    data = json.loads((DIAG_DIR / "summary.json").read_text())
    truth_dps = data['truth_omega_dps']
    print(f"Seed {data['seed']}, truth ω-mag = {truth_dps:.3f} dps")

    # Split cells into dir-perturbation series and mag-perturbation series.
    cells = data['cells']
    dir_series = [(c['d_deg'], c['best_q0_err_deg'], c['min_q0_err_deg'],
                   c['best_final_mse'])
                  for c in cells if c['d_mag_pct'] == 0.0]
    mag_series = [(c['d_mag_pct'], c['best_q0_err_deg'], c['min_q0_err_deg'],
                   c['best_final_mse'])
                  for c in cells if c['d_deg'] == 0.0]

    print("\nω-direction perturbation (mag at truth):")
    print(f"  d_dir (°) | best_q0_err (°) | min_q0_err (°) | final_mse")
    print(f"  ----------+-----------------+-----------------+----------")
    for d_deg, b, m, mse in sorted(set(dir_series)):
        in_basin = "*" if b < 5 else " "
        print(f"   {d_deg:7.1f}  | {b:14.3f}  | {m:14.3f}  | {mse:.3e}{in_basin}")

    print("\nω-mag perturbation (dir at truth):")
    print(f"  d_mag (%) | best_q0_err (°) | min_q0_err (°) | final_mse")
    print(f"  ----------+-----------------+-----------------+----------")
    for d_pct, b, m, mse in sorted(set(mag_series)):
        in_basin = "*" if b < 5 else " "
        print(f"   {d_pct:+7.1f} | {b:14.3f}  | {m:14.3f}  | {mse:.3e}{in_basin}")

    # Determine LM grab radius
    dir_grab_deg = max([d for d, b, _, _ in dir_series if b < 5], default=0)
    mag_grab_pct = max([abs(d) for d, b, _, _ in mag_series if b < 5], default=0)
    print(f"\nLM grab estimates:")
    print(f"  ω-dir: |Δθ| ≤ {dir_grab_deg:.0f}°  (Band-A boundary)")
    print(f"  ω-mag: |Δp| ≤ {mag_grab_pct:.0f}% (Band-A boundary)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if dir_series:
        ds = sorted(set(dir_series))
        x = [d[0] for d in ds]
        y = [d[1] for d in ds]
        axes[0].plot(x, y, "o-", color="tab:blue")
        axes[0].axhline(5.0, color="tab:red", linestyle="--",
                        label="basin (q0_err<5°)")
        axes[0].set_xlabel("ω-direction perturbation (°)")
        axes[0].set_ylabel("best q0_err after LM (°)")
        axes[0].set_title("ω-direction LM grab")
        axes[0].set_yscale("log")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    if mag_series:
        ms = sorted(set(mag_series))
        x = [d[0] for d in ms]
        y = [d[1] for d in ms]
        axes[1].plot(x, y, "o-", color="tab:orange")
        axes[1].axhline(5.0, color="tab:red", linestyle="--",
                        label="basin (q0_err<5°)")
        axes[1].set_xlabel("ω-magnitude perturbation (% of truth)")
        axes[1].set_ylabel("best q0_err after LM (°)")
        axes[1].set_title("ω-magnitude LM grab")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    fig.suptitle(f"s018c LM grab in ω (seed {data['seed']}, truth ω-mag = {truth_dps:.3f} dps)")
    fig.tight_layout()
    out_png = DIAG_DIR / "lm_grab_in_omega.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    # Recommended ω-grid for pilot
    print("\n=== ω-grid recommendation for production pilot ===")
    if dir_grab_deg < 5 or mag_grab_pct < 5:
        print("  LM grab is too tight for any feasible ω-grid.")
        print("  Recommended: pivot to S016-A coarse adaptive ω-grid")
        print("  (s018c is bottlenecked by ω-search density, not q0 IC primitive).")
    else:
        # Number of dir cells: solid angle 2π(1-cos(grab°)) per cell;
        # 4π / per-cell solid angle = N_dir for full S² coverage.
        if dir_grab_deg >= 60:
            n_dir_rec = 8
        elif dir_grab_deg >= 30:
            n_dir_rec = 16
        elif dir_grab_deg >= 20:
            n_dir_rec = 24
        elif dir_grab_deg >= 10:
            n_dir_rec = 48
        else:
            n_dir_rec = 96
        # Number of mag cells over [0.1, 1.5] dps for grab radius:
        n_mag_rec = max(3, int(np.ceil(1.4 / (truth_dps * mag_grab_pct / 100))))
        n_mag_rec = min(n_mag_rec, 30)
        print(f"  ω-dir grid (Fibonacci on S²): N = {n_dir_rec}")
        print(f"  ω-mag grid (linspace 0.1-1.5 dps): N = {n_mag_rec}")
        print(f"  Total cells: {n_dir_rec * n_mag_rec}")
        print(f"  Pilot wall estimate (5 seeds): "
              f"~{5 * n_dir_rec * n_mag_rec * 144 * 0.05 / 60 / 8:.1f} min score-phase + "
              f"~{5 * 128 * 18 / 60 / 8:.1f} min LM-phase ≈ "
              f"~{(5 * n_dir_rec * n_mag_rec * 144 * 0.05 / 60 / 8) + (5 * 128 * 18 / 60 / 8):.0f} min total")


if __name__ == "__main__":
    main()
