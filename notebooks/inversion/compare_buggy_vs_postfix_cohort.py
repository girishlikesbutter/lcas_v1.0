#!/usr/bin/env python3
"""Compare per-seed buggy vs post-fix m048 trajectories on the corrected
propagator. Produces a population summary (CSV + 4-panel PNG) plus a
seed-level NPZ.

Each seed's `traj_seedNNN_buggy.npz` is the pre-m139-fix output (conv-(b),
RIGHT-mult kinematic) and `traj_seedNNN.npz` is the post-fix (conv-(a),
LEFT-mult). The (q0, ω, start_et, satellite, observation geometry) inputs
are deterministic per seed so the only difference is the kinematic
integration. Same physical scenario, different propagator output.

Reports per seed:
  * `lc_rms_delta_mag`, `lc_max_delta_mag`, `lc_rho` (ρ = √(MSE/0.05²))
  * `quat_geodesic_max_deg`, `quat_geodesic_mean_deg`
  * `peak_count_buggy`, `peak_count_post`, `peak_count_delta`
  * `mag_hifi_min_buggy/post`, `mag_hifi_max_buggy/post`
  * `bright_n_buggy/post` (epochs with mag < 9)
  * `bright_peak_displacement_epochs` (median absolute displacement of
    bright-peak epochs between buggy and post-fix)

Population:
  * Histograms of ρ, max-LC-delta, peak-count delta
  * Scatter of phase-angle-median vs ρ (does bug effect correlate with PA?)
  * Identify outliers (worst/best ρ seeds)

Usage:
    python3 notebooks/inversion/compare_buggy_vs_postfix_cohort.py
    python3 notebooks/inversion/compare_buggy_vs_postfix_cohort.py --seeds 6 91
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJ_DIR = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
            / "m048_trajectories" / "per_trajectory")
OUT_DIR = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
           / "m048_buggy_vs_postfix_compare_2026_04_30")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_SIGMA = 0.05  # mag — matches canonical_observed_lc convention


def quat_geodesic_deg(q1, q2):
    """Angular distance between two unit quaternions (wxyz), in degrees.
    Returns array shape (T,) for (T,4)+(T,4) inputs."""
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def compare_seed(seed):
    p_buggy = TRAJ_DIR / f"traj_seed{seed:03d}_buggy.npz"
    p_post = TRAJ_DIR / f"traj_seed{seed:03d}.npz"
    if not p_buggy.exists() or not p_post.exists():
        return None
    db = np.load(p_buggy, allow_pickle=True)
    dp = np.load(p_post, allow_pickle=True)

    mag_b = db["mag_hifi"]
    mag_p = dp["mag_hifi"]
    valid = np.isfinite(mag_b) & np.isfinite(mag_p)

    delta = mag_p - mag_b
    delta_v = delta[valid]
    rms = float(np.sqrt(np.mean(delta_v ** 2))) if delta_v.size else float("nan")
    mx = float(np.max(np.abs(delta_v))) if delta_v.size else float("nan")
    rho = rms / NOISE_SIGMA  # ρ on the buggy-vs-post-fix LC delta

    qg = quat_geodesic_deg(db["quaternions"], dp["quaternions"])
    qg_max = float(np.nanmax(qg))
    qg_mean = float(np.nanmean(qg))
    qg_p50 = float(np.nanpercentile(qg, 50))
    qg_p99 = float(np.nanpercentile(qg, 99))

    # Bright/dim regime split (mag < 9 = "bright")
    bright_b = valid & (mag_b < 9.0)
    bright_p = valid & (mag_p < 9.0)
    dim_b = valid & (mag_b >= 9.0)

    bright_rms = float(np.sqrt(np.mean(delta[bright_b] ** 2))) if bright_b.any() else 0.0
    dim_rms = float(np.sqrt(np.mean(delta[dim_b] ** 2))) if dim_b.any() else 0.0

    # Peak count + bright-peak displacement
    pk_b, _ = find_peaks(-mag_b, distance=5, prominence=0.3)
    pk_p, _ = find_peaks(-mag_p, distance=5, prominence=0.3)

    # Bright-peak epoch shift: pair each buggy bright peak with nearest post-fix
    # bright peak (within 50 epochs); record the median absolute shift.
    bp_b = pk_b[mag_b[pk_b] < 9.0] if pk_b.size else np.array([])
    bp_p = pk_p[mag_p[pk_p] < 9.0] if pk_p.size else np.array([])
    if bp_b.size and bp_p.size:
        # Nearest-post-peak displacement per buggy bright peak
        disp = np.array([np.min(np.abs(bp_p - eb)) for eb in bp_b])
        bright_peak_disp_median = float(np.median(disp))
        bright_peak_disp_max = float(np.max(disp))
    else:
        bright_peak_disp_median = float("nan")
        bright_peak_disp_max = float("nan")

    # Phase-angle range — independent of attitude, but useful to cross-tab vs ρ
    pa3d = dp["phase_angle_3d"]
    pa_median = float(np.nanmedian(pa3d))
    pa_max = float(np.nanmax(pa3d))

    # ω and q0 magnitudes — same in both, but echo for sanity
    omega_dps = float(dp["omega_mag_dps"])

    return {
        "seed": int(seed),
        "lc_rms_delta_mag": rms,
        "lc_max_delta_mag": mx,
        "lc_rho": float(rho),
        "lc_bright_rms_mag": bright_rms,
        "lc_dim_rms_mag": dim_rms,
        "n_valid_epochs": int(valid.sum()),
        "n_bright_post": int(bright_p.sum()),
        "n_dim_post": int(dim_b.sum()),
        "quat_geodesic_max_deg": qg_max,
        "quat_geodesic_mean_deg": qg_mean,
        "quat_geodesic_p50_deg": qg_p50,
        "quat_geodesic_p99_deg": qg_p99,
        "peak_count_buggy": int(pk_b.size),
        "peak_count_post": int(pk_p.size),
        "peak_count_delta": int(pk_p.size - pk_b.size),
        "bright_peak_count_buggy": int(bp_b.size),
        "bright_peak_count_post": int(bp_p.size),
        "bright_peak_disp_median_epochs": bright_peak_disp_median,
        "bright_peak_disp_max_epochs": bright_peak_disp_max,
        "mag_hifi_min_buggy": float(np.nanmin(mag_b)),
        "mag_hifi_max_buggy": float(np.nanmax(mag_b)),
        "mag_hifi_min_post": float(np.nanmin(mag_p)),
        "mag_hifi_max_post": float(np.nanmax(mag_p)),
        "phase_angle_median_deg": pa_median,
        "phase_angle_max_deg": pa_max,
        "omega_mag_dps": omega_dps,
    }


def write_csv(rows, csv_path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")


def plot_summary(rows, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = np.array([r["lc_rho"] for r in rows])
    max_d = np.array([r["lc_max_delta_mag"] for r in rows])
    pk_d = np.array([r["peak_count_delta"] for r in rows])
    qg_p99 = np.array([r["quat_geodesic_p99_deg"] for r in rows])
    pa_med = np.array([r["phase_angle_median_deg"] for r in rows])
    omega = np.array([r["omega_mag_dps"] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].hist(rho, bins=30, color="steelblue", edgecolor="black")
    axes[0, 0].axvline(2, color="green", ls="--", label="Band-A bound")
    axes[0, 0].axvline(4, color="orange", ls="--", label="Band-B bound")
    axes[0, 0].axvline(8, color="red", ls="--", label="Band-C bound")
    axes[0, 0].set_xlabel("ρ (post-fix vs buggy LC delta / 0.05)")
    axes[0, 0].set_ylabel("# seeds")
    axes[0, 0].set_title(f"ρ distribution (n={len(rows)}; med={np.median(rho):.1f})")
    axes[0, 0].legend()

    axes[0, 1].hist(max_d, bins=30, color="firebrick", edgecolor="black")
    axes[0, 1].set_xlabel("max |Δmag| over LC")
    axes[0, 1].set_ylabel("# seeds")
    axes[0, 1].set_title(f"max LC delta (med={np.median(max_d):.1f} mag)")

    axes[0, 2].hist(pk_d, bins=range(int(pk_d.min()) - 1, int(pk_d.max()) + 2),
                    color="seagreen", edgecolor="black")
    axes[0, 2].set_xlabel("peak count delta (post − buggy)")
    axes[0, 2].set_ylabel("# seeds")
    axes[0, 2].set_title(f"peak count delta (med={np.median(pk_d):+.0f})")

    axes[1, 0].scatter(pa_med, rho, c=omega, cmap="viridis", s=30)
    axes[1, 0].set_xlabel("median phase angle (deg)")
    axes[1, 0].set_ylabel("ρ")
    axes[1, 0].set_title("ρ vs phase angle (color = ω mag)")
    cb = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cb.set_label("ω mag (dps)")

    axes[1, 1].scatter(qg_p99, rho, c=omega, cmap="viridis", s=30)
    axes[1, 1].set_xlabel("quat geodesic p99 (deg)")
    axes[1, 1].set_ylabel("ρ")
    axes[1, 1].set_title("ρ vs attitude divergence (p99)")

    axes[1, 2].hist(qg_p99, bins=30, color="purple", edgecolor="black")
    axes[1, 2].set_xlabel("quat geodesic p99 (deg)")
    axes[1, 2].set_ylabel("# seeds")
    axes[1, 2].set_title(f"attitude divergence (med p99={np.median(qg_p99):.1f}°)")

    plt.tight_layout()
    plt.savefig(png_path, dpi=110)
    plt.close()


def print_population(rows):
    rho = np.array([r["lc_rho"] for r in rows])
    max_d = np.array([r["lc_max_delta_mag"] for r in rows])
    pk_d = np.array([r["peak_count_delta"] for r in rows])
    qg_p99 = np.array([r["quat_geodesic_p99_deg"] for r in rows])
    bp_disp = np.array([r["bright_peak_disp_median_epochs"]
                        for r in rows if not np.isnan(r["bright_peak_disp_median_epochs"])])

    print(f"\n=== Population (n={len(rows)}) ===")
    print(f"  ρ         min={rho.min():.2f}  med={np.median(rho):.2f}  "
          f"p90={np.percentile(rho, 90):.2f}  max={rho.max():.2f}")
    print(f"  Band breakdown:")
    print(f"    A (ρ<2):   {int((rho < 2).sum())}")
    print(f"    B (2-4):   {int(((rho >= 2) & (rho < 4)).sum())}")
    print(f"    C (4-8):   {int(((rho >= 4) & (rho < 8)).sum())}")
    print(f"    D (≥8):    {int((rho >= 8).sum())}")
    print(f"  max |Δmag|   med={np.median(max_d):.2f}  max={max_d.max():.2f}")
    print(f"  peak Δ       med={np.median(pk_d):+.1f}  range={pk_d.min():+d}..{pk_d.max():+d}")
    print(f"  quat geo p99 med={np.median(qg_p99):.1f}°  max={qg_p99.max():.1f}°")
    if bp_disp.size:
        print(f"  bright-peak displacement med={np.median(bp_disp):.1f} epochs  "
              f"max={bp_disp.max():.1f} epochs")

    print(f"\n=== Worst 5 by ρ (largest bug effect) ===")
    order = np.argsort(rho)[::-1][:5]
    for i in order:
        r = rows[i]
        print(f"  seed {r['seed']:3d}: ρ={r['lc_rho']:.2f}  "
              f"max_Δ={r['lc_max_delta_mag']:.2f}  "
              f"pk_Δ={r['peak_count_delta']:+d}  "
              f"qg_p99={r['quat_geodesic_p99_deg']:.1f}°  "
              f"PA_med={r['phase_angle_median_deg']:.1f}°  "
              f"ω={r['omega_mag_dps']:.2f}dps")

    print(f"\n=== Best 5 by ρ (smallest bug effect) ===")
    order = np.argsort(rho)[:5]
    for i in order:
        r = rows[i]
        print(f"  seed {r['seed']:3d}: ρ={r['lc_rho']:.2f}  "
              f"max_Δ={r['lc_max_delta_mag']:.2f}  "
              f"pk_Δ={r['peak_count_delta']:+d}  "
              f"qg_p99={r['quat_geodesic_p99_deg']:.1f}°  "
              f"PA_med={r['phase_angle_median_deg']:.1f}°  "
              f"ω={r['omega_mag_dps']:.2f}dps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Specific seeds to compare (default: all available)")
    args = ap.parse_args()

    if args.seeds is None:
        seeds = sorted({int(p.stem.replace("traj_seed", "").rstrip("_buggy"))
                        for p in TRAJ_DIR.glob("traj_seed*.npz")
                        if "_buggy" not in p.stem})
        # filter to those that have BOTH a non-buggy AND a _buggy sibling
        seeds = [s for s in seeds
                 if (TRAJ_DIR / f"traj_seed{s:03d}_buggy.npz").exists()
                 and (TRAJ_DIR / f"traj_seed{s:03d}.npz").exists()]
    else:
        seeds = args.seeds

    print(f"Comparing {len(seeds)} seeds (buggy vs post-fix)...", flush=True)
    rows = []
    for s in seeds:
        r = compare_seed(s)
        if r is None:
            print(f"  seed {s:3d}: missing one side, skipping")
            continue
        rows.append(r)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(seeds)} done...", flush=True)

    csv_path = OUT_DIR / "summary.csv"
    npz_path = OUT_DIR / "summary.npz"
    png_path = OUT_DIR / "summary.png"
    json_path = OUT_DIR / "summary.json"

    write_csv(rows, csv_path)
    np.savez(npz_path,
             seeds=np.array([r["seed"] for r in rows], dtype=int),
             rows=np.array(rows, dtype=object))
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    plot_summary(rows, png_path)
    print_population(rows)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
