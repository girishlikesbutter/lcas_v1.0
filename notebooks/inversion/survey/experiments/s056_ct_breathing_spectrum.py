"""s056 — |C_t|(t) breathing-spectrum sanity look on seed 89.

Loads the dense (500-epoch, 100k-pool) cached cloud-viewer run and asks:
- Does |C_t|(t) constrict at hi-fi LC peaks (PAB-aligned bright facets)?
- What spectral content does |C_t|(t) carry?
- How does it compare to the LC magnitude spectrum?

This is a SANITY LOOK — not an architecture proposal. Outputs: figure +
summary numbers; no model fit. Decides whether the breathing-spectrum
direction merits a proper experiment or punts to cross-epoch ω-aggregation
or cloud-centroid drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, lombscargle

SURVEY = Path(__file__).resolve().parents[1]

DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
S053_COHORT = SURVEY / "results" / "s053_cohort_polhode_survey" / "cohort.npz"
TRAJ089 = SURVEY / "data" / "trajectories" / "traj_seed089.npz"

OUT = SURVEY / "results" / "s056_ct_breathing_spectrum"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    z = np.load(DENSE_RUN)
    obs_times = z["obs_times"]
    n_surv = z["n_survivors"].astype(float)
    mag_hifi = z["mag_hifi"]
    peak_epochs = np.asarray(z["hifi_peak_epochs"])
    closest_deg = z["closest_deg_per_epoch"]
    omega_mag_dps = float(z["omega_mag_dps"])

    t = obs_times - obs_times[0]
    dt = float(np.median(np.diff(obs_times)))
    duration_s = float(t[-1])

    # polhode params from s053 cohort (no integration needed)
    cohort = np.load(S053_COHORT)
    seeds = cohort["seeds"]
    i89 = int(np.where(seeds == 89)[0][0])
    pol_diam_dps = float(cohort["pol_diam_dps"][i89])
    cone_max_deg = float(cohort["cone_max_deg"][i89])
    D_seed = float(cohort["D"][i89])

    rot_period_s = 360.0 / omega_mag_dps
    f_rot_per_min = 60.0 / rot_period_s

    # --- constriction vs LC peak coincidence ---
    n_at_peaks = n_surv[peak_epochs]
    off_mask = np.ones(len(n_surv), bool)
    off_mask[peak_epochs] = False
    n_off = n_surv[off_mask]
    ratio_peak_vs_off = float(np.median(n_at_peaks) / np.median(n_off))

    # also: are |C_t| MINIMA close to LC peaks in time?
    # find local minima in n_surv (with prominence)
    log_n = np.log10(n_surv)
    minima_idx, _ = find_peaks(-log_n, prominence=0.3)  # ~factor-2 dip
    # for each LC peak, distance (in epochs) to nearest |C_t| minimum
    if len(minima_idx) > 0:
        dist_peak_to_min = np.array([
            float(np.min(np.abs(minima_idx - p))) for p in peak_epochs
        ])
        median_dist_peak_to_min_s = float(np.median(dist_peak_to_min) * dt)
    else:
        dist_peak_to_min = np.array([])
        median_dist_peak_to_min_s = float("nan")

    # --- LC-slope comparator: is |C_t| just a tolerance-rescaled |dmag/dt|⁻¹ proxy? ---
    dmag_dt = np.gradient(mag_hifi, t)
    slope_proxy = 1.0 / (np.abs(dmag_dt) + 1e-3)  # avoid blow-ups at peak stationary points

    # --- ω_body(t) from cached q(t) by finite-diff ---
    # Quaternion convention check: post-fix propagator emits passive convention.
    # ω_body finite-diff via q_dot = 0.5 * q ⊗ ω  (Hamiltonian, scalar-first).
    # Equivalently:  ω_body = 2 * (q^-1 ⊗ q_dot)_vec.
    traj = np.load(TRAJ089)
    q_t = traj["quaternions"]  # (500, 4) wxyz
    q_dot = np.gradient(q_t, t, axis=0)
    # quaternion product q^-1 ⊗ q_dot, scalar-first
    qw, qx, qy, qz = q_t[:, 0], q_t[:, 1], q_t[:, 2], q_t[:, 3]
    dw, dx, dy, dz = q_dot[:, 0], q_dot[:, 1], q_dot[:, 2], q_dot[:, 3]
    # q^-1 = (w, -x, -y, -z) for unit q
    omx_b = 2.0 * (qw * dx - qx * dw + qy * dz - qz * dy)
    omy_b = 2.0 * (qw * dy - qx * dz - qy * dw + qz * dx)
    omz_b = 2.0 * (qw * dz + qx * dy - qy * dx - qz * dw)
    om_body = np.stack([omx_b, omy_b, omz_b], axis=1)  # rad/s
    om_mag_t = np.linalg.norm(om_body, axis=1) * (180.0 / np.pi)  # dps
    # sanity: mean should match cohort omega_mag_dps within ~0.5%
    om_mag_check = float(np.median(om_mag_t))

    # --- spectra ---
    f_min = 1.0 / (3 * duration_s)
    f_max = 0.4 / dt  # below Nyquist
    freqs = np.linspace(f_min, f_max, 4000)
    omg_rad = 2.0 * np.pi * freqs
    n_norm = (n_surv - n_surv.mean()) / (n_surv.std() + 1e-12)
    mag_norm = (mag_hifi - mag_hifi.mean()) / (mag_hifi.std() + 1e-12)
    slope_norm = (slope_proxy - slope_proxy.mean()) / (slope_proxy.std() + 1e-12)
    om_mag_norm = (om_mag_t - om_mag_t.mean()) / (om_mag_t.std() + 1e-12)
    pgram_n = lombscargle(t, n_norm, omg_rad, normalize=True)
    pgram_m = lombscargle(t, mag_norm, omg_rad, normalize=True)
    pgram_slope = lombscargle(t, slope_norm, omg_rad, normalize=True)
    pgram_om = lombscargle(t, om_mag_norm, omg_rad, normalize=True)

    # --- correlations: is |C_t| more like LC-slope or |ω|(t)? ---
    # Pearson on log|C_t| (since |C_t| spans 3 decades)
    log_n = np.log10(n_surv)
    rho_n_slope = float(np.corrcoef(log_n, slope_proxy)[0, 1])
    rho_n_lcslopelog = float(np.corrcoef(log_n, np.log10(slope_proxy))[0, 1])
    rho_n_ommag = float(np.corrcoef(log_n, om_mag_t)[0, 1])
    rho_n_mag = float(np.corrcoef(log_n, mag_hifi)[0, 1])

    # top-10 spectral peaks in |C_t|
    pgram_peaks_idx, _ = find_peaks(pgram_n, height=0.05)
    if len(pgram_peaks_idx):
        top = pgram_peaks_idx[np.argsort(pgram_n[pgram_peaks_idx])[::-1][:10]]
    else:
        top = np.argsort(pgram_n)[::-1][:10]
    top_specs = []
    for i in top:
        f_pm = float(freqs[i] * 60)
        top_specs.append({
            "f_per_min": f_pm,
            "period_min": float(1.0 / f_pm) if f_pm > 0 else float("inf"),
            "psd": float(pgram_n[i]),
        })

    # --- 4-panel figure ---
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)

    ax = axes[0]
    ax.plot(t / 60, mag_hifi, "k-", lw=0.7)
    ax.plot(t[peak_epochs] / 60, mag_hifi[peak_epochs], "rv", ms=8,
            label=f"{len(peak_epochs)} hi-fi peaks")
    ax.invert_yaxis()
    ax.set_ylabel("hi-fi mag")
    ax.set_xlabel("t (min)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        f"seed 89 — |ω|={omega_mag_dps:.4f} dps, "
        f"rot period {rot_period_s/60:.1f} min, "
        f"pol_diam={pol_diam_dps:.3f} dps, D={D_seed:.4f}, "
        f"cone_max={cone_max_deg:.1f}°"
    )

    ax = axes[1]
    ax.semilogy(t / 60, n_surv, "b-", lw=0.7)
    for p in peak_epochs:
        ax.axvline(t[p] / 60, color="r", alpha=0.25, lw=0.6)
    if len(minima_idx):
        ax.semilogy(t[minima_idx] / 60, n_surv[minima_idx], "g^", ms=6,
                    label=f"{len(minima_idx)} |C_t| minima")
    ax.axhline(np.median(n_surv), ls=":", color="grey", lw=0.6,
               label=f"median |C_t|={np.median(n_surv):.0f}")
    ax.set_ylabel("|C_t|  (log)")
    ax.set_xlabel("t (min)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"breathing — peak |C_t| median={np.median(n_at_peaks):.0f}, "
        f"off-peak {np.median(n_off):.0f}, ratio={ratio_peak_vs_off:.2f} "
        f"(median dist LC-peak → |C_t|-min: {median_dist_peak_to_min_s:.0f}s)"
    )

    ax = axes[2]
    f_per_min = freqs * 60
    ax.semilogy(f_per_min, pgram_n, "b-", lw=0.9, label="|C_t|(t) PSD")
    ax.semilogy(f_per_min, pgram_slope, "g-", lw=0.7, alpha=0.7,
                label="|dmag/dt|⁻¹ PSD  (LC-slope proxy)")
    ax.semilogy(f_per_min, pgram_om, "m-", lw=0.7, alpha=0.7,
                label="|ω|_body(t) PSD")
    ax.semilogy(f_per_min, pgram_m, "k-", lw=0.5, alpha=0.4,
                label="mag_hifi(t) PSD")
    ax.axvline(f_rot_per_min, color="r", ls="--", lw=1,
               label=f"f_rot = {f_rot_per_min:.4f}/min  (P={rot_period_s/60:.1f} min)")
    ax.axvline(2 * f_rot_per_min, color="r", ls=":", lw=0.8, alpha=0.6,
               label="2·f_rot")
    ax.set_xlabel("frequency  (1/min)")
    ax.set_ylabel("PSD  (Lomb-Scargle, normalized)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"top |C_t| peak P={top_specs[0]['period_min']:.2f} min  "
        f"(corr log|C_t| vs LC-slope ρ={rho_n_lcslopelog:+.3f}, "
        f"vs |ω|(t) ρ={rho_n_ommag:+.3f}, vs mag_hifi ρ={rho_n_mag:+.3f})"
    )

    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(t / 60, om_mag_t, "m-", lw=0.6, label="|ω|_body(t) (dps)")
    ax2.plot(t / 60, n_surv, "b-", lw=0.5, alpha=0.6, label="|C_t|(t)")
    ax2.set_yscale("log")
    ax.set_xlabel("t (min)")
    ax.set_ylabel("|ω|_body  (dps)", color="m")
    ax2.set_ylabel("|C_t|  (log)", color="b")
    ax.set_title(
        f"|ω|_body(t) finite-diff: median={om_mag_check:.4f} dps  "
        f"(cohort {omega_mag_dps:.4f}, std/mean={om_mag_t.std()/om_mag_t.mean()*100:.3f}%)  "
        f"|  pool→truth max={closest_deg.max():.2f}°"
    )

    plt.tight_layout()
    fig_p = OUT / "ct_breathing_overview.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_p}")

    summary = {
        "seed": 89,
        "n_epochs": int(len(t)),
        "duration_s": duration_s,
        "dt_s": dt,
        "omega_mag_dps": omega_mag_dps,
        "rotation_period_s": rot_period_s,
        "rotation_period_min": rot_period_s / 60,
        "pol_diam_dps": pol_diam_dps,
        "cone_max_deg": cone_max_deg,
        "D_seed": D_seed,
        "n_lc_peaks": int(len(peak_epochs)),
        "n_ct_minima": int(len(minima_idx)),
        "n_surv_min": int(n_surv.min()),
        "n_surv_max": int(n_surv.max()),
        "n_surv_median": float(np.median(n_surv)),
        "n_surv_at_peaks_median": float(np.median(n_at_peaks)),
        "n_surv_off_peaks_median": float(np.median(n_off)),
        "ratio_peak_vs_off": ratio_peak_vs_off,
        "median_dist_peak_to_ct_min_s": median_dist_peak_to_min_s,
        "spectrum_top_peaks": top_specs,
        "f_rot_per_min": f_rot_per_min,
        "om_mag_finite_diff_median_dps": om_mag_check,
        "om_mag_std_over_mean_pct": float(om_mag_t.std() / om_mag_t.mean() * 100),
        "rho_log_ct_vs_lcslope_log": rho_n_lcslopelog,
        "rho_log_ct_vs_om_mag_t": rho_n_ommag,
        "rho_log_ct_vs_mag_hifi": rho_n_mag,
    }
    out_p = OUT / "summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_p}")

    np.savez(
        OUT / "breathing_data.npz",
        t=t, n_surv=n_surv, mag_hifi=mag_hifi,
        peak_epochs=peak_epochs, ct_minima=minima_idx,
        freqs=freqs, pgram_n=pgram_n, pgram_m=pgram_m,
        pgram_slope=pgram_slope, pgram_om=pgram_om,
        slope_proxy=slope_proxy, om_body=om_body, om_mag_t=om_mag_t,
        omega_mag_dps=omega_mag_dps, pol_diam_dps=pol_diam_dps,
        closest_deg=closest_deg,
    )
    print(f"Saved: {OUT / 'breathing_data.npz'}")

    return summary


if __name__ == "__main__":
    s = main()
    print()
    print("=== summary ===")
    print(json.dumps(s, indent=2))
