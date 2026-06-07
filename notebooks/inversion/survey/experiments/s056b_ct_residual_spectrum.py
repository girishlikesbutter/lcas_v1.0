"""s056b — does |C_t|(t) carry info BEYOND mag_hifi(t)?

Builds on s056. The dominant |C_t| spectrum peak (P=20.5 min) coincides
with the LC magnitude PSD shape (Pearson ρ=+0.78 between log|C_t| and
mag_hifi). Question: after regressing the LC-magnitude-explained
component out of log|C_t|(t), does the RESIDUAL carry any geometric /
dynamic structure (rotation, polhode, or pool-discretisation effects)?

If residual spectrum is white → direction (2) breathing-spectrum is
mostly redundant with LC, closed on seed 89.
If residual spectrum has peaks at f_rot or polhode frequencies → there
IS a different signal channel; pursue.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, lombscargle

SURVEY = Path(__file__).resolve().parents[1]
S056_DATA = SURVEY / "results" / "s056_ct_breathing_spectrum" / "breathing_data.npz"
OUT = SURVEY / "results" / "s056b_ct_residual_spectrum"
OUT.mkdir(parents=True, exist_ok=True)


def fit_residual(y: np.ndarray, x_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS of y on [1, x_cols]. Returns (beta, residual, R²)."""
    X = np.column_stack([np.ones(len(y))] + list(x_cols.T))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return beta, resid, r2


def main() -> dict:
    z = np.load(S056_DATA)
    t = z["t"]
    n_surv = z["n_surv"].astype(float)
    mag_hifi = z["mag_hifi"]
    slope_proxy = z["slope_proxy"]
    om_mag_t = z["om_mag_t"]
    closest_deg = z["closest_deg"]
    omega_mag_dps = float(z["omega_mag_dps"])
    pol_diam_dps = float(z["pol_diam_dps"])

    log_n = np.log10(n_surv)
    duration_s = float(t[-1])
    dt = float(np.median(np.diff(t)))
    rot_period_s = 360.0 / omega_mag_dps
    f_rot_per_min = 60.0 / rot_period_s

    # --- regress log|C_t| on mag_hifi only ---
    _, resid_mag, r2_mag = fit_residual(log_n, mag_hifi[:, None])
    # --- regress log|C_t| on (mag_hifi, log|dmag/dt|⁻¹) ---
    _, resid_both, r2_both = fit_residual(
        log_n, np.column_stack([mag_hifi, np.log10(slope_proxy)])
    )
    # --- regress log|C_t| on (mag_hifi, log|dmag/dt|⁻¹, mag²) ---
    _, resid_quad, r2_quad = fit_residual(
        log_n,
        np.column_stack([mag_hifi, np.log10(slope_proxy), mag_hifi ** 2]),
    )

    # --- spectra of three series + residuals ---
    f_min = 1.0 / (3 * duration_s)
    f_max = 0.4 / dt
    freqs = np.linspace(f_min, f_max, 4000)
    omg_rad = 2.0 * np.pi * freqs

    def _psd(y):
        yn = (y - y.mean()) / (y.std() + 1e-12)
        return lombscargle(t, yn, omg_rad, normalize=True)

    pgram_n = _psd(log_n)
    pgram_resid_mag = _psd(resid_mag)
    pgram_resid_both = _psd(resid_both)
    pgram_resid_quad = _psd(resid_quad)
    pgram_om = _psd(om_mag_t)
    pgram_closest = _psd(closest_deg)

    # top peaks in residual-mag spectrum
    pk_idx, _ = find_peaks(pgram_resid_mag, height=0.02)
    if len(pk_idx) == 0:
        pk_idx = np.argsort(pgram_resid_mag)[::-1][:5]
    top_resid_mag = []
    for i in pk_idx[np.argsort(pgram_resid_mag[pk_idx])[::-1][:8]]:
        f_pm = float(freqs[i] * 60)
        top_resid_mag.append({
            "f_per_min": f_pm,
            "period_min": float(1.0 / f_pm) if f_pm > 0 else float("inf"),
            "psd": float(pgram_resid_mag[i]),
        })

    # is residual spectrum dominated by short-period / pool-discretisation noise?
    # split power below vs above f=0.2/min (P<5 min, well above rotation)
    cutoff = 0.2 / 60  # Hz
    lo = freqs < cutoff
    hi = freqs >= cutoff
    p_low_n = float(pgram_n[lo].sum())
    p_high_n = float(pgram_n[hi].sum())
    p_low_resid = float(pgram_resid_mag[lo].sum())
    p_high_resid = float(pgram_resid_mag[hi].sum())

    # --- 3-panel figure ---
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    ax = axes[0]
    ax.plot(t / 60, log_n, "b-", lw=0.7, label="log|C_t|(t)")
    ax.plot(t / 60, log_n - resid_mag, "r--", lw=0.7,
            label=f"OLS fit on mag_hifi  (R²={r2_mag:.3f})")
    ax.set_xlabel("t (min)")
    ax.set_ylabel("log|C_t|")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        f"OLS regression: R² (mag_hifi)={r2_mag:.3f}, "
        f"R² (mag + log|dmag/dt|⁻¹)={r2_both:.3f}, "
        f"R² (+mag²)={r2_quad:.3f}"
    )

    ax = axes[1]
    ax.plot(t / 60, resid_mag, "b-", lw=0.7, label="resid (mag_hifi)")
    ax.plot(t / 60, resid_both, "g-", lw=0.5, alpha=0.7, label="resid (mag + slope)")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("t (min)")
    ax.set_ylabel("residual log|C_t|")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        f"residual std (mag_hifi)={resid_mag.std():.3f} dex, "
        f"(mag+slope)={resid_both.std():.3f} dex; "
        f"≡ ×{10**resid_mag.std():.2f} factor on |C_t|"
    )

    ax = axes[2]
    f_per_min = freqs * 60
    ax.semilogy(f_per_min, pgram_n, "b-", lw=0.9, label="log|C_t|(t) PSD")
    ax.semilogy(f_per_min, pgram_resid_mag, "r-", lw=0.7,
                label=f"residual (mag) PSD — total power below {cutoff*60:.2f}/min: "
                      f"{p_low_resid/(p_low_resid+p_high_resid)*100:.0f}%")
    ax.semilogy(f_per_min, pgram_resid_both, "g-", lw=0.6, alpha=0.7,
                label="residual (mag+slope) PSD")
    ax.semilogy(f_per_min, pgram_om, "m-", lw=0.6, alpha=0.7,
                label="|ω|_body(t) PSD")
    ax.semilogy(f_per_min, pgram_closest, "c-", lw=0.5, alpha=0.5,
                label="closest_deg(t) PSD")
    ax.axvline(f_rot_per_min, color="r", ls="--", lw=1, alpha=0.6,
               label=f"f_rot ({rot_period_s/60:.1f} min)")
    ax.axvline(2 * f_rot_per_min, color="r", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("frequency (1/min)")
    ax.set_ylabel("PSD (LS, normalized)")
    ax.legend(loc="upper right", fontsize=7)
    if len(top_resid_mag):
        ax.set_title(
            f"top residual peak: P={top_resid_mag[0]['period_min']:.2f} min "
            f"(f={top_resid_mag[0]['f_per_min']:.4f}/min, "
            f"PSD={top_resid_mag[0]['psd']:.3f})"
        )

    plt.tight_layout()
    fig_p = OUT / "ct_residual_spectrum.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_p}")

    summary = {
        "seed": 89,
        "r2_mag_only": r2_mag,
        "r2_mag_plus_slope": r2_both,
        "r2_mag_plus_slope_plus_quadratic": r2_quad,
        "resid_std_mag_only_dex": float(resid_mag.std()),
        "resid_std_mag_plus_slope_dex": float(resid_both.std()),
        "resid_factor_mag_only": float(10 ** resid_mag.std()),
        "resid_power_low_freq_frac": p_low_resid / (p_low_resid + p_high_resid),
        "log_ct_power_low_freq_frac": p_low_n / (p_low_n + p_high_n),
        "f_rot_per_min": f_rot_per_min,
        "top_residual_peaks": top_resid_mag,
    }
    out_p = OUT / "summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_p}")

    np.savez(
        OUT / "residual_data.npz",
        t=t, log_n=log_n, resid_mag=resid_mag, resid_both=resid_both,
        freqs=freqs, pgram_n=pgram_n, pgram_resid_mag=pgram_resid_mag,
        pgram_resid_both=pgram_resid_both, pgram_om=pgram_om,
        pgram_closest=pgram_closest,
    )
    print(f"Saved: {OUT / 'residual_data.npz'}")
    return summary


if __name__ == "__main__":
    s = main()
    print()
    print("=== summary ===")
    print(json.dumps(s, indent=2))
