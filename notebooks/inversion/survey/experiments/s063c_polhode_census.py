"""s063c — Cohort polhode census.

For each of 120 cohort seeds, extract polhode invariants via the closed-form
Jacobi machinery: regime, k², |L|, 2T, τ_dot, polhode period T_pol = 4K(m)/|τ_dot|,
and analytical pol_diam = max - min of |ω_PA| along the polhode.

Use this to:
  - Map the cohort distribution of polhode parameters.
  - Identify near-separatrix seeds (k² > 0.99) where Path 2 (s062b)
    closed-form q(t) may need numerical conditioning fallback.
  - Identify short-period seeds (T_pol < 60 s) where s061 constant-ω
    propagation is only valid over a small fraction of LC duration.

Cheap (~5 s), no surrogate, no propagation."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ellipj, ellipk

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))

from lib.jacobi_propagator import (  # noqa: E402
    _build_omega_func,
    _eigendecompose_inertia,
)

INERTIA = np.diag([37985.15566495171, 38305.70560133419, 7749.014672251076])
TRAJ_DIR = SURVEY / "data" / "trajectories"

OUT_DIR = SURVEY / "results" / "s063c"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def polhode_stats(omega_pa, I_pa):
    info = _build_omega_func(I_pa, omega_pa)[1]
    m = info["m"]
    tau_dot = info["tau_dot"]
    K_m = ellipk(m)
    T_pol = 4.0 * K_m / abs(tau_dot)

    # Analytical pol_diam: evaluate |ω_PA| along full polhode.
    taus = np.linspace(0, 4.0 * K_m, 500, endpoint=False)
    sn, cn, dn, _ = ellipj(taus, m)
    a1, a2, a3 = info["a"]
    if info["regime"] == "A":
        sgn_1 = info["sgn_1"]
        omega_curve = np.column_stack([sgn_1 * a1 * dn, a2 * sn, a3 * cn])
    else:
        sgn_3 = info["sgn_3"]
        omega_curve = np.column_stack([a1 * cn, a2 * sn, sgn_3 * a3 * dn])

    omega_mags = np.linalg.norm(omega_curve, axis=1)
    pol_diam_omega = float((omega_mags.max() - omega_mags.min()) * 180.0 / np.pi)

    # |L| (constant)
    L_mag = float(np.sqrt(info["L2_0"]))
    twoT = float(info["twoT_0"])

    return dict(
        regime=info["regime"],
        m_k_sq=float(m),
        K_of_m=float(K_m),
        tau_dot_abs=float(abs(tau_dot)),
        T_pol_seconds=float(T_pol),
        L_magnitude=L_mag,
        twoT=twoT,
        a_amplitudes=[float(a) for a in info["a"]],
        omega_mag_max_dps=float(omega_mags.max() * 180.0 / np.pi),
        omega_mag_min_dps=float(omega_mags.min() * 180.0 / np.pi),
        pol_diam_omega_dps=pol_diam_omega,
    )


def main():
    t0 = time.time()
    I_pa, R_pa = _eigendecompose_inertia(INERTIA)

    rows = []
    for seed in range(120):
        npz_path = TRAJ_DIR / f"traj_seed{seed:03d}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        omega_body = d["omega0_rad"].astype(np.float64)
        omega_pa = R_pa.T @ omega_body
        stats = polhode_stats(omega_pa, I_pa)
        stats["seed"] = int(seed)
        stats["omega0_mag_dps"] = float(np.linalg.norm(omega_body) * 180.0 / np.pi)
        rows.append(stats)

    wall = time.time() - t0

    # Summary
    arr = lambda key: np.array([r[key] for r in rows], dtype=float)
    m_vals = arr("m_k_sq")
    T_pol = arr("T_pol_seconds")
    omega_mag = arr("omega0_mag_dps")
    pol_diam = arr("pol_diam_omega_dps")
    L_mag = arr("L_magnitude")
    regimes = [r["regime"] for r in rows]
    regime_A = sum(1 for r in regimes if r == "A")
    regime_B = sum(1 for r in regimes if r == "B")

    def stats(a):
        return {
            "n": int(len(a)),
            "min": float(np.min(a)),
            "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "max": float(np.max(a)),
        }

    near_sep_99 = [r["seed"] for r in rows if r["m_k_sq"] > 0.99]
    near_sep_999 = [r["seed"] for r in rows if r["m_k_sq"] > 0.999]
    short_period = [r["seed"] for r in rows if r["T_pol_seconds"] < 60.0]

    summary = dict(
        n_seeds=len(rows),
        wall_seconds=wall,
        regime_A_count=regime_A,
        regime_B_count=regime_B,
        m_k_sq=stats(m_vals),
        T_pol_seconds=stats(T_pol),
        omega_mag_dps=stats(omega_mag),
        pol_diam_omega_dps=stats(pol_diam),
        L_magnitude=stats(L_mag),
        n_near_separatrix_99=len(near_sep_99),
        n_near_separatrix_999=len(near_sep_999),
        seeds_near_separatrix_99=near_sep_99,
        seeds_near_separatrix_999=near_sep_999,
        seeds_short_period_under_60s=short_period,
    )

    out_summary = OUT_DIR / "summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    out_per_seed = OUT_DIR / "per_seed.json"
    with open(out_per_seed, "w") as f:
        json.dump(rows, f, indent=2)

    # Cohort plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.hist(m_vals, bins=40, color="steelblue", edgecolor="k", alpha=0.85)
    ax.axvline(0.99, color="crimson", ls="--", label="separatrix 0.99")
    ax.set_xlabel("k² (elliptic modulus²)")
    ax.set_ylabel("# seeds")
    ax.set_title(f"Cohort k² distribution  (≥0.99: {len(near_sep_99)}, ≥0.999: {len(near_sep_999)})")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(T_pol, bins=40, color="darkgreen", edgecolor="k", alpha=0.85)
    ax.axvline(60, color="crimson", ls="--", label="60 s")
    ax.axvline(np.median(T_pol), color="orange", ls="--", label=f"median {np.median(T_pol):.0f}s")
    ax.set_xlabel("Polhode period T_pol [s]")
    ax.set_ylabel("# seeds")
    ax.set_title(f"Cohort polhode-period distribution (median {np.median(T_pol):.0f}s)")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[0, 2]
    ax.scatter(omega_mag, m_vals, c=[("steelblue" if r == "A" else "tab:orange") for r in regimes],
               s=18, alpha=0.8)
    ax.axhline(0.99, color="crimson", ls=":", lw=1)
    ax.set_xlabel("|ω₀| [dps]")
    ax.set_ylabel("k²")
    ax.set_title("|ω₀| vs k² (blue=Case A small-axis, orange=Case B large-axis)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(omega_mag, T_pol, c=[("steelblue" if r == "A" else "tab:orange") for r in regimes],
               s=18, alpha=0.8)
    ax.set_xlabel("|ω₀| [dps]")
    ax.set_ylabel("T_pol [s]")
    ax.set_title("|ω₀| vs polhode period")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(omega_mag, pol_diam, c=[("steelblue" if r == "A" else "tab:orange") for r in regimes],
               s=18, alpha=0.8)
    ax.plot([0, omega_mag.max()], [0, omega_mag.max()], "k:", lw=0.5, label="pol_diam = |ω|")
    ax.set_xlabel("|ω₀| [dps]")
    ax.set_ylabel("pol_diam_ω [dps]")
    ax.set_title("|ω₀| vs polhode diameter (PA frame)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(m_vals, pol_diam, c=[("steelblue" if r == "A" else "tab:orange") for r in regimes],
               s=18, alpha=0.8)
    ax.set_xlabel("k²")
    ax.set_ylabel("pol_diam_ω [dps]")
    ax.set_title("k² vs polhode diameter")
    ax.grid(alpha=0.3)

    fig.suptitle("s063c — Cohort polhode census (120 m048 seeds via closed-form Jacobi)",
                 fontsize=13)
    fig.tight_layout()
    plot_path = OUT_DIR / "cohort_polhode_census.png"
    fig.savefig(str(plot_path), dpi=130)
    plt.close(fig)

    print(f"Wall: {wall:.2f}s for {len(rows)} seeds")
    print(f"Regime A (encloses I_1): {regime_A}    Regime B (encloses I_3): {regime_B}")
    print(f"k² median: {np.median(m_vals):.4f}  max: {np.max(m_vals):.4f}")
    print(f"T_pol median: {np.median(T_pol):.1f}s  range: [{np.min(T_pol):.1f}, {np.max(T_pol):.1f}]")
    print(f"|ω| median: {np.median(omega_mag):.3f} dps  range: [{np.min(omega_mag):.3f}, {np.max(omega_mag):.3f}]")
    print(f"pol_diam median: {np.median(pol_diam):.3f} dps")
    print(f"Near-separatrix k²≥0.99: {near_sep_99}")
    print(f"Near-separatrix k²≥0.999: {near_sep_999}")
    print(f"Short-period <60s: {short_period}")
    print(f"\nSaved: {out_summary}")
    print(f"Saved: {out_per_seed}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
