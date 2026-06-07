"""s057 — anchor-propagation ω-direction recovery on seed 89.

Architecture (proposed by user):
- Find anchor epoch t_a where |C_t| is smallest (deepest constriction).
- For each q_a ∈ C_{t_a} and each q_b ∈ C_{t_a + Δt}, compute the implied
  ω-vector via finite-diff: q_b = q_a ⊗ exp(0.5 · ω_body · Δt).
- Filter on |ω|-prior (s055a-style pol_diam-derived bracket).
- Aggregate surviving ω-vectors; ω-direction is recoverable if the
  hypothesis cloud concentrates near truth-ω-direction.

Δt choice: |ω|·Δt must exceed the pool angular resolution (~1° at 100k
Sobol). At |ω|=0.24 dps, Δt=10 epochs (72s) gives |ω|·Δt≈17° — well
above resolution. v1 pilot uses Δt=10.

|ω|-prior settings: ORACLE (truth ± 0%), TIGHT (±5%, ≈ s055a holdout
MAPE), LOOSE (±25%, s055a worst-case).

Output: per-prior histogram of angular distance to truth-ω̂ + 3D scatter
+ concentration metrics (fraction within 10°, 30°, 90°). Compares to
uniform-sphere baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
TRAJ089 = SURVEY / "data" / "trajectories" / "traj_seed089.npz"
OUT = SURVEY / "results" / "s057_anchor_propagation"
OUT.mkdir(parents=True, exist_ok=True)

DELTA_EPOCH = 10


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def finite_diff_omega(q_a_wxyz: np.ndarray, q_b_wxyz: np.ndarray,
                     dt_s: float, convention: str) -> np.ndarray:
    """ω-body finite-diff. Returns rad/s, shape (..., 3)."""
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a_wxyz))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b_wxyz))
    if convention == "active":
        dR = R_a.inv() * R_b
    elif convention == "passive":
        dR = R_b * R_a.inv()
    else:
        raise ValueError(convention)
    rotvec = dR.as_rotvec()
    return rotvec / dt_s


def smoke_test_convention(traj: dict, dt_s: float) -> tuple[str, float]:
    """Finite-diff cached truth q(t). Pick convention recovering omega0_rad."""
    q_t = traj["quaternions"]
    om0 = traj["omega0_rad"]
    om0_mag = float(np.linalg.norm(om0))
    om0_mag_dps = om0_mag * 180 / np.pi

    # truth ω at t=0+ via central diff between epochs 1 and 2
    q_a = q_t[1:2]
    q_b = q_t[2:3]
    om_active = finite_diff_omega(q_a, q_b, dt_s, "active")[0]
    om_passive = finite_diff_omega(q_a, q_b, dt_s, "passive")[0]
    err_active = np.linalg.norm(om_active - om0)
    err_passive = np.linalg.norm(om_passive - om0)
    # also check sign-flipped (some conventions emit -ω)
    err_active_neg = np.linalg.norm(-om_active - om0)
    err_passive_neg = np.linalg.norm(-om_passive - om0)

    candidates = {
        "active": (om_active, err_active),
        "passive": (om_passive, err_passive),
        "-active": (-om_active, err_active_neg),
        "-passive": (-om_passive, err_passive_neg),
    }
    best = min(candidates, key=lambda k: candidates[k][1])
    om_best, err_best = candidates[best]
    err_pct = err_best / om0_mag * 100
    return best, err_pct


def angular_distance_to_axis(omega_3vec: np.ndarray, om_truth: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between each ω-hypothesis direction and truth-ω̂.
    Antipodal-aware (|cos|)."""
    om_norm = np.linalg.norm(omega_3vec, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = omega_3vec / safe
    truth_hat = om_truth / np.linalg.norm(om_truth)
    cos_a = np.abs(np.einsum("...i,i->...", om_hat, truth_hat))
    cos_a = np.clip(cos_a, 0.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def main() -> dict:
    z = np.load(DENSE_RUN)
    n_surv = z["n_survivors"]
    obs_times = z["obs_times"]
    q_pool_wxyz = z["q_pool_wxyz"]                 # (100000, 4)
    survive_all = z["survive_all"]                 # (500, 100000) bool
    closest_deg = z["closest_deg_per_epoch"]
    omega_mag_dps_meta = float(z["omega_mag_dps"])

    traj = np.load(TRAJ089)
    q_truth_t = traj["quaternions"]                # (500, 4) wxyz
    om0_rad = traj["omega0_rad"]
    om_mag_dps_truth = float(traj["omega_mag_dps"])

    dt_s = float(np.median(np.diff(obs_times)))

    # --- step 1: smoke-test q convention ---
    convention, conv_err_pct = smoke_test_convention({"quaternions": q_truth_t,
                                                       "omega0_rad": om0_rad}, dt_s)
    print(f"q convention: {convention}, recovery err = {conv_err_pct:.3f}%")

    # parse "-active" or "-passive" → sign + base
    sign = -1 if convention.startswith("-") else +1
    base = convention.lstrip("-")

    # --- step 2: anchor selection ---
    t_a = int(np.argmin(n_surv))
    t_b = t_a + DELTA_EPOCH
    if t_b >= len(n_surv):
        t_a = int(np.argmin(n_surv[:len(n_surv) - DELTA_EPOCH]))
        t_b = t_a + DELTA_EPOCH

    n_a = int(n_surv[t_a])
    n_b = int(n_surv[t_b])
    closest_a = float(closest_deg[t_a])
    closest_b = float(closest_deg[t_b])

    # --- step 3: truth ω at t_a from cached q(t) finite-diff ---
    q_truth_a = q_truth_t[t_a]
    q_truth_b = q_truth_t[t_b]
    Δt_pair = (t_b - t_a) * dt_s
    om_truth_at_ta = sign * finite_diff_omega(
        q_truth_a[None, :], q_truth_b[None, :], Δt_pair, base
    )[0]
    om_truth_mag_at_ta = float(np.linalg.norm(om_truth_at_ta))
    om_truth_mag_at_ta_dps = om_truth_mag_at_ta * 180 / np.pi
    om_truth_dir = om_truth_at_ta / om_truth_mag_at_ta

    # --- step 4: extract clouds + sanity check truth in pool at t_a ---
    idx_a = np.where(survive_all[t_a])[0]
    idx_b = np.where(survive_all[t_b])[0]
    C_a = q_pool_wxyz[idx_a]
    C_b = q_pool_wxyz[idx_b]
    assert len(C_a) == n_a and len(C_b) == n_b

    # is the closest-to-truth pool member at t_a in C_a?
    closest_idx_a = int(z["closest_idx_per_epoch"][t_a])
    truth_q_a_in_cloud = bool(closest_idx_a in idx_a.tolist())
    closest_idx_b = int(z["closest_idx_per_epoch"][t_b])
    truth_q_b_in_cloud = bool(closest_idx_b in idx_b.tolist())

    # --- step 5: all-pair ω-finite-diff ---
    # Tile into shape (n_a * n_b, 4)
    Q_A = np.repeat(C_a, n_b, axis=0)               # (n_a*n_b, 4)
    Q_B = np.tile(C_b, (n_a, 1))                    # (n_a*n_b, 4)
    omegas = sign * finite_diff_omega(Q_A, Q_B, Δt_pair, base)  # (n_a*n_b, 3) rad/s
    om_mags = np.linalg.norm(omegas, axis=1)        # rad/s
    om_mags_dps = om_mags * 180 / np.pi

    # angular distance to truth-ω-direction (ALL pairs, before filter)
    ang_all = angular_distance_to_axis(omegas, om_truth_at_ta)

    # --- step 6: |ω|-prior filters ---
    # Three settings: oracle (±0.5%), tight (±5%), loose (±25%)
    # 'oracle' is ±0.5% to allow finite-diff numerical jitter
    target_mag = om_truth_mag_at_ta  # rad/s
    settings = {
        "oracle (±0.5%)": (0.995, 1.005),
        "tight (±5%)":    (0.95, 1.05),
        "loose (±25%)":   (0.75, 1.25),
    }

    results_per_setting = {}
    for label, (lo_frac, hi_frac) in settings.items():
        mask = (om_mags >= target_mag * lo_frac) & (om_mags <= target_mag * hi_frac)
        n_pass = int(mask.sum())
        ang_pass = ang_all[mask]
        if n_pass == 0:
            results_per_setting[label] = {
                "n_pass": 0, "frac_within_10deg": float("nan"),
                "frac_within_30deg": float("nan"),
                "frac_within_90deg": float("nan"),
                "median_ang_deg": float("nan"),
            }
            continue
        # uniform-on-sphere baseline: P(within X°) = (1-cos(X°))/2 for axes, but
        # since we're measuring axis-distance via |dot|, P(within X°) ≈ 1 - cos(X°).
        # (X° axis-distance ↔ X° apex angle; cap area on unit sphere)
        results_per_setting[label] = {
            "n_pass": n_pass,
            "n_pass_pct": n_pass / len(om_mags) * 100,
            "frac_within_10deg": float((ang_pass < 10).mean()),
            "frac_within_30deg": float((ang_pass < 30).mean()),
            "frac_within_90deg": float((ang_pass < 90).mean()),
            "median_ang_deg": float(np.median(ang_pass)),
            "p10_ang_deg": float(np.percentile(ang_pass, 10)),
            "min_ang_deg": float(ang_pass.min()),
            "ang_pass": ang_pass,
            "om_pass": omegas[mask],
        }

    # uniform-sphere baseline: fraction within X° apex
    def baseline_frac(x_deg: float) -> float:
        return float(1.0 - np.cos(np.radians(x_deg)))

    # --- step 7: figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) histogram of |ω|_dps for all pairs (no filter)
    ax = axes[0, 0]
    ax.hist(om_mags_dps, bins=60, color="lightblue", edgecolor="grey", alpha=0.7)
    ax.axvline(om_truth_mag_at_ta_dps, color="r", lw=1.5,
               label=f"truth |ω|={om_truth_mag_at_ta_dps:.3f} dps")
    for label, (lo_f, hi_f) in settings.items():
        ax.axvspan(om_truth_mag_at_ta_dps * lo_f, om_truth_mag_at_ta_dps * hi_f,
                   alpha=0.1, color={"oracle (±0.5%)": "green",
                                     "tight (±5%)": "orange",
                                     "loose (±25%)": "purple"}[label])
    ax.set_xlim(0, om_truth_mag_at_ta_dps * 3)
    ax.set_xlabel("implied |ω|  (dps)")
    ax.set_ylabel("# pairs")
    ax.set_title(f"all-pair |ω|: {len(om_mags)} pairs from C_{{t_a}}×C_{{t_a+10}}")
    ax.legend(loc="best", fontsize=8)

    # (b) ang-dist histograms by prior setting, log-scale
    ax = axes[0, 1]
    colors = {"oracle (±0.5%)": "green", "tight (±5%)": "orange", "loose (±25%)": "purple"}
    bins = np.linspace(0, 90, 46)
    for label in settings:
        r = results_per_setting[label]
        if r["n_pass"] == 0:
            continue
        ax.hist(r["ang_pass"], bins=bins, alpha=0.5, color=colors[label],
                label=f"{label}: n={r['n_pass']}, p10={r['p10_ang_deg']:.1f}°, "
                      f"f<30°={r['frac_within_30deg']:.2f}",
                density=True)
    # uniform baseline density: dP/dθ = sin(θ)·π/180 (rad)
    theta = np.linspace(0, 90, 200)
    uni_dens = np.sin(np.radians(theta)) * (np.pi / 180)
    ax.plot(theta, uni_dens, "k--", lw=1, label="uniform-sphere baseline")
    ax.set_xlabel("angular distance to truth-ω̂  (deg)")
    ax.set_ylabel("density")
    ax.set_title("ω-direction recovery: hypothesis cloud vs truth-ω̂")
    ax.legend(loc="upper right", fontsize=7)

    # (c) ω-vector 3D scatter projected as polar (axis on unit sphere)
    # Use longitude/latitude in body frame
    ax = axes[1, 0]
    om_truth_hat = om_truth_at_ta / np.linalg.norm(om_truth_at_ta)
    truth_lon = float(np.degrees(np.arctan2(om_truth_hat[1], om_truth_hat[0])))
    truth_lat = float(np.degrees(np.arcsin(om_truth_hat[2])))
    for label in settings:
        r = results_per_setting[label]
        if r["n_pass"] == 0:
            continue
        omp = r["om_pass"]
        omp_hat = omp / np.linalg.norm(omp, axis=1, keepdims=True)
        # antipodal fold: flip if dot(om_hat, truth_hat) < 0
        flip = np.einsum("ij,j->i", omp_hat, om_truth_hat) < 0
        omp_hat[flip] = -omp_hat[flip]
        lon = np.degrees(np.arctan2(omp_hat[:, 1], omp_hat[:, 0]))
        lat = np.degrees(np.arcsin(omp_hat[:, 2]))
        ax.scatter(lon, lat, s=2, alpha=0.3, color=colors[label],
                   label=f"{label} (n={r['n_pass']})")
    ax.scatter(truth_lon, truth_lat, s=200, marker="*", color="red",
               edgecolor="black", linewidth=1.5, label="truth-ω̂", zorder=10)
    ax.set_xlabel("body-frame longitude  (deg)")
    ax.set_ylabel("body-frame latitude  (deg)")
    ax.set_title("ω-direction hypotheses on body-frame unit sphere (antipodal-folded)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) summary text
    ax = axes[1, 1]
    ax.axis("off")
    txt = []
    txt.append(f"q convention: {convention} (recovery err {conv_err_pct:.3f}%)")
    txt.append(f"anchor epoch t_a={t_a}, |C_{{t_a}}|={n_a}, closest pool→truth={closest_a:.2f}°")
    txt.append(f"forward epoch t_b={t_b}, |C_{{t_b}}|={n_b}, closest pool→truth={closest_b:.2f}°")
    txt.append(f"truth q_a in C_a: {truth_q_a_in_cloud}")
    txt.append(f"truth q_b in C_b: {truth_q_b_in_cloud}")
    txt.append(f"Δt = {Δt_pair:.2f}s, total pairs = {len(om_mags):,}")
    txt.append(f"truth |ω| at t_a (finite-diff): {om_truth_mag_at_ta_dps:.4f} dps")
    txt.append(f"truth |ω| (cohort meta): {om_mag_dps_truth:.4f} dps")
    txt.append("")
    txt.append("Concentration vs prior bracket:")
    for label in settings:
        r = results_per_setting[label]
        if r["n_pass"] == 0:
            txt.append(f"  {label}: 0 pairs survive")
            continue
        bl_30 = baseline_frac(30) * 100
        bl_10 = baseline_frac(10) * 100
        txt.append(
            f"  {label}: n={r['n_pass']} ({r['n_pass_pct']:.1f}% of pairs)\n"
            f"     f<10° = {r['frac_within_10deg']*100:.2f}% (baseline {bl_10:.2f}%, "
            f"× {r['frac_within_10deg']*100/bl_10:.1f})\n"
            f"     f<30° = {r['frac_within_30deg']*100:.2f}% (baseline {bl_30:.2f}%, "
            f"× {r['frac_within_30deg']*100/bl_30:.1f})\n"
            f"     median ang = {r['median_ang_deg']:.2f}°, p10 = {r['p10_ang_deg']:.2f}°, "
            f"min = {r['min_ang_deg']:.2f}°"
        )
    ax.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=9,
            family="monospace", transform=ax.transAxes)

    plt.tight_layout()
    fig_p = OUT / "anchor_propagation_overview.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_p}")

    summary = {
        "seed": 89,
        "convention": convention,
        "convention_err_pct": conv_err_pct,
        "anchor_epoch_t_a": t_a,
        "forward_epoch_t_b": t_b,
        "delta_t_s": Δt_pair,
        "n_C_a": n_a,
        "n_C_b": n_b,
        "closest_pool_to_truth_at_t_a_deg": closest_a,
        "closest_pool_to_truth_at_t_b_deg": closest_b,
        "truth_q_a_in_cloud": truth_q_a_in_cloud,
        "truth_q_b_in_cloud": truth_q_b_in_cloud,
        "om_truth_mag_at_ta_dps": om_truth_mag_at_ta_dps,
        "n_total_pairs": int(len(om_mags)),
        "settings": {
            label: {k: v for k, v in r.items() if k not in ("ang_pass", "om_pass")}
            for label, r in results_per_setting.items()
        },
        "baseline_uniform_frac_within_10deg": baseline_frac(10),
        "baseline_uniform_frac_within_30deg": baseline_frac(30),
        "baseline_uniform_frac_within_90deg": baseline_frac(90),
    }
    out_p = OUT / "summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_p}")
    return summary


if __name__ == "__main__":
    s = main()
    print()
    print("=== summary ===")
    print(json.dumps(s, indent=2))
