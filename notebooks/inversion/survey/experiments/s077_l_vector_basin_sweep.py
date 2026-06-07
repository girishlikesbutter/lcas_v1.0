"""s077 — L-vector ↔ basin cohort sweep (analytical, no new compute).

Background
----------
s073/s073d found that one Band-A multi-solution basin on seed 89
(`cluster_457`) has nearly the same angular-momentum *magnitude* as truth
(|L| match 0.55%) but a wildly different inertial *direction* (L_J2000 off
128.6°). The s073e/cat-4 framing escalated this single pair to a
"continuous family of solutions". This experiment drops the hypothesis and
asks the general, no-assumptions question instead:

    Across a cohort of converged inversion basins, what is the empirical
    relationship between a basin's L vector (magnitude AND direction) and
    the truth L vector?

Substrate
---------
results/s011/runs.npz — 640 converged basins from the post-fix s011/s068
cohort pilot (10 seeds × 64 multi-start Sobol ICs, LM-polished). Confirmed
post-fix: last written by commit 7bf3352. No new optimisation is run here;
this is a pure re-score of cached endpoints.

What we compute, per basin
--------------------------
  L_J2000 = R(q0_final).T @ I @ omega_final          (inertial momentum)
  |L|, body-frame Casimirs 2T and |L|^2, regime A/B, k^2, T_pol
  vs that seed's truth:
    |dL|/|L_truth|, L_J2000 direction angle, d(2T)/2T, d|L^2|/|L^2|,
    magnitude-only |L| relative diff.

Basin classes (from cached s011 flags + final surrogate MSE):
  truth   — truth_basin_strict
  twin    — twin_basin_strict
  competing_low_mse — neither, and final_mse < MSE_LOW (0.5, the s011
            convention for "competing basin")
  high_mse — neither, final_mse >= MSE_LOW

NOTE final_mse is the LM surrogate MSE, not a hi-fi rho. Language in the
writeup keeps that distinction.

Saves
-----
  results/s077/basin_l_metrics.npz — every per-basin array
  results/s077/summary.json        — per-seed aggregates + cohort headline
  results/s077/*.png               — four diagnostic figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ellipk

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.jacobi_propagator import _quat_to_matrix, omega_jacobi  # noqa: E402
from lib.traj_load import truth_state  # noqa: E402
from lib.hifi_render import _build_model  # noqa: E402

RESULTS_DIR = SURVEY_ROOT / "results" / "s077"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MSE_LOW = 0.5  # s011 convention for "competing basin below mse"
_T0 = np.array([0.0])


def l_j2000(q_wxyz: np.ndarray, omega_rad: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Inertial angular momentum. Same formula as s073 / s066."""
    R = _quat_to_matrix(q_wxyz)        # passive J2000 -> body
    return R.T @ (inertia @ omega_rad)  # body -> J2000


def casimirs(omega_rad: np.ndarray, inertia: np.ndarray) -> dict:
    """Body-frame Casimir invariants + polhode descriptors via omega_jacobi.

    Returns a dict with twoT, L2, regime, k2, T_pol. On any numerical
    failure (near-axis spin, near-separatrix, near-symmetric inertia)
    returns regime='FAIL' and NaN descriptors — the scalar 2T / |L|^2 are
    still computed directly so they are always finite.
    """
    # 2T and |L|^2 are basis-independent; compute directly so they never fail.
    Iw = inertia @ omega_rad
    twoT = float(omega_rad @ Iw)
    L2 = float(Iw @ Iw)
    out = {"twoT": twoT, "L2": L2, "regime": "FAIL", "k2": np.nan, "T_pol": np.nan}
    try:
        _, info = omega_jacobi(_T0, omega_rad, inertia)
        k2 = float(info["m"])
        tau_dot = float(info["tau_dot"])
        out["regime"] = str(info["regime"])
        out["k2"] = k2
        if np.isfinite(k2) and 0.0 <= k2 < 1.0 and tau_dot != 0.0:
            out["T_pol"] = float(4.0 * ellipk(k2) / abs(tau_dot))
    except Exception:
        pass
    return out


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return np.nan
    c = float(np.clip(u @ v / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def main() -> int:
    # --- Inertia (constant across all m048 seeds) ---
    _, inertia = _build_model()
    inertia = np.asarray(inertia, dtype=np.float64)

    # --- Cached cohort basins ---
    runs_path = SURVEY_ROOT / "results" / "s011" / "runs.npz"
    runs = np.load(runs_path)
    n = int(runs["seed"].shape[0])
    seeds = runs["seed"]
    q0_final = runs["q0_final_wxyz"]
    omega_final = runs["omega_final_rad"]
    final_mse = runs["final_mse"]
    q0_err_deg = runs["q0_err_deg"]
    omega_dir_err_deg = runs["omega_dir_err_deg"]
    omega_mag_err_pct = runs["omega_mag_err_pct"]
    truth_basin = runs["truth_basin_strict"]
    twin_basin = runs["twin_basin_strict"]
    success = runs["success"]

    unique_seeds = sorted(set(int(s) for s in seeds))
    print(f"=== s077 — L-vector / basin cohort sweep ===")
    print(f"  substrate: {runs_path.relative_to(SURVEY_ROOT)}  ({n} basins, "
          f"{len(unique_seeds)} seeds: {unique_seeds})")
    print(f"  inertia diag (kg m^2): {np.diag(inertia)}")
    print()

    # --- Truth L / Casimirs per seed ---
    truth_L = {}
    truth_cas = {}
    for s in unique_seeds:
        t = truth_state(s)
        q0t = np.asarray(t["q0_wxyz"], dtype=np.float64)
        om0t = np.asarray(t["omega0_rad"], dtype=np.float64)
        truth_L[s] = l_j2000(q0t, om0t, inertia)
        truth_cas[s] = casimirs(om0t, inertia)

    # --- Per-basin metrics ---
    L_vec = np.full((n, 3), np.nan)
    L_mag = np.full(n, np.nan)
    twoT = np.full(n, np.nan)
    L2 = np.full(n, np.nan)
    k2 = np.full(n, np.nan)
    T_pol = np.full(n, np.nan)
    regime = np.empty(n, dtype="<U4")
    rel_dL = np.full(n, np.nan)             # |dL| / |L_truth|
    L_dir_angle = np.full(n, np.nan)        # deg, vs truth L_J2000
    L_mag_rel_diff = np.full(n, np.nan)     # ||L|-|L_truth|| / |L_truth|
    d_twoT_rel = np.full(n, np.nan)
    d_L2_rel = np.full(n, np.nan)
    basin_class = np.empty(n, dtype="<U20")

    for i in range(n):
        s = int(seeds[i])
        qf = np.asarray(q0_final[i], dtype=np.float64)
        omf = np.asarray(omega_final[i], dtype=np.float64)
        Lb = l_j2000(qf, omf, inertia)
        L_vec[i] = Lb
        L_mag[i] = np.linalg.norm(Lb)
        cas = casimirs(omf, inertia)
        twoT[i] = cas["twoT"]
        L2[i] = cas["L2"]
        k2[i] = cas["k2"]
        T_pol[i] = cas["T_pol"]
        regime[i] = cas["regime"]

        Lt = truth_L[s]
        Lt_mag = np.linalg.norm(Lt)
        rel_dL[i] = np.linalg.norm(Lb - Lt) / Lt_mag
        L_dir_angle[i] = angle_between(Lb, Lt)
        L_mag_rel_diff[i] = abs(L_mag[i] - Lt_mag) / Lt_mag
        ct = truth_cas[s]
        d_twoT_rel[i] = abs(cas["twoT"] - ct["twoT"]) / ct["twoT"]
        d_L2_rel[i] = abs(cas["L2"] - ct["L2"]) / ct["L2"]

        if truth_basin[i]:
            basin_class[i] = "truth"
        elif twin_basin[i]:
            basin_class[i] = "twin"
        elif final_mse[i] < MSE_LOW:
            basin_class[i] = "competing_low_mse"
        else:
            basin_class[i] = "high_mse"

    # --- Save NPZ checkpoint ---
    npz_path = RESULTS_DIR / "basin_l_metrics.npz"
    np.savez(
        npz_path,
        seed=seeds, success=success,
        q0_final_wxyz=q0_final, omega_final_rad=omega_final,
        final_mse=final_mse, q0_err_deg=q0_err_deg,
        omega_dir_err_deg=omega_dir_err_deg, omega_mag_err_pct=omega_mag_err_pct,
        truth_basin_strict=truth_basin, twin_basin_strict=twin_basin,
        basin_class=basin_class,
        L_vec=L_vec, L_mag=L_mag, twoT=twoT, L2=L2, k2=k2, T_pol=T_pol,
        regime=regime,
        rel_dL=rel_dL, L_dir_angle_deg=L_dir_angle,
        L_mag_rel_diff=L_mag_rel_diff,
        d_twoT_rel=d_twoT_rel, d_L2_rel=d_L2_rel,
        inertia=inertia,
        truth_L=np.array([truth_L[s] for s in unique_seeds]),
        truth_seeds=np.array(unique_seeds),
    )

    # --- Aggregates ---
    comp = basin_class == "competing_low_mse"
    truth_m = basin_class == "truth"
    twin_m = basin_class == "twin"

    def _stats(mask, arr):
        v = arr[mask & np.isfinite(arr)]
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "median": float(np.median(v)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75))}

    per_seed = {}
    for s in unique_seeds:
        sm = seeds == s
        cm = sm & comp
        # distinct polhodes among competing basins: bin (2T, |L|^2) at 1% tol
        ct = truth_cas[s]
        comp_2T = twoT[cm]
        comp_L2 = L2[cm]
        # count clusters of (2T,|L|2) within 1% of each other
        n_distinct = 0
        if comp_2T.size:
            taken = np.zeros(comp_2T.size, dtype=bool)
            for j in range(comp_2T.size):
                if taken[j]:
                    continue
                near = (np.abs(comp_2T - comp_2T[j]) / ct["twoT"] < 0.01) & \
                       (np.abs(comp_L2 - comp_L2[j]) / ct["L2"] < 0.01)
                taken |= near
                n_distinct += 1
        per_seed[f"seed_{s:03d}"] = {
            "seed": s,
            "n_basins": int(sm.sum()),
            "n_truth": int((sm & truth_m).sum()),
            "n_twin": int((sm & twin_m).sum()),
            "n_competing_low_mse": int(cm.sum()),
            "competing_L_dir_angle_deg": _stats(cm, L_dir_angle),
            "competing_rel_dL": _stats(cm, rel_dL),
            "competing_L_mag_rel_diff": _stats(cm, L_mag_rel_diff),
            "competing_d_twoT_rel": _stats(cm, d_twoT_rel),
            "competing_distinct_polhodes_1pct": n_distinct,
            "truth_regime": truth_cas[s]["regime"],
            "truth_k2": truth_cas[s]["k2"],
        }

    summary = {
        "experiment": "s077",
        "substrate": str(runs_path.relative_to(SURVEY_ROOT)),
        "n_basins": n,
        "seeds": unique_seeds,
        "mse_low_threshold": MSE_LOW,
        "inertia_diag_kg_m2": np.diag(inertia).tolist(),
        "cohort": {
            "n_truth": int(truth_m.sum()),
            "n_twin": int(twin_m.sum()),
            "n_competing_low_mse": int(comp.sum()),
            "n_high_mse": int((basin_class == "high_mse").sum()),
            "competing_L_dir_angle_deg": _stats(comp, L_dir_angle),
            "competing_rel_dL": _stats(comp, rel_dL),
            "competing_L_mag_rel_diff": _stats(comp, L_mag_rel_diff),
            "competing_d_twoT_rel": _stats(comp, d_twoT_rel),
            "competing_d_L2_rel": _stats(comp, d_L2_rel),
        },
        "per_seed": per_seed,
    }
    json_path = RESULTS_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # --- Figures ---
    cls_color = {"truth": "tab:green", "twin": "tab:blue",
                 "competing_low_mse": "tab:red", "high_mse": "lightgray"}

    # Fig 1: L direction angle vs final_mse, colored by class
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cls in ["high_mse", "competing_low_mse", "twin", "truth"]:
        m = basin_class == cls
        ax.scatter(final_mse[m], L_dir_angle[m], s=22, alpha=0.7,
                   c=cls_color[cls], label=f"{cls} (n={int(m.sum())})",
                   edgecolors="none")
    ax.axvline(MSE_LOW, color="k", ls=":", lw=1, label=f"mse={MSE_LOW}")
    ax.set_xscale("log")
    ax.set_xlabel("final surrogate MSE (LM endpoint)")
    ax.set_ylabel("L_J2000 direction angle vs truth (deg)")
    ax.set_title("s077 — basin L-direction offset vs fit quality (640 basins, 10 seeds)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = RESULTS_DIR / "s077_Ldir_vs_mse.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # Fig 2: histogram of |dL|/|L_truth| and L_mag_rel_diff for competing basins
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cdir = L_dir_angle[comp]
    crel = rel_dL[comp]
    cmag = L_mag_rel_diff[comp]
    axes[0].hist(cmag[np.isfinite(cmag)], bins=30, color="tab:red", alpha=0.8)
    axes[0].set_xlabel("||L| - |L_truth|| / |L_truth|  (magnitude-only diff)")
    axes[0].set_ylabel("competing-low-mse basin count")
    axes[0].set_title("Is there an equal-|L| peak near 0?")
    axes[0].grid(alpha=0.3)
    axes[1].hist(cdir[np.isfinite(cdir)], bins=30, color="tab:purple", alpha=0.8)
    axes[1].set_xlabel("L_J2000 direction angle vs truth (deg)")
    axes[1].set_ylabel("competing-low-mse basin count")
    axes[1].set_title("Direction-offset distribution")
    axes[1].grid(alpha=0.3)
    fig.suptitle("s077 — competing low-MSE basins: L magnitude vs direction")
    fig.tight_layout()
    f2 = RESULTS_DIR / "s077_competing_hist.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    # Fig 3: (|L| mag rel diff, L dir angle) plane for competing basins
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(L_mag_rel_diff[comp], L_dir_angle[comp],
                    c=np.log10(final_mse[comp]), cmap="viridis", s=40,
                    edgecolors="k", linewidths=0.3)
    ax.set_xlabel("magnitude-only diff  ||L| - |L_truth|| / |L_truth|")
    ax.set_ylabel("L_J2000 direction angle vs truth (deg)")
    ax.set_title("s077 — competing low-MSE basins in the (|L|, L-direction) plane\n"
                 "equal-|L|/spread-direction = points hug the left edge")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10(final surrogate MSE)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = RESULTS_DIR / "s077_L_plane.png"
    fig.savefig(f3, dpi=130)
    plt.close(fig)

    # Fig 4: per-seed (2T, |L|^2) of competing basins vs truth
    ncol = 5
    nrow = int(np.ceil(len(unique_seeds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, unique_seeds):
        sm = seeds == s
        cm = sm & comp
        tm = sm & truth_m
        ax.scatter(twoT[cm], L2[cm], s=30, c="tab:red", alpha=0.7,
                   label="competing")
        ax.scatter(twoT[tm], L2[tm], s=120, marker="*", c="tab:green",
                   edgecolors="k", label="truth", zorder=5)
        ax.set_title(f"seed {s} (n_comp={int(cm.sum())}, "
                     f"distinct={per_seed[f'seed_{s:03d}']['competing_distinct_polhodes_1pct']})",
                     fontsize=9)
        ax.set_xlabel("2T", fontsize=8)
        ax.set_ylabel("|L|^2", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes[len(unique_seeds):]:
        ax.axis("off")
    fig.suptitle("s077 — Casimir (2T, |L|^2) of competing basins vs truth, per seed")
    fig.tight_layout()
    f4 = RESULTS_DIR / "s077_casimir_per_seed.png"
    fig.savefig(f4, dpi=120)
    plt.close(fig)

    # --- Console report ---
    co = summary["cohort"]
    print(f"  basin classes: truth={co['n_truth']}  twin={co['n_twin']}  "
          f"competing_low_mse={co['n_competing_low_mse']}  high_mse={co['n_high_mse']}")
    print()
    print(f"  competing low-MSE basins (n={co['n_competing_low_mse']}):")
    cda = co["competing_L_dir_angle_deg"]
    cmr = co["competing_L_mag_rel_diff"]
    crd = co["competing_rel_dL"]
    cdt = co["competing_d_twoT_rel"]
    print(f"    L direction angle vs truth : median {cda['median']:.1f}°  "
          f"[{cda['min']:.1f}, {cda['max']:.1f}]  IQR [{cda['p25']:.1f}, {cda['p75']:.1f}]")
    print(f"    |L| magnitude rel diff     : median {cmr['median']:.3%}  "
          f"[{cmr['min']:.3%}, {cmr['max']:.3%}]  IQR [{cmr['p25']:.3%}, {cmr['p75']:.3%}]")
    print(f"    |dL|/|L_truth| (full vec)  : median {crd['median']:.3f}  "
          f"[{crd['min']:.3f}, {crd['max']:.3f}]")
    print(f"    d(2T)/2T                   : median {cdt['median']:.3%}  "
          f"[{cdt['min']:.3%}, {cdt['max']:.3%}]")
    print()
    print("  per-seed competing-basin polhode count (distinct 2T,|L|^2 @ 1%):")
    for s in unique_seeds:
        ps = per_seed[f"seed_{s:03d}"]
        print(f"    seed {s:3d}: n_comp={ps['n_competing_low_mse']:2d}  "
              f"distinct_polhodes={ps['competing_distinct_polhodes_1pct']:2d}  "
              f"truth_regime={ps['truth_regime']}")
    print()
    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {f1}")
    print(f"Saved: {f2}")
    print(f"Saved: {f3}")
    print(f"Saved: {f4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
