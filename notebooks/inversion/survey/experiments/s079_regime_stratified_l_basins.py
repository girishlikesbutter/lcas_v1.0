"""s079 — regime-stratified re-score of the s077 L-vector basin sweep.

Background
----------
s077 swept 640 converged basins (10 pilot seeds x 64 multi-start ICs) and
found, among the 104 "competing low-MSE" basins, that |L| magnitude is
pinned (median 0.73% off truth) while L_J2000 *direction* is free (median
90.8 deg off) — discrete clumps, strongly seed-dependent. s073f then showed
the lone seed-89 example (cluster_457) is a *locally isolated* point, not a
soft sheet. The user's reframe: the "equal-|L|, different-direction" multi-
solution structure is not expected on every seed — it should depend on the
*type of tumbling motion*, i.e. short-axis mode (SAM) vs long-axis mode
(LAM) and how near the separatrix the truth state sits.

This experiment is the regime-stratified re-score that tests that. It is a
PURE re-score of results/s077/basin_l_metrics.npz — no propagation, no
rendering, no new compute.

Regime convention
-----------------
Torque-free rigid body, principal inertias I_1 <= I_2 <= I_3.
  disc = 2T*I_2 - |L|^2.
  disc >= 0  -> regime A: polhode encloses I_1 (min-inertia axis)
                = rotation predominantly about the long axis = LAM.
  disc <  0  -> regime B: polhode encloses I_3 (max-inertia axis)
                = rotation predominantly about the short axis = SAM.
  k^2 (elliptic modulus m): -> 1 at the separatrix, -> 0 deep in a regime
  (pure principal-axis spin). Formulas copied from
  lib/jacobi_propagator.py::_build_omega_func (regime A line 121, regime B
  line 154) so s079 is self-contained.

Question
--------
Does the count / character of competing low-MSE basins (the s077 "equal-|L|,
free-direction" multi-solution class) correlate with the truth seed's
tumbling regime (LAM vs SAM) and its separatrix proximity (k^2)?

Caveat (binding, stated up front)
---------------------------------
n = 10 pilot seeds. This is a first-look correlation, NOT a cohort claim.
"competing low-MSE" is the loose surrogate gate from s077 (final_mse < 0.5),
not a hi-fi rho-band. The split reported here is descriptive of these 10
seeds; generalising needs Band A|B multi-sols on more seeds (new compute).

Saves
-----
  results/s079/summary.json   — per-seed regime/k2 + competing-basin stats,
                                per-regime aggregates, the LAM/SAM split
  results/s079/s079_regime_stratified.png — 2-panel diagnostic figure

Cross-references
----------------
  experiments/s077_l_vector_basin_sweep.{py,md}
  experiments/s073f_cluster457_local_geometry.{py,md}
  lib/jacobi_propagator.py  (_build_omega_func — regime/k2 formulas)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.traj_load import truth_state  # noqa: E402
from lib.hifi_render import _build_model  # noqa: E402

RESULTS_DIR = SURVEY_ROOT / "results" / "s079"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

S077_NPZ = SURVEY_ROOT / "results" / "s077" / "basin_l_metrics.npz"


def classify_regime(omega0: np.ndarray, inertia: np.ndarray,
                    I_pa: np.ndarray) -> dict:
    """Truth tumbling regime + separatrix proximity from (omega0, inertia).

    omega0 is in the body frame; `inertia` is the full body-frame tensor;
    I_pa is its ascending eigenvalues (I_1 <= I_2 <= I_3). 2T and |L|^2 are
    computed basis-independently (omega.I.omega, |I omega|^2) so the body
    frame need NOT coincide with the principal-axis frame — this matches
    lib/jacobi_propagator.py::omega_jacobi, which eigendecomposes internally.
    The k^2 formulas mirror _build_omega_func (regime A line 121, B line 154).
    """
    I_1, I_2, I_3 = (float(x) for x in I_pa)
    Iw = inertia @ omega0
    twoT = float(omega0 @ Iw)
    L2 = float(Iw @ Iw)
    disc = twoT * I_2 - L2
    if disc >= 0.0:
        regime, mode = "A", "LAM"  # polhode encloses I_1 (long axis)
        m = (I_3 - I_2) * (L2 - twoT * I_1) / ((I_2 - I_1) * (twoT * I_3 - L2))
    else:
        regime, mode = "B", "SAM"  # polhode encloses I_3 (short axis)
        m = (I_2 - I_1) * (twoT * I_3 - L2) / ((I_3 - I_2) * (L2 - twoT * I_1))
    # signed, normalised separatrix distance: 0 at separatrix, +/-1 at a
    # pure-axis spin edge. disc normalised by 2T*I_2.
    sep_dist = disc / (twoT * I_2)
    return {"regime": regime, "mode": mode, "k2": float(m),
            "twoT": twoT, "L2": L2, "disc": disc,
            "sep_dist_norm": float(sep_dist)}


def _stats(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75))}


def main() -> int:
    # ----------------------------------------------------------- substrate
    if not S077_NPZ.exists():
        print(f"ERROR: missing {S077_NPZ} — run s077 first.", file=sys.stderr)
        return 1
    d = np.load(S077_NPZ, allow_pickle=True)
    seeds = d["seed"].astype(int)
    basin_class = d["basin_class"].astype(str)
    L_dir_angle = d["L_dir_angle_deg"]
    L_mag_rel_diff = d["L_mag_rel_diff"]
    rel_dL = d["rel_dL"]
    d_twoT_rel = d["d_twoT_rel"]
    twoT = d["twoT"]
    L2 = d["L2"]

    _models, inertia = _build_model()
    inertia = np.asarray(inertia, dtype=np.float64)
    I_vals, _ = np.linalg.eigh(inertia)   # ascending
    I_pa = np.sort(I_vals)

    unique_seeds = sorted(set(int(s) for s in seeds))
    comp = basin_class == "competing_low_mse"

    # cross-check substrate: s077's per-seed truth_regime (computed via the
    # validated omega_jacobi path) — our classify_regime must reproduce it.
    s077_summary = json.load(open(S077_NPZ.parent / "summary.json"))
    s077_regime = {ps["seed"]: ps["truth_regime"]
                   for ps in s077_summary["per_seed"].values()}

    print("=== s079 — regime-stratified re-score of s077 ===")
    print(f"  substrate: {S077_NPZ.relative_to(SURVEY_ROOT)}  "
          f"({seeds.size} basins, {len(unique_seeds)} seeds)")
    print(f"  principal inertias (ascending, kg m^2): {I_pa}")
    print(f"  competing low-MSE basins (cohort): {int(comp.sum())}")
    print()

    # --------------------------------------------------- per-seed classification
    per_seed = {}
    for s in unique_seeds:
        t = truth_state(s)
        om0 = np.asarray(t["omega0_rad"], dtype=np.float64)
        reg = classify_regime(om0, inertia, I_pa)
        assert reg["regime"] == s077_regime[s], (
            f"seed {s}: regime {reg['regime']} disagrees with s077's "
            f"omega_jacobi label {s077_regime[s]}")

        sm = seeds == s
        cm = sm & comp
        # distinct polhodes among competing basins: bin (2T,|L|^2) at 1% tol,
        # normalised by this seed's truth Casimirs.
        n_distinct = 0
        if cm.sum():
            c2T, cL2 = twoT[cm], L2[cm]
            taken = np.zeros(c2T.size, dtype=bool)
            for j in range(c2T.size):
                if taken[j]:
                    continue
                near = (np.abs(c2T - c2T[j]) / reg["twoT"] < 0.01) & \
                       (np.abs(cL2 - cL2[j]) / reg["L2"] < 0.01)
                taken |= near
                n_distinct += 1

        per_seed[f"seed_{s:03d}"] = {
            "seed": s,
            "regime": reg["regime"],
            "mode": reg["mode"],
            "k2": reg["k2"],
            "sep_dist_norm": reg["sep_dist_norm"],
            "n_basins": int(sm.sum()),
            "n_competing_low_mse": int(cm.sum()),
            "competing_distinct_polhodes_1pct": n_distinct,
            "competing_L_dir_angle_deg": _stats(L_dir_angle[cm]),
            "competing_L_mag_rel_diff": _stats(L_mag_rel_diff[cm]),
            "competing_rel_dL": _stats(rel_dL[cm]),
            "competing_d_twoT_rel": _stats(d_twoT_rel[cm]),
        }

    # ---------------------------------------------------- per-regime aggregate
    def _regime_agg(mode: str) -> dict:
        ps = [v for v in per_seed.values() if v["mode"] == mode]
        seed_ids = [v["seed"] for v in ps]
        n_comp_per_seed = np.array([v["n_competing_low_mse"] for v in ps])
        n_distinct_per_seed = np.array(
            [v["competing_distinct_polhodes_1pct"] for v in ps])
        # pooled competing basins across this regime's seeds
        regime_seed_set = set(seed_ids)
        pool = comp & np.array([int(s) in regime_seed_set for s in seeds])
        return {
            "mode": mode,
            "n_seeds": len(ps),
            "seeds": seed_ids,
            "n_competing_total": int(pool.sum()),
            "n_competing_per_seed": n_comp_per_seed.tolist(),
            "n_competing_per_seed_median": float(np.median(n_comp_per_seed))
            if n_comp_per_seed.size else 0.0,
            "n_distinct_polhodes_per_seed": n_distinct_per_seed.tolist(),
            "pooled_competing_L_dir_angle_deg": _stats(L_dir_angle[pool]),
            "pooled_competing_L_mag_rel_diff": _stats(L_mag_rel_diff[pool]),
            "pooled_competing_d_twoT_rel": _stats(d_twoT_rel[pool]),
        }

    agg_lam = _regime_agg("LAM")
    agg_sam = _regime_agg("SAM")

    summary = {
        "experiment": "s079",
        "substrate": str(S077_NPZ.relative_to(SURVEY_ROOT)),
        "n_basins": int(seeds.size),
        "n_seeds": len(unique_seeds),
        "principal_inertias_ascending": I_pa.tolist(),
        "caveat": ("n=10 pilot seeds, first-look correlation only; "
                   "'competing low-MSE' = s077 surrogate gate final_mse<0.5, "
                   "not a hi-fi rho-band."),
        "regime_split": {
            "LAM": agg_lam,
            "SAM": agg_sam,
        },
        "per_seed": per_seed,
    }
    json_path = RESULTS_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    mode_color = {"LAM": "tab:red", "SAM": "tab:blue"}

    # Panel 1: n_competing vs k2, colored by regime, sized by distinct polhodes
    ax = axes[0]
    for mode in ("LAM", "SAM"):
        ps = [v for v in per_seed.values() if v["mode"] == mode]
        x = [v["k2"] for v in ps]
        y = [v["n_competing_low_mse"] for v in ps]
        sz = [30 + 18 * v["competing_distinct_polhodes_1pct"] for v in ps]
        ax.scatter(x, y, s=sz, c=mode_color[mode], alpha=0.75,
                   edgecolors="k", linewidths=0.4,
                   label=f"{mode} (n={len(ps)} seeds)")
        for v in ps:
            ax.annotate(str(v["seed"]), (v["k2"], v["n_competing_low_mse"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("truth k^2  (elliptic modulus; -> 1 at separatrix)")
    ax.set_ylabel("competing low-MSE basins on this seed")
    ax.set_title("s079 — competing-basin count vs regime + separatrix proximity\n"
                 "marker size ~ distinct polhodes among competing basins")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: pooled L-direction-offset distribution by regime
    ax = axes[1]
    for mode in ("LAM", "SAM"):
        seed_set = {v["seed"] for v in per_seed.values() if v["mode"] == mode}
        pool = comp & np.array([int(s) in seed_set for s in seeds])
        vals = L_dir_angle[pool]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=np.arange(0, 185, 15), alpha=0.6,
                    color=mode_color[mode],
                    label=f"{mode} (n={vals.size} competing basins)")
    ax.set_xlabel("L_J2000 direction angle vs truth (deg)")
    ax.set_ylabel("competing low-MSE basin count")
    ax.set_title("s079 — L-direction offset of competing basins, by regime")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"s079 — regime-stratified L-basin re-score "
        f"(10 pilot seeds; LAM {agg_lam['n_competing_total']} vs "
        f"SAM {agg_sam['n_competing_total']} competing basins)",
        fontsize=11)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "s079_regime_stratified.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- console
    print("  per-seed regime classification + competing-basin count:")
    print(f"    {'seed':>5} {'mode':>5} {'regime':>7} {'k2':>8} "
          f"{'sep_dist':>9} {'n_comp':>7} {'distinct':>9} {'Ldir_med':>9}")
    for s in unique_seeds:
        v = per_seed[f"seed_{s:03d}"]
        cda = v["competing_L_dir_angle_deg"]
        ldm = f"{cda['median']:.1f}" if cda.get("n", 0) else "--"
        print(f"    {v['seed']:>5} {v['mode']:>5} {v['regime']:>7} "
              f"{v['k2']:>8.4f} {v['sep_dist_norm']:>9.4f} "
              f"{v['n_competing_low_mse']:>7d} "
              f"{v['competing_distinct_polhodes_1pct']:>9d} {ldm:>9}")
    print()
    print("  regime split (competing low-MSE basins):")
    for agg in (agg_lam, agg_sam):
        ld = agg["pooled_competing_L_dir_angle_deg"]
        lm = agg["pooled_competing_L_mag_rel_diff"]
        lds = (f"median {ld['median']:.1f} deg [{ld['min']:.1f}, {ld['max']:.1f}]"
               if ld.get("n", 0) else "n/a")
        lms = (f"median {lm['median']:.3%}" if lm.get("n", 0) else "n/a")
        print(f"    {agg['mode']}: {agg['n_seeds']} seeds, "
              f"{agg['n_competing_total']} competing basins total "
              f"(median {agg['n_competing_per_seed_median']:.1f}/seed)")
        print(f"         pooled L-direction offset: {lds}")
        print(f"         pooled |L|-magnitude rel diff: {lms}")
    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
