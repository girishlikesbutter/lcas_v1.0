"""s081 — hi-fi rho-bands + twin classification for the s011/s068 basins.

Background
----------
s080 transformed the cached surrogate-v2 MSE of all 640 s011/s068 basins
into a surrogate-rho proxy. It found the loose "competing" gate
(final_mse < 0.5) admits basins deep into Band D, and that seed 10 in
particular has ~12 distinct surrogate-Band-A|B basins — some of which look
like approximate body-twins. Two things s080 could NOT settle:

  1. surrogate-rho is a PROXY. The phantom-basin failure mode (surrogate
     says good, hi-fi says bad) is real — s073f / the local-window-polish
     memory put the divergence at 30-70 rho. Only a hi-fi render gives the
     true rho-band.
  2. some surrogate-Band-A|B basins sit ~180 deg from truth in q0 with
     omega-direction nearly aligned — candidates for the body-X twin
     degeneracy (`lib/twin.py`), an EXACT LC-preserving map, not a new
     physical solution. They need the twin convention applied.

This experiment renders the 145 non-garbage basins hi-fi, classifies each
against truth AND its body-twin, and re-checks the s079 LAM/SAM regime
split on the *hi-fi* Band A|B subset.

Scope
-----
Render the 41 truth-basin + 104 competing-low-MSE basins (145 total). The
495 high-MSE basins are skipped: surrogate rho >= 14 there, and the
surrogate failure mode is optimism out-of-basin, not pessimism — a hi-fi
Band A|B hiding in surrogate-Band-D is not a credible risk. Truth basins
are rendered as a pipeline control (they must come back hi-fi Band A).

Twin classification
-------------------
For each basin, the body-X twin of *truth* is `twin(q0_truth, omega_truth)`
= `(q_180x (X) q0_truth, R_180x . omega_truth)` (lib/twin.py). A basin is
labelled:
  truth    — q-geodesic to truth state < TWIN_Q_TOL and omega within tol
  twin     — q-geodesic to twin-of-truth < TWIN_Q_TOL and omega within tol
  distinct — neither
Raw geodesic / omega distances to both anchors are saved so the label is
auditable, not a black box.

Saves
-----
  results/s081/basin_hifi.npz   — per-basin hi-fi rho, band, twin metrics
  results/s081/summary.json     — band counts by class/regime, hi-fi vs
                                  surrogate agreement, s079 split re-checked
                                  on the hi-fi Band A|B subset, per-seed
                                  distinct-multisol dedup
  results/s081/s081_hifi_vs_surrogate.png
  results/s081/s081_rho_bands.png

Cross-references
----------------
  experiments/s080_basin_rho_bands.{py,md}
  experiments/s079_regime_stratified_l_basins.{py,md}
  experiments/s073f_cluster457_local_geometry.{py,md}
  lib/twin.py, lib/hifi_render.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402
from lib.traj_load import truth_state  # noqa: E402
from lib.twin import twin  # noqa: E402

RESULTS_DIR = SURVEY_ROOT / "results" / "s081"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

S077_NPZ = SURVEY_ROOT / "results" / "s077" / "basin_l_metrics.npz"
S079_SUMMARY = SURVEY_ROOT / "results" / "s079" / "summary.json"
S080_NPZ = SURVEY_ROOT / "results" / "s080" / "basin_rho.npz"

# Twin / truth proximity tolerances (LM polish drifts a few deg / few %).
TWIN_Q_TOL_DEG = 8.0
TWIN_OMEGA_REL_TOL = 0.08
BANDS = ["A", "B", "C", "D"]


def q_geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """Geodesic angle between two unit quaternions (double-cover aware)."""
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    c = min(1.0, abs(float(qa @ qb)))
    return float(np.degrees(2.0 * np.arccos(c)))


def omega_rel_dist(oa: np.ndarray, ob: np.ndarray) -> float:
    """Relative omega distance, normalised by |ob|."""
    nb = np.linalg.norm(ob)
    return float(np.linalg.norm(oa - ob) / nb) if nb > 0 else float("nan")


# --------------------------------------------------------------------- worker
_CTX: dict = {}


def _get_ctx(seed: int) -> dict:
    if seed not in _CTX:
        _CTX[seed] = build_context(seed)
    return _CTX[seed]


def _render_one(args):
    idx, seed, q0, om0 = args
    try:
        ctx = _get_ctx(int(seed))
        pred = render_hifi(np.asarray(q0, dtype=np.float64),
                           np.asarray(om0, dtype=np.float64), ctx)
        rho = rho_from_hifi(pred, np.asarray(ctx["mag_hifi_truth"],
                                             dtype=np.float64))
    except Exception:
        rho = float("nan")
    # compute_shadows (trimesh) leaves cyclic garbage that refcounting can't
    # free; without an explicit collect a Pool worker doing many renders
    # back-to-back climbs from ~1.1 GB to ~1.55+ GB. gc.collect() pins each
    # worker at a flat ~1.13 GB steady-state (verified: 14-render probe).
    gc.collect()
    return idx, float(rho)


def _band_counts(bands: np.ndarray) -> dict:
    return {b: int((bands == b).sum()) for b in BANDS}


def main() -> int:
    t_start = time.time()

    # --------------------------------------------------------------- substrate
    d = np.load(S077_NPZ, allow_pickle=True)
    seeds = d["seed"].astype(int)
    q0_final = d["q0_final_wxyz"]
    omega_final = d["omega_final_rad"]
    basin_class = d["basin_class"].astype(str)
    L_dir_angle = d["L_dir_angle_deg"]
    L_mag_rel_diff = d["L_mag_rel_diff"]
    n_all = seeds.size

    s080 = np.load(S080_NPZ, allow_pickle=True)
    rho_surr = s080["rho_surr"].astype(np.float64)
    band_surr = s080["band"].astype(str)

    s079 = json.load(open(S079_SUMMARY))
    seed_mode = {ps["seed"]: ps["mode"] for ps in s079["per_seed"].values()}
    mode_per_basin = np.array([seed_mode[int(s)] for s in seeds])

    # render set: truth + competing (skip the 495 surrogate-Band-D high_mse)
    render_mask = np.isin(basin_class, ["truth", "twin", "competing_low_mse"])
    render_idx = np.where(render_mask)[0]

    print("=== s081 — hi-fi rho-bands + twin classification ===")
    print(f"  substrate: {S077_NPZ.relative_to(SURVEY_ROOT)}  ({n_all} basins)")
    print(f"  rendering {render_idx.size} basins hi-fi "
          f"(truth + competing; skipping {n_all - render_idx.size} high-MSE)")
    print()

    # --------------------------------------------- truth + twin anchors / seed
    truth_anchor, twin_anchor = {}, {}
    for s in sorted(set(int(x) for x in seeds)):
        t = truth_state(s)
        q0t = np.asarray(t["q0_wxyz"], dtype=np.float64)
        om0t = np.asarray(t["omega0_rad"], dtype=np.float64)
        truth_anchor[s] = (q0t, om0t)
        twin_anchor[s] = twin(q0t, om0t)

    # ------------------------------------------------------- render hi-fi Pool
    args = [(int(i), int(seeds[i]), q0_final[i], omega_final[i])
            for i in render_idx]
    # Each hi-fi render holds ~1.13 GB resident in its worker. The box has
    # 30 GB RAM; Pool(24) (the s081-v1 default) needed ~27+ GB just for
    # workers and OOM-killed once trimesh garbage pushed workers past 1.5 GB.
    # 16 workers x 1.13 GB ~= 18 GB leaves comfortable headroom.
    nproc = min(16, mp.cpu_count())
    t_render = time.time()
    with mp.Pool(nproc) as pool:
        out = pool.map(_render_one, args, chunksize=2)
    wall_render = time.time() - t_render

    rho_hifi = np.full(n_all, np.nan)
    for idx, rho in out:
        rho_hifi[idx] = rho
    band_hifi = np.array([rho_band(r) if np.isfinite(r) else "X"
                          for r in rho_hifi])
    print(f"  rendered {render_idx.size} basins in {wall_render:.1f}s "
          f"on Pool({nproc})")

    # ----------------------------------------------- twin / truth classification
    twin_label = np.array(["unrendered"] * n_all, dtype="<U10")
    q_to_truth = np.full(n_all, np.nan)
    q_to_twin = np.full(n_all, np.nan)
    om_to_truth = np.full(n_all, np.nan)
    om_to_twin = np.full(n_all, np.nan)
    for i in render_idx:
        s = int(seeds[i])
        qf = np.asarray(q0_final[i], dtype=np.float64)
        omf = np.asarray(omega_final[i], dtype=np.float64)
        q0t, om0t = truth_anchor[s]
        q0w, om0w = twin_anchor[s]
        q_to_truth[i] = q_geodesic_deg(qf, q0t)
        q_to_twin[i] = q_geodesic_deg(qf, q0w)
        om_to_truth[i] = omega_rel_dist(omf, om0t)
        om_to_twin[i] = omega_rel_dist(omf, om0w)
        is_truth = (q_to_truth[i] < TWIN_Q_TOL_DEG
                    and om_to_truth[i] < TWIN_OMEGA_REL_TOL)
        is_twin = (q_to_twin[i] < TWIN_Q_TOL_DEG
                   and om_to_twin[i] < TWIN_OMEGA_REL_TOL)
        if is_truth and not is_twin:
            twin_label[i] = "truth"
        elif is_twin and not is_truth:
            twin_label[i] = "twin"
        elif is_truth and is_twin:
            twin_label[i] = "truth_twin"  # truth and twin coincide for this seed
        else:
            twin_label[i] = "distinct"

    # ----------------------------------------------------------------- arrays
    np.savez(
        RESULTS_DIR / "basin_hifi.npz",
        seed=seeds, basin_class=basin_class, mode=mode_per_basin,
        q0_final_wxyz=q0_final, omega_final_rad=omega_final,
        rho_surr=rho_surr, band_surr=band_surr,
        rho_hifi=rho_hifi, band_hifi=band_hifi,
        twin_label=twin_label,
        q_to_truth_deg=q_to_truth, q_to_twin_deg=q_to_twin,
        om_to_truth_rel=om_to_truth, om_to_twin_rel=om_to_twin,
        L_dir_angle_deg=L_dir_angle, L_mag_rel_diff=L_mag_rel_diff,
        render_mask=render_mask,
    )

    # --------------------------------------------------------------- aggregates
    rendered = render_mask
    comp = basin_class == "competing_low_mse"
    band_AB_hifi = np.isin(band_hifi, ["A", "B"])

    overall = _band_counts(band_hifi[rendered])
    by_class = {c: _band_counts(band_hifi[basin_class == c])
                for c in ["truth", "competing_low_mse"]}
    by_twin = {tl: _band_counts(band_hifi[twin_label == tl])
               for tl in ["truth", "twin", "truth_twin", "distinct"]}

    # the headline class: genuinely distinct (not truth, not twin) AND hi-fi A|B
    distinct_AB = (twin_label == "distinct") & band_AB_hifi

    # surrogate vs hi-fi agreement on the rendered set
    sa = band_surr[rendered]
    ha = band_hifi[rendered]
    agree = int((sa == ha).sum())
    surr_AB_hifi_not = int((np.isin(sa, ["A", "B"])
                            & ~np.isin(ha, ["A", "B"])).sum())
    surr_not_hifi_AB = int((~np.isin(sa, ["A", "B"])
                            & np.isin(ha, ["A", "B"])).sum())

    # s079 LAM/SAM split, re-checked on the hi-fi Band A|B distinct subset
    split = {}
    for m in ["LAM", "SAM"]:
        mm = mode_per_basin == m
        seed_ids = sorted({int(s) for s in seeds[mm]})
        split[m] = {
            "n_seeds": len(seed_ids),
            "seeds": seed_ids,
            "n_rendered": int((mm & rendered).sum()),
            "n_hifi_band_AB": int((mm & band_AB_hifi).sum()),
            "n_competing_hifi_band_AB": int((mm & comp & band_AB_hifi).sum()),
            "n_distinct_hifi_band_AB": int((mm & distinct_AB).sum()),
            "distinct_hifi_AB_per_seed": {
                int(s): int((distinct_AB & (seeds == s)).sum())
                for s in seed_ids},
        }

    # per-seed dedup of genuinely-distinct hi-fi Band A|B basins
    def _dedup(idx_list) -> list:
        clusters = []
        used = set()
        for a in idx_list:
            if a in used:
                continue
            grp = [a]
            used.add(a)
            for b in idx_list:
                if b in used:
                    continue
                if (q_geodesic_deg(q0_final[a], q0_final[b]) < 5.0
                        and omega_rel_dist(omega_final[a],
                                           omega_final[b]) < 0.05):
                    grp.append(b)
                    used.add(b)
            clusters.append(grp)
        return clusters

    per_seed = {}
    for s in sorted(set(int(x) for x in seeds)):
        sm = seeds == s
        s_distinct_AB = list(np.where(sm & distinct_AB)[0])
        clusters = _dedup(s_distinct_AB)
        rep = []
        for c in sorted(clusters, key=lambda g: rho_hifi[g[0]]):
            k = c[0]
            rep.append({
                "n_ic": len(c),
                "rho_hifi": float(rho_hifi[k]),
                "band_hifi": str(band_hifi[k]),
                "rho_surr": float(rho_surr[k]),
                "q_to_truth_deg": float(q_to_truth[k]),
                "q_to_twin_deg": float(q_to_twin[k]),
                "L_dir_angle_deg": float(L_dir_angle[k]),
            })
        per_seed[f"seed_{s:03d}"] = {
            "seed": s,
            "mode": seed_mode[s],
            "n_rendered": int((sm & rendered).sum()),
            "bands_hifi": _band_counts(band_hifi[sm & rendered]),
            "n_truth_basin_hifi_A": int((sm & (twin_label == "truth")
                                         & (band_hifi == "A")).sum()),
            "n_twin_basin": int((sm & (twin_label == "twin")).sum()),
            "n_distinct_hifi_band_AB": len(s_distinct_AB),
            "n_distinct_hifi_band_AB_dedup": len(clusters),
            "distinct_basins": rep,
        }

    summary = {
        "experiment": "s081",
        "substrate": str(S077_NPZ.relative_to(SURVEY_ROOT)),
        "n_rendered": int(render_idx.size),
        "render_wall_s": wall_render,
        "render_nproc": nproc,
        "twin_q_tol_deg": TWIN_Q_TOL_DEG,
        "twin_omega_rel_tol": TWIN_OMEGA_REL_TOL,
        "band_distribution_hifi": {
            "overall_rendered": overall,
            "by_basin_class": by_class,
            "by_twin_label": by_twin,
        },
        "surrogate_vs_hifi": {
            "n_rendered": int(render_idx.size),
            "band_exact_agree": agree,
            "surrogate_AB_but_hifi_worse": surr_AB_hifi_not,
            "surrogate_worse_but_hifi_AB": surr_not_hifi_AB,
        },
        "s079_split_on_hifi_band_AB": split,
        "per_seed": per_seed,
        "caveat": ("twin/truth labels use q-geodesic < 8 deg & omega rel "
                   "< 8% tolerances; raw distances saved in basin_hifi.npz "
                   "for audit. 'distinct' = not within tol of truth OR its "
                   "body-X twin — other symmetry degeneracies not checked."),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------ figures
    band_color = {"A": "tab:green", "B": "tab:olive",
                  "C": "tab:orange", "D": "lightgray", "X": "black"}

    # Fig 1: surrogate rho vs hi-fi rho, colored by twin label
    fig, ax = plt.subplots(figsize=(7.5, 7))
    tl_color = {"truth": "tab:green", "twin": "tab:blue",
                "truth_twin": "tab:cyan", "distinct": "tab:red"}
    for tl, c in tl_color.items():
        m = (twin_label == tl) & rendered
        if m.sum():
            ax.scatter(rho_surr[m], rho_hifi[m], s=28, alpha=0.75, c=c,
                       edgecolors="k", linewidths=0.3,
                       label=f"{tl} (n={int(m.sum())})")
    lim = [0, np.nanmax(rho_hifi[rendered]) * 1.05]
    ax.plot(lim, lim, "k:", lw=1, label="surrogate = hi-fi")
    for thr in (2.0, 4.0):
        ax.axhline(thr, color="gray", ls="--", lw=0.7)
        ax.axvline(thr, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("surrogate-v2 rho (s080 proxy)")
    ax.set_ylabel("hi-fi rho (s081)")
    ax.set_title("s081 — surrogate-v2 rho vs hi-fi rho\n"
                 "points above the diagonal = surrogate over-optimistic")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = RESULTS_DIR / "s081_hifi_vs_surrogate.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # Fig 2: hi-fi band stacked bars by twin label and by regime
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    labels = ["truth", "twin", "truth_twin", "distinct"]
    bottoms = np.zeros(len(labels))
    for b in BANDS:
        vals = np.array([by_twin[tl][b] for tl in labels])
        ax.bar(labels, vals, bottom=bottoms, color=band_color[b],
               label=f"Band {b}")
        bottoms += vals
    ax.set_ylabel("basin count")
    ax.set_title("s081 — hi-fi rho-band by twin classification")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    modes = ["LAM", "SAM"]
    bd = {m: _band_counts(band_hifi[(mode_per_basin == m) & rendered])
          for m in modes}
    bottoms = np.zeros(len(modes))
    for b in BANDS:
        vals = np.array([bd[m][b] for m in modes])
        ax.bar(modes, vals, bottom=bottoms, color=band_color[b],
               label=f"Band {b}")
        bottoms += vals
    ax.set_ylabel("rendered basin count")
    ax.set_title("s081 — hi-fi rho-band by tumbling regime (rendered set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"s081 — hi-fi rho-bands, {render_idx.size} rendered basins  "
        f"(A={overall['A']} B={overall['B']} C={overall['C']} D={overall['D']})",
        fontsize=11)
    fig.tight_layout()
    f2 = RESULTS_DIR / "s081_rho_bands.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ console
    print()
    print(f"  hi-fi band distribution (rendered set, n={render_idx.size}):")
    for b in BANDS:
        print(f"    Band {b}: {overall[b]:4d}")
    print()
    print("  by twin classification:")
    for tl in labels:
        bc = by_twin[tl]
        tot = sum(bc.values())
        print(f"    {tl:12s} (n={tot:3d}): "
              f"A={bc['A']:3d} B={bc['B']:3d} C={bc['C']:3d} D={bc['D']:3d}")
    print()
    print("  truth-basin control: "
          f"{by_class['truth']['A']}/{sum(by_class['truth'].values())} "
          "truth basins came back hi-fi Band A")
    print()
    print(f"  surrogate vs hi-fi (n={render_idx.size}): "
          f"exact-band agree {agree}, "
          f"surrogate-A|B but hi-fi worse {surr_AB_hifi_not}, "
          f"surrogate-worse but hi-fi-A|B {surr_not_hifi_AB}")
    print()
    print("  s079 LAM/SAM split on the hi-fi Band A|B *distinct* subset:")
    for m in modes:
        sp = split[m]
        print(f"    {m}: {sp['n_distinct_hifi_band_AB']} distinct hi-fi-A|B "
              f"basins ({sp['n_competing_hifi_band_AB']} competing-class "
              f"hi-fi-A|B total); per seed {sp['distinct_hifi_AB_per_seed']}")
    print()
    print("  per-seed genuinely-distinct hi-fi Band A|B (deduped):")
    for s in sorted(set(int(x) for x in seeds)):
        ps = per_seed[f"seed_{s:03d}"]
        print(f"    seed {s:3d} ({ps['mode']}): "
              f"{ps['n_distinct_hifi_band_AB_dedup']} distinct basins "
              f"(from {ps['n_distinct_hifi_band_AB']} endpoints), "
              f"n_twin={ps['n_twin_basin']}")
        for rb in ps["distinct_basins"]:
            print(f"        rho_hifi={rb['rho_hifi']:6.2f} ({rb['band_hifi']})  "
                  f"rho_surr={rb['rho_surr']:5.2f}  "
                  f"q_to_truth={rb['q_to_truth_deg']:6.1f}d  "
                  f"q_to_twin={rb['q_to_twin_deg']:6.1f}d  "
                  f"L_dir={rb['L_dir_angle_deg']:6.1f}d")
    print()
    print(f"Saved: {RESULTS_DIR / 'summary.json'}")
    print(f"Saved: {RESULTS_DIR / 'basin_hifi.npz'}")
    print(f"Saved: {f1}")
    print(f"Saved: {f2}")
    print(f"Total wall: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
