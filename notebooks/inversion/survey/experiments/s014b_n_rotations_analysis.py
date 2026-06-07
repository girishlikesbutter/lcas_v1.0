"""s014b — cohort-scale n_rotations / PA correlation with ρ-min and
multi-solution candidate counts (cached-data analysis).

Combines two cheap analyses on already-rendered data, recommended in
PROGRESS.md after s014 confirmed cohort architecture trust:

  (1) Mechanism verification — low-rotation under-determination.
      Per-seed (n_rotations, ρ-min) on the 10 s011 pilot seeds (9
      recoverable + seed 10). Tests whether seeds with sparse rotational
      coverage have higher ρ-min, consistent with the s007/s008/s012a/s013
      hypothesis that LC under-determination scales with low rotation
      count.

  (2) Multi-solution cluster classification.
      The 17 ρ<4 candidates s014 found outside in-basin span 4 seeds
      (28, 41, 48, 84). Greedy 5°-geodesic clustering on q0_final →
      distinct attractors per seed. Each cluster classified into the
      three structural classes from the s014 wind-down:
        class 1 — near-truth-q0 + off-truth-ω
        class 2 — q0 ∈ [30°, 150°] + near-truth ω
        class 3 — q0 ≥ 170° (near-180° flip) + near-truth ω
      Cross-tabulated against n_rotations + mean PA. Includes seed
      10's ρ=3.90 cluster from s013 (cohort B) for completeness.

Inputs:
  results/s011/runs.npz
  results/s013/rho_in_basin.npz, rho_seed10.npz
  results/s014/rho_s011_nb.npz, rho_s005.npz, analysis_summary.json
  data/trajectories/traj_seedXXX.npz (via lib.traj_load)

Outputs:
  results/s014b/seed_metadata.npz   — n_rotations / mean_PA / ρ-min /
                                       ms_count for all 100 cohort seeds
  results/s014b/clusters.json       — per-seed multi-solution clusters +
                                       structural classes
  results/s014b/summary.json        — correlations + key findings
  results/s014b/n_rotations_vs_rho.png
  results/s014b/pa_vs_rho.png
  results/s014b/multi_solution_breakdown.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

from lib.traj_load import load_truth, list_seeds  # noqa: E402

OUT_DIR = SURVEY_DIR / "results" / "s014b"
S011_DIR = SURVEY_DIR / "results" / "s011"
S013_DIR = SURVEY_DIR / "results" / "s013"
S014_DIR = SURVEY_DIR / "results" / "s014"

PILOT_SEEDS = [6, 10, 21, 28, 41, 44, 48, 60, 84, 91]
RHO_BAND_BAR = 4.0  # Band A ∪ B
GEO_CLUSTER_DEG = 5.0  # greedy clustering threshold (matches s013)


# --------------------------- numerics helpers ---------------------------

def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    return float(np.corrcoef(_rankdata(x[finite]), _rankdata(y[finite]))[0, 1])


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _q_norm(q):
    return q / np.linalg.norm(q)


def geodesic_deg(q1_wxyz, q2_wxyz):
    """Geodesic distance on SO(3) in degrees, via |w| of the relative quat."""
    q1 = _q_norm(q1_wxyz)
    q2 = _q_norm(q2_wxyz)
    dot = float(abs(np.dot(q1, q2)))
    dot = min(1.0, max(0.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def greedy_cluster_q0(q0_wxyz_arr, threshold_deg):
    """Return cluster_id per row, one centroid per cluster (first member)."""
    n = len(q0_wxyz_arr)
    cluster_id = np.full(n, -1, dtype=np.int64)
    centroids = []
    for i in range(n):
        assigned = -1
        for c_idx, c_q in enumerate(centroids):
            if geodesic_deg(q0_wxyz_arr[i], c_q) < threshold_deg:
                assigned = c_idx
                break
        if assigned < 0:
            cluster_id[i] = len(centroids)
            centroids.append(q0_wxyz_arr[i])
        else:
            cluster_id[i] = assigned
    return cluster_id, centroids


def classify_cluster(q0_err, omega_dir_err, omega_mag_err_pct):
    """Return 'class_1' / 'class_2' / 'class_3' / 'unclassified'.

    Convention matches the s014 writeup:
      class_1 — near-truth-q0 (q0_err < 10°), ω outside STRICT basin
                (|ωd|≥1° OR |ωm|≥5%). Includes seed 28 q0=2.1°/ωd=2.0°.
                Often a "strict basin definition was conservative" case.
      class_2 — q0 ∈ [30°, 150°], ω near truth (|ωd|<5°, |ωm|<5%).
                LC under-determination class.
      class_3 — q0 ≥ 170° (near-180° flip), ω near truth (|ωd|<5°, |ωm|<5%).
                Near-flip-degeneracy class (NOT pure body-twin, twin
                recoveries 0/640+50 in s011/s005).
    """
    in_strict_basin_omega = (abs(omega_dir_err) < 1.0) and (abs(omega_mag_err_pct) < 5.0)
    near_truth_omega_loose = (abs(omega_dir_err) < 5.0) and (abs(omega_mag_err_pct) < 5.0)
    if q0_err < 10.0 and not in_strict_basin_omega:
        return "class_1"
    if 30.0 <= q0_err <= 150.0 and near_truth_omega_loose:
        return "class_2"
    if q0_err >= 170.0 and near_truth_omega_loose:
        return "class_3"
    return "unclassified"


# --------------------------- main analysis ---------------------------


def per_seed_metadata(seeds):
    """Compute n_rotations, mean PA from cached truth NPZs."""
    rows = []
    for sd in seeds:
        d = load_truth(sd)
        dur_s = float(d["observation_times"][-1] - d["observation_times"][0])
        omg_dps = float(d["omega_mag_dps"])
        n_rot = omg_dps * dur_s / 360.0
        pa = d["phase_angle_3d"]
        rows.append({
            "seed": int(sd),
            "n_rotations": n_rot,
            "omega_mag_dps": omg_dps,
            "duration_s": dur_s,
            "mean_pa_deg": float(np.mean(pa)),
            "min_pa_deg": float(np.min(pa)),
            "max_pa_deg": float(np.max(pa)),
        })
    return rows


def per_seed_rho_min(pilot_seeds):
    """Compute min ρ across all hi-fi-rendered ICs per pilot seed.

    For 9 recoverable seeds (6/21/28/41/44/48/60/84/91): merge s013
    in-basin + s014 non-basin (64 ICs each).
    For seed 10: from s013 cohort B (256 ICs).
    """
    a_ib = np.load(S013_DIR / "rho_in_basin.npz", allow_pickle=True)
    a_nb = np.load(S014_DIR / "rho_s011_nb.npz", allow_pickle=True)
    s10 = np.load(S013_DIR / "rho_seed10.npz", allow_pickle=True)

    out = {}
    for sd in pilot_seeds:
        if sd == 10:
            rho = s10["rho"]
            finite = np.isfinite(rho)
            n_finite = int(finite.sum())
            n_total = int(len(rho))
            rho_min = float(np.min(rho[finite])) if n_finite else float("nan")
            out[sd] = {
                "rho_min": rho_min,
                "rho_p10": float(np.percentile(rho[finite], 10)) if n_finite else float("nan"),
                "rho_median": float(np.median(rho[finite])) if n_finite else float("nan"),
                "n_total": n_total,
                "n_finite": n_finite,
                "source": "s013_seed10_n256",
            }
        else:
            ib_mask = (a_ib["seed"] == sd)
            nb_mask = (a_nb["seed"] == sd)
            rho_combined = np.concatenate([a_ib["rho"][ib_mask], a_nb["rho"][nb_mask]])
            finite = np.isfinite(rho_combined)
            n_finite = int(finite.sum())
            n_total = int(len(rho_combined))
            rho_min = float(np.min(rho_combined[finite])) if n_finite else float("nan")
            out[sd] = {
                "rho_min": rho_min,
                "rho_p10": float(np.percentile(rho_combined[finite], 10)) if n_finite else float("nan"),
                "rho_median": float(np.median(rho_combined[finite])) if n_finite else float("nan"),
                "n_total": n_total,
                "n_finite": n_finite,
                "source": "s013_in_basin + s014_non_basin",
            }
    return out


def collect_multisolution_candidates(seeds_with_ms):
    """Pull all Band A∪B candidates outside in-basin from cached NPZs.

    Includes:
      - s011 non-basin (s014 cohort A, 16 candidates across 4 seeds)
      - s005 non-basin (s014 cohort B, 1 candidate cross-confirming seed 41)
      - s013 cohort B (seed 10 N=256, 2 candidates ρ=3.90)

    Returns list of dicts with q0_final_wxyz attached for clustering.
    """
    out = []

    a_nb = np.load(S014_DIR / "rho_s011_nb.npz", allow_pickle=True)
    msol_mask = (a_nb["rho"] < RHO_BAND_BAR)
    for i in np.where(msol_mask)[0]:
        out.append({
            "cohort": "s011_non_basin",
            "seed": int(a_nb["seed"][i]),
            "ic_idx": int(a_nb["ic_idx"][i]),
            "rho": float(a_nb["rho"][i]),
            "band": str(a_nb["band"][i]),
            "q0_err_deg": float(a_nb["q0_err_deg"][i]),
            "omega_dir_err_deg": float(a_nb["omega_dir_err_deg"][i]),
            "omega_mag_err_pct": float(a_nb["omega_mag_err_pct"][i]),
            "q0_final_wxyz": a_nb["q0_final_wxyz"][i].tolist(),
            "omega_final_rad": a_nb["omega_final_rad"][i].tolist(),
        })

    b = np.load(S014_DIR / "rho_s005.npz", allow_pickle=True)
    msol_mask = (b["rho"] < RHO_BAND_BAR) & (~b["truth_basin_strict"])
    for i in np.where(msol_mask)[0]:
        out.append({
            "cohort": "s005_non_basin",
            "seed": int(b["seed"][i]),
            "ic_idx": int(b["ic_idx"][i]),
            "rho": float(b["rho"][i]),
            "band": str(b["band"][i]),
            "q0_err_deg": float(b["q0_err_deg"][i]),
            "omega_dir_err_deg": float(b["omega_dir_err_deg"][i]),
            "omega_mag_err_pct": float(b["omega_mag_err_pct"][i]),
            "q0_final_wxyz": b["q0_final_wxyz"][i].tolist(),
            "omega_final_rad": b["omega_final_rad"][i].tolist(),
        })

    s10 = np.load(S013_DIR / "rho_seed10.npz", allow_pickle=True)
    msol_mask = (s10["rho"] < RHO_BAND_BAR) & (~s10["truth_basin_strict"])
    for i in np.where(msol_mask)[0]:
        out.append({
            "cohort": "s013_seed10_n256",
            "seed": int(s10["seed"][i]),
            "ic_idx": int(s10["ic_idx"][i]),
            "rho": float(s10["rho"][i]),
            "band": str(s10["band"][i]),
            "q0_err_deg": float(s10["q0_err_deg"][i]),
            "omega_dir_err_deg": float(s10["omega_dir_err_deg"][i]),
            "omega_mag_err_pct": float(s10["omega_mag_err_pct"][i]),
            "q0_final_wxyz": s10["q0_final_wxyz"][i].tolist(),
            "omega_final_rad": s10["omega_final_rad"][i].tolist(),
        })

    return out


def cluster_multisolution_per_seed(candidates):
    """Greedy 5°-geodesic cluster on q0_final, separately per seed.

    Returns a dict { seed -> [cluster_dict, ...] } where each cluster_dict
    has its members + assigned structural class.
    """
    by_seed = {}
    for c in candidates:
        by_seed.setdefault(c["seed"], []).append(c)

    out = {}
    for seed, cs in by_seed.items():
        q_arr = np.array([c["q0_final_wxyz"] for c in cs])
        cluster_ids, centroids = greedy_cluster_q0(q_arr, GEO_CLUSTER_DEG)
        clusters = []
        for cid in range(len(centroids)):
            members = [cs[i] for i in range(len(cs)) if cluster_ids[i] == cid]
            best = min(members, key=lambda m: m["rho"])
            cls = classify_cluster(
                q0_err=best["q0_err_deg"],
                omega_dir_err=best["omega_dir_err_deg"],
                omega_mag_err_pct=best["omega_mag_err_pct"],
            )
            clusters.append({
                "cluster_id": cid,
                "n_members": len(members),
                "rho_min": min(m["rho"] for m in members),
                "rho_max": max(m["rho"] for m in members),
                "best_q0_err_deg": best["q0_err_deg"],
                "best_omega_dir_err_deg": best["omega_dir_err_deg"],
                "best_omega_mag_err_pct": best["omega_mag_err_pct"],
                "structural_class": cls,
                "cohorts": sorted(set(m["cohort"] for m in members)),
                "member_ic_idx": [(m["cohort"], m["ic_idx"]) for m in members],
            })
        clusters.sort(key=lambda c: c["rho_min"])
        out[int(seed)] = clusters
    return out


# --------------------------- plotting ---------------------------


def plot_n_rotations_vs_rho(meta_by_seed, rho_by_seed, ms_count_by_seed, path):
    seeds = PILOT_SEEDS
    n_rot = np.array([meta_by_seed[s]["n_rotations"] for s in seeds])
    rho_min = np.array([rho_by_seed[s]["rho_min"] for s in seeds])
    ms_count = np.array([ms_count_by_seed.get(s, 0) for s in seeds])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(n_rot, rho_min, c=ms_count, s=120, cmap="plasma",
                    edgecolor="k", vmin=0, vmax=max(1, ms_count.max()))
    for s, x, y in zip(seeds, n_rot, rho_min):
        ax.annotate(f"{s}", (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=9)
    ax.axhline(2.0, color="green", ls="--", lw=0.8, label="ρ=2 (Band A bar)")
    ax.axhline(4.0, color="orange", ls="--", lw=0.8, label="ρ=4 (Band B bar)")
    ax.set_xlabel("n_rotations in 1-h window")
    ax.set_ylabel("ρ-min (best hi-fi candidate)")
    ax.set_yscale("log")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("multi-solution candidate count (ρ<4 outside basin)")
    sp = spearman(n_rot, rho_min)
    pe = pearson(np.log10(n_rot), np.log10(rho_min))
    sp_ms = spearman(n_rot, ms_count)
    ax.set_title(
        f"n_rotations vs ρ-min (s011 pilot, n=10)\n"
        f"Spearman(n_rot, ρ_min) = {sp:.3f}  |  "
        f"Pearson(log, log) = {pe:.3f}  |  "
        f"Spearman(n_rot, ms_count) = {sp_ms:.3f}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pa_vs_rho(meta_by_seed, rho_by_seed, ms_count_by_seed, path):
    seeds = PILOT_SEEDS
    pa = np.array([meta_by_seed[s]["mean_pa_deg"] for s in seeds])
    rho_min = np.array([rho_by_seed[s]["rho_min"] for s in seeds])
    ms_count = np.array([ms_count_by_seed.get(s, 0) for s in seeds])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(pa, rho_min, c=ms_count, s=120, cmap="plasma",
                    edgecolor="k", vmin=0, vmax=max(1, ms_count.max()))
    for s, x, y in zip(seeds, pa, rho_min):
        ax.annotate(f"{s}", (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=9)
    ax.axhline(2.0, color="green", ls="--", lw=0.8, label="ρ=2 (Band A bar)")
    ax.axhline(4.0, color="orange", ls="--", lw=0.8, label="ρ=4 (Band B bar)")
    ax.set_xlabel("mean phase angle (deg)")
    ax.set_ylabel("ρ-min (best hi-fi candidate)")
    ax.set_yscale("log")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("multi-solution candidate count")
    sp = spearman(pa, rho_min)
    sp_ms = spearman(pa, ms_count)
    ax.set_title(
        f"mean PA vs ρ-min (s011 pilot, n=10)\n"
        f"Spearman(PA, ρ_min) = {sp:.3f}  |  "
        f"Spearman(PA, ms_count) = {sp_ms:.3f}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_multi_solution_breakdown(clusters_by_seed, meta_by_seed, path):
    """Stacked bar: one bar per seed-with-ms, segments coloured by class."""
    classes = ["class_1", "class_2", "class_3", "unclassified"]
    class_colors = {
        "class_1": "#1f77b4",
        "class_2": "#ff7f0e",
        "class_3": "#2ca02c",
        "unclassified": "#888888",
    }

    seeds = sorted(clusters_by_seed.keys())
    counts_by_class = {c: [] for c in classes}
    n_rot_by_seed = []
    pa_by_seed = []
    for sd in seeds:
        cluster_list = clusters_by_seed[sd]
        for c in classes:
            counts_by_class[c].append(sum(1 for cl in cluster_list if cl["structural_class"] == c))
        n_rot_by_seed.append(meta_by_seed[sd]["n_rotations"])
        pa_by_seed.append(meta_by_seed[sd]["mean_pa_deg"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(seeds))
    bottom = np.zeros(len(seeds))
    for c in classes:
        h = np.array(counts_by_class[c])
        if h.sum() == 0:
            continue
        ax.bar(x, h, bottom=bottom, label=c, color=class_colors[c], edgecolor="k")
        bottom = bottom + h

    labels = [
        f"seed {s}\nn_rot={n_rot_by_seed[i]:.2f}\nPA={pa_by_seed[i]:.1f}°"
        for i, s in enumerate(seeds)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("distinct multi-solution attractors (ρ<4, 5° greedy clust)")
    ax.set_title(
        "Multi-solution attractor structural classes per seed\n"
        "(s014 cohort A non-basin + s005 cohort B + s013 seed10 cohort B)",
        fontsize=11,
    )
    ax.legend(title="structural class", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------- driver ---------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading per-seed metadata for all 100 cohort seeds ...", flush=True)
    all_seeds = list_seeds()
    cohort_meta = per_seed_metadata(all_seeds)
    cohort_n_rot = np.array([m["n_rotations"] for m in cohort_meta])
    cohort_pa = np.array([m["mean_pa_deg"] for m in cohort_meta])

    print(f"Cohort n_rotations: median={np.median(cohort_n_rot):.2f}, "
          f"p10={np.percentile(cohort_n_rot, 10):.2f}, "
          f"p90={np.percentile(cohort_n_rot, 90):.2f}, "
          f"min={cohort_n_rot.min():.2f}, max={cohort_n_rot.max():.2f}")
    print(f"Cohort mean PA: median={np.median(cohort_pa):.2f}, "
          f"p10={np.percentile(cohort_pa, 10):.2f}, "
          f"p90={np.percentile(cohort_pa, 90):.2f}")

    n_low_rot = (cohort_n_rot < 2.0).sum()
    n_mid_rot = ((cohort_n_rot >= 2.0) & (cohort_n_rot < 5.0)).sum()
    n_high_rot = (cohort_n_rot >= 5.0).sum()
    print(f"Cohort rotation bands: low (<2): {n_low_rot}, mid [2-5): {n_mid_rot}, "
          f"high (≥5): {n_high_rot}")

    meta_by_seed = {m["seed"]: m for m in cohort_meta}

    print()
    print("Computing per-seed ρ-min on s011 pilot (10 seeds) ...", flush=True)
    rho_by_seed = per_seed_rho_min(PILOT_SEEDS)
    print()
    print(f"  {'seed':>4} {'n_rot':>6} {'mean_PA':>8} {'rho_min':>9} {'rho_p10':>9} "
          f"{'n_fin':>6} {'source':<35}")
    for sd in PILOT_SEEDS:
        m = meta_by_seed[sd]
        r = rho_by_seed[sd]
        print(f"  {sd:>4} {m['n_rotations']:>6.2f} {m['mean_pa_deg']:>8.2f} "
              f"{r['rho_min']:>9.3f} {r['rho_p10']:>9.3f} "
              f"{r['n_finite']:>4}/{r['n_total']:<3} {r['source']:<35}")

    print()
    print("Collecting multi-solution candidates (ρ<4 outside in-basin) ...", flush=True)
    candidates = collect_multisolution_candidates([])
    print(f"  total candidates: {len(candidates)} across "
          f"{len(set(c['seed'] for c in candidates))} seeds")
    by_cohort = {}
    for c in candidates:
        by_cohort.setdefault(c["cohort"], []).append(c)
    for k, v in by_cohort.items():
        print(f"    {k}: {len(v)} candidates, seeds {sorted(set(c['seed'] for c in v))}")

    clusters_by_seed = cluster_multisolution_per_seed(candidates)
    ms_count_by_seed = {sd: len(cls) for sd, cls in clusters_by_seed.items()}

    print()
    print("Per-seed multi-solution clustering (5° geodesic on q0):")
    for sd in sorted(clusters_by_seed):
        m = meta_by_seed[sd]
        cls_counts = {}
        for cl in clusters_by_seed[sd]:
            cls_counts[cl["structural_class"]] = cls_counts.get(cl["structural_class"], 0) + 1
        print(f"  seed {sd:>3}  n_rot={m['n_rotations']:>5.2f}  "
              f"PA={m['mean_pa_deg']:>5.1f}°  "
              f"n_clusters={len(clusters_by_seed[sd])}  classes={cls_counts}")
        for cl in clusters_by_seed[sd]:
            print(f"      cl={cl['cluster_id']} ρ_min={cl['rho_min']:.2f} "
                  f"(n={cl['n_members']}) "
                  f"q0_err={cl['best_q0_err_deg']:.1f}° "
                  f"ωd={cl['best_omega_dir_err_deg']:.1f}° "
                  f"ωm={cl['best_omega_mag_err_pct']:+.1f}%  "
                  f"class={cl['structural_class']}  cohorts={cl['cohorts']}")

    # ------- correlations on the 10 pilot seeds -------
    n_rot_pilot = np.array([meta_by_seed[s]["n_rotations"] for s in PILOT_SEEDS])
    pa_pilot = np.array([meta_by_seed[s]["mean_pa_deg"] for s in PILOT_SEEDS])
    rho_pilot = np.array([rho_by_seed[s]["rho_min"] for s in PILOT_SEEDS])
    msc_pilot = np.array([ms_count_by_seed.get(s, 0) for s in PILOT_SEEDS])

    # Correlations on full pilot (n=10) AND with seed 10 dropped (n=9 recoverable)
    seed10_idx = PILOT_SEEDS.index(10)
    keep_mask = np.ones(len(PILOT_SEEDS), dtype=bool)
    keep_mask[seed10_idx] = False
    corrs = {
        "spearman_n_rot_vs_rho_min__n10": spearman(n_rot_pilot, rho_pilot),
        "pearson_log_n_rot_vs_log_rho_min__n10": pearson(np.log10(n_rot_pilot), np.log10(rho_pilot)),
        "spearman_n_rot_vs_ms_count__n10": spearman(n_rot_pilot, msc_pilot),
        "spearman_pa_vs_rho_min__n10": spearman(pa_pilot, rho_pilot),
        "spearman_pa_vs_ms_count__n10": spearman(pa_pilot, msc_pilot),
        "spearman_omega_mag_dps_vs_rho_min__n10": spearman(
            np.array([meta_by_seed[s]["omega_mag_dps"] for s in PILOT_SEEDS]),
            rho_pilot,
        ),
        "spearman_n_rot_vs_rho_min__n9_no_seed10": spearman(n_rot_pilot[keep_mask], rho_pilot[keep_mask]),
        "pearson_log_n_rot_vs_log_rho_min__n9_no_seed10": pearson(
            np.log10(n_rot_pilot[keep_mask]), np.log10(rho_pilot[keep_mask])
        ),
        "spearman_n_rot_vs_ms_count__n9_no_seed10": spearman(n_rot_pilot[keep_mask], msc_pilot[keep_mask]),
        "spearman_pa_vs_rho_min__n9_no_seed10": spearman(pa_pilot[keep_mask], rho_pilot[keep_mask]),
        "spearman_pa_vs_ms_count__n9_no_seed10": spearman(pa_pilot[keep_mask], msc_pilot[keep_mask]),
    }

    print()
    print("Correlations (n=10 s011 pilot seeds):")
    for k, v in corrs.items():
        print(f"  {k:<40} {v:+.3f}")

    # ------- plots -------
    p1 = OUT_DIR / "n_rotations_vs_rho.png"
    plot_n_rotations_vs_rho(meta_by_seed, rho_by_seed, ms_count_by_seed, p1)
    print(f"Saved: {p1}")

    p2 = OUT_DIR / "pa_vs_rho.png"
    plot_pa_vs_rho(meta_by_seed, rho_by_seed, ms_count_by_seed, p2)
    print(f"Saved: {p2}")

    p3 = OUT_DIR / "multi_solution_breakdown.png"
    plot_multi_solution_breakdown(clusters_by_seed, meta_by_seed, p3)
    print(f"Saved: {p3}")

    # ------- save NPZ + JSON -------
    npz_path = OUT_DIR / "seed_metadata.npz"
    seed_arr = np.array([m["seed"] for m in cohort_meta])
    np.savez(
        npz_path,
        seed=seed_arr,
        n_rotations=cohort_n_rot,
        omega_mag_dps=np.array([m["omega_mag_dps"] for m in cohort_meta]),
        mean_pa_deg=cohort_pa,
        min_pa_deg=np.array([m["min_pa_deg"] for m in cohort_meta]),
        max_pa_deg=np.array([m["max_pa_deg"] for m in cohort_meta]),
        pilot_seeds=np.array(PILOT_SEEDS),
        pilot_rho_min=rho_pilot,
        pilot_ms_count=msc_pilot,
    )
    print(f"Saved: {npz_path}")

    clusters_json_path = OUT_DIR / "clusters.json"
    with open(clusters_json_path, "w") as f:
        json.dump({
            "rho_band_bar": RHO_BAND_BAR,
            "geo_cluster_deg": GEO_CLUSTER_DEG,
            "n_candidates_total": len(candidates),
            "clusters_by_seed": {str(sd): cls for sd, cls in clusters_by_seed.items()},
        }, f, indent=2)
    print(f"Saved: {clusters_json_path}")

    summary = {
        "pilot_seeds": PILOT_SEEDS,
        "cohort_n_rotations_stats": {
            "median": float(np.median(cohort_n_rot)),
            "p10": float(np.percentile(cohort_n_rot, 10)),
            "p90": float(np.percentile(cohort_n_rot, 90)),
            "min": float(cohort_n_rot.min()),
            "max": float(cohort_n_rot.max()),
            "n_low_rot_lt_2": int(n_low_rot),
            "n_mid_rot_2_to_5": int(n_mid_rot),
            "n_high_rot_ge_5": int(n_high_rot),
        },
        "cohort_pa_stats": {
            "median": float(np.median(cohort_pa)),
            "p10": float(np.percentile(cohort_pa, 10)),
            "p90": float(np.percentile(cohort_pa, 90)),
        },
        "pilot_table": [
            {
                "seed": int(s),
                "n_rotations": meta_by_seed[s]["n_rotations"],
                "mean_pa_deg": meta_by_seed[s]["mean_pa_deg"],
                "rho_min": rho_by_seed[s]["rho_min"],
                "rho_p10": rho_by_seed[s]["rho_p10"],
                "rho_median": rho_by_seed[s]["rho_median"],
                "ms_count": int(ms_count_by_seed.get(s, 0)),
                "ms_classes": sorted(set(
                    cl["structural_class"] for cl in clusters_by_seed.get(s, [])
                )),
                "n_finite_hifi": rho_by_seed[s]["n_finite"],
                "n_total_hifi": rho_by_seed[s]["n_total"],
            }
            for s in PILOT_SEEDS
        ],
        "correlations": corrs,
        "multi_solution_summary": {
            "n_seeds_with_ms": int(len(clusters_by_seed)),
            "seeds": sorted(int(s) for s in clusters_by_seed.keys()),
            "n_clusters_total": int(sum(len(v) for v in clusters_by_seed.values())),
            "by_class": {
                "class_1": int(sum(
                    1 for cls in clusters_by_seed.values() for cl in cls
                    if cl["structural_class"] == "class_1"
                )),
                "class_2": int(sum(
                    1 for cls in clusters_by_seed.values() for cl in cls
                    if cl["structural_class"] == "class_2"
                )),
                "class_3": int(sum(
                    1 for cls in clusters_by_seed.values() for cl in cls
                    if cl["structural_class"] == "class_3"
                )),
                "unclassified": int(sum(
                    1 for cls in clusters_by_seed.values() for cl in cls
                    if cl["structural_class"] == "unclassified"
                )),
            },
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
