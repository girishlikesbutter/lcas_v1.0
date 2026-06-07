"""s027 — Filter-conditioned cluster topology (post-processing on s021).

After filtering (alignment AND geo at calibrated thresholds), what does
the surviving candidate set look like in (q0, ω) space?
  - One cluster around truth → filter localises.
  - Two clusters (truth + body-twin) → filter respects symmetry.
  - N clusters → multi-solution structure surfaces naturally post-filter.
  - Amorphous / many tiny clusters → filter thins but doesn't localise.

Greedy clustering: 6-D distance with q0 geodesic + ω-dir-angle + ω-mag-pct.
A candidate joins a cluster if (q0_geodesic < 5° AND ω-dir < 5° AND
ω-mag-pct < 10%) of any existing cluster member; else starts a new cluster.

Output:
  results/s027/clusters.json
  results/s027/cluster_topology.png
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib.traj_load import load_truth                # noqa: E402
from lib.forward import quat_geodesic_deg            # noqa: E402

S021 = SURVEY_DIR / "results" / "s021" / "score_distributions.npz"
RESULTS = SURVEY_DIR / "results" / "s027"
RESULTS.mkdir(parents=True, exist_ok=True)


def cluster_candidates(q0s, omegas, q_thresh_deg=5.0, w_thresh_deg=5.0,
                       wmag_thresh_pct=10.0):
    """Greedy single-link clustering in 6-D."""
    if q0s.shape[0] == 0:
        return []
    clusters = [[0]]  # list of lists of indices
    cluster_reps = [(q0s[0], omegas[0])]  # first member as representative

    for i in range(1, q0s.shape[0]):
        q = q0s[i]
        w = omegas[i]
        wmag = np.linalg.norm(w)
        joined = False
        for ci, (q_rep, w_rep) in enumerate(cluster_reps):
            dq = quat_geodesic_deg(q, q_rep)
            wmag_rep = np.linalg.norm(w_rep)
            if wmag < 1e-9 or wmag_rep < 1e-9:
                wd = 0.0
            else:
                cos_w = np.clip(np.dot(w / wmag, w_rep / wmag_rep), -1, 1)
                wd = np.degrees(np.arccos(cos_w))
            wmag_pct = abs(wmag - wmag_rep) / wmag_rep * 100 if wmag_rep > 1e-9 else 0.0
            if dq < q_thresh_deg and wd < w_thresh_deg and wmag_pct < wmag_thresh_pct:
                clusters[ci].append(i)
                joined = True
                break
        if not joined:
            clusters.append([i])
            cluster_reps.append((q, w))
    return clusters


def main():
    print("=" * 72)
    print("s027 — Filter-conditioned cluster topology")
    print("=" * 72)
    if not S021.exists():
        print(f"FATAL: {S021} not found.")
        sys.exit(1)

    d = np.load(S021)
    seeds = d["seeds"]
    rand_align = d["rand_align"]
    rand_geo = d["rand_geo"]
    rand_q0s = d["random_q0s"]      # (N_RAND, 4)
    rand_omegas = d["random_omegas"]  # (N_RAND, 3)
    truth_align = d["truth_align"]
    truth_geo = d["truth_geo"]
    n_random = int(d["n_random"])

    n_seeds = seeds.shape[0]
    per_seed = {}
    cluster_counts = []

    for i in range(n_seeds):
        seed = int(seeds[i])
        ra = rand_align[i]
        rg = rand_geo[i]

        # Filter pass: alignment >= truth_align (treats truth as the floor)
        # AND geo >= truth_geo (or no constraint if truth_geo is NaN)
        thresh_a = truth_align[i] if np.isfinite(truth_align[i]) else 0.999
        if np.isfinite(truth_geo[i]):
            thresh_g = truth_geo[i]
            pass_mask = (ra >= thresh_a) & (rg >= thresh_g)
        else:
            pass_mask = (ra >= thresh_a)

        survivors = np.where(pass_mask)[0]
        n_surv = int(survivors.size)

        if n_surv == 0:
            per_seed[seed] = {
                "n_survivors": 0,
                "n_clusters": 0,
                "max_cluster_size": 0,
                "thresh_align": float(thresh_a),
                "thresh_geo": float(thresh_g) if np.isfinite(truth_geo[i]) else None,
            }
            cluster_counts.append(0)
            continue

        # Cluster the survivors in 6-D
        q_surv = rand_q0s[survivors]
        w_surv = rand_omegas[survivors]
        clusters = cluster_candidates(q_surv, w_surv)
        cluster_counts.append(len(clusters))
        # Largest-cluster size
        max_size = max(len(c) for c in clusters) if clusters else 0

        # For each cluster: distance from cluster representative to truth + twin
        truth = load_truth(seed)
        q0_truth = truth["q0_wxyz"]
        omega_truth = truth["omega0_rad"]
        # Twin
        import quaternion as q_pkg
        q0t_q = q_pkg.quaternion(*q0_truth)
        q_180x = q_pkg.quaternion(0, 1, 0, 0)
        q_twin = q_180x * q0t_q
        q0_twin = np.array([q_twin.w, q_twin.x, q_twin.y, q_twin.z])
        R_180x = np.diag([1.0, -1.0, -1.0])
        omega_twin = R_180x @ omega_truth

        cluster_classifications = []
        for c in clusters:
            rep = c[0]
            q_rep = q_surv[rep]
            w_rep = w_surv[rep]
            dq_truth = quat_geodesic_deg(q_rep, q0_truth)
            dq_twin = quat_geodesic_deg(q_rep, q0_twin)
            wmag_rep = np.linalg.norm(w_rep)
            wmag_truth = np.linalg.norm(omega_truth)
            wmag_pct = abs(wmag_rep - wmag_truth) / wmag_truth * 100
            if wmag_rep > 1e-9:
                wd_truth = np.degrees(np.arccos(np.clip(np.dot(
                    w_rep / wmag_rep, omega_truth / wmag_truth), -1, 1)))
                wd_twin = np.degrees(np.arccos(np.clip(np.dot(
                    w_rep / wmag_rep, omega_twin / np.linalg.norm(omega_twin)), -1, 1)))
            else:
                wd_truth = wd_twin = 180.0
            label = "other"
            if dq_truth < 10 and wd_truth < 5 and wmag_pct < 10:
                label = "near_truth"
            elif dq_twin < 10 and wd_twin < 5 and wmag_pct < 10:
                label = "near_twin"
            cluster_classifications.append({
                "size": len(c),
                "label": label,
                "dq_truth_deg": float(dq_truth),
                "dq_twin_deg": float(dq_twin),
                "wd_truth_deg": float(wd_truth),
                "wd_twin_deg": float(wd_twin),
                "wmag_pct_off_truth": float(wmag_pct),
            })

        per_seed[seed] = {
            "n_survivors": n_surv,
            "n_clusters": len(clusters),
            "max_cluster_size": max_size,
            "thresh_align": float(thresh_a),
            "thresh_geo": float(thresh_g) if np.isfinite(truth_geo[i]) else None,
            "clusters": cluster_classifications,
        }

    # Aggregate
    cluster_counts_arr = np.array(cluster_counts)
    n_seeds_with_survivors = int(np.sum(cluster_counts_arr > 0))
    n_seeds_zero_survivors = int(np.sum(cluster_counts_arr == 0))
    summary = {
        "n_seeds": int(n_seeds),
        "n_random_per_seed": n_random,
        "filter_intersection_at_truth_threshold": True,
        "n_seeds_with_zero_survivors": n_seeds_zero_survivors,
        "n_seeds_with_survivors": n_seeds_with_survivors,
        "cluster_count_stats": {
            "min": int(cluster_counts_arr.min()),
            "p25": float(np.percentile(cluster_counts_arr, 25)),
            "median": float(np.median(cluster_counts_arr)),
            "p75": float(np.percentile(cluster_counts_arr, 75)),
            "p90": float(np.percentile(cluster_counts_arr, 90)),
            "max": int(cluster_counts_arr.max()),
            "mean": float(np.mean(cluster_counts_arr)),
        },
        "per_seed": per_seed,
    }
    with open(RESULTS / "clusters.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {RESULTS / 'clusters.json'}")

    # Plot: histogram of cluster counts; scatter survivor count vs cluster count
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].hist(cluster_counts_arr, bins=np.arange(cluster_counts_arr.max() + 2) - 0.5,
                color="C0", edgecolor="k")
    axs[0].set_xlabel("number of distinct clusters in survivor set")
    axs[0].set_ylabel("seeds")
    axs[0].set_title(f"cluster count per seed (n_seeds={n_seeds}, "
                     f"n_seeds_no_survivor={n_seeds_zero_survivors})")
    n_surv = np.array([per_seed[int(s)]["n_survivors"] for s in seeds])
    axs[1].scatter(n_surv, cluster_counts_arr, alpha=0.6)
    axs[1].set_xlabel("# survivors")
    axs[1].set_ylabel("# clusters")
    axs[1].set_xscale("symlog")
    axs[1].set_title("survivors vs clusters (one dot per seed)")
    fig.tight_layout()
    fig.savefig(RESULTS / "cluster_topology.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'cluster_topology.png'}")

    # Print headline
    print()
    print("=" * 72)
    print(f"  Seeds with zero survivors: {n_seeds_zero_survivors}/{n_seeds}")
    print(f"  Seeds with at least 1 survivor: {n_seeds_with_survivors}/{n_seeds}")
    print(f"  Cluster count: median={np.median(cluster_counts_arr):.0f}, "
          f"max={cluster_counts_arr.max()}, mean={cluster_counts_arr.mean():.1f}")
    print(f"  Among non-zero seeds, top cluster classification:")
    near_truth = 0
    near_twin = 0
    other = 0
    for sd in seeds:
        cl = per_seed[int(sd)].get("clusters", [])
        # Largest cluster
        if cl:
            largest = max(cl, key=lambda c: c["size"])
            if largest["label"] == "near_truth":
                near_truth += 1
            elif largest["label"] == "near_twin":
                near_twin += 1
            else:
                other += 1
    print(f"    near_truth: {near_truth}, near_twin: {near_twin}, other: {other}")


if __name__ == "__main__":
    main()
