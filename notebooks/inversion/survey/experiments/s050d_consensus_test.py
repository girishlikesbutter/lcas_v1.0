"""s050d — refined C1 test: per-bin distinct-t_k consensus.

Hypothesis (stronger than s050c's max-bin-count): for each q_a, identify
the ω-bin that gets votes from the MOST distinct t_k's. Truth-q_a should
have a bin reached by all K t_k's (the truth-ω region attracts one
truth-adjacent ω-estimate per t_k). Random q_a should have at most 1-2
t_k's voting per bin (each t_k's per-q_a ω-distribution lives in its
own random region).

Also tests Q_k pre-filtering: restrict to top-N_QB tightest q_b at each
t_k (lowest mag-discrepancy from measured), so noise from large loose Q_k
doesn't dominate.

Sweep: K_TIGHT_TK ∈ {3, 5, 10}, N_QB ∈ {None, 200, 500}, BINS ∈ {50, 100, 200}.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
SURROGATE_PATH = Path("/home/girish/surrogate_model")
sys.path.insert(0, str(SURROGATE_PATH))

from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1  # noqa: E402

CASCADE_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "cascade.npz"
SCAN_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "scan.npz"
TRAJ_NPZ = (
    PROJECT_ROOT
    / "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed014.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s050d_consensus_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_OM_LO = 0.05
COHORT_OM_HI = 2.0
HIST_RANGE_DPS = (-2.5, 2.5)

SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.ndim == 1:
        q1 = q1[None, :]
    if q2.ndim == 1:
        q2 = q2[None, :]
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = np.broadcast_to(q1, q2.shape).copy()
    if q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = np.broadcast_to(q2, q1.shape).copy()
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def omega_from_pair(qA: np.ndarray, qB: np.ndarray, dt: float) -> np.ndarray:
    qA_inv = qA.copy()
    qA_inv[:, 1:] *= -1.0
    qr = quat_mul_batch(qB, qA_inv)
    flip = qr[:, 0] < 0
    qr[flip] *= -1
    w = np.clip(qr[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sin(angle / 2.0)
    axis = np.zeros((qr.shape[0], 3))
    ok = s > 1e-9
    axis[ok] = qr[ok, 1:] / s[ok, None]
    return (angle / dt)[:, None] * axis


def quat_angular_dist_deg(q_pool: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    cos_half = np.abs(q_pool @ q_ref)
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cos_half))


def consensus_metric(
    Q0: np.ndarray,
    truth_qa_idx: int,
    chosen_meta: list,
    nb: int,
    om_truth_dps: np.ndarray,
    om_truth_mag_dps: float,
) -> dict:
    """For each q_a, compute the max # of distinct t_k's contributing to any ω-bin.

    Also reports max single-bin count (s050c metric) for comparison.
    Returns dict with per-q_a consensus_score and ranking info for truth-q_a.
    """
    n_q_a = len(Q0)
    edges = np.linspace(*HIST_RANGE_DPS, nb + 1)
    K = len(chosen_meta)

    consensus_score = np.zeros(n_q_a, dtype=np.int8)  # max # distinct t_k's per bin
    max_bin_count = np.zeros(n_q_a, dtype=np.int32)  # for comparison
    truth_bin_consensus = np.zeros(n_q_a, dtype=np.int8)

    truth_bin = np.floor(
        (om_truth_dps - HIST_RANGE_DPS[0])
        / (HIST_RANGE_DPS[1] - HIST_RANGE_DPS[0])
        * nb
    ).astype(int)
    truth_bin = np.clip(truth_bin, 0, nb - 1)

    t_h = time.time()
    for i in range(n_q_a):
        q_a = Q0[i]
        per_tk_bin_indices_set = []  # one set per t_k of bin-tuple-keys
        all_om = []
        for kk, (idx, ep, dk, Qk, dt_k) in enumerate(chosen_meta):
            qa_tiled = np.broadcast_to(q_a, Qk.shape).copy()
            ω_dps = np.degrees(omega_from_pair(qa_tiled, Qk, dt_k))
            ω_mag = np.linalg.norm(ω_dps, axis=1)
            in_cohort = (ω_mag >= COHORT_OM_LO) & (ω_mag <= COHORT_OM_HI)
            ω_dps = ω_dps[in_cohort]
            if len(ω_dps) == 0:
                per_tk_bin_indices_set.append(set())
                continue
            # Bin
            bx = np.floor(
                (ω_dps[:, 0] - HIST_RANGE_DPS[0])
                / (HIST_RANGE_DPS[1] - HIST_RANGE_DPS[0])
                * nb
            ).astype(int)
            by = np.floor(
                (ω_dps[:, 1] - HIST_RANGE_DPS[0])
                / (HIST_RANGE_DPS[1] - HIST_RANGE_DPS[0])
                * nb
            ).astype(int)
            bz = np.floor(
                (ω_dps[:, 2] - HIST_RANGE_DPS[0])
                / (HIST_RANGE_DPS[1] - HIST_RANGE_DPS[0])
                * nb
            ).astype(int)
            bx = np.clip(bx, 0, nb - 1)
            by = np.clip(by, 0, nb - 1)
            bz = np.clip(bz, 0, nb - 1)
            # Bin-tuple keys for this t_k (use packed int64 for speed)
            keys = bx.astype(np.int64) * (nb * nb) + by.astype(np.int64) * nb + bz.astype(
                np.int64
            )
            per_tk_bin_indices_set.append(set(keys.tolist()))
            all_om.append((bx, by, bz))

        # Per-bin: count how many distinct t_k's contributed
        # Use a dict bin_key -> count of distinct t_k contributors.
        bin_tk_count: dict[int, int] = {}
        for s in per_tk_bin_indices_set:
            for key in s:
                bin_tk_count[key] = bin_tk_count.get(key, 0) + 1

        if bin_tk_count:
            consensus_score[i] = max(bin_tk_count.values())
        truth_key = (
            int(truth_bin[0]) * (nb * nb) + int(truth_bin[1]) * nb + int(truth_bin[2])
        )
        truth_bin_consensus[i] = bin_tk_count.get(truth_key, 0)

        # Also compute max-bin-count (sum over t_k contributions in same bin)
        bin_total_count: dict[int, int] = {}
        for kk, (bx, by, bz) in enumerate(all_om):
            keys = bx.astype(np.int64) * (nb * nb) + by.astype(np.int64) * nb + bz.astype(
                np.int64
            )
            unique_keys, counts = np.unique(keys, return_counts=True)
            for k, c in zip(unique_keys, counts):
                bin_total_count[int(k)] = bin_total_count.get(int(k), 0) + int(c)
        if bin_total_count:
            max_bin_count[i] = max(bin_total_count.values())

    print(f"    consensus computed in {time.time()-t_h:.1f}s")
    return {
        "consensus_score": consensus_score,
        "max_bin_count": max_bin_count,
        "truth_bin_consensus": truth_bin_consensus,
    }


def main() -> None:
    t_start = time.time()

    # Load
    print(f"[load] scan: {SCAN_NPZ}")
    scan = np.load(SCAN_NPZ)
    q_pool = scan["q_pool_wxyz"]
    scan_eps = scan["scan_epochs"]
    survive_mask = scan["survive_mask"]
    n_surv_per_ep = scan["n_survivors"]
    obs_times_at_scan = scan["obs_times_at_scan"]
    measured_at_scan = scan["measured_at"]

    print(f"[load] cascade: {CASCADE_NPZ}")
    casc = np.load(CASCADE_NPZ)
    truth_q_a = casc["truth_q_a"]
    om_truth_at_t0 = casc["om_truth_at_t0"]
    om_truth_dps = np.degrees(om_truth_at_t0)
    om_truth_mag_dps = float(np.linalg.norm(om_truth_dps))
    t0_ep = int(casc["t0_ep"])

    # Truth-q_a in Q_0
    idx_t0 = int(np.where(scan_eps == t0_ep)[0][0])
    Q0 = q_pool[survive_mask[idx_t0]]
    d_truth_Q0 = quat_angular_dist_deg(Q0, truth_q_a)
    truth_qa_idx = int(np.argmin(d_truth_Q0))

    # Sweep configs
    K_VALUES = [3, 5, 10, 15]
    BINS_VALUES = [50, 100, 200]

    # Pre-load surrogate (for Q_k pre-filter via fresh mag-agreement scoring)
    # Not needed if we use the cached scan survivor mask as Q_k
    print("[load] surrogate v1 (for Q_k tightening if needed)")
    v1_weights = SURROGATE_PATH / "surrogate_model" / "s10_5M_weights.npz"
    v1_norm = SURROGATE_PATH / "surrogate_model" / "s10_5M_normalization.npz"
    model = SurrogateV1(str(v1_weights), str(v1_norm))

    # Choose K tightest non-anchor scan epochs
    sort_idx = np.argsort(n_surv_per_ep)
    candidate_t_k_indices = []
    for i in sort_idx:
        i = int(i)
        if i == idx_t0:
            continue
        if n_surv_per_ep[i] == 0:
            continue
        candidate_t_k_indices.append(i)

    print(
        f"[scan] candidate t_k indices (sorted by |Q_k|): "
        f"top10 epochs={[int(scan_eps[i]) for i in candidate_t_k_indices[:10]]}, "
        f"|Q_k|={[int(n_surv_per_ep[i]) for i in candidate_t_k_indices[:10]]}"
    )

    results = []
    for K in K_VALUES:
        print(f"\n=== K={K} t_k epochs ===")
        chosen_indices = candidate_t_k_indices[:K]
        chosen_meta = []
        for idx in chosen_indices:
            ep = int(scan_eps[idx])
            Qk = q_pool[survive_mask[idx]]
            dt_k = float(obs_times_at_scan[idx] - obs_times_at_scan[idx_t0])
            chosen_meta.append((idx, ep, ep - t0_ep, Qk, dt_k))

        for nb in BINS_VALUES:
            bin_width_pct = 5.0 / nb / om_truth_mag_dps * 100
            print(
                f"  bins/axis={nb}  (bin width={5.0/nb:.3f} dps = "
                f"{bin_width_pct:.1f}% of |ω|)"
            )
            m = consensus_metric(
                Q0, truth_qa_idx, chosen_meta, nb, om_truth_dps, om_truth_mag_dps
            )
            consensus_score = m["consensus_score"]
            max_bin_count = m["max_bin_count"]
            truth_bin_consensus = m["truth_bin_consensus"]

            # Rank truth-q_a by consensus_score (higher better)
            tc = consensus_score[truth_qa_idx]
            rank_c = int((consensus_score > tc).sum())
            ties_c = int((consensus_score == tc).sum())
            # Rank by max_bin_count
            tmc = max_bin_count[truth_qa_idx]
            rank_mc = int((max_bin_count > tmc).sum())
            # Rank by truth-bin consensus (truth_bin_consensus[truth_qa_idx])
            tbc = truth_bin_consensus[truth_qa_idx]
            rank_tbc = int((truth_bin_consensus > tbc).sum())

            print(
                f"    consensus_score: truth-q_a={tc}/{K}  rank={rank_c+1}/1113  "
                f"({100*(rank_c+1)/1113:.1f}%)  ties_at_truth={ties_c}  "
                f"pool max/p99/p90/p50 = {consensus_score.max()}/"
                f"{int(np.percentile(consensus_score, 99))}/"
                f"{int(np.percentile(consensus_score, 90))}/"
                f"{int(np.percentile(consensus_score, 50))}"
            )
            print(
                f"    max_bin_count:   truth-q_a={tmc}  rank={rank_mc+1}/1113  "
                f"pool max/p99/p90/p50 = {max_bin_count.max()}/"
                f"{int(np.percentile(max_bin_count, 99))}/"
                f"{int(np.percentile(max_bin_count, 90))}/"
                f"{int(np.percentile(max_bin_count, 50))}"
            )
            print(
                f"    truth-bin consensus: truth-q_a={tbc}/{K}  rank={rank_tbc+1}/1113"
            )

            results.append(
                {
                    "K": K,
                    "bins_per_axis": nb,
                    "bin_width_dps": float(5.0 / nb),
                    "bin_width_pct": float(bin_width_pct),
                    "truth_consensus": int(tc),
                    "consensus_rank": int(rank_c + 1),
                    "consensus_ties_at_truth": int(ties_c),
                    "truth_max_bin_count": int(tmc),
                    "max_bin_count_rank": int(rank_mc + 1),
                    "truth_bin_consensus": int(tbc),
                    "truth_bin_consensus_rank": int(rank_tbc + 1),
                    "consensus_pool_max": int(consensus_score.max()),
                    "consensus_pool_p50": int(np.percentile(consensus_score, 50)),
                    "consensus_pool_p90": int(np.percentile(consensus_score, 90)),
                    "consensus_pool_p99": int(np.percentile(consensus_score, 99)),
                }
            )

    # Save
    summary = {
        "seed": 14,
        "t_0_ep": t0_ep,
        "om_truth_dps": om_truth_dps.tolist(),
        "om_truth_mag_dps": om_truth_mag_dps,
        "truth_qa_idx_in_Q0": truth_qa_idx,
        "results": results,
        "wall_total_s": float(time.time() - t_start),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[saved] {OUT_DIR / 'summary.json'}")

    # Quick plot: rank-vs-K-and-bins heatmap
    K_arr = sorted(set(r["K"] for r in results))
    nb_arr = sorted(set(r["bins_per_axis"] for r in results))
    rank_consensus = np.zeros((len(K_arr), len(nb_arr)))
    rank_max_bin = np.zeros((len(K_arr), len(nb_arr)))
    truth_consensus_val = np.zeros((len(K_arr), len(nb_arr)))
    for r in results:
        i = K_arr.index(r["K"])
        j = nb_arr.index(r["bins_per_axis"])
        rank_consensus[i, j] = r["consensus_rank"]
        rank_max_bin[i, j] = r["max_bin_count_rank"]
        truth_consensus_val[i, j] = r["truth_consensus"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(
        rank_consensus, cmap="viridis_r", aspect="auto", origin="lower"
    )
    axes[0].set_xticks(range(len(nb_arr)))
    axes[0].set_xticklabels(nb_arr)
    axes[0].set_yticks(range(len(K_arr)))
    axes[0].set_yticklabels(K_arr)
    axes[0].set_xlabel("bins/axis")
    axes[0].set_ylabel("K (# t_k epochs)")
    axes[0].set_title("Truth-q_a rank by consensus_score (lower=better)")
    for i in range(len(K_arr)):
        for j in range(len(nb_arr)):
            axes[0].text(
                j,
                i,
                f"{int(rank_consensus[i, j])}/1113\n"
                f"score={int(truth_consensus_val[i, j])}/{K_arr[i]}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        rank_max_bin, cmap="viridis_r", aspect="auto", origin="lower"
    )
    axes[1].set_xticks(range(len(nb_arr)))
    axes[1].set_xticklabels(nb_arr)
    axes[1].set_yticks(range(len(K_arr)))
    axes[1].set_yticklabels(K_arr)
    axes[1].set_xlabel("bins/axis")
    axes[1].set_ylabel("K (# t_k epochs)")
    axes[1].set_title("Truth-q_a rank by max_bin_count (lower=better)")
    for i in range(len(K_arr)):
        for j in range(len(nb_arr)):
            axes[1].text(
                j,
                i,
                f"{int(rank_max_bin[i, j])}/1113",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    plt.colorbar(im1, ax=axes[1])

    truth_consensus_score_max = max(K_arr)
    rel_score = np.array(
        [
            [
                truth_consensus_val[i, j] / K_arr[i]
                for j in range(len(nb_arr))
            ]
            for i in range(len(K_arr))
        ]
    )
    im2 = axes[2].imshow(rel_score, cmap="viridis", aspect="auto", origin="lower")
    axes[2].set_xticks(range(len(nb_arr)))
    axes[2].set_xticklabels(nb_arr)
    axes[2].set_yticks(range(len(K_arr)))
    axes[2].set_yticklabels(K_arr)
    axes[2].set_xlabel("bins/axis")
    axes[2].set_ylabel("K (# t_k epochs)")
    axes[2].set_title("Truth consensus score / K (1.0 = full consensus)")
    for i in range(len(K_arr)):
        for j in range(len(nb_arr)):
            axes[2].text(
                j,
                i,
                f"{rel_score[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if rel_score[i, j] < 0.5 else "black",
                fontsize=9,
            )
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("s050d — consensus discriminator sweep, seed 14", fontsize=14)
    plt.tight_layout()
    out_png = OUT_DIR / "consensus_sweep.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[saved] {out_png}")
    print(f"[done] wall = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
