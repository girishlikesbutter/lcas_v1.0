"""s050c — C1 quick test: per-q_a ω-estimate concentration via Hough voting.

Tests whether multi-pair ω derivation gives a per-q_a "concentration score"
that discriminates truth-adjacent q_a from random q_a in the cascade pool.

Mechanism: for each q_a in Q_0, derive ω from (q_a, q_b) for q_b in Q_k at
each of K tightest non-anchor scan epochs. Pool all ω-estimates per q_a;
3D-histogram them; max bin count = concentration. Truth-q_a should have
higher concentration because the truth-adjacent q_b at each t_k contributes
ω near truth-ω (clustering across t_k's), while random q_a gets random
ω-estimates spread across the cohort range.

If truth-q_a ranks in top-K (K << 1113), C1 is a viable primitive.
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

CASCADE_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "cascade.npz"
SCAN_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "scan.npz"
TRAJ_NPZ = (
    PROJECT_ROOT
    / "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed014.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s050c_multipair_concentration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_TIGHT_TK = 5             # use 5 tightest non-anchor scan eps as t_k
COHORT_OM_LO = 0.05        # dps
COHORT_OM_HI = 2.0         # dps
HIST_BINS_PER_AXIS = [25, 50, 100]  # try multiple resolutions
HIST_RANGE_DPS = (-2.5, 2.5)


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


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"[load] scan: {SCAN_NPZ}")
    scan = np.load(SCAN_NPZ)
    q_pool = scan["q_pool_wxyz"]  # (500000, 4)
    scan_eps = scan["scan_epochs"]  # (30,)
    survive_mask = scan["survive_mask"]  # (30, 500000)
    n_surv_per_ep = scan["n_survivors"]  # (30,)
    obs_times_at_scan = scan["obs_times_at_scan"]  # (30,)
    quats_truth_at_scan = scan["quats_truth_at_scan"]  # (30, 4)

    print(f"[load] cascade: {CASCADE_NPZ}")
    casc = np.load(CASCADE_NPZ)
    truth_q_a = casc["truth_q_a"]
    om_truth_at_t0 = casc["om_truth_at_t0"]  # rad/s
    om_truth_at_t0_dps = np.degrees(om_truth_at_t0)
    om_truth_mag_dps = float(np.linalg.norm(om_truth_at_t0_dps))
    t0_ep = int(casc["t0_ep"])
    print(
        f"[truth] t_0_ep={t0_ep}, |ω|={om_truth_mag_dps:.4f} dps, "
        f"ω_dps={om_truth_at_t0_dps}"
    )

    # Find t_0 in scan
    idx_t0 = int(np.where(scan_eps == t0_ep)[0][0])
    Q0_mask = survive_mask[idx_t0]
    Q0 = q_pool[Q0_mask]  # (1113, 4)
    print(
        f"[scan] t_0 at scan idx {idx_t0}, |Q_0|={len(Q0)}, "
        f"nearest_truth_to_Q0 = {scan['nearest_truth_deg'][idx_t0]:.2f}°"
    )

    # Truth-q_a in Q_0
    d_truth_Q0 = quat_angular_dist_deg(Q0, truth_q_a)
    truth_qa_idx_in_Q0 = int(np.argmin(d_truth_Q0))
    print(
        f"[scan] truth-q_a in Q_0: idx={truth_qa_idx_in_Q0}, "
        f"q_dist_to_truth_q_a={d_truth_Q0[truth_qa_idx_in_Q0]:.4f}° "
        f"(should be 0)"
    )

    # ------------------------------------------------------------------
    # Pick K_TIGHT_TK tightest non-anchor scan epochs as t_k
    # ------------------------------------------------------------------
    sort_idx = np.argsort(n_surv_per_ep)
    chosen_t_k_idx = []
    for i in sort_idx:
        i = int(i)
        if i == idx_t0:
            continue
        if n_surv_per_ep[i] == 0:
            continue
        chosen_t_k_idx.append(i)
        if len(chosen_t_k_idx) == K_TIGHT_TK:
            break

    chosen_meta = []
    for idx in chosen_t_k_idx:
        ep = int(scan_eps[idx])
        Qk = q_pool[survive_mask[idx]]
        dt_k = float(obs_times_at_scan[idx] - obs_times_at_scan[idx_t0])
        dk = ep - t0_ep
        chosen_meta.append((idx, ep, dk, Qk, dt_k))
        print(
            f"[scan] t_k ep={ep} Δk={dk:+d} |Q_k|={len(Qk)} "
            f"Δt={dt_k:+.2f}s  truth-rotation={om_truth_mag_dps*abs(dt_k):.2f}°"
        )

    n_q_a = len(Q0)

    # ------------------------------------------------------------------
    # Per-q_a Hough voting
    # ------------------------------------------------------------------
    print(
        f"\n[hough] computing ω-estimates and per-q_a 3D histograms "
        f"(N_q_a={n_q_a}, K_tk={K_TIGHT_TK})"
    )

    # Storage
    n_estimates_in_cohort = np.zeros(n_q_a, dtype=np.int64)
    median_om_dps = np.zeros((n_q_a, 3))
    std_om_dps = np.zeros((n_q_a, 3))
    concentration_per_bin = {nb: np.zeros(n_q_a) for nb in HIST_BINS_PER_AXIS}
    truth_om_bin_count = {nb: np.zeros(n_q_a) for nb in HIST_BINS_PER_AXIS}
    truth_om_bin_index = {
        nb: np.floor(
            (om_truth_at_t0_dps - HIST_RANGE_DPS[0])
            / (HIST_RANGE_DPS[1] - HIST_RANGE_DPS[0])
            * nb
        ).astype(int)
        for nb in HIST_BINS_PER_AXIS
    }
    print(
        f"[hough] truth-ω bin index per resolution: "
        + ", ".join(
            f"{nb}b={truth_om_bin_index[nb].tolist()}" for nb in HIST_BINS_PER_AXIS
        )
    )

    edges_per_bin = {
        nb: np.linspace(*HIST_RANGE_DPS, nb + 1) for nb in HIST_BINS_PER_AXIS
    }

    # Pre-compute Q_k tiled offsets for vectorisation per q_a
    t_h0 = time.time()
    for i in range(n_q_a):
        q_a = Q0[i]
        all_om_dps_chunks = []
        for idx, ep, dk, Qk, dt_k in chosen_meta:
            qa_tiled = np.broadcast_to(q_a, Qk.shape).copy()
            ω_rad = omega_from_pair(qa_tiled, Qk, dt_k)
            ω_dps = np.degrees(ω_rad)
            ω_mag = np.linalg.norm(ω_dps, axis=1)
            in_cohort = (ω_mag >= COHORT_OM_LO) & (ω_mag <= COHORT_OM_HI)
            all_om_dps_chunks.append(ω_dps[in_cohort])
        all_om_dps = np.concatenate(all_om_dps_chunks, axis=0)
        n_estimates_in_cohort[i] = len(all_om_dps)
        if len(all_om_dps) == 0:
            continue
        median_om_dps[i] = np.median(all_om_dps, axis=0)
        std_om_dps[i] = np.std(all_om_dps, axis=0)
        for nb in HIST_BINS_PER_AXIS:
            edges = edges_per_bin[nb]
            H, _ = np.histogramdd(
                all_om_dps,
                bins=(edges, edges, edges),
            )
            concentration_per_bin[nb][i] = H.max()
            tb = truth_om_bin_index[nb]
            if 0 <= tb[0] < nb and 0 <= tb[1] < nb and 0 <= tb[2] < nb:
                truth_om_bin_count[nb][i] = H[tb[0], tb[1], tb[2]]
        if (i + 1) % 100 == 0:
            print(
                f"  q_a {i+1}/{n_q_a}  wall={time.time()-t_h0:.1f}s  "
                f"median estimate count={np.median(n_estimates_in_cohort[:i+1]):.0f}"
            )
    print(f"[hough] wall total: {time.time()-t_h0:.1f}s")

    # ------------------------------------------------------------------
    # Diagnostics: where does truth-q_a rank?
    # ------------------------------------------------------------------
    print("\n=== Diagnostics ===")
    for nb in HIST_BINS_PER_AXIS:
        # Concentration ranking (max bin count)
        c = concentration_per_bin[nb]
        truth_c = c[truth_qa_idx_in_Q0]
        rank_c = int((c > truth_c).sum())
        rank_pct = 100.0 * (rank_c + 1) / n_q_a
        # Truth-bin specific count
        tb = truth_om_bin_count[nb]
        truth_tb = tb[truth_qa_idx_in_Q0]
        rank_tb = int((tb > truth_tb).sum())
        rank_tb_pct = 100.0 * (rank_tb + 1) / n_q_a
        print(
            f"  bins/axis={nb:>3d} (bin width={5.0/nb:.3f} dps = "
            f"{(5.0/nb)/om_truth_mag_dps*100:.1f}% of |ω|)"
        )
        print(
            f"    max-bin-count: truth-q_a={truth_c:.0f}  rank={rank_c+1}/{n_q_a}  "
            f"({rank_pct:.1f}%)  pool p50/p90/p99 = "
            f"{np.percentile(c, 50):.0f}/{np.percentile(c, 90):.0f}/"
            f"{np.percentile(c, 99):.0f}"
        )
        print(
            f"    truth-bin-count: truth-q_a={truth_tb:.0f}  rank={rank_tb+1}/{n_q_a}  "
            f"({rank_tb_pct:.1f}%)  pool p50/p90/p99 = "
            f"{np.percentile(tb, 50):.0f}/{np.percentile(tb, 90):.0f}/"
            f"{np.percentile(tb, 99):.0f}"
        )

    # Median-ω-error metric — does truth-q_a's median ω match truth?
    median_om_err = np.linalg.norm(
        median_om_dps - om_truth_at_t0_dps, axis=1
    ) / om_truth_mag_dps
    truth_med_err = median_om_err[truth_qa_idx_in_Q0]
    rank_me = int((median_om_err < truth_med_err).sum())
    print(
        f"\n  median-ω error (relative): truth-q_a={truth_med_err*100:.2f}%  "
        f"rank={rank_me+1}/{n_q_a}  pool p10/p50 = "
        f"{np.percentile(median_om_err, 10)*100:.2f}%/{np.percentile(median_om_err, 50)*100:.2f}%"
    )

    # std-ω metric (lower = more concentrated)
    total_std = np.linalg.norm(std_om_dps, axis=1) / om_truth_mag_dps
    truth_std = total_std[truth_qa_idx_in_Q0]
    rank_std = int((total_std < truth_std).sum())
    print(
        f"  total-std-ω (relative): truth-q_a={truth_std*100:.2f}%  "
        f"rank={rank_std+1}/{n_q_a} (lower=better)  pool p10/p50 = "
        f"{np.percentile(total_std, 10)*100:.2f}%/{np.percentile(total_std, 50)*100:.2f}%"
    )

    # Plot: rank-ordered concentration scores; mark truth-q_a
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: concentration ranking (max bin count)
    nb = 50
    c = concentration_per_bin[nb]
    sort_c = np.sort(c)[::-1]
    truth_c = c[truth_qa_idx_in_Q0]
    truth_rank_c = int((c > truth_c).sum())
    ax = axes[0, 0]
    ax.plot(sort_c, "C0-", lw=1)
    ax.axhline(truth_c, color="red", ls="--", label=f"truth-q_a={truth_c:.0f}")
    ax.axvline(truth_rank_c, color="red", ls=":", lw=1, label=f"rank={truth_rank_c+1}")
    ax.set_xlabel("q_a rank (highest concentration first)")
    ax.set_ylabel(f"max-bin-count ({nb}-bins/axis)")
    ax.set_title(
        f"Concentration ranking ({nb} bins/axis = "
        f"{(5.0/nb)/om_truth_mag_dps*100:.1f}% of |ω|)"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: truth-bin count
    tb = truth_om_bin_count[nb]
    sort_tb = np.sort(tb)[::-1]
    truth_tb = tb[truth_qa_idx_in_Q0]
    truth_rank_tb = int((tb > truth_tb).sum())
    ax = axes[0, 1]
    ax.plot(sort_tb, "C2-", lw=1)
    ax.axhline(truth_tb, color="red", ls="--", label=f"truth-q_a={truth_tb:.0f}")
    ax.axvline(truth_rank_tb, color="red", ls=":", lw=1, label=f"rank={truth_rank_tb+1}")
    ax.set_xlabel("q_a rank (highest truth-bin count first)")
    ax.set_ylabel(f"count in truth-ω bin ({nb}-bins/axis)")
    ax.set_title("Truth-bin count ranking")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: median ω error vs concentration
    ax = axes[1, 0]
    ax.scatter(
        median_om_err * 100,
        c,
        s=5,
        alpha=0.4,
        label="random q_a",
    )
    ax.scatter(
        [median_om_err[truth_qa_idx_in_Q0] * 100],
        [truth_c],
        s=100,
        color="red",
        marker="*",
        label="truth-q_a",
        zorder=5,
    )
    ax.set_xlabel("median ω error (% of |ω|)")
    ax.set_ylabel(f"max-bin-count ({nb}-bins/axis)")
    ax.set_title("Concentration vs median-ω error per q_a")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: std-ω vs concentration
    ax = axes[1, 1]
    ax.scatter(
        total_std * 100,
        c,
        s=5,
        alpha=0.4,
        label="random q_a",
    )
    ax.scatter(
        [total_std[truth_qa_idx_in_Q0] * 100],
        [truth_c],
        s=100,
        color="red",
        marker="*",
        label="truth-q_a",
        zorder=5,
    )
    ax.set_xlabel("total std ω (% of |ω|)")
    ax.set_ylabel(f"max-bin-count ({nb}-bins/axis)")
    ax.set_title("Concentration vs std-ω per q_a")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"s050c — multi-pair concentration test (seed 14, K={K_TIGHT_TK} t_k)",
        fontsize=14,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "concentration_test.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[saved] {out_png}")

    np.savez(
        OUT_DIR / "concentration.npz",
        Q0_idx=truth_qa_idx_in_Q0,
        n_estimates_in_cohort=n_estimates_in_cohort,
        median_om_dps=median_om_dps,
        std_om_dps=std_om_dps,
        median_om_err_rel=median_om_err,
        total_std_rel=total_std,
        truth_q_a=truth_q_a,
        om_truth_at_t0_dps=om_truth_at_t0_dps,
        chosen_t_k_eps=np.array([m[1] for m in chosen_meta]),
        **{f"concentration_{nb}b": concentration_per_bin[nb] for nb in HIST_BINS_PER_AXIS},
        **{f"truth_bin_count_{nb}b": truth_om_bin_count[nb] for nb in HIST_BINS_PER_AXIS},
    )
    print(f"[saved] {OUT_DIR / 'concentration.npz'}")

    summary = {
        "seed": 14,
        "t_0_ep": t0_ep,
        "K_tight_tk": K_TIGHT_TK,
        "chosen_t_k_eps": [m[1] for m in chosen_meta],
        "chosen_t_k_dks": [m[2] for m in chosen_meta],
        "chosen_t_k_n_Qk": [int(len(m[3])) for m in chosen_meta],
        "n_q_a_in_Q0": n_q_a,
        "truth_qa_idx_in_Q0": truth_qa_idx_in_Q0,
        "om_truth_at_t0_dps": om_truth_at_t0_dps.tolist(),
        "om_truth_mag_dps": om_truth_mag_dps,
        "n_estimates_in_cohort": {
            "min": int(n_estimates_in_cohort.min()),
            "p10": int(np.percentile(n_estimates_in_cohort, 10)),
            "p50": int(np.percentile(n_estimates_in_cohort, 50)),
            "p90": int(np.percentile(n_estimates_in_cohort, 90)),
            "max": int(n_estimates_in_cohort.max()),
            "truth_qa": int(n_estimates_in_cohort[truth_qa_idx_in_Q0]),
        },
        "median_om_err": {
            "truth_qa_pct": float(median_om_err[truth_qa_idx_in_Q0] * 100),
            "rank": int((median_om_err < median_om_err[truth_qa_idx_in_Q0]).sum())
            + 1,
            "p10_pct": float(np.percentile(median_om_err, 10) * 100),
            "p50_pct": float(np.percentile(median_om_err, 50) * 100),
        },
        "total_std_om": {
            "truth_qa_pct": float(total_std[truth_qa_idx_in_Q0] * 100),
            "rank": int((total_std < total_std[truth_qa_idx_in_Q0]).sum()) + 1,
            "p10_pct": float(np.percentile(total_std, 10) * 100),
            "p50_pct": float(np.percentile(total_std, 50) * 100),
        },
        "concentration_results": {
            f"bins_{nb}": {
                "bin_width_dps": 5.0 / nb,
                "bin_width_rel_omega": 5.0 / nb / om_truth_mag_dps,
                "max_bin_count": {
                    "truth_qa": float(concentration_per_bin[nb][truth_qa_idx_in_Q0]),
                    "rank": int(
                        (
                            concentration_per_bin[nb]
                            > concentration_per_bin[nb][truth_qa_idx_in_Q0]
                        ).sum()
                    )
                    + 1,
                    "p50": float(np.percentile(concentration_per_bin[nb], 50)),
                    "p90": float(np.percentile(concentration_per_bin[nb], 90)),
                    "p99": float(np.percentile(concentration_per_bin[nb], 99)),
                },
                "truth_bin_count": {
                    "truth_qa": float(truth_om_bin_count[nb][truth_qa_idx_in_Q0]),
                    "rank": int(
                        (
                            truth_om_bin_count[nb]
                            > truth_om_bin_count[nb][truth_qa_idx_in_Q0]
                        ).sum()
                    )
                    + 1,
                    "p50": float(np.percentile(truth_om_bin_count[nb], 50)),
                    "p90": float(np.percentile(truth_om_bin_count[nb], 90)),
                    "p99": float(np.percentile(truth_om_bin_count[nb], 99)),
                },
            }
            for nb in HIST_BINS_PER_AXIS
        },
        "wall_total_s": float(time.time() - t_start),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"[done] wall = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
