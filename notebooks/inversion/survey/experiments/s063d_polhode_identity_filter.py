"""s063d — Jacobi-exact polhode-identity filter retest on s055d cached pool.

s055d found that filtering the s049 cascade pool (39,591 survivors on seed 14)
by SCALAR pol_diam gives at best 1.24× truth enrichment — well below the 3× gate
for "operational filter". The pol_diam-only filter conflates many different
polhodes that happen to share a similar diameter.

Jacobi reframe: each polhode is uniquely determined by (|L|, 2T) given I. Compute
both invariants analytically per candidate ω (zero propagation), then filter by
2D tube around truth's (|L|, 2T) and measure the resulting truth enrichment.

This is an UPPER-BOUND check (oracle filter using truth invariants). If even
truth-oracle 2D filter fails to beat 3× enrichment, the s055d 1.24× was not
limited by the scalar approximation — the cascade-pool ω-noise simply lands
many random candidates ON truth's polhode by chance.

Compares two filters:
  - 1D: scalar pol_diam disagreement (replicates s055d ranking)
  - 2D: joint (|L|, 2T) tube around truth (Jacobi-exact)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))

POOL = SURVEY / "results" / "s055d_cascade_pol_diam_filter" / "pool.npz"
POL_DIAM = SURVEY / "results" / "s055d_cascade_pol_diam_filter" / "pol_diam.npz"
OUT_DIR = SURVEY / "results" / "s063d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INERTIA = np.diag([37985.15566495171, 38305.70560133419, 7749.014672251076])


def main():
    pool = np.load(POOL)
    pol = np.load(POL_DIAM)

    om_surv = pool["om_kept_survivors"].astype(np.float64)            # (N, 3) rad/s
    om_truth_t0 = pool["om_truth_at_t0"].astype(np.float64)          # (3,) rad/s
    truth_mask = pool["truth_qa_in_survivors"].astype(bool)           # (N,)

    N = len(om_surv)
    n_truth = int(truth_mask.sum())
    rate_before = n_truth / N
    print(f"N_survivors = {N}, n_truth = {n_truth}, rate = {rate_before:.6f}")

    # Cached scalar pol_diam (s055d, exact via DOP853)
    pol_diam_all = pol["pol_diam_implied"]    # (N,) dps
    truth_pol_diam = float(pol["truth_pol_diam"])
    print(f"truth pol_diam = {truth_pol_diam:.4f} dps")

    # --- Jacobi-exact polhode invariants (vectorized linear algebra) ---
    # L = I·ω, |L|² = (I·ω)·(I·ω); 2T = ω·I·ω
    L_vec = om_surv @ INERTIA.T               # (N, 3)
    L2_all = np.sum(L_vec ** 2, axis=1)       # (N,)
    twoT_all = np.einsum("ni,ij,nj->n", om_surv, INERTIA, om_surv)

    L_truth_vec = INERTIA @ om_truth_t0
    L2_truth = float(np.sum(L_truth_vec ** 2))
    twoT_truth = float(om_truth_t0 @ INERTIA @ om_truth_t0)
    print(f"truth |L|² = {L2_truth:.4e}    truth 2T = {twoT_truth:.4e}")

    # Truth-candidate invariant subset stats
    L2_truth_cands = L2_all[truth_mask]
    twoT_truth_cands = twoT_all[truth_mask]
    print(f"truth-q_a candidate |L|² range: [{L2_truth_cands.min():.3e}, "
          f"{L2_truth_cands.max():.3e}]  median={np.median(L2_truth_cands):.3e}")
    print(f"truth-q_a candidate 2T range:   [{twoT_truth_cands.min():.3e}, "
          f"{twoT_truth_cands.max():.3e}]  median={np.median(twoT_truth_cands):.3e}")

    # Relative deltas
    rel_dL2 = np.abs(L2_all - L2_truth) / L2_truth
    rel_dT = np.abs(twoT_all - twoT_truth) / twoT_truth

    # --- Filter sweeps ---
    # 1D scalar pol_diam (replicate of s055d ranking, for direct comparison)
    pol_diam_rel = np.abs(pol_diam_all - truth_pol_diam) / truth_pol_diam
    thresholds_1d_pct = [5, 10, 15, 25, 35, 50, 100]
    rows_1d = []
    for pct in thresholds_1d_pct:
        thr = pct / 100.0
        keep = pol_diam_rel < thr
        n_keep = int(keep.sum())
        n_truth_keep = int((keep & truth_mask).sum())
        if n_keep == 0:
            enr = 0.0
        else:
            enr = (n_truth_keep / n_keep) / rate_before
        rows_1d.append(dict(
            threshold_pct=pct, n_keep=n_keep, n_truth_keep=n_truth_keep,
            decimation=n_keep / N, enrichment=enr,
        ))
        print(f"  1D-poldiam thr={pct}%: keep={n_keep:5d} truth={n_truth_keep:3d} enr={enr:.3f}")

    # 2D Jacobi (|L|, 2T) joint tube
    thresholds_2d_pct = [1, 2, 3, 5, 10, 15, 25, 50]
    rows_2d = []
    for pct in thresholds_2d_pct:
        thr = pct / 100.0
        keep = (rel_dL2 < thr) & (rel_dT < thr)
        n_keep = int(keep.sum())
        n_truth_keep = int((keep & truth_mask).sum())
        if n_keep == 0:
            enr = 0.0
        else:
            enr = (n_truth_keep / n_keep) / rate_before
        rows_2d.append(dict(
            threshold_pct=pct, n_keep=n_keep, n_truth_keep=n_truth_keep,
            decimation=n_keep / N, enrichment=enr,
        ))
        print(f"  2D-Jacobi thr={pct:3d}%: keep={n_keep:5d} truth={n_truth_keep:3d} enr={enr:.3f}")

    # --- Fine asymmetric sweep on 2D ---
    # Test thr_L = small, thr_T = small simultaneously
    print("\n2D asymmetric thresholds:")
    rows_2d_asym = []
    for thr_L_pct in [1, 2, 5]:
        for thr_T_pct in [1, 2, 5, 10]:
            keep = (rel_dL2 < thr_L_pct / 100.0) & (rel_dT < thr_T_pct / 100.0)
            n_keep = int(keep.sum())
            n_truth_keep = int((keep & truth_mask).sum())
            enr = ((n_truth_keep / n_keep) / rate_before) if n_keep else 0.0
            rows_2d_asym.append(dict(
                thr_L_pct=thr_L_pct, thr_T_pct=thr_T_pct,
                n_keep=n_keep, n_truth_keep=n_truth_keep,
                decimation=n_keep / N, enrichment=enr,
            ))
            print(f"  thr_L={thr_L_pct}%, thr_T={thr_T_pct}%: keep={n_keep:5d} "
                  f"truth={n_truth_keep:3d} enr={enr:.3f}")

    # --- Pareto best ---
    best_1d = max(rows_1d, key=lambda r: r["enrichment"])
    best_2d = max(rows_2d, key=lambda r: r["enrichment"])
    best_2d_asym = max(rows_2d_asym, key=lambda r: r["enrichment"])

    out = dict(
        seed=14,
        N_survivors=int(N),
        n_truth_in_survivors=n_truth,
        rate_before=rate_before,
        truth_pol_diam=truth_pol_diam,
        L2_truth=L2_truth,
        twoT_truth=twoT_truth,
        truth_qa_L2_range=[float(L2_truth_cands.min()), float(L2_truth_cands.max())],
        truth_qa_twoT_range=[float(twoT_truth_cands.min()), float(twoT_truth_cands.max())],
        truth_qa_L2_relspread=float((L2_truth_cands.max() - L2_truth_cands.min()) / L2_truth),
        truth_qa_twoT_relspread=float((twoT_truth_cands.max() - twoT_truth_cands.min()) / twoT_truth),
        filter_1d_poldiam=rows_1d,
        filter_2d_jacobi_LT_symmetric=rows_2d,
        filter_2d_jacobi_LT_asymmetric=rows_2d_asym,
        best_1d_enrichment=best_1d,
        best_2d_enrichment_symmetric=best_2d,
        best_2d_enrichment_asymmetric=best_2d_asym,
    )
    json_path = OUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBest 1D enr (s055d-style): {best_1d['enrichment']:.3f} at thr={best_1d['threshold_pct']}%")
    print(f"Best 2D enr (sym):         {best_2d['enrichment']:.3f} at thr={best_2d['threshold_pct']}%")
    print(f"Best 2D enr (asym):        {best_2d_asym['enrichment']:.3f} "
          f"at thr_L={best_2d_asym['thr_L_pct']}%, thr_T={best_2d_asym['thr_T_pct']}%")

    # --- Plot 2D scatter colored by truth membership ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(L2_all[~truth_mask] / L2_truth, twoT_all[~truth_mask] / twoT_truth,
               s=2, alpha=0.10, color="gray", label=f"random ({(~truth_mask).sum()})")
    ax.scatter(L2_all[truth_mask] / L2_truth, twoT_all[truth_mask] / twoT_truth,
               s=20, color="crimson", label=f"truth-q_a ({n_truth})")
    ax.axhline(1, color="black", lw=0.5)
    ax.axvline(1, color="black", lw=0.5)
    ax.set_xlabel("|L|² / |L|²_truth")
    ax.set_ylabel("2T / 2T_truth")
    ax.set_title("s055d pool: (|L|², 2T) joint distribution (seed 14, log axes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 30)
    ax.set_ylim(0.05, 30)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    enrs_1d = [r["enrichment"] for r in rows_1d]
    decs_1d = [r["decimation"] for r in rows_1d]
    enrs_2d = [r["enrichment"] for r in rows_2d]
    decs_2d = [r["decimation"] for r in rows_2d]
    ax.plot(decs_1d, enrs_1d, "o-", label="1D pol_diam (s055d)", lw=1.5, ms=6)
    ax.plot(decs_2d, enrs_2d, "s-", label="2D Jacobi (|L|, 2T)", lw=1.5, ms=6)
    enrs_a = [r["enrichment"] for r in rows_2d_asym]
    decs_a = [r["decimation"] for r in rows_2d_asym]
    ax.scatter(decs_a, enrs_a, color="tab:green", marker="x",
               s=80, label="2D asym", zorder=5)
    ax.axhline(1, color="gray", ls=":")
    ax.axhline(3, color="crimson", ls="--", label="3× gate")
    ax.set_xlabel("Decimation factor (n_keep / N)")
    ax.set_ylabel("Truth enrichment")
    ax.set_title("Truth enrichment vs decimation (seed 14)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("s063d — Jacobi (|L|, 2T) polhode-identity filter vs s055d scalar pol_diam",
                 fontsize=12)
    fig.tight_layout()
    plot_path = OUT_DIR / "polhode_identity_filter.png"
    fig.savefig(str(plot_path), dpi=130)
    plt.close(fig)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
