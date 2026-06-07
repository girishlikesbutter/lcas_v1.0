"""stage1b_anchor_intersect.py — multi-anchor intersection analysis.

Reads the per-epoch candidate NPZs produced by stage1_anchor_scan.py and
computes the intersection of candidate sets across the top-N tightest
anchor epochs. No new forward-model compute — pure post-processing.

The hypothesis: a single tight anchor can be a "false positive" — its
candidate set may be compact but miss truth (we saw t=98 behave this way).
But any attitude consistent with TWO or more independent tight anchors
(within an angular tolerance) must be much closer to truth, because
independent anchors' false-positive basins generally don't line up.

Output: per-intersection size + truth survival analysis.

Usage:
    python stage1b_anchor_intersect.py --seed 0 \
        --anchors 400 313 22 204 \
        --angle-tol-deg 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import itertools

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lib.data import load_seed  # noqa: E402
from stage1_per_epoch_q_search import (  # noqa: E402
    construct_frame, rotmat_to_quat, quat_geodesic_deg, truth_q_per_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
            / "13_clean_slate_omega" / "d_per_epoch_search")


def quat_all_pairwise_dist_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Pairwise geodesic angle (deg) between rows of qa (Ma,4) and qb (Mb,4).
    Returns (Ma, Mb)."""
    dots = np.abs(qa @ qb.T)
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def intersect_pair(qa: np.ndarray, qb: np.ndarray, tol_deg: float) -> np.ndarray:
    """Indices of qa whose min-distance to qb is < tol_deg."""
    if len(qa) == 0 or len(qb) == 0:
        return np.zeros(0, dtype=int)
    D = quat_all_pairwise_dist_deg(qa, qb)
    min_dist = D.min(axis=1)
    return np.where(min_dist < tol_deg)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchors", type=int, nargs="+",
                    help="epoch indices to intersect (overrides auto-pick)")
    ap.add_argument("--top-n-by-c", type=int, default=5,
                    help="if --anchors not given: use the top-N tightest-|C| "
                         "anchors from anchor_scan_summary.npz")
    ap.add_argument("--angle-tol-deg", type=float, default=5.0)
    args = ap.parse_args()

    bundle = load_seed(args.seed)
    truth_q = truth_q_per_epoch(bundle)

    seed_dir = OUT_ROOT / f"seed{args.seed:03d}"
    cand_dir = seed_dir / "anchor_scan_candidates"

    # Pick anchors
    if args.anchors:
        anchors = list(args.anchors)
    else:
        summary = np.load(seed_dir / "anchor_scan_summary.npz")
        t = summary["t"]; n_c = summary["n_candidates"]
        mask_nonzero = n_c > 0
        order = np.argsort(n_c[mask_nonzero])
        top_idx = np.where(mask_nonzero)[0][order][:args.top_n_by_c]
        anchors = t[top_idx].tolist()

    print(f"seed={args.seed}   anchors = {anchors}")
    print(f"angle tolerance = {args.angle_tol_deg}°")

    # Load per-anchor candidate sets
    sets = []
    for a in anchors:
        p = cand_dir / f"epoch_{a:04d}.npz"
        if not p.exists():
            print(f"  [warn] missing {p} — skipping")
            continue
        d = np.load(p)
        q = np.asarray(d["q"], dtype=np.float64)
        sets.append({
            "t": int(a),
            "q": q,
            "truth_dist": float(quat_geodesic_deg(q, truth_q[a][None, :]).min()),
            "n": int(len(q)),
        })
    if not sets:
        print("No candidate sets found — run stage1_anchor_scan first.")
        return

    print(f"\nPer-anchor:")
    print(f"{'t':>4} {'|C|':>6} {'truth_dist°':>12}")
    for s in sets:
        print(f"{s['t']:>4} {s['n']:>6} {s['truth_dist']:>12.2f}")

    # Pairwise intersections
    print(f"\nPairwise intersections (candidates in anchor i within {args.angle_tol_deg}° of some candidate in anchor j):")
    print(f"{'i(t)':>5} {'j(t)':>5} {'|Ci|':>6} {'|Cj|':>6} {'|Ci∩Cj|':>9} {'i_in_j%':>8} {'truth_in_∩':>10}")
    pair_results = []
    for (i, j) in itertools.combinations(range(len(sets)), 2):
        si, sj = sets[i], sets[j]
        idx_in = intersect_pair(si["q"], sj["q"], args.angle_tol_deg)
        idx_ji = intersect_pair(sj["q"], si["q"], args.angle_tol_deg)
        # Truth survives the intersection if truth is within angle_tol of
        # BOTH si and sj (i.e. each set has a candidate within tol of truth).
        truth_in_i = si["truth_dist"] < args.angle_tol_deg
        truth_in_j = sj["truth_dist"] < args.angle_tol_deg
        truth_in_int = truth_in_i and truth_in_j
        pct = 100.0 * len(idx_in) / max(len(si["q"]), 1)
        print(f"{si['t']:>5} {sj['t']:>5} {si['n']:>6} {sj['n']:>6} "
              f"{len(idx_in):>9} {pct:>7.1f}% {str(truth_in_int):>10}")
        pair_results.append({
            "i": si["t"], "j": sj["t"],
            "n_i": si["n"], "n_j": sj["n"],
            "n_i_in_j": int(len(idx_in)),
            "n_j_in_i": int(len(idx_ji)),
            "truth_in_intersection": bool(truth_in_int),
        })

    # Full N-way intersection
    if len(sets) >= 2:
        # Running intersection: keep candidates of set[0] that have a neighbor
        # within tol in every other set.
        ref = sets[0]["q"]
        keep = np.ones(len(ref), dtype=bool)
        for s_other in sets[1:]:
            idx = intersect_pair(ref, s_other["q"], args.angle_tol_deg)
            mask = np.zeros(len(ref), dtype=bool)
            mask[idx] = True
            keep &= mask
        surviving = ref[keep]
        truth_min = float(quat_geodesic_deg(surviving, truth_q[sets[0]["t"]][None, :]).min()) if len(surviving) > 0 else float("nan")
        print(f"\nFull {len(sets)}-way intersection (from anchor t={sets[0]['t']}'s perspective):")
        print(f"  surviving: {int(keep.sum())}/{len(ref)} candidates")
        print(f"  min-truth-dist among survivors: {truth_min:.2f}°")
        truth_in_all = all(s['truth_dist'] < args.angle_tol_deg for s in sets)
        print(f"  truth within {args.angle_tol_deg}° of EVERY anchor's set: {truth_in_all}")

    # Save
    out = {
        "seed": args.seed,
        "anchors": anchors,
        "angle_tol_deg": float(args.angle_tol_deg),
        "per_anchor": [{"t": s["t"], "n": s["n"], "truth_dist": s["truth_dist"]}
                        for s in sets],
        "pairs": pair_results,
        "full_intersection_size": int(keep.sum()) if len(sets) >= 2 else None,
        "full_intersection_truth_dist_deg": truth_min if len(sets) >= 2 else None,
    }
    out_path = seed_dir / "stage1b_intersect.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
