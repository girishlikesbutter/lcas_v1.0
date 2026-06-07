"""s059d — early-anchor variant of the cloud-data → polish pipeline.

s059c (anchor-frame polish) didn't bridge on seed 28 because the residual
gradient w.r.t. ω at the anchor still has the |ω|·t_a back-prop sensitivity
baked into it (every Jacobian column rewinds from t_a to t=0). At T_A=312
on seed 28 |ω|·t_a = 1.44 dps × 2250s = 3240° accumulated rotation — the
cost surface is non-convex over the relevant scales.

The fix is structural: pick the anchor NEAR t=0 so the back-prop length is
short. For seed 28, T_A=25 (|C_t|=383, t_a=180s) shrinks the back-prop
factor 12.5× vs T_A=312. Cloud is bigger (less raw discrimination per
forward-prop hit) but the trade is the right way: cloud-data is a SEED
GENERATOR; the LM polish is what wins or loses, and short back-prop is
how LM keeps a sane Jacobian.

Reuses cached cloud + s059_pilot stages + s059c anchor-frame LM polish.
Only the anchor-pick logic changes.

Usage:
    python experiments/s059d_early_anchor.py --seed 28
    python experiments/s059d_early_anchor.py --seed 28 --anchor-window 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import (
    stage_cloud_generation, stage_forward_prop, stage_cluster,
    SURROGATE_RHO_HIFI_GATE,
)
from experiments.s059c_anchor_polish import stage_polish_anchor
from lib.hifi_render import build_context


def stage_pick_early_anchor(survive_all, log, edge_lo=3, window=30):
    """Pick T_A as argmin |C_t| within [edge_lo, edge_lo + window]."""
    cv = survive_all.sum(axis=1)
    hi = min(edge_lo + window, len(cv) - 5)
    T_A = int(edge_lo + np.argmin(cv[edge_lo:hi]))
    log(f"early anchor: search window [{edge_lo}, {hi}); "
        f"T_A = {T_A}, |C_{{T_A}}| = {cv[T_A]} "
        f"(global min |C_t| = {cv.min()} at t={int(np.argmin(cv))})")
    return T_A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--anchor-window", type=int, default=30,
                   help="search T_A in epochs [3, 3+window)")
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059d_seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir = SURVEY / "results" / f"s059_seed{args.seed:03d}"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059d early-anchor — seed {args.seed} (anchor window={args.anchor_window}) ===")

    log("\n[1/4] cloud generation (reuse s059's cache)")
    cloud = stage_cloud_generation(args.seed, cloud_dir, log)

    import experiments.s059_pilot as s059
    s059._R_CACHE = s059._SUN_UNIT = s059._OBS_UNIT = None
    s059._OBS_DIST = s059._MAG_TARGET = s059._SURROGATE = None
    import gc; gc.collect()

    log("\n[2/4] anchor + forward-prop scoring")
    ctx = build_context(seed=args.seed)
    T_A = stage_pick_early_anchor(cloud["survive_all"], log,
                                   window=args.anchor_window)
    fp = stage_forward_prop(args.seed, cloud, T_A, ctx, log)

    log("\n[3/4] canonicalise + cluster")
    cl = stage_cluster(fp, log)

    log("\n[4/4] anchor-frame LM polish + surrogate-ρ gate + hi-fi classify")
    polished = stage_polish_anchor(args.seed, fp, cl, ctx, log)

    bands = [p["band_polished_hifi"] for p in polished]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for b in bands:
        counts[b] = counts.get(b, 0) + 1
    n_AB = counts["A"] + counts["B"]
    log(f"\n=== HEADLINE: seed {args.seed} (early-anchor T_A={T_A}) ===")
    log(f"  Bands: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']} "
        f"GATED={counts['GATED']} ERR={counts['ERR']}")
    log(f"  Band A∪B yield: {n_AB}/{len(polished)} polished candidates")
    log(f"  Wall total: {(time.time()-t_overall)/60:.1f} min "
        f"({time.time()-t_overall:.0f}s)")

    summary = {
        "seed": args.seed, "T_A": T_A,
        "anchor_window": args.anchor_window,
        "n_candidates": int(len(fp["Q_A_pass"])),
        "n_clusters": int(len(cl["clusters"])),
        "truth_cluster_rank": cl["truth_cluster_rank"],
        "discrimination_ratio": float(
            fp["scores"].max() / max(fp["null_score"], 1e-6)),
        "band_counts": counts, "n_band_AB": n_AB, "n_polished": len(polished),
        "n_passed_surrogate_gate": sum(
            1 for p in polished if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE),
        "wall_total_s": time.time() - t_overall,
        "polished": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in p.items() if k != "pred_hifi"}
            for p in polished
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
