#!/usr/bin/env python3
"""Inspect the patched-m103 checkpoints (nm_prededup, multi_phi, geo) for a
single seed. Prints: pool sizes, surr_mse summary, jointly-truth-near
candidate counts and ranks per stage.

Usage:
    python3 notebooks/inversion/inspect_m103_patched.py --seed 91 --traj-source m048
"""
import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def m103_dir(seed, source):
    sub = "m103_hybrid" if source == "m046" else f"m103_hybrid_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def fmt_rank(arr_rank, n):
    return f"{arr_rank+1}/{n}"


def report_nm(d, joint_q0=60.0, joint_w=30.0):
    n = int(d["n"])
    surr_mse = d["surr_mse"] if "surr_mse" in d.files else np.full(n, np.nan)
    refined_costs = d["refined_costs"]
    q0_err = d["q0_err"]
    w0_err = d["w0_err"]
    nm_rerank_by = str(d["nm_rerank_by"]) if "nm_rerank_by" in d.files else "<legacy:align>"
    print(f"  n={n}, nm_rerank_by={nm_rerank_by}")
    finite = np.isfinite(surr_mse).sum()
    if finite > 0:
        print(f"  surr_mse: min={np.nanmin(surr_mse):.4f}  "
              f"median={np.nanmedian(surr_mse):.4f}  "
              f"finite={finite}/{n}")
    else:
        print(f"  surr_mse: ALL NAN (rerank not run)")
    align_order = np.argsort(refined_costs)
    align_rank = np.empty(n, dtype=int)
    align_rank[align_order] = np.arange(n)
    if finite > 0:
        surr_order = np.argsort(np.where(np.isfinite(surr_mse), surr_mse, np.inf))
        surr_rank = np.empty(n, dtype=int)
        surr_rank[surr_order] = np.arange(n)
    else:
        surr_rank = np.full(n, -1, dtype=int)

    joint = (q0_err < joint_q0) & (w0_err < joint_w)
    print(f"  jointly-truth-near (q0<{joint_q0} AND w<{joint_w}): {int(joint.sum())}/{n}")
    if joint.any():
        idxs = np.where(joint)[0]
        for i in idxs[:8]:
            ar = fmt_rank(align_rank[i], n)
            sr = fmt_rank(surr_rank[i], n) if finite > 0 else "-"
            print(f"    idx={i:3d}  q0={q0_err[i]:7.2f}  w={w0_err[i]:6.2f}  "
                  f"surr_mse={surr_mse[i] if np.isfinite(surr_mse[i]) else float('nan'):.4f}  "
                  f"align_rk={ar}  surr_rk={sr}")
        if len(idxs) > 8:
            print(f"    ... and {len(idxs) - 8} more")


def report_multi_phi(d, joint_q0=60.0, joint_w=30.0):
    n = int(d["n_candidates"])
    omega_ranks = d["omega_ranks"]
    phi_ranks = d["phi_ranks"]
    glint_costs = d["glint_costs"]
    q0_errs = d["q0_errs"]
    w0_errs = d["w0_errs"]
    print(f"  n_candidates={n}, n_omega_clusters={len(np.unique(omega_ranks))}")
    print(f"  glint_cost: min={glint_costs.min():.4e}  median={np.median(glint_costs):.4e}")
    joint = (q0_errs < joint_q0) & (w0_errs < joint_w)
    print(f"  jointly-truth-near (q0<{joint_q0} AND w<{joint_w}): {int(joint.sum())}/{n}")
    align_order = np.argsort(glint_costs)
    align_rank = np.empty(n, dtype=int)
    align_rank[align_order] = np.arange(n)
    if joint.any():
        idxs = np.where(joint)[0]
        # Sort joint candidates by their multi_phi index (insertion order)
        for i in idxs[:10]:
            print(f"    idx={i:3d}  w_rank={omega_ranks[i]}  phi_rank={phi_ranks[i]:2d}  "
                  f"q0={q0_errs[i]:7.2f}  w={w0_errs[i]:6.2f}  "
                  f"glint={glint_costs[i]:.4e}  align_rk={fmt_rank(align_rank[i], n)}")
        if len(idxs) > 10:
            print(f"    ... and {len(idxs) - 10} more")
    print(f"\n  Top-5 by glint_cost ascending:")
    print(f"  {'rank':>4}  {'idx':>4}  {'w_rk':>4}  {'phi_rk':>6}  "
          f"{'q0_err':>8}  {'w_err':>7}  {'glint':>10}")
    for new_rk in range(min(5, n)):
        i = align_order[new_rk]
        print(f"  {new_rk+1:>4d}  {i:>4d}  {omega_ranks[i]:>4d}  {phi_ranks[i]:>6d}  "
              f"{q0_errs[i]:>8.2f}  {w0_errs[i]:>7.2f}  {glint_costs[i]:>10.4e}")


def report_geo(d, joint_q0=60.0, joint_w=30.0):
    n = int(d["n_candidates"])
    omega_ranks = d["omega_ranks"]
    phi_ranks = d["phi_ranks"]
    geo_costs = d["geo_costs"]
    q0_ref_errs = d["q0_ref_errs"]
    w0_ref_errs = d["w0_ref_errs"]
    print(f"  n_candidates={n}, n_omega_clusters={len(np.unique(omega_ranks))}")
    print(f"  geo_cost: min={geo_costs.min():.4e}  median={np.median(geo_costs):.4e}")
    joint = (q0_ref_errs < joint_q0) & (w0_ref_errs < joint_w)
    print(f"  jointly-truth-near (q0<{joint_q0} AND w<{joint_w}): {int(joint.sum())}/{n}")
    geo_order = np.argsort(geo_costs)
    geo_rank = np.empty(n, dtype=int)
    geo_rank[geo_order] = np.arange(n)
    if joint.any():
        idxs = np.where(joint)[0]
        for i in idxs[:10]:
            print(f"    idx={i:3d}  w_rank={omega_ranks[i]}  phi_rank={phi_ranks[i]:2d}  "
                  f"q0={q0_ref_errs[i]:7.2f}  w={w0_ref_errs[i]:6.2f}  "
                  f"geo={geo_costs[i]:.4e}  geo_rk={fmt_rank(geo_rank[i], n)}")
        if len(idxs) > 10:
            print(f"    ... and {len(idxs) - 10} more")
    print(f"\n  Top-10 by geo_cost ascending (this is what m115 K consumes):")
    print(f"  {'geo_rk':>6}  {'idx':>4}  {'w_rk':>4}  {'phi_rk':>6}  "
          f"{'q0_err':>8}  {'w_err':>7}  {'geo':>10}")
    for new_rk in range(min(10, n)):
        i = geo_order[new_rk]
        print(f"  {new_rk+1:>6d}  {i:>4d}  {omega_ranks[i]:>4d}  {phi_ranks[i]:>6d}  "
              f"{q0_ref_errs[i]:>8.2f}  {w0_ref_errs[i]:>7.2f}  {geo_costs[i]:>10.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    ap.add_argument("--joint-q0", type=float, default=60.0,
                    help="q0_err threshold for joint-truth-near (default 60)")
    ap.add_argument("--joint-w", type=float, default=30.0,
                    help="w_err threshold for joint-truth-near (default 30)")
    args = ap.parse_args()

    seed_dir = m103_dir(args.seed, args.traj_source)
    print(f"=== seed {args.seed} ({args.traj_source}) ===")
    print(f"dir: {seed_dir}\n")

    nm_path = seed_dir / "nm_prededup_ckpt.npz"
    mp_path = seed_dir / "multi_phi_ckpt.npz"
    geo_path = seed_dir / "geo_ckpt.npz"

    if nm_path.exists():
        print("--- NM-prededup ---")
        report_nm(np.load(nm_path, allow_pickle=True),
                  joint_q0=args.joint_q0, joint_w=args.joint_w)
        print()

    if mp_path.exists():
        print("--- Multi-phi ---")
        report_multi_phi(np.load(mp_path, allow_pickle=True),
                         joint_q0=args.joint_q0, joint_w=args.joint_w)
        print()

    if geo_path.exists():
        print("--- Geo (post Step 4) ---")
        report_geo(np.load(geo_path, allow_pickle=True),
                   joint_q0=args.joint_q0, joint_w=args.joint_w)
        print()
    else:
        print("--- Geo: geo_ckpt.npz NOT FOUND (step timed out or skipped) ---\n")


if __name__ == "__main__":
    main()
