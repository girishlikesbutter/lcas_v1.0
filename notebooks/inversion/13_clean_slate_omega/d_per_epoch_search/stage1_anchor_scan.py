"""stage1_anchor_scan.py — find the tightest-|C| anchor epoch on a seed.

Strategy:
    1. Rank 500 epochs by brightness (lowest magnitude first).
    2. Greedy-dedupe: skip any that's within ±DEDUPE_WINDOW of an already-picked
       epoch (same glint peak shows up as adjacent epochs — we only want one).
    3. Take the top-K unique brightest epochs.
    4. Run the 3-DOF (PAB × azimuth) q-search on all K in a SINGLE batched
       v2 call (chunked to keep peak RAM manageable).
    5. Filter each epoch's samples by |mag_pred − mag_obs| < tol, save its
       candidate attitudes.
    6. Report |C_t| for every anchor epoch; the anchor is argmin(|C_t|).

This replaces the "500-epoch sweep" idea — we only probe the handful of epochs
most likely to be tightly constrained.

Outputs to data/.../13_clean_slate_omega/d_per_epoch_search/seed{NNN}/:
    anchor_scan_summary.npz — per-anchor-epoch |C|, phi, mag_obs, truth_dist.
    anchor_scan_candidates/epoch_{t:04d}.npz — per-epoch candidate attitudes.
    anchor_scan_result.json — scalars + tightest epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Let numpy use all cores — single process, no Pool contention.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path.home() / "surrogate_model" / "surrogate_model"))

from lib.data import load_seed  # noqa: E402
from surrogate import SurrogateModel  # noqa: E402

# Reuse grid + frame math from stage1_per_epoch
from stage1_per_epoch_q_search import (  # noqa: E402
    fibonacci_sphere, per_pab_perp_basis, build_body_samples,
    construct_frame, rotmat_to_quat, quat_geodesic_deg, truth_q_per_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
            / "13_clean_slate_omega" / "d_per_epoch_search")

PANEL_DEG = 0.0
DISH_DEG = 15.0
MAX_CANDIDATES_PER_EPOCH = 5000
# Batch size (samples per v2 forward call). Tuned for ~2 GB peak feature
# intermediates — v2 MLP hidden-layer activations at fp64 float64 256-wide
# are ~2 kB/sample; 500k samples ≈ 1 GB per layer, safe headroom on 30 GB RAM.
V2_CHUNK_SIZE = 500_000


def greedy_dedupe_brightest(mag: np.ndarray, k: int, window: int = 3) -> list[int]:
    """Pick top-k brightest epoch indices, greedy-deduped so no two are within
    ±window epochs of each other.
    """
    order = np.argsort(mag)  # ascending (brightest first)
    picked: list[int] = []
    for t in order:
        t = int(t)
        if any(abs(t - p) <= window for p in picked):
            continue
        picked.append(t)
        if len(picked) >= k:
            break
    return picked


def batched_predict(model, k1_all: np.ndarray, k2_all: np.ndarray,
                    dist_all: np.ndarray, chunk: int = V2_CHUNK_SIZE) -> np.ndarray:
    """v2.predict_magnitude over possibly-very-large arrays, chunked for RAM."""
    N = k1_all.shape[0]
    out = np.empty(N, dtype=np.float64)
    t0 = time.perf_counter()
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        out[s:e] = np.asarray(model.predict_magnitude(
            k1_all[s:e], k2_all[s:e], PANEL_DEG, DISH_DEG, dist_all[s:e]
        ))
        print(f"    v2 chunk {s}-{e} of {N}: {time.perf_counter() - t0:.1f}s elapsed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pab", type=int, default=1500)
    ap.add_argument("--n-az", type=int, default=72)
    ap.add_argument("--n-anchors", type=int, default=30)
    ap.add_argument("--dedupe-window", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.15)
    args = ap.parse_args()

    seed = args.seed
    bundle = load_seed(seed)
    N_epoch = bundle["observation_times"].shape[0]
    print(f"seed={seed}  N_epoch={N_epoch}  |ω|={bundle['omega_mag_dps']:.3f} dps")
    print(f"Grid: {args.n_pab} PAB × {args.n_az} az = {args.n_pab * args.n_az}/epoch")
    print(f"Anchors: top-{args.n_anchors} brightest (±{args.dedupe_window} dedupe)")
    print(f"Tolerance: {args.tol} mag")

    anchors = greedy_dedupe_brightest(
        bundle["mag_hifi"], args.n_anchors, args.dedupe_window,
    )
    print(f"\nPicked {len(anchors)} anchor epochs:")
    for i, t in enumerate(anchors[:10]):
        print(f"  [{i}] t={t:3d}  mag={bundle['mag_hifi'][t]:.3f}")
    if len(anchors) > 10:
        print(f"  ... and {len(anchors) - 10} more")

    # Grid + truth once
    pab_verts = fibonacci_sphere(args.n_pab)
    az_angles = np.linspace(0.0, 2.0 * np.pi, args.n_az, endpoint=False)
    truth_q = truth_q_per_epoch(bundle)
    N_per_epoch = args.n_pab * args.n_az

    # Build per-epoch inertial vectors + phase
    sun = bundle["sun_j2k"] - bundle["sat_j2k"]
    obs = bundle["obs_j2k"] - bundle["sat_j2k"]
    k1_i_all = sun / np.linalg.norm(sun, axis=-1, keepdims=True)
    k2_i_all = obs / np.linalg.norm(obs, axis=-1, keepdims=True)

    # Build batched body samples across all anchor epochs
    # For each anchor t we have a phi(t); body samples depend on phi.
    # Accumulate the per-anchor sample arrays, then concatenate.
    print(f"\nBuilding batched body samples...")
    t0 = time.perf_counter()
    per_epoch_k1b = []
    per_epoch_k2b = []
    per_epoch_dist = []
    per_epoch_phi = []
    per_epoch_slice = []   # (start, end) indices into the batched array
    cursor = 0
    for t in anchors:
        k1_i = k1_i_all[t]
        k2_i = k2_i_all[t]
        phi = float(np.arccos(np.clip(k1_i @ k2_i, -1.0, 1.0)))
        k1_b, k2_b = build_body_samples(pab_verts, az_angles, phi)
        per_epoch_k1b.append(k1_b)
        per_epoch_k2b.append(k2_b)
        per_epoch_dist.append(np.full(N_per_epoch, float(bundle["obs_dist"][t]),
                                       dtype=np.float64))
        per_epoch_phi.append(phi)
        per_epoch_slice.append((cursor, cursor + N_per_epoch))
        cursor += N_per_epoch
    k1_all = np.concatenate(per_epoch_k1b, axis=0)
    k2_all = np.concatenate(per_epoch_k2b, axis=0)
    dist_all = np.concatenate(per_epoch_dist, axis=0)
    print(f"  Built {cursor} total samples across {len(anchors)} anchor epochs "
          f"in {time.perf_counter() - t0:.2f}s")

    # Load v2 once
    t0 = time.perf_counter()
    model = SurrogateModel.load_default()
    print(f"v2 load: {time.perf_counter() - t0:.2f}s")

    # Warm-up (first call has JIT / blas-init overhead)
    _ = model.predict_magnitude(k1_all[:100], k2_all[:100], PANEL_DEG, DISH_DEG,
                                 dist_all[:100])

    # BIG batched call
    print(f"\nBatched v2 forward on {cursor} samples...")
    t0 = time.perf_counter()
    mag_pred_all = batched_predict(model, k1_all, k2_all, dist_all)
    print(f"  v2 total: {time.perf_counter() - t0:.2f}s")

    # Per-epoch filter + save
    out_dir = OUT_ROOT / f"seed{seed:03d}"
    cand_dir = out_dir / "anchor_scan_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for p in cand_dir.glob("epoch_*.npz"):
        p.unlink()

    rows = []
    for i, t in enumerate(anchors):
        s, e = per_epoch_slice[i]
        mag_obs = float(bundle["mag_hifi"][t])
        resid = np.abs(mag_pred_all[s:e] - mag_obs)
        mask = resid < args.tol
        n_pass = int(mask.sum())

        row = {
            "t": int(t),
            "mag_obs": mag_obs,
            "phi_deg": float(np.degrees(per_epoch_phi[i])),
            "n_candidates": n_pass,
            "min_residual": float(resid.min()),
        }

        if n_pass == 0:
            row["best_truth_dist_deg"] = float("nan")
            row["truth_in_set"] = False
        else:
            k1_p = k1_all[s:e][mask]
            k2_p = k2_all[s:e][mask]
            resid_p = resid[mask]
            if n_pass > MAX_CANDIDATES_PER_EPOCH:
                order = np.argsort(resid_p)[:MAX_CANDIDATES_PER_EPOCH]
                k1_p = k1_p[order]; k2_p = k2_p[order]; resid_p = resid_p[order]

            F_i = construct_frame(k1_i_all[t], k2_i_all[t])
            F_b = construct_frame(k1_p, k2_p)
            R = np.einsum("mij,kj->mik", F_b, F_i)
            q = rotmat_to_quat(R)
            dists = quat_geodesic_deg(q, truth_q[t][None, :])
            best = int(np.argmin(dists))
            row["best_truth_dist_deg"] = float(dists[best])
            row["truth_in_set"] = bool(dists[best] < 5.0)

            # Save per-epoch candidates
            np.savez_compressed(
                cand_dir / f"epoch_{t:04d}.npz",
                q=q.astype(np.float32),
                residual=resid_p.astype(np.float32),
                k1_body=k1_p.astype(np.float32),
                k2_body=k2_p.astype(np.float32),
                truth_dist_deg=dists.astype(np.float32),
            )
        rows.append(row)

    # Summary
    rows_by_c = sorted(rows, key=lambda r: r["n_candidates"])
    print("\n=== anchor scan results (sorted by |C|) ===")
    print(f"{'rank':>4} {'t':>4} {'mag':>6} {'phi':>6} {'|C|':>6} "
          f"{'truth_dist°':>12} {'truth_in':>9}")
    for rank, r in enumerate(rows_by_c):
        print(f"{rank:>4} {r['t']:>4} {r['mag_obs']:>6.3f} {r['phi_deg']:>6.2f} "
              f"{r['n_candidates']:>6} {r['best_truth_dist_deg']:>12.2f} "
              f"{str(r['truth_in_set']):>9}")

    # Pick the tightest non-empty
    tight = next((r for r in rows_by_c if r["n_candidates"] > 0), None)
    if tight is None:
        print("\nWARN: no anchor epoch had any candidates! Loosen tol.")

    # Save aggregated summary
    np.savez_compressed(
        out_dir / "anchor_scan_summary.npz",
        t=np.array([r["t"] for r in rows], dtype=np.int32),
        mag_obs=np.array([r["mag_obs"] for r in rows], dtype=np.float32),
        phi_deg=np.array([r["phi_deg"] for r in rows], dtype=np.float32),
        n_candidates=np.array([r["n_candidates"] for r in rows], dtype=np.int32),
        min_residual=np.array([r["min_residual"] for r in rows], dtype=np.float32),
        best_truth_dist_deg=np.array([r["best_truth_dist_deg"] for r in rows],
                                      dtype=np.float32),
        truth_in_set=np.array([r["truth_in_set"] for r in rows], dtype=bool),
        anchors_picked=np.array(anchors, dtype=np.int32),
    )
    with open(out_dir / "anchor_scan_result.json", "w") as f:
        json.dump({
            "seed": seed,
            "grid": {"n_pab": args.n_pab, "n_az": args.n_az,
                     "samples_per_epoch": N_per_epoch},
            "n_anchors": len(anchors),
            "tol": float(args.tol),
            "tightest": tight,
            "rows_sorted_by_c": rows_by_c,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'anchor_scan_summary.npz'}")
    print(f"Saved: {out_dir / 'anchor_scan_result.json'}")
    if tight is not None:
        print(f"\nTightest anchor: t={tight['t']}  |C|={tight['n_candidates']}  "
              f"truth_dist={tight['best_truth_dist_deg']:.2f}°")


if __name__ == "__main__":
    main()
