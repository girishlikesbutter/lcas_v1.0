"""s031b — Stage-B-only hi-fi rerank, restart with proper BLAS=1 env setup.

The original `s031_hifi_rerank_seed6.py` was killed because each Pool worker
was using ~400% CPU (BLAS env vars set in worker init are too late;
OpenBLAS / OpenMP thread pools are initialised at numpy/torch import time,
not at runtime). Stage A had already saved
`results/s031/seed006/surrogate_mse_union.npz`, so we resume Stage B from
that cache.

This file MUST set OMP/OPENBLAS/MKL env vars BEFORE any numpy/torch import.
"""

from __future__ import annotations

# ─── BEFORE ANY IMPORT THAT TRANSITIVELY PULLS NUMPY / TORCH ──────────────
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# ──────────────────────────────────────────────────────────────────────────

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# Use fork (not forkserver) so workers INHERIT the master's already-built
# satellite + BRDF + inertia. Forkserver = each worker re-imports + re-loads
# = 22 min stall observed on prior s031b attempts.
_MP_CONTEXT = mp.get_context("fork")

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
if str(SURVEY_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S020 = SURVEY_DIR / "results" / "s020" / "seed006"
S030 = SURVEY_DIR / "results" / "s030" / "seed006"
OUT = SURVEY_DIR / "results" / "s031" / "seed006"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = [
    ("L0_strict_1p0_1p0", 1.0, 1.0),
    ("L1_relaxed_0p5_6of7", 0.5, 6.0 / 7.0),
    ("L2_relaxed_0p5_0p5", 0.5, 0.5),
    ("L3_align_only_6of7", 0.0, 6.0 / 7.0),
]
# TOP_N_CAP reduced from 2000 (user request) → 500 → 50 because Stage A
# diagnostic was decisive: n_pred_BandC = 0 for ALL 4 levels (min ρ_pred ≥ 28
# across the entire union of 121k survivors). Per s014 surrogate↔hi-fi
# Spearman 0.9952 outside basin, hi-fi will track surrogate; running 2000
# Band-D-predicted candidates per level just confirms what Stage A already
# said. Top-50 per level + the 5 basin extras (which all 4 levels miss
# because they score align ≤ 0.429) is sufficient to deliver the per-level
# ρ-band table and answer the binary "does any blind candidate hit Band A∪B"
# question.
TOP_N_CAP = 50
PROGRESS_EVERY = 10  # print progress every N completions
N_POOL_WORKERS = 8
RHO_NOISE_SIGMA = 0.05
SURROGATE_MSE_BAND_C_THRESHOLD = 0.16


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1n = q1 / (np.linalg.norm(q1) + 1e-30)
    q2n = q2 / (np.linalg.norm(q2) + 1e-30)
    d = float(np.abs(np.dot(q1n, q2n)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


# Build context ONCE in master; workers inherit via fork. Avoids 8× STL/BRDF
# rebuild observed in forkserver mode. NB: Pool with fork context inherits
# all in-memory state including these globals.
_MASTER_CTX = None  # populated in main() before Pool spawn


def _hifi_pool_init() -> None:
    """Per-worker init (post-fork). Env vars already inherited from master.

    Workers inherit `_MASTER_CTX` from fork — no rebuild. Just clamp torch
    threads as belt-and-braces.
    """
    try:
        import torch  # noqa: WPS433
        torch.set_num_threads(1)
    except Exception:
        pass


def _hifi_one(args):
    cand_id, q0, omega = args
    from lib.hifi_render import render_hifi  # noqa: WPS433  (cached after first call)
    pred = render_hifi(q0, omega, _MASTER_CTX)
    truth = _MASTER_CTX["mag_hifi_truth"]
    mask = np.isfinite(pred) & np.isfinite(truth)
    if mask.sum() == 0:
        rho = float("inf")
    else:
        diff = pred[mask] - truth[mask]
        rho = float(np.sqrt(np.mean(diff ** 2)) / RHO_NOISE_SIGMA)
    return cand_id, rho, int(mask.sum())


def main():
    print("=" * 72)
    print("s031b — Stage-B-only hi-fi rerank (BLAS=1, restart)")
    print("=" * 72)

    cands = np.load(S020 / "candidates_meta.npz")
    omega_grid = np.load(S020 / "omega_grid.npz")
    diag = np.load(S020 / "survivor_diagnostics.npz")
    relax = np.load(S030 / "survivors_per_level.npz")
    surr_npz = np.load(OUT / "surrogate_mse_union.npz")

    q0_all = cands["q0"]
    cell_idx_all = cands["omega_cell_idx"]
    omega_vectors = omega_grid["omega_vectors"]
    truth_q0 = diag["truth_q0"]
    twin_q0 = diag["twin_q0"]
    truth_omega = diag["truth_omega"]
    truth_omega_mag = float(np.linalg.norm(truth_omega))
    truth_omega_dir = truth_omega / truth_omega_mag
    twin_omega = np.array(
        [truth_omega[0], -truth_omega[1], -truth_omega[2]], dtype=np.float64
    )
    twin_omega_dir = twin_omega / np.linalg.norm(twin_omega)

    union_idx = surr_npz["union_idx"]
    surrogate_mse = surr_npz["surrogate_mse"]
    union_to_local = {int(i): k for k, i in enumerate(union_idx)}

    level_indices = {
        "L0_strict_1p0_1p0": relax["L0_idx"],
        "L1_relaxed_0p5_6of7": relax["L1_idx"],
        "L2_relaxed_0p5_0p5": relax["L2_idx"],
        "L3_align_only_6of7": relax["L3_idx"],
    }
    truth_basin_idx = relax["truth_basin_idx"]
    twin_basin_idx = relax["twin_basin_idx"]

    per_level_top: dict[str, np.ndarray] = {}
    print("\n[Per-level top-N selection]")
    for name, _, _ in LEVELS:
        idx = level_indices[name]
        if idx.size == 0:
            per_level_top[name] = np.array([], dtype=np.int64)
            continue
        local_pos = np.array([union_to_local[int(i)] for i in idx])
        mse_local = surrogate_mse[local_pos]
        n_pred_c = int((mse_local < SURROGATE_MSE_BAND_C_THRESHOLD).sum())
        order = np.argsort(mse_local, kind="stable")
        n_take = min(max(TOP_N_CAP, n_pred_c), idx.size)
        per_level_top[name] = idx[order[:n_take]]
        print(
            f"  {name}: take top-{n_take} (cap={TOP_N_CAP}, "
            f"n_pred_BandC={n_pred_c}, level_total={idx.size}); "
            f"min surrogate-MSE = {mse_local[order[0]]:.5f}, "
            f"min ρ_pred = {np.sqrt(mse_local[order[0]])/RHO_NOISE_SIGMA:.2f}"
        )

    hifi_targets = np.unique(np.concatenate(list(per_level_top.values())))
    print(f"\n  Hi-fi target set: {hifi_targets.size} unique candidates")

    # --- INCLUDE the 5 basin candidates that all 4 levels miss (closes
    # the s030 coverage gap — basin candidates score align ≤ 0.429 < 0.5).
    basin_extra = np.setdiff1d(
        np.concatenate([truth_basin_idx, twin_basin_idx]).astype(np.int64),
        hifi_targets,
    )
    print(f"  Basin candidates to add (truth+twin not in hi-fi set): "
          f"{basin_extra.size}")
    hifi_targets = np.unique(np.concatenate([hifi_targets, basin_extra]))
    print(f"  Hi-fi target set (after basin add): {hifi_targets.size}")

    target_q0 = q0_all[hifi_targets].astype(np.float64)
    target_cells = cell_idx_all[hifi_targets]
    target_omegas = omega_vectors[target_cells].astype(np.float64)

    args_iter = [
        (int(hifi_targets[k]), target_q0[k], target_omegas[k])
        for k in range(hifi_targets.size)
    ]
    target_to_local = {int(i): k for k, i in enumerate(hifi_targets)}
    rho_per_target = np.full(hifi_targets.size, np.nan)
    n_finite_per_target = np.zeros(hifi_targets.size, dtype=np.int32)

    # Build master context ONCE (heavy: STL + BRDF). Workers inherit via fork.
    print("\n  Building master context (satellite + BRDF + inertia + SPICE) ...")
    global _MASTER_CTX
    from lib.hifi_render import build_context  # noqa: WPS433
    t_ctx = time.time()
    _MASTER_CTX = build_context(6)
    print(f"  Master context built in {time.time()-t_ctx:.1f}s")

    t_start = time.time()
    print(f"\n  Pool({N_POOL_WORKERS}) hi-fi render for "
          f"{hifi_targets.size} candidates ...")
    with _MP_CONTEXT.Pool(
        processes=N_POOL_WORKERS,
        initializer=_hifi_pool_init,
    ) as pool:
        progress = 0
        for cand_id, rho, n_fin in pool.imap_unordered(
            _hifi_one, args_iter, chunksize=4
        ):
            k = target_to_local[cand_id]
            rho_per_target[k] = rho
            n_finite_per_target[k] = n_fin
            progress += 1
            if progress % PROGRESS_EVERY == 0 or progress == hifi_targets.size:
                elapsed = time.time() - t_start
                rate = progress / elapsed if elapsed > 0 else 0.0
                eta = (hifi_targets.size - progress) / rate if rate > 0 else 0.0
                print(
                    f"    [Stage B] {progress}/{hifi_targets.size}  "
                    f"elapsed={elapsed/60:.1f}m  rate={rate:.2f}/s  "
                    f"eta={eta/60:.1f}m",
                    flush=True,
                )
    print(f"\n  Stage B done in {(time.time()-t_start)/60:.1f}m")

    np.savez(
        OUT / "hifi_rho_union.npz",
        hifi_target_idx=hifi_targets.astype(np.int64),
        rho=rho_per_target.astype(np.float64),
        n_finite_epochs=n_finite_per_target,
        target_q0=target_q0.astype(np.float32),
        target_omega=target_omegas.astype(np.float64),
        target_cell_idx=target_cells.astype(np.int32),
        basin_added=basin_extra.astype(np.int64),
    )
    print(f"  Saved: {OUT / 'hifi_rho_union.npz'}")

    # ----- Per-level ρ-band breakdown ----------------------------------
    summary: dict = {
        "seed": 6,
        "n_total_candidates": int(q0_all.shape[0]),
        "rho_noise_sigma": RHO_NOISE_SIGMA,
        "top_n_cap": TOP_N_CAP,
        "n_pool_workers": N_POOL_WORKERS,
        "hifi_target_count_unique": int(hifi_targets.size),
        "n_basin_added": int(basin_extra.size),
        "levels": {},
    }

    print("\n[Per-level ρ-band breakdown]\n")
    print(
        f"  {'level':<22} {'top_N':>6} {'A(<2)':>6} {'B(2-4)':>7} "
        f"{'C(4-8)':>7} {'D(>=8)':>7} {'ρ_min':>8} "
        f"{'best_q→truth°':>15}"
    )
    print("  " + "-" * 88)

    for name, _, _ in LEVELS:
        ids = per_level_top[name]
        if ids.size == 0:
            print(f"  {name:<22} (empty)")
            summary["levels"][name] = {"n_top": 0}
            continue
        rhos = np.array([
            rho_per_target[target_to_local[int(i)]] for i in ids
        ])
        finite = np.isfinite(rhos)
        nA = int(((rhos < 2.0) & finite).sum())
        nB = int(((rhos >= 2.0) & (rhos < 4.0) & finite).sum())
        nC = int(((rhos >= 4.0) & (rhos < 8.0) & finite).sum())
        nD = int(((rhos >= 8.0) & finite).sum())
        n_inf = int((~finite).sum())
        rho_min = float(np.nanmin(rhos)) if finite.any() else float("inf")

        if finite.any():
            best_idx_in_level = int(
                ids[np.argmin(np.where(finite, rhos, np.inf))]
            )
            best_q0 = q0_all[best_idx_in_level].astype(np.float64)
            best_omega = omega_vectors[cell_idx_all[best_idx_in_level]]
            best_q0_to_truth = quat_geodesic_deg(best_q0, truth_q0)
            best_q0_to_twin = quat_geodesic_deg(best_q0, twin_q0)
            best_omag = float(np.linalg.norm(best_omega))
            best_omag_rel_to_truth = (
                abs(best_omag - truth_omega_mag) / truth_omega_mag
            )
            best_dir_to_truth = float(
                np.degrees(np.arccos(np.clip(
                    best_omega @ truth_omega_dir / best_omag, -1.0, 1.0
                )))
            )
            best_dir_to_twin = float(
                np.degrees(np.arccos(np.clip(
                    best_omega @ twin_omega_dir / best_omag, -1.0, 1.0
                )))
            )
        else:
            best_idx_in_level = -1
            best_q0_to_truth = float("nan")
            best_q0_to_twin = float("nan")
            best_omag = float("nan")
            best_omag_rel_to_truth = float("nan")
            best_dir_to_truth = float("nan")
            best_dir_to_twin = float("nan")
            best_omega = np.array([np.nan, np.nan, np.nan])
            best_q0 = np.array([np.nan] * 4)

        truth_in_top = np.intersect1d(ids, truth_basin_idx)
        twin_in_top = np.intersect1d(ids, twin_basin_idx)
        truth_rhos = (
            np.array([rho_per_target[target_to_local[int(i)]] for i in truth_in_top])
            if truth_in_top.size else np.array([])
        )
        twin_rhos = (
            np.array([rho_per_target[target_to_local[int(i)]] for i in twin_in_top])
            if twin_in_top.size else np.array([])
        )

        print(
            f"  {name:<22} {ids.size:>6d} {nA:>6d} {nB:>7d} {nC:>7d} {nD:>7d} "
            f"{rho_min:>8.3f} {best_q0_to_truth:>15.2f}"
        )
        summary["levels"][name] = {
            "n_top": int(ids.size),
            "n_band_A": nA,
            "n_band_B": nB,
            "n_band_C": nC,
            "n_band_D": nD,
            "n_rho_inf": n_inf,
            "rho_min": rho_min,
            "best_cand_idx": best_idx_in_level,
            "best_q0_wxyz": best_q0.tolist(),
            "best_omega_rad": best_omega.tolist() if not isinstance(best_omega, list) else best_omega,
            "best_q0_err_to_truth_deg": best_q0_to_truth,
            "best_q0_err_to_twin_deg": best_q0_to_twin,
            "best_omega_mag_rel_err_truth": best_omag_rel_to_truth,
            "best_omega_dir_err_truth_deg": best_dir_to_truth,
            "best_omega_dir_err_twin_deg": best_dir_to_twin,
            "truth_basin_in_top_n": int(truth_in_top.size),
            "twin_basin_in_top_n": int(twin_in_top.size),
            "truth_basin_rhos": truth_rhos.tolist(),
            "twin_basin_rhos": twin_rhos.tolist(),
        }

    # ------- Basin-candidate hi-fi (always included) -------------------
    print("\n[Basin candidate hi-fi — added separately to hi-fi target set]")
    print(f"  {'tag':<6} {'cand':>7} {'q→ref°':>8} {'|dω-mag|':>10} "
          f"{'ω-dir-deg':>10} {'ρ_hifi':>10} {'band':>5}")
    basin_summary: list[dict] = []
    for tag, basin in [("TRUTH", truth_basin_idx), ("TWIN", twin_basin_idx)]:
        for c in basin:
            c = int(c)
            local = target_to_local.get(c)
            if local is None:
                continue
            rho_v = rho_per_target[local]
            q0_c = q0_all[c].astype(np.float64)
            omg = omega_vectors[cell_idx_all[c]]
            mag = float(np.linalg.norm(omg))
            mag_rel = abs(mag - truth_omega_mag) / truth_omega_mag
            ref_q = truth_q0 if tag == "TRUTH" else twin_q0
            ref_dir = truth_omega_dir if tag == "TRUTH" else twin_omega_dir
            q_to_ref = quat_geodesic_deg(q0_c, ref_q)
            d_to_ref = float(
                np.degrees(np.arccos(np.clip(omg @ ref_dir / mag, -1, 1)))
            )
            band = "A" if rho_v < 2 else "B" if rho_v < 4 else "C" if rho_v < 8 else "D"
            print(
                f"  {tag:<6} {c:>7d} {q_to_ref:>8.2f} {mag_rel*100:>9.2f}% "
                f"{d_to_ref:>10.2f} {rho_v:>10.3f} {band:>5}"
            )
            basin_summary.append({
                "tag": tag,
                "cand_idx": c,
                "rho_hifi": float(rho_v),
                "q_to_ref_deg": q_to_ref,
                "omega_mag_rel": mag_rel,
                "omega_dir_to_ref_deg": d_to_ref,
                "band": band,
            })

    truth_pool_top = np.intersect1d(hifi_targets, truth_basin_idx)
    twin_pool_top = np.intersect1d(hifi_targets, twin_basin_idx)
    print(
        f"\n  TRUE truth-basin candidates in hi-fi target set: "
        f"{truth_pool_top.size}/{truth_basin_idx.size}"
    )
    print(
        f"  TRUE twin-basin  candidates in hi-fi target set: "
        f"{twin_pool_top.size}/{twin_basin_idx.size}"
    )
    summary["truth_basin_in_pool"] = int(truth_basin_idx.size)
    summary["twin_basin_in_pool"] = int(twin_basin_idx.size)
    summary["truth_basin_in_hifi_set"] = int(truth_pool_top.size)
    summary["twin_basin_in_hifi_set"] = int(twin_pool_top.size)
    summary["basin_candidate_hifi"] = basin_summary

    out_json = OUT / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()
