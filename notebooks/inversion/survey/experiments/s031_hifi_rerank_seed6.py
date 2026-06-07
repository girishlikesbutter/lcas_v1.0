"""s031 — surrogate-MSE rank + hi-fi rerank survivors per relaxation level on seed 6.

Two stages in one script.

Stage A (single process, default torch threading): for each candidate in the
union of L0..L3 survivors, evaluate the surrogate at the candidate's
(k1_body, k2_body) trajectory (Δ-factorisation: q_full[e] = Δ(e) ⊗ q0).
Compute surrogate-MSE vs `mag_hifi` from the cached truth NPZ. Per level,
pick top-min(level_count, 2000) by surrogate-MSE.

Stage B (Pool(8), BLAS=1 + torch_threads=1 in workers): hi-fi render the
deduplicated union of per-level top-N candidates via lib.hifi_render.
Compute ρ = sqrt(MSE) / 0.05 against truth mag_hifi. Report per-level
ρ-band breakdown (A: ρ<2, B: 2≤ρ<4, C: 4≤ρ<8, D: ρ≥8) and best (q0, ω)
per level with q0_err, ω-dir-err, ω-mag-err to truth and twin.

Cached input artefacts (from s020 + s030):
  results/s020/seed006/{candidates_meta.npz, delta_trajectories.npz,
                         omega_grid.npz, survivor_diagnostics.npz}
  results/s030/seed006/survivors_per_level.npz

Output:
  results/s031/seed006/{
    surrogate_mse_union.npz, hifi_rho_union.npz, summary.json
  }
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
if str(SURVEY_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S020_DIR = SURVEY_DIR / "results" / "s020" / "seed006"
S030_DIR = SURVEY_DIR / "results" / "s030" / "seed006"
OUT_DIR = SURVEY_DIR / "results" / "s031" / "seed006"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = [
    ("L0_strict_1p0_1p0", 1.0, 1.0),
    ("L1_relaxed_0p5_6of7", 0.5, 6.0 / 7.0),
    ("L2_relaxed_0p5_0p5", 0.5, 0.5),
    ("L3_align_only_6of7", 0.0, 6.0 / 7.0),
]

# TOP_N_CAP: nominal user request was 2000 per level. Pool(8) hi-fi at ~5-9s
# amortised per render and 4 levels would be 6-10 hours wall worst-case (4×2000
# pre-dedup ≈ 4-6k unique deduped). We instead run Stage A first (surrogate-MSE
# on the full union, sub-second per cand), then choose a level-N that keeps
# Stage B within a tractable wall budget. The default 500 with Stage-A-driven
# narrowing typically yields ≤1.5k unique hi-fi targets (~2 hr wall).
#
# Override via CLI flag --top-n-cap.
TOP_N_CAP = 500
N_POOL_WORKERS = 8
RHO_NOISE_SIGMA = 0.05
SURROGATE_MSE_BAND_C_THRESHOLD = 0.16   # surrogate-MSE for predicted ρ < 8
SURROGATE_MSE_BAND_B_THRESHOLD = 0.04   # surrogate-MSE for predicted ρ < 4
SURROGATE_MSE_BAND_A_THRESHOLD = 0.01   # surrogate-MSE for predicted ρ < 2


# ---------------------------------------------------------------------------
# Helpers (vectorised quaternion utilities, mirrored from s020).
# ---------------------------------------------------------------------------


def quat_mul_outer_left(delta_arr: np.ndarray, q0_arr: np.ndarray) -> np.ndarray:
    """LEFT outer-product q[n, m] = delta_arr[n] ⊗ q0_arr[m].

    delta_arr: (N, 4), q0_arr: (M, 4) → (N, M, 4)
    """
    dw = delta_arr[:, 0:1]; dx = delta_arr[:, 1:2]
    dy = delta_arr[:, 2:3]; dz = delta_arr[:, 3:4]
    w = q0_arr[None, :, 0]; x = q0_arr[None, :, 1]
    y = q0_arr[None, :, 2]; z = q0_arr[None, :, 3]
    return np.stack([
        dw*w - dx*x - dy*y - dz*z,
        dw*x + dx*w + dy*z - dz*y,
        dw*y - dx*z + dy*w + dz*x,
        dw*z + dx*y - dy*x + dz*w,
    ], axis=-1)


def quat_to_R_i2b_batch(q_arr_wxyz: np.ndarray) -> np.ndarray:
    qxyzw = q_arr_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1n = q1 / (np.linalg.norm(q1) + 1e-30)
    q2n = q2 / (np.linalg.norm(q2) + 1e-30)
    d = float(np.abs(np.dot(q1n, q2n)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def rho_band(rho: float) -> str:
    if rho < 2.0:
        return "A"
    if rho < 4.0:
        return "B"
    if rho < 8.0:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Stage A — surrogate-MSE on the union of survivors.
# ---------------------------------------------------------------------------


def stage_a_surrogate_mse(
    union_idx: np.ndarray,
    q0_all: np.ndarray,
    cell_idx_all: np.ndarray,
    delta_quats: np.ndarray,
    sun_unit: np.ndarray,
    obs_unit: np.ndarray,
    obs_dist: np.ndarray,
    mag_hifi_truth: np.ndarray,
) -> np.ndarray:
    """Return surrogate-MSE for each candidate in union_idx."""
    from lib.surrogate_eval import (  # noqa: WPS433 — import-on-call to defer torch
        get_model,
        DEFAULT_SP_ANGLE_DEG,
        DEFAULT_AD_ANGLE_DEG,
    )

    model = get_model()

    valid_mask = np.isfinite(mag_hifi_truth)
    n_valid = int(valid_mask.sum())
    print(f"  truth mag_hifi: {n_valid}/{mag_hifi_truth.size} finite epochs")

    union_cells = cell_idx_all[union_idx]
    union_q0 = q0_all[union_idx].astype(np.float64)
    n_union = union_idx.size
    n_obs = sun_unit.shape[0]

    mse_out = np.full(n_union, np.nan, dtype=np.float64)

    print(f"  Computing surrogate-MSE for {n_union} union candidates ...")
    print(
        f"  Grouping by ω-cell ({len(np.unique(union_cells))} unique cells)"
        f"; mean = {n_union/max(1,len(np.unique(union_cells))):.1f} cands/cell"
    )

    sort_order = np.argsort(union_cells, kind="stable")
    boundaries = np.searchsorted(
        union_cells[sort_order],
        np.arange(int(union_cells.max()) + 2),
    )

    t_start = time.time()
    cells_done = 0
    cells_total = 0

    for c in range(int(union_cells.max()) + 1):
        lo = boundaries[c]
        hi = boundaries[c + 1]
        if hi == lo:
            continue
        cells_total += 1
        local_idx = sort_order[lo:hi]
        K_c = local_idx.size
        q0_chunk = union_q0[local_idx]

        delta_q = delta_quats[c].astype(np.float64)
        q_full = quat_mul_outer_left(delta_q, q0_chunk)
        q_full_flat = q_full.reshape(n_obs * K_c, 4)
        R_flat = quat_to_R_i2b_batch(q_full_flat).reshape(n_obs, K_c, 3, 3)

        k1_body = np.einsum('eqij,ej->qei', R_flat, sun_unit)
        k2_body = np.einsum('eqij,ej->qei', R_flat, obs_unit)
        k1_flat = k1_body.reshape(K_c * n_obs, 3)
        k2_flat = k2_body.reshape(K_c * n_obs, 3)
        obs_dist_flat = np.broadcast_to(
            obs_dist[None, :], (K_c, n_obs)
        ).reshape(-1)
        mag_flat = model.predict_magnitude(
            k1_flat, k2_flat,
            DEFAULT_SP_ANGLE_DEG, DEFAULT_AD_ANGLE_DEG,
            obs_dist_flat,
        )
        mag_pred = mag_flat.reshape(K_c, n_obs)

        diff = mag_pred[:, valid_mask] - mag_hifi_truth[valid_mask][None, :]
        mse = np.mean(diff ** 2, axis=1)
        mse_out[local_idx] = mse

        cells_done += 1
        if cells_done % 100 == 0 or cells_done == cells_total:
            elapsed = time.time() - t_start
            print(
                f"    [Stage A] cell-progress {cells_done}/{cells_total}; "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"  Stage A done in {elapsed:.1f}s")
    return mse_out


# ---------------------------------------------------------------------------
# Stage B — hi-fi rerank the deduplicated top-N union.
# ---------------------------------------------------------------------------


_WORKER_CTX = None


def _hifi_pool_init(seed: int) -> None:
    global _WORKER_CTX
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch  # noqa: WPS433
        torch.set_num_threads(1)
    except Exception:
        pass
    from lib.hifi_render import build_context  # noqa: WPS433
    _WORKER_CTX = build_context(seed)


def _hifi_one(args):
    cand_id, q0, omega = args
    from lib.hifi_render import render_hifi  # noqa: WPS433
    pred = render_hifi(q0, omega, _WORKER_CTX)
    truth = _WORKER_CTX["mag_hifi_truth"]
    mask = np.isfinite(pred) & np.isfinite(truth)
    if mask.sum() == 0:
        rho = float("inf")
    else:
        diff = pred[mask] - truth[mask]
        rho = float(np.sqrt(np.mean(diff ** 2)) / RHO_NOISE_SIGMA)
    return cand_id, rho, int(mask.sum())


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("s031 — surrogate-MSE rank + hi-fi rerank for seed 6")
    print("=" * 72)

    cands = np.load(S020_DIR / "candidates_meta.npz")
    deltas = np.load(S020_DIR / "delta_trajectories.npz")
    omega_grid = np.load(S020_DIR / "omega_grid.npz")
    diag = np.load(S020_DIR / "survivor_diagnostics.npz")
    relax = np.load(S030_DIR / "survivors_per_level.npz")

    q0_all = cands["q0"]
    cell_idx_all = cands["omega_cell_idx"]
    delta_quats = deltas["delta_quats"]
    omega_vectors = omega_grid["omega_vectors"]
    truth_q0 = diag["truth_q0"]
    twin_q0 = diag["twin_q0"]
    truth_omega = diag["truth_omega"]
    twin_omega = np.array(
        [truth_omega[0], -truth_omega[1], -truth_omega[2]],
        dtype=np.float64,
    )
    truth_omega_dir = truth_omega / np.linalg.norm(truth_omega)
    twin_omega_dir = twin_omega / np.linalg.norm(twin_omega)
    truth_omega_mag = float(np.linalg.norm(truth_omega))

    level_indices = {
        "L0_strict_1p0_1p0": relax["L0_idx"],
        "L1_relaxed_0p5_6of7": relax["L1_idx"],
        "L2_relaxed_0p5_0p5": relax["L2_idx"],
        "L3_align_only_6of7": relax["L3_idx"],
    }
    union_idx = relax["union_idx"]
    print(
        f"  Per level sizes: L0={level_indices['L0_strict_1p0_1p0'].size}, "
        f"L1={level_indices['L1_relaxed_0p5_6of7'].size}, "
        f"L2={level_indices['L2_relaxed_0p5_0p5'].size}, "
        f"L3={level_indices['L3_align_only_6of7'].size}; "
        f"union={union_idx.size}"
    )

    # Per-seed inertial geometry from the cached trajectory NPZ.
    traj = np.load(SURVEY_DIR / "data" / "trajectories" / "traj_seed006.npz")
    sun_pos = traj["sun_pos"]; obs_pos = traj["obs_pos"]; sat_pos = traj["sat_pos"]
    obs_dist = traj["obs_dist"].astype(float)
    mag_hifi = traj["mag_hifi"].astype(float)
    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)

    # ------------------------------------------------------------------ STAGE A
    print("\n[Stage A] Surrogate-MSE on the 4-level UNION of survivors")
    print("-" * 72)
    surrogate_mse_union = stage_a_surrogate_mse(
        union_idx=union_idx,
        q0_all=q0_all,
        cell_idx_all=cell_idx_all,
        delta_quats=delta_quats,
        sun_unit=sun_unit,
        obs_unit=obs_unit,
        obs_dist=obs_dist,
        mag_hifi_truth=mag_hifi,
    )
    union_to_local = {int(i): k for k, i in enumerate(union_idx)}

    np.savez(
        OUT_DIR / "surrogate_mse_union.npz",
        union_idx=union_idx.astype(np.int64),
        surrogate_mse=surrogate_mse_union.astype(np.float64),
    )
    print(f"  Saved: {OUT_DIR / 'surrogate_mse_union.npz'}")

    # ---- Stage A diagnostics: predicted ρ histogram per level. Drives whether
    # to bother with full Stage B or restrict the cap.
    print("\n  [Stage A diagnostic] Predicted-ρ counts per level "
          "(predicted ρ = √surrogate-MSE / 0.05):")
    print(f"    {'level':<22} {'<2 (A)':>8} {'<4 (B)':>8} {'<8 (C)':>8} "
          f"{'<16':>8} {'level_total':>12}")
    print("    " + "-" * 70)
    for name, _, _ in LEVELS:
        idx = level_indices[name]
        if idx.size == 0:
            print(f"    {name:<22} (empty)")
            continue
        local_pos = np.array([union_to_local[int(i)] for i in idx])
        mse_local = surrogate_mse_union[local_pos]
        pred_rho = np.sqrt(mse_local) / RHO_NOISE_SIGMA
        nA = int((pred_rho < 2).sum())
        nB = int((pred_rho < 4).sum())
        nC = int((pred_rho < 8).sum())
        n16 = int((pred_rho < 16).sum())
        print(
            f"    {name:<22} {nA:>8d} {nB:>8d} {nC:>8d} {n16:>8d} {idx.size:>12d}"
        )
    print("    (A and B counts are 'predicted' — only certain after hi-fi rerank.)")

    # Per-level top-N by surrogate-MSE.
    # N = max(TOP_N_CAP, n_predicted_BandC) so we always cover all candidates
    # with predicted ρ < 8 (i.e., not guaranteed-Band-D).
    per_level_top: dict[str, np.ndarray] = {}
    for name, _, _ in LEVELS:
        idx = level_indices[name]
        if idx.size == 0:
            per_level_top[name] = np.array([], dtype=np.int64)
            continue
        local_pos = np.array([union_to_local[int(i)] for i in idx])
        mse_local = surrogate_mse_union[local_pos]
        n_pred_c = int(((mse_local < SURROGATE_MSE_BAND_C_THRESHOLD)).sum())
        order = np.argsort(mse_local, kind="stable")
        n_take = min(max(TOP_N_CAP, n_pred_c), idx.size)
        per_level_top[name] = idx[order[:n_take]]
        print(f"  {name}: take top-{n_take} (cap={TOP_N_CAP}, "
              f"n_pred_BandC={n_pred_c}, level_total={idx.size}); "
              f"min surrogate-MSE = {mse_local[order[0]]:.5f}, "
              f"min ρ_pred = {np.sqrt(mse_local[order[0]])/RHO_NOISE_SIGMA:.2f}")

    hifi_targets = np.unique(np.concatenate(list(per_level_top.values())))
    print(f"\n  Hi-fi target set: {hifi_targets.size} unique candidates "
          f"(deduped across 4 levels)")

    # ------------------------------------------------------------------ STAGE B
    print("\n[Stage B] Hi-fi rerank")
    print("-" * 72)

    # Per-target (q0, ω) for the Pool inputs.
    target_q0 = q0_all[hifi_targets].astype(np.float64)
    target_cells = cell_idx_all[hifi_targets]
    target_omegas = omega_vectors[target_cells].astype(np.float64)

    args_iter = [
        (int(hifi_targets[k]), target_q0[k], target_omegas[k])
        for k in range(hifi_targets.size)
    ]
    rho_per_target = np.full(hifi_targets.size, np.nan)
    n_finite_per_target = np.zeros(hifi_targets.size, dtype=np.int32)

    target_to_local = {int(i): k for k, i in enumerate(hifi_targets)}

    t_start = time.time()
    print(f"  Pool({N_POOL_WORKERS}) hi-fi render for {hifi_targets.size} candidates ...")
    with Pool(
        processes=N_POOL_WORKERS,
        initializer=_hifi_pool_init,
        initargs=(6,),
    ) as pool:
        progress = 0
        for cand_id, rho, n_fin in pool.imap_unordered(_hifi_one, args_iter, chunksize=4):
            k = target_to_local[cand_id]
            rho_per_target[k] = rho
            n_finite_per_target[k] = n_fin
            progress += 1
            if progress % 100 == 0 or progress == hifi_targets.size:
                elapsed = time.time() - t_start
                rate = progress / elapsed if elapsed > 0 else 0.0
                eta = (hifi_targets.size - progress) / rate if rate > 0 else 0.0
                print(
                    f"    [Stage B] {progress}/{hifi_targets.size}  "
                    f"elapsed={elapsed/60:.1f}m  rate={rate:.2f}/s  "
                    f"eta={eta/60:.1f}m",
                    flush=True,
                )
    print(f"  Stage B done in {(time.time()-t_start)/60:.1f}m")

    np.savez(
        OUT_DIR / "hifi_rho_union.npz",
        hifi_target_idx=hifi_targets.astype(np.int64),
        rho=rho_per_target.astype(np.float64),
        n_finite_epochs=n_finite_per_target,
        target_q0=target_q0.astype(np.float32),
        target_omega=target_omegas.astype(np.float64),
        target_cell_idx=target_cells.astype(np.int32),
    )
    print(f"  Saved: {OUT_DIR / 'hifi_rho_union.npz'}")

    # --------------------------------------------------------- Per-level report
    print("\n[Per-level ρ-band breakdown]\n")

    summary: dict = {
        "seed": 6,
        "n_total_candidates": int(q0_all.shape[0]),
        "rho_noise_sigma": RHO_NOISE_SIGMA,
        "top_n_cap": TOP_N_CAP,
        "n_pool_workers": N_POOL_WORKERS,
        "hifi_target_count_unique": int(hifi_targets.size),
        "levels": {},
    }

    truth_basin_idx = relax["truth_basin_idx"]
    twin_basin_idx = relax["twin_basin_idx"]

    print(
        f"  {'level':<22} {'top_N':>6} {'A(<2)':>6} {'B(2-4)':>7} "
        f"{'C(4-8)':>7} {'D(>=8)':>7} {'ρ_min':>8} {'best_q0_to_truth_deg':>22}"
    )
    print("  " + "-" * 95)
    for name, _, _ in LEVELS:
        ids = per_level_top[name]
        if ids.size == 0:
            print(f"  {name:<22} {'(empty)':>6}")
            summary["levels"][name] = {"n_top": 0}
            continue
        # Map each level-id to its rho via target_to_local.
        rhos = np.array([
            rho_per_target[target_to_local[int(i)]] for i in ids
        ])
        # Bands.
        finite = np.isfinite(rhos)
        nA = int(((rhos < 2.0) & finite).sum())
        nB = int(((rhos >= 2.0) & (rhos < 4.0) & finite).sum())
        nC = int(((rhos >= 4.0) & (rhos < 8.0) & finite).sum())
        nD = int(((rhos >= 8.0) & finite).sum())
        n_inf = int((~finite).sum())
        rho_min = float(np.nanmin(rhos)) if finite.any() else float("inf")
        # Best by ρ:
        if finite.any():
            best_idx_in_level = int(ids[np.argmin(np.where(finite, rhos, np.inf))])
            best_q0 = q0_all[best_idx_in_level].astype(np.float64)
            best_omega = omega_vectors[cell_idx_all[best_idx_in_level]]
            best_q0_to_truth = quat_geodesic_deg(best_q0, truth_q0)
            best_q0_to_twin = quat_geodesic_deg(best_q0, twin_q0)
            best_omag = float(np.linalg.norm(best_omega))
            best_omag_rel_to_truth = abs(best_omag - truth_omega_mag) / truth_omega_mag
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
            best_q0 = np.array([np.nan]*4)

        # Truth/twin basin survivors with hi-fi ρ.
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
            f"{rho_min:>8.3f} {best_q0_to_truth:>22.2f}"
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
            "best_omega_rad": best_omega.tolist(),
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

    # -------------------------------------------------- Truth/twin pool diag
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

    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()
