"""s047b — cell-filter rank match on seed 14's full s019 bracket (97 cells).

Pre-flight gate for the s047 hybrid pilot. s047a established that on s032's
5-cell bracket the cell with the highest `max_geo` matches the closest-to-
truth cell on 71.6% of seeds (top-1) / 97.5% (top-2). But s032's 5-cell
bracket has 4/5 cells at gross offset; the discriminating power may be
"elimination by clear rejection" rather than fine-grained rank quality.

s047 will use s019's bracket (median 74 cells, seed 14: 97 cells) where
many cells are 1-3% off-truth and the filter must rank them on subtler
geometric cues. This script tests that rank quality on seed 14, the
binding cohort case (|ω|=1.229 dps, 0.5% basin per s042).

Approach:
  1. Reconstruct seed 14's s019 bracket: geomspace(lo, hi, 97) where
     lo = 0.5 * min(LS peaks), hi = 2.0 * max(LS peaks). Identical to
     s019's `bracket_full_grid` for the s019 strategy.
  2. For each of 97 ω-mag cells, run s020's _process_cell worker
     (300 ω-dirs × M_q q-targets, geo + align scoring). Pool(8).
  3. Aggregate per cell: max_geo, max_align, max_geo_align_sum,
     n_pass_both.
  4. Find closest-to-truth-|ω| cell by abs offset.
  5. Rank all 97 cells by each metric; report rank position of the
     closest cell. Top-K coverage at K ∈ {1, 2, 3, 5, 10}.

Gate criterion (from s047 brief): top-K rank match must hit ≥80% on a
budget K ≤ 3 to commit s047 to the rank-then-densify architecture. If
the closest cell ranks 5–10, K can be widened. If it ranks worse than
20, stop and rethink the metric.

Cost: 97 cells × ~12s/cell on Pool(8) = ~24 min compute. Within budget.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle, find_peaks

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR / "experiments"))

# Reuse s020 internals: worker, helpers, configuration. Use a regular
# import (not importlib.util.spec_from_file_location) so the worker
# function is picklable for forkserver Pool.
import s020_seed_pipeline as s020  # noqa: E402

from lib.filter_costs import load_static_geometry, load_tier_table, assign_tier
from lib.surrogate_eval import get_model
from src.dynamics.attitude_propagator import propagate_attitude


# ---- s019 bracket parameters (mirror of s019's logic on the LS spectrum) ----
N_LS_FREQS = 4000
LS_THRESHOLD_FRAC = 0.1
LS_MIN_PEAK_DIST = 5
N_MAGS_PER_BASIS = 20  # not used here directly, but kept for parity


def s019_bracket_grid(mag_hifi, observation_times):
    """Reconstruct s019's `bracket_full_grid` from cached LC.

    Returns the full geomspace from 0.5*min_peak_omega to 2.0*max_peak_omega
    at ~5% spacing — the same geomspace s019 stores as `bracket_full_grid`.
    """
    valid = np.isfinite(mag_hifi)
    valid_lc = mag_hifi[valid]
    valid_t = observation_times[valid]

    s = -valid_lc
    s = s - np.mean(s)
    dt = float(np.median(np.diff(valid_t)))
    f_min = 1.0 / (valid_t[-1] - valid_t[0])
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, N_LS_FREQS)
    ang = 2 * np.pi * freqs
    power = lombscargle(valid_t, s, ang, normalize=True)

    pmax = float(power.max())
    idx, _ = find_peaks(power, distance=LS_MIN_PEAK_DIST,
                        height=LS_THRESHOLD_FRAC * pmax)
    if len(idx) == 0:
        raise RuntimeError("No significant LS peaks")
    peak_omegas = 2 * np.pi * freqs[idx]   # rad/s

    lo = 0.5 * peak_omegas.min()
    hi = 2.0 * peak_omegas.max()
    n_full = max(N_MAGS_PER_BASIS,
                 int(np.ceil(np.log(hi / max(lo, 1e-12)) / np.log(1.05))))
    full_grid = np.geomspace(lo, hi, n_full)
    return {
        "lo": float(lo),
        "hi": float(hi),
        "n_cells": int(n_full),
        "cells": full_grid,
        "ls_peak_omegas": np.sort(peak_omegas),
    }


def run_s047b(seed: int, out_dir: Path, n_dir: int = 300,
              smoke: bool = False, verbose: bool = True):
    t_total_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load truth ----
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
    truth_npz = np.load(traj_path)
    truth = {k: truth_npz[k] for k in truth_npz.files}
    obs_times = truth["observation_times"].astype(float)
    obs_dist = truth["obs_dist"].astype(float)
    mag_hifi = truth["mag_hifi"].astype(float)
    sun_pos = truth["sun_pos"].astype(float)
    obs_pos = truth["obs_pos"].astype(float)
    sat_pos = truth["sat_pos"].astype(float)
    pab_j2000 = truth["pab_j2000"].astype(float)
    min_ang = truth["min_ang_dist"].astype(float)
    hifi_peak_eps = truth["hifi_peak_epochs"].astype(int)
    q0_truth = truth["q0_wxyz"].astype(float)
    omega0_truth = truth["omega0_rad"].astype(float)
    omega_mag_truth = float(np.linalg.norm(omega0_truth))
    N_obs = obs_times.size

    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)

    geo = load_static_geometry()
    face_normals = geo["face_normals"]
    inertia = geo["inertia_tensor"]
    tier_table = load_tier_table()

    # ---- s019 bracket (full geomspace) ----
    bracket = s019_bracket_grid(mag_hifi, obs_times)
    bracket_cells = bracket["cells"]
    if smoke:
        # Pick 2 cells: closest to truth + farthest, for fastest sanity.
        dists = np.abs(bracket_cells - omega_mag_truth) / omega_mag_truth
        keep = np.array([int(np.argmin(dists)), int(np.argmax(dists))])
        bracket_cells = bracket_cells[keep]
    cell_offsets_pct = (bracket_cells - omega_mag_truth) / omega_mag_truth * 100
    abs_offsets_pct = np.abs(cell_offsets_pct)
    closest_cell_idx = int(np.argmin(abs_offsets_pct))
    if verbose:
        print(f"=== s047b seed {seed} ===", flush=True)
        print(f"truth |ω| = {omega_mag_truth:.6f} rad/s "
              f"({np.degrees(omega_mag_truth):.4f} dps)", flush=True)
        print(f"s019 bracket: lo={bracket['lo']:.5f}, hi={bracket['hi']:.5f}, "
              f"n_cells={bracket['n_cells']}", flush=True)
        print(f"  cell offsets to truth: min |Δ|={abs_offsets_pct.min():.3f}%, "
              f"median={np.median(abs_offsets_pct):.2f}%, "
              f"max={abs_offsets_pct.max():.1f}%", flush=True)
        print(f"  closest cell (idx in bracket): {closest_cell_idx}, "
              f"|Δ|={abs_offsets_pct[closest_cell_idx]:.3f}%", flush=True)
        print(f"smoke={smoke}, running {len(bracket_cells)} cells", flush=True)

    # ---- ω-direction grid ----
    omega_dirs = s020.fibonacci_sphere(n_dir)
    N_dir = omega_dirs.shape[0]
    truth_dir = omega0_truth / omega_mag_truth
    omega_dir_dist_to_truth = np.degrees(np.arccos(
        np.clip(omega_dirs @ truth_dir, -1, 1)
    ))
    nearest_dir_idx = int(np.argmin(omega_dir_dist_to_truth))
    nearest_dir_deg = float(omega_dir_dist_to_truth[nearest_dir_idx])
    if verbose:
        print(f"  ω-dirs: {N_dir}; nearest dir to truth: {nearest_dir_deg:.3f}° "
              f"(idx {nearest_dir_idx})", flush=True)

    # ---- q-target pool (s020 mechanics, identical) ----
    pool = s020.build_q_target_pool(truth, tier_table)
    if pool is None:
        raise RuntimeError("Zero classifiable peaks — phi-sweep cannot generate ICs.")
    M_q = pool["n_q_target"]
    n_peaks_used = pool["n_peaks_classifiable"]
    q_target_peak_eps = pool["peak_epoch_idx"]
    if verbose:
        print(f"  q-target pool: M_q={M_q}, classifiable peaks={n_peaks_used}",
              flush=True)

    # ---- Spec events ----
    is_spec_at_peak = (
        (min_ang[hifi_peak_eps] < s020.SPEC_THRESHOLD_DEG)
        & (pool["tier_at_peak"] >= 0)
    )
    spec_event_eps = hifi_peak_eps[is_spec_at_peak]
    spec_event_tier = pool["tier_at_peak"][is_spec_at_peak]
    n_spec = spec_event_eps.size
    pab_at_spec_inertial = pab_j2000[spec_event_eps]
    if verbose:
        print(f"  spec events: {n_spec}", flush=True)

    bright_mask = mag_hifi[hifi_peak_eps] < s020.BRIGHT_MAG_THRESHOLD
    bright_peak_idx_arr = hifi_peak_eps[bright_mask]
    if verbose:
        print(f"  bright peaks for align cost: {bright_peak_idx_arr.size}",
              flush=True)

    # Bring up surrogate
    _ = get_model()

    # ---- Truth's own scores (per-seed thresholds, for cat_both bookkeeping) ----
    truth_q_traj, _ = propagate_attitude(
        q0_truth, omega0_truth, obs_times, mode="tumbling",
        inertia_tensor=inertia,
    )
    truth_R_i2b = s020.quat_to_R_i2b_batch(truth_q_traj)
    truth_pab_body = np.einsum('nij,nj->ni', truth_R_i2b, pab_j2000)
    truth_pab_body_at_spec = truth_pab_body[spec_event_eps]
    truth_geo_score = float(s020.geo_cost_batch(
        truth_pab_body_at_spec[None, :, :], spec_event_eps, spec_event_tier,
        face_normals, tier_table["tier_face_idx"], s020.GEO_THRESHOLD_DEG,
    )[0])
    truth_k1 = np.einsum('nij,nj->ni', truth_R_i2b, sun_unit)
    truth_k2 = np.einsum('nij,nj->ni', truth_R_i2b, obs_unit)
    from lib.surrogate_eval import predict as surrogate_predict
    truth_mag_pred = surrogate_predict(
        truth_k1, truth_k2, obs_dist,
        s020.SP_ANGLE_DEG, s020.AD_ANGLE_DEG,
    )
    truth_align_score = float(s020.alignment_cost_one(
        truth_mag_pred, bright_peak_idx_arr, s020.ALIGN_WINDOW_EPOCHS,
        s020.BRIGHT_MAG_THRESHOLD,
    ))
    if verbose:
        print(f"  truth_geo_score   = {truth_geo_score:.4f}", flush=True)
        print(f"  truth_align_score = {truth_align_score:.4f}", flush=True)

    # ---- Worker state (mirrors s020) ----
    worker_state = {
        "obs_times": obs_times, "inertia": inertia,
        "q_target": pool["q_target"], "q_target_peak_eps": q_target_peak_eps,
        "sun_unit": sun_unit, "obs_unit": obs_unit,
        "obs_dist": obs_dist,
        "spec_event_eps": spec_event_eps,
        "spec_event_tier": spec_event_tier,
        "pab_at_spec_inertial": pab_at_spec_inertial,
        "face_normals": face_normals,
        "tier_face_idx": tier_table["tier_face_idx"],
        "geo_threshold": float(truth_geo_score),
        "align_threshold": float(truth_align_score),
        "bright_peak_idx_arr": bright_peak_idx_arr,
        "M_q": M_q, "N_obs": N_obs, "n_spec": n_spec,
        "measure_geo_fail": False,
    }

    # ---- ω-vectors per cell (cell expands to N_dir 3D vectors) ----
    # We don't aggregate per (cell, dir) — we run one cell at a time with
    # ALL N_dir ω-vectors of that cell, mirroring s020's per-cell worker.
    # Each call to _process_cell takes (cell_idx, omega_vec_at_dir0). To
    # process N_dir directions for a given mag-cell we'd have to call the
    # worker once per (mag, dir). That's what s020 does at the per-cell
    # level (1 cell = 1 ω-vector of length 3).
    #
    # For s047b we want the FILTER score aggregate per *mag-cell* (over
    # all dirs and q-targets). So construct N_mag * N_dir cell args, each
    # carrying its (mag_idx, dir_idx) so we can group later.

    n_mag = bracket_cells.size
    cell_args = []
    cell_meta = np.empty((n_mag * N_dir, 2), dtype=np.int32)
    for mi, mag in enumerate(bracket_cells):
        for di in range(N_dir):
            omega_vec = mag * omega_dirs[di]
            arg_idx = len(cell_args)
            cell_args.append((arg_idx, omega_vec))
            cell_meta[arg_idx, 0] = mi
            cell_meta[arg_idx, 1] = di
    n_total_cells = len(cell_args)
    if verbose:
        print(f"\nProcessing {n_total_cells} (mag×dir) cells "
              f"= {n_mag} mags × {N_dir} dirs", flush=True)
        print(f"  per-cell rollup: max/mean over {M_q} q-targets each "
              f"({n_total_cells * M_q:,} total candidates)", flush=True)

    # Per-(mag×dir) aggregates: tiny arrays of length n_total_cells.
    per_cell_max_geo = np.full(n_total_cells, np.nan, dtype=np.float32)
    per_cell_mean_geo = np.full(n_total_cells, np.nan, dtype=np.float32)
    per_cell_max_align = np.full(n_total_cells, np.nan, dtype=np.float32)
    per_cell_mean_align = np.full(n_total_cells, np.nan, dtype=np.float32)
    per_cell_n_pass = np.zeros(n_total_cells, dtype=np.int32)

    from multiprocessing import Pool
    t_loop = time.time()
    n_workers = 8
    print(f"  Pool({n_workers}); chunksize=4", flush=True)
    last_print = time.time()
    with Pool(n_workers, initializer=s020._pool_init,
              initargs=(worker_state,)) as p:
        results_iter = p.imap_unordered(s020._process_cell, cell_args,
                                        chunksize=4)
        done = 0
        for r in results_iter:
            ci = r["cell_idx"]
            g = r["geo_scores"]
            a = r["align_scores"]
            gf = g[np.isfinite(g)]
            af = a[np.isfinite(a)]
            per_cell_max_geo[ci] = float(gf.max()) if gf.size else np.nan
            per_cell_mean_geo[ci] = float(gf.mean()) if gf.size else np.nan
            per_cell_max_align[ci] = float(af.max()) if af.size else np.nan
            per_cell_mean_align[ci] = float(af.mean()) if af.size else np.nan
            per_cell_n_pass[ci] = int(
                (np.isfinite(g) & np.isfinite(a) &
                 (g >= truth_geo_score) & (a >= truth_align_score)).sum()
            )
            done += 1
            if (time.time() - last_print) > 30.0 or done == n_total_cells:
                elapsed = time.time() - t_loop
                rate = done / max(elapsed, 1e-6)
                eta = (n_total_cells - done) / max(rate, 1e-6)
                print(f"  {done}/{n_total_cells} "
                      f"({100*done/n_total_cells:.1f}%); "
                      f"elapsed {elapsed/60:.1f} min; "
                      f"rate {rate:.1f} cells/s; "
                      f"ETA {eta/60:.1f} min", flush=True)
                last_print = time.time()
    loop_wall = time.time() - t_loop
    print(f"\nLoop wall: {loop_wall/60:.1f} min "
          f"({n_total_cells / loop_wall:.1f} cells/s on Pool({n_workers}))",
          flush=True)

    # ---- Roll up per-mag (max/mean across all dirs of that mag) ----
    per_mag = []
    for mi in range(n_mag):
        mask = cell_meta[:, 0] == mi
        gv = per_cell_max_geo[mask]
        av = per_cell_max_align[mask]
        gv_finite = gv[np.isfinite(gv)]
        av_finite = av[np.isfinite(av)]
        per_mag.append({
            "mag_idx": mi,
            "cell_rad_s": float(bracket_cells[mi]),
            "cell_dps": float(np.degrees(bracket_cells[mi])),
            "offset_pct": float(cell_offsets_pct[mi]),
            "abs_offset_pct": float(abs_offsets_pct[mi]),
            "max_geo": float(gv_finite.max()) if gv_finite.size else float("nan"),
            "mean_geo": float(per_cell_mean_geo[mask].mean()),
            "max_align": float(av_finite.max()) if av_finite.size else float("nan"),
            "mean_align": float(per_cell_mean_align[mask].mean()),
            "n_pass_both": int(per_cell_n_pass[mask].sum()),
        })

    # ---- Rank analysis ----
    metrics = {
        "max_geo": [c["max_geo"] for c in per_mag],
        "max_align": [c["max_align"] for c in per_mag],
        "max_geo_align_sum": [c["max_geo"] + c["max_align"] for c in per_mag],
        "n_pass_both": [c["n_pass_both"] for c in per_mag],
    }
    closest_idx = closest_cell_idx if not smoke else int(np.argmin(abs_offsets_pct))
    rank_report = {}
    for name, vals in metrics.items():
        order = sorted(range(n_mag), key=lambda i: vals[i], reverse=True)
        rank_of_closest = order.index(closest_idx) + 1   # 1-based
        rank_report[name] = {
            "rank_of_closest_cell": rank_of_closest,
            "value_at_closest": float(vals[closest_idx]),
            "value_at_top1": float(vals[order[0]]),
            "top5_offsets_pct": [
                float(per_mag[i]["abs_offset_pct"]) for i in order[:5]
            ],
            "ordered_indices_top10": order[:10],
        }

    # ---- Print summary ----
    print(f"\n=== HEADLINES ===", flush=True)
    print(f"closest-to-truth cell idx: {closest_idx}, "
          f"offset = {abs_offsets_pct[closest_idx]:.3f}%", flush=True)
    print(f"\nrank of closest-to-truth cell under each metric:", flush=True)
    for name, rep in rank_report.items():
        print(f"  {name:>20}: rank {rep['rank_of_closest_cell']:>3d} / {n_mag} "
              f"  (value@closest={rep['value_at_closest']:.4f} | "
              f"@top1={rep['value_at_top1']:.4f})", flush=True)

    # Top-K coverage table
    print(f"\nTop-K coverage of closest cell under each metric:", flush=True)
    for K in [1, 2, 3, 5, 10]:
        line = f"  K={K:>2d}:  "
        for name in metrics:
            r = rank_report[name]["rank_of_closest_cell"]
            mark = "Y" if r <= K else "."
            line += f"{name}={mark}({r})  "
        print(line, flush=True)

    # ---- Save artefacts (small only — we don't keep the full M_q-per-cell
    # candidate scores; per-(mag,dir) and per-mag rollups are sufficient
    # for s047b's gating question). ----
    np.savez_compressed(out_dir / "per_mag_aggregate.npz",
                        bracket_cells_rad=bracket_cells,
                        cell_offsets_pct=cell_offsets_pct,
                        abs_offsets_pct=abs_offsets_pct,
                        closest_cell_idx=np.int32(closest_idx),
                        max_geo=np.array([c["max_geo"] for c in per_mag]),
                        mean_geo=np.array([c["mean_geo"] for c in per_mag]),
                        max_align=np.array([c["max_align"] for c in per_mag]),
                        mean_align=np.array([c["mean_align"] for c in per_mag]),
                        n_pass_both=np.array([c["n_pass_both"] for c in per_mag]),
                        truth_omega_mag_rad=omega_mag_truth)

    np.savez_compressed(out_dir / "per_mag_dir_aggregate.npz",
                        cell_meta=cell_meta,         # (n_total, 2): [mag_idx, dir_idx]
                        max_geo=per_cell_max_geo,
                        mean_geo=per_cell_mean_geo,
                        max_align=per_cell_max_align,
                        mean_align=per_cell_mean_align,
                        n_pass_both=per_cell_n_pass,
                        bracket_cells_rad=bracket_cells,
                        omega_dirs=omega_dirs,
                        truth_dir=truth_dir,
                        omega_mag_truth=omega_mag_truth)

    summary = {
        "seed": seed,
        "config": {
            "smoke": bool(smoke),
            "N_OMEGA_DIR": int(N_dir),
            "N_PHI_STEPS": int(s020.N_PHI_STEPS),
            "BRIGHT_MAG_THRESHOLD": float(s020.BRIGHT_MAG_THRESHOLD),
            "GEO_THRESHOLD_DEG": float(s020.GEO_THRESHOLD_DEG),
        },
        "truth": {
            "omega_mag_rad_s": float(omega_mag_truth),
            "omega_mag_dps": float(np.degrees(omega_mag_truth)),
        },
        "bracket": {
            "lo_rad_s": float(bracket["lo"]),
            "hi_rad_s": float(bracket["hi"]),
            "n_cells": int(n_mag),
            "closest_cell_idx": int(closest_idx),
            "closest_cell_offset_pct": float(abs_offsets_pct[closest_idx]),
            "min_offset_pct": float(abs_offsets_pct.min()),
            "median_offset_pct": float(np.median(abs_offsets_pct)),
            "max_offset_pct": float(abs_offsets_pct.max()),
        },
        "thresholds": {
            "truth_geo_score": float(truth_geo_score),
            "truth_align_score": float(truth_align_score),
        },
        "scale": {
            "M_q_target": int(M_q),
            "n_total_cell_args": int(n_total_cells),
            "n_total_candidates": int(n_total_cells * M_q),
        },
        "rank_report": rank_report,
        "timing": {
            "loop_wall_s": float(loop_wall),
            "total_wall_s": float(time.time() - t_total_start),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}", flush=True)
    print(f"Saved: {out_dir / 'per_mag_aggregate.npz'}", flush=True)
    print(f"Saved: {out_dir / 'per_mag_dir_aggregate.npz'}", flush=True)
    print(f"\nTotal wall: {(time.time() - t_total_start)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=14)
    p.add_argument("--n-dir", type=int, default=300)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny mode: 2 mag-cells (closest + farthest from truth)")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    if args.out_dir is None:
        out_dir = (SURVEY_DIR / "results"
                   / "s047b_seed14_full_bracket_cell_filter"
                   / ("smoke" if args.smoke else f"seed{args.seed:03d}"))
    else:
        out_dir = Path(args.out_dir)

    print(f"=== s047b | seed={args.seed} | n_dir={args.n_dir} | "
          f"smoke={args.smoke} | out={out_dir} ===", flush=True)
    run_s047b(args.seed, out_dir, n_dir=args.n_dir, smoke=args.smoke,
              verbose=True)


if __name__ == "__main__":
    main()
