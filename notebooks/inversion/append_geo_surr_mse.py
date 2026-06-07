#!/usr/bin/env python3
"""Append `surr_mse` and `surr_bright_mse` fields to an existing geo_ckpt.npz
in-place. Use when m103 was run before the M103_GEO_RERANK_BY=surr_mse patch
and you don't want to re-run the full pipeline.

Same scoring chain as the inline Step 4.5 in m103_hybrid.py.

Usage:
    python3 notebooks/inversion/append_geo_surr_mse.py --seeds 91 --traj-source m048
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model")

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from lib.experiment_setup import setup_experiment  # noqa: E402
from lib.traj_source import canonical_observed_lc  # noqa: E402
from surrogate_model.surrogate import SurrogateModel  # noqa: E402

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def m103_dir(seed, source):
    sub = "m103_hybrid" if source == "m046" else f"m103_hybrid_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def append_seed(seed, source):
    print(f"=== seed {seed} ({source}) — appending surr_mse to geo_ckpt ===", flush=True)
    seed_dir = m103_dir(seed, source)
    geo_path = seed_dir / "geo_ckpt.npz"
    if not geo_path.exists():
        raise FileNotFoundError(f"geo_ckpt.npz not found at {geo_path}")
    geo = dict(np.load(geo_path, allow_pickle=True))
    n = int(geo["n_candidates"])
    q0_refs = geo["q0_refs"]
    w0_refs = geo["w0_refs"]
    print(f"  n={n}", flush=True)

    if "surr_mse" in geo and np.any(np.isfinite(geo["surr_mse"])):
        print(f"  surr_mse already populated — skipping (use --force to overwrite)")
        return

    # Setup
    if source == "m048":
        master = np.load(str(DIAG / "m048_trajectories" / "m048_trajectories.npz"),
                         allow_pickle=True)
        start_et = float(master["start_ets"][seed])
        end_time_utc = None
    else:
        master = np.load(str(DIAG / "m046_trajectories" / "m046_trajectories.npz"),
                         allow_pickle=True)
        start_et = None
        end_time_utc = "2020-02-05T11:00:00"
    obs_times = master["observation_times"][seed] if source == "m048" else master["observation_times"]
    I_tensor = master["inertia_tensor"]
    true_lc = master["mag_hifi"][seed]
    observed_lc = canonical_observed_lc(true_lc)

    print(f"  loading SPICE setup (start_et={start_et})...", flush=True)
    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           start_et=start_et, end_time_utc=end_time_utc,
                           skip_true_lc=True)
    sun_dirs = (ctx.sun_pos - ctx.sat_pos)
    sun_dirs /= np.linalg.norm(sun_dirs, axis=1, keepdims=True)
    obs_dirs = (ctx.obs_pos - ctx.sat_pos)
    obs_dirs /= np.linalg.norm(obs_dirs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    print(f"  loading surrogate...", flush=True)
    model = SurrogateModel.load_default()

    obs_valid = np.isfinite(observed_lc)
    bright_mask = obs_valid & (observed_lc < 9.0)
    surr_mse = np.full(n, np.nan, dtype=float)
    surr_bright_mse = np.full(n, np.nan, dtype=float)

    print(f"  scoring {n} geo-refined candidates...", flush=True)
    t0 = time.time()
    for i in range(n):
        q0 = q0_refs[i] / np.linalg.norm(q0_refs[i])
        w0 = w0_refs[i]
        quats, _ = propagate_attitude(q0, w0, obs_times, "tumbling", I_tensor)
        R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        k1 = np.einsum('nij,nj->ni', R_all, sun_dirs)
        k2 = np.einsum('nij,nj->ni', R_all, obs_dirs)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        pred = model.predict_magnitude(k1, k2, 0.0, 15.0, obs_dist_km)
        v = obs_valid & np.isfinite(pred)
        if v.sum() >= 10:
            surr_mse[i] = float(np.mean((pred[v] - observed_lc[v]) ** 2))
        bv = bright_mask & np.isfinite(pred)
        if bv.sum() >= 5:
            surr_bright_mse[i] = float(np.mean((pred[bv] - observed_lc[bv]) ** 2))
    print(f"  scoring done in {time.time() - t0:.1f}s; "
          f"min={np.nanmin(surr_mse):.4f}  median={np.nanmedian(surr_mse):.4f}", flush=True)

    geo["surr_mse"] = surr_mse
    geo["surr_bright_mse"] = surr_bright_mse
    geo["geo_rerank_by"] = "surr_mse"
    np.savez(geo_path, **geo)
    print(f"  re-saved {geo_path} with surr_mse / surr_bright_mse fields", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    args = ap.parse_args()
    for seed in args.seeds:
        append_seed(seed, args.traj_source)


if __name__ == "__main__":
    main()
