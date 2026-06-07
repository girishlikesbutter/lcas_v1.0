#!/usr/bin/env python3
"""Score the 300 lofi candidates with surrogate full-LC MSE.

Produces `lofi_surr_ckpt.npz` next to `lofi_ckpt.npz` so the pipeline_viz
cost-landscape panel can plot the same 300 candidates colored by the
surrogate cost (instead of m103's alignment cost).

Cheap (~1.5 sec/seed) — single forward surrogate evaluation per candidate.
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


def quaternion_multiply(q1, q2):
    """wxyz quaternion multiplication. q1 (4,), q2 (..., 4) -> result has q2's shape."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def score_seed(seed, source):
    print(f"=== seed {seed} ===", flush=True)
    t0 = time.time()
    seed_dir = m103_dir(seed, source)
    lofi = np.load(seed_dir / "lofi_ckpt.npz", allow_pickle=True)
    n = int(lofi["n"])
    q0_arr = lofi["q0"]            # (300, 4) wxyz
    w0_arr = lofi["w0"]            # (300, 3) rad/s

    # Master to get truth + obs window for this seed
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

    print(f"  loading SPICE setup...", flush=True)
    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           start_et=start_et, end_time_utc=end_time_utc,
                           skip_true_lc=True)
    sun_vecs = ctx.sun_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    print(f"  loading surrogate...", flush=True)
    model = SurrogateModel.load_default()

    obs_valid = np.isfinite(observed_lc)
    surr_mse = np.full(n, np.inf, dtype=np.float64)
    surr_bright_mse = np.full(n, np.inf, dtype=np.float64)
    bright_mask = obs_valid & (observed_lc < 9.0)

    print(f"  scoring {n} (q0, w0) candidates...", flush=True)
    t_score = time.time()
    for i in range(n):
        q0 = q0_arr[i]
        q0 = q0 / np.linalg.norm(q0)
        w0 = w0_arr[i]
        # Forward propagate from t=0 over the 500-epoch window.
        quats, _ = propagate_attitude(q0, w0, obs_times, "tumbling", I_tensor)
        quats_xyzw = quats[:, [1, 2, 3, 0]]
        R_all = Rotation.from_quat(quats_xyzw).as_matrix()
        k1 = np.einsum('nij,nj->ni', R_all, sun_dirs)
        k2 = np.einsum('nij,nj->ni', R_all, obs_dirs)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        pred = model.predict_magnitude(k1, k2, 0.0, 15.0, obs_dist_km)
        valid = obs_valid & np.isfinite(pred)
        if valid.sum() >= 10:
            surr_mse[i] = float(np.mean((pred[valid] - observed_lc[valid]) ** 2))
        bv = bright_mask & np.isfinite(pred)
        if bv.sum() >= 5:
            surr_bright_mse[i] = float(np.mean((pred[bv] - observed_lc[bv]) ** 2))
    elapsed = time.time() - t_score

    out_path = seed_dir / "lofi_surr_ckpt.npz"
    np.savez(out_path, n=n, surr_mse=surr_mse, surr_bright_mse=surr_bright_mse,
             elapsed_s=elapsed)
    print(f"  saved {out_path}  (min surr_mse={surr_mse.min():.4f}, "
          f"median={np.median(surr_mse):.4f}, scoring {elapsed:.1f}s)", flush=True)
    print(f"  total wall {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    args = ap.parse_args()
    for seed in args.seeds:
        score_seed(seed, args.traj_source)


if __name__ == "__main__":
    main()
