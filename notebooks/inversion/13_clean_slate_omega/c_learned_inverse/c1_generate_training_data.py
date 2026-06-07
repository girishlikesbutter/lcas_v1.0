"""c1_generate_training_data.py

Generate synthetic (q0, omega, geometry) -> mag_lc training data using the
v2 surrogate as the forward model.

Plan:
  - Iterate over the 100 m048 seeds' saved geometries (sun_j2k, obs_j2k,
    sat_j2k, obs_dist, observation_times). This gives real-world-like
    geometry with phase angles spanning ~20-45 deg and varied viewing.
  - For each geometry, draw N_PER_GEO random (q0, omega):
      q0 uniform on S^3.
      |omega| log-uniform in [0.5, 15] dps.
      omega axis uniform on S^2.
  - Propagate attitude (tumbling, shared m048 inertia), compute body-frame
    k1/k2, run v2 surrogate to get a 500-mag LC.
  - Save shards of 10k samples as NPZ plus a manifest JSON.

Matches the c_learned_inverse README contract. Outputs to
  data/results/inversion_diagnostics/13_clean_slate_omega/c_learned_inverse/train_data/

Parallelisation: multiprocessing.Pool(8). Each worker initialises its own
surrogate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must set BLAS thread caps BEFORE numpy/surrogate import. Each worker inherits
# these via the Pool fork. Without this, 8 workers × 16 BLAS threads thrash
# the scheduler and can hang the machine.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# Allow importing the shared lib
HERE = Path(__file__).resolve().parent
SUB = HERE.parent  # .../13_clean_slate_omega
sys.path.insert(0, str(SUB))

from lib.data import load_seed, all_seeds
from lib.forward import body_vectors_from_attitude, get_surrogate
from src.dynamics.attitude_propagator import propagate_attitude


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
           / "13_clean_slate_omega" / "c_learned_inverse" / "train_data")


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------

def sample_q0(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform on S^3 via 4-D Gaussian normalisation. Returns (n, 4) wxyz."""
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Enforce w >= 0 to kill double-cover sign ambiguity — makes regression targets nicer.
    # (We'll use the quat for propagation; both hemispheres give same rotation.)
    flip = q[:, 0] < 0
    q[flip] *= -1
    return q


def sample_omega(
    rng: np.random.Generator,
    n: int,
    mag_dps_range: tuple[float, float] = (0.5, 15.0),
) -> np.ndarray:
    """Random omega. |omega| log-uniform in given dps range; direction uniform on S^2."""
    lo, hi = np.log(mag_dps_range[0]), np.log(mag_dps_range[1])
    mag_dps = np.exp(rng.uniform(lo, hi, size=n))
    mag_rad = np.deg2rad(mag_dps)
    axis = rng.standard_normal((n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis * mag_rad[:, None]


# ---------------------------------------------------------------------------
# Geometry cache — load all 100 m048 seeds once
# ---------------------------------------------------------------------------

def load_geometry_bank():
    """Pull the geometry arrays from every m048 seed.

    Returns dict with:
      - obs_times: (N_geo, 500) relative times, t[0]=0
      - sun_j2k, obs_j2k, sat_j2k: (N_geo, 500, 3)
      - obs_dist: (N_geo, 500)
      - start_et: (N_geo,)
      - seeds: (N_geo,)
      - inertia_tensor: (3,3)  (shared across all seeds)
    """
    seeds = all_seeds()
    N = len(seeds)
    obs_times = np.empty((N, 500), dtype=np.float64)
    sun_j2k = np.empty((N, 500, 3), dtype=np.float64)
    obs_j2k = np.empty((N, 500, 3), dtype=np.float64)
    sat_j2k = np.empty((N, 500, 3), dtype=np.float64)
    obs_dist = np.empty((N, 500), dtype=np.float64)
    start_et = np.empty(N, dtype=np.float64)
    inertia = None
    for i, s in enumerate(seeds):
        d = load_seed(s)
        obs_times[i] = d["observation_times"] - d["observation_times"][0]
        sun_j2k[i] = d["sun_j2k"]
        obs_j2k[i] = d["obs_j2k"]
        sat_j2k[i] = d["sat_j2k"]
        obs_dist[i] = d["obs_dist"]
        start_et[i] = d["start_et"]
        if inertia is None:
            inertia = d["inertia_tensor"]
    return {
        "obs_times": obs_times,
        "sun_j2k": sun_j2k,
        "obs_j2k": obs_j2k,
        "sat_j2k": sat_j2k,
        "obs_dist": obs_dist,
        "start_et": start_et,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "inertia_tensor": inertia,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _init_worker(bank_path: str):
    # Load geometry bank using mmap so all workers share the kernel page cache
    # rather than each duplicating 100 × 500 × ... arrays into heap. Surrogate
    # is small (~4 MB) — it's fine to load per-worker.
    bank = np.load(bank_path, allow_pickle=False, mmap_mode="r")
    _WORKER_STATE["bank"] = {k: bank[k] for k in bank.files}
    _WORKER_STATE["model"] = get_surrogate("v2")
    _WORKER_STATE["pid"] = os.getpid()


def _worker_make_shard(args):
    (shard_id, seed_rng, n_samples, shard_out) = args
    bank = _WORKER_STATE["bank"]
    model = _WORKER_STATE["model"]
    I = bank["inertia_tensor"]

    rng = np.random.default_rng(seed_rng)
    N_geo = bank["obs_times"].shape[0]

    # Pick geometry index per sample
    geo_idx = rng.integers(0, N_geo, size=n_samples)

    q0_all = sample_q0(rng, n_samples)           # (B, 4)
    omega_all = sample_omega(rng, n_samples)     # (B, 3)

    mag_lc = np.empty((n_samples, 500), dtype=np.float32)
    # We also store the geometry per sample so training doesn't need the bank.
    sun_out = np.empty((n_samples, 500, 3), dtype=np.float32)
    obs_out = np.empty((n_samples, 500, 3), dtype=np.float32)
    dist_out = np.empty((n_samples, 500), dtype=np.float32)
    start_et_out = np.empty((n_samples,), dtype=np.float64)

    t0 = time.time()
    for i in range(n_samples):
        gi = int(geo_idx[i])
        t_rel = bank["obs_times"][gi]
        try:
            quats, _ = propagate_attitude(
                q0_all[i], omega_all[i], t_rel,
                mode="tumbling", inertia_tensor=I,
                rtol=1e-8, atol=1e-10,  # looser tol, ~3x faster than default
            )
        except Exception as e:
            # Fallback to principal-axis if ODE fails
            quats, _ = propagate_attitude(
                q0_all[i], omega_all[i], t_rel,
                mode="principal_axis",
            )
        k1b, k2b = body_vectors_from_attitude(
            quats, bank["sun_j2k"][gi], bank["obs_j2k"][gi], bank["sat_j2k"][gi],
        )
        mag = model.predict_magnitude(k1b, k2b, 0.0, 15.0, bank["obs_dist"][gi])
        mag_lc[i] = mag.astype(np.float32)
        # Store per-sample geometry (sun/obs positions relative to sat — the features the NN wants)
        sun_vec = bank["sun_j2k"][gi] - bank["sat_j2k"][gi]
        obs_vec = bank["obs_j2k"][gi] - bank["sat_j2k"][gi]
        sun_out[i] = sun_vec.astype(np.float32)
        obs_out[i] = obs_vec.astype(np.float32)
        dist_out[i] = bank["obs_dist"][gi].astype(np.float32)
        start_et_out[i] = bank["start_et"][gi]

    elapsed = time.time() - t0
    np.savez_compressed(
        shard_out,
        q0=q0_all.astype(np.float32),
        omega=omega_all.astype(np.float32),
        mag_lc=mag_lc,
        sun_j2k=sun_out,
        obs_j2k=obs_out,
        obs_dist=dist_out,
        start_et=start_et_out,
        geo_idx=geo_idx,
    )
    return {"shard_id": shard_id, "path": str(shard_out), "n": n_samples, "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 1 training data generator")
    ap.add_argument("--n-samples", type=int, default=100_000,
                    help="total number of (q0, omega) samples to generate")
    ap.add_argument("--shard-size", type=int, default=10_000,
                    help="samples per NPZ shard")
    ap.add_argument("--n-workers", type=int, default=4,
                    help="DEFAULT 4 — higher values caused a thread storm hang on 2026-04-18; do not raise without re-testing.")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--master-seed", type=int, default=20260418)
    ap.add_argument("--omega-mag-lo-dps", type=float, default=0.5)
    ap.add_argument("--omega-mag-hi-dps", type=float, default=15.0)
    ap.add_argument("--tag", type=str, default="",
                    help="optional label appended to output dir (e.g. 'pilot')")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    if args.tag:
        out_root = out_root.parent / f"{out_root.name}_{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Persist bank once so workers can mmap it
    bank = load_geometry_bank()
    bank_path = out_root / "geometry_bank.npz"
    np.savez(bank_path, **bank)
    print(f"Saved geometry bank: {bank_path}")

    # Build shard task list
    shards = []
    remaining = args.n_samples
    sid = 0
    master_rng = np.random.default_rng(args.master_seed)
    while remaining > 0:
        n = min(args.shard_size, remaining)
        seed_rng = int(master_rng.integers(0, 2**31 - 1))
        shard_out = out_root / f"shard_{sid:04d}.npz"
        shards.append((sid, seed_rng, n, shard_out))
        sid += 1
        remaining -= n

    print(f"Launching {args.n_workers} workers for {len(shards)} shards, "
          f"{args.n_samples} total samples, "
          f"omega mag range=[{args.omega_mag_lo_dps}, {args.omega_mag_hi_dps}] dps")

    manifest = {
        "n_samples": args.n_samples,
        "shard_size": args.shard_size,
        "n_shards": len(shards),
        "master_seed": args.master_seed,
        "omega_mag_range_dps": [args.omega_mag_lo_dps, args.omega_mag_hi_dps],
        "geometry_bank": str(bank_path),
        "inertia_tensor": bank["inertia_tensor"].tolist(),
        "shards": [],
        "started_at": time.time(),
    }

    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(args.n_workers, initializer=_init_worker,
                 initargs=(str(bank_path),)) as pool:
        for result in pool.imap_unordered(_worker_make_shard, shards):
            manifest["shards"].append(result)
            n_done = sum(s["n"] for s in manifest["shards"])
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (args.n_samples - n_done) / max(rate, 1e-6)
            print(f"[{n_done}/{args.n_samples}] shard {result['shard_id']} "
                  f"({result['n']} samples, {result['elapsed_s']:.1f}s) "
                  f"rate={rate:.1f}/s eta={eta/60:.1f} min")

    manifest["elapsed_s"] = time.time() - t0
    manifest["finished_at"] = time.time()
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")
    print(f"Total elapsed: {manifest['elapsed_s']/60:.1f} min")
    for s in sorted(manifest["shards"], key=lambda r: r["shard_id"]):
        print(f"  shard {s['shard_id']:04d}: {s['path']}")


if __name__ == "__main__":
    main()
