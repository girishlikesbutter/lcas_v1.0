"""C_t pipeline orchestrator — runs the per-epoch survival sweep and
saves an NPZ + meta.json for a given config.

Cache-aware: if a run with the same `config_id` already exists on disk,
returns its path immediately without recomputing. The Flask route
handler in `serve.py` redirects on cache hit; this module is the only
thing that touches `lib.c_t_pipeline`.

Usage:
    npz_path = compute_c_t(seed=89, surrogate='v1', n_samples=100_000,
                           sample_seed=42, tolerance_mag=0.10,
                           epoch_indices=None, progress_cb=cb,
                           epoch_spec_str='all')
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
SURROGATE_PATH = Path("/home/girish/surrogate_model")

for _p in (str(PROJECT_ROOT), str(SURVEY_DIR), str(SURROGATE_PATH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool,
    compute_j2000_units,
    project_directions,
    survive_at_epoch,
    nearest_in_pool_to_truths,
    quats_to_rotvec_wxyz,
)
from lib.traj_load import load_truth  # noqa: E402

from s048c_viewer.cache import make_config_id, run_dir  # noqa: E402

DEFAULT_SP_ANGLE_DEG = 0.0
DEFAULT_AD_ANGLE_DEG = 15.0
HAZE_SUBSAMPLE_SEED = 0  # documented; never change without invalidating cache

_MODEL_CACHE: dict[str, object] = {}


def get_compute_model(name: str):
    """Lazy-load and memoize the surrogate model. Server preloads v1 at
    startup to amortize the slow init."""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    if name == "v1":
        from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1

        weights = SURROGATE_PATH / "surrogate_model" / "s10_5M_weights.npz"
        norm = SURROGATE_PATH / "surrogate_model" / "s10_5M_normalization.npz"
        _MODEL_CACHE[name] = SurrogateV1(str(weights), str(norm))
    elif name == "v2":
        from lib.surrogate_eval import get_model  # type: ignore

        _MODEL_CACHE[name] = get_model()
    else:
        raise ValueError(f"unknown surrogate {name!r}; expected 'v1' or 'v2'")
    return _MODEL_CACHE[name]


def compute_c_t(
    seed: int,
    surrogate: str,
    n_samples: int = 100_000,
    sample_seed: int = 42,
    tolerance_mag: float = 0.10,
    epoch_indices: Iterable[int] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    epoch_spec_str: str | None = None,
    sp_angle_deg: float = DEFAULT_SP_ANGLE_DEG,
    ad_angle_deg: float = DEFAULT_AD_ANGLE_DEG,
) -> Path:
    """Run the C_t survival sweep and persist results. Returns spread.npz path.

    `progress_cb(fraction, message)` is called at named milestones (see code).
    `epoch_spec_str` is purely for meta.json provenance — it doesn't affect
    the cache id (the resolved epoch_indices already does).

    Returns the spread.npz path. Side effect: writes meta.json alongside.
    Cache-aware: if a run dir for this config already has spread.npz, returns
    its path without recomputing.
    """
    t0 = time.time()
    cb = progress_cb or (lambda *_a, **_k: None)

    cb(0.01, "loading trajectory")
    traj = load_truth(int(seed))
    n_obs = int(traj["mag_hifi"].shape[0])

    if epoch_indices is None:
        epoch_indices = np.arange(n_obs, dtype=np.int64)
    epoch_indices = np.asarray(epoch_indices, dtype=np.int64)
    n_ep = int(epoch_indices.size)

    config_id = make_config_id(
        seed, surrogate, n_samples, sample_seed, tolerance_mag, epoch_indices
    )
    out_dir = run_dir(seed, config_id)
    npz_path = out_dir / "spread.npz"

    if npz_path.exists():
        cb(1.0, "cache hit")
        return npz_path

    out_dir.mkdir(parents=True, exist_ok=True)

    cb(0.02, f"loading surrogate {surrogate}")
    model = get_compute_model(surrogate)

    cb(0.04, f"sampling {n_samples} q on SO(3)")
    pool = sample_so3_pool(int(n_samples), int(sample_seed))
    R_cache = pool["R_cache"]
    q_pool_wxyz = pool["q_pool_wxyz"]
    rotvec_pool = pool["rotvec_pool"]

    sun_unit, obs_unit = compute_j2000_units(
        traj["sun_pos"], traj["obs_pos"], traj["sat_pos"]
    )
    obs_dist_all = np.asarray(traj["obs_dist"], float)
    mag_hifi = np.asarray(traj["mag_hifi"], float)

    pred_all = np.zeros((n_ep, n_samples), dtype=np.float32)
    survive_all = np.zeros((n_ep, n_samples), dtype=bool)
    measured_at = np.zeros(n_ep, dtype=np.float64)
    n_survivors = np.zeros(n_ep, dtype=np.int64)

    cb(0.05, "starting per-epoch loop")
    for j, ep in enumerate(epoch_indices):
        ep_int = int(ep)
        k1, k2 = project_directions(R_cache, sun_unit[ep_int], obs_unit[ep_int])
        pred, keep = survive_at_epoch(
            model,
            k1,
            k2,
            obs_dist_all[ep_int],
            sp_angle_deg,
            ad_angle_deg,
            mag_hifi[ep_int],
            tolerance_mag,
        )
        pred_all[j] = pred
        survive_all[j] = keep
        measured_at[j] = mag_hifi[ep_int]
        n_survivors[j] = int(keep.sum())
        cb(0.05 + 0.85 * (j + 1) / n_ep, f"epoch {j + 1}/{n_ep}")

    cb(0.92, "computing closest-q per epoch")
    quats_truth = np.asarray(traj["quaternions"], float)
    q_truth_at = quats_truth[epoch_indices]
    rotvec_truth_at = quats_to_rotvec_wxyz(q_truth_at)
    closest_deg_per_epoch, closest_idx_per_epoch = nearest_in_pool_to_truths(
        q_pool_wxyz, q_truth_at
    )
    closest_pos_per_epoch = rotvec_pool[closest_idx_per_epoch].astype(np.float32)

    obs_times = np.asarray(traj["observation_times"], float)
    omega_mag_dps = float(traj["omega_mag_dps"])
    pe = np.asarray(traj.get("hifi_peak_epochs", np.array([], dtype=int)), int)
    pp = np.asarray(traj.get("hifi_peak_prominences", np.array([], dtype=float)), float)

    cb(0.97, "saving NPZ")
    np.savez(
        npz_path,
        q_pool_wxyz=q_pool_wxyz,
        rotvec_pool=rotvec_pool,
        epoch_indices=epoch_indices,
        obs_times=obs_times[epoch_indices],
        measured_at=measured_at,
        pred_all=pred_all,
        survive_all=survive_all,
        n_survivors=n_survivors,
        q_truth_at=q_truth_at,
        rotvec_truth_at=rotvec_truth_at,
        closest_idx_per_epoch=closest_idx_per_epoch,
        closest_deg_per_epoch=closest_deg_per_epoch,
        closest_pos_per_epoch=closest_pos_per_epoch,
        tolerance_mag=tolerance_mag,
        omega_mag_dps=omega_mag_dps,
        mag_hifi=mag_hifi,
        hifi_peak_epochs=pe,
        hifi_peak_prominences=pp,
    )

    wall = time.time() - t0
    meta = {
        "config_id": config_id,
        "seed": int(seed),
        "surrogate": str(surrogate),
        "n_samples": int(n_samples),
        "sample_seed": int(sample_seed),
        "tolerance_mag": float(tolerance_mag),
        "sp_angle_deg": float(sp_angle_deg),
        "ad_angle_deg": float(ad_angle_deg),
        "epoch_indices": epoch_indices.tolist(),
        "epoch_spec_str": epoch_spec_str,
        "n_epochs": n_ep,
        "n_obs": n_obs,
        "omega_mag_dps": omega_mag_dps,
        "wall_compute_s": wall,
        "wall_render_s": None,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "n_survivors_per_epoch": n_survivors.tolist(),
        "closest_deg_min_max": [
            float(closest_deg_per_epoch.min()),
            float(closest_deg_per_epoch.max()),
        ],
        "haze_subsample_seed": HAZE_SUBSAMPLE_SEED,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)

    cb(1.0, "compute done")
    return npz_path


if __name__ == "__main__":
    # CLI smoke: python -m s048c_viewer.compute --seed 89 --epochs 0,100,200
    import argparse

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=89)
    ap.add_argument("--surrogate", default="v1")
    ap.add_argument("--n-samples", type=int, default=100_000)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--tolerance-mag", type=float, default=0.10)
    ap.add_argument("--epochs", default="all", help="epoch spec (all / ::N / M-N / M,N,P)")
    args = ap.parse_args()

    from s048c_viewer.parser import parse_epoch_spec  # noqa: E402

    traj = load_truth(args.seed)
    eps = parse_epoch_spec(args.epochs, int(traj["mag_hifi"].shape[0]))

    def _print(frac, msg):
        print(f"[{frac:5.2f}] {msg}")

    p = compute_c_t(
        args.seed,
        args.surrogate,
        n_samples=args.n_samples,
        sample_seed=args.sample_seed,
        tolerance_mag=args.tolerance_mag,
        epoch_indices=eps,
        progress_cb=_print,
        epoch_spec_str=args.epochs,
    )
    print(f"[done] {p}")
