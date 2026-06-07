"""s085 convention smoke test (CLAUDE.md: verify q->R before trusting geodesics).

Confirms the c_t_pipeline projection convention matches the cached truth on a
post-fix holdout seed, BEFORE s085 trusts any anchor q_geo number.

Three checks on seed 119:
  (0) Round-trip: does R(truth_q[T]) @ sun_unit_j2000[T] reproduce cached
      k1_body[T] (and obs -> k2_body)? Tries both R and R^T to disambiguate
      the active/passive convention empirically.
  (1) Does truth survive at its own epoch? (surrogate pred ~ measured mag)
  (2) Does nearest_in_pool_to_truth return ~0 when truth is inserted into pool?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions,
    survive_at_epoch, nearest_in_pool_to_truth,
)
from lib.surrogate_eval import get_model  # noqa: E402
from lib import traj_load  # noqa: E402

SEED = 119
T = 100  # arbitrary mid-LC epoch for the convention check
TOL_MAG = 0.10
SP_DEG, AD_DEG = 0.0, 15.0


def wxyz_to_xyzw(q):
    q = np.asarray(q, float)
    return q[[1, 2, 3, 0]]


def main():
    d = traj_load.load_truth(SEED)
    quats = d["quaternions"]           # (N,4) wxyz, cached truth attitude
    sun_pos, obs_pos, sat_pos = d["sun_pos"], d["obs_pos"], d["sat_pos"]
    k1_cached, k2_cached = d["k1_body"], d["k2_body"]
    obs_dist = d["obs_dist"]
    mag_meas = d["mag_hifi"]
    sun_unit, obs_unit = compute_j2000_units(sun_pos, obs_pos, sat_pos)

    # --- Check 0: round-trip convention ---
    q_xyzw = wxyz_to_xyzw(quats[T])
    R = Rotation.from_quat(q_xyzw).as_matrix()   # 3x3
    k1_R = R @ sun_unit[T]
    k1_RT = R.T @ sun_unit[T]
    err_R = float(np.linalg.norm(k1_R - k1_cached[T]))
    err_RT = float(np.linalg.norm(k1_RT - k1_cached[T]))
    print(f"=== Check 0: q->R convention (seed {SEED}, epoch {T}) ===")
    print(f"  ||R   @ sun_unit - k1_body_cached|| = {err_R:.3e}")
    print(f"  ||R^T @ sun_unit - k1_body_cached|| = {err_RT:.3e}")
    which = "R (no transpose, matches pool)" if err_R < err_RT else "R^T (pool would be WRONG)"
    print(f"  -> matching convention: {which}")
    k2_R = R @ obs_unit[T]
    k2_RT = R.T @ obs_unit[T]
    print(f"  obs: ||R@obs - k2||={np.linalg.norm(k2_R-k2_cached[T]):.3e}  "
          f"||R^T@obs - k2||={np.linalg.norm(k2_RT-k2_cached[T]):.3e}")

    # --- Check 1: does truth survive at its own epoch? ---
    model = get_model()
    R_truth = R[None]  # (1,3,3)
    k1_b, k2_b = project_directions(R_truth, sun_unit[T], obs_unit[T])
    pred, keep = survive_at_epoch(model, k1_b, k2_b, float(obs_dist[T]),
                                  SP_DEG, AD_DEG, float(mag_meas[T]), TOL_MAG)
    print(f"\n=== Check 1: truth survives at its own epoch {T} ===")
    print(f"  surrogate pred = {float(pred[0]):.4f}   measured = {float(mag_meas[T]):.4f}"
          f"   |diff| = {abs(float(pred[0])-float(mag_meas[T])):.4f}  (tol {TOL_MAG})")
    print(f"  survives: {bool(keep[0])}")

    # --- Check 2: nearest_in_pool_to_truth ~ 0 when truth is in the pool ---
    pool = sample_so3_pool(50_000, sample_seed=42)
    q_pool = np.vstack([pool["q_pool_wxyz"], quats[T][None]])  # inject truth as last row
    deg, idx = nearest_in_pool_to_truth(q_pool, quats[T])
    print(f"\n=== Check 2: nearest_in_pool_to_truth with truth injected ===")
    print(f"  nearest deg = {deg:.4e}   idx = {idx} (injected truth idx = {len(q_pool)-1})")

    ok = (err_R < 1e-9) and bool(keep[0]) and (deg < 1e-6)
    print(f"\nCONVENTION SMOKE: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
