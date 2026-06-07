"""s031 addendum — hi-fi the 5 basin candidates that ALL relaxation levels miss.

The s030 levels (L0..L3) all require align >= 0.5. The 5 basin candidates
(2 truth, 3 twin) score align ∈ {0.286, 0.429} and so are not in the s031
union/hi-fi target set. To answer "do basin candidates ever hit Band A∪B in
hi-fi?" we render them directly.

This script is fast: 5 hi-fi renders, ~3 min Pool(8) wall.

Output: results/s031/seed006/basin_hifi.json + .npz.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY.parent.parent.parent
if str(SURVEY) not in sys.path:
    sys.path.insert(0, str(SURVEY))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S020 = SURVEY / "results" / "s020" / "seed006"
S030 = SURVEY / "results" / "s030" / "seed006"
OUT = SURVEY / "results" / "s031" / "seed006"
OUT.mkdir(parents=True, exist_ok=True)


_CTX = None


def _init():
    global _CTX
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from lib.hifi_render import build_context
    _CTX = build_context(6)


def _render(args):
    cand_id, q0, omega = args
    from lib.hifi_render import render_hifi
    pred = render_hifi(q0, omega, _CTX)
    truth = _CTX["mag_hifi_truth"]
    mask = np.isfinite(pred) & np.isfinite(truth)
    if mask.sum() == 0:
        return cand_id, float("inf"), 0
    diff = pred[mask] - truth[mask]
    return cand_id, float(np.sqrt(np.mean(diff ** 2)) / 0.05), int(mask.sum())


def main():
    cands = np.load(S020 / "candidates_meta.npz")
    omega_grid = np.load(S020 / "omega_grid.npz")
    relax = np.load(S030 / "survivors_per_level.npz")

    q0_all = cands["q0"]
    cell_idx_all = cands["omega_cell_idx"]
    omega_vectors = omega_grid["omega_vectors"]

    truth_idx = relax["truth_basin_idx"]
    twin_idx = relax["twin_basin_idx"]
    all_basin = np.concatenate([truth_idx, twin_idx]).astype(np.int64)

    print(f"  Hi-fi'ing {all_basin.size} basin candidates "
          f"(truth: {truth_idx.size}, twin: {twin_idx.size})")

    args_iter = [
        (int(c), q0_all[c].astype(np.float64),
         omega_vectors[cell_idx_all[c]].astype(np.float64))
        for c in all_basin
    ]
    rho_out = {}
    t0 = time.time()
    with Pool(processes=min(8, all_basin.size), initializer=_init) as pool:
        for cand_id, rho, n_fin in pool.imap_unordered(_render, args_iter):
            rho_out[cand_id] = {"rho": rho, "n_finite": n_fin}
            print(f"    cand {cand_id:>7d}  ρ = {rho:.4f}  ({n_fin} finite epochs)")
    print(f"  Wall: {time.time()-t0:.1f}s")

    truth_q0 = relax["truth_q0"]
    twin_q0 = relax["twin_q0"]
    truth_omega = relax["truth_omega"]
    twin_omega = relax["twin_omega"]

    summary = {
        "n_truth_basin": int(truth_idx.size),
        "n_twin_basin": int(twin_idx.size),
        "truth_q0": truth_q0.tolist(),
        "twin_q0": twin_q0.tolist(),
        "truth_omega": truth_omega.tolist(),
        "twin_omega": twin_omega.tolist(),
        "truth_basin": [
            {
                "cand_idx": int(c),
                "q0_wxyz": q0_all[c].tolist(),
                "omega_rad_s": omega_vectors[cell_idx_all[c]].tolist(),
                "rho_hifi": rho_out[int(c)]["rho"],
                "n_finite_epochs": rho_out[int(c)]["n_finite"],
            }
            for c in truth_idx
        ],
        "twin_basin": [
            {
                "cand_idx": int(c),
                "q0_wxyz": q0_all[c].tolist(),
                "omega_rad_s": omega_vectors[cell_idx_all[c]].tolist(),
                "rho_hifi": rho_out[int(c)]["rho"],
                "n_finite_epochs": rho_out[int(c)]["n_finite"],
            }
            for c in twin_idx
        ],
    }
    out_json = OUT / "basin_hifi.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_json}")

    out_npz = OUT / "basin_hifi.npz"
    rhos = np.array([rho_out[int(c)]["rho"] for c in all_basin])
    np.savez(
        out_npz,
        cand_idx=all_basin,
        rho=rhos,
        is_truth=np.isin(all_basin, truth_idx),
        is_twin=np.isin(all_basin, twin_idx),
    )
    print(f"  Saved: {out_npz}")


if __name__ == "__main__":
    main()
