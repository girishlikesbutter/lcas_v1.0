#!/usr/bin/env python3
"""Compute extras costs from saved per-seed LC matrices (cheap, no surrogate).

Reads seed_XXX_scores.json + seed_XXX_lcmatrix.npz, adds EXTRAS costs, and
re-writes the JSON in place. Run after rerank.py completes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from costs_extras import EXTRAS

sys.path.insert(0, str(ROOT))
from notebooks.inversion.lib.traj_source import canonical_observed_lc

DIAG = ROOT / "data/results/inversion_diagnostics"
OUT = DIAG / "rerank_experiment"
M048 = DIAG / "m048_trajectories/per_trajectory"

COHORT = [6, 7, 8, 11, 16, 17, 34, 45, 47, 48, 51, 57, 59, 64, 67,
          71, 78, 79, 84, 89, 91, 99]


def main():
    for seed in COHORT:
        sj = OUT / f"seed_{seed:03d}_scores.json"
        lm = OUT / f"seed_{seed:03d}_lcmatrix.npz"
        if not (sj.exists() and lm.exists()):
            print(f"  seed {seed}: missing, skip")
            continue
        scores = json.load(open(sj))
        lcmat = np.load(lm)["lc_matrix"]
        traj = np.load(M048 / f"traj_seed{seed:03d}.npz")
        observed = canonical_observed_lc(traj["mag_hifi"].astype(np.float64))
        obs_peaks, _ = find_peaks(-observed, distance=3, prominence=0.2)
        for name, fn in EXTRAS.items():
            if name in scores["costs"]:
                continue
            col = []
            for i in range(lcmat.shape[0]):
                lc = lcmat[i]
                col.append(fn(lc, observed, obs_peaks=list(obs_peaks)))
            scores["costs"][name] = col
        with open(sj, "w") as f:
            json.dump(scores, f)
        print(f"  seed {seed}: added {list(EXTRAS.keys())}")


if __name__ == "__main__":
    main()
