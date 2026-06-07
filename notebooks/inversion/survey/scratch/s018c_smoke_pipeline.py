"""Pipeline smoke for s018c — seed 6 only, reduced grid, validates end-to-end."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR / "experiments"))


def main():
    import s018c_phi_sweep_pilot as s018c
    # Override config for smoke
    s018c.PILOT_SEEDS = [6]
    s018c.N_PHI = 6
    s018c.N_OMEGA_DIR = 6
    s018c.N_OMEGA_MAG = 2
    s018c.OMEGA_MAG_BINS_DPS = np.geomspace(0.1, 1.5, s018c.N_OMEGA_MAG)
    s018c.TOP_K_LM = 32
    s018c.MAX_NFEV = 60
    s018c.main()


if __name__ == "__main__":
    main()
