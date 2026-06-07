"""Diagnostic: does s018c phi-sweep IC + LM converge to truth when truth-ω is
explicitly in the ω-grid? Tests whether the architecture is fundamentally sound,
isolating the ω-grid-density question.

Approach:
  Replace the coarse ω-grid with a SINGLE cell at truth-ω. Run the rest of
  the s018c pipeline. If the cohort selector picks a Band A landing
  (q0_err<5°, surr_MSE near truth_mse_ref), the IC architecture is sound and
  the failure mode is purely ω-grid density. If it still fails, something
  more fundamental is wrong.
"""

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

    # Inject truth ω as the ONLY cell in the grid
    seed = 6
    d = np.load(SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz")
    truth_omega = d['omega0_rad'].astype(float)
    print(f"Seed {seed} truth omega = {truth_omega} rad/s "
          f"(|w|={np.linalg.norm(truth_omega):.6f}={np.degrees(np.linalg.norm(truth_omega)):.4f} dps)")

    # Monkey-patch build_omega_grid to return a single-cell grid at truth
    truth_only = truth_omega.reshape(1, 3)
    s018c.build_omega_grid = lambda: truth_only

    # Override config
    s018c.PILOT_SEEDS = [seed]
    s018c.N_PHI = 12         # denser to ensure phi grab
    s018c.TOP_K_LM = 64
    s018c.MAX_NFEV = 100
    s018c.OUT_DIR = SURVEY_DIR / "results" / "s018c_truth_omega_diag"
    s018c.OUT_DIR.mkdir(parents=True, exist_ok=True)

    s018c.main()

    # Print diagnostic comparison: surrogate_MSE distribution at truth-ω
    # vs cached s001 truth_mse_ref.
    import json
    summary = json.loads((s018c.OUT_DIR / "summary.json").read_text())
    print("\n=== DIAGNOSTIC SUMMARY ===")
    for s, info in summary['per_seed'].items():
        print(f"  seed {s}: best_final_mse={info.get('best_final_mse')}")
        print(f"           best_q0_err_deg={info.get('best_q0_err_deg')}")
        print(f"           best_omega_dir_err_deg={info.get('best_omega_dir_err_deg')}")
        print(f"           best_omega_mag_err_pct={info.get('best_omega_mag_err_pct')}")
    print(f"\n  s001 cached truth_mse_ref for seed 6 = 2.92e-3 mag²")
    print(f"  Band A bar: q0_err < 5° and surr_MSE near 2.92e-3")


if __name__ == "__main__":
    main()
