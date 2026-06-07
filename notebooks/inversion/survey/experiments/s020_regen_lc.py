"""s020 — regenerate a single candidate's LC from stored (q0, ω).

Use this to inspect a reject candidate's full LC after the fact (since
non-survivor LCs are not stored — only their q0 + ω + scores).

Usage:
  python notebooks/inversion/survey/experiments/s020_regen_lc.py 6 --cand-idx 12345
  python notebooks/inversion/survey/experiments/s020_regen_lc.py 6 --smoke --cand-idx 100 --plot
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.surrogate_eval import predict as surrogate_predict
from lib.filter_costs import load_static_geometry
from lib.traj_load import load_truth
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--cand-idx", type=int, required=True,
                    help="global candidate index (0..N_total-1)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="save matplotlib LC PNG next to checkpoint")
    args = ap.parse_args()

    sub = "smoke" if args.smoke else f"seed{args.seed:03d}"
    cp_dir = SURVEY_DIR / "results" / "s020" / sub

    meta = np.load(cp_dir / "candidates_meta.npz")
    og = np.load(cp_dir / "omega_grid.npz")
    th = np.load(cp_dir / "thresholds.npz")

    N_total = meta["q0"].shape[0]
    if not (0 <= args.cand_idx < N_total):
        raise IndexError(f"cand-idx {args.cand_idx} out of range [0, {N_total})")

    q0 = meta["q0"][args.cand_idx].astype(float)
    cell_idx = int(meta["omega_cell_idx"][args.cand_idx])
    qt_idx = int(meta["q_target_idx"][args.cand_idx])
    omega = og["omega_vectors"][cell_idx]
    geo_score = float(meta["geo_score"][args.cand_idx])
    align_score = float(meta["align_score"][args.cand_idx])

    print(f"--- candidate idx {args.cand_idx} ---")
    print(f"  ω-cell idx     : {cell_idx} of {og['omega_vectors'].shape[0]}")
    print(f"  q_target idx   : {qt_idx}")
    print(f"  q0 (wxyz)      : {q0}")
    print(f"  ω (rad/s)      : {omega}  |ω|={np.linalg.norm(omega):.6f}")
    print(f"  geo_score      : {geo_score:.4f}  (threshold {float(th['geo_threshold']):.4f})")
    print(f"  align_score    : {align_score:.4f}  (threshold {float(th['align_threshold']):.4f})")

    geo_pass = geo_score >= float(th['geo_threshold'])
    align_pass = align_score >= float(th['align_threshold'])
    cat = ("passed_both" if geo_pass and align_pass
           else "passed_geo_only" if geo_pass
           else "passed_align_only" if align_pass
           else "rejected_both")
    print(f"  category       : {cat}")

    # Regenerate LC from (q0, ω)
    truth = load_truth(args.seed)
    obs_times = truth["observation_times"].astype(float)
    sun_pos = truth["sun_pos"].astype(float)
    obs_pos = truth["obs_pos"].astype(float)
    sat_pos = truth["sat_pos"].astype(float)
    obs_dist = truth["obs_dist"].astype(float)
    mag_hifi = truth["mag_hifi"].astype(float)
    geo = load_static_geometry()
    inertia = geo["inertia_tensor"]

    quats, _ = propagate_attitude(q0, omega, obs_times, mode="tumbling",
                                   inertia_tensor=inertia)
    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
    qxyzw = quats[:, [1, 2, 3, 0]]
    R_i2b = Rotation.from_quat(qxyzw).as_matrix()
    k1_body = np.einsum('nij,nj->ni', R_i2b, sun_unit)
    k2_body = np.einsum('nij,nj->ni', R_i2b, obs_unit)
    mag_pred = surrogate_predict(k1_body, k2_body, obs_dist)

    # MSE vs truth
    mask = np.isfinite(mag_pred) & np.isfinite(mag_hifi)
    mse = float(np.mean((mag_pred[mask] - mag_hifi[mask]) ** 2))
    rho = float(np.sqrt(mse) / 0.05)
    print(f"  surrogate MSE  : {mse:.6f} mag²")
    print(f"  ρ (proxy)      : {rho:.3f}  (Band {'A' if rho<2 else 'B' if rho<4 else 'C' if rho<8 else 'D'})")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(obs_times, mag_hifi, 'k-', lw=1.5, label='truth hi-fi')
        ax.plot(obs_times, mag_pred, 'r--', lw=1.0, alpha=0.8, label='candidate (surrogate)')
        ax.invert_yaxis()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("magnitude")
        ax.set_title(f"seed {args.seed} candidate {args.cand_idx} ({cat}, ρ={rho:.2f})")
        ax.legend()
        ax.grid(alpha=0.3)
        png = cp_dir / f"regen_lc_cand{args.cand_idx}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
