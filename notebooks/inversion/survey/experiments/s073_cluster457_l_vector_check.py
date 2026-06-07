"""s073 — Does L_J2000 discriminate cluster_457 from truth on seed 89?

Background
----------
Post-fix s069 surfaced cluster id=457 as the single Band A multi-solution
attractor on seed 89 (q0_err=59.58°, ω_dir_err=25.05°, hi-fi ρ=0.90).
After the s066/s067 propagator fix, real torque-free dynamics conserves
L_J2000 = R(q).T @ I @ ω exactly. Two candidates that produce different
L_J2000 at any epoch represent physically distinct trajectories — and
because conservation is exact post-fix, comparing L_J2000 at t=0 is
sufficient.

Question (gating)
-----------------
If |L_J2000(cluster_457) − L_J2000(truth)| / |L_J2000(truth)| is large
(say, > 5%), L-conservation cross-anchor matching DISCRIMINATES the
Band A multi-sol class from truth, and the L-conservation architecture
(s062d candidate) is worth designing. If the difference is small
(<< noise floor), L-matching is dead weight against this class.

What we compute
---------------
For both states at t=0:
    L_J2000 = R_J2000_to_body(q0).T @ (I_body @ ω0)

Saves
-----
  results/s073/summary.json — load-bearing numbers
  results/s073/l_vectors.npz — both L_J2000 vectors + diagnostics

Convention
----------
Quaternions scalar-first (w, x, y, z). `_quat_to_matrix(q)` is the
post-fix conv-(a) passive J2000→body matrix. L_J2000 = R.T @ I @ ω.
Same formula s066 used (`experiments/s066_lj2000_nonconservation.py::measure_drift`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.jacobi_propagator import _quat_to_matrix  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402


RESULTS_DIR = SURVEY_ROOT / "results" / "s073"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def l_j2000(q_wxyz: np.ndarray, omega_rad: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    R = _quat_to_matrix(q_wxyz)        # passive J2000 -> body
    L_body = inertia @ omega_rad
    return R.T @ L_body                # body -> J2000


def main() -> int:
    seed = 89

    # --- Truth state (t=0, body-frame) ---
    truth = load_truth(seed)
    q0_truth = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    om0_truth = np.asarray(truth["omega0_rad"], dtype=np.float64)
    ctx = build_context(seed)
    I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)

    # --- Cluster 457 polished state (from s059k_full_lc on post-fix data) ---
    #   results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json
    s059k_summary = json.load(open(
        SURVEY_ROOT / "results" / "s059k_nd800_seed89" / "seed089"
        / "full_lc_seeds" / "summary.json"
    ))
    cluster_entries = [e for e in s059k_summary["polished"] if e["cluster_id"] == 457]
    if not cluster_entries:
        print("FAIL: cluster_id=457 not found in s059k full_lc_seeds summary", file=sys.stderr)
        return 1
    # The same cluster appears with multiple mag_pct_offset starts; take the
    # mag_pct_offset=0.0 row as the canonical polished endpoint (matches s069 writeup).
    entry = next(e for e in cluster_entries if e["mag_pct_offset"] == 0.0)
    q0_c457 = np.asarray(entry["q0_pol_wxyz"], dtype=np.float64)
    om0_c457 = np.asarray(entry["om0_pol_rad"], dtype=np.float64)
    rho_c457 = float(entry["rho_polished_hifi"])
    band_c457 = entry["band"]
    q0_err_c457 = float(entry["q0_err_deg"])

    # --- L_J2000 at t=0 ---
    L_truth = l_j2000(q0_truth, om0_truth, I)
    L_c457 = l_j2000(q0_c457, om0_c457, I)

    L_truth_mag = float(np.linalg.norm(L_truth))
    L_c457_mag = float(np.linalg.norm(L_c457))

    # Body-frame |L| (invariant under rotation; sanity check)
    L_body_truth_mag = float(np.linalg.norm(I @ om0_truth))
    L_body_c457_mag = float(np.linalg.norm(I @ om0_c457))

    # Discrimination metrics
    dL = L_c457 - L_truth
    dL_mag = float(np.linalg.norm(dL))
    rel_dL = dL_mag / L_truth_mag

    # Direction angle (cluster 457 L vs truth L, both in J2000)
    cos_ang = float(np.clip(L_c457 @ L_truth / (L_c457_mag * L_truth_mag), -1.0, 1.0))
    L_dir_angle_deg = float(np.degrees(np.arccos(cos_ang)))

    # Magnitude-only diff (could match in direction but differ in |L|)
    L_mag_rel_diff = float(abs(L_c457_mag - L_truth_mag) / L_truth_mag)

    summary = {
        "experiment": "s073",
        "seed": seed,
        "cluster_id": 457,
        "cluster_band_postfix": band_c457,
        "cluster_rho_hifi": rho_c457,
        "cluster_q0_err_deg_vs_truth": q0_err_c457,
        "inertia_diag_kg_m2": np.diag(I).tolist(),
        "inertia_offdiag_max_kg_m2": float(np.abs(I - np.diag(np.diag(I))).max()),

        "truth": {
            "q0_wxyz": q0_truth.tolist(),
            "om0_rad": om0_truth.tolist(),
            "L_J2000_vec": L_truth.tolist(),
            "L_J2000_mag_kg_m2_per_s": L_truth_mag,
            "L_body_mag_check": L_body_truth_mag,
        },
        "cluster_457": {
            "q0_wxyz": q0_c457.tolist(),
            "om0_rad": om0_c457.tolist(),
            "L_J2000_vec": L_c457.tolist(),
            "L_J2000_mag_kg_m2_per_s": L_c457_mag,
            "L_body_mag_check": L_body_c457_mag,
        },

        "discrimination": {
            "delta_L_vec": dL.tolist(),
            "delta_L_mag_abs": dL_mag,
            "delta_L_mag_relative_to_truth": rel_dL,
            "L_J2000_direction_angle_deg": L_dir_angle_deg,
            "L_J2000_magnitude_relative_diff": L_mag_rel_diff,
        },

        "gate": {
            "threshold_relative_diff": 0.05,
            "is_discriminator": rel_dL > 0.05,
            "verdict": (
                "L-conservation cross-anchor filter DISCRIMINATES cluster_457 "
                "from truth — propagation-free 3-DOF gate is worth designing "
                "as s062d candidate." if rel_dL > 0.05 else
                "L-conservation matches between cluster_457 and truth — "
                "L-matching is DEAD WEIGHT against this multi-sol Band A "
                "class. Need a different discriminator."
            ),
        },
    }

    out_json = RESULTS_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    out_npz = RESULTS_DIR / "l_vectors.npz"
    np.savez(
        out_npz,
        L_truth=L_truth, L_c457=L_c457, dL=dL,
        q0_truth=q0_truth, om0_truth=om0_truth,
        q0_c457=q0_c457, om0_c457=om0_c457,
        inertia=I,
    )

    # --- Console report ---
    print("=== s073 — cluster_457 L-vector check (seed 89, post-fix) ===")
    print(f"  truth q0_err=0 (by definition);  cluster_457 q0_err={q0_err_c457:.2f}°")
    print(f"  cluster_457 hi-fi ρ={rho_c457:.3f} (band {band_c457})")
    print()
    print(f"  inertia diag (kg m²): {np.diag(I)}")
    print(f"  inertia off-diag max abs: {summary['inertia_offdiag_max_kg_m2']:.3e}")
    print()
    print(f"  |L_J2000 truth |        = {L_truth_mag:.6e} kg m² / s")
    print(f"  |L_J2000 c457  |        = {L_c457_mag:.6e} kg m² / s")
    print(f"  |Δ L_J2000     |        = {dL_mag:.6e} kg m² / s")
    print(f"  |Δ L| / |L_truth|       = {rel_dL:.4%}")
    print(f"  L direction angle       = {L_dir_angle_deg:.3f}°")
    print(f"  ||L|_c457 − |L|_truth|/|L|_truth = {L_mag_rel_diff:.4%}")
    print()
    print(f"  verdict: {summary['gate']['verdict']}")
    print()
    print(f"Saved: {out_json}")
    print(f"Saved: {out_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
