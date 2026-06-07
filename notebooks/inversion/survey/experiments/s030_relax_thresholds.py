"""s030 — relax-threshold sweep on cached s020 seed-6 data.

Re-categorise the 432k cached candidates at 4 relaxation levels:
  L0: geo>=1.0  AND align>=1.0    (s020 baseline = 875)
  L1: geo>=0.5  AND align>=0.857  (mentioned in s020 writeup ~ 1864)
  L2: geo>=0.5  AND align>=0.5
  L3: geo>=0.0  AND align>=0.857  (no geo filter)

For each level report:
  - total survivor count
  - TRUE truth-basin count (q0_to_truth < 30 AND |dω-mag|/truth_mag < 0.10
    AND ω-dir to truth-dir < 10°)
  - TRUE twin-basin count (twin = q_180x · q0_truth (LEFT-multiply) +
    R_180x · ω_truth, R_180x = diag(1,-1,-1) i.e. flip y, z)

Output: results/s030/seed006/{summary.json, survivors_per_level.npz}.

The script is post-process only: no propagation, no surrogate eval, no hi-fi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

S020_DIR = SURVEY_DIR / "results" / "s020" / "seed006"
OUT_DIR = SURVEY_DIR / "results" / "s030" / "seed006"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = [
    ("L0_strict_1p0_1p0", 1.0, 1.0),
    ("L1_relaxed_0p5_6of7", 0.5, 6.0 / 7.0),
    ("L2_relaxed_0p5_0p5", 0.5, 0.5),
    ("L3_align_only_6of7", 0.0, 6.0 / 7.0),
]

BASIN_Q_DEG = 30.0
BASIN_OMEGA_MAG_REL = 0.10
BASIN_OMEGA_DIR_DEG = 10.0


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Rotation-angle geodesic (handles double cover via |dot|).

    q1: (N, 4) wxyz. q2: (4,) wxyz. Returns (N,) degrees.
    """
    q1n = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-30)
    q2n = q2 / (np.linalg.norm(q2) + 1e-30)
    dot = np.abs(q1n @ q2n)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def main() -> None:
    print("=" * 70)
    print("s030 — relax-threshold sweep on cached seed-6 data")
    print("=" * 70)

    cands = np.load(S020_DIR / "candidates_meta.npz")
    omega_grid = np.load(S020_DIR / "omega_grid.npz")
    diag = np.load(S020_DIR / "survivor_diagnostics.npz")

    q0_all = cands["q0"]
    cell_idx = cands["omega_cell_idx"]
    geo = cands["geo_score"]
    align = cands["align_score"]
    N = q0_all.shape[0]

    omega_vectors = omega_grid["omega_vectors"]
    omega_mags_grid = omega_grid["omega_mags"]
    truth_omega_mag = float(omega_grid["truth_omega_mag_rad"])
    truth_omega_dir = omega_grid["truth_omega_dir"].astype(np.float64)
    truth_omega_dir = truth_omega_dir / np.linalg.norm(truth_omega_dir)

    truth_q0 = diag["truth_q0"]
    twin_q0 = diag["twin_q0"]
    truth_omega = diag["truth_omega"]
    twin_omega = np.array(
        [truth_omega[0], -truth_omega[1], -truth_omega[2]], dtype=np.float64
    )
    twin_omega_dir = twin_omega / np.linalg.norm(twin_omega)

    print(f"  N_candidates              = {N}")
    print(f"  truth_omega_mag (rad/s)   = {truth_omega_mag:.6f}")
    print(f"  truth_q0 (wxyz)           = {truth_q0.tolist()}")
    print(f"  twin_q0 (wxyz)            = {twin_q0.tolist()}")
    print(f"  twin_omega (rad/s)        = {twin_omega.tolist()}")

    print("\n[1/4] Per-candidate omega vectors / dirs / mags ...", flush=True)
    omega_vec_per_cand = omega_vectors[cell_idx]
    omega_mag_per_cand = np.linalg.norm(omega_vec_per_cand, axis=-1)
    omega_dir_per_cand = omega_vec_per_cand / (
        omega_mag_per_cand[:, None] + 1e-30
    )

    print("[2/4] Q-geodesics (q0 -> truth, q0 -> twin) ...", flush=True)
    q_to_truth_deg = quat_geodesic_deg(q0_all.astype(np.float64), truth_q0)
    q_to_twin_deg = quat_geodesic_deg(q0_all.astype(np.float64), twin_q0)

    print("[3/4] Omega errors (mag-rel, dir-deg) to truth & twin ...", flush=True)
    omega_mag_rel_err_truth = np.abs(
        omega_mag_per_cand - truth_omega_mag
    ) / truth_omega_mag
    omega_mag_rel_err_twin = omega_mag_rel_err_truth.copy()  # twin |ω| identical

    cos_to_truth = np.clip(omega_dir_per_cand @ truth_omega_dir, -1.0, 1.0)
    omega_dir_err_truth_deg = np.degrees(np.arccos(cos_to_truth))
    cos_to_twin = np.clip(omega_dir_per_cand @ twin_omega_dir, -1.0, 1.0)
    omega_dir_err_twin_deg = np.degrees(np.arccos(cos_to_twin))

    print("[4/4] Basin masks ...", flush=True)
    truth_basin_mask = (
        (q_to_truth_deg < BASIN_Q_DEG)
        & (omega_mag_rel_err_truth < BASIN_OMEGA_MAG_REL)
        & (omega_dir_err_truth_deg < BASIN_OMEGA_DIR_DEG)
    )
    twin_basin_mask = (
        (q_to_twin_deg < BASIN_Q_DEG)
        & (omega_mag_rel_err_twin < BASIN_OMEGA_MAG_REL)
        & (omega_dir_err_twin_deg < BASIN_OMEGA_DIR_DEG)
    )

    print(
        f"\n  TRUE truth-basin candidates in IC pool: {int(truth_basin_mask.sum())}"
    )
    print(
        f"  TRUE twin-basin candidates in IC pool : {int(twin_basin_mask.sum())}"
    )

    print("\n" + "=" * 70)
    print("Per-level survivor counts")
    print("=" * 70)
    header = (
        f"  {'level':<22} {'geo_thr':>8} {'align_thr':>10} "
        f"{'survivors':>10} {'truth_basin':>12} {'twin_basin':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    per_level: dict = {}
    surv_idx_by_level: dict = {}
    for name, geo_thr, align_thr in LEVELS:
        mask = (geo >= geo_thr) & (align >= align_thr)
        n_surv = int(mask.sum())
        n_truth = int((mask & truth_basin_mask).sum())
        n_twin = int((mask & twin_basin_mask).sum())
        idx = np.where(mask)[0].astype(np.int64)
        per_level[name] = {
            "geo_threshold": geo_thr,
            "align_threshold": align_thr,
            "n_survivors": n_surv,
            "n_truth_basin": n_truth,
            "n_twin_basin": n_twin,
        }
        surv_idx_by_level[name] = idx
        print(
            f"  {name:<22} {geo_thr:>8.4f} {align_thr:>10.4f} "
            f"{n_surv:>10d} {n_truth:>12d} {n_twin:>12d}"
        )

    union_idx = np.unique(
        np.concatenate([surv_idx_by_level[n] for n, _, _ in LEVELS])
    )
    print(f"\n  UNION across 4 levels: {union_idx.size} unique survivor candidates")

    print("\n[ Truth/twin pool details (out of 432000) ]")
    truth_idx = np.where(truth_basin_mask)[0]
    twin_idx = np.where(twin_basin_mask)[0]
    for tag, idx in [("TRUTH", truth_idx), ("TWIN", twin_idx)]:
        print(f"\n  {tag} basin candidates:")
        for k in idx:
            print(
                f"    cand {k:>7d}: q_to_{tag.lower():<5} = "
                f"{(q_to_truth_deg if tag=='TRUTH' else q_to_twin_deg)[k]:6.2f}°  "
                f"|dω-mag|/truth = {omega_mag_rel_err_truth[k]*100:5.2f}%  "
                f"ω-dir-deg = "
                f"{(omega_dir_err_truth_deg if tag=='TRUTH' else omega_dir_err_twin_deg)[k]:5.2f}°  "
                f"geo={geo[k]:.3f} align={align[k]:.3f}"
            )

    out_npz = OUT_DIR / "survivors_per_level.npz"
    np.savez(
        out_npz,
        L0_idx=surv_idx_by_level["L0_strict_1p0_1p0"],
        L1_idx=surv_idx_by_level["L1_relaxed_0p5_6of7"],
        L2_idx=surv_idx_by_level["L2_relaxed_0p5_0p5"],
        L3_idx=surv_idx_by_level["L3_align_only_6of7"],
        union_idx=union_idx.astype(np.int64),
        truth_basin_idx=truth_idx.astype(np.int64),
        twin_basin_idx=twin_idx.astype(np.int64),
        truth_q0=truth_q0,
        twin_q0=twin_q0,
        truth_omega=truth_omega,
        twin_omega=twin_omega,
        q_to_truth_deg=q_to_truth_deg.astype(np.float32),
        q_to_twin_deg=q_to_twin_deg.astype(np.float32),
        omega_mag_rel_err=omega_mag_rel_err_truth.astype(np.float32),
        omega_dir_err_truth_deg=omega_dir_err_truth_deg.astype(np.float32),
        omega_dir_err_twin_deg=omega_dir_err_twin_deg.astype(np.float32),
    )
    print(f"\n  Saved: {out_npz}")

    out_json = OUT_DIR / "summary.json"
    summary = {
        "seed": 6,
        "n_total_candidates": int(N),
        "n_truth_basin_in_pool": int(truth_basin_mask.sum()),
        "n_twin_basin_in_pool": int(twin_basin_mask.sum()),
        "basin_def": {
            "q_deg": BASIN_Q_DEG,
            "omega_mag_rel": BASIN_OMEGA_MAG_REL,
            "omega_dir_deg": BASIN_OMEGA_DIR_DEG,
        },
        "twin_convention": "q_180x · q0_truth (left-mul) + R_180x · ω_truth (flip y,z)",
        "levels": per_level,
        "union_size": int(union_idx.size),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_json}")


if __name__ == "__main__":
    main()
