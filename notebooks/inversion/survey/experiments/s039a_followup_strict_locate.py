"""s039a follow-up #2: at STRICT 1.0/1.0, where do the surviving ICs sit?

Per seed where strict survivors exist: report the closest-to-truth survivor's
q0_err and its ω-cell offsets. This is the cohort-relevant question — strict
filter survivors are the ones forming the existing s032 architecture's
working set.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

S032_DIR = SURVEY_DIR / "results" / "s032_cohort_fast"
TRAJ_DIR = SURVEY_DIR / "data" / "trajectories"
OUT_DIR = SURVEY_DIR / "results" / "s039a_relaxed_filter_rescore"


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def angular_dist_batch(q_arr, q_ref):
    dots = np.abs(q_arr @ q_ref)
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def main():
    out = []
    seeds = sorted(int(d.name[4:]) for d in S032_DIR.glob("seed*/"))
    print(f"=== s039a follow-up #2: strict-survivor closest-IC locations ===")
    print(f"{'seed':>4} {'n_surv':>7} {'q_min':>6} {'q_t':>6} {'q_w':>6} "
          f"{'mag%':>7} {'dir°':>6} {'br%':>6}")
    for seed in seeds:
        seed_dir = S032_DIR / f"seed{seed:03d}"
        summ_p = seed_dir / "summary.json"
        if not summ_p.exists():
            continue
        summ = json.loads(summ_p.read_text())
        if summ["categorisation"]["passed_both"] == 0:
            continue
        # Strict survivors are stored explicitly
        sd = np.load(seed_dir / "survivor_diagnostics.npz")
        if sd["survivor_q0"].shape[0] == 0:
            continue
        og = np.load(seed_dir / "omega_grid.npz")
        omega_vectors = og["omega_vectors"]
        truth_omega_dir = og["truth_omega_dir"]
        truth_omega_mag = float(og["truth_omega_mag_rad"])

        truth = np.load(TRAJ_DIR / f"traj_seed{seed:03d}.npz")
        q0_truth = truth["q0_wxyz"].astype(np.float64)
        twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), q0_truth)

        e_t = sd["survivor_geodesic_to_truth_deg"]
        e_w = sd["survivor_geodesic_to_twin_deg"]
        e_min = np.minimum(e_t, e_w)
        idx = int(np.argmin(e_min))

        cell_i = int(sd["survivor_omega_cell"][idx])
        omega_cell = omega_vectors[cell_i]
        cell_mag = float(np.linalg.norm(omega_cell))
        mag_pct = (cell_mag - truth_omega_mag) / truth_omega_mag * 100
        dir_deg = float(np.degrees(np.arccos(np.clip(
            omega_cell / cell_mag @ truth_omega_dir, -1, 1))))
        n_surv = int(sd["survivor_q0"].shape[0])
        br_pct = float(summ["bracket"]["nearest_cell_pct"])

        print(f"{seed:>4} {n_surv:>7} {e_min[idx]:>6.2f} {e_t[idx]:>6.1f} {e_w[idx]:>6.1f} "
              f"{mag_pct:>+7.2f} {dir_deg:>6.2f} {br_pct:>6.2f}")
        out.append({
            "seed": seed,
            "n_strict_survivors": n_surv,
            "q0_err_truth_deg": float(e_t[idx]),
            "q0_err_twin_deg": float(e_w[idx]),
            "q0_err_min_deg": float(e_min[idx]),
            "omega_cell_mag_pct": float(mag_pct),
            "omega_cell_dir_deg": float(dir_deg),
            "bracket_pct": br_pct,
        })

    # Cohort summary
    qerrs = np.array([r['q0_err_min_deg'] for r in out])
    pcts = np.array([r['omega_cell_mag_pct'] for r in out])
    dirs = np.array([r['omega_cell_dir_deg'] for r in out])
    n = len(out)

    print(f"\nn={n} seeds with ≥1 strict-1.0/1.0 survivor")
    print(f"\nq0_err of closest strict survivor:")
    print(f"  median={np.median(qerrs):.2f}°, p25={np.percentile(qerrs,25):.2f}°, "
          f"p75={np.percentile(qerrs,75):.2f}°, max={qerrs.max():.2f}°")
    print(f"\nω-mag offset of closest survivor's cell:")
    print(f"  median={np.median(np.abs(pcts)):.2f}%, p25={np.percentile(np.abs(pcts),25):.2f}%, "
          f"p75={np.percentile(np.abs(pcts),75):.2f}%, max={np.max(np.abs(pcts)):.2f}%")
    print(f"\nω-dir offset of closest survivor's cell:")
    print(f"  median={np.median(dirs):.2f}°, p25={np.percentile(dirs,25):.2f}°, "
          f"p75={np.percentile(dirs,75):.2f}°, max={dirs.max():.2f}°")

    print(f"\nLM-plausible (q0<10° AND |ω-mag|<10% AND ω-dir<5°): "
          f"{int(((qerrs<10)&(np.abs(pcts)<10)&(dirs<5)).sum())}/{n}")
    print(f"LM-plausible (q0<30° AND |ω-mag|<20% AND ω-dir<10°): "
          f"{int(((qerrs<30)&(np.abs(pcts)<20)&(dirs<10)).sum())}/{n}")

    # Also: of the n seeds with strict survivors, how many have the BRACKET cell within 5%?
    n_bracket_in_tube = sum(1 for r in out if r["bracket_pct"] < 5)
    print(f"\nOf the {n} seeds with strict survivors, {n_bracket_in_tube} have bracket within 5% off truth-ω")

    with open(OUT_DIR / "strict_best_locations.json", "w") as f:
        json.dump({"per_seed": out}, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'strict_best_locations.json'}")


if __name__ == "__main__":
    main()
