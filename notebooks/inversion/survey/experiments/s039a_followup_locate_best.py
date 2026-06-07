"""s039a follow-up — locate the closest-to-truth IC per seed:
where in the (q0, ω) space does it sit?

For each cohort seed at threshold geo >= 1/n_spec, identify the IC with
minimum q0_err to truth/twin. Report:
  - q0_err to truth, twin, min
  - ω-cell index, ω-mag % off truth, ω-dir ° off truth
  - geo_score, align_score (NaN if geo-fail)
  - whether the cell is the near-truth bracket cell

This tells us: of the 75/75 'in-basin' seeds, how many have the close IC at
a cell where LM is plausible (small ω-mag offset, small ω-dir offset)?
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


def locate(seed):
    seed_dir = S032_DIR / f"seed{seed:03d}"
    summ_p = seed_dir / "summary.json"
    if not summ_p.exists():
        return None
    summ = json.loads(summ_p.read_text())
    n_spec = int(summ["scale"]["n_spec_events"])
    if n_spec == 0:
        return None

    cands = np.load(seed_dir / "candidates_meta.npz", mmap_mode="r")
    geo_score = np.array(cands["geo_score"])
    align_score = np.array(cands["align_score"])
    omega_cell_idx = np.array(cands["omega_cell_idx"])
    q0_arr = np.array(cands["q0"]).astype(np.float64)

    omega_grid = np.load(seed_dir / "omega_grid.npz")
    omega_vectors = omega_grid["omega_vectors"]
    truth_omega_dir = omega_grid["truth_omega_dir"]
    truth_omega_mag = float(omega_grid["truth_omega_mag_rad"])

    truth = np.load(TRAJ_DIR / f"traj_seed{seed:03d}.npz")
    q0_truth = truth["q0_wxyz"].astype(np.float64)
    twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), q0_truth)

    thr = 1.0 / n_spec - 1e-6
    mask = np.isfinite(geo_score) & (geo_score >= thr)
    if not mask.any():
        return None

    e_t = angular_dist_batch(q0_arr[mask], q0_truth)
    e_w = angular_dist_batch(q0_arr[mask], twin_q0)
    e_min = np.minimum(e_t, e_w)
    idx_in_mask = int(np.argmin(e_min))
    global_idx = int(np.where(mask)[0][idx_in_mask])
    cell_i = int(omega_cell_idx[global_idx])
    omega_cell = omega_vectors[cell_i]
    omega_cell_mag = float(np.linalg.norm(omega_cell))
    omega_cell_dir = omega_cell / omega_cell_mag
    mag_pct = (omega_cell_mag - truth_omega_mag) / truth_omega_mag * 100
    dir_deg = float(np.degrees(np.arccos(np.clip(omega_cell_dir @ truth_omega_dir, -1, 1))))

    return {
        "seed": seed,
        "q0_err_truth_deg": float(e_t[idx_in_mask]),
        "q0_err_twin_deg": float(e_w[idx_in_mask]),
        "q0_err_min_deg": float(e_min[idx_in_mask]),
        "cell_idx": cell_i,
        "omega_cell_mag_pct": float(mag_pct),
        "omega_cell_dir_deg": float(dir_deg),
        "geo_score": float(geo_score[global_idx]),
        "align_score": (float(align_score[global_idx])
                        if np.isfinite(align_score[global_idx]) else None),
        "n_spec": n_spec,
        "nearest_cell_pct_in_bracket": float(summ["bracket"]["nearest_cell_pct"]),
    }


def main():
    t0 = time.time()
    out = []
    seeds = sorted(int(d.name[4:]) for d in S032_DIR.glob("seed*/"))
    print(f"=== s039a follow-up: locate best IC per seed ===")
    print(f"{'seed':>4} {'q_err':>6} {'q_t':>6} {'q_w':>6} {'cell_id':>7} "
          f"{'mag%':>7} {'dir°':>6} {'geo':>5} {'aln':>5} {'br_pct':>7}")
    for s in seeds:
        r = locate(s)
        if r is None:
            continue
        aln = f"{r['align_score']:.2f}" if r['align_score'] is not None else "  - "
        print(f"{r['seed']:>4} {r['q0_err_min_deg']:>6.2f} {r['q0_err_truth_deg']:>6.1f} "
              f"{r['q0_err_twin_deg']:>6.1f} {r['cell_idx']:>7} "
              f"{r['omega_cell_mag_pct']:>+7.2f} {r['omega_cell_dir_deg']:>6.2f} "
              f"{r['geo_score']:>5.2f} {aln:>5} {r['nearest_cell_pct_in_bracket']:>7.2f}")
        out.append(r)

    print(f"\nWall: {time.time()-t0:.1f}s, n={len(out)}")
    with open(OUT_DIR / "best_ic_locations.json", "w") as f:
        json.dump({"per_seed": out}, f, indent=2)

    # Headline analysis
    print(f"\n=== Cohort: where do the closest-to-truth ICs live? ===")
    pcts = np.array([r['omega_cell_mag_pct'] for r in out])
    dirs = np.array([r['omega_cell_dir_deg'] for r in out])
    qerrs = np.array([r['q0_err_min_deg'] for r in out])

    # Define "LM-plausible cell": mag offset within s003 tube (~5%) AND dir within Fibonacci grid resolution (~3°)
    plausible = (np.abs(pcts) < 5) & (dirs < 3)
    print(f"  q0_err < 5° AND |ω-mag|<5% AND ω-dir<3°: {int((plausible & (qerrs<5)).sum())}/{len(out)} seeds")
    print(f"  q0_err < 5° AND |ω-mag|<10% AND ω-dir<5°: "
          f"{int(((np.abs(pcts)<10)&(dirs<5)&(qerrs<5)).sum())}/{len(out)} seeds")
    print(f"  q0_err < 10° AND |ω-mag|<10% AND ω-dir<5°: "
          f"{int(((np.abs(pcts)<10)&(dirs<5)&(qerrs<10)).sum())}/{len(out)} seeds")
    print(f"  q0_err < 30° AND |ω-mag|<20% AND ω-dir<10°: "
          f"{int(((np.abs(pcts)<20)&(dirs<10)&(qerrs<30)).sum())}/{len(out)} seeds")

    print()
    print(f"  ω-mag offset of best-IC cell: median={np.median(np.abs(pcts)):.2f}%, "
          f"p25={np.percentile(np.abs(pcts), 25):.2f}%, p75={np.percentile(np.abs(pcts), 75):.2f}%")
    print(f"  ω-dir offset of best-IC cell: median={np.median(dirs):.2f}°, "
          f"p25={np.percentile(dirs, 25):.2f}°, p75={np.percentile(dirs, 75):.2f}°")
    print(f"  q0_err of best IC: median={np.median(qerrs):.2f}°, "
          f"p25={np.percentile(qerrs, 25):.2f}°, p75={np.percentile(qerrs, 75):.2f}°")

    print()
    print(f"=== Bracket coverage vs best-IC ω-cell ===")
    bracket_pcts = np.array([r['nearest_cell_pct_in_bracket'] for r in out])
    print(f"  bracket nearest-cell-pct: median={np.median(bracket_pcts):.2f}%, "
          f"p90={np.percentile(bracket_pcts, 90):.2f}%")
    print(f"  Of seeds where best-IC ω-cell is at <5%: their bracket nearest_pct distribution:")
    near_mask = np.abs(pcts) < 5
    if near_mask.any():
        print(f"    n={near_mask.sum()}, bracket median={np.median(bracket_pcts[near_mask]):.2f}%")


if __name__ == "__main__":
    main()
