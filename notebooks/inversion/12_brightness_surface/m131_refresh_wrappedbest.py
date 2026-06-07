#!/usr/bin/env python3
"""Rebuild wrappedbest_seed{NNN}/ dirs with post-fix winners + cached hi-fi LCs.

Reads the current best basin per seed from m126_wrapped (seeds 0, 6, 12, 24,
33, 36) and m124/m125_keep_better (seeds 14, 27, 46, 74, 93), writes a
result.json in the schema lc_compare.py expects, and copies the corresponding
saved hi-fi LC as pred_lc.npy.

No hi-fi regeneration. No surrogate runs. Purely plumbing.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
M126_DIR = RESULTS_DIR / "m126_wrapped"
M124_DIR = RESULTS_DIR / "m124"
M125_DIR = RESULTS_DIR / "m125_keep_better"
TRAJ_PATH = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"


def quat_angle_deg(q1, q2):
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * np.degrees(np.arccos(dot))


def vec_angle_deg(v1, v2, signed=True):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-15 or n2 < 1e-15:
        return float('nan')
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = min(1.0, max(-1.0, cos))
    if signed:
        return np.degrees(np.arccos(cos))
    return np.degrees(np.arccos(abs(cos)))


def build_winner_record(q0_wxyz, w0_rad, true_q0, true_w0, hifi_mse,
                         source_label, w_dir_signed=True):
    q0 = np.asarray(q0_wxyz)
    w0 = np.asarray(w0_rad)
    q0_err = quat_angle_deg(q0, true_q0)
    w_dir_err = vec_angle_deg(w0, true_w0, signed=w_dir_signed)
    mag_est = np.linalg.norm(w0)
    mag_true = np.linalg.norm(true_w0)
    w_mag_err_pct = 100.0 * (mag_est - mag_true) / mag_true
    return {
        "q0_wxyz": q0.tolist(),
        "w0_rad": w0.tolist(),
        "w0_dps": np.rad2deg(w0).tolist(),
        "q0_err": q0_err,
        "w0_err": w_dir_err,
        "w_mag_err_pct": w_mag_err_pct,
        "hifi": float(hifi_mse),
        "source_label": source_label,
    }


def refresh_m126_seed(seed, master):
    """Rebuild wrappedbest for seeds run in m126_wrapped batch."""
    seed_dir = M126_DIR / f"seed_{seed:03d}"
    res = json.load(open(seed_dir / "result.json"))
    basins = res["basins"]
    winner_idx = int(np.argmin([b["hifi_wrapped"] for b in basins]))
    b = basins[winner_idx]

    use_after = b["hifi_after"] <= b["hifi_before"]
    q0_src = b["q0_after"] if use_after else b["q0_before"]
    w0_src = b["omega_after"] if use_after else b["omega_before"]

    hifi_ckpt = np.load(seed_dir / "hifi_ckpt.npz", allow_pickle=True)
    idx_in_ckpt = int(np.where(hifi_ckpt["basin_idx"] == winner_idx)[0][0])
    pred_lc = hifi_ckpt["hifi_mags_after"][idx_in_ckpt]
    assert pred_lc.shape == (500,), f"seed {seed} LC shape mismatch: {pred_lc.shape}"

    winner = build_winner_record(
        q0_src, w0_src,
        master["q0s"][seed], master["omega0s"][seed],
        b["hifi_wrapped"],
        source_label=f"m126_basin{winner_idx}_{'after' if use_after else 'before'}",
        w_dir_signed=True,
    )
    winner["hifi"] = float(b["hifi_wrapped"])
    return winner, pred_lc, res["classification"], b


def refresh_m124_seed(seed, master, m124_summary, m125_summary, hifi_npz):
    """Rebuild wrappedbest for seeds in m124+m125_keep_better cohort."""
    seed_entry = next(s for s in m125_summary["per_seed"] if s["seed"] == seed)
    per_basin = seed_entry["per_basin"]
    winner_basin_label = per_basin[int(np.argmin(
        [pb["hifi_best_wrapped"] for pb in per_basin]))]["label"]

    winner_cand = next(
        c for c in m124_summary["per_candidate"]
        if c["seed"] == seed and c["start_label"] == winner_basin_label
    )
    q0_src = winner_cand["final_q0_wxyz"]
    w0_src = winner_cand["final_omega_rad"]
    hifi_mse = winner_cand["hifi_mse_after"]

    kinds = hifi_npz["kinds"]
    tags = hifi_npz["tags"]
    seeds = hifi_npz["seeds"]
    target_tag = f"_seed_{seed:03d}_{winner_basin_label}"
    match = np.where(
        (seeds == seed) &
        (kinds == b"polished") &
        np.array([target_tag in t.decode() for t in tags])
    )[0]
    assert len(match) == 1, f"seed {seed}: expected 1 match, got {len(match)}"
    pred_lc = hifi_npz["hifi_mags"][int(match[0])]
    assert pred_lc.shape == (500,)

    winner = build_winner_record(
        q0_src, w0_src,
        master["q0s"][seed], master["omega0s"][seed],
        hifi_mse,
        source_label=f"m124_{winner_basin_label}_polished",
        w_dir_signed=True,
    )
    return winner, pred_lc, None, winner_basin_label


def main():
    print("Loading truth trajectories...")
    master = np.load(str(TRAJ_PATH), allow_pickle=True)

    m126_seeds = [0, 6, 12, 24, 33, 36]
    m124_seeds = [14, 27, 46, 74, 93]
    all_seeds = sorted(m126_seeds + m124_seeds)

    print("Loading m124/m125 summaries...")
    m124_summary = json.load(open(M124_DIR / "summary.json"))
    m125_summary = json.load(open(M125_DIR / "summary.json"))
    hifi_npz = np.load(M124_DIR / "hifi_results.npz", allow_pickle=True)

    print("\nRebuilding wrappedbest dirs:")
    print(f"{'seed':>5} {'source':>25} {'q0_err':>8} {'w_dir':>8} {'w_mag%':>8} "
          f"{'hifi':>9}  class")
    print("-" * 85)

    for seed in all_seeds:
        if seed in m126_seeds:
            winner, pred_lc, classification, basin = refresh_m126_seed(seed, master)
        else:
            winner, pred_lc, _, basin = refresh_m124_seed(
                seed, master, m124_summary, m125_summary, hifi_npz)
            if winner["hifi"] < 0.01:
                classification = "OK"
            elif winner["hifi"] < 0.1:
                classification = "PARTIAL"
            else:
                classification = "FAIL"

        out_dir = RESULTS_DIR / f"wrappedbest_seed{seed:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment": f"wrappedbest (postfix 2026-04-17)",
            "seed": seed,
            "classification": classification,
            "winner": winner,
        }
        with open(out_dir / "result.json", "w") as f:
            json.dump(payload, f, indent=2)

        np.save(str(out_dir / "pred_lc.npy"), pred_lc.astype(np.float64))

        w = winner
        print(f"{seed:>5d} {w['source_label']:>25s} "
              f"{w['q0_err']:>7.2f}° {w['w0_err']:>7.3f}° "
              f"{w['w_mag_err_pct']:>+7.3f}% {w['hifi']:>8.5f}  {classification}")

    print(f"\nWrote {len(all_seeds)} wrappedbest_seed*/ dirs under {RESULTS_DIR}")
    print(f"Next: python3 notebooks/inversion/lib/lc_compare.py --prefix wrappedbest "
          f"{' '.join(str(s) for s in all_seeds)}")


if __name__ == "__main__":
    sys.exit(main())
