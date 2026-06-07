"""Tune alignment-cost peak detector on seed 6 truth.

Goal: pick (window, prominence) such that truth scores ≥ 0.95.
Investigates why prominence=0.3, window=3 gives 0.70 on truth.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY.parent.parent.parent))

from lib.traj_load import load_truth                  # noqa: E402
from lib import filter_costs as fc                    # noqa: E402

static = fc.load_static_geometry()
tier_table = fc.load_tier_table()


def truth_align_score(seed, window, prominence):
    truth = load_truth(seed)
    seed_data = fc.precompute_seed_filter_data(truth, tier_table)
    res = fc.evaluate_candidate(
        truth["q0_wxyz"], truth["omega0_rad"], seed_data,
        static["inertia_tensor"], static["face_normals"], tier_table["tier_face_idx"],
        align_window_epochs=window, align_prominence_mag=prominence,
    )
    return res["score_alignment"], res["mag_pred"], seed_data["truth_peak_idx"], truth["mag_hifi"]


# Detail diagnostic on seed 6
score, mag_pred, peak_idx, mag_hifi = truth_align_score(6, 3, 0.3)
print(f"seed 6 with W=3, prom=0.3: align={score:.3f}")
print(f"  truth peak count: {peak_idx.size}")

# Show the surrogate-LC peaks vs truth peaks
cand_pk_idx, props = find_peaks(-mag_pred, prominence=0.3)
print(f"  surrogate peaks (prom=0.3): {cand_pk_idx.size}")

# For each truth peak, show whether it's matched
print("\nPer-peak detail (W=3, prom=0.3):")
print(f"  {'peak#':>5} {'idx':>4} {'mag_truth':>10} {'mag_surr':>10} {'match?':>8}")
for i, tp in enumerate(peak_idx):
    matched = np.any(np.abs(cand_pk_idx - tp) <= 3)
    print(f"  {i:>5} {tp:>4} {mag_hifi[tp]:>10.3f} {mag_pred[tp]:>10.3f} "
          f"{'YES' if matched else 'no':>8}")

# Param sweep
print("\nParameter sweep — TRUTH alignment cost:")
print(f"  {'W':>3} {'prom':>5} | {'seed6':>6}  {'seed28':>6}  {'seed44':>6}  {'seed91':>6}  {'mean5':>6}")
for window in [2, 3, 5, 7]:
    for prom in [0.05, 0.1, 0.2, 0.3]:
        scores = []
        for sd in [6, 13, 28, 41, 44, 49, 79, 84, 91]:
            try:
                s, _, _, _ = truth_align_score(sd, window, prom)
                scores.append(s)
            except Exception as e:
                scores.append(float("nan"))
        s6 = scores[0]
        s28 = scores[2]
        s44 = scores[4]
        s91 = scores[8]
        mean = np.nanmean(scores)
        print(f"  {window:>3} {prom:>5.2f} | {s6:>6.3f}  {s28:>6.3f}  {s44:>6.3f}  "
              f"{s91:>6.3f}  {mean:>6.3f}")
