"""Does the ω≈truth subset of s081's Band A|B basins follow a searchable form?

Splits the 30 distinct hi-fi Band A|B basins by ω-direction error, then for
the small-ω-error group reads s077's cached polhode invariants (2T, |L|^2)
to test: same body-frame polhode as truth (-> the s077 'fixed-ω, free
L-direction' form), or merely nearby (the s073d ambiguity)?

Pure cached-data pass: s081/basin_hifi.npz + s077/basin_l_metrics.npz are
both 640-ordered over the same s011/s068 substrate, so indices align.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.traj_load import truth_state  # noqa: E402

S081 = SURVEY_ROOT / "results" / "s081" / "basin_hifi.npz"
S077 = SURVEY_ROOT / "results" / "s077" / "basin_l_metrics.npz"

W_DIR_SPLIT_DEG = 15.0  # clean gap in the table: group A 0-8 deg, group B 52-89


def omega_dir_deg(oa, ob):
    na, nb = np.linalg.norm(oa), np.linalg.norm(ob)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(oa @ ob) / (na * nb),
                                              -1.0, 1.0))))


def main():
    a = np.load(S081, allow_pickle=True)
    b = np.load(S077, allow_pickle=True)

    seeds = a["seed"].astype(int)
    om = a["omega_final_rad"]
    rho_hifi = a["rho_hifi"]
    band_hifi = a["band_hifi"].astype(str)
    twin_label = a["twin_label"].astype(str)
    basin_class = a["basin_class"].astype(str)

    # sanity: same substrate ordering
    assert np.array_equal(seeds, b["seed"].astype(int)), "index misalignment"

    d_twoT = b["d_twoT_rel"]
    d_L2 = b["d_L2_rel"]
    L_dir = b["L_dir_angle_deg"]
    L_mag = b["L_mag_rel_diff"]
    regime = b["regime"].astype(str)

    truth = {int(s): truth_state(int(s)) for s in sorted(set(seeds.tolist()))}

    sel = ((basin_class == "competing_low_mse")
           & np.isin(band_hifi, ["A", "B"])
           & (twin_label == "distinct"))
    idx = np.where(sel)[0]

    wdir = np.array([omega_dir_deg(om[i],
                                   np.asarray(truth[int(seeds[i])]["omega0_rad"]))
                     for i in idx])
    grpA = idx[wdir < W_DIR_SPLIT_DEG]
    grpB = idx[wdir >= W_DIR_SPLIT_DEG]

    print(f"=== s081 Band A|B distinct basins: searchable-form check ===")
    print(f"  n total = {idx.size}   group A (w_dir<{W_DIR_SPLIT_DEG:.0f} deg) "
          f"= {grpA.size}   group B = {grpB.size}")
    print()
    print("  GROUP A — omega ~ truth in the body frame. If this is the s077")
    print("  'fixed-omega, free-L-direction' form, 2T and |L|^2 must MATCH")
    print("  truth (same polhode); if they differ ~1%, it's s073d 'nearby'.")
    print()
    hdr = (f"  {'seed':>4} {'reg':>3} {'rho':>5} {'w_dir':>6} "
           f"{'d_2T_rel':>9} {'d_L2_rel':>9} {'L_dir_deg':>9} {'L_mag_rel':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    seen = set()
    for i in sorted(grpA, key=lambda j: (seeds[j], rho_hifi[j])):
        key = (int(seeds[i]), round(float(rho_hifi[i]), 2))
        if key in seen:
            continue
        seen.add(key)
        wd = omega_dir_deg(om[i], np.asarray(truth[int(seeds[i])]["omega0_rad"]))
        print(f"  {int(seeds[i]):>4} {regime[i]:>3} {rho_hifi[i]:>5.2f} "
              f"{wd:>6.1f} {d_twoT[i]:>+9.3%} {d_L2[i]:>+9.3%} "
              f"{L_dir[i]:>9.1f} {L_mag[i]:>+9.3%}")

    gA_2T = np.abs(d_twoT[grpA])
    gA_L2 = np.abs(d_L2[grpA])
    print()
    print(f"  group A |d_2T_rel| : min {gA_2T.min():.3%}  "
          f"median {np.median(gA_2T):.3%}  max {gA_2T.max():.3%}")
    print(f"  group A |d_L2_rel| : min {gA_L2.min():.3%}  "
          f"median {np.median(gA_L2):.3%}  max {gA_L2.max():.3%}")
    print()
    print("  GROUP B — omega moved 50-90 deg. Does NOT fit a fixed-omega form.")
    seen = set()
    for i in sorted(grpB, key=lambda j: (seeds[j], rho_hifi[j])):
        key = (int(seeds[i]), round(float(rho_hifi[i]), 2))
        if key in seen:
            continue
        seen.add(key)
        wd = omega_dir_deg(om[i], np.asarray(truth[int(seeds[i])]["omega0_rad"]))
        print(f"  {int(seeds[i]):>4} {regime[i]:>3} {rho_hifi[i]:>5.2f} "
              f"{wd:>6.1f} {d_twoT[i]:>+9.3%} {d_L2[i]:>+9.3%} "
              f"{L_dir[i]:>9.1f} {L_mag[i]:>+9.3%}")


if __name__ == "__main__":
    main()
