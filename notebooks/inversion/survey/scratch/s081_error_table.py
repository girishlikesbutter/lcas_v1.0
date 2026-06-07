"""Re-score s081's rendered basins: q0 / omega-dir / omega-mag errors.

Pure analytical pass over cached results/s081/basin_hifi.npz — no re-render.
s081 saved a single combined `om_to_truth_rel`; this splits omega into the
direction error (deg) and magnitude error (relative) per the reporting rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.traj_load import truth_state  # noqa: E402

NPZ = SURVEY_ROOT / "results" / "s081" / "basin_hifi.npz"


def q_geodesic_deg(qa, qb):
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(qa @ qb))))))


def omega_dir_deg(oa, ob):
    na, nb = np.linalg.norm(oa), np.linalg.norm(ob)
    if na == 0 or nb == 0:
        return float("nan")
    c = np.clip(float(oa @ ob) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def omega_mag_rel(oa, ob):
    nb = np.linalg.norm(ob)
    return float((np.linalg.norm(oa) - nb) / nb) if nb > 0 else float("nan")


def main():
    d = np.load(NPZ, allow_pickle=True)
    seeds = d["seed"].astype(int)
    q0 = d["q0_final_wxyz"]
    om = d["omega_final_rad"]
    rho_hifi = d["rho_hifi"]
    band_hifi = d["band_hifi"].astype(str)
    twin_label = d["twin_label"].astype(str)
    mode = d["mode"].astype(str)
    basin_class = d["basin_class"].astype(str)

    truth = {int(s): truth_state(int(s)) for s in sorted(set(seeds.tolist()))}

    # focus: the genuinely-distinct hi-fi Band A|B basins (the multi-solutions)
    sel = ((basin_class == "competing_low_mse")
           & np.isin(band_hifi, ["A", "B"])
           & (twin_label == "distinct"))
    idx = np.where(sel)[0]

    print(f"=== s081 distinct hi-fi Band A|B basins — error breakdown "
          f"(n={idx.size}) ===")
    print(f"  source: {NPZ.relative_to(SURVEY_ROOT)}  (cached, no re-render)")
    print()
    hdr = (f"  {'seed':>4} {'reg':>3} {'rho':>6} {'bd':>2}  "
           f"{'q0_err_deg':>10}  {'w_dir_deg':>9}  {'w_mag_rel':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for i in idx:
        s = int(seeds[i])
        t = truth[s]
        qt = np.asarray(t["q0_wxyz"], dtype=np.float64)
        ot = np.asarray(t["omega0_rad"], dtype=np.float64)
        qf = np.asarray(q0[i], dtype=np.float64)
        of = np.asarray(om[i], dtype=np.float64)
        q_err = q_geodesic_deg(qf, qt)
        wd = omega_dir_deg(of, ot)
        wm = omega_mag_rel(of, ot)
        rows.append((s, mode[i], float(rho_hifi[i]), band_hifi[i],
                     q_err, wd, wm))
    for s, reg, rho, bd, q_err, wd, wm in sorted(rows,
                                                 key=lambda r: (r[0], r[2])):
        print(f"  {s:>4} {reg:>3} {rho:>6.2f} {bd:>2}  "
              f"{q_err:>10.1f}  {wd:>9.1f}  {wm:>+9.3%}")

    qe = np.array([r[4] for r in rows])
    wde = np.array([r[5] for r in rows])
    wme = np.array([r[6] for r in rows])
    print()
    print(f"  q0_err   deg : min {qe.min():6.1f}  median {np.median(qe):6.1f}"
          f"  max {qe.max():6.1f}")
    print(f"  w_dir    deg : min {wde.min():6.1f}  median {np.median(wde):6.1f}"
          f"  max {wde.max():6.1f}")
    print(f"  w_mag    rel : min {wme.min():+6.2%}  "
          f"median {np.median(wme):+6.2%}  max {wme.max():+6.2%}")
    print()
    print("  (q0_err = quaternion geodesic to truth; w_dir = angle between "
          "omega vectors;\n   w_mag = (|w_basin|-|w_truth|)/|w_truth|. "
          "All basins are >8 deg from truth AND the body-X twin.)")


if __name__ == "__main__":
    main()
