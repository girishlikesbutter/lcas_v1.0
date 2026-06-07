"""s057c — locate the truth-representative pair and check the ω it implies.

s057 measured the angular distance of all 14872 pair-implied ω-vectors to
truth-ω. The aggregate concentration was weak (1.79× uniform baseline at
±25% prior). The right diagnostic question: what ω does the SPECIFIC pair
(closest-survivor-to-truth at t_a, closest-survivor-to-truth at t_b) imply?

If that ω is close to truth-ω (within, say, 30°), then the architecture's
"true positive" pair exists in the pool and the problem reduces to finding
it. If that ω is far from truth-ω, finite-diff noise from pool
discretisation washes out the signal regardless of search strategy.

Also reports k=5 nearest survivors on each side and the cross-product k=25
pairs to bound the in-pool achievable accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
TRAJ089 = SURVEY / "data" / "trajectories" / "traj_seed089.npz"
OUT = SURVEY / "results" / "s057c_truth_pair_omega"
OUT.mkdir(parents=True, exist_ok=True)

T_A = 411  # deepest |C_t| anchor (from s057)
T_B = 421
K = 5      # k nearest survivors per side


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def finite_diff_omega_passive(q_a_wxyz, q_b_wxyz, dt_s):
    """Passive convention (verified by s057 smoke test, 0.76% recovery err)."""
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a_wxyz))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b_wxyz))
    dR = R_b * R_a.inv()
    return -dR.as_rotvec() / dt_s   # negative because s057 found "-passive"

# wait — re-check. s057 returned `convention="passive"` directly,
# not "-passive". Let me confirm by setting sign=+1 and checking truth recovery.


def main() -> dict:
    z = np.load(DENSE_RUN)
    survive_all = z["survive_all"]
    q_pool = z["q_pool_wxyz"]
    obs_times = z["obs_times"]

    traj = np.load(TRAJ089)
    q_truth_t = traj["quaternions"]
    om0_rad = traj["omega0_rad"]
    om0_mag_dps = float(np.linalg.norm(om0_rad)) * 180 / np.pi

    dt_s = float(np.median(np.diff(obs_times)))
    Δt_pair = (T_B - T_A) * dt_s

    # Verify convention: at t=0 epoch 1↔2, should recover om0_rad
    # (passive, no sign flip from s057)
    R1 = Rotation.from_quat(wxyz_to_xyzw(q_truth_t[1]))
    R2 = Rotation.from_quat(wxyz_to_xyzw(q_truth_t[2]))
    dR = R2 * R1.inv()
    om_check = dR.as_rotvec() / dt_s
    err_pos = np.linalg.norm(om_check - om0_rad)
    err_neg = np.linalg.norm(-om_check - om0_rad)
    if err_neg < err_pos:
        sign = -1
        conv_err = err_neg / np.linalg.norm(om0_rad)
    else:
        sign = +1
        conv_err = err_pos / np.linalg.norm(om0_rad)
    print(f"convention sign: {sign}, recovery err {conv_err*100:.3f}%")

    def fd_omega(q_a, q_b, dt):
        R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
        R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
        dR = R_b * R_a.inv()
        return sign * dR.as_rotvec() / dt

    # truth ω at t_a from cached q(t) finite-diff over Δt_pair
    om_truth_at_ta = fd_omega(q_truth_t[T_A:T_A+1], q_truth_t[T_B:T_B+1], Δt_pair)[0]
    om_truth_mag = float(np.linalg.norm(om_truth_at_ta))
    om_truth_hat = om_truth_at_ta / om_truth_mag

    # Survivors at each epoch
    idx_a = np.where(survive_all[T_A])[0]
    idx_b = np.where(survive_all[T_B])[0]
    C_a = q_pool[idx_a]
    C_b = q_pool[idx_b]

    # Find k closest survivors to truth on each side (by quaternion |dot|)
    dots_a = np.abs(C_a @ q_truth_t[T_A])
    order_a = np.argsort(dots_a)[::-1][:K]
    dots_b = np.abs(C_b @ q_truth_t[T_B])
    order_b = np.argsort(dots_b)[::-1][:K]

    deg_a = np.degrees(2 * np.arccos(np.clip(dots_a[order_a], 0, 1)))
    deg_b = np.degrees(2 * np.arccos(np.clip(dots_b[order_b], 0, 1)))

    print(f"\nT_A={T_A}: |C_a|={len(C_a)}; top-{K} closest survivors to truth:")
    for r, (oa, da) in enumerate(zip(order_a, deg_a)):
        print(f"  rank {r}: pool_idx={idx_a[oa]:6d}  ang_to_truth={da:.3f}°")
    print(f"T_B={T_B}: |C_b|={len(C_b)}; top-{K} closest survivors to truth:")
    for r, (ob, db) in enumerate(zip(order_b, deg_b)):
        print(f"  rank {r}: pool_idx={idx_b[ob]:6d}  ang_to_truth={db:.3f}°")

    # ω implied by k=1 closest pair (q_a*, q_b*)
    qa_star = C_a[order_a[0]]
    qb_star = C_b[order_b[0]]
    om_star = fd_omega(qa_star[None, :], qb_star[None, :], Δt_pair)[0]
    om_star_mag = float(np.linalg.norm(om_star))
    om_star_hat = om_star / om_star_mag
    cos_dir_star = float(np.abs(np.dot(om_star_hat, om_truth_hat)))
    ang_star_deg = float(np.degrees(np.arccos(np.clip(cos_dir_star, 0, 1))))
    mag_err_star_pct = float((om_star_mag - om_truth_mag) / om_truth_mag * 100)

    print(f"\n=== closest-pair (q_a*, q_b*) implied ω ===")
    print(f"truth |ω| at t_a: {om_truth_mag*180/np.pi:.4f} dps")
    print(f"truth-pair ω-axis (body): [{om_truth_hat[0]:+.4f}, {om_truth_hat[1]:+.4f}, {om_truth_hat[2]:+.4f}]")
    print(f"star-pair ω-axis  (body): [{om_star_hat[0]:+.4f}, {om_star_hat[1]:+.4f}, {om_star_hat[2]:+.4f}]")
    print(f"angular distance: {ang_star_deg:.2f}°")
    print(f"|ω| relative error: {mag_err_star_pct:+.2f}%")

    # all k×k pairs with the top-K nearest survivors per side
    pair_results = []
    Q_A = np.repeat(C_a[order_a], K, axis=0)
    Q_B = np.tile(C_b[order_b], (K, 1))
    om_kk = fd_omega(Q_A, Q_B, Δt_pair)  # (K*K, 3) rad/s
    om_kk_mag = np.linalg.norm(om_kk, axis=1)
    om_kk_hat = om_kk / om_kk_mag[:, None]
    cos_d = np.abs(om_kk_hat @ om_truth_hat)
    ang_d = np.degrees(np.arccos(np.clip(cos_d, 0, 1)))
    mag_err_kk = (om_kk_mag - om_truth_mag) / om_truth_mag * 100

    print(f"\n=== top-{K}×{K}={K*K} truth-adjacent pairs: ω-direction error ===")
    print(f"  min={ang_d.min():.2f}°, p10={np.percentile(ang_d,10):.2f}°, "
          f"median={np.median(ang_d):.2f}°, max={ang_d.max():.2f}°")
    print(f"  |ω| error (signed %): "
          f"min={mag_err_kk.min():+.1f}%, median={np.median(mag_err_kk):+.1f}%, "
          f"max={mag_err_kk.max():+.1f}%")

    # how many of these K*K pairs would have passed the |ω|-prior at ±5%, ±25%?
    target_mag = om_truth_mag
    pass_5 = ((om_kk_mag >= 0.95 * target_mag) & (om_kk_mag <= 1.05 * target_mag)).sum()
    pass_25 = ((om_kk_mag >= 0.75 * target_mag) & (om_kk_mag <= 1.25 * target_mag)).sum()
    print(f"  |ω| ±5%  bracket: {pass_5}/{K*K} pairs survive prior")
    print(f"  |ω| ±25% bracket: {pass_25}/{K*K} pairs survive prior")

    summary = {
        "seed": 89,
        "convention_sign": sign,
        "convention_err_pct": float(conv_err * 100),
        "T_A": T_A,
        "T_B": T_B,
        "delta_t_s": Δt_pair,
        "om_truth_mag_dps": om_truth_mag * 180 / np.pi,
        "om_truth_hat": om_truth_hat.tolist(),
        "K_per_side": K,
        "top_k_distances_a_deg": deg_a.tolist(),
        "top_k_distances_b_deg": deg_b.tolist(),
        "star_pair": {
            "q_a_idx_in_pool": int(idx_a[order_a[0]]),
            "q_b_idx_in_pool": int(idx_b[order_b[0]]),
            "ang_to_truth_a_deg": float(deg_a[0]),
            "ang_to_truth_b_deg": float(deg_b[0]),
            "om_implied_dps": om_star_mag * 180 / np.pi,
            "om_axis": om_star_hat.tolist(),
            "ang_to_truth_omega_deg": ang_star_deg,
            "mag_err_pct": mag_err_star_pct,
        },
        "kxk_distribution": {
            "ang_min_deg": float(ang_d.min()),
            "ang_median_deg": float(np.median(ang_d)),
            "ang_p10_deg": float(np.percentile(ang_d, 10)),
            "ang_max_deg": float(ang_d.max()),
            "mag_err_min_pct": float(mag_err_kk.min()),
            "mag_err_median_pct": float(np.median(mag_err_kk)),
            "mag_err_max_pct": float(mag_err_kk.max()),
            "n_pass_5pct_bracket": int(pass_5),
            "n_pass_25pct_bracket": int(pass_25),
        },
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT / 'summary.json'}")
    return summary


if __name__ == "__main__":
    s = main()
