#!/usr/bin/env python3
"""Cost-at-truth probe — diagnose m103 sampling-failure seeds.

For each seed in --seeds, replicates m103's anchor / constraint / phi-sweep
setup and evaluates the alignment cost at the TRUE omega (and a fine ring
of fibonacci-grid directions around truth) to settle whether m103's
sampling-failure mode (truth-close omega absent from top-3 pool) is driven
by:

    (a) cost-function inadequacy at truth (cost(truth_w) is high — denser
        grid won't help), OR
    (b) grid-resolution miss (cost(truth_w) is low but the nearest fibonacci
        grid neighbour is too far off-basin to score in top-500).

Output:  data/results/inversion_diagnostics/probe_cost_at_truth/seed_NNN.json
plus a summary CSV at .../summary.csv.

Usage
-----
    python3 notebooks/inversion/probe_cost_at_truth.py \
        --seeds 47 51 79 84 89 91 --traj-source m048
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from lib.traj_source import canonical_observed_lc  # noqa: E402

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# m103-equivalent constants
N_DIRS = 2000
N_PHI_FINE = 360
PEAK_WINDOW = 3  # noqa
CONSTRAINT_WEIGHT = 10.0
Z_NORMALS = {4, 5}


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def propagate_delta_qs(omega_vec, dt_arr, I_tensor):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
                                   np.concatenate([[0.0], fwd_dt]),
                                   "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
                                   np.concatenate([[0.0], bwd_dt]),
                                   "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def vectorized_phi_cost_excl(q_anchors_xyzw, delta_qs, pab_arr,
                             allowed_per_constraint, normals, w):
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per_constraint[ci]
        bds = (pbs @ normals[allowed].T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


def setup_seed(seed, source):
    """Replicate m103 Step 1 setup for one seed."""
    if source == "m048":
        master = np.load(str(DIAG / "m048_trajectories" / "m048_trajectories.npz"),
                         allow_pickle=True)
        obs_times = master["observation_times"][seed]
        pab_j2000 = master["pab_j2000"][seed]
    else:
        master = np.load(str(DIAG / "m046_trajectories" / "m046_trajectories.npz"),
                         allow_pickle=True)
        obs_times = master["observation_times"]
        pab_j2000 = master["pab_j2000"]

    unique_normals = master["unique_normals"]
    I_tensor = master["inertia_tensor"]
    true_q0 = master["q0s"][seed]
    true_omega0 = master["omega0s"][seed]
    true_lc = master["mag_hifi"][seed]
    observed_lc = canonical_observed_lc(true_lc)

    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

    smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
    smooth_mags = smoothed_lc[spec_peaks]
    sr = np.argsort(smooth_mags)
    if len(sr) >= 2 and abs(smooth_mags[sr[0]] - smooth_mags[sr[1]]) < 0.05:
        anchor_rank = sr[:2][np.argmin(spec_peaks[sr[:2]])]
    else:
        anchor_rank = sr[0]
    anchor_idx = int(spec_peaks[anchor_rank])
    anchor_time = obs_times[anchor_idx]
    anchor_mag = observed_lc[anchor_idx]
    anchor_allowed = get_allowed_normals(anchor_mag)

    non_anchor = spec_peaks[spec_peaks != anchor_idx]
    constraint_epochs = non_anchor
    constraint_mags = observed_lc[constraint_epochs]
    constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
    dt_constraints = obs_times[constraint_epochs] - anchor_time
    pab_at_constraints = pab_j2000[constraint_epochs]

    # True omega in body frame at anchor epoch
    _, w_hist = propagate_attitude(true_q0, true_omega0,
                                   np.array([0.0, anchor_time]),
                                   "tumbling", I_tensor)
    true_omega_at_anchor = w_hist[1]
    true_w_mag = float(np.linalg.norm(true_omega_at_anchor))
    true_w_hat = true_omega_at_anchor / true_w_mag

    return dict(
        seed=seed, source=source,
        unique_normals=unique_normals, I_tensor=I_tensor,
        true_q0=true_q0, true_omega0=true_omega0,
        true_omega_at_anchor=true_omega_at_anchor,
        true_w_hat=true_w_hat, true_w_mag=true_w_mag,
        anchor_idx=anchor_idx, anchor_time=anchor_time,
        anchor_mag=anchor_mag, anchor_allowed=anchor_allowed,
        n_spec_peaks=len(spec_peaks),
        n_constraints=len(constraint_epochs),
        dt_constraints=dt_constraints,
        pab_at_constraints=pab_at_constraints,
        constraint_allowed=constraint_allowed,
        pab_at_anchor=pab_j2000[anchor_idx],
    )


def cost_at_omega(omega_vec, ctx, anchor_ni_list=None):
    """Min cost over all (anchor_ni, phi) for a given omega.

    Returns (best_cost, best_ni, best_phi_idx, full_phi_cost_per_ni).
    """
    if anchor_ni_list is None:
        anchor_ni_list = ctx["anchor_allowed"]
    phi_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
    phi_z = np.linspace(0, 2*np.pi, N_PHI_FINE, endpoint=False)
    dqs = propagate_delta_qs(omega_vec, ctx["dt_constraints"], ctx["I_tensor"])
    best = np.inf
    best_ni = -1
    best_phi = -1
    per_ni = {}
    for ni in anchor_ni_list:
        phi_arr = phi_z if ni in Z_NORMALS else phi_xy
        n_body = ctx["unique_normals"][ni]
        qa_wxyz = np.array([anchor_q_from_phi(p, n_body, ctx["pab_at_anchor"])
                            for p in phi_arr])
        qa_xyzw = qa_wxyz[:, [1, 2, 3, 0]]
        cs = vectorized_phi_cost_excl(qa_xyzw, dqs, ctx["pab_at_constraints"],
                                      ctx["constraint_allowed"],
                                      ctx["unique_normals"],
                                      CONSTRAINT_WEIGHT)
        per_ni[int(ni)] = cs.tolist()
        i = int(np.argmin(cs))
        if cs[i] < best:
            best = float(cs[i])
            best_ni = int(ni)
            best_phi = i
    return best, best_ni, best_phi, per_ni


def probe_seed(seed, source):
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.time()
    ctx = setup_seed(seed, source)
    print(f"  anchor ep={ctx['anchor_idx']}, mag={ctx['anchor_mag']:.2f}, "
          f"allowed_normals={ctx['anchor_allowed']}, "
          f"n_constraints={ctx['n_constraints']}", flush=True)
    print(f"  truth |w|={ctx['true_w_mag']:.4f} rad/s "
          f"({np.degrees(ctx['true_w_mag']):.3f} dps)", flush=True)

    # 1. Cost at TRUE omega (unrestricted normal)
    cost_truth, ni_truth, phi_truth, _ = cost_at_omega(
        ctx["true_omega_at_anchor"], ctx, anchor_ni_list=range(len(ctx["unique_normals"])))
    print(f"  cost(truth_w) = {cost_truth:.4e} (best ni={ni_truth}, phi={phi_truth})",
          flush=True)

    # 2. Cost at TRUE omega — restricted to anchor_allowed (what m103 actually does)
    cost_truth_restr, ni_truth_r, phi_truth_r, _ = cost_at_omega(
        ctx["true_omega_at_anchor"], ctx, anchor_ni_list=ctx["anchor_allowed"])
    print(f"  cost(truth_w, anchor-allowed) = {cost_truth_restr:.4e}", flush=True)

    # 3. Cost at the m103 grid's NEAREST fibonacci direction to truth, at truth |w|
    grid = fibonacci_sphere(N_DIRS)
    dots = grid @ ctx["true_w_hat"]
    nn_idx = int(np.argmax(dots))
    nn_dir = grid[nn_idx]
    nn_offset_deg = float(np.degrees(np.arccos(np.clip(dots[nn_idx], -1, 1))))
    omega_nn = nn_dir * ctx["true_w_mag"]
    cost_nn, ni_nn, phi_nn, _ = cost_at_omega(omega_nn, ctx,
                                              anchor_ni_list=ctx["anchor_allowed"])
    print(f"  fibonacci nearest neighbour: idx={nn_idx}, offset={nn_offset_deg:.2f}&deg;",
          flush=True)
    print(f"  cost(grid_nn_w) = {cost_nn:.4e}", flush=True)

    # 4. Cost at fibonacci grid points within 5 deg of truth (sample of basin width)
    near_mask = dots > np.cos(np.radians(5.0))
    near_idx = np.where(near_mask)[0]
    near_costs = []
    for gi in near_idx[:30]:
        omega_g = grid[gi] * ctx["true_w_mag"]
        c, _, _, _ = cost_at_omega(omega_g, ctx,
                                   anchor_ni_list=ctx["anchor_allowed"])
        offset = float(np.degrees(np.arccos(np.clip(dots[gi], -1, 1))))
        near_costs.append((int(gi), offset, float(c)))
    print(f"  evaluated {len(near_costs)} fibonacci points within 5&deg; of truth",
          flush=True)

    # 5. Reference: cost at m103's geo_ckpt BEST omega (sanity)
    geo_path = (DIAG / ("m103_hybrid_m048" if source == "m048" else "m103_hybrid") /
                f"seed_{seed:03d}" / "geo_ckpt.npz")
    cost_geo_best = None
    geo_best_w_dir_err_deg = None
    if geo_path.exists():
        geo = np.load(str(geo_path), allow_pickle=True)
        idx_best = int(np.argmin(geo["geo_costs"]))
        omega_geo = geo["w0_refs"][idx_best]
        # propagate to anchor frame so units match
        # geo's w0_refs are body-frame omegas at t=0; we need at anchor_time
        _, w_hist = propagate_attitude(geo["q0_refs"][idx_best], omega_geo,
                                       np.array([0.0, ctx["anchor_time"]]),
                                       "tumbling", ctx["I_tensor"])
        omega_geo_at_anchor = w_hist[1]
        cost_geo, ni_geo, phi_geo, _ = cost_at_omega(omega_geo_at_anchor, ctx,
                                                    anchor_ni_list=ctx["anchor_allowed"])
        cost_geo_best = float(cost_geo)
        d1 = omega_geo_at_anchor / np.linalg.norm(omega_geo_at_anchor)
        geo_best_w_dir_err_deg = float(np.degrees(np.arccos(
            np.clip(np.abs(d1 @ ctx["true_w_hat"]), -1, 1))))
        print(f"  cost(geo_best_w, propagated) = {cost_geo:.4e} "
              f"[geo_best omega offset from truth = {geo_best_w_dir_err_deg:.2f}&deg;]",
              flush=True)

    elapsed = time.time() - t0
    payload = {
        "seed": seed, "source": source,
        "anchor_idx": int(ctx["anchor_idx"]),
        "anchor_mag": float(ctx["anchor_mag"]),
        "anchor_allowed_normals": ctx["anchor_allowed"],
        "n_spec_peaks": int(ctx["n_spec_peaks"]),
        "n_constraints": int(ctx["n_constraints"]),
        "true_omega_at_anchor": ctx["true_omega_at_anchor"].tolist(),
        "true_w_mag_radps": float(ctx["true_w_mag"]),
        "cost_truth_unrestricted": float(cost_truth),
        "cost_truth_anchor_allowed": float(cost_truth_restr),
        "cost_truth_best_ni": int(ni_truth_r),
        "cost_truth_best_phi_idx": int(phi_truth_r),
        "fibonacci_nn_offset_deg": nn_offset_deg,
        "cost_grid_nn": float(cost_nn),
        "near_grid_costs_within_5deg": near_costs,
        "cost_geo_best_propagated": cost_geo_best,
        "geo_best_w_dir_err_deg": geo_best_w_dir_err_deg,
        "elapsed_s": elapsed,
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    args = ap.parse_args()

    out_dir = DIAG / "probe_cost_at_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        payload = probe_seed(seed, args.traj_source)
        with open(out_dir / f"seed_{seed:03d}.json", "w") as f:
            json.dump(payload, f, indent=2)
        rows.append(payload)

    # Summary CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "n_spec_peaks", "n_constraints", "anchor_mag",
            "cost_truth_anchor_allowed", "cost_truth_unrestricted",
            "cost_grid_nn", "fibonacci_nn_offset_deg",
            "cost_geo_best", "geo_best_w_dir_err_deg",
        ])
        for r in rows:
            w.writerow([
                r["seed"], r["n_spec_peaks"], r["n_constraints"],
                f"{r['anchor_mag']:.3f}",
                f"{r['cost_truth_anchor_allowed']:.4e}",
                f"{r['cost_truth_unrestricted']:.4e}",
                f"{r['cost_grid_nn']:.4e}",
                f"{r['fibonacci_nn_offset_deg']:.3f}",
                "" if r["cost_geo_best_propagated"] is None
                else f"{r['cost_geo_best_propagated']:.4e}",
                "" if r["geo_best_w_dir_err_deg"] is None
                else f"{r['geo_best_w_dir_err_deg']:.3f}",
            ])
    print(f"\nWrote {csv_path}")
    print(f"Wrote {len(rows)} per-seed JSONs in {out_dir}")


if __name__ == "__main__":
    main()
