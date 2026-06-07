"""s061a — 1-step cloud threading smoke test on seed 89.

The idea (user, 2026-05-10): instead of generating per-epoch C_t clouds
independently, start at the most-constrained anchor and EXPAND outward
under a connectability filter. A candidate q' at adjacent epoch t±1 must
lie within geodesic ball of radius `ω_max · Δt` around some q ∈ C_anchor
(in addition to satisfying brightness at t±1). Any anchor cluster whose
descendants all fail brightness at t±1 is dead.

This smoke checks:
  1. Truth's quaternion at t=208 lies in the v2 cloud (sanity).
  2. Adjacent-epoch truth quaternions (t=207, t=209) lie within
     geodesic ball of radius ω_max·Δt around truth at t=208 — i.e. the
     ω_max=1.5 dps budget is genuinely an upper bound for this seed.
  3. Threading 1 step left + 1 step right produces a smaller, brightness-
     filtered candidate set; cluster-disappearance kills some clusters.
  4. Per-pair derived ω = 2·log(q_a^{-1} ⊗ q_c) / Δt agrees with truth ω
     (within Δq quantization).

Anchor: t=208 (clean-dim per s060_anchor_topology — 5 clusters, top-3
mass 0.92). Seed 89 |ω|_truth = 0.24 dps, so ω_max=1.5 dps is ~6× truth.

Output: results/s061a_thread_smoke/seed089/{summary.json, thread.npz, plot.png}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

# Force single-thread BLAS BEFORE any heavy imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions, survive_at_epoch,
)
from lib.hifi_render import build_context  # noqa: E402
from lib.surrogate_eval import get_model  # noqa: E402

TOL_MAG = 0.10
SP_DEG = 0.0
AD_DEG = 15.0


# ---------------------------------------------------------------------------
# Quaternion utilities (wxyz convention; matches lib.twin and lib.c_t_pipeline)
# ---------------------------------------------------------------------------


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product. q1, q2 are (..., 4) wxyz; broadcasts."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (wxyz)."""
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_log(q: np.ndarray) -> np.ndarray:
    """Log of unit quaternion → (3,) rotvec. Antipode-aware (picks the rep
    with rotation angle in [0, π])."""
    q = np.asarray(q, dtype=np.float64)
    # Sign-fix so w >= 0 (rotation angle in [0, π])
    sign = np.where(q[..., 0:1] >= 0, 1.0, -1.0)
    q = q * sign
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    theta = 2.0 * np.arccos(w)
    sin_half = np.sin(theta / 2.0)
    safe = np.where(np.abs(sin_half) > 1e-12, sin_half, 1.0)
    axis = v / safe[..., None]
    rotvec = axis * theta[..., None]
    rotvec = np.where(np.abs(sin_half[..., None]) > 1e-12, rotvec, np.zeros_like(rotvec))
    return rotvec


def quat_geodesic_deg_batch(q1_wxyz, q_pool_wxyz):
    """Geodesic angle in degrees, antipode-aware. q1: (4,), pool: (N,4)."""
    dots = np.abs(q_pool_wxyz @ np.asarray(q1_wxyz, float))
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def quat_geodesic_deg_pairwise(qA: np.ndarray, qB: np.ndarray) -> np.ndarray:
    """Pairwise geodesic deg. qA: (M,4), qB: (N,4) → (M,N) array."""
    dots = np.abs(qA @ qB.T)
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def axis_angle_to_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Build (w,x,y,z) quaternion from axis (..., 3) and angle (...,)."""
    half = angle / 2.0
    s = np.sin(half)
    c = np.cos(half)
    return np.concatenate([c[..., None], axis * s[..., None]], axis=-1)


def sample_perturbations_in_geodesic_ball(
    n: int, r_max_rad: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample n random rotations with rotation angle uniform in [0, r_max].

    Axis uniform on S². Returns (n, 4) wxyz quaternions. NOT Haar measure
    on the ball (which has angular density ∝ sin²(θ/2)); just uniform
    coverage of [0, r_max] in angle. For r_max small this matters little.
    """
    # Uniform axis on S² via normalizing 3D Gaussians.
    g = rng.standard_normal((n, 3))
    axis = g / np.linalg.norm(g, axis=1, keepdims=True)
    # Uniform angle.
    angle = rng.uniform(0.0, r_max_rad, size=n)
    return axis_angle_to_quat(axis, angle)


def body_twin_canonicalize_q(q: np.ndarray) -> np.ndarray:
    """Pick the body-twin canonical representative of each q (wxyz).

    The twin map is q → q' = (-x, w, -z, y). Among {q, q'} pick the one
    whose representation satisfies a deterministic ordering. We use:
    ``q[1] >= 0`` (x-component non-negative); if x=0, fall back to
    z-component non-negative; etc. After picking, also sign-normalize so
    q[0] >= 0.

    NOTE: this collapses body-twin pairs at the QUATERNION level, mirroring
    `lib.twin.canonical_batch` but operating on q alone (without ω). The
    LC at any single epoch is invariant under twin (s043); for clustering
    we want body-twin pairs to merge.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    q_twin = np.column_stack([-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]])
    # Pick whichever has q[1] >= 0; tie-break on q[3].
    keep_orig = (q[:, 1] > 0) | ((q[:, 1] == 0) & (q[:, 3] >= 0))
    out = np.where(keep_orig[:, None], q, q_twin)
    # Sign-normalize: q[0] >= 0
    flip = out[:, 0] < 0
    out = np.where(flip[:, None], -out, out)
    return out


def greedy_cluster(q_wxyz: np.ndarray, threshold_deg: float = 40.0):
    """Greedy clustering on geodesic distance. Returns list of index arrays."""
    n = q_wxyz.shape[0]
    unassigned = np.ones(n, dtype=bool)
    clusters = []
    while unassigned.any():
        idx_pool = np.where(unassigned)[0]
        seed_idx = idx_pool[0]
        d = quat_geodesic_deg_batch(q_wxyz[seed_idx], q_wxyz[idx_pool])
        members_local = idx_pool[d < threshold_deg]
        clusters.append(members_local)
        unassigned[members_local] = False
    clusters.sort(key=lambda c: -len(c))
    return clusters


# ---------------------------------------------------------------------------
# Threading step
# ---------------------------------------------------------------------------


def thread_one_step(
    anchor_q_wxyz: np.ndarray,        # (M, 4) anchor cloud (canonicalized)
    anchor_cluster_id: np.ndarray,    # (M,) cluster id per anchor survivor
    target_t: int,                    # epoch index to evaluate
    delta_t_sec: float,               # |t_target - t_anchor| in seconds
    omega_max_rad_per_sec: float,
    n_per_anchor: int,
    model,
    sun_unit, obs_unit, obs_dist, mag_truth,
    rng: np.random.Generator,
):
    """Sample tube candidates around each anchor q, brightness-filter at
    target_t, return (q_survivors, predecessor_idx, derived_omega_body).

    Sampling: for each anchor q_a, draw n_per_anchor random δq from the
    geodesic ball of radius ω_max·Δt. Build candidate q_c = δq ⊗ q_a (so
    q_c is "δq applied in inertial frame" = body-frame ω flipped by q_a;
    for ball coverage the order doesn't matter — both left and right
    multiplication produce the same set of points within geodesic radius
    r_max around q_a).

    Derived ω is computed as: q_a → q_c via q_a ⊗ exp(0.5·ω_body·Δt),
    so ω_body = 2 · log(q_a^{-1} ⊗ q_c) / Δt. (Body-frame, since q_a is
    inertial-from-body and right-multiplication is the body-frame
    perturbation.)
    """
    M = anchor_q_wxyz.shape[0]
    r_max = omega_max_rad_per_sec * delta_t_sec  # radians

    # Tile anchor q (each anchor produces n_per_anchor children)
    q_anchor_tiled = np.repeat(anchor_q_wxyz, n_per_anchor, axis=0)  # (M*N, 4)
    pred_idx_tiled = np.repeat(np.arange(M), n_per_anchor)            # (M*N,)
    cluster_tiled = anchor_cluster_id[pred_idx_tiled]                 # (M*N,)

    # Sample δq for each child
    delta_q = sample_perturbations_in_geodesic_ball(M * n_per_anchor, r_max, rng)

    # Build child q. Use right-multiplication q_c = q_a ⊗ δq so δq encodes
    # the body-frame perturbation directly. Then derived ω_body = 2·log(δq)/Δt.
    q_child = quat_mul_batch(q_anchor_tiled, delta_q)

    # Brightness check at target_t
    R_child = Rotation.from_quat(q_child[:, [1, 2, 3, 0]]).as_matrix()
    k1_b = R_child @ sun_unit[target_t]
    k2_b = R_child @ obs_unit[target_t]
    pred_mag = model.predict_magnitude(
        k1_b, k2_b, SP_DEG, AD_DEG,
        np.full(R_child.shape[0], float(obs_dist[target_t]))
    )
    keep = np.abs(pred_mag - float(mag_truth[target_t])) < TOL_MAG

    # Survivor outputs
    q_survivors = q_child[keep]
    delta_q_survivors = delta_q[keep]
    pred_idx_survivors = pred_idx_tiled[keep]
    cluster_survivors = cluster_tiled[keep]

    # Derived ω = 2·log(δq) / Δt (body frame)
    rotvec_survivors = quat_log(delta_q_survivors)  # (k, 3) rotvec
    omega_body_derived = rotvec_survivors / delta_t_sec  # rad/s

    return {
        "q_survivors": q_survivors,
        "delta_q": delta_q_survivors,
        "predecessor_idx": pred_idx_survivors,
        "predecessor_cluster": cluster_survivors,
        "omega_body_derived": omega_body_derived,
        "n_candidates": M * n_per_anchor,
        "n_survivors": int(keep.sum()),
        "r_max_rad": r_max,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=89)
    ap.add_argument("--anchor-t", type=int, default=208)
    ap.add_argument("--n-pool", type=int, default=25_000)
    ap.add_argument("--cluster-threshold-deg", type=float, default=40.0)
    ap.add_argument("--omega-max-dps", type=float, default=1.5)
    ap.add_argument("--n-per-anchor", type=int, default=50)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== s061a — 1-step thread smoke ===")
    print(f"seed={args.seed}, anchor_t={args.anchor_t}, "
          f"ω_max={args.omega_max_dps} dps, n_per_anchor={args.n_per_anchor}\n")

    out_dir = SURVEY / "results" / "s061a_thread_smoke" / f"seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build context (cached SPICE state, satellite, surrogate)
    print("loading context + v2 surrogate...")
    ctx = build_context(seed=args.seed)
    obs_dist = np.asarray(ctx["obs_dist"])
    sun_unit, obs_unit = compute_j2000_units(ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    mag_truth = np.asarray(ctx["mag_hifi_truth"])
    obs_times = np.asarray(ctx["observation_times"])
    truth_quaternions = np.asarray(ctx["q0_truth"])  # placeholder
    # The full truth quaternion array is in the NPZ — re-load it:
    from lib.traj_load import load_truth
    full_truth = load_truth(args.seed)
    truth_quaternions = np.asarray(full_truth["quaternions"])  # (N, 4)
    omega0_truth = np.asarray(ctx["omega0_truth_rad"])         # (3,)

    n_obs = obs_times.shape[0]
    omega_max_rad = args.omega_max_dps * np.pi / 180.0
    print(f"n_obs={n_obs}, |ω|_truth={np.linalg.norm(omega0_truth)*180/np.pi:.3f} dps")
    print(f"ω_max_rad={omega_max_rad:.4f}")

    # === STAGE 0: anchor cloud =================================================
    t_a = args.anchor_t
    delta_t_left = obs_times[t_a] - obs_times[t_a - 1]
    delta_t_right = obs_times[t_a + 1] - obs_times[t_a]
    print(f"\nΔt_left={delta_t_left:.3f}s, Δt_right={delta_t_right:.3f}s")
    r_max_left = omega_max_rad * delta_t_left
    r_max_right = omega_max_rad * delta_t_right
    print(f"r_max_left={r_max_left:.4f} rad ({np.degrees(r_max_left):.2f}°)")
    print(f"r_max_right={r_max_right:.4f} rad ({np.degrees(r_max_right):.2f}°)")

    print(f"\n--- ANCHOR cloud at t={t_a} ---")
    t0 = time.time()
    pool = sample_so3_pool(args.n_pool, sample_seed=args.rng_seed)
    model = get_model()
    k1_b, k2_b = project_directions(pool["R_cache"], sun_unit[t_a], obs_unit[t_a])
    _, keep = survive_at_epoch(
        model, k1_b, k2_b, float(obs_dist[t_a]),
        SP_DEG, AD_DEG, float(mag_truth[t_a]), TOL_MAG,
    )
    survivors_q = pool["q_pool_wxyz"][keep]
    print(f"  raw |C_t={t_a}| = {survivors_q.shape[0]} (wall {time.time()-t0:.1f}s)")

    # Body-twin canonicalize for clustering
    survivors_q_canon = body_twin_canonicalize_q(survivors_q)
    print(f"  after body-twin canon: |C| = {survivors_q_canon.shape[0]} (note: count "
          f"unchanged; collapse happens in cluster-space — pairs merge into single rep)")

    # Cluster
    clusters = greedy_cluster(survivors_q_canon, args.cluster_threshold_deg)
    cluster_id = np.full(survivors_q_canon.shape[0], -1, dtype=np.int64)
    for cid, members in enumerate(clusters):
        cluster_id[members] = cid
    cluster_sizes = [len(c) for c in clusters]
    print(f"  clusters at {args.cluster_threshold_deg}°: n={len(clusters)} "
          f"sizes={cluster_sizes[:6]}{'...' if len(cluster_sizes) > 6 else ''}")

    # Truth quaternion at t_a
    q_truth_a = truth_quaternions[t_a]
    q_truth_a_canon = body_twin_canonicalize_q(q_truth_a[None, :])[0]
    truth_dist_to_canon = quat_geodesic_deg_batch(q_truth_a_canon, survivors_q_canon)
    truth_in_cloud_idx = int(np.argmin(truth_dist_to_canon))
    truth_in_cloud_dist = float(truth_dist_to_canon[truth_in_cloud_idx])
    truth_cluster = int(cluster_id[truth_in_cloud_idx])
    print(f"  truth at t={t_a}: closest survivor at {truth_in_cloud_dist:.2f}°, "
          f"in cluster {truth_cluster} (size {cluster_sizes[truth_cluster]})")

    # === STAGE 1: thread one step right =======================================
    print(f"\n--- THREAD t={t_a} → t={t_a+1} ---")
    t0 = time.time()
    rng = np.random.default_rng(args.rng_seed + 1)
    res_right = thread_one_step(
        anchor_q_wxyz=survivors_q_canon,
        anchor_cluster_id=cluster_id,
        target_t=t_a + 1,
        delta_t_sec=delta_t_right,
        omega_max_rad_per_sec=omega_max_rad,
        n_per_anchor=args.n_per_anchor,
        model=model,
        sun_unit=sun_unit, obs_unit=obs_unit,
        obs_dist=obs_dist, mag_truth=mag_truth,
        rng=rng,
    )
    print(f"  candidates: {res_right['n_candidates']} → survivors: {res_right['n_survivors']} "
          f"(retention {res_right['n_survivors']/res_right['n_candidates']*100:.2f}%, "
          f"wall {time.time()-t0:.1f}s)")

    surviving_clusters_right = np.unique(res_right["predecessor_cluster"])
    print(f"  clusters with ≥1 surviving descendant: "
          f"{len(surviving_clusters_right)}/{len(clusters)} "
          f"(IDs: {sorted(surviving_clusters_right.tolist())})")

    # Per-cluster survival count
    for cid in range(len(clusters)):
        n_surv = int((res_right["predecessor_cluster"] == cid).sum())
        n_anchor = cluster_sizes[cid]
        n_candidates_in_cluster = n_anchor * args.n_per_anchor
        ret_pct = n_surv / n_candidates_in_cluster * 100 if n_candidates_in_cluster > 0 else 0.0
        marker = "  ←TRUTH" if cid == truth_cluster else ""
        print(f"    cluster {cid} (size {n_anchor}): {n_surv}/{n_candidates_in_cluster} "
              f"survived ({ret_pct:.1f}%){marker}")

    # === STAGE 2: thread one step left ========================================
    print(f"\n--- THREAD t={t_a} → t={t_a-1} ---")
    t0 = time.time()
    rng2 = np.random.default_rng(args.rng_seed + 2)
    res_left = thread_one_step(
        anchor_q_wxyz=survivors_q_canon,
        anchor_cluster_id=cluster_id,
        target_t=t_a - 1,
        delta_t_sec=delta_t_left,
        omega_max_rad_per_sec=omega_max_rad,
        n_per_anchor=args.n_per_anchor,
        model=model,
        sun_unit=sun_unit, obs_unit=obs_unit,
        obs_dist=obs_dist, mag_truth=mag_truth,
        rng=rng2,
    )
    print(f"  candidates: {res_left['n_candidates']} → survivors: {res_left['n_survivors']} "
          f"(retention {res_left['n_survivors']/res_left['n_candidates']*100:.2f}%, "
          f"wall {time.time()-t0:.1f}s)")

    surviving_clusters_left = np.unique(res_left["predecessor_cluster"])
    print(f"  clusters with ≥1 surviving descendant: "
          f"{len(surviving_clusters_left)}/{len(clusters)} "
          f"(IDs: {sorted(surviving_clusters_left.tolist())})")

    # === STAGE 3: truth tracking ==============================================
    print(f"\n--- TRUTH TRACKING ---")
    q_truth_left = truth_quaternions[t_a - 1]
    q_truth_right = truth_quaternions[t_a + 1]

    # Truth geodesic step
    geo_left = quat_geodesic_deg_batch(q_truth_a, q_truth_left[None, :])[0]
    geo_right = quat_geodesic_deg_batch(q_truth_a, q_truth_right[None, :])[0]
    print(f"  truth geodesic |q[208]→q[207]| = {geo_left:.3f}°")
    print(f"  truth geodesic |q[208]→q[209]| = {geo_right:.3f}°")
    print(f"  expected from truth |ω|·Δt: {np.linalg.norm(omega0_truth)*delta_t_left*180/np.pi:.3f}° "
          f"({np.linalg.norm(omega0_truth)*delta_t_right*180/np.pi:.3f}° right)")
    print(f"  ω_max budget {args.omega_max_dps} dps × Δt → {np.degrees(r_max_left):.2f}° "
          f"({np.degrees(r_max_right):.2f}° right)")
    truth_within_budget_left = geo_left < np.degrees(r_max_left)
    truth_within_budget_right = geo_right < np.degrees(r_max_right)
    print(f"  truth within ω_max budget? left={truth_within_budget_left}, right={truth_within_budget_right}")

    # Truth-derived ω from finite-diff (truth quaternions only, sanity)
    delta_q_truth_right = quat_mul_batch(quat_conj(q_truth_a), q_truth_right)
    omega_truth_derived_right = quat_log(delta_q_truth_right) / delta_t_right
    omega_err_dps = (np.linalg.norm(omega_truth_derived_right) - np.linalg.norm(omega0_truth)) * 180 / np.pi
    print(f"  truth FD ω at t=208→209: |ω|={np.linalg.norm(omega_truth_derived_right)*180/np.pi:.4f} dps "
          f"(truth |ω|={np.linalg.norm(omega0_truth)*180/np.pi:.4f}, err {omega_err_dps:+.4f} dps)")

    # === STAGE 4: derived-ω quality from threaded survivors ===================
    print(f"\n--- DERIVED ω FROM THREADED SURVIVORS ---")
    if res_right["n_survivors"] > 0:
        omega_mag_dps = np.linalg.norm(res_right["omega_body_derived"], axis=1) * 180 / np.pi
        print(f"  RIGHT thread: |ω_derived| min/median/max = "
              f"{omega_mag_dps.min():.3f}/{np.median(omega_mag_dps):.3f}/{omega_mag_dps.max():.3f} dps")

        # Subset to truth cluster
        truth_cluster_mask = res_right["predecessor_cluster"] == truth_cluster
        if truth_cluster_mask.sum() > 0:
            omega_in_truth_cluster = res_right["omega_body_derived"][truth_cluster_mask]
            mag_in_truth = np.linalg.norm(omega_in_truth_cluster, axis=1) * 180 / np.pi
            # Compare derived ω vector to truth ω
            cos_to_truth = (omega_in_truth_cluster @ omega0_truth) / (
                np.linalg.norm(omega_in_truth_cluster, axis=1) * np.linalg.norm(omega0_truth)
            )
            angle_to_truth = np.degrees(np.arccos(np.clip(np.abs(cos_to_truth), 0, 1)))
            print(f"    in TRUTH cluster ({truth_cluster_mask.sum()} survivors): "
                  f"|ω| {mag_in_truth.min():.3f}-{mag_in_truth.max():.3f} dps, "
                  f"axis err {angle_to_truth.min():.2f}-{angle_to_truth.max():.2f}°")
        else:
            print(f"    truth cluster has NO surviving descendants → architecture broken on truth basin")

    # === STAGE 5: save artefacts ==============================================
    summary = {
        "seed": int(args.seed),
        "anchor_t": int(args.anchor_t),
        "n_pool": int(args.n_pool),
        "tol_mag": float(TOL_MAG),
        "omega_max_dps": float(args.omega_max_dps),
        "n_per_anchor": int(args.n_per_anchor),
        "cluster_threshold_deg": float(args.cluster_threshold_deg),
        "anchor_cloud": {
            "Ct": int(survivors_q.shape[0]),
            "n_clusters": int(len(clusters)),
            "cluster_sizes_top6": cluster_sizes[:6],
            "truth_cluster_id": int(truth_cluster),
            "truth_dist_to_nearest_canon_deg": float(truth_in_cloud_dist),
            "truth_cluster_size": int(cluster_sizes[truth_cluster]),
        },
        "thread_right": {
            "target_t": int(t_a + 1),
            "delta_t_sec": float(delta_t_right),
            "r_max_deg": float(np.degrees(r_max_right)),
            "n_candidates": int(res_right["n_candidates"]),
            "n_survivors": int(res_right["n_survivors"]),
            "n_clusters_with_descendants": int(len(surviving_clusters_right)),
            "truth_cluster_survives": bool(truth_cluster in surviving_clusters_right.tolist()),
        },
        "thread_left": {
            "target_t": int(t_a - 1),
            "delta_t_sec": float(delta_t_left),
            "r_max_deg": float(np.degrees(r_max_left)),
            "n_candidates": int(res_left["n_candidates"]),
            "n_survivors": int(res_left["n_survivors"]),
            "n_clusters_with_descendants": int(len(surviving_clusters_left)),
            "truth_cluster_survives": bool(truth_cluster in surviving_clusters_left.tolist()),
        },
        "truth_tracking": {
            "geo_truth_207_to_208_deg": float(geo_left),
            "geo_truth_208_to_209_deg": float(geo_right),
            "omega_truth_dps": float(np.linalg.norm(omega0_truth) * 180 / np.pi),
            "omega_truth_FD_right_dps": float(np.linalg.norm(omega_truth_derived_right) * 180 / np.pi),
            "omega_max_budget_deg_right": float(np.degrees(r_max_right)),
            "truth_within_budget_left": bool(truth_within_budget_left),
            "truth_within_budget_right": bool(truth_within_budget_right),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Save NPZ with full survivor data for further analysis
    npz_path = out_dir / "thread.npz"
    np.savez(
        npz_path,
        anchor_q_canon=survivors_q_canon,
        anchor_cluster_id=cluster_id,
        right_q_survivors=res_right["q_survivors"],
        right_predecessor_idx=res_right["predecessor_idx"],
        right_predecessor_cluster=res_right["predecessor_cluster"],
        right_omega_body=res_right["omega_body_derived"],
        left_q_survivors=res_left["q_survivors"],
        left_predecessor_idx=res_left["predecessor_idx"],
        left_predecessor_cluster=res_left["predecessor_cluster"],
        left_omega_body=res_left["omega_body_derived"],
        omega0_truth_rad=omega0_truth,
        q_truth_left=q_truth_left,
        q_truth_anchor=q_truth_a,
        q_truth_right=q_truth_right,
    )
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()
