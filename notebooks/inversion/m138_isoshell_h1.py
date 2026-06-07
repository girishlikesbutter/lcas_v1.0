"""m138 — H1 pilot: surrogate-attitude-isoshell with q0-hypothesis-cluster cost.

Pipeline per seed (single script for fast iteration):

  Stage 1: per-epoch L(t) build. Constraint epochs are bright-mag (<11)
           peaks ∪ uniform mid-LC samples. For each, sample 64k uniform-random
           SO(3) rotations, batched-surrogate-forward, threshold |pred-obs|<2σ.
  Stage 2: DBSCAN cluster L(t) into components (chord-distance metric on
           sign-canonicalised quaternions). Save per-epoch component centroids.
  Stage 3: ω-grid (16k Fibonacci dirs × 5 |ω| from peak count). For each
           candidate ω: propagate q0=I to all constraint epochs (batched RK4),
           compute q0-hypothesis cloud { q_hat[t,i] = c_i(t) · q_world(t)^-1 },
           cost = -(max neighbour count within 10° geodesic in the cloud).
  Stage 4: sort by cost, audit top-K vs truth (oracle).

Output: data/results/inversion_diagnostics/m138_isoshell_h1/seed_NNN/{
  levelset_ckpt.npz   - per-epoch kept-q indices + component centroids
  isoshell_ckpt.npz   - top-K (q0_hat, ω) candidates with cost + oracle errs
  result.json         - summary (pilot wall, rank-1 ω-err, etc.)
}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model/surrogate_model")

from lib.experiment_setup import setup_experiment  # noqa: E402
from lib.traj_source import load_truth, CANONICAL_NOISE_SIGMA  # noqa: E402
from surrogate import SurrogateModel  # noqa: E402

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
OUT_BASE = DIAG / "m138_isoshell_h1"


# ── Quaternion utilities (wxyz) ────────────────────────────────────────────

def quat_mul(q1, q2):
    """wxyz quaternion multiplication. Both can be (...,4); broadcast."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_conj(q):
    """wxyz quaternion conjugate. q (...,4) -> same shape."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_canonicalise(q):
    """Flip sign so scalar part is non-negative. Maps SO(3) double-cover to single hemisphere."""
    sign = np.where(q[..., 0] >= 0, 1.0, -1.0)
    return q * sign[..., None]


def geodesic_deg(q1, q2):
    """Geodesic angle in degrees between two unit quaternions (SO(3) double-cover-aware)."""
    d = np.abs(np.sum(q1 * q2, axis=-1))
    d = np.clip(d, -1.0, 1.0)
    return 2.0 * np.degrees(np.arccos(d))


def quat_to_R(q):
    """wxyz quaternion → 3x3 rotation matrix. q (...,4) -> (...,3,3).
    Rotation matrix maps inertial vectors to body frame: v_body = R · v_inertial,
    matching the convention used in score_lofi_surrogate.py.
    """
    qxyzw = q[..., [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


# ── SO(3) sampling ─────────────────────────────────────────────────────────

def sample_so3_random(n, rng):
    """n uniform-random unit quaternions (wxyz). Marsaglia / scipy."""
    rot = Rotation.random(n, random_state=rng)
    qxyzw = rot.as_quat()
    return qxyzw[:, [3, 0, 1, 2]]  # → wxyz


def fibonacci_sphere(n):
    """n Fibonacci-spiral points on S². Returns (n, 3) unit vectors."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


# ── Batched RK4 propagation (q0=identity, varying ω) ───────────────────────

def euler_dynamics_batch(omega, I_diag):
    """ω_dot = I^-1 · (-ω × Iω) for diagonal-inertia rigid body, batched.
    omega: (N, 3). I_diag: (3,) diagonal entries. Returns (N, 3) ω_dot."""
    Iw = omega * I_diag
    # cross(omega, Iw) per row
    cross = np.cross(omega, Iw)
    return -cross / I_diag  # (N, 3)


def quat_dot_batch(q, omega):
    """q_dot = 0.5 * Ω(omega) · q for q wxyz. q: (N, 4), omega: (N, 3) -> (N, 4).
    Equivalent to q_dot = 0.5 * q ⊗ (0, omega) in wxyz convention.
    """
    w_omega = np.zeros((q.shape[0], 4))
    w_omega[:, 1:] = omega
    return 0.5 * quat_mul(q, w_omega)


def propagate_identity_batch(omega_batch, I_diag, t_eval, dt=1.0):
    """Batched RK4 propagation of q0=identity under varying ω.

    omega_batch: (N, 3) angular velocities (rad/s).
    I_diag: (3,) diagonal inertia.
    t_eval: (T,) target times in seconds (must include 0 if needed; first
            output is at t_eval[0]).
    dt: integrator step (≤ smallest dt between t_eval recommended).

    Returns: q_eval (N, T, 4) wxyz quaternions.
    """
    N = omega_batch.shape[0]
    T = t_eval.shape[0]
    # State: (N, 4) q + (N, 3) ω, packed to (N, 7)
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))  # identity
    omega = omega_batch.copy()

    out = np.empty((N, T, 4), dtype=np.float64)
    t_now = 0.0
    next_idx = 0

    # If t_eval[0] is 0, output identity immediately
    if t_eval[0] == 0.0:
        out[:, 0, :] = q
        next_idx = 1

    while next_idx < T:
        target = t_eval[next_idx]
        # Step from t_now → target with steps of size <= dt
        n_steps = max(1, int(np.ceil((target - t_now) / dt)))
        h = (target - t_now) / n_steps
        for _ in range(n_steps):
            # RK4 on (q, ω)
            k1_q = quat_dot_batch(q, omega)
            k1_w = euler_dynamics_batch(omega, I_diag)

            q2 = q + 0.5 * h * k1_q
            w2 = omega + 0.5 * h * k1_w
            k2_q = quat_dot_batch(q2, w2)
            k2_w = euler_dynamics_batch(w2, I_diag)

            q3 = q + 0.5 * h * k2_q
            w3 = omega + 0.5 * h * k2_w
            k3_q = quat_dot_batch(q3, w3)
            k3_w = euler_dynamics_batch(w3, I_diag)

            q4 = q + h * k3_q
            w4 = omega + h * k3_w
            k4_q = quat_dot_batch(q4, w4)
            k4_w = euler_dynamics_batch(w4, I_diag)

            q = q + (h / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
            omega = omega + (h / 6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
            # Renormalise quaternions to unit length each step
            q = q / np.linalg.norm(q, axis=-1, keepdims=True)

        out[:, next_idx, :] = q
        t_now = target
        next_idx += 1

    return out


# ── Constraint epoch selection ─────────────────────────────────────────────

def select_constraint_epochs(observed_lc, mag_cap=11.0, n_target=60, rng=None):
    """Pick constraint epoch indices: bright peaks ∪ uniform fillers, mag<cap.

    Returns array of indices into observed_lc (0..len-1).
    """
    valid = np.isfinite(observed_lc)
    bright = np.where(valid & (observed_lc < mag_cap))[0]
    if len(bright) == 0:
        raise ValueError("no bright epochs (mag<cap) found")

    # Find peaks: local minima of magnitude (lower mag = brighter) in bright set.
    peak_mask = np.zeros(len(observed_lc), dtype=bool)
    for i in bright:
        if i == 0 or i == len(observed_lc) - 1:
            continue
        if observed_lc[i] < observed_lc[i-1] and observed_lc[i] < observed_lc[i+1]:
            peak_mask[i] = True
    peaks = np.where(peak_mask)[0]

    if len(bright) <= n_target:
        return bright

    # If peaks dominate, use them; otherwise take all peaks + uniform sample of remaining bright
    n_remain = n_target - len(peaks)
    if n_remain > 0:
        non_peak_bright = np.setdiff1d(bright, peaks)
        if rng is None:
            rng = np.random.default_rng(0)
        chosen = rng.choice(non_peak_bright, size=min(n_remain, len(non_peak_bright)),
                             replace=False)
        return np.sort(np.concatenate([peaks, chosen]))
    return peaks[:n_target]


# ── Stage 1: per-epoch L(t) build ──────────────────────────────────────────

def build_levelsets(seed, source, model, ctx, observed_lc,
                    n_so3=64000, sigma_combined=0.06, mag_cap=11.0,
                    n_constraint_target=60, rng_seed=42, verbose=True):
    """Build L(t) for each constraint epoch.

    Returns dict with:
      constraint_idx: (T,) indices into observation array
      q_grid: (n_so3, 4) the shared SO(3) sample (wxyz)
      kept_indices_per_epoch: list of (n_kept[t],) int arrays
    """
    rng = np.random.default_rng(rng_seed)
    constraint_idx = select_constraint_epochs(observed_lc, mag_cap=mag_cap,
                                               n_target=n_constraint_target, rng=rng)
    if verbose:
        print(f"  selected {len(constraint_idx)} constraint epochs (mag<{mag_cap})", flush=True)

    sun_vecs = ctx.sun_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    # SO(3) sample (shared across epochs).
    q_grid = sample_so3_random(n_so3, rng)
    R_grid = quat_to_R(q_grid)  # (n_so3, 3, 3)

    threshold = 2.0 * sigma_combined  # 2σ tolerance band

    kept_per_epoch = []
    n_kept_per_epoch = []
    t_total = 0.0
    for it, t_idx in enumerate(constraint_idx):
        t_eval0 = time.time()
        k1_J = sun_dirs[t_idx]  # (3,)
        k2_J = obs_dirs[t_idx]
        dist = obs_dist_km[t_idx]
        # k1_body = R · k1_inertial for each candidate rotation
        k1_body = R_grid @ k1_J  # (n_so3, 3)
        k2_body = R_grid @ k2_J
        # Surrogate forward
        pred = model.predict_magnitude(k1_body, k2_body, 0.0, 15.0, dist)
        residual = pred - observed_lc[t_idx]
        kept = np.where(np.isfinite(pred) & (np.abs(residual) < threshold))[0]
        kept_per_epoch.append(kept.astype(np.int32))
        n_kept_per_epoch.append(len(kept))
        t_total += time.time() - t_eval0
        if verbose and (it % 10 == 0 or it == len(constraint_idx) - 1):
            print(f"    epoch {it+1}/{len(constraint_idx)} (idx={t_idx}, "
                  f"obs_mag={observed_lc[t_idx]:.2f}, n_kept={len(kept)}, "
                  f"elapsed={t_total:.1f}s)", flush=True)
    if verbose:
        print(f"  Stage 1 wall: {t_total:.1f}s, "
              f"n_kept median={int(np.median(n_kept_per_epoch))}, "
              f"min={min(n_kept_per_epoch)}, max={max(n_kept_per_epoch)}",
              flush=True)
    return {
        "constraint_idx": constraint_idx,
        "q_grid": q_grid,
        "kept_per_epoch": kept_per_epoch,
        "n_kept_per_epoch": np.array(n_kept_per_epoch),
    }


# ── Stage 2: per-epoch component clustering ────────────────────────────────

def cluster_levelsets(levelset, eps_deg=5.0, min_samples=5, verbose=True):
    """For each epoch, DBSCAN-cluster the kept q's into components.

    Distance metric: chord distance (Euclidean on canonicalised quaternions).
    eps_deg=5° corresponds to chord distance 2*sin(eps/2) for small angles.

    Returns: list of {labels, centroids, sizes} per epoch. Fallback: if DBSCAN
    finds 0 clusters but kept count > 0, use the kept q's themselves as a
    flat "component cloud" (sample up to 30) so Stage 3 isn't starved.
    """
    q_grid = levelset["q_grid"]
    chord_eps = 2.0 * np.sin(np.deg2rad(eps_deg) / 2.0)

    components_per_epoch = []
    t0 = time.time()
    fallback_ct = 0
    empty_ct = 0
    for it, kept_idx in enumerate(levelset["kept_per_epoch"]):
        if len(kept_idx) == 0:
            empty_ct += 1
            components_per_epoch.append({
                "labels": np.array([], dtype=np.int32),
                "centroids": np.zeros((0, 4)),
                "sizes": np.array([], dtype=np.int32),
            })
            continue
        q_kept = quat_canonicalise(q_grid[kept_idx])
        if len(kept_idx) < min_samples:
            # Too few for DBSCAN; treat each kept q as its own component
            components_per_epoch.append({
                "labels": np.arange(len(kept_idx), dtype=np.int32),
                "centroids": q_kept,
                "sizes": np.ones(len(kept_idx), dtype=np.int32),
            })
            fallback_ct += 1
            continue
        db = DBSCAN(eps=chord_eps, min_samples=min_samples, metric='euclidean',
                    n_jobs=1)
        labels = db.fit_predict(q_kept)
        n_clusters = labels.max() + 1 if (labels >= 0).any() else 0
        if n_clusters == 0:
            # All noise points — sample up to 30 of them as a flat cloud
            n_take = min(30, len(kept_idx))
            sel = np.random.default_rng(it).choice(len(kept_idx), n_take, replace=False)
            components_per_epoch.append({
                "labels": np.full(len(kept_idx), -1, dtype=np.int32),
                "centroids": q_kept[sel],
                "sizes": np.ones(n_take, dtype=np.int32),
            })
            fallback_ct += 1
            continue
        centroids = np.zeros((n_clusters, 4))
        sizes = np.zeros(n_clusters, dtype=np.int32)
        for c in range(n_clusters):
            mask = labels == c
            sizes[c] = mask.sum()
            mean_q = q_kept[mask].mean(axis=0)
            mean_q /= np.linalg.norm(mean_q)
            centroids[c] = mean_q
        components_per_epoch.append({
            "labels": labels,
            "centroids": centroids,
            "sizes": sizes,
        })
    elapsed = time.time() - t0
    n_comps = [len(c["centroids"]) for c in components_per_epoch]
    if verbose:
        print(f"  Stage 2 wall: {elapsed:.1f}s, components/epoch median={int(np.median(n_comps))}, "
              f"min={min(n_comps)}, max={max(n_comps)}, "
              f"fallback={fallback_ct}, empty={empty_ct}", flush=True)
    return components_per_epoch


# ── Stage 3: ω-grid + q0-hypothesis-cluster cost ───────────────────────────

def estimate_omega_mag_grid(observed_lc, obs_times, n_mags=20,
                              span_low=0.3, span_high=3.0):
    """ω-magnitude grid: log-spaced bracket around peak-count estimate.

    Default: 20 values spanning 0.3× to 3× base, factor ~1.13 per step.
    Tighter than 5-value grid; fixes signal-collapse from |ω| mismatch.
    """
    valid = np.isfinite(observed_lc)
    bright_mask = valid & (observed_lc < observed_lc[valid].mean() - 1.0)
    transitions = np.diff(bright_mask.astype(int))
    n_peaks = max(1, (transitions > 0).sum())
    window = obs_times[-1] - obs_times[0]
    base = 2 * np.pi * n_peaks / window
    return np.geomspace(span_low * base, span_high * base, n_mags)


def score_isoshell_omega_grid(levelset, components, ctx, omega_dirs, omega_mags,
                              I_diag, eps_cluster_deg=8.0, verbose=True):
    """Score each (omega_dir, |omega|) candidate by max unique-epoch count.

    For each candidate ω:
      1. Propagate q0=I forward to all constraint epochs → q_world(t)
      2. For each kept point q_kept ∈ L(t): hypothesised q0_hat = q_kept · conj(q_world(t))
      3. For TRUE ω: the kept point closest to q_truth(t) maps to q0_hat ≈ q0_truth
         (or its ±X twin) at every constraint epoch. Across all 30 epochs,
         these q0_hats cluster tightly at q0_truth → max unique-epoch count ≈ 30.
         For WRONG ω: q0_hats scatter, only a few epochs accidentally co-locate.
      4. Cost = -(max unique-epoch count within eps_cluster_deg).

    Also returns the best q0 cluster centroid per candidate as q0_estimate, useful
    as a warm-start for downstream m115.
    """
    constraint_idx = levelset["constraint_idx"]
    obs_times = ctx.observation_times
    t_eval = obs_times[constraint_idx]

    n_dirs = omega_dirs.shape[0]
    n_mags = omega_mags.shape[0]
    omega_batch = (omega_dirs[:, None, :] * omega_mags[None, :, None]).reshape(-1, 3)
    N = omega_batch.shape[0]

    if verbose:
        print(f"  Stage 3: {N} candidates ({n_dirs} dirs × {n_mags} mags)", flush=True)

    # Pre-canonicalise all kept points per epoch + collect epoch labels
    q_grid = levelset["q_grid"]
    kept_q_per_epoch = []
    valid_mask = []
    for ti, kept_idx in enumerate(levelset["kept_per_epoch"]):
        if len(kept_idx) == 0:
            valid_mask.append(False)
            continue
        kept_q_per_epoch.append(quat_canonicalise(q_grid[kept_idx]))
        valid_mask.append(True)
    valid_mask = np.array(valid_mask)
    t_eval_used = t_eval[valid_mask]
    n_used_per_epoch = np.array([s.shape[0] for s in kept_q_per_epoch])
    n_total_kept = n_used_per_epoch.sum()
    epoch_labels_template = np.concatenate(
        [np.full(n, i, dtype=np.int32) for i, n in enumerate(n_used_per_epoch)])
    if verbose:
        print(f"    {valid_mask.sum()} epochs used; total {n_total_kept} kept-q hypotheses per candidate",
              flush=True)

    t0 = time.time()
    chunk = 2000
    cost = np.empty(N, dtype=np.float64)
    q0_estimate = np.empty((N, 4), dtype=np.float64)
    eps_cluster_chord = 2.0 * np.sin(np.deg2rad(eps_cluster_deg) / 2.0)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        omega_chunk = omega_batch[start:end]
        q_world_chunk = propagate_identity_batch(
            omega_chunk, I_diag, t_eval_used, dt=2.0
        )  # (n_chunk, T_used, 4)

        for ci in range(end - start):
            qw_inv = quat_conj(q_world_chunk[ci])  # (T_used, 4)
            cloud_pts = []
            for ti, q_kept_t in enumerate(kept_q_per_epoch):
                qw_inv_t = np.broadcast_to(qw_inv[ti], q_kept_t.shape).copy()
                q_hat = quat_mul(q_kept_t, qw_inv_t)
                cloud_pts.append(q_hat)
            cloud = quat_canonicalise(np.concatenate(cloud_pts, axis=0))
            tree = BallTree(cloud)
            neigh = tree.query_radius(cloud, r=eps_cluster_chord)
            best_unique = 0
            best_idx = 0
            for qi, ni in enumerate(neigh):
                u = len(np.unique(epoch_labels_template[ni]))
                if u > best_unique:
                    best_unique = u
                    best_idx = qi
            cost[start + ci] = -float(best_unique)
            # q0 estimate: mean of points at best cluster
            best_neighbours = neigh[best_idx]
            cluster_pts = cloud[best_neighbours]
            mean_q0 = cluster_pts.mean(axis=0)
            mean_q0 /= np.linalg.norm(mean_q0)
            q0_estimate[start + ci] = mean_q0

        if verbose and ((start // chunk) % 4 == 0):
            elapsed = time.time() - t0
            done = end
            eta = elapsed / max(1, done) * (N - done)
            print(f"    progress {done}/{N}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                  flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Stage 3 wall: {elapsed:.1f}s", flush=True)

    return {
        "omega_batch": omega_batch,
        "cost": cost,
        "q0_estimate": q0_estimate,
        "elapsed_s": elapsed,
    }


# ── Top-level driver ──────────────────────────────────────────────────────

def run_seed(seed, source, n_so3=64000, n_omega_dirs=16000, n_constraint_target=60,
             eps_cluster_deg=10.0, mag_cap=11.0, verbose=True):
    print(f"=== H1 pilot, seed {seed}, source {source} ===", flush=True)
    t_total = time.time()
    out_dir = OUT_BASE / f"seed_{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load truth + setup
    truth = load_truth(seed, source)
    obs_lc = truth["observed_lc"]
    true_q0 = truth["q0_wxyz"]
    true_omega = truth["omega0_rad"]
    I_tensor = truth["inertia_tensor"]
    I_diag = np.diag(I_tensor) if I_tensor.shape == (3, 3) else np.asarray(I_tensor)

    print(f"  truth ω = {true_omega}, |ω|={np.linalg.norm(true_omega):.5f}, "
          f"|ω|_deg/s={np.degrees(np.linalg.norm(true_omega)):.3f}", flush=True)

    if source == "m048":
        ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                               random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                               start_et=truth["start_et"], skip_true_lc=True)
    else:
        ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                               random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                               end_time_utc=truth["end_time_utc"], skip_true_lc=True)

    # 2. Surrogate
    print(f"  loading surrogate v2...", flush=True)
    model = SurrogateModel.load_default()

    # 3. Stage 1: levelsets
    print(f"  Stage 1: building L(t) over constraint epochs...", flush=True)
    t1 = time.time()
    levelset = build_levelsets(seed, source, model, ctx, obs_lc,
                                n_so3=n_so3, mag_cap=mag_cap,
                                n_constraint_target=n_constraint_target,
                                verbose=verbose)
    levelset_path = out_dir / "levelset_ckpt.npz"
    # Save kept indices as object array (variable-length per epoch)
    kept_obj = np.empty(len(levelset["kept_per_epoch"]), dtype=object)
    for i, k in enumerate(levelset["kept_per_epoch"]):
        kept_obj[i] = k
    np.savez_compressed(levelset_path,
                        constraint_idx=levelset["constraint_idx"],
                        q_grid=levelset["q_grid"],
                        kept_per_epoch=kept_obj,
                        n_kept_per_epoch=levelset["n_kept_per_epoch"],
                        n_so3=n_so3, mag_cap=mag_cap, sigma=CANONICAL_NOISE_SIGMA)
    print(f"  saved {levelset_path}  (Stage 1+select={time.time()-t1:.1f}s)",
          flush=True)

    # 4. Stage 2: cluster
    print(f"  Stage 2: clustering L(t) into components...", flush=True)
    components = cluster_levelsets(levelset, eps_deg=15.0, min_samples=5,
                                    verbose=verbose)
    # Save component centroids
    centroids_obj = np.empty(len(components), dtype=object)
    sizes_obj = np.empty(len(components), dtype=object)
    for i, c in enumerate(components):
        centroids_obj[i] = c["centroids"]
        sizes_obj[i] = c["sizes"]
    np.savez_compressed(out_dir / "components_ckpt.npz",
                        constraint_idx=levelset["constraint_idx"],
                        centroids=centroids_obj, sizes=sizes_obj)

    # 4b. Oracle diagnostic: does truth-ω propagation land inside L(t)?
    #    For each constraint epoch, propagate truth (q0=truth, ω=truth_omega) and
    #    measure geodesic to nearest kept-q. Should be ≤ 5° if framework is right.
    obs_times_full = ctx.observation_times
    constraint_t_eval = obs_times_full[levelset["constraint_idx"]]
    truth_quats = propagate_identity_batch(true_omega[None, :], I_diag,
                                            constraint_t_eval, dt=2.0)[0]
    # truth_quats are q_world(t) under (q0=identity, ω=true_omega).
    # The full truth trajectory is q_truth(t) = q0_truth · q_world(t)  (left-mul).
    q_truth_at_constraints = quat_mul(np.broadcast_to(true_q0, truth_quats.shape).copy(),
                                       truth_quats)
    q_truth_canon = quat_canonicalise(q_truth_at_constraints)
    # For each epoch, geodesic to nearest kept-q
    truth_dist_per_epoch = []
    for ti, kept_idx in enumerate(levelset["kept_per_epoch"]):
        if len(kept_idx) == 0:
            truth_dist_per_epoch.append(180.0)
            continue
        q_kept = quat_canonicalise(levelset["q_grid"][kept_idx])
        # geodesic in degrees
        dots = np.abs(q_kept @ q_truth_canon[ti])
        d = 2.0 * np.degrees(np.arccos(dots.max().clip(-1, 1)))
        truth_dist_per_epoch.append(d)
    truth_dist_per_epoch = np.array(truth_dist_per_epoch)
    print(f"  Oracle truth-trajectory check: median dist to nearest kept-q = "
          f"{np.median(truth_dist_per_epoch):.2f}°, max = "
          f"{truth_dist_per_epoch.max():.2f}°, "
          f"epochs ≤5° = {(truth_dist_per_epoch<=5).sum()}/{len(truth_dist_per_epoch)}",
          flush=True)

    # 5. Stage 3: ω-grid + cost
    print(f"  Stage 3: scoring ω-grid via q0-hypothesis cluster cost...", flush=True)
    omega_dirs = fibonacci_sphere(n_omega_dirs)  # (n_dirs, 3)
    omega_mags = estimate_omega_mag_grid(obs_lc, ctx.observation_times)
    print(f"    |ω| candidates: {omega_mags}", flush=True)
    print(f"    truth |ω|: {np.linalg.norm(true_omega):.5f}", flush=True)

    score = score_isoshell_omega_grid(levelset, components, ctx, omega_dirs,
                                       omega_mags, I_diag,
                                       eps_cluster_deg=eps_cluster_deg,
                                       verbose=verbose)

    # 6. Audit: oracle ω-direction error of top-K
    omega_batch = score["omega_batch"]
    cost = score["cost"]
    order = np.argsort(cost)
    top_k_n = 30
    top_k_idx = order[:top_k_n]
    top_k_omegas = omega_batch[top_k_idx]
    top_k_costs = cost[top_k_idx]

    truth_dir = true_omega / max(np.linalg.norm(true_omega), 1e-12)
    truth_mag = float(np.linalg.norm(true_omega))
    ranked_omegas_normed = top_k_omegas / np.linalg.norm(top_k_omegas, axis=1, keepdims=True)
    dot = np.clip(ranked_omegas_normed @ truth_dir, -1.0, 1.0)
    ω_dir_errs = np.degrees(np.arccos(dot))
    ω_mag_errs = (np.linalg.norm(top_k_omegas, axis=1) - truth_mag) / truth_mag * 100

    pool_min_dir = float(ω_dir_errs.min())
    rank1_dir = float(ω_dir_errs[0])

    # Also: best-overall in entire grid (oracle-only diagnostic)
    all_norms = np.linalg.norm(omega_batch, axis=1)
    all_dirs = omega_batch / all_norms[:, None].clip(1e-12)
    all_dot = np.clip(all_dirs @ truth_dir, -1.0, 1.0)
    all_dir_errs = np.degrees(np.arccos(all_dot))
    grid_min_dir = float(all_dir_errs.min())
    grid_min_idx = int(np.argmin(all_dir_errs))
    grid_min_cost_rank = int((cost <= cost[grid_min_idx]).sum() - 1)

    summary = {
        "seed": int(seed), "source": source,
        "n_so3": n_so3, "n_omega_dirs": n_omega_dirs, "n_omega_mags": len(omega_mags),
        "n_constraint_epochs": len(levelset["constraint_idx"]),
        "constraint_mag_cap": mag_cap,
        "rank1_omega_dir_err_deg": rank1_dir,
        "rank1_omega_mag_err_pct": float(ω_mag_errs[0]),
        "rank1_cost": float(top_k_costs[0]),
        "top_k_omega_dir_errs": ω_dir_errs.tolist(),
        "top_k_costs": top_k_costs.tolist(),
        "pool_min_omega_dir_err_deg_in_topK": pool_min_dir,
        "grid_min_omega_dir_err_deg": grid_min_dir,
        "grid_min_cost_rank": grid_min_cost_rank,
        "truth_omega_mag": truth_mag,
        "truth_omega_mag_deg_per_s": float(np.degrees(truth_mag)),
        "stage_walls_s": {
            "stage1": float(time.time() - t1) if False else None,
            "stage3": float(score["elapsed_s"]),
        },
        "total_wall_s": float(time.time() - t_total),
    }

    q0_est = score["q0_estimate"]
    np.savez_compressed(out_dir / "isoshell_ckpt.npz",
                        omega_batch=omega_batch, cost=cost,
                        q0_estimate=q0_est,
                        top_k_idx=top_k_idx, top_k_omegas=top_k_omegas,
                        top_k_q0_estimates=q0_est[top_k_idx],
                        top_k_omega_dir_errs=ω_dir_errs,
                        top_k_omega_mag_errs_pct=ω_mag_errs)
    # Compute q0 errors of top-K vs truth (and twin)
    twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
    q0_to_truth = []
    q0_to_twin = []
    for q0_e in q0_est[top_k_idx]:
        q0_to_truth.append(float(geodesic_deg(q0_e, true_q0)))
        q0_to_twin.append(float(geodesic_deg(q0_e, twin_q0)))
    summary["top_k_q0_to_truth_deg"] = q0_to_truth
    summary["top_k_q0_to_twin_deg"] = q0_to_twin

    with open(out_dir / "result.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"  === seed {seed} summary ===")
    print(f"  rank-1 ω-dir err: {rank1_dir:.2f}°   |ω|-err: {ω_mag_errs[0]:+.1f}%")
    print(f"  pool_min in top-{top_k_n}: {pool_min_dir:.2f}°")
    print(f"  grid_min ω-dir err: {grid_min_dir:.2f}°  (cost rank: {grid_min_cost_rank})")
    print(f"  total wall: {time.time()-t_total:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--traj-source", default="m048")
    ap.add_argument("--n-so3", type=int, default=64000)
    ap.add_argument("--n-omega-dirs", type=int, default=16000)
    ap.add_argument("--n-constraint", type=int, default=60)
    ap.add_argument("--eps-cluster-deg", type=float, default=10.0)
    ap.add_argument("--mag-cap", type=float, default=11.0)
    args = ap.parse_args()

    run_seed(args.seed, args.traj_source,
             n_so3=args.n_so3, n_omega_dirs=args.n_omega_dirs,
             n_constraint_target=args.n_constraint,
             eps_cluster_deg=args.eps_cluster_deg,
             mag_cap=args.mag_cap)


if __name__ == "__main__":
    main()
