#!/usr/bin/env python3
"""
m118 costs — Library of 5 alignment-cost variants.

Imported by m118_diag. No execution at import.

All scorers take:
  kernel_data  : dict-like (from np.load(kernel.npz))
  ipl_data     : dict-like (from np.load(ipl_all_epochs.npz))
  phi_arr      : (N_PHI,) float64, uniform [0, 2pi)
  mode         : 'spec_peaks_only' or 'extended' (which epochs to score over)

Return shape: cost[N_DIRS, N_MAGS, N_ANCHOR_CENTROIDS, N_PHI] float32.

Convention: quaternions are wxyz. scipy Rotation uses xyzw; we permute
explicitly at every boundary.

Cost formula:
  For each constraint epoch ep:
    q_total = q_anchor * q_delta[dir, mag, ep]   (Hamilton product)
    pab_body = R(q_total) @ pab_j2000[ep]
    best_dot = max_{t in targets(ep)} (pab_body . t)
    cost += w_ep * (1 - best_dot)**2
"""

import numpy as np
from scipy.spatial.transform import Rotation

Z_NORMALS = {4, 5}  # match m103 convention

# ---- IS-901 facet normal gating (copied verbatim from m103_hybrid) -----
def get_allowed_normals(mag):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))


# ---- Anchor attitude builder (copied verbatim from m103_hybrid) --------
def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# ---- Hamilton quaternion product (wxyz), batched ---------------------------
def _qmul_wxyz(q1, q2):
    """q1 * q2, both arrays broadcast-compatible with trailing dim 4 (wxyz)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _rotate_by_q(q_wxyz, v):
    """Rotate vector v by quaternion q (wxyz). Supports broadcasting.

    q_wxyz: (..., 4), v: (..., 3). Returns (..., 3).
    """
    w = q_wxyz[..., 0:1]
    u = q_wxyz[..., 1:4]
    # Rodrigues: v' = v + 2 w (u x v) + 2 u x (u x v)
    uxv = np.cross(u, v)
    return v + 2.0 * w * uxv + 2.0 * np.cross(u, uxv)


# ---- Anchor attitude grid: (N_C, N_PHI) quaternions ------------------------
def _build_anchor_quats(anchor_centroids, pab_anchor_j2000, truth_q0,
                       anchor_time, I_tensor, phi_arr):
    """Build q_anchor[c, phi] — attitude at anchor_time.

    We need the body-frame image of pab_j2000[anchor_epoch] to sit on centroid c.
    The anchor centroids are already body-frame directions (from IPL), so the
    target body vector IS the centroid. Use anchor_q_from_phi(phi, centroid, pab).
    """
    n_c = anchor_centroids.shape[0]
    n_phi = len(phi_arr)
    q_anchor = np.zeros((n_c, n_phi, 4), dtype=np.float64)
    for ci in range(n_c):
        n_body = anchor_centroids[ci].astype(np.float64)
        nb_norm = np.linalg.norm(n_body)
        if nb_norm < 1e-8:
            # degenerate — identity
            q_anchor[ci] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        n_body = n_body / nb_norm
        for pi, phi in enumerate(phi_arr):
            q_anchor[ci, pi] = anchor_q_from_phi(phi, n_body, pab_anchor_j2000)
    return q_anchor


# ---- Core cost evaluation --------------------------------------------------
def _score_chunk(args):
    """Worker: score a direction chunk. Module-level so it's picklable."""
    (q_delta_chunk, q_anchor, pab_j2000_at_constraints,
     target_list_per_epoch, weights_per_epoch) = args
    Dc, M, E, _ = q_delta_chunk.shape
    C, P, _ = q_anchor.shape
    cost = np.zeros((Dc, M, C, P), dtype=np.float64)
    q_delta64 = q_delta_chunk.astype(np.float64)
    for ep in range(E):
        q_d = q_delta64[:, :, ep, :]
        q_a = q_anchor[None, None, :, :, :]
        q_d_b = q_d[:, :, None, None, :]
        q_total = _qmul_wxyz(q_a, q_d_b)
        pab = pab_j2000_at_constraints[ep]
        pab_body = _rotate_by_q(q_total, pab[None, None, None, None, :])
        targets = np.atleast_2d(target_list_per_epoch[ep])
        if targets.shape[0] == 0:
            continue
        dots = np.einsum('dmcpi,ti->dmcpt', pab_body, targets)
        best = dots.max(axis=-1)
        cost += float(weights_per_epoch[ep]) * (1.0 - best) ** 2
    return cost.astype(np.float32)


def _score_chunk_ring(args):
    """Worker: score a direction chunk under RING cost.
    cost_ep = (best_dot - expected_cos[ep])**2 — penalises deviation from the
    expected angular distance to the nearest target (loop radius proxy).
    """
    (q_delta_chunk, q_anchor, pab_j2000_at_constraints,
     target_list_per_epoch, weights_per_epoch, expected_cos_per_epoch) = args
    Dc, M, E, _ = q_delta_chunk.shape
    C, P, _ = q_anchor.shape
    cost = np.zeros((Dc, M, C, P), dtype=np.float64)
    q_delta64 = q_delta_chunk.astype(np.float64)
    for ep in range(E):
        q_d = q_delta64[:, :, ep, :]
        q_a = q_anchor[None, None, :, :, :]
        q_d_b = q_d[:, :, None, None, :]
        q_total = _qmul_wxyz(q_a, q_d_b)
        pab = pab_j2000_at_constraints[ep]
        pab_body = _rotate_by_q(q_total, pab[None, None, None, None, :])
        targets = np.atleast_2d(target_list_per_epoch[ep])
        if targets.shape[0] == 0:
            continue
        dots = np.einsum('dmcpi,ti->dmcpt', pab_body, targets)
        # For ring cost we pick the target closest to PAB (max dot), compute
        # its angular distance, and penalise deviation from expected.
        best = dots.max(axis=-1)
        exp_cos = float(expected_cos_per_epoch[ep])
        cost += float(weights_per_epoch[ep]) * (best - exp_cos) ** 2
    return cost.astype(np.float32)


def _score_with_targets_ring(q_delta, q_anchor, pab_j2000_at_constraints,
                             target_list_per_epoch, weights_per_epoch,
                             expected_cos_per_epoch, n_workers=None):
    """Ring-cost version. expected_cos_per_epoch: (E,) float — cos(expected angle)."""
    import os
    import multiprocessing as mp
    if n_workers is None:
        n_workers = int(os.environ.get('MICRO118_COST_WORKERS', '16'))
    D, M, E, _ = q_delta.shape
    C, P, _ = q_anchor.shape
    if n_workers <= 1 or D <= n_workers:
        return _score_chunk_ring((q_delta, q_anchor, pab_j2000_at_constraints,
                                  target_list_per_epoch, weights_per_epoch,
                                  expected_cos_per_epoch))
    chunks = np.array_split(np.arange(D), n_workers)
    args_list = []
    for chunk_idx in chunks:
        if len(chunk_idx) == 0:
            continue
        args_list.append((q_delta[chunk_idx[0]:chunk_idx[-1] + 1],
                         q_anchor, pab_j2000_at_constraints,
                         target_list_per_epoch, weights_per_epoch,
                         expected_cos_per_epoch))
    with mp.Pool(len(args_list)) as pool:
        chunk_results = pool.map(_score_chunk_ring, args_list)
    cost = np.empty((D, M, C, P), dtype=np.float32)
    for chunk_idx, chunk_cost in zip(chunks, chunk_results):
        if len(chunk_idx) == 0:
            continue
        cost[chunk_idx[0]:chunk_idx[-1] + 1] = chunk_cost
    return cost


def _score_with_targets(q_delta, q_anchor, pab_j2000_at_constraints,
                       target_list_per_epoch, weights_per_epoch,
                       n_workers=None):
    """Evaluate cost tensor with direction-axis parallelism.

    q_delta: (D, M, E, 4) float32 (wxyz) — propagated from anchor time
    q_anchor: (C, P, 4) float64 (wxyz)
    pab_j2000_at_constraints: (E, 3) float64
    target_list_per_epoch: list length E, each item shape (T_e, 3) or (3,)
    weights_per_epoch: (E,) float64
    n_workers: parallel worker count. Default from MICRO118_COST_WORKERS env (def 16).

    Returns cost[D, M, C, P] float32.
    """
    import os
    import multiprocessing as mp
    if n_workers is None:
        n_workers = int(os.environ.get('MICRO118_COST_WORKERS', '16'))

    D, M, E, _ = q_delta.shape
    C, P, _ = q_anchor.shape

    if n_workers <= 1 or D <= n_workers:
        return _score_chunk((q_delta, q_anchor, pab_j2000_at_constraints,
                             target_list_per_epoch, weights_per_epoch))

    # Split directions into ~equal chunks
    chunks = np.array_split(np.arange(D), n_workers)
    args_list = []
    for chunk_idx in chunks:
        if len(chunk_idx) == 0:
            continue
        args_list.append((q_delta[chunk_idx[0]:chunk_idx[-1] + 1],
                         q_anchor, pab_j2000_at_constraints,
                         target_list_per_epoch, weights_per_epoch))

    with mp.Pool(len(args_list)) as pool:
        chunk_results = pool.map(_score_chunk, args_list)

    # Stitch
    cost = np.empty((D, M, C, P), dtype=np.float32)
    offset = 0
    for chunk_idx, chunk_cost in zip(chunks, chunk_results):
        if len(chunk_idx) == 0:
            continue
        cost[chunk_idx[0]:chunk_idx[-1] + 1] = chunk_cost
    return cost


# ---- Target builders -------------------------------------------------------
def _targets_facet_normal(constraint_epochs, observed_lc, unique_normals):
    targets = []
    for ep in constraint_epochs:
        mag = float(observed_lc[ep])
        allowed = get_allowed_normals(mag)
        targets.append(np.asarray(unique_normals[allowed], dtype=np.float64))
    return targets


def _targets_all_centroids(constraint_epochs, centroids_obj):
    targets = []
    for ep in constraint_epochs:
        c = centroids_obj[ep]
        if len(c) == 0:
            targets.append(np.zeros((0, 3), dtype=np.float64))
        else:
            arr = np.asarray(c, dtype=np.float64).reshape(-1, 3)
            # normalize just in case
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            targets.append(arr / norms)
    return targets


def _targets_active_centroid(constraint_epochs, active_dirs):
    targets = []
    for ep in constraint_epochs:
        d = np.asarray(active_dirs[ep], dtype=np.float64).reshape(3)
        n = np.linalg.norm(d)
        if n < 1e-8:
            targets.append(np.zeros((0, 3), dtype=np.float64))
        else:
            targets.append((d / n).reshape(1, 3))
    return targets


def _weights_uniform(n_ep):
    return np.ones(n_ep, dtype=np.float64)


def _weights_inverse_length(lengths_per_ep, n_ep, eps=1e-3):
    w = 1.0 / (np.asarray(lengths_per_ep, dtype=np.float64) + eps)
    total = w.sum()
    if total > 0:
        w = w * (n_ep / total)
    return w


# ---- Public scoring API ----------------------------------------------------
def _load_common(kernel_data, ipl_data):
    q_delta = kernel_data['q_delta']
    constraint_epochs = kernel_data['constraint_epochs']
    pab_j2000_at_constraints = kernel_data['pab_j2000_at_constraints']
    anchor_centroids = kernel_data['anchor_centroids']
    anchor_epoch = int(kernel_data['anchor_epoch'])
    anchor_time = float(kernel_data['anchor_time'])
    pab_anchor = np.asarray(kernel_data['pab_j2000_at_constraints'])  # unused here
    # The J2000 PAB at the anchor epoch itself — reconstruct from master via
    # the caller. Caller must include 'pab_anchor_j2000' in kernel_data.
    return q_delta, constraint_epochs, pab_j2000_at_constraints, anchor_centroids


def _get_pab_anchor_j2000(kernel_data):
    # If explicitly stored, use it. Otherwise error.
    if 'pab_anchor_j2000' in kernel_data.files if hasattr(kernel_data, 'files') else kernel_data:
        return kernel_data['pab_anchor_j2000']
    raise KeyError("kernel_data must contain 'pab_anchor_j2000'")


def _build_q_anchor_from_kernel(kernel_data, phi_arr):
    """Anchor attitudes using the J2000 PAB stored at the anchor epoch."""
    # Read pab at anchor from the 'pab_anchor_j2000' field if present,
    # else synthesize via the kernel's stored full array.
    anchor_centroids = kernel_data['anchor_centroids']
    pab_anchor = np.asarray(kernel_data['pab_anchor_j2000']).reshape(3)
    n_c = anchor_centroids.shape[0]
    n_phi = len(phi_arr)
    q_anchor = np.zeros((n_c, n_phi, 4), dtype=np.float64)
    for ci in range(n_c):
        n_body = np.asarray(anchor_centroids[ci], dtype=np.float64)
        nb_norm = np.linalg.norm(n_body)
        if nb_norm < 1e-8:
            q_anchor[ci] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        n_body = n_body / nb_norm
        for pi, phi in enumerate(phi_arr):
            q_anchor[ci, pi] = anchor_q_from_phi(phi, n_body, pab_anchor)
    return q_anchor


def _select_epoch_mask(kernel_data, mode):
    """Return boolean mask over constraint_epochs indicating which to score."""
    constraint_epochs = np.asarray(kernel_data['constraint_epochs'])
    if mode == 'spec_peaks_only':
        spec_peaks = np.asarray(kernel_data['spec_peaks'])
        return np.isin(constraint_epochs, spec_peaks)
    elif mode == 'extended':
        # tight_epochs OR spec_peaks
        spec_peaks = np.asarray(kernel_data['spec_peaks'])
        tight_epochs = np.asarray(kernel_data['tight_epochs'])
        union = np.union1d(spec_peaks, tight_epochs)
        return np.isin(constraint_epochs, union)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _apply_mask(q_delta, pab, target_list, weights, mask):
    if mask.all():
        return q_delta, pab, target_list, weights
    idx = np.where(mask)[0]
    return (q_delta[:, :, idx, :],
            pab[idx],
            [target_list[i] for i in idx],
            weights[idx])


def score_facet_normal(kernel_data, ipl_data, phi_arr,
                       mode='spec_peaks_only', **kw):
    """Variant 1: allowed facet normals per epoch (m103-style)."""
    q_delta = kernel_data['q_delta']
    pab = kernel_data['pab_j2000_at_constraints']
    constraint_epochs = np.asarray(kernel_data['constraint_epochs'])
    observed_lc = kernel_data['observed_lc']
    unique_normals = kernel_data['unique_normals']
    q_anchor = _build_q_anchor_from_kernel(kernel_data, phi_arr)

    targets = _targets_facet_normal(constraint_epochs, observed_lc, unique_normals)
    weights = _weights_uniform(len(constraint_epochs))

    mask = _select_epoch_mask(kernel_data, mode)
    q_d, pab_m, t_m, w_m = _apply_mask(q_delta, pab, targets, weights, mask)
    return _score_with_targets(q_d, q_anchor, pab_m, t_m, w_m)


def score_ipl_centroid_uniform(kernel_data, ipl_data, phi_arr,
                               mode='spec_peaks_only', **kw):
    """Variant 2: all IPL centroids at each epoch, uniform weight."""
    q_delta = kernel_data['q_delta']
    pab = kernel_data['pab_j2000_at_constraints']
    constraint_epochs = np.asarray(kernel_data['constraint_epochs'])
    seed = int(kernel_data['seed'])
    centroids_obj = ipl_data[f"s{seed:03d}_centroids"]
    q_anchor = _build_q_anchor_from_kernel(kernel_data, phi_arr)

    targets = _targets_all_centroids(constraint_epochs, centroids_obj)
    weights = _weights_uniform(len(constraint_epochs))

    mask = _select_epoch_mask(kernel_data, mode)
    q_d, pab_m, t_m, w_m = _apply_mask(q_delta, pab, targets, weights, mask)
    return _score_with_targets(q_d, q_anchor, pab_m, t_m, w_m)


def score_ipl_centroid_weighted(kernel_data, ipl_data, phi_arr,
                                mode='spec_peaks_only',
                                tightness_norm='inverse_length', **kw):
    """Variant 3: all centroids, weighted by IPL tightness (1/length+eps)."""
    q_delta = kernel_data['q_delta']
    pab = kernel_data['pab_j2000_at_constraints']
    constraint_epochs = np.asarray(kernel_data['constraint_epochs'])
    seed = int(kernel_data['seed'])
    centroids_obj = ipl_data[f"s{seed:03d}_centroids"]
    lengths = ipl_data[f"s{seed:03d}_lengths"]
    q_anchor = _build_q_anchor_from_kernel(kernel_data, phi_arr)

    targets = _targets_all_centroids(constraint_epochs, centroids_obj)
    lengths_per_ep = lengths[constraint_epochs]
    if tightness_norm == 'inverse_length':
        weights = _weights_inverse_length(lengths_per_ep, len(constraint_epochs))
    else:
        weights = _weights_uniform(len(constraint_epochs))

    mask = _select_epoch_mask(kernel_data, mode)
    q_d, pab_m, t_m, w_m = _apply_mask(q_delta, pab, targets, weights, mask)
    return _score_with_targets(q_d, q_anchor, pab_m, t_m, w_m)


def score_ipl_active_centroid(kernel_data, ipl_data, phi_arr,
                              mode='spec_peaks_only', **kw):
    """Variant 4 (ORACLE): truth-containing active centroid only."""
    q_delta = kernel_data['q_delta']
    pab = kernel_data['pab_j2000_at_constraints']
    constraint_epochs = np.asarray(kernel_data['constraint_epochs'])
    seed = int(kernel_data['seed'])
    active_dirs = ipl_data[f"s{seed:03d}_active_centroid_dirs"]
    q_anchor = _build_q_anchor_from_kernel(kernel_data, phi_arr)

    targets = _targets_active_centroid(constraint_epochs, active_dirs)
    weights = _weights_uniform(len(constraint_epochs))

    mask = _select_epoch_mask(kernel_data, mode)
    q_d, pab_m, t_m, w_m = _apply_mask(q_delta, pab, targets, weights, mask)
    return _score_with_targets(q_d, q_anchor, pab_m, t_m, w_m)


def score_ipl_active_ring(kernel_data, ipl_data, phi_arr,
                          mode='spec_peaks_only', **kw):
    """Variant 6 (RING-ORACLE): cost = (PAB·c_active - cos(ang_dist[ep]))^2.

    Instead of penalising deviation from dot=1 (centroid exactly), penalise
    deviation from the expected dot product — PAB is expected to sit at
    angular distance ang_dist[ep] from the active centroid (on the IPL loop).
    Uses ang_dists as oracle loop-radius per epoch.
    """
    q_delta = kernel_data['q_delta']
    pab = kernel_data['pab_j2000_at_constraints']
    constraint_epochs = np.asarray(kernel_data['constraint_epochs'])
    seed = int(kernel_data['seed'])
    active_dirs = ipl_data[f"s{seed:03d}_active_centroid_dirs"]
    ang_dists = ipl_data[f"s{seed:03d}_ang_dists"]  # degrees
    q_anchor = _build_q_anchor_from_kernel(kernel_data, phi_arr)

    targets = _targets_active_centroid(constraint_epochs, active_dirs)
    weights = _weights_uniform(len(constraint_epochs))
    # Expected cos(angle) per epoch — used by ring cost
    expected_cos = np.cos(np.deg2rad(
        np.asarray([ang_dists[ep] for ep in constraint_epochs], dtype=np.float64)))

    mask = _select_epoch_mask(kernel_data, mode)
    q_d, pab_m, t_m, w_m = _apply_mask(q_delta, pab, targets, weights, mask)
    # Also mask expected_cos
    if mask is None:
        exp_cos_m = expected_cos
    else:
        exp_cos_m = expected_cos[mask]
    return _score_with_targets_ring(q_d, q_anchor, pab_m, t_m, w_m, exp_cos_m)


def score_ipl_centroid_weighted_extended(kernel_data, ipl_data, phi_arr,
                                         length_threshold='median',
                                         tightness_norm='inverse_length', **kw):
    """Variant 5: all tight-IPL epochs (length < median) with inverse-length weights.

    Epoch set = union(spec_peaks, tight_epochs) — kernel already built for this.
    """
    return score_ipl_centroid_weighted(
        kernel_data, ipl_data, phi_arr,
        mode='extended', tightness_norm=tightness_norm)


# Expose scorers by name for convenience in the driver
SCORERS = {
    'facet_normal':               score_facet_normal,
    'ipl_centroid_uniform':       score_ipl_centroid_uniform,
    'ipl_centroid_weighted':      score_ipl_centroid_weighted,
    'ipl_active_centroid':        score_ipl_active_centroid,
    'ipl_active_ring':            score_ipl_active_ring,
    'ipl_centroid_weighted_ext':  score_ipl_centroid_weighted_extended,
}
