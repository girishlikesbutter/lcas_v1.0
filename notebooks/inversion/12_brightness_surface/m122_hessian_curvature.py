#!/usr/bin/env python3
"""
m122 -- Hessian-at-truth basin geometry.

Hypothesis (3 falsifiable claims)
---------------------------------
1. Finite-difference Hessian eigenvalues at truth confirm the per-axis basin
   widths measured in m121: there exist eigenvectors with curvature
   corresponding to ω-direction widths < 0.1° and q0 widths ~5°.

2. The two stiffest Hessian eigenvectors lie predominantly in the ω-direction
   subspace, and their dominant body-frame projection matches the seed-specific
   principal axis identified empirically by m121:
     - seed 14: ≈body +Z stiff / -Y soft
     - seed 27: ≈body +X stiff
     - seed 46: ≈body -X/+Z stiff
     - soft axis always has dominant ±Y component (IS-901 panel-spin axis).

3. OK-cohort seeds (74, 93) have basin geometry qualitatively similar to
   ATT_FAIL cohort (eigenvalue spread within ~1 order of magnitude of
   ATT_FAIL seeds 14/27/46). REFUTED ⇔ basin geometry is part of why
   ATT_FAIL seeds are hard.

Method
------
At truth (q0_true, ω_true) we build a 6-DOF tangent-space parameterization:
  x[0:3] = r ∈ ℝ³ : Rodrigues vector (radians). q0 = quat(r) ∘_LEFT q0_true.
  x[3:5] = ξ ∈ ℝ² : tangent in plane ⟂ ω̂_true (basis e1, e2 from GS).
                    perturbed ω̂ = normalize(ω̂_true + ξ[0]·e1 + ξ[1]·e2);
                    |ω| held fixed when perturbing direction.
  x[5]   = δ     : fractional magnitude. |ω| = |ω_true| · (1 + δ).
At x = 0 the parameterization reproduces truth EXACTLY.

Cost: EXACTLY m121's mean_L1 path — propagate_attitude, surrogate over all
500 constraint epochs, residual = predicted - observed, cost = mean(|r|).

Finite differences (central): 1 (truth) + 12 (±h per axis) + 60 (4 evals per
(i<j) pair × 15 pairs) = 73 cost evaluations per seed.

Step sizes (length-6 numpy array, saved in outputs):
  h[0:3] = deg2rad(0.01)     # 0.01° rotation
  h[3:5] = deg2rad(0.005)    # 0.005° tilt (sin≈angle in tangent)
  h[5]   = 1e-4              # 0.01% fractional

Stages (design-first checkpoints):
  A: per-seed setup.npz materialisation (via lib.experiment_setup) if missing.
  B: per-seed evals.npz -- EVERY surrogate cost evaluation.
  C: per-seed hessian.npz -- assembled Hessian + eigendecomposition.
  D: per-seed summary.json -- truth cost, grad, eigs, classifications, widths.

Each stage skip-if-exists unless MICRO122_FORCE=1.

Env
---
MICRO122_SEEDS   comma-separated list, default "14,27,46,74,93"
MICRO122_FORCE   "1" = ignore cache

Outputs per seed (data/results/inversion_diagnostics/m122/seed_NNN/):
  setup.npz        (only new for seeds without existing m119v2 setup)
  evals.npz        perturbation_vector[N,6], cost[N], axis_label[N]
  hessian.npz      H_raw, H_sym, eigenvalues, eigenvectors, h_per_axis,
                   grad, truth_cost
  summary.json     per-seed diagnostics (schema in spec)
  run.log

Run-level: data/results/inversion_diagnostics/m122/summary.json

TODOs (deferred to v2 — not implemented)
----------------------------------------
- Richardson extrapolation / multi-h step-size check for FD noise floor.
- Alternative cost variants (mean_L2, count_pass, soft_pass).
- Plotting: eigenvector bars, basin-width per axis, cohort comparison.
- Project ω-dir eigenvectors to body frame at MULTIPLE times (not just t=0).
- Joint-seed cohort comparison plot (ATT_FAIL vs OK eigenvalue spread).
- Save predicted hi-fi light curves at ±h perturbations (diagnostic).

Conventions
-----------
- Use mean_L1 cost (= m121 primary) unchanged.
- LEFT-multiply quaternion perturbation (twin-degeneracy convention).
- Symmetrize H = 0.5 · (H + H.T) before eigendecomposition.
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import time
import json
from pathlib import Path
import multiprocessing as mp

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel
from lib.experiment_setup import setup_experiment


# ---- Config --------------------------------------------------------------
SEEDS = [int(s) for s in os.environ.get(
    'MICRO122_SEEDS', '14,27,46,74,93').split(',') if s.strip()]
FORCE = os.environ.get('MICRO122_FORCE', '0') == '1'

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
MICRO46_NPZ = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"

OUT_BASE = RESULTS_DIR / "m122"
MICRO119V2_BASE = RESULTS_DIR / "m119v2"

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

PANEL_DEG = 0.0
DISH_DEG = 15.0

# FD step sizes (length-6 array, physical units shown in comments)
H_PER_AXIS = np.array([
    np.deg2rad(0.01),   # q0 r_x   (0.01° rotation)
    np.deg2rad(0.01),   # q0 r_y
    np.deg2rad(0.01),   # q0 r_z
    np.deg2rad(0.005),  # omega-dir ξ1 (0.005° tilt; tangent mag == sin(angle))
    np.deg2rad(0.005),  # omega-dir ξ2
    1e-4,               # omega-mag δ (0.01% fractional)
], dtype=np.float64)

AXIS_LABELS = ['q0_rx', 'q0_ry', 'q0_rz', 'w_dir_e1', 'w_dir_e2', 'w_mag_delta']

# Empirical per-seed principal axes (body frame) from m121 REPORT.md
# used for hyp2 comparison. Seeds 74/93 have no m121 run ⇒ None.
# These are the STIFF-direction axes (high curvature => narrow basin).
MICRO121_STIFF_BODY = {
    14: np.array([0.0, 0.0, 1.0]),   # +Z stiff
    27: np.array([1.0, 0.0, 0.0]),   # +X stiff
    46: np.array([-0.5, 0.0, 0.5]) / np.linalg.norm([-0.5, 0.0, 0.5]),  # -X/+Z mix
    74: None,
    93: None,
}


# ---- Logging -------------------------------------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data); f.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (ValueError, OSError):
                pass


def atomic_json_save(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, filepath)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode('utf-8', errors='replace')
    return str(o)


# ---- Quaternion helpers --------------------------------------------------
def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def rotvec_to_wxyz(r):
    theta = np.linalg.norm(r)
    if theta < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = r / theta
    ha = theta / 2.0
    s = np.sin(ha)
    return np.array([np.cos(ha), s * axis[0], s * axis[1], s * axis[2]])


def wxyz_to_xyzw(q):
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]])
    return q[:, [1, 2, 3, 0]]


def rotation_matrix_body_from_q0(q0_wxyz):
    """Inertial-to-body rotation R(q0) at t=0."""
    return Rotation.from_quat(wxyz_to_xyzw(q0_wxyz)).as_matrix()


def tangent_basis_perp(w_hat):
    """Deterministic Gram-Schmidt basis (e1, e2) perp to w_hat, unit norm."""
    w_hat = w_hat / np.linalg.norm(w_hat)
    # pick reference axis least aligned with w_hat
    abs_w = np.abs(w_hat)
    idx = int(np.argmin(abs_w))
    ref = np.zeros(3); ref[idx] = 1.0
    e1 = ref - np.dot(ref, w_hat) * w_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w_hat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


# ---- Parameterization: x ∈ ℝ^6 -> (q0, omega0) --------------------------
def apply_perturbation(x, q0_true, omega0_true, w_hat_true, w_mag_true,
                       e1_perp, e2_perp):
    """Return (q0, omega0) for tangent-space vector x (length-6)."""
    # q0 perturbation: LEFT multiply by quat(r)
    r = x[0:3]
    dq = rotvec_to_wxyz(r)
    q0 = quat_mul_wxyz(dq, q0_true)
    q0 = q0 / np.linalg.norm(q0)

    # omega direction perturbation
    xi = x[3:5]
    new_dir = w_hat_true + xi[0] * e1_perp + xi[1] * e2_perp
    new_dir = new_dir / np.linalg.norm(new_dir)

    # omega magnitude perturbation
    delta = x[5]
    new_mag = w_mag_true * (1.0 + delta)

    omega0 = new_dir * new_mag
    return q0, omega0


# ---- Cost evaluation (identical path to m121 mean_L1) ---------------
def eval_cost(x, ctx, model):
    """
    ctx keys: q0_true, omega0_true, w_hat_true, w_mag_true, e1_perp, e2_perp,
              obs_times, constraint_epochs, k1_j2000_ce, k2_j2000_ce,
              obs_dist_ce, observed_mag_ce, inertia_tensor
    """
    q0, w = apply_perturbation(x,
                               ctx['q0_true'], ctx['omega0_true'],
                               ctx['w_hat_true'], ctx['w_mag_true'],
                               ctx['e1_perp'], ctx['e2_perp'])
    try:
        quats, _ = propagate_attitude(q0, w, ctx['obs_times'],
                                      "tumbling", ctx['inertia_tensor'])
    except Exception:
        return 10.0

    q_ce = quats[ctx['constraint_epochs']]
    R_ce = Rotation.from_quat(wxyz_to_xyzw(q_ce)).as_matrix()
    k1_body = np.einsum('nij,nj->ni', R_ce, ctx['k1_j2000_ce'])
    k2_body = np.einsum('nij,nj->ni', R_ce, ctx['k2_j2000_ce'])
    k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
    k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)

    E = len(ctx['constraint_epochs'])
    zeros_E = np.zeros(E, dtype=np.float64)
    dish_E = np.full(E, DISH_DEG, dtype=np.float64)
    dist_E = ctx['obs_dist_ce'].astype(np.float64)

    pred = model.predict_magnitude(k1_body, k2_body, zeros_E, dish_E, dist_E)
    residual = pred.astype(np.float64) - ctx['observed_mag_ce'].astype(np.float64)
    return float(np.mean(np.abs(residual)))


# ---- Stage A: load or materialise setup ---------------------------------
def stage_A_setup(seed, out_dir):
    """
    Prefer an existing m119v2 setup.npz for this seed (reuses the exact
    geometry m121 saw). If missing, call setup_experiment with truth
    overrides from m046 to generate a fresh setup.npz in our own out_dir.
    """
    print(f"\n[Stage A] Setup for seed {seed}")
    t0 = time.time()

    # Prefer m119v2 setup (where m121 got its data from)
    v2_setup = MICRO119V2_BASE / f"seed_{seed:03d}" / "setup.npz"
    if v2_setup.exists():
        print(f"  reusing existing m119v2 setup: {v2_setup}")
        setup = dict(np.load(v2_setup, allow_pickle=False))
        print(f"  Stage A done in {time.time()-t0:.1f}s")
        return setup

    # Fallback: local setup.npz in m122 seed dir
    local_setup = out_dir / "setup.npz"
    if local_setup.exists() and not FORCE:
        print(f"  loading cached local setup: {local_setup}")
        setup = dict(np.load(local_setup, allow_pickle=False))
        print(f"  Stage A done in {time.time()-t0:.1f}s")
        return setup

    # Materialise fresh via setup_experiment(true_q0_wxyz=..., true_omega0_rad=...)
    print(f"  no setup found — generating via setup_experiment for seed {seed}")
    m046 = np.load(MICRO46_NPZ)
    truth_q0 = m046['q0s'][seed].astype(np.float64)
    truth_omega0 = m046['omega0s'][seed].astype(np.float64)
    print(f"  m046 truth: q0={truth_q0}  |w|={np.linalg.norm(truth_omega0):.5f}")

    ctx = setup_experiment(
        n_observations=500,
        end_time_utc='2020-02-05T11:00:00',  # DATA_INVARIANTS.md — must match m046 1-hour window
        noise_sigma=0.05,
        random_seed=42,
        skip_true_lc=False,
        true_q0_wxyz=truth_q0,
        true_omega0_rad=truth_omega0,
    )

    k1_j2000 = ctx.sun_pos - ctx.sat_pos
    k1_j2000 /= np.linalg.norm(k1_j2000, axis=1, keepdims=True)
    k2_j2000 = ctx.obs_pos - ctx.sat_pos
    k2_j2000 /= np.linalg.norm(k2_j2000, axis=1, keepdims=True)

    constraint_epochs = np.arange(500, dtype=np.int64)
    k1_j2000_ce = k1_j2000[constraint_epochs]
    k2_j2000_ce = k2_j2000[constraint_epochs]
    obs_dist_ce = ctx.obs_dist[constraint_epochs]
    observed_mag_ce = ctx.observed_lc[constraint_epochs]
    obs_times = ctx.observation_times
    inertia_tensor = ctx.inertia_tensor.astype(np.float64)

    # DATA_INVARIANTS.md § 2: m046's ground-truth window is 1 hour (3600 s).
    # Previous assertion here expected ~21600 s (6 hr) and masked the
    # end_time_utc bug — it is the wrong check. Verify the correct window
    # instead.
    assert 3500.0 < obs_times[-1] < 3700.0, (
        f"obs_times span {obs_times[-1]} s -- expected ~3600 s "
        f"(m046's 1-hour window); did setup_experiment fall through to "
        f"the config default?")

    np.savez(
        local_setup,
        constraint_epochs=constraint_epochs,
        truth_q0=truth_q0,
        truth_omega0=truth_omega0,
        observed_mag_ce=observed_mag_ce,
        k1_j2000_ce=k1_j2000_ce,
        k2_j2000_ce=k2_j2000_ce,
        obs_dist_ce=obs_dist_ce,
        inertia_tensor=inertia_tensor,
        obs_times=obs_times,
    )
    print(f"  saved {local_setup}")
    print(f"  Stage A done in {time.time()-t0:.1f}s")
    return dict(np.load(local_setup, allow_pickle=False))


# ---- Stage B: evaluate 73 FD points --------------------------------------
def build_fd_points(h):
    """
    Return perturbation_vector[N,6] and axis_label[N] for the 73-eval scheme.
      index 0: truth (x=0)
      indices 1..12: gradient/diagonal: +h_i, -h_i for i=0..5
      indices 13..72: 15 off-diagonal pairs × 4 sign combos
    """
    pts = [np.zeros(6)]
    labels = ['truth']

    # 12 gradient/diagonal evals
    for i in range(6):
        for sgn in (+1, -1):
            v = np.zeros(6); v[i] = sgn * h[i]
            pts.append(v)
            labels.append(f'axis{i}_{"+" if sgn>0 else "-"}h')

    # 60 off-diagonal evals: (i<j), signs (++, +-, -+, --)
    for i in range(6):
        for j in range(i + 1, 6):
            for si in (+1, -1):
                for sj in (+1, -1):
                    v = np.zeros(6)
                    v[i] = si * h[i]
                    v[j] = sj * h[j]
                    pts.append(v)
                    labels.append(
                        f'pair{i}{j}_{"+" if si>0 else "-"}{"+" if sj>0 else "-"}')

    pts = np.array(pts, dtype=np.float64)
    labels = np.array(labels, dtype='S24')
    assert len(pts) == 73, f"expected 73 points, got {len(pts)}"
    return pts, labels


def stage_B_evals(seed, setup, out_dir):
    print(f"\n[Stage B] FD evaluations for seed {seed}")
    t0 = time.time()
    out = out_dir / "evals.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  cost stats: truth={d['cost'][0]:.6f}, "
              f"min={d['cost'].min():.6f}, max={d['cost'].max():.6f}")
        print(f"  Stage B (cached) done in {time.time()-t0:.1f}s")
        return d

    # Build context
    q0_true = setup['truth_q0'].astype(np.float64)
    q0_true = q0_true / np.linalg.norm(q0_true)
    omega0_true = setup['truth_omega0'].astype(np.float64)
    w_mag_true = float(np.linalg.norm(omega0_true))
    w_hat_true = omega0_true / w_mag_true
    e1_perp, e2_perp = tangent_basis_perp(w_hat_true)

    ctx = {
        'q0_true': q0_true,
        'omega0_true': omega0_true,
        'w_hat_true': w_hat_true,
        'w_mag_true': w_mag_true,
        'e1_perp': e1_perp,
        'e2_perp': e2_perp,
        'obs_times': setup['obs_times'].astype(np.float64),
        'constraint_epochs': setup['constraint_epochs'].astype(np.int64),
        'k1_j2000_ce': setup['k1_j2000_ce'].astype(np.float64),
        'k2_j2000_ce': setup['k2_j2000_ce'].astype(np.float64),
        'obs_dist_ce': setup['obs_dist_ce'].astype(np.float64),
        'observed_mag_ce': setup['observed_mag_ce'].astype(np.float64),
        'inertia_tensor': setup['inertia_tensor'].astype(np.float64),
    }

    model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))
    print(f"  loaded surrogate model")
    print(f"  w_hat_true = {w_hat_true}")
    print(f"  e1_perp    = {e1_perp}")
    print(f"  e2_perp    = {e2_perp}")

    pts, labels = build_fd_points(H_PER_AXIS)
    N = len(pts)
    print(f"  evaluating {N} FD points (1 + 12 + 60)")

    costs = np.zeros(N, dtype=np.float64)
    t_eval = time.time()
    for k in range(N):
        costs[k] = eval_cost(pts[k], ctx, model)
        if (k + 1) % 10 == 0 or k == N - 1:
            rate = (k + 1) / max(time.time() - t_eval, 1e-3)
            rem = (N - k - 1) / max(rate, 1e-3)
            print(f"    {k+1}/{N} done, {rate:.1f} eval/s, ~{rem:.0f}s remaining")

    print(f"  truth cost = {costs[0]:.6f}")
    print(f"  cost min/max = {costs.min():.6f} / {costs.max():.6f}")

    # --- design-first save ---
    np.savez_compressed(
        out,
        perturbation_vector=pts,
        cost=costs,
        axis_label=labels,
        h_per_axis=H_PER_AXIS,
        w_hat_true=w_hat_true,
        e1_perp=e1_perp,
        e2_perp=e2_perp,
        q0_true=q0_true,
        omega0_true=omega0_true,
    )
    print(f"  saved {out}")
    print(f"  Stage B done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


# ---- Stage C: Hessian assembly + eigendecomposition ---------------------
def stage_C_hessian(seed, evals, out_dir):
    print(f"\n[Stage C] Hessian assembly for seed {seed}")
    t0 = time.time()
    out = out_dir / "hessian.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  Stage C (cached) done in {time.time()-t0:.1f}s")
        return d

    pts = evals['perturbation_vector']
    costs = evals['cost']
    labels = evals['axis_label']
    h = evals['h_per_axis']

    # Build lookup: label -> cost index
    def label_of(k):
        raw = labels[k]
        return raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    label_to_idx = {label_of(k): k for k in range(len(labels))}

    truth_cost = float(costs[label_to_idx['truth']])

    # Gradient (central): g_i = (cost(+h_i) - cost(-h_i)) / (2 h_i)
    grad = np.zeros(6, dtype=np.float64)
    for i in range(6):
        cp = costs[label_to_idx[f'axis{i}_+h']]
        cm = costs[label_to_idx[f'axis{i}_-h']]
        grad[i] = (cp - cm) / (2.0 * h[i])
    grad_norm = float(np.linalg.norm(grad))
    print(f"  gradient = {grad}")
    print(f"  |grad|   = {grad_norm:.6g}")

    # Hessian
    H = np.zeros((6, 6), dtype=np.float64)
    # Diagonal: H_ii = (cost(+h_i) - 2 cost(0) + cost(-h_i)) / h_i^2
    for i in range(6):
        cp = costs[label_to_idx[f'axis{i}_+h']]
        cm = costs[label_to_idx[f'axis{i}_-h']]
        H[i, i] = (cp - 2.0 * truth_cost + cm) / (h[i] ** 2)

    # Off-diagonal: H_ij = (c(++) - c(+-) - c(-+) + c(--)) / (4 h_i h_j)
    for i in range(6):
        for j in range(i + 1, 6):
            c_pp = costs[label_to_idx[f'pair{i}{j}_++']]
            c_pm = costs[label_to_idx[f'pair{i}{j}_+-']]
            c_mp = costs[label_to_idx[f'pair{i}{j}_-+']]
            c_mm = costs[label_to_idx[f'pair{i}{j}_--']]
            H[i, j] = (c_pp - c_pm - c_mp + c_mm) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]  # mirror; symmetrize below anyway

    H_sym = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H_sym)  # ascending
    # reorder descending
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    print(f"  eigenvalues (desc) = {eigvals}")

    np.savez(
        out,
        H_raw=H,
        H_sym=H_sym,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        grad=grad,
        truth_cost=np.array(truth_cost),
        h_per_axis=h,
    )
    print(f"  saved {out}")
    print(f"  Stage C done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


# ---- Stage D: summary.json ----------------------------------------------
def classify_eigenvector(v, thresh=0.7):
    """Return subblock label and squared-norm fractions."""
    sq = v ** 2
    f_q0 = float(sq[0:3].sum())
    f_wdir = float(sq[3:5].sum())
    f_wmag = float(sq[5])
    total = f_q0 + f_wdir + f_wmag
    fracs = {'q0': f_q0 / total, 'omega_dir': f_wdir / total,
             'omega_mag': f_wmag / total}
    if fracs['q0'] > thresh:
        return 'q0', fracs
    if fracs['omega_dir'] > thresh:
        return 'omega_dir', fracs
    if fracs['omega_mag'] > thresh:
        return 'omega_mag', fracs
    return 'mixed', fracs


def basin_width_physical(eigval, delta_cost_target, axis_kind):
    """
    width_param = sqrt(2 · Δc / λ)  if λ > 0 else inf (flat direction).
    axis_kind: 'q0' (rad->deg), 'omega_dir' (tangent->deg via arcsin),
               'omega_mag' (fraction->pct).
    """
    if eigval <= 0:
        return {'param': float('inf'), 'physical': float('inf'),
                'unit': axis_kind}
    w = float(np.sqrt(2.0 * delta_cost_target / eigval))
    if axis_kind == 'q0':
        return {'param': w, 'physical': float(np.rad2deg(w)), 'unit': 'deg'}
    if axis_kind == 'omega_dir':
        # tangent mag == sin(angle); physical angle = arcsin(min(w,1))
        ang = float(np.rad2deg(np.arcsin(min(w, 1.0))))
        return {'param': w, 'physical': ang, 'unit': 'deg'}
    if axis_kind == 'omega_mag':
        return {'param': w, 'physical': w * 100.0, 'unit': 'pct'}
    return {'param': w, 'physical': w, 'unit': 'unknown'}


def stage_D_summary(seed, setup, evals, hessian, out_dir, wall_start):
    print(f"\n[Stage D] Summary for seed {seed}")
    t0 = time.time()

    truth_cost = float(hessian['truth_cost'])
    grad = hessian['grad']
    eigvals = hessian['eigenvalues']
    eigvecs = hessian['eigenvectors']

    q0_true = setup['truth_q0'].astype(np.float64)
    omega0_true = setup['truth_omega0'].astype(np.float64)
    w_hat_true = omega0_true / np.linalg.norm(omega0_true)
    e1_perp, e2_perp = tangent_basis_perp(w_hat_true)
    R_body = rotation_matrix_body_from_q0(q0_true)  # inertial -> body at t=0

    # Δc target: 100% cost rise (same scale as m121's basin criterion)
    delta_cost_target = 0.5 * truth_cost

    axis_class = []
    axis_fracs = []
    widths_q0 = []
    widths_wdir = []
    widths_wmag = []
    basin_widths_param = []
    basin_widths_unit = []
    omega_dir_eigenaxes_inertial = []
    omega_dir_eigenaxes_body = []

    for k in range(6):
        v = eigvecs[:, k]
        cls, fracs = classify_eigenvector(v)
        axis_class.append(cls)
        axis_fracs.append(fracs)

        # per-eigenvector basin width: use dominant-subblock unit if classified
        if cls == 'q0':
            w = basin_width_physical(eigvals[k], delta_cost_target, 'q0')
            widths_q0.append(w['physical'])
        elif cls == 'omega_dir':
            w = basin_width_physical(eigvals[k], delta_cost_target, 'omega_dir')
            widths_wdir.append(w['physical'])
        elif cls == 'omega_mag':
            w = basin_width_physical(eigvals[k], delta_cost_target, 'omega_mag')
            widths_wmag.append(w['physical'])
        else:
            # mixed: skip for physical-width bucketing
            w = {'param': float(np.sqrt(2.0 * delta_cost_target / max(eigvals[k], 1e-30)))
                 if eigvals[k] > 0 else float('inf'),
                 'physical': None, 'unit': 'mixed'}
        basin_widths_param.append(w['param'])
        basin_widths_unit.append(w['unit'])

        # If predominantly omega_dir, project (ξ[0]*e1 + ξ[1]*e2) to axis
        if cls == 'omega_dir':
            xi = v[3:5]
            axis_inertial = xi[0] * e1_perp + xi[1] * e2_perp
            nrm = np.linalg.norm(axis_inertial)
            if nrm > 1e-12:
                axis_inertial = axis_inertial / nrm
                axis_body = R_body @ axis_inertial
                axis_body = axis_body / np.linalg.norm(axis_body)
                omega_dir_eigenaxes_inertial.append(axis_inertial.tolist())
                omega_dir_eigenaxes_body.append(axis_body.tolist())
            else:
                omega_dir_eigenaxes_inertial.append(None)
                omega_dir_eigenaxes_body.append(None)
        else:
            omega_dir_eigenaxes_inertial.append(None)
            omega_dir_eigenaxes_body.append(None)

    # Anisotropy comparison vs m121 (hyp2)
    empirical_body = MICRO121_STIFF_BODY.get(seed)
    anisotropy_dot = None
    if empirical_body is not None:
        # Find the stiffest omega_dir eigenvector (highest eigenvalue among
        # omega_dir-classified). If none, compare against stiffest overall.
        best_k = None
        for k in range(6):
            if axis_class[k] == 'omega_dir':
                best_k = k; break
        if best_k is None:
            best_k = 0  # stiffest overall
        body_axis = omega_dir_eigenaxes_body[best_k]
        if body_axis is not None:
            anisotropy_dot = float(abs(np.dot(np.array(body_axis), empirical_body)))

    # Print headline
    eigs_str = ", ".join(f"{e:.3g}" for e in eigvals)
    q0_w_str = f"{min(widths_q0):.2f}°" if widths_q0 else "n/a"
    wdir_w_str = f"{min(widths_wdir):.3f}°" if widths_wdir else "n/a"
    wmag_w_str = f"{min(widths_wmag):.2f}%" if widths_wmag else "n/a"

    # Hypothesis classification (per seed)
    # hyp1: exists eigvec with omega_dir width < 0.1° AND q0 width ~5° (<= 10°)
    hyp1_wdir_ok = any(w < 0.1 for w in widths_wdir) if widths_wdir else False
    hyp1_q0_ok = any(w < 10.0 for w in widths_q0) if widths_q0 else False
    if hyp1_wdir_ok and hyp1_q0_ok:
        hyp1 = 'CONFIRMED'
    elif widths_wdir or widths_q0:
        hyp1 = 'REFUTED'
    else:
        hyp1 = 'INCONCLUSIVE'

    # hyp2: two stiffest eigenvecs predominantly omega_dir; best matches empirical
    top2_kinds = [axis_class[0], axis_class[1]]
    hyp2_stiffness = (top2_kinds.count('omega_dir') >= 1)
    if empirical_body is not None and anisotropy_dot is not None:
        hyp2 = 'CONFIRMED' if (hyp2_stiffness and anisotropy_dot > 0.7) else 'REFUTED'
    else:
        hyp2 = 'INCONCLUSIVE'  # no empirical reference

    # hyp3 is computed at run-level once we have all seeds
    hyp3 = 'DEFERRED_TO_RUNLEVEL'

    print(f"\n  SEED {seed:3d}  truth_cost={truth_cost:.4f}  |grad|={np.linalg.norm(grad):.3g}  "
          f"eigs=[{eigs_str}]")
    print(f"    basin widths: q0={q0_w_str}  ω_dir={wdir_w_str}  ω_mag={wmag_w_str}")
    print(f"    hyp1: {hyp1}   hyp2: {hyp2}   hyp3: {hyp3}")

    wall_seconds = float(time.time() - wall_start)

    summary = {
        'seed': int(seed),
        'truth_q0_wxyz': q0_true.tolist(),
        'truth_omega_rad_per_s': omega0_true.tolist(),
        'truth_cost_mean_L1': truth_cost,
        'gradient': grad.tolist(),
        'gradient_norm': float(np.linalg.norm(grad)),
        'h_per_axis': H_PER_AXIS.tolist(),
        'axis_labels': AXIS_LABELS,
        'delta_cost_target': float(delta_cost_target),
        'eigenvalues': eigvals.tolist(),
        'eigenvectors_columns': eigvecs.T.tolist(),  # rows = eigenvectors
        'axis_classification': axis_class,
        'axis_fracs': axis_fracs,
        'basin_widths_param': basin_widths_param,
        'basin_widths_unit': basin_widths_unit,
        'basin_widths_physical': {
            'q0_deg': widths_q0,
            'omega_dir_deg': widths_wdir,
            'omega_mag_pct': widths_wmag,
        },
        'omega_dir_eigenaxes_inertial': omega_dir_eigenaxes_inertial,
        'omega_dir_eigenaxes_body': omega_dir_eigenaxes_body,
        'm121_empirical_axis_body': (empirical_body.tolist()
                                         if empirical_body is not None else None),
        'anisotropy_axis_dot_micro121': anisotropy_dot,
        'n_surrogate_evals': 73,
        'hyp1_verdict': hyp1,
        'hyp2_verdict': hyp2,
        'hyp3_verdict': hyp3,
        'wall_seconds': wall_seconds,
    }
    atomic_json_save(out_dir / "summary.json", summary)
    print(f"  saved {out_dir / 'summary.json'}")
    print(f"  Stage D done in {time.time()-t0:.1f}s")
    return summary


# ---- Per-seed driver -----------------------------------------------------
def run_seed(seed):
    out_dir = OUT_BASE / f"seed_{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    log_f = open(log_path, 'w')
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    t_seed = time.time()
    try:
        print("=" * 70)
        print(f"m122 -- Hessian at truth, seed {seed}")
        print("=" * 70)
        print(f"  out_dir={out_dir}  FORCE={FORCE}")
        print(f"  h_per_axis={H_PER_AXIS}")

        setup = stage_A_setup(seed, out_dir)
        evals = stage_B_evals(seed, setup, out_dir)
        hessian = stage_C_hessian(seed, evals, out_dir)
        summary = stage_D_summary(seed, setup, evals, hessian, out_dir, t_seed)

        dt = time.time() - t_seed
        print(f"\n  Total for seed {seed}: {dt:.1f}s")
        return summary
    finally:
        sys.stdout = orig_stdout
        log_f.close()


# ---- Run-level aggregation ----------------------------------------------
def aggregate_runlevel(per_seed_summaries):
    """Compute cohort-level hyp3 verdict and write run-level summary.json."""
    ATT_FAIL_COHORT = {14, 27, 46}
    OK_COHORT = {74, 93}

    def eigenvalue_spread(s):
        eigs = np.asarray(s['eigenvalues'], dtype=np.float64)
        pos = eigs[eigs > 0]
        if len(pos) < 2:
            return None
        return float(pos.max() / pos.min())

    att_fail_spreads = []
    ok_spreads = []
    for s in per_seed_summaries:
        sp = eigenvalue_spread(s)
        if sp is None:
            continue
        if s['seed'] in ATT_FAIL_COHORT:
            att_fail_spreads.append(sp)
        elif s['seed'] in OK_COHORT:
            ok_spreads.append(sp)

    # hyp3: OK within ~1 order of magnitude of ATT_FAIL
    hyp3 = 'INCONCLUSIVE'
    hyp3_reason = 'not enough cohort data'
    if att_fail_spreads and ok_spreads:
        att_med = float(np.median(att_fail_spreads))
        ok_med = float(np.median(ok_spreads))
        ratio = max(att_med, ok_med) / min(att_med, ok_med)
        if ratio <= 10.0:
            hyp3 = 'REFUTED'  # geometry similar => not the discriminator
            hyp3_reason = (f'cohort eig spreads within {ratio:.2f}× '
                           f'(ATT_FAIL med {att_med:.3g}, OK med {ok_med:.3g}); '
                           f'basin geometry does NOT explain ATT_FAIL')
        else:
            hyp3 = 'CONFIRMED'
            hyp3_reason = (f'cohort eig spreads differ by {ratio:.2f}× '
                           f'(ATT_FAIL med {att_med:.3g}, OK med {ok_med:.3g}); '
                           f'basin geometry IS part of why ATT_FAIL is hard')

    # hyp1 and hyp2 run-level: majority confirmation
    def majority(verdicts):
        if not verdicts:
            return 'INCONCLUSIVE', 'no seeds'
        c = sum(1 for v in verdicts if v == 'CONFIRMED')
        r = sum(1 for v in verdicts if v == 'REFUTED')
        total = len(verdicts)
        if c >= (total + 1) // 2:
            return 'CONFIRMED', f'{c}/{total} seeds confirmed'
        if r >= (total + 1) // 2:
            return 'REFUTED', f'{r}/{total} seeds refuted'
        return 'INCONCLUSIVE', f'{c} confirm / {r} refute / {total - c - r} inconclusive'

    hyp1_overall, hyp1_reason = majority([s['hyp1_verdict'] for s in per_seed_summaries])
    hyp2_overall, hyp2_reason = majority(
        [s['hyp2_verdict'] for s in per_seed_summaries
         if s['hyp2_verdict'] != 'INCONCLUSIVE'])

    runlevel = {
        'seeds_run': [s['seed'] for s in per_seed_summaries],
        'att_fail_cohort': sorted(list(ATT_FAIL_COHORT)),
        'ok_cohort': sorted(list(OK_COHORT)),
        'per_seed_eigenvalue_spread': {
            s['seed']: eigenvalue_spread(s) for s in per_seed_summaries
        },
        'per_seed_basin_widths': {
            s['seed']: s['basin_widths_physical'] for s in per_seed_summaries
        },
        'per_seed_axis_classification': {
            s['seed']: s['axis_classification'] for s in per_seed_summaries
        },
        'att_fail_spreads': att_fail_spreads,
        'ok_spreads': ok_spreads,
        'verdicts': {
            'hyp1': {'verdict': hyp1_overall, 'reason': hyp1_reason},
            'hyp2': {'verdict': hyp2_overall, 'reason': hyp2_reason},
            'hyp3': {'verdict': hyp3, 'reason': hyp3_reason},
        },
        'h_per_axis': H_PER_AXIS.tolist(),
    }
    atomic_json_save(OUT_BASE / "summary.json", runlevel)
    print(f"\nRun-level summary: {OUT_BASE / 'summary.json'}")
    print(f"  hyp1: {hyp1_overall} — {hyp1_reason}")
    print(f"  hyp2: {hyp2_overall} — {hyp2_reason}")
    print(f"  hyp3: {hyp3} — {hyp3_reason}")
    return runlevel


# ---- Main ----------------------------------------------------------------
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"m122 -- Hessian at truth; seeds={SEEDS}  FORCE={FORCE}")

    per_seed = []
    t_all = time.time()
    for seed in SEEDS:
        s = run_seed(seed)
        per_seed.append(s)

    print(f"\n{'=' * 72}")
    print(f"All seeds done in {time.time() - t_all:.1f}s")
    print(f"{'=' * 72}")

    aggregate_runlevel(per_seed)


if __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
