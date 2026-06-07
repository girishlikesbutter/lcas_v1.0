"""s100 — 5-step coverage-fix prototype (user's plan), tested on seed 119.

The s099 pipeline FAILED on 119 (top hi-fi rho 33, Band D) for two distinct
UPSTREAM reasons, neither of which densify/bracket/speed can touch:
  (1) cloud STARVATION: the sharpest anchor B admitted only |B|=20 pool members
      at 30k -> nearest-truth 37 deg -> no truth-near candidate to cross;
  (2) winding ALIASING (s095 near%=0): even a truth-near pair connects to the
      WRONG omega winding under a single finite-diff-init shoot.

This script implements the user's redesign, which attacks (1) by COVERAGE
(keep sharp anchors for tight constraint, but DENSE-FILL so truth's region is
sampled, then thin to 2 deg so the cross stays bounded):

  STEP 1  blind all-epoch sharpness scan      (removes the T_pol leak)
  STEP 2  pick anchors A/B/C, BLIND spacing    (LC-fraction epoch offsets)
  STEP 3  adaptive DENSE sampling at A,B        (millions, full SO(3), isophote)
  STEP 4  decimate each cloud to 2-deg reps     (one rep / 2-deg cell)
  STEP 5  cross the reps -> shoot -> filters -> coarse RMSE -> full-500 rank

TWO-PART MEASUREMENT (the science, baked in):
  (a) does a <=2 deg truth-near PAIR now enter the cross?   [I expect YES]
  (b) is the omega that the shoot returns truth-near?        [I expect NOT
      reliably under single-shoot; the multi-start probe tests whether the
      s088/s089 companion fix is necessary-and-sufficient.]

Blindness: truth is used ONLY for (i) the infra LC floor check, (ii) LABELS
(nt distances, dir_err) — never to steer candidate selection. The |w| window
is the s019 LS bracket (blind). NO T_pol, NO truth-centered |w| band.

Pool(24), BLAS pinned, fork CoW. v2 surrogate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import importlib
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.spatial.transform import Rotation

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import (sample_so3_pool, compute_j2000_units, project_directions,
                              survive_at_epoch, nearest_in_pool_to_truth)
from lib.shoot import (m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg,
                       geodesic_angle)
from lib.jacobi_propagator import propagate_jacobi_path2

s019 = importlib.import_module("s019_ls_bracket_omega_mag")   # blind |w| bracket

SEED = int(os.environ.get("S100_SEED", 119))
OUT = SURVEY / "results" / "s100" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- knobs ----
SCAN_POOL_N = int(os.environ.get("S100_SCAN_POOL", 30_000))   # step-1 scan pool
RNG_SEED = 42
TOL_MAG = 0.10
SP_DEG, AD_DEG = 0.0, 15.0
INERTIA = m048_inertia()
CONNECT_TOL_DEG = 1e-3

# blind anchor spacing (epoch offsets == LC-fraction; NO truth, NO T_pol)
EDGE = 5
A_HI_FRAC = 0.45                 # A chosen in [EDGE, A_HI_FRAC*N]
GAP_MIN = int(os.environ.get("S100_GAP_MIN", 20))   # ~144 s on 119
GAP_MAX = int(os.environ.get("S100_GAP_MAX", 80))   # ~577 s on 119

# step-3 dense fill (adaptive: sample until >=DENSE_TARGET survivors or budget hit)
DENSE_POOL = int(os.environ.get("S100_DENSE_POOL", 1_000_000))   # 1M clouds (user standing pref; 6M+ wasteful)
DENSE_TARGET = int(os.environ.get("S100_DENSE_TARGET", 0))        # 0 -> NO adaptive growth (flat 1M)
DENSE_MAX_POOL = int(os.environ.get("S100_DENSE_MAX", 1_000_000))
N_WORK = 24

# step-4 decimation
DECIM_DEG = float(os.environ.get("S100_DECIM_DEG", 2.0))
MAX_REPS = int(os.environ.get("S100_MAX_REPS", 2000))          # cap reps/anchor -> bound cross
REP_TARGET = int(os.environ.get("S100_REP_TARGET", 2000))     # anchor-cap mode: coarsen decim until cells<=this

# step-5 scoring
COARSE_K = int(os.environ.get("S100_COARSE_K", 50))
FULL_TOPN = int(os.environ.get("S100_FULL_TOPN", 4000))
PAIR_BUDGET = int(os.environ.get("S100_PAIR_BUDGET", 50_000_000))  # s111 fix #4: raised from 6M so fixed-2deg cross does NOT random-cull (cull can drop the truth cell, blind). kill-at-2x backstops a runaway cross.

# coarse-then-fine cross (contract_slow-tumbler-generality v1): anchor-cap path only.
# PASS 1 localizes connectable (A,B) regions on COARSE_DEG cells w/ relaxed C-tol;
# PASS 2 re-decimates dense members in surviving coarse cells at DECIM_DEG (tight tol).
COARSE_DEG = float(os.environ.get("S100_COARSE_DEG", 6.0))
COARSE_TOL_MAG = float(os.environ.get("S100_COARSE_TOL_MAG", 0.30))

# multi-start probe
MS_NMAG = 10
MS_NDIR = 16

# ---- worker globals (fork CoW) ----
_SURR = None
_TOL_MAG_RUN = TOL_MAG          # C-brightness tol used by the active cross pass (coarse relaxes it)
_RC = _SUN = _OBS = _OD = _MAG = None
_SUN_A = _OBS_A = _OD_A = _MAG_A = None
_SUN_B = _OBS_B = _OD_B = _MAG_B = None
_TIMES0 = _EP_A = _DT_AB = _DT_AC = None
_W_LO = _W_HI = None
_QB = _CB = None
_SUNC = _OBSC = _ODC = _MAGC = None
_TF = _TB = _NB = _DROPF = None
_SUNc = _OBSc = _ODc = _MAGc = None


def _winit():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


# ======================================================================
# STEP 1 — sharpness of one epoch (survivor count over the scan pool)
# ======================================================================
def _sharp_count(ep):
    k1, k2 = project_directions(_RC, _SUN[ep], _OBS[ep])
    _, keep = survive_at_epoch(_SURR, k1, k2, float(_OD[ep]), SP_DEG, AD_DEG,
                               float(_MAG[ep]), TOL_MAG)
    return ep, int(keep.sum())


# ======================================================================
# STEP 3 — dense isophote sampling at A and B (one Haar batch per worker)
# ======================================================================
def _dense_worker(args):
    wseed, n = args
    rng = np.random.default_rng(wseed)
    R = Rotation.random(int(n), random_state=rng)
    Rm = R.as_matrix()
    out = {}
    for tag, sun, obs, od, mg in (("A", _SUN_A, _OBS_A, _OD_A, _MAG_A),
                                  ("B", _SUN_B, _OBS_B, _OD_B, _MAG_B)):
        k1 = np.einsum("nij,j->ni", Rm, sun)
        k2 = np.einsum("nij,j->ni", Rm, obs)
        pred = _SURR.predict_magnitude(k1, k2, SP_DEG, AD_DEG, np.full(int(n), od))
        keep = np.abs(pred - mg) < TOL_MAG
        out[tag] = R[keep].as_quat()[:, [3, 0, 1, 2]].astype(np.float64)  # wxyz
    return out


def dense_fill(ctx, total_pool):
    """Adaptive: sample `total_pool` Haar rotations split over workers, isophote-
    filter at A and B. Returns (clouds_A, clouds_B) as (M,4) wxyz arrays."""
    per = int(np.ceil(total_pool / N_WORK))
    args = [(7000 + i, per) for i in range(N_WORK)]
    qa, qb = [], []
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for r in p.imap_unordered(_dense_worker, args):
            if len(r["A"]):
                qa.append(r["A"])
            if len(r["B"]):
                qb.append(r["B"])
    QA = np.vstack(qa) if qa else np.empty((0, 4))
    QB = np.vstack(qb) if qb else np.empty((0, 4))
    return QA, QB


# ======================================================================
# STEP 4 — decimate a cloud to ~one rep per DECIM_DEG geodesic cell
# ======================================================================
def decimate_2deg(q_wxyz, deg=DECIM_DEG, max_reps=MAX_REPS, rng=None):
    """Grid-bucket in rotvec space; keep the member nearest each cell center.
    Double-cover canonicalised (w>=0). rotvec~geodesic for small cells, so a
    `deg`-sized cell keeps reps >= ~deg apart (approx; exact only for small angles)."""
    if len(q_wxyz) == 0:
        return q_wxyz, 0
    q = np.array(q_wxyz, float)
    q[q[:, 0] < 0] *= -1.0                                    # canonical hemisphere
    rv = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec()   # |rv|<=pi
    cell = np.radians(deg)
    keys = np.floor(rv / cell).astype(np.int64)
    centers = (keys + 0.5) * cell
    d2 = np.sum((rv - centers) ** 2, axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.lexsort((d2, inv))                             # by cell, then dist asc
    inv_s = inv[order]
    first = np.ones(len(order), bool)
    first[1:] = inv_s[1:] != inv_s[:-1]
    sel = order[first]
    n_cells = len(sel)
    if max_reps and len(sel) > max_reps:
        rng = rng or np.random.default_rng(0)
        sel = np.sort(rng.choice(sel, max_reps, replace=False))
    return q[sel], n_cells


def decimate_adaptive(q_wxyz, target, start_deg=DECIM_DEG, max_deg=15.0):
    """Coarsen the cell size until n_cells <= target, keeping ALL cells (one rep
    each, nearest-to-center). Deterministic: no random cull, so the truth-near
    cell is never dropped. Blind — we can't keep truth preferentially (we don't
    know which cell it is), so we keep every cell and just bound their count."""
    deg = start_deg
    rep, ncells = decimate_2deg(q_wxyz, deg=deg, max_reps=0)
    while ncells > target and deg < max_deg:
        deg *= 1.3
        rep, ncells = decimate_2deg(q_wxyz, deg=deg, max_reps=0)
    return rep, ncells, deg


# ======================================================================
# STEP 5 — cross / shoot / filters / coarse + full RMSE  (adapted from s099)
# ======================================================================
def _set_coarse_globals(times0, ep_a, sun_unit, obs_unit, obs_dist, mag):
    global _TF, _TB, _NB, _DROPF, _SUNc, _OBSc, _ODc, _MAGc
    cidx = np.unique(np.linspace(0, len(times0) - 1, COARSE_K, dtype=int))
    rel = times0[cidx] - times0[ep_a]
    relf = rel[rel >= 0]
    relb = rel[rel < 0]
    if relf.size and relf[0] == 0.0:
        _TF, _DROPF = relf, 0
    else:
        _TF, _DROPF = np.concatenate([[0.0], relf]), 1
    _TB = np.concatenate([[0.0], relb[::-1]]) if relb.size else np.array([])
    _NB = relb.size
    _SUNc, _OBSc = sun_unit[cidx], obs_unit[cidx]
    _ODc, _MAGc = obs_dist[cidx], mag[cidx]


def _coarse_rmse(q_a, w):
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, _TF)
    qf = qf[_DROPF:]
    if _NB:
        qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, _TB)
        quats = np.vstack([qb[1:][::-1], qf])
    else:
        quats = qf
    if not np.all(np.isfinite(quats)):
        return np.inf
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUNc),
                                   np.einsum("nij,nj->ni", R, _OBSc), SP_DEG, AD_DEG, _ODc)
    return float(np.sqrt(np.mean((pred - _MAGc) ** 2)))


def _propagate_full(q_a, w):
    tf = _TIMES0[_EP_A:] - _TIMES0[_EP_A]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EP_A == 0:
        return qf
    tb = (_TIMES0[:_EP_A + 1] - _TIMES0[_EP_A])[::-1]
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])


def _full_rmse(args):
    q_a, w = np.asarray(args[0]), np.asarray(args[1])
    try:
        with np.errstate(all="ignore"):
            quats = _propagate_full(q_a, w)
            if not np.all(np.isfinite(quats)):
                return np.inf
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                           np.einsum("nij,nj->ni", R, _OBS), SP_DEG, AD_DEG, _OD)
            return float(np.sqrt(np.mean((pred - _MAG) ** 2)))
    except (ValueError, FloatingPointError):
        return np.inf


def _filters_and_coarse(q_a, q_b):
    with np.errstate(all="ignore"):
        w_fd = finite_diff_omega(q_a, q_b, _DT_AB)
        if not np.all(np.isfinite(w_fd)) or np.linalg.norm(w_fd) < 1e-9:
            return None
        s = shoot(q_a, q_b, _DT_AB, INERTIA, w_fd)
        if s["geo_err_deg"] >= CONNECT_TOL_DEG:
            return None
        w = s["omega"]; wmag = float(np.linalg.norm(w))
        if not (_W_LO <= wmag <= _W_HI):
            return None
        qch, _ = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, _DT_AC]))
        qc = qch[-1]
        if not np.all(np.isfinite(qc)):
            return None
        Rc = Rotation.from_quat(qc[[1, 2, 3, 0]]).as_matrix()
        predc = float(_SURR.predict_magnitude((Rc @ _SUNC)[None, :], (Rc @ _OBSC)[None, :],
                                              SP_DEG, AD_DEG, np.array([_ODC]))[0])
        if abs(predc - _MAGC) >= _TOL_MAG_RUN:
            return None
        return w, _coarse_rmse(q_a, w)


def _cross_one_qa(args):
    a_idx, q_a = args
    out = []
    for b_idx in range(_QB.shape[0]):
        try:
            r = _filters_and_coarse(q_a, _QB[b_idx])
        except (ValueError, FloatingPointError):
            continue
        if r is None:
            continue
        w, crmse = r
        out.append((crmse, w.tolist(), q_a.tolist(), _QB[b_idx].tolist(), int(a_idx), int(b_idx)))
    return out


def cell_codes(q_wxyz, deg):
    """Integer cell code per quaternion at `deg` geodesic cells (canonical hemisphere).
    Matches decimate_2deg's bucketing so reps map back to their dense members' cells."""
    q = np.atleast_2d(np.array(q_wxyz, float)).copy()
    q[q[:, 0] < 0] *= -1.0
    rv = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec()
    k = np.floor(rv / np.radians(deg)).astype(np.int64) + 1024   # |rv|<=pi -> |k| small; +1024 -> >=0
    return (k[:, 0] << 22) + (k[:, 1] << 11) + k[:, 2]


def run_cross(ctx, rA, rB, tol_mag=TOL_MAG):
    """Cross every A-rep against every B-rep (finite-diff -> shoot -> connect/bracket/C-tol),
    returning C-pass survivor rows. `tol_mag` sets the C-brightness tol for this pass
    (coarse pass relaxes it). No PAIR_BUDGET cull — callers bound the product structurally."""
    global _QB, _CB, _TOL_MAG_RUN
    _TOL_MAG_RUN = tol_mag
    _QB, _CB = rB, np.arange(len(rB))
    work = [(int(i), rA[i]) for i in range(len(rA))]
    rows = []
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_cross_one_qa, work, chunksize=2):
            rows.extend(res)
    return rows


# ======================================================================
# MEASUREMENT (b) — single vs bounded multi-start shoot on the truth-near pair
# ======================================================================
def multistart_shoot(q_a, q_b, dt, w_lo, w_hi, w_true, n_mag=MS_NMAG, n_dir=MS_NDIR):
    """Bounded multi-start under the blind |w| bracket. Returns list of distinct
    connecting roots and the best (min) dir-err vs truth among them."""
    w_fd = finite_diff_omega(q_a, q_b, dt)
    fd_dir = w_fd / (np.linalg.norm(w_fd) + 1e-30)
    rng = np.random.default_rng(123)
    dirs = [fd_dir]
    rd = rng.normal(size=(n_dir, 3)); rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    dirs.extend(list(rd))
    mags = np.geomspace(w_lo, w_hi, n_mag)
    roots = []
    for d in dirs:
        for m in mags:
            try:
                s = shoot(q_a, q_b, dt, INERTIA, d * m)
            except (ValueError, FloatingPointError):
                continue
            if s["geo_err_deg"] >= CONNECT_TOL_DEG:
                continue
            w = s["omega"]; wm = float(np.linalg.norm(w))
            if not (w_lo <= wm <= w_hi):
                continue
            roots.append(w)
    # dedup
    uniq = []
    for w in roots:
        if not any(np.linalg.norm(w - u) < 1e-4 for u in uniq):
            uniq.append(w)
    if not uniq:
        return [], np.nan
    derrs = [omega_dir_err_deg(w, w_true) for w in uniq]
    return uniq, float(np.nanmin(derrs))


def main():
    t0 = time.time()
    ctx = get_context("fork")
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)   # LABELS + infra only
    mag = d["mag_hifi"]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    obs_dist = d["obs_dist"]
    N = len(times0)
    model = get_model()
    pool = sample_so3_pool(SCAN_POOL_N, RNG_SEED)
    qP = pool["q_pool_wxyz"]

    print(f"===== s100 5-STEP PROTO | seed {SEED} | N={N} | scan_pool {SCAN_POOL_N} =====", flush=True)

    # blind |w| bracket (s019, observed-LC only)
    br = s019.run_seed(SEED)
    w_lo, w_hi = float(br["bracket_lo"]), float(br["bracket_hi"])
    # physical clamp (feedback_omega_prior_physical_bracket): intersect blind
    # bracket with [0.1,1.5] deg/s. env-gated so the 119 reference run is untouched.
    if os.environ.get("S100_CLAMP", "0") == "1":
        phys_lo, phys_hi = np.radians(0.05), np.radians(1.6)   # s109-validated envelope
        w_lo, w_hi = max(w_lo, phys_lo), min(w_hi, phys_hi)
        print(f"[clamp] physical [0.05,1.6] deg/s applied", flush=True)
    wmag_true = float(np.linalg.norm(w0))
    print(f"[blind |w|] [{w_lo:.6f},{w_hi:.6f}] rad/s "
          f"([{np.degrees(w_lo):.4f},{np.degrees(w_hi):.4f}] deg/s, factor {w_hi/w_lo:.1f}x) | "
          f"truth |w|={wmag_true:.6f} ({np.degrees(wmag_true):.4f} deg/s) "
          f"in-bracket={w_lo<=wmag_true<=w_hi}", flush=True)

    # ---------- STEP 1: blind scan of ALL epochs ----------
    global _RC, _SUN, _OBS, _OD, _MAG
    _RC, _SUN, _OBS, _OD, _MAG = pool["R_cache"], sun_unit, obs_unit, obs_dist, mag
    finite = np.where(np.isfinite(mag))[0]
    cand = [int(e) for e in finite if EDGE <= e <= N - 1 - EDGE]
    ts = time.time()
    counts = {}
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for ep, c in p.imap_unordered(_sharp_count, cand, chunksize=4):
            counts[ep] = c
    print(f"[1] scanned {len(cand)} epochs in {time.time()-ts:.0f}s "
          f"(survivors min {min(counts.values())} / med {int(np.median(list(counts.values())))} "
          f"/ max {max(counts.values())})", flush=True)

    # ---------- STEP 2: pick anchors A/B/C, blind spacing ----------
    # The three GLOBALLY-sharpest epochs (smallest isophote = tightest constraint),
    # greedily spaced >= GAP_MIN epochs apart. Sharp anchors keep the rep count
    # (hence the cross) small AND dense-fill covers truth on the small patch.
    # Blind: index spacing only, no truth, no T_pol.
    sorted_eps = sorted((e for e in counts if counts[e] > 0), key=lambda e: counts[e])
    if os.environ.get("S100_ANCHOR_CAP", "0") == "1":
        # A->B finite-diff aliases past half a turn. Bound dt_ab so NO |w| in the
        # bracket can rotate A->B more than pi: dt_ab <= pi / w_hi. A=sharpest;
        # B=sharpest within [GAP_MIN, dt_cap] of A; C=sharpest elsewhere. Blind.
        dt_cap = float(np.pi / w_hi)
        ep_a = sorted_eps[0]
        # B = sharpest epoch with dt_ab in [gmin_s, dt_cap]. Prefer a wide gap for
        # cross discrimination; relax the lower bound if the (wide-bracket -> tight
        # dt_cap) window is otherwise empty. Time-based so it's cadence-robust.
        b_cands = []
        for gmin_s in (200.0, 120.0, 60.0, 30.0):
            b_cands = [e for e in sorted_eps if abs(e - ep_a) >= 5
                       and gmin_s <= abs(times0[e] - times0[ep_a]) <= dt_cap]
            if b_cands:
                break
        if not b_cands:
            print(f"anchor cap: no B in [30s,{dt_cap:.0f}s] of A=ep{ep_a} — abort", flush=True); return
        ep_b = b_cands[0]
        c_cands = [e for e in sorted_eps if abs(e - ep_a) >= GAP_MIN and abs(e - ep_b) >= GAP_MIN]
        if not c_cands:
            print("anchor cap: no C — abort", flush=True); return
        ep_c = c_cands[0]
        ep_a, ep_b = sorted([ep_a, ep_b])               # A before B in time (fwd shoot)
        print(f"[2] anchor-cap ON: dt_cap={dt_cap:.0f}s (pi/w_hi); "
              f"max windings @w_hi={w_hi*dt_cap/(2*np.pi):.2f}", flush=True)
    else:
        sel = []
        for e in sorted_eps:
            if all(abs(e - s) >= GAP_MIN for s in sel):
                sel.append(e)
            if len(sel) == 3:
                break
        if len(sel) < 3:
            print("anchor selection failed — abort", flush=True); return
        ep_a, ep_b, ep_c = sorted(sel)
    dt_ab = float(times0[ep_b] - times0[ep_a]); dt_ac = float(times0[ep_c] - times0[ep_a])
    print(f"[2] anchors A=ep{ep_a}(|surv|={counts[ep_a]}) B=ep{ep_b}({counts[ep_b]}) "
          f"C=ep{ep_c}({counts[ep_c]}) | dt_ab={dt_ab:.0f}s dt_ac={dt_ac:.0f}s | "
          f"|w|*dt_ab={wmag_true*dt_ab:.2f} rad", flush=True)

    # set globals for dense-fill + cross
    global _SUN_A, _OBS_A, _OD_A, _MAG_A, _SUN_B, _OBS_B, _OD_B, _MAG_B
    global _TIMES0, _EP_A, _DT_AB, _DT_AC, _W_LO, _W_HI, _SUNC, _OBSC, _ODC, _MAGC
    _SUN_A, _OBS_A, _OD_A, _MAG_A = sun_unit[ep_a], obs_unit[ep_a], float(obs_dist[ep_a]), float(mag[ep_a])
    _SUN_B, _OBS_B, _OD_B, _MAG_B = sun_unit[ep_b], obs_unit[ep_b], float(obs_dist[ep_b]), float(mag[ep_b])
    _TIMES0, _EP_A, _DT_AB, _DT_AC = times0, ep_a, dt_ab, dt_ac
    _W_LO, _W_HI = w_lo, w_hi
    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    _SUNC, _OBSC, _ODC, _MAGC = sunc, obsc, float(obs_dist[ep_c]), float(mag[ep_c])
    _set_coarse_globals(times0, ep_a, sun_unit, obs_unit, obs_dist, mag)

    # INFRA: truth-LC floor (must be small else gauge bug)
    _winit()
    truth_floor = _full_rmse((q_hist[ep_a], w_hist[ep_a]))
    print(f"[infra] truth-LC full-500 RMSE {truth_floor:.4f} "
          f"({'OK' if truth_floor < 0.05 else 'BUG'})", flush=True)

    # ---------- STEP 3: adaptive dense fill at A,B ----------
    ts = time.time()
    QA, QB = dense_fill(ctx, DENSE_POOL)
    used = DENSE_POOL
    while (len(QA) < DENSE_TARGET or len(QB) < DENSE_TARGET) and used < DENSE_MAX_POOL:
        add = min(DENSE_POOL, DENSE_MAX_POOL - used)
        QA2, QB2 = dense_fill(ctx, add)
        QA = np.vstack([QA, QA2]); QB = np.vstack([QB, QB2]); used += add
    nt_a_d, _ = nearest_in_pool_to_truth(QA, q_hist[ep_a]) if len(QA) else (np.nan, -1)
    nt_b_d, _ = nearest_in_pool_to_truth(QB, q_hist[ep_b]) if len(QB) else (np.nan, -1)
    print(f"[3] dense fill {used/1e6:.1f}M pool in {time.time()-ts:.0f}s | "
          f"|A_dense|={len(QA)} (nt {nt_a_d:.2f}deg) |B_dense|={len(QB)} (nt {nt_b_d:.2f}deg)", flush=True)

    # ===== MEASUREMENT (b): truth-near pair single vs multi-start shoot (diagnostic) =====
    # Uses the nearest DENSE member at each anchor — isolates the SHOOT question from the
    # decimation: does the connecting omega come out truth-near when a truth-near pair IS
    # supplied? (label-based diagnostic; runs on dense, independent of the reps below.)
    probe = {}
    if len(QA) and len(QB):
        _, ia_d = nearest_in_pool_to_truth(QA, q_hist[ep_a])
        _, ib_d = nearest_in_pool_to_truth(QB, q_hist[ep_b])
        qa_near, qb_near = QA[ia_d], QB[ib_d]
        w_fd = finite_diff_omega(qa_near, qb_near, dt_ab)
        s = shoot(qa_near, qb_near, dt_ab, INERTIA, w_fd)
        single_dir = omega_dir_err_deg(s["omega"], w_hist[ep_a])
        single_mag = float(np.linalg.norm(s["omega"]))
        roots, multi_best = multistart_shoot(qa_near, qb_near, dt_ab, w_lo, w_hi, w_hist[ep_a])
        probe = dict(nt_a_dense=float(nt_a_d), nt_b_dense=float(nt_b_d),
                     single_dir_err=float(single_dir), single_geo=float(s["geo_err_deg"]),
                     single_wmag=single_mag, single_in_bracket=bool(w_lo <= single_mag <= w_hi),
                     n_roots_multistart=len(roots), multistart_best_dir_err=float(multi_best))
        print(f"\n[(b) truth-near pair probe] nearest dense: A {nt_a_d:.2f}deg  B {nt_b_d:.2f}deg from truth", flush=True)
        print(f"    single-shoot:  dir_err {single_dir:6.2f}deg  geo {s['geo_err_deg']:.1e}  "
              f"|w| {single_mag:.5f} in-bracket {probe['single_in_bracket']}", flush=True)
        print(f"    multi-start :  {len(roots)} distinct roots, BEST dir_err {multi_best:6.2f}deg", flush=True)

    # ---------- STEP 4+5: decimate to reps + cross ----------
    if os.environ.get("S100_ANCHOR_CAP", "0") == "1":
        # COARSE-THEN-FINE (contract_slow-tumbler-generality v1). s111 fix #1 retained:
        # FIXED cell sizes, NO coarsening-by-growth, NO blind cull. The v0 single-pass 2deg
        # cross was unbounded on weak anchors (116 B=40941 cells -> 211M pairs). PASS 1
        # localizes connectable (A,B) regions on COARSE_DEG cells with a relaxed C-tol
        # (over-include the truth region); PASS 2 re-decimates the dense members inside
        # surviving coarse cells at DECIM_DEG (2deg) and re-crosses with the tight tol.
        repA_c, ncA_c = decimate_2deg(QA, deg=COARSE_DEG, max_reps=0)
        repB_c, ncB_c = decimate_2deg(QB, deg=COARSE_DEG, max_reps=0)
        print(f"[4c] coarse decimate {COARSE_DEG}deg | A {len(QA)}->{ncA_c} cells | "
              f"B {len(QB)}->{ncB_c} cells | coarse cross {ncA_c*ncB_c/1e6:.1f}M pairs", flush=True)
        ts = time.time()
        cross_c = run_cross(ctx, repA_c, repB_c, tol_mag=COARSE_TOL_MAG)
        print(f"[5c] coarse cross: {len(cross_c)} survivors in {time.time()-ts:.0f}s "
              f"(C-tol {COARSE_TOL_MAG})", flush=True)
        if not cross_c:
            print("NO coarse cross survivors.", flush=True)
            _save(OUT, dict(seed=SEED, aborted="no_coarse_survivors", probe=probe,
                            nt_a_dense=float(nt_a_d), nt_b_dense=float(nt_b_d),
                            ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, wall_s=time.time()-t0), None)
            return
        surv_qa_c = np.array([r[2] for r in cross_c])
        surv_qb_c = np.array([r[3] for r in cross_c])
        codeA = np.unique(cell_codes(surv_qa_c, COARSE_DEG))
        codeB = np.unique(cell_codes(surv_qb_c, COARSE_DEG))
        maskA = np.isin(cell_codes(QA, COARSE_DEG), codeA)
        maskB = np.isin(cell_codes(QB, COARSE_DEG), codeB)
        repA, ncA = decimate_2deg(QA[maskA], deg=DECIM_DEG, max_reps=0)
        repB, ncB = decimate_2deg(QB[maskB], deg=DECIM_DEG, max_reps=0)
        nt_a_r, ia_r = nearest_in_pool_to_truth(repA, q_hist[ep_a]) if len(repA) else (np.nan, -1)
        nt_b_r, ib_r = nearest_in_pool_to_truth(repB, q_hist[ep_b]) if len(repB) else (np.nan, -1)
        print(f"[4f] fine decimate {DECIM_DEG}deg in {len(codeA)}A/{len(codeB)}B surviving coarse "
              f"cells | A {int(maskA.sum())}->{len(repA)} reps (nt {nt_a_r:.2f}deg) | "
              f"B {int(maskB.sum())}->{len(repB)} reps (nt {nt_b_r:.2f}deg) | "
              f"fine cross {len(repA)*len(repB)/1e6:.1f}M pairs", flush=True)
        ts = time.time()
        cross_rows = run_cross(ctx, repA, repB, tol_mag=TOL_MAG)
        print(f"[5f] fine cross: {len(cross_rows)} C-pass survivors in {time.time()-ts:.0f}s", flush=True)
    else:
        repA, ncA = decimate_2deg(QA, rng=np.random.default_rng(1))
        repB, ncB = decimate_2deg(QB, rng=np.random.default_rng(2))
        nt_a_r, ia_r = nearest_in_pool_to_truth(repA, q_hist[ep_a]) if len(repA) else (np.nan, -1)
        nt_b_r, ib_r = nearest_in_pool_to_truth(repB, q_hist[ep_b]) if len(repB) else (np.nan, -1)
        print(f"[4] decimate {DECIM_DEG}deg | A: {len(QA)}->{ncA} cells ->{len(repA)} reps (nt {nt_a_r:.2f}deg) | "
              f"B: {len(QB)}->{ncB} cells ->{len(repB)} reps (nt {nt_b_r:.2f}deg)", flush=True)
        rA, rB = repA, repB
        if len(rA) * len(rB) > PAIR_BUDGET:
            cap = int(np.sqrt(PAIR_BUDGET)); rng = np.random.default_rng(0)
            if len(rA) > cap: rA = rA[np.sort(rng.choice(len(rA), cap, replace=False))]
            if len(rB) > cap: rB = rB[np.sort(rng.choice(len(rB), cap, replace=False))]
        print(f"\n[5] cross {len(rA)}x{len(rB)}={len(rA)*len(rB)} rep-pairs ...", flush=True)
        ts = time.time()
        cross_rows = run_cross(ctx, rA, rB, tol_mag=TOL_MAG)
        print(f"[5] cross: {len(cross_rows)} C-pass survivors in {time.time()-ts:.0f}s", flush=True)
    if not cross_rows:
        print("NO cross survivors.", flush=True)
        _save(OUT, dict(seed=SEED, aborted="no_cross_survivors", probe=probe,
                        nt_a_dense=float(nt_a_d), nt_b_dense=float(nt_b_d),
                        nt_a_rep=float(nt_a_r), nt_b_rep=float(nt_b_r),
                        ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, wall_s=time.time()-t0), None)
        return

    # rank by coarse RMSE -> full-500 on coarse-top-N
    crmse = np.array([r[0] for r in cross_rows])
    order = np.argsort(crmse)[:FULL_TOPN]
    full_work = [(np.array(cross_rows[i][2]), np.array(cross_rows[i][1])) for i in order]
    ts = time.time()
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        full_rmse = np.array(list(p.imap(_full_rmse, full_work, chunksize=16)))
    print(f"[5] full-500 on {len(full_work)} coarse-top in {time.time()-ts:.0f}s", flush=True)

    fo = np.argsort(full_rmse)
    qa_arr = np.array([cross_rows[order[i]][2] for i in range(len(order))])
    qb_arr = np.array([cross_rows[order[i]][3] for i in range(len(order))])
    om_arr = np.array([cross_rows[order[i]][1] for i in range(len(order))])
    cr_arr = crmse[order]
    qa_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qa_arr @ q_hist[ep_a]), 0, 1)))
    qb_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qb_arr @ q_hist[ep_b]), 0, 1)))
    dir_err = np.array([omega_dir_err_deg(w, w_hist[ep_a]) for w in om_arr])

    print(f"\n--- TOP-12 by full-500 RMSE (BLIND rank) | floor {truth_floor:.4f} ---", flush=True)
    print("rank | full-RMSE | coarse | dir    | qa    qb   (oracle labels)", flush=True)
    for k in range(min(12, len(fo))):
        i = fo[k]
        print(f"  {k+1:3d} | {full_rmse[i]:8.4f} | {cr_arr[i]:.4f} | {dir_err[i]:6.2f} | "
              f"{qa_geo[i]:5.2f} {qb_geo[i]:5.2f}", flush=True)

    # (a) does a truth-near pair survive the cross?
    pair_score = qa_geo + qb_geo
    j = int(np.argmin(pair_score))
    print(f"\n[(a) truth-near survivor] best (qa+qb) survivor: qa {qa_geo[j]:.2f} qb {qb_geo[j]:.2f} "
          f"dir {dir_err[j]:.2f} | full-RMSE {full_rmse[j]:.4f} (rank {int(np.where(fo==j)[0][0])+1})", flush=True)

    K = min(int(os.environ.get("S100_SAVE_K", 500)), len(fo))   # save many -> polish many (s110)
    top = fo[:K]
    np.savez(OUT / "invert.npz", qa=qa_arr[top], qb=qb_arr[top], omega=om_arr[top],
             full_rmse=full_rmse[top], coarse_rmse=cr_arr[top], qa_geo=qa_geo[top],
             qb_geo=qb_geo[top], dir_err=dir_err[top], ep_a=ep_a, ep_b=ep_b, ep_c=ep_c,
             truth_floor=truth_floor, w_true=w_hist[ep_a], times0=times0,
             repA=repA, repB=repB)
    meta = dict(seed=SEED, blind_bracket=[w_lo, w_hi], bracket_factor=w_hi / w_lo,
                truth_in_bracket=bool(w_lo <= wmag_true <= w_hi),
                ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, dt_ab=dt_ab, dt_ac=dt_ac,
                dense_pool_used=int(used), n_A_dense=len(QA), n_B_dense=len(QB),
                nt_a_dense=float(nt_a_d), nt_b_dense=float(nt_b_d),
                n_A_reps=len(repA), n_B_reps=len(repB),
                nt_a_rep=float(nt_a_r), nt_b_rep=float(nt_b_r),
                n_cross_survivors=len(cross_rows), n_full_scored=len(full_work),
                truth_floor=truth_floor, best_full_rmse=float(full_rmse[fo[0]]),
                best_pair_survivor=dict(qa=float(qa_geo[j]), qb=float(qb_geo[j]),
                                        dir=float(dir_err[j]), full_rmse=float(full_rmse[j])),
                probe=probe,
                top12=[dict(full_rmse=float(full_rmse[fo[k]]), coarse=float(cr_arr[fo[k]]),
                            dir=float(dir_err[fo[k]]), qa=float(qa_geo[fo[k]]), qb=float(qb_geo[fo[k]]))
                       for k in range(min(12, len(fo)))],
                wall_s=time.time() - t0)
    _save(OUT, meta, None)
    print(f"\nSaved: {OUT/'invert.npz'}\nSaved: {OUT/'invert.json'}\nTOTAL WALL: "
          f"{meta['wall_s']:.0f}s ({meta['wall_s']/60:.1f} min)", flush=True)


def _save(out, meta, _):
    with open(out / "invert.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)


if __name__ == "__main__":
    main()
