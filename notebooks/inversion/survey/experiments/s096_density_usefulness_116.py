"""s096 — does pulling the truth-near candidate to <1deg make connectability +
windowed-RMSE USEFUL on seed 116? (the load-bearing assumption behind densifying)

s092/s093/s094 all ran at 30k pool, where the nearest-truth cloud member is
3.2-5.7deg (s092) -- outside the ~1deg coherent tube (s003). The densification
plan rests on one untested assumption: that IF a denser pool put a candidate at
<1deg, the connectability omega-dir error would drop below the s087 ~1.25deg
recovery threshold AND its windowed RMSE would separate from junk. This measures
that basin profile directly, before we build the adaptive coarse->fine cross.

This is a forward-model BASIN PROBE (truth known at characterization time, like
s083/s084), NOT an inversion: we place candidates at controlled geodesic distance
delta from truth at the two solve anchors, shoot for omega, and score the same
W_ACb window s094 found best. We sweep delta and read off:
  (a) nearest-truth distance  = delta (by construction)
  (b) connecting omega-dir error vs delta  -> crosses 1.25deg (s087) at what delta?
  (c) windowed RMSE vs delta  -> drops below the JUNK band at what delta?

Junk reference = the REAL s092 connectable+C-pass survivors (no oracle), re-shot
and scored over the same window. Floor = exact (q_truth, omega_truth) propagated.
Anchors / window are taken from results/s092/cross.json for faithfulness.
Pool(24), BLAS pinned, fork CoW, v2 surrogate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.spatial.transform import Rotation

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import sample_so3_pool, compute_j2000_units
from lib.shoot import (
    m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg, polhode_period,
)
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s096"
OUT.mkdir(parents=True, exist_ok=True)
S092 = Path(__file__).resolve().parent.parent / "results" / "s092" / "cross.json"

SEED = 116
POOL_N = 30_000                 # must match s092 to map survivor pool indices
RNG_SEED = 42                   # must match s092
BUFFER = 60                     # W_ACb = [ep_a, ep_c + BUFFER]  (s094 best window)
DELTAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]   # truth-near distances (deg)
M = 120                         # random-direction perturbations per delta
N_JUNK = 2500                   # real survivors subsampled for the junk band
SP_DEG, AD_DEG = 0.0, 15.0
CONNECT_TOL_DEG = 1e-3
OMEGA_DIR_THRESH = 1.25         # s087 fast-seed recovery threshold (deg)
INERTIA = m048_inertia()

_SURR = None
# Geometry sliced from the anchor epoch onward, so propagate() always gets times[0]==0
# (the propagator pins q_a using theta_hist[0]/psi_hist[0] under a phi(0)=0 gauge, line
#  669-674 of jacobi_propagator.py -- only self-consistent when times[0]==0).
_TSUB = _SUNS = _OBSS = _ODS = _MAGW = None
_WHI = _DT_AB = None
_TREL_FULL = _SUN_FULL = _OBS_FULL = _OD_FULL = _MAG_FULL = _EPA = _EPCB = None  # for buggy-way demo
_QA0 = _QB0 = _WTRUE = None     # truth orientations at A,B and truth omega@A


def _winit():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


def _perturb(q_wxyz, theta_deg, rng):
    axis = rng.normal(size=3)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
    dq = Rotation.from_rotvec(np.radians(theta_deg) * axis)
    q_xyzw = np.asarray(q_wxyz, float)[[1, 2, 3, 0]]
    out = (dq * Rotation.from_quat(q_xyzw)).as_quat()
    return out[[3, 0, 1, 2]]


def _windowed_rmse(q_a, w):
    """Propagate from q_a at the anchor (times[0]==0) and RMSE over the W_ACb window."""
    quats, _ = propagate_jacobi_path2(q_a, w, INERTIA, _TSUB)
    if not np.all(np.isfinite(quats)):
        return np.inf
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUNS),
                                   np.einsum("nij,nj->ni", R, _OBSS), SP_DEG, AD_DEG, _ODS)
    return float(np.sqrt(np.mean((pred[:_WHI + 1] - _MAGW) ** 2)))


def _windowed_rmse_buggy(q_a, w):
    """The s094/s096-v1 way: times[0] != 0 -> q_a pinned at the wrong epoch (for demo only)."""
    quats, _ = propagate_jacobi_path2(q_a, w, INERTIA, _TREL_FULL)
    if not np.all(np.isfinite(quats)):
        return np.inf
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN_FULL),
                                   np.einsum("nij,nj->ni", R, _OBS_FULL), SP_DEG, AD_DEG, _OD_FULL)
    return float(np.sqrt(np.mean((pred[_EPA:_EPCB + 1] - _MAG_FULL[_EPA:_EPCB + 1]) ** 2)))


def _score_truthnear(args):
    """delta-perturb both truth anchors, shoot, score. Returns (delta, ddir, rmse, geo)."""
    delta, m = args
    rng = np.random.default_rng([int(delta * 1000), m])
    q_a = _perturb(_QA0, delta, rng)
    q_b = _perturb(_QB0, delta, rng)
    try:
        with np.errstate(all="ignore"):
            wfd = finite_diff_omega(q_a, q_b, _DT_AB)
            if not np.all(np.isfinite(wfd)) or np.linalg.norm(wfd) < 1e-9:
                return delta, np.nan, np.inf, np.inf
            sol = shoot(q_a, q_b, _DT_AB, INERTIA, wfd)
            w, geo = sol["omega"], sol["geo_err_deg"]
            ddir = omega_dir_err_deg(w, _WTRUE)
            rmse = _windowed_rmse(q_a, w)
    except (ValueError, FloatingPointError):
        return delta, np.nan, np.inf, np.inf
    return delta, float(ddir), float(rmse), float(geo)


def _score_junk(args):
    """Real s092 survivor pair (q from pool indices), re-shot + scored."""
    qa, qb = args
    try:
        with np.errstate(all="ignore"):
            wfd = finite_diff_omega(qa, qb, _DT_AB)
            if not np.all(np.isfinite(wfd)) or np.linalg.norm(wfd) < 1e-9:
                return np.nan, np.inf, np.inf
            sol = shoot(qa, qb, _DT_AB, INERTIA, wfd)
            w, geo = sol["omega"], sol["geo_err_deg"]
            ddir = omega_dir_err_deg(w, _WTRUE)
            rmse = _windowed_rmse(qa, w)
    except (ValueError, FloatingPointError):
        return np.nan, np.inf, np.inf
    return float(ddir), float(rmse), float(geo)


def main():
    t0 = time.time()
    s092 = json.load(open(S092))
    ep_a, ep_b, ep_c, dt_ab = s092["ep_a"], s092["ep_b"], s092["ep_c"], s092["dt_ab_s"]
    a_pool = np.array(s092["cpass_a_pool"], int); b_pool = np.array(s092["cpass_b_pool"], int)

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    T_pol = polhode_period(w0, INERTIA)
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    pool = sample_so3_pool(POOL_N, RNG_SEED); qP = pool["q_pool_wxyz"]
    epcb = min(ep_c + BUFFER, len(times0) - 1)
    mag = d["mag_hifi"].astype(np.float64)
    mag_std_win = float(np.std(mag[ep_a:epcb + 1]))

    global _SURR, _TSUB, _SUNS, _OBSS, _ODS, _MAGW, _WHI, _DT_AB
    global _TREL_FULL, _SUN_FULL, _OBS_FULL, _OD_FULL, _MAG_FULL, _EPA, _EPCB
    global _QA0, _QB0, _WTRUE
    obs_dist = d["obs_dist"].astype(np.float64)
    _TSUB = times0[ep_a:] - times0[ep_a]                       # correct: times[0]==0
    _SUNS, _OBSS, _ODS = sun_unit[ep_a:], obs_unit[ep_a:], obs_dist[ep_a:]
    _MAGW, _WHI, _DT_AB = mag[ep_a:epcb + 1], epcb - ep_a, dt_ab
    _TREL_FULL, _SUN_FULL, _OBS_FULL = times0 - times0[ep_a], sun_unit, obs_unit  # buggy-way demo
    _OD_FULL, _MAG_FULL, _EPA, _EPCB = obs_dist, mag, ep_a, epcb
    _QA0, _QB0, _WTRUE = q_hist[ep_a], q_hist[ep_b], w_hist[ep_a]

    print(f"===== s096 density-usefulness | seed {SEED} =====", flush=True)
    print(f"|w|={np.degrees(np.linalg.norm(w0)):.4f}dps  T_pol={T_pol:.0f}s  "
          f"anchors A={ep_a} B={ep_b} C={ep_c}  window W_ACb=[{ep_a},{epcb}] "
          f"({epcb-ep_a+1}ep)  mag_std_win={mag_std_win:.3f}", flush=True)

    # ---- infra validation ----
    _SURR = get_model()
    # does propagating truth from the anchor reproduce the truth trajectory?
    q_corr, _ = propagate_jacobi_path2(_QA0, _WTRUE, INERTIA, _TSUB)
    repro_corr = float(np.degrees(2 * np.arccos(np.clip(
        np.abs(np.einsum("ni,ni->n", q_corr, q_hist[ep_a:])), 0, 1))).max())
    q_bug, _ = propagate_jacobi_path2(_QA0, _WTRUE, INERTIA, _TREL_FULL)
    repro_bug = float(np.degrees(2 * np.arccos(np.clip(
        np.abs(np.einsum("ni,ni->n", q_bug, q_hist)), 0, 1))).max())
    # exact-truth floor: should be ~surrogate floor (correct) vs junk-level (buggy)
    floor_rmse = _windowed_rmse(_QA0, _WTRUE)
    floor_buggy = _windowed_rmse_buggy(_QA0, _WTRUE)
    # exact-anchor shoot (delta=0): does shoot recover omega from exact anchors?
    s0 = shoot(_QA0, _QB0, _DT_AB, INERTIA, finite_diff_omega(_QA0, _QB0, _DT_AB))
    floor_shoot_ddir = omega_dir_err_deg(s0["omega"], _WTRUE)
    floor_shoot_rmse = _windowed_rmse(_QA0, s0["omega"])
    print(f"\n[infra] truth reproduction (max geodesic): correct-way={repro_corr:.2e}deg  "
          f"buggy-way(times[0]!=0)={repro_bug:.1f}deg", flush=True)
    print(f"[infra] exact-truth floor RMSE: correct={floor_rmse:.4f} mag  "
          f"buggy(s094-style)={floor_buggy:.4f} mag", flush=True)
    print(f"[infra] exact-anchor shoot: omega-dir={floor_shoot_ddir:.3f}deg  "
          f"RMSE={floor_shoot_rmse:.4f} mag  geo={s0['geo_err_deg']:.1e}deg", flush=True)

    ctx = get_context("fork")

    # ---- truth-near sweep ----
    work = [(delta, m) for delta in DELTAS for m in range(M)]
    tn = {delta: dict(ddir=[], rmse=[], geo=[]) for delta in DELTAS}
    ts = time.time()
    with ctx.Pool(24, initializer=_winit) as p:
        for delta, ddir, rmse, geo in p.imap_unordered(_score_truthnear, work, chunksize=16):
            tn[delta]["ddir"].append(ddir); tn[delta]["rmse"].append(rmse); tn[delta]["geo"].append(geo)
    print(f"\ntruth-near sweep ({len(work)} evals) in {time.time()-ts:.0f}s", flush=True)

    # ---- junk reference (real survivors) ----
    rng = np.random.default_rng(0)
    sel = rng.choice(len(a_pool), min(N_JUNK, len(a_pool)), replace=False)
    jwork = [(qP[a_pool[i]], qP[b_pool[i]]) for i in sel]
    ts = time.time()
    j_ddir, j_rmse, j_geo = [], [], []
    with ctx.Pool(24, initializer=_winit) as p:
        for ddir, rmse, geo in p.imap_unordered(_score_junk, jwork, chunksize=16):
            j_ddir.append(ddir); j_rmse.append(rmse); j_geo.append(geo)
    j_rmse = np.array(j_rmse); j_ddir = np.array(j_ddir)
    jr = j_rmse[np.isfinite(j_rmse)]
    junk_p1, junk_p50 = float(np.percentile(jr, 1)), float(np.percentile(jr, 50))
    print(f"junk reference ({len(jwork)} real survivors) in {time.time()-ts:.0f}s", flush=True)
    print(f"junk windowed-RMSE: p1={junk_p1:.4f}  p50={junk_p50:.4f} mag  "
          f"(min {jr.min():.4f})", flush=True)

    # ---- per-delta summary ----
    def stat(a, q):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return float(np.percentile(a, q)) if len(a) else np.nan
    print(f"\n{'delta':>6} {'ddir_p50':>9} {'ddir_p90':>9} {'rmse_p50':>9} {'rmse_p90':>9} "
          f"{'norm_p50':>9} {'<junk_p1%':>9} {'<1.25deg%':>10}", flush=True)
    summ = {}
    for delta in DELTAS:
        dd = np.array(tn[delta]["ddir"], float); rr = np.array(tn[delta]["rmse"], float)
        rr_f = rr[np.isfinite(rr)]; dd_f = dd[np.isfinite(dd)]
        below_junk = float(100 * np.mean(rr_f < junk_p1)) if len(rr_f) else np.nan
        below_th = float(100 * np.mean(dd_f < OMEGA_DIR_THRESH)) if len(dd_f) else np.nan
        summ[delta] = dict(
            ddir_p50=stat(dd, 50), ddir_p90=stat(dd, 90),
            rmse_p50=stat(rr, 50), rmse_p90=stat(rr, 90),
            norm_p50=stat(rr, 50) / mag_std_win,
            frac_rmse_below_junk_p1=below_junk, frac_ddir_below_thresh=below_th,
        )
        print(f"{delta:>6.2f} {summ[delta]['ddir_p50']:>9.2f} {summ[delta]['ddir_p90']:>9.2f} "
              f"{summ[delta]['rmse_p50']:>9.4f} {summ[delta]['rmse_p90']:>9.4f} "
              f"{summ[delta]['norm_p50']:>9.3f} {below_junk:>8.0f}% {below_th:>9.0f}%", flush=True)

    # ---- required-density translation ----
    # SO(3) NN gap ~ N^(-1/3); 30k -> ~5deg (s092). pool(delta) = 30k*(5/delta)^3
    def pool_for(delta):
        return int(POOL_N * (5.0 / delta) ** 3)

    meta = dict(
        seed=SEED, pool_n=POOL_N, ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, epcb=epcb,
        window_len=epcb - ep_a + 1, mag_std_win=mag_std_win, deltas=DELTAS, M=M,
        floor_rmse=floor_rmse, floor_rmse_buggy=floor_buggy,
        repro_correct_deg=repro_corr, repro_buggy_deg=repro_bug,
        floor_shoot_ddir=floor_shoot_ddir, floor_shoot_rmse=floor_shoot_rmse,
        junk_n=len(jwork), junk_rmse_p1=junk_p1, junk_rmse_p50=junk_p50, junk_rmse_min=float(jr.min()),
        junk_ddir_med=float(np.nanmedian(j_ddir)), omega_dir_thresh=OMEGA_DIR_THRESH,
        per_delta=summ, pool_for_1deg=pool_for(1.0), pool_for_0p5deg=pool_for(0.5),
        wall_s=time.time() - t0,
    )
    with open(OUT / "basin.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    np.savez_compressed(OUT / "basin.npz",
                        deltas=np.array(DELTAS),
                        **{f"ddir_{delta}": np.array(tn[delta]["ddir"]) for delta in DELTAS},
                        **{f"rmse_{delta}": np.array(tn[delta]["rmse"]) for delta in DELTAS},
                        junk_rmse=j_rmse, junk_ddir=j_ddir)
    _plot(DELTAS, tn, floor_rmse, junk_p1, junk_p50, jr, mag_std_win)

    print(f"\npool size for nearest-truth 1.0deg: ~{pool_for(1.0):,}  | "
          f"0.5deg: ~{pool_for(0.5):,}  (uniform; N^-1/3 from 30k->5deg)", flush=True)
    print(f"\nSaved: {OUT / 'basin.json'}\nSaved: {OUT / 'basin.npz'}", flush=True)
    print(f"Total wall: {meta['wall_s']:.0f}s", flush=True)


def _plot(deltas, tn, floor, junk_p1, junk_p50, jr, mag_std_win):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    dd_p50 = [np.nanpercentile(tn[dl]["ddir"], 50) for dl in deltas]
    dd_p10 = [np.nanpercentile(tn[dl]["ddir"], 10) for dl in deltas]
    dd_p90 = [np.nanpercentile(tn[dl]["ddir"], 90) for dl in deltas]
    axes[0].fill_between(deltas, dd_p10, dd_p90, alpha=0.2, color="tab:blue")
    axes[0].plot(deltas, dd_p50, "o-", color="tab:blue", label="truth-near omega-dir (p50, 10-90 band)")
    axes[0].axhline(OMEGA_DIR_THRESH, color="k", ls="--", lw=1.2, label=f"s087 recovery thresh {OMEGA_DIR_THRESH}deg")
    axes[0].set_xlabel("truth-near distance delta (deg)"); axes[0].set_ylabel("connecting omega-dir error (deg)")
    axes[0].set_yscale("log"); axes[0].set_title("(b) connectability vs density")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)

    rr_p50 = [np.nanpercentile(np.array(tn[dl]["rmse"])[np.isfinite(tn[dl]["rmse"])], 50) for dl in deltas]
    rr_p10 = [np.nanpercentile(np.array(tn[dl]["rmse"])[np.isfinite(tn[dl]["rmse"])], 10) for dl in deltas]
    rr_p90 = [np.nanpercentile(np.array(tn[dl]["rmse"])[np.isfinite(tn[dl]["rmse"])], 90) for dl in deltas]
    axes[1].fill_between(deltas, rr_p10, rr_p90, alpha=0.2, color="tab:green")
    axes[1].plot(deltas, rr_p50, "o-", color="tab:green", label="truth-near windowed-RMSE (p50, 10-90 band)")
    axes[1].axhspan(junk_p1, junk_p50, alpha=0.15, color="tab:red", label="real junk RMSE band (p1-p50)")
    axes[1].axhline(junk_p1, color="tab:red", ls=":", lw=1)
    axes[1].axhline(floor, color="k", ls="--", lw=1.2, label=f"exact-truth floor {floor:.3f} mag")
    axes[1].axhline(mag_std_win, color="grey", ls="-.", lw=1, label=f"LC dyn-range (std {mag_std_win:.2f})")
    axes[1].set_xlabel("truth-near distance delta (deg)"); axes[1].set_ylabel("windowed RMSE (mag)")
    axes[1].set_yscale("log"); axes[1].set_title("(c) discriminator vs density")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.suptitle(f"s096 seed 116: does pulling truth-near to <1deg make connectability + RMSE useful?", fontsize=12)
    fig.tight_layout()
    path = OUT / "basin.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
