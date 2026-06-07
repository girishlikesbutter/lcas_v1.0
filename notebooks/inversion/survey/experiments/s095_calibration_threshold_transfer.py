"""s095 — does a cohort-calibrated truth-near windowed-RMSE threshold TRANSFER?

The filter question (user, 2026-05-22): we can't perturb the *real* truth at
inversion time. But we CAN calibrate a retention threshold on a cohort split
where truth IS known, then apply that fixed number to a held-out seed. This is
operational ONLY IF the threshold transfers across seeds. This script tests that.

Per seed we model a "pool-quantized truth-near candidate" the cheap way: take
truth's orientation at the two solve anchors A,B (from the propagated truth
trajectory), perturb each by the 30k-pool nearest-neighbour scale (s092 measured
3.2-5.7 deg), shoot for omega, propagate, and score the W_ACb window RMSE -- the
SAME scoring primitive as s094. The distribution of those windowed RMSEs is the
"truth-near" distribution: where an imperfect-but-truth-quality candidate lands.

Calibration: pool the truth-near RMSEs over the CALIBRATION seeds, set the
threshold T = 99th percentile. Transfer test: on HELD-OUT seeds (never in
calibration), what fraction of truth-near candidates fall below T? ~99% => the
threshold retains truth on unseen seeds (transfers). << 99% => it does not, and
the cohort-calibration idea is dead (you'd throw away truth on a new target).

Wrinkle tested: raw mag-RMSE likely won't transfer (a fixed angular error makes
more mag-RMSE on a high-contrast LC). So we also compute a dynamic-range-
NORMALIZED RMSE (divided by the std of the observed LC in the window) and test
whether the dimensionless threshold transfers better.

Anchors: frac-of-span (I_A, AB_FRAC, AC_FRAC) exactly like s092 but WITHOUT the
pool sharpness refinement -- deterministic, pool-free, consistent across seeds,
and removes the only per-seed setup cost. Window W_ACb = [ep_a, ep_c + BUFFER]
(the best window from s094). Pool(24), BLAS pinned, fork CoW, v2 surrogate.
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
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import (
    m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg, polhode_period,
)
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s095"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 15                    # quick first look
P = 300                         # perturbations (truth-near candidates) per seed
I_A = 100                       # base anchor epoch (matches s092)
AB_FRAC, AC_FRAC = 0.30, 0.45   # B, C as frac of span past A (matches s092)
BUFFER = 60                     # epochs past C for W_ACb (matches s094)
PERT_LO, PERT_HI = 2.0, 7.0     # per-anchor q jitter (deg); brackets 30k pool NN gap (s092)
SP_DEG, AD_DEG = 0.0, 15.0
CONNECT_TOL_DEG = 1e-3          # only count candidates that connect (match s092 screen)
DDIR_NEAR = 15.0                # "near-truth-omega" gate (deg) — isolates winding-alias from RMSE
PCTILE = 99.0                   # retention threshold percentile
INERTIA = m048_inertia()

_SURR = None
_SEEDS = {}                     # seed -> per-seed scoring data (fork CoW)


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


def _wmag_dps(seed):
    """Read just |omega| (dps) from a trajectory NPZ without loading big arrays."""
    p = tl.SURVEY_DATA_DIR / f"traj_seed{seed:03d}.npz"
    with np.load(p) as z:
        return float(z["omega_mag_dps"])


def _perturb(q_wxyz, theta_deg, rng):
    """Rotate a scalar-first quaternion by theta_deg about a random axis."""
    axis = rng.normal(size=3)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
    dq = Rotation.from_rotvec(np.radians(theta_deg) * axis)
    q_xyzw = np.asarray(q_wxyz, float)[[1, 2, 3, 0]]
    out = (dq * Rotation.from_quat(q_xyzw)).as_quat()
    return out[[3, 0, 1, 2]]


def _score(args):
    seed, pidx = args
    s = _SEEDS[seed]
    rng = np.random.default_rng([seed, pidx])
    qa = _perturb(s["q_a"], rng.uniform(PERT_LO, PERT_HI), rng)
    qb = _perturb(s["q_b"], rng.uniform(PERT_LO, PERT_HI), rng)
    try:
        with np.errstate(all="ignore"):
            wfd = finite_diff_omega(qa, qb, s["dt_ab"])
            if not np.all(np.isfinite(wfd)) or np.linalg.norm(wfd) < 1e-9:
                return seed, pidx, np.inf, np.inf, np.nan, np.nan
            sol = shoot(qa, qb, s["dt_ab"], INERTIA, wfd)
            w, geo = sol["omega"], sol["geo_err_deg"]
            quats, _ = propagate_jacobi_path2(qa, w, INERTIA, s["tsub"])  # q_a pinned at ep_a
            if not np.all(np.isfinite(quats)):
                return seed, pidx, np.inf, np.inf, np.nan, geo
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(
                np.einsum("nij,nj->ni", R, s["sun"]),
                np.einsum("nij,nj->ni", R, s["obs"]), SP_DEG, AD_DEG, s["od"])
            rmse = float(np.sqrt(np.mean((pred[:s["whi"] + 1] - s["mag_w"]) ** 2)))
            rmse_n = rmse / s["mag_std_win"]
            ddir = omega_dir_err_deg(w, s["w_true"])
    except (ValueError, FloatingPointError):
        return seed, pidx, np.inf, np.inf, np.nan, np.nan
    return seed, pidx, rmse, rmse_n, float(ddir), float(geo)


def _build_seed(seed):
    """Per-seed scoring data; returns (data_dict, info) or (None, reason)."""
    d = tl.load_truth(seed)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    if len(times0) <= I_A + 10:
        return None, f"too short ({len(times0)}ep)"
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    T_pol = polhode_period(w0, INERTIA)
    span = min(T_pol, times0[-1] - times0[I_A])
    ep_a = I_A
    ep_b = int(np.argmin(np.abs(times0 - (times0[I_A] + AB_FRAC * span))))
    ep_c = int(np.argmin(np.abs(times0 - (times0[I_A] + AC_FRAC * span))))
    epcb = min(ep_c + BUFFER, len(times0) - 1)
    if not (ep_a < ep_b < ep_c < epcb):
        return None, f"degenerate window a{ep_a} b{ep_b} c{ep_c} cb{epcb}"
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    mag = d["mag_hifi"].astype(np.float64)
    mag_std_win = float(np.std(mag[ep_a:epcb + 1]))
    # Geometry sliced from ep_a so propagate() gets times[0]==0 (the propagator pins
    # q_a at the first sample under a phi(0)=0 gauge; W_ACb is entirely >= ep_a).
    od = d["obs_dist"].astype(np.float64)
    data = dict(
        q_a=q_hist[ep_a], q_b=q_hist[ep_b], dt_ab=float(times0[ep_b] - times0[ep_a]),
        tsub=times0[ep_a:] - times0[ep_a], sun=sun_unit[ep_a:], obs=obs_unit[ep_a:],
        od=od[ep_a:], mag_w=mag[ep_a:epcb + 1], whi=epcb - ep_a,
        ep_a=ep_a, epcb=epcb, w_true=w_hist[ep_a], mag_std_win=mag_std_win,
    )
    info = dict(ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, epcb=epcb, T_pol=float(T_pol),
                wmag_dps=float(d["omega_mag_dps"]), mag_std_win=mag_std_win,
                n_ep=len(times0))
    return data, info


def main():
    t0 = time.time()
    all_seeds = tl.list_seeds()
    wmags = {s: _wmag_dps(s) for s in all_seeds}
    order = sorted(all_seeds, key=lambda s: wmags[s])
    pick = sorted({order[i] for i in np.linspace(0, len(order) - 1, N_SEEDS).round().astype(int)})
    if 116 not in pick:
        pick.append(116)
    pick = sorted(pick, key=lambda s: wmags[s])
    cal = pick[0::2]
    holdout = pick[1::2]
    print(f"===== s095 calibration-threshold transfer | {len(pick)} seeds, P={P} =====", flush=True)
    print(f"|omega| range {wmags[order[0]]:.3f}-{wmags[order[-1]]:.3f} dps over cohort", flush=True)
    print(f"CAL ({len(cal)}):     {cal}", flush=True)
    print(f"HOLDOUT ({len(holdout)}): {holdout}", flush=True)

    # ---- build per-seed scoring data ----
    info = {}
    used = []
    for seed in pick:
        data, meta = _build_seed(seed)
        if data is None:
            print(f"  seed {seed}: SKIP ({meta})", flush=True)
            continue
        _SEEDS[seed] = data
        info[seed] = meta
        used.append(seed)
    cal = [s for s in cal if s in used]
    holdout = [s for s in holdout if s in used]
    print(f"built {len(used)} seeds; per-seed anchors/windows ready", flush=True)

    work = [(seed, pidx) for seed in used for pidx in range(P)]
    ts = time.time()
    ctx = get_context("fork")
    rows = []
    with ctx.Pool(24, initializer=_winit) as p:
        for r in p.imap_unordered(_score, work, chunksize=64):
            rows.append(r)
    print(f"scored {len(rows)} truth-near candidates in {time.time()-ts:.0f}s\n", flush=True)

    # ---- per-seed distributions ----
    # Two candidate sets per seed:
    #   conn = connecting (geo < tol)               -- everything the prefilter keeps
    #   near = connecting AND ddir < DDIR_NEAR       -- the GENUINELY truth-near-omega
    #          candidate we actually want to retain (excludes winding-alias contamination)
    def pct(a, q): return float(np.percentile(a, q)) if len(a) else np.nan
    per = {}
    for seed in used:
        r = [x for x in rows if x[0] == seed]
        rmse = np.array([x[2] for x in r]); rmse_n = np.array([x[3] for x in r])
        ddir = np.array([x[4] for x in r]); geo = np.array([x[5] for x in r])
        conn = np.isfinite(rmse) & np.isfinite(geo) & (geo < CONNECT_TOL_DEG)
        near = conn & (ddir < DDIR_NEAR)
        per[seed] = dict(
            wmag_dps=wmags[seed], conn_frac=float(conn.mean()), near_frac=float(near.mean()),
            ddir_med=float(np.nanmedian(ddir[conn])) if conn.any() else np.nan,
            raw_p50=pct(rmse[conn], 50), raw_p99=pct(rmse[conn], PCTILE),
            norm_p50=pct(rmse_n[conn], 50), norm_p99=pct(rmse_n[conn], PCTILE),
            near_raw_p50=pct(rmse[near], 50), near_raw_p99=pct(rmse[near], PCTILE),
            near_norm_p50=pct(rmse_n[near], 50), near_norm_p99=pct(rmse_n[near], PCTILE),
            _rc=rmse[conn], _rnc=rmse_n[conn], _rc_n=rmse[near], _rnc_n=rmse_n[near],
        )

    # ---- calibrate on CAL, test retention on HOLDOUT, for each (set, metric) ----
    def calibrate(cal_key, hold_key, p99_key):
        cal_pool = np.concatenate([per[s][cal_key] for s in cal if len(per[s][cal_key])])
        T = float(np.percentile(cal_pool, PCTILE)) if len(cal_pool) else np.nan
        ret = {s: (float(np.mean(per[s][hold_key] < T)) if len(per[s][hold_key]) else np.nan)
               for s in holdout}
        p99s = np.array([per[s][p99_key] for s in used], float)
        spread = float(np.nanmax(p99s) / np.nanmin(p99s))
        return T, ret, spread

    T_raw,  ret_raw,  spread_raw  = calibrate("_rc",   "_rc",   "raw_p99")
    T_norm, ret_norm, spread_norm = calibrate("_rnc",  "_rnc",  "norm_p99")
    T_nr,   ret_nr,   spread_nr    = calibrate("_rc_n", "_rc_n", "near_raw_p99")
    T_nn,   ret_nn,   spread_nn    = calibrate("_rnc_n","_rnc_n","near_norm_p99")

    def mret(d):
        v = [x for x in d.values() if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan

    # ---- report ----
    print(f"{'seed':>4} {'set':>4} {'|w|dps':>7} {'ddir':>6} {'near%':>6} {'raw_p50':>8} "
          f"{'raw_p99':>8} {'nrm_p99':>8} {'NEAR raw_p99':>12} {'NEAR nrm_p99':>12}", flush=True)
    for s in sorted(used, key=lambda s: wmags[s]):
        p = per[s]; tag = "CAL" if s in cal else "HOLD"
        print(f"{s:>4} {tag:>4} {p['wmag_dps']:>7.3f} {p['ddir_med']:>6.1f} {100*p['near_frac']:>5.0f}% "
              f"{p['raw_p50']:>8.4f} {p['raw_p99']:>8.4f} {p['norm_p99']:>8.3f} "
              f"{p['near_raw_p99']:>12.4f} {p['near_norm_p99']:>12.3f}", flush=True)

    print(f"\n--- CALIBRATION (on {len(cal)} CAL seeds), per-seed p99 spread max/min ---", flush=True)
    print(f"  ALL-CONNECTING : T_raw={T_raw:.4f}mag (spread {spread_raw:.1f}x) | "
          f"T_norm={T_norm:.3f} (spread {spread_norm:.1f}x)", flush=True)
    print(f"  NEAR-OMEGA(<{DDIR_NEAR:.0f}): T_raw={T_nr:.4f}mag (spread {spread_nr:.1f}x) | "
          f"T_norm={T_nn:.3f} (spread {spread_nn:.1f}x)", flush=True)
    print(f"\n--- TRANSFER: retention on HELD-OUT seeds (want ~{PCTILE:.0f}%) ---", flush=True)
    print(f"{'seed':>4} {'|w|dps':>7} {'near%':>6} {'raw':>7} {'norm':>7} {'NEARraw':>8} {'NEARnrm':>8}", flush=True)
    for s in sorted(holdout, key=lambda s: wmags[s]):
        def f(d): return f"{100*d[s]:>6.1f}%" if np.isfinite(d[s]) else "   n/a"
        print(f"{s:>4} {wmags[s]:>7.3f} {100*per[s]['near_frac']:>5.0f}% "
              f"{f(ret_raw)} {f(ret_norm)} {f(ret_nr)} {f(ret_nn)}", flush=True)
    print(f"\nmean held-out retention:  raw {100*mret(ret_raw):.1f}%  norm {100*mret(ret_norm):.1f}%  "
          f"NEARraw {100*mret(ret_nr):.1f}%  NEARnorm {100*mret(ret_nn):.1f}%", flush=True)

    # ---- save ----
    meta = dict(
        n_seeds=len(used), P=P, pert_deg=[PERT_LO, PERT_HI], pctile=PCTILE, ddir_near=DDIR_NEAR,
        i_a=I_A, ab_frac=AB_FRAC, ac_frac=AC_FRAC, buffer=BUFFER,
        cal=cal, holdout=holdout, seed_info=info,
        per_seed={s: {k: v for k, v in per[s].items() if not k.startswith("_")} for s in used},
        T_raw=T_raw, T_norm=T_norm, T_near_raw=T_nr, T_near_norm=T_nn,
        spread_raw_p99=spread_raw, spread_norm_p99=spread_norm,
        spread_near_raw_p99=spread_nr, spread_near_norm_p99=spread_nn,
        retention_raw=ret_raw, retention_norm=ret_norm, retention_near_raw=ret_nr, retention_near_norm=ret_nn,
        mean_retention_raw=mret(ret_raw), mean_retention_norm=mret(ret_norm),
        mean_retention_near_raw=mret(ret_nr), mean_retention_near_norm=mret(ret_nn),
        wall_s=time.time() - t0,
    )
    with open(OUT / "transfer.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    np.savez_compressed(OUT / "truthnear_rmse.npz", seeds=np.array(used),
                        cal=np.array(cal), holdout=np.array(holdout),
                        rows=np.array(rows, dtype=float))   # (n,6) seed,pidx,rmse,rmse_n,ddir,geo
    _plot(used, cal, holdout, per, wmags, T_raw, T_norm)
    print(f"\nSaved: {OUT / 'transfer.json'}\nSaved: {OUT / 'truthnear_rmse.npz'}", flush=True)
    print(f"Total wall: {meta['wall_s']:.0f}s", flush=True)


def _plot(used, cal, holdout, per, wmags, T_raw, T_norm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(used, key=lambda s: wmags[s])
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, key, T, ttl in [
        (axes[0], "_rc", T_raw, "Raw windowed-RMSE (mag)"),
        (axes[1], "_rnc", T_norm, "Dynamic-range-normalized windowed-RMSE"),
    ]:
        for i, s in enumerate(order):
            a = per[s][key]
            col = "tab:blue" if s in cal else "tab:red"
            ax.scatter(np.full(len(a), i) + np.random.uniform(-0.12, 0.12, len(a)),
                       a, s=3, alpha=0.18, color=col)
            ax.plot([i - 0.3, i + 0.3], [np.median(a)] * 2, color=col, lw=2)        # median
            ax.plot([i - 0.3, i + 0.3], [np.percentile(a, 99)] * 2, color=col, lw=1, ls=":")  # p99
        ax.axhline(T, color="k", lw=1.5, ls="--",
                   label=f"CAL p99 threshold = {T:.4g}")
        ax.set_ylabel(ttl)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.2)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([f"{s}\n{wmags[s]:.2f}" for s in order], fontsize=8)
    axes[1].set_xlabel("seed (ordered by |omega| dps)   —   blue=CAL, red=HELD-OUT")
    axes[0].set_title("s095 truth-near windowed-RMSE per seed: does the CAL p99 threshold "
                      "retain truth on held-out seeds?", fontsize=11)
    fig.tight_layout()
    path = OUT / "transfer.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
