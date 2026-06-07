"""s085 — anchor-accuracy go/no-go gate on fast tumblers (the s084 open question).

s084 found the binding constraint for blind inversion moved to "anchor
q-accuracy": with q0 held FIXED and only |ω| refined, the fast seed needs
q within ~1-3° to clear Band B. The handoff asks: "can v3's sharp-|C_t|
anchoring deliver ~1-3° q on a fast seed?"

But s084's "1-3° budget" is a STATIC ρ-floor (q0 fixed, only |ω| refined).
The v3 pipeline POLISHES q0 jointly (s064). Whether 1-3° is truly binding
depends on the LM grab radius (s005, pre-fix: 5-15° good seeds, ~2° tight
seed-28). This experiment measures both, post-fix, on fast holdout seeds.

LAYER A — anchor delivery (the handoff's literal ask):
  * sharpness map (coarse 50k pool, all epochs) -> T_A = argmin |C_t|
    (`project_anchor_selection_coarse_then_dense`: coarse for WHERE).
  * dense q-cloud at T_A, density scan {100k,200k,400k,800k}: best q_geo to
    truth-q(T_A) = nearest survivor (DENSE for HOW CLOSE). 400k = headline.
  NOT oracle: truth is the comparison TARGET, never inserted into the pool.

LAYER B — does the joint polish bridge the gap (resolves the static-floor
  vs LM-basin ambiguity):
  * controlled-perturbation grab-radius test (the s005/s064-gate2 method, NOT
    oracle): perturb truth-q(T_A) by {2,3,5}° over 3 random axes, seed ω at
    two realism levels, back_propagate to t=0, lm_polish_jacobi (s064), hi-fi
    classify ρ<4.
    - L0 (truth ω): q0_seed_err == q_pert exactly (left-invariance under
      identical body-ω) -> isolates the pure q grab radius.
    - L1 (dir+1.5°, |ω|+0.5%): realistic blind ω; back-prop through a fast
      tumbler's many revolutions amplifies the ω error into q0.

Seeds: 119/108/100 (fast LAM, |ω| 1.35-1.48 dps) + 116 (slow control, |ω|
0.13; s084 says it tolerates 15°, so it must pass — pipeline sanity).

Usage:
    python experiments/s085_anchor_accuracy_gate.py            # full run
    python experiments/s085_anchor_accuracy_gate.py --smoke    # 1 seed, quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions,
    survive_at_epoch, nearest_in_pool_to_truth,
)
from lib.surrogate_eval import get_model  # noqa: E402
from lib import traj_load  # noqa: E402
from lib.hifi_render import build_context, rho_band  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from experiments.s059_pilot import back_propagate, TOL_MAG, SP_DEG, AD_DEG  # noqa: E402
from experiments.s064_jacobi_polish import (  # noqa: E402
    lm_polish_jacobi, hifi_classify, perturb_q, perturb_omega, quat_geodesic_deg,
)

FAST_SEEDS = [119, 108, 100]
SLOW_SEED = 116
DENSITIES = [100_000, 200_000, 400_000, 800_000]
HEADLINE_N = 400_000
SHARP_POOL_N = 50_000
EDGE_SKIP = 5            # exclude first/last 5 epochs from anchor selection
RNG_SEED = 42

# Layer B ladder
Q_PERT_FAST = [2.0, 3.0, 5.0]
Q_PERT_SLOW = [2.0, 5.0]
N_AXES = 3
# ω realism levels: (label, dir_deg, mag_pct)
OMEGA_LEVELS = [
    ("L0_truth_omega", 0.0, 0.0),
    ("L1_realistic", 1.5, 0.5),   # grid-dir mid (s082: 0.94-2.69°) + refine |ω| (s084: <=0.4%)
]


# ----------------------- Layer A: sharpness map (parallel) -----------------------

_R_CACHE = None
_MODEL = None
_SUN = None
_OBS = None
_ODIST = None
_MAG = None


def _sharp_init(R_cache, sun_unit, obs_unit, obs_dist, mag_meas):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _R_CACHE, _MODEL, _SUN, _OBS, _ODIST, _MAG
    _R_CACHE = R_cache
    _MODEL = get_model()
    _SUN, _OBS, _ODIST, _MAG = sun_unit, obs_unit, obs_dist, mag_meas


def _sharp_epoch(t):
    k1, k2 = project_directions(_R_CACHE, _SUN[t], _OBS[t])
    _, keep = survive_at_epoch(_MODEL, k1, k2, float(_ODIST[t]),
                               SP_DEG, AD_DEG, float(_MAG[t]), TOL_MAG)
    return t, int(keep.sum())


def sharpness_map(d, n_workers):
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    obs_dist, mag_meas = d["obs_dist"], d["mag_hifi"]
    n_ep = sun_unit.shape[0]
    pool = sample_so3_pool(SHARP_POOL_N, sample_seed=RNG_SEED)
    R_cache = pool["R_cache"]
    epochs = list(range(EDGE_SKIP, n_ep - EDGE_SKIP))
    Ct = np.full(n_ep, -1, dtype=np.int64)
    t0 = time.time()
    if n_workers <= 1:
        _sharp_init(R_cache, sun_unit, obs_unit, obs_dist, mag_meas)
        for t in epochs:
            _, c = _sharp_epoch(t)
            Ct[t] = c
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(n_workers, initializer=_sharp_init,
                           initargs=(R_cache, sun_unit, obs_unit, obs_dist, mag_meas)) as pool_:
            for t, c in pool_.imap_unordered(_sharp_epoch, epochs, chunksize=8):
                Ct[t] = c
    wall = time.time() - t0
    valid = Ct >= 0
    T_A = int(np.argmin(np.where(valid, Ct, np.iinfo(np.int64).max)))
    return Ct, T_A, float(wall)


# ----------------------- Layer A: density scan at T_A -----------------------

def density_scan(d, T_A, pools):
    """For each pre-sampled pool, project at T_A, survive, nearest-to-truth."""
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    obs_dist, mag_meas = d["obs_dist"], d["mag_hifi"]
    q_truth_at = d["quaternions"][T_A]
    model = get_model()
    rows = []
    for N, pool in pools.items():
        R_cache = pool["R_cache"]
        q_pool = pool["q_pool_wxyz"]
        k1, k2 = project_directions(R_cache, sun_unit[T_A], obs_unit[T_A])
        _, keep = survive_at_epoch(model, k1, k2, float(obs_dist[T_A]),
                                   SP_DEG, AD_DEG, float(mag_meas[T_A]), TOL_MAG)
        n_surv = int(keep.sum())
        if n_surv > 0:
            best_deg, _ = nearest_in_pool_to_truth(q_pool[keep], q_truth_at)
        else:
            best_deg = float("nan")
        # pool-wide nearest (no survival filter) — the quantization floor
        floor_deg, _ = nearest_in_pool_to_truth(q_pool, q_truth_at)
        rows.append({"N": int(N), "n_surv": n_surv,
                     "surv_rate_pct": 100.0 * n_surv / N,
                     "best_q_geo_deg": float(best_deg),
                     "pool_floor_deg": float(floor_deg)})
    return rows


# ----------------------- Layer B: polish ladder (parallel) -----------------------

_CTX_B = None
_TARGET_B = None


def _polishB_init(ctx, target):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _CTX_B, _TARGET_B
    _CTX_B = ctx
    _TARGET_B = target


def _polishB_worker(job):
    (q_pert, om_label, om_dir, om_mag, axis_i, rng_seed,
     q_a_truth, om_a_truth, t_a_seconds, inertia, q0_truth, om0_truth) = job
    rng = np.random.default_rng(rng_seed)
    q_a_truth = np.asarray(q_a_truth); om_a_truth = np.asarray(om_a_truth)
    qa_p = perturb_q(q_a_truth, q_pert, rng)
    om_p = perturb_omega(om_a_truth, om_dir, om_mag, rng) if (om_dir or om_mag) else om_a_truth
    q0_seed, om0_seed = back_propagate(qa_p, om_p, t_a_seconds, np.asarray(inertia))
    q0_seed_err = quat_geodesic_deg(np.asarray(q0_seed), np.asarray(q0_truth))

    res = lm_polish_jacobi(q0_seed, om0_seed, _CTX_B, _TARGET_B,
                           label=f"q{q_pert}_{om_label}_ax{axis_i}")
    q0p = np.asarray(res["q0_pol_wxyz"]); om0p = np.asarray(res["om0_pol_rad"])
    om_truth_mag = float(np.linalg.norm(om0_truth))
    q0_err = quat_geodesic_deg(q0p, np.asarray(q0_truth))
    om_mag_err = float((np.linalg.norm(om0p) - om_truth_mag) / om_truth_mag * 100)
    om_dir_err = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                   np.asarray(om0_truth) / om_truth_mag)), 0, 1))))
    return {
        "q_pert_deg": float(q_pert), "om_level": om_label, "axis": int(axis_i),
        "q0_seed_err_deg": float(q0_seed_err),
        "rho_seed_surr": float(res["surrogate_rho_seed"]),
        "rho_pol_surr": float(res["surrogate_rho_polished"]),
        "q0_err_deg": float(q0_err),
        "om_mag_err_pct": float(om_mag_err),
        "om_dir_err_deg": float(om_dir_err),
        "n_eval": int(res["n_eval"]),
        "q0_pol_wxyz": res["q0_pol_wxyz"], "om0_pol_rad": res["om0_pol_rad"],
    }


def layer_b(seed, T_A, ctx, q_pert_levels, omega_levels, n_workers):
    target = ctx["mag_hifi_truth"]
    inertia = ctx["inertia_tensor"]
    q0_truth = np.asarray(ctx["q0_truth"], float)
    om0_truth = np.asarray(ctx["omega0_truth_rad"], float)
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    quats_t, omegas_t = propagate_attitude(
        q0=q0_truth, omega0=om0_truth, times=ctx["observation_times"],
        mode="tumbling", inertia_tensor=inertia,
    )
    q_a_truth = np.asarray(quats_t[T_A], float)
    om_a_truth = np.asarray(omegas_t[T_A], float)

    jobs = []
    for q_pert in q_pert_levels:
        for (om_label, om_dir, om_mag) in omega_levels:
            for ax in range(N_AXES):
                rng_seed = 7000 + seed * 100 + int(q_pert * 10) + ax + (1 if om_dir else 0) * 50
                jobs.append((q_pert, om_label, om_dir, om_mag, ax, rng_seed,
                             q_a_truth, om_a_truth, t_a_seconds, inertia, q0_truth, om0_truth))

    t0 = time.time()
    out = []
    if n_workers <= 1:
        _polishB_init(ctx, target)
        for j in jobs:
            out.append(_polishB_worker(j))
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(n_workers, initializer=_polishB_init,
                           initargs=(ctx, target)) as pool_:
            for r in pool_.imap_unordered(_polishB_worker, jobs):
                out.append(r)
    polish_wall = time.time() - t0

    # Surrogate-first: band from ρ_surr for ALL candidates (s081 established
    # 145/145 surrogate↔hi-fi band agreement). Convergence is read off q0_err
    # (bimodal per s005: <1° converged, >20° escaped). hi-fi is the EXPENSIVE
    # step (~56s/render), so confirm only the single best-ρ_surr converged
    # candidate per seed as a spot-check against surrogate-wide bias.
    for r in out:
        r["band_surr"] = rho_band(r["rho_pol_surr"])
    out_conv = [r for r in out if r["rho_pol_surr"] < 4.0]
    rho_hifi_check = None
    t1 = time.time()
    if out_conv:
        best = min(out_conv, key=lambda r: r["rho_pol_surr"])
        rho_h, band_h = hifi_classify(best["q0_pol_wxyz"], best["om0_pol_rad"], ctx, target)
        rho_hifi_check = {
            "q_pert_deg": best["q_pert_deg"], "om_level": best["om_level"],
            "axis": best["axis"], "rho_surr": best["rho_pol_surr"],
            "rho_hifi": float(rho_h), "band_surr": best["band_surr"],
            "band_hifi": band_h, "q0_err_deg": best["q0_err_deg"],
        }
    hifi_wall = time.time() - t1

    out.sort(key=lambda r: (r["q_pert_deg"], r["om_level"], r["axis"]))
    return out, t_a_seconds, float(polish_wall), float(hifi_wall), rho_hifi_check


# ----------------------- per-seed driver + plots -----------------------

def aggregate_b(rows):
    """Median over axes per (q_pert, om_level). Primary metric = ρ_surr (band
    proxy validated 145/145 by s081) + q0_err (bimodal convergence per s005)."""
    agg = {}
    for r in rows:
        key = (r["q_pert_deg"], r["om_level"])
        agg.setdefault(key, []).append(r)
    summary = []
    for (q_pert, om_level), grp in sorted(agg.items()):
        rhos = [g["rho_pol_surr"] for g in grp]
        bands = [g["band_surr"] for g in grp]
        n_AB = sum(1 for b in bands if b in {"A", "B"})
        # converged = q0_err < 5° (s005 strict basin); bimodal so threshold robust
        n_conv = sum(1 for g in grp if g["q0_err_deg"] < 5.0)
        summary.append({
            "q_pert_deg": q_pert, "om_level": om_level,
            "n_axes": len(grp),
            "q0_seed_err_med": float(np.median([g["q0_seed_err_deg"] for g in grp])),
            "rho_surr_med": float(np.median(rhos)),
            "rho_surr_min": float(np.min(rhos)),
            "q0_err_med": float(np.median([g["q0_err_deg"] for g in grp])),
            "n_band_AB": n_AB,
            "n_converged": n_conv,
            "bands": bands,
        })
    return summary


def plot_seed(seed, Ct, T_A, mag, dens_rows, b_summary, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    n_ep = len(Ct)
    t_arr = np.arange(n_ep)
    valid = Ct >= 0

    ax = axes[0, 0]
    ax.plot(t_arr[valid], mag[valid], "k-", lw=0.7)
    ax.axvline(T_A, color="red", ls="--", lw=1)
    ax.scatter([T_A], [mag[T_A]], c="red", s=50, zorder=5, label=f"T_A={T_A}")
    ax.invert_yaxis(); ax.set_ylabel("mag"); ax.set_xlabel("epoch")
    ax.set_title(f"seed {seed}: LC + anchor"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(t_arr[valid], np.maximum(Ct[valid], 1), "b-", lw=0.7)
    ax.scatter([T_A], [max(Ct[T_A], 1)], c="red", s=50, zorder=5)
    ax.set_ylabel("|C_t| (log)"); ax.set_xlabel("epoch")
    ax.set_title(f"sharpness map (50k); |C_TA|={Ct[T_A]}"); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    Ns = [r["N"] for r in dens_rows]
    bq = [r["best_q_geo_deg"] for r in dens_rows]
    fl = [r["pool_floor_deg"] for r in dens_rows]
    ax.plot(Ns, bq, "o-", label="best survivor q_geo")
    ax.plot(Ns, fl, "s--", color="gray", label="pool floor (no filter)")
    ax.axhspan(1, 3, color="green", alpha=0.12, label="s084 budget 1-3°")
    ax.set_xscale("log"); ax.set_xlabel("pool N"); ax.set_ylabel("q_geo to truth (deg)")
    ax.set_title("Layer A: anchor accuracy vs density"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for om_level in sorted({s["om_level"] for s in b_summary}):
        xs = [s["q_pert_deg"] for s in b_summary if s["om_level"] == om_level]
        ys = [s["rho_surr_med"] for s in b_summary if s["om_level"] == om_level]
        ax.plot(xs, ys, "o-", label=om_level)
    ax.axhline(4, color="orange", ls="--", lw=1, label="Band B edge (ρ=4)")
    ax.axhline(2, color="green", ls=":", lw=1, label="Band A edge (ρ=2)")
    ax.set_xlabel("anchor q-perturbation (deg)"); ax.set_ylabel("ρ_surr (median over axes)")
    ax.set_title("Layer B: does polish converge?"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"seed{seed:03d}_anchor_gate.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    return p


def run_seed(seed, pools, n_workers, q_pert_levels, omega_levels, out_dir):
    print(f"\n{'='*64}\n=== seed {seed} ===\n{'='*64}")
    d = traj_load.load_truth(seed)
    wdps = float(d["omega_mag_dps"])
    mag = d["mag_hifi"]

    print("Layer A.1 — sharpness map (50k, Pool)...")
    Ct, T_A, sharp_wall = sharpness_map(d, n_workers)
    mag_pct = float((mag[T_A] - np.nanmin(mag)) / (np.nanmax(mag) - np.nanmin(mag)) * 100)
    print(f"  T_A={T_A}  |C_TA|={Ct[T_A]}  mag={mag[T_A]:.2f} ({mag_pct:.0f}% of range)  "
          f"wall={sharp_wall:.1f}s")

    print("Layer A.2 — density scan at T_A...")
    dens_rows = density_scan(d, T_A, pools)
    for r in dens_rows:
        print(f"  N={r['N']:>7d}  |C_a|={r['n_surv']:>6d} ({r['surv_rate_pct']:.2f}%)  "
              f"best q_geo={r['best_q_geo_deg']:.2f}°  (pool floor {r['pool_floor_deg']:.2f}°)")
    headline = next(r["best_q_geo_deg"] for r in dens_rows if r["N"] == HEADLINE_N)
    print(f"  HEADLINE anchor accuracy @ {HEADLINE_N}: {headline:.2f}°")

    print("Layer B — polish ladder (build hifi context)...")
    ctx = build_context(seed=seed)
    b_rows, t_a_seconds, polish_wall, hifi_wall, rho_hifi_check = layer_b(
        seed, T_A, ctx, q_pert_levels, omega_levels, n_workers)
    b_summary = aggregate_b(b_rows)
    print(f"  t_a_seconds={t_a_seconds:.1f}  ({wdps*t_a_seconds:.0f}° truth rotation to anchor)")
    print(f"  {'q_pert':>6} {'om_lvl':>14} {'q0_seed_err':>11} {'ρsurr_med':>10} "
          f"{'ρsurr_min':>10} {'conv':>5} {'A∪B/axes':>9}")
    for s in b_summary:
        print(f"  {s['q_pert_deg']:>5.1f}° {s['om_level']:>14} "
              f"{s['q0_seed_err_med']:>10.1f}° {s['rho_surr_med']:>10.3f} "
              f"{s['rho_surr_min']:>10.3f} {s['n_converged']:>3}/{s['n_axes']} "
              f"{s['n_band_AB']:>4}/{s['n_axes']}")
    if rho_hifi_check:
        print(f"  hi-fi spot-check (best ρ_surr cand): ρ_surr={rho_hifi_check['rho_surr']:.3f} "
              f"({rho_hifi_check['band_surr']}) → ρ_hifi={rho_hifi_check['rho_hifi']:.3f} "
              f"({rho_hifi_check['band_hifi']})  q0_err={rho_hifi_check['q0_err_deg']:.2f}°")
    print(f"  polish wall={polish_wall:.1f}s  hifi wall={hifi_wall:.1f}s")

    png = plot_seed(seed, Ct, T_A, mag, dens_rows, b_summary, out_dir)
    np.savez(out_dir / f"seed{seed:03d}_data.npz",
             seed=np.int64(seed), Ct=Ct, T_A=np.int64(T_A),
             mag=mag, t_a_seconds=np.float64(t_a_seconds))
    print(f"  Saved: {png}")
    print(f"  Saved: {out_dir / f'seed{seed:03d}_data.npz'}")

    return {
        "seed": int(seed), "omega_mag_dps": wdps,
        "T_A": int(T_A), "Ct_TA": int(Ct[T_A]),
        "anchor_mag": float(mag[T_A]), "anchor_mag_pct": mag_pct,
        "t_a_seconds": t_a_seconds,
        "truth_rotation_to_anchor_deg": float(wdps * t_a_seconds),
        "sharp_wall_s": sharp_wall,
        "density_scan": dens_rows,
        "headline_anchor_q_geo_deg": float(headline),
        "layer_b_summary": b_summary,
        "layer_b_raw": b_rows,
        "layer_b_hifi_check": rho_hifi_check,
        "polish_wall_s": polish_wall, "hifi_wall_s": hifi_wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-workers", type=int, default=24)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (SURVEY / "results" / "s085")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds_cfg = [(119, Q_PERT_FAST[:1], OMEGA_LEVELS[:1])]
        densities = [100_000, 400_000]
    else:
        seeds_cfg = [(s, Q_PERT_FAST, OMEGA_LEVELS) for s in FAST_SEEDS]
        seeds_cfg.append((SLOW_SEED, Q_PERT_SLOW, OMEGA_LEVELS[1:]))  # slow: realistic ω only
        densities = DENSITIES

    print("pre-sampling Haar pools (shared across seeds)...")
    t0 = time.time()
    pools = {N: sample_so3_pool(N, sample_seed=RNG_SEED) for N in densities}
    print(f"  {len(pools)} pools in {time.time()-t0:.1f}s")

    all_seeds = []
    t_start = time.time()
    for seed, qlev, omlev in seeds_cfg:
        all_seeds.append(run_seed(seed, pools, args.n_workers, qlev, omlev, out_dir))
    total_wall = time.time() - t_start

    summary = {
        "experiment": "s085_anchor_accuracy_gate",
        "tol_mag": TOL_MAG, "sp_deg": SP_DEG, "ad_deg": AD_DEG,
        "sharp_pool_n": SHARP_POOL_N, "headline_N": HEADLINE_N,
        "densities": densities, "n_axes": N_AXES,
        "omega_levels": [{"label": l, "dir_deg": dd, "mag_pct": mm} for (l, dd, mm) in OMEGA_LEVELS],
        "total_wall_s": total_wall,
        "seeds": all_seeds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*64}\nTOTAL wall: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
