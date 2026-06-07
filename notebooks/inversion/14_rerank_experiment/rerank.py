#!/usr/bin/env python3
"""
Rerank m103 candidate pools using alternative cost functions.

Goal: find a non-oracle cost c(q0, w) that, when used to rank the 26
checkpointed candidates per seed, surfaces the lowest-w0_err candidates
to the top. Evaluation uses oracle w0_ref_errs but ONLY as a judge — the
costs themselves see only (q0, w, observed_lc, geometry).

Outputs per-seed + aggregate JSON. No pipeline re-runs; checkpoints only.
"""
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

# Python 3.14 defaults to forkserver which requires pickleable init funcs.
# fork is fine here: we're CPU-bound, no threads, no GPU.
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass  # already set

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/home/girish/surrogate_model")
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude
from surrogate_model.surrogate import SurrogateModel

from notebooks.inversion.lib.traj_source import canonical_observed_lc

# -----------------------------------------------------------------------------
# Cohort: 22 seeds that reached m126 (excludes geo_timeout 35, 69 + crash 42)
# -----------------------------------------------------------------------------
COHORT = [6, 7, 8, 11, 16, 17, 34, 45, 47, 48, 51, 57, 59, 64, 67,
          71, 78, 79, 84, 89, 91, 99]

DIAG = PROJECT_ROOT / "data/results/inversion_diagnostics"
M048_TRAJ_DIR = DIAG / "m048_trajectories/per_trajectory"
M103_DIR = DIAG / "m103_hybrid_m048"
OUT_DIR = DIAG / "rerank_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed articulation (matches m103 CTX in experiment_setup.py)
PANEL_DEG = 0.0
DISH_DEG = 15.0


# -----------------------------------------------------------------------------
def load_seed_data(seed):
    """Return (candidates, traj_geometry, observed_lc, inertia)."""
    ckpt = np.load(M103_DIR / f"seed_{seed:03d}/geo_ckpt.npz")
    traj = np.load(M048_TRAJ_DIR / f"traj_seed{seed:03d}.npz")
    shared = np.load(DIAG / "m048_trajectories/m048_trajectories.npz",
                     allow_pickle=True)

    candidates = {
        "q0": ckpt["q0_refs"].copy(),            # (26, 4) wxyz
        "w0": ckpt["w0_refs"].copy(),            # (26, 3) rad/s
        "geo_cost": ckpt["geo_costs"].copy(),    # baseline ranker
        "w0_ref_err": ckpt["w0_ref_errs"].copy(),   # oracle (eval only)
        "q0_ref_err": ckpt["q0_ref_errs"].copy(),   # oracle (eval only)
    }
    geom = {
        "obs_times": traj["observation_times"].astype(np.float64).copy(),
        "sun_pos": traj["sun_pos"].astype(np.float64).copy(),
        "obs_pos": traj["obs_pos"].astype(np.float64).copy(),
        "sat_pos": traj["sat_pos"].astype(np.float64).copy(),
        "obs_dist": traj["obs_dist"].astype(np.float64).copy(),
    }
    observed = canonical_observed_lc(traj["mag_hifi"].astype(np.float64))
    inertia = shared["inertia_tensor"].astype(np.float64).copy()
    return candidates, geom, observed, inertia


def predict_lc(q0, w0, geom, inertia, surrogate):
    """Propagate (q0, w0) and return surrogate-predicted LC (500,)."""
    quats, _ = propagate_attitude(q0, w0, geom["obs_times"],
                                  "tumbling", inertia)
    # Rotate inertial vectors into body frame per epoch.
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()  # (N, 3, 3)
    sv = geom["sun_pos"] - geom["sat_pos"]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = geom["obs_pos"] - geom["sat_pos"]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum("nij,nj->ni", R, sv)
    k2 = np.einsum("nij,nj->ni", R, ov)
    n_ep = len(quats)
    lc = surrogate.predict_magnitude(
        k1, k2,
        panel_deg=np.full(n_ep, PANEL_DEG),
        dish_deg=np.full(n_ep, DISH_DEG),
        observer_distance_km=geom["obs_dist"])
    return lc, k1, k2


# -----------------------------------------------------------------------------
# Cost function catalog. Each takes (lc_pred, observed, k1, k2, ...) and
# returns a scalar. Lower = better.
# -----------------------------------------------------------------------------
def cost_surr_mse(lc_pred, observed, **_):
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    return float(np.mean((lc_pred[finite] - observed[finite]) ** 2))


def cost_surr_mae(lc_pred, observed, **_):
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    return float(np.mean(np.abs(lc_pred[finite] - observed[finite])))


def cost_peak_count(lc_pred, observed, obs_peaks=None, window=3, **_):
    """m103 stage-2b peak-match cost, but on surrogate LC.
    Returns n_observed_peaks - n_matched (lower = better)."""
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    cand_peaks, _ = find_peaks(-lc_pred, distance=3, prominence=0.2)
    cps = set(int(p) for p in cand_peaks)
    nm = sum(1 for op in obs_peaks
             if any((op + o) in cps for o in range(-window, window + 1)))
    return float(len(obs_peaks) - nm)


def cost_bright_mse(lc_pred, observed, bright_mask=None, **_):
    """MSE restricted to observed bright epochs (mag < 10)."""
    finite = np.isfinite(lc_pred)
    m = finite & bright_mask
    if m.sum() < 5:
        return np.inf
    return float(np.mean((lc_pred[m] - observed[m]) ** 2))


def cost_peak_mse(lc_pred, observed, obs_peaks=None, **_):
    """MSE restricted to observed peak epochs ± 2."""
    finite = np.isfinite(lc_pred)
    N = len(observed)
    m = np.zeros(N, dtype=bool)
    for op in obs_peaks:
        lo, hi = max(0, op - 2), min(N, op + 3)
        m[lo:hi] = True
    m &= finite
    if m.sum() < 5:
        return np.inf
    return float(np.mean((lc_pred[m] - observed[m]) ** 2))


def cost_detrended_mse(lc_pred, observed, **_):
    """MSE after subtracting per-LC means — removes absolute-mag offset."""
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    p = lc_pred[finite] - np.mean(lc_pred[finite])
    o = observed[finite] - np.mean(observed[finite])
    return float(np.mean((p - o) ** 2))


def cost_derivative_mse(lc_pred, observed, **_):
    """MSE of first derivatives — invariant to absolute magnitude, rewards
    matching tumbling rate / peak timing."""
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    dp = np.diff(lc_pred)
    do = np.diff(observed)
    m = np.isfinite(dp)
    return float(np.mean((dp[m] - do[m]) ** 2))


def cost_peak_timing_rms(lc_pred, observed, obs_peaks=None, **_):
    """For each observed peak, distance to nearest predicted peak (in epochs).
    Ignores peak magnitudes — purely about WHERE they land in time.
    Returns sqrt(mean d²) — lower = better ω-period match."""
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    cand_peaks, _ = find_peaks(-lc_pred, distance=3, prominence=0.2)
    if len(cand_peaks) == 0:
        return float(len(observed))  # no peaks at all = bad
    d2 = []
    for op in obs_peaks:
        dist = np.min(np.abs(cand_peaks - op))
        d2.append(dist * dist)
    return float(np.sqrt(np.mean(d2)))


def cost_peak_xcorr(lc_pred, observed, obs_peaks=None, max_lag=30, **_):
    """Cross-correlation of peak trains. Max xcorr over lags in [-max_lag, max_lag].
    Returns -max_xcorr (so lower = better)."""
    finite = np.isfinite(lc_pred)
    if finite.sum() < 50:
        return np.inf
    cand_peaks, _ = find_peaks(-lc_pred, distance=3, prominence=0.2)
    if len(cand_peaks) == 0:
        return 0.0
    N = len(observed)
    a = np.zeros(N); a[obs_peaks] = 1
    b = np.zeros(N); b[cand_peaks] = 1
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            v = np.sum(a[:N - lag] * b[lag:])
        else:
            v = np.sum(a[-lag:] * b[:N + lag])
        if v > best:
            best = v
    return float(-best)


sys.path.insert(0, str(Path(__file__).resolve().parent))
from costs_extras import EXTRAS

COST_FUNCS = {
    "surr_mse":          cost_surr_mse,
    "surr_mae":          cost_surr_mae,
    "surr_peak_cnt":     cost_peak_count,
    "surr_bright_mse":   cost_bright_mse,
    "surr_peak_mse":     cost_peak_mse,
    "surr_detrend_mse":  cost_detrended_mse,
    "surr_deriv_mse":    cost_derivative_mse,
    "surr_peak_time":    cost_peak_timing_rms,
    "surr_peak_xcorr":   cost_peak_xcorr,
    **EXTRAS,
}


# -----------------------------------------------------------------------------
def q0_marginalized_lc_mse(q0_ref, w0, geom, inertia, surrogate, observed,
                           n_perturbations=20, sigma_deg=15.0, rng=None):
    """For fixed ω, try many q0 perturbations and return the best LC MSE.
    This measures 'how well can this ω possibly fit the LC' — ω-isolating."""
    if rng is None:
        rng = np.random.default_rng(42)
    # Also include twin ±X rotation (180° around X) — legitimate IS-901 twin.
    q_twin_x = np.array([0.0, 1.0, 0.0, 0.0])
    best_mse = np.inf
    best_bright_mse = np.inf
    bright = observed < 10.0
    q0s = [q0_ref.copy()]
    # Twin:
    # quaternion product q_twin * q0_ref (scalar-first, Hamilton)
    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    q0s.append(qmul(q_twin_x, q0_ref))
    # Random perturbations:
    for _ in range(n_perturbations - 2):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = np.deg2rad(rng.normal(0, sigma_deg))
        dq = np.array([np.cos(ang / 2),
                       *(np.sin(ang / 2) * axis)])
        q0s.append(qmul(dq, q0_ref))

    for q0 in q0s:
        try:
            lc, _, _ = predict_lc(q0, w0, geom, inertia, surrogate)
            finite = np.isfinite(lc)
            if finite.sum() < 50:
                continue
            mse = np.mean((lc[finite] - observed[finite]) ** 2)
            if mse < best_mse:
                best_mse = float(mse)
            m = finite & bright
            if m.sum() >= 5:
                bmse = np.mean((lc[m] - observed[m]) ** 2)
                if bmse < best_bright_mse:
                    best_bright_mse = float(bmse)
        except Exception:
            continue
    return best_mse, best_bright_mse


_WORKER_STATE = {}  # per-process cache


def _init_worker(weights_path=None):
    # fork → inherits the parent's already-loaded surrogate, nothing to do.
    # Kept as a hook in case we switch back to forkserver later.
    pass


_SHARED_SURR = [None]


def _worker_score_cand(args):
    (i, q0, w0, geom, inertia, observed, obs_peaks_list, bright_mask,
     n_pert, seed_rng_key, enable_q0_marginal, cost_names) = args
    surrogate = _SHARED_SURR[0]
    result = {"i": i, "costs": {}, "lc": None}
    try:
        lc, k1, k2 = predict_lc(q0, w0, geom, inertia, surrogate)
        result["lc"] = lc.tolist()
        for name in cost_names:
            fn = COST_FUNCS[name]
            result["costs"][name] = fn(
                lc, observed, obs_peaks=obs_peaks_list,
                bright_mask=bright_mask, k1=k1, k2=k2)
        if enable_q0_marginal:
            rng = np.random.default_rng(seed_rng_key + i)
            mse_m, bmse_m = q0_marginalized_lc_mse(
                q0, w0, geom, inertia, surrogate, observed,
                n_perturbations=n_pert, rng=rng)
            result["costs"]["surr_q0marg_mse"] = mse_m
            result["costs"]["surr_q0marg_bright_mse"] = bmse_m
    except Exception as e:
        result["error"] = repr(e)
    return result


def score_seed(seed, surrogate, verbose=True,
               enable_q0_marginal=True, n_pert=20, n_workers=8):
    from multiprocessing import Pool
    t0 = time.time()
    # Cache surrogate for fork'd workers
    _SHARED_SURR[0] = surrogate
    cands, geom, observed, inertia = load_seed_data(seed)
    obs_peaks, _ = find_peaks(-observed, distance=3, prominence=0.2)
    bright_mask = observed < 10.0
    n = len(cands["q0"])

    lc_matrix = np.full((n, len(observed)), np.nan)
    cost_names = list(COST_FUNCS.keys())
    cost_matrix = {k: np.full(n, np.inf) for k in cost_names}
    if enable_q0_marginal:
        cost_matrix["surr_q0marg_mse"] = np.full(n, np.inf)
        cost_matrix["surr_q0marg_bright_mse"] = np.full(n, np.inf)

    args_list = [
        (i, cands["q0"][i].copy(), cands["w0"][i].copy(),
         geom, inertia, observed, list(obs_peaks), bright_mask,
         n_pert, 1000 * seed, enable_q0_marginal, cost_names)
        for i in range(n)
    ]

    with Pool(n_workers, initializer=_init_worker) as pool:
        results = pool.map(_worker_score_cand, args_list)

    for r in results:
        i = r["i"]
        if r.get("lc") is not None:
            lc_matrix[i] = np.array(r["lc"])
        for name, val in r["costs"].items():
            cost_matrix[name][i] = val

    # baseline from checkpoint
    cost_matrix["geo_cost"] = cands["geo_cost"].copy()

    out = {
        "seed": seed,
        "n_candidates": n,
        "w0_ref_err": cands["w0_ref_err"].tolist(),
        "q0_ref_err": cands["q0_ref_err"].tolist(),
        "costs": {k: v.tolist() for k, v in cost_matrix.items()},
    }
    wall = time.time() - t0
    if verbose:
        print(f"  seed {seed}: {n} cands scored in {wall:.1f}s "
              f"(pool_min_w_err={cands['w0_ref_err'].min():.1f}°)")
    return out, lc_matrix


# -----------------------------------------------------------------------------
def evaluate_ranking(seed_scores, cost_name):
    """For each seed, sort by cost and report where truth-closest lands."""
    metrics = []
    for r in seed_scores:
        costs = np.array(r["costs"][cost_name])
        w_errs = np.array(r["w0_ref_err"])
        # Rank the best (lowest-w-err) candidate under this cost
        order = np.argsort(costs)
        inv_rank = np.empty_like(order)
        inv_rank[order] = np.arange(len(order))
        true_best_idx = int(np.argmin(w_errs))
        rank_of_best = int(inv_rank[true_best_idx])
        # Min w-err within top-3 under this cost
        topK_w_err = {
            "top1": float(w_errs[order[0]]),
            "top3": float(w_errs[order[:3]].min()),
            "top5": float(w_errs[order[:5]].min()),
        }
        # Spearman between cost and w_err (ideally positive: low cost → low err)
        mask = np.isfinite(costs)
        if mask.sum() >= 3:
            rho, _ = spearmanr(costs[mask], w_errs[mask])
        else:
            rho = np.nan
        metrics.append({
            "seed": r["seed"],
            "rank_of_best_omega": rank_of_best,
            "top1_w_err": topK_w_err["top1"],
            "top3_w_err": topK_w_err["top3"],
            "top5_w_err": topK_w_err["top5"],
            "spearman": float(rho) if rho == rho else None,
            "pool_min_w_err": float(w_errs.min()),
        })
    return metrics


def aggregate(metrics, pool_min_thresh_deg=20.0):
    """Aggregate per-seed metrics for one cost function."""
    ranks = [m["rank_of_best_omega"] for m in metrics]
    top3s = [m["top3_w_err"] for m in metrics]
    spear = [m["spearman"] for m in metrics if m["spearman"] is not None]
    # "rescuable" = seeds where pool has w_err < 20° (truth-adjacent reachable)
    rescuable = [m for m in metrics if m["pool_min_w_err"] < pool_min_thresh_deg]
    n_top3_hit = sum(1 for m in rescuable if m["top3_w_err"] < pool_min_thresh_deg)
    return {
        "mean_rank_of_best": float(np.mean(ranks)),
        "median_rank_of_best": float(np.median(ranks)),
        "mean_top3_w_err": float(np.mean(top3s)),
        "median_top3_w_err": float(np.median(top3s)),
        "mean_spearman": float(np.mean(spear)) if spear else None,
        "n_rescuable": len(rescuable),
        "n_top3_hit_among_rescuable": n_top3_hit,
    }


# -----------------------------------------------------------------------------
def main():
    print(f"Loading surrogate...")
    surrogate = SurrogateModel.load_default()
    print(f"Cohort: {len(COHORT)} seeds")

    all_scores = []
    t_global = time.time()
    for seed in COHORT:
        per_seed_json = OUT_DIR / f"seed_{seed:03d}_scores.json"
        if per_seed_json.exists():
            # Resume: load instead of recompute.
            all_scores.append(json.load(open(per_seed_json)))
            print(f"  seed {seed}: skipping (already scored)")
            continue
        out, lc_matrix = score_seed(seed, surrogate, verbose=True)
        all_scores.append(out)
        np.savez(OUT_DIR / f"seed_{seed:03d}_lcmatrix.npz",
                 lc_matrix=lc_matrix,
                 w0_ref_err=np.array(out["w0_ref_err"]),
                 q0_ref_err=np.array(out["q0_ref_err"]))
        with open(per_seed_json, "w") as f:
            json.dump(out, f)
        print(f"    saved {per_seed_json.name}", flush=True)

    # --- Build composite rankings (rank-sum across base costs) ---
    # Rank composites avoid scale-mismatch across heterogeneous costs.
    def add_composite(name, component_keys):
        for s in all_scores:
            n = s["n_candidates"]
            total_rank = np.zeros(n)
            for ck in component_keys:
                c = np.array(s["costs"][ck])
                # infinite costs → worst rank (n-1)
                finite = np.isfinite(c)
                rk = np.empty(n, dtype=float)
                rk[finite] = np.argsort(np.argsort(c[finite]))
                rk[~finite] = finite.sum() if finite.sum() < n else n - 1
                total_rank += rk
            s["costs"][name] = total_rank.tolist()

    add_composite("rank_sum_surr_geo",
                  ["surr_mse", "geo_cost"])
    add_composite("rank_sum_bright_geo",
                  ["surr_bright_mse", "geo_cost"])
    add_composite("rank_sum_peakmse_geo",
                  ["surr_peak_mse", "geo_cost"])
    add_composite("rank_sum_deriv_geo",
                  ["surr_deriv_mse", "geo_cost"])
    add_composite("rank_sum_all_surr",
                  ["surr_mse", "surr_bright_mse", "surr_peak_mse",
                   "surr_deriv_mse"])
    add_composite("rank_sum_everything",
                  ["surr_mse", "surr_bright_mse", "surr_peak_mse",
                   "surr_deriv_mse", "surr_peak_time", "geo_cost"])

    add_composite("rank_sum_q0marg_geo",
                  ["surr_q0marg_mse", "geo_cost"])
    add_composite("rank_sum_q0marg_bright_geo",
                  ["surr_q0marg_bright_mse", "geo_cost"])

    # Evaluate each cost
    all_evals = {}
    base = list(COST_FUNCS.keys()) + ["geo_cost",
                                      "surr_q0marg_mse",
                                      "surr_q0marg_bright_mse"]
    composites = [
        "rank_sum_surr_geo", "rank_sum_bright_geo", "rank_sum_peakmse_geo",
        "rank_sum_deriv_geo", "rank_sum_all_surr", "rank_sum_everything",
        "rank_sum_q0marg_geo", "rank_sum_q0marg_bright_geo",
    ]
    cost_names = base + composites
    for cn in cost_names:
        metrics = evaluate_ranking(all_scores, cn)
        agg = aggregate(metrics)
        all_evals[cn] = {"per_seed": metrics, "aggregate": agg}

    # Save everything
    out_json = {
        "cohort": COHORT,
        "pool_min_thresh_deg": 20.0,
        "cost_funcs": cost_names,
        "per_seed_scores": all_scores,
        "evaluations": all_evals,
    }
    with open(OUT_DIR / "rerank_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Summary table
    print(f"\n\n=== Aggregate comparison ({time.time()-t_global:.1f}s total) ===")
    print(f"{'cost':<18} | {'rank_best':>10} | {'top3_w_err':>11} | "
          f"{'spearman':>9} | {'top3_hit':>10}")
    print("-" * 72)
    for cn in cost_names:
        agg = all_evals[cn]["aggregate"]
        print(f"{cn:<18} | "
              f"{agg['mean_rank_of_best']:>10.2f} | "
              f"{agg['mean_top3_w_err']:>10.2f}° | "
              f"{(agg['mean_spearman'] or 0):>9.3f} | "
              f"{agg['n_top3_hit_among_rescuable']:>3}/{agg['n_rescuable']:<6}")

    print(f"\nSaved: {OUT_DIR}/rerank_results.json")
    print(f"        {OUT_DIR}/seed_XXX_lcmatrix.npz (per seed)")


if __name__ == "__main__":
    main()
