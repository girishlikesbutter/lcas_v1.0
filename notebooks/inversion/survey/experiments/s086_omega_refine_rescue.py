"""s086 — does a 1-D |ω| refine before the joint polish rescue seed 119 at L1?

s085 found the fast seed 119 FAILS the realistic-blind-ω level (L1: ω-dir off
1.5°, |ω| off 0.5%): only 1/3 axes reach Band A∪B, median ρ≈25 (D). Cause: that
small ω error is amplified into a 12-20° q0_seed_err by back-propagating through
738° of accumulated rotation, beyond the polish grab radius. Seeds 108/100/116
passed L1. So the open question (PROGRESS revised stage order
"anchor q → grid ω-dir → 1-D refine |ω| → joint polish"): does refining |ω| FIRST
shrink the seed error enough to rescue 119?

The decisive uncertainty is decomposition: of the 12-20° q0_seed_err, how much
comes from the 0.5% |ω|-MAGNITUDE error (which the refine fixes) vs the 1.5°
ω-DIRECTION error (which it does NOT)?

LAYER 1 — analytical decomposition (back_propagate only, ~instant, NO polish):
  for each (q_pert, axis) measure q0_seed_err under 4 ω-seed treatments:
    (dir 1.5, mag 0.5)  baseline   = s085 L1
    (dir 1.5, mag 0.0)  oracle_mag = ceiling of a perfect |ω| refine
    (dir 0.0, mag 0.5)  oracle_dir
    (dir 0.0, mag 0.0)  = L0 (≈ q_pert)
  This predicts whether refining |ω| CAN help before we spend any polish.

LAYER 2 — the real test (polish + hi-fi):
  three treatments per (q_pert, axis):
    baseline   : seed ω at anchor = (dir 1.5, mag +0.5%)        [reproduces s085 L1]
    oracle_mag : seed ω at anchor = (dir 1.5, mag exact)        [upper bound]
    refined    : seed ω at anchor = (dir 1.5, |ω| from a 1-D    [the production fix]
                 full-LC surrogate-MSE scan over the ls_bracket, s084 method)
  back_propagate -> lm_polish_jacobi (s064) -> hi-fi classify the best converged
  candidate per (seed, treatment).

PREDICTION (stated before running): 0.5% over 738° ≈ 3.7° of pure phase slip plus
nonlinear winding. I expect the refine to lower q0_seed_err substantially but I am
NOT confident it clears the grab radius given the residual 1.5° dir error. If
oracle_mag ALSO fails -> direction dominates, conclusion flips to "tighten the
ω-dir grid for fast seeds (1.5° is too coarse)".

Not oracle in the production sense: the refine scans |ω| blindly over the
ls_bracket; truth is only the comparison target + the perturbation centre (the
s005/s064-gate2 controlled-perturbation method).

Usage:
    python experiments/s086_omega_refine_rescue.py            # 119 + 100
    python experiments/s086_omega_refine_rescue.py --smoke    # 119, q_pert=3, 1 axis
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

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.forward import propagate_to_body_frame  # noqa: E402
from lib.surrogate_eval import predict as surrogate_predict  # noqa: E402
from lib.lc_features import ls_bracket  # noqa: E402
from lib.hifi_render import build_context, rho_band  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from experiments.s059_pilot import back_propagate  # noqa: E402
from experiments.s064_jacobi_polish import (  # noqa: E402
    lm_polish_jacobi, hifi_classify, perturb_q, perturb_omega, quat_geodesic_deg,
)

SEEDS = [119, 100]            # 119 = the L1 failure; 100 = marginal-converge confirm
Q_PERT_LEVELS = [2.0, 3.0]    # bracket the s085 anchor delivery (119: 3.30°@400k)
N_AXES = 3
OM_DIR_DEG = 1.5             # L1 direction error (s082: grid delivers 0.94-2.69°)
OM_MAG_PCT = 0.5             # L1 magnitude error
# 1-D refine scan over the full ls_bracket (s084 P1: truth-|ω| unique global min)
N_COARSE = 150
N_FINE = 121
FINE_HALFWIDTH_PCT = 6.0
# rng scheme MATCHES s085 layer_b exactly so qa_p / om_p-direction are identical
def _rng_seed(seed, q_pert, ax, om_dir):
    return 7000 + seed * 100 + int(q_pert * 10) + ax + (1 if om_dir else 0) * 50


# ----------------------- worker globals -----------------------

_CTX = None
_TARGET = None
_BRACKET = None   # dict seed -> (lo, hi)


def _winit(ctx_by_seed, target_by_seed, bracket_by_seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _CTX, _TARGET, _BRACKET
    _CTX = ctx_by_seed
    _TARGET = target_by_seed
    _BRACKET = bracket_by_seed


def _score_full_lc(q0, om0, ctx, target):
    """Full-LC surrogate ρ for a (q0, ω0) at t=0 — same surface the polish uses."""
    try:
        k1, k2, _ = propagate_to_body_frame(
            q0_wxyz=q0, omega0_rad=om0,
            observation_times=ctx["observation_times"],
            sun_pos=ctx["sun_pos"], obs_pos=ctx["obs_pos"], sat_pos=ctx["sat_pos"],
            inertia_tensor=ctx["inertia_tensor"], mode="tumbling",
        )
        pred = surrogate_predict(k1, k2, ctx["obs_dist"])
        r = pred - target
        m = np.isfinite(r)
        if not m.any():
            return float("inf")
        return float(np.sqrt(np.mean(r[m] ** 2)) / 0.05)
    except Exception:
        return float("inf")


def refine_omega_1d(qa_p, om_dir_unit, t_a_seconds, ctx, target, lo, hi):
    """1-D |ω| refine: hold anchor-q and (wrong) ω-direction; scan |ω| over the
    full ls_bracket, score full-LC surrogate ρ on the back-propagated state, pick
    the global min, then a fine local scan. Returns (w_best, q0_best, om0_best,
    rho_best, n_eval)."""
    inertia = ctx["inertia_tensor"]
    n_eval = 0

    def eval_w(w):
        nonlocal n_eval
        n_eval += 1
        om_a = om_dir_unit * w
        q0_c, om0_c = back_propagate(qa_p, om_a, t_a_seconds, inertia)
        return _score_full_lc(q0_c, om0_c, ctx, target), q0_c, om0_c

    # coarse full-bracket
    w_coarse = np.geomspace(lo, hi, N_COARSE)
    best = (float("inf"), None, None, None)
    for w in w_coarse:
        rho, q0_c, om0_c = eval_w(w)
        if rho < best[0]:
            best = (rho, q0_c, om0_c, float(w))
    # fine local scan around coarse min
    w0 = best[3]
    w_fine = np.linspace(w0 * (1 - FINE_HALFWIDTH_PCT / 100),
                         w0 * (1 + FINE_HALFWIDTH_PCT / 100), N_FINE)
    for w in w_fine:
        rho, q0_c, om0_c = eval_w(w)
        if rho < best[0]:
            best = (rho, q0_c, om0_c, float(w))
    rho_best, q0_best, om0_best, w_best = best
    return w_best, q0_best, om0_best, rho_best, n_eval


def _worker(job):
    (seed, q_pert, ax, treatment) = job
    ctx = _CTX[seed]
    target = _TARGET[seed]
    inertia = ctx["inertia_tensor"]
    q0_truth = np.asarray(ctx["q0_truth"], float)
    om0_truth = np.asarray(ctx["omega0_truth_rad"], float)
    om_truth_mag = float(np.linalg.norm(om0_truth))
    T_A = ctx["T_A"]
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])

    # truth state at anchor (matches s085)
    quats_t, omegas_t = propagate_attitude(
        q0=q0_truth, omega0=om0_truth, times=ctx["observation_times"],
        mode="tumbling", inertia_tensor=inertia,
    )
    q_a_truth = np.asarray(quats_t[T_A], float)
    om_a_truth = np.asarray(omegas_t[T_A], float)

    # reproduce s085 L1 perturbations with the SAME rng draws
    rng = np.random.default_rng(_rng_seed(seed, q_pert, ax, OM_DIR_DEG))
    qa_p = perturb_q(q_a_truth, q_pert, rng)
    om_p = perturb_omega(om_a_truth, OM_DIR_DEG, OM_MAG_PCT, rng)  # (dir 1.5, mag +0.5%)
    om_dir_unit = om_p / np.linalg.norm(om_p)

    refine_info = None
    if treatment == "baseline":
        om_a_seed = om_p
    elif treatment == "oracle_mag":
        om_a_seed = om_dir_unit * om_truth_mag           # perfect |ω|, wrong dir 1.5°
    elif treatment == "refined":
        lo, hi = _BRACKET[seed]
        w_best, q0_seed, om0_seed, rho_refine, n_ev = refine_omega_1d(
            qa_p, om_dir_unit, t_a_seconds, ctx, target, lo, hi)
        refine_info = {"w_best_rad": float(w_best),
                       "w_err_pct": float((w_best - om_truth_mag) / om_truth_mag * 100),
                       "rho_refine_surr": float(rho_refine), "n_eval_refine": int(n_ev)}
        om_a_seed = None  # already back-propagated inside refine
    else:
        raise ValueError(treatment)

    if treatment != "refined":
        q0_seed, om0_seed = back_propagate(qa_p, om_a_seed, t_a_seconds, inertia)

    q0_seed_err = quat_geodesic_deg(np.asarray(q0_seed), q0_truth)

    res = lm_polish_jacobi(q0_seed, om0_seed, ctx, target,
                           label=f"s{seed}_q{q_pert}_ax{ax}_{treatment}")
    q0p = np.asarray(res["q0_pol_wxyz"]); om0p = np.asarray(res["om0_pol_rad"])
    q0_err = quat_geodesic_deg(q0p, q0_truth)
    om_mag_err = float((np.linalg.norm(om0p) - om_truth_mag) / om_truth_mag * 100)
    om_dir_err = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)), om0_truth / om_truth_mag)), 0, 1))))

    return {
        "seed": int(seed), "q_pert_deg": float(q_pert), "axis": int(ax),
        "treatment": treatment,
        "q0_seed_err_deg": float(q0_seed_err),
        "rho_pol_surr": float(res["surrogate_rho_polished"]),
        "band_surr": rho_band(res["surrogate_rho_polished"]),
        "q0_err_deg": float(q0_err), "om_mag_err_pct": om_mag_err, "om_dir_err_deg": om_dir_err,
        "n_eval": int(res["n_eval"]),
        "q0_pol_wxyz": res["q0_pol_wxyz"], "om0_pol_rad": res["om0_pol_rad"],
        "refine_info": refine_info,
    }


# ----------------------- Layer 1: analytical decomposition -----------------------

def layer1_decomposition(seed, ctx):
    """q0_seed_err under 4 ω-seed treatments — back_propagate only, no polish."""
    inertia = ctx["inertia_tensor"]
    q0_truth = np.asarray(ctx["q0_truth"], float)
    om0_truth = np.asarray(ctx["omega0_truth_rad"], float)
    T_A = ctx["T_A"]
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    quats_t, omegas_t = propagate_attitude(
        q0=q0_truth, omega0=om0_truth, times=ctx["observation_times"],
        mode="tumbling", inertia_tensor=inertia)
    q_a_truth = np.asarray(quats_t[T_A], float)
    om_a_truth = np.asarray(omegas_t[T_A], float)

    combos = [("dir1.5_mag0.5", 1.5, 0.5), ("dir1.5_mag0.0", 1.5, 0.0),
              ("dir0.0_mag0.5", 0.0, 0.5), ("dir0.0_mag0.0", 0.0, 0.0)]
    rows = {}
    for label, dd, mm in combos:
        errs = []
        for q_pert in Q_PERT_LEVELS:
            for ax in range(N_AXES):
                rng = np.random.default_rng(_rng_seed(seed, q_pert, ax, dd))
                qa_p = perturb_q(q_a_truth, q_pert, rng)
                om_p = (perturb_omega(om_a_truth, dd, mm, rng)
                        if (dd or mm) else om_a_truth)
                q0_seed, _ = back_propagate(qa_p, om_p, t_a_seconds, inertia)
                errs.append(quat_geodesic_deg(np.asarray(q0_seed), q0_truth))
        rows[label] = {"q0_seed_err_median": float(np.median(errs)),
                       "q0_seed_err_min": float(np.min(errs)),
                       "q0_seed_err_max": float(np.max(errs))}
    return rows


# ----------------------- aggregation + plot -----------------------

def aggregate(rows, seed):
    out = {}
    for treatment in ["baseline", "oracle_mag", "refined"]:
        grp = [r for r in rows if r["seed"] == seed and r["treatment"] == treatment]
        if not grp:
            continue
        bands = [r["band_surr"] for r in grp]
        out[treatment] = {
            "n": len(grp),
            "q0_seed_err_med": float(np.median([r["q0_seed_err_deg"] for r in grp])),
            "rho_surr_med": float(np.median([r["rho_pol_surr"] for r in grp])),
            "rho_surr_min": float(np.min([r["rho_pol_surr"] for r in grp])),
            "n_band_AB": sum(1 for b in bands if b in {"A", "B"}),
            "n_converged": sum(1 for r in grp if r["q0_err_deg"] < 5.0),
            "bands": bands,
            "refine_w_err_pct_med": (
                float(np.median([r["refine_info"]["w_err_pct"] for r in grp]))
                if treatment == "refined" else None),
        }
    return out


def plot_seed(seed, l1, agg, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    treatments = ["baseline", "oracle_mag", "refined"]
    colors = {"baseline": "tab:red", "oracle_mag": "tab:green", "refined": "tab:blue"}

    # Layer 1 decomposition (q0_seed_err)
    labels = list(l1.keys())
    xs = np.arange(len(labels))
    meds = [l1[k]["q0_seed_err_median"] for k in labels]
    lo = [l1[k]["q0_seed_err_median"] - l1[k]["q0_seed_err_min"] for k in labels]
    hi = [l1[k]["q0_seed_err_max"] - l1[k]["q0_seed_err_median"] for k in labels]
    ax1.bar(xs, meds, yerr=[lo, hi], capsize=4, color="tab:purple", alpha=0.7)
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, rotation=20, fontsize=8)
    ax1.set_ylabel("q0_seed_err (deg, median±range)")
    ax1.set_title(f"seed {seed} Layer 1: back-prop error decomposition")
    ax1.grid(alpha=0.3, axis="y")

    # Layer 2 polish outcome (rho_surr)
    for t in treatments:
        if t not in agg:
            continue
        ax2.scatter([t], [agg[t]["rho_surr_med"]], s=80, color=colors[t],
                    label=f"{t} (A∪B {agg[t]['n_band_AB']}/{agg[t]['n']})")
    ax2.axhline(4, color="orange", ls="--", lw=1, label="Band B edge ρ=4")
    ax2.axhline(2, color="green", ls=":", lw=1, label="Band A edge ρ=2")
    ax2.set_ylabel("ρ_surr (median over q_pert×axes)")
    ax2.set_title(f"seed {seed} Layer 2: does the refine rescue the polish?")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"seed{seed:03d}_refine_rescue.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-workers", type=int, default=24)
    args = ap.parse_args()

    out_dir = SURVEY / "results" / "s086"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[:1] if args.smoke else SEEDS
    q_levels = Q_PERT_LEVELS[-1:] if args.smoke else Q_PERT_LEVELS
    n_axes = 1 if args.smoke else N_AXES
    treatments = ["baseline", "oracle_mag", "refined"]

    # Build contexts; find T_A per seed via the cached s085 anchor (same T_A).
    import json as _json
    s085 = _json.load(open(SURVEY / "results" / "s085" / "summary.json"))
    T_A_by_seed = {s["seed"]: s["T_A"] for s in s085["seeds"]}

    ctx_by_seed, target_by_seed, bracket_by_seed = {}, {}, {}
    l1_by_seed = {}
    for seed in seeds:
        ctx = build_context(seed=seed)
        ctx["T_A"] = T_A_by_seed[seed]
        target = ctx["mag_hifi_truth"]
        br = ls_bracket(ctx["observation_times"], target, n_cells=5)
        ctx_by_seed[seed] = ctx
        target_by_seed[seed] = target
        bracket_by_seed[seed] = (float(br[0]), float(br[-1]))
        print(f"seed {seed}: T_A={ctx['T_A']}  ls_bracket=[{br[0]:.4e},{br[-1]:.4e}] "
              f"(span {br[-1]/br[0]:.0f}×)  truth|ω|={np.linalg.norm(ctx['omega0_truth_rad']):.4e}")
        # Layer 1 — analytical, instant
        l1_by_seed[seed] = layer1_decomposition(seed, ctx)
        print(f"  Layer 1 (q0_seed_err median over {len(q_levels)*n_axes if False else len(Q_PERT_LEVELS)*N_AXES} (q_pert×axis)):")
        for k, v in l1_by_seed[seed].items():
            print(f"    {k:>16}: {v['q0_seed_err_median']:6.2f}° "
                  f"(range {v['q0_seed_err_min']:.2f}-{v['q0_seed_err_max']:.2f})")

    jobs = [(seed, q, ax, t) for seed in seeds for q in q_levels
            for ax in range(n_axes) for t in treatments]
    print(f"\nLayer 2: {len(jobs)} polish jobs (Pool {args.n_workers}) ...", flush=True)

    t0 = time.time()
    rows = []
    if args.n_workers <= 1:
        _winit(ctx_by_seed, target_by_seed, bracket_by_seed)
        for j in jobs:
            rows.append(_worker(j))
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(args.n_workers, initializer=_winit,
                           initargs=(ctx_by_seed, target_by_seed, bracket_by_seed)) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                rows.append(r)
    polish_wall = time.time() - t0
    print(f"  polish+refine wall {polish_wall:.1f}s", flush=True)

    # hi-fi spot-check: best converged (ρ_surr) per (seed, treatment)
    print("\nhi-fi spot-checks (best ρ_surr per seed×treatment) ...", flush=True)
    t1 = time.time()
    hifi_checks = []
    for seed in seeds:
        ctx = ctx_by_seed[seed]; target = target_by_seed[seed]
        for t in treatments:
            grp = [r for r in rows if r["seed"] == seed and r["treatment"] == t
                   and r["rho_pol_surr"] < 4.0]
            if not grp:
                continue
            best = min(grp, key=lambda r: r["rho_pol_surr"])
            rho_h, band_h = hifi_classify(best["q0_pol_wxyz"], best["om0_pol_rad"], ctx, target)
            chk = {"seed": seed, "treatment": t, "rho_surr": best["rho_pol_surr"],
                   "band_surr": best["band_surr"], "rho_hifi": float(rho_h),
                   "band_hifi": band_h, "q0_err_deg": best["q0_err_deg"]}
            hifi_checks.append(chk)
            print(f"  seed {seed} {t:>10}: ρ_surr={best['rho_pol_surr']:.3f}({best['band_surr']}) "
                  f"→ ρ_hifi={rho_h:.3f}({band_h})  q0_err={best['q0_err_deg']:.2f}°")
    hifi_wall = time.time() - t1

    # aggregate + plots + console table
    agg_by_seed = {}
    for seed in seeds:
        agg = aggregate(rows, seed)
        agg_by_seed[seed] = agg
        png = plot_seed(seed, l1_by_seed[seed], agg, out_dir)
        print(f"\n=== seed {seed} Layer 2 summary ===")
        print(f"  {'treatment':>10} {'q0_seed_err':>11} {'rho_med':>8} {'conv':>5} {'A∪B':>5} {'refine|ω|err':>12}")
        for t in treatments:
            if t not in agg:
                continue
            a = agg[t]
            we = f"{a['refine_w_err_pct_med']:+.2f}%" if a['refine_w_err_pct_med'] is not None else "  --"
            print(f"  {t:>10} {a['q0_seed_err_med']:>10.1f}° {a['rho_surr_med']:>8.3f} "
                  f"{a['n_converged']:>3}/{a['n']} {a['n_band_AB']:>3}/{a['n']} {we:>12}")
        print(f"  Saved: {png}")

    summary = {
        "experiment": "s086_omega_refine_rescue",
        "seeds": seeds, "q_pert_levels": q_levels, "n_axes": n_axes,
        "om_dir_deg": OM_DIR_DEG, "om_mag_pct": OM_MAG_PCT,
        "refine": {"n_coarse": N_COARSE, "n_fine": N_FINE, "fine_halfwidth_pct": FINE_HALFWIDTH_PCT},
        "layer1_decomposition": {str(s): l1_by_seed[s] for s in seeds},
        "layer2_aggregate": {str(s): agg_by_seed[s] for s in seeds},
        "hifi_checks": hifi_checks,
        "rows": rows,
        "polish_wall_s": polish_wall, "hifi_wall_s": hifi_wall,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {out_dir / 'summary.json'}")
    print(f"TOTAL wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
