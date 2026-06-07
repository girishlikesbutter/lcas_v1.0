"""s059j — cloud-data ω-grid search (anchor-aware), single-seed pilot.

Production cloud-data inversion pipeline that the s059i_* validators
de-risked end-to-end. The two-stage anchor optimisation (coarse 50k pool
to pick T_A, dense 400k pool to densify only at T_A) gives a 20× speedup
on cloud setup with zero quality loss. The 9-stage pipeline:

    [1] Anchor select    coarse 50k pool, project+survive [3, 30), T_A=argmin |C_t|
    [2] Dense at anchor  400k pool, project+survive ONLY at T_A → C_a
    [3] |C_a|-cap        random subsample to ≤ CA_CAP=3000
    [4] ω-grid           Fibonacci(N_DIRS=200) × linspace(0.7, 1.3, N_MAGS=6) × |ω|_anchor
    [5] Joint score      surrogate-MSE on [T_A-W, T_A+W] (W=10) over C_a × ω_grid, Pool(24)
    [6] Top-K + cluster  top-K-by-score → canonical_batch → greedy cluster (s057h)
    [7] LM polish        TOP_K_POLISH=20 cluster reps, NO TRUTH INJECTION (surrogate v2 only)
    [8] Hi-fi gate       render iff surrogate_rho_local_polished < 4
    [9] Headline yield   Band A∪B count over polished candidates (no truth row)

PILOT regime: ω-grid centered on the ORACLE truth |ω_a_body| at T_A (matches the
s059i validators). Cohort runner will swap to the s055a pol_diam estimator.

Critical correctness rules (per `feedback_oracle_injection_taints_yield.md`):
  - NEVER inject the truth cluster into the polish set.
  - LM polish uses surrogate v2 full-LC residual, not hi-fi.
  - Hi-fi reserved for the final ρ-band classification.

Usage:
    python experiments/s059j_cloud_data_omega_grid.py --seed 28
    python experiments/s059j_cloud_data_omega_grid.py --seed 28 --n-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions, survive_at_epoch,
)
from lib.forward import propagate_to_body_frame  # noqa: E402  # noqa: F401  (imported for fork-CoW availability)
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402
from lib.surrogate_eval import get_model, predict as surrogate_predict  # noqa: E402  # noqa: F401
from lib.twin import canonical_batch  # noqa: E402  # noqa: F401  (used inside stage_cluster import)
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from experiments.s059_pilot import (  # noqa: E402
    back_propagate, wxyz_to_xyzw, xyzw_to_wxyz,
    quat_ang_deg, quat_ang_deg_batch, ang_to_axis,
    stage_cluster,
    LM_MAX_NFEV, LM_FTOL, LM_XTOL, RESIDUAL_CAP,
    SURROGATE_RHO_HIFI_GATE, TOL_MAG, SP_DEG, AD_DEG,
    CLUSTER_Q_DEG, CLUSTER_OM_DEG, CLUSTER_OM_MAG_PCT,
)
from experiments.s059e_local_window import (  # noqa: E402
    lm_polish_local,
)
from experiments.s059i_validator import (  # noqa: E402
    fibonacci_sphere, score_local_window,
)


# ----- defaults --------------------------------------------------------------

N_COARSE = 50_000
N_DENSE = 400_000
RNG_SEED = 42
EARLY_LO, EARLY_HI = 3, 30
CA_CAP = 3_000
N_DIRS = 200
N_MAGS = 6
OMEGA_BRACKET = 0.30
WINDOW_W = 10
TOP_K_FOR_CLUSTERING = 5_000
TOP_K_POLISH = 20
N_WORKERS = 24


# ----- Pool worker globals (set in parent before fork) ----------------------

_CTX_GLOBAL = None
_C_A_GLOBAL = None
_OMEGA_GRID_GLOBAL = None
_T_A_GLOBAL = None
_W_GLOBAL = None


def _worker_init():
    """Pin BLAS to 1 thread per worker; warm the surrogate cache.

    Surrogate `_MODEL` is already populated in the parent process via
    `get_model()` before fork — workers inherit via CoW, so this just
    ensures the cache is hot in case fork didn't already.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    get_model()


def _worker_score_qa_chunk(qa_indices_chunk):
    qa_indices_chunk = np.asarray(qa_indices_chunk, dtype=np.int64)
    n_om = len(_OMEGA_GRID_GLOBAL)
    out = np.empty((len(qa_indices_chunk), n_om), dtype=np.float64)
    for ii, qa_idx in enumerate(qa_indices_chunk):
        q_a = _C_A_GLOBAL[int(qa_idx)]
        for j in range(n_om):
            mse, _, _ = score_local_window(
                q_a, _OMEGA_GRID_GLOBAL[j], _T_A_GLOBAL, _W_GLOBAL, _CTX_GLOBAL,
            )
            out[ii, j] = mse
    return qa_indices_chunk, out


# ----- stages ---------------------------------------------------------------


def stage_anchor_select_coarse(ctx, *, n_coarse, edge_lo, edge_hi, rng_seed, log):
    """Coarse Haar pool, project+survive across [edge_lo, edge_hi), pick argmin |C_t|."""
    t0 = time.time()
    pool = sample_so3_pool(n_coarse, sample_seed=rng_seed)
    sun_unit, obs_unit = compute_j2000_units(
        ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"]
    )
    model = get_model()
    n_ep = edge_hi - edge_lo
    cv = np.zeros(n_ep, dtype=np.int64)
    for j, ep in enumerate(range(edge_lo, edge_hi)):
        k1, k2 = project_directions(pool["R_cache"], sun_unit[ep], obs_unit[ep])
        _, keep = survive_at_epoch(
            model, k1, k2, ctx["obs_dist"][ep], SP_DEG, AD_DEG,
            ctx["mag_hifi_truth"][ep], TOL_MAG,
        )
        cv[j] = int(keep.sum())
    T_A = int(edge_lo + np.argmin(cv))
    wall = time.time() - t0
    log(
        f"anchor select (coarse N={n_coarse}): T_A={T_A}, "
        f"|C_{{T_A}}|={int(cv[T_A - edge_lo])}, "
        f"cv range [{int(cv.min())}, {int(cv.max())}], wall {wall:.2f}s"
    )
    return {
        "T_A": T_A, "cv": cv,
        "edge_lo": edge_lo, "edge_hi": edge_hi,
        "wall_s": wall,
    }


def stage_dense_at_anchor(ctx, T_A, *, n_dense, rng_seed, log):
    """Dense Haar pool, project+survive ONLY at T_A → C_a."""
    t0 = time.time()
    pool = sample_so3_pool(n_dense, sample_seed=rng_seed)
    sun_unit, obs_unit = compute_j2000_units(
        ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"]
    )
    model = get_model()
    k1, k2 = project_directions(pool["R_cache"], sun_unit[T_A], obs_unit[T_A])
    _, keep = survive_at_epoch(
        model, k1, k2, ctx["obs_dist"][T_A], SP_DEG, AD_DEG,
        ctx["mag_hifi_truth"][T_A], TOL_MAG,
    )
    C_a = pool["q_pool_wxyz"][keep]
    wall = time.time() - t0
    n_raw = int(keep.sum())
    log(f"dense at T_A={T_A} (N={n_dense}): |C_a|={n_raw}, wall {wall:.2f}s")
    return {"C_a": C_a, "n_C_a_raw": n_raw, "wall_s": wall}


def stage_cap_subsample(C_a, *, cap, rng_seed, log):
    if len(C_a) <= cap:
        log(f"|C_a|-cap not triggered: |C_a|={len(C_a)} <= cap={cap}")
        return C_a
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(C_a), cap, replace=False)
    log(f"|C_a|-cap subsample: {len(C_a)} → {cap}")
    return C_a[idx]


def truth_qa_om_at_anchor(ctx, T_A):
    """Diagnostic: truth (q_a, ω_a_body) at T_A. Oracle-pilot grid centering."""
    quats_truth, omegas_truth = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=ctx["observation_times"], mode="tumbling",
        inertia_tensor=ctx["inertia_tensor"],
    )
    q_a_truth = np.asarray(quats_truth[T_A], dtype=np.float64)
    om_a_truth = np.asarray(omegas_truth[T_A], dtype=np.float64)
    return q_a_truth, om_a_truth


def build_omega_grid(omega_mag_anchor, *, n_dirs, n_mags, bracket):
    """Fibonacci(n_dirs) × linspace(1-b, 1+b, n_mags) × |ω|."""
    dirs = fibonacci_sphere(n_dirs)
    factors = np.linspace(1.0 - bracket, 1.0 + bracket, n_mags)
    mags = factors * omega_mag_anchor
    return (dirs[:, None, :] * mags[None, :, None]).reshape(-1, 3)


def stage_score_grid(C_a, omega_grid, T_A, W, ctx, *, n_workers, log):
    """Joint surrogate-MSE on local window for every (q_a, ω) cell. Pool(N)."""
    global _CTX_GLOBAL, _C_A_GLOBAL, _OMEGA_GRID_GLOBAL, _T_A_GLOBAL, _W_GLOBAL
    _CTX_GLOBAL = ctx
    _C_A_GLOBAL = C_a
    _OMEGA_GRID_GLOBAL = omega_grid
    _T_A_GLOBAL = T_A
    _W_GLOBAL = W

    n_a = len(C_a)
    n_om = len(omega_grid)
    log(
        f"score grid: |C_a|={n_a} × |ω_grid|={n_om} = {n_a*n_om} cells, "
        f"Pool({n_workers}) over q_a chunks, W={W}"
    )

    # Warm the surrogate in the parent so workers inherit via fork CoW.
    get_model()

    # Finer chunks improve load balance when score-per-q_a is roughly constant.
    chunks = [c.tolist() for c in np.array_split(np.arange(n_a), max(1, n_workers * 4))]
    scores = np.empty((n_a, n_om), dtype=np.float64)
    n_done = 0
    t0 = time.time()
    last_report = t0

    ctx_pool = get_context("fork")
    with ctx_pool.Pool(n_workers, initializer=_worker_init) as pool:
        for chunk_indices, chunk_out in pool.imap_unordered(
            _worker_score_qa_chunk, chunks
        ):
            scores[chunk_indices] = chunk_out
            n_done += len(chunk_indices)
            now = time.time()
            if now - last_report > 30.0:
                rate = n_done / max(1e-9, now - t0)
                eta = (n_a - n_done) / rate if rate > 0 else 0.0
                log(
                    f"  {n_done}/{n_a} q_a done ({100 * n_done / n_a:.1f}%), "
                    f"rate {rate:.1f}/s, ETA {eta/60:.1f} min"
                )
                last_report = now
    wall = time.time() - t0
    log(
        f"score grid wall: {wall:.1f}s ({wall/60:.2f} min); "
        f"min MSE={float(scores.min()):.6e}, "
        f"min ρ_local={float(np.sqrt(scores.min())/0.05):.3f}"
    )
    return scores, wall


def stage_topK_extract(C_a, omega_grid, scores, *, top_K, log):
    """Take top-K (q_a, ω) joint cells by lowest MSE."""
    n_a, n_om = scores.shape
    flat = scores.reshape(-1)
    order = np.argsort(flat)
    top_K = min(top_K, len(flat))
    top_flat = order[:top_K]
    qa_idx = (top_flat // n_om).astype(np.int64)
    om_idx = (top_flat % n_om).astype(np.int64)
    Q_top = C_a[qa_idx]
    Om_top = omega_grid[om_idx]
    mse_top = flat[top_flat]
    log(
        f"top-K extract: K={top_K}; "
        f"min MSE in top-K={float(mse_top.min()):.6e} "
        f"(ρ={float(np.sqrt(mse_top.min())/0.05):.3f}), "
        f"max MSE in top-K={float(mse_top.max()):.6e} "
        f"(ρ={float(np.sqrt(mse_top.max())/0.05):.3f})"
    )
    return Q_top, Om_top, mse_top, qa_idx, om_idx


def diagnostic_truth_idx_in_topK(Q_top, Om_top, q_a_truth, om_a_truth):
    qa_dist = quat_ang_deg_batch(Q_top, q_a_truth)
    om_dist = ang_to_axis(Om_top, om_a_truth)
    truth_idx = int(np.argmin(qa_dist + om_dist))
    return truth_idx, qa_dist, om_dist


def stage_polish_local_no_truth(
    clusters_sorted, Q_top, Om_top, T_A, W, ctx, *, top_K_polish, log
):
    """LM polish top-K cluster reps. NO truth-cluster injection."""
    target = ctx["mag_hifi_truth"]
    to_polish = list(clusters_sorted[:top_K_polish])
    log(
        f"polishing {len(to_polish)} cluster reps (NO truth injection); "
        f"local window W={W}"
    )

    polished = []
    for c in to_polish:
        rank = next(
            i for i, cc in enumerate(clusters_sorted)
            if cc["cluster_id"] == c["cluster_id"]
        ) + 1
        bm = c["best_member_idx"]
        q_a = Q_top[bm]
        om_a = Om_top[bm]

        result = lm_polish_local(q_a, om_a, T_A, W, ctx, target)
        result["cluster_rank"] = rank
        result["cluster_id"] = c["cluster_id"]
        result["score_sum"] = c["score_sum"]
        result["qa_dist_t_a_init"] = c["min_qa_dist_to_truth"]
        result["om_dist_t_a_init"] = c["min_om_dist_to_truth"]

        # Diagnostic errors vs truth (post-polish, at t=0).
        result["q0_err_polished_deg"] = quat_ang_deg(
            result["q0_pol_wxyz"], ctx["q0_truth"]
        )
        om_truth = ctx["omega0_truth_rad"]
        om_pol = result["om0_pol_rad"]
        om_truth_mag = float(np.linalg.norm(om_truth))
        result["om_mag_err_pct"] = float(
            (np.linalg.norm(om_pol) - om_truth_mag) / om_truth_mag * 100
        )
        result["om_dir_err_deg"] = float(
            np.degrees(np.arccos(np.clip(
                abs(np.dot(
                    om_pol / max(1e-12, np.linalg.norm(om_pol)),
                    om_truth / om_truth_mag,
                )), 0, 1,
            )))
        )

        log(
            f"  rank {rank:3d}/{len(clusters_sorted)}  "
            f"cluster_id={c['cluster_id']:3d}  "
            f"local ρ_seed={result['surrogate_rho_local_seed']:6.2f} → "
            f"ρ_polished={result['surrogate_rho_local_polished']:6.3f}  "
            f"(rotvec_a={result['rotvec_a_pol_mag_deg']:6.2f}°, "
            f"|ω_a|Δ={result['om_a_change_pct']:+.2f}%, "
            f"q0_err={result['q0_err_polished_deg']:.2f}°, "
            f"|ω|err={result['om_mag_err_pct']:+.2f}%, "
            f"ω_dir_err={result['om_dir_err_deg']:.2f}°, "
            f"n_eval={result['n_eval']:3d}, wall={result['wall_s']:.1f}s)"
        )
        polished.append(result)
    return polished


def stage_hifi_gate_classify(polished, ctx, *, gate, log):
    """Hi-fi render only candidates with surrogate_rho_local_polished < gate."""
    target = ctx["mag_hifi_truth"]
    n_pass = sum(
        1 for p in polished if p["surrogate_rho_local_polished"] < gate
    )
    log(
        f"hi-fi gate (surrogate_rho_local_polished < {gate}): "
        f"{n_pass}/{len(polished)} pass"
    )
    for p in polished:
        if p["surrogate_rho_local_polished"] < gate:
            t0 = time.time()
            try:
                pred = render_hifi(p["q0_pol_wxyz"], p["om0_pol_rad"], ctx)
                rho_h = rho_from_hifi(pred, target)
                band = rho_band(rho_h)
            except Exception as e:
                log(f"  cluster_id={p['cluster_id']}: hi-fi FAILED: {e}")
                rho_h, band = float("nan"), "ERR"
                pred = np.full_like(target, np.nan)
            p["pred_hifi"] = pred
            p["rho_polished_hifi"] = float(rho_h)
            p["band_polished_hifi"] = band
            p["hifi_render_s"] = time.time() - t0
            log(
                f"  HI-FI cluster_id={p['cluster_id']:3d}  "
                f"local ρ={p['surrogate_rho_local_polished']:.3f} → "
                f"hi-fi ρ={rho_h:.3f}  band={band}  "
                f"(q0_err={p['q0_err_polished_deg']:.2f}°, "
                f"|ω|err={p['om_mag_err_pct']:+.2f}%, "
                f"ω_dir_err={p['om_dir_err_deg']:.2f}°)"
            )
        else:
            p["pred_hifi"] = None
            p["rho_polished_hifi"] = float("nan")
            p["band_polished_hifi"] = "GATED"
            p["hifi_render_s"] = 0.0
    return polished


# ----- main ----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--n-coarse", type=int, default=N_COARSE)
    ap.add_argument("--n-dense", type=int, default=N_DENSE)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--edge-lo", type=int, default=EARLY_LO)
    ap.add_argument("--edge-hi", type=int, default=EARLY_HI)
    ap.add_argument("--ca-cap", type=int, default=CA_CAP)
    ap.add_argument("--n-dirs", type=int, default=N_DIRS)
    ap.add_argument("--n-mags", type=int, default=N_MAGS)
    ap.add_argument("--mag-bracket", type=float, default=OMEGA_BRACKET)
    ap.add_argument("--W", type=int, default=WINDOW_W,
                    help="local cost window radius (epochs)")
    ap.add_argument("--top-k-for-clustering", type=int, default=TOP_K_FOR_CLUSTERING)
    ap.add_argument("--top-k-polish", type=int, default=TOP_K_POLISH)
    ap.add_argument("--n-workers", type=int, default=N_WORKERS)
    ap.add_argument(
        "--out-root",
        default=str(SURVEY / "results" / "s059j_cloud_data_omega_grid"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_root) / f"seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059j: cloud-data ω-grid search — seed {args.seed} ===")
    log(
        f"config: N_COARSE={args.n_coarse} N_DENSE={args.n_dense} "
        f"rng={args.rng_seed} CA_CAP={args.ca_cap} "
        f"N_DIRS={args.n_dirs} N_MAGS={args.n_mags} "
        f"omega_bracket=±{args.mag_bracket*100:.0f}% W={args.W} "
        f"TOP_K_FOR_CLUSTERING={args.top_k_for_clustering} "
        f"TOP_K_POLISH={args.top_k_polish} workers={args.n_workers}"
    )
    log(
        "ORACLE FLAG: ω-grid centered on truth |ω_a_body| at T_A — "
        "PILOT regime, NOT operational. Cohort runner will swap to s055a estimator."
    )

    ctx = build_context(args.seed)

    log("\n[1/8] anchor selection (coarse pool)")
    anchor = stage_anchor_select_coarse(
        ctx,
        n_coarse=args.n_coarse,
        edge_lo=args.edge_lo, edge_hi=args.edge_hi,
        rng_seed=args.rng_seed, log=log,
    )
    T_A = anchor["T_A"]
    cv_at_T_A = int(anchor["cv"][T_A - args.edge_lo])
    (out_dir / "anchor_summary.json").write_text(json.dumps({
        "T_A": T_A,
        "cv": anchor["cv"].tolist(),
        "cv_at_T_A": cv_at_T_A,
        "edge_lo": args.edge_lo,
        "edge_hi": args.edge_hi,
        "wall_s": anchor["wall_s"],
    }, indent=2))
    log(f"Saved: {out_dir / 'anchor_summary.json'}")

    log("\n[2/8] dense at anchor")
    dense = stage_dense_at_anchor(
        ctx, T_A,
        n_dense=args.n_dense, rng_seed=args.rng_seed, log=log,
    )
    C_a = dense["C_a"]

    log("\n[3/8] |C_a|-cap subsample")
    C_a_capped = stage_cap_subsample(
        C_a, cap=args.ca_cap, rng_seed=args.rng_seed, log=log,
    )
    n_C_a_capped = int(len(C_a_capped))

    # Diagnostic: closest q_a in the capped C_a to truth q_a, plus oracle ω.
    q_a_truth, om_a_truth = truth_qa_om_at_anchor(ctx, T_A)
    qa_dists_capped = quat_ang_deg_batch(C_a_capped, q_a_truth)
    closest_qa_to_truth_deg = float(qa_dists_capped.min())
    log(
        f"diagnostic: closest q_a in capped C_a to truth = "
        f"{closest_qa_to_truth_deg:.3f}°"
    )
    om_a_truth_mag = float(np.linalg.norm(om_a_truth))
    log(
        f"oracle |ω_a_body|_truth at T_A = {om_a_truth_mag:.6f} rad/s "
        f"({np.degrees(om_a_truth_mag):.4f} dps)"
    )

    log("\n[4/8] ω-grid build (oracle truth |ω_a| × [1±bracket])")
    omega_grid = build_omega_grid(
        om_a_truth_mag,
        n_dirs=args.n_dirs, n_mags=args.n_mags,
        bracket=args.mag_bracket,
    )
    log(
        f"  ω-grid: N_DIRS={args.n_dirs} × N_MAGS={args.n_mags} = "
        f"{len(omega_grid)} cells"
    )

    log("\n[5/8] joint score grid (Pool over q_a chunks)")
    scores, score_wall = stage_score_grid(
        C_a_capped, omega_grid, T_A, args.W, ctx,
        n_workers=args.n_workers, log=log,
    )

    np.savez_compressed(
        out_dir / "score_grid.npz",
        C_a=C_a_capped,
        omega_grid=omega_grid,
        scores=scores,
        T_A=np.int64(T_A),
        W=np.int64(args.W),
        n_dirs=np.int64(args.n_dirs),
        n_mags=np.int64(args.n_mags),
        omega_bracket=np.float64(args.mag_bracket),
        omega_mag_anchor_used=np.float64(om_a_truth_mag),
        omega_mag_anchor_truth=np.float64(om_a_truth_mag),
        q_a_truth=q_a_truth,
        om_a_truth=om_a_truth,
        closest_qa_to_truth_deg=np.float64(closest_qa_to_truth_deg),
        wall_s=np.float64(score_wall),
    )
    log(f"Saved: {out_dir / 'score_grid.npz'}")

    log("\n[6/8] top-K extract + canonicalise + cluster")
    Q_top, Om_top, mse_top, qa_idx_top, om_idx_top = stage_topK_extract(
        C_a_capped, omega_grid, scores,
        top_K=args.top_k_for_clustering, log=log,
    )

    truth_idx_diag, qa_dist_diag, om_dist_diag = diagnostic_truth_idx_in_topK(
        Q_top, Om_top, q_a_truth, om_a_truth,
    )
    truth_score_rank_in_topK = int(
        (mse_top < mse_top[truth_idx_diag]).sum()
    ) + 1
    log(
        f"diagnostic: truth-equivalent in top-K: "
        f"idx={truth_idx_diag}, qa_d={qa_dist_diag[truth_idx_diag]:.2f}°, "
        f"ω_d={om_dist_diag[truth_idx_diag]:.2f}°, "
        f"score-rank in top-K = {truth_score_rank_in_topK}/{len(mse_top)}"
    )

    # Build a synthetic fp dict so we can reuse stage_cluster verbatim.
    # `scores` field in fp is interpreted as HIGHER=BETTER by stage_cluster
    # (it does `np.argsort(-scores)` and `np.argmax(scores[members])`), so we
    # pass NEGATIVE MSE — smallest MSE → largest -MSE → ranked first.
    fp = {
        "Q_A_pass": Q_top,
        "om_pass": Om_top,
        "scores": -mse_top,
        "qa_dist_to_truth": qa_dist_diag,
        "om_dist_to_truth": om_dist_diag,
        "truth_idx": truth_idx_diag,
    }
    cl = stage_cluster(fp, log)

    np.savez_compressed(
        out_dir / "clusters.npz",
        Q_top=Q_top, Om_top=Om_top, mse_top=mse_top,
        qa_idx_top=qa_idx_top, om_idx_top=om_idx_top,
        truth_idx_diagnostic=np.int64(truth_idx_diag),
        truth_cluster_rank_diagnostic=np.int64(cl["truth_cluster_rank"]),
        truth_score_rank_in_topK=np.int64(truth_score_rank_in_topK),
    )
    log(f"Saved: {out_dir / 'clusters.npz'}")

    log("\n[7/8] LM polish (NO truth injection)")
    polished = stage_polish_local_no_truth(
        cl["clusters_sorted"], Q_top, Om_top, T_A, args.W, ctx,
        top_K_polish=args.top_k_polish, log=log,
    )

    log("\n[8/8] hi-fi gate + ρ-band classify")
    polished = stage_hifi_gate_classify(
        polished, ctx, gate=SURROGATE_RHO_HIFI_GATE, log=log,
    )

    # ----- aggregate + save ----------------------------------------------

    bands = [p["band_polished_hifi"] for p in polished]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for b in bands:
        counts[b] = counts.get(b, 0) + 1
    n_AB = counts["A"] + counts["B"]
    n_pass_gate = sum(
        1 for p in polished
        if p["surrogate_rho_local_polished"] < SURROGATE_RHO_HIFI_GATE
    )

    log(f"\n=== HEADLINE: seed {args.seed} (s059j, NO truth injection) ===")
    log(
        f"  T_A={T_A}, |C_a|_capped={n_C_a_capped}, "
        f"closest_qa_to_truth={closest_qa_to_truth_deg:.3f}°"
    )
    log(
        f"  diagnostic truth score-rank in top-{args.top_k_for_clustering} = "
        f"{truth_score_rank_in_topK}; truth-cluster rank = {cl['truth_cluster_rank']}"
    )
    log(
        f"  Bands: A={counts['A']} B={counts['B']} C={counts['C']} "
        f"D={counts['D']} GATED={counts['GATED']} ERR={counts['ERR']}"
    )
    log(
        f"  Band A∪B yield: {n_AB}/{len(polished)} polished candidates"
    )
    log(
        f"  Wall total: {(time.time()-t_overall)/60:.1f} min "
        f"({time.time()-t_overall:.0f}s)"
    )

    np.savez_compressed(
        out_dir / "polished_states.npz",
        q0_pol_wxyz=np.array([p["q0_pol_wxyz"] for p in polished]),
        om0_pol_rad=np.array([p["om0_pol_rad"] for p in polished]),
        q_a_pol_wxyz=np.array([p["q_a_pol_wxyz"] for p in polished]),
        om_a_pol_rad=np.array([p["om_a_pol_rad"] for p in polished]),
        cluster_rank=np.array([p["cluster_rank"] for p in polished]),
        cluster_id=np.array([p["cluster_id"] for p in polished]),
        surrogate_rho_local_seed=np.array(
            [p["surrogate_rho_local_seed"] for p in polished]),
        surrogate_rho_local_polished=np.array(
            [p["surrogate_rho_local_polished"] for p in polished]),
        rho_polished_hifi=np.array(
            [p["rho_polished_hifi"] for p in polished]),
        band=np.array([p["band_polished_hifi"] for p in polished]),
        q0_err_deg=np.array([p["q0_err_polished_deg"] for p in polished]),
        om_mag_err_pct=np.array([p["om_mag_err_pct"] for p in polished]),
        om_dir_err_deg=np.array([p["om_dir_err_deg"] for p in polished]),
        n_eval=np.array([p["n_eval"] for p in polished]),
        wall_s=np.array([p["wall_s"] for p in polished]),
        pred_hifi=np.array([
            p["pred_hifi"] if p["pred_hifi"] is not None
            else np.full_like(ctx["mag_hifi_truth"], np.nan)
            for p in polished
        ]),
    )
    log(f"Saved: {out_dir / 'polished_states.npz'}")

    finite_q0_errs = [p["q0_err_polished_deg"] for p in polished
                      if np.isfinite(p["q0_err_polished_deg"])]
    finite_om_dir_errs = [p["om_dir_err_deg"] for p in polished
                          if np.isfinite(p["om_dir_err_deg"])]
    finite_om_mag_errs = [abs(p["om_mag_err_pct"]) for p in polished
                          if np.isfinite(p["om_mag_err_pct"])]

    summary = {
        "experiment": "s059j",
        "seed": int(args.seed),
        "config": {
            "n_coarse": int(args.n_coarse),
            "n_dense": int(args.n_dense),
            "rng_seed": int(args.rng_seed),
            "ca_cap": int(args.ca_cap),
            "tol_mag": float(TOL_MAG),
            "edge_lo": int(args.edge_lo),
            "edge_hi": int(args.edge_hi),
            "n_dirs": int(args.n_dirs),
            "n_mags": int(args.n_mags),
            "omega_bracket": float(args.mag_bracket),
            "window_w": int(args.W),
            "top_k_for_clustering": int(args.top_k_for_clustering),
            "top_k_polish": int(args.top_k_polish),
            "surrogate_rho_hifi_gate": float(SURROGATE_RHO_HIFI_GATE),
            "n_workers": int(args.n_workers),
            "omega_centering": "oracle_truth_at_anchor",
            "cluster_q_deg": float(CLUSTER_Q_DEG),
            "cluster_om_deg": float(CLUSTER_OM_DEG),
            "cluster_om_mag_pct": float(CLUSTER_OM_MAG_PCT),
            "lm_max_nfev": int(LM_MAX_NFEV),
            "lm_ftol": float(LM_FTOL),
        },
        "stages": {
            "anchor": {
                "T_A": int(T_A),
                "cv_at_T_A": cv_at_T_A,
                "wall_s": float(anchor["wall_s"]),
            },
            "dense": {
                "n_C_a_raw": int(dense["n_C_a_raw"]),
                "n_C_a_capped": n_C_a_capped,
                "closest_qa_to_truth_deg": closest_qa_to_truth_deg,
                "wall_s": float(dense["wall_s"]),
            },
            "score": {
                "min_mse": float(scores.min()),
                "min_rho": float(np.sqrt(scores.min()) / 0.05),
                "wall_s": float(score_wall),
            },
            "cluster": {
                "n_top_K": int(args.top_k_for_clustering),
                "n_clusters_after_canon": int(len(cl["clusters"])),
                "diagnostic_truth_cluster_rank": int(cl["truth_cluster_rank"]),
                "diagnostic_truth_score_rank_in_topK":
                    int(truth_score_rank_in_topK),
            },
            "polish": {
                "n_polished": int(len(polished)),
                "best_surrogate_rho_local_polished": float(
                    min((p["surrogate_rho_local_polished"] for p in polished),
                        default=float("nan"))),
                "n_pass_hifi_gate": int(n_pass_gate),
                "wall_s": float(sum(p["wall_s"] for p in polished)),
            },
            "hifi": {
                "band_counts": counts,
                "n_band_AB": int(n_AB),
                "headline_yield": int(n_AB),
                "wall_s": float(sum(p.get("hifi_render_s", 0.0) for p in polished)),
            },
        },
        "diagnostic_truth": {
            "best_polish_q0_err_deg": float(min(finite_q0_errs)) if finite_q0_errs else float("nan"),
            "best_polish_om_dir_err_deg": float(min(finite_om_dir_errs)) if finite_om_dir_errs else float("nan"),
            "best_polish_om_mag_err_pct_abs": float(min(finite_om_mag_errs)) if finite_om_mag_errs else float("nan"),
            "om_a_truth_mag_dps": float(np.degrees(om_a_truth_mag)),
            "qa_dist_truth_to_topK_deg": float(qa_dist_diag[truth_idx_diag]),
            "om_dist_truth_to_topK_deg": float(om_dist_diag[truth_idx_diag]),
        },
        "wall_total_s": float(time.time() - t_overall),
        "polished": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in p.items() if k != "pred_hifi"}
            for p in polished
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved: {out_dir / 'summary.json'}")

    # ----- plot -----------------------------------------------------------

    log("\nplotting score grid heatmap...")
    rho_grid = np.sqrt(scores) / 0.05
    row_min_rho = rho_grid.min(axis=1)
    row_order = np.argsort(row_min_rho)
    rho_sorted = rho_grid[row_order]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        np.log10(np.maximum(rho_sorted, 1e-3)),
        aspect="auto", origin="lower", cmap="viridis",
    )
    ax.set_xlabel("ω-grid flat index")
    ax.set_ylabel("q_a index (sorted by row-min ρ)")
    ax.set_title(
        f"log10(ρ_local) over (q_a, ω) — seed {args.seed}, T_A={T_A}, W={args.W}\n"
        f"|C_a|={n_C_a_capped}, |ω_grid|={len(omega_grid)}, "
        f"min ρ={float(rho_grid.min()):.3f}, Band A∪B yield={n_AB}/{len(polished)}"
    )
    fig.colorbar(im, ax=ax, label="log10(ρ_local)")
    plt.tight_layout()
    plot_path = out_dir / "score_grid.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
