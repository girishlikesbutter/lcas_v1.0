"""s082 — joint (q0, ω) grid pivot on 3 fresh holdout seeds.

Measures the rank of the truth-nearest grid cell under full-LC Jacobi Path 2
+ surrogate-MSE scoring on a Sobol-Shoemake(q0) × Fibonacci(ω-dir) × LS-bracket(|ω|)
joint grid. The rank picks the Day-2 architecture per the plan §5.4 decision tree:

    rank ≤ 200       → Branch v2 (dense joint grid + top-K polish)
    rank 200-10000   → Branch v3 (multi-anchor consistency filter via Path 2)
    rank > 10000     → reconsider; probably densify N_dir or roll elliprj first

Three seeds picked from the post-fix holdout pool 100-119 via the stratification
procedure in plan §5.2 (1 LAM-slow, 1 LAM-fast, 1 SAM). The pipeline runs blind
on each seed: truth is loaded ONLY for the stratification metadata + final
error reporting, never inside the scoring loop.

Diagnostic axes saved per seed so a bad rank can be attributed:
  - |ω|-offset of nearest-truth cell (5-cell LS-bracket, ratio √2 spacing,
    worst-case ~17%; smoke shows 3-37% across slow/fast seeds — see lc_features
    docstring).
  - ω-direction offset (Fibonacci N=2000, avg spacing 4.5°).
  - q-geodesic to nearest Sobol q in the pool (N=64 → ~32 after twin
    canonicalisation; pool quantisation typically 18-25° on SO(3)).

Source: blind_inversion_15min_plan_2026-05-20.md §5, §14.

Wall budget (plan §5.6): 3 seeds × ~320k candidates × ~20 ms/24 workers ≈
13 min Pool(24) scoring + ~5 min stratification + writeup ≈ 18-20 min total.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from lib.c_t_pipeline import sample_so3_pool, compute_j2000_units  # noqa: E402
from lib.twin import canonical_batch  # noqa: E402
from lib.jacobi_propagator import (  # noqa: E402
    propagate_jacobi_path2,
    _eigendecompose_inertia,
)
from lib.surrogate_eval import get_model  # noqa: E402
from lib.lc_features import ls_bracket  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration (plan §3.4, §5.3)
# ---------------------------------------------------------------------------

N_Q = 64
N_DIR = 2000
N_MAG = 5
SAMPLE_SEED = 42
HOLDOUT_SEEDS = list(range(100, 120))
POOL_SIZE = 24
CHUNKSIZE = 256

OUT_DIR = SURVEY_DIR / "results" / "s082"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stratification — uses truth ONLY for picking seeds, never inside the search.
# ---------------------------------------------------------------------------


def fibonacci_sphere(n: int) -> np.ndarray:
    """Golden-angle Fibonacci sphere; (n, 3) unit vectors. avg ang ≈ √(41253/n)°."""
    i = np.arange(n) + 0.5
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * i / n
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = 2.0 * np.pi * i / golden
    return np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1)


def compute_holdout_metadata(seeds, inertia_tensor):
    """Per-seed truth metadata for stratification. NOT used in pipeline."""
    records = []
    I_pa, R_pa = _eigendecompose_inertia(inertia_tensor)
    # Plan §14.1 uses I_pa[1] = middle eigenvalue; _eigendecompose_inertia returns
    # eigenvalues sorted descending (I_pa[0] >= I_pa[1] >= I_pa[2]). Verify by
    # checking that the propagator's own asymmetric-Jacobi build interprets
    # I_pa indices accordingly.
    # By convention in src.dynamics.attitude_propagator the principal axes are
    # sorted I_1 >= I_2 >= I_3 so I_pa[1] = intermediate = I_2.
    I_2 = float(sorted(I_pa)[1])
    for s in seeds:
        truth = load_truth(s)
        om = truth["omega0_rad"].astype(np.float64)
        om_pa = R_pa.T @ om
        twoT = float(np.sum(I_pa * om_pa ** 2))
        L_pa = I_pa * om_pa
        L_sq = float(np.sum(L_pa ** 2))
        disc = twoT * I_2 - L_sq
        regime = "LAM" if disc > 0 else "SAM"
        om_mag_rad = float(np.linalg.norm(om))
        om_mag_dps = float(np.degrees(om_mag_rad))
        k1 = truth["k1_body"]
        k2 = truth["k2_body"]
        pa = np.degrees(
            np.arccos(np.clip(np.sum(k1 * k2, axis=1), -1.0, 1.0))
        )
        mean_pa = float(np.nanmean(pa))
        records.append(
            {
                "seed": int(s),
                "regime": regime,
                "omega_mag_rad_s": om_mag_rad,
                "omega_mag_dps": om_mag_dps,
                "mean_pa_deg": mean_pa,
                "two_T": twoT,
                "L_sq": L_sq,
                "disc": disc,
            }
        )
    return records


def pick_3_seeds(records):
    """Pick 1 LAM-slow, 1 LAM-fast, 1 SAM. Fall back to LAM-mid if SAM unavailable."""
    lams = sorted([r for r in records if r["regime"] == "LAM"], key=lambda r: r["omega_mag_dps"])
    sams = sorted([r for r in records if r["regime"] == "SAM"], key=lambda r: r["omega_mag_dps"])
    picks = []
    if len(lams) >= 2:
        picks.append({"label": "LAM-slow", "seed": lams[0]["seed"]})
        picks.append({"label": "LAM-fast", "seed": lams[-1]["seed"]})
    elif len(lams) == 1:
        picks.append({"label": "LAM", "seed": lams[0]["seed"]})
    if sams:
        picks.append({"label": "SAM", "seed": sams[len(sams) // 2]["seed"]})
    while len(picks) < 3 and lams:
        used = {p["seed"] for p in picks}
        for r in lams[len(lams) // 2 :]:
            if r["seed"] not in used:
                picks.append({"label": "LAM-mid", "seed": r["seed"]})
                break
        else:
            break
    return picks[:3]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def build_joint_candidates(n_q=N_Q, n_dir=N_DIR, n_mag=N_MAG, om_mag_grid=None):
    """Sobol-Shoemake(q0) × Fibonacci(ω-dir, canonical hemisphere) × LS-bracket(|ω|).

    Body-X twin halving (s043): the (q, ω) → (q_180x·q, R_180x·ω) map is a
    bit-exact LC equivalence. Filtering the Fibonacci sphere to the canonical
    ω-hemisphere (ω_y > 0, per `lib/twin.py:is_canonical_batch`) enumerates
    one representative per equivalence class — the free 2× speedup. `n_dir`
    here is the FULL-sphere Fibonacci count; the function returns roughly
    half (n_dir / 2) ω-direction cells. canonical_batch is then called as a
    safety/idempotency pass — for ω_y > 0 it acts as q-sign normalisation only.
    """
    if om_mag_grid is None or len(om_mag_grid) == 0:
        raise ValueError("om_mag_grid required (ls_bracket failed)")

    pool = sample_so3_pool(n_q, SAMPLE_SEED)
    q_pool = pool["q_pool_wxyz"]  # (n_q, 4)

    dirs_full = fibonacci_sphere(n_dir)  # (n_dir, 3) — both hemispheres
    keep = dirs_full[:, 1] > 0.0  # canonical hemisphere only
    dirs = dirs_full[keep]  # (~n_dir/2, 3)

    om_full = (
        dirs[:, None, :] * np.asarray(om_mag_grid)[None, :, None]
    ).reshape(-1, 3)  # (n_dir_canon * n_mag, 3)

    # Cartesian product
    n_om = om_full.shape[0]
    q_rep = np.repeat(q_pool, n_om, axis=0)  # (n_q * n_om, 4)
    om_rep = np.tile(om_full, (n_q, 1))  # (n_q * n_om, 3)

    # canonical_batch acts as q-sign normalisation here (all ω are canonical by
    # construction; q_180x flip is never applied).
    q_canon, om_canon = canonical_batch(q_rep, om_rep)
    return q_canon, om_canon


# ---------------------------------------------------------------------------
# Per-candidate scoring (Pool worker)
# ---------------------------------------------------------------------------


_WORKER_CTX = None


def init_worker(ctx_for_pool):
    global _WORKER_CTX
    import torch  # noqa: F401

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_CTX = ctx_for_pool
    # warm surrogate
    _ = get_model()


def _quat_hist_to_R_passive(q_hist_wxyz):
    """Vectorised wxyz quaternion → (N, 3, 3) passive J2000→body matrices.

    Matches the existing pattern in lib/forward.py:
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    Then k1_body[i] = R @ sun_vec_J2000[i].
    """
    from scipy.spatial.transform import Rotation

    q_xyzw = q_hist_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(q_xyzw).as_matrix()


def score_one(args):
    q0, omega0 = args
    ctx = _WORKER_CTX
    try:
        q_hist, _ = propagate_jacobi_path2(
            q0, omega0, ctx["inertia_tensor"], ctx["observation_times"]
        )
    except Exception:
        return float("inf")
    try:
        R_hist = _quat_hist_to_R_passive(q_hist)  # (N, 3, 3)
        # ctx["sun_unit"], ctx["obs_unit"] are J2000 unit vectors (N, 3)
        k1_body = np.einsum("nij,nj->ni", R_hist, ctx["sun_unit"])
        k2_body = np.einsum("nij,nj->ni", R_hist, ctx["obs_unit"])
        pred = get_model().predict_magnitude(
            k1_body, k2_body, 0.0, 15.0, ctx["obs_dist"]
        )
        diff = pred - ctx["mag_observed"]
        mask = np.isfinite(diff)
        if mask.sum() == 0:
            return float("inf")
        return float(np.mean(diff[mask] ** 2))
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Per-seed pivot run
# ---------------------------------------------------------------------------


def run_seed_pivot(seed, label, out_dir):
    """Run the 320k joint-grid pivot on one seed."""
    print(f"\n=== s082 pivot seed {seed} ({label}) ===", flush=True)
    seed_dir = out_dir / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(seed)
    times = ctx["observation_times"]
    mag_observed = ctx["mag_hifi_truth"]  # used only as the observed LC
    sun_unit, obs_unit = compute_j2000_units(
        ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"]
    )

    # Stage 1: |ω| LS-bracket
    t0 = time.time()
    om_mag_grid = ls_bracket(times, mag_observed, n_cells=N_MAG)
    print(
        f"  LS-bracket: {len(om_mag_grid)} cells, "
        f"[{om_mag_grid[0]:.5f}..{om_mag_grid[-1]:.5f}] rad/s",
        flush=True,
    )

    # Stage 4: joint candidates
    q_cands, om_cands = build_joint_candidates(om_mag_grid=om_mag_grid)
    M = q_cands.shape[0]
    wall_gen = time.time() - t0
    print(f"  candidates: {M} (post-twin-dedup) — gen wall {wall_gen:.1f}s", flush=True)

    # Stage 5: Pool(24) scoring
    ctx_for_pool = {
        "inertia_tensor": ctx["inertia_tensor"],
        "observation_times": times,
        "sun_unit": sun_unit,
        "obs_unit": obs_unit,
        "obs_dist": ctx["obs_dist"],
        "mag_observed": mag_observed,
    }
    cand_list = list(zip(q_cands, om_cands))

    t0 = time.time()
    with Pool(POOL_SIZE, initializer=init_worker, initargs=(ctx_for_pool,)) as p:
        mses = list(
            p.imap(score_one, cand_list, chunksize=CHUNKSIZE)
        )
    mses = np.array(mses, dtype=np.float64)
    wall_score = time.time() - t0
    print(
        f"  scoring: {wall_score:.1f}s ({wall_score/M*1000:.2f} ms/candidate amortised)",
        flush=True,
    )

    # Truth-anchored diagnostics (loaded ONLY here, after the search)
    q0_truth = ctx["q0_truth"]
    om_truth = ctx["omega0_truth_rad"]
    om_mag_truth = float(np.linalg.norm(om_truth))
    om_dir_truth = om_truth / om_mag_truth if om_mag_truth > 0 else np.zeros(3)

    # Twin-canonicalise truth so we compare in the same chart as the grid
    q0_truth_c, om_truth_c = canonical_batch(
        q0_truth[None, :], om_truth[None, :]
    )
    q0_truth_c = q0_truth_c[0]
    om_truth_c = om_truth_c[0]
    om_mag_truth_c = float(np.linalg.norm(om_truth_c))
    om_dir_truth_c = om_truth_c / om_mag_truth_c

    # Errors per candidate (all 320k)
    om_dir_cands = om_cands / np.linalg.norm(om_cands, axis=1, keepdims=True)
    om_mag_cands = np.linalg.norm(om_cands, axis=1)
    om_dir_err_deg = np.degrees(
        np.arccos(np.clip(om_dir_cands @ om_dir_truth_c, -1.0, 1.0))
    )
    om_mag_err_rel = np.abs(om_mag_cands - om_mag_truth_c) / om_mag_truth_c
    # quaternion geodesic, antipode-aware
    q_dots = np.abs(q_cands @ q0_truth_c)
    q_geo_deg = np.degrees(2.0 * np.arccos(np.clip(q_dots, 0.0, 1.0)))

    # "Truth-nearest grid cell": combined metric (q-geo + 100×ω-dir-err + 100×ω-mag-err-rel%)
    # Each ~1° / 1% maps to ~1 unit so they're roughly commensurable.
    combined_err = q_geo_deg + om_dir_err_deg + 100.0 * om_mag_err_rel
    truth_nearest_idx = int(np.argmin(combined_err))
    # Rank of this candidate by MSE (1 = best, M = worst)
    rank = int(1 + np.sum(mses < mses[truth_nearest_idx]))

    # Top-50 statistics
    order = np.argsort(mses)
    top50 = order[:50]
    top200 = order[:200]
    top1000 = order[:1000]

    summary = {
        "seed": int(seed),
        "label": str(label),
        "n_candidates": int(M),
        "config": {
            "N_Q_sobol": N_Q,
            "N_DIR_fibonacci": N_DIR,
            "N_MAG_lsbracket": N_MAG,
            "SAMPLE_SEED": SAMPLE_SEED,
            "POOL_SIZE": POOL_SIZE,
        },
        "walls_sec": {
            "candidate_gen": wall_gen,
            "scoring": wall_score,
        },
        "ls_bracket": {
            "n_cells": int(len(om_mag_grid)),
            "lo_rad_s": float(om_mag_grid[0]),
            "hi_rad_s": float(om_mag_grid[-1]),
            "cells_rad_s": [float(x) for x in om_mag_grid],
        },
        "mse_distribution": {
            "min": float(np.nanmin(mses)),
            "p1": float(np.nanpercentile(mses, 1)),
            "p5": float(np.nanpercentile(mses, 5)),
            "p50": float(np.nanpercentile(mses, 50)),
            "p90": float(np.nanpercentile(mses, 90)),
            "max": float(np.nanmax(mses)),
            "n_inf": int(np.sum(~np.isfinite(mses))),
        },
        "truth_metrics": {
            "q0_truth_wxyz": q0_truth.tolist(),
            "omega0_truth_rad": om_truth.tolist(),
            "omega_mag_truth_rad_s": om_mag_truth,
            "omega_mag_truth_dps": float(np.degrees(om_mag_truth)),
            "truth_nearest_cell": {
                "idx": truth_nearest_idx,
                "q_geo_deg": float(q_geo_deg[truth_nearest_idx]),
                "om_dir_err_deg": float(om_dir_err_deg[truth_nearest_idx]),
                "om_mag_err_rel_pct": float(
                    100.0 * om_mag_err_rel[truth_nearest_idx]
                ),
                "mse": float(mses[truth_nearest_idx]),
                "rho_surrogate": float(np.sqrt(mses[truth_nearest_idx]) / 0.05),
                "rank_by_mse": rank,
            },
            "min_axis_errors_unconditional": {
                "min_q_geo_deg": float(np.min(q_geo_deg)),
                "min_om_dir_err_deg": float(np.min(om_dir_err_deg)),
                "min_om_mag_err_rel_pct": float(100.0 * np.min(om_mag_err_rel)),
            },
            "top_50_error_distribution": {
                "q_geo_p50_deg": float(np.median(q_geo_deg[top50])),
                "q_geo_p10_deg": float(np.percentile(q_geo_deg[top50], 10)),
                "om_dir_err_p50_deg": float(np.median(om_dir_err_deg[top50])),
                "om_mag_err_p50_pct": float(
                    100.0 * np.median(om_mag_err_rel[top50])
                ),
                "rho_surrogate_top1": float(np.sqrt(mses[order[0]]) / 0.05),
                "rho_surrogate_top10": float(np.sqrt(mses[order[9]]) / 0.05),
                "rho_surrogate_top50": float(np.sqrt(mses[order[49]]) / 0.05),
            },
            "rank_truth_nearest_in_topK": {
                "top200": bool(rank <= 200),
                "top1000": bool(rank <= 1000),
                "top10000": bool(rank <= 10000),
            },
        },
    }

    # Save NPZ with all per-candidate rows (raw material for Day 2 design)
    np.savez(
        seed_dir / "candidates.npz",
        q_cands=q_cands,
        om_cands=om_cands,
        mses=mses,
        q_geo_deg=q_geo_deg,
        om_dir_err_deg=om_dir_err_deg,
        om_mag_err_rel=om_mag_err_rel,
    )

    with open(seed_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    # Histogram of MSE
    finite = np.isfinite(mses)
    ax.hist(np.log10(mses[finite] + 1e-12), bins=80, color="tab:gray", alpha=0.7)
    ax.axvline(
        np.log10(mses[truth_nearest_idx]),
        color="tab:red",
        linestyle="--",
        label=f"truth-nearest (rank {rank})",
    )
    ax.axvline(np.log10(mses[order[0]]), color="tab:blue", linestyle="--", label="grid min")
    ax.set_xlabel("log10(MSE)")
    ax.set_ylabel("count")
    ax.set_title(f"seed {seed} ({label}) MSE distribution")
    ax.legend()

    ax = axes[1]
    # Top-1000 scatter: ω-dir err vs q-geo
    sc = ax.scatter(
        q_geo_deg[top1000],
        om_dir_err_deg[top1000],
        c=np.log10(mses[top1000] + 1e-12),
        s=8,
        cmap="viridis",
    )
    ax.scatter(
        [q_geo_deg[truth_nearest_idx]],
        [om_dir_err_deg[truth_nearest_idx]],
        c="red",
        s=80,
        marker="x",
        label=f"truth-nearest (rank {rank})",
    )
    plt.colorbar(sc, ax=ax, label="log10(MSE)")
    ax.set_xlabel("q-geodesic to truth (deg)")
    ax.set_ylabel("ω-direction err to truth (deg)")
    ax.set_title(f"seed {seed} top-1000 candidates")
    ax.legend()

    fig.tight_layout()
    fig.savefig(seed_dir / "pivot_diagnostics.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(
        f"  truth-nearest cell: rank={rank}/{M}, "
        f"q-geo={q_geo_deg[truth_nearest_idx]:.2f}°, "
        f"ω-dir={om_dir_err_deg[truth_nearest_idx]:.2f}°, "
        f"|ω|-err={100*om_mag_err_rel[truth_nearest_idx]:.2f}%, "
        f"ρ_surr={summary['truth_metrics']['truth_nearest_cell']['rho_surrogate']:.2f}",
        flush=True,
    )
    print(f"  Saved: {seed_dir/'summary.json'}", flush=True)
    print(f"  Saved: {seed_dir/'candidates.npz'}", flush=True)
    print(f"  Saved: {seed_dir/'pivot_diagnostics.png'}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_total = time.time()
    print("=== s082 — joint-grid pivot on 3 fresh holdout seeds ===", flush=True)

    # Stratify holdout 100-119
    ctx_template = build_context(0)  # any seed; we just need inertia
    inertia_tensor = ctx_template["inertia_tensor"]
    records = compute_holdout_metadata(HOLDOUT_SEEDS, inertia_tensor)
    picks = pick_3_seeds(records)
    n_lam = sum(1 for r in records if r["regime"] == "LAM")
    n_sam = sum(1 for r in records if r["regime"] == "SAM")
    print(
        f"Holdout regime split: {n_lam} LAM / {n_sam} SAM out of {len(records)}",
        flush=True,
    )
    print(f"Picked: {picks}", flush=True)

    with open(OUT_DIR / "holdout_stratification.json", "w") as f:
        json.dump(
            {
                "n_lam": n_lam,
                "n_sam": n_sam,
                "picks": picks,
                "records": records,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"Saved: {OUT_DIR/'holdout_stratification.json'}", flush=True)

    # Run pivot on each picked seed
    per_seed = []
    for p in picks:
        per_seed.append(
            run_seed_pivot(int(p["seed"]), p["label"], OUT_DIR)
        )

    # Cohort-level summary + branch decision
    ranks = [
        s["truth_metrics"]["truth_nearest_cell"]["rank_by_mse"]
        for s in per_seed
    ]
    median_rank = int(np.median(ranks))
    if median_rank <= 200:
        branch = "v2"
        rationale = (
            f"Median truth-nearest rank {median_rank} ≤ 200 — top-K=200 polish "
            "captures truth-near candidates. Branch v2 simple dense grid + top-K polish."
        )
    elif median_rank <= 10000:
        branch = "v3"
        rationale = (
            f"Median truth-nearest rank {median_rank} in (200, 10000] — top-200 polish "
            "misses truth-near. Branch v3 multi-anchor consistency filter required."
        )
    else:
        branch = "reconsider"
        rationale = (
            f"Median truth-nearest rank {median_rank} > 10000 — joint grid is too coarse. "
            "Re-evaluate Day 2 plan; consider densifying N_dir (requires elliprj Path 2) "
            "or revisiting candidate-gen strategy."
        )

    cohort = {
        "per_seed_rank_by_mse": ranks,
        "median_rank": median_rank,
        "branch_decision": branch,
        "rationale": rationale,
        "per_seed": [
            {
                "seed": s["seed"],
                "label": s["label"],
                "rank": s["truth_metrics"]["truth_nearest_cell"]["rank_by_mse"],
                "q_geo_deg": s["truth_metrics"]["truth_nearest_cell"]["q_geo_deg"],
                "om_dir_err_deg": s["truth_metrics"]["truth_nearest_cell"][
                    "om_dir_err_deg"
                ],
                "om_mag_err_rel_pct": s["truth_metrics"]["truth_nearest_cell"][
                    "om_mag_err_rel_pct"
                ],
                "rho_surrogate_truth_nearest": s["truth_metrics"][
                    "truth_nearest_cell"
                ]["rho_surrogate"],
                "wall_score_sec": s["walls_sec"]["scoring"],
            }
            for s in per_seed
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(cohort, f, indent=2, default=float)
    wall_total = time.time() - t_total
    print(f"\n=== Cohort summary ===", flush=True)
    print(f"per-seed ranks: {ranks}", flush=True)
    print(f"median rank: {median_rank}", flush=True)
    print(f"BRANCH DECISION: {branch}", flush=True)
    print(f"  {rationale}", flush=True)
    print(f"\nTotal wall: {wall_total:.1f}s", flush=True)
    print(f"Saved: {OUT_DIR/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
