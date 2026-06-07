"""s021 — Truth-vs-random score distributions, full cohort.

For each of 100 seeds:
  - Score (alignment, geo) on truth (q0, ω) — should be 1.0/1.0 by construction.
  - Score (alignment, geo) on N_RANDOM uniform Sobol-like random (q0, ω):
      q0 ← Rotation.random
      ω-dir ← uniform on S² (Gaussian then normalised)
      ω-mag ← uniform [0.1, 1.5] dps
  - Score the body-twin (q_180x · q0_truth, R_180x · ω_truth).
  - Save per-seed score arrays + truth scores + twin scores to NPZ.

Outputs:
  results/s021/score_distributions.npz
  results/s021/summary.json

Wall on Pool(8): ~18 min for N_RANDOM=1000.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib.traj_load import load_truth, list_seeds  # noqa: E402
from lib import filter_costs as fc                # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s021"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---- config ----
N_RANDOM = 1000
N_WORKERS = 8
RNG_SEED = 20260504  # date-based for reproducibility
ALIGN_BRIGHT_MAG = 11.0
ALIGN_WINDOW_EPOCHS = 3
GEO_THRESHOLD_DEG = 5.0
SPEC_THRESHOLD_DEG = 5.0  # for spec event identification at truth

# ---- per-worker state (initialised once per process) ----
_STATIC = None
_TIER = None


def _init_worker():
    global _STATIC, _TIER
    _STATIC = fc.load_static_geometry()
    _TIER = fc.load_tier_table()
    # Pre-warm surrogate
    from lib.surrogate_eval import get_model
    get_model()


def _generate_random_candidates(n: int, rng: np.random.Generator):
    """Generate (q0, omega) for n random candidates."""
    qs = []
    for _ in range(n):
        qxyzw = Rotation.random(random_state=rng).as_quat()
        qs.append(np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
    qs = np.array(qs)
    # ω-direction: Gaussian on S²
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mags = rng.uniform(0.1, 1.5, size=n)
    omegas = np.deg2rad(mags[:, None] * dirs)
    return qs, omegas


def _score_one_seed(args):
    """Score a single seed's full panel: truth + twin + N_RANDOM."""
    seed, q0s_random, omegas_random = args

    truth = load_truth(seed)
    seed_data = fc.precompute_seed_filter_data(
        truth, _TIER, spec_threshold_deg=SPEC_THRESHOLD_DEG,
        bright_mag_threshold=ALIGN_BRIGHT_MAG,
    )

    n_bright = int(seed_data["bright_peak_idx"].size)
    n_spec = int(seed_data["spec_event_idx"].size)

    # Truth
    res_t = fc.evaluate_candidate(
        truth["q0_wxyz"], truth["omega0_rad"], seed_data,
        _STATIC["inertia_tensor"], _STATIC["face_normals"], _TIER["tier_face_idx"],
        align_window_epochs=ALIGN_WINDOW_EPOCHS,
        align_bright_mag=ALIGN_BRIGHT_MAG,
        geo_threshold_deg=GEO_THRESHOLD_DEG,
    )
    truth_align = res_t["score_alignment"]
    truth_geo = res_t["score_geo"]

    # Twin: q_180x · q0_truth + R_180x · ω
    import quaternion as q_pkg
    q0t = truth["q0_wxyz"]
    q0_q = q_pkg.quaternion(q0t[0], q0t[1], q0t[2], q0t[3])
    q_180x = q_pkg.quaternion(0, 1, 0, 0)
    q_twin_q = q_180x * q0_q
    q_twin = np.array([q_twin_q.w, q_twin_q.x, q_twin_q.y, q_twin_q.z])
    R_180x = np.diag([1.0, -1.0, -1.0])
    omega_twin = R_180x @ truth["omega0_rad"]
    res_w = fc.evaluate_candidate(
        q_twin, omega_twin, seed_data,
        _STATIC["inertia_tensor"], _STATIC["face_normals"], _TIER["tier_face_idx"],
        align_window_epochs=ALIGN_WINDOW_EPOCHS,
        align_bright_mag=ALIGN_BRIGHT_MAG,
        geo_threshold_deg=GEO_THRESHOLD_DEG,
    )
    twin_align = res_w["score_alignment"]
    twin_geo = res_w["score_geo"]

    # Random panel
    n_rand = q0s_random.shape[0]
    rand_align = np.full(n_rand, np.nan)
    rand_geo = np.full(n_rand, np.nan)
    for i in range(n_rand):
        res = fc.evaluate_candidate(
            q0s_random[i], omegas_random[i], seed_data,
            _STATIC["inertia_tensor"], _STATIC["face_normals"], _TIER["tier_face_idx"],
            align_window_epochs=ALIGN_WINDOW_EPOCHS,
            align_bright_mag=ALIGN_BRIGHT_MAG,
            geo_threshold_deg=GEO_THRESHOLD_DEG,
        )
        rand_align[i] = res["score_alignment"]
        rand_geo[i] = res["score_geo"]

    return {
        "seed": seed,
        "n_bright": n_bright,
        "n_spec": n_spec,
        "truth_align": truth_align,
        "truth_geo": truth_geo,
        "twin_align": twin_align,
        "twin_geo": twin_geo,
        "rand_align": rand_align,
        "rand_geo": rand_geo,
    }


def main():
    print("=" * 72, flush=True)
    print(f"s021 — truth-vs-random filter scoring (N_RANDOM={N_RANDOM})", flush=True)
    print("=" * 72, flush=True)

    seeds = list_seeds()
    print(f"  seeds: {len(seeds)}", flush=True)
    print(f"  workers: {N_WORKERS}", flush=True)

    # Generate random candidates (same for all seeds — fair comparison)
    rng = np.random.default_rng(RNG_SEED)
    q0s_rand, omegas_rand = _generate_random_candidates(N_RANDOM, rng)
    print(f"  random candidates per seed: {q0s_rand.shape[0]}", flush=True)

    args_list = [(sd, q0s_rand, omegas_rand) for sd in seeds]

    t0 = time.time()
    print(f"  starting Pool({N_WORKERS}) at {time.strftime('%H:%M:%S')}", flush=True)
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_score_one_seed, args_list)):
            results.append(r)
            wall = time.time() - t0
            print(f"    [{i+1:3d}/{len(seeds)}] seed {r['seed']:3d} "
                  f"truth(a={r['truth_align']:.2f},g={r['truth_geo']:.2f}) "
                  f"twin(a={r['twin_align']:.2f},g={r['twin_geo']:.2f}) "
                  f"n_bright={r['n_bright']:2d} n_spec={r['n_spec']:2d} "
                  f"[{wall:.0f}s]", flush=True)
    wall_total = time.time() - t0
    print(f"  total wall: {wall_total:.1f}s = {wall_total/60:.1f} min", flush=True)

    # Sort by seed
    results.sort(key=lambda x: x["seed"])

    # Build arrays
    seeds_arr = np.array([r["seed"] for r in results])
    n_bright = np.array([r["n_bright"] for r in results])
    n_spec = np.array([r["n_spec"] for r in results])
    truth_align = np.array([r["truth_align"] for r in results])
    truth_geo = np.array([r["truth_geo"] for r in results])
    twin_align = np.array([r["twin_align"] for r in results])
    twin_geo = np.array([r["twin_geo"] for r in results])
    rand_align = np.stack([r["rand_align"] for r in results], axis=0)  # (100, N_RANDOM)
    rand_geo = np.stack([r["rand_geo"] for r in results], axis=0)

    np.savez_compressed(
        RESULTS / "score_distributions.npz",
        seeds=seeds_arr,
        n_bright=n_bright,
        n_spec=n_spec,
        truth_align=truth_align,
        truth_geo=truth_geo,
        twin_align=twin_align,
        twin_geo=twin_geo,
        rand_align=rand_align,
        rand_geo=rand_geo,
        random_q0s=q0s_rand,
        random_omegas=omegas_rand,
        n_random=N_RANDOM,
        align_bright_mag=ALIGN_BRIGHT_MAG,
        align_window_epochs=ALIGN_WINDOW_EPOCHS,
        geo_threshold_deg=GEO_THRESHOLD_DEG,
        spec_threshold_deg=SPEC_THRESHOLD_DEG,
    )

    # Summary statistics
    truth_align_finite = truth_align[np.isfinite(truth_align)]
    truth_geo_finite = truth_geo[np.isfinite(truth_geo)]

    # Truth percentile rank in random distribution (per seed)
    truth_align_rank = np.full(len(seeds), np.nan)
    truth_geo_rank = np.full(len(seeds), np.nan)
    for i in range(len(seeds)):
        ra = rand_align[i][np.isfinite(rand_align[i])]
        rg = rand_geo[i][np.isfinite(rand_geo[i])]
        if np.isfinite(truth_align[i]) and ra.size > 0:
            truth_align_rank[i] = (np.sum(ra >= truth_align[i]) + 1) / (ra.size + 1)
        if np.isfinite(truth_geo[i]) and rg.size > 0:
            truth_geo_rank[i] = (np.sum(rg >= truth_geo[i]) + 1) / (rg.size + 1)

    summary = {
        "n_seeds": int(len(seeds)),
        "n_random_per_seed": int(N_RANDOM),
        "wall_seconds": wall_total,
        "params": {
            "align_bright_mag_threshold": ALIGN_BRIGHT_MAG,
            "align_window_epochs": ALIGN_WINDOW_EPOCHS,
            "geo_threshold_deg": GEO_THRESHOLD_DEG,
            "spec_threshold_deg": SPEC_THRESHOLD_DEG,
        },
        "truth_align": {
            "n_finite": int(np.sum(np.isfinite(truth_align))),
            "min": float(np.nanmin(truth_align)),
            "median": float(np.nanmedian(truth_align)),
            "n_at_1.0": int(np.sum(truth_align >= 0.999)),
            "n_below_0.95": int(np.sum(truth_align < 0.95)),
            "n_below_0.5": int(np.sum(truth_align < 0.5)),
        },
        "truth_geo": {
            "n_finite": int(np.sum(np.isfinite(truth_geo))),
            "n_undefined_no_spec_events": int(np.sum(~np.isfinite(truth_geo))),
            "median": float(np.nanmedian(truth_geo)),
            "n_at_1.0": int(np.sum(truth_geo >= 0.999)),
        },
        "twin_align": {
            "median": float(np.nanmedian(twin_align)),
            "n_at_1.0": int(np.sum(twin_align >= 0.999)),
            "n_below_0.5": int(np.sum(twin_align < 0.5)),
        },
        "twin_geo": {
            "median": float(np.nanmedian(twin_geo)),
            "n_at_1.0": int(np.sum(twin_geo >= 0.999)),
        },
        "truth_percentile_rank_in_random": {
            "align_median": float(np.nanmedian(truth_align_rank)),
            "align_n_top1pct": int(np.sum(truth_align_rank <= 0.01)),
            "align_n_top5pct": int(np.sum(truth_align_rank <= 0.05)),
            "geo_median": float(np.nanmedian(truth_geo_rank)),
            "geo_n_top1pct": int(np.sum(truth_geo_rank <= 0.01)),
            "geo_n_top5pct": int(np.sum(truth_geo_rank <= 0.05)),
        },
        "random_score_global": {
            "align_p10": float(np.nanpercentile(rand_align, 10)),
            "align_p50": float(np.nanpercentile(rand_align, 50)),
            "align_p90": float(np.nanpercentile(rand_align, 90)),
            "align_p99": float(np.nanpercentile(rand_align, 99)),
            "align_max": float(np.nanmax(rand_align)),
            "geo_p10": float(np.nanpercentile(rand_geo, 10)),
            "geo_p50": float(np.nanpercentile(rand_geo, 50)),
            "geo_p90": float(np.nanpercentile(rand_geo, 90)),
            "geo_p99": float(np.nanpercentile(rand_geo, 99)),
            "geo_max": float(np.nanmax(rand_geo)),
        },
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {RESULTS / 'score_distributions.npz'}", flush=True)
    print(f"Saved: {RESULTS / 'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
