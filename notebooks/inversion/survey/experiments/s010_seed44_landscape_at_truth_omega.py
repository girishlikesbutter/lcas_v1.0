"""s010 — seed-44 surrogate landscape probe at fixed truth-ω.

Survey sub-question opened by s009. s009's cohort basin-radius probe (100
seeds × 2 ICs along body-X axis) found 3 tight-tail seeds (7, 44, 76) where
T1_inside (q0=2°, ωd=0.3°, ωm=-1%) does NOT recover truth. Seeds 44 and 76
are particularly striking — LM converges from inside the s005 tube to a
competing q0 basin ~100° from truth at NEAR-TRUTH ω with low final_mse:

  seed 44 T1: q0_err=99.6°, ωd_err=0.14°, |ωm_err|=0.31%, mse=6.005e-2 mag²
  seed 76 T1: q0_err=112.0°, ωd_err=0.76°, |ωm_err|=1.01%, mse=1.828e-2 mag²

These are real (q0,ω)-jointly-coupled competing-basin attractors at
near-truth ω, NOT stall points (mse is ~100× truth_mse, not the 1000×+
typical of mid-Sobol stalls). Twin recoveries 0/200, so it is not a
twin-degeneracy artifact.

Question: at fixed truth-ω on seed 44, is the LM-found competing basin
visible at s002/s006-class Sobol density (2046 candidates), or is it
sub-Sobol-resolution narrow?

Decision tree (extends s006's A/B/C):
  (A) Sub-Sobol narrow basin. argmin = truth, n_sobol_below_truth = 0,
      AND no Sobol point within 10° of LM-landing has mse below
      LM-landing's 0.06 mag². → Competing basin is real but
      sub-Sobol-density at 2046 points; like seed 28 it requires per-
      candidate LM polish to discover. Q4c-cohort is structurally fine
      but seeds 44/76 join seed 28 as tight-tail seeds.
  (B) Sobol-visible competing basin. n_sobol_below_K_truth ≥ 1 with the
      Sobol point near the LM-landing q0 region. → Refines s002's
      universal "argmin=truth at fixed truth-ω on 8/8" claim — small-
      sample artifact at 8 seeds. Q4c-cohort needs a "find ALL candidate
      basins" search, not "polish toward truth".
  (C) argmin = LM-landing region, not truth. → Most disruptive: refutes
      s002 universally on seed 44; truth is NOT the global argmin even
      at fixed truth-ω on this seed. Would imply the surrogate landscape
      itself has multi-basin pathology at correct ω on a non-trivial
      cohort fraction.

Method:
  - Single seed: 44 (s009 tight-tail, T1 lands ~100° from truth at
    near-truth ω with mse 0.06 mag²).
  - 2046 Sobol-Shoemake quats + truth + twin (180°-body-x · q0_truth)
    → 2048 candidates.
  - SOBOL_SEED = 42 (matches s002 / s006 — Sobol point set identical).
  - ω fixed at truth-ω (NOT s009's near-truth converged ω; we want to
    test the seed-44 surrogate landscape under ideal-ω conditions).
  - Score: surrogate full-LC MSE vs cached mag_hifi.
  - Pool(8), BLAS=1.

Diagnostics (s006 + new):
  - All s006 diagnostics: n_sobol_below_K_truth for K ∈ {2, 5, 10, 100},
    sobol-min-geo-to-truth, MSE distribution percentiles.
  - **NEW — LM-landing ring probe.** geo_to_lm_landing_deg for every
    Sobol candidate. Bin into 5° rings around LM-landing (0-5°, 5-10°,
    10-20°, 20-30°, ≥30°). Report MSE distribution per ring. Decision:
    if any ring within 10° of LM-landing has its MSE p10 below
    LM-landing's 0.06 mag², we are in case (B); if MSE p10 stays > 1
    mag² in all near-rings, case (A).

Outputs:
  - results/s010/landscape.npz      all candidate quats + MSEs + diagnostics
  - results/s010/summary.json       argmin / truth-rank / competing-basin stats
  - results/s010/landscape.png      MSE vs geodesic-to-truth panel
  - results/s010/lm_landing_ring.png  MSE distribution per ring around LM-landing q0
"""

# BLAS=1 BEFORE Pool fork (per project memory).
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
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg_batch  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
S009_RUNS = SURVEY_DIR / "results" / "s009" / "runs.npz"
OUT_DIR = SURVEY_DIR / "results" / "s010"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 44
N_SOBOL = 2046  # +2 for truth and twin → 2048 total candidates
SOBOL_SEED = 42  # MATCH s002 / s006
N_WORKERS = 8

# Twin: 180° about body x.
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Diagnostic MSE thresholds (multiples of truth_mse).
COMPETING_K = [2.0, 5.0, 10.0, 100.0]

# LM-landing ring bin edges (degrees).
RING_EDGES = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 60.0, 180.01])


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a2 = 2.0 * np.pi * u2
    a3 = 2.0 * np.pi * u3
    x = s1 * np.sin(a2)
    y = s1 * np.cos(a2)
    z = s2 * np.sin(a3)
    w = s2 * np.cos(a3)
    return np.column_stack([w, x, y, z])


def build_grid(q0_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sobol = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    u = sobol.random(N_SOBOL)
    q_sobol = shoemake_to_quat(u)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    q_grid = np.vstack([q0_truth[None, :], q_twin[None, :], q_sobol])
    kind = np.concatenate([
        np.array([0], dtype=int),  # 0 = truth
        np.array([1], dtype=int),  # 1 = twin
        np.full(N_SOBOL, 2, dtype=int),  # 2 = sobol
    ])
    return q_grid, kind


def load_lm_landing(seed: int) -> tuple[np.ndarray, dict]:
    """Pull the s009 T1_inside LM-converged q0 + summary stats for `seed`."""
    r = np.load(str(S009_RUNS), allow_pickle=True)
    seeds = r["seed"]
    tiers = r["tier"]
    idx = np.where((seeds == seed) & (tiers == 1))[0]  # tier 1 = T1_inside
    if len(idx) == 0:
        raise RuntimeError(f"No T1_inside row for seed {seed} in s009 runs.npz")
    i = int(idx[0])
    q0_lm = np.asarray(r["q0_final_wxyz"][i], dtype=float)
    summary = {
        "seed": int(seed),
        "tier": "T1_inside",
        "q0_lm_wxyz": q0_lm.tolist(),
        "q0_err_deg": float(r["q0_err_deg"][i]),
        "omega_dir_err_deg": float(r["omega_dir_err_deg"][i]),
        "omega_mag_err_pct": float(r["omega_mag_err_pct"][i]),
        "final_mse": float(r["final_mse"][i]),
        "omega_final_rad": np.asarray(r["omega_final_rad"][i], dtype=float).tolist(),
    }
    return q0_lm, summary


# Worker globals.
_W_OMEGA = None
_W_TIMES = None
_W_SUN = None
_W_OBS = None
_W_SAT = None
_W_INERTIA = None
_W_OBS_DIST = None
_W_MAG_HIFI = None


def init_worker(omega0_rad, observation_times, sun_pos, obs_pos, sat_pos,
                inertia_tensor, obs_dist, mag_hifi):
    global _W_OMEGA, _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_INERTIA, _W_OBS_DIST, _W_MAG_HIFI
    _W_OMEGA = omega0_rad
    _W_TIMES = observation_times
    _W_SUN = sun_pos
    _W_OBS = obs_pos
    _W_SAT = sat_pos
    _W_INERTIA = inertia_tensor
    _W_OBS_DIST = obs_dist
    _W_MAG_HIFI = mag_hifi
    surrogate_eval.get_model()


def score_candidate(q0_wxyz: np.ndarray) -> tuple[float, float]:
    k1b, k2b, _ = propagate_to_body_frame(
        q0_wxyz, _W_OMEGA, _W_TIMES, _W_SUN, _W_OBS, _W_SAT, _W_INERTIA,
    )
    pred = surrogate_eval.predict(k1b, k2b, _W_OBS_DIST)
    full = surrogate_eval.full_lc_mse(pred, _W_MAG_HIFI)
    bright = surrogate_eval.bright_mse(pred, _W_MAG_HIFI, 11.0)
    return float(full), float(bright)


def run_seed(seed: int, inertia_tensor: np.ndarray, q0_lm: np.ndarray) -> dict:
    d = traj_load.load_truth(seed)
    q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
    omega0 = np.asarray(d["omega0_rad"], dtype=float)

    q_grid, kind = build_grid(q0_truth)
    n_cand = q_grid.shape[0]

    t0 = time.time()
    init_args = (
        omega0, d["observation_times"], d["sun_pos"], d["obs_pos"], d["sat_pos"],
        inertia_tensor, d["obs_dist"], d["mag_hifi"],
    )
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=init_args) as pool:
        results = pool.map(score_candidate, list(q_grid), chunksize=16)
    wall = time.time() - t0

    full_mse = np.array([r[0] for r in results], dtype=float)
    bright_mse = np.array([r[1] for r in results], dtype=float)

    geo_to_truth = quat_geodesic_deg_batch(q_grid, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    geo_to_twin = quat_geodesic_deg_batch(q_grid, q_twin)
    geo_to_truth_or_twin = np.minimum(geo_to_truth, geo_to_twin)
    # NEW — distance to s009 T1_inside LM-landing q0:
    geo_to_lm_landing = quat_geodesic_deg_batch(q_grid, q0_lm)

    argmin = int(np.argmin(full_mse))
    truth_mse = float(full_mse[0])
    twin_mse = float(full_mse[1])
    argmin_geo = float(geo_to_truth[argmin])
    argmin_geo_twin = float(geo_to_twin[argmin])
    argmin_geo_lm = float(geo_to_lm_landing[argmin])

    print(
        f"seed {seed:3d}  N={n_cand}  "
        f"truth_mse={truth_mse:.4e}  twin_mse={twin_mse:.4e}  "
        f"argmin={argmin} ({'truth' if argmin == 0 else 'twin' if argmin == 1 else 'sobol'})  "
        f"argmin_mse={float(full_mse[argmin]):.4e}  "
        f"argmin_geo_to_truth={argmin_geo:.1f}°  "
        f"argmin_geo_to_lm={argmin_geo_lm:.1f}°  "
        f"wall={wall:.1f}s",
        flush=True,
    )

    return {
        "seed": int(seed),
        "n_candidates": int(n_cand),
        "q_grid": q_grid,
        "kind": kind,
        "full_mse": full_mse,
        "bright_mse": bright_mse,
        "geo_to_truth_deg": geo_to_truth,
        "geo_to_twin_deg": geo_to_twin,
        "geo_to_truth_or_twin_deg": geo_to_truth_or_twin,
        "geo_to_lm_landing_deg": geo_to_lm_landing,
        "argmin_idx": argmin,
        "argmin_full_mse": float(full_mse[argmin]),
        "argmin_geo_to_truth_deg": argmin_geo,
        "argmin_geo_to_twin_deg": argmin_geo_twin,
        "argmin_geo_to_lm_deg": argmin_geo_lm,
        "truth_full_mse": truth_mse,
        "twin_full_mse": twin_mse,
        "wall_s": wall,
    }


def diagnose_competing_basins(r: dict) -> dict:
    sobol_mask = r["kind"] == 2
    sobol_full = r["full_mse"][sobol_mask]
    sobol_geo = r["geo_to_truth_deg"][sobol_mask]
    truth_mse = r["truth_full_mse"]

    sobol_min_geo = float(np.min(sobol_geo))
    n_sobol_below_truth = int(np.sum(sobol_full < truth_mse))

    competing = {}
    for k in COMPETING_K:
        mask = sobol_full < (k * truth_mse)
        n = int(np.sum(mask))
        far_mask = mask & (sobol_geo > 30.0)
        n_far = int(np.sum(far_mask))
        if n > 0:
            geos = sobol_geo[mask]
            geo_min = float(np.min(geos))
            geo_max = float(np.max(geos))
            geo_median = float(np.median(geos))
        else:
            geo_min = geo_max = geo_median = float("nan")
        competing[f"K{int(k)}"] = {
            "K": float(k),
            "n_sobol_below_K_truth": n,
            "n_sobol_below_K_truth_far_from_truth": n_far,
            "geo_min_deg": geo_min,
            "geo_max_deg": geo_max,
            "geo_median_deg": geo_median,
        }

    return {
        "truth_full_mse": truth_mse,
        "twin_full_mse": r["twin_full_mse"],
        "sobol_min_geo_to_truth_deg": sobol_min_geo,
        "n_sobol_below_truth": n_sobol_below_truth,
        "competing_basin_counts": competing,
    }


def diagnose_lm_landing_rings(r: dict, lm_summary: dict) -> dict:
    """For each ring band around LM-landing q0, compute MSE distribution
    of Sobol candidates inside the ring + count of candidates with mse
    below LM-landing's final_mse."""
    sobol_mask = r["kind"] == 2
    sobol_full = r["full_mse"][sobol_mask]
    sobol_geo_lm = r["geo_to_lm_landing_deg"][sobol_mask]
    lm_mse = lm_summary["final_mse"]

    rings = []
    for i in range(len(RING_EDGES) - 1):
        lo, hi = float(RING_EDGES[i]), float(RING_EDGES[i + 1])
        mask = (sobol_geo_lm >= lo) & (sobol_geo_lm < hi)
        n = int(np.sum(mask))
        if n == 0:
            rings.append({"lo_deg": lo, "hi_deg": hi, "n": 0,
                          "mse_min": float("nan"), "mse_p10": float("nan"),
                          "mse_p50": float("nan"), "mse_p90": float("nan"),
                          "n_below_lm_mse": 0})
            continue
        msev = sobol_full[mask]
        rings.append({
            "lo_deg": lo, "hi_deg": hi, "n": n,
            "mse_min": float(np.min(msev)),
            "mse_p10": float(np.percentile(msev, 10)),
            "mse_p50": float(np.percentile(msev, 50)),
            "mse_p90": float(np.percentile(msev, 90)),
            "n_below_lm_mse": int(np.sum(msev < lm_mse)),
        })

    sobol_min_geo_lm = float(np.min(sobol_geo_lm))
    return {
        "lm_landing_final_mse": lm_mse,
        "sobol_min_geo_to_lm_landing_deg": sobol_min_geo_lm,
        "n_sobol_below_lm_mse": int(np.sum(sobol_full < lm_mse)),
        "rings": rings,
    }


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    q0_lm, lm_summary = load_lm_landing(SEED)
    print(f"Loaded s009 LM-landing for seed {SEED} (T1_inside):")
    print(f"  q0_lm_wxyz   = {np.round(q0_lm, 5).tolist()}")
    print(f"  q0_err_deg   = {lm_summary['q0_err_deg']:.2f}")
    print(f"  ωd_err_deg   = {lm_summary['omega_dir_err_deg']:.3f}")
    print(f"  |ωm_err|_pct = {abs(lm_summary['omega_mag_err_pct']):.3f}")
    print(f"  final_mse    = {lm_summary['final_mse']:.4e} mag²")
    print()

    print(f"Running s010 surrogate-landscape probe on seed {SEED} "
          f"({N_SOBOL + 2} candidates, Pool({N_WORKERS}), BLAS=1).", flush=True)

    r = run_seed(SEED, inertia_tensor, q0_lm)
    diag = diagnose_competing_basins(r)
    ring_diag = diagnose_lm_landing_rings(r, lm_summary)

    # Print key diagnostic line.
    print()
    print(f"=== s010 seed {SEED} diagnostic ===")
    print(f"  truth_full_mse           = {diag['truth_full_mse']:.4e}")
    print(f"  twin_full_mse            = {diag['twin_full_mse']:.4e}")
    print(f"  lm_landing_final_mse     = {ring_diag['lm_landing_final_mse']:.4e}  "
          f"(s009 T1_inside)")
    print(f"  sobol_min_geo_truth      = {diag['sobol_min_geo_to_truth_deg']:.2f}°")
    print(f"  sobol_min_geo_lm_landing = {ring_diag['sobol_min_geo_to_lm_landing_deg']:.2f}°")
    print(f"  n_sobol_below_truth      = {diag['n_sobol_below_truth']}/{N_SOBOL}")
    print(f"  n_sobol_below_lm_mse     = {ring_diag['n_sobol_below_lm_mse']}/{N_SOBOL}")
    for k_label, c in diag["competing_basin_counts"].items():
        print(f"  competing  {k_label:5s} (mse < {c['K']:.0f}× truth_mse): "
              f"{c['n_sobol_below_K_truth']:4d} sobol  "
              f"({c['n_sobol_below_K_truth_far_from_truth']:4d} > 30° from truth)  "
              f"geo[{c['geo_min_deg']:5.1f}, {c['geo_max_deg']:5.1f}], "
              f"med {c['geo_median_deg']:5.1f}°")
    print()
    print(f"  --- LM-landing ring probe (seed {SEED}, s009 T1 q0_final) ---")
    print(f"  {'ring':>10s}  {'n':>5s}  {'mse_min':>11s}  {'mse_p10':>11s}  "
          f"{'mse_p50':>11s}  {'mse_p90':>11s}  {'<lm_mse':>8s}")
    for ring in ring_diag["rings"]:
        print(f"  [{ring['lo_deg']:4.1f}, {ring['hi_deg']:5.1f})  "
              f"{ring['n']:5d}  {ring['mse_min']:11.4e}  {ring['mse_p10']:11.4e}  "
              f"{ring['mse_p50']:11.4e}  {ring['mse_p90']:11.4e}  "
              f"{ring['n_below_lm_mse']:8d}")
    print()

    # Decision summary.
    case = decide_case(diag, ring_diag)
    print(f"  → DECISION: case ({case})")
    print()

    # ── Save NPZ ──
    npz_path = OUT_DIR / "landscape.npz"
    np.savez(
        npz_path,
        seed=np.array([SEED], dtype=int),
        n_sobol=N_SOBOL,
        sobol_seed=SOBOL_SEED,
        q_grid=r["q_grid"],
        kind=r["kind"],
        full_mse=r["full_mse"],
        bright_mse=r["bright_mse"],
        geo_to_truth_deg=r["geo_to_truth_deg"],
        geo_to_twin_deg=r["geo_to_twin_deg"],
        geo_to_truth_or_twin_deg=r["geo_to_truth_or_twin_deg"],
        geo_to_lm_landing_deg=r["geo_to_lm_landing_deg"],
        q0_lm_wxyz=q0_lm,
        ring_edges_deg=RING_EDGES,
    )
    print(f"Saved: {npz_path}")

    # ── Save summary JSON ──
    summary = {
        "seed": SEED,
        "n_sobol": N_SOBOL,
        "sobol_seed": SOBOL_SEED,
        "n_candidates": r["n_candidates"],
        "argmin_idx": r["argmin_idx"],
        "argmin_kind": int(r["kind"][r["argmin_idx"]]),
        "argmin_full_mse": r["argmin_full_mse"],
        "argmin_geo_to_truth_deg": r["argmin_geo_to_truth_deg"],
        "argmin_geo_to_twin_deg": r["argmin_geo_to_twin_deg"],
        "argmin_geo_to_lm_deg": r["argmin_geo_to_lm_deg"],
        "truth_full_mse": r["truth_full_mse"],
        "twin_full_mse": r["twin_full_mse"],
        "lm_landing": lm_summary,
        "diagnostics": diag,
        "lm_landing_ring_diagnostics": ring_diag,
        "decision_case": case,
        "wall_s": r["wall_s"],
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    save_landscape_plot(r, ring_diag, OUT_DIR / "landscape.png")
    save_lm_landing_ring_plot(r, ring_diag, lm_summary, OUT_DIR / "lm_landing_ring.png")

    print(f"\nTotal wall: {time.time() - t_main:.1f}s")


def decide_case(diag: dict, ring_diag: dict) -> str:
    """Map the diagnostics to A / B / C as in the docstring."""
    n_sobol_below_truth = diag["n_sobol_below_truth"]
    # We need to inspect the argmin_kind from outside this function — passed
    # in via `case` but cleaner: use the K=1 proxy.
    if n_sobol_below_truth >= 1:
        # Truth is not the global argmin among scored points → case (C).
        return "C"
    # Else look for Sobol-visible competing basin:
    near_lm = [r for r in ring_diag["rings"] if r["hi_deg"] <= 10.0 and r["n"] > 0]
    if any(r["n_below_lm_mse"] >= 1 for r in near_lm):
        return "B"
    return "A"


def save_landscape_plot(r: dict, ring_diag: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    sobol_mask = r["kind"] == 2
    x = r["geo_to_truth_deg"][sobol_mask]
    y = r["full_mse"][sobol_mask]
    ax.scatter(x, y, s=5, alpha=0.45, color="steelblue", linewidths=0,
               label=f"Sobol (N={N_SOBOL})")

    ax.scatter([0], [r["truth_full_mse"]], marker="*", s=160,
               color="crimson", zorder=10,
               label=f"truth = {r['truth_full_mse']:.2e}")
    twin_geo_to_truth = r["geo_to_truth_deg"][1]
    ax.scatter([twin_geo_to_truth], [r["twin_full_mse"]], marker="D", s=70,
               color="darkorange", zorder=10,
               label=f"twin = {r['twin_full_mse']:.2e}")

    ax.scatter(
        [r["geo_to_truth_deg"][r["argmin_idx"]]],
        [r["argmin_full_mse"]],
        marker="X", s=110, color="black", zorder=11,
        label=f"argmin = {r['argmin_full_mse']:.2e}",
    )

    # NEW: vertical line at LM-landing geo (~100° for seed 44).
    # Pull from any sobol candidate's geo_to_lm... no — easier: use
    # quat_geodesic_deg_batch on q_grid[0] (truth) vs q0_lm. That equals
    # the truth-to-lm distance == the lm-landing's geo_to_truth.
    # ring_diag has lm_landing_final_mse but not the geo. Recompute from r:
    #   r["geo_to_truth_deg"] indexed by argmin_idx tells us truth-to-argmin,
    #   but we need truth-to-LM-landing — that's q0_lm geo_to_truth, which
    #   isn't stored on q_grid (LM landing is NOT in the candidate set).
    # Use the same field we have: print as text annotation on the plot.
    lm_mse = ring_diag["lm_landing_final_mse"]
    # The s009 q0_err_deg gives truth-to-lm in degrees. We don't have that
    # in `ring_diag` directly; use sobol-min-geo-to-lm + ring info to put
    # a horizontal line at lm_mse instead.
    ax.axhline(lm_mse, color="purple", linestyle="-.", alpha=0.6,
               label=f"s009 LM-landing mse = {lm_mse:.2e}")
    ax.axhline(2.0 * r["truth_full_mse"], color="gray", linestyle="--",
               alpha=0.4, label=f"2×truth_mse = {2*r['truth_full_mse']:.2e}")

    ax.set_yscale("log")
    ax.set_xlabel("geodesic to truth-q0  [deg]")
    ax.set_ylabel("surrogate full-LC MSE  [mag²]")
    ax.set_title(
        f"s010 — seed {r['seed']} surrogate landscape at fixed truth-ω "
        f"({N_SOBOL} Sobol-Shoemake quats)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


def save_lm_landing_ring_plot(r: dict, ring_diag: dict, lm_summary: dict, path: Path):
    """MSE scatter against geodesic-to-LM-landing, with ring percentiles
    overplotted. Decisive view for case A/B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sobol_mask = r["kind"] == 2
    geo_lm = r["geo_to_lm_landing_deg"][sobol_mask]
    mse = r["full_mse"][sobol_mask]
    truth_mse = r["truth_full_mse"]
    lm_mse = ring_diag["lm_landing_final_mse"]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.scatter(geo_lm, mse, s=6, color="steelblue", alpha=0.45, linewidths=0,
               label=f"Sobol (N={N_SOBOL})")

    # Overlay per-ring p10 and p90 as a band.
    centres, p10s, p90s, p50s = [], [], [], []
    for ring in ring_diag["rings"]:
        if ring["n"] == 0:
            continue
        centres.append(0.5 * (ring["lo_deg"] + ring["hi_deg"]))
        p10s.append(ring["mse_p10"])
        p50s.append(ring["mse_p50"])
        p90s.append(ring["mse_p90"])
    if centres:
        ax.fill_between(centres, p10s, p90s, color="orange", alpha=0.18,
                        label="ring p10–p90")
        ax.plot(centres, p50s, color="orange", linestyle="-", marker="o",
                markersize=5, label="ring median")

    ax.axhline(truth_mse, color="crimson", linestyle=":", linewidth=1.2,
               label=f"truth_mse = {truth_mse:.2e}")
    ax.axhline(lm_mse, color="purple", linestyle="-.", linewidth=1.4,
               label=f"LM-landing mse = {lm_mse:.2e}")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("geodesic to s009 LM-landing q0  [deg]")
    ax.set_ylabel("surrogate full-LC MSE  [mag²]")
    ax.set_title(
        f"s010 — seed {r['seed']} Sobol-MSE vs distance to s009 T1 LM-landing\n"
        f"q0_err(LM→truth)={lm_summary['q0_err_deg']:.1f}°, "
        f"ω near truth (ωd={lm_summary['omega_dir_err_deg']:.2f}°, "
        f"|ωm|={abs(lm_summary['omega_mag_err_pct']):.2f}%), "
        f"final_mse={lm_mse:.2e} mag²"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
