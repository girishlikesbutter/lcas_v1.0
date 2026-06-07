"""s006 — seed-28 surrogate landscape probe at fixed truth-ω.

Survey question Q4d. s005 showed seed 28 has a tight joint-LM basin (~2°
in q0_geodesic): T1 (2°), R2 (1.2°), R3 (1.9°) converge, but T2 (5°) and
7 other ICs escape to MSE 5-9 mag² with q0_err 20°-113°. The escapes
land at finite distances from truth — NOT at twin (twin would be
~180° away with the IS-901 q_180x convention) — so deep competing
minima exist somewhere on the surrogate-MSE surface even at exactly
truth-ω.

Question: are those competing basins visible at s002-class Sobol
density (2046 candidates), or are they sub-Sobol-resolution and only
LM-discoverable?

Decision tree:
  (A) truth still global argmin, n_sobol_below_truth = 0, sobol-min-geo
      to truth ~4-15° (s002 baseline range). Competing basins are
      narrow and sub-Sobol-resolution.  → Q4c implication: Sobol
      density alone won't seed seed-28-class basins; mandatory LM
      polish per Sobol candidate.
  (B) n_sobol_below_truth ≥ 1 OR multiple Sobol points with full_mse
      < 2 × truth_mse at >30° geodesic from truth. Competing basins
      visible at Sobol resolution.  → Q4c implication: define basins
      by enumeration; counting truth-basin landings is insufficient.
  (C) argmin NOT at truth. Would refute s002's 8/8 baseline-correctness
      claim by extension to seed 28.

Method:
  - Single seed: 28 (PA-high, s004 alignment-cost multi-basin pathology,
    s005 tight ~2° joint-LM basin).
  - 2046 Sobol-Shoemake quats + truth + twin (180°-body-x · q0_truth)
    → 2048 candidates.
  - SOBOL_SEED = 42 (same as s002) so the Sobol point set is identical
    to s002's seed-28-equivalent draw — direct comparability.
  - ω fixed at truth-ω.
  - Score: surrogate full-LC MSE vs cached mag_hifi.
  - Pool(8), BLAS=1 in workers (project memory).

Diagnostics (beyond s002):
  - "competing basin" candidates: count of Sobol points with
    full_mse < K × truth_mse for K ∈ {2, 5, 10, 100}, with their
    geodesic-distance distribution. (Catches case B even when truth
    remains the argmin.)
  - sobol-min-geo-to-truth — does seed 28's basin reach Sobol's
    coverage radius or sit below it?
  - Sobol candidates near (within 5°/10° of) the s005 LM-escape
    landing q0s — does Sobol see those landing zones at all?

Outputs:
  - results/s006/landscape.npz      all candidate quats + MSEs + diagnostics
  - results/s006/summary.json       argmin / truth-rank / competing-basin stats
  - results/s006/landscape.png      MSE vs geodesic-to-truth panel
  - results/s006/competing_basins.png  hist + scatter of Sobol points by MSE band
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
OUT_DIR = SURVEY_DIR / "results" / "s006"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 28
N_SOBOL = 2046  # +2 for truth-q0 and twin → 2048 total candidates
SOBOL_SEED = 42  # MATCH s002 — Sobol point set is identical.
N_WORKERS = 8

# Twin: 180° about body x (concepts/twin_degeneracy.md).
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Diagnostic MSE thresholds (multiples of truth_mse).
COMPETING_K = [2.0, 5.0, 10.0, 100.0]


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
    """Shoemake's uniform-on-S^3 mapping (matches s002)."""
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
    """Truth + twin + 2046 Sobol → (2048, 4) wxyz array."""
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


def run_seed(seed: int, inertia_tensor: np.ndarray) -> dict:
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

    argmin = int(np.argmin(full_mse))
    truth_mse = float(full_mse[0])
    twin_mse = float(full_mse[1])
    argmin_geo = float(geo_to_truth[argmin])
    argmin_geo_twin = float(geo_to_twin[argmin])

    print(
        f"seed {seed:3d}  N={n_cand}  "
        f"truth_mse={truth_mse:.4e}  twin_mse={twin_mse:.4e}  "
        f"argmin={argmin} ({'truth' if argmin == 0 else 'twin' if argmin == 1 else 'sobol'})  "
        f"argmin_mse={float(full_mse[argmin]):.4e}  "
        f"argmin_geo_to_truth={argmin_geo:.1f}°  "
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
        "argmin_idx": argmin,
        "argmin_full_mse": float(full_mse[argmin]),
        "argmin_geo_to_truth_deg": argmin_geo,
        "argmin_geo_to_twin_deg": argmin_geo_twin,
        "truth_full_mse": truth_mse,
        "twin_full_mse": twin_mse,
        "wall_s": wall,
    }


def diagnose_competing_basins(r: dict) -> dict:
    """Sobol-visibility diagnostic: count candidates beating truth_mse by
    multiplicative factor K, and look at where they live geodesically."""
    sobol_mask = r["kind"] == 2
    sobol_full = r["full_mse"][sobol_mask]
    sobol_geo = r["geo_to_truth_deg"][sobol_mask]
    truth_mse = r["truth_full_mse"]

    # Sobol min geodesic to truth (how close does Sobol get?).
    sobol_min_geo = float(np.min(sobol_geo))

    # n Sobol candidates with full_mse strictly below truth.
    n_sobol_below_truth = int(np.sum(sobol_full < truth_mse))

    # For each K, how many Sobol points have mse < K · truth_mse, and
    # of those, how many sit > 30° from truth? "Far" candidates are
    # the competing-basin signature.
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


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    print(f"Running s006 surrogate-landscape probe on seed {SEED} "
          f"({N_SOBOL + 2} candidates, Pool({N_WORKERS}), BLAS=1).", flush=True)

    r = run_seed(SEED, inertia_tensor)
    diag = diagnose_competing_basins(r)

    # Print key diagnostic line.
    print()
    print(f"=== s006 seed {SEED} diagnostic ===")
    print(f"  truth_full_mse       = {diag['truth_full_mse']:.4e}")
    print(f"  twin_full_mse        = {diag['twin_full_mse']:.4e}")
    print(f"  sobol_min_geo_truth  = {diag['sobol_min_geo_to_truth_deg']:.2f}°")
    print(f"  n_sobol_below_truth  = {diag['n_sobol_below_truth']}/{N_SOBOL}")
    for k_label, c in diag["competing_basin_counts"].items():
        print(f"  competing  {k_label:5s} (mse < {c['K']:.0f}× truth_mse): "
              f"{c['n_sobol_below_K_truth']:4d} sobol  "
              f"({c['n_sobol_below_K_truth_far_from_truth']:4d} > 30° from truth)  "
              f"geo[{c['geo_min_deg']:5.1f}, {c['geo_max_deg']:5.1f}], "
              f"med {c['geo_median_deg']:5.1f}°")
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
    )
    print(f"Saved: {npz_path}")

    # ── Save summary JSON ──
    summary = {
        "seed": SEED,
        "n_sobol": N_SOBOL,
        "sobol_seed": SOBOL_SEED,
        "n_candidates": r["n_candidates"],
        "argmin_idx": r["argmin_idx"],
        "argmin_kind": int(r["kind"][r["argmin_idx"]]),  # 0 truth 1 twin 2 sobol
        "argmin_full_mse": r["argmin_full_mse"],
        "argmin_geo_to_truth_deg": r["argmin_geo_to_truth_deg"],
        "argmin_geo_to_twin_deg": r["argmin_geo_to_twin_deg"],
        "truth_full_mse": r["truth_full_mse"],
        "twin_full_mse": r["twin_full_mse"],
        "diagnostics": diag,
        "wall_s": r["wall_s"],
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # ── Plots ──
    save_landscape_plot(r, OUT_DIR / "landscape.png")
    save_competing_basins_plot(r, diag, OUT_DIR / "competing_basins.png")

    print(f"\nTotal wall: {time.time() - t_main:.1f}s")


def save_landscape_plot(r: dict, path: Path):
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
    twin_geo = r["geo_to_twin_deg"][0]  # twin index 1 -> use index-1 row?
    # Actually for the truth row, geo_to_twin = 180° (truth is far from twin).
    # We want to plot the twin point's geo-to-truth; that's row 1.
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

    # Reference horizontal: 2× truth_mse
    ax.axhline(2.0 * r["truth_full_mse"], color="gray", linestyle="--",
               alpha=0.4, label=f"2×truth_mse = {2*r['truth_full_mse']:.2e}")

    ax.set_yscale("log")
    ax.set_xlabel("geodesic to truth-q0  [deg]")
    ax.set_ylabel("surrogate full-LC MSE  [mag²]")
    ax.set_title(
        f"s006 — seed {r['seed']} surrogate landscape at fixed truth-ω "
        f"({N_SOBOL} Sobol-Shoemake quats)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


def save_competing_basins_plot(r: dict, diag: dict, path: Path):
    """Two-panel: histogram of MSE-bin counts vs geodesic-to-truth, and
    scatter highlighting "competing" candidates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sobol_mask = r["kind"] == 2
    geo = r["geo_to_truth_deg"][sobol_mask]
    mse = r["full_mse"][sobol_mask]
    truth_mse = r["truth_full_mse"]

    bands = [
        ("< truth_mse",      mse < truth_mse,                              "tab:red"),
        ("[truth, 2×truth)", (mse >= truth_mse) & (mse < 2 * truth_mse),  "tab:orange"),
        ("[2×, 10×truth)",   (mse >= 2 * truth_mse) & (mse < 10 * truth_mse), "tab:olive"),
        ("[10×, 100×truth)", (mse >= 10 * truth_mse) & (mse < 100 * truth_mse), "steelblue"),
        ("≥ 100× truth",     mse >= 100 * truth_mse,                        "lightgray"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: stacked histogram of geodesic-to-truth, colored by MSE band.
    ax = axes[0]
    bins = np.arange(0, 185, 5)
    bottom = np.zeros(len(bins) - 1)
    for label, mask, color in bands:
        if not np.any(mask):
            continue
        counts, _ = np.histogram(geo[mask], bins=bins)
        ax.bar(bins[:-1], counts, width=5, bottom=bottom, color=color,
               label=f"{label}  (n={int(np.sum(mask))})", align="edge",
               edgecolor="white", linewidth=0.2)
        bottom = bottom + counts
    ax.set_xlabel("geodesic to truth-q0  [deg]")
    ax.set_ylabel("Sobol-candidate count per 5° bin")
    ax.set_title("Sobol coverage stacked by MSE band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: scatter — like landscape.png but highlighting competing-basin
    # bands explicitly.
    ax = axes[1]
    for label, mask, color in bands[::-1]:  # plot lightgray first → red on top
        if not np.any(mask):
            continue
        ax.scatter(geo[mask], mse[mask], s=8, color=color, alpha=0.6,
                   linewidths=0, label=f"{label}  (n={int(np.sum(mask))})")
    ax.scatter([0], [truth_mse], marker="*", s=180, color="crimson",
               edgecolors="white", linewidths=0.8, zorder=20,
               label=f"truth_mse = {truth_mse:.2e}")
    ax.axhline(truth_mse, color="crimson", linestyle=":", alpha=0.5)
    ax.axhline(2 * truth_mse, color="darkorange", linestyle=":", alpha=0.4)
    ax.axhline(10 * truth_mse, color="olive", linestyle=":", alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("geodesic to truth-q0  [deg]")
    ax.set_ylabel("surrogate full-LC MSE  [mag²]")
    ax.set_title("Sobol scatter colored by MSE band")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"s006 — seed {r['seed']} competing-basin diagnostic at fixed truth-ω",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
