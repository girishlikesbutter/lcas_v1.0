"""s083 — |ω| basin-sharpness probe (the s082 Day-2 #1 "N_mag sensitivity" check).

s082 found truth-nearest rank 70823/320000 (median), dominated by |ω|-magnitude
error of 20-43% — because the 5-cell LS-bracket spans 8.6x-142x (NOT the "4x /
√2-spacing" the writeup assumed; multiple LS harmonic peaks blow up hi/lo).

This probe answers the question s082 said needed answering ("re-run at N_mag=15+"):
HOW FINE must the |ω| grid be? Equivalently: how wide is the |ω| basin?

Design (cheap, ~minutes; uses the NEW elliprj Path 2 propagator):
  Layer A (basin sharpness):  hold q0 + ω-direction at TRUTH, sweep |ω| over
      ±60% of truth in fine steps. ρ_surr(|ω|) gives the Band A (ρ<2) and
      Band B (ρ<4) basin half-widths in %. This is a forward-model sensitivity
      measurement (à la s073f), NOT a search-yield claim — truth is used only
      to centre the sweep.
  Layer B (realistic grid):   hold q0 + ω-direction at the BEST cells the s082
      grid actually achieved (min q_geo ~16-33°, min dir_err ~1-3°, pulled from
      the cached candidates.npz), sweep |ω| the same way. This is what a denser
      N_mag could actually reach given the existing q/dir resolution.

For both layers we also report the rank of the best swept candidate inside the
cached 320k MSE distribution. Caveat: cached MSEs used the OLD solve_ivp Path 2;
these use the NEW elliprj path. s074 gates show the two agree to <=2.75e-10 in q
across all cases, so the MSE difference is negligible and the rank is valid.

Run only the 3 s082 seeds (the uncertain ones). No truth in any production grid.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from lib.hifi_render import build_context  # noqa: E402
from lib.c_t_pipeline import compute_j2000_units  # noqa: E402
from lib.jacobi_propagator import propagate_jacobi_path2  # noqa: E402
from lib.surrogate_eval import get_model  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SEEDS = [(116, "LAM-slow"), (119, "LAM-fast"), (103, "SAM")]
REL_GRID = np.linspace(0.40, 1.60, 241)  # ±60% of truth-|ω|, 0.5% steps
OUT_DIR = SURVEY_DIR / "results" / "s083"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL = get_model()


def quat_hist_to_R(q_hist_wxyz):
    q_xyzw = q_hist_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(q_xyzw).as_matrix()


def score(q0, omega0, ctx):
    """Full-LC surrogate MSE for one (q0, ω0). Returns (mse, rho_surr)."""
    q_hist, _ = propagate_jacobi_path2(
        q0, omega0, ctx["inertia_tensor"], ctx["observation_times"]
    )
    R_hist = quat_hist_to_R(q_hist)
    k1 = np.einsum("nij,nj->ni", R_hist, ctx["sun_unit"])
    k2 = np.einsum("nij,nj->ni", R_hist, ctx["obs_unit"])
    pred = _MODEL.predict_magnitude(k1, k2, 0.0, 15.0, ctx["obs_dist"])
    diff = pred - ctx["mag_observed"]
    mask = np.isfinite(diff)
    mse = float(np.mean(diff[mask] ** 2)) if mask.sum() else float("inf")
    return mse, float(np.sqrt(mse) / 0.05)


def basin_halfwidth(rel, rho, band_rho):
    """Half-width (in % of truth-|ω|) of the contiguous ρ<band_rho region
    containing rel=1.0. Returns (lo_pct, hi_pct) offsets from truth, or None if
    truth itself is outside the band."""
    i_truth = int(np.argmin(np.abs(rel - 1.0)))
    if rho[i_truth] >= band_rho:
        return None
    lo = i_truth
    while lo > 0 and rho[lo - 1] < band_rho:
        lo -= 1
    hi = i_truth
    while hi < len(rel) - 1 and rho[hi + 1] < band_rho:
        hi += 1
    return (float((rel[lo] - 1.0) * 100.0), float((rel[hi] - 1.0) * 100.0))


def run_seed(seed, label):
    print(f"\n=== s083 |ω| basin probe — seed {seed} ({label}) ===", flush=True)
    ctx_raw = build_context(seed)
    sun_unit, obs_unit = compute_j2000_units(
        ctx_raw["sun_pos"], ctx_raw["obs_pos"], ctx_raw["sat_pos"]
    )
    ctx = {
        "inertia_tensor": ctx_raw["inertia_tensor"],
        "observation_times": ctx_raw["observation_times"],
        "sun_unit": sun_unit,
        "obs_unit": obs_unit,
        "obs_dist": ctx_raw["obs_dist"],
        "mag_observed": ctx_raw["mag_hifi_truth"],
    }
    q0_truth = ctx_raw["q0_truth"]
    om_truth = ctx_raw["omega0_truth_rad"]
    om_mag_truth = float(np.linalg.norm(om_truth))
    dir_truth = om_truth / om_mag_truth

    # Sanity anchor: truth exactly
    mse_truth, rho_truth = score(q0_truth, om_truth, ctx)
    print(f"  truth-exact: ρ_surr={rho_truth:.3f} (MSE={mse_truth:.5f})", flush=True)

    # ---- Layer A: q0+dir at truth, sweep |ω| ----
    t0 = time.time()
    rhoA = np.empty(len(REL_GRID))
    mseA = np.empty(len(REL_GRID))
    for i, r in enumerate(REL_GRID):
        mseA[i], rhoA[i] = score(q0_truth, dir_truth * (om_mag_truth * r), ctx)
    wallA = time.time() - t0

    # ---- Layer B: q0+dir at best achievable grid cells, sweep |ω| ----
    cnpz = np.load(SURVEY_DIR / "results" / "s082" / f"seed_{seed:03d}" / "candidates.npz")
    q_cands = cnpz["q_cands"]
    om_cands = cnpz["om_cands"]
    cached_mses = cnpz["mses"]
    q_geo = cnpz["q_geo_deg"]
    dir_err = cnpz["om_dir_err_deg"]
    i_qbest = int(np.argmin(q_geo))
    i_dbest = int(np.argmin(dir_err))
    q_best = q_cands[i_qbest]
    dir_best = om_cands[i_dbest] / np.linalg.norm(om_cands[i_dbest])
    q_best_geo = float(q_geo[i_qbest])
    dir_best_err = float(dir_err[i_dbest])
    print(
        f"  best-grid q_geo={q_best_geo:.2f}°, best-grid dir_err={dir_best_err:.2f}°",
        flush=True,
    )

    rhoB = np.empty(len(REL_GRID))
    mseB = np.empty(len(REL_GRID))
    for i, r in enumerate(REL_GRID):
        mseB[i], rhoB[i] = score(q_best, dir_best * (om_mag_truth * r), ctx)

    # Ranks of best swept candidate in cached 320k distribution
    finite = np.isfinite(cached_mses)
    bestA_mse = float(np.min(mseA))
    bestB_mse = float(np.min(mseB))
    rankA = int(1 + np.sum(cached_mses[finite] < bestA_mse))
    rankB = int(1 + np.sum(cached_mses[finite] < bestB_mse))

    # Basin half-widths
    bandA_A = basin_halfwidth(REL_GRID, rhoA, 2.0)
    bandB_A = basin_halfwidth(REL_GRID, rhoA, 4.0)
    bandA_B = basin_halfwidth(REL_GRID, rhoB, 2.0)
    bandB_B = basin_halfwidth(REL_GRID, rhoB, 4.0)

    res = {
        "seed": seed,
        "label": label,
        "omega_mag_truth_dps": float(np.degrees(om_mag_truth)),
        "truth_exact": {"mse": mse_truth, "rho_surr": rho_truth},
        "layer_A_truth_q_dir": {
            "min_rho_surr": float(np.min(rhoA)),
            "rho_at_truth_mag": float(rhoA[int(np.argmin(np.abs(REL_GRID - 1.0)))]),
            "bandA_halfwidth_pct": bandA_A,
            "bandB_halfwidth_pct": bandB_A,
            "rank_in_cached_320k": rankA,
            "wall_sec": wallA,
        },
        "layer_B_bestgrid_q_dir": {
            "q_best_geo_deg": q_best_geo,
            "dir_best_err_deg": dir_best_err,
            "min_rho_surr": float(np.min(rhoB)),
            "bandA_halfwidth_pct": bandA_B,
            "bandB_halfwidth_pct": bandB_B,
            "rank_in_cached_320k": rankB,
        },
    }
    with open(OUT_DIR / f"seed_{seed:03d}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot((REL_GRID - 1) * 100, rhoA, color="tab:blue", label="Layer A: q,dir=truth")
    ax.plot((REL_GRID - 1) * 100, rhoB, color="tab:orange",
            label=f"Layer B: q_geo={q_best_geo:.0f}°, dir={dir_best_err:.1f}°")
    ax.axhline(2.0, color="green", ls=":", label="Band A (ρ=2)")
    ax.axhline(4.0, color="red", ls=":", label="Band B (ρ=4)")
    ax.axvline(0.0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("|ω| offset from truth (%)")
    ax.set_ylabel("ρ_surr")
    ax.set_ylim(0, max(8, float(np.percentile(np.r_[rhoA, rhoB], 90))))
    ax.set_title(f"s083 seed {seed} ({label}) — |ω| basin at fixed q,dir")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pp = OUT_DIR / f"seed_{seed:03d}_omega_basin.png"
    fig.savefig(pp, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"  Layer A: min ρ_surr={res['layer_A_truth_q_dir']['min_rho_surr']:.3f}, "
          f"BandA hw={bandA_A}, BandB hw={bandB_A}, rank={rankA}", flush=True)
    print(f"  Layer B: min ρ_surr={res['layer_B_bestgrid_q_dir']['min_rho_surr']:.3f}, "
          f"BandA hw={bandA_B}, BandB hw={bandB_B}, rank={rankB}", flush=True)
    print(f"  Saved: {OUT_DIR / f'seed_{seed:03d}.json'}", flush=True)
    print(f"  Saved: {pp}", flush=True)
    return res


def main():
    t0 = time.time()
    allres = [run_seed(s, lab) for s, lab in SEEDS]
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"seeds": allres}, f, indent=2, default=float)
    print(f"\n=== s083 done in {time.time()-t0:.1f}s ===", flush=True)
    print(f"Saved: {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
