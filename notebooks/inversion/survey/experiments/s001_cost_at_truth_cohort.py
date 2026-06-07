"""s001 — cost-at-truth, cohort-scale (100 m048 seeds, post-fix truth).

Survey question Q1. For each of four cost surfaces — surrogate full-LC MSE,
surrogate bright-MSE, m103 alignment cost, lofi peak-match — compute the
value at the truth state `(q0_truth, ω_truth)` for every seed in the post-fix
m048 cohort. Pure post-hoc scoring; no propagation, no search.

Why no propagation: every cached `traj_seedXXX.npz` already stores the
truth-attitude body-frame quantities (`k1_body`, `k2_body`, `pab_body`,
`mag_hifi`, `mag_lofi`). The cost-at-truth values are functions of these
cached arrays plus a peak-finder on the canonical noisy observed LC.

Faithful reproduction of m103 machinery (copied + adapted from
`notebooks/inversion/11_casadi_formulation/m103_hybrid.py`, NOT imported):
  - canonical observed LC: rng=default_rng(42), sigma=0.05
  - peak detection: find_peaks(-observed_lc, distance=5, prominence=0.3)
  - spec peaks: subset where observed_lc < 9.0
  - anchor: savgol-smoothed brightest spec peak (tiebreak on epoch index)
  - constraint epochs: spec peaks excluding anchor
  - alignment cost = sum_{ci in constraints} w * (1 - max(pab_body[ci] · normals[allowed]))^2
    with w = CONSTRAINT_WEIGHT = 10.0, allowed = get_allowed_normals(observed_lc[ci]).
  - lofi peak-match: count obs_peaks (full peaks_idx, NOT just spec) matched by
    cand_peaks = find_peaks(-mag_lofi, distance=3, prominence=0.2) within ±3 epochs.

Outputs:
  - results/s001/per_seed.csv         one row per seed, all metrics
  - results/s001/summary.json         population statistics
  - results/s001/cost_at_truth_distributions.png    4-panel histogram
"""

# BLAS = 1 per project memory; we don't use Pool here but keep it tidy in case
# surrogate-internal threading would interact poorly with future parallelism.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

from lib import surrogate_eval, traj_load  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent

# --- m103-era constants (copied from m103_hybrid.py, NOT imported) ---
CONSTRAINT_WEIGHT = 10.0
PEAK_WINDOW = 3
NOISE_SEED = 42
NOISE_SIGMA = 0.05
SPEC_THRESHOLD = 9.0
OBS_PEAK_DISTANCE = 5
OBS_PEAK_PROMINENCE = 0.3
PRED_PEAK_DISTANCE = 3
PRED_PEAK_PROMINENCE = 0.2
SAVGOL_WINDOW = 7
SAVGOL_POLY = 3
SURROGATE_BRIGHT_THRESHOLD = 11.0  # mag; survey default for "bright" subset

OUT_DIR = SURVEY_DIR / "results" / "s001"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# unique_normals are a satellite-static array — same for all seeds — pulled
# from the m048 master NPZ. Treat as a fixed constant of the satellite.
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)


def get_allowed_normals(mag: float):
    """m103 magnitude-banded allowed-normal lookup (m103_hybrid.py:104)."""
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))


def canonical_observed_lc(mag_hifi: np.ndarray) -> np.ndarray:
    """Canonical noisy observed LC (rng=default_rng(42), sigma=0.05).

    Single source of truth — reproduces the parent project's
    `lib/traj_source.canonical_observed_lc` exactly.
    """
    rng = np.random.default_rng(NOISE_SEED)
    return mag_hifi + rng.normal(0.0, NOISE_SIGMA, mag_hifi.shape[0])


def select_anchor(observed_lc: np.ndarray, spec_peaks: np.ndarray) -> int:
    """Replicate m103_hybrid.py:198-204 anchor-selection logic.

    - Smooth observed LC with savgol.
    - Take the spec peak with the lowest smoothed magnitude (= brightest).
    - On ties (within 0.05 mag), prefer the earlier-epoch peak.
    """
    smoothed = savgol_filter(observed_lc, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
    smooth_mags = smoothed[spec_peaks]
    sr = np.argsort(smooth_mags)
    if len(sr) >= 2 and abs(smooth_mags[sr[0]] - smooth_mags[sr[1]]) < 0.05:
        anchor_rank = sr[:2][np.argmin(spec_peaks[sr[:2]])]
    else:
        anchor_rank = int(sr[0])
    return int(spec_peaks[anchor_rank])


def alignment_cost_at_truth(
    pab_body: np.ndarray,
    observed_lc: np.ndarray,
    constraint_epochs: np.ndarray,
    unique_normals: np.ndarray,
) -> tuple[float, int]:
    """m103 alignment cost evaluated at truth.

    At truth, R(q_truth_at_constraint_epoch) @ pab_J2000[ci] = pab_body[ci]
    (verified at machine precision on seed 6, see s001 design notes), so
    the cost reduces to a pure dot-product sum over constraint epochs:
        cost = sum_{ci in constraints} w * (1 - max_a (pab_body[ci] · normals[allowed_a]))^2

    Returns (total_cost, n_constraints).
    """
    if len(constraint_epochs) == 0:
        return 0.0, 0
    total = 0.0
    for ci in constraint_epochs:
        allowed = get_allowed_normals(observed_lc[ci])
        bds = float(np.max(pab_body[ci] @ unique_normals[allowed].T))
        total += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
    return total, len(constraint_epochs)


def lofi_peak_match_at_truth(
    observed_lc: np.ndarray,
    mag_lofi_truth: np.ndarray,
) -> tuple[int, int]:
    """m103 peak-match score evaluated at truth (mag_lofi cached at truth).

    Returns (n_matched, n_obs_peaks). Higher n_matched is better.
    """
    obs_peaks, _ = find_peaks(
        -observed_lc, distance=OBS_PEAK_DISTANCE, prominence=OBS_PEAK_PROMINENCE
    )
    cand_peaks, _ = find_peaks(
        -mag_lofi_truth, distance=PRED_PEAK_DISTANCE, prominence=PRED_PEAK_PROMINENCE
    )
    cps = set(int(c) for c in cand_peaks)
    nm = sum(
        1
        for op in obs_peaks
        if any((int(op) + o) in cps for o in range(-PEAK_WINDOW, PEAK_WINDOW + 1))
    )
    return int(nm), int(len(obs_peaks))


def main():
    t0 = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    unique_normals = np.asarray(master["unique_normals"])
    assert unique_normals.shape == (10, 3), f"unexpected normals: {unique_normals.shape}"

    # Eager-load surrogate once.
    print("Loading surrogate model...", flush=True)
    surrogate_eval.get_model()
    print(f"Surrogate ready ({time.time() - t0:.1f}s).")

    seeds = traj_load.list_seeds()
    print(f"Scoring {len(seeds)} seeds at truth...")

    rows = []
    for seed in seeds:
        d = traj_load.load_truth(seed)
        omega_mag_dps = float(d["omega_mag_dps"])
        mag_hifi = d["mag_hifi"]
        mag_lofi = d["mag_lofi"]
        k1_body = d["k1_body"]
        k2_body = d["k2_body"]
        obs_dist = d["obs_dist"]
        phase_angle = d["phase_angle_3d"]
        pab_body = d["pab_body"]

        observed_lc = canonical_observed_lc(mag_hifi)

        # Surrogate at truth k1/k2.
        pred_surr = surrogate_eval.predict(k1_body, k2_body, obs_dist)
        surr_full_mse_vs_hifi = surrogate_eval.full_lc_mse(pred_surr, mag_hifi)
        surr_bright_mse_vs_hifi = surrogate_eval.bright_mse(
            pred_surr, mag_hifi, SURROGATE_BRIGHT_THRESHOLD
        )
        surr_full_mse_vs_obs = surrogate_eval.full_lc_mse(pred_surr, observed_lc)
        surr_bright_mse_vs_obs = surrogate_eval.bright_mse(
            pred_surr, observed_lc, SURROGATE_BRIGHT_THRESHOLD
        )

        # Peak detection on observed LC for anchor + alignment-cost setup.
        peaks_idx, _ = find_peaks(
            -observed_lc, distance=OBS_PEAK_DISTANCE, prominence=OBS_PEAK_PROMINENCE
        )
        spec_peaks = peaks_idx[observed_lc[peaks_idx] < SPEC_THRESHOLD]

        if len(spec_peaks) == 0:
            anchor_idx = -1
            constraint_epochs = np.array([], dtype=int)
            align_cost = float("nan")
            n_constraints = 0
        elif len(spec_peaks) == 1:
            # Only one spec peak → it's the anchor; no constraints.
            anchor_idx = int(spec_peaks[0])
            constraint_epochs = np.array([], dtype=int)
            align_cost = 0.0  # by definition the empty sum
            n_constraints = 0
        else:
            anchor_idx = select_anchor(observed_lc, spec_peaks)
            constraint_epochs = spec_peaks[spec_peaks != anchor_idx]
            align_cost, n_constraints = alignment_cost_at_truth(
                pab_body, observed_lc, constraint_epochs, unique_normals
            )

        align_cost_per_constraint = (
            float("nan") if n_constraints == 0 else align_cost / n_constraints
        )

        # Lofi peak-match at truth.
        n_matched, n_obs_peaks = lofi_peak_match_at_truth(observed_lc, mag_lofi)
        match_frac = float("nan") if n_obs_peaks == 0 else n_matched / n_obs_peaks

        row = {
            "seed": seed,
            "omega_mag_dps": omega_mag_dps,
            "phase_angle_min_deg": float(np.min(phase_angle)),
            "phase_angle_med_deg": float(np.median(phase_angle)),
            "phase_angle_max_deg": float(np.max(phase_angle)),
            "n_total_peaks": int(len(peaks_idx)),
            "n_spec_peaks": int(len(spec_peaks)),
            "n_constraints": int(n_constraints),
            "anchor_idx": int(anchor_idx),
            "surr_full_mse_vs_hifi": float(surr_full_mse_vs_hifi),
            "surr_bright_mse_vs_hifi": float(surr_bright_mse_vs_hifi),
            "surr_full_mse_vs_obs": float(surr_full_mse_vs_obs),
            "surr_bright_mse_vs_obs": float(surr_bright_mse_vs_obs),
            "align_cost": float(align_cost),
            "align_cost_per_constraint": align_cost_per_constraint,
            "lofi_n_matched": int(n_matched),
            "lofi_n_obs_peaks": int(n_obs_peaks),
            "lofi_match_frac": match_frac,
        }
        rows.append(row)
        if seed % 10 == 0:
            print(
                f"  seed {seed:03d}: "
                f"surr_full={surr_full_mse_vs_hifi:.4f} "
                f"surr_bright={surr_bright_mse_vs_hifi:.4f} "
                f"align={align_cost:.3f} (n={n_constraints}) "
                f"lofi={n_matched}/{n_obs_peaks}",
                flush=True,
            )

    # --- Save per-seed CSV ---
    csv_path = OUT_DIR / "per_seed.csv"
    with open(csv_path, "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r[c]) for c in cols) + "\n")
    print(f"Saved: {csv_path}")

    # --- Save summary JSON ---
    summary = build_summary(rows)
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # --- Save 4-panel distribution plot ---
    plot_path = OUT_DIR / "cost_at_truth_distributions.png"
    save_distributions_plot(rows, plot_path)
    print(f"Saved: {plot_path}")

    print(f"\nTotal wall: {time.time() - t0:.1f}s for {len(rows)} seeds.")


def _fmt(x):
    if isinstance(x, float):
        if np.isnan(x):
            return "nan"
        return f"{x:.6g}"
    return str(x)


def build_summary(rows):
    arr = lambda key: np.array([r[key] for r in rows], dtype=float)
    surr_full = arr("surr_full_mse_vs_hifi")
    surr_bright = arr("surr_bright_mse_vs_hifi")
    align = arr("align_cost")
    align_pc = arr("align_cost_per_constraint")
    match_frac = arr("lofi_match_frac")
    n_matched = arr("lofi_n_matched")
    n_constraints = arr("n_constraints")
    n_spec = arr("n_spec_peaks")

    def stats(a):
        a = np.asarray(a, dtype=float)
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            return {"n_finite": 0}
        return {
            "n_finite": int(len(finite)),
            "min": float(np.min(finite)),
            "p05": float(np.percentile(finite, 5)),
            "p25": float(np.percentile(finite, 25)),
            "median": float(np.median(finite)),
            "mean": float(np.mean(finite)),
            "p75": float(np.percentile(finite, 75)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }

    summary = {
        "n_seeds": len(rows),
        "surrogate_full_lc_mse_vs_hifi": stats(surr_full),
        "surrogate_bright_mse_vs_hifi": stats(surr_bright),
        "surrogate_full_lc_mse_vs_obs": stats(arr("surr_full_mse_vs_obs")),
        "surrogate_bright_mse_vs_obs": stats(arr("surr_bright_mse_vs_obs")),
        "align_cost": stats(align),
        "align_cost_per_constraint": stats(align_pc),
        "lofi_match_frac": stats(match_frac),
        "lofi_n_matched": stats(n_matched),
        "n_constraints": stats(n_constraints),
        "n_spec_peaks": stats(n_spec),
    }

    # Notable seeds for downstream cross-references.
    summary["seeds_zero_constraints"] = sorted(
        int(r["seed"]) for r in rows if r["n_constraints"] == 0
    )
    summary["seeds_all_peaks_matched"] = sorted(
        int(r["seed"]) for r in rows if r["lofi_n_obs_peaks"] > 0 and r["lofi_match_frac"] == 1.0
    )
    # The surrogate-MSE outliers (highest 5 vs hifi).
    sorted_by_full = sorted(rows, key=lambda r: r["surr_full_mse_vs_hifi"], reverse=True)
    summary["top5_surrogate_full_mse"] = [
        {"seed": int(r["seed"]), "mse": float(r["surr_full_mse_vs_hifi"])}
        for r in sorted_by_full[:5]
    ]
    summary["bottom5_surrogate_full_mse"] = [
        {"seed": int(r["seed"]), "mse": float(r["surr_full_mse_vs_hifi"])}
        for r in sorted_by_full[-5:]
    ]
    sorted_by_align = sorted(
        [r for r in rows if r["n_constraints"] > 0],
        key=lambda r: r["align_cost_per_constraint"],
        reverse=True,
    )
    summary["top5_align_cost_per_constraint"] = [
        {
            "seed": int(r["seed"]),
            "align_cost": float(r["align_cost"]),
            "align_per_constraint": float(r["align_cost_per_constraint"]),
            "n_constraints": int(r["n_constraints"]),
        }
        for r in sorted_by_align[:5]
    ]
    summary["bottom5_align_cost_per_constraint"] = [
        {
            "seed": int(r["seed"]),
            "align_cost": float(r["align_cost"]),
            "align_per_constraint": float(r["align_cost_per_constraint"]),
            "n_constraints": int(r["n_constraints"]),
        }
        for r in sorted_by_align[-5:]
    ]
    return summary


def save_distributions_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surr_full = np.array([r["surr_full_mse_vs_hifi"] for r in rows])
    surr_bright = np.array([r["surr_bright_mse_vs_hifi"] for r in rows], dtype=float)
    align_pc = np.array([r["align_cost_per_constraint"] for r in rows], dtype=float)
    match_frac = np.array([r["lofi_match_frac"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(surr_full, bins=30, color="steelblue", edgecolor="k", alpha=0.85)
    ax.axvline(np.median(surr_full), color="crimson", linestyle="--", label=f"median={np.median(surr_full):.4f}")
    ax.set_xlabel("MSE [mag²]")
    ax.set_ylabel("# seeds")
    ax.set_title("Surrogate full-LC MSE at truth (vs mag_hifi)")
    ax.legend()
    ax.set_yscale("linear")

    ax = axes[0, 1]
    surr_bright_finite = surr_bright[np.isfinite(surr_bright)]
    ax.hist(surr_bright_finite, bins=30, color="darkgreen", edgecolor="k", alpha=0.85)
    if len(surr_bright_finite) > 0:
        ax.axvline(
            np.median(surr_bright_finite), color="crimson", linestyle="--",
            label=f"median={np.median(surr_bright_finite):.4f}"
        )
    ax.text(
        0.95, 0.95,
        f"{int(np.sum(~np.isfinite(surr_bright)))} seeds w/ no mag_hifi<11",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="gray",
    )
    ax.set_xlabel("MSE [mag²]  (mag_hifi < 11)")
    ax.set_ylabel("# seeds")
    ax.set_title("Surrogate bright-MSE at truth (vs mag_hifi)")
    ax.legend()

    ax = axes[1, 0]
    finite_pc = align_pc[np.isfinite(align_pc)]
    if len(finite_pc) > 0:
        ax.hist(finite_pc, bins=30, color="darkorange", edgecolor="k", alpha=0.85)
        ax.axvline(
            np.median(finite_pc), color="crimson", linestyle="--",
            label=f"median={np.median(finite_pc):.3f}"
        )
    ax.set_xlabel("alignment cost / constraint  [unitless, w=10]")
    ax.set_ylabel("# seeds")
    ax.set_title("m103 alignment cost (per-constraint) at truth")
    n_zero_constraints = int(np.sum(np.isnan(align_pc)))
    ax.text(
        0.95, 0.95, f"{n_zero_constraints} seeds w/ 0 constraints",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="gray",
    )
    ax.legend()

    ax = axes[1, 1]
    finite_mf = match_frac[np.isfinite(match_frac)]
    if len(finite_mf) > 0:
        ax.hist(finite_mf, bins=20, color="purple", edgecolor="k", alpha=0.85)
        ax.axvline(
            np.median(finite_mf), color="crimson", linestyle="--",
            label=f"median={np.median(finite_mf):.2f}"
        )
    ax.set_xlabel("matched / observed peaks")
    ax.set_ylabel("# seeds")
    ax.set_title("lofi peak-match fraction at truth (±3 epoch window)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend()

    fig.suptitle(
        "s001 — cost-at-truth distributions across 100 m048 seeds (post-fix forward model)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
