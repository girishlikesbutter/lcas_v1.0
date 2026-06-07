"""s055b — L̂ (inertial angular momentum direction) regression from LC features.

Hypothesis: L̂ in J2000 — the inertial axis around which ω̂(t) precesses
under torque-free rigid-body motion — is partially recoverable from
LC features (s008 set) augmented by simple geometric features (PAB
direction at brightest LC peak + mean PAB direction over the window).

Why this matters: L̂ is conserved under torque-free dynamics, so it is
a single 3-vector per seed. If recoverable, the ω-direction grid in
inertial frame can be reduced from the full unit sphere (4π sr) to a
small precession disc around L̂ (the polhode-projected ω̂ trajectory
covers a cone of half-angle related to pol_diam / |L|·I_b). At a
±20° L̂ uncertainty, the ω̂ candidate region shrinks by ~1/40 vs
uniform on the sphere, and at ±10° by ~1/160. Combined with s055a's
pol_diam adaptive bracket on |ω|, this collapses the (ω_x, ω_y, ω_z)
3-D search space to a tube around the precession orbit.

Truth: For each m048 seed, with cached body-frame `omega0_rad` and
`q0_wxyz` (passive J2000→body convention per `attitude_propagator.py`)
and the post-fix forward-model inertia tensor I from
`lib.hifi_render._build_model()`,
    L_inertial = R(q0)ᵀ · I · ω0_body
    L̂ = L_inertial / |L_inertial|
L is conserved under torque-free motion, so this is invariant in t.

Method:
  • Train cohort: m048 seeds 0..99 (n=100), L̂ from cached truth.
  • Test holdout: m048 seeds 100..119 (n=20).
  • Features (34 total):
      - 28 s008 LC features from `lib.lc_features`
      - 6 geometric features:
          - `pab_brightest[3]`: PAB unit vector in J2000 at the LC's
            brightest epoch (argmin mag).
          - `pab_mean[3]`:      mean PAB unit vector over the window
            (sun + observer geometry slowly varies; this captures the
            seed's overall observation geometry).
  • Targets: 3-component L̂ in J2000. At inference, predictions are
    renormalised onto the unit sphere.
  • Models: same set as s055a — LinearRegression, RidgeCV,
    RandomForestRegressor (300×8), GradientBoostingRegressor (300×4).
    GBR wrapped in MultiOutputRegressor (sklearn GBR is single-output).
    Training on raw 3-component target (no log10; direction has no
    scale).
  • Metrics: angular error in DEGREES between predicted unit vector and
    truth unit vector. Median, p90, max, fraction within {10°, 20°,
    45°, 90°}. Random-on-sphere baseline: median ~90°, frac<20° ≈ 3%.

Decision gate:
  • holdout median angular error < 20° → operational; ω-direction grid
    pruning gets a ~1/40-of-sphere prior. Strong promotion.
  • 20° ≤ MAE < 45° → useful for half-sphere pruning, factor ~2-4×
    grid reduction. Worth integrating.
  • MAE ≥ 45° → close negative; L̂ is not recoverable from this
    feature set. Possible v2: peak-times × PAB(peak_time) richer
    geometric features. Or accept that L̂ is ω-direction-degenerate.

Outputs (under results/s055b_lhat_lc_regression/):
  features.npz   - X[120,34], y_lhat[120,3], y_lhat_norm[120], seeds,
                   is_cohort_mask, feature_names
  regression.npz - per-model {cohort-LOO, holdout-test} predictions
  summary.json   - decision-grade scalars + cohort-vs-random baseline
  angular_error_hist.png - cohort LOO + holdout angular error
                           histograms with random-baseline reference
  predicted_vs_truth_lhat.png - 3D scatter of predicted L̂ vs truth L̂
                                on unit sphere (cohort LOO + holdout)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# BLAS threading limits — the single-output regressions are cheap, but
# RF/GBR with multi-output can spawn threads via openblas.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quaternion as nq  # numpy-quaternion package
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hifi_render import _build_model  # noqa: E402  (cached module-level)
from lib.lc_features import FEATURE_NAMES as LC_FEATURE_NAMES  # noqa: E402
from lib.lc_features import lc_features  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402

OUT_DIR = ROOT / "results" / "s055b_lhat_lc_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_SEEDS = list(range(100))
HOLDOUT_SEEDS = list(range(100, 120))
ALL_SEEDS = COHORT_SEEDS + HOLDOUT_SEEDS

GEO_FEATURE_NAMES = [
    "pab_bright_x",
    "pab_bright_y",
    "pab_bright_z",
    "pab_mean_x",
    "pab_mean_y",
    "pab_mean_z",
]
ALL_FEATURE_NAMES = list(LC_FEATURE_NAMES) + GEO_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Truth L̂ + features
# ---------------------------------------------------------------------------

def truth_lhat_inertial(q0_wxyz: np.ndarray, omega0_rad: np.ndarray, I_body: np.ndarray) -> np.ndarray:
    """Compute inertial-frame L̂ from truth (q0, ω0_body) and inertia.

    Convention from `src/dynamics/attitude_propagator.py`:
      as_rotation_matrix(q) = passive J2000 → body
    so v_body = R(q) · v_inertial, and v_inertial = R(q)ᵀ · v_body.

    L_body = I · ω_body
    L_inertial = R(q0)ᵀ · L_body
    L̂ = L_inertial / |L_inertial|
    """
    q_obj = nq.quaternion(*q0_wxyz)
    R_inertial_to_body = nq.as_rotation_matrix(q_obj)
    L_body = I_body @ np.asarray(omega0_rad, dtype=float)
    L_inertial = R_inertial_to_body.T @ L_body
    return L_inertial / np.linalg.norm(L_inertial)


def pab_features(
    mag: np.ndarray,
    sun_pos: np.ndarray,
    obs_pos: np.ndarray,
    sat_pos: np.ndarray,
) -> np.ndarray:
    """Compute 6 geometric features.

    PAB unit vector at observation epoch t:
        sun_unit(t) = (sun_pos(t) - sat_pos(t)) / |.|
        obs_unit(t) = (obs_pos(t) - sat_pos(t)) / |.|
        pab(t)      = (sun_unit + obs_unit) / |sun_unit + obs_unit|

    Returns
    -------
    feats : (6,) [pab_bright_x/y/z, pab_mean_x/y/z]
    """
    sun_to_sat = sun_pos - sat_pos
    obs_to_sat = obs_pos - sat_pos
    sun_unit = sun_to_sat / np.linalg.norm(sun_to_sat, axis=1, keepdims=True)
    obs_unit = obs_to_sat / np.linalg.norm(obs_to_sat, axis=1, keepdims=True)
    pab_all = sun_unit + obs_unit
    pab_all = pab_all / np.linalg.norm(pab_all, axis=1, keepdims=True)

    i_peak = int(np.argmin(mag))
    pab_bright = pab_all[i_peak]
    pab_mean = pab_all.mean(axis=0)
    pab_mean = pab_mean / np.linalg.norm(pab_mean)
    return np.concatenate([pab_bright, pab_mean])


def lhat_features(seed: int, I_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract (34,) features and (3,) truth L̂_J2000 for one seed."""
    truth = load_truth(seed)
    feats_lc, _ = lc_features(truth["observation_times"], truth["mag_hifi"])
    feats_geo = pab_features(
        mag=truth["mag_hifi"],
        sun_pos=truth["sun_pos"],
        obs_pos=truth["obs_pos"],
        sat_pos=truth["sat_pos"],
    )
    feats = np.concatenate([feats_lc, feats_geo])
    L_hat = truth_lhat_inertial(truth["q0_wxyz"], truth["omega0_rad"], I_body)
    return feats, L_hat


# ---------------------------------------------------------------------------
# Models, predictions, metrics
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """Multi-output regressors. Linear & Ridge & RF natively support
    multi-output; GBR is single-output and gets MultiOutputRegressor."""
    return {
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 2, 20)),
        "rf": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=0, n_jobs=1
        ),
        "gbr": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=4, random_state=0
            ),
            n_jobs=1,
        ),
    }


def loo_predict(model_proto, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Leave-one-out OOF predictions on (X, Y). Y has shape (n, 3). Returns (n, 3)."""
    loo = LeaveOneOut()
    preds = np.zeros_like(Y)
    for tr, te in loo.split(X):
        # Clone via re-construction. MultiOutputRegressor.get_params doesn't
        # round-trip cleanly across sklearn versions, so we just deep-clone
        # a fresh model_proto each time.
        from sklearn.base import clone

        m = clone(model_proto)
        m.fit(X[tr], Y[tr])
        preds[te] = m.predict(X[te])
    return preds


def normalize_rows(V: np.ndarray) -> np.ndarray:
    """Renormalise each row to unit length."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(norms, 1e-12)


def angular_error_deg(Y_pred: np.ndarray, Y_truth: np.ndarray) -> np.ndarray:
    """Both (n, 3) unit vectors. Returns (n,) angles in degrees."""
    cos_t = (Y_pred * Y_truth).sum(axis=1)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def metrics(errs_deg: np.ndarray) -> dict:
    return {
        "median_deg": float(np.median(errs_deg)),
        "mean_deg": float(np.mean(errs_deg)),
        "p90_deg": float(np.percentile(errs_deg, 90)),
        "max_deg": float(np.max(errs_deg)),
        "frac_within_10deg": float(np.mean(errs_deg < 10.0)),
        "frac_within_20deg": float(np.mean(errs_deg < 20.0)),
        "frac_within_45deg": float(np.mean(errs_deg < 45.0)),
        "frac_within_90deg": float(np.mean(errs_deg < 90.0)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_angular_error_hists(
    cohort_errs: dict[str, np.ndarray],
    holdout_errs: dict[str, np.ndarray],
    fname: Path,
):
    """4-panel histogram of angular errors (one per model), with cohort-LOO
    + holdout overlaid + random-baseline reference line."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    bins = np.linspace(0, 180, 37)  # 5° bins
    # Random-on-sphere baseline median ≈ 90° (acos(0.5) for cos~uniform).
    for ax, name in zip(axes.ravel(), ["linear", "ridge", "rf", "gbr"]):
        ax.hist(
            cohort_errs[name],
            bins=bins,
            alpha=0.55,
            label=f"cohort LOO (n={len(cohort_errs[name])}, "
            f"med {np.median(cohort_errs[name]):.1f}°)",
            color="tab:blue",
            density=True,
        )
        ax.hist(
            holdout_errs[name],
            bins=bins,
            alpha=0.55,
            label=f"holdout (n={len(holdout_errs[name])}, "
            f"med {np.median(holdout_errs[name]):.1f}°)",
            color="tab:orange",
            density=True,
        )
        ax.axvline(90.0, color="k", linestyle=":", alpha=0.6, label="random med (90°)")
        ax.axvline(20.0, color="g", linestyle="--", alpha=0.4, label="20° gate")
        ax.axvline(45.0, color="r", linestyle="--", alpha=0.4, label="45° gate")
        ax.set_title(name)
        ax.set_xlabel("angular error (deg)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("s055b — L̂ angular-error histograms (per model)")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_predicted_vs_truth_sphere(
    L_truth: np.ndarray,
    L_pred_loo: dict[str, np.ndarray],
    L_pred_hold: dict[str, np.ndarray],
    is_cohort: np.ndarray,
    fname: Path,
):
    """4 subplots, one per model. Each: 3D scatter of truth (blue) and
    predicted (red) L̂ on unit sphere. Lines from truth → pred."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 12))
    for k, name in enumerate(["linear", "ridge", "rf", "gbr"]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        # Cohort
        Tc = L_truth[is_cohort]
        Pc = L_pred_loo[name]
        ax.scatter(Tc[:, 0], Tc[:, 1], Tc[:, 2], c="tab:blue", s=8, alpha=0.6, label="truth (cohort)")
        for i in range(min(20, len(Tc))):
            ax.plot(
                [Tc[i, 0], Pc[i, 0]], [Tc[i, 1], Pc[i, 1]], [Tc[i, 2], Pc[i, 2]],
                color="tab:blue", alpha=0.2, linewidth=0.5,
            )
        # Holdout
        Th = L_truth[~is_cohort]
        Ph = L_pred_hold[name]
        ax.scatter(Th[:, 0], Th[:, 1], Th[:, 2], c="tab:orange", s=22, alpha=0.9, label="truth (holdout)")
        ax.scatter(Ph[:, 0], Ph[:, 1], Ph[:, 2], c="tab:red", s=22, alpha=0.9, marker="^", label="pred (holdout)")
        for i in range(len(Th)):
            ax.plot(
                [Th[i, 0], Ph[i, 0]], [Th[i, 1], Ph[i, 1]], [Th[i, 2], Ph[i, 2]],
                color="tab:red", alpha=0.5, linewidth=1.0,
            )
        ax.set_title(f"{name} — holdout med {np.median(angular_error_deg(Ph, Th)):.1f}°")
        ax.set_xlabel("L̂_x")
        ax.set_ylabel("L̂_y")
        ax.set_zlabel("L̂_z")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.legend(fontsize=7)
    fig.suptitle("s055b — predicted vs truth L̂ on unit sphere")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def random_baseline(L_truth: np.ndarray, n_trials: int = 1000, seed: int = 0) -> dict:
    """For each L_truth row, draw `n_trials` random unit vectors and compute
    angular error. Aggregate across truth rows × trials. Reports the same
    metric set as `metrics()` for fair comparison."""
    rng = np.random.default_rng(seed)
    n = L_truth.shape[0]
    Z = rng.normal(size=(n_trials, n, 3))
    Z = Z / np.linalg.norm(Z, axis=2, keepdims=True)
    errs_all = []
    for ti in range(n_trials):
        errs_all.append(angular_error_deg(Z[ti], L_truth))
    return metrics(np.concatenate(errs_all))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()

    print("s055b — L̂ regression on 120 m048 seeds (100 cohort + 20 holdout)")
    print("  building model (satellite + inertia)...")
    _, I_body = _build_model()
    eigvals = np.linalg.eigvalsh(I_body)
    print(f"    inertia tensor eigvals = {eigvals} kg·m² (I_a, I_b, I_c)")

    # --- 1. Extract features + truth L̂ for all seeds ---
    print("  extracting features + computing truth L̂...")
    X = np.empty((len(ALL_SEEDS), len(ALL_FEATURE_NAMES)))
    Y = np.empty((len(ALL_SEEDS), 3))
    for i, seed in enumerate(ALL_SEEDS):
        feats, L = lhat_features(seed, I_body)
        X[i] = feats
        Y[i] = L
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(ALL_SEEDS)} seeds processed")
    is_cohort = np.array([s in set(COHORT_SEEDS) for s in ALL_SEEDS])
    assert is_cohort.sum() == 100
    assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "non-finite features"
    # Sanity: truth L̂ rows are unit vectors
    Y_norms = np.linalg.norm(Y, axis=1)
    print(f"    truth L̂ row norms: min {Y_norms.min():.6f} max {Y_norms.max():.6f}")
    assert np.allclose(Y_norms, 1.0, atol=1e-9), "truth L̂ not unit"
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    # --- 2. Fit + LOO + holdout for each model ---
    Y_cohort = Y[is_cohort]
    Y_holdout = Y[~is_cohort]
    X_cohort = X[is_cohort]
    X_holdout = X[~is_cohort]

    cohort_errs: dict[str, np.ndarray] = {}
    holdout_errs: dict[str, np.ndarray] = {}
    L_pred_loo: dict[str, np.ndarray] = {}
    L_pred_hold: dict[str, np.ndarray] = {}
    cohort_metrics: dict[str, dict] = {}
    holdout_metrics_d: dict[str, dict] = {}

    for name, proto in build_models().items():
        # Cohort LOO
        loo_raw = loo_predict(proto, X_cohort, Y_cohort)
        loo_unit = normalize_rows(loo_raw)
        errs_loo = angular_error_deg(loo_unit, Y_cohort)
        # Holdout (fit full cohort, predict holdout)
        from sklearn.base import clone

        full = clone(proto)
        full.fit(X_cohort, Y_cohort)
        hold_raw = full.predict(X_holdout)
        hold_unit = normalize_rows(hold_raw)
        errs_hold = angular_error_deg(hold_unit, Y_holdout)

        cohort_errs[name] = errs_loo
        holdout_errs[name] = errs_hold
        L_pred_loo[name] = loo_unit
        L_pred_hold[name] = hold_unit
        cohort_metrics[name] = metrics(errs_loo)
        holdout_metrics_d[name] = metrics(errs_hold)

        c, h = cohort_metrics[name], holdout_metrics_d[name]
        print(
            f"  {name:6s}  LOO med {c['median_deg']:5.1f}°  p90 {c['p90_deg']:5.1f}°  "
            f"<20° {c['frac_within_20deg']:.0%}   "
            f"HOLDOUT med {h['median_deg']:5.1f}°  p90 {h['p90_deg']:5.1f}°  "
            f"<20° {h['frac_within_20deg']:.0%}"
        )

    # --- 3. Random baseline + sanity ---
    rb_cohort = random_baseline(Y_cohort, n_trials=1000, seed=0)
    rb_holdout = random_baseline(Y_holdout, n_trials=1000, seed=1)
    print(
        f"  RANDOM baseline (1000 trials):  cohort med {rb_cohort['median_deg']:.1f}°  "
        f"holdout med {rb_holdout['median_deg']:.1f}°  "
        f"cohort <20° {rb_cohort['frac_within_20deg']:.1%}"
    )

    # --- 4. Save ---
    np.savez(
        OUT_DIR / "features.npz",
        X=X,
        y_lhat=Y,
        seeds=np.array(ALL_SEEDS),
        is_cohort_mask=is_cohort,
        feature_names=np.array(ALL_FEATURE_NAMES),
    )
    reg_payload = {
        "seeds": np.array(ALL_SEEDS),
        "is_cohort_mask": is_cohort,
        "y_lhat_truth": Y,
    }
    for name in ["linear", "ridge", "rf", "gbr"]:
        reg_payload[f"{name}__loo_pred"] = L_pred_loo[name]
        reg_payload[f"{name}__holdout_pred"] = L_pred_hold[name]
        reg_payload[f"{name}__loo_errs_deg"] = cohort_errs[name]
        reg_payload[f"{name}__holdout_errs_deg"] = holdout_errs[name]
    np.savez(OUT_DIR / "regression.npz", **reg_payload)

    # --- 5. Summary + decision ---
    summary = {
        "n_cohort": int(is_cohort.sum()),
        "n_holdout": int((~is_cohort).sum()),
        "n_features": int(X.shape[1]),
        "wall_seconds": float(time.perf_counter() - t0),
        "cohort_metrics": cohort_metrics,
        "holdout_metrics": holdout_metrics_d,
        "random_baseline": {"cohort": rb_cohort, "holdout": rb_holdout},
    }

    # Best (model) by holdout median angular error
    best_name = min(
        holdout_metrics_d.keys(),
        key=lambda k: holdout_metrics_d[k]["median_deg"],
    )
    best_med_holdout = holdout_metrics_d[best_name]["median_deg"]
    summary["best_model_holdout"] = best_name
    summary["best_holdout_median_deg"] = best_med_holdout
    summary["best_holdout_frac_within_20deg"] = holdout_metrics_d[best_name]["frac_within_20deg"]
    summary["best_holdout_frac_within_45deg"] = holdout_metrics_d[best_name]["frac_within_45deg"]

    # Decision label
    if best_med_holdout < 20.0:
        decision = "OPERATIONAL"
        decision_msg = (
            f"L̂ recoverable to <20° on holdout with {best_name}; ω-direction "
            f"grid pruning collapses sphere to ~1/40 (small precession disc)."
        )
    elif best_med_holdout < 45.0:
        decision = "COARSE_USEFUL"
        decision_msg = (
            f"L̂ recoverable to <45° with {best_name}; useful for half-sphere "
            f"or quadrant pruning (factor ~2-4× ω-grid reduction)."
        )
    else:
        decision = "WEAK"
        decision_msg = (
            f"L̂ not directly recoverable with the s008+geo feature set "
            f"(holdout med {best_med_holdout:.1f}°). Possible v2: PAB at "
            f"top-K LC peaks, or accept that L̂ is geometry-degenerate."
        )

    # Beats random?
    beats_random = best_med_holdout < rb_holdout["median_deg"] - 5.0
    summary["decision"] = decision
    summary["decision_msg"] = decision_msg
    summary["beats_random_baseline_5deg"] = bool(beats_random)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- 6. Plots ---
    plot_angular_error_hists(cohort_errs, holdout_errs, OUT_DIR / "angular_error_hist.png")
    plot_predicted_vs_truth_sphere(
        Y, L_pred_loo, L_pred_hold, is_cohort,
        OUT_DIR / "predicted_vs_truth_lhat.png",
    )

    # --- 7. Final stdout ---
    print(f"\n=== s055b summary (wall {summary['wall_seconds']:.2f} s) ===")
    print(
        f"BEST holdout = {best_name}  median angular error = {best_med_holdout:.2f}°  "
        f"<20° {holdout_metrics_d[best_name]['frac_within_20deg']:.0%}  "
        f"<45° {holdout_metrics_d[best_name]['frac_within_45deg']:.0%}"
    )
    print(f"Random baseline holdout median: {rb_holdout['median_deg']:.1f}° "
          f"(beats random by ≥5°: {beats_random})")
    print(f"Decision: {decision} — {decision_msg}")
    print("Saved:")
    print(f"  {OUT_DIR / 'features.npz'}")
    print(f"  {OUT_DIR / 'regression.npz'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {OUT_DIR / 'angular_error_hist.png'}")
    print(f"  {OUT_DIR / 'predicted_vs_truth_lhat.png'}")


if __name__ == "__main__":
    main()
