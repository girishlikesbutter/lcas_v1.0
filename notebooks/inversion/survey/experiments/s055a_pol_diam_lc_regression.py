"""s055a — Polhode-diameter regression from LC features.

Hypothesis: pol_diam (polhode L2 diameter, dps) — the s053 load-bearing
basin-width predictor (Spearman ρ=−0.94 vs basin width, n=10) — is
recoverable from light-curve features alone, making the polhode prior an
operational adaptive-bracket controller rather than a post-hoc
explanatory framework.

Method (mirrors s008's pipeline, swap target ω-mag → pol_diam):
  • Train cohort: m048 seeds 0..99 (n=100), pol_diam from s053.
  • Test holdout: m048 seeds 100..119 (n=20), pol_diam from s054b.
    True held-out per `feedback_holdout_validation.md` — never seen
    during training.
  • Features: 28-dim s008 set, lifted to `lib.lc_features`.
    Spectral (LS top-5 freqs/powers, hi/lo power ratio), time-domain
    (mean/std/min/max/skew/kurt, |dmag/dt| stats), autocorrelation
    (lag-1/5/20, first-peak lag), glints (count, spacing). LC-only —
    no truth ω, no geometry.
  • Models: LinearRegression, RidgeCV, RandomForestRegressor (300x8),
    GradientBoostingRegressor (300x4) — identical to s008.
  • Targets: direct (pol_diam) and log10(pol_diam).
  • Eval: LOO CV on cohort → cohort_MAPE, p90, frac<{3,5,10,30}%.
          Fit on full cohort → holdout test MAPE etc.
  • Sanity control: run the same pipeline with om_mag_dps as target.
    Must reproduce s008's LOO MAPE ~16.4% within ±2 percentage points.
    If not, lc_features extraction in lib/ has drifted from s008.

Decision gate (interpreted in the writeup, not enforced by the script):
  • holdout MAPE < 20% → polhode prior operational; proceed to s055b
    oracle adaptive-bracket pilot, then operational s055c.
  • 20% ≤ MAPE < 50% → coarse but useful (factor of ~1.5 bracket
    error); s055b still informative; defer regression refinement to a
    focused s056 (e.g. 3-bucket classification).
  • MAPE ≥ 50% → polhode prior remains analytical-only; reframe via
    hierarchical search or accept as explanatory.
  Bar reasoning: pol_diam appears multiplicatively in the s053 rule
  `δ|ω|/|ω| ≤ 0.5% × (pol_diam_ref / pol_diam_seed)`, so a 2× error in
  pol_diam → 2× bracket error (still recoverable); 10× error → broken.

Outputs (under results/s055a_pol_diam_lc_regression/):
  features.npz       - X[120,28], y_pol_diam[120], y_om_mag[120], seeds,
                       is_cohort_mask, feature_names
  regression.npz     - per-model {cohort-LOO, holdout-test} predictions
                       for pol_diam target (direct + log10) and om_mag
                       sanity control
  summary.json       - decision-grade scalars
  actual_vs_predicted_loo.png      - cohort LOO scatter, 2x4 grid
  actual_vs_predicted_holdout.png  - holdout test scatter, 2x4 grid
  feature_importance.png           - RF importances on pol_diam target
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.lc_features import FEATURE_NAMES, lc_features  # noqa: E402
from lib.traj_load import truth_state  # noqa: E402

OUT_DIR = ROOT / "results" / "s055a_pol_diam_lc_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_SEEDS = list(range(100))
HOLDOUT_SEEDS = list(range(100, 120))
ALL_SEEDS = COHORT_SEEDS + HOLDOUT_SEEDS  # 120 total

POL_COHORT_NPZ = ROOT / "results" / "s053_cohort_polhode_survey" / "cohort.npz"
POL_HOLDOUT_NPZ = ROOT / "results" / "s054_holdout" / "holdout_polhode.npz"


def build_models() -> dict:
    return {
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 2, 20)),
        "rf": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=0, n_jobs=1
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, random_state=0
        ),
    }


def loo_predict(model_proto, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out OOF predictions on (X, y)."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = model_proto.__class__(**model_proto.get_params())
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rel = 100.0 * np.abs(y_pred - y_true) / np.abs(y_true)
    return {
        "median_pct": float(np.median(rel)),
        "mean_pct": float(np.mean(rel)),
        "p90_pct": float(np.percentile(rel, 90)),
        "max_pct": float(np.max(rel)),
        "frac_within_3pct": float(np.mean(rel < 3.0)),
        "frac_within_5pct": float(np.mean(rel < 5.0)),
        "frac_within_10pct": float(np.mean(rel < 10.0)),
        "frac_within_30pct": float(np.mean(rel < 30.0)),
    }


def evaluate_target(
    target_name: str,
    X: np.ndarray,
    y: np.ndarray,
    is_cohort: np.ndarray,
) -> dict:
    """Run all four models on a target. Both direct and log10 targets.

    Returns nested dict: {model: {direct: {loo, holdout}, log10: {loo, holdout}}}.
    """
    out: dict = {}
    Xc = X[is_cohort]
    yc = y[is_cohort]
    Xh = X[~is_cohort]
    yh = y[~is_cohort]
    yc_log = np.log10(yc)

    for name, proto in build_models().items():
        out[name] = {}
        # -- direct target --
        loo_pred = loo_predict(proto, Xc, yc)
        full = proto.__class__(**proto.get_params())
        full.fit(Xc, yc)
        hold_pred = full.predict(Xh)
        out[name]["direct"] = {
            "loo_preds": loo_pred,
            "holdout_preds": hold_pred,
            "loo_metrics": metrics(yc, loo_pred),
            "holdout_metrics": metrics(yh, hold_pred),
        }
        # -- log10 target --
        loo_pred_log = loo_predict(proto, Xc, yc_log)
        full_log = proto.__class__(**proto.get_params())
        full_log.fit(Xc, yc_log)
        hold_pred_log = full_log.predict(Xh)
        loo_pred_from_log = 10.0 ** loo_pred_log
        hold_pred_from_log = 10.0 ** hold_pred_log
        out[name]["log10"] = {
            "loo_preds": loo_pred_from_log,
            "holdout_preds": hold_pred_from_log,
            "loo_metrics": metrics(yc, loo_pred_from_log),
            "holdout_metrics": metrics(yh, hold_pred_from_log),
        }
        # diagnostic print
        d = out[name]["direct"]
        l = out[name]["log10"]
        print(
            f"  {target_name:10s} {name:6s}  LOO direct {d['loo_metrics']['median_pct']:6.2f}% "
            f"log {l['loo_metrics']['median_pct']:6.2f}%   "
            f"HOLDOUT direct {d['holdout_metrics']['median_pct']:6.2f}% "
            f"log {l['holdout_metrics']['median_pct']:6.2f}%"
        )
    return out


def plot_scatter_grid(
    y_true: np.ndarray,
    target_results: dict,
    split_name: str,
    target_name: str,
    fname: Path,
):
    """2x4 grid: rows = direct/log10, cols = 4 models. Log-log scatter."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    y_lo = float(np.min(y_true))
    y_hi = float(np.max(y_true))
    for col, (name, res) in enumerate(target_results.items()):
        for row, kind in enumerate(["direct", "log10"]):
            ax = axes[row, col]
            preds = res[kind][f"{split_name}_preds"]
            m = res[kind][f"{split_name}_metrics"]
            ax.plot([y_lo, y_hi], [y_lo, y_hi], "k--", alpha=0.5)
            ax.plot([y_lo, y_hi], [1.05 * y_lo, 1.05 * y_hi], "k:", alpha=0.3)
            ax.plot([y_lo, y_hi], [0.95 * y_lo, 0.95 * y_hi], "k:", alpha=0.3)
            color = "tab:blue" if kind == "direct" else "tab:orange"
            ax.scatter(y_true, preds, s=20, alpha=0.7, color=color)
            ax.set_title(
                f"{name} ({kind})\nmedian {m['median_pct']:.2f}%  "
                f"p90 {m['p90_pct']:.2f}%"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(f"predicted {target_name}")
            if row == 1:
                ax.set_xlabel(f"truth {target_name}")
    fig.suptitle(f"s055a — {split_name.upper()} predictions on {target_name}")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_feature_importance(
    X: np.ndarray, y: np.ndarray, names: list[str], fname: Path
):
    """RF feature importance on full cohort, pol_diam target."""
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=10, random_state=0, n_jobs=1
    )
    rf.fit(X, y)
    fi = rf.feature_importances_
    order = np.argsort(-fi)
    top_n = 15
    fig, ax = plt.subplots(figsize=(10, 7))
    idx = order[:top_n]
    ax.barh(range(top_n), fi[idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([names[i] for i in idx[::-1]])
    ax.set_xlabel("RF feature importance")
    ax.set_title("s055a — top-15 features (RF, pol_diam target, cohort fit)")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fi


def main():
    t0 = time.perf_counter()
    print(f"s055a — pol_diam regression on {len(ALL_SEEDS)} seeds (100 cohort + 20 holdout)")

    # --- 1. Load pol_diam targets ---
    cohort_npz = np.load(POL_COHORT_NPZ)
    holdout_npz = np.load(POL_HOLDOUT_NPZ)
    cohort_seeds_npz = cohort_npz["seeds"]
    holdout_seeds_npz = holdout_npz["seeds"]
    assert (cohort_seeds_npz == np.arange(100)).all(), "cohort seed order mismatch"
    assert (holdout_seeds_npz == np.arange(100, 120)).all(), "holdout seed order mismatch"
    pol_cohort = cohort_npz["pol_diam_dps"].astype(float)
    pol_holdout = holdout_npz["pol_diam_dps"].astype(float)
    y_pol = np.concatenate([pol_cohort, pol_holdout])
    print(f"  pol_diam range: cohort [{pol_cohort.min():.3f}, {pol_cohort.max():.3f}]  "
          f"holdout [{pol_holdout.min():.3f}, {pol_holdout.max():.3f}] dps")

    # --- 2. Extract LC features for all 120 seeds ---
    print("  extracting LC features...")
    X = np.empty((len(ALL_SEEDS), len(FEATURE_NAMES)))
    y_om = np.empty(len(ALL_SEEDS))
    for i, seed in enumerate(ALL_SEEDS):
        s = truth_state(seed)
        feats, _ = lc_features(s["observation_times"], s["mag_hifi"])
        X[i] = feats
        y_om[i] = float(s["omega_mag_dps"])
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(ALL_SEEDS)} seeds processed")
    is_cohort = np.array([s in set(COHORT_SEEDS) for s in ALL_SEEDS])
    assert is_cohort.sum() == 100
    assert not np.any(np.isnan(X)), "NaN in feature matrix"
    assert not np.any(np.isinf(X)), "Inf in feature matrix"
    print(f"  X shape: {X.shape},  pol_diam y range [{y_pol.min():.3f}, {y_pol.max():.3f}] dps,  "
          f"|ω| y range [{y_om.min():.3f}, {y_om.max():.3f}] dps")

    # --- 3. Evaluate targets ---
    print("\n--- pol_diam target ---")
    results_pol = evaluate_target("pol_diam", X, y_pol, is_cohort)
    print("\n--- om_mag_dps SANITY (must reproduce s008 LOO ~16.4%) ---")
    results_om = evaluate_target("om_mag", X, y_om, is_cohort)

    # --- 4. Feature importance (pol_diam) ---
    print("\n  computing RF feature importance on pol_diam (cohort fit)")
    fi = plot_feature_importance(
        X[is_cohort],
        y_pol[is_cohort],
        list(FEATURE_NAMES),
        OUT_DIR / "feature_importance.png",
    )
    fi_order = np.argsort(-fi)
    print("  top-10 features:")
    for j in fi_order[:10]:
        print(f"    {FEATURE_NAMES[j]:25s}  {fi[j]:.4f}")

    # --- 5. Plots ---
    plot_scatter_grid(
        y_pol[is_cohort],
        results_pol,
        "loo",
        "pol_diam (dps)",
        OUT_DIR / "actual_vs_predicted_loo.png",
    )
    plot_scatter_grid(
        y_pol[~is_cohort],
        results_pol,
        "holdout",
        "pol_diam (dps)",
        OUT_DIR / "actual_vs_predicted_holdout.png",
    )

    # --- 6. Save NPZs ---
    np.savez(
        OUT_DIR / "features.npz",
        X=X,
        y_pol_diam=y_pol,
        y_om_mag=y_om,
        seeds=np.array(ALL_SEEDS),
        is_cohort_mask=is_cohort,
        feature_names=np.array(FEATURE_NAMES),
        rf_feature_importance=fi,
    )

    reg_payload = {
        "seeds": np.array(ALL_SEEDS),
        "is_cohort_mask": is_cohort,
        "y_pol_diam": y_pol,
        "y_om_mag": y_om,
    }
    for tgt_name, results in [("pol_diam", results_pol), ("om_mag", results_om)]:
        for mname, mres in results.items():
            for kind in ("direct", "log10"):
                key_loo = f"{tgt_name}__{mname}__{kind}__loo_preds"
                key_hold = f"{tgt_name}__{mname}__{kind}__holdout_preds"
                reg_payload[key_loo] = mres[kind]["loo_preds"]
                reg_payload[key_hold] = mres[kind]["holdout_preds"]
    np.savez(OUT_DIR / "regression.npz", **reg_payload)

    # --- 7. Summary ---
    summary = {
        "n_cohort": int(is_cohort.sum()),
        "n_holdout": int((~is_cohort).sum()),
        "n_features": int(X.shape[1]),
        "wall_seconds": float(time.perf_counter() - t0),
        "s008_om_mag_loo_baseline_pct": 16.4,
        "results": {},
    }

    def _flatten(target_results):
        flat = {}
        for mname, mres in target_results.items():
            flat[mname] = {}
            for kind in ("direct", "log10"):
                flat[mname][kind] = {
                    "loo": mres[kind]["loo_metrics"],
                    "holdout": mres[kind]["holdout_metrics"],
                }
        return flat

    summary["results"]["pol_diam"] = _flatten(results_pol)
    summary["results"]["om_mag_sanity"] = _flatten(results_om)

    # Top-line decision scalars: best (model, kind) for pol_diam by holdout median
    best_pol_holdout = None
    for mname, mres in results_pol.items():
        for kind in ("direct", "log10"):
            med = mres[kind]["holdout_metrics"]["median_pct"]
            if best_pol_holdout is None or med < best_pol_holdout[2]:
                best_pol_holdout = (mname, kind, med)
    summary["best_pol_diam_holdout_model"] = best_pol_holdout[0]
    summary["best_pol_diam_holdout_kind"] = best_pol_holdout[1]
    summary["best_pol_diam_holdout_median_pct"] = best_pol_holdout[2]

    best_pol_loo = None
    for mname, mres in results_pol.items():
        for kind in ("direct", "log10"):
            med = mres[kind]["loo_metrics"]["median_pct"]
            if best_pol_loo is None or med < best_pol_loo[2]:
                best_pol_loo = (mname, kind, med)
    summary["best_pol_diam_loo_model"] = best_pol_loo[0]
    summary["best_pol_diam_loo_kind"] = best_pol_loo[1]
    summary["best_pol_diam_loo_median_pct"] = best_pol_loo[2]

    best_om_loo = None
    for mname, mres in results_om.items():
        for kind in ("direct", "log10"):
            med = mres[kind]["loo_metrics"]["median_pct"]
            if best_om_loo is None or med < best_om_loo[2]:
                best_om_loo = (mname, kind, med)
    summary["best_om_mag_loo_model"] = best_om_loo[0]
    summary["best_om_mag_loo_kind"] = best_om_loo[1]
    summary["best_om_mag_loo_median_pct"] = best_om_loo[2]
    summary["om_mag_sanity_pass"] = bool(abs(best_om_loo[2] - 16.4) < 2.0)

    summary["top10_features_pol_diam"] = [
        {"name": FEATURE_NAMES[j], "rf_importance": float(fi[j])} for j in fi_order[:10]
    ]

    # Decision-gate label (interpreted in the writeup)
    h = best_pol_holdout[2]
    if h < 20.0:
        summary["decision"] = "OPERATIONAL"
    elif h < 50.0:
        summary["decision"] = "COARSE_USEFUL"
    else:
        summary["decision"] = "ANALYTICAL_ONLY"

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- 8. Final stdout ---
    print(f"\n=== s055a summary (wall {summary['wall_seconds']:.2f} s) ===")
    print(
        f"pol_diam best LOO     = {best_pol_loo[0]} ({best_pol_loo[1]})  "
        f"median MAPE = {best_pol_loo[2]:.2f}%"
    )
    print(
        f"pol_diam best HOLDOUT = {best_pol_holdout[0]} ({best_pol_holdout[1]})  "
        f"median MAPE = {best_pol_holdout[2]:.2f}%   →  decision: {summary['decision']}"
    )
    print(
        f"om_mag SANITY  best LOO = {best_om_loo[0]} ({best_om_loo[1]})  "
        f"median MAPE = {best_om_loo[2]:.2f}%   "
        f"(s008 baseline 16.4%, sanity_pass = {summary['om_mag_sanity_pass']})"
    )
    print("Saved:")
    print(f"  {OUT_DIR / 'features.npz'}")
    print(f"  {OUT_DIR / 'regression.npz'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {OUT_DIR / 'actual_vs_predicted_loo.png'}")
    print(f"  {OUT_DIR / 'actual_vs_predicted_holdout.png'}")
    print(f"  {OUT_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
