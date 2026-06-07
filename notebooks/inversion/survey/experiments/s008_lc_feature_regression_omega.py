"""s008 — Q4c-iv pilot: model-aware omega-mag regression from LC features.

Hypothesis: a multi-feature regression model trained on the 100 m048 truth
trajectories can map LC features (spectral + time-domain) to omega-mag
within s005's 3-5% basin radius. If yes, Q4c becomes tractable. If the
regressor caps near s007's oracle ceiling (~7.4%), LC-only priors are
fundamentally limited and Q4c needs a different strategy.

Method:
  100 seeds (full m048 cohort).
  Features per seed (extracted from truth hi-fi LC, no truth-omega use):
    - top-K Lomb-Scargle peak frequencies (Hz)        [K=5]
    - top-K LS peak powers (normalised)               [K=5]
    - mean / std / min / max of magnitude
    - skewness / kurtosis of magnitude
    - mean / std of |dmag/dt|
    - autocorrelation lag-1 / lag-5 / lag-20
    - ACF first non-zero peak lag (s)
    - count of "glints" (local minima with mag < 9, brightness threshold)
    - mean / std of glint spacing (s)
    - integrated power above 1/100 Hz vs below
  Total: ~25 features.

  Targets: omega_mag_dps (primary), and as a sanity-check we also try
  predicting log(omega_mag_dps) which often regresses better on power-law
  distributed quantities.

  Models compared (all sklearn):
    - LinearRegression (baseline)
    - Ridge (alpha auto-tuned)
    - RandomForestRegressor (n_estimators=200, max_depth=8)
    - GradientBoostingRegressor (n_estimators=300)

  Eval: leave-one-out cross-validation across all 100 seeds. Report
  out-of-fold MAPE (median, p90, max) and the s007-comparable
  "fraction of seeds within 3% / 5% / 10% of truth".

Decision (3% bar):
  - If best model out-of-fold median MAPE < 3% on omega_mag: Q4c-iv passes.
    Q4c becomes tractable with this prior + Sobol(q0) × Sobol(omega_dir) +
    LM polish.
  - If 3% <= median MAPE < 7%: marginal — improves on s007 oracle (7.4%)
    but still wider than basin. Worth keeping as a coarse prior.
  - If median MAPE >= 7%: LC-only priors are dead. Pivot to brute-force
    omega-mag grid or architecture rethink.

Outputs:
  results/s008/features.npz       - per-seed feature vectors and targets
  results/s008/predictions.npz    - per-model out-of-fold predictions
  results/s008/summary.json       - decision-grade scalars
  results/s008/scatter.png        - truth-vs-prediction scatter, per model
  results/s008/feature_importance.png - top-10 RF feature importances
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.traj_load import truth_state, list_seeds  # noqa: E402

OUT_DIR = ROOT / "results" / "s008"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LS_PEAKS = 5
N_FREQ = 4000  # LS frequency grid


def lc_features(t: np.ndarray, mag: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract a fixed-length feature vector from a single LC.

    Does NOT use truth omega - only the observed LC.
    """
    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])

    # ---- Spectral (Lomb-Scargle) ----
    f_min = 1.0 / span
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, N_FREQ)
    power = LombScargle(t, mag).power(freqs)
    peaks, _ = find_peaks(power)
    if len(peaks) >= K_LS_PEAKS:
        order = np.argsort(-power[peaks])
        top = peaks[order[:K_LS_PEAKS]]
        top_f = freqs[top]
        top_p = power[top]
    else:
        # pad with zeros if fewer peaks than K
        top_f = np.zeros(K_LS_PEAKS)
        top_p = np.zeros(K_LS_PEAKS)
        if len(peaks) > 0:
            order = np.argsort(-power[peaks])
            top_f[: len(peaks)] = freqs[peaks[order]]
            top_p[: len(peaks)] = power[peaks[order]]

    # spectral energy ratio: above vs below the median frequency-grid value
    f_split = 0.01  # 1/100 Hz, a coarse cut
    pwr_low = float(power[freqs < f_split].sum())
    pwr_high = float(power[freqs >= f_split].sum())
    pwr_ratio = pwr_high / max(pwr_low, 1e-12)

    # ---- Time-domain stats ----
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_min = float(np.min(mag))
    mag_max = float(np.max(mag))
    mag_skew = float(skew(mag))
    mag_kurt = float(kurtosis(mag))

    dmag = np.diff(mag) / dt
    dmag_mean_abs = float(np.mean(np.abs(dmag)))
    dmag_std = float(np.std(dmag))

    # autocorrelation
    yc = mag - mag.mean()
    acf_full = np.correlate(yc, yc, mode="full")[len(yc) - 1 :]
    acf = acf_full / max(acf_full[0], 1e-12)
    acf_lag1 = float(acf[1]) if len(acf) > 1 else 0.0
    acf_lag5 = float(acf[5]) if len(acf) > 5 else 0.0
    acf_lag20 = float(acf[20]) if len(acf) > 20 else 0.0

    # first non-zero ACF peak lag
    min_lag = max(int(2.0 / dt), 2)
    cand_peaks, _ = find_peaks(acf[min_lag:], height=0.05)
    if len(cand_peaks) > 0:
        acf_first_peak_lag = float((cand_peaks[0] + min_lag) * dt)
    else:
        acf_first_peak_lag = 0.0  # signals "no clear ACF peak"

    # ---- Glint features (bright local minima — note: lower mag = brighter) ----
    # Use 10th-percentile threshold for brightness (top-decile bright epochs).
    bright_thresh = float(np.percentile(mag, 10))
    glint_idx, _ = find_peaks(-mag, height=-bright_thresh)
    glint_count = float(len(glint_idx))
    if len(glint_idx) >= 2:
        glint_spacings = np.diff(t[glint_idx])
        glint_spacing_mean = float(np.mean(glint_spacings))
        glint_spacing_std = float(np.std(glint_spacings))
    else:
        glint_spacing_mean = 0.0
        glint_spacing_std = 0.0

    feats = np.concatenate([
        top_f,
        top_p,
        [pwr_low, pwr_high, pwr_ratio],
        [mag_mean, mag_std, mag_min, mag_max, mag_skew, mag_kurt],
        [dmag_mean_abs, dmag_std],
        [acf_lag1, acf_lag5, acf_lag20, acf_first_peak_lag],
        [glint_count, glint_spacing_mean, glint_spacing_std],
    ])

    names = (
        [f"ls_top{i}_f" for i in range(K_LS_PEAKS)]
        + [f"ls_top{i}_p" for i in range(K_LS_PEAKS)]
        + ["pwr_low", "pwr_high", "pwr_ratio"]
        + ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_skew", "mag_kurt"]
        + ["dmag_mean_abs", "dmag_std"]
        + ["acf_lag1", "acf_lag5", "acf_lag20", "acf_first_peak_lag"]
        + ["glint_count", "glint_spacing_mean", "glint_spacing_std"]
    )

    return feats, names


def main():
    t0 = time.perf_counter()
    seeds = list_seeds()
    print(f"s008 — extracting features for {len(seeds)} seeds")

    X_rows = []
    y_omega = []
    feature_names = None
    for seed in seeds:
        s = truth_state(seed)
        feats, names = lc_features(s["observation_times"], s["mag_hifi"])
        X_rows.append(feats)
        y_omega.append(float(s["omega_mag_dps"]))
        if feature_names is None:
            feature_names = names

    X = np.array(X_rows)
    y = np.array(y_omega)
    log_y = np.log(y)
    print(f"  X shape: {X.shape}, y range [{y.min():.3f}, {y.max():.3f}] dps")

    # ----- Models -----
    models = {
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 2, 20)),
        "rf": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0, n_jobs=1),
        "gbr": GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0),
    }

    results = {}
    loo = LeaveOneOut()

    for name, model in models.items():
        # Predict omega directly
        preds = np.zeros(len(seeds))
        for tr, te in loo.split(X):
            m = model.__class__(**model.get_params())
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        # also predict log-omega and exp back
        preds_log = np.zeros(len(seeds))
        for tr, te in loo.split(X):
            m = model.__class__(**model.get_params())
            m.fit(X[tr], log_y[tr])
            preds_log[te] = m.predict(X[te])
        preds_from_log = np.exp(preds_log)

        rel_err = 100.0 * np.abs(preds - y) / y
        rel_err_log = 100.0 * np.abs(preds_from_log - y) / y

        results[name] = {
            "preds": preds,
            "preds_from_log": preds_from_log,
            "rel_err_pct": rel_err,
            "rel_err_log_pct": rel_err_log,
            "median_err": float(np.median(rel_err)),
            "p90_err": float(np.percentile(rel_err, 90)),
            "max_err": float(np.max(rel_err)),
            "median_err_log": float(np.median(rel_err_log)),
            "p90_err_log": float(np.percentile(rel_err_log, 90)),
            "max_err_log": float(np.max(rel_err_log)),
            "frac_within_3pct": float(np.mean(rel_err < 3.0)),
            "frac_within_5pct": float(np.mean(rel_err < 5.0)),
            "frac_within_10pct": float(np.mean(rel_err < 10.0)),
            "frac_within_3pct_log": float(np.mean(rel_err_log < 3.0)),
            "frac_within_5pct_log": float(np.mean(rel_err_log < 5.0)),
            "frac_within_10pct_log": float(np.mean(rel_err_log < 10.0)),
        }
        print(
            f"  {name:8s}: median {rel_err.mean():.2f}% (direct) {rel_err_log.mean():.2f}% (log)  "
            f"  median direct={results[name]['median_err']:.2f}%  log={results[name]['median_err_log']:.2f}%  "
            f"  <3%: direct={results[name]['frac_within_3pct']:.0%}  log={results[name]['frac_within_3pct_log']:.0%}  "
            f"  <5%: direct={results[name]['frac_within_5pct']:.0%}  log={results[name]['frac_within_5pct_log']:.0%}"
        )

    # ----- Feature importance from final RF on full data -----
    rf_full = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=0, n_jobs=1)
    rf_full.fit(X, y)
    fi = rf_full.feature_importances_
    fi_order = np.argsort(-fi)
    print("\nTop-10 feature importances (RF on full data, omega target):")
    for i in fi_order[:10]:
        print(f"  {feature_names[i]:25s}  {fi[i]:.4f}")

    # ----- Best model -----
    best_name = min(results.keys(), key=lambda k: min(results[k]["median_err"], results[k]["median_err_log"]))
    best_kind = "direct" if results[best_name]["median_err"] <= results[best_name]["median_err_log"] else "log"
    best_median = (
        results[best_name]["median_err"] if best_kind == "direct" else results[best_name]["median_err_log"]
    )
    print(f"\nBest model: {best_name} ({best_kind}-target), median LOO MAPE = {best_median:.2f}%")

    # ----- Save -----
    np.savez(
        OUT_DIR / "features.npz",
        X=X,
        y_omega=y,
        log_y=log_y,
        feature_names=np.array(feature_names),
        seeds=np.array(seeds),
        rf_feature_importance=fi,
    )
    pred_payload = {
        "seeds": np.array(seeds),
        "y_truth": y,
    }
    for name, r in results.items():
        pred_payload[f"{name}_preds"] = r["preds"]
        pred_payload[f"{name}_preds_log"] = r["preds_from_log"]
        pred_payload[f"{name}_rel_err_pct"] = r["rel_err_pct"]
        pred_payload[f"{name}_rel_err_log_pct"] = r["rel_err_log_pct"]
    np.savez(OUT_DIR / "predictions.npz", **pred_payload)

    summary = {
        "n_seeds": len(seeds),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "wall_seconds": float(time.perf_counter() - t0),
        "decision_threshold_pct": 3.0,
        "s007_oracle_baseline_pct": 7.4,
        "best_model": best_name,
        "best_kind": best_kind,
        "best_median_pct": best_median,
        "passes_3pct_bar": bool(best_median < 3.0),
        "beats_s007_oracle": bool(best_median < 7.4),
        "models": {},
    }
    for name, r in results.items():
        summary["models"][name] = {
            "direct": {
                "median_pct": r["median_err"],
                "p90_pct": r["p90_err"],
                "max_pct": r["max_err"],
                "frac_within_3pct": r["frac_within_3pct"],
                "frac_within_5pct": r["frac_within_5pct"],
                "frac_within_10pct": r["frac_within_10pct"],
            },
            "log": {
                "median_pct": r["median_err_log"],
                "p90_pct": r["p90_err_log"],
                "max_pct": r["max_err_log"],
                "frac_within_3pct": r["frac_within_3pct_log"],
                "frac_within_5pct": r["frac_within_5pct_log"],
                "frac_within_10pct": r["frac_within_10pct_log"],
            },
        }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----- Plots -----
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    for col, (name, r) in enumerate(results.items()):
        ax = axes[0, col]
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", alpha=0.5)
        ax.plot([y.min(), y.max()], [1.05 * y.min(), 1.05 * y.max()], "k:", alpha=0.3)
        ax.plot([y.min(), y.max()], [0.95 * y.min(), 0.95 * y.max()], "k:", alpha=0.3)
        ax.scatter(y, r["preds"], s=15, alpha=0.7)
        ax.set_title(f"{name} (direct)\nmedian {r['median_err']:.2f}%")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("predicted ω-mag (dps)")
        ax = axes[1, col]
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", alpha=0.5)
        ax.plot([y.min(), y.max()], [1.05 * y.min(), 1.05 * y.max()], "k:", alpha=0.3)
        ax.plot([y.min(), y.max()], [0.95 * y.min(), 0.95 * y.max()], "k:", alpha=0.3)
        ax.scatter(y, r["preds_from_log"], s=15, alpha=0.7, color="tab:orange")
        ax.set_title(f"{name} (log target)\nmedian {r['median_err_log']:.2f}%")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("truth ω-mag (dps)")
        if col == 0:
            ax.set_ylabel("predicted ω-mag (dps)")
    fig.suptitle(f"s008 — LOO ω-mag regression on 100 m048 truth LCs (s007 oracle = 7.4%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter.png", dpi=120)
    plt.close(fig)

    # feature importance bar
    fig, ax = plt.subplots(figsize=(10, 7))
    top_n = 15
    idx = fi_order[:top_n]
    ax.barh(range(top_n), fi[idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
    ax.set_xlabel("RF feature importance")
    ax.set_title("s008 — top-15 features (RF, ω-mag target, full-data fit)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    print(f"\n=== s008 summary (wall {summary['wall_seconds']:.2f} s) ===")
    print(f"best: {best_name} ({best_kind})  median LOO MAPE = {best_median:.2f}%")
    print(f"3% bar: {'PASS' if summary['passes_3pct_bar'] else 'FAIL'}")
    print(f"beats s007 oracle (7.4%): {'YES' if summary['beats_s007_oracle'] else 'NO'}")
    print("Saved:")
    print(f"  {OUT_DIR / 'features.npz'}")
    print(f"  {OUT_DIR / 'predictions.npz'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {OUT_DIR / 'scatter.png'}")
    print(f"  {OUT_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
