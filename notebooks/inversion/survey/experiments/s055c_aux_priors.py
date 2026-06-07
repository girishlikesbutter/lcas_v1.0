"""s055c — Two auxiliary LC-derived priors after s055b L̂ direction failed.

After s055b found L̂ direction is operationally weak (74° holdout median, only
15° better than random), this experiment tests two SCALAR priors that should
be more directly recoverable from the LC and that prune ω-direction in
useful, complementary ways:

  1. **abs_cos_L_PAB ∈ [0, 1]** — the unsigned cosine of the angle between
     L̂ and the mean PAB direction. The LC modulation depth (and acf
     features) is monotonically tied to the L̂-PAB angle: when L̂≈PAB the
     body's tumble axis points at the viewer (low modulation); when L̂⊥PAB
     the tumble sweeps facets across the half-vector (high modulation).
     Sign is degenerate (PAB and -PAB indistinguishable in modulation), so
     we regress |cos|. **Operational use**: an estimate of |cos(L̂, PAB)|
     constrains L̂ to an "annulus" on the unit sphere (a cone of one
     half-angle around PAB), reducing ω-direction grid by ~3-5×.

  2. **polhode_class ∈ {I_a, separatrix, I_c}** — the polhode topology
     class determined by `D = 2T · I_b / |L|²`:
        D < 0.95           → I_a-enclosing  (class 0)
        0.95 ≤ D ≤ 1.05    → near-separatrix (class 1)
        D > 1.05           → I_c-enclosing  (class 2)
     m048 cohort distribution (s053): ~13/87/50% I_a/I_c/separatrix overlap.
     **Operational use**: knowing the topology class rejects ω-grid points
     whose implied D lies in the wrong class — different classes have
     structurally different ω̂(t) precession patterns in body frame.

Both targets are computed analytically from cached truth (q0, ω0, I).

Method:
  • Train cohort: 100 seeds. Test holdout: 20 seeds (m048 100..119).
  • Features: 28 s008 features (LC-only). |cos(L̂, PAB)| in particular is
    a rotation-invariant scalar so PAB direction features aren't needed.
  • cos regression: linear, ridge, RF (300×8), GBR (300×4). Direct target.
    Report RMSE on [0,1], MAE, p90 absolute error, frac within {0.1, 0.2}.
  • polhode classification: LogisticRegression (multinomial, balanced),
    RidgeClassifier, RandomForest (300×8), GradientBoosting (multinomial
    via OneVsRest) — actually GBR multiclass is built-in in sklearn 1.x;
    use RandomForestClassifier and HistGradientBoostingClassifier.
    Report LOO accuracy, balanced accuracy, holdout accuracy, confusion
    matrix.

Decision:
  • cos RMSE < 0.15 (≈1 std for [0,1] range with cohort spread) →
    operational; ω-direction grid pruning by |cos|-cone constraint.
  • polhode_class accuracy > 75% on holdout → operational; rejects
    grid points on wrong topology.
  • Both must beat random baselines (cos: predict cohort mean; class:
    predict majority class).

Outputs (under results/s055c_aux_priors/):
  features.npz        - X[120,28], y_abs_cos[120], y_polclass[120],
                        seeds, is_cohort_mask
  cos_regression.npz  - per-model {LOO, holdout} predictions for |cos|
  class_regression.npz- per-model {LOO, holdout} predictions for polclass
  summary.json        - decision-grade scalars
  cos_scatter.png     - actual vs predicted |cos|, 2x4 grid
  class_confusion.png - confusion matrices, 1x4 grid
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quaternion as nq
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    RidgeClassifier,
    RidgeCV,
)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hifi_render import _build_model  # noqa: E402
from lib.lc_features import FEATURE_NAMES, lc_features  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402

OUT_DIR = ROOT / "results" / "s055c_aux_priors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_SEEDS = list(range(100))
HOLDOUT_SEEDS = list(range(100, 120))
ALL_SEEDS = COHORT_SEEDS + HOLDOUT_SEEDS

POLHODE_CLASS_NAMES = ["I_a-encl (D<1)", "I_c-encl (D≥1)"]
N_CLASSES = 2


def _polclass(D: float) -> int:
    """Binary topology class: 0 = I_a-enclosing (D<1), 1 = I_c-enclosing (D≥1).

    MEMORY.md / s053 reports ~13/87 split for the cohort under this rule.
    A 3-class refinement (sub-classifying near-separatrix) gave 0 seeds
    in the I_a class with the 0.05 separatrix half-width, so we use the
    cleaner binary cut here.
    """
    return 0 if D < 1.0 else 1


# ---------------------------------------------------------------------------
# Truth + features
# ---------------------------------------------------------------------------

def truth_targets_and_features(seed: int, I_body: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return (X[28], abs_cos_L_PAB, polhode_class) for one seed."""
    truth = load_truth(seed)
    feats, _ = lc_features(truth["observation_times"], truth["mag_hifi"])

    # L̂ in J2000 (passive convention)
    q0 = truth["q0_wxyz"]
    omega0 = truth["omega0_rad"]
    q_obj = nq.quaternion(*q0)
    R = nq.as_rotation_matrix(q_obj)  # passive J2000 → body
    L_body = I_body @ omega0
    L_inertial = R.T @ L_body
    L_hat = L_inertial / np.linalg.norm(L_inertial)

    # PAB mean direction in J2000
    sun_unit = (truth["sun_pos"] - truth["sat_pos"])
    sun_unit = sun_unit / np.linalg.norm(sun_unit, axis=1, keepdims=True)
    obs_unit = (truth["obs_pos"] - truth["sat_pos"])
    obs_unit = obs_unit / np.linalg.norm(obs_unit, axis=1, keepdims=True)
    pab = sun_unit + obs_unit
    pab = pab / np.linalg.norm(pab, axis=1, keepdims=True)
    pab_mean = pab.mean(axis=0)
    pab_mean = pab_mean / np.linalg.norm(pab_mean)

    cos_L_PAB = float(np.dot(L_hat, pab_mean))
    abs_cos = float(abs(cos_L_PAB))

    # Polhode topology class via D = 2T · I_b / |L|²
    twoT = float(omega0 @ (I_body @ omega0))
    L_mag_sq = float(L_body @ L_body)
    I_b = float(np.linalg.eigvalsh(I_body)[1])  # middle eigenvalue
    D = twoT * I_b / L_mag_sq
    polclass = _polclass(D)

    return feats, abs_cos, polclass, D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loo_predict_reg(model_proto, X, y) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = clone(model_proto)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def loo_predict_clf(model_proto, X, y) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    for tr, te in loo.split(X):
        m = clone(model_proto)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])[0]
    return preds


def cos_metrics(y_true, y_pred) -> dict:
    abs_err = np.abs(y_pred - y_true)
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(abs_err)),
        "p90_abs": float(np.percentile(abs_err, 90)),
        "max_abs": float(np.max(abs_err)),
        "frac_within_0p1": float(np.mean(abs_err < 0.1)),
        "frac_within_0p2": float(np.mean(abs_err < 0.2)),
        "frac_within_0p3": float(np.mean(abs_err < 0.3)),
    }


def class_metrics(y_true, y_pred) -> dict:
    labels = list(range(N_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(np.mean(y_true == y_pred))
    # balanced_accuracy needs both classes present in y_true; if only one
    # class appears, balanced_accuracy is the per-class recall of that
    # class, which is just accuracy. Wrap in try.
    try:
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bacc = acc
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cos_scatter(y_true_loo, y_true_hold, preds_loo, preds_hold, fname):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for col, name in enumerate(["linear", "ridge", "rf", "gbr"]):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.scatter(y_true_loo, preds_loo[name], s=15, alpha=0.7, color="tab:blue")
        m = cos_metrics(y_true_loo, preds_loo[name])
        ax.set_title(f"{name} (LOO)\nrmse {m['rmse']:.3f}  <0.1 {m['frac_within_0p1']:.0%}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("predicted |cos(L̂, PAB)|")
        ax = axes[1, col]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.scatter(y_true_hold, preds_hold[name], s=22, alpha=0.85, color="tab:orange")
        m = cos_metrics(y_true_hold, preds_hold[name])
        ax.set_title(f"{name} (holdout)\nrmse {m['rmse']:.3f}  <0.1 {m['frac_within_0p1']:.0%}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("truth |cos(L̂, PAB)|")
        if col == 0:
            ax.set_ylabel("predicted |cos(L̂, PAB)|")
    fig.suptitle("s055c — |cos(L̂, PAB)| regression  (rows: LOO / holdout)")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_class_confusion(metrics_dict, split_name, fname):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    short_names = ["I_a", "I_c"][:N_CLASSES]
    for ax, name in zip(axes, ["logreg", "ridge", "rf", "hgb"]):
        cm = np.array(metrics_dict[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=11)
        ax.set_xticks(list(range(N_CLASSES)))
        ax.set_yticks(list(range(N_CLASSES)))
        ax.set_xticklabels(short_names)
        ax.set_yticklabels(short_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("truth")
        ax.set_title(
            f"{name}  acc {metrics_dict[name]['accuracy']:.2f}  "
            f"bacc {metrics_dict[name]['balanced_accuracy']:.2f}"
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"s055c — polhode_class confusion matrix ({split_name})")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    print(f"s055c — auxiliary priors on {len(ALL_SEEDS)} seeds (100 cohort + 20 holdout)")
    print("  building model (satellite + inertia)...")
    _, I_body = _build_model()

    # --- 1. Truth + features ---
    print("  extracting truth + features...")
    X = np.empty((len(ALL_SEEDS), len(FEATURE_NAMES)))
    y_abscos = np.empty(len(ALL_SEEDS))
    y_polclass = np.empty(len(ALL_SEEDS), dtype=int)
    y_D = np.empty(len(ALL_SEEDS))
    for i, seed in enumerate(ALL_SEEDS):
        feats, abs_cos, polclass, D = truth_targets_and_features(seed, I_body)
        X[i] = feats
        y_abscos[i] = abs_cos
        y_polclass[i] = polclass
        y_D[i] = D
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(ALL_SEEDS)}")
    is_cohort = np.array([s in set(COHORT_SEEDS) for s in ALL_SEEDS])
    print(f"  X={X.shape}  |cos| range [{y_abscos.min():.3f}, {y_abscos.max():.3f}]  "
          f"D range [{y_D.min():.3f}, {y_D.max():.3f}]")
    counts_co = np.bincount(y_polclass[is_cohort], minlength=N_CLASSES)
    counts_ho = np.bincount(y_polclass[~is_cohort], minlength=N_CLASSES)
    print(f"  polclass counts COHORT  I_a/I_c = {counts_co.tolist()}")
    print(f"  polclass counts HOLDOUT I_a/I_c = {counts_ho.tolist()}")

    Xc, Xh = X[is_cohort], X[~is_cohort]
    yc_cos, yh_cos = y_abscos[is_cohort], y_abscos[~is_cohort]
    yc_cls, yh_cls = y_polclass[is_cohort], y_polclass[~is_cohort]

    # --- 2. cos(L̂, PAB) regression ---
    print("\n--- |cos(L̂, PAB)| regression ---")
    reg_models = {
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 2, 20)),
        "rf": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=0, n_jobs=1
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, random_state=0
        ),
    }
    cos_loo_pred = {}
    cos_hold_pred = {}
    cos_loo_metrics = {}
    cos_hold_metrics = {}
    for name, proto in reg_models.items():
        loo = loo_predict_reg(proto, Xc, yc_cos)
        full = clone(proto)
        full.fit(Xc, yc_cos)
        hold = full.predict(Xh)
        # Clip predictions to [0, 1] (target range)
        loo = np.clip(loo, 0.0, 1.0)
        hold = np.clip(hold, 0.0, 1.0)
        cos_loo_pred[name] = loo
        cos_hold_pred[name] = hold
        cos_loo_metrics[name] = cos_metrics(yc_cos, loo)
        cos_hold_metrics[name] = cos_metrics(yh_cos, hold)
        c, h = cos_loo_metrics[name], cos_hold_metrics[name]
        print(
            f"  {name:6s}  LOO rmse {c['rmse']:.3f}  mae {c['mae']:.3f}  "
            f"<0.1 {c['frac_within_0p1']:.0%}   "
            f"HOLDOUT rmse {h['rmse']:.3f}  mae {h['mae']:.3f}  "
            f"<0.1 {h['frac_within_0p1']:.0%}"
        )

    # Cohort-mean baseline
    yc_mean = float(yc_cos.mean())
    base_loo = cos_metrics(yc_cos, np.full_like(yc_cos, yc_mean))
    base_hold = cos_metrics(yh_cos, np.full_like(yh_cos, yc_mean))
    print(f"  COHORT-MEAN baseline ({yc_mean:.3f}):  "
          f"LOO rmse {base_loo['rmse']:.3f}   HOLDOUT rmse {base_hold['rmse']:.3f}")

    best_cos_name = min(cos_hold_metrics.keys(), key=lambda k: cos_hold_metrics[k]["rmse"])
    best_cos_rmse = cos_hold_metrics[best_cos_name]["rmse"]
    cos_decision = (
        "OPERATIONAL" if best_cos_rmse < 0.15
        else ("COARSE_USEFUL" if best_cos_rmse < 0.25 else "WEAK")
    )
    cos_beats_baseline = best_cos_rmse < base_hold["rmse"] - 0.02

    # --- 3. polhode_class classification ---
    print("\n--- polhode_class classification (3-class) ---")
    clf_models = {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=0,
        ),
        "ridge": RidgeClassifier(class_weight="balanced", random_state=0),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=0, n_jobs=1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, random_state=0,
            class_weight="balanced",
        ),
    }
    cls_loo_pred = {}
    cls_hold_pred = {}
    cls_loo_metrics = {}
    cls_hold_metrics = {}
    for name, proto in clf_models.items():
        loo = loo_predict_clf(proto, Xc, yc_cls)
        full = clone(proto)
        full.fit(Xc, yc_cls)
        hold = full.predict(Xh)
        cls_loo_pred[name] = loo
        cls_hold_pred[name] = hold
        cls_loo_metrics[name] = class_metrics(yc_cls, loo)
        cls_hold_metrics[name] = class_metrics(yh_cls, hold)
        c, h = cls_loo_metrics[name], cls_hold_metrics[name]
        print(
            f"  {name:7s}  LOO acc {c['accuracy']:.2f}  bacc {c['balanced_accuracy']:.2f}   "
            f"HOLDOUT acc {h['accuracy']:.2f}  bacc {h['balanced_accuracy']:.2f}"
        )

    # Majority-class baseline
    majority = int(np.bincount(yc_cls).argmax())
    base_acc_loo = float(np.mean(yc_cls == majority))
    base_acc_hold = float(np.mean(yh_cls == majority))
    print(f"  MAJORITY-CLASS baseline (class {majority} = {POLHODE_CLASS_NAMES[majority]}):  "
          f"LOO acc {base_acc_loo:.2f}   HOLDOUT acc {base_acc_hold:.2f}")

    best_cls_name = max(cls_hold_metrics.keys(), key=lambda k: cls_hold_metrics[k]["balanced_accuracy"])
    best_cls_bacc = cls_hold_metrics[best_cls_name]["balanced_accuracy"]
    cls_decision = (
        "OPERATIONAL" if best_cls_bacc > 0.75
        else ("COARSE_USEFUL" if best_cls_bacc > 0.55 else "WEAK")
    )

    # --- 4. Save ---
    np.savez(
        OUT_DIR / "features.npz",
        X=X, y_abs_cos=y_abscos, y_polclass=y_polclass, y_D=y_D,
        seeds=np.array(ALL_SEEDS), is_cohort_mask=is_cohort,
        feature_names=np.array(FEATURE_NAMES),
    )
    cos_payload = {
        "y_truth_loo": yc_cos, "y_truth_holdout": yh_cos,
        "seeds_loo": np.array(COHORT_SEEDS), "seeds_holdout": np.array(HOLDOUT_SEEDS),
    }
    for name in cos_loo_pred:
        cos_payload[f"{name}__loo_pred"] = cos_loo_pred[name]
        cos_payload[f"{name}__holdout_pred"] = cos_hold_pred[name]
    np.savez(OUT_DIR / "cos_regression.npz", **cos_payload)
    cls_payload = {
        "y_truth_loo": yc_cls, "y_truth_holdout": yh_cls,
        "seeds_loo": np.array(COHORT_SEEDS), "seeds_holdout": np.array(HOLDOUT_SEEDS),
    }
    for name in cls_loo_pred:
        cls_payload[f"{name}__loo_pred"] = cls_loo_pred[name]
        cls_payload[f"{name}__holdout_pred"] = cls_hold_pred[name]
    np.savez(OUT_DIR / "class_regression.npz", **cls_payload)

    summary = {
        "n_cohort": int(is_cohort.sum()),
        "n_holdout": int((~is_cohort).sum()),
        "n_features": int(X.shape[1]),
        "wall_seconds": float(time.perf_counter() - t0),
        "cohort_polclass_counts": counts_co.tolist(),
        "holdout_polclass_counts": counts_ho.tolist(),
        "cos_results": {
            "loo": cos_loo_metrics,
            "holdout": cos_hold_metrics,
            "best_holdout_model": best_cos_name,
            "best_holdout_rmse": best_cos_rmse,
            "decision": cos_decision,
            "cohort_mean_baseline_rmse_holdout": base_hold["rmse"],
            "beats_baseline_by_0p02": cos_beats_baseline,
        },
        "polclass_results": {
            "loo": cls_loo_metrics,
            "holdout": cls_hold_metrics,
            "best_holdout_model": best_cls_name,
            "best_holdout_balanced_accuracy": best_cls_bacc,
            "decision": cls_decision,
            "majority_class_baseline_acc_holdout": base_acc_hold,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_cos_scatter(yc_cos, yh_cos, cos_loo_pred, cos_hold_pred,
                     OUT_DIR / "cos_scatter.png")
    plot_class_confusion(cls_hold_metrics, "holdout",
                         OUT_DIR / "class_confusion.png")

    print(f"\n=== s055c summary (wall {summary['wall_seconds']:.2f} s) ===")
    print(f"|cos(L̂,PAB)|  best holdout = {best_cos_name}  rmse = {best_cos_rmse:.3f}  "
          f"(baseline {base_hold['rmse']:.3f})  →  {cos_decision}")
    print(f"polhode_class best holdout = {best_cls_name}  bacc = {best_cls_bacc:.3f}  "
          f"(majority baseline {base_acc_hold:.3f})  →  {cls_decision}")
    print("Saved:")
    for f_ in [
        "features.npz", "cos_regression.npz", "class_regression.npz",
        "summary.json", "cos_scatter.png", "class_confusion.png",
    ]:
        print(f"  {OUT_DIR / f_}")


if __name__ == "__main__":
    main()
