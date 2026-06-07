"""s055d — cascade-pool pol_diam filter on cached s049 seed-14 pool.

After s050a/b/c/d/e closed every "re-aggregate the cascade pool" idea
(mag-agreement, multipair, q-space agreement, peak stationarity) by showing
the survivor pool is a uniform thinning with 0.59× anti-enrichment of
truth-q_a — and after s055a confirmed pol_diam IS LC-recoverable at ~25%
holdout MAPE — this experiment tests whether **pol_diam disagreement** is
an orthogonal filter that the cascade hasn't tried.

Mechanism:
  • The s049 cascade pool on seed 14 has 141,706 (q_a, ω) hypotheses; at
    tol=0.10/K=3 it decimates to 35,835 survivors.
  • Each ω hypothesis (a 3-vector in body-frame, dps) implies a polhode
    of computable size: integrate Euler's equations from ω_body for ~1
    polhode period and measure max pairwise L2 distance in body-frame
    ω-space. This is the SAME pol_diam s053 measured for cohort seeds.
  • LC-derived pol_diam estimate for seed 14 is available from s055a
    (LOO prediction at index 14).
  • Filter: reject hypotheses whose IMPLIED pol_diam disagrees with the
    LC-PREDICTED pol_diam by more than X% (or factor of N).
  • This is orthogonal to mag-agreement (per-epoch surrogate brightness
    fit) — pol_diam is a global polhode-shape property derived from the
    full ω trajectory, not a per-epoch test.

Hypothesis: a 50-100% pol_diam disagreement filter rejects most random
hypotheses while keeping truth-q_a survivors (whose ω is approximately
truth and therefore implies truth-pol_diam ≈ 1.97 dps for seed 14).

Method:
  1. Load `s049_cascade_seed14/cascade.npz` (qA_kept, om_kept, delta_mag,
     val_eps, truth_q_a). Apply tol=0.10/K=3 filter (`delta_mag < 0.10` at
     ≥3 of 5 validation epochs) → 35,835 survivors.
  2. Compute truth-q_a mask: q-distance from each qA_kept to truth_q_a;
     threshold at 0.5° → 153 in raw 141k pool, 23 in 35k survivor pool
     (per s049/s050a).
  3. For each survivor's `om_kept` hypothesis (in dps, converted to
     rad/s), integrate Euler's equations on body-frame ω over 1 hour via
     scipy `solve_ivp` (DOP853 method, 500 epochs to match s053 sampling
     density). Compute pol_diam in dps as max pairwise L2 distance on the
     trajectory.
  4. Load LC-predicted pol_diam for seed 14 from s055a's LOO predictions
     (best LOO model = rf-log10).
  5. Apply pol_diam filter at thresholds {25%, 50%, 100%, 200%} of LC
     prediction. Compute total survivors after filter and truth-q_a
     survivors after filter. Enrichment = (truth_after/total_after) /
     (truth_before/total_before).

Decision:
  • Enrichment ≥ 3× → operational; cascade-pool pol_diam filter is the
    first non-trivial truth-discriminator on the s049 pool.
  • 1.5× ≤ enrichment < 3× → modest but useful as one term in a
    multi-filter score.
  • Enrichment < 1.5× (or < 1×, anti-enrichment) → close negative; the
    polhode-prior reframe is operational on |ω| basin width but not on
    cascade-pool truth-discrimination.

Outputs (under results/s055d_cascade_pol_diam_filter/):
  pool.npz       - all survivor indices, om_kept, qA_kept, truth_q_a_mask,
                   delta_mag, validation epochs
  pol_diam.npz   - implied pol_diam per survivor (35k floats), wall time
  enrichment.json- enrichment results at multiple thresholds + decision
  pol_diam_hist.png - histogram of implied pol_diam vs LC pred / truth
  enrichment_curve.png - enrichment vs filter threshold
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
from multiprocessing import Pool
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hifi_render import _build_model  # noqa: E402

CASCADE_NPZ = ROOT / "results" / "s049_cascade_seed14" / "cascade.npz"
S055A_REG_NPZ = ROOT / "results" / "s055a_pol_diam_lc_regression" / "regression.npz"
S053_COHORT_NPZ = ROOT / "results" / "s053_cohort_polhode_survey" / "cohort.npz"

OUT_DIR = ROOT / "results" / "s055d_cascade_pol_diam_filter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 14
TOL_MAG = 0.10
K_REQUIRED = 3
TRUTH_QA_TOL_DEG = 0.5  # q-distance threshold to flag truth-q_a hypothesis

# Polhode integration: 1 hour matches LC observation window + s053 sampling
INTEGRATE_T_TOTAL_S = 3600.0
N_INTEGRATE_SAMPLES = 500
T_EVAL = np.linspace(0.0, INTEGRATE_T_TOTAL_S, N_INTEGRATE_SAMPLES)


# Module-level inertia tensor (set in main; workers reuse via fork)
I_BODY: np.ndarray | None = None
I_INV: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Polhode integration
# ---------------------------------------------------------------------------

def _euler_rhs(t, w, I_inv, I):
    """ω̇ = I^{-1} (-ω × (I·ω)). Body frame, torque-free."""
    Iw = I @ w
    return I_inv @ (-np.cross(w, Iw))


def _pol_diam_one(om_rad: np.ndarray) -> float:
    """Integrate Euler's equations from body-frame ω0 (rad/s) for 1 hour
    and compute pol_diam (max pairwise L2 distance) in dps.

    NOTE: cascade.npz `om_kept` is in RAD/S despite the variable name.
    Verified against `om_truth_at_t0` whose |ω|=0.0217 rad/s matches the
    known seed-14 truth |ω|=1.229 dps.
    """
    sol = solve_ivp(
        _euler_rhs,
        (0.0, INTEGRATE_T_TOTAL_S),
        om_rad,
        t_eval=T_EVAL,
        method="DOP853",
        rtol=1e-8,
        atol=1e-11,
        args=(I_INV, I_BODY),
    )
    if not sol.success:
        return float("nan")
    om_history_dps = np.rad2deg(sol.y.T)  # (N, 3) in dps
    diff = om_history_dps[:, None, :] - om_history_dps[None, :, :]
    return float(np.sqrt(np.sum(diff * diff, axis=2)).max())


def _worker_init(I_body_arr: np.ndarray):
    """Init worker — set module-level I_BODY + I_INV."""
    global I_BODY, I_INV
    I_BODY = I_body_arr
    I_INV = np.linalg.inv(I_body_arr)


def _worker_compute(om_dps_row: np.ndarray) -> float:
    return _pol_diam_one(om_dps_row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quaternion_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """q1: (4,) wxyz reference; q2: (N, 4) wxyz array. Returns (N,) geodesic
    angle in degrees. Quaternions are taken modulo sign (q ≡ -q on SO(3))."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    cos_half = np.abs(q2 @ q1)
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cos_half))


def apply_K_filter(delta_mag: np.ndarray, tol: float, K: int) -> np.ndarray:
    """delta_mag (N, n_eps). Survivor mask: |delta| < tol at >=K of n_eps epochs."""
    return (np.abs(delta_mag) < tol).sum(axis=1) >= K


def enrichment_at_filter(
    n_total_before: int,
    n_truth_before: int,
    n_total_after: int,
    n_truth_after: int,
) -> float:
    """Truth-q_a enrichment factor.

    rate_after = n_truth_after / n_total_after
    rate_before = n_truth_before / n_total_before
    enrichment = rate_after / rate_before
    """
    if n_total_after == 0 or n_total_before == 0 or n_truth_before == 0:
        return float("nan")
    rate_before = n_truth_before / n_total_before
    rate_after = n_truth_after / n_total_after
    return rate_after / rate_before


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pol_diam_hist(
    pol_diam_all: np.ndarray,
    truth_qa_mask: np.ndarray,
    lc_pred: float,
    truth_value: float,
    fname: Path,
):
    """Histogram of implied pol_diam: full pool vs truth-q_a survivors.
    Overlay LC prediction and truth."""
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, max(pol_diam_all.max(), truth_value * 1.5), 60)
    ax.hist(
        pol_diam_all,
        bins=bins,
        alpha=0.5,
        color="tab:blue",
        density=True,
        label=f"all survivors (n={len(pol_diam_all)}, med {np.median(pol_diam_all):.2f})",
    )
    pd_truth_qa = pol_diam_all[truth_qa_mask]
    if len(pd_truth_qa) > 0:
        ax.hist(
            pd_truth_qa,
            bins=bins,
            alpha=0.7,
            color="tab:orange",
            density=True,
            label=f"truth-q_a survivors (n={len(pd_truth_qa)}, "
            f"med {np.median(pd_truth_qa):.2f})",
        )
    ax.axvline(lc_pred, color="g", linestyle="--", linewidth=2,
               label=f"LC prediction = {lc_pred:.2f}")
    ax.axvline(truth_value, color="r", linestyle="--", linewidth=2,
               label=f"truth = {truth_value:.2f}")
    ax.set_xlabel("implied pol_diam (dps)")
    ax.set_ylabel("density")
    ax.set_title(f"s055d — implied pol_diam in cascade pool (seed {SEED}, "
                 f"tol={TOL_MAG}/K={K_REQUIRED})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_enrichment_curve(
    thresholds_pct: list,
    n_total: list,
    n_truth: list,
    enrichments: list,
    n_total_before: int,
    n_truth_before: int,
    fname: Path,
):
    """Enrichment vs filter threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(thresholds_pct, n_total, "o-", label="all survivors")
    ax.plot(thresholds_pct, n_truth, "o-", color="tab:orange", label="truth-q_a survivors")
    ax.axhline(n_total_before, color="tab:blue", linestyle=":", alpha=0.5,
               label=f"pre-filter (all={n_total_before})")
    ax.axhline(n_truth_before, color="tab:orange", linestyle=":", alpha=0.5,
               label=f"pre-filter (truth={n_truth_before})")
    ax.set_yscale("log")
    ax.set_xlabel("pol_diam disagreement threshold (% of LC prediction)")
    ax.set_ylabel("survivor count")
    ax.legend()
    ax.set_title("Survivor counts vs threshold")

    ax = axes[1]
    ax.plot(thresholds_pct, enrichments, "o-", color="tab:purple")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="no enrichment")
    ax.axhline(3.0, color="g", linestyle="--", alpha=0.5, label="3× operational gate")
    ax.set_xlabel("pol_diam disagreement threshold (% of LC prediction)")
    ax.set_ylabel("truth-q_a enrichment factor")
    ax.legend()
    ax.set_title("Truth-q_a enrichment factor")

    fig.suptitle(f"s055d — cascade-pool pol_diam filter (seed {SEED})")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    print(f"s055d — cascade-pool pol_diam filter on seed {SEED}")

    # 1. Inertia
    print("  building inertia tensor...")
    _, I_body = _build_model()
    eigvals = np.linalg.eigvalsh(I_body)
    print(f"    I eigvals = {eigvals}")

    # 2. Load cascade pool
    print(f"  loading {CASCADE_NPZ.name}...")
    casc = np.load(CASCADE_NPZ)
    qA_kept = casc["qA_kept"]
    om_kept = casc["om_kept"]            # RAD/S (verified via om_truth_at_t0 cross-check)
    delta_mag = casc["delta_mag"]        # (N, n_eps)
    val_eps = casc["val_eps"]
    truth_q_a = casc["truth_q_a"]
    om_truth_at_t0 = casc["om_truth_at_t0"]
    om_truth_cascade = casc["om_truth_cascade"]
    n_eps = delta_mag.shape[1]
    print(f"    raw pool: {len(qA_kept)} hypotheses, {n_eps} val_eps")
    print(f"    val_eps = {val_eps.tolist()}")
    print(f"    truth_q_a = {truth_q_a}")
    om_truth_at_t0_dps = np.rad2deg(om_truth_at_t0)
    om_truth_cascade_dps = np.rad2deg(om_truth_cascade)
    print(f"    om_truth_at_t0 (rad/s) → |ω|_dps = {np.linalg.norm(om_truth_at_t0_dps):.4f}")
    print(f"    om_truth_cascade (rad/s) → |ω|_dps = {np.linalg.norm(om_truth_cascade_dps):.4f}")

    # 3. Truth-q_a mask + K-filter survivor mask
    qA_dist_deg = quaternion_geodesic_deg(truth_q_a, qA_kept)
    truth_qa_mask_raw = qA_dist_deg < TRUTH_QA_TOL_DEG
    n_truth_raw = int(truth_qa_mask_raw.sum())
    survivor_mask = apply_K_filter(delta_mag, TOL_MAG, K_REQUIRED)
    n_survivors = int(survivor_mask.sum())
    truth_qa_survivor_mask = survivor_mask & truth_qa_mask_raw
    n_truth_survivors = int(truth_qa_survivor_mask.sum())
    print(f"  truth-q_a in raw pool   ({TRUTH_QA_TOL_DEG}° tol): {n_truth_raw}")
    print(f"  K=3/tol=0.10 survivors: {n_survivors}")
    print(f"  truth-q_a ∩ survivors: {n_truth_survivors}")

    # 4. Truth pol_diam (analytical from cohort)
    cohort = np.load(S053_COHORT_NPZ)
    truth_pol_diam = float(cohort["pol_diam_dps"][SEED])
    print(f"  truth pol_diam (s053, seed {SEED}): {truth_pol_diam:.4f} dps")

    # 5. LC-predicted pol_diam from s055a LOO predictions for seed 14
    s055a_reg = np.load(S055A_REG_NPZ)
    # All 4 model × 2 kind × {loo, holdout}.  Best LOO was rf-log10.
    keys_loo = [k for k in s055a_reg.files if k.startswith("pol_diam__") and k.endswith("__loo_preds")]
    print(f"  s055a LOO keys: {keys_loo}")
    lc_preds_for_seed14 = {}
    for k in keys_loo:
        lc_preds_for_seed14[k] = float(s055a_reg[k][SEED])
    print(f"  s055a seed-{SEED} LOO predictions: {lc_preds_for_seed14}")
    # Use best LOO (rf-log10 per s055a summary). Fallback: linear-direct.
    lc_pred = lc_preds_for_seed14.get("pol_diam__rf__log10__loo_preds")
    if lc_pred is None:
        lc_pred = lc_preds_for_seed14["pol_diam__linear__direct__loo_preds"]
    print(f"  using LC-pred = {lc_pred:.4f} dps (truth {truth_pol_diam:.4f}, "
          f"err {100*abs(lc_pred - truth_pol_diam)/truth_pol_diam:.1f}%)")

    # 6. Compute implied pol_diam for ALL 141k raw hypotheses (so we can
    # compare pol_diam filter applied BEFORE vs AFTER the K-filter).
    print(f"  computing implied pol_diam for ALL {len(om_kept)} raw hypotheses "
          f"(Pool(8), N_int={N_INTEGRATE_SAMPLES})...")
    t_int_start = time.perf_counter()
    with Pool(processes=8, initializer=_worker_init, initargs=(I_body,)) as pool:
        pol_diam_all = np.array(pool.map(_worker_compute, om_kept))
    t_int = time.perf_counter() - t_int_start
    print(f"    wall {t_int:.1f}s ({1000*t_int/len(om_kept):.2f} ms / hyp)")
    print(f"    raw pool pol_diam: median {np.median(pol_diam_all):.3f} "
          f"min {pol_diam_all.min():.3f} max {pol_diam_all.max():.3f} dps")
    pol_diam_implied = pol_diam_all[survivor_mask]
    print(f"    K=3-survivors pol_diam: median {np.median(pol_diam_implied):.3f} "
          f"min {pol_diam_implied.min():.3f} max {pol_diam_implied.max():.3f} dps")
    n_nan = int(np.isnan(pol_diam_all).sum())
    if n_nan > 0:
        print(f"    WARNING: {n_nan} integration failures (NaN)")

    # 7. Apply pol_diam filter at multiple thresholds — TWO scenarios:
    #    (A) pol_diam-only on RAW 141k (replaces K-filter) — orthogonality test
    #    (B) pol_diam ON TOP OF K=3 filter (combined filter) — cumulative test
    truth_qa_in_survivors = truth_qa_mask_raw[survivor_mask]
    print("\n  pol_diam disagreement filter:")
    print(f"  RAW pre-filter: {len(qA_kept)} hypotheses, {n_truth_raw} truth-q_a "
          f"({100*n_truth_raw/len(qA_kept):.4f}%)")
    print(f"  K=3 pre-filter: {n_survivors} survivors, {n_truth_survivors} truth-q_a "
          f"({100*n_truth_survivors/n_survivors:.4f}%)")

    thresholds_pct = [10, 15, 25, 50, 100, 150, 200, 300, 500, 1000]
    results_raw: list[dict] = []
    results_combined: list[dict] = []

    for thr in thresholds_pct:
        rel_err_pct_all = 100.0 * np.abs(pol_diam_all - lc_pred) / lc_pred
        # Scenario A: pol_diam-only on raw pool
        keep_A = (rel_err_pct_all < thr) & ~np.isnan(pol_diam_all)
        n_A = int(keep_A.sum())
        n_truth_A = int((keep_A & truth_qa_mask_raw).sum())
        enr_A = enrichment_at_filter(len(qA_kept), n_truth_raw, n_A, n_truth_A)
        results_raw.append({
            "threshold_pct": thr,
            "n_total_after": n_A,
            "n_truth_after": n_truth_A,
            "enrichment": enr_A,
            "decimation_factor": n_A / max(len(qA_kept), 1),
        })
        # Scenario B: pol_diam ∩ K=3
        keep_B = keep_A & survivor_mask
        n_B = int(keep_B.sum())
        n_truth_B = int((keep_B & truth_qa_mask_raw).sum())
        enr_B = enrichment_at_filter(n_survivors, n_truth_survivors, n_B, n_truth_B)
        results_combined.append({
            "threshold_pct": thr,
            "n_total_after": n_B,
            "n_truth_after": n_truth_B,
            "enrichment": enr_B,
            "decimation_factor": n_B / max(n_survivors, 1),
        })
        print(
            f"    thr {thr:>5}%   "
            f"RAW {n_A:>6}/{len(qA_kept):6} ({100*n_A/len(qA_kept):4.1f}%)  "
            f"truth {n_truth_A:>3}/{n_truth_raw:<3} enr {enr_A:5.2f}×    "
            f"K3∩pol {n_B:>6}/{n_survivors:6} ({100*n_B/n_survivors:4.1f}%)  "
            f"truth {n_truth_B:>3}/{n_truth_survivors:<3} enr {enr_B:5.2f}×"
        )

    # For backward compat in saving + plotting, results = combined K3+pol_diam
    results = results_combined

    # 8. Save
    np.savez(
        OUT_DIR / "pool.npz",
        qA_dist_deg=qA_dist_deg,
        survivor_mask=survivor_mask,
        truth_qa_mask_raw=truth_qa_mask_raw,
        truth_qa_in_survivors=truth_qa_in_survivors,
        delta_mag_subset=delta_mag[survivor_mask],
        om_kept_survivors=om_kept[survivor_mask],
        truth_q_a=truth_q_a,
        om_truth_at_t0=om_truth_at_t0,
        om_truth_cascade=om_truth_cascade,
    )
    np.savez(
        OUT_DIR / "pol_diam.npz",
        pol_diam_all=pol_diam_all,                # raw 141k
        pol_diam_implied=pol_diam_implied,        # K=3 survivors only
        lc_pred=lc_pred,
        truth_pol_diam=truth_pol_diam,
        wall_seconds=t_int,
    )

    enrichment_summary = {
        "seed": SEED,
        "tol_mag": TOL_MAG,
        "K_required": K_REQUIRED,
        "n_raw": int(len(qA_kept)),
        "n_truth_raw": n_truth_raw,
        "n_survivors": n_survivors,
        "n_truth_survivors": n_truth_survivors,
        "truth_qa_tol_deg": TRUTH_QA_TOL_DEG,
        "truth_pol_diam": truth_pol_diam,
        "lc_pred_pol_diam": lc_pred,
        "lc_pred_err_vs_truth_pct": 100*abs(lc_pred - truth_pol_diam)/truth_pol_diam,
        "filter_results_raw": results_raw,         # pol_diam-only on raw 141k
        "filter_results_combined": results_combined,  # pol_diam ∩ K=3
        "best_enrichment_raw": float(np.nanmax([r["enrichment"] for r in results_raw])),
        "best_enrichment_combined": float(np.nanmax([r["enrichment"] for r in results_combined])),
        "wall_seconds": float(time.perf_counter() - t0),
    }
    best_enr = max(
        enrichment_summary["best_enrichment_raw"],
        enrichment_summary["best_enrichment_combined"],
    )
    enrichment_summary["best_enrichment"] = best_enr
    if best_enr >= 3.0:
        enrichment_summary["decision"] = "OPERATIONAL"
    elif best_enr >= 1.5:
        enrichment_summary["decision"] = "MODEST"
    else:
        enrichment_summary["decision"] = "WEAK"

    with open(OUT_DIR / "enrichment.json", "w") as f:
        json.dump(enrichment_summary, f, indent=2)

    # 9. Plots
    # Histogram on K=3-survivor pool (matches original intent)
    plot_pol_diam_hist(
        pol_diam_implied, truth_qa_in_survivors, lc_pred, truth_pol_diam,
        OUT_DIR / "pol_diam_hist.png",
    )
    # Enrichment curve: 2 scenarios overlaid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(thresholds_pct, [r["n_total_after"] for r in results_raw],
            "o-", color="tab:blue", label="raw pool kept")
    ax.plot(thresholds_pct, [r["n_truth_after"] for r in results_raw],
            "s--", color="tab:cyan", label="raw pool truth-q_a kept")
    ax.plot(thresholds_pct, [r["n_total_after"] for r in results_combined],
            "o-", color="tab:red", label="K3 ∩ pol_diam kept")
    ax.plot(thresholds_pct, [r["n_truth_after"] for r in results_combined],
            "s--", color="tab:orange", label="K3 ∩ pol_diam truth-q_a kept")
    ax.axhline(len(qA_kept), color="tab:blue", linestyle=":", alpha=0.4,
               label=f"raw total ({len(qA_kept)})")
    ax.axhline(n_truth_raw, color="tab:cyan", linestyle=":", alpha=0.4,
               label=f"raw truth-q_a ({n_truth_raw})")
    ax.axhline(n_survivors, color="tab:red", linestyle=":", alpha=0.4,
               label=f"K3 total ({n_survivors})")
    ax.axhline(n_truth_survivors, color="tab:orange", linestyle=":", alpha=0.4,
               label=f"K3 truth-q_a ({n_truth_survivors})")
    ax.set_yscale("log")
    ax.set_xlabel("pol_diam threshold (% of LC pred)")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    ax.set_title("survivor counts vs threshold (2 scenarios)")

    ax = axes[1]
    ax.plot(thresholds_pct, [r["enrichment"] for r in results_raw],
            "o-", color="tab:blue", label="pol_diam only on raw 141k")
    ax.plot(thresholds_pct, [r["enrichment"] for r in results_combined],
            "o-", color="tab:red", label="pol_diam ∩ K=3 (39591 → ?)")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="no enrichment")
    ax.axhline(3.0, color="g", linestyle="--", alpha=0.5, label="3× operational gate")
    # Cascade K=3 alone enrichment (s050a found 0.59×)
    cascade_enr = (n_truth_survivors / n_survivors) / (n_truth_raw / len(qA_kept))
    ax.axhline(cascade_enr, color="m", linestyle="--", alpha=0.5,
               label=f"K=3 alone ({cascade_enr:.2f}×)")
    ax.set_xlabel("pol_diam threshold (% of LC pred)")
    ax.set_ylabel("truth-q_a enrichment factor")
    ax.legend(fontsize=8)
    ax.set_title("Truth enrichment factor")

    fig.suptitle(f"s055d — cascade-pool pol_diam filter (seed {SEED}, tol={TOL_MAG}/K={K_REQUIRED})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "enrichment_curve.png", dpi=120)
    plt.close(fig)

    print(f"\n=== s055d summary (wall {enrichment_summary['wall_seconds']:.1f} s) ===")
    print(f"Best enrichment = {best_enr:.2f}× → DECISION: {enrichment_summary['decision']}")
    print(f"Truth-q_a survival before filter: {n_truth_survivors}/{n_survivors} "
          f"({100*n_truth_survivors/n_survivors:.3f}%)")
    print(f"Saved:")
    for f_ in ["pool.npz", "pol_diam.npz", "enrichment.json",
              "pol_diam_hist.png", "enrichment_curve.png"]:
        print(f"  {OUT_DIR / f_}")


if __name__ == "__main__":
    main()
