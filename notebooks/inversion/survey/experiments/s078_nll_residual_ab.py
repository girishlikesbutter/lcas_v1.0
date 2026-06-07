"""s078 — RF25 NLL residual: staged A/B against plain magnitude MSE.

Background
----------
s073e's #1 implementation transplant from Robinson & Frueh 2025 is their Eq. 3
NLL loss: a flux-space negative-log-likelihood with per-timestep σ_k weighting
and a ‖S‖/‖Ŝ‖ signal rescaling. RF25 argue magnitude-space ℓ2 overfits bright
specular glints. s059/s069 phantom basins are the symptom we have seen.

The primitive landed in `lib/surrogate_eval.py::nll_residual` / `nll_cost`
(opt-in; production default unchanged). Scope note from that module: with the
m048 fixed 0.05-mag noise floor, the σ_k weighting is ~first-order equivalent
to magnitude MSE, so the genuine lever is the ‖S‖/‖Ŝ‖ rescaling — the `rescale`
flag isolates the two effects.

Staged A/B (user-chosen design)
-------------------------------
Step 1a — re-score the 20 cached s069 polished states (pred_hifi is cached →
          pure file-load). Does NLL rank hi-fi ρ better than plain MSE?
Step 1b — render + re-score all 5000 s059k score-grid candidates. Where does
          the truth-nearest grid cell (idx 1709, cached score rank 1710) land
          under full-LC MSE vs NLL?
Step 2  — re-polish the non-Band-A polished states with the NLL residual.
          Does NLL's basin move any phantom basin toward Band A?

Substrate (all cached, post-fix seed 89)
----------------------------------------
  results/s059k_nd800_seed89/seed089/polished_states.npz  (20 polished states)
  results/s059k_nd800_seed89/seed089/clusters.npz         (5000 grid candidates)

Saves
-----
  results/s078/step1a_polished_rescore.npz / .json
  results/s078/step1b_grid_rescore.npz
  results/s078/step2_nll_repolish.npz
  results/s078/summary.json
  results/s078/*.png
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# BLAS threads = 1 before any heavy import (Pool discipline).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import spearmanr

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import _build_model  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg  # noqa: E402
from lib import surrogate_eval  # noqa: E402

RESULTS_DIR = SURVEY_ROOT / "results" / "s078"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 89
S059K = SURVEY_ROOT / "results" / "s059k_nd800_seed89" / "seed089"

# rho-band thresholds (concepts/rho_band.md): A<2, B<4, C<8, else D.
def rho_band(rho: float) -> str:
    if rho < 2.0:
        return "A"
    if rho < 4.0:
        return "B"
    if rho < 8.0:
        return "C"
    return "D"


# --- worker globals (filled in _init_worker) ---
_W = {}


def _init_worker(times, sun, obs, sat, inertia, obs_dist, target):
    surrogate_eval.get_model()
    _W.update(dict(times=times, sun=sun, obs=obs, sat=sat,
                   inertia=inertia, obs_dist=obs_dist, target=target))


def _render(q0, om0):
    """(q0, ω0) -> surrogate full-LC magnitude prediction on seed 89 geometry."""
    k1, k2, _ = propagate_to_body_frame(
        q0, om0, _W["times"], _W["sun"], _W["obs"], _W["sat"], _W["inertia"])
    return surrogate_eval.predict(k1, k2, _W["obs_dist"])


def _score_one(args):
    """Step 1b worker: render one grid candidate, return the three scores."""
    idx, q0, om0 = args
    try:
        pred = _render(q0, om0)
    except Exception:
        return idx, np.nan, np.nan, np.nan
    mse = surrogate_eval.full_lc_mse(pred, _W["target"])
    nll_r = surrogate_eval.nll_cost(pred, _W["target"], rescale=True)
    nll_n = surrogate_eval.nll_cost(pred, _W["target"], rescale=False)
    return idx, mse, nll_r, nll_n


def _repolish_one(args):
    """Step 2 worker: re-polish one state.

    mode='nll'  -> full-LC NLL residual (rescale=True)
    mode='mse'  -> full-LC MSE residual (the control: isolates NLL from the
                   full-LC-vs-local-window effect, since the cached states
                   were local-window polished).
    """
    tag, q0_start, om0_start, mode = args
    q0_start = np.asarray(q0_start, dtype=np.float64)
    om0_start = np.asarray(om0_start, dtype=np.float64)

    # 6-DOF state: 3 rotation-vector components (delta from q0_start) + 3 omega.
    def _q_from_dtheta(dtheta):
        ang = np.linalg.norm(dtheta)
        if ang < 1e-12:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            ax = dtheta / ang
            dq = np.array([np.cos(ang / 2.0), *(np.sin(ang / 2.0) * ax)])
        # Hamilton product dq ⊗ q0_start (scalar-first).
        w0, x0, y0, z0 = q0_start
        w1, x1, y1, z1 = dq
        q = np.array([
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ])
        return q / np.linalg.norm(q)

    def residuals(x):
        q0 = _q_from_dtheta(x[:3])
        om0 = x[3:6]
        try:
            pred = _render(q0, om0)
        except Exception:
            return np.full_like(_W["target"], 1e3)
        if mode == "nll":
            return surrogate_eval.nll_residual(pred, _W["target"], rescale=True)
        # mode == "mse": plain magnitude residual (full-LC).
        r = pred - _W["target"]
        return np.where(np.isfinite(r), r, 0.0)

    x0 = np.concatenate([np.zeros(3), om0_start])
    t0 = time.time()
    sol = least_squares(residuals, x0, method="lm", max_nfev=400)
    wall = time.time() - t0
    q0_fin = _q_from_dtheta(sol.x[:3])
    om0_fin = sol.x[3:6]
    pred_fin = _render(q0_fin, om0_fin)
    mse_fin = surrogate_eval.full_lc_mse(pred_fin, _W["target"])
    return dict(tag=tag, mode=mode, q0_fin=q0_fin, om0_fin=om0_fin,
                mse_fin=mse_fin, n_eval=int(sol.nfev), wall=wall)


def main() -> int:
    import multiprocessing as mp

    # --- shared seed-89 geometry / target ---
    d = load_truth(SEED)
    times = d["observation_times"]
    sun, obs, sat = d["sun_pos"], d["obs_pos"], d["sat_pos"]
    obs_dist = d["obs_dist"]
    target = d["mag_hifi"]                 # truth hi-fi LC = the A/B target
    q0_truth = np.asarray(d["q0_wxyz"], dtype=np.float64)
    om0_truth = np.asarray(d["omega0_rad"], dtype=np.float64)
    _, inertia = _build_model()
    inertia = np.asarray(inertia, dtype=np.float64)

    # Populate worker globals in the main process too (post-Pool rendering).
    _init_worker(times, sun, obs, sat, inertia, obs_dist, target)

    n_valid = int(np.isfinite(target).sum())
    print(f"=== s078 — RF25 NLL residual A/B (seed {SEED}) ===")
    print(f"  target: truth hi-fi LC, {n_valid}/{target.shape[0]} finite epochs")
    print()

    # ===================================================================
    # Step 1a — re-score the 20 cached polished states (pure file-load)
    # ===================================================================
    pol = np.load(S059K / "polished_states.npz", allow_pickle=True)
    pred_hifi = pol["pred_hifi"]            # (20, 500) cached predictions
    band_cached = pol["band"]
    rho_hifi = pol["rho_polished_hifi"]
    cluster_id = pol["cluster_id"]
    cluster_rank = pol["cluster_rank"]
    q0_err_pol = pol["q0_err_deg"]
    n_pol = pred_hifi.shape[0]

    s1a_mse = np.array([surrogate_eval.full_lc_mse(pred_hifi[i], target)
                        for i in range(n_pol)])
    s1a_nll_r = np.array([surrogate_eval.nll_cost(pred_hifi[i], target, rescale=True)
                          for i in range(n_pol)])
    s1a_nll_n = np.array([surrogate_eval.nll_cost(pred_hifi[i], target, rescale=False)
                          for i in range(n_pol)])

    # How well does each score track the hi-fi ground truth (rho)?
    rho_mse = spearmanr(s1a_mse, rho_hifi).correlation
    rho_nllr = spearmanr(s1a_nll_r, rho_hifi).correlation
    rho_nlln = spearmanr(s1a_nll_n, rho_hifi).correlation

    # Rank of the best (lowest hi-fi rho) state under each score.
    best_hifi = int(np.argmin(rho_hifi))
    def _rank_of(score, idx):
        return int(np.sum(score < score[idx]) + 1)
    rank_best_mse = _rank_of(s1a_mse, best_hifi)
    rank_best_nllr = _rank_of(s1a_nll_r, best_hifi)
    rank_best_nlln = _rank_of(s1a_nll_n, best_hifi)

    print("--- Step 1a: re-score 20 cached polished states ---")
    print(f"  Spearman ρ(score, hi-fi ρ):  MSE={rho_mse:.3f}  "
          f"NLL-rescale={rho_nllr:.3f}  NLL-norescale={rho_nlln:.3f}")
    print(f"  best hi-fi state (idx {best_hifi}, hi-fi ρ={rho_hifi[best_hifi]:.3f}, "
          f"band {band_cached[best_hifi]}) ranks: "
          f"MSE #{rank_best_mse}  NLL-rescale #{rank_best_nllr}  "
          f"NLL-norescale #{rank_best_nlln}  (of {n_pol})")
    n_A = int(np.sum(band_cached == "A"))
    print(f"  cached bands: {n_A} A, {int(np.sum(band_cached=='B'))} B, "
          f"{int(np.sum(band_cached=='C'))} C, {int(np.sum(band_cached=='D'))} D")
    print()

    np.savez(RESULTS_DIR / "step1a_polished_rescore.npz",
             mse=s1a_mse, nll_rescale=s1a_nll_r, nll_norescale=s1a_nll_n,
             rho_hifi=rho_hifi, band_cached=band_cached,
             cluster_id=cluster_id, cluster_rank=cluster_rank,
             q0_err_deg=q0_err_pol)

    # ===================================================================
    # Step 1b — render + re-score all 5000 score-grid candidates
    # ===================================================================
    clu = np.load(S059K / "clusters.npz", allow_pickle=True)
    Q_top, Om_top = clu["Q_top"], clu["Om_top"]
    mse_cached = clu["mse_top"]                       # local-window score (control)
    truth_idx = int(clu["truth_idx_diagnostic"])      # 1709
    n_grid = Q_top.shape[0]
    print(f"--- Step 1b: render + re-score {n_grid} grid candidates ---")
    print(f"  truth-nearest grid cell: idx {truth_idx}, cached score rank "
          f"{int(clu['truth_score_rank_in_topK'])}")

    args = [(i, Q_top[i], Om_top[i]) for i in range(n_grid)]
    t0 = time.time()
    nproc = min(24, mp.cpu_count())
    with mp.Pool(nproc, initializer=_init_worker,
                 initargs=(times, sun, obs, sat, inertia, obs_dist, target)) as pool:
        results = pool.map(_score_one, args, chunksize=32)
    wall_1b = time.time() - t0

    g_mse = np.full(n_grid, np.nan)
    g_nllr = np.full(n_grid, np.nan)
    g_nlln = np.full(n_grid, np.nan)
    for idx, mse, nr, nn in results:
        g_mse[idx], g_nllr[idx], g_nlln[idx] = mse, nr, nn

    def _grid_rank(score, idx):
        finite = np.isfinite(score)
        return int(np.sum(score[finite] < score[idx]) + 1), int(finite.sum())

    r_cached, _ = _grid_rank(mse_cached, truth_idx)
    r_mse, n_fin = _grid_rank(g_mse, truth_idx)
    r_nllr, _ = _grid_rank(g_nllr, truth_idx)
    r_nlln, _ = _grid_rank(g_nlln, truth_idx)

    print(f"  rendered in {wall_1b:.1f}s on Pool({nproc})  ({n_fin}/{n_grid} finite)")
    print(f"  truth-cell rank:  cached-local-window #{r_cached}  "
          f"full-LC-MSE #{r_mse}  NLL-rescale #{r_nllr}  NLL-norescale #{r_nlln}")
    print()

    np.savez(RESULTS_DIR / "step1b_grid_rescore.npz",
             mse=g_mse, nll_rescale=g_nllr, nll_norescale=g_nlln,
             mse_cached=mse_cached, truth_idx=truth_idx)

    # ===================================================================
    # Step 2 — re-polish the non-Band-A polished states: NLL vs MSE control
    #
    # The cached states were LOCAL-WINDOW MSE polished. To attribute any
    # change to NLL (and not just to full-LC-vs-local-window), Step 2 runs
    # BOTH a full-LC NLL re-polish and a full-LC MSE re-polish control from
    # the same start. All comparisons in surrogate ρ (MSE-rooted).
    # ===================================================================
    flagged = [i for i in range(n_pol) if band_cached[i] != "A"]
    print(f"--- Step 2: re-polish {len(flagged)} non-Band-A states "
          f"(NLL vs full-LC-MSE control) ---")
    repol_args = []
    for i in flagged:
        for mode in ("nll", "mse"):
            repol_args.append((int(i), pol["q0_pol_wxyz"][i],
                               pol["om0_pol_rad"][i], mode))
    t0 = time.time()
    with mp.Pool(nproc, initializer=_init_worker,
                 initargs=(times, sun, obs, sat, inertia, obs_dist, target)) as pool:
        repol = pool.map(_repolish_one, repol_args)
    wall_2 = time.time() - t0

    by_key = {(r["tag"], r["mode"]): r for r in repol}
    s2_rows = []
    for i in flagged:
        pred_pre = _render(pol["q0_pol_wxyz"][i], pol["om0_pol_rad"][i])
        rho_pre = surrogate_eval.rho(pred_pre, target)
        row = dict(idx=i, cluster_id=int(cluster_id[i]),
                   band_pre=str(band_cached[i]), rho_hifi_pre=float(rho_hifi[i]),
                   surrogate_rho_pre=float(rho_pre))
        for mode in ("nll", "mse"):
            r = by_key[(i, mode)]
            pred_post = _render(r["q0_fin"], r["om0_fin"])
            rho_post = surrogate_eval.rho(pred_post, target)
            row[f"surrogate_rho_post_{mode}"] = float(rho_post)
            row[f"band_post_{mode}"] = rho_band(rho_post)
            row[f"q0_moved_deg_{mode}"] = float(
                quat_geodesic_deg(pol["q0_pol_wxyz"][i], r["q0_fin"]))
            row[f"n_eval_{mode}"] = r["n_eval"]
        s2_rows.append(row)

    n_nll_beats_mse = sum(1 for r in s2_rows
                          if r["surrogate_rho_post_nll"] < r["surrogate_rho_post_mse"] - 1e-6)
    n_nll_to_AB = sum(1 for r in s2_rows if r["band_post_nll"] in ("A", "B"))
    n_mse_to_AB = sum(1 for r in s2_rows if r["band_post_mse"] in ("A", "B"))
    print(f"  re-polished in {wall_2:.1f}s ({len(repol_args)} polishes); "
          f"NLL beats MSE-control on surrogate ρ in {n_nll_beats_mse}/{len(s2_rows)}; "
          f"reached band A/B: NLL {n_nll_to_AB}, MSE-control {n_mse_to_AB}")
    for r in s2_rows:
        print(f"    idx {r['idx']:2d} cl{r['cluster_id']:4d}  pre ρ={r['surrogate_rho_pre']:6.2f}"
              f"  ->  MSE-ctrl {r['surrogate_rho_post_mse']:6.2f} ({r['band_post_mse']})"
              f"   NLL {r['surrogate_rho_post_nll']:6.2f} ({r['band_post_nll']})")
    print()

    np.savez(RESULTS_DIR / "step2_nll_repolish.npz",
             flagged_idx=np.array(flagged),
             q0_fin_nll=np.array([by_key[(i, "nll")]["q0_fin"] for i in flagged]),
             om0_fin_nll=np.array([by_key[(i, "nll")]["om0_fin"] for i in flagged]),
             q0_fin_mse=np.array([by_key[(i, "mse")]["q0_fin"] for i in flagged]),
             om0_fin_mse=np.array([by_key[(i, "mse")]["om0_fin"] for i in flagged]))

    # ===================================================================
    # Summary + figures
    # ===================================================================
    summary = {
        "experiment": "s078",
        "seed": SEED,
        "primitive": "lib/surrogate_eval.py::nll_residual / nll_cost (opt-in)",
        "step1a": {
            "n_polished": n_pol,
            "spearman_score_vs_hifi_rho": {
                "mse": float(rho_mse), "nll_rescale": float(rho_nllr),
                "nll_norescale": float(rho_nlln)},
            "best_hifi_state_rank": {
                "mse": rank_best_mse, "nll_rescale": rank_best_nllr,
                "nll_norescale": rank_best_nlln},
            "cached_band_counts": {b: int(np.sum(band_cached == b))
                                   for b in ("A", "B", "C", "D")},
        },
        "step1b": {
            "n_grid": n_grid, "n_finite": n_fin,
            "wall_s": float(wall_1b), "nproc": nproc,
            "truth_cell_idx": truth_idx,
            "truth_cell_rank": {
                "cached_local_window": r_cached, "full_lc_mse": r_mse,
                "nll_rescale": r_nllr, "nll_norescale": r_nlln},
        },
        "step2": {
            "n_flagged": len(s2_rows), "n_polishes": len(repol_args),
            "wall_s": float(wall_2),
            "n_nll_beats_mse_control": n_nll_beats_mse,
            "n_reached_band_AB_nll": n_nll_to_AB,
            "n_reached_band_AB_mse_control": n_mse_to_AB,
            "rows": s2_rows,
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Fig 1: Step 1a — each score vs hi-fi rho
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sc, name, rr in [
        (axes[0], s1a_mse, "plain MSE", rho_mse),
        (axes[1], s1a_nll_r, "NLL (rescale)", rho_nllr),
        (axes[2], s1a_nll_n, "NLL (no rescale)", rho_nlln)]:
        colors = ["tab:green" if b == "A" else "tab:orange" if b == "B"
                  else "tab:red" if b == "C" else "darkred" for b in band_cached]
        ax.scatter(sc, rho_hifi, c=colors, s=55, edgecolors="k", linewidths=0.4)
        ax.set_xlabel(f"{name} score")
        ax.set_ylabel("hi-fi ρ (cached ground truth)")
        ax.set_title(f"{name}\nSpearman ρ = {rr:.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle("s078 Step 1a — does each score track hi-fi ρ? (20 s069 polished states)")
    fig.tight_layout()
    f1 = RESULTS_DIR / "s078_step1a_score_vs_hifi.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # Fig 2: Step 1b — truth-cell rank under each score
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["cached\n(local-window)", "full-LC\nMSE", "NLL\n(rescale)",
              "NLL\n(no rescale)"]
    ranks = [r_cached, r_mse, r_nllr, r_nlln]
    bars = ax.bar(labels, ranks, color=["gray", "tab:blue", "tab:green", "tab:olive"])
    for b, r in zip(bars, ranks):
        ax.text(b.get_x() + b.get_width() / 2, r, f"#{r}", ha="center",
                va="bottom", fontsize=10)
    ax.set_ylabel(f"rank of truth-nearest grid cell (of {n_fin})")
    ax.set_title(f"s078 Step 1b — where the truth cell lands under each score\n"
                 f"(seed {SEED}, {n_grid} grid candidates; lower = better)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    f2 = RESULTS_DIR / "s078_step1b_truth_rank.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    # Fig 3: Step 2 — pre vs MSE-control vs NLL re-polish, per flagged state
    fig, ax = plt.subplots(figsize=(13, 5))
    xs = np.arange(len(s2_rows))
    w = 0.27
    ax.bar(xs - w, [r["surrogate_rho_pre"] for r in s2_rows], w,
           label="pre (cached local-window MSE polish)", color="gray")
    ax.bar(xs, [r["surrogate_rho_post_mse"] for r in s2_rows], w,
           label="full-LC MSE re-polish (control)", color="tab:blue")
    ax.bar(xs + w, [r["surrogate_rho_post_nll"] for r in s2_rows], w,
           label="full-LC NLL re-polish", color="tab:green")
    ax.axhline(8.0, color="k", ls=":", lw=1, label="band C/D edge (ρ=8)")
    ax.axhline(2.0, color="tab:red", ls=":", lw=1, label="band A edge (ρ=2)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"cl{r['cluster_id']}" for r in s2_rows], rotation=60,
                       fontsize=7)
    ax.set_ylabel("surrogate ρ (MSE-rooted)")
    ax.set_title(f"s078 Step 2 — NLL re-polish vs full-LC-MSE control "
                 f"(seed {SEED}, {len(s2_rows)} non-Band-A states)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    f3 = RESULTS_DIR / "s078_step2_repolish.png"
    fig.savefig(f3, dpi=130)
    plt.close(fig)

    print(f"Saved: {RESULTS_DIR / 'summary.json'}")
    print(f"Saved: {RESULTS_DIR / 'step1a_polished_rescore.npz'}")
    print(f"Saved: {RESULTS_DIR / 'step1b_grid_rescore.npz'}")
    print(f"Saved: {RESULTS_DIR / 'step2_nll_repolish.npz'}")
    print(f"Saved: {f1}")
    print(f"Saved: {f2}")
    print(f"Saved: {f3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
