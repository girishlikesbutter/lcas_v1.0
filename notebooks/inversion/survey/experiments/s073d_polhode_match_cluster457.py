"""s073d — Body-frame polhode match check for cluster_457 vs truth (seed 89).

Background
----------
s073 established that on post-fix seed 89, cluster_457 (the lone Band A
multi-sol attractor, q0_err=59.58°, hi-fi ρ=0.904) shares |L| with truth
to 0.55% but its L_J2000 direction is 128.6° off truth. s073c is the
scope-correction audit: this is N=1; "same body-frame polhode" was
asserted as plausible but not directly verified.

This experiment performs that direct verification, the cheapest decisive
check in the cat-4 hypothesis-conversion sequence (s073c §Next #1).

Theory
------
Under torque-free rigid-body motion, ω(t) in the body frame is constrained
to lie on the closed curve given by the simultaneous level sets

    energy:   ω · I · ω = 2T  (constant of motion)
    momentum: |I · ω|²  = |L|² (constant of motion, body-frame magnitude)

This curve in body-frame ω-space is the polhode. Two trajectories share
the same polhode IFF both (2T, |L|²) match — the polhode is fully determined
by (I, 2T, |L|²). The torque-free orbit on that curve is unique up to
phase (where on the curve ω₀ sits) and direction (sign of traversal).

Question
--------
Do truth and cluster_457 lie on the same body-frame polhode?

Three answers, in order of decisiveness:
1. Scalar invariants: do (2T, |L|²) match?
2. Regime / k² (m): does omega_jacobi place them in the same case (A or B)
   with the same elliptic modulus? This is redundant with (1) but a useful
   cross-check that the Jacobi parameterisation is consistent.
3. Set match: does cluster_457's traced ω_body(t) lie on truth's polhode
   (and vice versa) modulo discretisation? Visual + nearest-neighbour distance.

What we compute
---------------
For both states:
  - 2T = ω₀ · I · ω₀
  - |L|² = |I · ω₀|²
  - omega_jacobi(times, ω₀, I) → ω_body(t) on a 1001-point grid spanning
    [0, T_LC] where T_LC = (observation_times[-1] - observation_times[0]).
  - omega_jacobi.info regime, m (k²), polhode period
  - Conservation residuals along the trace (2T, |L|² should be flat)

Cross-comparison:
  - Same-curve set-distance: for each cluster_457 ω(t), nearest distance to
    the truth ω(t) trace, normalised by |ω₀_truth|. Same in reverse.
  - Discrete polhode sample: 1001 points covers ~one full polhode period at
    most cohort cadences; if shorter than one period the test is local.

Convention
----------
Scalar-first quaternions (w, x, y, z). `_quat_to_matrix(q)` is the
post-fix conv-(a) passive J2000→body matrix. ω is body-frame (rad/s),
inertia is body-frame (kg m²). Polhode lives in body-frame ω-space.

Saves
-----
  results/s073d/summary.json — invariants, regime, set-distance, verdict
  results/s073d/omega_traces.npz — ω_body(t) for both states + diagnostics
  results/s073d/polhode_match.png — 3D + 2D-projection scatter

Cross-references
----------------
  experiments/s073_cluster457_l_vector_check.{py,md}  (parent measurement)
  experiments/s073c_literature_novelty_audit.md       (scope correction + plan)
  lib/jacobi_propagator.py::omega_jacobi              (closed-form ω(t))
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# BLAS threading sanity (tiny compute; kept for hygiene)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.jacobi_propagator import omega_jacobi  # noqa: E402
from lib.traj_load import load_truth            # noqa: E402
from lib.hifi_render import build_context       # noqa: E402


RESULTS_DIR = SURVEY_ROOT / "results" / "s073d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def scalar_invariants(omega0: np.ndarray, inertia: np.ndarray) -> tuple[float, float]:
    """Return (2T, |L|²) at a single state."""
    twoT = float(omega0 @ inertia @ omega0)
    L_body = inertia @ omega0
    L2 = float(L_body @ L_body)
    return twoT, L2


def nearest_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of a, distance to nearest row of b. Both (N, 3)."""
    # O(N²) but N=1001 → 1M pairs, trivial.
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(d2.min(axis=1))


def main() -> int:
    seed = 89

    # --- Truth ---
    truth = load_truth(seed)
    q0_truth = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    om0_truth = np.asarray(truth["omega0_rad"], dtype=np.float64)
    obs_times = np.asarray(truth["observation_times"], dtype=np.float64)
    T_LC = float(obs_times[-1] - obs_times[0])  # seconds

    ctx = build_context(seed)
    I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)

    # --- Cluster 457 polished state (same source as s073) ---
    s059k_summary = json.load(open(
        SURVEY_ROOT / "results" / "s059k_nd800_seed89" / "seed089"
        / "full_lc_seeds" / "summary.json"
    ))
    cluster_entries = [e for e in s059k_summary["polished"] if e["cluster_id"] == 457]
    if not cluster_entries:
        print("FAIL: cluster_id=457 not found in s059k full_lc_seeds summary", file=sys.stderr)
        return 1
    entry = next(e for e in cluster_entries if e["mag_pct_offset"] == 0.0)
    q0_c457 = np.asarray(entry["q0_pol_wxyz"], dtype=np.float64)
    om0_c457 = np.asarray(entry["om0_pol_rad"], dtype=np.float64)
    rho_c457 = float(entry["rho_polished_hifi"])

    # --- Scalar invariants ---
    twoT_truth, L2_truth = scalar_invariants(om0_truth, I)
    twoT_c457, L2_c457 = scalar_invariants(om0_c457, I)

    twoT_rel_diff = abs(twoT_c457 - twoT_truth) / abs(twoT_truth)
    L2_rel_diff = abs(L2_c457 - L2_truth) / abs(L2_truth)

    # --- omega_jacobi traces over LC + polhode period diagnostic ---
    # We want one full polhode period to make the set-match test global.
    # Use the closed-form `tau_dot` from info to know the period analytically.
    # Quick info-only call to get period; reuse output below.
    times_short = np.array([0.0])
    _, info_truth = omega_jacobi(times_short, om0_truth, I)
    _, info_c457 = omega_jacobi(times_short, om0_c457, I)

    # Polhode period T_pol = 4 * K(m) / |tau_dot| (full sn/cn period is 4K).
    from scipy.special import ellipk
    T_pol_truth = 4.0 * float(ellipk(info_truth["m"])) / abs(info_truth["tau_dot"])
    T_pol_c457 = 4.0 * float(ellipk(info_c457["m"])) / abs(info_c457["tau_dot"])

    # Span the LARGER of T_LC and T_pol so we always cover at least one full polhode.
    T_span_truth = max(T_LC, T_pol_truth)
    T_span_c457 = max(T_LC, T_pol_c457)
    N_grid = 1001

    times_truth = np.linspace(0.0, T_span_truth, N_grid)
    times_c457 = np.linspace(0.0, T_span_c457, N_grid)

    om_hist_truth, _ = omega_jacobi(times_truth, om0_truth, I)
    om_hist_c457, _ = omega_jacobi(times_c457, om0_c457, I)

    # Conservation along trace (sanity check on closed-form ω)
    twoT_truth_trace = (om_hist_truth * (om_hist_truth @ I.T)).sum(axis=1)
    L2_truth_trace = ((om_hist_truth @ I.T) ** 2).sum(axis=1)
    twoT_c457_trace = (om_hist_c457 * (om_hist_c457 @ I.T)).sum(axis=1)
    L2_c457_trace = ((om_hist_c457 @ I.T) ** 2).sum(axis=1)

    twoT_drift_truth = float((twoT_truth_trace.max() - twoT_truth_trace.min()) / twoT_truth)
    L2_drift_truth = float((L2_truth_trace.max() - L2_truth_trace.min()) / L2_truth)
    twoT_drift_c457 = float((twoT_c457_trace.max() - twoT_c457_trace.min()) / twoT_c457)
    L2_drift_c457 = float((L2_c457_trace.max() - L2_c457_trace.min()) / L2_c457)

    # Set-match: nearest neighbour distance, normalised by |ω₀_truth|.
    om_truth_norm = float(np.linalg.norm(om0_truth))
    d_c457_to_truth = nearest_distance(om_hist_c457, om_hist_truth)
    d_truth_to_c457 = nearest_distance(om_hist_truth, om_hist_c457)
    set_dist_c457_to_truth_rel = float(d_c457_to_truth.max() / om_truth_norm)
    set_dist_truth_to_c457_rel = float(d_truth_to_c457.max() / om_truth_norm)
    # Median set-distance is also informative — max may be dominated by phase
    # endpoints if periods differ slightly.
    set_dist_c457_to_truth_med = float(np.median(d_c457_to_truth) / om_truth_norm)
    set_dist_truth_to_c457_med = float(np.median(d_truth_to_c457) / om_truth_norm)

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(om_hist_truth[:, 0], om_hist_truth[:, 1], om_hist_truth[:, 2],
                 s=4, c="C0", alpha=0.6, label="truth ω(t)")
    ax3d.scatter(om_hist_c457[:, 0], om_hist_c457[:, 1], om_hist_c457[:, 2],
                 s=4, c="C3", alpha=0.6, label="cluster_457 ω(t)")
    ax3d.scatter(*om0_truth, c="C0", s=80, edgecolor="k", label="truth ω₀")
    ax3d.scatter(*om0_c457, c="C3", s=80, marker="^", edgecolor="k", label="c457 ω₀")
    ax3d.set_xlabel("ω_x (rad/s)")
    ax3d.set_ylabel("ω_y (rad/s)")
    ax3d.set_zlabel("ω_z (rad/s)")
    ax3d.set_title(f"Body-frame polhode (seed {seed})")
    ax3d.legend(loc="upper left", fontsize=8)

    # 2D projections
    for k, (i, j, name) in enumerate([(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")]):
        ax = fig.add_subplot(2, 2, 2 + k)
        ax.plot(om_hist_truth[:, i], om_hist_truth[:, j], "-", color="C0",
                lw=0.8, alpha=0.7, label="truth")
        ax.plot(om_hist_c457[:, i], om_hist_c457[:, j], "-", color="C3",
                lw=0.8, alpha=0.7, label="c457")
        ax.scatter(om0_truth[i], om0_truth[j], c="C0", s=60, edgecolor="k", zorder=5)
        ax.scatter(om0_c457[i], om0_c457[j], c="C3", s=60, marker="^",
                   edgecolor="k", zorder=5)
        ax.set_xlabel(f"ω_{'xyz'[i]} (rad/s)")
        ax.set_ylabel(f"ω_{'xyz'[j]} (rad/s)")
        ax.set_title(f"projection {name}")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        if k == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"s073d — polhode match: truth vs cluster_457 (seed {seed})\n"
        f"2T rel diff = {twoT_rel_diff:.3e}   |L|² rel diff = {L2_rel_diff:.3e}   "
        f"regime: truth={info_truth['regime']} c457={info_c457['regime']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = RESULTS_DIR / "polhode_match.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Verdict logic ---
    # "Same polhode" gate: 2T and |L|² match within 1% (the bar set by s073's
    # |L| match at 0.55%; tighter would over-claim, looser would let phantoms in).
    same_polhode = (twoT_rel_diff < 0.01) and (L2_rel_diff < 0.01) and \
                   (info_truth["regime"] == info_c457["regime"])

    # The set-distance gate: max nearest-neighbour distance < 1% of |ω₀_truth|
    # (a 0.01-relative tolerance covers Jacobi residual + grid discretisation).
    set_match = max(set_dist_c457_to_truth_rel, set_dist_truth_to_c457_rel) < 0.01

    if same_polhode and set_match:
        verdict = (
            "PASS: truth and cluster_457 share the same body-frame polhode "
            "(scalar invariants match, regime matches, set-distance < 1%). "
            "The 'same body-frame polhode' leg of the s073 cat-4 hypothesis "
            "is verified for this single pair on this single seed."
        )
    elif same_polhode and not set_match:
        verdict = (
            "PARTIAL: scalar invariants match within 1% but set-distance "
            "exceeds the 1% gate. Investigate whether the discrepancy is "
            "discretisation/period-mismatch artefact or a genuine bug."
        )
    else:
        verdict = (
            "FAIL: scalar invariants differ beyond 1% or regimes disagree. "
            "Cluster_457 is NOT on truth's polhode — the cat-4 framing "
            "does not apply to this seed-89 multi-sol. Redirect."
        )

    summary = {
        "experiment": "s073d",
        "seed": seed,
        "cluster_id": 457,
        "cluster_rho_hifi_from_s073": rho_c457,

        "inertia_diag_kg_m2": np.diag(I).tolist(),

        "scalar_invariants": {
            "twoT_truth": twoT_truth,
            "twoT_c457": twoT_c457,
            "twoT_rel_diff": twoT_rel_diff,
            "L2_truth": L2_truth,
            "L2_c457": L2_c457,
            "L2_rel_diff": L2_rel_diff,
        },

        "jacobi_info": {
            "truth": {
                "regime": info_truth["regime"],
                "m_k2": float(info_truth["m"]),
                "tau_dot": float(info_truth["tau_dot"]),
                "T_pol_s": T_pol_truth,
            },
            "c457": {
                "regime": info_c457["regime"],
                "m_k2": float(info_c457["m"]),
                "tau_dot": float(info_c457["tau_dot"]),
                "T_pol_s": T_pol_c457,
            },
        },

        "trace_diagnostics": {
            "T_LC_s": T_LC,
            "T_span_truth_s": T_span_truth,
            "T_span_c457_s": T_span_c457,
            "N_grid": N_grid,
            "twoT_drift_along_truth_trace_rel": twoT_drift_truth,
            "L2_drift_along_truth_trace_rel": L2_drift_truth,
            "twoT_drift_along_c457_trace_rel": twoT_drift_c457,
            "L2_drift_along_c457_trace_rel": L2_drift_c457,
        },

        "set_match": {
            "max_nn_dist_c457_to_truth_rel_omega0": set_dist_c457_to_truth_rel,
            "max_nn_dist_truth_to_c457_rel_omega0": set_dist_truth_to_c457_rel,
            "median_nn_dist_c457_to_truth_rel_omega0": set_dist_c457_to_truth_med,
            "median_nn_dist_truth_to_c457_rel_omega0": set_dist_truth_to_c457_med,
            "gate_max_rel": 0.01,
        },

        "gate": {
            "same_polhode_scalar": same_polhode,
            "set_match": set_match,
            "verdict": verdict,
        },
    }

    out_json = RESULTS_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    np.savez(
        RESULTS_DIR / "omega_traces.npz",
        times_truth=times_truth, times_c457=times_c457,
        om_hist_truth=om_hist_truth, om_hist_c457=om_hist_c457,
        om0_truth=om0_truth, om0_c457=om0_c457,
        inertia=I,
        twoT_truth_trace=twoT_truth_trace, L2_truth_trace=L2_truth_trace,
        twoT_c457_trace=twoT_c457_trace, L2_c457_trace=L2_c457_trace,
    )

    # --- Console report ---
    print("=== s073d — body-frame polhode match: truth vs cluster_457 (seed 89) ===")
    print()
    print(f"  inertia diag (kg m²): {np.diag(I)}")
    print()
    print("  Scalar invariants (decisive):")
    print(f"    2T     truth = {twoT_truth:.6e}   c457 = {twoT_c457:.6e}   "
          f"rel diff = {twoT_rel_diff:.3e}")
    print(f"    |L|²   truth = {L2_truth:.6e}   c457 = {L2_c457:.6e}   "
          f"rel diff = {L2_rel_diff:.3e}")
    print()
    print("  Jacobi parameterisation:")
    print(f"    regime truth = {info_truth['regime']}    c457 = {info_c457['regime']}")
    print(f"    k² (m) truth = {float(info_truth['m']):.6e}   "
          f"c457 = {float(info_c457['m']):.6e}")
    print(f"    T_pol  truth = {T_pol_truth:.2f} s   c457 = {T_pol_c457:.2f} s   "
          f"(LC = {T_LC:.2f} s)")
    print()
    print("  Conservation along Jacobi trace (sanity, expect ≤ 1e-12):")
    print(f"    truth: 2T drift = {twoT_drift_truth:.3e}, "
          f"|L|² drift = {L2_drift_truth:.3e}")
    print(f"    c457:  2T drift = {twoT_drift_c457:.3e}, "
          f"|L|² drift = {L2_drift_c457:.3e}")
    print()
    print(f"  Set-match (max nearest-neighbour, normalised by |ω₀_truth| = "
          f"{om_truth_norm:.4e} rad/s):")
    print(f"    c457 -> truth: max = {set_dist_c457_to_truth_rel:.3e}, "
          f"median = {set_dist_c457_to_truth_med:.3e}")
    print(f"    truth -> c457: max = {set_dist_truth_to_c457_rel:.3e}, "
          f"median = {set_dist_truth_to_c457_med:.3e}")
    print()
    print(f"  Verdict: {verdict}")
    print()
    print(f"Saved: {out_json}")
    print(f"Saved: {RESULTS_DIR / 'omega_traces.npz'}")
    print(f"Saved: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
