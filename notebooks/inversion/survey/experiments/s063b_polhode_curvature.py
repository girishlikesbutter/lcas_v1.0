"""s063b — Polhode tangent vs polhode-normal cost curvature at truth.

For each of 3 seeds (89 slow, 28 fast, 14 near-separatrix candidate), at truth
(q0, ω0), take 1D ω-slices along three orthonormal directions:
  - tangent   : Euler-equation direction dω/dt = I^-1 ((I·ω) × ω)
  - normal_E  : grad(2T) projected away from tangent, normalized (changes energy)
  - normal_L  : grad(L²) projected away from (tangent, normal_E), normalized

For each direction d (body frame): vary ε ∈ {-0.05, ..., +0.05} × |ω| (21 points).
For each perturbed ω, propagate q via DOP853 + render via surrogate, compute
full-LC MSE vs cached mag_hifi.

Fit quadratic c(ε) ≈ c0 + α·ε + β·ε² and report β (curvature). The ratio
β_normal / β_tangent quantifies how much sharper off-polhode is than along-polhode.

Hypothesis: polhode tangent is the SOFT direction (small β), off-polhode normals
are SHARP (large β). If confirmed, LM polish should reparameterize ω onto the
polhode basis to drop the noisy dimensions.

Reading these results:
  - β >> β_tangent on both normals: off-polhode reparameterization is worth it.
  - All three β's similar: cost surface is isotropic in ω, polhode prior gives
    no conditioning benefit (still useful for sampling, not for polish).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))

from lib import surrogate_eval  # noqa: E402
from lib.forward import propagate_to_body_frame  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402

INERTIA = np.diag([37985.15566495171, 38305.70560133419, 7749.014672251076])
SEEDS = [89, 28, 14]
N_EPS = 21
EPS_FRAC = 0.05  # max fractional perturbation: ±5% of |ω|

OUT_DIR = SURVEY / "results" / "s063b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gram_schmidt_basis(omega: np.ndarray, I: np.ndarray):
    """Return orthonormal (tangent, normal_E, normal_L) in BODY frame.

    tangent  : Euler dω/dt = I^-1 ((I·ω) × ω)   -- along the polhode
    normal_E : grad(2T) = 2 I·ω, Gram-Schmidt against tangent
    normal_L : grad(L²) = 2 I²·ω, Gram-Schmidt against (tangent, normal_E)
    """
    L = I @ omega
    raw_t = np.linalg.solve(I, np.cross(L, omega))
    t_norm = np.linalg.norm(raw_t)
    if t_norm < 1e-15:
        # Pure rotation about a principal axis: tangent ill-defined.
        # Fall back to any vector perpendicular to ω.
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(omega, e1) / np.linalg.norm(omega)) > 0.95:
            e1 = np.array([0.0, 1.0, 0.0])
        raw_t = np.cross(omega, e1)
        t_norm = np.linalg.norm(raw_t)
    t_hat = raw_t / t_norm

    g_e = 2.0 * L
    nE_raw = g_e - np.dot(g_e, t_hat) * t_hat
    nE_hat = nE_raw / np.linalg.norm(nE_raw)

    g_l = 2.0 * I @ L  # gradient of L² wrt ω is 2 I² ω
    nL_raw = g_l - np.dot(g_l, t_hat) * t_hat - np.dot(g_l, nE_hat) * nE_hat
    nL_hat = nL_raw / np.linalg.norm(nL_raw)

    return t_hat, nE_hat, nL_hat


def cost_at_omega(q0, omega, d, mag_hifi, obs_dist):
    k1, k2, _ = propagate_to_body_frame(
        q0, omega, d["observation_times"], d["sun_pos"], d["obs_pos"], d["sat_pos"],
        INERTIA, mode="tumbling",
    )
    pred = surrogate_eval.predict(k1, k2, obs_dist)
    return surrogate_eval.full_lc_mse(pred, mag_hifi)


def quadratic_fit(eps, costs):
    """Fit c = c0 + α·ε + β·ε² and return (c0, alpha, beta)."""
    A = np.column_stack([np.ones_like(eps), eps, eps ** 2])
    coef, *_ = np.linalg.lstsq(A, costs, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def main():
    # Eager-load surrogate.
    print("Loading surrogate...", flush=True)
    surrogate_eval.get_model()

    results = {}
    eps_frac = np.linspace(-EPS_FRAC, EPS_FRAC, N_EPS)

    for seed in SEEDS:
        d = load_truth(seed)
        q0 = d["q0_wxyz"].astype(np.float64)
        omega0 = d["omega0_rad"].astype(np.float64)
        mag_hifi = d["mag_hifi"]
        obs_dist = d["obs_dist"]
        omega_mag = float(np.linalg.norm(omega0))
        omega_mag_dps = omega_mag * 180.0 / np.pi

        t_hat, nE_hat, nL_hat = gram_schmidt_basis(omega0, INERTIA)

        # Sanity: c0 at truth (should be a small surrogate-floor value).
        t0 = time.time()
        c0_truth = cost_at_omega(q0, omega0, d, mag_hifi, obs_dist)
        truth_wall = time.time() - t0

        per_dir = {}
        for name, d_hat in [("tangent", t_hat), ("normal_E", nE_hat), ("normal_L", nL_hat)]:
            eps_abs = eps_frac * omega_mag    # absolute ω perturbation magnitude
            costs = np.empty(N_EPS, dtype=float)
            tic = time.time()
            for i, e in enumerate(eps_abs):
                omega_p = omega0 + e * d_hat
                costs[i] = cost_at_omega(q0, omega_p, d, mag_hifi, obs_dist)
            wall = time.time() - tic

            c0, alpha, beta = quadratic_fit(eps_abs, costs)
            # Also compute the curvature in fractional-eps units for cross-seed comparison.
            c0_f, alpha_f, beta_f = quadratic_fit(eps_frac, costs)

            rho_at_ends = np.sqrt(costs.max()) / 0.05
            per_dir[name] = dict(
                eps_frac=eps_frac.tolist(),
                eps_abs=eps_abs.tolist(),
                costs=costs.tolist(),
                rho_at_ends=float(rho_at_ends),
                fit_abs=dict(c0=c0, alpha=alpha, beta=beta),
                fit_frac=dict(c0=c0_f, alpha=alpha_f, beta=beta_f),
                wall_seconds=wall,
            )

        # Ratios on fractional-eps units (dimensionless comparison across seeds).
        bt = per_dir["tangent"]["fit_frac"]["beta"]
        bE = per_dir["normal_E"]["fit_frac"]["beta"]
        bL = per_dir["normal_L"]["fit_frac"]["beta"]
        ratio_E = bE / bt if abs(bt) > 1e-12 else float("inf")
        ratio_L = bL / bt if abs(bt) > 1e-12 else float("inf")

        results[seed] = dict(
            omega_mag_dps=omega_mag_dps,
            c0_truth_surr_mse=float(c0_truth),
            rho_truth=float(np.sqrt(c0_truth) / 0.05),
            t_hat_body=t_hat.tolist(),
            nE_hat_body=nE_hat.tolist(),
            nL_hat_body=nL_hat.tolist(),
            beta_tangent_frac=bt,
            beta_normal_E_frac=bE,
            beta_normal_L_frac=bL,
            ratio_E_over_tangent=ratio_E,
            ratio_L_over_tangent=ratio_L,
            per_direction=per_dir,
            truth_eval_wall=truth_wall,
        )
        print(f"seed {seed:03d} |ω|={omega_mag_dps:.3f} dps  "
              f"ρ_truth={results[seed]['rho_truth']:.3f}  "
              f"β: tan={bt:.2e}  nE={bE:.2e}  nL={bL:.2e}  "
              f"ratio_E={ratio_E:.2f}  ratio_L={ratio_L:.2f}",
              flush=True)

    out = OUT_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


if __name__ == "__main__":
    main()
