"""s059k_smoke_multistart — verify multi-mag-start rescues the failed TEST 1.

Take the EXACT TEST 1 perturbation that landed Band D under single-start
(q_a 1.7°, ω_dir 3.5°, |ω|+6%) and run full-LC polish from offsets
[0, ±3, ±6]% relative to the perturbed seed mag. Expectation: the offset
−6% gives effective mag ≈ 0% (truth) and should land Band A.

If this confirms, multi-mag-start in s059k_full_lc_from_seeds.py is
robust on Phase 2 grid.

Single-threaded ~8 min wall.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import back_propagate  # noqa: E402
from experiments.s058_lm_polish_clusters import lm_polish  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from experiments.s059k_smoke_seed89 import perturb_q, perturb_omega  # reuse  # noqa: E402

SEED = 89
T_A = 3
RNG_SEED = 17
OUT = SURVEY / "results" / "s059k_smoke_multistart"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=== s059k_smoke_multistart — multi-mag-start on TEST 1 perturbation ===\n")

    ctx = build_context(seed=SEED)
    target = ctx["mag_hifi_truth"]
    obs_times = ctx["observation_times"]
    inertia_tensor = ctx["inertia_tensor"]
    t_a_seconds = float(obs_times[T_A] - obs_times[0])

    quats_truth, omegas_truth = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=obs_times, mode="tumbling", inertia_tensor=inertia_tensor,
    )
    q_a_truth = np.asarray(quats_truth[T_A])
    om_a_truth = np.asarray(omegas_truth[T_A])

    # Reproduce TEST 1 EXACT perturbation (RNG_SEED=17, then advance the RNG)
    rng = np.random.default_rng(RNG_SEED)
    qa_pert = perturb_q(q_a_truth, 1.7, rng)
    om_pert = perturb_omega(om_a_truth, 3.5, 6.0, rng)
    print(f"perturbed q_a (1.7° from truth): {qa_pert}")
    print(f"perturbed ω_a (dir 3.5°, mag +6%): {om_pert}")
    print(f"  perturbed |ω|: {np.degrees(np.linalg.norm(om_pert)):.4f} dps")
    print(f"  truth |ω|:     {np.degrees(np.linalg.norm(om_a_truth)):.4f} dps")

    om_truth_t0 = ctx["omega0_truth_rad"]
    om_truth_t0_mag = float(np.linalg.norm(om_truth_t0))

    # Multi-mag-start from this perturbation
    offsets = [0.0, +3.0, -3.0, +6.0, -6.0]
    print(f"\nrunning {len(offsets)} polishes with mag offsets {offsets}%")
    results = []
    for offset in offsets:
        scale = 1.0 + offset / 100.0
        om_seed = om_pert * scale
        om_eff_mag_pct = (np.linalg.norm(om_seed) - om_truth_t0_mag) / om_truth_t0_mag * 100
        print(f"\n  offset {offset:+5.1f}%  → effective seed |ω| pct vs truth = {om_eff_mag_pct:+5.2f}%")

        t0 = time.time()
        q0_seed, om0_seed = back_propagate(qa_pert, om_seed, t_a_seconds, inertia_tensor)
        res = lm_polish(q0_seed, om0_seed, ctx, target, label=f"offset{offset:+.0f}")
        try:
            pred = render_hifi(np.asarray(res["q0_pol_wxyz"]),
                               np.asarray(res["om0_pol_rad"]), ctx)
            rho_h = float(rho_from_hifi(pred, target))
            band = rho_band(rho_h)
        except Exception:
            rho_h, band = float("nan"), "ERR"
        wall = time.time() - t0

        q0p = np.asarray(res["q0_pol_wxyz"])
        om0p = np.asarray(res["om0_pol_rad"])
        q0_err = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(q0p, ctx["q0_truth"]))))))
        om_mag_err = float((np.linalg.norm(om0p) - om_truth_t0_mag) / om_truth_t0_mag * 100)
        om_dir_err = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                       om_truth_t0 / om_truth_t0_mag)), 0, 1))))
        print(f"    ρ_seed={res['surrogate_rho_seed']:6.2f} → ρ_pol={res['surrogate_rho_polished']:6.3f}  "
              f"hi-fi ρ={rho_h:6.3f} band={band}  q0_err={q0_err:.2f}° |ω|err={om_mag_err:+.2f}% "
              f"ω_dir={om_dir_err:.2f}°  wall={wall:.1f}s")
        results.append({
            "offset_pct": float(offset),
            "om_eff_mag_pct": float(om_eff_mag_pct),
            "rho_polished_full_lc": float(res["surrogate_rho_polished"]),
            "rho_polished_hifi": float(rho_h),
            "band": band,
            "q0_err_deg": q0_err,
            "om_mag_err_pct": om_mag_err,
            "om_dir_err_deg": om_dir_err,
            "wall_s": float(wall),
        })

    # Headline: best yield among the 5
    best = min(results, key=lambda r: r["rho_polished_hifi"] if not np.isnan(r["rho_polished_hifi"]) else np.inf)
    print(f"\n{'='*60}")
    print(f"MULTI-MAG-START RESULT — best of 5 polishes:")
    print(f"  best at offset {best['offset_pct']:+.1f}% (effective |ω| pct = {best['om_eff_mag_pct']:+.2f}%)")
    print(f"  → hi-fi ρ={best['rho_polished_hifi']:.3f} band={best['band']} q0_err={best['q0_err_deg']:.2f}°")
    print(f"\nVERDICT: " + ("multi-mag-start RESCUES TEST 1 → fix architecture is robust"
                            if best["band"] in {"A", "B"}
                            else "multi-mag-start did NOT rescue TEST 1 → need finer scan or different fix"))
    print(f"{'='*60}")

    summary = {
        "experiment": "s059k_smoke_multistart",
        "seed": SEED, "T_A": T_A, "rng_seed": RNG_SEED,
        "perturbation": {"q_deg": 1.7, "om_dir_deg": 3.5, "om_mag_pct": 6.0,
                          "qa_pert_wxyz": qa_pert.tolist(), "om_pert_rad": om_pert.tolist()},
        "offsets_pct": offsets,
        "results": results,
        "best_band": best["band"],
        "best_rho_hifi": best["rho_polished_hifi"],
    }
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
