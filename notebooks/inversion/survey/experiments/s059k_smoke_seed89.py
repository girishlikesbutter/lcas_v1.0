"""s059k_smoke_seed89 — quick single-threaded polish-bridge test on seed 89.

Tests whether Phase 2's local-window polish is likely to suffer the same
PHANTOM BASIN issue as seed 28 (Phase 1 a-local found ρ_local=1.98 but
hi-fi ρ=70). Synthesize a seed at the noise level Phase 2 will see at
N_DIRS=800 (q_a ~ 1.7°, ω_dir ~ 3.5°, |ω| ±6%), run both local-window
and full-LC polish, hi-fi each.

Single-threaded (~2-3 min wall) so it doesn't compete with Phase 2's
Pool(24).

If local-window phantom-basins on seed 89 too → expect Phase 2 to fail
and rescue.py (full-LC re-polish) is the next move.
If local-window converges to truth on seed 89 → Phase 2 likely succeeds.

Usage:
    python experiments/s059k_smoke_seed89.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import back_propagate  # noqa: E402
from experiments.s059e_local_window import lm_polish_local  # noqa: E402
from experiments.s058_lm_polish_clusters import lm_polish, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

SEED = 89
T_A = 3       # matches Phase 2 anchor
W = 10
RNG_SEED = 17  # deterministic perturbation
OUT = SURVEY / "results" / "s059k_smoke_seed89"
OUT.mkdir(parents=True, exist_ok=True)


def perturb_q(q_wxyz, deg, rng):
    """Apply a random rotation of given magnitude (deg) to q."""
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    rotvec = np.deg2rad(deg) * axis
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    q_perturbed_xyzw = (Rotation.from_rotvec(rotvec) * Rotation.from_quat(q_xyzw)).as_quat()
    return quat_xyzw_to_wxyz(q_perturbed_xyzw)


def perturb_omega(om_rad, dir_deg, mag_pct, rng):
    """Perturb ω: rotate direction by dir_deg, scale magnitude by (1+mag_pct/100)."""
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    om_mag = float(np.linalg.norm(om_rad))
    om_dir = om_rad / om_mag
    rotvec = np.deg2rad(dir_deg) * axis
    new_dir = Rotation.from_rotvec(rotvec).apply(om_dir)
    new_mag = om_mag * (1 + mag_pct / 100.0)
    return new_dir * new_mag


def main():
    print(f"=== s059k_smoke_seed89 — polish bridge test on seed {SEED} at T_A={T_A} ===\n")

    print(f"building hifi context for seed {SEED}...")
    t0 = time.time()
    ctx = build_context(seed=SEED)
    target = ctx["mag_hifi_truth"]
    print(f"  built in {time.time()-t0:.1f}s; target len={len(target)}")

    # Get truth (q_a, om_a) at T_A
    print(f"\npropagating truth to T_A={T_A}...")
    quats_truth, omegas_truth = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=ctx["observation_times"], mode="tumbling",
        inertia_tensor=ctx["inertia_tensor"],
    )
    q_a_truth = np.asarray(quats_truth[T_A], dtype=np.float64)
    om_a_truth = np.asarray(omegas_truth[T_A], dtype=np.float64)
    om_a_truth_mag_dps = float(np.degrees(np.linalg.norm(om_a_truth)))
    print(f"  truth q_a = {q_a_truth}")
    print(f"  truth ω_a = {om_a_truth} (|ω|={om_a_truth_mag_dps:.4f} dps)")

    rng = np.random.default_rng(RNG_SEED)

    # Test 1: light perturbation matching Phase 2 N=800 expected (q_a 1.7°, ω_dir 3.5°, |ω|+6%)
    print(f"\n=== TEST 1: q_a 1.7° + ω_dir 3.5° + |ω|+6% (Phase 2 N_DIRS=800 expected noise) ===")
    qa_pert = perturb_q(q_a_truth, 1.7, rng)
    om_pert = perturb_omega(om_a_truth, 3.5, 6.0, rng)
    om_pert_mag_dps = float(np.degrees(np.linalg.norm(om_pert)))
    print(f"  perturbed q_a = {qa_pert}")
    print(f"  perturbed ω_a = {om_pert} (|ω|={om_pert_mag_dps:.4f} dps)")

    results_test1 = {}

    # Local-window polish
    print(f"\n  (1a) lm_polish_local (architecture match)")
    t0 = time.time()
    res1a = lm_polish_local(qa_pert, om_pert, T_A, W, ctx, target)
    try:
        pred = render_hifi(res1a["q0_pol_wxyz"], res1a["om0_pol_rad"], ctx)
        rho_h = float(rho_from_hifi(pred, target))
        band = rho_band(rho_h)
    except Exception as e:
        rho_h, band = float("nan"), "ERR"
    wall1a = time.time() - t0
    q0_err = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(res1a["q0_pol_wxyz"], ctx["q0_truth"]))))))
    om_truth_t0 = ctx["omega0_truth_rad"]
    om_truth_t0_mag = float(np.linalg.norm(om_truth_t0))
    om_pol_t0 = res1a["om0_pol_rad"]
    om_mag_err = float((np.linalg.norm(om_pol_t0) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    om_dir_err = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om_pol_t0 / max(1e-12, np.linalg.norm(om_pol_t0)),
                   om_truth_t0 / om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_local_seed={res1a['surrogate_rho_local_seed']:.3f} → ρ_local_pol={res1a['surrogate_rho_local_polished']:.3f}")
    print(f"    hi-fi ρ={rho_h:.3f} band={band}")
    print(f"    q0_err={q0_err:.2f}° |ω|err={om_mag_err:+.2f}% ω_dir={om_dir_err:.2f}°  wall={wall1a:.1f}s")
    results_test1["1a_local_window"] = {
        "rho_local_polished": float(res1a["surrogate_rho_local_polished"]),
        "rho_polished_hifi": rho_h, "band": band,
        "q0_err_deg": q0_err, "om_mag_err_pct": om_mag_err, "om_dir_err_deg": om_dir_err,
        "wall_s": wall1a,
    }

    # Full-LC polish from same perturbation
    print(f"\n  (1b) lm_polish (full-LC residual)")
    t0 = time.time()
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    q0_seed_b, om0_seed_b = back_propagate(qa_pert, om_pert, t_a_seconds, ctx["inertia_tensor"])
    res1b = lm_polish(q0_seed_b, om0_seed_b, ctx, target, label="test1_full_lc")
    try:
        pred = render_hifi(np.array(res1b["q0_pol_wxyz"]), np.array(res1b["om0_pol_rad"]), ctx)
        rho_h_b = float(rho_from_hifi(pred, target))
        band_b = rho_band(rho_h_b)
    except Exception as e:
        rho_h_b, band_b = float("nan"), "ERR"
    wall1b = time.time() - t0
    q0_err_b = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(np.array(res1b["q0_pol_wxyz"]), ctx["q0_truth"]))))))
    om_pol_b = np.array(res1b["om0_pol_rad"])
    om_mag_err_b = float((np.linalg.norm(om_pol_b) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    om_dir_err_b = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om_pol_b / max(1e-12, np.linalg.norm(om_pol_b)),
                   om_truth_t0 / om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_seed={res1b['surrogate_rho_seed']:.3f} → ρ_polished={res1b['surrogate_rho_polished']:.3f}")
    print(f"    hi-fi ρ={rho_h_b:.3f} band={band_b}")
    print(f"    q0_err={q0_err_b:.2f}° |ω|err={om_mag_err_b:+.2f}% ω_dir={om_dir_err_b:.2f}°  wall={wall1b:.1f}s")
    results_test1["1b_full_lc"] = {
        "rho_polished": float(res1b["surrogate_rho_polished"]),
        "rho_polished_hifi": rho_h_b, "band": band_b,
        "q0_err_deg": q0_err_b, "om_mag_err_pct": om_mag_err_b, "om_dir_err_deg": om_dir_err_b,
        "wall_s": wall1b,
    }

    # Test 2: tighter noise (N=1600 expected) — q_a 0.85°, ω_dir 1.75°, |ω|+0%
    print(f"\n=== TEST 2: q_a 0.85° + ω_dir 1.75° + |ω|+0% (Phase 4 N=1600 / no mag noise) ===")
    rng2 = np.random.default_rng(RNG_SEED + 1)
    qa_pert2 = perturb_q(q_a_truth, 0.85, rng2)
    om_pert2 = perturb_omega(om_a_truth, 1.75, 0.0, rng2)
    print(f"  perturbed q_a = {qa_pert2}")
    print(f"  perturbed ω_a = {om_pert2}")
    results_test2 = {}
    print(f"\n  (2a) lm_polish_local")
    t0 = time.time()
    res2a = lm_polish_local(qa_pert2, om_pert2, T_A, W, ctx, target)
    try:
        pred = render_hifi(res2a["q0_pol_wxyz"], res2a["om0_pol_rad"], ctx)
        rho_h2a = float(rho_from_hifi(pred, target)); band2a = rho_band(rho_h2a)
    except Exception:
        rho_h2a, band2a = float("nan"), "ERR"
    wall2a = time.time() - t0
    q0e2a = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(res2a["q0_pol_wxyz"], ctx["q0_truth"]))))))
    om2a = res2a["om0_pol_rad"]
    omme2a = float((np.linalg.norm(om2a) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    omde2a = float(np.degrees(np.arccos(np.clip(abs(np.dot(om2a/max(1e-12,np.linalg.norm(om2a)), om_truth_t0/om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_local_pol={res2a['surrogate_rho_local_polished']:.3f} → hi-fi ρ={rho_h2a:.3f} band={band2a}")
    print(f"    q0_err={q0e2a:.2f}° |ω|err={omme2a:+.2f}% ω_dir={omde2a:.2f}°  wall={wall2a:.1f}s")
    results_test2["2a_local_window"] = {"rho_local_polished": float(res2a["surrogate_rho_local_polished"]),
                                         "rho_polished_hifi": rho_h2a, "band": band2a,
                                         "q0_err_deg": q0e2a, "om_mag_err_pct": omme2a, "om_dir_err_deg": omde2a, "wall_s": wall2a}

    print(f"\n  (2b) lm_polish (full-LC)")
    t0 = time.time()
    q0_seed_2b, om0_seed_2b = back_propagate(qa_pert2, om_pert2, t_a_seconds, ctx["inertia_tensor"])
    res2b = lm_polish(q0_seed_2b, om0_seed_2b, ctx, target, label="test2_full_lc")
    try:
        pred = render_hifi(np.array(res2b["q0_pol_wxyz"]), np.array(res2b["om0_pol_rad"]), ctx)
        rho_h2b = float(rho_from_hifi(pred, target)); band2b = rho_band(rho_h2b)
    except Exception:
        rho_h2b, band2b = float("nan"), "ERR"
    wall2b = time.time() - t0
    q0e2b = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(np.array(res2b["q0_pol_wxyz"]), ctx["q0_truth"]))))))
    om2b = np.array(res2b["om0_pol_rad"])
    omme2b = float((np.linalg.norm(om2b) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    omde2b = float(np.degrees(np.arccos(np.clip(abs(np.dot(om2b/max(1e-12,np.linalg.norm(om2b)), om_truth_t0/om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_pol={res2b['surrogate_rho_polished']:.3f} → hi-fi ρ={rho_h2b:.3f} band={band2b}")
    print(f"    q0_err={q0e2b:.2f}° |ω|err={omme2b:+.2f}% ω_dir={omde2b:.2f}°  wall={wall2b:.1f}s")
    results_test2["2b_full_lc"] = {"rho_polished": float(res2b["surrogate_rho_polished"]),
                                    "rho_polished_hifi": rho_h2b, "band": band2b,
                                    "q0_err_deg": q0e2b, "om_mag_err_pct": omme2b, "om_dir_err_deg": omde2b, "wall_s": wall2b}

    # Test 3: N=800 q+dir noise but ZERO |ω| mag noise — isolates mag contribution
    print(f"\n=== TEST 3: q_a 1.7° + ω_dir 3.5° + |ω|+0% (N=800 noise, ZERO mag) ===")
    rng3 = np.random.default_rng(RNG_SEED + 2)
    qa_pert3 = perturb_q(q_a_truth, 1.7, rng3)
    om_pert3 = perturb_omega(om_a_truth, 3.5, 0.0, rng3)
    print(f"  perturbed q_a = {qa_pert3}")
    print(f"  perturbed ω_a = {om_pert3}")
    results_test3 = {}

    print(f"\n  (3b) lm_polish (full-LC)")
    t0 = time.time()
    q0_seed_3b, om0_seed_3b = back_propagate(qa_pert3, om_pert3, t_a_seconds, ctx["inertia_tensor"])
    res3b = lm_polish(q0_seed_3b, om0_seed_3b, ctx, target, label="test3_full_lc")
    try:
        pred = render_hifi(np.array(res3b["q0_pol_wxyz"]), np.array(res3b["om0_pol_rad"]), ctx)
        rho_h3b = float(rho_from_hifi(pred, target)); band3b = rho_band(rho_h3b)
    except Exception:
        rho_h3b, band3b = float("nan"), "ERR"
    wall3b = time.time() - t0
    q0e3b = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(np.array(res3b["q0_pol_wxyz"]), ctx["q0_truth"]))))))
    om3b = np.array(res3b["om0_pol_rad"])
    omme3b = float((np.linalg.norm(om3b) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    omde3b = float(np.degrees(np.arccos(np.clip(abs(np.dot(om3b/max(1e-12,np.linalg.norm(om3b)), om_truth_t0/om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_pol={res3b['surrogate_rho_polished']:.3f} → hi-fi ρ={rho_h3b:.3f} band={band3b}")
    print(f"    q0_err={q0e3b:.2f}° |ω|err={omme3b:+.2f}% ω_dir={omde3b:.2f}°  wall={wall3b:.1f}s")
    results_test3["3b_full_lc"] = {"rho_polished": float(res3b["surrogate_rho_polished"]),
                                    "rho_polished_hifi": rho_h3b, "band": band3b,
                                    "q0_err_deg": q0e3b, "om_mag_err_pct": omme3b, "om_dir_err_deg": omde3b, "wall_s": wall3b}

    # Headline
    print(f"\n{'='*60}")
    print(f"SMOKE HEADLINE — seed 89 polish bridges:")
    print(f"  TEST 1 (q_a=1.7°, ω=3.5°, |ω|+6%) — Phase 2 N=800 grade noise:")
    print(f"    (1a) local-window: hi-fi ρ={rho_h:.3f} band={band}")
    print(f"    (1b) full-LC:      hi-fi ρ={rho_h_b:.3f} band={band_b}")
    print(f"  TEST 2 (q_a=0.85°, ω=1.75°, |ω|+0%) — N=1600 + zero mag noise:")
    print(f"    (2a) local-window: hi-fi ρ={rho_h2a:.3f} band={band2a}")
    print(f"    (2b) full-LC:      hi-fi ρ={rho_h2b:.3f} band={band2b}")
    print(f"  TEST 3 (q_a=1.7°, ω=3.5°, |ω|+0%) — N=800 noise, ZERO mag:")
    print(f"    (3b) full-LC:      hi-fi ρ={rho_h3b:.3f} band={band3b}")
    print(f"{'='*60}")
    print()
    if band in {"A", "B"}:
        print("  → local-window polish converges on seed 89 — Phase 2 LIKELY succeeds")
    elif band in {"C"}:
        print("  → local-window polish converges to a near-Band-A∪B state — Phase 2 borderline")
    else:
        print("  → local-window polish lands phantom on seed 89 too — Phase 2 likely fails")
        if band_b in {"A", "B"}:
            print("    BUT full-LC polish converges — rescue.py should fix it after Phase 2 done")
        else:
            print("    AND full-LC polish ALSO fails — architectural pivot needed")

    summary = {
        "experiment": "s059k_smoke_seed89",
        "seed": SEED, "T_A": T_A, "W": W, "rng_seed": RNG_SEED,
        "truth_at_T_A": {
            "q_a_wxyz": q_a_truth.tolist(),
            "om_a_rad": om_a_truth.tolist(),
            "om_a_mag_dps": om_a_truth_mag_dps,
        },
        "test1": {
            "perturbation": {"q_deg": 1.7, "om_dir_deg": 3.5, "om_mag_pct": 6.0,
                              "qa_pert_wxyz": qa_pert.tolist(), "om_pert_rad": om_pert.tolist()},
            "results": results_test1,
        },
        "test2": {
            "perturbation": {"q_deg": 0.85, "om_dir_deg": 1.75, "om_mag_pct": 0.0,
                              "qa_pert_wxyz": qa_pert2.tolist(), "om_pert_rad": om_pert2.tolist()},
            "results": results_test2,
        },
        "test3": {
            "perturbation": {"q_deg": 1.7, "om_dir_deg": 3.5, "om_mag_pct": 0.0,
                              "qa_pert_wxyz": qa_pert3.tolist(), "om_pert_rad": om_pert3.tolist()},
            "results": results_test3,
        },
    }
    out = OUT / "smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
