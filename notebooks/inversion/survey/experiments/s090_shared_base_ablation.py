"""s090 — shared-base ablation: does letting q_a float (ON the brightness isophote)
drop the fast-seed omega-dir floor below the s087 1.25 deg gate?

s089 found the third anchor is a DISAMBIGUATOR, not a noise-reducer: on fast LAM
seed 119 the joint 3-pt solve tracked the 2-pt almost exactly (1.49 deg vs 1.57 deg
@ sigma_q=2.5 deg), still above the 1.25 deg cliff. The interpretation [hypothesis,
not ablated]: every anchor shares the BASE orientation q_a, whose noise is common
to all constraints and cannot average out. Over-determination fixes far-endpoint
noise + winding multiplicity, not the shared-base term.

This experiment tests that hypothesis -- and incorporates the user's correction
(2026-05-22): freeing q_a as an UNCONSTRAINED unknown makes the geometry-only
system SQUARE (6 unknowns vs 6 constraints, K=2 targets) -> it threads the NOISY
targets exactly, over-fitting cloud noise. The fix is to keep q_a on its observed
brightness isophote (the same |dmag| < TOL_MAG bound that defined the cloud), so it
may slide ALONG the isophote (the shared-base component that CAN average out) but
not off it.

Three arms, init-at-truth (isolates truth-branch conditioning, matches s089 Part 1):
  (1) PINNED q_a            -- shoot_multianchor, reproduces s089's 1.49 deg.
  (2) FREE q_a, no bright   -- square system; expect WORSE (over-fits noisy q_b/q_c).
  (3) FREE q_a + brightness -- isophote-constrained; THE test. < 1.25 deg @2.5 deg
                               => shared-base mechanism confirmed + a real lever.

Seed 119 only (the binding fast LAM seed; 116/103 already clear or aren't binding,
per s089). sigma_q in {2.0, 2.5} deg (the binding cell + the s089 clear-point).
Pool + BLAS pinned. Closed-form propagation + v2 surrogate for the brightness term.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.spatial.transform import Rotation

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
import lib.surrogate_eval as se
from lib.shoot import (
    m048_inertia, omega_dir_err_deg, omega_mag_err_frac, polhode_period,
    shoot_multianchor, shoot_multianchor_freebase,
)
from lib.jacobi_propagator import propagate_jacobi_path2, _quat_multiply

OUT = Path(__file__).resolve().parent.parent / "results" / "s090"
OUT.mkdir(parents=True, exist_ok=True)

I_A = 100
S087_GATE_DEG = 1.25
TOL_MAG = 0.10            # cloud-survival tolerance (source: experiments/s059_pilot.py:67)
SP_DEG, AD_DEG = 0.0, 15.0
INERTIA = m048_inertia()
DEG = np.pi / 180.0

SEED = 119
AB_FRAC = 0.30
AC_FRACS = [0.45, 0.60]
SIGMAS = [2.0, 2.5]
N_NOISE = 300


def perturb_q(q, sigma_deg, rng):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    a = np.radians(sigma_deg)
    qp = np.array([np.cos(a / 2), *(np.sin(a / 2) * ax)])
    out = _quat_multiply(qp, q)
    return out / np.linalg.norm(out)


def brightness_at(q_a, sun_hat, obs_hat, obs_dist):
    """v2-surrogate magnitude for orientation q_a at the anchor epoch.

    Builds R(q_a) with the validated forward.py convention (scipy from_quat,
    xyzw order) and projects the (fixed) J2000 sun/observer directions into body.
    """
    R = Rotation.from_quat([q_a[1], q_a[2], q_a[3], q_a[0]]).as_matrix()
    k1 = (R @ sun_hat)[None, :]
    k2 = (R @ obs_hat)[None, :]
    return float(se.predict(k1, k2, np.array([obs_dist]), SP_DEG, AD_DEG)[0])


def _init_worker():
    se.get_model()  # warm the surrogate once per worker


def _trial(trial_seed, q_a, q_b, q_c, dt_ab, dt_ac, w_true, sigma,
           sun_hat, obs_hat, obs_dist, mag_obs):
    rng = np.random.default_rng(trial_seed)
    qa_n = perturb_q(q_a, sigma, rng)
    qb_n = perturb_q(q_b, sigma, rng)
    qc_n = perturb_q(q_c, sigma, rng)
    anchors = [(qb_n, dt_ab), (qc_n, dt_ac)]
    w_geo = 1.0 / np.radians(sigma)               # ML weight: geo residual / sigma_q

    # arm 1: pinned q_a (= s089 joint solve)
    s_pin = shoot_multianchor(qa_n, anchors, INERTIA, w_true)

    # arm 2: free q_a, NO brightness constraint (square -> over-fits noise)
    s_free = shoot_multianchor_freebase(qa_n, anchors, INERTIA, w_true,
                                        base_resid=None, w_geo=w_geo)

    # arm 3: free q_a + brightness-isophote residual (ML weight / TOL_MAG)
    def b_resid(qa):
        return (brightness_at(qa, sun_hat, obs_hat, obs_dist) - mag_obs) / TOL_MAG
    s_bri = shoot_multianchor_freebase(qa_n, anchors, INERTIA, w_true,
                                       base_resid=b_resid, w_geo=w_geo)

    return (
        omega_dir_err_deg(s_pin["omega"], w_true), abs(omega_mag_err_frac(s_pin["omega"], w_true)),
        omega_dir_err_deg(s_free["omega"], w_true), abs(omega_mag_err_frac(s_free["omega"], w_true)),
        s_free["base_shift_deg"], s_free["geo_err_max_deg"],
        omega_dir_err_deg(s_bri["omega"], w_true), abs(omega_mag_err_frac(s_bri["omega"], w_true)),
        s_bri["base_shift_deg"], s_bri["geo_err_max_deg"],
    )


def main():
    t0 = time.time()
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64)
    times0 = times0 - times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[I_A]
    T_pol = polhode_period(w0, INERTIA)

    # fixed anchor-epoch geometry for the brightness residual (J2000 -> body)
    sun_vec = d["sun_pos"][I_A] - d["sat_pos"][I_A]
    obs_vec = d["obs_pos"][I_A] - d["sat_pos"][I_A]
    sun_hat = sun_vec / np.linalg.norm(sun_vec)
    obs_hat = obs_vec / np.linalg.norm(obs_vec)
    obs_dist = float(d["obs_dist"][I_A])
    se.get_model()
    mag_obs = brightness_at(q_hist[I_A], sun_hat, obs_hat, obs_dist)  # noiseless "observed" brightness

    span = min(T_pol, times0[-1] - times0[I_A])
    i_b = int(np.argmin(np.abs(times0 - (times0[I_A] + AB_FRAC * span))))
    dt_ab = float(times0[i_b] - times0[I_A])
    q_a, q_b = q_hist[I_A], q_hist[i_b]

    print(f"===== s090 shared-base ablation | seed {SEED} =====")
    print(f"|w|={np.degrees(np.linalg.norm(w0)):.4f} dps  T_pol={T_pol:.0f}s  "
          f"len={len(times0)}ep  span={span:.0f}s  mag_obs(A)={mag_obs:.3f}")

    results = []
    with Pool(12, initializer=_init_worker) as pool:
        for ac_frac in AC_FRACS:
            i_c = int(np.argmin(np.abs(times0 - (times0[I_A] + ac_frac * span))))
            q_c = q_hist[i_c]
            dt_ac = float(times0[i_c] - times0[I_A])
            omdt_ab = np.degrees(np.linalg.norm(w_true) * dt_ab)
            omdt_ac = np.degrees(np.linalg.norm(w_true) * dt_ac)

            # noiseless self-checks (init +5%): every arm must snap to truth.
            chk_pin = shoot_multianchor(q_a, [(q_b, dt_ab), (q_c, dt_ac)], INERTIA, w_true * 1.05)
            chk_bri = shoot_multianchor_freebase(
                q_a, [(q_b, dt_ab), (q_c, dt_ac)], INERTIA, w_true * 1.05,
                base_resid=lambda qa: (brightness_at(qa, sun_hat, obs_hat, obs_dist) - mag_obs) / TOL_MAG,
                w_geo=1.0)
            print(f"\n--- ab={i_b - I_A}ep(|w|dt={omdt_ab:.0f}deg)  "
                  f"ac={i_c - I_A}ep(|w|dt={omdt_ac:.0f}deg) ---")
            print(f"  [self-check +5%] pinned dir={omega_dir_err_deg(chk_pin['omega'], w_true):.1e}deg  "
                  f"free+bri dir={omega_dir_err_deg(chk_bri['omega'], w_true):.1e}deg "
                  f"base_shift={chk_bri['base_shift_deg']:.1e}deg")
            print(f"  {'sig':>4} | {'PIN dir':>8} {'FREE dir':>8} {'BRI dir':>8} | "
                  f"{'FREEshift':>9} {'BRIshift':>9} | gate(bri)")

            for sg in SIGMAS:
                args = [(7_000_000 + 10_000 * int(sg * 10) + k, q_a, q_b, q_c, dt_ab, dt_ac,
                         w_true, sg, sun_hat, obs_hat, obs_dist, mag_obs) for k in range(N_NOISE)]
                r = np.array(pool.starmap(_trial, args))   # (N, 10)
                (pin_d, pin_m, fr_d, fr_m, fr_sh, fr_geo,
                 br_d, br_m, br_sh, br_geo) = r.T
                row = dict(
                    sigma_q_deg=sg,
                    pin_dir_med=float(np.median(pin_d)), pin_dir_p90=float(np.percentile(pin_d, 90)),
                    free_dir_med=float(np.median(fr_d)), free_dir_p90=float(np.percentile(fr_d, 90)),
                    bri_dir_med=float(np.median(br_d)), bri_dir_p90=float(np.percentile(br_d, 90)),
                    pin_mag_med_pct=float(100 * np.median(pin_m)),
                    bri_mag_med_pct=float(100 * np.median(br_m)),
                    free_base_shift_med=float(np.median(fr_sh)),
                    bri_base_shift_med=float(np.median(br_sh)),
                    bri_geo_max_med=float(np.median(br_geo)),
                )
                gate = "PASS" if row["bri_dir_med"] < S087_GATE_DEG else "----"
                print(f"  {sg:>4} | {row['pin_dir_med']:8.2f} {row['free_dir_med']:8.2f} "
                      f"{row['bri_dir_med']:8.2f} | {row['free_base_shift_med']:9.2f} "
                      f"{row['bri_base_shift_med']:9.2f} | {gate}")
                results.append(dict(seed=SEED, ab_epochs=i_b - I_A, ac_epochs=i_c - I_A,
                                    dt_ab_s=dt_ab, dt_ac_s=dt_ac,
                                    omega_dt_ab_deg=float(omdt_ab), omega_dt_ac_deg=float(omdt_ac),
                                    n_noise=N_NOISE, **row))

    meta = dict(seed=SEED, tol_mag=TOL_MAG, gate_deg=S087_GATE_DEG, mag_obs=mag_obs,
                T_pol_s=float(T_pol), wall_s=time.time() - t0, rows=results)
    with open(OUT / "ablation.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'ablation.json'}")
    print(f"Compute wall: {meta['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
