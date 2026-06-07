"""s089 — third-anchor over-determination: break the s088 conditioning<->multiplicity crossover.

s088 Phase B left the fast tumbler (seed 119) at omega-dir 1.45 deg @ sigma_q=2.5 deg
(Delta t=60, the 2-point sweet spot) -- just above the s087 1.25 deg cliff, clearing
only at sigma_q <~ 2.0 deg. The LONG baseline (Delta t=120) would be better
conditioned but Phase B showed it BLOWS UP to 28.8 deg even initialised at truth,
because winding aliases crowd into the noise ball (the conditioning<->multiplicity
crossover). Gate A2 quantified the ladder: 11 in-prior distinct roots at Delta t=120.

A THIRD intermediate anchor turns the 2-point BVP into an over-determined one
(3 unknowns omega_a vs 3K geodesic constraints). Hypotheses:

  Q1 (noise floor): joint 3-point LS averages independent endpoint noise AND uses
      the long baseline for conditioning while the intermediate anchor pins the
      winding -> omega-dir below 1.25 deg at sigma_q=2.5 deg on seed 119, where
      2-point only cleared <~2.0 deg.
  Q2 (ladder collapse): the intermediate-anchor constraint kills the spurious
      windings -> the 11-root ladder at the long baseline collapses toward a unique
      in-prior root.

Design decisions (agreed with user, 2026-05-22):
  - INIT-AT-TRUTH in Part 1: isolates truth-branch conditioning from blind search
    (Part 2 is the blind multi-start). NOT a blind end-to-end test.
  - Seeds span both tumbling regimes: 119 (LAM-fast, binding), 116 (LAM-slow,
    s088 continuity), 103 (SAM control). Regime governs winding spacing, hence
    whether aliasing -- the thing the 3rd anchor fixes -- even appears.
  - Anchor placement = FRACTIONS of the polhode period T_pol (the spin-axis wobble
    cycle). ab=0.30*T_pol, ac in {0.45,0.60}*T_pol reproduces s088's measured 119
    geometry (Delta t=60 ~ 0.30*T_pol, Delta t=120 ~ 0.60*T_pol) and transfers
    fairly across regimes -- same point in the wobble cycle for every seed. If
    T_pol exceeds the remaining LC (near-separatrix SAM), span the LC instead.
  - Metric = omega-DIRECTION error vs the s087 1.25 deg gate. That gate was measured
    on LAM seed 119 ONLY, so on the SAM seed it is reported as INDICATIVE (no hard
    PASS/FAIL) -- the real SAM gate would need its own s087-style threshold or the
    downstream rho-band.

Pool(24), BLAS pinned. Pure closed-form propagation + cached truth (no surrogate).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.shoot import (
    m048_inertia, geodesic_angle, omega_dir_err_deg, omega_mag_err_frac,
    polhode_period, shoot, shoot_multianchor,
)
from lib.jacobi_propagator import propagate_jacobi_path2, _quat_multiply

OUT = Path(__file__).resolve().parent.parent / "results" / "s089"
OUT.mkdir(parents=True, exist_ok=True)

I_A = 100
S087_GATE_DEG = 1.25
INERTIA = m048_inertia()
DEG = np.pi / 180.0

# seed -> tumbling regime (source: PROGRESS.md line 49; 103 SAM from s082/83/84).
SEEDS = {119: "LAM", 116: "LAM", 103: "SAM"}
# anchor placement as fractions of the polhode period T_pol.
AB_FRAC = 0.30
AC_FRACS = [0.45, 0.60]
SIGMAS = [1.5, 2.0, 2.5, 3.3]
N_NOISE = 300

# Part 2 (ladder) multi-start grid -- mirrors s088 Gate A2.
N_DIR = 200
MAG_DPS = np.geomspace(0.04, 2.2, 11)
CONNECT_TOL_DEG = 1e-4
DEDUP_DIR_DEG = 2.0
DEDUP_MAG_FRAC = 0.01


def perturb_q(q, sigma_deg, rng):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    a = np.radians(sigma_deg)
    qp = np.array([np.cos(a / 2), *(np.sin(a / 2) * ax)])
    out = _quat_multiply(qp, q)
    return out / np.linalg.norm(out)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def dedup(roots):
    uniq = []
    for w in roots:
        if all(not (omega_dir_err_deg(w, u) < DEDUP_DIR_DEG and
                    abs(omega_mag_err_frac(w, u)) < DEDUP_MAG_FRAC) for u in uniq):
            uniq.append(w)
    return uniq


def anchor_epochs(times0, T_pol):
    """Place the intermediate (ab) and long (ac) anchors at fractions of T_pol.

    Anchor A is fixed at epoch I_A. Returns (i_b, [i_c, ...]) as epoch indices,
    spanning the LC if T_pol exceeds the remaining light curve.
    """
    remaining = times0[-1] - times0[I_A]
    span = min(T_pol, remaining)
    i_b = int(np.argmin(np.abs(times0 - (times0[I_A] + AB_FRAC * span))))
    i_cs = []
    for f in AC_FRACS:
        i_c = int(np.argmin(np.abs(times0 - (times0[I_A] + f * span))))
        if i_c > i_b and i_c < len(times0):
            i_cs.append(i_c)
    return i_b, i_cs, float(span), bool(T_pol > remaining)


# ---------------------------------------------------------------------------
# PART 1 -- conditioning sigma-sweep (init-at-truth)
# ---------------------------------------------------------------------------
def _trial(trial_seed, q_a, q_b, q_c, dt_ab, dt_ac, w_true, sigma):
    rng = np.random.default_rng(trial_seed)
    qa_n = perturb_q(q_a, sigma, rng)
    qb_n = perturb_q(q_b, sigma, rng)
    qc_n = perturb_q(q_c, sigma, rng)
    s_short = shoot(qa_n, qb_n, dt_ab, INERTIA, w_true)
    s_long = shoot(qa_n, qc_n, dt_ac, INERTIA, w_true)
    s_joint = shoot_multianchor(qa_n, [(qb_n, dt_ab), (qc_n, dt_ac)], INERTIA, w_true)
    return (
        omega_dir_err_deg(s_short["omega"], w_true), abs(omega_mag_err_frac(s_short["omega"], w_true)),
        omega_dir_err_deg(s_long["omega"], w_true), abs(omega_mag_err_frac(s_long["omega"], w_true)),
        omega_dir_err_deg(s_joint["omega"], w_true), abs(omega_mag_err_frac(s_joint["omega"], w_true)),
    )


def run_conditioning(seed, regime, q_hist, times0, w_true, i_b, i_cs, pool):
    dt_ab = float(times0[i_b] - times0[I_A])
    q_a, q_b = q_hist[I_A], q_hist[i_b]
    omega_mag = np.linalg.norm(w_true)
    out = []
    for i_c in i_cs:
        q_c = q_hist[i_c]
        dt_ac = float(times0[i_c] - times0[I_A])
        omdt_ab = np.degrees(omega_mag * dt_ab)
        omdt_ac = np.degrees(omega_mag * dt_ac)
        print(f"\n  --- seed {seed} [{regime}]  ab={i_b - I_A}ep(|w|dt={omdt_ab:.0f}deg) "
              f"ac={i_c - I_A}ep(|w|dt={omdt_ac:.0f}deg) ---")
        print(f"  {'sig':>4} | {'short2pt':>9} {'long2pt':>9} {'JOINT3pt':>9}"
              f" | {'sh|w|%':>7} {'jo|w|%':>7} | gate(joint)")
        rows = []
        for sg in SIGMAS:
            args = [(10_000 * int(sg * 10) + k, q_a, q_b, q_c, dt_ab, dt_ac, w_true, sg)
                    for k in range(N_NOISE)]
            r = np.array(pool.starmap(_trial, args))   # (N,6)
            sh_d, sh_m, lo_d, lo_m, jo_d, jo_m = r.T
            row = dict(
                sigma_q_deg=sg,
                short_dir_med=float(np.median(sh_d)), short_dir_p90=float(np.percentile(sh_d, 90)),
                long_dir_med=float(np.median(lo_d)), long_dir_p90=float(np.percentile(lo_d, 90)),
                joint_dir_med=float(np.median(jo_d)), joint_dir_p90=float(np.percentile(jo_d, 90)),
                short_mag_med_pct=float(100 * np.median(sh_m)),
                joint_mag_med_pct=float(100 * np.median(jo_m)),
            )
            rows.append(row)
            if regime == "LAM":
                gate = "PASS" if row["joint_dir_med"] < S087_GATE_DEG else "----"
            else:
                gate = "indic"   # 1.25 deg is LAM-derived; no hard pass/fail on SAM
            print(f"  {sg:>4} | {row['short_dir_med']:9.2f} {row['long_dir_med']:9.2f} "
                  f"{row['joint_dir_med']:9.2f} | {row['short_mag_med_pct']:7.2f} "
                  f"{row['joint_mag_med_pct']:7.2f} | {gate}")
        out.append(dict(seed=seed, regime=regime, gate_applies=bool(regime == "LAM"),
                        ab_epochs=i_b - I_A, ac_epochs=i_c - I_A,
                        dt_ab_s=dt_ab, dt_ac_s=dt_ac,
                        omega_dt_ab_deg=float(omdt_ab), omega_dt_ac_deg=float(omdt_ac),
                        n_noise=N_NOISE, rows=rows))
    return out


# ---------------------------------------------------------------------------
# PART 2 -- ladder collapse (noiseless multi-start)
# ---------------------------------------------------------------------------
def _ms_long(w_init, q_a, q_c, dt_ac):
    s = shoot(q_a, q_c, dt_ac, INERTIA, w_init)
    return s["omega"], s["geo_err_deg"]


def _ms_joint(w_init, q_a, q_b, q_c, dt_ab, dt_ac):
    s = shoot_multianchor(q_a, [(q_b, dt_ab), (q_c, dt_ac)], INERTIA, w_init)
    return s["omega"], s["geo_err_max_deg"]


def _count_inprior(uniq, w_true, lo=0.7, hi=1.3):
    mr = np.linalg.norm(w_true)
    rows = sorted(((omega_dir_err_deg(u, w_true), float(np.linalg.norm(u) / mr)) for u in uniq),
                  key=lambda x: x[1])
    in_band = [(d, r) for d, r in rows if lo <= r <= hi]
    truth_found = min((d for d, _ in rows), default=999.0) < 1.0
    near = [d for d, _ in in_band if d < 5.0]
    return dict(n_distinct=len(uniq), n_in_prior=len(in_band),
                n_in_prior_near_truth=len(near),
                truth_recovered=bool(truth_found),
                truth_unique=bool(len(in_band) == 1 and truth_found),
                per_root_in_prior=[[float(d), float(r)] for d, r in in_band])


def run_ladder(seed, regime, q_hist, times0, w_true, i_b, i_c, pool):
    q_a, q_b, q_c = q_hist[I_A], q_hist[i_b], q_hist[i_c]
    dt_ab = float(times0[i_b] - times0[I_A])
    dt_ac = float(times0[i_c] - times0[I_A])
    inits = np.vstack([(m * DEG) * fib_sphere(N_DIR) for m in MAG_DPS])
    print(f"\n  --- ladder seed {seed} [{regime}]: long2pt(ac={i_c - I_A}ep) vs "
          f"joint(ab={i_b - I_A},ac={i_c - I_A}), {len(inits)} inits ---")

    res_long = pool.map(partial(_ms_long, q_a=q_a, q_c=q_c, dt_ac=dt_ac), inits, chunksize=64)
    long_stats = _count_inprior(dedup([w for (w, geo) in res_long if geo < CONNECT_TOL_DEG]), w_true)

    res_joint = pool.map(partial(_ms_joint, q_a=q_a, q_b=q_b, q_c=q_c, dt_ab=dt_ab, dt_ac=dt_ac),
                         inits, chunksize=64)
    joint_stats = _count_inprior(dedup([w for (w, geo) in res_joint if geo < CONNECT_TOL_DEG]), w_true)

    print(f"    long 2pt : distinct={long_stats['n_distinct']:3d}  in-prior(+-30%)="
          f"{long_stats['n_in_prior']:3d}  truth_unique={long_stats['truth_unique']}")
    print(f"    JOINT 3pt: distinct={joint_stats['n_distinct']:3d}  in-prior(+-30%)="
          f"{joint_stats['n_in_prior']:3d}  truth_unique={joint_stats['truth_unique']}")
    return dict(seed=seed, regime=regime, ab_epochs=i_b - I_A, ac_epochs=i_c - I_A,
                long=long_stats, joint=joint_stats)


# ---------------------------------------------------------------------------
def make_plot(cond_results):
    seeds = list(SEEDS)
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.5 * len(seeds), 4.6), squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        cs = [x for x in cond_results if x["seed"] == seed]
        if not cs:
            continue
        c = cs[-1]                       # the longest ac geometry for this seed
        sig = [r["sigma_q_deg"] for r in c["rows"]]
        ax.plot(sig, [r["short_dir_med"] for r in c["rows"]], "s--", color="tab:orange",
                label=f"short 2-pt (Δt={c['ab_epochs']})")
        ax.plot(sig, [r["long_dir_med"] for r in c["rows"]], "^:", color="tab:red",
                label=f"long 2-pt (Δt={c['ac_epochs']})")
        ax.plot(sig, [r["joint_dir_med"] for r in c["rows"]], "o-", color="tab:blue",
                label=f"JOINT 3-pt ({c['ab_epochs']},{c['ac_epochs']})")
        ax.fill_between(sig, [r["joint_dir_med"] for r in c["rows"]],
                        [r["joint_dir_p90"] for r in c["rows"]], color="tab:blue", alpha=0.15)
        gl = f"s087 gate {S087_GATE_DEG}°" + ("" if c["gate_applies"] else " (LAM-derived)")
        ax.axhline(S087_GATE_DEG, color="k", ls=":", label=gl)
        ax.set_title(f"seed {seed} [{c['regime']}]")
        ax.set_xlabel("per-anchor q-cloud noise σ_q (deg)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("ω-direction error, median (deg)")
    fig.suptitle("s089 third-anchor over-determination: joint 3-point vs 2-point ω-dir error")
    fig.tight_layout()
    p = OUT / "conditioning.png"
    fig.savefig(p, dpi=120)
    print(f"\nSaved: {p}")


def main():
    cond_all, ladder_all = [], []
    with Pool(24) as pool:
        for seed, regime in SEEDS.items():
            d = tl.load_truth(seed)
            times0 = d["observation_times"].astype(np.float64)
            times0 = times0 - times0[0]
            q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
            q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
            w_true = w_hist[I_A]
            T_pol = polhode_period(w0, INERTIA)
            i_b, i_cs, span, capped = anchor_epochs(times0, T_pol)
            print(f"\n===== seed {seed} [{regime}]  |w|={np.degrees(np.linalg.norm(w0)):.4f} dps "
                  f"T_pol={T_pol:.0f}s  len={len(times0)}ep  span={span:.0f}s"
                  f"{'  (T_pol>LC: spanning LC)' if capped else ''} =====")
            if not i_cs:
                print(f"  (no valid long anchor within LC; skipping seed {seed})")
                continue
            # noiseless self-check: joint solve at the truth triple, init 5% off,
            # must recover w_true exactly (validates the stacked-residual convention).
            _anch = [(q_hist[i_b], float(times0[i_b] - times0[I_A])),
                     (q_hist[i_cs[-1]], float(times0[i_cs[-1]] - times0[I_A]))]
            _chk = shoot_multianchor(q_hist[I_A], _anch, INERTIA, w_true * 1.05)
            print(f"  [self-check] joint@truth (init +5%): dir_err="
                  f"{omega_dir_err_deg(_chk['omega'], w_true):.2e}deg  "
                  f"|w|_err={100 * abs(omega_mag_err_frac(_chk['omega'], w_true)):.2e}%  "
                  f"geo_max={_chk['geo_err_max_deg']:.2e}deg")
            cond_all.extend(run_conditioning(seed, regime, q_hist, times0, w_true, i_b, i_cs, pool))
            ladder_all.append(run_ladder(seed, regime, q_hist, times0, w_true, i_b, i_cs[-1], pool))

    with open(OUT / "conditioning.json", "w") as f:
        json.dump(cond_all, f, indent=2, default=float)
    print(f"Saved: {OUT / 'conditioning.json'}")
    with open(OUT / "ladder.json", "w") as f:
        json.dump(ladder_all, f, indent=2, default=float)
    print(f"Saved: {OUT / 'ladder.json'}")
    make_plot(cond_all)


if __name__ == "__main__":
    main()
