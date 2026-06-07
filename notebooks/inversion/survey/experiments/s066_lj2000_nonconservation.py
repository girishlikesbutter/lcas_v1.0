"""s066 — L_J2000 non-conservation in the codebase propagator: m048 cohort is
   non-physical, NOT a "convention residual" as s065 claimed.

Empirical finding (this session, post-s064): the codebase's `propagate_attitude`
output `(q(t), omega(t))` does NOT satisfy `dL_J2000/dt = 0` — the J2000-frame
angular momentum precesses by tens of percent over an LC. In real torque-free
physics, L_J2000 is exactly conserved (no external torque ⇒ no change in
inertial angular momentum). The m048 cohort therefore does NOT correspond to
any real tumbling rigid body, regardless of how its omega is interpreted.

This supersedes the s065 framing (which claimed the codebase's `omega = -omega_phys`
gave physical R via `-omega` substitution): that holds only at t=0 (matches SPICE
at 3e-16) and approximately over short ~1 s windows, but DOES NOT hold over the
60-min LC durations of the m048 cohort.

Mechanism (already known from s065):
    - dω/dt = -I^-1 (ω × I·ω)  -- textbook Euler (correct)
    - dq/dt = +0.5 ω ⊗ q       -- OPPOSITE SIGN of textbook passive J2000→body kinematic

The new finding: pairing textbook Euler with the wrong-signed kinematic does NOT
produce a physical trajectory under any sign-substitution. ω → -ω only makes
the kinematic textbook *locally*; the Euler equation is sign-asymmetric over
time (its RHS is even in ω, so trajectories starting from ±ω0 have the same
initial derivative but diverge thereafter). The integrated `(q(t), omega(t))`
satisfies its own coupled ODE but doesn't lie on any physical (torque-free)
phase-space trajectory.

Quantitative consequence on m048 cohort (this script):
    seed 89  (|ω|=0.240 dps, 60 min):  62.5% L_J2000 drift
    seed 28  (|ω|=1.438 dps, 60 min):  36.4% L_J2000 drift
    seed 14  (|ω|=1.229 dps, 60 min): 125.5% L_J2000 drift
plus a toy diagnostic on I=diag(1,2,3), ω0=(0.5,0.3,0.7) over 50s.

Implications:
    - Self-consistency: codebase forward model is deterministic + invertible.
      Inversion benchmark is well-posed, methodology survives, qualitative
      findings (multi-sol, polhode invariants, basin scaling) survive.
    - Architecture: Path 2 closed-form q(t) (s062b redux) is BLOCKED because
      there is no constant L_J2000 to precess around. Would require deriving a
      closed-form for the non-physical coupled ODE.
    - Real-world: m048 LCs are NOT what a telescope would see for any tumbling
      satellite at any (q0_phys, omega_phys). Eventual fix-and-regen needed to
      connect to real telescope data.

Usage:
    python experiments/s066_lj2000_nonconservation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

OUT = SURVEY / "results" / "s066_lj2000"
OUT.mkdir(parents=True, exist_ok=True)


def R_of_q(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def measure_drift(q_hist, om_hist, inertia, indices=None):
    """Compute L_J2000 = R(t).T @ I @ omega(t) at given epoch indices."""
    N = len(q_hist)
    if indices is None:
        indices = list(range(N))
    L_J = []
    for i in indices:
        R = R_of_q(q_hist[i])
        L_body = inertia @ om_hist[i]
        L_J.append(R.T @ L_body)
    L_J = np.asarray(L_J)
    drifts = np.linalg.norm(L_J - L_J[0], axis=1)
    return L_J, drifts


def gate_textbook_conservation(I_toy, omega0_toy, q0_toy, times):
    """Independent textbook integration (dR/dt = -[omega]_x R). MUST conserve L_J2000."""
    def textbook_dyn(t, y):
        R = y[:9].reshape(3, 3)
        omega = y[9:12]
        I_om = I_toy @ omega
        om_dot = np.linalg.inv(I_toy) @ (-np.cross(omega, I_om))
        om_skew = np.array([[0, -omega[2], omega[1]],
                             [omega[2], 0, -omega[0]],
                             [-omega[1], omega[0], 0]])
        R_dot = -om_skew @ R  # textbook passive J2000→body
        return np.concatenate([R_dot.flatten(), om_dot])

    R0_q = R_of_q(q0_toy)
    y0 = np.concatenate([R0_q.flatten(), omega0_toy])
    sol = solve_ivp(textbook_dyn, (times[0], times[-1]), y0, method="DOP853",
                    t_eval=times, rtol=1e-12, atol=1e-14)
    R_tb = sol.y[:9].T.reshape(-1, 3, 3)
    om_tb = sol.y[9:12].T
    L_J = []
    for i in range(len(times)):
        L_body = I_toy @ om_tb[i]
        L_J.append(R_tb[i].T @ L_body)
    L_J = np.asarray(L_J)
    drift = np.max(np.linalg.norm(L_J - L_J[0], axis=1))
    return drift, L_J


def main():
    print("=== s066 — L_J2000 non-conservation in m048 codebase propagator ===\n")
    results = {}

    # --- Part 1: control on toy with independent textbook integrator ---
    print("--- Part 1: control — independent textbook integrator MUST conserve L_J2000 ---")
    I_toy = np.diag([1.0, 2.0, 3.0])
    omega0_toy = np.array([0.5, 0.3, 0.7])
    q0_toy = np.array([1.0, 0.0, 0.0, 0.0])
    times_toy = np.linspace(0.0, 50.0, 51)

    drift_tb, L_J_tb = gate_textbook_conservation(I_toy, omega0_toy, q0_toy, times_toy)
    print(f"  textbook (dR/dt = -[ω]_x R) drift over 50s: {drift_tb:.2e}")
    assert drift_tb < 1e-10, f"Textbook integrator failed conservation (drift={drift_tb:.2e})"

    # --- Part 2: codebase on the same toy ---
    print("\n--- Part 2: codebase propagator on the same toy ---")
    q_cb, om_cb = propagate_attitude(q0=q0_toy, omega0=omega0_toy, times=times_toy,
                                      mode="tumbling", inertia_tensor=I_toy)
    L_J_cb, drifts_cb = measure_drift(q_cb, om_cb, I_toy)
    L0_norm = np.linalg.norm(L_J_cb[0])
    drift_rel_cb = drifts_cb.max() / L0_norm * 100
    print(f"  codebase drift over 50s: max ||L_J - L_J(0)|| = {drifts_cb.max():.3f} "
          f"({drift_rel_cb:.1f}% of |L_body|={L0_norm:.3f})")
    print(f"  |L_body| at sample epochs: {[float(np.linalg.norm(I_toy@om_cb[i])) for i in [0, 1, 10, 25, 50]]}")
    print(f"  -> codebase DOES NOT conserve L_J2000 even on a controlled toy.")
    results["toy"] = {
        "I": I_toy.diagonal().tolist(), "omega0": omega0_toy.tolist(), "q0": q0_toy.tolist(),
        "duration_s": float(times_toy[-1] - times_toy[0]),
        "textbook_drift_abs": float(drift_tb),
        "codebase_drift_abs_max": float(drifts_cb.max()),
        "codebase_drift_rel_pct": float(drift_rel_cb),
        "L_body_mag": float(L0_norm),
    }

    # --- Part 3: m048 cohort (3 representative seeds) ---
    print("\n--- Part 3: m048 cohort (slow / fast / near-separatrix) ---")
    ctx = build_context(seed=89)
    inertia = np.asarray(ctx["inertia_tensor"])
    print(f"  I_body (m048): diag {np.diag(inertia)}")
    cohort = {}
    for seed in [89, 28, 14]:
        d = load_truth(seed)
        q0 = np.asarray(d["q0_wxyz"])
        om0 = np.asarray(d["omega0_rad"])
        times = np.asarray(d["observation_times"])
        q_hist, om_hist = propagate_attitude(
            q0=q0, omega0=om0, times=times, mode="tumbling",
            inertia_tensor=inertia,
        )
        sample_idx = [0, 1, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
        L_J, drifts = measure_drift(q_hist, om_hist, inertia, indices=sample_idx)
        L0_norm = np.linalg.norm(L_J[0])
        drift_rel = drifts.max() / L0_norm * 100
        om_mag_dps = float(np.linalg.norm(om0) * 180.0 / np.pi)
        T_min = float((times[-1] - times[0]) / 60.0)
        print(f"  seed {seed:03d}: |ω|={om_mag_dps:.3f} dps, T={T_min:.1f} min  ->  "
              f"max drift = {drifts.max():.1f} ({drift_rel:.1f}% of |L_body|={L0_norm:.1f})")
        cohort[seed] = {
            "omega_mag_dps": om_mag_dps, "duration_min": T_min,
            "L_body_mag": float(L0_norm),
            "drift_abs_max": float(drifts.max()),
            "drift_rel_pct": float(drift_rel),
            "sample_indices": sample_idx,
            "sample_L_J2000": L_J.tolist(),
        }
    results["m048_cohort"] = cohort

    # --- Headline ---
    print(f"\n{'='*60}")
    print(f"HEADLINE: L_J2000 NOT conserved by codebase propagator.")
    print(f"  Textbook control: drift {drift_tb:.2e} (machine precision)")
    print(f"  Codebase toy:     drift {drift_rel_cb:.1f}% over 50 s")
    print(f"  m048 cohort:")
    for s, c in cohort.items():
        print(f"    seed {s:03d}: drift {c['drift_rel_pct']:.1f}% over {c['duration_min']:.0f} min")
    print(f"  Conclusion: m048 trajectories DO NOT correspond to any physical")
    print(f"  torque-free rigid body. The forward model is self-consistent but")
    print(f"  non-physical. s065 's 'omega-sign convention residual' framing")
    print(f"  understated the issue.")
    print(f"{'='*60}")

    out = OUT / "summary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
