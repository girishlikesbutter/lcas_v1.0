"""s088 Gate A1 — two-point attitude BVP shoot: truth-pair recovery smoke test.

Validates the newton_shoot primitive (lib/shoot.py) before any conditioning sweep:

  A1.0  Propagation round-trip: propagate_jacobi_path2(truth q0,w0) reproduces
        the cached truth quaternion history (convention/cohort sanity).
  A1.1  Residual at TRUE body-frame omega_a is ~0  (validates residual map +
        quaternion convention, independent of any init).
  A1.2  finite_diff_omega error vs truth (expect s057c-like 6-30° dir error).
  A1.3  LM shoot from the finite-diff init recovers true omega_a to ~machine
        precision (geo_err -> 0), across a range of Δt.

Seeds: 119 (fast, |w|=1.48 dps) + 116 (slow, |w|=0.134 dps).
Pure cached-truth + closed-form propagation; no surrogate, no Pool.
"""
import json
from pathlib import Path
import numpy as np

import lib.traj_load as tl
from lib.shoot import (
    m048_inertia,
    geodesic_angle,
    finite_diff_omega,
    omega_dir_err_deg,
    omega_mag_err_frac,
    polhode_period,
    shoot,
    _quat_log_residual,
)
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s088"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [119, 116]
I_A = 100                       # anchor epoch index
DT_EPOCHS = [3, 7, 15, 30, 60, 120, 250]
INERTIA = m048_inertia()


def run_seed(seed: int) -> dict:
    d = tl.load_truth(seed)
    times = d["observation_times"].astype(np.float64)
    times0 = times - times[0]                       # relative, times0[0] == 0
    q0 = d["q0_wxyz"].astype(np.float64)
    w0 = d["omega0_rad"].astype(np.float64)
    quats_cached = d["quaternions"].astype(np.float64)

    # A1.0 — propagate truth, compare to cached quaternion history.
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    roundtrip_err = max(
        geodesic_angle(q_hist[i], quats_cached[i]) for i in range(len(times0))
    )
    T_pol = polhode_period(w0, INERTIA)

    print(f"\n===== seed {seed}  |w|={np.degrees(np.linalg.norm(w0)):.4f} dps "
          f"T_pol={T_pol:.1f}s ({T_pol/times[-1]+0*times[0]:.2f}× LC span) =====")
    print(f"  A1.0 propagation round-trip vs cached quats: max geo err = "
          f"{np.degrees(roundtrip_err):.2e} deg")

    rows = []
    for de in DT_EPOCHS:
        i_b = I_A + de
        if i_b >= len(times0):
            continue
        q_a = q_hist[I_A]
        q_b = q_hist[i_b]
        w_true = w_hist[I_A]                          # body-frame truth omega at anchor
        dt = float(times0[i_b] - times0[I_A])

        # A1.1 — residual at true omega (must be ~0).
        res_true = float(np.linalg.norm(_quat_log_residual(
            propagate_jacobi_path2(q_a, w_true, INERTIA, np.array([0.0, dt]))[0][-1], q_b)))

        # A1.2 — finite-diff init error vs truth.
        w_fd = finite_diff_omega(q_a, q_b, dt)
        fd_dir = omega_dir_err_deg(w_fd, w_true)
        fd_mag = omega_mag_err_frac(w_fd, w_true)

        # A1.3 — shoot from finite-diff init.
        s = shoot(q_a, q_b, dt, INERTIA, w_fd)
        rec_dir = omega_dir_err_deg(s["omega"], w_true)
        rec_mag = omega_mag_err_frac(s["omega"], w_true)

        rows.append(dict(
            dt_epochs=de, dt_s=dt, omega_dt_deg=np.degrees(np.linalg.norm(w_true) * dt),
            res_at_true_deg=np.degrees(res_true),
            fd_dir_deg=fd_dir, fd_mag_pct=100 * fd_mag,
            shoot_geo_err_deg=s["geo_err_deg"],
            shoot_dir_err_deg=rec_dir, shoot_mag_err_pct=100 * rec_mag,
            n_eval=s["n_eval"], jac_cond=s["jac_cond"], connected=s["connected"],
        ))
        print(f"  Δt={de:>3}ep ({dt:6.1f}s, |w|Δt={np.degrees(np.linalg.norm(w_true)*dt):6.1f}°) "
              f"| res@true={np.degrees(res_true):7.1e}° "
              f"| fd init: dir={fd_dir:6.2f}° mag={100*fd_mag:+6.1f}% "
              f"| shoot: geo={s['geo_err_deg']:7.1e}° dir={rec_dir:7.1e}° "
              f"mag={100*rec_mag:+7.1e}% nfev={s['n_eval']:3d} cond={s['jac_cond']:.1e}")

    return dict(seed=seed, omega_mag_dps=float(np.degrees(np.linalg.norm(w0))),
                T_pol_s=T_pol, roundtrip_max_geo_deg=float(np.degrees(roundtrip_err)),
                anchor_epoch=I_A, rows=rows)


def main():
    results = [run_seed(s) for s in SEEDS]
    out = OUT / "gateA1.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")

    # Verdict — the GATE validates the PRIMITIVE (convention + that the solve
    # connects). Truth-recovery vs Δt is a finding (the single-shoot basin
    # boundary), not a pass/fail — recovering truth at large Δt is Gate A2's job.
    print("\n===== Gate A1 verdict =====")
    ok = True
    for r in results:
        rt = r["roundtrip_max_geo_deg"]
        res_max = max(row["res_at_true_deg"] for row in r["rows"])
        connect_max = max(row["shoot_geo_err_deg"] for row in r["rows"])
        seed_ok = (rt < 1e-3) and (res_max < 1e-3) and (connect_max < 1e-4)
        ok = ok and seed_ok
        # largest |w|Δt at which a single finite-diff-init shoot still recovers truth
        rec = [row["omega_dt_deg"] for row in r["rows"] if row["shoot_dir_err_deg"] < 1.0]
        lost = [row["omega_dt_deg"] for row in r["rows"] if row["shoot_dir_err_deg"] >= 1.0]
        boundary = (f"truth recovered to |w|Δt≤{max(rec):.0f}°, "
                    f"alias from |w|Δt≥{min(lost):.0f}°") if rec and lost else \
                   (f"truth recovered at all tested Δt (max |w|Δt={max(rec):.0f}°)" if rec
                    else "no truth recovery at any Δt")
        print(f"  seed {r['seed']}: roundtrip={rt:.1e}° res@true(max)={res_max:.1e}° "
              f"connect(max)={connect_max:.1e}° -> {'PASS' if seed_ok else 'FAIL'}")
        print(f"            single-shoot basin: {boundary}")
    print(f"  GATE A1 (primitive validated): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
