"""s105b — does a THIRD anchor collapse the under-determined pair-connection family?

s105a showed a single (q_a, q_b) pair is connection-degenerate: ~every direction
on the sphere admits a magnitude that threads q_a->q_b over dt_ab (2-parameter
family). So pairwise connection cannot pick truth-omega. The classic fix is
over-determination: require the SAME omega to also thread q_a->q_c over dt_ac.

This measures, on seed 119's truth TRIPLE (A=ep69, B=ep172, C=ep377):
  for N_dir Fibonacci directions, sweep |omega| over [0.1,1.6] deg/s, find every
  return that connects q_a->q_b (geodesic < QB_TOL), and for each such (dir, |w|)
  candidate measure the geodesic residual to q_c at the SAME omega. Question:
  is small q_c-residual confined to near-truth directions, or is the triple
  ALSO degenerate? If confined -> 3-anchor over-determination is the discriminator
  and the s105 architecture is a 3-anchor return-map cross. deg/s throughout.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
from pathlib import Path
import numpy as np
from scipy.signal import argrelmin

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia, geodesic_angle
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED, EP_A, EP_B, EP_C = 119, 69, 172, 377
S_LO, S_HI = np.radians(0.1), np.radians(1.6)
N_SCAN = 600
N_DIR = 800
QB_TOL = np.radians(3.0)        # accept as "connects to q_b"


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])


def angle_between(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), 0, 1))))


def main():
    d = tl.load_truth(SEED)
    t = d["observation_times"].astype(float); t -= t[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, t)
    q_a, q_b, q_c = qh[EP_A], qh[EP_B], qh[EP_C]
    w_a = wh[EP_A]; true_mag = float(np.linalg.norm(w_a)); true_dir = w_a / true_mag
    dt_ab = float(t[EP_B] - t[EP_A]); dt_ac = float(t[EP_C] - t[EP_A])
    s_grid = np.linspace(S_LO, S_HI, N_SCAN)

    print(f"=== s105b 3-anchor collapse | seed {SEED} A=ep{EP_A} B=ep{EP_B} C=ep{EP_C} ===")
    print(f"truth |w_a|={true_mag*R2D:.4f} deg/s | dt_ab={dt_ab:.0f}s ({true_mag*dt_ab/(2*np.pi):.2f} turns) "
          f"dt_ac={dt_ac:.0f}s ({true_mag*dt_ac/(2*np.pi):.2f} turns)", flush=True)
    print(f"scanning {N_DIR} dirs x {N_SCAN} mags; accept q_b-connect < {QB_TOL*R2D:.0f} deg\n", flush=True)

    dirs = fib_sphere(N_DIR)
    cand = []   # (dir_offset, qb_depth, qc_resid, wmag_dps)
    for dv in dirs:
        gb = np.array([geodesic_angle(
            propagate_jacobi_path2(q_a, sv * dv, INERTIA, np.array([0.0, dt_ab]))[0][-1], q_b)
            for sv in s_grid])
        mins = argrelmin(gb, order=3)[0]
        mins = mins[gb[mins] < QB_TOL]
        off = angle_between(dv, true_dir)
        for m in mins:
            w = s_grid[m] * dv
            qc_pred = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, dt_ac]))[0][-1]
            qc = geodesic_angle(qc_pred, q_c)
            cand.append((off, float(gb[m]), float(qc), float(s_grid[m] * R2D)))

    cand = np.array(cand)
    n_b = len(cand)
    print(f"[connect] {n_b} (dir,|w|) candidates thread q_a->q_b (< {QB_TOL*R2D:.0f} deg)", flush=True)
    if n_b == 0:
        print("none — abort"); return

    # how does q_c residual gate things?
    for qc_tol in (1, 3, 5, 10, 20):
        sel = cand[cand[:, 2] < qc_tol]
        if len(sel):
            off_max = sel[:, 0].max(); off_min = sel[:, 0].min()
            print(f"[3-anchor] q_c-resid < {qc_tol:2d} deg: {len(sel):4d}/{n_b} survive | "
                  f"dir-offset-from-truth range [{off_min:5.1f}, {off_max:5.1f}] deg | "
                  f"|w| range [{sel[:,3].min():.3f},{sel[:,3].max():.3f}] deg/s", flush=True)
        else:
            print(f"[3-anchor] q_c-resid < {qc_tol:2d} deg: 0/{n_b}", flush=True)

    # the best q_c survivor
    k = int(np.argmin(cand[:, 2]))
    print(f"\n[best q_c] resid {cand[k,2]:.3f} deg | dir-offset {cand[k,0]:.2f} deg | "
          f"|w| {cand[k,3]:.4f} deg/s (truth {true_mag*R2D:.4f})", flush=True)
    # count near-truth-direction connectors
    near = cand[cand[:, 0] < 5]
    print(f"[context] connectors within 5 deg of truth dir: {len(near)} | "
          f"their q_c-resid range [{near[:,2].min():.2f},{near[:,2].max():.2f}] deg" if len(near) else
          "[context] no connector within 5 deg of truth dir", flush=True)

    out = SURVEY / "results" / "s105"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "third_anchor.npz", cand=cand, true_mag_dps=true_mag * R2D,
             ep=(EP_A, EP_B, EP_C), dt_ab=dt_ab, dt_ac=dt_ac)
    summ = dict(seed=SEED, ep=[EP_A, EP_B, EP_C], n_connect_b=n_b,
                best_qc_resid=float(cand[k, 2]), best_qc_dir_offset=float(cand[k, 0]),
                best_qc_wmag_dps=float(cand[k, 3]), true_mag_dps=true_mag * R2D,
                gate={str(q): int((cand[:, 2] < q).sum()) for q in (1, 3, 5, 10, 20)})
    with open(out / "third_anchor.json", "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(f"\nSaved: {out/'third_anchor.json'}\nSaved: {out/'third_anchor.npz'}", flush=True)


if __name__ == "__main__":
    main()
