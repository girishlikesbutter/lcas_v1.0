"""s105c — null control for s105b's "16/16 q_b-connectors also hit q_c".

Two readings of that result:
  (i) q_c-connection is ALSO degenerate (any ~ω lands near q_c) -> 16/16 trivial,
      3-anchor adds nothing.
  (ii) q_b-connection SPECIFICALLY implies q_c-connection (a real structure) ->
      surprising; would mean the triple is satisfied by a discrete multi-sol set.

Distinguish by measuring, for RANDOM ω in the bracket (NOT filtered by q_b):
  - fraction landing within {1,3,10} deg of q_b at dt_ab
  - fraction landing within {1,3,10} deg of q_c at dt_ac
  - fraction within 3 deg of BOTH
If random ω hits q_c as often as the q_b-connectors do, q_c is degenerate (i).
Also independently re-verify one s105b connector (full-LC surrogate RMSE) to see
if these are genuine alternative trajectories vs numerical artifacts. deg/s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia, geodesic_angle
from lib.jacobi_propagator import propagate_jacobi_path2
from scipy.spatial.transform import Rotation

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED, EP_A, EP_B, EP_C = 119, 69, 172, 377
S_LO, S_HI = np.radians(0.1), np.radians(1.6)
N_RAND = 20000
SP, AD = 0.0, 15.0


def main():
    d = tl.load_truth(SEED)
    t = d["observation_times"].astype(float); t -= t[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, t)
    q_a, q_b, q_c = qh[EP_A], qh[EP_B], qh[EP_C]
    dt_ab = float(t[EP_B] - t[EP_A]); dt_ac = float(t[EP_C] - t[EP_A])
    true_mag = float(np.linalg.norm(wh[EP_A]))

    print(f"=== s105c connection null | seed {SEED} A=ep{EP_A} B=ep{EP_B} C=ep{EP_C} ===")
    print(f"truth |w_a|={true_mag*R2D:.4f} deg/s | dt_ab={dt_ab:.0f}s dt_ac={dt_ac:.0f}s")
    print(f"random sample: {N_RAND} omega, dir~uniform sphere, |w|~U[{S_LO*R2D:.1f},{S_HI*R2D:.1f}] deg/s\n", flush=True)

    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(N_RAND, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mags = rng.uniform(S_LO, S_HI, N_RAND)
    W = dirs * mags[:, None]

    gb = np.empty(N_RAND); gc = np.empty(N_RAND)
    for i in range(N_RAND):
        gb[i] = geodesic_angle(propagate_jacobi_path2(q_a, W[i], INERTIA, np.array([0.0, dt_ab]))[0][-1], q_b)
        gc[i] = geodesic_angle(propagate_jacobi_path2(q_a, W[i], INERTIA, np.array([0.0, dt_ac]))[0][-1], q_c)
    gb *= R2D; gc *= R2D

    print(f"{'tol':>5} | {'hit q_b':>12} | {'hit q_c':>12} | {'hit BOTH':>12}")
    for tol in (1, 3, 10, 30):
        fb = (gb < tol).mean(); fc = (gc < tol).mean(); fboth = ((gb < tol) & (gc < tol)).mean()
        print(f"{tol:>4}d | {100*fb:9.3f} %  | {100*fc:9.3f} %  | {100*fboth:9.4f} %", flush=True)

    # conditional: among random ω that hit q_b<3, what fraction also hit q_c<3?
    mask_b = gb < 3
    if mask_b.sum():
        cond = (gc[mask_b] < 3).mean()
        print(f"\nP(hit q_c<3 | hit q_b<3) = {100*cond:.1f}%  (n_b={int(mask_b.sum())})", flush=True)
        print(f"P(hit q_c<3) unconditional = {100*(gc<3).mean():.3f}%", flush=True)
    print(f"\nmedian geodesic of random ω: to q_b {np.median(gb):.1f} deg, to q_c {np.median(gc):.1f} deg", flush=True)


if __name__ == "__main__":
    main()
