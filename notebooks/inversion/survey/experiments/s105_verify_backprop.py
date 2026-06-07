"""s105_verify_backprop — is the anchor->full-LC propagation in s105d/e/f correct?

Challenge: omega is at the ANCHOR (ep_a), not t=0. If the full-LC reconstruction
doesn't back-propagate correctly, every RMSE is garbage.

Three independent checks on seed 119, ep_a=69:
  A. reconstruct truth from (q_a, w_a) via s105's _propagate_full (fwd from ep_a +
     bwd from ep_a) and compare to the direct truth trajectory from t=0.
  B. back-propagate (q_a, w_a) to t=0 -> (q0', w0'), compare to true (q0, w0).
  C. for a PERTURBED omega (truth dir + 4.25 deg, same |w|), compute full-LC RMSE
     two ways: (1) s105 anchor-stitch path, (2) back-prop-to-t0-then-forward path
     (the s099/s100_hifi path). If they agree, the anchor propagation is sound and
     the s105f 0.687 result is real.
All angles deg.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED, EP_A = 119, 69
SP, AD = 0.0, 15.0


def geo(qx, qy):
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qx * qy, axis=-1)), 0, 1)))


def propagate_full_anchor(q_a, w, times0, ep_a):
    """s105 path: forward from ep_a + backward from ep_a, stitched."""
    tf = times0[ep_a:] - times0[ep_a]
    qf = propagate_jacobi_path2(q_a, w, INERTIA, tf)[0]
    if ep_a == 0:
        return qf
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
    qb = propagate_jacobi_path2(q_a, w, INERTIA, tb)[0]
    return np.vstack([qb[::-1][:-1], qf])


def state_at_t0(q_a, w_a, times0, ep_a):
    """s099/s100_hifi path: back-propagate to t=0."""
    if ep_a == 0:
        return np.asarray(q_a, float), np.asarray(w_a, float)
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
    qb, wb = propagate_jacobi_path2(np.asarray(q_a, float), np.asarray(w_a, float), INERTIA, tb)
    return qb[-1], wb[-1]


def lc_rmse(quats, sun, obs, od, mag):
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    model = get_model()
    pred = model.predict_magnitude(np.einsum("nij,nj->ni", R, sun),
                                   np.einsum("nij,nj->ni", R, obs), SP, AD, od)
    m = np.isfinite(mag)
    return float(np.sqrt(np.mean((pred[m] - mag[m]) ** 2)))


def main():
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a, w_a = qh[EP_A], wh[EP_A]
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    N = len(times0)
    total_turns = np.linalg.norm(w0) * (times0[-1] - times0[0]) / (2 * np.pi)

    print(f"=== s105 backprop verification | seed {SEED} ep_a={EP_A} N={N} ===")
    print(f"truth |w|={np.linalg.norm(w0)*R2D:.4f} deg/s | FULL-LC spans {total_turns:.2f} turns\n", flush=True)

    # A. reconstruct truth from anchor
    qrec = propagate_full_anchor(q_a, w_a, times0, EP_A)
    errA = geo(qrec, qh)
    print(f"[A] reconstruct truth from (q_a,w_a) via anchor-stitch vs direct-from-t0:")
    print(f"    max geodesic over 500 epochs = {errA.max():.3e} deg  (mean {errA.mean():.3e})", flush=True)

    # B. back-propagate to t=0
    q0b, w0b = state_at_t0(q_a, w_a, times0, EP_A)
    print(f"[B] back-prop (q_a,w_a) -> t=0:  q0 err {geo(q0b,q0):.3e} deg | "
          f"w0 err {np.linalg.norm(w0b-w0)*R2D:.3e} deg/s", flush=True)

    # C. perturbed omega, RMSE two ways
    rng = np.random.default_rng(0)
    perp = np.cross(w_a, rng.normal(size=3)); perp /= np.linalg.norm(perp)
    ang = np.radians(4.25); wmag = np.linalg.norm(w_a)
    w_pert = wmag * (np.cos(ang) * (w_a / wmag) + np.sin(ang) * perp)
    # path 1: anchor stitch
    q1 = propagate_full_anchor(q_a, w_pert, times0, EP_A)
    rmse1 = lc_rmse(q1, sun_u, obs_u, od, mag)
    # path 2: back-prop to t0 then forward
    q0p, w0p = state_at_t0(q_a, w_pert, times0, EP_A)
    q2 = propagate_jacobi_path2(q0p, w0p, INERTIA, times0)[0]
    rmse2 = lc_rmse(q2, sun_u, obs_u, od, mag)
    # also truth rmse both ways
    rt1 = lc_rmse(propagate_full_anchor(q_a, w_a, times0, EP_A), sun_u, obs_u, od, mag)
    rt2 = lc_rmse(qh, sun_u, obs_u, od, mag)

    print(f"\n[C] perturbed omega = truth dir + 4.25 deg, same |w|:")
    print(f"    full-LC RMSE  anchor-stitch path = {rmse1:.4f}")
    print(f"    full-LC RMSE  backprop-to-t0 path = {rmse2:.4f}   (agree if backprop correct)")
    print(f"    trajectory disagreement path1 vs path2: max geodesic {geo(q1,q2).max():.3e} deg")
    print(f"\n[control] truth full-LC RMSE: anchor-stitch {rt1:.4f} | direct {rt2:.4f}", flush=True)
    print(f"\nVERDICT: backprop correct if [A] ~0, [B] ~0, [C] paths agree, control ~0.018.", flush=True)


if __name__ == "__main__":
    main()
