"""s102 — ground the multi-start cross timing under the user's revised plan.

Measures, on seed 119's actual anchors (ep_a=69, ep_b=172 from
results/s100/seed119/invert.json):
  (1) how many 2-deg reps a 1M-sample isophote cloud yields at A and B
      (-> the pair count for the cross), plus nearest-truth coverage;
  (2) the wall-time of ONE 170-start multistart_shoot (17 dir x 10 mag) over
      the PHYSICAL |w| bracket [0.1, 1.5] deg/s.

Then projects the full multi-start-cross wall = (reps_A * reps_B) * t_pair / 24.
All rates reported in deg/s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, importlib
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY)); sys.path.insert(0, str(SURVEY / "experiments"))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units, nearest_in_pool_to_truth
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2
s100 = importlib.import_module("s100_5step_proto")   # reuse decimate_2deg + multistart_shoot

R2D = 180.0 / np.pi
SEED = 119
EP_A, EP_B = 69, 172          # source: results/s100/seed119/invert.json
TOL_MAG, SP, AD = 0.10, 0.0, 15.0
INERTIA = m048_inertia()
N_SAMPLES = 1_000_000
N_WORK = 24
# physical bracket
W_LO = np.radians(0.1); W_HI = np.radians(1.5)

_SURR = None
_ANCH = None   # (sun, obs, od, mg)


def _winit(anch):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    global _SURR, _ANCH
    _SURR = get_model(); _ANCH = anch


def _dense_worker(args):
    wseed, n = args
    sun, obs, od, mg = _ANCH
    rng = np.random.default_rng(wseed)
    R = Rotation.random(int(n), random_state=rng); Rm = R.as_matrix()
    k1 = np.einsum("nij,j->ni", Rm, sun); k2 = np.einsum("nij,j->ni", Rm, obs)
    pred = _SURR.predict_magnitude(k1, k2, SP, AD, np.full(int(n), od))
    m = np.abs(pred - mg) < TOL_MAG
    return R[m].as_quat()[:, [3, 0, 1, 2]].astype(np.float64) if m.any() else np.empty((0, 4))


def dense_1m_par(ctx, sun, obs, od, mg, n=N_SAMPLES):
    per = int(np.ceil(n / N_WORK))
    args = [(7000 + i, per) for i in range(N_WORK)]
    out = []
    with ctx.Pool(N_WORK, initializer=_winit, initargs=((sun, obs, od, mg),)) as p:
        for r in p.imap_unordered(_dense_worker, args):
            if len(r):
                out.append(r)
    return np.vstack(out) if out else np.empty((0, 4))


def main():
    t0 = time.time()
    ctx = get_context("fork")
    model = get_model()
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    dt_ab = float(times0[EP_B] - times0[EP_A])
    print(f"=== s102 | seed {SEED} | anchors A=ep{EP_A} B=ep{EP_B} | dt_ab={dt_ab:.0f}s | "
          f"truth |w|={np.linalg.norm(w0)*R2D:.3f} deg/s ===", flush=True)
    print(f"physical bracket [{W_LO*R2D:.1f},{W_HI*R2D:.1f}] deg/s -> windings "
          f"{(W_HI-W_LO)*dt_ab/(2*np.pi):.2f}", flush=True)

    # (1) 1M clouds -> reps
    reps = {}
    for tag, ep in (("A", EP_A), ("B", EP_B)):
        ts = time.time()
        Q = dense_1m_par(ctx, sun_u[ep], obs_u[ep], float(od[ep]), float(mag[ep]))
        rep, ncells = s100.decimate_2deg(Q, rng=np.random.default_rng(1))
        ntr, _ = nearest_in_pool_to_truth(rep, q_hist[ep]) if len(rep) else (np.nan, -1)
        reps[tag] = rep
        print(f"[1] {tag}: 1M -> {len(Q)} survivors -> {ncells} cells -> {len(rep)} reps "
              f"(nearest truth {ntr:.2f} deg) [{time.time()-ts:.0f}s]", flush=True)

    # (2) time ONE 170-start multistart_shoot on a representative pair
    rA, rB = reps["A"], reps["B"]
    qa = rA[len(rA)//2]; qb = rB[len(rB)//2]
    n_reps = 3
    t_pairs = []
    for _ in range(n_reps):
        ts = time.time()
        roots, _ = s100.multistart_shoot(qa, qb, dt_ab, W_LO, W_HI, w_hist[EP_A])
        t_pairs.append(time.time() - ts)
    t_pair = float(np.median(t_pairs))
    print(f"[2] one 170-start (17 dir x 10 mag) multistart_shoot: {t_pair*1000:.0f} ms "
          f"(median of {n_reps}); produced {len(roots)} roots", flush=True)

    # projection
    npairs = len(rA) * len(rB)
    for ncore in (1, 24):
        wall = npairs * t_pair / ncore
        print(f"[proj] {len(rA)}x{len(rB)}={npairs:,} pairs x {t_pair*1000:.0f}ms / {ncore} cores "
              f"= {wall:.0f}s = {wall/60:.1f} min = {wall/3600:.2f} h", flush=True)
    print(f"total measure wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
