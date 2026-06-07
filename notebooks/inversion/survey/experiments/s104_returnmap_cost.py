"""s104 — atomic cost of the return-map approach vs the LM-multistart, then
project the full qa-qb cross under the user's revised plan (1M clouds ->
~549x651 pairs on seed 119, physical bracket [0.1,1.6] deg/s -> ~4 windings).

Measures:
  t_prop = one closed-form propagate_jacobi_path2(q_a, w, [0,dt])  (return-map atom)
  t_LM   = one shoot() LM solve                                    (old multistart atom)
Then projects per-pair and full-cross wall for both schemes. deg/s throughout.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia, finite_diff_omega, shoot
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180/np.pi
SEED, EP_A, EP_B = 119, 69, 172
I = m048_inertia()
N_PAIRS = 549 * 651          # measured 1M-cloud reps (s102)
N_CORES = 24
N_DIR = 17                   # same direction sampling as the probe
N_SCAN = 40                  # mags per direction to resolve ~4 windings
N_DEEP = 4                   # deep minima LM-polished per pair (generous)


def main():
    d = tl.load_truth(SEED)
    t = d["observation_times"].astype(float); t -= t[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, I, t)
    q_a, q_b = qh[EP_A], qh[EP_B]
    dt = float(t[EP_B] - t[EP_A])
    times = np.array([0.0, dt])
    w_fd = finite_diff_omega(q_a, q_b, dt)

    # t_prop
    n = 5000
    ts = time.perf_counter()
    for _ in range(n):
        propagate_jacobi_path2(q_a, w_fd, I, times)
    t_prop = (time.perf_counter() - ts) / n

    # t_LM
    nl = 50
    ts = time.perf_counter()
    for _ in range(nl):
        shoot(q_a, q_b, dt, I, w_fd)
    t_LM = (time.perf_counter() - ts) / nl

    print(f"=== s104 atomic costs (single core) ===")
    print(f"t_prop (1 closed-form propagation) = {t_prop*1e3:.3f} ms")
    print(f"t_LM   (1 LM shoot)                = {t_LM*1e3:.3f} ms   ({t_LM/t_prop:.0f} props/solve)")
    print(f"\nN_pairs = {N_PAIRS:,} | cores = {N_CORES}\n")

    def proj(per_pair_s, label):
        wall = N_PAIRS * per_pair_s / N_CORES
        print(f"  {label:38s}: {per_pair_s*1e3:7.1f} ms/pair -> {wall/60:6.1f} min ({wall/3600:.2f} h)")

    print("--- OLD: 170-start LM multistart (17 dir x 10 mag, all LM solves) ---")
    proj(170 * t_LM, "170 LM solves/pair")
    print("\n--- RETURN-MAP: cheap scan + LM only on deep minima ---")
    scan = N_DIR * N_SCAN * t_prop
    proj(scan + N_DEEP * t_LM, f"{N_DIR}dir x {N_SCAN} scan-props + {N_DEEP} LM")
    print("    (breakdown: scan = {:.1f} ms/pair, LM-polish = {:.1f} ms/pair)".format(
        scan*1e3, N_DEEP*t_LM*1e3))
    print("\n--- RETURN-MAP, scan-discard: non-connecting pairs skip LM entirely ---")
    for frac in (1.0, 0.1, 0.01):
        proj(scan + frac * N_DEEP * t_LM, f"scan all + LM on {frac*100:.0f}% of pairs")
    print("\n--- middle option: 68 LM solves/pair (17 dir x 4 winding rungs, no scan) ---")
    proj(68 * t_LM, "68 LM solves/pair")


if __name__ == "__main__":
    main()
