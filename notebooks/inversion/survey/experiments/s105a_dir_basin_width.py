"""s105a — measure the DIRECTION-BASIN WIDTH of the propagator return-map.

The s104 cost estimate (~11 min full cross) assumed N_dir=17 directions on the
sphere are enough for the return-map to detect a connecting pair's deep return.
That is only true if the "deep return" (geodesic(propagate(q_a, s*dir, dt), q_b)
~ 0) survives a direction offset wider than the grid spacing. A 17-dir Fibonacci
sphere has ~40 deg nearest-neighbour spacing; if the basin is narrower than
that, a blind 17-dir grid MISSES truth and the cost estimate is invalid.

This measures, on seed 119's truth pair (the hard 3.06-turn case) and 116's
(the easy 0.35-turn case):
  (1) OFFSET SCAN: best return depth as a function of angular offset of the
      scanned direction from the TRUTH omega direction -> basin half-width.
  (2) FIBONACCI SWEEP: for blind grids of N_dir in {20,50,100,200,500,1000},
      the deepest return achieved and the angular offset of the winning
      direction from truth -> the operational N_dir needed for a hit.

deg/s throughout; magnitude bracket [0.1, 1.6] deg/s (physical prior widened to
absorb polhode |omega| variability). Truth direction used ONLY as a measurement
label, never to steer a production choice.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
import json
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia, geodesic_angle
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
S_LO, S_HI = np.radians(0.1), np.radians(1.6)
N_SCAN = 400                                   # magnitude resolution per direction
CASES = [(119, 69, 172), (116, None, None)]    # 116 anchors filled from its invert.npz


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])


def best_return_for_dir(q_a, q_b, dt, ndir, s_grid):
    """Min geodesic-to-q_b over the magnitude sweep along one direction."""
    g = np.array([geodesic_angle(
        propagate_jacobi_path2(q_a, sv * ndir, INERTIA, np.array([0.0, dt]))[0][-1], q_b)
        for sv in s_grid])
    k = int(np.argmin(g))
    return float(g[k]), float(s_grid[k])


def offset_dirs(true_dir, delta_deg, n_az=8):
    """n_az directions at angular offset delta_deg from true_dir (ring of azimuths)."""
    n = true_dir / np.linalg.norm(true_dir)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    d = np.radians(delta_deg)
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    return [np.cos(d) * n + np.sin(d) * (np.cos(a_) * e1 + np.sin(a_) * e2) for a_ in az]


def angle_between(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), 0, 1))))


def run_case(seed, ep_a, ep_b):
    d = tl.load_truth(seed)
    t = d["observation_times"].astype(float); t -= t[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    if ep_a is None:
        inv = np.load(SURVEY / "results" / "s099" / "invert.npz", allow_pickle=True) \
            if seed == 116 else None
        ep_a = int(inv["ep_a"]) if inv is not None else 0
        # fall back: use s100-style anchors saved for 116 if present
        cand = SURVEY / "results" / "s100" / f"seed{seed:03d}" / "invert.json"
        if cand.exists():
            j = json.load(open(cand)); ep_a, ep_b = int(j["ep_a"]), int(j["ep_b"])
        else:
            ep_b = ep_a + 100
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, t)
    q_a, q_b = qh[ep_a], qh[ep_b]
    w_a = wh[ep_a]
    dt = float(t[ep_b] - t[ep_a])
    true_mag = float(np.linalg.norm(w_a)); true_dir = w_a / true_mag
    turns = true_mag * dt / (2 * np.pi)
    s_grid = np.linspace(S_LO, S_HI, N_SCAN)

    print(f"\n===== seed {seed} | A=ep{ep_a} B=ep{ep_b} dt={dt:.0f}s | "
          f"truth |w_a|={true_mag*R2D:.4f} deg/s | turns/baseline={turns:.2f} =====", flush=True)

    # (1) offset scan
    print("[1] OFFSET SCAN (best return depth vs direction offset from truth):", flush=True)
    offsets = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45]
    off_rows = []
    for dd in offsets:
        dirs = offset_dirs(true_dir, dd) if dd > 0 else [true_dir]
        depths = [best_return_for_dir(q_a, q_b, dt, dv, s_grid)[0] for dv in dirs]
        best = min(depths); worst = max(depths)
        off_rows.append(dict(offset_deg=dd, best_depth=best, worst_depth=worst))
        print(f"    offset {dd:3d} deg -> return depth: best {best:6.2f} deg  worst {worst:6.2f} deg", flush=True)

    # (2) blind fibonacci sweep
    print("[2] BLIND FIBONACCI SWEEP (deepest return, offset of winner from truth):", flush=True)
    fib_rows = []
    for nd in (20, 50, 100, 200, 500, 1000):
        dirs = fib_sphere(nd)
        depths = np.array([best_return_for_dir(q_a, q_b, dt, dv, s_grid)[0] for dv in dirs])
        k = int(np.argmin(depths))
        win_off = angle_between(dirs[k], true_dir)
        fib_rows.append(dict(n_dir=nd, deepest=float(depths[k]), winner_offset_deg=win_off,
                             n_under_5deg=int((depths < 5).sum())))
        print(f"    N_dir {nd:4d} -> deepest {depths[k]:6.2f} deg @ offset {win_off:5.1f} deg "
              f"from truth | dirs<5deg-return: {int((depths<5).sum())}", flush=True)

    return dict(seed=seed, ep_a=ep_a, ep_b=ep_b, dt=dt, true_mag_dps=true_mag * R2D,
                turns=turns, offset_scan=off_rows, fib_sweep=fib_rows)


def main():
    out = SURVEY / "results" / "s105"; out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed, ea, eb in CASES:
        try:
            results.append(run_case(seed, ea, eb))
        except Exception as e:
            print(f"seed {seed} skipped: {e}", flush=True)
    with open(out / "dir_basin_width.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out/'dir_basin_width.json'}", flush=True)


if __name__ == "__main__":
    main()
