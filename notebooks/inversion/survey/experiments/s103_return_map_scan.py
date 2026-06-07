"""s103 — propagator return-map: build |w| winding candidates WITHOUT the
constant-w 2*pi ladder. For a fixed direction, sweep |w| through the physical
bracket with the closed-form propagator and find where the torque-free
trajectory returns near q_b (local minima of geodesic-to-q_b). Demonstrates on
seed 119's truth pair that (a) the minima sit on truth, (b) they are NOT
2*pi/dt-spaced, (c) a wrong (finite-diff) direction gives different/shallower
minima. All rates deg/s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelmin

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia, finite_diff_omega, geodesic_angle
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
SEED = int(os.environ.get("S103_SEED", 119))
EP_A = int(os.environ.get("S103_EP_A", 69))
EP_B = int(os.environ.get("S103_EP_B", 172))
INERTIA = m048_inertia()
N_SCAN = 600
S_LO, S_HI = np.radians(0.1), np.radians(1.6)


def scan(q_a, q_b, dt, ndir):
    s = np.linspace(S_LO, S_HI, N_SCAN)
    g = np.array([geodesic_angle(propagate_jacobi_path2(q_a, sv * ndir, INERTIA,
                                                        np.array([0.0, dt]))[0][-1], q_b)
                  for sv in s])
    mins = argrelmin(g, order=3)[0]
    mins = mins[g[mins] < np.radians(20)]            # keep genuine returns (<20 deg)
    return s, g, mins


def main():
    d = tl.load_truth(SEED)
    t = d["observation_times"].astype(float); t -= t[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, t)
    q_a, q_b = qh[EP_A], qh[EP_B]
    w_a = wh[EP_A]
    dt = float(t[EP_B] - t[EP_A])
    truth_mag = float(np.linalg.norm(w_a))
    true_dir = w_a / truth_mag
    fd = finite_diff_omega(q_a, q_b, dt); fd_dir = fd / np.linalg.norm(fd)

    print(f"=== s103 return-map | seed {SEED} A=ep{EP_A} B=ep{EP_B} dt={dt:.0f}s ===")
    print(f"truth |w_a| = {truth_mag*R2D:.4f} deg/s | finite-diff |w| = {np.linalg.norm(fd)*R2D:.4f} deg/s")
    print(f"finite-diff DIR err vs truth = {np.degrees(np.arccos(np.clip(abs(fd_dir@true_dir),0,1)))*2:.1f} deg")
    print(f"constant-w spacing 2pi/dt = {(2*np.pi/dt)*R2D:.4f} deg/s\n")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for dirv, name, col in ((true_dir, "TRUE w direction", "tab:green"),
                            (fd_dir, "finite-diff direction", "tab:red")):
        s, g, mins = scan(q_a, q_b, dt, dirv)
        ax.plot(s * R2D, g * R2D, color=col, lw=1.3, label=name)
        ax.plot(s[mins] * R2D, g[mins] * R2D, "o", color=col, ms=6)
        mm = (s[mins] * R2D).tolist()
        depths = (g[mins] * R2D).tolist()
        print(f"{name:22s}: return minima at |w| = {[round(x,3) for x in mm]} deg/s")
        print(f"{'':22s}  geodesic DEPTH at each = {[round(x,2) for x in depths]} deg  (0 = exact return)")
        if len(mm) > 1:
            print(f"{'':22s}  spacings = {[round(mm[i+1]-mm[i],3) for i in range(len(mm)-1)]} deg/s")
    ax.axvline(truth_mag * R2D, color="k", ls="--", lw=1, label=f"truth |w|={truth_mag*R2D:.3f}")
    ax.set_xlabel("|w| (deg/s)"); ax.set_ylabel("geodesic( propagate(q_a,|w|*dir,dt), q_b )  (deg)")
    ax.set_title(f"s103 propagator return-map vs |w| (seed {SEED} truth pair)")
    ax.legend(fontsize=8); ax.set_ylim(0, 60)
    fig.tight_layout()
    out = SURVEY / "results" / "s103"; out.mkdir(parents=True, exist_ok=True)
    p = out / f"return_map_seed{SEED:03d}.png"; fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
