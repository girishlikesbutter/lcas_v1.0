"""s109 — |omega(t)| envelope over each trajectory vs the bracket clamp bounds.

The cohort is generated with |omega(0)| in [0.1, 1.5] deg/s, but |omega| is NOT
conserved under torque-free Euler dynamics (only |L| and 2T are) — it oscillates
along the polhode. The s100 |w| bracket FILTER is applied to the omega at the
ANCHOR epoch (the shoot returns omega at A), not at t=0. So if a blind anchor
lands in a polhode trough where |omega(t_a)| < clamp_floor, the TRUE anchor omega
is rejected and truth is unresolvable. Symmetrically, |omega| can exceed the 1.5
generation cap at polhode peaks -> the ceiling must sit above 1.5.

This measures, per seed, min/max |omega(t)| over the OBSERVATION epochs (the set
the anchor is actually chosen from) and aggregates against candidate clamp bounds.
All rates deg/s. Closed-form propagation only — no surrogate, no render.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
OUT = SURVEY / "results" / "s109"
OUT.mkdir(parents=True, exist_ok=True)


def envelope(seed):
    try:
        d = tl.load_truth(seed)
    except Exception:
        return None
    t = d["observation_times"].astype(float); t -= t[0]
    q0 = d["q0_wxyz"].astype(float); w0 = d["omega0_rad"].astype(float)
    _, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, t)
    wmag = np.linalg.norm(w_hist, axis=1) * R2D     # |omega(t)| deg/s at each obs epoch
    return dict(seed=seed, w0=float(np.linalg.norm(w0) * R2D),
                wmin=float(wmag.min()), wmax=float(wmag.max()),
                swing=float(wmag.max() / wmag.min()))


def main():
    rows = [r for s in range(120) if (r := envelope(s)) is not None]
    wmin = np.array([r["wmin"] for r in rows])
    wmax = np.array([r["wmax"] for r in rows])
    w0 = np.array([r["w0"] for r in rows])

    print(f"=== s109 |omega(t)| trajectory envelope | {len(rows)} seeds ===\n")
    print(f"|omega(0)| (generation):  min {w0.min():.3f}  max {w0.max():.3f} deg/s")
    print(f"min_t |omega(t)|       :  GLOBAL MIN {wmin.min():.4f} deg/s "
          f"(seed {rows[int(np.argmin(wmin))]['seed']})")
    print(f"max_t |omega(t)|       :  GLOBAL MAX {wmax.max():.4f} deg/s "
          f"(seed {rows[int(np.argmax(wmax))]['seed']})\n")

    print("--- FLOOR: seeds whose |omega(t)| dips below a candidate floor (anchor could land there) ---")
    for fl in (0.10, 0.05, 0.04, 0.03):
        n = int((wmin < fl).sum())
        print(f"  floor {fl:.2f} deg/s: {n:3d}/{len(rows)} seeds dip below it "
              f"-> truth unresolvable if anchor lands in trough")
    print(f"  => a floor <= {wmin.min():.4f} deg/s clips NO seed at any anchor")

    print("\n--- CEILING: seeds whose |omega(t)| rises above a candidate ceiling ---")
    for ce in (1.50, 1.60, 1.70):
        n = int((wmax > ce).sum())
        print(f"  ceiling {ce:.2f} deg/s: {n:3d}/{len(rows)} seeds exceed it")
    print(f"  => a ceiling >= {wmax.max():.4f} deg/s clips NO seed at any anchor")

    # detail on the slow seeds (where the floor matters)
    order = np.argsort(w0)
    print("\n--- SLOWEST 10 (floor-relevant): how deep does the polhode trough go? ---")
    print(f"{'seed':>4} {'|w0|':>6} {'wmin':>7} {'wmax':>7} {'swing':>6} {'<0.10?':>6} {'<0.05?':>6}")
    for i in order[:10]:
        r = rows[i]
        print(f"{r['seed']:>4} {r['w0']:>6.3f} {r['wmin']:>7.4f} {r['wmax']:>7.4f} "
              f"{r['swing']:>6.2f} {str(r['wmin'] < 0.10):>6} {str(r['wmin'] < 0.05):>6}")

    print("\n--- FASTEST 6 (ceiling-relevant): how high does the polhode peak go? ---")
    for i in order[::-1][:6]:
        r = rows[i]
        print(f"  seed {r['seed']:>3}: |w0| {r['w0']:.3f}  wmax {r['wmax']:.4f} deg/s "
              f"(over 1.5: {r['wmax'] > 1.5}, over 1.6: {r['wmax'] > 1.6})")

    with open(OUT / "omega_envelope.json", "w") as f:
        json.dump(dict(n=len(rows), global_wmin=float(wmin.min()),
                       global_wmax=float(wmax.max()), rows=rows), f, indent=2)
    print(f"\nSaved: {OUT/'omega_envelope.json'}")


if __name__ == "__main__":
    main()
