#!/usr/bin/env python3
"""Step 1: Brightness filter on anchor candidates.

For each of 720 candidate attitudes at anchor 1, evaluate single-epoch
hi-fi brightness and compare with observed magnitude. This filters out
candidates that don't produce the right brightness BEFORE bridging.

Question: how many of 720 survive? Does the correct attitude survive?

Output: m059_step1_brightness_traj19.npz
Runtime: ~2 min (720 × hi-fi single-epoch ~47ms each ≈ 34s, but shadows
are slow for single-epoch; let's see)
"""
import sys, os, time
import numpy as np
from pathlib import Path
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

from lib.experiment_setup import (setup_experiment, attitude_error_deg,
                                   brightness_single_epoch)
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# Worker for parallel brightness eval
_CTX_g = None
_epoch_g = None
_shadows_g = None


def _eval_brightness(q_wxyz):
    try:
        return brightness_single_epoch(q_wxyz, _epoch_g, _CTX_g,
                                        use_shadows=_shadows_g)
    except Exception:
        return 20.0


# =========================================================================
print("=" * 60, flush=True)
print("Step 1: Brightness filter on 720 anchor candidates", flush=True)
print("=" * 60, flush=True)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0_true = master['q0s'][19]
omega_true = master['omega0s'][19]
mags = master['mag_hifi'][19]
n_normals = len(normals)

# Peak detection
peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
bright = peaks[mags[peaks] < 9.0]
sorted_by_mag = bright[np.argsort(mags[bright])]
a1 = int(sorted_by_mag[0])
a2 = int(sorted_by_mag[1])
observed_mag_a1 = mags[a1]
observed_mag_a2 = mags[a2]
print(f"Anchor 1: epoch {a1}, observed mag {observed_mag_a1:.2f}", flush=True)
print(f"Anchor 2: epoch {a2}, observed mag {observed_mag_a2:.2f}", flush=True)

# Build 720 candidates at each anchor
N_PHI = 72
phi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

c1_all = []
c1_hyp = []
c1_phi = []
for h in range(n_normals):
    for pi, phi in enumerate(phi_vals):
        c1_all.append(anchor_q_from_phi(phi, normals[h], pab_j2000[a1]))
        c1_hyp.append(h)
        c1_phi.append(phi)
c1_all = np.array(c1_all)

c2_all = []
c2_hyp = []
for h in range(n_normals):
    for pi, phi in enumerate(phi_vals):
        c2_all.append(anchor_q_from_phi(phi, normals[h], pab_j2000[a2]))
        c2_hyp.append(h)
c2_all = np.array(c2_all)

# Propagate truth to anchors for attitude error comparison
prop_t1 = np.array([0.0, obs_times[a1]])
qt1, _ = propagate_attitude(q0_true, omega_true, prop_t1, 'tumbling', I_tensor)
q_true_a1 = qt1[-1]

prop_t2 = np.array([0.0, obs_times[a2]])
qt2, _ = propagate_attitude(q0_true, omega_true, prop_t2, 'tumbling', I_tensor)
q_true_a2 = qt2[-1]

att_errors_a1 = np.array([attitude_error_deg(c1_all[i], q_true_a1)
                           for i in range(len(c1_all))])
att_errors_a2 = np.array([attitude_error_deg(c2_all[i], q_true_a2)
                           for i in range(len(c2_all))])

print(f"\nBest attitude error at anchor 1: {att_errors_a1.min():.1f}° "
      f"(G{c1_hyp[np.argmin(att_errors_a1)]})", flush=True)
print(f"Best attitude error at anchor 2: {att_errors_a2.min():.1f}° "
      f"(G{c2_hyp[np.argmin(att_errors_a2)]})", flush=True)

# ===== Evaluate brightness at BOTH anchors =====
for anchor_label, candidates, epoch, observed_mag, att_errors in [
    ("Anchor 1", c1_all, a1, observed_mag_a1, att_errors_a1),
    ("Anchor 2", c2_all, a2, observed_mag_a2, att_errors_a2),
]:
    print(f"\n--- {anchor_label} (epoch {epoch}, obs={observed_mag:.2f}) ---",
          flush=True)

    # Lo-fi first (fast)
    _CTX_g = CTX
    _epoch_g = epoch
    _shadows_g = False

    t_lo = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(8) as pool:
        mags_lofi = pool.map(_eval_brightness, list(candidates))
    mags_lofi = np.array(mags_lofi)
    dt_lo = time.time() - t_lo
    print(f"  Lo-fi: {dt_lo:.1f}s", flush=True)

    # Hi-fi (slower but accurate)
    _shadows_g = True
    t_hi = time.time()
    with ctx.Pool(8) as pool:
        mags_hifi = pool.map(_eval_brightness, list(candidates))
    mags_hifi = np.array(mags_hifi)
    dt_hi = time.time() - t_hi
    print(f"  Hi-fi: {dt_hi:.1f}s", flush=True)

    # Save per-anchor checkpoint
    anchor_num = 1 if "1" in anchor_label else 2
    np.savez(str(RESULTS_DIR / f"m059_step1_brightness_a{anchor_num}_traj19.npz"),
             candidates=candidates,
             mags_lofi=mags_lofi,
             mags_hifi=mags_hifi,
             att_errors=att_errors,
             observed_mag=observed_mag,
             epoch=epoch)

    # Report
    print(f"  Predicted mag range: lo-fi [{mags_lofi.min():.2f}, {mags_lofi.max():.2f}], "
          f"hi-fi [{mags_hifi.min():.2f}, {mags_hifi.max():.2f}]", flush=True)

    for tol in [0.5, 1.0, 1.5, 2.0, 3.0]:
        match_lo = np.abs(mags_lofi - observed_mag) < tol
        match_hi = np.abs(mags_hifi - observed_mag) < tol
        n_lo = np.sum(match_lo)
        n_hi = np.sum(match_hi)
        n_correct_lo = np.sum(att_errors[match_lo] < 10) if n_lo > 0 else 0
        n_anti_lo = np.sum(att_errors[match_lo] > 170) if n_lo > 0 else 0
        n_correct_hi = np.sum(att_errors[match_hi] < 10) if n_hi > 0 else 0
        n_anti_hi = np.sum(att_errors[match_hi] > 170) if n_hi > 0 else 0
        print(f"  |Δmag| < {tol}: lo-fi {n_lo:3d}/720 "
              f"({n_correct_lo} correct, {n_anti_lo} anti) | "
              f"hi-fi {n_hi:3d}/720 "
              f"({n_correct_hi} correct, {n_anti_hi} anti)", flush=True)

    # Where does the best attitude rank by brightness match?
    best_idx = np.argmin(att_errors)
    mag_err_best = abs(mags_hifi[best_idx] - observed_mag)
    print(f"  Best attitude ({att_errors[best_idx]:.1f}°): "
          f"hi-fi mag={mags_hifi[best_idx]:.2f}, |Δ|={mag_err_best:.2f}",
          flush=True)

    # Per-normal summary
    print(f"  Per-normal (hi-fi, best match):", flush=True)
    for h in range(n_normals):
        mask = np.array([i for i in range(len(candidates))
                        if h == (i // N_PHI)])
        if len(mask) == 0:
            continue
        best_match_idx = mask[np.argmin(np.abs(mags_hifi[mask] - observed_mag))]
        dm = abs(mags_hifi[best_match_idx] - observed_mag)
        ae = att_errors[best_match_idx]
        marker = " ← correct" if ae < 10 else (" ← anti" if ae > 170 else "")
        print(f"    G{h:2d}: best |Δmag|={dm:.2f}, "
              f"att_err={ae:.1f}°{marker}", flush=True)

print(f"\nTotal: {time.time() - t_global:.0f}s", flush=True)
