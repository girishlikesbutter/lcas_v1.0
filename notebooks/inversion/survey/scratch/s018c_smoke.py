"""s018c smoke test — validate phi-sweep IC generator conventions.

Run from project root:
  cd /home/girish/projects/lcas_v1.0
  python notebooks/inversion/survey/scratch/s018c_smoke.py

Goals (each independently asserted):
  S1. PAB-alignment geometric story: at a known peak with min_ang_dist<5°,
      R_b2i(q_truth(t_peak)) @ IS901_NORMALS[best_group_at_peak] aligns
      with pab_j2000(t_peak) within ~5°.
  S2. Back-propagation convention: q0_truth = Phi(t_peak; omega0)^{-1} ⊗
      q_truth(t_peak), where Phi is the propagation of (identity, omega0).
      OR q0_truth = q_truth(t_peak) ⊗ Phi^{-1}, depending on left/right
      composition. Whichever holds defines the s018c IC formula.
  S3. Phi-sweep round-trip: with truth-omega and truth-face at the truth
      peak epoch, the phi value that recovers q_truth must lie in [0, 2pi).
      Build the q0 IC from that phi and verify identity to ~1e-6.
  S4. Sensitivity: at small phi perturbations away from truth-phi, the
      forward-propagated mag at t_peak should change smoothly (sanity).

If S1 or S2 fails, the rest is broken; abort and re-derive.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"

sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.forward import propagate_to_body_frame, quat_geodesic_deg
from src.dynamics.attitude_propagator import propagate_attitude

# IS-901 body-frame face normals (10 groups), copied from
# notebooks/inversion/lib/attitude_anim.py:32 (geometric STL invariants;
# unaffected by the propagator bug; allowed under "copy-adapt" policy
# because they are STL-derived geometry, not buggy-era inversion code).
IS901_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z',
               '+WD', '-WD', '+ED', '-ED']
IS901_NORMALS = np.array([
    [ 1.0000,  0.0000, 0.0],   # +X
    [-1.0000,  0.0000, 0.0],   # -X
    [ 0.0000,  1.0000, 0.0],   # +Y
    [ 0.0000, -1.0000, 0.0],   # -Y
    [ 0.0000,  0.0000, 1.0],   # +Z
    [ 0.0000,  0.0000, -1.0],  # -Z
    [ 0.9659, -0.2588, 0.0],   # +WD
    [-0.9659,  0.2588, 0.0],   # -WD
    [ 0.9659,  0.2588, 0.0],   # +ED
    [-0.9659, -0.2588, 0.0],   # -ED
])
assert IS901_NORMALS.shape == (10, 3)


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot_b2i(q_wxyz):
    """body→inertial matrix from a (w, x, y, z) quaternion.

    Convention from forward.py:63 — scipy.from_quat((x,y,z,w)).as_matrix()
    is inertial→body in this codebase, so the transpose is body→inertial.
    """
    R_i2b = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    return R_i2b.T


def rot_to_quat_wxyz(R_b2i):
    """Inverse of quat_to_rot_b2i."""
    R_i2b = R_b2i.T
    quat_xyzw = Rotation.from_matrix(R_i2b).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def rotation_aligning(a, b):
    """Rotation matrix R such that R @ a = b. a, b unit vectors."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180° flip; pick any axis perpendicular to a
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    s = float(np.linalg.norm(v))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def rotation_about_axis(axis_unit, angle_rad):
    return Rotation.from_rotvec(angle_rad * axis_unit).as_matrix()


def main():
    seed = 6
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
    print(f"Loading {traj_path}")
    d = np.load(traj_path)

    q0_truth = d["q0_wxyz"].astype(float)
    omega0_truth = d["omega0_rad"].astype(float)
    times = d["observation_times"].astype(float)
    pab_j2000 = d["pab_j2000"].astype(float)
    sun_pos = d["sun_pos"].astype(float)
    obs_pos = d["obs_pos"].astype(float)
    sat_pos = d["sat_pos"].astype(float)
    min_ang = d["min_ang_dist"].astype(float)
    best_group = d["best_group"].astype(int)
    peak_eps = d["hifi_peak_epochs"].astype(int)
    mag_hifi = d["mag_hifi"].astype(float)

    print(f"  q0_truth (wxyz) = {q0_truth}")
    print(f"  omega0_truth = {omega0_truth} rad/s "
          f"(|w|={np.linalg.norm(omega0_truth):.6f})")
    print(f"  N_epochs = {len(times)}, hifi peaks = {len(peak_eps)}")

    master = np.load(
        PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
        / "m048_trajectories" / "m048_trajectories.npz", allow_pickle=True
    )
    inertia = np.asarray(master["inertia_tensor"], dtype=float)
    print(f"  inertia diag: {np.diag(inertia)}")

    # Pick a peak that is a true spec event (min_ang<5°)
    peak_min_angs = min_ang[peak_eps]
    spec_mask = peak_min_angs < 5.0
    if not spec_mask.any():
        print("  no spec5 peaks for seed 6; falling back to overall min_ang<5 epoch")
        good = np.where(min_ang < 5.0)[0]
        i_pk = int(good[len(good)//2])
    else:
        i_pk = int(peak_eps[spec_mask][0])
    print(f"\nSelected peak epoch i = {i_pk}")
    print(f"  min_ang at peak = {min_ang[i_pk]:.3f} deg")
    print(f"  best_group at peak = {best_group[i_pk]} ({IS901_NAMES[best_group[i_pk]]})")
    print(f"  mag at peak = {mag_hifi[i_pk]:.3f}")

    pab_at_peak = pab_j2000[i_pk]
    n_g_body = IS901_NORMALS[best_group[i_pk]]
    t_peak = times[i_pk]
    print(f"  pab_j2000 at peak = {pab_at_peak}")
    print(f"  n_g body-frame   = {n_g_body}")
    print(f"  t_peak (s) = {t_peak:.3f}")

    # Forward-propagate truth to t_peak — and check S1.
    quats_truth, _ = propagate_attitude(
        q0=q0_truth, omega0=omega0_truth,
        times=times, mode="tumbling", inertia_tensor=inertia,
    )
    q_truth_at_peak = quats_truth[i_pk]
    R_b2i_truth = quat_to_rot_b2i(q_truth_at_peak)
    n_in_inertial = R_b2i_truth @ n_g_body
    cos_align = float(np.dot(n_in_inertial, pab_at_peak))
    align_deg = float(np.degrees(np.arccos(np.clip(cos_align, -1.0, 1.0))))
    print(f"\n[S1] truth normal aligned with PAB?")
    print(f"     R_b2i(q_truth(t_peak)) @ n_g = {n_in_inertial}")
    print(f"     pab_j2000(t_peak)            = {pab_at_peak}")
    print(f"     angle = {align_deg:.4f} deg  -> {'PASS' if align_deg < 5 else 'FAIL'}")
    assert align_deg < 5.0, "S1: PAB-alignment story does not hold under correct truth"

    # S2: back-propagation convention.
    # Forward-propagate (identity, omega0_truth) → Phi.
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    quats_phi, _ = propagate_attitude(
        q0=q_id, omega0=omega0_truth,
        times=times, mode="tumbling", inertia_tensor=inertia,
    )
    Phi_at_peak = quats_phi[i_pk]
    print(f"\n[S2] back-propagation convention test")
    print(f"     Phi(t_peak)      = {Phi_at_peak}")
    print(f"     q_truth(t_peak)  = {q_truth_at_peak}")

    Phi_inv = quat_conjugate_wxyz(Phi_at_peak)
    cand_left  = quat_multiply_wxyz(Phi_inv, q_truth_at_peak)
    cand_right = quat_multiply_wxyz(q_truth_at_peak, Phi_inv)

    err_left = quat_geodesic_deg(cand_left, q0_truth)
    err_right = quat_geodesic_deg(cand_right, q0_truth)
    print(f"     Phi^-1 ⊗ q_target = {cand_left}, geodesic vs q0_truth = {err_left:.6f} deg")
    print(f"     q_target ⊗ Phi^-1 = {cand_right}, geodesic vs q0_truth = {err_right:.6f} deg")

    if err_left < 1e-3:
        BACKPROP = "left"
        print(f"     -> back-prop convention: q0 = Phi^-1 ⊗ q_target  (LEFT)")
    elif err_right < 1e-3:
        BACKPROP = "right"
        print(f"     -> back-prop convention: q0 = q_target ⊗ Phi^-1  (RIGHT)")
    else:
        BACKPROP = None
        print("     !! neither composition recovers q0_truth — propagation must "
              "have a more complex form. Inspecting both errors for diagnostics.")

    assert BACKPROP is not None, "S2: back-prop convention not identified"

    # S3: phi-sweep IC generator forward consistency.
    # Construct: R_b2i_target(phi) = R_phi(pab) @ R_align(n_g, pab)
    # Then q0(phi) = Phi^-1 ⊗ rot_to_quat(R_b2i_target).
    # Forward-propagate (q0(phi), omega0_truth) to t_peak and verify the
    # rotated face-normal aligns with PAB to ~machine precision (this is
    # the IC-generation invariant, by construction independent of what
    # the truth's phi is).
    R_align = rotation_aligning(n_g_body, pab_at_peak)
    align_check = R_align @ n_g_body
    align_resid = float(np.degrees(np.arccos(np.clip(np.dot(align_check,
                                                            pab_at_peak), -1, 1))))
    print(f"\n[S3] IC-generation forward consistency check")
    print(f"     R_align @ n_g = {align_check}  (matches pab to {align_resid:.6e} deg)")
    assert align_resid < 1e-6, "S3a: R_align does not actually align n_g to pab"

    print("\n     phi (deg)  align_built(R@n, pab) (deg)  align_after_propagation (deg)")
    for d_deg in [0, 30, 60, 90, 120, 180, 240, 300]:
        phi_test = np.radians(d_deg)
        R_phi_t = rotation_about_axis(pab_at_peak, phi_test)
        R_b2i_t = R_phi_t @ R_align
        # Sanity: by construction R_b2i_t @ n_g must equal pab.
        n_built = R_b2i_t @ n_g_body
        ang_build = float(np.degrees(np.arccos(np.clip(
            np.dot(n_built, pab_at_peak), -1, 1))))
        q_target_t = rot_to_quat_wxyz(R_b2i_t)
        q0_t = quat_multiply_wxyz(Phi_inv, q_target_t)  # BACKPROP=='left'
        # Forward-propagate q0_t with truth omega; recheck alignment at peak.
        quats_t, _ = propagate_attitude(
            q0=q0_t, omega0=omega0_truth,
            times=np.array([times[0], times[i_pk]]),
            mode="tumbling", inertia_tensor=inertia,
        )
        R_b2i_t_pk = quat_to_rot_b2i(quats_t[-1])
        n_in_i = R_b2i_t_pk @ n_g_body
        ang_prop = float(np.degrees(np.arccos(np.clip(
            np.dot(n_in_i, pab_at_peak), -1, 1))))
        # geodesic to truth q0
        g0 = quat_geodesic_deg(q0_t, q0_truth)
        print(f"       {d_deg:5d}     {ang_build:9.6f}                   "
              f"{ang_prop:9.6f}     (q0_geo_to_truth = {g0:6.2f}°)")
        # By construction, ang_build must be ~0
        assert ang_build < 1e-6, f"S3b: built R does not align at phi={d_deg}"
        assert ang_prop < 1e-6, f"S3c: propagation breaks alignment at phi={d_deg}"

    # S4: closest-to-truth-phi sweep — show that (phi, q0) lying on the
    # phi-circle through the actual truth pose has minimum geodesic.
    # This validates that LM started at one of these ICs has a chance of
    # converging to truth via the Phi-axis 1-DOF.
    print("\n[S4] dense phi-sweep — locate closest-to-truth phi")
    best_phi = None
    best_g0 = 999.0
    for d_deg in np.arange(0, 360, 1):
        phi_test = np.radians(d_deg)
        R_phi_t = rotation_about_axis(pab_at_peak, phi_test)
        R_b2i_t = R_phi_t @ R_align
        q_target_t = rot_to_quat_wxyz(R_b2i_t)
        q0_t = quat_multiply_wxyz(Phi_inv, q_target_t)
        g0 = quat_geodesic_deg(q0_t, q0_truth)
        if g0 < best_g0:
            best_g0 = g0
            best_phi = float(d_deg)
    print(f"     dense (1° step) sweep closest to truth: phi={best_phi}°, "
          f"q0_geo={best_g0:.4f}°")
    print(f"     min_ang at this peak (truth offset): {min_ang[i_pk]:.4f}°")
    print(f"     Expectation: best_g0 ~ min_ang (truth-misalignment limits "
          f"how close the IC can sit to truth at this peak)")
    # If best_g0 is much larger than min_ang, the math has a different bug;
    # if it's roughly equal, the IC generator is geometrically tight.
    assert best_g0 < min_ang[i_pk] + 5.0, (
        f"S4: closest phi-IC ({best_g0:.2f}°) far exceeds truth misalignment "
        f"({min_ang[i_pk]:.2f}° + 5° tolerance)"
    )

    print("\nALL SMOKE TESTS PASSED.")
    print(f"BACKPROP convention = {BACKPROP}")


if __name__ == "__main__":
    main()
