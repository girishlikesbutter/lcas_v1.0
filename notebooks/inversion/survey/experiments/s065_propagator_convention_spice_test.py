"""s065 — Propagator-vs-SPICE convention test.

Definitive test of the propagator's rotation convention against the SPICE
reference frame `IS901_BUS_FRAME` over a short propagation interval (1–60 s).

Conclusion (see s065_propagator_convention_spice_test.md and
concepts/quaternion_convention.md):

  - q-convention is correct: `scipy.from_quat(q[xyzw]).as_matrix()` agrees
    with SPICE's `pxform('J2000', body, et)` at t=0 to 3e-16. The cached
    `q0_wxyz` in every m048 trajectory IS the physical initial attitude.

  - omega-sign is FLIPPED relative to physics: propagating from a SPICE
    q0 with the textbook-finite-difference body-frame omega gives an
    R that drifts at ~2e-4 per second from SPICE. Negating omega closes
    that to 9.7e-11 per second — machine precision.

This is a convention bookkeeping choice, not a physics bug. The cached
m048 LCs are physically valid LCs of satellites whose PHYSICAL body-frame
angular velocity is `-omega0_rad`. STL geometry, BRDF, surrogate, and
inversion comparisons are all internally consistent under this convention.

Run: `python experiments/s065_propagator_convention_spice_test.py`
"""
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.spatial.transform import Rotation

from src.dynamics.attitude_propagator import propagate_euler
from src.spice.spice_handler import SpiceHandler


def main():
    metakernel = Path("data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm")
    sh = SpiceHandler()
    sh.load_metakernel_programmatically(str(metakernel))

    t0 = sh.utc_to_et("2020-02-05T10:00:00")
    dt_for_omega = 0.01  # s, for finite-difference omega from SPICE
    frame_from, frame_to = "J2000", "IS901_BUS_FRAME"

    # SPICE reference at t0
    R0 = np.asarray(spice.pxform(frame_from, frame_to, t0))
    R0_plus = np.asarray(spice.pxform(frame_from, frame_to, t0 + dt_for_omega))

    # Seed propagator from SPICE: q such that scipy.as_matrix(q) == R0 (codebase
    # convention). This is the q the propagator should produce at t0.
    q_xyzw = Rotation.from_matrix(R0).as_quat()
    q0_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # Textbook body-frame omega via dR/dt = -[omega]_x R for R: J2000->body passive
    dR_dt = (R0_plus - R0) / dt_for_omega
    om_skew = -dR_dt @ R0.T
    omega_textbook = np.array([om_skew[2, 1], om_skew[0, 2], om_skew[1, 0]])

    print(f"t0 (ET):                 {t0:.3f}")
    print(f"|R0_pxform_det - 1|:     {abs(np.linalg.det(R0) - 1):.2e}")
    print(f"q0 from R0 (wxyz):       {q0_wxyz}")
    print(f"omega_textbook (rad/s):  {omega_textbook}   |w|={np.linalg.norm(omega_textbook):.3e}")

    I_test = np.eye(3) * 1000.0  # identity inertia isolates the q-kinematic
    test_dts = np.array([0.0, 1.0, 10.0, 60.0])

    def run(omega_in, label):
        q_hist, _ = propagate_euler(q0_wxyz, omega_in, I_test, test_dts,
                                    rtol=1e-12, atol=1e-14)
        print(f"\n--- {label} ---")
        print(f"     dt      ||R_prop - R_SPICE||")
        for i, dt in enumerate(test_dts):
            R_ref = np.asarray(spice.pxform(frame_from, frame_to, t0 + dt))
            q = q_hist[i]
            R_prop = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            d = np.linalg.norm(R_prop - R_ref)
            print(f"  {dt:6.1f} s       {d:.3e}")

    run(+omega_textbook, "feeding propagator +omega_textbook (current convention)")
    run(-omega_textbook, "feeding propagator -omega_textbook (matches SPICE)")


if __name__ == "__main__":
    main()
