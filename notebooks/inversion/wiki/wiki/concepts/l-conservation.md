---
title: "Angular Momentum Conservation"
type: concept
sources:
  - "raw/inversion_diagnostics/"
related:
  - "[[bridge-solver]]"
created: 2026-03-10
updated: 2026-04-09
confidence: high
---

# Angular Momentum Conservation

For torque-free rigid body motion, the angular momentum vector in the inertial frame is conserved:

```
L_inertial = R(q) @ (I @ omega_body) = const
```

where R(q) is the rotation matrix from body to inertial frame, I is the inertia tensor, and omega_body is the angular velocity in body coordinates.

## As a Constraint

At a shared glint (peak node), two trajectory segments must have the same L_inertial. This provides a powerful discriminator:

- **||DL|| ranks true pair #1 out of 64**, with a gap of 113 kg*m^2/s (9 orders of magnitude above noise)
- Robust to 10 deg endpoint attitude error: still 100% correct, gap = 95
- Shared-node attitude error is **mathematically invariant** (R cancels in the difference)

## What Doesn't Help

- **Kinetic energy T** adds nothing beyond L. Since T = 0.5 * omega . L, it is algebraically dependent on L for torque-free motion.

## Role in Pipeline

L-conservation is the primary ranking criterion for the [[bridge-solver]] when connecting trajectory segments. It provides a physics-based filter that is far more reliable than LC-based scoring for segment matching.
