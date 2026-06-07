# Angular Momentum Conservation Constraint for Multi-Peak Inversion

**Date:** 2026-03-02
**Origin:** Girish + Bast discussion, pre-Jack meeting

## Core Insight

For a rigid body with no external torques, angular momentum L is conserved in the inertial frame. This means:

Given candidate quaternions (q₁, q₂, q₃) at peaks 1, 2, 3 and a choice of ω₁ at peak 1:

1. Compute L_inertial = q₁ ⊗ (I_body · ω₁_body) ⊗ q₁*
2. L is constant — so at any other peak k:
   - ω_k_body = I_body⁻¹ · (q_k* ⊗ L_inertial ⊗ q_k)
3. Only need to search ω at ONE epoch — all others are determined by conservation

## Consistency Check

Having the same L doesn't guarantee dynamic connectivity. The test is:
- Propagate Euler equations from (q₁, ω₁) forward
- Check if propagated state arrives at (q₂, ω₂) at time t₂
- Check if it continues to (q₃, ω₃) at time t₃

Multi-peak consistency is a brutally strong filter — getting L to work across 3+ independent peak candidates simultaneously should annihilate false positives.

## Parameterisation Options

**Option A: Search over ω₁_body (3 DOF)**
- Natural starting point, body-frame angular velocity at peak 1

**Option B: Search over L_inertial (3 DOF)**
- L = (Lx, Ly, Lz) inertial — the actual conserved quantity
- Conceptually appealing but mathematically equivalent to Option A (related by linear transform for fixed q₁)

## Problem Reduction

The full problem reduces to 6 DOF at epoch 1:
- 3 DOF discrete: which candidate q₁ (from ~1,400 peak candidates)
- 3 DOF continuous: ω₁ (or equivalently L)
- Everything else is determined by physics

## Quaternion Formulation

Angular momentum rotation between frames:

```
L_inertial = q ⊗ (I_body · ω_body) ⊗ q*
```

Recovering ω at another epoch given L and a candidate q:

```
ω₂_body = I_body⁻¹ · (q₂* ⊗ L_inertial ⊗ q₂)
```

Full chain:

```
ω₂_body = I_body⁻¹ · (q₂* ⊗ q₁ ⊗ (I_body · ω₁_body) ⊗ q₁* ⊗ q₂)
```

The middle part (q₂* ⊗ q₁) is the relative rotation from body frame 1 to body frame 2.

## Propagation

Euler's equations (body frame):
```
I_body · ω̇_body = -ω_body × (I_body · ω_body)
```

Quaternion kinematics:
```
q̇ = ½ q ⊗ ω_body
```

Propagation stays in body frame. Inertial frame only needed when converting L ↔ ω at candidates.

## Next Steps

- Benchmark propagator cost (wall time vs simulation time at various step sizes)
- Prototype: for each q₁ candidate, search over ω₁, propagate forward, check arrival at q₂ candidates
- Explore lo-fi global search over L_inertial space (L is 3 DOF, lo-fi is cheap)
