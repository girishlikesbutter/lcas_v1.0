# Inversion Strategy Ideas — February 2026

## Table of Contents

1. [Problem Summary and Current State](#1-problem-summary-and-current-state)
2. [Critical Bug: Constant-omega Approximation](#2-critical-bug-constant-omega-approximation)
3. [Idea 1: Conservation Law Constraints (Roberto's Idea)](#3-idea-1-conservation-law-constraints-robertos-idea)
4. [Idea 2: Reparameterize omega to Angular Momentum](#4-idea-2-reparameterize-omega-to-angular-momentum)
5. [Idea 3: Jacobi Elliptic Function Analytical Solution](#5-idea-3-jacobi-elliptic-function-analytical-solution)
6. [Idea 4: Particle Filter / Sequential Monte Carlo](#6-idea-4-particle-filter--sequential-monte-carlo)
7. [Idea 5: Progressive Time Window (Short to Long)](#7-idea-5-progressive-time-window-short-to-long)
8. [Idea 6: Multiple Shooting](#8-idea-6-multiple-shooting)
9. [Idea 7: Brightness Rate of Change Constraint](#9-idea-7-brightness-rate-of-change-constraint)
10. [Prioritised Roadmap](#10-prioritised-roadmap)
11. [Connections to Existing Code](#11-connections-to-existing-code)

---

## 1. Problem Summary and Current State

### 1.1 The Inverse Problem

Given:
- Observed light curve `m_obs(t)` at N epochs
- Satellite geometry (STL meshes, component list)
- BRDF material properties per component
- Orbit ephemeris (satellite, sun, observer positions via SPICE)
- Inertia tensor `I` (3x3, from STL meshes + assumed masses)
- Articulation model (which components rotate, about which axes)

Estimate:
- Initial attitude `q_0` (quaternion, 3 DOF via axis-angle)
- Initial angular velocity `omega_0` (3 DOF, rad/s in body frame)
- (Future: articulation angles per movable component)

For a torque-free tumbling body, Euler's equations + quaternion kinematics propagate `(q_0, omega_0)` to the full attitude history `q(t)`, `omega(t)` for all epochs, so the full estimation problem reduces to 6 parameters.

### 1.2 The Fundamental Difficulty

The joint 6-parameter space has an extremely narrow convergence basin:

| Subproblem | Basin Width | Evidence |
|------------|-------------|----------|
| Attitude only (omega fixed at truth) | ~5-6 degrees | L-BFGS-B converges from 5 degrees |
| Omega only (attitude fixed at truth) | ~0.02 deg/s | Fails at 0.03 deg/s |
| Joint (attitude + omega) | <1 deg attitude + <0.001 deg/s omega | 3 deg + 0.01 deg/s diverges |

**Root cause**: Small omega errors compound over the observation window. At `|omega_err| = 0.01 deg/s` over `T = 3600s`, the accumulated attitude error is `0.01 * 3600 = 36 degrees` — far outside the attitude basin.

### 1.3 What Has Been Tried and Failed

| Approach | Result | Error |
|----------|--------|-------|
| Differential Evolution (6D, hi-fi) | Too slow | — |
| Dual Annealing (6D, lo-fi) | FAILED | 129 deg |
| CMA-ES (6D) | FAILED | 211 deg |
| Tight-start 0.5 deg (lo-fi joint) | FAILED | 36 deg |
| FFT omega-first + grid | FAILED | 162 deg |
| Multi-start L-BFGS-B (joint, within 5 deg) | FAILED | 8/8 failed |

### 1.4 What Works

- **Lo-fi is 163x faster** (0.037s vs 6.1s per eval), Spearman rho = 0.978
- **Attitude-only L-BFGS-B** converges from ~5 degrees (lo-fi), even 10 degrees sometimes works
- **Iso-brightness search** finds ~2000 candidate attitudes per epoch from 10k random starts
- **Epoch chaining** (rotation angle filter) achieves ~99% culling per 7.2s bridge
- **Hi-fi refinement from good lo-fi starting point** works (6/6 success in Phase 4 of basin study)

### 1.5 Current Best Approaches

1. **Two-epoch matching** (`exp_two_epoch_matching.py`): iso-brightness at epoch 0 and epoch 1, pair matching with omega derivation, progressive culling at validation epochs, lo-fi ranking, hi-fi refinement
2. **Epoch chaining** (`exp_chain7_tight.py`): 10 consecutive epochs, 5k candidates each, forward-backward consistency filtering by quaternion distance
3. **Forward propagation** (`exp_forward_prop.py`): 10k candidates at epoch 0, omega grid (11^3 = 1331 points), propagate and cull at 20 check epochs

---

## 2. Critical Bug: Constant-omega Approximation

### 2.1 The Problem

Several experiment scripts use constant-omega (principal-axis) propagation instead of proper Euler dynamics:

**`exp_forward_prop.py` (line 223-225):**
```python
# Propagate: R(t) = R(omega*t) * R0 (constant omega approximation)
rotvec = omega * dt  # omega in rad/s, dt in seconds
R_t = Rotation.from_rotvec(rotvec) * R0
```

**`exp_two_epoch_matching.py` (line 527-529):**
```python
R_check = R_0 * Rotation.from_rotvec(omega_rad * check_dt)
```

**`exp_chain7_tight.py`:** Uses quaternion distance (rotation angle) between adjacent epochs, which is purely kinematic and doesn't account for Euler dynamics.

### 2.2 Why This Matters

For a torque-free asymmetric rigid body, the angular velocity vector **precesses** in the body frame, tracing out a **polhode** (the intersection of the energy and momentum ellipsoids). The constant-omega approximation `q(t) = q_0 * exp(omega * t / 2)` is ONLY valid when:
- The body is axisymmetric (I_1 = I_2), OR
- omega is aligned with a principal axis, OR
- The time interval is very short

For Intelsat 901 with significant asymmetry (bus + two solar panels + two antenna dishes), the precession rate can be substantial. Over the full 3600s observation window, the constant-omega trajectory diverges significantly from the true Euler dynamics trajectory.

### 2.3 Quantifying the Error

The precession period for a torque-free body is approximately:

```
T_precession ~ 2*pi / (|omega| * |I_1 - I_2| / I_3)
```

For the Intelsat 901 inertia tensor (typical values: I_1 ~ 5000, I_2 ~ 15000, I_3 ~ 18000 kg*m^2 — bus-dominated with solar panels extending in one axis), and |omega| ~ 2 deg/s = 0.035 rad/s:

```
T_precession ~ 2*pi / (0.035 * 10000/18000) ~ 320s
```

Over 3600s, the omega vector completes ~11 precession cycles. The constant-omega approximation accumulates catastrophic error after the first half-cycle (~160s).

### 2.4 Impact on Experiments

- **`exp_forward_prop.py`**: Propagates to check epochs at 5-499 (up to 3600s). After ~160s, the constant-omega trajectory is completely wrong. This likely explains why the experiment finds no survivors or only false positives.
- **`exp_two_epoch_matching.py`**: Omega derivation from adjacent epochs (7.2s gap) is fine. But Phase 4 validation propagates to epochs at 2%, 5%, 10%, 25% through the window (72s to 900s). The 900s propagation is invalid.
- **`exp_chain7_tight.py`**: Uses only adjacent-epoch quaternion distances, so the constant-omega issue is less severe (7.2s steps). But the omega derived from pairs is the "average" omega, not accounting for precession within the step.

### 2.5 Fix

Use `propagate_attitude(..., mode="tumbling", inertia_tensor=I)` consistently everywhere. This calls `propagate_euler()` which integrates Euler's equations with DOP853 (8th-order Dormand-Prince). The per-eval cost of `solve_ivp` for a 7-state system over 500 timesteps is small compared to the BRDF evaluation.

For the iso-brightness search and epoch-chaining where you only need one-step propagation, the constant-omega approximation over 7.2s is acceptable. But for any propagation beyond ~50s, use Euler dynamics.

---

## 3. Idea 1: Conservation Law Constraints (Roberto's Idea)

### 3.1 Physics Background

For a torque-free rigid body, two quantities are exactly conserved:

**Angular momentum** (constant in inertial frame):

```
L_inertial = R(q) * I * omega_body = constant vector
```

where `R(q)` is the rotation matrix from body to inertial frame, `I` is the inertia tensor in body frame, and `omega_body` is the angular velocity in body frame.

This gives THREE conserved scalar quantities: `L_x`, `L_y`, `L_z` (inertial components).

**Rotational kinetic energy** (scalar):

```
T = (1/2) * omega_body^T * I * omega_body = constant
```

Additionally, the magnitude of angular momentum is conserved:

```
|L|^2 = omega_body^T * I^2 * omega_body = constant
```

Note: `T` and `|L|^2` are the two conserved quantities that define the **polhode** in body-frame omega-space.

### 3.2 Application to Epoch-Pair Filtering

Currently, the epoch-chaining experiments (`exp_chain7_tight.py`) filter pairs by:
- Quaternion distance (rotation angle) < max_angle_rad

This is a purely kinematic constraint. Conservation laws add dynamical constraints that are much tighter.

**For each consecutive pair `(q_i, q_{i+1})` with a derived `omega_i`:**

1. Compute `omega_i` from the rotation: `omega_i = rotvec(conj(q_i) * q_{i+1}) / dt`
2. Compute the conserved quantities:
   - `T_i = 0.5 * omega_i^T * I * omega_i`
   - `L_i_inertial = R(q_i) * I * omega_i`
   - `|L_i|^2 = L_i_inertial^T * L_i_inertial`

**For a chain of N epochs, require consistency:**
- `T_0 ~ T_1 ~ ... ~ T_{N-1}` (all pairs give the same energy)
- `|L_0| ~ |L_1| ~ ... ~ |L_{N-1}|` (all pairs give the same momentum magnitude)
- `L_0_inertial ~ L_1_inertial ~ ... ~ L_{N-1}_inertial` (all pairs give the same momentum VECTOR)

The last constraint is the strongest: it requires agreement of a 3-vector across all pairs. For a 10-epoch chain, you get 9 estimates of `L_inertial` that should all point in the same direction with the same magnitude.

### 3.3 Why This Is Tighter Than Rotation Angle Filtering

The rotation angle filter only constrains `|omega| * dt < theta_max`. Two candidates can pass this filter while giving wildly different omega directions, which would imply different `L` vectors.

Conservation filtering adds:
- **Energy consistency**: Constrains `|omega|` AND its projection onto principal axes
- **Momentum direction consistency**: Constrains the omega direction (not just magnitude)
- **Cross-epoch consistency**: A chain must agree on a single (T, L) pair

Consider: for a 10-epoch chain with ~2000 candidates per epoch, the rotation angle filter might keep ~1% per step (99% culling), leaving ~20 candidates per epoch. The conservation filter could reduce this further to ~1-5 candidates at epoch 0 by requiring ALL 9 momentum estimates to agree.

### 3.4 Mathematical Detail: Tolerance Computation

The derived omega from a pair `(q_i, q_{i+1})` separated by dt has error due to:
1. **Brightness noise** propagating to attitude uncertainty (~0.5-2 deg per candidate)
2. **Constant-omega approximation** over dt = 7.2s (small for short dt)
3. **Lo-fi vs hi-fi bias** in iso-brightness matching

The omega error translates to T and L errors via:
```
delta_T = omega^T * I * delta_omega  (first-order)
delta_L = R(q) * I * delta_omega     (first-order, inertial frame)
```

For `|delta_omega| ~ 0.1 deg/s = 0.0017 rad/s`, `I ~ 10000 kg*m^2`, `|omega| ~ 0.035 rad/s`:
```
delta_T ~ 10000 * 0.035 * 0.0017 ~ 0.6 J
T_true ~ 0.5 * 10000 * 0.035^2 ~ 6.1 J
delta_T / T ~ 10%
```

So a tolerance of ~20-30% on T and ~20% on |L| should retain the truth while culling bad candidates. This can be tightened empirically.

### 3.5 Implementation Plan

```python
def compute_conserved_quantities(q_wxyz, omega_body, inertia_tensor):
    """Compute T and L_inertial from attitude and body-frame omega."""
    I = inertia_tensor
    T = 0.5 * omega_body @ I @ omega_body
    L_body = I @ omega_body

    # Rotate L from body to inertial frame
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    L_inertial = R @ L_body  # Note: R is body-to-inertial here

    return T, L_inertial, np.linalg.norm(L_inertial)

def conservation_filter(chain_pairs, inertia_tensor, T_tol_frac=0.3, L_tol_frac=0.3):
    """
    Filter chain paths by conservation law consistency.

    chain_pairs: list of (q_i, q_{i+1}, omega_derived) for each step
    Returns: boolean mask of which paths are consistent
    """
    all_T = []
    all_L = []
    for q_i, q_next, omega_i in chain_pairs:
        T_i, L_i, L_mag_i = compute_conserved_quantities(q_i, omega_i, inertia_tensor)
        all_T.append(T_i)
        all_L.append(L_i)

    # Check T consistency
    T_mean = np.mean(all_T)
    T_consistent = all(abs(T - T_mean) / T_mean < T_tol_frac for T in all_T)

    # Check L direction consistency (all L vectors should be parallel)
    L_mean = np.mean(all_L, axis=0)
    L_mean_hat = L_mean / np.linalg.norm(L_mean)
    L_consistent = all(
        np.arccos(np.clip(np.dot(L / np.linalg.norm(L), L_mean_hat), -1, 1)) < np.deg2rad(15)
        for L in all_L
    )

    return T_consistent and L_consistent
```

**Integration with `exp_chain7_tight.py`:**

After the forward-backward pass identifies surviving candidate pairs at each epoch:
1. For each surviving path through the chain, extract the (q_i, q_{i+1}) pairs at each step
2. Derive omega_i from each pair
3. Compute T and L_inertial for each step
4. Reject paths where T or L are inconsistent across steps

This is a post-filter applied to the existing chain output. Computational cost is negligible (just matrix multiplications, no forward model evaluations).

### 3.6 Expected Impact

- **Culling power**: Should eliminate ~50-90% of surviving candidates from the kinematic-only filter, particularly those that have the right rotation magnitude but wrong rotation axis
- **Truth retention**: The conservation laws are exact for the true solution, so the truth should always pass (within noise tolerance)
- **Cost**: Negligible — just linear algebra on already-computed quantities
- **Risk**: Low — purely additive filter, can't make things worse

### 3.7 Extension: Using Conservation to Constrain the Global Search

Beyond filtering, conservation can GUIDE the search:
- Given a candidate q_0 and a target T and |L|, the constraint `T = 0.5 * omega^T * I * omega` and `|L|^2 = omega^T * I^2 * omega` restricts omega to a 1D curve (the polhode)
- This reduces the omega search from 3D to 1D (parameterized by phase along the polhode)
- This connects directly to Idea 3 (Jacobi elliptic functions)

---

## 4. Idea 2: Reparameterize omega to Angular Momentum

### 4.1 Core Insight

The narrow convergence basin in omega-space exists because **omega errors compound over time**. A perturbation `delta_omega_0` grows as `delta_q(t) ~ delta_omega_0 * t` (linearly in time for the attitude error).

Angular momentum `L_inertial`, by contrast, is **exactly constant** for torque-free motion. A perturbation `delta_L` does NOT grow with time. The attitude trajectory is smoothly and continuously dependent on L: nearby L values produce nearby trajectories for ALL time.

**Therefore, the convergence basin in L-space should be dramatically wider than in omega-space.**

### 4.2 Mathematical Framework

**Current parameterization** (in `ObjectiveFunction.evaluate()`):
```
params = [axis_angle_x, axis_angle_y, axis_angle_z, omega_x, omega_y, omega_z]
```

**Proposed L-parameterization:**
```
params = [axis_angle_x, axis_angle_y, axis_angle_z, L_x, L_y, L_z]
```

where `L_x, L_y, L_z` are the components of angular momentum in the **inertial (J2000) frame**.

**The conversion from (q_0, L_inertial) to (q_0, omega_0) is:**
```python
R_0 = quaternion_to_rotation_matrix(q_0)  # body-to-inertial
L_body = R_0.T @ L_inertial               # transform L to body frame
omega_0 = np.linalg.solve(I, L_body)      # omega = I^{-1} * L_body
```

Then propagate `(q_0, omega_0)` using the existing `propagate_euler()` as before. The only change is the parameterization of the search space.

### 4.3 Why the Basin Should Be Wider

Consider two candidate L vectors, L_true and L_true + delta_L:

1. Both produce the SAME L throughout the entire trajectory (it's constant)
2. The omega at any time t is `omega(t) = I^{-1} * R(q(t))^T * L`
3. The attitude trajectory q(t) is a smooth function of L
4. The key difference from omega parameterization: at t=0, `omega_0 = I^{-1} * R(q_0)^T * L`, so a perturbation in L maps to `delta_omega_0 = I^{-1} * R(q_0)^T * delta_L`. The same `delta_omega_0` value. **But at time t, the omega perturbation doesn't compound** — it stays as `delta_omega(t) = I^{-1} * R(q(t))^T * delta_L`, which has the same magnitude (L is constant, so |delta_L| is constant, and I^{-1} is fixed, and R is orthogonal).

**The attitude error at time t** for a perturbation in L does NOT grow linearly with t. Instead, it oscillates and remains bounded. This is the fundamental difference: omega perturbations accumulate, L perturbations don't.

More precisely:
- An omega_0 error of `epsilon` produces attitude error `~ epsilon * t` after time t
- An L error of `epsilon` produces attitude error `~ epsilon * C` where C is a bounded constant that depends on the geometry but NOT on t

For `t = 3600s` and `C ~ 50` (rough estimate), the L-basin should be `3600/50 = 72x` wider than the omega-basin. The exact factor depends on the specific inertia tensor and omega regime.

### 4.4 Experimental Validation

**Test 1: Basin width characterization in L-space**

Redo the basin study (`exp3_basin_resume.py` methodology) with L-parameterization:
1. Compute the true L_inertial from (q_0_true, omega_0_true)
2. Perturb L by various amounts (0.1%, 1%, 5%, 10%, 20%)
3. For each perturbed L, compute omega_0, propagate, generate lightcurve, compute residual
4. Run L-BFGS-B from the perturbed (q_0_true, L_perturbed) starting point
5. Measure convergence rate and basin width

**Test 2: Direct comparison**
- Fix q_0 = q_0_true
- Run L-BFGS-B in omega-space from (q_0_true, omega_0 + perturbation) → measure basin
- Run L-BFGS-B in L-space from (q_0_true, L_true + perturbation) → measure basin
- Compare basin widths

**Expected result**: L-space basin should be at least 10x wider, possibly 50-100x wider.

### 4.5 Implementation

Modify `ObjectiveFunction.evaluate()` to accept L-parameterized input:

```python
class ObjectiveFunction:
    def evaluate_L_param(self, params):
        """
        Evaluate with L-parameterization.

        params: [axis_angle(3), L_inertial(3)]
        """
        axis_angle = params[:3]
        L_inertial = params[3:6]

        # Convert axis-angle to quaternion
        q0 = axis_angle_to_quaternion(axis_angle)

        # Convert L_inertial to omega_0 in body frame
        R_0 = _quaternion_to_rotation_matrix(q0)
        L_body = R_0.T @ L_inertial  # inertial to body
        omega_0 = np.linalg.solve(self.inertia_tensor, L_body)

        # Propagate using existing Euler dynamics
        quaternions, _ = propagate_attitude(
            q0=q0, omega0=omega_0, times=self.observation_times,
            mode="tumbling", inertia_tensor=self.inertia_tensor)

        # ... rest is the same (compute body-frame vectors, generate LC, compute residual)
```

**Bounds for L-space:**

The relationship between L and omega is `L = I * omega`, so:
```
L_max = I_max * omega_max
```
For `I_max ~ 18000 kg*m^2` and `omega_max ~ 3.5 deg/s = 0.061 rad/s`:
```
L_max ~ 18000 * 0.061 ~ 1100 kg*m^2/s
```

Bounds: `L_i in [-1100, 1100]` for each component.

### 4.6 Integration with Decoupled Approach

The L-parameterization is especially powerful when combined with the decoupled approach:

1. **Stage 1**: Iso-brightness search for q_0 candidates (unchanged, 3D attitude search)
2. **Stage 2**: For each q_0 candidate, search over L_inertial (3D) instead of omega_0 (3D)
   - L-space has a wider basin → higher success rate
   - L-space may be smoother → faster convergence
3. **Stage 3**: Joint refinement in (q_0, L) space — should converge better than (q_0, omega)

### 4.7 Theoretical Basis

This approach has connections to the **Hamiltonian formulation** of rigid body dynamics:
- The canonical variables are (attitude angles, angular momenta)
- Angular momenta (L) are the "slow" variables (conserved for torque-free motion)
- Angles are the "fast" variables (evolving at rate omega)
- Optimization over slow variables is always better conditioned than over fast variables

This is a well-known principle in numerical methods: **parameterize in terms of integrals of motion whenever possible**.

---

## 5. Idea 3: Jacobi Elliptic Function Analytical Solution

### 5.1 Background

For a torque-free rigid body with distinct principal moments `I_1 < I_2 < I_3`, Euler's equations have a closed-form solution in terms of **Jacobi elliptic functions**. This is a classical result from mechanics (Landau & Lifshitz, "Mechanics", Section 37; Goldstein, "Classical Mechanics", Chapter 5).

### 5.2 The Solution

Euler's equations in the principal axis frame are:
```
I_1 * omega_1_dot = (I_2 - I_3) * omega_2 * omega_3
I_2 * omega_2_dot = (I_3 - I_1) * omega_3 * omega_1
I_3 * omega_3_dot = (I_1 - I_2) * omega_1 * omega_2
```

Given two conserved quantities:
- Energy: `2T = I_1 * omega_1^2 + I_2 * omega_2^2 + I_3 * omega_3^2`
- Momentum squared: `L^2 = I_1^2 * omega_1^2 + I_2^2 * omega_2^2 + I_3^2 * omega_3^2`

The solution depends on whether `2*T*I_2` is less than, equal to, or greater than `L^2`:

**Case 1: `2*T*I_2 < L^2` (rotation closer to the I_3 axis):**
```
omega_1(t) = Omega_1 * cn(sigma*t + phi_0, k)
omega_2(t) = Omega_2 * sn(sigma*t + phi_0, k)
omega_3(t) = Omega_3 * dn(sigma*t + phi_0, k)
```

**Case 2: `2*T*I_2 > L^2` (rotation closer to the I_1 axis):**
```
omega_1(t) = Omega_1 * dn(sigma*t + phi_0, k)
omega_2(t) = Omega_2 * sn(sigma*t + phi_0, k)
omega_3(t) = Omega_3 * cn(sigma*t + phi_0, k)
```

where:
- `cn`, `sn`, `dn` are Jacobi elliptic functions (`scipy.special.ellipj`)
- The amplitudes are:
  ```
  Omega_1^2 = (L^2 - 2*T*I_3) / (I_1 * (I_1 - I_3))
  Omega_2^2 = (L^2 - 2*T*I_3) / (I_2 * (I_2 - I_3))  [Case 1]
            = (2*T*I_1 - L^2) / (I_2 * (I_2 - I_1))  [Case 2]
  Omega_3^2 = (2*T*I_1 - L^2) / (I_3 * (I_3 - I_1))
  ```
- The frequency parameter is:
  ```
  sigma^2 = (I_1 - I_2)(L^2 - 2*T*I_3) / (I_1 * I_2 * I_3)  [Case 1]
           = (I_2 - I_3)(2*T*I_1 - L^2) / (I_1 * I_2 * I_3)  [Case 2]
  ```
- The elliptic modulus is:
  ```
  k^2 = (I_2 - I_1)(2*T*I_3 - L^2) / ((I_3 - I_2)(L^2 - 2*T*I_1))  [Case 1]
      = (I_2 - I_3)(L^2 - 2*T*I_1) / ((I_1 - I_2)(2*T*I_3 - L^2))  [Case 2]
  ```
- `phi_0` is the initial phase on the polhode (1 parameter)

### 5.3 Polhode Period

The period of the polhode (how long it takes omega to complete one cycle in the body frame) is:
```
T_polhode = 4 * K(k) / sigma
```
where `K(k)` is the complete elliptic integral of the first kind (`scipy.special.ellipk`).

This is analytically known given (T, L^2, I).

### 5.4 Quaternion Integration

Once omega(t) is known analytically, the quaternion kinematics `q_dot = 0.5 * q * omega_quat` must still be integrated. However, this is much simpler because omega(t) is now a known analytical function rather than coupled to q.

Options:
- **Numerical integration** of the quaternion ODE with the analytical omega(t) feeding in — much faster than the full coupled system because the ODE is 4D (quaternion only) instead of 7D, and the right-hand side is cheaper to evaluate
- **Series expansion** for short time intervals: for each polhode period, compute a few terms of the Magnus expansion

### 5.5 New Parameterization

Instead of (q_0, omega_0) with 6 parameters, use:

```
params = [q_0 (3 DOF, axis-angle), T (1 DOF), L^2 (1 DOF), phi_0 (1 DOF)]
```

Still 6 parameters, but:
- `T` and `L^2` are conserved — they don't cause error accumulation
- `phi_0` determines where on the polhode the body starts — this does evolve, but its effect on the observable (light curve) is periodic, not growing
- The relationship between (T, L^2) and the light curve morphology (period, amplitude) is relatively direct and smooth

**Bounds:**
```
T: [0, T_max] where T_max = 0.5 * I_3 * omega_max^2
L^2: [2*T*I_1, 2*T*I_3] (polhode existence condition)
phi_0: [0, 4*K(k)/sigma] (one polhode period)
```

### 5.6 Benefits

1. **No ODE integration for omega**: omega(t) is a closed-form function. This makes the forward model significantly faster (the `solve_ivp` call is eliminated for the omega part).
2. **Analytically differentiable**: The Jacobi elliptic functions have known derivatives: `d/du cn(u,k) = -sn(u,k)*dn(u,k)`, etc. This enables exact gradient computation for L-BFGS-B, which should dramatically improve convergence.
3. **Physical parameterization**: The parameters (T, L^2, phi_0) have direct physical meaning and smooth relationships to observables.
4. **Period is known**: The polhode period `T_polhode` is directly computable, which can be compared to observed periodicities.

### 5.7 Implementation Sketch

```python
from scipy.special import ellipj, ellipk

def omega_analytical(t, I1, I2, I3, T_energy, L_sq, phi0):
    """
    Compute omega(t) in principal axis frame using Jacobi elliptic functions.

    Parameters
    ----------
    t : float or array
        Time(s) at which to evaluate
    I1, I2, I3 : float
        Principal moments of inertia (I1 < I2 < I3)
    T_energy : float
        Rotational kinetic energy
    L_sq : float
        Squared angular momentum magnitude
    phi0 : float
        Initial phase on polhode

    Returns
    -------
    omega : array (3,) or (N, 3)
        Angular velocity in principal axis frame
    """
    # Determine case
    if 2 * T_energy * I2 < L_sq:  # Case 1: near I3 axis
        Om1_sq = (L_sq - 2*T_energy*I3) / (I1 * (I1 - I3))
        Om2_sq = (L_sq - 2*T_energy*I3) / (I2 * (I2 - I3))
        Om3_sq = (2*T_energy*I1 - L_sq) / (I3 * (I3 - I1))

        sigma_sq = (I1 - I2) * (L_sq - 2*T_energy*I3) / (I1 * I2 * I3)
        k_sq = (I2 - I1) * (2*T_energy*I3 - L_sq) / ((I3 - I2) * (L_sq - 2*T_energy*I1))

        sigma = np.sqrt(abs(sigma_sq))
        k = np.sqrt(abs(np.clip(k_sq, 0, 0.9999)))

        u = sigma * t + phi0
        sn, cn, dn, _ = ellipj(u, k**2)  # Note: ellipj takes m = k^2

        omega = np.column_stack([
            np.sqrt(abs(Om1_sq)) * cn,
            np.sqrt(abs(Om2_sq)) * sn,
            np.sqrt(abs(Om3_sq)) * dn,
        ]) if np.ndim(t) > 0 else np.array([
            np.sqrt(abs(Om1_sq)) * cn,
            np.sqrt(abs(Om2_sq)) * sn,
            np.sqrt(abs(Om3_sq)) * dn,
        ])
    else:  # Case 2: near I1 axis
        # ... analogous with dn, sn, cn ordering
        pass

    return omega
```

**Caveat:** The principal axis frame may differ from the body frame used in LCAS. The inertia tensor must be diagonalized first, and the quaternion must be defined relative to the principal axis frame. The transformation is:
```python
eigenvalues, eigenvectors = np.linalg.eigh(I)  # I = V * diag(I1,I2,I3) * V^T
# omega_principal = V^T @ omega_body
# q_principal relates to q_body by the V rotation
```

### 5.8 Complexity and Risk

- **Implementation effort**: Medium-high. The Jacobi elliptic function solution requires careful handling of the two cases, sign conventions, principal axis transformation, and the quaternion integration.
- **Risk**: The quaternion still needs numerical integration (just a 4D ODE instead of 7D), so the speedup for a single evaluation may be 2-3x rather than transformative. The bigger win is differentiability.
- **When to attempt**: After validating that the L-parameterization (Idea 2) widens the basin. If the basin is still narrow even in L-space, the Jacobi solution offers a different angle (exact gradients).

---

## 6. Idea 4: Particle Filter / Sequential Monte Carlo

### 6.1 Concept

Instead of optimizing a global objective over the full observation window, process the light curve **sequentially in time** using a particle filter. This is a Bayesian approach where we maintain a population of state hypotheses and update them epoch by epoch.

### 6.2 Algorithm

```
INITIALIZE:
  For i = 1 to N_particles:
    q_i ~ sample from iso-brightness candidates at epoch 0
    omega_i ~ sample from prior (uniform in [-omega_max, omega_max]^3 or polhode-constrained)
    w_i = 1 / N_particles  (uniform weights)

FOR each epoch t = 1, 2, ..., N_obs:

  PREDICT:
    For each particle i:
      Propagate (q_i, omega_i) by one timestep dt using Euler's equations
      q_i, omega_i = propagate_euler(q_i, omega_i, I, [0, dt])

  UPDATE:
    For each particle i:
      Compute predicted brightness: mag_pred_i = brightness(q_i, epoch_t)
      Compute weight: w_i = exp(-0.5 * ((mag_pred_i - mag_obs_t) / sigma)^2)
    Normalize: w_i = w_i / sum(w_j)

  RESAMPLE:
    N_eff = 1 / sum(w_i^2)  (effective sample size)
    If N_eff < N_particles / 2:
      Resample N_particles particles with replacement, proportional to weights
      Add small jitter to prevent collapse: omega_i += N(0, sigma_jitter)
      Reset weights: w_i = 1 / N_particles

RESULT:
  Weighted mean of surviving particles gives (q_0, omega_0) estimate
  Particle spread gives uncertainty
```

### 6.3 Why This Bypasses the Narrow Basin Problem

The key insight: **each timestep only propagates by dt = 7.2s**, not 3600s. Over 7.2s:
- The attitude changes by `|omega| * dt ~ 2 deg/s * 7.2s ~ 14.4 degrees` — substantial but manageable
- The omega changes by `|omega_dot| * dt`, which for Euler dynamics is `|(I^{-1} * omega x I*omega)| * dt`. For typical values, this is `~ 0.001 rad/s * 7.2s = 0.007 rad/s ~ 0.4 deg/s` — a small change

So the single-step propagation error is small for ALL particles, and the brightness update at each epoch progressively eliminates wrong hypotheses. Over 500 epochs, the population converges to the true trajectory.

**Comparison with the current approach:**
- Current: Find candidates at epoch 0, propagate ALL THE WAY to epoch 499, check. Errors accumulate catastrophically.
- Particle filter: Propagate one step, check, propagate one step, check, ... Errors never accumulate because they're corrected at each step.

### 6.4 Initialization Strategy

The initial particle distribution is critical. Options:

**Option A: Iso-brightness + uniform omega**
- q_i: sample from iso-brightness candidates at epoch 0 (already available from existing code)
- omega_i: uniform random in `[-omega_max, omega_max]^3`, filtered by `|omega| < omega_max`
- Problem: omega space is 3D, need many particles for coverage

**Option B: Iso-brightness + conservation-constrained omega**
- q_i: sample from iso-brightness candidates at epoch 0
- For each q_i, also use the iso-brightness search at epoch 1 to get a rough omega estimate
- omega_i: sample around the estimated omega with some spread
- Better: uses information from two epochs to initialize

**Option C: Iso-brightness + polhode-constrained omega**
- q_i: sample from iso-brightness candidates at epoch 0
- Sample (T, |L|) from a prior
- For each (T, |L|, q_i), compute the set of allowable omega values (polhode curve)
- Sample omega from this 1D curve
- Most physically informed initialization

### 6.5 Computational Cost

Per particle per epoch:
- Propagate Euler (7-state ODE, 1 step): ~0.01 ms (trivial, just one RK step)
- Compute brightness (lo-fi): ~0.04 ms (your benchmark: 37ms for 500 epochs = 0.074ms/epoch, but some of that is overhead)
- Actually, the lo-fi single-epoch eval benchmark was ~2.5ms (from your experiment setup). For the particle filter we need a lighter evaluation.

**For N = 10,000 particles, 500 epochs:**
```
Total evals = 10,000 * 500 = 5,000,000
At 0.5ms per eval = 2500s = 42 minutes (serial)
At 0.05ms per eval (vectorized) = 250s = 4 minutes
```

The key is to **vectorize** the brightness evaluation. Currently each call goes through `generate_lightcurves()` which has Python-loop overhead. A vectorized "evaluate N attitudes at one epoch" function would be transformative.

### 6.6 Vectorized Brightness Evaluation

For the particle filter to be practical, we need a function that evaluates brightness for many quaternions at a single epoch simultaneously:

```python
def brightness_batch_lofi(quats_wxyz, epoch_idx, ctx):
    """
    Evaluate lo-fi brightness for N quaternions at one epoch.

    quats_wxyz: (N, 4) array of quaternions
    epoch_idx: int

    Returns: (N,) array of magnitudes
    """
    N = len(quats_wxyz)

    # Convert quaternions to rotation matrices (vectorized)
    # ... (scipy Rotation supports batched operations)

    # Compute body-frame sun/observer vectors for all N attitudes
    # Each is a batch matrix-vector multiply: R @ v

    # Evaluate BRDF for all N attitudes (vectorize the facet sum)
    # This is the key bottleneck — can be done with numpy broadcasting
```

The BRDF evaluation per facet is:
```
flux = sum_over_facets(area * rho(n, k1, k2) * cos_theta_i * delta_omega)
```

For N attitudes, each facet gets N different (k1, k2) pairs but the facet geometry (area, normal) is the same. This is embarrassingly parallel and can be vectorized with numpy:
```python
# normals: (F, 3) - F facets
# k1: (N, 3) - N sun directions
# k2: (N, 3) - N observer directions
# cos_theta_i = (N, F) = k1 @ normals.T
# etc.
```

This could bring the per-particle-per-epoch cost down to ~0.001ms, making the particle filter very fast.

### 6.7 Resampling and Degeneracy

Standard particle filter issues:
- **Particle degeneracy**: After many resampling steps, all particles may descend from a single ancestor → loss of diversity
- **Solution**: Jitter after resampling, MCMC rejuvenation moves, or regularized resampling (move particles toward the posterior mode with added noise)

For this application, a practical mitigation is:
- Use **stratified resampling** (reduces variance vs multinomial)
- Apply **omega jitter** after resampling: `omega_i += N(0, sigma_jitter * I)` where sigma_jitter decreases over time as the filter converges
- Maintain a **minimum diversity** by injecting fresh particles if the effective sample size drops too low

### 6.8 Expected Outcome

A well-tuned particle filter with 10,000 particles should:
- Converge to the true (q_0, omega_0) within ~100-200 epochs
- Provide uncertainty estimates (particle spread at convergence)
- Handle multimodality gracefully (different particle clusters)
- Runtime: 5-15 minutes with vectorized brightness evaluation

---

## 7. Idea 5: Progressive Time Window (Short to Long)

### 7.1 Concept

The narrow basin problem arises because we evaluate the objective function over the full 3600s window, where omega errors accumulate catastrophically. Over SHORTER windows, the effective basin is wider because errors haven't accumulated.

**Strategy**: Start optimization with a short time window (where the basin is wide), find a coarse solution, then progressively extend the window (tightening the solution at each stage).

### 7.2 Analysis: Basin Width vs Window Length

The accumulated attitude error from an omega perturbation `delta_omega` over time `T` is approximately:
```
delta_q(T) ~ delta_omega * T  (in radians)
```

The attitude-only basin is ~5 degrees. For the joint solution to remain within this basin, we need:
```
delta_omega * T < 5 deg = 0.087 rad
```

Therefore:
```
delta_omega_max ~ 0.087 / T
```

| Window T | omega basin width | Relative to full window |
|----------|-------------------|------------------------|
| 50s (7 epochs) | 0.0017 rad/s = 0.1 deg/s | 100x wider |
| 100s (14 epochs) | 0.00087 rad/s = 0.05 deg/s | 50x wider |
| 200s (28 epochs) | 0.00044 rad/s = 0.025 deg/s | 25x wider |
| 500s (70 epochs) | 0.00017 rad/s = 0.01 deg/s | 10x wider |
| 3600s (500 epochs) | 0.000024 rad/s = 0.0014 deg/s | baseline |

At `T = 50s`, the omega basin is ~0.1 deg/s, which is 100x wider than at 3600s. This should be easily findable by DE or multi-start L-BFGS-B.

### 7.3 Algorithm

```
Stage 1: T = 50s (7 epochs)
  - Use lo-fi DE to search (q_0, omega_0) with omega bounds [-omega_max, omega_max]
  - Basin is wide → DE should succeed with moderate budget (~2000 evals)
  - Get coarse estimate (q_0*, omega_0*)

Stage 2: T = 100s (14 epochs)
  - Start from (q_0*, omega_0*) found in Stage 1
  - Refine with L-BFGS-B using the 100s window
  - Tighter omega precision, but starting point is already close

Stage 3: T = 200s (28 epochs)
  - Start from Stage 2 result
  - Refine with L-BFGS-B using the 200s window

Stage 4: T = 500s (70 epochs)
  - Refine further

Stage 5: T = 3600s (500 epochs, full window)
  - Final joint refinement with all data
  - Switch to hi-fi (shadows enabled) at this stage

ALTERNATIVELY: use exponential window doubling:
  50s → 100s → 200s → 400s → 800s → 1600s → 3200s → 3600s
```

### 7.4 Key Consideration: Information Content

Very short windows have limited information content because the attitude hasn't changed much. Specifically:
- Over 50s at |omega| ~ 2 deg/s, the body rotates ~100 degrees
- The light curve shows ~1-2 features (peaks/valleys) in 50s
- This constrains the (q_0, omega_0) combination but with significant ambiguity

The progressive extension resolves this ambiguity: each stage adds more information while the basin at each stage is wide enough for the optimizer to handle.

### 7.5 Implementation

Modify the objective function to accept a window parameter:

```python
def evaluate_windowed(params, t_window):
    """Evaluate objective using only epochs within [0, t_window]."""
    axis_angle = params[:3]
    omega = params[3:6]
    q0 = axis_angle_to_quaternion(axis_angle)

    # Find epochs within window
    mask = observation_times <= t_window
    times_window = observation_times[mask]

    # Propagate only within window
    quats, _ = propagate_attitude(q0, omega, times_window, "tumbling", I)

    # Compute residual only for windowed epochs
    ...
```

Then the progressive pipeline:

```python
windows = [50, 100, 200, 500, 3600]
x_current = initial_guess  # from DE or random

for T_win in windows:
    result = minimize(
        lambda x: evaluate_windowed(x, T_win),
        x_current, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 200})
    x_current = result.x
```

### 7.6 Combination with L-parameterization

If combined with Idea 2, the progressive window approach becomes even stronger:
- Stage 1 (50s): DE search in (q_0, L) space — wide basin in both q and L
- Later stages: L-BFGS-B refinement in (q_0, L) space with progressively longer windows
- L doesn't accumulate errors, so the basin stays wide even as T increases

This combination could potentially make the straightforward DE + L-BFGS-B approach work on the joint problem directly, eliminating the need for decoupled estimation entirely.

### 7.7 Risk

The main risk is that short windows are **underdetermined** — the 7-epoch window at 2 deg/s omega may have multiple (q_0, omega) solutions that produce identical brightness. The optimizer may lock onto a wrong solution early and fail to escape in later stages.

Mitigation: multi-start in Stage 1, keep top-N candidates through stages.

---

## 8. Idea 6: Multiple Shooting

### 8.1 Concept

Multiple shooting is a classical numerical method for solving boundary-value problems and parameter estimation in ODEs. Instead of propagating from a single initial condition over the full time span (which is sensitive to initial condition errors), the time span is divided into segments, each with its own initial condition, and continuity constraints couple adjacent segments.

### 8.2 Formulation

Divide the 3600s observation window into K segments of length ~T_seg:

```
Segment 1: [0, T_seg]         — variables: (q_1, omega_1) at t=0
Segment 2: [T_seg, 2*T_seg]   — variables: (q_2, omega_2) at t=T_seg
...
Segment K: [(K-1)*T_seg, T]   — variables: (q_K, omega_K) at t=(K-1)*T_seg
```

**Decision variables**: (q_k, omega_k) for k = 1, ..., K → total 6K parameters

**Objective function**:
```
min  sum_{k=1}^{K} sum_{t in segment_k} (m_pred(t) - m_obs(t))^2 / sigma^2

subject to:
  propagate(q_k, omega_k, T_seg) = (q_{k+1}, omega_{k+1})   for k = 1, ..., K-1
  (continuity constraints)
```

### 8.3 Advantages

1. **Each segment is short** (~60s for K=60), so the sensitivity to initial conditions within each segment is low
2. **The convergence basin for each segment** is wide (omega basin ~0.05 deg/s for 60s segments vs 0.001 deg/s for the full window)
3. **Parallelizable**: Segments can be evaluated independently given their initial conditions
4. **Well-studied**: Multiple shooting is a workhorse method in trajectory optimization, orbit determination, and optimal control. Efficient solvers exist.

### 8.4 Solution Method

**Option A: Direct transcription + NLP solver**

Convert to a nonlinear programming (NLP) problem:
- Variables: x = [q_1, omega_1, q_2, omega_2, ..., q_K, omega_K] ∈ R^{6K}
- Objective: lightcurve residual
- Equality constraints: continuity at segment boundaries

Use SciPy's `minimize` with method `SLSQP` (Sequential Least Squares Programming) or the `trust-constr` method, which handles equality constraints.

**Option B: Penalty method**

Add continuity as a penalty term:
```
min  sum_k LC_residual(segment_k) + lambda * sum_k ||end(segment_k) - start(segment_{k+1})||^2
```

Start with small lambda, increase gradually. No constrained solver needed — standard L-BFGS-B works.

**Option C: Sequential approach**

1. Solve each segment independently (no continuity constraints) — this gives K independent 6-parameter problems, each with a wide basin
2. Use the "stitch" constraint as a regularizer to pull segments into consistency
3. Iterate between (a) refining each segment given its neighbors, and (b) enforcing consistency

### 8.5 Segment Length Choice

- **Too short** (< 20s, < 3 epochs): Not enough brightness measurements to constrain 6 parameters per segment. Underdetermined.
- **Too long** (> 300s, > 40 epochs): Basin narrows, losing the advantage of multiple shooting.
- **Sweet spot**: ~60-120s (8-16 epochs per segment), giving K = 30-60 segments.

For K = 60 segments: 360 total parameters, but each segment's 6 parameters interact only with its neighbors. The constraint Jacobian is block-tridiagonal — highly structured and efficient to solve.

### 8.6 Conservation Constraints

Multiple shooting naturally accommodates conservation constraints:
- Add `T_k = T_{k+1}` and `L_k = L_{k+1}` as additional equality constraints
- Or equivalently: require `T_k = T_1` and `L_k = L_1` for all k (all segments share the same energy and momentum)
- This couples all segments and prevents drift in the conserved quantities

### 8.7 Implementation Sketch

```python
from scipy.optimize import minimize

def multiple_shooting_objective(x, K, T_seg, ctx, lam=100):
    """
    x: array of shape (6*K,), containing [aa_1, omega_1, aa_2, omega_2, ...]
    """
    total_residual = 0
    continuity_penalty = 0

    for k in range(K):
        # Extract segment k's initial conditions
        aa_k = x[6*k : 6*k+3]
        omega_k = x[6*k+3 : 6*k+6]
        q_k = axis_angle_to_quaternion(aa_k)

        # Propagate segment k
        t_start = k * T_seg
        t_end = min((k+1) * T_seg, T_total)
        epoch_mask = (observation_times >= t_start) & (observation_times < t_end)
        times_segment = observation_times[epoch_mask] - t_start  # relative times

        quats_k, omegas_k = propagate_attitude(
            q_k, omega_k, times_segment, "tumbling", I)

        # Lightcurve residual for this segment
        # ... compute body-frame vectors, predict brightness, sum residuals
        total_residual += segment_residual

        # Continuity constraint: end of segment k = start of segment k+1
        if k < K - 1:
            q_end = quats_k[-1]
            omega_end = omegas_k[-1]
            q_next = axis_angle_to_quaternion(x[6*(k+1) : 6*(k+1)+3])
            omega_next = x[6*(k+1)+3 : 6*(k+1)+6]

            # Quaternion distance
            dq = attitude_error_deg(q_end, q_next)
            # Omega distance
            domega = np.linalg.norm(omega_end - omega_next)

            continuity_penalty += dq**2 + domega**2

    return total_residual + lam * continuity_penalty
```

### 8.8 Expected Benefit

- Each segment's 6-parameter optimization has a ~50x wider basin than the full-window optimization
- The continuity constraints couple segments without requiring error-free propagation over the full window
- The method is well-established in the trajectory optimization literature
- Scales well: K segments can be evaluated in parallel

### 8.9 Risk

- 6K parameters is a high-dimensional optimization (360 for K=60). But the structure (block-tridiagonal) means it's much easier than a generic 360-parameter problem.
- The penalty method can be sensitive to lambda tuning. The constrained optimization approach (SLSQP/trust-constr) is more robust.

---

## 9. Idea 7: Brightness Rate of Change Constraint

### 9.1 Concept

The instantaneous rate of brightness change `dB/dt` is directly related to the angular velocity. At any epoch, the brightness depends on the body-frame sun and observer vectors. The rate of change of brightness depends on how fast these vectors are sweeping — which is proportional to omega.

This provides a constraint on omega that is **instantaneous** (no propagation needed, no error accumulation).

### 9.2 Mathematical Detail

The apparent magnitude `m(t)` depends on the attitude `q(t)` through the body-frame vectors:
```
k1(t) = R(q(t))^T * (r_sun - r_sat)  / |r_sun - r_sat|     (sun direction, body frame)
k2(t) = R(q(t))^T * (r_obs - r_sat)  / |r_obs - r_sat|     (observer direction, body frame)
```

The rate of change is:
```
dk1/dt = d/dt [R(q)^T] * s_inertial = -[omega x] * R(q)^T * s_inertial = -omega x k1
dk2/dt = -omega x k2
```

where `[omega x]` is the skew-symmetric matrix of omega (body frame).

Therefore:
```
dB/dt = (dB/dk1) * (dk1/dt) + (dB/dk2) * (dk2/dt)
      = (dB/dk1) * (-omega x k1) + (dB/dk2) * (-omega x k2)
```

This is LINEAR in omega. Given a known attitude q (and therefore known k1, k2), and a computable gradient dB/dk1, dB/dk2 from the forward model, the observed dB/dt provides a linear constraint on the 3 components of omega.

### 9.3 Practical Computation

**Step 1: Estimate dB/dt from observations**
```python
dBdt_obs = np.gradient(observed_lc, dt_sampling)
```

This gives a noisy estimate of dB/dt at each epoch. The noise is amplified by differentiation, but can be smoothed with a Savitzky-Golay filter.

**Step 2: Compute dB/dk1 and dB/dk2 from the forward model**

For a candidate attitude q at epoch t:
- Compute brightness B(q)
- Perturb k1 slightly: B(q; k1 + epsilon*e_i) for i=1,2,3 → finite-difference gradient dB/dk1
- Similarly for dB/dk2

This is 6 extra brightness evaluations per epoch (or use analytical BRDF gradient).

**Step 3: Set up the linear system**

At epoch t with candidate attitude q:
```
dBdt_obs(t) = -(dB/dk1) * (omega x k1) - (dB/dk2) * (omega x k2)
```

Rewrite using the identity `a x b = -b x a` and the hat map:
```
dBdt_obs(t) = (dB/dk1)^T * [k1]_x * omega + (dB/dk2)^T * [k2]_x * omega
            = [(dB/dk1)^T * [k1]_x + (dB/dk2)^T * [k2]_x] * omega
            = A(t) * omega
```

where `A(t)` is a (1 x 3) row vector that depends on the attitude and geometry at epoch t.

With N epochs, we get an overdetermined linear system:
```
[A(t_1)]        [dBdt(t_1)]
[A(t_2)] * omega = [dBdt(t_2)]
[  ...  ]        [  ...    ]
[A(t_N)]        [dBdt(t_N)]
```

This is solvable by least squares, giving an estimate of omega.

### 9.4 Limitations

1. **Assumes constant omega**: The linear system treats omega as constant across all epochs. For a tumbling body, omega changes over time. This means the method works best for:
   - Short time windows (where omega is approximately constant)
   - Principal-axis rotation (where omega IS constant)
   - As a rough estimate to initialize refinement

2. **Requires known attitude**: The gradient dB/dk1, dB/dk2 depends on the attitude. If the attitude is wrong, the omega estimate will be wrong. However, this can be iterated: estimate omega given attitude, update attitude given omega, repeat.

3. **Noise amplification**: Numerical differentiation amplifies measurement noise. Smoothing helps but reduces temporal resolution.

### 9.5 Application as an Initializer

The most practical use of dB/dt is as an **omega initializer** for the decoupled approach:

1. Find iso-brightness attitude candidates at epoch 0
2. For each candidate q_0:
   a. Propagate with omega=0 for a short window (7 epochs)
   b. At each epoch, compute the gradient matrix A(t)
   c. Solve the linear system for omega
   d. Use this omega as a starting point for optimization

This replaces the current omega grid search (11^3 = 1331 points) with a single least-squares solve per candidate, which is both faster and more accurate.

### 9.6 Advanced: Instantaneous Omega Estimation

For a tumbling body, omega changes over time. By solving the linear system at EACH epoch independently (using just dBdt at that epoch and nearby epochs), you get a time series `omega(t)`. This omega(t) trajectory should:
- Satisfy Euler's equations: `I * omega_dot = -omega x (I*omega)`
- Have constant T and |L|

Fitting the Euler dynamics to the estimated omega(t) trajectory gives another route to (T, |L|, q_0).

---

## 10. Prioritised Roadmap

### Phase 0: Bug Fix (1 day)
- Fix constant-omega propagation in `exp_forward_prop.py`, `exp_two_epoch_matching.py`
- Use `propagate_attitude(..., mode="tumbling")` everywhere beyond ~50s propagation
- Re-run `exp_two_epoch_matching.py` and `exp_forward_prop.py` with proper Euler dynamics

### Phase 1: Quick Wins (1-2 days)
1. **Conservation filter** (Idea 1): Add T and L consistency checks to `exp_chain7_tight.py`
   - Implement `compute_conserved_quantities()` helper
   - Add as a post-filter after the existing forward-backward pass
   - Measure additional culling power and truth retention

2. **L-parameterization basin study** (Idea 2): Quick experiment to test whether the basin in L-space is wider than in omega-space
   - Implement `evaluate_L_param()` in `ObjectiveFunction`
   - Run perturbation analysis: fix q_0 = true, perturb L by 1%, 5%, 10%, 20%
   - Compare basin width with omega-space results from `exp3_basin_resume.py`

### Phase 2: Core Method Development (1-2 weeks)
Based on Phase 1 results:

3. **If L-basin is wider**: Build a complete inversion pipeline using L-parameterization
   - Decoupled approach: iso-brightness q_0 search → L search (per candidate) → joint refinement in (q_0, L) space
   - Test with progressive time window (Idea 5) for extra robustness

4. **If L-basin is NOT wider**: Implement particle filter (Idea 4)
   - Build vectorized brightness evaluation function
   - Implement bootstrap particle filter with systematic resampling
   - Initialize with iso-brightness candidates + prior on omega
   - Test on synthetic data

### Phase 3: Advanced Methods (2-4 weeks)
5. **Jacobi elliptic function solution** (Idea 3): Only if the analytical solution is needed for:
   - Speed (current `solve_ivp` is a bottleneck)
   - Differentiability (gradient-based methods not converging without exact gradients)

6. **Multiple shooting** (Idea 6): If progressive window has convergence issues
   - Implement penalty-method version first (simpler)
   - Graduate to constrained optimization if needed

### Phase 4: Towards Articulation-Aware Estimation
7. Once the basic (q_0, omega_0) inversion works reliably:
   - Add articulation angles to the parameter vector
   - The forward model already supports articulation — just need to make the angles free parameters
   - Start with one movable component, validate, then add more

---

## 11. Connections to Existing Code

### 11.1 Files to Modify

| Idea | Files | Nature of Change |
|------|-------|-----------------|
| Bug fix | `exp_forward_prop.py`, `exp_two_epoch_matching.py` | Replace `Rotation.from_rotvec(omega*dt)` with `propagate_attitude()` |
| Conservation filter | New file or addition to `exp_chain7_tight.py` | Add `compute_conserved_quantities()` helper |
| L-parameterization | `src/inversion/objective_function.py` | Add `evaluate_L_param()` method |
| Jacobi solution | New file `src/dynamics/jacobi_propagator.py` | New analytical propagation module |
| Particle filter | New file `notebooks/inversion/exp_particle_filter.py` | New experiment script |
| Progressive window | `src/inversion/objective_function.py` | Add window parameter to `evaluate()` |
| Multiple shooting | New file `notebooks/inversion/exp_multiple_shooting.py` | New experiment script |
| dB/dt constraint | New file `notebooks/inversion/exp_brightness_gradient.py` | New experiment script |

### 11.2 Key Existing Code to Reuse

- **`src/dynamics/attitude_propagator.py`**: `propagate_euler()` for proper Euler dynamics
- **`src/computation/inertia_calculator.py`**: `InertiaResult` with `inertia_tensor`, `principal_moments`, `principal_axes`
- **`src/inversion/objective_function.py`**: `ObjectiveFunction` class — the main forward model wrapper
- **`notebooks/inversion/lib/experiment_setup.py`**: `ExperimentContext`, `setup_experiment()`, `brightness_single_epoch()`
- **`src/inversion/optimizers.py`**: `global_optimize()` (DE), `local_refine()` (L-BFGS-B)
- **`src/inversion/constraints.py`**: Bound computation for axis-angle and omega

### 11.3 Key Parameters and Constants

From `experiment_setup.py` and the experiment scripts:

| Parameter | Value | Source |
|-----------|-------|--------|
| True q_0 | axis=[0.6,0.3,0.8]/norm, angle=45 deg | `experiment_setup.py:150-155` |
| True omega_0 | [0.005, -0.003, 0.05] deg/s (setup) or [0.5, -0.3, 2.0] deg/s (chain7) | `experiment_setup.py:79` and `exp_chain7_tight.py:74` |
| Inertia tensor | From IS901 STL + masses {Bus:1532, SP:170x2, AD:50x2} kg | `experiment_setup.py:104-111` |
| N_observations | 500 | Standard across experiments |
| dt_sampling | ~7.2s (for 500 obs over 1 hour) | Computed |
| Noise sigma | 0.05 mag | Standard |
| Lo-fi single-epoch eval | ~2.5 ms | Benchmark |
| Hi-fi single-epoch eval | ~12 ms | Benchmark |
| Lo-fi full-LC eval | ~37 ms | Benchmark |

### 11.4 Two Omega Regimes

Note: There are TWO different omega regimes in the experiments:
1. **Slow tumbler** (`experiment_setup.py` default): omega = [0.005, -0.003, 0.05] deg/s, |omega| = 0.05 deg/s. The body rotates ~180 degrees over 3600s. This is the original test case.
2. **Fast tumbler** (`exp_chain7_tight.py`, `exp_forward_prop.py`): omega = [0.5, -0.3, 2.0] deg/s, |omega| = 2.1 deg/s. The body makes ~21 full rotations over 3600s. This is more challenging and more realistic for debris.

The fast tumbler is harder because:
- More aliasing and ambiguity in the light curve
- Faster omega precession on the polhode
- Shorter polhode period → constant-omega approximation breaks down faster

Most of the "FAILED" experiments used the slow tumbler (exp3_basin, dual_annealing, CMA-ES). The chain and forward-prop experiments used the fast tumbler. Be careful to compare like with like.
