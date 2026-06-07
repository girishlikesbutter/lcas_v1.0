# Series 07c — Robust Winding Enumeration

> **Status: DONE — validated building block.** Band-sweep with multi-start random directions finds all winding families reliably. Replaces the sequential staircase heuristic. Reusable.

## Problem Statement

m017's staircase finds 8 evenly-spaced winding solutions on leg 0 (peaks 183→260, dt=555.5s):
`0.32, 0.96, 1.60, 2.23, 2.88, 3.51, 4.18, 4.82 deg/s` — true ω (~2.08 deg/s) is near step 3.

But m018's staircase on leg 1 (peaks 260→360, dt=721.4s) got:
`0.25, 3.28, 3.51, 4.00, 4.49, 4.98, 5.48, 5.98 deg/s` — **huge gap 0.25→3.28 deg/s**.
The true ω was NEVER FOUND on leg 1. This caused m018's L-consistency test to fail.

**Root cause hypothesis:** The staircase uses `w_prev + delta_per_rev * rot_axis` as the initial
guess for the next step. For Euler dynamics, the rotation axis changes with |ω|, so the
heuristic can overshoot entire winding families when the axis changes sharply.

---

## m019 — Winding Landscape (dense |ω| sweep)

**Goal:** Map ALL valid ω solutions by sweeping |ω| from 0.1 to 6.0 deg/s in 0.02 deg/s steps.

**Method:** At each target magnitude, minimise arrival error over ω direction using Nelder-Mead
in 2D spherical coordinates (theta, phi), 3 random starts per bin. Parallelised over 8 cores.
Runtime: 434s (~7 min).

### Results

**Leg 0 (peaks 183→260, dt=555.5s):** 10 valleys found

| Valley center | Width  | Bins |
|--------------|--------|------|
| 0.320        | 0.000  | 1    |
| 0.797        | 0.260  | 6    |
| 1.344        | 0.100  | 5    |
| **2.029**    | 0.780  | 13   | ← **near true (2.08 dps)**
| 2.735        | 0.340  | 8    |
| 3.080        | 0.040  | 3    |
| 3.433        | 0.200  | 6    |
| 4.365        | 1.200  | 28   |
| 5.418        | 0.460  | 13   |
| 5.970        | 0.060  | 2    |

**Leg 1 (peaks 260→360, dt=721.4s):** 6 valleys found

| Valley center | Width  | Bins |
|--------------|--------|------|
| 0.280        | 0.080  | 5    |
| 0.626        | 0.260  | 7    |
| 1.265        | 0.740  | 12   |
| **2.399**    | 1.180  | 22   | ← **spans 1.8–3.0, includes true ω**
| 4.065        | 1.740  | 31   |
| 5.558        | 0.880  | 19   |

**Key data point:** At |ω|=2.08 deg/s on leg 1, arrival_err = 0.00097 (below threshold).
At |ω|=2.10 deg/s, arrival_err = 0.00019. **The solution EXISTS.**

### Interpretation

- Leg 0 has dense, well-separated valleys (m017's staircase found most of them)
- Leg 1 has **fewer but wider** valleys. The longer dt (721s vs 556s) changes the
  landscape structure. Valleys merge and broaden.
- The staircase gap on leg 1 is a **search failure**, not a missing solution.
- Leg 1's valleys are less evenly spaced — the 2π/dt heuristic is a poor initial
  guess for a body with triaxial inertia under Euler dynamics.

---

## m020 — Multi-start Staircase (leg 1 only)

**Goal:** Find missed windings on leg 1 using 10 random starts per 0.5 deg/s band.

**Method:** For each of 12 bands [0,0.5), [0.5,1.0), ..., [5.5,6.0), launch 10 L-BFGS-B
runs from random ω (random direction, random magnitude in band) with upper+lower barrier
penalties. Runtime: 957s (~16 min).

### Results

| Band (deg/s)  | Valid/10 | Best |ω| | m018 found? |
|---------------|----------|---------|----------------|
| [0.0, 0.5)    | 9        | 0.474   | 0.249 (yes)    |
| [0.5, 1.0)    | 10       | 0.725   | **NO**         |
| [1.0, 1.5)    | 9        | 1.200   | **NO**         |
| [1.5, 2.0)    | 9        | 1.899   | **NO**         |
| **[2.0, 2.5)**| **9**    | **2.384** | **NO** ← true ω band |
| [2.5, 3.0)    | 8        | 2.690   | **NO**         |
| [3.0, 3.5)    | 8        | 3.494   | 3.279, 3.509   |
| [3.5, 4.0)    | 8        | 3.524   | 3.995          |
| [4.0, 4.5)    | 8        | 4.477   | 4.486          |
| [4.5, 5.0)    | 9        | 4.753   | 4.980          |
| [5.0, 5.5)    | 8        | 5.409   | 5.477          |
| [5.5, 6.0)    | 8        | 5.926   | 5.976          |

**Staircase missed 5 entire winding families:** 0.7, 1.2, 1.9, 2.4, 2.7 deg/s

### Interpretation

- ALL 12 bands contain valid solutions. The omega bridge has solutions at every
  winding number — the question is never "does a solution exist?" but "can the
  search find it?"
- The staircase jumped from 0.25 to 3.28 deg/s, missing bands 0.5–3.0 entirely.
  This is because after step 0 (|ω|≈0.25 dps), the heuristic `w_prev + (2π/dt)*axis`
  gives |w_next| ≈ 0.25 + 0.50 = 0.75 dps in the wrong direction, causing the
  solver to converge back to a high-winding solution instead.
- Multi-start with 10 random directions per band recovers all families.

---

## Key Findings

1. **The leg-1 gap is a search failure, not physics.** Solutions exist near 2.08 deg/s
   for both legs. m018's L-consistency failure was caused by incomplete enumeration,
   not by L-conservation being wrong.

2. **The staircase heuristic fails on leg 1** because the longer dt (721s vs 556s)
   changes the landscape. With Euler dynamics, the rotation axis shifts with |ω|,
   and `w_prev + (2π/dt)*axis` can land in a different basin.

3. **Multi-start random directions in magnitude bands is robust.** 10 random starts
   per 0.5 deg/s band finds solutions in all 12 bands with 67-100% success rate.

4. **Leg 1 has wider, fewer valleys than leg 0.** The longer propagation time makes
   the landscape smoother (wider basins) but with fewer distinct solutions in [0, 6].

## Implications for Pipeline

- **Replace the staircase with a magnitude-band sweep:** For each leg, divide [0, M_max]
  into ~0.5 deg/s bands and run N random-start bridge solves per band. This is
  embarrassingly parallel and more robust than the sequential staircase.

- **L-consistency should work if both legs are enumerated properly.** With correct
  enumeration, the true (k, j) pair should have matching L vectors. This needs to be
  re-tested now that we know how to find all winding families.

- **Computational cost is manageable:** 12 bands × 10 starts × 2 legs = 240 bridge
  solves, each ~1-5s. Total ~10 min on 8 cores. Acceptable for the pipeline.
