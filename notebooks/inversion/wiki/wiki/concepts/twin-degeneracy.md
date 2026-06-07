---
title: "Twin Degeneracy"
type: concept
sources:
  - "raw/inversion_diagnostics/m091_twin_test/twin_lcs.npz"
  - "raw/inversion_diagnostics/m092_twin_axis_analysis.png"
related:
  - "[[m091_twin_state_test]]"
  - "[[m092_twin_axis_visualization]]"
  - "[[m090_robust_peak_selection]]"
  - "[[glint-physics]]"
  - "[[symmetry-degeneracies]]"
created: 2026-04-04
updated: 2026-04-16
confidence: high
---

# Twin Degeneracy

IS-901 has 180-degree rotational symmetry about the +X body axis. The twin state is:

```
q0_twin = q_180x * q0_true       (LEFT multiply — body-frame rotation)
omega_twin = omega_true           (SAME omega, unchanged)
```

This produces an **identical** light curve (surrogate MSE at noise floor ~0.004-0.006 across all seeds tested; hi-fi RMS = 6e-6 mag from m091).

## Quaternion Convention (CRITICAL)

The LCAS pipeline uses **J2000→body** convention: `R(q) @ v_J2000 = v_body`, with `q(t) = q0 * Δq(t)`.

- **LEFT multiply** `q_180x * q0`: applies R_180x in body space → `k1_body_twin(t) = R_180x @ k1_body_true(t)` → correct twin (same omega)
- **RIGHT multiply** `q0 * q_180x`: applies R_180x in J2000 space → body-frame vectors NOT R_180x-related → WRONG twin (would require a different omega to compensate)

Both give `attitude_error_deg = 180°`, so the error metric doesn't distinguish them. But only left-multiply produces matching LCs with the same omega.

**m114's `check_twin_degeneracy`** correctly uses left-multiply.

## Why the Twin Works (Mechanically)

1. Same ω_body + same I_tensor → same Δq(t) (Euler equations are q0-independent)
2. q_twin(t) = q_180x * q_true(t) → R_twin(t) = R_180x @ R_true(t)
3. k1_body_twin(t) = R_180x @ k1_body_true(t) at every timestep
4. ±X geometric symmetry: brightness(R_180x·k1, R_180x·k2) = brightness(k1, k2)
5. Therefore LC_twin = LC_true

Omega does NOT need to change because ω_body describes body-frame dynamics. The flipped body tumbles identically from its own perspective; externally the rotation looks different, but the ±X-symmetric geometry makes the brightness indistinguishable.

## Why Only +X

| Axis | RMS residual |
|------|-------------|
| +/-X | 0.000000 |
| +/-Y | 0.542 |
| +/-Z | 0.542 |
| +WD / +ED | ~1.6 |

Shadow geometry (solar panel self-occlusion) is symmetric about X but not Y or Z. See [[symmetry-degeneracies]] for full analysis of IS-901 symmetries including near-symmetries.

Confirmed definitively by [[m091_twin_state_test]] hi-fi twin test.

## What About "Near-Twin" DE Solutions?

m113/114 found solutions at q0_err ≈ 170° with good MSE (~0.3) when using estimated omega (3° error). These are **NOT** geometric ±X twins — the exact twin at 180° gives MSE ≈ 0.004 with truth omega but MSE ≈ 1.5 with estimated omega (same as truth q0 with estimated omega). The ~170° solutions are local minima of the omega-error-distorted MSE landscape. They are valid solutions (low MSE) but arise from a different mechanism than ±X symmetry.

## Implications for Pipeline

- Any estimated q0 with ~180deg error and axis near +X is a **valid twin** (equivalent solution, not a failure).
- Seeds 6, 24, 36 show q0 ~ 180deg error but axis near -Y -- these are **not** valid twins ([[m092_twin_axis_visualization]]).
- The pipeline must report twin-aware error metrics: check both q0 and q0_twin, take the minimum error.
- When evaluating with estimated omega, both truth and twin have ~same elevated MSE — neither is "better."

## Diagnostic

To test if a candidate is a twin: `q_twin = q_180x * q_true` (LEFT multiply), then `geodesic_distance(q_found, q_twin) < threshold`.
