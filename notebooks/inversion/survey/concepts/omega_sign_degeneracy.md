---
title: "ω-sign degeneracy — flipping ω while compensating q0 produces an LC-equivalent state"
type: concept
created: 2026-04-30
updated: 2026-04-30
confidence: medium
---

# ω-sign degeneracy

## The structural claim

For some `(q0, ω, satellite_geometry)` combinations, there exists a "compensating" rotation `q_compensate` such that `(q_compensate · q0, -ω)` produces an LC indistinguishable from `(q0, +ω)` to within the surrogate / hi-fi noise floor. The flip in ω is time-reversal of the tumble; the rotated q0 reorients the satellite to absorb the geometric difference.

This is **kinematic / geometric**, not buggy-era. It depends on the satellite's symmetry properties and the observation geometry. Stays true post-fix.

## Where it was first quantified (frozen reference)

Discovered on m048 seed 33 in the m126 wrapped-pipeline analysis (parent wiki: `m126_wrapped_pipeline.md`, concept `omega-sign-degeneracy`). Key facts from that work:
- 1 of 11 baseline seeds (seed 33) had a clean ω-sign-flip degeneracy.
- The compensating rotation is ~98.5° about an axis ≈ body +Z (within 2.5°).
- Body +Z on IS-901 is the solar-panel deployment axis; bus + antennas have approximate 4-fold symmetry about it. Combined with time-reversal of the tumble, this produces an LC-equivalent attractor.
- An inline probe at `(q0_truth, -ω_truth)` across 11 seeds found no UNIVERSAL ω-sign symmetry — surrogate MSE was 3–10 on the other 10 seeds. Seed 33's degeneracy is satellite-symmetry + geometry conditional, not generic.

These numbers are buggy-era (the pipeline that found them ran on the buggy forward model). The STRUCTURAL claim — that some seeds have ω-sign-flip degeneracy — should survive the bug fix because the symmetry argument is renderer-independent. **The numbers do not.**

## Convention for reporting ω errors

Always use **signed** ω-direction error, i.e. report whether the candidate ω is forward (signed angle ≤ 90° from truth) or retrograde (signed angle > 90°). The unsigned axis-angle treats `ω` and `-ω` as equivalent, which hides the degeneracy.

When the survey computes ω errors, do:
```python
cos_angle = np.dot(omega_candidate, omega_truth) / (
    np.linalg.norm(omega_candidate) * np.linalg.norm(omega_truth)
)
signed_w_dir_err_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
```
NOT the axis-angle reduction.

## What the survey should check post-fix

- Is the ω-sign-flip degeneracy still present on seed 33 specifically? (Direct probe: surrogate MSE at `(q0_truth, +ω_truth)` vs `(q_compensate · q0_truth, -ω_truth)`. Cheap.)
- Are there OTHER seeds in the 100-seed cohort with a similar degeneracy that the buggy forward model masked or fabricated? Population probe at `(q0_truth, -ω_truth)`: count seeds with surrogate MSE near the truth-state value.
- Does the compensating rotation axis correlate with satellite symmetry directions on those seeds?

These probes are cheap (no propagation needed if you're willing to evaluate at `(q0_truth, -ω_truth)` and let the renderer + surrogate handle the rest, OR a single propagation to compute the body-frame trajectory). Defer until the survey's main thread (cost-at-truth + landscape) is well underway, but it's a natural side-quest.

## Cross-references

- Frozen-reference: `m116_omega_sign_probe.md`, `m126_wrapped_pipeline.md`, concept `omega-sign-degeneracy.md` in parent wiki
- Survey concept: `q_omega_coupling.md` (joint variation rules)
- Concept: `twin_degeneracy.md` (other satellite-symmetry degeneracy: ±X for IS-901)
