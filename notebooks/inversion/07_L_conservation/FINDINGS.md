# Series 07b — L-Conservation as Winding Filter

## Summary

**L-conservation is a viable and extremely robust winding filter.** With oracle attitudes, it uniquely identifies the correct winding pair with 9 orders of magnitude separation. With up to 10 deg endpoint attitude errors, it still achieves 100% accuracy.

---

## Background

micro18 tested L-consistency across two legs sharing a peak node: angular momentum L = R(q) * (I * omega_body) must be conserved in the inertial frame. It FAILED — but not because L-consistency is wrong. It failed because leg 1's staircase missed the true omega entirely (jumped from 0.25 to 3.28 deg/s, skipping the true 2.08 deg/s). With the correct omega absent from leg 1's candidate set, no pair could match.

This series tests whether L-consistency works **in principle** by injecting the true omega into both legs' candidate sets.

---

## micro23 — Oracle L-consistency test

**Question:** With oracle attitudes and truth injected, does L-conservation identify the correct winding pair?

**Answer: YES — unambiguously.**

| Metric | True pair | Next-best | Gap |
|--------|-----------|-----------|-----|
| \|\|DL\|\| (kg*m^2/s) | 8.2e-08 | 113.2 | **9 orders of magnitude** |
| \|DT\| (kg*m^2*rad^2/s^2) | 8.3e-10 | 0.34 | 8 orders of magnitude |
| Combined (normalised) | 0.0 | — | rank 1/64 |

The L-error heatmap shows row k=3 (true leg-0 winding) has a perfect minimum at j=3 (true leg-1 winding) with L_err = 0. All other pairs have L_err >= 113 kg*m^2/s. The pattern is clean and monotonic — wrong windings produce L-errors proportional to their distance from the true winding.

### Does kinetic energy (T) add discrimination beyond L?

T also ranks the true pair as #1/64 with gap = 0.34. But T provides no additional information beyond L in this test case. Since T = 0.5 * omega^T * I * omega depends only on omega (not q), it's a simpler check — but L already captures the omega mismatch AND includes directional information via R(q). **Use L alone; T is redundant.**

---

## micro24 — Nudge sensitivity

**Question:** How does attitude error at the shared node and endpoints degrade L-consistency?

### Part A: Shared node (q_mid) nudge

**Completely invariant.** Nudging q_mid has **zero effect** on the L-error ranking. This is a mathematical identity:

```
||DL|| = ||R(q_mid) @ I @ (omega_arr - omega_dep)|| = ||I @ (omega_arr - omega_dep)||
```

Since R(q) is orthogonal and the SAME R appears in both L vectors, it cancels in the norm. The L-consistency check is equivalent to checking omega matching in the body frame, weighted by I. **Shared node attitude errors are irrelevant.**

### Part B: Endpoint (q_A, q_B) nudge

Nudging endpoints changes the bridging omega by approximately delta_omega ~ rotvec(dq) / dt.

| Endpoint nudge | P(correct) | Avg rank | Gap (kg*m^2/s) | True pair \|\|DL\|\| |
|----------------|------------|----------|----------------|---------------------|
| 0.5 deg | 1.00 | 1.0 | 112.3 | 0.53 |
| 1.0 deg | 1.00 | 1.0 | 111.3 | 1.21 |
| 2.0 deg | 1.00 | 1.0 | 109.6 | 2.39 |
| 3.0 deg | 1.00 | 1.0 | 107.6 | 3.51 |
| 5.0 deg | 1.00 | 1.0 | 104.3 | 5.67 |
| 7.0 deg | 1.00 | 1.0 | 101.5 | 7.31 |
| 10.0 deg | 1.00 | 1.0 | 95.2 | 11.9 |

**100% correct at all nudge levels.** True pair DL grows linearly at ~1.2 kg*m^2/s per degree. The gap degrades from 113 to 95 at 10 deg — still 8x the true pair's error. Extrapolating, L-consistency would break at ~80+ deg endpoint error, far beyond any realistic scenario.

**Tolerance estimate:** The gap-to-noise ratio at 10 deg endpoint error is 95/12 = 8x. The filter is practical with attitude candidates as coarse as 10 deg.

---

## micro25 — Three-leg test

**Question:** Does a 3rd leg (4 peaks, 2 shared nodes) improve discrimination?

Found a 4th peak at epoch 15. With 4 peaks [15, 183, 260, 360] and 3 legs:

| Test | Candidates | True rank | Gap |
|------|-----------|-----------|-----|
| 2-leg (node B only) | 64 pairs | 1/64 | 113.2 |
| 3-leg (nodes B + C) | 512 triples | 1/512 | 113.2 |

The 3rd leg provides an additional constraint but was **not needed** for this test case — the 2-leg filter already achieves perfect discrimination. The 3-leg test confirms consistency: the true triple (3,3,3) has score = 1.4e-07, and adding the third node didn't change the gap. The additional constraint would be valuable if winding candidates were closer together (smaller dt, more ambiguous windings).

---

## Key Physics Insight

L-conservation at a shared node is equivalent to checking whether the body-frame omegas match (weighted by I):

```
||DL|| = ||R(q) @ I @ (omega_leg0_arrived - omega_leg1_departing)|| = ||I @ delta_omega||
```

Since R is orthogonal, only the omega mismatch matters. Different windings produce omega differences of ~delta_omega ~ 2*pi/dt ~ 0.01 rad/s, which yields ||I @ delta_omega|| ~ 100 kg*m^2/s. This is enormous compared to propagation errors (~1e-8) or endpoint attitude errors (~1 per degree).

The winding spacing (2*pi/dt) divided by the attitude-error-induced omega noise (delta_q/dt) gives the signal-to-noise ratio. Both scale as 1/dt, so the ratio is approximately 2*pi/delta_q — independent of leg duration. For delta_q = 10 deg: SNR ~ 2*pi/0.175 ~ 36x. This explains the extreme robustness.

---

## Conclusion

**L-conservation is a viable winding filter.** It is:

1. **Perfectly discriminating** with oracle attitudes (9 orders of magnitude gap)
2. **Robust to shared-node attitude errors** (mathematically invariant — R cancels)
3. **Robust to endpoint attitude errors** up to at least 10 deg (100% correct, gap = 95)
4. **Simple to implement** — just compute L = R(q) * (I * omega) at each shared node
5. **Independent of fidelity** — does not require lightcurve evaluation, only quaternion propagation

**The blocking problem is not L-consistency itself, but ensuring the staircase on each leg finds the correct winding.** micro18 failed because leg 1's staircase jumped over the true winding. The fix is to ensure denser staircase coverage, or to run the staircase with better initial guesses that don't miss windings.

**Next step:** Fix the leg-1 staircase (smaller barrier gaps, adaptive stepping, or bidirectional search from both ends) so that it reliably includes the true winding, then apply L-conservation as the winding selector.
