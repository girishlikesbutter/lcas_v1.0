# Series 08: Integration Pipeline Findings

## Micro-26: End-to-end integration with oracle attitudes

**Question:** Does wiring band-sweep enumeration (micro20) + L-conservation filter (micro23)
recover the correct omega pair when oracle attitudes are provided at 3 peaks?

**Setup:**
- 3 oracle peaks at epoch indices [183, 260, 360], dt_0=555.5s, dt_1=721.4s
- Band-sweep: 13 bands × 0.5 deg/s from [0, 6.5], 10 random-start L-BFGS-B per band
- Convergence: arrival_err < 1e-6, dedup by |omega| within 0.05 deg/s
- L-conservation: score all (leg0 × leg1) pairs by ||L_arriving - L_departing|| at peak B

**Results:**
- Leg 0: 34 candidates (13 bands non-empty)
- Leg 1: 32 candidates (14 bands non-empty, extra from solutions crossing band edges)
- True pair rank: **423/1088** — L-conservation completely fails to identify it
- Best pair: (k=0, j=0) — lowest-magnitude candidates (~0.3 deg/s)
- Best omega error: 2.37 / 2.28 deg/s, direction error: 155 / 140 deg
- Runtime: 2291s (~38 min, over 15-min target)

**Root cause:** Magnitude-only deduplication is fundamentally incompatible with L-conservation filtering.

At each |omega| magnitude, the bridge problem has multiple solutions with different omega
directions. The dedup keeps one arbitrary solution per magnitude band. The kept solution
at the true magnitude (~2.08 deg/s) has the wrong direction, so its angular momentum
vector L does not match between legs.

In micro23, this worked because candidates were constructed along the true omega direction
(winding number enumeration adds/subtracts full revolutions along omega_hat). Here,
random-start bridge solves find arbitrary directions.

**Implications for pipeline design:**
1. Must keep **all** directionally-distinct bridge solutions, not just one per magnitude
2. Or: seed bridge solves more intelligently (e.g., use multiple fixed directions per band)
3. L-conservation filter is still valid physics — the dedup strategy defeats it
4. Runtime concern: 130 bridge solves/leg × ~8s each ≈ 17 min/leg with Pool(8).
   Keeping all directions would multiply candidate count significantly.

**Next steps (from micro26):**
- micro26c: Direction-aware dedup → see below
- micro27: Bridge screening feasibility → infeasible (11s/solve, no cheap alternative)
- micro28: Nudged attitudes → PA-mode fundamentally wrong, v4 incomplete

---

## Micro-26c: Direction-aware dedup in full pipeline

**Question:** With direction-aware dedup (|ω| within 0.05 deg/s AND angular distance < 5°), does L-conservation identify the correct omega pair?

**Setup:** Same as micro26, but dedup now requires both magnitude AND direction proximity to merge candidates. 10 random starts per band, 13 bands.

**Results:**
- Leg 0: 54 candidates (vs 34 with magnitude-only dedup)
- Leg 1: 49 candidates (vs 32)
- Leg 0: true omega found exactly (0° error)
- Leg 1: true omega **NOT found**. Nearest candidate at |ω|≈2.07 deg/s has **125° direction error** — a completely different omega vector.
- Total pairs: 2646. "True pair" (leg0 exact × leg1 nearest) has L_err=1518 — meaningless since leg 1's candidate is wrong.
- Runtime: 2039s

**Root cause:** The dedup fix is correct and working — the problem is upstream. At dt=721s (~4 full revolutions), the bridge problem has many directionally-distinct local minima at each |ω|. 10 random starts per band is insufficient to discover the narrow basin containing the true omega direction. Leg 0 (dt=555s, ~3 rev) finds truth exactly with the same 10 starts, suggesting the number of basins grows with dt.

**Comparison with micro26:**
- micro26 (mag-only dedup): 34/32 candidates, rank 423/1088
- micro26c (direction-aware): 54/49 candidates, true omega absent from leg 1
- The dedup bug is fixed. The new bottleneck is bridge solver coverage.

**Implication:** Increasing n_starts per band is the direct fix. Cost: ~1.4s per solve with Pool(8). 50 starts = 15 min/leg, 100 starts = 30 min/leg. Need to sweep n_starts to find the minimum for reliable coverage.

---

## Micro-29: Peak-shape omega filter

**Question:** Can a peak-shape consistency check (brightness is a local max at the peak epoch) filter wrong-winding candidates?

**Method:** For each candidate omega at a peak:
1. Compute brightness at peak epoch (using oracle attitude)
2. Propagate omega ±1 epoch (7.2s) to get neighboring attitudes
3. Compute brightness at peak±1
4. Pass if both neighbors are dimmer (higher magnitude) than the peak
5. Test at both start and end peaks of each leg; require both to pass

**Results:**
- Leg 0: 27 candidates, 24 pass combined (kill rate 11%)
- Leg 1: true omega not found (same issue as micro26c — only 5 starts per band here)
- Killed candidates are mostly low-frequency (|ω| < 1 deg/s)
- True omega survives on leg 0

**Verdict:** Too weak. At ±1 epoch (7.2s), even wrong-winding omegas barely rotate the satellite, so the peak shape is preserved. Not useful as a standalone pre-filter.

---

## Micro-30: Hi-fi pruning of lo-fi iso-brightness candidates

**Question:** Does hi-fi (shadow) evaluation prune lo-fi iso-brightness candidates, and does the true candidate survive?

**Method:** At each of 3 peaks, generate 1000 random iso-brightness seeds via L-BFGS-B (lo-fi), dedup, then evaluate each at hi-fi. Score by |m_hifi - m_target|. Sweep thresholds {0.2, 0.1, 0.05, 0.025, 0.01} mag.

**Results:**

| Peak | n_unique | Nearest to truth | Hi-fi resid (mag) | Survives 0.05? |
|------|----------|------------------|--------------------|----------------|
| 183  | 691      | 6.94°            | 0.242              | No             |
| 260  | 680      | 5.86°            | 0.109              | No             |
| 360  | 918      | 2.69°            | 7e-6               | Yes (all)      |

- Hi-fi pruning removes 20-33% of candidates depending on threshold
- At peak 360 (nearest 2.69°, within basin), hi-fi residual is essentially zero — truth survives all thresholds
- At peaks 183/260 (nearest 5.9-6.9°, outside basin), hi-fi correctly kills these wrong candidates

**Verdict:** Inconclusive. 1000 seeds is too few — we already know from micro10 that 10K seeds bring nearest within ~1° at all peaks. The failures at peaks 183/260 are seed coverage failures, not hi-fi pruning failures. Needs re-run with 10K seeds to properly evaluate.

---

## Summary: Bridge Solver Coverage is THE Bottleneck (2026-03-12)

All three experiments (micro26c, micro29, micro30) point to the same conclusion: **the pipeline's blocking failure mode is that the bridge solver cannot reliably find the true omega direction on longer legs.**

| Leg | dt (s) | Revolutions | 10 starts/band | True ω found? |
|-----|--------|-------------|-----------------|---------------|
| 0   | 555    | ~3          | Yes             | Exactly (0°)  |
| 1   | 721    | ~4          | No              | 125° error    |

The number of directionally-distinct local minima grows with dt. More random starts is the direct fix, with linear cost scaling. Next experiment: sweep n_starts = {10, 25, 50, 100} on leg 1.
