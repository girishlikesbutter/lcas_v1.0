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

**Next steps:**
- micro27: Bridge screening with direction-aware dedup or no dedup
- micro28: Nudged (non-oracle) attitudes to test sensitivity to attitude errors
