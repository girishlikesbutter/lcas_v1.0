# Series 11 — Glint-Constrained Inversion Pipeline

> **Status: MULTI-SEED VALIDATED.** m073 alpha pipeline: 4/6 seeds recover omega to < 5° direction error, 1/6 fully resolves attitude. 180° twin degeneracy confirmed fundamental (m071e). Pipeline runs in 4.5–7.4 min per seed.

## m060 — L-parameterization basin test

**Question:** Is the convergence basin wider with L-parameterization (q0, L_inertial) vs omega-parameterization (q0, omega0)?

**Answer: NO.** L-parameterization is slightly worse at small angles (0.5-5°) and comparable at large angles. Both fail at 0.5° — only 4/20 (omega) and 2/20 (L) converge at the 5× baseline threshold. The narrow basin is fundamental to the LC physics, not the parameterization.

Key number: angle between omega_dir and L_dir = 61° (due to triaxial inertia), yet basin widths are comparable.

**L-parameterization is a dead end.**

---

## m061 — Basin width vs observation window length

**Question:** Does a shorter observation window produce a wider convergence basin?

**Answer: YES — dramatically.**

### Omega-only basin (q0 fixed at truth, L-BFGS-B):

| Window | From 2° | From 5° | From 10° | From 20° | Effective basin |
|--------|---------|---------|----------|----------|----------------|
| 180s (25 ep) | 5/5 | 3/5 | 3/5 | 0/5 | ~10° |
| 360s (50 ep) | 4/5 | 4/5 | 1/5 | 0/5 | ~5° |
| 720s (100 ep) | 5/5 | 3/5 | 0/5 | 0/5 | ~5° |

The 180s window has a **~10° omega direction basin** — 5× wider than the full 3600s window.

### Raw residual scaling (Part A):

The residual ratio grows with window length. At 2° omega perturbation:
- 180s: 10.7× baseline
- 360s: 49.8× baseline
- 3600s: 16.2× baseline

The ratio is not monotonic because baselines differ (lo-fi/hi-fi mismatch accumulates), but the absolute residual penalty for a given perturbation grows with window length, confirming the error-compounding mechanism.

---

## m062 — Joint 6-DOF basin on short windows

**Question:** Does the JOINT (q0 + omega) basin also widen on short windows?

**Answer: YES, but more constrained than omega-only.**

### 180s window — joint 6-DOF convergence (L-BFGS-B):

| q0 pert | ω pert | Converged |
|---------|--------|-----------|
| 5° | 5° | **5/5** |
| 5° | 10° | **4/5** |
| 10° | 5° | **4/5** |
| 10° | 10° | 2/5 |
| 20° | 5° | **3/5** |
| 20° | 10° | 1/5 |
| any | 20° | 0/5 |

**Joint basin at 180s: q < ~10° AND ω < ~10° for reliable convergence.** With q well-known (< 5°), omega basin extends to ~10°. With omega well-known (< 5°), q basin extends to ~20°.

### 360s window — already narrower:

Only 3/5 converge from q±5° + ω±5°. The 360s window is too narrow for the initial grid search — **180s is the right starting window.**

### Progressive estimation (q±20°, ω±10° → extend) — BREAKTHROUGH:

**Full results (10 trials):** 2/10 converge to sub-degree accuracy, 1 more converges partially (1.4° final). The 2 successful trials show the mechanism working perfectly:

**Trial 7 (start q±20°, ω±10°):**
```
T=  180s | q_err=  4.2° | ω_err= 1.5°  ← first window converges
T=  360s | q_err=  1.4° | ω_err= 0.3°  ← sharpens
T=  720s | q_err=  0.4° | ω_err= 0.1°  ← near-perfect
T= 3600s | q_err=  0.4° | ω_err= 0.0°  ← CONVERGED
```

**Trial 8:**
```
T=  180s | q_err=  9.4° | ω_err= 2.5°
T=  360s | q_err=  6.3° | ω_err= 5.4°  ← stumbled on 360s
T=  720s | q_err=  2.5° | ω_err= 0.3°  ← 720s recovers!
T= 3600s | q_err=  1.4° | ω_err= 0.1°  ← CONVERGED
```

**The architecture is proven.** Progressive windowed estimation works when the first window captures the correct basin. The 2/10 rate from random starts can be increased with better seeding (phi-sweep for q, grid for ω, magnitude constraint).

---

## Key Insight: The Pipeline Architecture

The findings suggest a concrete pipeline:

1. **Estimate |omega|** from peak count (13% error — validated, m052)
2. **At earliest specular glint** (mag < 6): phi sweep gives 360 candidate q's at the glint epoch
3. **Grid search omega direction** on Fibonacci sphere (~100-500 points, ~5-10° spacing)
4. **Evaluate 180s window** (25 epochs, centered on glint) for all 360 × 100 = 36,000 (q, ω) pairs
   - Cost: 36,000 × ~1ms (25-epoch lo-fi) = ~36s
5. **Top-10 → L-BFGS-B** on 180s window (joint 6-DOF)
6. **Progressive extension:** 180s → 360s → 720s → 1800s → 3600s
7. **Best final solution** across survivors

**Total estimated cost:** ~5-10 minutes.

**Required accuracy for step 4 to work:**
- q within ~10° (phi sweep gives ~2° with known omega, maybe ~5-10° with approximate)
- ω direction within ~10° (100-point Fibonacci gives ~10° spacing — marginal, 500 points gives ~5°)

---

## m063 / m063b — Grid search + progressive/multi-start estimation

### m063: Progressive extension from single best candidate — FAILS

The 180s grid search (200 dirs × 10 q candidates) finds the correct omega at rank #5 (3.3° error). But the 180s window can't discriminate: false candidates at 70-90° get better residuals than the correct omega. After L-BFGS-B refinement on 180s, the best-by-residual candidate has 81° omega error. Progressive extension from this wrong candidate fails catastrophically.

### m063b: Multi-start on 720s window — IN PROGRESS

Grid search (500 dirs × 10 q): correct omega at 1.3° error, ranked #26/500 on 180s window. In top-20: one candidate at 4.4° (#19) and one at 11.3° (#12). Multi-start L-BFGS-B on 720s window from all 20 candidates: results for 18/20 show all converge to wrong local minima (ω_dir 16-88°, ω_mag errors 2-2800%). Candidate #19 (the close one) still running.

### The fundamental tension

| Window | Basin width | Discrimination power |
|--------|-----------|---------------------|
| 180s | ~10° | **Poor** — correct omega at rank #19-26/500 |
| 360s | ~5° | Moderate (not tested at grid scale) |
| 720s | ~5° | Better but L-BFGS-B finds local minima from far starts |
| 3600s | ~2° | Strong but unusable basin |

Short windows widen the basin but lose the ability to identify the correct solution. Long windows can discriminate but the basin is too narrow to reach. **This is a fundamental trade-off, not a parameter tuning problem.**

### Remaining hope

If m063b candidate #19 (starting at 4.4° from truth) converges correctly on the 720s window with the lowest residual, then the architecture works — but requires the correct omega to be in the top-20 of the 180s grid, which needs a top-N of ~30+ for this test case. The residual-based selection on the 720s window is the real discriminator, not the 180s grid.

### What's needed next

1. ~~omega magnitude constraint~~ — addressed by geometric refinement
2. ~~Glint-anchored initialization~~ — addressed by phi sweep
3. ~~CasADi multiple shooting~~ — not needed, glint geometry sufficient
4. ~~Hi-fi short-window evaluation~~ — full-curve hi-fi affordable for 10 candidates

---

## m069 / m069b — Bug fixes + geometric refinement (2026-03-25)

### Three bugs fixed from m068:

1. **Phi sweep scored all 10 normals** instead of ±X only at specular epochs. Grid search correctly used ±X; phi sweep was inconsistent. Fix: restrict to `max(dot(+X, pb), dot(-X, pb))`.

2. **Phi sweep used 36 bins (10°)** instead of 360 (1°). This introduced ~5° phi error at anchor, compounding to 32° q0 error after back-propagation. Fix: use 360 bins.

3. **Windowed evaluation time-offset bug.** `ObjectiveFunction.evaluate()` places initial state at `times[0]`. With shifted times (anchor=0), `times[0]` is negative, so the state is placed ~195s before the anchor. Fix: back-propagate to window start, or use forward-only windows, or propagate from t=0 with original obs_times.

### m069 — Fixed phi sweep + full hi-fi scoring

With 360 phi bins and ±X-only scoring, full-curve hi-fi correctly selects truth at rank #1/10 (residual 2.37 vs 3.60 for best false positive). Pre-refinement: q0=19.1°, ω=1.5°.

### m069b — Geometric refinement

**Question:** Can a purely geometric cost (no BRDF, no shadows) refine from 19° to sub-degree?

**Answer: YES.**

Cost function: specular alignment (±X, weight 10) + bright alignment (any normal, weight 5) at 19 peak epochs. Propagate from (q0, ω0) at t=0 using original obs_times. L-BFGS-B, 1050 evals, 50s.

| | q0 error | ω direction | ω magnitude |
|---|---------|------------|-------------|
| Before | 19.1° | 1.5° | -0.0% |
| After | **1.94°** | **0.14°** | **-0.03%** |

**Key finding:** anti-glint constraints are INVALID. "Alignment → glint" is wrong because shadows suppress glints even with perfect alignment. Only "glint → alignment" is valid. The geometric cost must use positive constraints only (where we see peaks, we require alignment).

---

## m070 — Full end-to-end pipeline (2026-03-25)

**THE WORKING PIPELINE.** Blind inversion on seed 93, +X anchor, 7.8 minutes.

### Pipeline steps:

| Step | What | Time | Result |
|------|------|------|--------|
| 1 | Peak count → \|ω\| estimate | 0.0s | 1.193 dps (true: 1.283, -7%) |
| 2 | Grid search (2000 dirs × 20 mags, 16 cores) | 134.5s | Truth at rank #13 |
| 3 | NM refinement (top 50, 16 cores) | 32.1s | Truth at rank #1, 1.8° |
| 4 | Phi sweep (360 bins, ±X only) | 0.4s | 10 candidates |
| 5 | Full hi-fi (10 cands, 8 cores) | 197.1s | Truth at rank #1, q0=19.1° |
| 6 | Geometric refinement (L-BFGS-B) | 49.3s | **q0=1.94° ω=0.14°** |
| **Total** | | **7.8 min** | **SUCCESS** |

### What's next:

1. **±X disambiguation:** run with ANCHOR_GROUP=1 (-X), compare hi-fi residuals
2. **Multi-seed validation:** run on 5-10 seeds from m046 dataset
3. **Edge cases:** seeds with few specular glints, very slow/fast tumblers
4. **Articulated components:** current pipeline assumes static solar panels

---

## m071 — Geometric selection failure + 180° twin degeneracy (2026-03-25)

### m071: Geometric cost selects wrong candidate

**Question:** Can geometric refinement (L-BFGS-B on specular ±X + bright alignment cost) SELECT the correct candidate without hi-fi?

**Answer: NO.** The candidate with the lowest geometric cost has **88.9° omega error**. The true candidate (1.6° omega error) ranks #5-6 by geometric cost.

| Candidate | geo_cost | ω error |
|-----------|----------|---------|
| ω#2 (wrong) | 0.001200 | 88.9° |
| ω#3 (truth) | 0.002215 | 1.6° |

Geometric cost is degenerate — multiple omega directions produce similar specular/bright alignment patterns.

### m071b: Wrong winner light curve confirmation

The geometrically-favored wrong winner produces **RMS 8.3 mag residual** vs truth — clearly distinguishable by hi-fi LC. Confirms hi-fi is necessary for final selection.

### m071c-d: IS-901 symmetry analysis

IS-901 has near-perfect 180° rotational symmetry about the +X body axis. Solar panels are symmetric; only antennas/dishes break symmetry (centroid mismatch < 0.01 m). This explains the phi+180° degeneracy.

### m071e-f: The fundamental optical twin

**Key finding:** Rotating the true attitude by 180° about +X in the body frame (and transforming omega accordingly) produces **IDENTICAL** light curves.

- True ω_body: [0.5, -0.3, 2.0] deg/s
- Twin ω_body: [0.5, +0.3, -2.0] deg/s (y,z signs flip)
- RMS LC difference: **6×10⁻⁶ magnitude** — numerically indistinguishable
- |ω| preserved, direction error ~0° within the twin pair

**Implication:** The 178.9° q0 error seen in m070 is not a bug — it's the satellite's optical twin. Any single-observer, broadband photometric pipeline will face this ambiguity for satellites with approximate 180° symmetry about a body axis.

---

## m072 — MVP pipeline + twin investigation (2026-03-27)

### m072: Streamlined pipeline (seed 93)

Implements MICRO72_SPEC: bright constraints in grid/NM, vectorized phi loops, geometric pre-filtering with cluster detection, composite scoring.

| Step | Time | Result |
|------|------|--------|
| Grid search (2000 × 20 × 36) | 98.5s | Grid#3 at 4.3° |
| NM refinement (top 20, 360 phis) | 14.1s | Truth at 1.6° (ω#3) |
| Geometric refinement (10 cands) | 63.9s | 6 in low-cost cluster (74.7× gap) |
| Hi-fi LC (6 cands) | 154.0s | Winner: ω#3 +X, 1.2° dir |
| **Total** | **330.7s (5.5 min)** | **ω correct, q0=178.9° (twin)** |

Pipeline correctly recovers omega (1.25° direction, -0.01% magnitude) but selects the degenerate twin attitude.

### m072b: Twin animation

Visual confirmation via 3D Plotly animations: true and twin satellites are indistinguishable from the observer's perspective, despite 180° different body-frame orientations.

### m072c: Peak count filtering

Non-glint peak count (mag > 9.0) cannot break the twin degeneracy. The twin with correct peak count (13 = observed) has worse hi-fi MSE (0.2087 vs 0.0306). Composite scoring with alpha up to 5.0 still favors the lower-MSE twin.

---

## m073 — Multi-seed alpha pipeline (2026-03-27)

**Question:** How robust is the pipeline across different trajectories?

Ran on 6 seeds from the m046 realistic-tumble dataset:

| Seed | |ω| true | q0 err | ω dir err | ω mag err | Time | Status |
|------|---------|--------|-----------|-----------|------|--------|
| 000 | 1.42 | **4.4°** | **1.7°** | +0.1% | 4.5m | **FULL SUCCESS** |
| 014 | 1.23 | 5.2° | 2.2° | -0.0% | 6.7m | PARTIAL (q ~5°) |
| 027 | 1.14 | 129.8° | 42.8° | -9.6% | 6.7m | **FAILED** |
| 036 | 1.44 | 176.3° | 5.5° | +0.2% | 6.6m | PARTIAL (twin + 5.5° ω) |
| 074 | 1.09 | 178.6° | **2.0°** | -0.1% | 7.4m | ω SUCCESS (twin) |
| 093 | 1.28 | 178.9° | **1.2°** | -0.0% | 5.5m | ω SUCCESS (twin) |

### Summary:
- **4/6 recover omega direction** to < 5° (seeds 0, 14, 74, 93)
- **1/6 fully resolves attitude** to < 5° (seed 0)
- **1/6 total failure** (seed 27 — 42.8° omega error)
- 180° attitude twin appears in 3/6 seeds (expected)
- Runtime: **4.5–7.4 min** per seed

### Failure analysis (seed 27):
Grid search failed to rank the correct omega direction in the top 20. Likely cause: unfavorable geometry (viewing angle or tumble axis alignment reduces discriminating power of specular constraints).

### What's next:
1. **Diagnose seed 27 failure** — check specular count, grid ranking of true omega
2. **Break the twin degeneracy** — multi-epoch polarimetry, or accept ±180° as inherent
3. **Extend to more seeds** — need 10+ for reliable success rate statistics
4. **Report preparation** — pipeline is mature enough for a technical report
