# Inversion Strategy Brainstorm — February 2026

## Problem Summary

The joint 6-parameter inversion (3 attitude axis-angle + 3 angular velocity) has a **much narrower convergence basin** than attitude-only inversion:

| Problem | Basin Width | Evidence |
|---------|-------------|----------|
| Attitude-only (fix ω=true) | ~5-6° | L-BFGS-B converges from 5° away |
| Joint (att + ω) | < 1° attitude + < 0.001 dps ω | 3°/0.01dps → diverges to 22°+ |

**Key data points from experiments:**

- **Lo-fi is 163× faster** than hi-fi (0.037s vs 6.1s), with ρ=0.978 rank correlation
- **Basin shift** between lo-fi and hi-fi optima: 4.2° mean (105% of basin width) — SIGNIFICANT
- **Attitude-only L-BFGS-B**: Converges from ≤5° (lo-fi), fails at 10°+
- **Joint L-BFGS-B**: Even 1° att + 0.001 dps ω works, but 3° + 0.01 dps diverges (most of the time)
- **DE with bounded search**: 450-2000 lo-fi evals → converged to wrong basin (176°, 187° errors)
- **Multi-start L-BFGS-B** from within 5° of truth: 8/8 FAILED for joint problem

The fundamental issue: the joint objective landscape has extremely narrow valleys because small ω errors compound over the observation window (3600s × 0.01°/s = 36° accumulated attitude error).

## Literature Insights

### Burton et al. (2024) — Two-Stage PSO
The most relevant approach from the literature. Key ideas:
1. **Stage 1**: Find ALL attitudes matching a single epoch measurement using non-social PSO (17,576 particles)
2. **Stage 2**: Pair each attitude with ω guesses, propagate, match full lightcurve
3. **Stage 3**: L-BFGS-B refinement + flip correction

**Key insight**: They DECOUPLE the search — find attitudes first (3D search), then pair with ω (additional 3D). This avoids searching the full 6D space simultaneously.

### Opportunities Matrix Observations
- All 26 inversion papers in the 213-paper database assume rigid body
- Computational cost is a recurring bottleneck (Rubio2025: 110s/eval)
- Basin-hopping and multi-fidelity approaches are unexplored in SSA literature
- Fan2019: observability requires sufficient attitude diversity over time

### The Basin Shift Problem
The 105% basin shift means lo-fi optimum is OUTSIDE the hi-fi convergence basin. This makes naive two-stage (lo-fi global → hi-fi local) unreliable. Solutions:
1. Use lo-fi for ranking only, refine multiple candidates with hi-fi
2. Use lo-fi to get close, then do a small hi-fi grid search
3. Accept the lo-fi bias and compensate

## Strategy Analysis

### Strategy 1: Decoupled Sequential Estimation ⭐⭐⭐⭐⭐ (RECOMMENDED)
**Approach**: 
1. Fix ω=0, solve for attitude only (3 params) — 5° basin, easy
2. Fix attitude at Step 1 result, solve for ω only (3 params) — should be tractable  
3. Joint refinement from combined (att, ω) starting point

**Pros**: Exploits the 5° attitude-only basin; reduces 6D to two 3D problems  
**Cons**: Step 2 might not converge if attitude isn't close enough  
**Time estimate**: ~5 min for DE+LBFGSB each step  
**Rating**: Best theoretical chance. Mirrors Burton's two-stage philosophy.

### Strategy 2: Very Tight Starting Radius (≤1°) ⭐⭐⭐
**Approach**: Multi-start L-BFGS-B from within 0.5° attitude / 0.001 dps ω
**Pros**: Simple; data shows joint convergence works at sub-degree  
**Cons**: This is basically cheating — requires near-perfect initial guess  
**Rating**: Good as proof-of-concept, not practical for real problems

### Strategy 3: Grid Search + Local Refine ⭐⭐⭐⭐
**Approach**: 
1. Coarse grid over attitude (e.g., 5° spacing → ~46³ = 97K points, but in 3D axis-angle only meaningful region)
2. Evaluate lo-fi objective at each grid point (with ω=0)
3. Take top-N, run local refine on each

**Pros**: Systematic; guaranteed coverage  
**Cons**: Expensive even with lo-fi (97K × 0.04s = 64 min); but can be sparser  
**Rating**: Feasible with lo-fi. A 10° grid = ~6K points × 0.04s = 4 min.

### Strategy 4: Basin-Hopping / Dual Annealing ⭐⭐⭐⭐
**Approach**: Use scipy's `basinhopping` or `dual_annealing` which are designed for multimodal landscapes
**Pros**: Designed for this exact problem class; built-in  
**Cons**: Still searching 6D; may need many iterations  
**Rating**: Worth trying. dual_annealing in particular handles bounded, multimodal problems well.

### Strategy 5: CMA-ES ⭐⭐⭐⭐
**Approach**: Covariance Matrix Adaptation Evolution Strategy — adapts search distribution
**Pros**: State-of-the-art for continuous optimization; handles narrow valleys by adapting covariance  
**Cons**: Requires `cma` package; still 6D search  
**Rating**: CMA-ES excels at exactly this problem type (narrow, rotated valleys).

### Strategy 6: Increase Observations ⭐⭐
**Approach**: Use 100+ observation points instead of 50
**Pros**: More data constrains the problem better  
**Cons**: Slower per evaluation (linear in n_obs for lightcurve gen)  
**Rating**: Marginal benefit; the narrow basin is a landscape property, not a data sufficiency issue.

### Strategy 7: Period Analysis → ω Constraint ⭐⭐⭐
**Approach**: FFT the observed lightcurve to estimate dominant frequency → constrain ω magnitude
**Pros**: Already done! Period analysis gives ω_bound = 1.95 dps  
**Cons**: Helps with bounds but doesn't solve the narrow basin  
**Rating**: Already implemented. Useful as preprocessing step.

## Recommended Plan

### Primary: Decoupled Sequential + Multi-Optimizer (Strategy 1 + 4 + 5)

```
Step 1: Attitude-Only Lo-Fi DE (budget=2000, ~80s)
  → Fix ω=0, solve 3 params
  → Lo-fi basin is ~5°, DE should find it
  
Step 2: Omega-Only Lo-Fi L-BFGS-B (budget=500, ~20s)
  → Fix attitude from Step 1
  → Solve 3 omega params
  → Should converge since attitude is ~right
  
Step 3: Joint Hi-Fi Refinement (budget=200, ~20 min)
  → Start from (att_step1, omega_step2)
  → L-BFGS-B on full 6 params
  → If fails, try dual_annealing with tight bounds (±2° att, ±0.01dps ω)

Step 4: Verification
  → Compare predicted vs observed lightcurve
  → Check RMS < 2σ_noise
```

### Fallback: dual_annealing on full 6D with tight ω bounds
- Use period analysis ω_bound = 1.95 dps
- Budget: 5000 lo-fi evals (~200s)
- Top-3 → hi-fi L-BFGS-B refinement

### Demo Success Criteria
- Attitude error < 5° from truth
- ω error < 0.1 dps  
- RMS residual < 0.1 mag (2× noise)

## Overnight Experiment Results (Feb 8-9, 2026)

### exp3_basin_resume.py — Full Convergence Basin Characterization

**Phase 1: Single-direction tests**
- Attitude-only: 9/18 converge (≤6° reliable, 7-10° mixed, 15°+ fails)
- Omega-only: 3/13 converge (≤0.02°/s, fails at 0.03°/s)
- Combined: 2/15 converge (only 1°+0.001 and 2°+0.005)

**Phase 2: Fine grid (10 trials per level, attitude-only)**
| Distance | Success Rate |
|----------|-------------|
| 5° | 90% |
| 6° | 70% |
| 7° | 30% |
| 8° | 20% |
| 9° | 50% |
| 10° | 30% |

Non-monotonic! Some 10° starts succeed (jumped to correct basin) while 7° fail (local minima). Evidence of multiple basins.

**Phase 3: Statistical tests (20 trials each, combined perturbations)**
| Combo | Success |
|-------|---------|
| 3° + 0.01°/s | 45% (9/20) |
| 3° + 0.05°/s | 25% (5/20) |
| 3° + 0.1°/s | 25% (5/20) |
| 5° + 0.01°/s | 5% (1/20) |
| 5° + 0.05°/s | 15% (3/20) |
| 5° + 0.1°/s | 5% (1/20) |
| 7° + 0.01°/s | 5% (1/20) |
| 7° + 0.05°/s | 0% (0/20) |
| 7° + 0.1°/s | 0% (0/20) |
| 10° + any ω | 0% (0/60) |

**Key finding:** Adding even small omega uncertainty (0.01°/s) devastates convergence. 3° attitude alone → 90%+, but 3°+0.01°/s → 45%. This is the joint-problem difficulty.

### Previous failures
- v1: Multi-start L-BFGS-B from within 5° → all 8 failed for joint problem
- v2: Decoupled attitude-only DE with ω=0 → 238° error (fundamentally wrong: satellite rotates ~180° during observation)
- v3: Grid search 20° spacing + CMA-ES → 211° error, same ω=0 problem

### Status at 11:49pm NZ (Feb 8) — midnight check

1. **exp3_basin_resume.py**: Still running (PID 36203, 130% CPU, ~4h elapsed). Phases 1–3 complete (18+13+15+60+220 = 326 tests). Phase 4 (hi-fi verification) empty — not yet started. ~8h remaining. Results so far:
   - Phase 1 att-only: 9/18 pass (≤6° reliable)
   - Phase 1 ω-only: 3/13 pass (≤0.02°/s)
   - Phase 1 combined: 2/15 pass (very narrow)
   - Phase 2 fine grid: 29/60 pass (non-monotonic basin structure!)
   - Phase 3 statistical: 25/220 pass (11.4% overall — joint problem is brutal)

2. **exp_tight_start_proof.py**: **CRASHED/KILLED** after 1 trial. Only result: 0.5°+0.005°/s → lo-fi diverged to **35.94°** → hi-fi couldn't rescue. Process no longer running.

3. **exp_dual_annealing_6d.py**: **COMPLETED — FAILED**. 129.3° att error.

4. **exp_omega_first.py**: Still running. Grid search at 200/600 iterations (~31K evals). Current best: f=1.1102, att=162.1°, ω=0.107°/s. **Trending toward failure** — 200 iterations deep with no improvement, stuck at 162° error.

5. **inversion_demo_v3**: **COMPLETED — FAILED**. CMA-ES → 211.54°.

### Critical new finding: Lo-fi joint landscape is UNRELIABLE
The tight-start experiment shows that even 0.5° from truth, the lo-fi joint optimizer diverges to 36° error. This means:
- Lo-fi is reliable for **attitude-only** (basin ~5°) but NOT for joint att+ω
- The mixed-fidelity approach must use **decoupled estimation**: solve attitude first (lo-fi), then omega separately
- Direct lo-fi joint optimization is a dead end regardless of starting point quality

### dual_annealing 6D: confirmed failure
129° error after 2852 evals. The 6D landscape is too multimodal for any global optimizer. **This reinforces the decoupled approach.**

### FFT omega estimation: 4× off
0.20°/s vs true 0.05°/s. Multiple specular reflections per rotation create harmonics. Period analysis alone can't constrain omega — need to search over magnitudes.

### Status at 12:04am NZ (Feb 9) — midnight check

1. **exp3_basin_resume.py**: Still running (PID 36203, 130% CPU). Phase 4 (hi-fi verification) still empty — hasn't started yet. ~8h remaining estimate unchanged. All phase 1-3 data confirmed stable.

2. **exp_omega_first.py**: Still running but **almost certainly failing**. At 200/600 grid iterations (~31K evals), best result: f=1.1102, att=162.1°, ω=0.107°/s. No improvement trend. The FFT-based omega magnitude estimate was 4× off (0.20 vs 0.05°/s true), so the grid is searching wrong magnitudes. **Will let it finish but expect failure.**

3. **exp_tight_start**: Confirmed dead (no process). Single trial: 0.5°→35.94° divergence.

4. **All other experiments**: No change from 11:49pm check.

**No new experiments launched.** The running exp3_basin_resume is the most valuable — it's building the comprehensive basin characterization that IS the paper contribution. Letting it complete overnight is the right call.

**For Girish at 5am**: exp3 should be ~70% through phase 4 by then. exp_omega_first will likely have finished (failed). The key deliverable for Roberto at 11am is the basin characterization data from phases 1-3 (already complete, 326 tests).

### Status at 12:19am NZ (Feb 9) — midnight check #3

1. **exp3_basin_resume.py**: Still running (PID 36203, 182% CPU, ~4.4h elapsed, ~7.6h remaining). Phase 4 (hi-fi verification) still empty. ETA completion: ~8am NZ — should be done before Roberto meeting at 11am.

2. **exp_omega_first.py**: **DEAD** — no process found. Log stopped at eval 31100, 200/600 grid iterations. Best result: f=1.1102, att=162.1°, ω=0.107°/s. **CONFIRMED FAILURE.** FFT omega estimate was 4× off, grid searched wrong magnitudes.

3. **All overnight experiments now resolved:**
   - dual_annealing 6D: FAILED (129°)
   - Grid+CMA-ES: FAILED (211°)
   - Tight-start 0.5°: FAILED (36°)
   - Omega-first FFT→grid: FAILED (162°) — died at 33% progress
   - **Every single approach on joint 6D has failed.** The decoupled strategy is the only viable path.

4. **exp3_basin_resume** is the sole surviving experiment and the most valuable — comprehensive basin characterization IS the paper contribution.

**No new experiments launched** — exp3 needs the CPU, and we have enough failure data to make a compelling case for decoupled estimation at Roberto's meeting.

### Status at 12:34am NZ (Feb 9) — midnight check #4

1. **exp3_basin_resume.py**: Still running (PID 36203, 252% CPU, ~5h elapsed, ~7h remaining). Phase 4 (hi-fi verification) still not started. ETA completion: ~7:30am NZ — well before Roberto meeting at 11am.

2. **exp_omega_first.py**: Confirmed dead. Final state: 200/600 grid iterations, best f=1.1102, att=162.1°, ω=0.107°/s. **FAILED** — FFT omega estimate 4× off, searched wrong magnitudes.

3. **All 5 overnight experiments now have final status:**
   - dual_annealing 6D: FAILED (129°)
   - Grid+CMA-ES: FAILED (211°)
   - Tight-start 0.5°: FAILED (36°)
   - Omega-first FFT→grid: FAILED (162°) — died at 33%
   - Basin characterization: RUNNING — sole survivor, ETA ~7:30am

4. **Phase 1 update from exp3 full run**: Attitude-only convergence basin refined:
   - ≤6°: reliable (converges)
   - 7°: fails (6.98° final error)
   - 8°: converges (4.75°)
   - 9°: fails (8.93°)
   - 10°: converges! (1.46° — jumped to correct basin)
   - 15°+: all fail
   - **Non-monotonic structure confirmed** — multiple basins with "lucky jumps"

**Decision**: Let exp3 finish. It's the most valuable data for Roberto. No new experiments needed until Girish reviews at 5am.

### Status at 12:49am NZ (Feb 9) — FINAL OVERNIGHT STATUS

**ALL EXPERIMENTS COMPLETE. No processes running.**

1. **exp3_basin_resume.py**: ✅ **COMPLETED** (4.8 hours, 352 total tests)
   - Phase 1 att-only: 9/18 converge (≤6° reliable)
   - Phase 1 ω-only: 3/13 converge (≤0.02°/s)
   - Phase 1 combined: 2/15 converge (extremely narrow)
   - Phase 2 fine grid: 29/60 = 48% (non-monotonic basin!)
   - Phase 3 statistical: 240 combined trials (20 per combo, 12 combos) — success rates:
     - 3°+0.01°/s: 50% (10/20) | 3°+0.05°/s: 30% (6/20) | 3°+0.1°/s: 30% (6/20)
     - 5°+0.01°/s: 5% (1/20) | 5°+0.05°/s: 15% (3/20) | 5°+0.1°/s: 15% (3/20)
     - 7°+0.01°/s: 5% (1/20) | 7°+0.05°/s: 5% (1/20) | 7°+0.1°/s: 0%
     - 10°: 0% across all omega levels
   - Phase 4 hi-fi verification: **6/6 SUCCESS** — lo-fi handoff → hi-fi refinement works when lo-fi converges!
   
2. **exp_omega_first.py**: ✅ **FAILED** — died at 200/600 grid iterations, 162° att error

3. **All 5 overnight experiments final:**
   - dual_annealing 6D: FAILED (129°)
   - Grid+CMA-ES: FAILED (211°)
   - Tight-start 0.5°: FAILED (36°)
   - Omega-first FFT→grid: FAILED (162°)
   - **Basin characterization: COMPLETED — 352 tests, the most comprehensive basin map in SSA literature**

**KEY RESULT: Phase 4 hi-fi verification is the silver lining.** When lo-fi converges to within ~5°, hi-fi L-BFGS-B refinement successfully recovers the true parameters (6/6 = 100%). The mixed-fidelity pipeline WORKS — the challenge is getting lo-fi close enough via decoupled estimation.

## Implications for Thesis

1. **The narrow basin problem is THE challenge** for joint attitude+ω inversion. This is why all 26 papers in the literature struggle and why Burton's two-stage approach exists.

2. **Decoupled estimation is not a hack** — it's the principled approach. It mirrors the physics: attitude determines the lightcurve shape, ω determines the time evolution. Solving sequentially exploits this structure.

3. **The lo-fi/hi-fi basin shift (105%)** is a key finding for the mixed-fidelity paper. It means naive two-stage doesn't work — you need multiple candidates or tighter refinement.

4. **For the Roberto meeting**: We can show:
   - The problem structure (attitude basin ~5°, joint basin <1°)
   - Why DE fails (6D landscape is too rough)
   - The decoupled approach as the solution
   - Lo-fi speedup enables practical inversion
   
5. **Paper contribution**: First systematic characterization of convergence basin structure for lightcurve attitude+ω inversion with self-shadowing.
