# Exp 3 — Convergence Basin Status Note
**2026-02-08**

## What's Done

### Phase 1: Lo-fi Basin Mapping ✅
- **Attitude-only**: 18 perturbation levels (1°–180°), all complete
- **Omega-only**: 13 perturbation levels (0.001–1.0 °/s), all complete
- **Combined**: 15 perturbation combinations, all complete

### Phase 2: Fine Grid (5°–10°) ✅
- 10 trials each at 5°, 6°, 7°, 8°, 9°, 10° — all 60 runs complete
- Success rates: 5°=90%, 6°=70%, 7°=30%, 8°=20%, 9°=50%, 10°=30%

### Phase 3: Statistical Tests ⚠️ PARTIAL
- Only completed `stat_3d_0.01dps` (20 trials, 8/20 = 40% success)
- **Remaining**: 3°+0.05°/s, 3°+0.1°/s, 5°+{0.01,0.05,0.1}°/s, 7°+{0.01,0.05,0.1}°/s, 10°+{0.01,0.05,0.1}°/s
- That's 11 more conditions × 20 trials = **220 remaining runs**
- At ~30s each lo-fi ≈ **~2 hours**

### Phase 4: Hi-fi Verification ❌ NOT STARTED
- Plan: take lo-fi successes, refine with hi-fi objective
- Each hi-fi run: ~30 evals × 6s = ~3 min per test
- Planned: ~5–10 tests = **30–60 min**

## Total Remaining Runtime Estimate
- Phase 3: ~2 hours
- Phase 4: ~1 hour  
- **Total: ~3 hours** (with margin: ~4–6 hours)

## Separate Runs Already Done
- `exp3_convergence_basin_lofi.json`: Quick initial lofi test (sanity through 45°). Ran in ~3 min. Confirmed basin boundary around 5–10°.
- `exp3_timed_results.json`: Timed two-stage DE→L-BFGS-B with period analysis. Ran ~12 min. Failed — random evaluations are slow (4.6s avg because some trigger expensive dynamics). Speedup only 1.7× (not 167×) due to evaluation cost variation.
- `exp3_v3_bounded_results.json`: Bounded DE+L-BFGS-B two-stage at "tiny" (100 lo-fi + 20 hi-fi) and "small" (1980 lo-fi + 100 hi-fi) budgets. Both failed completely — DE converged to wrong basin (176° and 187° attitude errors). The omega bound of ±1.95°/s was correctly computed from frequency analysis but search budget was insufficient.

## Key Takeaway
The convergence basin is **narrow** (~5–6° attitude, ~0.02°/s omega for single-start L-BFGS-B). The two-stage DE→L-BFGS-B approach needs significantly more evaluations to find the correct basin globally. This motivates multi-start strategies and/or better initialisation (e.g., grid + refine, or particle swarm).
