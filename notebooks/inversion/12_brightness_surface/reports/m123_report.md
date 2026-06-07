# m123 — L-BFGS polish from truth + m115 DE basins

## Hypotheses

1. **Surrogate-truth offset is small.** L-BFGS from truth converges within < 1° q0, < 0.01° ω-dir, < 0.01% |ω| of truth. Refuted ⇔ the surrogate minimum is displaced from truth beyond inversion precision.
2. **DE basins do not benefit from GD polish.** L-BFGS from each m115 DE basin stays within 0.1° q0 and 0.005° ω-dir of its start. Confirmed ⇔ basins sit in the saturated region (dark-mag-saturation); refuted ⇔ gradient-based post-polish would materially improve DE hits.
3. **Attractor count ≤ DE basin count per seed.** GD can only land on minima already near a start point. Equality => DE found every gradient attractor; strict inequality => multiple DE basins share one GD minimum.

## Method

Per-start 6-DOF tangent parameterisation (Rodrigues q0 + ω̂ tangent + |ω| scale) freshly rebuilt at each start — unlike m122 where the tangent lived at truth. Cost is the m121/122 mean_L1 over 500 surrogate-evaluated epochs (panel=0°, dish=15°). Optimiser: `scipy.optimize.minimize(L-BFGS-B, jac=None)` with 2-sided finite differences; `ftol=1e-6, gtol=1e-3, maxiter=100, maxfun=500`. Starts per seed = 1 truth + N m115 basins (N taken dynamically from `de_results.basins`).

Seeds: {14, 27, 46, 74, 93}. Reuses `m122/seed_NNN/setup.npz`.

## What to look at when results land

- **Run-level `summary.json`** → `verdicts.hyp1/hyp2/hyp3` (plus per-seed `hyp2_fraction` and `n_attractors_vs_n_basins`).
- **Per-seed `summary.json`** → `records` (per-start moves + final errors), `clusters` (attractor groupings).
- **Console headlines** per seed show truth-start final errors and per-basin SAME_ATTRACTOR/MOVED flags.

## What refutes each hypothesis

- **H1** refuted if any seed's truth-start final exceeds 1° q0, 0.01° ω-dir, or 0.01% |ω|.
- **H2** refuted if aggregate stayed-put fraction across non-truth starts < 0.2 (DE basins actually have gradient to exploit → pipeline should add a polish stage).
- **H3** refuted if any seed produces more attractors than DE basins (unexpected; would indicate DE merged distinct minima).

## TODOs punted

FD-step sensitivity on the L-BFGS jac; alternate cost variants; plotting; hi-fi LC capture at each final; single LBFGS option setting only.
