# Worker 3 Report: m028 — Integration Pipeline with Nudged Attitudes

## What was attempted

**Task:** Run the m026 pipeline (band-sweep enumeration + L-conservation filter) with attitudes perturbed by 1, 2, 5 degrees from truth. 10 trials per nudge level (30 total).

**Scripts written:**
- `notebooks/inversion/08_integration/m028_integration_nudged.py` — multiple iterations

### Iteration history

1. **v1 (committed as `ecc1193` on `exp/integration-nudged`):** Direct port of m026 with nudge loop. Full tumbling-mode bridge (13 bands × 10 starts = 130 solves per leg, Pool(8)). First trial took **~1080s** — extrapolates to ~9 hours for 30 trials. Killed.

2. **v2 (PA-only bridge):** Used `principal_axis` mode for bridge solves (763x faster per propagation call). Completed all 30 trials in ~54s. **But 0/30 correct** — PA-mode bridge solutions are fundamentally different from tumbling solutions. Direction error was consistently ~160 deg. The winding topology differs between PA and tumbling dynamics.

3. **v3 (PA enumeration → tumbling refinement, two-phase):** PA sweep to find candidate bands (~0.7s), then tumbling refinement of each PA candidate with Pool(8). Hit Python 3.14 forkserver issue (`mp.get_context('fork')` fix needed). After fixing, first trial still took **~450s** — PA warm starts are too far from tumbling solutions to converge quickly.

4. **v4 (oracle sweep + nudged refinement, current on disk):** One full tumbling oracle sweep (like m026), then short warm-started refinement (maxiter=30) for each nudged trial. Oracle sweep was still running when killed (~15 min per leg expected). This is the most promising approach but was not completed.

## What results exist

- `data/results/inversion_diagnostics/m028_integration_nudged.json` — from v2 (PA-only), all 30 trials MISS. **Not useful.**
- `data/results/inversion_diagnostics/m028_integration_nudged.png` — from v2, shows 0% accuracy across all nudge levels. **Not useful.**

## What went wrong

### Core bottleneck: tumbling-mode bridge solves
Each tumbling-mode bridge solve (L-BFGS-B with `propagate_attitude(..., "tumbling", I)`) takes **~30-60 seconds** on this system due to the ODE integration cost (~66ms per `propagate_attitude` call × ~500 function evaluations per solve). This makes:
- Full 130-solve sweep: ~15 min per leg
- Even 10-candidate refinement: ~5 min per leg

### PA mode doesn't work as a substitute
PA-mode (`principal_axis`) bridge solutions find different omega directions than tumbling-mode solutions. They share the same magnitude bands but the specific omegas are physically inconsistent with Euler dynamics. L-conservation scoring requires tumbling-mode solutions.

### Python 3.14 multiprocessing
Default start method is `forkserver`, which fails with `os.chdir()` at module level. Fix: `mp.get_context('fork').Pool(...)`.

## Current state

- **Branch:** Currently on `inversion_q_w`. The work branch is `exp/integration-nudged`.
- **Committed:** `ecc1193` (v1 script) on `exp/integration-nudged`
- **Uncommitted:** v4 script is on disk at `notebooks/inversion/08_integration/m028_integration_nudged.py` (oracle sweep + nudged refinement approach)
- **Nothing running.**
- The v2 results (JSON/PNG) are untracked on `inversion_q_w`.

## Recommendation for next steps

The **oracle sweep + nudged refinement** (v4) approach is the right one. To complete:

1. Switch to `exp/integration-nudged`
2. Run the v4 script — expect ~20 min for oracle sweep + ~15 min for 30 nudged trials = ~35 min total
3. If still too slow, reduce to 5 trials per nudge (15 total) or reduce `maxiter` in refinement from 30 to 15
4. Consider relaxing `ARRIVAL_THRESH` for nudged refinements to 1e-4 (since nudged endpoints don't lie on the true trajectory anyway)
