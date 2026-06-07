# m119 — CRITICAL BUG REPORT (2026-04-15 evening)

## Status: all m119 results on `inversion_q_w` branch are CORRUPTED

Everything written about m119 in this session — the "rank 0 / 60000 for seed 14", the multi-seed tables, the "differentiable inversion is the next paradigm" concept page, the gradient-based-inversion branch — is built on numerically wrong geometry. Do not act on any of it.

## The bug in one sentence

`m119_attitude_isoshell.py` mixes observation data generated at a 3600 s / 7.21 s-per-epoch window (pulled from `m046_trajectories.npz`) with J2000 geometry freshly computed at a 21600 s / 43.29 s-per-epoch window (from today's `setup_experiment()` via the current `intelsat_901_config.yaml`). Epoch index `i` in the two sources refers to physically different times, off by a factor of 6. Everything downstream is garbage.

## Evidence (how to reproduce the diagnosis)

```python
import numpy as np, sys
sys.path.insert(0, 'notebooks/inversion')
from lib.experiment_setup import setup_experiment

ctx = setup_experiment(n_observations=500, skip_true_lc=True)
master = np.load('data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz')

print(master['observation_times'][[0, 1, -1]])  # [0, 7.21, 3600]
print(ctx.observation_times[[0, 1, -1]])         # [0, 43.29, 21600]
```

The two `observation_times` arrays are different by a factor of 6. m046 was generated when the config's `simulation_defaults.end_time` was 1 hour after `start_time`; today the config has a 6-hour span.

### Symptom that tipped us off

In `m119_attitude_isoshell.py` Stage A, the script computes `k1_j2000` and `k2_j2000` from `ctx.sun_pos`, `ctx.obs_pos`, `ctx.sat_pos`. These are at the 6-hour (wrong) time grid. The `observed_lc` and the truth pose come from the kernel, which pulled them from master — 1-hour time grid.

Per-epoch angular mismatch between the true body-frame unit vectors (back-derived from master's saved `k1_body`, `k2_body`, `quaternions`) and the vectors m119 actually uses:

| Quantity | ep 0 | ep 100 | ep 250 | ep 499 |
|---|---|---|---|---|
| k1 angle | 0.00° | 0.05° | 0.11° | 0.22° |
| k2 angle | 0.00° | 14.79° | 36.95° | 73.69° |

k1 (sun direction) matches closely because the sun is 150 Mkm away and its direction barely moves in a few hours. k2 (observer direction) rotates with Earth at ~15°/hour, and the time offset at epoch `i` is `i × (43.29 − 7.21) s = 36.08 i s`. At ep 250 that's 9020 s of offset × 15°/3600 s = 37.6° — exactly matches the measured 36.9°. Bug confirmed.

### Why seed 14's "rank 0 / 60000 under mean_L1" is meaningless

The SO(3) grid candidates and the propagated truth trajectory are both evaluated against the same wrong `k1_j2000` / `k2_j2000` / `obs_dist_ce` arrays. Whichever of the 60,001 things (60k grid + truth) happens to score best under wrong geometry is noise, not signal.

Related: the earlier session even noted `ctx.obs_dist vs m046 obs_dist max diff: 5.583e+01 km` in the Stage A output and moved on. That 56-km max diff is another surface symptom of the same time-mismatch bug. Should have been treated as a load-bearing warning, not a harmless print.

## Files that need retraction

All added in this session's commits `eebb287`…`cc57f0a` (plus follow-ups):

- `data/results/inversion_diagnostics/m119/seed_*/` — every seed's residual_kernel.npz, cost_variants.npz, target_scores.npz, summary.json, plots/*.png. All corrupted.
- `data/results/inversion_diagnostics/m119/multiseed_summary.json` — corrupted aggregation.
- `notebooks/inversion/wiki/wiki/experiments/m119.md` — conclusions invalid.
- `notebooks/inversion/wiki/wiki/concepts/differentiable-inversion.md` — "validated partially by m119" claim is false; the concept may still be worth thinking about, but the supposed validation isn't there.
- `notebooks/inversion/wiki/wiki/branches/gradient-based-inversion.md` — gating "multi-seed cost generalisation" was never actually tested; branch status claim is false.
- `notebooks/inversion/wiki/wiki/experiments/m118.md` — See-also pointer to m119 is misleading if retained.
- `notebooks/inversion/wiki/wiki/concepts/surrogate-model.md` — "dim-regime caveat" added based on corrupted m119 residuals; the real cause was time misalignment, not surrogate coverage.
- `notebooks/inversion/wiki/wiki/index.md` — m119 row, differentiable-inversion row, gradient-based-inversion row.
- `notebooks/inversion/wiki/wiki/log.md` — recent m119 / differentiable-inversion entries.
- `notebooks/inversion/EXPERIMENTS.md` — Section 1 resume point, m119 block, updated NEXT STEPS.

Decision for the next agent: either full revert of this session's commits, or keep the artifacts but prepend RETRACTED banners and mark the wiki branches closed. I've left banners in place as a minimum; full revert is cleaner.

## Why m046 is in this pipeline at all

Legacy. `m046_trajectories.npz` is a pre-computed 100-seed dataset (q0s, omega0s, quaternions, k1_body, k2_body, mag_hifi, etc.) produced once upon a time when the current `setup_experiment()` flow was less mature. It was convenient to pull a seed's truth pose and a reference hi-fi LC from there instead of regenerating them. Later scripts (m102 onwards) kept using master data out of inertia.

**Nothing in the current pipeline architecturally requires m046.** `setup_experiment(n_observations=500, skip_true_lc=False)` produces the same quantities at runtime, consistent with today's config by construction. Mixing master-era observation data with current-config SPICE geometry is exactly the failure mode that happened here.

## Recommended fix (for the next agent)

**Option A — sanest fix:** stop using `m046_trajectories.npz` entirely.
- Rewrite Stage A of `m119_attitude_isoshell.py` to source truth `(q0, ω)` from the seed's own m046 entry ONLY for scalar fields that are seed-parameters (q0s, omega0s), and generate `observed_lc`, `obs_times`, `sun_pos`, `sat_pos`, `obs_pos` from `setup_experiment(skip_true_lc=False)` at today's config.
- Require `setup_experiment` to also generate `observed_lc` consistent with the current time grid (it already does when `skip_true_lc=False`, but the script needs to actually call with that flag and add noise itself).
- Do the same for `m118_kernel_computation.py` — it pulls observation_times, pab_j2000, true_lc all from master. Either regenerate them at runtime or confirm the config hasn't changed since.

**Option B — quick patch:** regenerate `m046_trajectories.npz` at today's config. One-time cost, preserves all existing scripts. Risk: the next time the config changes, the same bug reappears.

**Option C — defensive:** in `setup_experiment`, accept explicit `end_time_utc` parameter (it already does — line 79 of `experiment_setup.py`). Callers that read master data should pass `end_time_utc` consistent with master's observation_times span (3600 s after start). This keeps master usable with a tiny change at every call site.

Option A is cleanest. Option C is minimal-risk if the next agent wants to preserve ongoing work.

## Audit list (before trusting anything pre-bug)

Search for any script that BOTH reads `m046_trajectories.npz` AND calls `setup_experiment()` / uses `compute_observation_geometry`. These are candidates for the same bug:

```
grep -l m046_trajectories notebooks/inversion/**/*.py
```

### AUDIT RESOLVED 2026-04-15 (afternoon)

Empirical check (strategist, this session): scripts that explicitly pass `end_time_utc='2020-02-05T11:00:00'` to `setup_experiment()` produce `observation_times` that match `master['observation_times']` to machine precision, and reconstructed k1/k2 J2000 vectors match `ctx.sun_dirs/obs_dirs` to 10⁻⁶°. **These scripts are NOT affected by the time-mismatch bug.**

- `m119_attitude_isoshell.py` (v1) — **confirmed broken** (used default end_time → 6-hour window, mismatched m046's 1-hour window). Fixed by `m119v2_attitude_isoshell.py`.
- `m118_kernel_computation.py` — **OK** (pure m046 internally; no setup_experiment mixing).
- `m115_surrogate_pipeline.py` — **OK** (line 719: `end_time_utc='2020-02-05T11:00:00'`). The "10/10 surrogate multi-start DE breakthrough" stands.
- `m117_result_harvester.py` — **OK** (imports setup_experiment but never calls it; pure m046).
- `archive/inline_omega_selection_test.py` — **OK** (line 109: `end_time_utc='2020-02-05T11:00:00'`). The "4/4 surrogate omega-selection validation" stands.

The previous strategist's audit entry in EXPERIMENTS.md that flagged m115 and inline_omega_selection_test as "in doubt" was incorrect — it pattern-matched on "loads master + calls setup_experiment" without checking whether `end_time_utc` was forced to match the 1-hour window. **Both scripts are valid.** Only v1 of m119 actually had the bug.

## Other lessons from this blowup

1. **Weak test design compounded the bug.** The grid of "60,000 SO(3) static rotations" used as candidates is not a meaningful comparison set for a tumbling-satellite inversion problem. All grid "candidates" are ω=0 trajectories. Truth beating them under any cost function is near-tautological and doesn't validate a cost function. If the test had used 60,000 *tumbling* (q0, ω) candidates drawn from the actual 6-DOF parameter space — or just the handful of real wrong-basin competitors from m118 — the bug would have been caught immediately because the results would have been obviously broken instead of "interestingly mixed."

2. **Silent-skip logic hid the issue.** The script silently skips competitor files that don't exist (`topK_facet_normal.npz` for seeds other than 14). That meant `truth_beats_competitor_any_variant` was only a meaningful check on seed 14; for every other seed it was a vacuous pass/fail. The reporting didn't flag this clearly enough to notice.

3. **The "56 km obs_dist mismatch" print was a real clue, dismissed.** When Stage A consistently prints `ctx.obs_dist vs m046 obs_dist max diff: 5.58e+01 km` and no one investigates, that's a failed sanity check, not a nicety. Next agent: sanity checks that print non-zero values should block, not log.

4. **The strategist (me) over-claimed on a single seed's result.** I turned a single-seed "rank 0 under one of 13 cost variants" into a wiki concept page and a horizon branch before multi-seed validation existed. Multi-seed validation — had the geometry been correct — might or might not have supported the claim; the failure mode here is that I celebrated without waiting for it.

## Useful things that DO survive this blowup

- `super_fibonacci_quats()` implementation in `m119_attitude_isoshell.py` is correct and reusable.
- The kernel-factorization pattern (build expensive residual tensor once, score many cost variants cheaply) is good architecture regardless of today's bug.
- `m119_multiseed_aggregate.py` is a reusable pattern for summarising per-seed result JSONs.
- The checkpointing discipline (save all candidates under 13 cost variants per seed) gave us the ability to diagnose after the fact — without it we'd have had to re-run to investigate. Keep this.

## Current git state

Branch: `inversion_q_w`. Last good commit before this session's false findings: `eebb287` (m118 kernel-factored IPL cost diagnostic). Subsequent commits `0a748f2` and onward were added in the preceding session AND this one — the preceding-session ones are fine; this-session ones are suspect.

Commits added this session (all based on the corrupted m119):
- `ec41cf0` — "m119: surrogate attitude-isoshell POC on seed 14"
- `f468a52` — "wiki: differentiable-inversion concept + gradient-based-inversion branch"
- `cc57f0a` — "m119: multi-seed runner + aggregator scripts"

At the next agent's discretion: revert these three, or keep them with RETRACTED banners.
