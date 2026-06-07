---
title: "s111 — anchor-cap cures finite-diff aliasing (116 inverts, 7-attractor multi-sol); slow-tumbler sweep BROKEN by a decimation/sampling bug"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s100_5step_proto.py
  - notebooks/inversion/survey/experiments/s109_omega_traj_envelope.py
  - notebooks/inversion/survey/experiments/s110_polish_116.py
  - notebooks/inversion/survey/experiments/run_slow_sweep.sh
related:
  - experiments/s099_blind_fast_invert_116.md
  - experiments/s100_5step_proto.py
  - experiments/s106_hybrid_loss_polish.md
created: 2026-05-29
updated: 2026-05-29
confidence: high (116 result + envelope solid; slow-sweep results KNOWN-COMPROMISED, see "What went wrong")
---

# TL;DR
Two new levers on s100 invert **seed 116** as a clean multi-solution set: a physical **|ω| clamp [0.05,1.6] deg/s** (s109-validated envelope) and an **anchor-spacing cap `dt_ab < π/w_hi`** that keeps the A→B finite-diff under half a turn so the single-shoot stops aliasing. s100(clamp+cap) → s110 surrogate-v2 polish returns **7 distinct attractors, 4 at the surrogate floor** (surr-ρ≈0.20), with truth recovered at **q0 0.19°, ω-dir 0.03°, |ω| 0.1335 vs 0.1336 deg/s** (`results/s110/polish_116.json`). **BUT the slow-tumbler sweep (seeds 10/42/31) is BROKEN** and produced no usable result — see "What went wrong"; the next agent must fix three things before re-running.

# What
After s100-capped inverted 116 (this session), the question was whether the same config generalises to slow tumblers (10, 42, 31, chosen by `results/s101`). It does NOT yet — but the failure is a pipeline-mechanics bug, not a science result. This writeup records the working 116 path AND documents the bugs precisely so the sweep can be re-run clean.

# How
- **s109** (`s109_omega_traj_envelope.py`): closed-form `propagate_jacobi_path2` of every cohort trajectory; min/max |ω(t)| over the observation epochs. Decides the clamp bounds.
- **s100** (`s100_5step_proto.py`, env `S100_CLAMP=1 S100_ANCHOR_CAP=1`): scan→anchors→dense-fill→decimate→cross→coarse→full-500 ranking. New: env-gated clamp; anchor-cap selecting A=sharpest, B=sharpest with `dt_ab∈[gmin,π/w_hi]` (time-windowed, relaxed if empty), C=sharpest elsewhere.
- **s110** (`s110_polish_116.py`): reuses `s058.lm_polish` (surrogate-v2 full-LC, 6-DOF, LM) — **no hi-fi** — on the saved candidates, clusters distinct attractors, reports truth-in-set as diagnostic.

# Result

**s109 envelope** (`results/s109/omega_envelope.json`): global min |ω(t)| = **0.1053 deg/s** (seed 10), global max = **1.5034 deg/s** (seed 119). |ω| swing per trajectory ≈ **1%** (not constant but nearly so). → floor 0.05 safe (2× margin); ceiling 1.6 needed because 119 peaks above the 1.5 generation cap.

**Seed 116 (clamp+cap, WORKS)** — `results/s110/polish_116.json`, 7 attractors:

| attractor | surr-ρ | q0_err | ω_dir | \|ω\| dps | note |
|---|---|---|---|---|---|
| A1 | 0.202 | **0.19°** | **0.03°** | 0.1335 | truth |
| A2 | 0.205 | 179.9° | 174.3° | 0.1335 | truth body-twin |
| A3 | 0.214 | 38.5° | 6.4° | 0.1332 | alt-basin (same ω) |
| A4 | 0.215 | 141.7° | 176.4° | 0.1331 | A3 twin |
| A5–A7 | 1.71–1.86 | 65–120° | 13–19° | 0.169–0.171 | Band-B alts (shifted ω) |

surrogate fidelity floor ≈ 0.20 (truth's own surrogate-vs-hifi RMSE 0.0102/0.05). 4 attractors at floor (Band-A-equiv), 3 at Band-B-equiv. **Multi-solution success; truth in set.**

**Anchor-cap mechanism confirmed:** on 116 the cap pulled B from ep349 (dt 2467s, 0.92 windings → single-shoot **145° dir-err**, 0 survivors) to ep52 (dt 325s, 0.12 windings → single-shoot **0.94°**, 721→ survivors). Same un-aliasing on 10 (3.54°), 42 (3.50°), 31 (anchors 72s apart).

**Slow-tumbler sweep (10/42/31) — NO USABLE RESULT.** All three s100 runs completed but (a) every polish was skipped (padding bug) and (b) the runs used the wrong config (6M sampling + widened clumps). Coarse, unpolished, compromised diagnostics only:

| seed | cross surv | best truth-near pair (coarse) | rank | reps coarsened to |
|---|---|---|---|---|
| 10 | 2189 | dir 7.13° (qa 7.56, qb 5.83) | 2 | A 16.3°, B 12.5° |
| 42 | 3895 | dir 114° (qa 25.4, qb 2.11) | 327 | A 12.5°, B 9.7° |
| 31 | 2726 | dir 75.8° (qa 11.1, qb 8.40) | 26 | A 12.5°, B 4.4° |

# Why this matters
The anchor-cap + clamp are the real fixes: finite-diff single-shoot aliases past 0.5 windings (the 119 *and* slow-seed blocker), and capping `dt_ab < π/w_hi` keeps every in-bracket candidate under half a turn. With that, 116 inverts via the cheap single-shoot cross + polish — no multistart/return-map needed for slow seeds. The slow-tumbler GENERALITY claim is **unproven** because the sweep was run on a broken config; fix the bugs below and re-run.

# What went wrong (FIX THESE BEFORE RE-RUNNING — the next agent's job)

1. **DECIMATION WIDENS THE BUBBLE (the #1 bug, still in code).** I added `decimate_adaptive` (s100) which, when the 2° cell count exceeds `REP_TARGET=2000`, **grows the cell size** (2°→up to 16°) until cells ≤2000. This destroys resolution: seed 42 dense nearest-truth **1.45° → rep 6.61°**; seed 10 → 16.3° clumps. The intended behaviour (per user) is **FIXED 2° clumps, ONE rep per occupied clump, NO cap, NO coarsening**. **FIX:** in the `S100_ANCHOR_CAP` branch of step 4, replace `decimate_adaptive(...)` with `decimate_2deg(Q, deg=DECIM_DEG, max_reps=0)` (max_reps=0 → keep all cells, no random cull). Delete/ignore `decimate_adaptive` and `REP_TARGET`.

2. **SAMPLING WAS 6M, NOT 1M.** s100's default was `DENSE_POOL=6_000_000` (116 grew to 12M; 31 to 12M via the adaptive-growth loop). User standing pref is **1M flat**. This is *why* bug #1 fired — 6× the samples → ~6× the 2° cells → forced coarsening. **Already fixed in code this session** (default `DENSE_POOL=1_000_000`, `DENSE_TARGET=0`, `DENSE_MAX_POOL=1_000_000` → flat 1M, no growth). Verify before re-running.

3. **PADDING BUG skipped every slow-sweep polish.** s100 saves to `results/s100/seed{SEED:03d}/` (`seed010`, `seed042`, `seed031`) but `run_slow_sweep.sh`'s check and s110's loader used unpadded `seed{SEED}` (`seed10`...). Matches only for ≥3-digit seeds (116). **s110 loader fixed this session** (`f"seed{SEED:03d}"`); **`run_slow_sweep.sh` still has the unpadded `[ -f results/s100/seed${SEED}/invert.npz ]` check — FIX to `seed$(printf '%03d' $SEED)`.**

4. **PAIR_BUDGET random subsample can also drop truth.** The cross caps at `PAIR_BUDGET=6M` via random `rng.choice`. With fixed-2° + 1M clouds the cross should stay under budget, but if not, the random cull drops the truth rep. **FIX:** raise `PAIR_BUDGET` (e.g. 50M) or make the subsample deterministic so it never drops the truth cell.

5. **SAVE_K was 15 — buried 42's truth at rank 327.** s100 saved only top-15 by coarse full-RMSE; 42's truth-near pair ranked 327, outside the saved set, so even a correct polish couldn't reach it. **Already changed to `S100_SAVE_K=500` this session.**

# Numbers
- s109: min |ω(t)| 0.1053 dps, max 1.5034 dps (source: `results/s109/omega_envelope.json`); seed 119 wmax 1.5034 > 1.5 generation cap.
- 116 polish: truth attractor q0_err 0.19°, ω_dir 0.03°, |ω| 0.1335 vs truth 0.1336 dps, surr-ρ 0.202; 4 attractors surr-ρ<1, 3 in [1.7,1.9] (source: `results/s110/polish_116.json`).
- 116 anchor-cap: dt_cap 336s, B ep349→ep52, single-shoot 145°→0.94° (source: `results/s100/seed116/invert.json`, `seed116_capped_run.log`).
- 42 decimation loss: dense nt 1.45° → rep 6.61° at 12.5° cells (source: `results/s100/seed42_capped_run.log:[3],[4]`).
- Slow-sweep (COMPROMISED config 6M+widened): see Result table (source: `seed010/042/031_capped_run.log`).

# Artefacts
- `results/s110/polish_116.json` — the 116 multi-sol set (the one trustworthy result).
- `results/s109/omega_envelope.json` — clamp-bound validation.
- `results/s100/seed{010,042,031,116}/invert.{npz,json}` — s100 candidate sets (npz NOT committed; regeneratable).
- `results/s100/seed*_capped_run.log` — per-seed run logs (diagnostics).

# Out of scope
- Hi-fi ρ-band confirmation of any attractor (user: surrogate-only this session). The surr-ρ bands are surrogate; a final hi-fi pass on A1/A3/A5 would confirm true bands.
- The conservation/(|L|,2T,regime,phase) reparameterisation discussion (analysed, not implemented; concluded it's a computational reformulation, not new physical info, since the propagator already conserves L,T — see session transcript).

# Cross-references
- [[feedback_multi_sol_acceptance_is_the_goal]] — reaffirmed emphatically this session (multi-solution is the goal; polish is mandatory; never frame "truth rank" as the verdict).
- s099 (the densify-around-parents stage s100 dropped — relevant to the 42 resolution-loss; coarse-to-fine is the deeper fix once decimation is reverted).
- s106/s107/s108 — the 119 multistart/return-map line (separate; aliased-119 needs a different connector).
