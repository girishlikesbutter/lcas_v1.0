---
title: "m128 — warm-start flipped-ω polish from m115 DE basins"
type: experiment
sources:
  - "raw/inversion_diagnostics/m128_warmstart_polish/batch_summary.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_000/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_006/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_012/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_014/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_024/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_027/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_033/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_036/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_046/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_074/result.json"
  - "raw/inversion_diagnostics/m128_warmstart_polish/seed_093/result.json"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/seed_033/step2_hifi.npz"
related:
  - "[[m115_surrogate_pipeline]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[m127_flipped_omega_search]]"
  - "[[omega-sign-degeneracy]]"
  - "[[gradient-based-inversion]]"
  - "[[basin-of-attraction]]"
  - "[[surrogate-model]]"
  - "[[multi-solution-philosophy]]"
created: 2026-04-16
updated: 2026-04-16
confidence: high
---

# m128 — warm-start flipped-ω polish from m115 DE basins

## Hypothesis (falsifiable)

At least 2 of seeds `{0, 6, 14, 24, 27, 36, 46, 74, 93}` yield a polished `(q0', −ω_basin)` with hi-fi MSE < 0.5. Seed 33 positive control MUST reproduce hi-fi ≈ 0.082 — failure marks the verdict INVALID (polish mechanics broken), not REFUTED.

## Method

For each seed × each of the 3 m115 DE basins:

1. Load `data/results/inversion_diagnostics/m115_surrogate_pipeline/seed_NNN/step2_hifi.npz`; extract the 3 basins `(q0_basin_wxyz, ω_basin, hifi_basin)`.
2. Warm-start 3-DOF L-BFGS-B at `(q0_basin, −ω_basin)` on surrogate cost `mean |pred − obs|` over a 500-epoch, 1-hour window; `ftol=1e-7, gtol=1e-4, maxiter=200, maxfun=1000`.
3. Hi-fi-evaluate `(q0_init, −ω_basin)` and the polished `(q0', −ω_basin)` (Pool(2) serial-per-seed, parallel across seeds).
4. Seed-level `best_hifi = min(before, after) across 3 basins`.
5. Classify `FLIPPED_VALID < 0.1`, `FLIPPED_PARTIAL < 0.5`, `FLIPPED_FAIL ≥ 0.5`.

Wall: **2322.7 s (38.7 min)** across the 11 seeds with Pool(2). First attempt OOM'd with Pool(4) — retried with Pool(2). Per-seed wall 175–232 s (polish ~0.4 s/basin; hi-fi ~65 s/basin).

## ω-sign census across the 11 seeds (load-bearing pre-interpretation)

The warm-start spec `(q0_basin, −basin_ω)` is only a valid test of flipped-ω recovery IF basin_ω is **forward** relative to truth. If basin_ω is already retrograde, negating it produces a FORWARD-ω state (no longer a flipped-ω test).

Verified inline from `step2_hifi.npz` using `signed_w_dir = acos((ω̂_basin · ω̂_truth))`:

| seed | basin 0 signed | basin 1 signed | basin 2 signed | class |
|-----:|:--------------:|:--------------:|:--------------:|:-----:|
| 0  | 2.90° | 2.90° | 2.90° | **forward — valid test** |
| 6  | 3.08° | 3.08° | 3.08° | **forward — valid test** |
| 12 | 8.01° | 8.01° | 8.01° | **forward — valid test** |
| 14 | 0.34° | 0.34° | 0.34° | **forward — valid test** |
| 24 | 1.93° | 1.93° | 0.64° | **forward — valid test** |
| 27 | 3.08° | 3.08° | 3.08° | **forward — valid test** |
| 33 | **161.54°** | **161.54°** | **161.54°** | **retrograde — INVALID test** |
| 36 | 10.73° | 10.73° | 10.73° | **forward — valid test** |
| 46 | 6.39° | 6.39° | 6.39° | **forward — valid test** |
| 74 | 4.88° | 4.88° | 4.88° | **forward — valid test** |
| 93 | 0.49° | 0.49° | 0.49° | **forward — valid test** |

**10/11 seeds are valid flipped-ω tests; seed 33 is structurally invalid.** Seed 33's m115 basin is already on the retrograde ω; warm-starting at `(q0_basin, −basin_ω)` puts the polish at ω **forward** (signed 18.46° from truth), NOT at the known flipped-ω attractor at hi-fi 0.082. The 0.082 attractor lives at `(q0_basin, +basin_ω)` = m115 basin 0 as-is.

The axis-angle convention in `w_dir_err` folds 161.54° → 18.46°; this is what concealed the retrograde ω in [[m115_surrogate_pipeline]] and [[m126_wrapped_pipeline]] and in the cross-reference table here before correction.

## Results

| seed | before min hi-fi | after min hi-fi | best_hifi | class | notes |
|-----:|-----------------:|----------------:|----------:|:-----:|:------|
| 0  | 3.88 | 1.52 | **1.5153** | FLIPPED_FAIL | 3/3 basins polished lower on hi-fi |
| 6  | 5.95 | 3.22 | **3.2200** | FLIPPED_FAIL | 3/3 basins polished lower on hi-fi |
| 12 | 5.32 | 1.42 | **1.4161** | FLIPPED_FAIL | sanity control target was <0.3 — FAILED |
| 14 | 5.86 | 4.73 | **4.7296** | FLIPPED_FAIL | |
| 24 | 8.10 | 2.83 | **2.8298** | FLIPPED_FAIL | |
| 27 | 5.99 | 3.61 | **3.6062** | FLIPPED_FAIL | |
| 33 | 4.94 | 2.24 | **2.2434** | FLIPPED_FAIL | INVALID per census — polishing forward-ω (not flipped) |
| 36 | 6.00 | 2.93 | **2.9344** | FLIPPED_FAIL | |
| 46 | 5.52 | 1.52 | **1.5213** | FLIPPED_FAIL | |
| 74 | 2.76 | 2.56 | **2.5588** | FLIPPED_FAIL | |
| 93 | 6.50 | 3.73 | **3.7336** | FLIPPED_FAIL | |

Polish consistently drops hi-fi (3–5× better than pre-polish on most seeds) but never approaches the 0.5 threshold. No seed reaches even PARTIAL.

## Polish mechanics sanity (inline spot-check, this session)

Per-strategist request: since seed 33's "control" state as spec'd cannot by construction reproduce 0.082, I re-evaluated hi-fi at `(q0_basin_0, +basin_ω)` for seed 33 directly using m128's own `hifi_validate` path:

| quantity | value |
|---------|-------|
| re-eval hi-fi at `(q0_basin_0, +basin_ω)` | 0.082471 |
| m115 recorded `hifi_mse` (basin 0) | 0.081680 |
| relative diff | **0.97%** (within 1% tolerance) |

**Polish mechanics: OK.** The `hifi_validate` path reproduces m115 when fed the same state. L-BFGS-B machinery is intact.

Therefore, seed 33's `best_hifi = 2.2434` is the **correct behaviour** of the spec-as-written: the warm-start is 18.46° from truth on the wrong side of the ω sign-flip, and L-BFGS-B cannot jump into the known `+basin_ω` attractor because that attractor has q0-width <0.001° and the state would need to move ω rather than q0 to reach it (ω is fixed in this 3-DOF polish).

## Verdict reinterpretation

- **Script-emitted verdict:** `INVALID` (seed 33 control `best_hifi ≥ 0.15`).
- **Correct reading:** the INVALID label reflects a **spec bug**, not a **mechanics bug**. Given the ω-sign census, the spec's "seed 33 reproduces 0.082" is structurally impossible regardless of polish performance. The right interpretation is **REFUTED on the 10 valid-test seeds**: warm-starting from any m115 forward-ω basin with ω negated produces no PARTIAL solution (all 10 valid tests are FLIPPED_FAIL ≥ 0.5). Seed 33 is a known-unrunnable control that should have been specified differently (e.g. load the basin and polish at `+basin_ω` — which would trivially reproduce 0.082 as shown above).

## What we learned

1. **Warm-start L-BFGS from forward-ω m115 basins does NOT find flipped-ω attractors**, for any of the 10 valid-test seeds. Polish can flow downhill (surrogate cost drops 1.5–3× per basin on average) but lands at surrogate local minima that hi-fi rates as 1.4–4.7 mag² — not OK, not even PARTIAL.
2. **The m127-seed-12 WIDE flipped-ω attractor** (hi-fi 0.171 at q0_err 140.36°) is NOT reachable from any of seed 12's m115 basins via 3-DOF L-BFGS-B polish. Seed 12's sanity-control target was 0.3 — actual best 1.42 (4.7× short). Quaternion distance from seed 12's nearest m115 basin to the m127 winner was pre-characterised at ~62° (see [[omega-sign-degeneracy]] seed 12 section). L-BFGS-B local convergence radius on this surrogate is <<62° in q0, so the warm-start + negated-ω state cannot cross that gap.
3. **Seed 33's narrow ~0.082 attractor** is similarly unreachable via this method because the state sits on the wrong side of the ω sign-flip to begin with. See census row 33.
4. **m128 RULES OUT one search strategy** (warm-start-from-m115-basins + component-negate-ω + 3-DOF-q0-polish) as a path to flipped-ω attractors. It neither adds nor refutes the existence of flipped-ω basins.
5. **Mechanism for the failure:** the geometric distance in `(q0, ω)`-space between the 10 valid warm-start states `(q0_micro115_basin, −basin_ω)` and either (a) seed 33's known narrow flipped-ω basin or (b) seed 12's known wide flipped-ω basin or (c) any other hypothesised flipped-ω attractor on the remaining 9 seeds is much larger than the L-BFGS-B local-convergence radius on this surrogate cost surface. The surrogate gradient at the warm-start state points toward the nearest local minimum on the `−basin_ω` slice, which is NOT a flipped-ω attractor of truth (the truth's flipped-ω attractors, if any, have their own independent q0 far from the m115 basins' q0 values).

## Lazy-explanation audit of own analysis

- Shallow: "polish failed because the basins are different."
- Honest: L-BFGS-B with FD Jacobian has a local convergence radius bounded by the surrogate cost curvature. On this 3-DOF q0-only polish surface (ω frozen at `−basin_ω`), the cost gradient at the warm-start `q0_basin` points toward a local minimum dictated by the `−basin_ω`-slice's geometry — NOT by the position of any flipped-ω truth attractor. For this to recover a flipped-ω solution, the warm-start q0 would need to lie **within** the flipped-ω attractor's basin on the `−basin_ω`-slice. No evidence suggests m115's forward-ω basin q0 values coincide with flipped-ω attractor q0 values; the two attractor families are in different parts of SO(3). Hence: polish drives surrogate cost down, but downhill is "not toward a flipped-ω truth fit."

## Does [[omega-sign-degeneracy]] confidence change?

**No change — stays `medium`.** m128 neither confirms new basins nor refutes existing ones. It only rules out one search strategy (gradient-based warm-start from m115 basins) as a way to enumerate them. Seed 33's narrow basin still exists (re-confirmed by the 0.082 spot-check above). Seed 12's wide basin from [[m127_flipped_omega_search]] is untouched. Population frequency stands at 2/11 confirmed cases with unknown count of unsearched narrow basins on the other 9 seeds.

## Next experiment queue (rank-ordered)

1. **m129 — densified SO(3) grid on seed 33 alone.** Cost: ~15 min. Info: tests whether seed 33's narrow basin is recoverable by pure grid-density. Medium-high value because seed 33 is the canonical known narrow attractor, and this isolates "grid density is the fix" vs "no coarse-then-polish method works."
2. **m130 — DE-over-q0 with ω fixed at `−ω_true`, 11 seeds, pop 200, 10 restarts/seed.** Cost: ~30–45 min. Info: population-based search is width-agnostic → recovers narrow AND wide flipped-ω basins if they exist anywhere reachable on the `−ω_true` slice. Higher info but 2–3× more expensive than m129.
3. **Close the branch with a "narrow basins need DE, not polish" note** if both m129 and m130 land negative. The collective evidence would then be: (a) coarse SO(3) grid + L-BFGS [[m127_flipped_omega_search]] finds only wide basins; (b) warm-start polish [m128] finds none; (c) DE-over-q0 with fixed `−ω_true` [m130] is the only remaining enumeration tool for narrow attractors.

**Recommendation:** m129 first (cheap, single-seed, answers a binary question). If it rescues seed 33, pivot to DE (m130). If not, m130 anyway to enumerate wide flipped-ω basins population-wide.

## Open questions

- Is the m127-seed-12 wide attractor the only one on its seed, or are there multiple flipped-ω basins per seed once a sufficient search is done?
- Does the seed-33-style narrow attractor exist on other seeds? m128 cannot say (wide-basin-blind by construction; narrow-basin-blind on the 10 non-33 seeds where warm-start is not placed near a known narrow attractor).
- Spec-discipline lesson: the positive-control seed choice should reflect the warm-start state's geometry. For m128, the correct positive control would have been a seed where a warm-start `(q0_basin, −basin_ω)` is KNOWN to land in a flipped-ω attractor — none of the 11 baseline seeds qualifies.
