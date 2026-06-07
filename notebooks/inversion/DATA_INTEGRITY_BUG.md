# DATA INTEGRITY BUG — Pipeline Uses Wrong Trajectory Dataset & Wrong Observation Window

**Discovered:** 2026-04-16
**Status:** Bug 2 FIXED via Option A (2026-04-17 — commits `5d5938f`, `b906691`, `e96a866`). Bug 1 still open; Option B is the proposed remediation but not yet started. Option C (institutional layer) already landed in commit `5f4108c`.
**Severity:** Bug 1 still invalidates single-geometry generalization claims. Bug 2 was invalidating specific hi-fi MSE numbers from m122/m123/m124/m125/m126 — those numbers are now correct (correct-window re-run gave **10/11 seeds ≥10% improved, 33/33 basins helped, 5 OK + 4 PARTIAL + 2 FAIL**; [[gradient-based-inversion]] `#validated` REINSTATED).

---

## Bug 2 — Resolved 2026-04-17

The three script patches (`m122:328`, `m124:219`, `m126:347` — all three adding `end_time_utc='2020-02-05T11:00:00'`) landed in commit `5d5938f`. Re-run chain m122 → m123 → m124 → m125 → m126 (commit `b906691`) produced honest hi-fi numbers on the 1-hour window. Wiki sweep (commit `e96a866`) rewrote 10 pages + EXPERIMENTS.md Section 1 with the new numbers.

Additional findings from the re-run:

- **m119v2 also has Bug 2.** Bug-doc claim "m119v2 CORRECT" was wrong. Its setup_experiment lacks `end_time_utc`, so its stored setup.npz files spanned 21600 s. Those caches were quarantined as `setup_6hr_WRONG_WINDOW_retracted.npz` to prevent m122 from inheriting them. m119v2 page is already retracted per its own banner, so the script itself was not fixed in Option A. If m119v2 is ever revived, fix it first.
- **The bug hid BETTER results.** 18/33 basins "helped" on wrong window; 33/33 on correct window. Seed 12 went from "catastrophe (wrapper rejected)" to OK-class 99.2% improvement. Seeds 36 and 46 remain FAIL but improve 35% and 78% respectively.
- **m122 HYP3 flipped from CONFIRMED to REFUTED.** Basin-geometry cohort universality is gone — basin widths no longer explain ATT_FAIL. The driver is now upstream ω-direction quality.
- **Seed 33 flipped-ω degeneracy is real physics**, not a bug artefact. Hi-fi 0.0796 on correct window vs 0.0817 wrong — essentially unchanged.
- **[[surrogate-truth-offset]] retracted.** Truth IS a surrogate stationary point; April-16's non-zero `|grad|` at truth was FD-chart noise on the wrong window.

### Scripts that STILL have Bug 2 (not fixed by Option A)

These scripts call `setup_experiment(...)` without `end_time_utc` and will silently use the 6-hour default if re-run. Option A only fixed the three pipelines (`m122`, `m124`, `m126`) whose results were load-bearing for the `#validated` claim. The scripts below produced outputs that are either retracted or already characterised on wrong-window data; their data is on disk but their *scripts* are unfixed:

| script | line | status |
|---|---|---|
| `m111_shadow_isoshell.py` | 278 | UNFIXED — one-shot; results already logged on [[m111_shadow_isoshell]] |
| `m111b_shadow_epoch_hifi.py` | 190 | UNFIXED — one-shot |
| `m119v2_attitude_isoshell.py` | 314 | UNFIXED — page banner already retracted; setup.npz quarantined |

`m117_harvester.py` and `m118_kernel.py` import `setup_experiment` but do not call it — they read existing setup.npz files. No fix needed there.

**For Option B:** when you refactor `lib/experiment_setup.py` to accept per-seed `start_et`, grep all callers (`grep -rn "setup_experiment(" notebooks/inversion/ src/`) and thread the new argument through every one — including the three above. The list of consumers is NOT just m115/m122/m123/m124/m126/m127/m128/m129; it's ~15–20 scripts across 12_brightness_surface + 11_casadi_formulation. The refactor's first deliverable is that audit table.

### Bug 1 remains open

---

---

## TL;DR

1. **Bug 1 (architectural):** Every inversion pipeline reads `micro46_trajectories.npz` — a trajectory dataset where **all 100 seeds share a single fixed observation window (10:00–11:00 UTC on 2020-02-05)**. There is a correctly-designed alternative dataset `micro48_trajectories.npz` with **per-seed random start times** in `[08:35, 15:35 UTC]` with 1-hour duration each. No current pipeline uses it. Consequence: every result, finding, and "validated branch" in the project is on **ONE observation geometry**, not a diverse population.

2. **Bug 2 (script-level, compounding):** Within the micro46 architecture, `micro122_hessian_at_truth.py`, `micro124_hifi_validate.py`, and `micro126_wrapped_pipeline.py` call `setup_experiment(...)` **without** `end_time_utc='2020-02-05T11:00:00'`. The config default is `end_time: '2020-02-05T16:00:00'`, so these scripts silently ran on a **6-hour window** and scored their states against a completely different observed LC than micro46's truth. All hi-fi MSE values from these scripts are **on the wrong observation window** and not comparable to hi-fi MSE from micro115 / micro127 / micro128 / micro129 (which correctly pass `end_time_utc`).

---

## Bug 1 — Wrong Trajectory Dataset

### The two datasets

| dataset | file | `observation_times` shape | start-time logic | intent |
|---------|------|:-------------------------:|------------------|--------|
| `micro46_trajectories.npz` | `data/results/inversion_diagnostics/micro46_trajectories/micro46_trajectories.npz` | `(500,)` — shared | fixed `start=2020-02-05T10:00:00`, `end=2020-02-05T11:00:00` | prototype; generated by `notebooks/inversion/09_glint_analysis/micro46_generate_trajectories.py` |
| `micro48_trajectories.npz` | `data/results/inversion_diagnostics/micro48_trajectories/micro48_trajectories.npz` | `(100, 500)` — per-seed | `start_ets[seed]` uniform in `[2020-02-05T08:35:00, 2020-02-05T15:35:00]`; `duration=3600 s` | generalization across geometries; generated by `notebooks/inversion/09_glint_analysis/micro48_generate_trajectories_v2.py` |

### What the user stated was intended

> "the trajectory list was supposed to be generated with an arbitrary start time between 9am and 3pm or something like that, and to creat[e] a duration of 1 hr in order to get the end time"

This describes `micro48_trajectories.npz` exactly (08:35 → 15:35 is close to "9am–3pm", 1-hour duration).

### What the pipeline actually does

- `grep -rn "micro46_trajectories" notebooks/inversion` → **265 occurrences across 142 files** (including every pipeline: micro115 through micro129, lc_compare, attitude_viz, etc.).
- `grep -rn "micro48_trajectories" notebooks/inversion` → appears only in its own generator script.
- **Zero current pipelines use micro48.**

### Consequences of Bug 1

- **"100-seed population" statistics are actually "100 trajectories on ONE observation night."** The [[ipl-census]] (87% constraint-poor), the [[basin-of-attraction]] cohort-universality claim, and every per-seed solution count reflect that single geometry.
- **Generalization claims are untested.** The wrapped pipeline's "6/11 improve ≥10%", the flipped-ω frequency of "2/11 seeds confirmed", and the "±X twin frequency" all are geometry-specific.
- **Observation-geometry-induced degeneracies may disappear or reappear under micro48 windows.**
  - Seeds 14, 24, 93 have near-perfect ±X twin solutions under micro46's 10–11 window. Under a different start time, the symmetry may break earlier/later in the LC.
  - Seed 33's flipped-ω basin (hi-fi 0.082) exploits a time-reversal symmetry specific to the 10–11 geometry. Reproducing it on a different start time is uncertain.
  - The [[dark-mag-saturation]] plateau width and the seed-specific ω-anisotropy from [[m121_basin_width_metric]] depend on the specific sun/observer geometry.

---

## Bug 2 — Missing `end_time_utc` In Several Scripts

### Audit results (2026-04-16)

| script | passes `end_time_utc='2020-02-05T11:00:00'`? | scoring window | status |
|--------|:--------------------------------------------:|----------------|:------:|
| `micro115_surrogate_pipeline.py` | YES (line 719) | 1-hour | ✓ CORRECT |
| `micro119v2_attitude_isoshell.py` | YES | 1-hour | ✓ CORRECT |
| `micro122_hessian_at_truth.py` (line 328) | **NO** | 6-hour default | ✗ BROKEN |
| `micro123_lbfgs_polish.py` | inherits setup from micro122/micro119v2 | depends on source | ✗ likely BROKEN (inherits micro122) |
| `micro124_hifi_validate.py` (line 219) | **NO** | 6-hour default | ✗ BROKEN |
| `micro125_keep_better_inline.py` | inline re-score of micro124 | 6-hour | ✗ BROKEN (inherits micro124) |
| `micro126_wrapped_pipeline.py` (line 347) | **NO** | 6-hour default | ✗ BROKEN |
| `micro127_flipped_omega_search.py` (line 492) | YES | 1-hour | ✓ CORRECT |
| `micro128_warmstart_polish.py` (line 700) | YES | 1-hour | ✓ CORRECT |
| `micro129_densegrid.py` (line 505) | YES | 1-hour | ✓ CORRECT |

All other pipelines (micro60s–micro114) not yet audited. The pattern was:
- micro115 and earlier inherited the invariant.
- micro119 v1 broke it → [[m119v2_attitude_isoshell]] fixed it.
- micro122 (2026-04-16 morning) re-broke it. micro123/124/126 copy-pasted from micro122 without catching the bug.
- micro127 (2026-04-16 evening) built from micro115 template → correct again.

### Evidence (seed 0, micro126)

Loaded `data/results/inversion_diagnostics/micro126_wrapped/seed_000/hifi_ckpt.npz`:

| quantity | value |
|---|---|
| Reported `hifi_before[0]` (basin 0 pre-polish) | 0.1399 |
| Reported `hifi_after[0]` (basin 0 post-polish) | **0.1119** |
| Stored `hifi_mags_after[0]` vs `m46['mag_hifi'][0]` | **MSE = 6.362** (**RMS = 2.522 mag**) |
| Ratio claimed / actual-vs-m46-truth | **57×** |
| Regenerated LC from `(q0_after, omega_after)` on **correct 1-hour window** | MSE = 0.325 (RMS = 0.570) |
| Ratio claimed (on 6-hr) / actual (on 1-hr) | **2.9×** |

The stored LC was computed on the 6-hour ctx — its MSE vs the 6-hour observed_lc was genuinely 0.112, but the 6-hour observed_lc ≠ m46 truth. Regenerating the same state on the correct 1-hour window gives MSE = 0.325 — still **2.9× WORSE than micro126 reported**.

### Scope of corruption

**What's corrupted (hi-fi MSE numbers):**
- [[m122_hessian_curvature]]: all Hessian-at-truth numbers (eigenvalues, basin widths, gradient magnitudes)
- [[m123_lbfgs_polish]]: "polish improves cost 4-7×" claims — computed on wrong-window surrogate cost
- [[m124_hifi_validate]]: "2/12 basins agree within ±30%" ratio-agreement verdict (REFUTED framing)
- [[m125_keep_better_inline]]: "90%/33%/70% hi-fi MSE improvement on seeds 14/74/93" (the keep_better wrapper finding)
- [[m126_wrapped_pipeline]]: "6/11 improved ≥10%, 5/11 break-even, 0/11 regressed" — the core #validated claim
- The [[gradient-based-inversion]] branch's `#validated` status is UNVERIFIED

**What's NOT corrupted:**
- `q0_err_deg`, `w_dir_err_deg`, `w_mag_err_pct` — all computed from truth state independently of window
- State (q0, ω) discovery via DE (micro115) — search was on correct 1-hour window
- [[m115_surrogate_pipeline]] basin enumeration — valid
- [[m127_flipped_omega_search]], [[m128_warmstart_polish]], [[m129_dense_grid_eval]] (flipped-ω thread) — correctly windowed
- Structural / symmetry observations (twin frequency, etc.) — depend on geometry but not on hi-fi MSE accuracy

### Why it went undetected

- The micro126 reviewer checklist was scope-limited to mechanical conventions (`Pool(8)`, atomic JSON, N_MAGS≥20) per the `feedback_reviewer_scope.md` memory. Data-plumbing (`end_time_utc`, window match) was not in scope.
- The writer prompt did not list `end_time_utc='2020-02-05T11:00:00'` as a load-bearing invariant.
- The strategist's "pre-launch sanity check" was a spec-level review, not a runtime LC-comparison check.
- The analyst-phase did not include an LC visual-sanity step; it processed hi-fi MSE numbers face-value.
- No 1-seed dry-run + LC-compare step between writing the script and running the full 11-seed batch.

---

## Why This Matters for Every Claim Made So Far

Aggregated effect of both bugs:

1. **Research has been done on one fixed observation geometry** (Bug 1). Every "cohort-universal" / "X% of seeds" / "population-wide" finding is effectively a single-sample result about IS-901 under 10–11 UTC on 2020-02-05.
2. **The #validated wrapped pipeline's quality is unknown** (Bug 2). Regenerating on the correct window gives MSE 0.325 for seed 0 vs claimed 0.112 — nearly 3× worse.
3. **The "micro130 pre-run prior is NEGATIVE" assessment from the wind-down** was based on cost-landscape characterization (micro121, micro122) that itself may be on a wrong-window ctx. The flipped-ω thread closure rests on partly-compromised evidence.

---

## Suggested Fix Plan (for next research loop to evaluate — NOT to act on now)

### Option A (short-term, within micro46 architecture)

1. **Fix Bug 2 only.** Add `end_time_utc='2020-02-05T11:00:00'` to `setup_experiment(...)` calls in `micro122`, `micro123` (or verify its inherited setup source), `micro124`, `micro126`.
2. **Re-run** `micro126_wrapped_pipeline.py` for all 11 seeds. ~15 min wall with the existing Pool(8).
3. **Re-score** micro122/123/124/125 without re-running (load the stored states and compute hi-fi MSE on the correct window). ~20 min.
4. **Update all wiki / EXPERIMENTS.md claims** that depend on these numbers.

**Cost:** ~1 session. **Value:** restores honesty within the micro46 architecture but doesn't address Bug 1.

### Option B (medium-term, fixes Bug 1)

1. **Modify `lib/experiment_setup.py`** to accept a per-seed `start_et` (or equivalent) rather than the fixed-window config defaults.
2. **Migrate every pipeline** (at least micro115, micro126, micro127-129 for starters) to read `micro48_trajectories.npz` instead of `micro46_trajectories.npz`. Per-seed `start_et = m48['start_ets'][seed]`, `end_et = start_et + 3600`.
3. **Re-generate all truth LCs** on per-seed windows. The `mag_hifi` array in micro48 is already on per-seed windows — use it directly.
4. **Re-run the full research arc on micro48** — surrogate DE, wrapped pipeline, flipped-ω (if still interesting), basin characterization. This is a large session-count commitment.

**Cost:** ~5–15 sessions. **Value:** gives honest generalization claims across observation geometries. This is probably what the user actually wants.

### Option C (institutional)

Regardless of A or B, add to the research-loop skill:
1. **`DATA_INVARIANTS.md`** in the project root, listing load-bearing parameters (trajectory dataset, window, N_OBS, noise seed, inertia tensor source, etc.). Writer prompts MUST cite it by name.
2. **Reviewer checklist extension** — data-plumbing items (does this match micro48? does `end_time_utc` appear? is `n_observations == 500`?).
3. **Mandatory 1-seed dry-run + LC-compare sanity step** between writer review and full batch launch. If the dry-run's regenerated LC doesn't visually match truth, the full batch aborts.

---

## Commit Status as of This Document

- All the "broken window" experiments (micro122, 123, 124, 125, 126) are **already committed** to git. Their hi-fi MSE numbers are in the repo history.
- The wiki pages, EXPERIMENTS.md, and memory files reference those numbers as truth.
- Rolling back is NOT recommended — the states, scripts, and analyses have value. What's needed is a CORRECTION layer documenting which numbers are wrong and replacing them with recomputed values.

---

## Files With Quantitative Claims That Need Revision

**In priority order (highest = most load-bearing for current "best method" narrative):**

1. `notebooks/inversion/EXPERIMENTS.md` — Section 1 "Resume Point" for 2026-04-16 evening (micro126 promotion to #validated).
2. `notebooks/inversion/wiki/wiki/branches/gradient-based-inversion.md` — `#validated` status, the "6/11 improved" table, the three NOTE blocks.
3. `notebooks/inversion/wiki/wiki/experiments/micro126.md` — all per-seed hi-fi numbers.
4. `notebooks/inversion/wiki/wiki/experiments/micro124.md` — the "2/12 ratio" REFUTED framing.
5. `notebooks/inversion/wiki/wiki/experiments/micro125.md` — the 90%/33%/70% improvement claims.
6. `notebooks/inversion/wiki/wiki/experiments/micro122.md` — Hessian eigenvalues, basin-width numbers.
7. `notebooks/inversion/wiki/wiki/experiments/micro123.md` — polish improvement claims.
8. `notebooks/inversion/wiki/wiki/concepts/omega-sign-degeneracy.md` — seed 33's hi-fi 0.082 (computed on 6-hr ctx; correct value on 1-hr unknown).
9. `notebooks/inversion/wiki/wiki/concepts/basin-of-attraction.md` — [[m122_hessian_curvature]]-derived numbers.
10. `notebooks/inversion/wiki/wiki/concepts/surrogate-model.md` — "10-15× hi-fi disagreement" claim from [[m124_hifi_validate]].

**All of the above are annotated with numbers computed on the 6-hour ctx — EVERY one of them is incorrect.**

---

## Additional Inline [inline] Claims Affected

- Every [inline] entry in `log.md` dated 2026-04-16 that cites a hi-fi MSE number from micro122/123/124/125/126.
- The "wrapped pipeline is the best method" strategic summary I (strategist) gave the user this session — INCORRECT under the correct-window re-scoring (seed 0 best is 0.325 not 0.112; seed 33 flipped-ω at 0.082 is actually on wrong window, so the 1-hr value could be very different).

---

## What Was Correctly Measured (Still Trustworthy)

- DE discovery of basins (micro115 states): basins 0, 1, 2 per seed are real attractors of the 1-hour surrogate cost.
- The fact that most DE winners end up at ±X twin or near-truth q0: an observation about the cost landscape, not about hi-fi values.
- The [[m127_flipped_omega_search]]/[[m128_warmstart_polish]]/[[m129_dense_grid_eval]] flipped-ω findings: correct-window, correctly-REFUTED.
- The [[m115_surrogate_pipeline]] "10/10 seeds have hi-fi < 1.0" statement: computed on correct 1-hour window (micro115 scripts pass `end_time_utc`).

---

## Recommended Next Research Loop Step

**Do NOT run any new experiments until the audit and fix plan is chosen.** Running micro130 or any successor on the corrupted architecture just adds more invalid data.

The next loop's **first agenda item** should be:
- Decide between Option A (fix Bug 2 only) and Option B (migrate to micro48).
- If Option A: re-score the 5 affected experiments, update wiki, reassess "best method" claim.
- If Option B: design the setup_experiment refactor, then run small-scale validation on micro48 before full migration.

Either way, Option C (institutional fix) should happen in parallel.
