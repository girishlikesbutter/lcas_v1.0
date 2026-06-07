# Forward-Survey Workspace

This is a fresh workspace for surveying the **post-bug-fix forward model** of the LCAS inversion project. It is intentionally walled off from the prior inversion pipelines (`m103`, `m115`, `m126`, etc.) and prior numerical findings, almost all of which were measured against a buggy forward model and are tainted until re-validated.

If you are an agent working in this directory, read this entire README before doing anything else.

---

## Why this workspace exists

The propagator at `src/dynamics/attitude_propagator.py` was integrating the wrong quaternion-convention kinematic (RIGHT-mult Hamilton, producing convention-(b) `q`) for months. The renderer everywhere downstream applied `R(q) @ v_J2000`, which is correct only under convention (a). Conventions (a)/(b) are inverses (`q_a = q_b*`), so per-epoch every body-frame sun/observer vector was the conjugate of what it should have been.

The fix is two single-line edits to `_quaternion_multiply` ordering inside the propagator. It was identified, fixed, and validated end-to-end at machine precision on 2026-04-30 (commit `f7fabbe`, audit experiment `m139`). All 100 m048 trajectory NPZs were regenerated under the post-fix propagator (commit `ac1fdf4`); the buggy versions are preserved at `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed*_buggy.npz` if needed for reference. **They are NOT in this workspace by design.**

The buggy-vs-postfix LC delta (`m146`, commit `d16b9bc`) is catastrophic: all 100 seeds Band D, ρ median 48, max-|Δmag| median 8.9 mag, quaternion geodesic p99 ≈ 178.7°. The forward model under the bug was not a small perturbation of the correct model — it was approximately `q → q*` per epoch, geometrically scrambling every body-frame vector. **Every basin width, every cost-at-truth claim, every "previously-solved seed", every twin classification from months of pre-fix work is tainted until re-measured under correct truth.**

The pre-fix project lives at `notebooks/inversion/` (parent of this workspace). It is preserved frozen as reference. **Do not import from it; do not auto-load its docs.** If you need to look up what was tried, the wiki is at `notebooks/inversion/wiki/`. Reference only — do not let buggy-era heuristics leak in.

---

## Goal of this workspace

**The endgame is cohort-scale inversion of post-fix m048.** "Survey" names the principled starting phase — characterise the post-fix forward-model landscape (cost honesty, basin sizes, multi-solution presence, failure modes) before committing to a solver architecture — but the workspace's mission is to *do* the inversion, not just describe it.

**Success criterion (multi-solution acceptance, see `concepts/observational_indistinguishability.md`).** For each seed, return as many `(q0, ω)` candidates as exist that explain the observed LC at noise-comparable fidelity:

- **Band A (ρ < 2)** — admit. Truth-grade fit; below noise floor.
- **Band B (2 ≤ ρ < 4)** — admit. Acceptable fit; within ~2× noise.
- **Band C (4 ≤ ρ < 8)** — keep but flag as suboptimal. Visually presentable, not publishable.
- **Band D (ρ ≥ 8)** — reject.

Returning the geometric truth `(q0_truth, ω_truth)` is highly desirable and we hope it falls out, but it is a *subset* of the goal — not the gating criterion. The goal is to **maximise valid candidates per seed**. The truth basin is one of those candidates when LC information uniquely determines the attitude; on seeds where LC information under-determines `(q0, ω)` (e.g. low-rotation tumblers like seed 10, or seeds 28 / 41 / 48 / 84 where multi-solution candidates exist), the inversion correctly returns multiple valid attitudes — those are features of the data, not solver failures.

**Phasing — survey → proto-inversion → full inversion.**

The work proceeds incrementally from passive landscape characterisation to active cohort inversion:

1. **Survey-proper (s001–s010).** Where does cost place truth? Is the surrogate honest? How wide are the basins? — landscape and substrate questions.
2. **Proto-inversion at fixed truth-ω (s011–s014).** A concrete architecture (Sobol-Shoemake(q0) on SO(3) at N=64 + joint LM polish + lowest-surrogate-MSE selector) tested on a 9-seed pilot under controlled conditions (truth-ω given). Closed: surrogate ↔ hi-fi rank correlation cohort-wide (s014 Spearman 0.9952), surrogate-best == hi-fi-best on 9/9 seeds, multi-solution at cohort scale. **The q0-axis half of the global-search problem is solved at pilot scale.**
3. **Joint q0 × ω inversion pilot (next).** Extend the s011 architecture by adding the ω-axis: replace "joint LM at truth-ω" with "joint LM from a (q0, ω) IC distribution that doesn't assume truth-ω is known." Specific design pending — candidates include (a) Sobol-q0 × ω-grid (5×5 over the s003-measured tube) → 64 × 25 ICs/seed; (b) joint Sobol-(q0, ω) sampled from a wider product distribution; (c) adaptive ω refinement seeded by the q0-Sobol top candidates. Validation cohort: 5–10 PA-stratified seeds matching the s011 pilot, so the ω-axis cost can be isolated from the q0-axis result already in hand.
4. **Cohort-scale full inversion.** Once phase 3 settles the architecture, run on all 100 m048 seeds. Output: per-seed candidate list with `(q0_err, ω_dir_err, ω_mag_err, ρ, band)`, cohort-scale recovery yield distribution under multi-solution acceptance, failure-mode taxonomy for the genuine-failure tail.

**What about a 100-seed cohort scan at truth-ω (the natural extrapolation of s011)?**
This was originally penciled in as "phase 3" but is **not load-bearing for full inversion**: any joint q0 × ω scan that includes a truth-ω cell recovers it for free, and the s011 9-seed pilot + s014 cohort-architecture trust already smoke-test the q0 axis at pilot scale. Long-tail seed surprises (more seed-10-classes, more sub-Sobol-narrow seeds) absorb into multi-solution acceptance regardless. The truth-ω cohort scan remains an option as cheap diagnostic insurance before phase 3 launches but is not on the critical path.

**The one-page output the workspace eventually delivers** is no longer just a *map* of cost surfaces — it is the inversion result itself: per-seed, all candidates with `(q0_err, ω_dir_err, ω_mag_err, ρ, band)` reported, plus a cohort-scale recovery-yield distribution and a failure-mode taxonomy for the genuine-failure tail.

Survey-phase questions (still load-bearing inputs to the inversion phase):

1. **Cost-at-truth, cohort-scale.** Where does each cost place truth in its own pool? (s001 ✅: surrogate full-LC MSE Band-A on all 100 seeds; m103 alignment cost structurally inapplicable on 24/100 seeds; surrogate-MSE is the only cost defined on every seed.)

2. **Surrogate landscape probe.** Is the surrogate's argmin at truth-q0? (s002 ✅: argmin = truth on 8/8 PA-stratified seeds at truth-ω. s014 ✅: surrogate ↔ hi-fi rank correlation Spearman 0.9952 cohort-wide on the 9-seed pilot.)

3. **Failure-mode taxonomy.** Where does the global minimum live across `(q0, ω)`? Which regimes are honest vs deceptive? (s003 ✅: surrogate-MSE is ω-fragile — argmin = truth-q0 only inside a thin tube of ~1° dir / ~2-5% mag. s007 / s008 ✅: cheap LC-only ω priors structurally dead. s011 / s013 / s014 ✅: multi-solution is cohort-scale; 5 seeds out of 10 pilot have multi-solution candidates.)

---

## Hard rules for the agent working here

- **DO NOT import from `notebooks/inversion/11_casadi_formulation/m103_hybrid.py`, `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py`, `notebooks/inversion/m126_wrapped_pipeline.py`, or any other `mXXX*.py` script in the parent `notebooks/inversion/` tree.** These are buggy-era inversion pipelines. If you genuinely need a single subroutine (e.g. a glint-cost computation), copy + adapt + re-test it inside this workspace; do not import it.
- **DO use the forward-model substrate via `lib/`.** The propagator (post-fix), renderer, BRDF, surrogate, SPICE handler, satellite STL, articulation engine — all validated under correct truth. Import freely from `src.*` or use the `lib/` wrappers.
- **DO NOT cite buggy-era numbers** (m115 bridging radii, m121 basin widths, m135 cost-at-truth ratios, "previously-solved seeds", twin classifications, etc.) without re-measuring under correct truth in this workspace. See `concepts/known_pathologies_to_revalidate.md` for the catalogue of tainted claims.
- **DO start a fresh experiment numbering scheme.** First survey experiment is `s001_cost_at_truth_cohort`. The old `mXXX` series stays in `notebooks/inversion/wiki/` as frozen reference; do not extend it from here.
- **DO write a short experiment markdown for every numbered experiment** under `experiments/sXXX_*.md`, alongside the script `experiments/sXXX_*.py`. Frontmatter convention mirrors the old wiki: `title / type / sources / related / created / updated / confidence`.
- **DO append a one-line `## [YYYY-MM-DD] ingest | sXXX | one-liner` entry to `log.md`** per experiment. That's the survey's audit trail.
- **DO update `PROGRESS.md`** at the end of each session: current question, what was just measured, what's next.
- **DO commit when work is done.** Don't accumulate uncommitted survey artefacts.

---

## File layout

```
survey/
├── README.md                       # this file
├── PROGRESS.md                     # current state of the survey, hand-maintained
├── log.md                          # append-only ingest log (one line per experiment)
├── CLAUDE.md                       # local agent instructions (auto-loads alongside the project root CLAUDE.md)
├── concepts/                       # forward-model invariants and methodology rules
├── lib/                            # thin wrappers around src/ — the only inversion-side API
├── experiments/                    # sXXX_*.py scripts + matching sXXX_*.md write-ups
├── results/                        # NPZ/JSON/PNG outputs per experiment (sXXX_*/...)
└── data/trajectories/              # symlinks to post-fix m048 NPZs (no buggy siblings)
```

---

## Concept pages (read these once on first session)

These are forward-model invariants or methodology rules — they survive the bug fix and are NOT buggy-era findings.

- `concepts/quaternion_convention.md` — the bug, conv-(a) vs (b), the renderer formula, runtime gates worth adding
- `concepts/surrogate_model.md` — architecture, MAE, bridge-independent claim, ~50000× speedup, m145 caveat
- `concepts/q_omega_coupling.md` — (q0, ω_dir) jointly determine the LC; only |ω| is independent
- `concepts/omega_sign_degeneracy.md` — flipping ω while compensating with rotated q0 produces an LC-equivalent trajectory (kinematic, structural, NOT buggy-era)
- `concepts/twin_degeneracy.md` — LEFT-mult `q_180x · q0` for IS-901 ±X — geometric, structural
- `concepts/rho_band.md` — the metric: ρ = √(hifi_MSE / 0.05²), bands A/B/C/D, ρ < 4 acceptance bar
- `concepts/observational_indistinguishability.md` — ρ < 2 is a valid solution regardless of twin status
- `concepts/known_pathologies_to_revalidate.md` — catalogue of buggy-era claims that need re-checking under correct truth (NOT inherited as facts)

---

## Where to start

If this is the first session in this workspace:

1. Read this README.
2. Skim the eight concept pages.
3. Open `PROGRESS.md` to see current state of the survey.
4. Write `experiments/s001_cost_at_truth_cohort.py` to answer survey question 1.
5. Run it; produce `experiments/s001_cost_at_truth_cohort.md`.
6. Append a `log.md` entry, update `PROGRESS.md`, commit.

Each experiment is small, focused, and answers a single question. The survey ends when the three core questions above are answered.
