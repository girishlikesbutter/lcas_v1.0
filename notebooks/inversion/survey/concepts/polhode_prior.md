---
title: "ω lives on the polhode — a 1-D dynamics-informed prior"
type: concept
created: 2026-05-07
updated: 2026-05-22
confidence: medium-high
---

# Polhode-conditioned ω prior

## What

For a torque-free rigid body, body-frame `ω(t)` is confined to a closed 1-D curve — the **polhode** — set by the intersection of the energy and momentum ellipsoids. Polhodes form a 1-parameter family that fully tessellates the body-frame ω-space; the family is fixed by the inertia tensor `I` (known a priori for the m048 satellite via `lib.hifi_render`).

The unknowns for inversion are then:

- `|L|` (1 scalar) — overall scale of the polhode (equivalent to `|ω|` up to the inertia ratio)
- polhode-label (1 scalar, `2T·I_max / |L|²`) — which contour
- polhode-phase (1 scalar, `[0, τ_p)`) — where on the contour the trajectory starts

Three scalars, same dimensionality as raw ω, but each one carries dynamics — uniform sampling on this parameterization keeps every candidate dynamics-admissible.

## Visual signature

Polhode size in body frame ≡ amplitude of `|ω|(t)` wobble:

- **Tiny polhode** (near principal axis): `|ω|(t)` essentially flat; ω̂ traces a small loop near a fixed body axis. Quasi-stable rotation.
- **Large polhode** (near separatrix): `|ω|(t)` oscillates substantially; ω̂ sweeps large arcs across the body sphere. Tumbling.

The s048c+ viewer's body-frame ω-direction sphere panel renders this directly. Seed 89 frame 499 shows a small closed loop near body −X (`|ω|_std/mean = 0.5%`) — quasi-stable, principal-axis-locked.

## Why this matters for the survey

`s042` showed that the **ω-mag basin width inversely correlates with |ω|**. The plausible mechanism: high-|ω| cohort seeds are on near-separatrix polhodes (large amplitude, |ω|(t) oscillates substantially). Mis-specifying the polhode-label scalar moves you to a structurally different ω̂(t) trajectory and breaks the LC fit — narrow basin. Mis-specifying it on a small polhode (low-|ω| seed) is partially absorbed by q0 — wide basin.

If true, the architectural fix is to sample `(|L|, polhode-label, polhode-phase)` rather than raw ω, and to constrain LM polish so ω updates project onto the polhode tangent (1-DOF instead of 3-DOF).

## What to measure (open)

- Polhode size (max angular extent on the unit sphere) for the m048 high-|ω| cohort tail: **seeds 14, 17, 68, 81**. Predicted large, near-separatrix.
- Correlation between polhode size and s042's measured ω-mag basin width. Predicted strong positive.
- Polhode period τ_p vs LC peak spacing. Predicted recoverable via Lomb-Scargle on the LC.
- Sample-efficiency comparison: polhode-conditioned grid vs uniform-ω grid at fixed seed-yield.

## Status of this representation in the pipeline (2026-05-22 discussion)

Captures a session discussion of *why this representation has not been used as the search coordinate*, and the s087 evidence that bears on it.

**Used today (three roles):**
- **Propagation** — the closed-form Jacobi propagator (`lib/jacobi_propagator.py`, s072/s074) integrates from the conserved `(|L|, 2T)` analytically.
- **Polish** — `s064` reparameterizes the LM ω-update onto the polhode tangent basis (1-DOF along the curve vs 3-DOF in raw ω), the source of its 21–27× in-basin speedup (`results/s064_jacobi_polish/gate2_smoke_parity.json`).
- **Regime diagnostic** — `disc = 2T·I_2 − |L|²` classifies LAM/SAM per candidate (explicitly not an oracle look-up; plan §5).

**Not used:** the candidate-generation grid is still **raw** — a Fibonacci sphere of ω-directions × an LS-bracket of |ω| (the s082 architecture). The `(|L|, 2T, phase)` triple has never been the *search* coordinate; s083→s087 diagnosed the raw-grid architecture one axis at a time.

**The s087 connection (the new observation):** s087 measured that fast-tumbler truth recovery needs the ω-direction within **~1.25°** (seed 119 cliff 1.25°→1.50°, source: `results/s087/summary.json`). On a raw 2-sphere that tolerance needs ~9–10k grid points (derived from the s082 worst-case 2.69° + sphere-packing scaling — unverified). For a *fixed* `(|L|, 2T, regime, sign)` the body-frame ω-direction is pinned to the **1-D polhode curve**, with phase the only freedom — so the direction search would collapse from a 2-sphere to a 1-D phase scan, which is exactly the axis s087 found binding. This is the strongest concrete motivation to date for adopting the representation as the search coordinate.

**Honest caveats — what the discussion did NOT establish:**
- **Same continuous dimensionality.** `(|L|, 2T, phase)` is 3, like `(dir_θ, dir_φ, |ω|)`. The win is geometric (the tight-tolerance axis becomes 1-D) + admissibility, not fewer parameters.
- **You still search `(|L|, 2T)`** as a 2-D pair (`|L|` ~ `|ω|` via inertia, `2T` ~ energy); whether they are as cleanly refinable as the |ω| 1-D basin (s083/s084) is untested.
- **`(regime, sign)` is discrete** — classifiable from `(|L|, 2T)` via the discriminant, partly absorbed by the body-X twin halving, but a 2–4× branch enumeration.
- **The point-count win is UNMEASURED** — whether a `(|L|, 2T)`-grid × phase-scan hits <1.25° in fewer total candidates than the 2-sphere × |ω| grid depends on how direction-sensitivity distributes between `(|L|, 2T)` and phase.
- **Two prior "refutations" do NOT block this use.** `s055b`/`s055c` refuted *predicting* ω-direction from LC features (a regression, not a search reparameterization). `s052` refuted the polhode prior as a *downstream filter* on a noisy cascade pool — pre-bug-fix and a different role; the reparameterization survives for *upstream sampling* per `project_polhode_prior.md`.

## Cross-references

- **Natural integration basis: `concepts/jacobi_propagation.md`** — the polhode prior's `(|L|, label, phase)` triple ARE the Jacobi-elliptic-solution parameters. Sampling on this basis with closed-form Jacobi evaluation (queued as s062) replaces "sample uniform ω + project onto polhode + propagate numerically" with one analytical step. Auto-memory: `project_jacobi_propagation_priority.md`.
- Wiki concept: `notebooks/inversion/wiki/wiki/concepts/polhode-prior.md` (definitive page, Obsidian-linked from existing concepts; cites this survey concept as a source).
- Source experiment: `experiments/s051_polhode_observation.md`.
- Auto-memory: `project_polhode_prior.md`.
- Methodology lesson: `feedback_visual_artefacts_unlock_insight.md` — this insight came from direct visual observation of the s048c+ viewer; the same numerical data had been cached for months and never seeded this reframe.
- Related survey concept: `q_omega_coupling.md` — q0 and ω̂ jointly determine the LC; restricting ω to its polhode is the "ω̂-side" structural cut.
- Related wiki concepts: `[[l-conservation]]`, `[[omega-magnitude-estimation]]`, `[[brightness_surface_path_matching]]`, `[[basin-of-attraction]]`, `[[surrogate-attitude-isoshell]]`.
