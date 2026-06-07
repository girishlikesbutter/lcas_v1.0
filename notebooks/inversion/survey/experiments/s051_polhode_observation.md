---
title: "s051 — polhode-conditioned ω prior identified by direct visual observation"
type: insight
sources:
  - "s048c_cloud_viewer (s048c+ side panels)"
  - "data/trajectories/traj_seed089.npz"
  - "s042_basin_radius_cohort"
related:
  - "s048c_cloud_viewer"
  - "s042_basin_radius_cohort"
  - "concepts/polhode_prior"
  - "wiki/concepts/polhode-prior"
created: 2026-05-07
updated: 2026-05-07
confidence: medium-high (architectural; numbers below need cohort-scale confirmation)
---

# s051 — polhode observation: ω prior is a 1-D curve, not a 3-vector

## TL;DR

Direct visual observation of the s048c+ viewer's body-frame ω-direction
sphere panel on seed 89 frame 499 surfaced a structural reformulation of
the ω prior. The ω̂(t) trail traces a closed loop on the unit sphere — the
**polhode** of torque-free rigid-body motion. Polhodes are a 1-parameter
family of closed curves in body-frame ω-space, fully determined by the
satellite's inertia tensor (known a priori). The inversion's ω prior
should be parameterized as `(|L|, polhode-label, polhode-phase)` rather
than raw `(ω_x, ω_y, ω_z)` — same dimensionality, dynamics-informed,
every sample admissible. The amplitude of `|ω|(t)` wobble is a direct
geometric readout of polhode size.

## What

No code was run. The insight is an observation derived from the
s048c+ viewer (commit `6a789c3`) on the cached 500-epoch / seed-89 run:

1. The body-frame ω-direction sphere panel renders ω̂(t) as a moving
   point on the unit sphere with a permanent trail.
2. At frame 499 (one full hour of observation), the trail is a small
   nearly-closed loop sitting close to the body −X axis.
3. The `|ω|(t)` strip beside it shows a flat line at 0.241 dps with
   std/mean = 0.5%.

These two observations together identify the trajectory as a small
polhode near a principal axis (quasi-stable rotation with mild wobble).

## How (no compute)

The viewer was opened in a real browser (after the user reported the
s048c+ panels rendering correctly) at the 500-epoch seed-89 cached run.
Visual inspection at frame 499 showed the closed-loop polhode. Verbal
identification of the loop as a polhode (intersection of energy and
momentum ellipsoids in body frame for torque-free rigid-body motion)
followed immediately, leading to the architectural reframe documented in
`concepts/polhode_prior.md` and the wiki page `wiki/concepts/polhode-prior.md`.

## Result

**Architectural finding (not yet a measured number):** parameterize ω as

```
ω prior = (|L|, polhode_label, polhode_phase)
```

where `|L|` sets the overall scale (≈ |ω| × I_eff), `polhode_label =
2T·I_max / |L|²` selects the closed contour in body-frame ω-space,
and `polhode_phase ∈ [0, τ_p)` selects the starting position on that
contour. The inertia tensor `I` is known a priori for the m048
satellite (`lib.hifi_render` → `compute_inertia_from_config`).

**Empirical data point (seed 89, m048):**

| metric | value |
|---|---|
| `\|ω\|_mean` | 0.2415 dps |
| `\|ω\|_std/mean` (full traj) | 0.5% |
| polhode shape (visual) | small closed loop, near body −X |
| LC peak count | 14 |
| classification | tiny polhode → quasi-stable rotation |

**Predicted but unmeasured (forward path):**

For the high-|ω| / many-spike cohort tail (seeds 14, 17, 68, 81), the
visualisation should show:

- substantial `|ω|(t)` oscillation (predicted std/mean 5–30%)
- ω̂ sweeping a large polhode close to the separatrix
- correlation between polhode angular extent and the `s042`-measured
  ω-mag basin width

If both (large polhode) and (narrow basin) are confirmed jointly across
the cohort tail, the polhode prior promotes from `medium-high` to
`#validated` and becomes the load-bearing reformulation for
cohort-scale inversion sampling.

## Why this matters

Three downstream consequences:

1. **Sample efficiency.** Uniform sampling of `ω` on `S²` × magnitude is
   wasted on dynamically-inadmissible directions. Sampling on
   `(polhode_label, polhode_phase)` is dynamics-admissible by
   construction.

2. **LM polish.** The cohort-scale bottleneck per `[[gradient-based-inversion]]`
   is LM convergence radius in ω. Constraining ω updates to project on
   the polhode tangent reduces the ω step from 3-DOF to 1-DOF; the
   "ω-noise" cascade-failure mode of `s050b` may collapse from
   3-D random-walk in ω-space to 1-D random-walk along the polhode.

3. **Mechanism for `s042`.** The cohort-binding result that ω-mag basin
   width inversely scales with `|ω|` is currently a phenomenology with
   no underlying explanation. The polhode framing predicts: high-|ω|
   seeds happen to land on near-separatrix polhodes; mis-specifying the
   polhode-label scalar moves you to a structurally different
   trajectory and the LC fit breaks → narrow basin. Low-|ω| seeds are
   on small polhodes near principal axes; mis-specification is mostly
   absorbed by q0 sliding the orbit along the small contour → wide
   basin.

## Numbers (seed 89)

```
|ω|_mean      = 0.2415 dps
|ω|_std       = 0.0013 dps
|ω|_std/mean  = 0.5%
ω_body components std (dps):
   ω_bx       = 0.007  (mean -0.232)
   ω_by       = 0.031  (mean -0.031)
   ω_bz       = 0.032  (mean -0.036)
omega0_rad (initial body) = (-0.00405, +0.00054, -0.00091) rad/s
                          ≈ (-0.232, +0.031, -0.052) dps
                          mostly along body −X
```

Body-axis component variation (≈ 13% on y/z, 3% on x) is consistent
with a small polhode encircling the body −X principal axis.

## Out of scope (deferred to s052+)

- Computing the polhode family explicitly from the inertia tensor for
  m048's IS-901 satellite.
- Rendering polhodes for high-|ω| cohort seeds (14, 17, 68, 81) in the
  s048c+ viewer; measuring polhode angular extent.
- Lomb-Scargle on `|ω|(t)` to extract polhode period τ_p.
- Cross-reference τ_p against LC peak spacing.
- Polhode-conditioned grid sampler implementation.
- Polhode-tangent-projected LM polish.

## Methodology note (load-bearing)

This insight came from **direct visual observation** of a viewer the
agent built and screenshotted itself. The same numerical data
(body-frame ω at every epoch in `traj_seedXXX.npz`) has been cached for
months and was never the seed of this reframe in any prior session.
The visual feedback loop — multi-panel synchronized animation rendered
to HTML, screenshotted via headless Chrome, parsed visually as the next
turn's input — is what made the polhode topology immediate.

This is recorded as feedback memory `feedback_visual_artefacts_unlock_insight`:
when working on geometric or dynamic problems with non-obvious
structure, prioritize building visualization tooling. The cost of a
viewer is recouped many times over via insights that wouldn't otherwise
emerge.

## Artefacts

- `concepts/polhode_prior.md` (survey-side lean concept page)
- `notebooks/inversion/wiki/wiki/concepts/polhode-prior.md` (definitive wiki concept page)
- `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/project_polhode_prior.md`
- `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/feedback_visual_artefacts_unlock_insight.md`
- `s048c_viewer/viewer_template.html` (the visualization that surfaced the insight; commit `6a789c3`)

## Cross-references

- `s042_basin_radius_cohort.md` — the cohort-binding ω-mag basin / |ω| inverse correlation that the polhode framing claims to explain mechanistically.
- `s048c_cloud_viewer.md` — the viewer with the body-frame ω-direction sphere panel that surfaced the closed-loop polhode visually.
- `[[l-conservation]]` (wiki) — the polhode is on the intersection of the energy ellipsoid AND the momentum ellipsoid; momentum alone is insufficient.
- `[[omega-magnitude-estimation]]` (wiki) — needs revision: previously framed polhode as "complication that breaks periodicity"; under the polhode prior it becomes the structural prior that *replaces* periodicity-based estimation.
