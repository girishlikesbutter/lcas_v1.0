---
title: "s073f — cluster_457 local geometry: the seed-89 Band-A attractor is a locally isolated point"
type: experiment
sources:
  - results/s073f/summary.json
  - results/s073f/states.npz
  - results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json  (cluster_457 polished state)
related:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073d_polhode_match_cluster457.md
  - experiments/s077_l_vector_basin_sweep.md
created: 2026-05-14
updated: 2026-05-14
confidence: high (decisive on the narrow question; 63 hi-fi renders, every number traced to results/s073f/summary.json; N=1 — one cluster on one seed)
---

# TL;DR

On post-fix seed 89, `cluster_457` (the lone Band-A multi-solution attractor, q0 59.58°
off truth, hi-fi ρ=0.904) is a **locally isolated point** in 6-DOF state space: the hi-fi
LC residual grows fast in *every* direction probed — L-direction rotation, the orthogonal
in-plane rotation, the twist-about-L control, and polhode scaling. It exits Band A∪B at
±2° of any inertial rotation and ±0.5% of polhode scaling. This **closes the charitable
("it's just polish residual") reading** that s073d left open: the ~1.1% Casimir mismatch
between cluster_457 and truth is a real geometric distinction, not a soft direction the LC
can't see. It is a narrow, expected result — it settles one sub-question and does **not**
bear on whether cat-4-style structure exists cohort-wide.

# What

s073/s073d/s077 chain. s073d found that cluster_457 shares truth's body-frame polhode only
*approximately* — `2T` and `|L|²` each differ ~1.1% — while its inertial `L_J2000` direction
is 128.6° off truth. s073d could not decide, from one pair, between:

- **(i) strict** — the 1.1% Casimir mismatch is a real geometric distinction; cluster_457
  is genuinely off truth's polhode.
- **(ii) charitable** — the s059k LM polish minimised LC residual, not polhode distance;
  the 1.1% could just be polish residual, i.e. the LC is insensitive to a 1% polhode change.

This experiment is the local-geometry test that distinguishes them: perturb cluster_457's
state in controlled directions and measure how fast the hi-fi LC residual (vs seed-89 truth)
grows.

# How

`experiments/s073f_cluster457_local_geometry.py`. From cluster_457's polished `(q0, ω0)`,
build 62 perturbed states along two families and render all 63 hi-fi LCs on `Pool(24)`:

- **Direction (a) — L-direction sweep (body polhode held fixed exactly).** A global inertial
  rotation applied to `q0` with `ω0` (body) unchanged. Because `2T = ω·Iω` and `|L|² = |Iω|²`
  depend only on `ω0` and `I`, the body-frame polhode is preserved to machine precision; only
  the inertial direction of `L_J2000` rotates. Three rotation axes: `axis_1` rotates `L`
  *toward* truth's `L`; `axis_2` is the orthogonal in-plane direction; `axis_3` twists about
  `L` itself — a **control**, since `L_J2000` does not move at all.
- **Direction (b) — polhode/Casimir sweep (`L_J2000` direction held fixed exactly).** A
  uniform scaling `ω0 → (1+ε)·ω0` with `q0` unchanged: `|L|` scales by `(1+ε)`, `2T` by
  `(1+ε)²` — moving to a different polhode — while `I·ω0` keeps its body-frame direction so
  `L_J2000` keeps its inertial direction exactly. cluster_457 differs from truth by ~1.1% in
  *both* Casimirs, so this is the matched probe for the s073d offset.

Convention self-checks (Casimir preservation in (a), `L`-direction preservation and `|L|`
scaling in (b)) asserted to 1e-9 on all 62 perturbed states; base render cross-checked
against the cached s069 ρ=0.904 as a pipeline gate.

# Result

Base render ρ = 0.9041, band A — **exactly** matches cached s069; convention self-checks
and pipeline gate PASS. Reading the ρ sweeps (`results/s073f/summary.json`):

| Perturbation direction | Band-A edge | Band A∪B edge (ρ<4) |
|---|---|---|
| (a) `axis_1` — `L` rotated *toward* truth | ±1° | ±2° |
| (a) `axis_2` — `L` rotated orthogonally | +1° | ±2° |
| (a) `axis_3` — twist about `L` (**control**) | ±1° | ±2° |
| (b) ω-scale — polhode / Casimir change | (base already C-adjacent) | ±0.5% |

**Both directions are stiff.** The script's own decision rule: (a) soft + (b) stiff → strict
reading; (a) soft + (b) soft → charitable reading; **both stiff → isolated point**. The data
lands on *both stiff*.

The decisive piece is `axis_3`, the control — a pure body-twist about the `L` axis, which
moves neither `L_J2000` *nor* the polhode. If cluster_457 sat on a soft sheet with an
"L-direction" tangent, `axis_3` (the other fixed-`(L,2T)` degree of freedom) should have been
soft. It is not: ρ goes 1.67 → 3.23 → 7.07 over ±1° / ±2° / ±5°, the same stiffness as the
genuine L-direction sweeps. There is **no soft direction at all** in the local 6-DOF
neighbourhood of cluster_457.

# Why this matters

This **closes the charitable reading** of s073d. cluster_457's ~1.1% Casimir mismatch from
truth is a real geometric distinction — the LC *is* sensitive to a 1% polhode change (it
exits Band A∪B at ±0.5% scaling), so the mismatch cannot be dismissed as polish residual.
The strict reading of s073d holds: on seed 89, cluster_457 is genuinely off truth's polhode,
and it is an isolated Band-A point rather than a sample from a soft (let alone continuous)
family.

This is consistent with — and was the expected outcome under — the user's reframe: nobody
expected a single seed to host a continuum of attractors. The cat-4 question is a
*cohort/regime* question (does the equal-|L|/free-direction structure appear across seeds,
and does it track the tumbling regime?), not a local-geometry question about one point. s073f
just removes the polish-residual escape hatch so that s073d's "different polhodes" reading is
load-bearing for the cohort work (s079).

# What this does NOT establish

- **Nothing cohort-wide.** N=1 — one cluster on one seed. s073f says cluster_457 *specifically*
  is locally isolated; it does not say cat-4 is dead, nor that other seeds' multi-sols are
  isolated.
- **Nothing about global structure.** The probe is *local* (max ±128.6° rotation, ±10%
  scaling, from one base point). A separate isolated point elsewhere in state space is not
  excluded — s077 in fact found discrete competing basins on other seeds.
- **Not a statement about cat-4's publication status.** That still depends on the cohort
  sweep (s079 onward) and the theorem half-page, per the s073e banner.

# Numbers

- Base: ρ=0.9041 (band A), `|L_c457|`=156.27, `L_c457` vs `L_truth` = 128.56°, 2T=0.6479558,
  `|L|²`=24420.56. Pipeline gate PASS (|Δρ| vs cached s069 = 0.0000).
- (a) `axis_1`: ρ at ±1°/±2°/±5° = 1.22,1.34 / 2.06,2.18 / 5.14,4.34.
- (a) `axis_2`: ρ at ±1°/±2°/±5° = 2.12,1.98 / 3.98,3.52 / 10.94,6.83.
- (a) `axis_3` (control): ρ at ±1°/±2°/±5° = 1.67,1.91 / 3.00,3.23 / 6.72,7.07.
- (b) ω-scale: ρ at ±0.5%/±1%/±2% = 4.00,4.18 / 7.64,7.66 / 13.75,13.44.
- 63 states rendered; compute wall 518.8 s on Pool(24).
- Source for all of the above: `results/s073f/summary.json`.

# Out of scope

- Hi-fi ρ-band validation of *other* seeds' competing basins (s079+ thread).
- A tighter LM polish on cluster_457 targeting polhode-distance directly (would sharpen the
  base point but does not change the local-stiffness conclusion).
- The cat-4 theorem half-page and [RF74] acquisition (s073e deferred items).

# Cross-references

- `experiments/s073_cluster457_l_vector_check.md` — the original |L|-match / L-direction-off
  measurement on cluster_457.
- `experiments/s073d_polhode_match_cluster457.md` — the ~1.1% Casimir mismatch and the
  strict/charitable ambiguity this experiment resolves.
- `experiments/s077_l_vector_basin_sweep.md` — the cohort-scale |L|-pinned / direction-free
  finding (104 competing basins).
- `experiments/s079_regime_stratified_l_basins.md` — the regime-stratified cohort cut that
  s073f clears the way for.
