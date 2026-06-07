---
title: "s077 — L-vector ↔ basin cohort sweep: |L| magnitude is pinned, L direction is free"
type: experiment
sources:
  - results/s011/runs.npz  (post-fix s011/s068 cohort pilot, 640 converged basins)
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073d_polhode_match_cluster457.md
  - experiments/s073e_robinson_frueh_2025_full_audit.md
related:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073d_polhode_match_cluster457.md
created: 2026-05-14
updated: 2026-05-14
confidence: high (pure re-score of cached post-fix endpoints; 104 competing basins across 9 seeds; every number traced to results/s077/summary.json)
---

# TL;DR

Re-scored all 640 converged basins from the post-fix s011/s068 cohort pilot (10
seeds × 64 multi-start ICs) for their angular-momentum vector. Among the 104
"competing" low-MSE basins (not truth, not twin, surrogate MSE < 0.5):

- **The magnitude of L is pinned; the direction of L is not.** Competing basins
  match truth's |L| to a **median 0.73%** (IQR 0.37–3.0%) but their inertial
  L_J2000 direction is off by a **median 90.8°** (IQR 70.7–129.5°, full range
  5.7–166.8°). The magnitude histogram is a sharp spike at 0; the direction
  histogram is broad and near-uniform.
- **This is not a continuous family.** The direction offsets are discrete clumps,
  and the structure is strongly seed-dependent — no evidence for a continuous
  2-parameter L-direction family generic to asymmetric bodies.
- **The "shared polhode" sub-claim is partially true and seed-dependent.** On
  seed 60 all 13 competing basins sit on **one** polhode; seeds 91/6/44/21/41 use
  1–2; but seed 10 spreads 42 competing basins across **24** distinct polhodes.
  Where basins do share a polhode, they share it *with each other* — and that
  shared polhode is often displaced from truth's (clear in seeds 44, 60).
- **Casimirs match at ~1.4%, not exactly** — competing d(2T)/2T median 1.42%,
  d|L|²/|L|² median 1.47%. This is the cohort-scale echo of s073d's seed-89
  finding (2T differed 1.12%): "nearby, not identical" polhodes is cohort-typical.

The user's intuition — "different solutions with equal |L| at different
directions makes a kind of symmetric sense" — is **confirmed at cohort scale**.
The escalation to "continuous family" is **not** supported.

# What

s073 measured one Band-A basin on seed 89 (`cluster_457`): |L| matched truth to
0.55%, L_J2000 direction differed 128.6°. s073e's audit left the question of
whether this is a continuous family open, conditional on a cohort sweep beyond
N=1. The user asked to drop the continuous-family hypothesis and instead run a
**general, no-assumptions inquiry into the empirical relationship between a
solution basin's L vector and the truth L vector** across the cohort.

# How

Pure re-score of cached endpoints — no optimisation, no propagation beyond a
single-time call into the closed-form `omega_jacobi` for the polhode
descriptors. Substrate: `results/s011/runs.npz`, 640 LM-converged basins from the
post-fix s011/s068 pilot (seeds 6, 10, 21, 28, 41, 44, 48, 60, 84, 91; 64
multi-start Sobol ICs each). Confirmed post-fix: file last written by commit
`7bf3352` (post-fix replication).

Per basin: `L_J2000 = R(q0_final).T @ I @ ω_final` (the s073/s066 formula); body
Casimirs `2T`, `|L|²` (computed directly, basis-independent); polhode regime,
`k²`, `T_pol` from `omega_jacobi`'s `info` dict. Metrics vs that seed's truth:
`|ΔL|/|L_truth|`, L_J2000 direction angle, magnitude-only `||L|−|L_truth||/|L_truth|`,
`Δ(2T)/2T`, `Δ|L|²/|L|²`. Inertia tensor is constant across m048 seeds
(`diag = [37985, 38306, 7749]` kg·m²).

Basin class: `truth` (truth_basin_strict flag) / `twin` (twin_basin_strict) /
`competing_low_mse` (neither, final_mse < 0.5 — the s011 "competing basin"
convention) / `high_mse`.

Script: `experiments/s077_l_vector_basin_sweep.py`.

# Result

**Cohort split:** 41 truth, 0 twin, 104 competing_low_mse, 495 high_mse (640
total). (No twin basins were flagged in the s011 pilot.)

**The headline asymmetry — competing low-MSE basins (n=104):**

| metric | median | IQR | range |
|---|---|---|---|
| `|L|` magnitude rel diff vs truth | **0.73%** | 0.37%–3.0% | 0.002%–82.4% |
| L_J2000 direction angle vs truth | **90.8°** | 70.7°–129.5° | 5.7°–166.8° |
| `|ΔL|/|L_truth|` (full vector)   | 1.42 | 1.11–1.82 | 0.10–2.46 |
| d(2T)/2T                          | 1.42% | 0.71%–5.4% | 0.01%–229% |
| d`|L|²`/`|L|²`                    | 1.47% | 0.75%–6.0% | 0.005%–233% |

`s077_competing_hist.png`: magnitude diff is a sharp spike at 0 (77/104 basins
within ~3%); direction angle is broad and roughly flat across 0–180°.
`s077_L_plane.png`: competing basins hug the left edge (small magnitude diff) and
spread vertically across all direction angles — including the lowest-MSE ones.

**Per-seed polhode structure** (competing basins only; "distinct polhodes" =
distinct `(2T, |L|²)` clusters at 1% tol):

| seed | n_comp | distinct polhodes | L-dir median | truth regime |
|---|---|---|---|---|
| 6   | 15 | 2  | 90.8° | A |
| 10  | 42 | 24 | 83.6° | A |
| 21  | 2  | 2  | 143.4° | B |
| 28  | 0  | —  | — | B |
| 41  | 3  | 2  | 48.8° | B |
| 44  | 18 | 2  | 91.7° | A |
| 48  | 3  | 3  | 92.6° | A |
| 60  | 13 | **1** | 136.5° | A |
| 84  | 5  | 3  | 116.8° | A |
| 91  | 3  | **1** | 5.8° | B |

`s077_casimir_per_seed.png`: seeds 60 and 91 collapse all their competing basins
onto a single polhode (the cleanest "shared-polhode, free L-direction" cases);
seed 10 shows the opposite — 42 basins smeared along the (2T, |L|²) diagonal. In
seeds 44 and 60 the shared competing-basin polhode is visibly displaced from
truth's polhode (green star off to the side).

# Why this matters

1. **The user's "equal |L|, different direction" intuition is the right frame,
   and it now has cohort support.** The light curve tightly constrains the
   *magnitude* of L (median 0.73% — essentially `|ω|·I` scale, consistent with
   the known "ω-magnitude is the well-constrained DOF" forward-model finding) and
   leaves the *inertial direction* of L weakly constrained (median ~91°). This is
   a real, cohort-wide magnitude/direction asymmetry — not a property of one
   cluster on one seed.

2. **The continuous-family escalation is not supported.** Direction offsets are
   discrete clumps, not a continuum, and the polhode structure ranges from "one
   shared polhode" (seed 60) to "24 distinct polhodes" (seed 10). There is no
   single generic structure across asymmetric-inertia seeds. The s073e cat-4
   publication condition (i) — "cohort sweep showing cat-4 holds beyond N=1" — is
   **not met** by this data in its strong form; what *is* met is the weaker
   statement.

3. **It reframes the s073 cross-anchor filter — and confirms it at cohort
   scale.** An `|L|`-magnitude match between two candidates is near-useless here
   (competing basins match truth's `|L|` to <1%). The discriminating quantity is
   the *direction* of L_J2000. Two candidates claiming to be the same physical
   trajectory must share L_J2000 exactly; this sweep shows the direction part of
   that test is sharp (median 91° separation between competing basins and truth).
   s073 asserted this from N=1; s077 confirms it across 104 basins / 9 seeds.

4. **s073d's "nearby, not identical polhode" was cohort-typical, not an
   anomaly.** The seed-89 cluster_457 result (2T off 1.12%) sits right at the
   cohort median (1.42%). The strict "exact same polhode" reading of the original
   s073 banner is refuted cohort-wide; the "competing basins live on polhodes
   within ~1–2% of each other" reading is what the data supports.

# What this does NOT establish

s077 is a **descriptive** result, not an **operational** one — the distinction
matters and is easy to lose. The 640 basins were found by an LM optimiser
minimising light-curve MSE; L was never an input, a constraint, or a search
target. s077 ran *after* the search and measured what L looked like at each
already-found basin.

Consider the hypothesis "given a truth LC, you can *find* other LCs by *searching*
for solutions with |L| = |L_truth| in different L directions." That hypothesis has
two halves:

- **Necessary half** — "do the alternative LC-fitting solutions happen to preserve
  |L_truth| while their direction varies?" s077 tested this and found: yes for
  magnitude (median 0.73% off), yes for direction-spread (median 90.8° off). This
  half is supported at cohort scale.
- **Sufficient / generative half** — "does *imposing* |L| = |L_truth| at some
  chosen direction *produce* a valid solution?" s077 says **nothing** about this.
  No L-based search was run. "Found solutions have property X" does not imply
  "imposing X finds solutions." The hit rate of an arbitrary L direction at fixed
  |L| is unknown — most directions could be empty. And L_J2000 is only 3 DOF of
  the 6-DOF (q0, ω) state, so even a perfect L match leaves a 3-DOF subspace
  unsearched.

Additional limits on even the necessary half: |L| is *approximately* preserved
(~0.73% median, tail to 82%), not exactly; the Casimirs differ ~1.4% median, so
the basins are on *nearby* not *identical* polhodes; and "competing" used the loose
surrogate `final_mse < 0.5` gate, not hi-fi ρ-band validation. The experiment that
would test the generative half is the queued s073f local-geometry test (perturb
along L-direction at fixed |L| / Casimirs, measure LC-residual sensitivity).
**Honest status: necessary-side supported, generative-side open.**

# Numbers

All from `results/s077/summary.json` unless noted.

- 640 basins, 10 seeds (6,10,21,28,41,44,48,60,84,91), 64 multi-start ICs/seed
  (source: `results/s011/runs.npz`, `results/s011/summary.json`).
- Class split: 41 truth / 0 twin / 104 competing_low_mse / 495 high_mse.
- Competing-basin `|L|` magnitude rel diff: median 0.733%, IQR [0.372%, 3.005%]
  (source: summary.json `cohort.competing_L_mag_rel_diff`).
- Competing-basin L_J2000 direction angle: median 90.83°, IQR [70.66°, 129.50°],
  range [5.73°, 166.78°] (source: summary.json `cohort.competing_L_dir_angle_deg`).
- Competing-basin d(2T)/2T: median 1.425%; d|L|²/|L|²: median 1.472%
  (source: summary.json `cohort.competing_d_twoT_rel`, `competing_d_L2_rel`).
- Per-seed distinct-polhode counts and seed 60 single-polhode finding: summary.json
  `per_seed.seed_060.competing_distinct_polhodes_1pct = 1` over n_comp = 13.
- Inertia diag = [37985.16, 38305.71, 7749.01] kg·m² (source: s077 console /
  `_build_model()`).
- Cross-check vs s073d: seed-89 cluster_457 2T differed 1.118%
  (source: `results/s073d/summary.json`) — within the s077 cohort IQR.

# Out of scope

- **Hi-fi ρ-band validation of the 104 competing basins.** "Competing" here means
  surrogate `final_mse < 0.5` — the s011 convention, a loose gate. These are not
  individually hi-fi ρ-band classified. The magnitude/direction asymmetry holds
  even for the lowest-MSE subset (visible in `s077_L_plane.png`), but per-basin
  ρ-bands would need a render pass — deferred.
- **Seed 89 itself.** Not in the s011 pilot's 10 seeds. The s073/s073d/s059k
  seed-89 data is consistent with the cohort (cluster_457's numbers sit at the
  cohort medians) but was not re-folded into the s077 NPZ.
- **Why the polhode structure is seed-dependent** (seed 60 → 1 polhode vs seed 10
  → 24). Likely tied to tumble regime / polhode diameter / LC information content,
  but not investigated here.
- **The analytical theorem** (s073e publication condition ii) — untouched.

# Cross-references

- `experiments/s073_cluster457_l_vector_check.md` — the original N=1 observation
  (|L| 0.55%, direction 128.6°); s077 generalises it.
- `experiments/s073d_polhode_match_cluster457.md` — "nearby not identical
  polhode" on seed 89; s077 shows ~1.4% Casimir mismatch is cohort-typical.
- `experiments/s073e_robinson_frueh_2025_full_audit.md` — cat-4 publication
  conditions; s077 addresses condition (i) and finds the strong form unsupported,
  the weak form supported.
- Memory: `feedback_dont_overclaim_from_one_data_point.md` — observed; claims
  here are cohort-scale (n=104) and phrased accordingly.
- Memory: `project_omega_mag_basin_scales_with_omega.md` — the |L|-magnitude-pinned
  result is the L-space restatement of "|ω| magnitude is the well-constrained DOF".
