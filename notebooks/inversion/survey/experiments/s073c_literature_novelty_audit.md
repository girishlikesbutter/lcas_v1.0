---
title: "s073c — literature novelty audit of the L_J2000-ambiguity framing + scope correction on s073 claims"
type: audit
sources:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073b_path2_cprofile.md
related:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073b_path2_cprofile.md
  - experiments/s072_path2_closed_form_q.md
created: 2026-05-14
updated: 2026-05-14
confidence: high (literature triangulated from three independent searches; scope correction stated explicitly)
---

# TL;DR

Audit session (no compute). Two outcomes. **(1) Novelty: the framing implied by s073 — "the LC's dependence on body-frame sun/observer leaves the inertial orientation of L partially unobservable on asymmetric bodies" — is not in the published man-made-object LC inversion literature.** The Frueh-group flip problem (Burton & Frueh AMOS 2023; Burton, Robinson & Frueh, *Advances in Space Research* 74:5619, 2024 §4; Burton 2024 Purdue PhD thesis §8.5) is the closest prior art, but it covers a single discrete 180° rotation about the û-ŝ bisector derived from BRDF (Helmholtz) reciprocity. Burton's PhD enumerates three ambiguity categories (flip / object-visual-symmetry / noise); the framing implied by s073 would be a fourth category. The thesis contains zero occurrences of "polhode." **(2) Scope correction on s073: what we have is one striking data point (cluster_457 on seed 89), not a class.** The continuous-family, body-frame-polhode-invariance, and operational L-conservation cross-anchor filter claims made in the s073/s073b wind-down and in this session's earlier framings are *hypotheses built on the one measurement*, not findings.

# What

Triggered by user pushback after a literature review session: "What exactly are you claiming we have, and do we know we have that yet?" The session had two threads — (a) a literature audit to assess whether the s073 framing is publishable, and (b) honest accounting of what s073 has actually established vs what was being asserted.

# How

Three independent searches over the man-made-object LC inversion literature, plus the asteroid LC inversion literature as the closest analog (Kaasalainen 2001 A&A 376:302 is the cleanest published statement of inertial-L ambiguity for NPA rotators, but in the multi-precession-cycle multi-geometry regime that doesn't match ground-based satellite arcs).

- Two general-purpose subagents (RSO + asteroid axes); each produced a structured report with citations.
- Targeted Tavily search and Tavily extract on specific papers.
- Direct PDF reads of: Burton, Robinson & Frueh 2024 ASR (full paper, user-provided), Burton 2024 Purdue PhD thesis (234 pp, user-provided; selective reading + pdftotext full-text grep).
- Tavily research (the deep multi-source synthesis) was rate-limited on this plan and was substituted with the subagent runs above.

Triage method on the thesis: ToC + abstract + Conclusions + Future Work (≈ 15 pp), then targeted reads of §5.2.4, §8.5 (flip problem), §9.3.1 Case 3 (uniform cuboid), §9.3.2 Case 4 intro, with `pdftotext`-backed grep over the full text for decisive terms (`polhode`, `manifold`, `continuous family`, `L-conservation`, `cross-anchor`, `multi-anchor`, `cone of`, `infinite number of`, `inertial unobservable`).

# Result

## Literature audit — three ambiguity categories in the published RSO LC inversion literature

| Cat | Mechanism | Transformation | Cardinality | Requires shape symmetry | Disambiguator proposed |
|---|---|---|---|---|---|
| **1: flip problem** | Helmholtz reciprocity (Hapke 2012; Minnaert 1941) | 180° about the û-ŝ bisector (PAB) | Discrete (one mirror per truth) | No | Multi-observer joint LC fit (Burton thesis Case 4) |
| **2: object visual symmetry** | Body shape invariance under a discrete rotation group | Discrete 180° rotations about body principal axes (only for cuboid-like) | Discrete (finite group) | Yes | "Follow-up observations or inherent symmetry analysis" (Burton thesis §9.3.1) |
| **3: noise / measurement uncertainty** | Measurement noise floor | n/a | Continuous (within noise envelope) | n/a | Better noise model; Mahalanobis cost terms (Burton thesis Ch. 11) |

The s073 framing implies a **fourth category**, not present in the literature:

| Cat | Mechanism | Transformation | Cardinality | Requires shape symmetry | Disambiguator candidate |
|---|---|---|---|---|---|
| **4 (hypothesised)** | Body-frame-only LC dependence + slowly-varying inertial û, ŝ | Continuous rotation of L_J2000 with body-frame polhode held fixed | Continuous (≈ 2-parameter manifold) | **No** | L-conservation across anchor epochs (s073 idea, untested) |

## Burton 2024 PhD thesis full-text scan (load-bearing for the novelty claim)

- 234 pages, scanned cover-to-cover via TOC + targeted PDF reads + full-text grep.
- **Zero occurrences of "polhode" in the entire thesis** (source: `grep -in polhode /tmp/burton_thesis.txt` returned no matches).
- Zero hits on `manifold`, `continuous family`, `family of solutions`, `L-conservation`, `cross-anchor`, `multi-anchor`, `cone of`, `infinite number`, `inertial unobservable`.
- Future Work (Ch. 11, pp. 216-217) lists open problems as: measurement noise modelling, BRDF/shape/inertia errors, joint shape-attitude estimation, multispectral and polarised LCs, radar/laser ranging augmentation. The dynamical observability of inertial L is not enumerated.
- §8.5 "The flip problem" (pp. 167-172) is structurally identical to BRF 2024 §4; constructs `qflip = (0, p̂)` and uses Helmholtz reciprocity. Discrete only.
- §9.3.1 Case 3 "Tumbling Uniform Cuboid" (pp. 176-179) introduces cat 2 explicitly: *"the difference between the true attitude time history and one rotated 180° about one of the principal axes is unobservable."* Tied to object visual symmetry.
- §9.3.2 Case 4 (Landsat 8, two observers, pp. 180-187) is Burton's only multi-observation disambiguator. Joint LC fit, not propagation-free.

## Scope correction on s073 — what we have vs what was being asserted

What s073 measured (verified, factual):
- On seed 89, for cluster_457: `|L|` matches truth to 0.55%; `L_J2000` direction is 128.6° off truth at t=0; cluster_457 has q0_err = 59.58°, ω_dir_err = 25.05°, ω_mag_err small; hi-fi full-LC ρ = 0.904 (within noise floor) (source: `notebooks/inversion/survey/results/s073/summary.json`).
- This is **one cluster, one anchor time (t=0), one seed (89), one body (m048).**

What was being asserted in the wind-down banner and in this session's earlier framings (overclaim audit):

| Asserted | Status |
|---|---|
| "L-conservation cross-anchor matching IS a real discriminator" | **Untested as an operational filter.** What we have: this one (truth, cluster_457) pair would be separated by an L_J2000-direction filter at one anchor. We have not tested the filter on a candidate pool, measured truth-survival rate, measured discrimination ratio against non-truth Band A∪B candidates, or evaluated at a second anchor time. |
| "Continuous 2-parameter manifold of L_J2000 directions producing similar LCs" | **Hypothesis from N=1.** We have one point. The manifold structure has not been mapped. |
| "Same body-frame polhode" for truth and cluster_457 | **Plausible but not directly verified.** `|L|` match and (presumably) 2T match make this likely, but we have not computed ω_body(t) for cluster_457 and confirmed it traces truth's polhode curve modulo phase. |
| "Cat 4 generic to asymmetric bodies" | **N=1 anecdote.** One cluster on one cohort trajectory. |
| "Cat 4 distinct from cat 3 (noise)" | **Plausible but argued, not proved.** Cluster_457 ρ=0.90 is *within* noise floor of ρ=1.0, so cat 3 is a candidate explanation we haven't structurally ruled out. The structural argument (that cluster_457 is a different attitude whose forward-rendered noiseless LC sits within noise of truth's noiseless LC, not a noisy realisation of truth) is correct in principle but not yet demonstrated for this case. |

What is solid (and what is not):
- **Solid:** the literature novelty claim (the framing isn't in the published RSO LC inversion literature; triangulated three ways).
- **Solid:** the theoretical observation that the LC depends only on body-frame sun/observer (this is stated in BRF 2024 §4 itself).
- **Solid:** the one s073 measurement is real and quantitatively striking.
- **Not solid:** any statement that uses plural ("multiple solutions"), continuous ("family", "manifold"), or operational ("filter works") language. Those are all extrapolations from N=1.

# Why this matters

This is a focus-defining outcome, not a finding. Three consequences:

1. **The publication path is plausible but not yet earned.** If the hypothesis converts to a finding, the framing is novel relative to the literature and naturally extends the Frueh-group flip-problem framework (cat 1 + 2 + 3 → +cat 4). The closest competing paper is Robinson & Frueh 2025 J. Astronaut. Sci. ("Global Light Curve Attitude Estimation with Noisy Measurements and Inertia Uncertainty"), DOI 10.1007/s40295-025-00557-9 — not read in full, still a residual threat.

2. **The next compute should be the conversion experiments, in order from cheapest-and-most-decisive to most-expensive.** See "Next" below. The cheapest is the polhode-match verification (a few minutes); if it fails, the entire framing breaks for cluster_457 and the program redirects.

3. **The methodology lesson is generalised in memory:** when an empirical observation is N=1, write claims in the singular and factual; reserve plural / continuous / operational language for after cohort or filter validation. See `feedback_dont_overclaim_from_one_data_point.md`.

# Numbers (literature pointers, with citations)

**Cat 1 — flip problem.**
- Burton, A., Robinson, L., Frueh, C. (2024). "Light curve attitude estimation using particle swarm optimizers." *Adv. Space Res.* 74:5619-5638. DOI 10.1016/j.asr.2024.09.008. **§4 "The flip problem"** (p. 5627). Quote (verbatim): *"This attitude time history is found by rotating the object 180° about the bisector of û and ŝ so that the two unit vectors are 'flipped.' Because torque-free attitude motion expressions in Section 2 only depend on the initial angular momentum vector, the two unit vectors remain 'flipped' at all times, resulting in an unchanged light curve."*
- Burton, A. (2024). "Attitude Estimation Using Light Curves." Purdue PhD thesis. §8.5 (pp. 167-172). Constructs `qflip = (0, p̂)` from the û-ŝ bisector p̂.
- Burton, A., Frueh, C. (2023). AMOS conference, "Fast Light Curve Inversion for Regular and Tumbling Attitude Motion." (Original empirical observation of the 180° flip; theoretical explanation in BRF 2024.)

**Cat 2 — object visual symmetry.**
- Burton 2024 thesis §9.3.1 Case 3 (Tumbling Uniform Cuboid), p. 179. Quote: *"the Rank 2 estimate results from the visual symmetries introduced by making the cuboid's surface properties uniform rather than symmetries in the BRDF. These visual symmetries mean that the difference between the true attitude time history and one rotated 180° about one of the principal axes is unobservable."*

**Cat 3 — noise.**
- Burton 2024 thesis Ch. 11 Future Work (pp. 216-217). Mahalanobis cost terms, PHD-filter refinement, multispectral / polarised LCs as candidate disambiguators.

**Closest asteroid-side analog (different regime, not directly applicable).**
- Kaasalainen, M. (2001). "Interpretation of lightcurves of precessing asteroids." *A&A* 376:302-309. Quote: *"If the solar phase angle α at the one observing geometry is nonzero, there are always two possible mirror solutions for L… If the one geometry is at opposition (or close to it), there is an infinite number of L-solutions, all on a cone around the line of sight."* The asteroid regime assumes multi-precession-cycle arcs at multiple observing geometries — fundamentally different from a 60-minute ground-based satellite arc at one geometry. This is the closest published statement to the s073 framing but does not transfer directly.

**s073 measurement (verified, source: `notebooks/inversion/survey/results/s073/summary.json`):**
- Seed 89 cluster_457 vs truth at t=0
- `|L|` magnitudes: 155.41 vs 156.27 kg m²/s — relative diff 0.55%
- `L_J2000` direction error: 128.6°
- |ΔL|/|L_truth| = 181%
- q0_err = 59.58°; ω_dir_err = 25.05°; hi-fi ρ = 0.904

# Artefacts

- This writeup: `notebooks/inversion/survey/experiments/s073c_literature_novelty_audit.md`
- s073 underlying data: `notebooks/inversion/survey/results/s073/summary.json`, `results/s073/l_vectors.npz`
- Source PDFs read (user-local, not committed):
  - `/home/girish/Documents/1-s2.0-S0273117724009281-main.pdf` (Burton, Robinson & Frueh 2024 ASR)
  - `/home/girish/Downloads/burton_thesis_revised_v2.pdf` (Burton 2024 Purdue PhD thesis)
- pdftotext extract used for grep: `/tmp/burton_thesis.txt` (ephemeral)
- New memory: `feedback_dont_overclaim_from_one_data_point.md`

# Next (priority order — these are the cat-4-hypothesis-conversion experiments)

1. **(cheap, ~10 min compute) Body-frame polhode match verification for truth vs cluster_457 on seed 89.** Compute ω_body(t) over the LC for both attitudes. Check that they trace the same polhode curve in body frame (same |L|, same 2T, same trajectory modulo phase). If yes, the "same polhode" half of the hypothesis is verified for this case. If no, cluster_457 is not a cat-4 instance and the framing breaks for this seed — go to a different multi-sol candidate, or redirect.
2. **(cheap, ~30 min compute) Continuous-family test.** Take cluster_457's (q0, ω0). Apply small rotations in inertial space (parameterised over the unit 2-sphere of L_J2000 directions). Forward-render LCs. Check whether residual scales smoothly with rotation angle (manifold) or is sharply localised (isolated point). Gives empirical evidence for/against the continuous-family claim.
3. **(moderate, hours) Cohort sweep.** Apply the same s073 analysis (|L| match, L_J2000 direction comparison) to Band A∪B multi-sol clusters from other cohort seeds in the s069/s070 result tree (seeds 10, 14, 28, 91, others). If the pattern repeats — body-frame invariants match, L_J2000 differs, no shape-symmetry explanation — it's a class. If cluster_457 is the only such case, it's an anecdote.
4. **(moderate) Test L-conservation as an operational filter on a candidate pool at two anchor times.** Pick a pool with cat-4 candidates surfaced by experiments 2-3. Compute L_J2000 at t=0 and at t=t_B (e.g., t_B = 30 min). Measure: (a) does truth survive the filter? (b) how many cat-4 candidates does it kill? (c) does it preserve other true degeneracies (cat 1 / cat 2)?
5. **(theory)** Prove the body-frame-only-LC-dependence ⇒ inertial-L-invariance statement under fixed inertial û, ŝ. Should be a few lines of algebra. Converts hypothesis to theorem.

The natural sequence is 1 → 2 → 3 → 4, with the option to stop and redirect at the first failure.

# Out of scope (intentional gaps in this audit)

- **Robinson & Frueh 2025 J. Astronaut. Sci. paper** (DOI 10.1007/s40295-025-00557-9). Title contains "Global" and "Inertia Uncertainty" — could extend the flip problem direction. Springer paywall; not read. This is the highest-remaining-threat single paper.
- **Forward citation crawl on BRF 2024 and Hinks-Linares-Crassidis 2013** — papers from 2024-2026 citing these. Likely 5-15 hits each. Highest-yield remaining lit search step.
- **Roberto Furfaro recent work** — third major LC-inversion person not in the Frueh/Linares groups. Untested.
- **Foreign-language literature** (Russian / Chinese / Japanese RSO photometry).
- **Kucharski et al. 2021** (Acta Astronautica 187:115, TOPEX/Poseidon full attitude reconstruction). Solves for inertial L on a single pass; should be checked for whether they discuss the regime where this would fail.
- **Theoretical proof of the body-frame-invariance statement** (item 5 above) — left to future work.

# Cross-references

- `experiments/s073_cluster457_l_vector_check.md` — the underlying empirical observation.
- `experiments/s073b_path2_cprofile.md` — the parallel optimisation gate result.
- `experiments/s072_path2_closed_form_q.md` — the closed-form q(t) infrastructure on which the cat-4 framing builds.
- `concepts/known_pathologies_to_revalidate.md` — broader list of buggy-era findings being revisited.
- Memory: `feedback_dont_overclaim_from_one_data_point.md` (new), `feedback_single_seed_misleads_cohort.md`, `feedback_oracle_injection_taints_yield.md` (existing related lessons).
