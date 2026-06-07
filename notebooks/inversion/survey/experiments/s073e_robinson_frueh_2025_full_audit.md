---
title: "s073e — Robinson & Frueh 2025 full-paper audit: implementation lessons + encroachment assessment on the cat-4 framing"
type: audit
sources:
  - /home/girish/Documents/s40295-025-00557-9.pdf  (Robinson & Frueh 2025, J. Astronaut. Sci. 73:7, DOI 10.1007/s40295-025-00557-9)
  - experiments/s073c_literature_novelty_audit.md
  - experiments/s073d_polhode_match_cluster457.md
related:
  - experiments/s073_cluster457_l_vector_check.md
  - experiments/s073b_path2_cprofile.md
  - experiments/s073c_literature_novelty_audit.md
  - experiments/s073d_polhode_match_cluster457.md
created: 2026-05-14
updated: 2026-05-14
confidence: high (entire 37-page paper read; quotes pulled from the PDF with page numbers; encroachment analysis triangulated against s073/s073c/s073d data we already have)
---

# TL;DR

The full read of Robinson & Frueh 2025 (RF25 hereafter) — the residual-threat paper flagged at the end of s073c — settles two questions. **(1) Implementation lessons:** the single highest-value adoption is RF25's negative-log-likelihood loss with per-timestep $\sigma_k$ weighting plus $\|S\|/\|\hat S\|$ rescaling (their Eq. 3); it directly attacks the bright-peak overfitting we have seen empirically (s058/s059 local-window phantom basins). Second tier: shadow cache (50 MB body-frame lookup), $\alpha$-overfit gate, inertia-ratio + body-frame-$\omega$ visualisation. **(2) Encroachment:** RF25 does **not** pre-empt the cat-4 framing. They observe continuous 1D $\omega$-bands on the axisymmetric rocket body (Case 1a, four cylindrical sections about the body symmetry axis) and explicitly attribute them to cat-2 visual/dynamical symmetry. On the asymmetric box-wing (Cases 2a/2b) they find 90 and 112 widely-distributed solutions but **never analyse them through the body-frame Casimir lens** — zero occurrences of "polhode" in the paper, no plot of $L_{J2000}$ direction across solutions, no $(|L|, 2T)$ structure analysis. The publication path remains open conditional on (i) cohort sweep showing cat-4 holds beyond N=1, and (ii) writing the half-page analytical extension of Burton-Robinson-Frueh 2024 §4's flip argument to continuous rotations of $L_{J2000}$. Residual risk: [RF74] (Robinson & Frueh ECSD 2025 conference companion paper) is not in our possession — should be obtained before any cat-4 writeup.

Full audit artefact: `experiments/s073e_robinson_frueh_2025_full_audit.{tex,pdf}` (11 pages, two-part structure: implementation + encroachment).

# What

s073c flagged RF25 as the "remaining unread highest-threat paper" and recommended obtaining it before further cat-4 work. The user downloaded the PDF (37 pages, ~6 MB) on 2026-05-14 and asked for a deep two-part audit:

1. Which RF25 implementation pieces could materially improve our pipeline?
2. Does their work encroach on the s073 hypothesis — that the body-frame Casimirs $(|L|, 2T)$ plus polhode phase parameterise a continuous inertial-$L_{J2000}$ ambiguity generic to asymmetric-inertia bodies?

No new compute this session — pure reading, structured comparison, and writing.

# How

Sequential cover-to-cover read of the PDF (10 pages at a time via Read tool with pages= parameter), plus a `pdftotext -layout` extract for grep-based term scans:

- `grep -in -E "polhode|manifold|continuous family|family of|L_J2000|infinite number|cone of|inertial unobserv|mediatrix|cylindrical|symmetric|sphere|tube"` returned zero hits on "polhode", confirming the gap.
- `grep -in -E "energy|kinetic|momentum|magnitude|2T|conservation|conserved"` mapped every place RF25 references body-frame invariants — none of which are used as solution-space structure analyses.

Decisions catalogued by category: \textsc{adopt} (drop in), \textsc{adapt} (idea, our regime differs), \textsc{evaluate} (worth a small experiment), \textsc{pass} (their constraint, not ours).

# Result

## Part 1 — Implementation lessons (12 items, ranked by code-change leverage)

| # | RF25 element | Decision | Notes |
|---|---|---|---|
| 1 | NLL loss with $\sigma_k$ weighting + $\|S\|/\|\hat S\|$ rescaling (Eq. 3) | **adopt, high value** | $\sim$30-line change in `lib/forward.py` residual + LM cost. Directly addresses bright-peak overfitting empirically seen in s058/s059. |
| 2 | Full sensor + sky noise model (Eqs. 35–40) | **evaluate** | Implement in `lib/noise_rf.py` behind a flag; only adopt for real-data thread, do not invalidate m048 cohort. |
| 3 | Shadow cache: 4D body-frame lookup at $200\times200\times100\times100$ = 50 MB (Eqs. 25–29) | **evaluate** | High-leverage post-elliprj when forward-model wall = shadow time. Articulated panels need per-orientation caches. |
| 4 | Semi-analytic Sutherland-Hodgman shadowing (Eqs. 18–24) | **pass** | Our trimesh batched ray cast is fine; cache (item 3) likely subsumes this speedup. |
| 5 | MRP attitude state + inertia ratios $(J_y/J_x, J_z/J_x)$ in search (Eq. 4) | **pass + evaluate** | Pass on MRP. Evaluate inertia-ratio search for real-data work; not needed for m048. |
| 6 | $\alpha$-threshold overfit gate ($f_i < f_\text{best} \Rightarrow$ discard) (§2.5.1) | **adapt** | Define $\rho_\text{best}$ analogously; discard $\rho_i < \rho_\text{best}$ as overfits. Would have caught the s058 oracle-injection earlier. |
| 7 | Uniform-$SO(3)$ + uniform-ball $\omega$ sampling ($r^{1/3}$, Eq. 8) | **evaluate** | A/B-test vs our Fibonacci-direction × magnitude grid on 10-seed cohort slice. |
| 8 | BFGS from $10^5$ parallel starts in Taichi (§2.4–2.5) | **pass + evaluate** | Pass on BFGS itself. Evaluate Taichi GPU port if forward model becomes rate-limiter. |
| 9 | Inertia-ratio histogram visualisation (Fig. 8b, 13b) | **adopt** | Add to s074+ diagnostic kit. Cat-4 family at fixed $(|L|, 2T)$ should appear as a curve in $(J_y, J_z)$. |
| 10 | Body-frame $\omega$ 3D scatter (Fig. 7b) and (azimuth, elevation) histogram (Fig. 8d) | **adopt** | Standardise the plot; canonical view for polhode-family analysis. |
| 11 | "Don't average solutions — thin clusters instead" (§4.3) | **adopt as policy** | Add to `feedback_multi_solution.md` memory. |
| 12 | $f_\text{best}$ from truth in **assumed** (mismatched) model, not truth-model | **adopt** | Right denominator for "how good can any solution legally be" under model mismatch. |

## Part 2 — Encroachment assessment

s073 framing decomposed into three sub-claims, each judged separately against RF25:

| Claim | RF25 says | Encroachment | Risk |
|---|---|---|---|
| A. LC depends on body-frame geometry only | Acknowledged starting point (BRF 2024 §4 is cited as source) — not re-stated in RF25 | **Not novel.** Use as setup, not finding. | Low |
| B. Continuous 2-parameter $L_{J2000}$ family on asymmetric bodies | Demonstrate continuous 1D $\omega$-bands on **axisymmetric** body (Fig. 7b — "four cylindrical sections symmetric about the body's axis of symmetry"). On asymmetric box-wing (Figs. 13d/14d) solutions are widely distributed but not analysed as a continuous family. | **Partially open** — this is the gap our framing fills if it converts. | n/a |
| C. Generic to asymmetric-inertia bodies | Box-wing cases have 90 / 112 converged solutions; characterised by orientation/$\omega$ histograms only, never by $(|L|, 2T)$ or $L_{J2000}$ direction. | **Open.** Their data may contain cat-4 instances they did not surface. | Medium — we should build our case on our own m048 data, not theirs. |

**Most dangerous overlap (and the answer to a likely referee question):** RF25 Case 1a's continuous spin-rate ambiguity for the axisymmetric rocket body **is** structurally "LC depends on $(|L|, 2T)$ only, modulo spin phase about the symmetry axis." A careless referee will say "isn't your cat-4 just RF25 Case 1a generalised?" The answer is: RF25 Case 1a's ambiguity requires **both** axisymmetric inertia **and** body visual rotational symmetry (which their rocket model has — it's a cylinder). Cat-4 is the strictly-stronger claim that the **inertial rotation DOF of $L_{J2000}$** exists for bodies with **neither** symmetry. This distinction must be front and centre in any cat-4 writeup.

**Gaps in RF25 that cat-4 framing could fill:**

- No use of body-frame Casimir invariants. Zero occurrences of "polhode" in 37 pages.
- No computation or plotting of $L_{J2000}$ direction across solutions. They have $10^5$ solutions and never inspect this.
- No analytical proof of body-frame-only LC dependence extended to continuous $L$-direction ambiguity (BRF24 §4 proves it for the discrete 180° flip; the continuous version follows by the same argument applied to any inertial rotation preserving $(\hat u, \hat s)$ — but neither paper writes it down).
- No multi-anchor / two-epoch architecture.
- No connection to Kaasalainen 2001 asteroid LC inversion (the only published continuous-$L$-family discussion in any RSO-adjacent literature).

**Verdict:** publication path open, conditional on:

1. **Empirical** — cohort sweep showing cat-4 holds beyond N=1 (s073c §Next #3, deferred until s073d's polish-residual ambiguity is resolved by s073f / continuous-family local test).
2. **Theoretical** — half-page analytical extension of BRF24 §4 to continuous $L_{J2000}$ rotations.

# Why this matters

This was the decisive lit check the s073c audit flagged. Two outcomes that shape next-session work:

1. **The optimisation thread (Path 2, s074) is independent of this audit.** Nothing in RF25 changes the elliprj implementation plan. The NLL loss item (Part 1 #1) is a separate, cheap, valuable side-improvement worth slotting in alongside or after s074.

2. **The cat-4 thread has a clear next-step contract.** The two conditions above (cohort sweep + analytical extension) are concrete and bounded. The empirical part needs the s073d polish-residual ambiguity resolved first (s073c §Next #2 → s073f).

# Numbers

All numbers are from the RF25 PDF at `/home/girish/Documents/s40295-025-00557-9.pdf` unless otherwise noted. Page numbers refer to the journal's print page (top of each PDF page), e.g. "p. 5627" = journal page 5627.

**RF25 paper itself:**
- 37 pages, J. Astronaut. Sci. 73:7 (2026), DOI 10.1007/s40295-025-00557-9 (source: PDF front matter).
- Two test objects: ATLAS V rocket body 1987-084D (axisymmetric inertia, Cases 1a/1b) and HYLAS-4 box-wing 2014-004A (asymmetric inertia, Cases 2a/2b) (source: RF25 Table 5).
- $10^5$ uniform initial conditions per case, BFGS in Taichi, $\leq 10$ min per case on Apple M1 CPU (source: RF25 §2.5, Table 7).

**RF25 Case 1a (axisymmetric, low inertia uncertainty):**
- Converged solutions: 2357 (2.357% of $10^5$) (source: RF25 Table 7).
- Median spin-rate error: −95.60% — essentially undetermined (source: RF25 Table 7).
- Median $\|\omega(0)\|$ error: 21.34%; median precession-rate error: 0.45% (source: RF25 Table 7).
- Solution geometry: two disjoint MRP tubes (cat-1 flip about mediatrix plane); four cylindrical $\omega$-bands about body symmetry axis (cat-2 axisymmetric, source: RF25 §4.1.1 p. 23–25, Fig. 7).

**RF25 Case 2a (asymmetric box-wing, low inertia uncertainty):**
- Converged solutions: 90 (0.090%) (source: RF25 Table 10).
- Median $\|\omega(0)\|$ error: 0.78% (source: RF25 Table 10).
- Solution geometry: widely-distributed orientation errors with no clear clustering (source: RF25 §4.2.1, Fig. 13c/d).
- Authors' interpretation: "removing the spin rate ambiguity seen in the rocket body cases" (source: RF25 p. 27).

**RF25 Case 2b (asymmetric box-wing, high inertia uncertainty $\sigma_J=1.0$):**
- Converged solutions: 112 (0.005% — Table 11 says 0.005% but $112/10^5 = 0.112\%$; assume typo) (source: RF25 Table 11).
- Median $\|\omega(0)\|$ error: 2.36% (source: RF25 Table 11).

**Grep results on `/tmp/robinson_frueh_2025.txt`:**
- `grep -in polhode /tmp/robinson_frueh_2025.txt` → 0 matches.
- `grep -in -E "manifold|continuous family|cone of|infinite number|L_J2000"` → 0 matches.
- `grep -in -E "kinetic|2T|conserved"` → only BRDF energy-conservation context, no body-frame Casimir invariant analysis.

**Our s073 numbers (already cited in s073/s073c/s073d, repeated here for the encroachment table):**
- $|L|$ match truth vs cluster_457: 0.55% (source: `notebooks/inversion/survey/results/s073/summary.json`).
- $L_{J2000}$ direction error: 128.6° (source: same).
- s073d follow-up: $2T$ differs by 1.118%, $|L|^2$ differs by 1.112% — same scale, not the clean "same polhode" assumed by s073 banner (source: `notebooks/inversion/survey/results/s073d/summary.json`).

# Artefacts

- `notebooks/inversion/survey/experiments/s073e_robinson_frueh_2025_full_audit.tex` (40 KB LaTeX source).
- `notebooks/inversion/survey/experiments/s073e_robinson_frueh_2025_full_audit.pdf` (315 KB, 11 pages, compiled clean with `pdflatex`).
- New memory: `project_robinson_frueh_2025_no_encroachment.md` (one-line pointer to this audit + headline verdict).
- Source PDFs (user-local, not committed):
  - `/home/girish/Documents/s40295-025-00557-9.pdf` (RF25, 6.2 MB).
- Ephemeral grep extract: `/tmp/robinson_frueh_2025.txt`.

# Out of scope

- **The [RF74] Robinson & Frueh ECSD 2025 conference paper** ("Optimal light curve attitude inversion with measurement noise: two case studies", cited in RF25's bibliography). Not in our possession. Title overlap suggests it may revisit the same case studies with different emphasis. **Action item:** check ECSD 2025 proceedings before any cat-4 writeup.
- **Forward citation graph on BRF 2024** — papers from 2024–2026 citing the flip-problem paper. Not yet swept.
- **Roberto Furfaro's recent LC inversion work** — third major group besides Frueh/Linares; not in this audit.
- **Theoretical proof of the body-frame-only LC dependence extended to continuous $L_{J2000}$ rotation** (Part 2 condition ii). A half-page derivation, left to a follow-up.
- **Re-analysing RF25's published solution data** for cat-4 structure. Tempting but risky: any priority argument would still belong to them on their dataset. Build the case on our own m048 results.

# Cross-references

- `experiments/s073_cluster457_l_vector_check.md` — the original N=1 observation.
- `experiments/s073b_path2_cprofile.md` — the orthogonal optimisation gate (independent of encroachment).
- `experiments/s073c_literature_novelty_audit.md` — flagged RF25 as the residual threat; this audit answers it.
- `experiments/s073d_polhode_match_cluster457.md` — first cat-4-conversion check, returned AMBIGUOUS; informs the next-step contract.
- Memory: `feedback_dont_overclaim_from_one_data_point.md` (created by s073c, reinforced by s073d, still applicable).
- New memory: `project_robinson_frueh_2025_no_encroachment.md` (this session).
