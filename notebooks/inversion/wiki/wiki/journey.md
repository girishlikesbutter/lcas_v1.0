---
title: "The Research Journey — m001 to m129"
type: narrative
related: ["[[index]]", "[[log]]", "[[surrogate-de-search]]", "[[gradient-based-inversion]]", "[[omega-sign-degeneracy]]", "[[basin-of-attraction]]", "[[dark-mag-saturation]]", "[[surrogate-model]]", "[[twin-degeneracy]]"]
created: 2026-04-16
updated: 2026-04-16
confidence: reference
---

# The Research Journey — m001 to m129

> Chronological narrative of the inversion project from its first peak-graph prototype through the surrogate-DE breakthrough and the data-integrity reckoning. Written to re-frame the arc so that dead ends are legible as conditional dead ends — ideas that failed under one compute budget or one cost function may revive under another.
>
> Read this file when a new idea feels familiar but you can't remember whether it was tried. It almost certainly was, under a different name.

---

## The problem

Recover the attitude initial condition `q0` and body-frame angular velocity `ω` of a tumbling satellite (Intelsat 901) from a single ground-based light curve. Six free parameters against ~500 noisy brightness samples. The forward model is not the bottleneck — SPICE + trimesh ray tracing + Ashikhmin-Shirley BRDF at ~1 Hz is solved. The inverse is the entire project.

Every era of this research is a different guess at how to make the cost landscape tractable. They fail in characteristic ways, and the failures rhyme.

---

## Era 1 — Peak-graph and bridge methods (m001 → m045). Refuted.

**Hypothesis:** brightness peaks are sparse physical anchors. If we can label a handful of peaks with candidate attitudes, a graph of bridge solutions between them — filtered by angular-momentum conservation across legs — should recover the trajectory.

**m001–m003** established the problem: at any single epoch, hundreds of non-truth attitudes reproduce observed brightness within 1%. Brightness is catastrophically many-to-one ([[m001]] in DEAD_ENDS). Peak derivative constraints (m002) help marginally; the omega bridge BVP solver (m003) works within arrival tolerance and becomes a reusable building block.

**m004–m018** tried to turn peaks into a graph. Sample attitudes at peaks, bridge between them, score by residual at intermediate epochs, pick shortest path. Every variant — joint q0-ω optimisation (m008), dense sampling (m010b), interpolated peaks (m012), peak-connectivity graph (m013) — ranked truth at roughly the 10th percentile. Intermediate brightness residual is a signal-free coordinate: plausible-looking omegas produce plausible-looking intermediate brightness.

**m016 / m016b / m016c** — the dip/trough constraint trilogy — sought sharper discrimination in the derivative structure between peaks. m016c ("trough-constrained bridge") was never fully tested at scale because bridge solves cost ~11 s each. Flagged as revivable on the surrogate.

**m017 sequential staircase** tried to enumerate winding numbers along each bridge leg. It failed on leg 1 when triaxial polhode dynamics shifted the rotation axis with |ω|, jumping past the true omega. The structure was right; the propagator assumption was wrong.

**m019 – m025** rescued the winding-enumeration sub-thread. Band-sweep with multi-start random directions replaced the staircase (m019/m020), reliably finding all winding families. And m023–m025 validated the L-conservation filter **perfectly**: with true solutions present in the candidate pool, angular-momentum matching across legs separates the correct pair by nine orders of magnitude. This is a clean, reusable result that the later eras have not yet exploited.

**m026 – m030, m040** — integration into an end-to-end pipeline. The blocker turned out to be bridge coverage on long legs (10 random starts per 0.5°/s band was insufficient at dt≈721 s), not the filter logic. The magnitude-only dedup of m026 discarded correct directions; direction-aware dedup (m026b) found leg 0 but missed leg 1. m028 tried PA-mode dynamics as a bridge accelerator (763× faster); 0/30 correct because IS-901's triaxial asymmetry (0.556) is too far from axisymmetric. m029's peak-shape filter was too weak (11% kill rate). m030 hi-fi pruning was seed-count-limited.

**Verdict on Era 1.** The peak-graph architecture is refuted as an end-to-end pipeline, but three of its components survive: the band-sweep winding enumerator (m019/m020), the omega-bridge BVP solver (m003), and the L-conservation filter (m023–m025). Each is currently idle. None has been re-tried on top of the surrogate.

---

## Era 2 — Glint physics and the NLP pivot (m034 → m069).

**Hypothesis:** peaks brighter than magnitude 6 are specular glints — the phase-angle bisector PAB aligns with a body-frame facet normal. That is a hard geometric constraint, not a soft residual. Ride the constraint.

**m034 – m045** characterised the glint regime. m034 confirmed brightness peaks coincide with PAB-normal alignment. m041's gradient-boosted classifier put specular purity at 100% below mag 6 with F1 = 0.90. The specular threshold became load-bearing for every downstream pipeline. m044 visualised normals + inertial PAB on a sphere (glints = pass-throughs). m045 showed that a glint-alignment geometric filter has a clear basin for attitude but noisy direction.

Two attempts to seed L-BFGS-B with PAB-circle starts failed hard. m038 (30 k seeds across 14 normal families): 10× worse than random SO(3). m043 (oracle normal, focused seeding): still 18× worse. Unconstrained optimisation always walks off the circle — the manifold constraint has to be **imposed**, not suggested. **m042b** solved this correctly: coarse phi sweep (brute enumeration on the circle) for a correct-normal rank of #1, then Nelder-Mead on (phi, ω). With oracle omega this hit 1.77° q0 / 1.15° ω-dir. The two-phase brute-then-refine pattern became the template for everything that followed.

**m046 generated the trajectory dataset** — 100 seeds, all on a single 10:00–11:00 UTC window. **m048 generated the intended dataset** — 100 seeds with per-seed random start times in [08:35, 15:35 UTC]. Every pipeline from m050 onwards reads m046; m048 was never used. This is the architectural half of the data-integrity bug that would only be discovered in 2026-04-16.

**m049 – m059** was the long attempt at blind omega recovery. The working results were **m051b** — given known omega, hi-fi shadow physics recovers attitude 9/10 trajectories to ±180° (twin residual) — and **m057** — with the body-frame/inertial omega bug fixed, lo-fi LC residual ranks truth at #1 on trajectory 19. **m059 closed the blind-omega thread with nine explicit negatives**: stratified winding, anchor-centred scoring, phi-sweep-improved q1, alignment filtering, grid search, specular+anti-glint, focused bridging, arrival-error filtering, and brightness-profile shape scoring. All fail for the same chicken-and-egg reason — the bridge produces correct omega in its pool, but without a correct q1 the LC score buries it.

**m060 – m069** pivoted to the CasADi / multiple-shooting NLP formulation (the series-11 formulation). m060 tested whether an L-inertial parameterisation widens basins; it does not. **m061 is one of the load-bearing findings of the project**: a 180 s observation window has roughly a 10° omega basin, five times wider than at 3600 s. m062 showed that progressive window extension 180→3600 s converges 2/10 cold-start from joint ±10° boxes. m063b threaded grid search into the same architecture.

**m069b** distilled the geometric-refinement cost — specular alignment (±X, weight 10) plus bright-alignment at 19 peak epochs — and produced q0 19.1°→1.94°, ω 1.5°→0.14° in 50 s with 1050 evaluations. Anti-glint constraints were found to be **invalid** (shadows can suppress glints under perfect alignment, so absence of brightness is not absence of alignment). Only the positive constraint is safe. This lesson recurs — in m094's BRDF cost, m112's anchor-selection critique, and implicitly in every later alignment residual.

**Verdict on Era 2.** Forward-only phi-sweep+NM with a known omega works. Blind omega recovery from the bridge+LC architecture is refuted. The CasADi pivot introduced the short-window-basin insight that is arguably the most generalisable physics claim the project has produced, and is still largely unexploited.

---

## Era 3 — Cost-function engineering and NM expansion (m070 → m103). Plateau.

**Hypothesis:** the m069b/m070 pipeline can be generalised from seed 93 to a 10-seed cohort by smarter cost functions and wider candidate pools.

**m070** ran the first full four-stage pipeline (grid → NM → phi-sweep → hi-fi) end-to-end on seed 93 (1.94° q0, 0.14° ω, 7.8 min). **m073** extended to six seeds: 4/6 ω OK, 1/6 full OK, and for the first time three seeds resolved to their 180° twin. **m087/m088** studied speed knobs — SLERP attitude interpolation broke the cost landscape (4/10 regressed); relaxed ODE tolerance was safe (3×).

**m090** became the canonical baseline: 4 OK + 3 PARTIAL + 3 FAIL. Seeds 12/27/33 fail via scipy NM sensitivity (the same optimiser succeeds starting one grid point over); seeds 6/24/36 converge to a non-+X 180° rotation (wrong-axis twins). **m091** proved the twin is strictly ±X: every other rotation axis breaks LC identity. The wrong-axis convergence of m092 was not a twin but an **attitude-basin** pathology, which is what motivated the BRDF cost investigation.

**m093 – m099** were the cost-function era. m093 showed expected-dot cost does not help and diagnosed why: the Ashikhmin-Shirley BRDF encodes three geometric constraints per epoch (n·k1, n·k2, n·h) but alignment-cost uses only one. m094's adaptive BRDF+alignment cost was too selective — it rejects the truth when noise corrupts a glint. **m095 found the right lever: widen the NM pool (NM_TOP=200) and accept multi-window hi-fi rescoring.** Lo-fi re-ranking is **harmful** (it prematurely discards truth).

m096 ran a 100-seed constraint census and produced a blunt number: **87/100 seeds lack bright ±X constraints**. The entire glint-anchored pipeline was being tuned on the atypical 13%. m097 refuted lo-fi MSE as a grid-level replacement for alignment cost. m098/m099 found the sweet spot (NM_TOP=300, 2000-dir grid) and rescued seed 6.

m100/m101 introduced multi-phi selection (fixes 14/24) and discovered that full-window MSE beats multi-window voting. **m102 became the pre-surrogate best** — NM_TOP=300 + full-window MSE — at 3 OK + 3 PARTIAL + 4 FAIL. m103 rescued seed 14 via targeted multi-phi but left seed 27 at a 165° twin.

**Verdict on Era 3.** The cost-function branch hit a plateau. Alignment cost could not be engineered to both rank truth reliably and not regress edge cases. The bottleneck migrated — from grid density to NM pool size to phi parameterisation to anchor selection — but a rank-1 truth recovery across the cohort never materialised. The ground being prepared here — wider pools, honest ranking, the multi-solution philosophy — is what the surrogate would later cash in.

---

## Era 4 — IPL surface and the pab-contour limit (m104 → m112). Refuted.

**Hypothesis:** the shadow-less brightness surface over (k1, k2) — interpolated from a precomputed brightness table — can replace the expensive hi-fi BRDF evaluation in the alignment cost, giving a cheap grid-level discriminator.

**m104** proposed extracting omega from peak-crossing geometry — refuted on two grounds: the underlying kinematic identity is `dp_B/dt = Ω_L × p_B` not `ω_body × p_B` (you cannot extract ω without knowing attitude), and peak FWHM is too noisy at 7.2 s sampling (27% error). DEAD_ENDS logs this one clearly; do not revisit.

**m106 – m108** validated the constraint-satisfaction variant with oracle omega (14/14 peaks align) and then watched it fail at grid precision. A 1.16° ω-direction error accumulates 60–90° of attitude drift over the 3600 s window; late peaks become unmatchable. The method is mathematically correct and practically unusable at any density the grid search can afford.

**m109** tested whether any IPL-centroid metric has phi discrimination. None do — the zero-phase brightness surface is too symmetric, and **only shadows break the lobe symmetry**. This is the deep reason IPL cost fails: pab-contour assumes k1 = k2 (zero phase), and real PABs sit a median 25° off the IPL centroid. m118 later made this diagnosis explicit through kernel-factored decomposition: no IPL variant beats facet-normal alignment, and truth never ranks first.

**m110 – m112** tried sparse hi-fi phi (works only on seed 27) and post-NM best-anchor selection (fails because 3–5° ω error at NM exit corrupts distant-epoch PAB predictions by 20–150°). **m111 was the key diagnostic**: the anchor-alignment error — not shadows — is the bottleneck. cos²⁵⁰ at the brightest peak has zero phi sensitivity at the alignment maximum but amplifies a 2° misalignment by 100–1200×.

**Verdict on Era 4.** The IPL surface is not a drop-in cost replacement. The branch closes cleanly — pab-contour as an approximation is the blocker. The kernel-factored decomposition infrastructure from m118 survives as reusable diagnostic plumbing.

---

## Era 5 — Surrogate, DE, and the isoshell (m113 → m121). Breakthrough.

**Hypothesis:** if the forward cost is cheap enough, phi parameterisation becomes unnecessary — a 3-DOF DE over q0 can do what the phi sweep was approximating.

**m113** ran the ceiling test: at truth ω, a 3-DOF DE attitude search recovers q0 to 0.63°; phi sweep on the same ω gives 165°. Phi parameterisation, not the cost function, was the ceiling of Era 3. The trade is that DE costs 0.14 s per evaluation — ODE-bound.

**m114** answered the compute question with a learned surrogate (MLP at `~/surrogate_model`, ~50 000× speedup). Three-DOF multi-start DE is now cheap. Six-DOF cold start remains infeasible (the search space is too large even at zero eval cost), so the omega-first / attitude-second architecture is preserved.

**m115 is the Era-5 breakthrough.** 10/10 baseline seeds now have hi-fi MSE < 1.0 valid solutions. The m102 regression seeds 14 and 24 are OK. Seed 33 resolves at a 98.5° q0 that is not any standard ±X / ±Y / ±Z twin — a clue that the wiki would later crystallise as [[omega-sign-degeneracy]]. Five minutes per seed.

**m117** retired the "skip geometric refinement" lever: NM-only top-26 misses truth by 48.8°, L-BFGS-B geometric refinement is load-bearing. **m118** retired the IPL-variant cost-landscape search. **m119v2** (after the first m119 was retracted for a 6-hour/1-hour window mixup — a preview of the bigger bug to come) re-ran the attitude-isoshell POC on correct geometry; truth ranked 0/60 000 under all 13 variants.

**m120** is the validation plot twist: the tumbling-competitor test, multi-seed, ATT_FAIL cohort included. Truth ranks 0/10 004 under **every** variant for **every** seed. The `close_omega` bucket saturates at dark-mag ceiling ~10. The surrogate-attitude-isoshell branch is upgraded to #validated (multi-seed).

**m121** characterised the basins. q0 basin ≳ 5° (widest). ω-direction < 0.1° (saturated by 0.5°). ω-magnitude < 0.25% (narrowest — the m061-era hypothesis that |ω| was the wide direction is **refuted**). Strong 15–25× anisotropy in ω-direction, seed-specific preferred axis. The mechanism: ω errors compound over time (a 0.1° direction error drifts the attitude by 7.8° over an hour); q0 errors stay local. The time-integral is why ω is vastly tighter.

**Verdict on Era 5.** The surrogate resolved the Era-3 plateau by inverting the compute trade-off. m115's 10/10 claim is the strongest forward-progress moment in the project. The basin geometry from m121 is now the floor against which every later cost surface has to be checked.

---

## Era 6 — Wrapped pipeline and the flipped-ω thread (m122 → m129).

**Hypothesis:** DE finds wide basins; L-BFGS-B polishes them; hi-fi validates. Wrapping the three stages with a `keep_better(before, after)` guard promotes the surrogate-DE recipe to a production pipeline.

**m122 – m126** ran this wrap-up. m122 measured the Hessian at truth (tighter than m121, as expected for quadratic-local vs. saturation-dominated measurements) and found cohort universality within 2.11×. m123 revealed the hybrid architecture: DE enumerates q0 attractors, L-BFGS polishes |ω|-magnitude 5–10× (q0 and ω-direction are already locked). m124 refuted the raw polish (2/12 within ±30% hi-fi ratio; seed 27 catastrophic 12–16×). m125's inline `keep_better` wrapper rescued it (9/12 basins helped). m126 promoted the wrapper to default: **6/11 improved, 5/11 break-even, 0/11 regressed** on the full baseline, plus the seed-33 flipped-ω discovery (ω-dir 161.5° retrograde + q0 98.5° compensating, hi-fi 0.082) that became the [[omega-sign-degeneracy]] concept.

**Then the data-integrity bug broke the narrative.** m122/m123/m124/m125/m126 all called `setup_experiment(...)` without `end_time_utc='2020-02-05T11:00:00'`, silently using the 6-hour default window instead of m046's 1-hour truth window. Every hi-fi MSE number from those five scripts is on the wrong observed LC. Seed 0's claimed 0.112 regenerates to 0.325 (2.9× worse) on the correct window. The #validated status of the wrapped pipeline is retracted pending re-scoring. See [[DATA_INTEGRITY_BUG]] for the full audit.

What survives, because it does not depend on the hi-fi MSE number: q0 / ω-direction / ω-magnitude errors (window-independent), DE basin discovery, the Hessian/basin-width ratios, the polish mechanics, the flipped-ω structural observations.

**m127 – m129** form the flipped-ω enumeration mini-arc (correctly windowed). m127 (60 k SO(3) grid) found a second flipped-ω attractor in seed 12, independent of seed 33's. m128 (warm-start polish from m115 basins) refuted — seed 33's m115 basin ω is already retrograde, so `−ω_basin` points forward. m129 (600 k super-Fibonacci dense grid on seed 33) refuted the density lever: grid spacing is still ~1000× too coarse for a sub-0.001° basin width, and the surrogate cost has a saturation plateau at ~1.0 that blocks polish from crossing 7° of q0. The open thread is m130 — DE over q0 with ω = −ω_true — as the last blind-search enumerator.

**Verdict on Era 6.** The wrapped pipeline is **provisionally validated modulo re-scoring**. The flipped-ω thread converted a single-seed oddity into a cohort-level degeneracy with two confirmed seeds (33 narrow, 12 wide). The project now stands on a surrogate-discovered state set with an unresolved question about hi-fi honesty.

---

## Ideas that re-appeared under new names

- **PAB circle as a hard manifold.** m038/m043 proved unconstrained L-BFGS-B walks off it. m042b enforced it via brute phi sweep. m113's 3-DOF DE implicitly preserves it because the cost landscape is invariant to phi. Same insight, three implementations.
- **Short-window basins.** m061 measured 10° ω basin at 180 s. m121 measured < 0.1° at 3600 s. Same physics (time-integral of ω-error) seen from both ends. The progressive-extension idea (m062) was never scaled; it would interact interestingly with the surrogate, because the cost of running many short windows is now cheap.
- **Alignment as residual surface.** m069b's glint-alignment cost, m083's filtered alignment, m094's adaptive BRDF, and m119v2's per-epoch SO(3) isoshell are four takes on the same object: a cost surface on SO(3) whose minima are the compatible attitudes at each epoch. m119v2 is the most general, and the one with an honest surrogate noise model.
- **L-conservation.** m023–m025 delivered a nine-orders-of-magnitude discriminator and has not been re-used since Era 1. Every later architecture has ignored it.
- **Multi-solution philosophy.** First articulated in m042b (correct normal ranks #1 out of many), crystallised in m115 (enumerate valid basins, don't chase a single best), extended in m120 (validate the full SO(3) cost surface, not one point), and completed in m127 (twins are degeneracies, not failures).

## Revivable dead ends now that the surrogate exists

Everything below was compute-bound; none is physics-bound.

1. **m016c trough-constrained bridge.** Never finished because each bridge solve cost ~11 s. On the surrogate, ~0.2 ms/solve; a full exploration is minutes. Does the trough constraint meaningfully narrow the bridge pool?
2. **m018 L-consistency at end-to-end scale.** m023 showed L-conservation is perfect with correct solutions in the pool. Combine a surrogate-dense band-sweep with the L filter; does truth rank #1 end-to-end?
3. **m022a winding-score direction sweep.** m021 refuted multi-epoch winding selection because all staircase omegas had 13–25° direction error. Fix direction via the surrogate, isolate the winding-number question. The diagnosis should invert.
4. **m030 hi-fi pruning at 10 k seeds/peak.** Original used 1000. Surrogate allows 10 k cheaply. Is hi-fi pruning a general strategy or a coverage artefact?
5. **m061/m062 progressive-window extension.** Short-window basin was demonstrated but never operationalised into a cold-start pipeline. On the surrogate, a dense 180 s multi-start followed by window extension to 3600 s is tractable. This might be the cold-start recipe the project is still missing.

## Open thread: the data-integrity bug

Two compounding bugs define the current action gate.

**Bug 1** (architectural): every pipeline reads m046 (fixed 10:00–11:00 UTC, shared across 100 seeds). m048 (per-seed random start times, the intended design) is unused. Every "cohort-universal" claim is a single-geometry claim.

**Bug 2** (script-level): m122/m123/m124/m125/m126 silently defaulted to a 6-hour window instead of m046's 1-hour. All hi-fi MSE numbers from these five scripts are wrong.

Three responses are on the table. **A** — fix Bug 2 only, re-score the five experiments on the correct 1-hour window (~1 session). **B** — migrate the pipeline from m046 to m048 and regenerate the research arc on diverse per-seed geometries (5–15 sessions). **C** — institutional: `DATA_INVARIANTS.md`, reviewer-checklist extensions, mandatory 1-seed dry-run + LC-compare sanity step. **C** is already in place. The A/B choice is the user's.

Structural observations (basin anisotropy, polish mechanics, flipped-ω geometry, the m115 state discovery, everything in Eras 1–5) survive either way.
