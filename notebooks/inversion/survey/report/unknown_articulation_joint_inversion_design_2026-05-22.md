---
title: Recovering unknown articulation jointly with (q0, ω) — design / brainstorm
type: design-brainstorm
sources:
  - data/models/intelsat_901/intelsat_901_config.yaml
  - src/computation/inertia_calculator.py
  - notebooks/inversion/09_glint_analysis/m048_generate_trajectories_v2.py
  - notebooks/inversion/survey/lib/surrogate_eval.py
  - notebooks/inversion/survey/lib/hifi_render.py
  - notebooks/inversion/survey/lib/jacobi_propagator.py
  - notebooks/inversion/survey/lib/filter_costs.py
related: [s072 (jacobi path2), s059i/j (anchor + observability), s053 (polhode diameter), s043 (body-twin)]
created: 2026-05-22
updated: 2026-05-22
confidence: design-only (no experiment run; predictions stated, not measured)
---

# Recovering unknown articulation jointly with (q0, ω)

## 0. TL;DR

We assume the near-future pipeline solves the **known-articulation** inverse problem: from a 60-min light curve it recovers the tumble state `(q0, ω)` under the frozen-appendage assumption (panels at 0°, dishes at 15°). This report designs the extension where **the appendage angles are also unknown** and must be recovered.

The headline facts that shape every design choice:

1. **The search grows by only +2 dimensions, not +4.** The hi-fi model has four articulating bodies, but the **v2 surrogate that the inversion actually runs on collapses them to two scalars** — `sp_angle_deg`, `ad_angle_deg` (`survey/lib/surrogate_eval.py:42-63`). The whole inversion already routes through these two numbers, and their defaults (`0.0`, `15.0`) are exactly the m048 truth.
2. **The two unknowns are physically asymmetric.** `ad_angle` is **optical-only** — it touches per-epoch brightness and nothing else. `sp_angle` is **optical *and* dynamical** — the solar-panel angle reshapes the inertia tensor (`src/computation/inertia_calculator.py:725-841`; the m048 generator feeds SP angle into the inertia calc at `m048_generate_trajectories_v2.py:78`), which moves the principal axes, the polhode, and the entire torque-free ω(t). So a wrong panel angle is punished **twice** (wrong brightness *and* wrong trajectory), while a wrong dish angle is punished once.
3. **Frozen articulation keeps the dynamics torque-free with constant inertia**, which is precisely the regime the Jacobi closed-form propagator assumes. This is a strong, often-overlooked argument *for* the frozen model: it preserves the whole `propagate_jacobi_path2` machinery (`survey/lib/jacobi_propagator.py:435-499`). The "actively-moving panel" branch breaks constant-I and would force a return to numerical integration.
4. **No one in the SSA literature has done this.** Every published "joint" light-curve inversion augments attitude with a *static body-fixed* nuisance class — shape coefficients, BRDF parameters, optical/atmospheric properties. The most recent paper (Aerospace 2025) explicitly *names* "solar array orientations" as a source of non-uniqueness and then **fixes** the geometry. The gap is real and specific (see §9).

**Recommended architecture (one line):** a hybrid — a coarse **MMAE-style hypothesis bank over `sp_angle`** (re-propagate per panel-angle hypothesis, because panel angle is dynamical), with **`ad_angle` profiled out continuously inside each** (cheap, optical-only, no re-propagation), then a **bounded joint LM polish** over `(q0, ω, sp, ad)` on the survivors. This reuses the existing ω-grid / anchor / surrogate pipeline almost verbatim and adds the appendage angles where they are cheapest to add.

**Do the observability analysis first** (§3). It is a surrogate-only afternoon, it is what the literature says to do (Hinks et al. 2013), and it tells us *which* of the new `(q,ω,α)` degeneracies are structural before we commit a single batch.

---

## 1. Problem statement and notation

Forward model, per epoch *t*:

```
mag(t) = M( q0, ω, α ; t )      with   α = (sp_angle, ad_angle)
```

`M` factors as: propagate `(q0, ω)` to attitude `q(t)` → rotate sun/observer into the body frame → evaluate reflectance with the appendages set to `α`. In code, propagation is `jacobi_propagator.propagate_jacobi_path2`, and reflectance is `surrogate_eval.predict(k1_body, k2_body, obs_dist_km, sp_angle_deg, ad_angle_deg)`.

| Quantity | Known? (this report) | Role |
|---|---|---|
| `q0` (initial attitude) | unknown | dynamics + optics |
| `ω` (initial body-frame rate) | unknown | dynamics + optics |
| `sp_angle` (solar-panel gimbal) | **unknown (new)** | **optics + dynamics (via inertia)** |
| `ad_angle` (antenna-dish pointing) | **unknown (new)** | **optics only** |
| inertia tensor `I(sp_angle)` | function of `sp_angle` | dynamics |
| BRDF / masses / geometry | known | fixed |

**Temporal model: frozen.** Per the framing decision, both angles are *constant but unknown* over the pass. This matches an uncontrolled tumbler (a dead spacecraft cannot sun-track) and is exactly how the m048 truth was generated — `sp=0°`, `ad=15°` held constant across all epochs (`m048_generate_trajectories_v2.py:152-154`). The actively-moving and free-wheeling-appendage cases are sketched as harder branches in §8.

**Success metric.** Extend the existing convention. Today we report three errors (`q0`, ω-dir, ω-mag) in ρ-bands. Add two: `sp_err` and `ad_err` in degrees. A solution is accepted on the same multi-solution / observational-indistinguishability basis as before (hi-fi ρ < 2 is a valid fit regardless of whether it is the truth). An articulation "twin" that fits to ρ < 2 with a *different* `α` is a finding, not a failure — it is a degeneracy to characterize.

---

## 2. The two unknowns are not symmetric (grounded keystone)

This is the structural fact the architecture is built around.

### `ad_angle` — a clean optical nuisance parameter
The dishes are excluded from the inertia computation (the m048 generator passes only `{'SP_North','SP_South'}` to `compute_inertia_from_config`, `m048_generate_trajectories_v2.py:78`; dishes are negligible mass). So `ad_angle` enters **only** through `surrogate.predict(..., ad_angle_deg)`. Changing it is one cheap surrogate call — **no re-propagation**. It is the textbook nuisance parameter: profile it out, optimize it away, marginalize it — all cheap.

### `sp_angle` — an optical *and* dynamical parameter
`compute_inertia_from_config` rotates each articulating component mesh by its angle *before* the inertia integral (`src/computation/inertia_calculator.py:725-841`; docstring example `articulation_angles={'SP_North': 45.0, 'SP_South': -30.0}`). Therefore:

```
sp_angle  →  I(sp_angle)  →  principal axes + polhode  →  ω(t), q(t)  →  whole light curve
```

A hypothesized `sp_angle ≠ truth` produces a **different inertia tensor**, hence a **different polhode and a different ω(t)** — even before any reflectance change. Two consequences:

- **Cost (bad):** profiling `sp_angle` is *not* free. Each candidate `sp_angle` needs a re-diagonalized inertia and a fresh `propagate_jacobi_path2`. (Still cheap — closed-form q(t) — but not a single surrogate call.)
- **Observability (good):** a wrong `sp_angle` corrupts the **trajectory shape**, not just instantaneous brightness. Over a full 60-min LC the dynamical mismatch *accumulates* (the polhode beats against the true one). So `sp_angle` should be **more identifiable than a pure-optical parameter would be**, especially for fast tumblers where the polhode cycles many times per pass. This is the silver lining of the coupling and a testable prediction (Exp 4, §7).

### The polhode prior becomes `sp`-dependent
Prior survey work used a polhode-geometry prior to bound the ω search (the polhode-diameter→basin-width relationship; *pre-bug-fix numerics, treat as a question until re-confirmed post-fix*). Because the polhode family is a function of the inertia tensor, and inertia is a function of `sp_angle`, **the polhode prior itself is now conditioned on the panel-angle hypothesis.** This is the cleanest reason to treat `sp_angle` as a *discrete outer hypothesis* (one polhode family per hypothesis) rather than a continuously-swept parameter in the global search.

| | `sp_angle` (panels) | `ad_angle` (dishes) |
|---|---|---|
| Affects per-epoch brightness | ✓ | ✓ |
| Affects inertia / polhode / ω(t) | ✓ | ✗ |
| Cost to change in a candidate | re-propagate (cheap, closed-form) | one surrogate call |
| How to handle | **discrete outer hypothesis** (re-propagate per value), MMAE-style | **continuous inner profile** (no re-propagation) |
| Observability lever | dynamical fingerprint accumulates over LC | appendage glint timing/amplitude |

---

## 3. Observability first (analytical before computational)

Per project discipline (analytical before computational) and per the literature (Hinks et al. 2013 say compute the observability nullspace *before* estimating), the first deliverable is **not** an inversion — it is a surrogate-only identifiability map. It is cheap (an afternoon, no batch) and it tells us whether the problem is well-posed before we spend compute.

### Flux decomposition (the lever)
Total flux separates by source: `F_total(t) = F_body(q,ω; t) + F_panels(q,ω,sp; t) + F_dishes(q,ω,ad; t)`. Articulation enters only through the appendage terms. We only observe the sum, so we cannot subtract — but the decomposition tells us *when* each appendage carries information.

### Glint-driven observability
An appendage's angle is most observable at epochs where **that appendage glints** — when its specular lobe sweeps the observer (the PAB / phase-angle-bisector alignment that defines a glint in this project). Between glints the appendage contributes a broad diffuse term that is weakly sensitive to its angle. So articulation information is **concentrated at appendage-glint epochs**, and (consistent with the prior "sharpness ≠ brightness" finding) those epochs are not necessarily the bright peaks — a globally-dim moment can be the one that pins the dish angle. **Implication:** anchor the articulation estimate at appendage-glint epochs, not at LC peaks.

### Degeneracy modes to map
1. **Single-epoch α↔q trade (Hinks nullspace).** At one instant, a panel re-orientation can mimic a body re-orientation — the augmented Fisher information is rank-deficient at a single epoch (single-instant brightness is massively under-determined; observability comes from the time-history). *The exact rank claim in Hinks et al. is abstract-sourced — verify against full text before quoting a number.* The actionable version: **build the augmented FIM/Gramian for `(q,ω,sp,ad)` over the trajectory and inspect its smallest singular directions.** Those are the structural twins.
2. **Articulation twins.** Does freeing `α` create new `(q,ω,α) ≠ (q',ω',α')` pairs with identical LCs? And does the **known body-twin survive?** The body-twin (`q_180x · q0` with the *same* ω, a forward-model invariant that survives the bug fix) is a 180° rotation about a body axis — it permutes which facets face the observer. With panels free, the body-twin may pair with a *different* panel angle to stay degenerate, or may break. This is a direct, cheap surrogate experiment.
3. **The Kaasalainen shape–albedo analog.** The asteroid-inversion literature's canonical degeneracy — uniform-albedo non-spherical body vs. variegated-albedo round body — is the mathematical sibling of ours: **two distinct physical re-orientations (body vs. panel) producing the same brightness.** Its resolutions transfer: keep articulation a *constrained low-DOF* parameter (we have exactly 1 effective DOF per appendage — good), exploit **large phase-angle arcs** (better conditioning, mirrors our k1≈k2 manifold collapse), and if ever available add an **independent channel** (hyperspectral material separation — see §9).

### Cheap experiments (surrogate-only, no batch) — see Exp 0, §7.

---

## 4. Architecture options

For each: mechanism, use of Jacobi + surrogate, cost, failure mode, fit to the existing pipeline, literature anchor.

### A. Joint augmentation (global LM over `(q0, ω, sp, ad)`)
Add the two angles to the polish vector and let `least_squares` solve all of it. The cost functions already accept `sp_angle_deg`/`ad_angle_deg` (`survey/lib/filter_costs.py`), so the residual extends trivially.
- **Use:** best as the **final polish** stage, in-basin.
- **Cost:** +2 columns in the Jacobian; `sp` perturbations trigger re-propagation.
- **Failure mode:** enlarges the basin-of-attraction problem; `sp`–q coupling spawns new local minima. Weak as a *global* search.
- **Note:** must switch the polisher to a **bounded** method (`trf`/`dogbox`) because `α` has hard physical limits (`sp ∈ [-180,180]`, `ad ∈ [0,90]` per `intelsat_901_config.yaml:69-95`); the unbounded `lm` method we use for the q-from-ω polish cannot hold the box.
- **Lit anchor:** the rigid "6-param + surface params" baselines (IEEE astrometric+photometric; Robinson & Frueh 2025).

### B. Classify-then-invert (MMAE hypothesis bank) — **for the discrete part**
Discretize the configuration into hypotheses and run the *existing* known-α inverter under each, then pick by Bayesian weight. Natural hypotheses: a coarse `sp` grid (each defines a polhode family), a coarse `ad` grid, and — if we ever leave the frozen branch — a "sun-tracking vs frozen" discrete test (literally a two-hypothesis MMAE).
- **Use:** the **outer loop over `sp`** (because `sp` is dynamical, one polhode family per hypothesis); robust to multimodality and twins.
- **Cost:** (existing pipeline) × (#hypotheses), embarrassingly parallel (Pool).
- **Failure mode:** if the true `sp` falls between grid points, the best hypothesis is biased — mitigated by a continuous polish afterward.
- **Lit anchor:** Linares, Jah, Crassidis, Nebelecky (2014) — MMAE bank of UKFs over discrete shape hypotheses, Bayes-rule weights. The single most directly reusable architecture in the literature.

### C. Alternating block-coordinate (EM-like)
Fix `α`, solve `(q,ω)`; fix `(q,ω)`, solve `α`; iterate.
- **Use:** a cheap refinement once roughly in-basin.
- **Failure mode:** stalls in joint local minima; needs a good init (from B).

### D. Profile `α` as a nuisance — **for the continuous part (recommended core)**
For each candidate `(q,ω)` in the search, **inner-minimize the residual over `α`** and use the profiled cost `r*(q,ω) = min_α r(q,ω,α)` as the score. Because `ad` is optical-only and smooth, the `ad` inner solve is a 1-D minimization of ~10–50 surrogate calls — negligible. The `sp` inner solve needs a re-propagation per step, so in practice we **profile `ad` continuously and grid `sp` coarsely** (i.e., D for dishes nested inside B for panels).
- **Use:** keeps the **outer search dimensionality unchanged** — the ω-grid / anchor / Sobol machinery is untouched; only the scoring function changes.
- **Failure mode:** `α` can **over-fit** — absorb genuine model mismatch and fill in false basins. Mitigate with physical bounds, mild regularization toward nominal, and the standard hi-fi ρ-band gate at the end.
- **Lit anchor:** profile-likelihood (Cousins & Wasserman, arXiv:2404.17180): profiling (optimize out) is the cheap, principled choice when `α` is a nuisance; marginalization is the expensive full-uncertainty alternative (→ option E).

### E. Full Bayesian joint (adaptive Hamiltonian MCMC)
Sample the joint posterior over `(q0, ω, sp, ad)` with gradient-informed proposals.
- **Use:** when we want the **posterior / uncertainty** over `α`, not a point estimate; tolerates the known multimodality.
- **Cost:** highest; reserve for a few showcase seeds, not the cohort.
- **Lit anchor:** Linares & Crassidis (2018) — adaptive HMCMC for joint shape+attitude+BRDF, deployed precisely because the posterior is multimodal/non-Gaussian.

### Recommended hybrid
**Coarse MMAE over `sp` (B) → `ad` profiled continuously inside each (D) → bounded joint LM polish over `(q0,ω,sp,ad)` (A) on survivors → hi-fi ρ-band gate.** This mirrors the project's established **"coarse-find then dense-resample"** anchor philosophy: discretize the dynamical, expensive direction (`sp`); refine the cheap, smooth direction (`ad`) continuously; finish with a joint polish. Option E is the optional posterior layer for showcase seeds.

---

## 5. How it plugs into the existing pipeline

The extension is deliberately minimal — the surrogate and cost functions already speak `α`.

- **The lock to remove.** Articulation is currently pinned by the module-level constants `ART_ANGLES_DEG` in `survey/lib/hifi_render.py:70-75`, consumed in `build_context()` at `:143-144`. Make these a parameter (`build_context(seed, art_angles_deg=...)`) and thread it through `render_hifi`. The surrogate path needs nothing — `predict` already takes the two scalars.
- **The ω-grid / Fibonacci-direction search.** Unchanged in dimensionality if `ad` is profiled per candidate. `sp` becomes an outer loop (one grid pass per `sp` hypothesis).
- **The polhode prior.** Recompute `I(sp)` and the polhode family per `sp` hypothesis (cheap, once per grid value). This is the structural reason `sp` is an outer hypothesis, not an inner sweep.
- **The anchor architecture.** Profile `α` *at the appendage-glint anchor epochs* (§3), where it is most observable.
- **The LM polish.** Extend the parameter vector from `(q0,ω)` (local 3-param attitude chart + 3 ω) to `+ (sp, ad)` = 8 DOF; switch to a **bounded** least-squares method for the `α` box.
- **Surrogate-first / hi-fi-last (unchanged).** Every search/score/profile/polish step runs on the surrogate (~50,000× hi-fi per `survey/concepts/surrogate_model.md`). Hi-fi renders only the final per-basin winner for ρ-band classification — now with the recovered `α`.

---

## 6. Cost / tractability

- **Dimensionality:** +2 on the surrogate (`sp`, `ad`). Not +4 — the surrogate links the panel pair and the dish pair into one scalar each (`surrogate_eval.py:42-63`).
- **`ad` profiling:** optical-only, smooth, monotone-ish away from its single glint-driven minimum → ~10–50 surrogate calls per candidate. Negligible against the per-candidate propagation already being done.
- **`sp` gridding:** each value = one inertia re-diagonalization + one closed-form `propagate_jacobi_path2`. Closed-form q(t) is what makes a per-`sp` re-propagation affordable — **this is the concrete pay-off of the Jacobi work for *this* problem.** A coarse grid (say 5–9 panel angles spanning the physical range) is a 5–9× multiplier on the existing per-seed cost, fully parallelizable (Pool, BLAS threads = 1).
- **MMAE bank:** embarrassingly parallel across `sp` hypotheses and seeds.
- **Net:** the frozen-α extension is a **small constant-factor** cost increase over the known-α pipeline, dominated by the `sp` grid width. No new asymptotic wall.

*(Throughput note: the surrogate has previously been operated at tens of thousands of evals/sec under Pool in this workspace; re-verify the exact rate before quoting it in a results doc — the figure is not documented in `surrogate_eval.py`.)*

---

## 7. Recommended path & first experiments (cheap-first, with predictions)

Each experiment states the hypothesis, which cases should move, and what confirms/refutes — per project discipline. Surrogate-only unless noted. Next series is **s088+** at time of writing (latest committed is s087).

### Exp 0 — Articulation observability map (analytical, surrogate-only, ~hours, no batch)
**Do this before anything else.** On one seed already inverted cleanly by the current post-fix pipeline, at the *truth* `(q0,ω)`:
- compute the profiled-cost surface over `(sp, ad)` and the two sensitivity time-series `∂mag/∂sp(t)`, `∂mag/∂ad(t)`; mark the appendage-glint epochs;
- build the augmented FIM/Gramian for `(q,ω,sp,ad)` over the trajectory; inspect smallest singular directions (the structural twins);
- search for articulation twins, including whether the body-twin (`q_180x·q0`, same ω) survives with `α` free.

**Predict:** `ad` has a clean unimodal minimum at 15° with sensitivity spiking at dish-glint epochs; `sp` minimum at 0° but *broader optically* yet *sharpened dynamically* once propagation is included. **Refute well-posedness if** the cost is flat in `α` (→ unobservable for that seed/geometry) or the FIM has a hard zero direction mixing `q` and `sp` that the time-history does not lift.

### Exp 1 — Machinery validation (recover a known answer)
Take a seed the current pipeline inverts cleanly. Free `(sp, ad)`, **initialize them away** from truth (e.g. `sp=40°, ad=40°`), run the hybrid (B→D→A).
**Predict:** recovers `(0°, 15°)` within a few degrees and `(q,ω)` ρ-band is preserved. **Confirm:** small `α`-err, ρ-band unchanged vs known-α run. **Refute:** `α` drifts or ρ degrades → a real degeneracy, escalate to Exp 0's twin map.

### Exp 2 — Generalization across `α` (the real test) — requires new data
The entire m048 cohort was generated at `sp=0°, ad=15°` (`m048_generate_trajectories_v2.py:152-154`), so it **cannot** test `α`-recovery generalization — every truth is the same point. Generate a **small varied-α validation set**: ~3 seeds (slow + fast tumbler) × a coarse `{sp ∈ [0,30,60], ad ∈ [0,15,30]}` truth grid, re-using the existing generator with non-default angles (and the matching `I(sp)` for the panel cases). Invert each with `α` free.
**Predict:** recovery is easy where an appendage glints during the pass and hard/twinned where it never glints (the diffuse-only regime). Name the `(seed, α)` cells expected to twin *before* running. **This is the experiment that earns the publishable claim.**

### Exp 3 — Discrete classify baseline (MMAE)
Run the coarse `sp`-hypothesis bank with Bayesian weights on a couple of Exp-2 truths; report the weight posterior over `sp`.
**Predict:** the bank nails the discrete `sp` cell (and a sun-tracking-vs-frozen test, if exercised) even on seeds where the continuous polish is shaky — discrete classification is more robust than continuous recovery.

### Exp 4 — Is `sp` more identifiable than `ad`? (the asymmetry test)
On a **fast** tumbler vs a **slow** one, compare the `sp`-error cost curve from the *optical-only* surrogate (inertia held at truth) against the *full* curve (inertia recomputed per `sp`, re-propagated).
**Predict:** the full curve is markedly sharper for the fast tumbler — the polhode mismatch from a wrong `sp` accumulates over more cycles. Confirms `sp`'s dynamical fingerprint is an observability *asset*. **Refute if** the two curves coincide (→ inertia barely moves with `sp` for IS-901's panel geometry; quantify the `|dI/d(sp)|` magnitude, which is currently un-measured).

---

## 8. Risks, open questions, branches

- **`α` over-fitting (option D's main risk):** a free `α` absorbs genuine model mismatch and manufactures false basins. Mitigate with physical bounds, mild regularization toward nominal, and the hi-fi ρ-band gate.
- **Twin inflation:** freeing `α` may enlarge the valid multi-solution set. Consistent with the project's multi-solution philosophy — but every accepted twin must be *labeled* with its `α`, not silently averaged.
- **`sp` magnitude is unmeasured:** the whole `sp`-dynamical story hinges on `I` actually moving meaningfully with panel angle for IS-901. The coupling *exists* in code; its *magnitude* is unverified (Exp 4 measures it). If `|dI/d(sp)|` is tiny, `sp` collapses to a near-optical parameter and the architecture simplifies (profile both angles).
- **4-DOF vs 2-DOF:** the surrogate links the panel pair and dish pair. If a scenario ever needs *independent* panels/dishes (asymmetric failure), the surrogate must be retrained or hi-fi used (expensive). Out of scope for the frozen IS-901 case.
- **Data lifecycle (design-time gate):** Exp 2's varied-α set is new data — tag it, decide retention, keep candidate-level `α` in the checkpoints (every script saves `q0, ω, sp, ad, errors` to NPZ/JSON).
- **Branch: actively-moving panels.** Time-varying `α` makes `I` time-varying → torque-free-with-constant-I breaks → Jacobi closed-form no longer applies; you would estimate a *tracking-law* (axis, offset, lag) rather than a constant, and likely fall back to numerical propagation. Much harder; and physically at odds with a tumbler. Treat as a separate project.
- **Branch: free-wheeling appendage.** A sheared/unlatched panel windmilling at its own rate adds a *second* dynamical mode (the appendage's own ω) atop the body tumble. The closest analog in the literature is windmilling rocket bodies, but those are modeled as a *single* rigid spinner, not body-plus-appendage — genuinely novel and the hardest branch.

---

## 9. Literature review (synthesized)

**Bottom line:** no published work jointly estimates attitude *and* an independently-articulating appendage angle from a single optical light curve. Every "joint/simultaneous" estimator augments attitude with a *static body-fixed* nuisance class. The gap is real and specific.

| Source | Problem | Architecture | Transplant |
|---|---|---|---|
| Linares, Jah, Crassidis, Nebelecky **2014**, *JGCD* 37(1):13–25, DOI 10.2514/1.62986 | shape + state from LC+angles | **MMAE** bank of UKFs, Bayes-weight model selection | discrete `sp`/sun-tracking hypothesis bank (our option B) |
| Linares & Crassidis **2018**, *J. Astronaut. Sci.* (AAS 16-514) | joint shape+attitude+BRDF, no prior on any | **adaptive Hamiltonian MCMC** | continuous joint posterior over `(q,ω,sp,ad)` (option E) |
| Hinks, Linares, Crassidis **2013**, AIAA 2013-5005 | when is attitude observable from brightness? | **Fisher-information / nullspace** analysis | run the augmented FIM **first** (our Exp 0). *Exact rank claim abstract-sourced.* |
| "Dynamic observability…", *JGCD* DOI 10.2514/1.G002229 | observability over a trajectory | observability **Gramian** | decide whether an arc carries enough info to *add* `sp` vs profile it |
| Bradley & Axelrad **2014**, ISSFD S10-3 | asteroid-style inversion of man-made objects incl. **box-wing** | convex/photometric inversion + MCMC | **box-wing flat panels are the hard case**; unknown pole → **bilinear/non-convex** → favors global search. *URL mapping flagged uncertain.* |
| *Aerospace* **2025**, 12(10):942 (MDPI) | joint attitude + optical props, **known shape** | hybrid estimation/optimization | **names "solar array orientations" as a non-uniqueness source then fixes geometry — this defines our gap.** *Internals abstract-sourced.* |
| Cousins & Wasserman **2024**, arXiv:2404.17180 | profile vs marginalize nuisance params | statistics review | **profile `α` per candidate** (option D); marginalize if you want full uncertainty |
| Kaasalainen & Torppa et al. | asteroid convex inversion | regularized NLS / Bayesian | **shape–albedo degeneracy = sibling of attitude–articulation degeneracy**; resolutions: low-DOF param, large phase angle, extra channel |
| Hyperspectral LC inversion, arXiv:2401.05397 | attitude from *spectral* LC | (unread) | **material separation** could split panel vs body flux — a way to break the degeneracy if multi-band data exist. *Unverified — read full text.* |
| Robinson & Frueh **2025**, *J. Astronaut. Sci.* 73:7, DOI 10.1007/s40295-025-00557-9 | global LC attitude w/ noise + inertia uncertainty | global optimization (box-wing) | = our RF25; inertia-uncertainty handling is relevant to the `sp→I` coupling |

**Reusable architectural families:** (1) **MMAE** for the discrete configuration question (lowest risk; "sun-tracking vs frozen" is literally a two-hypothesis MMAE); (2) **profile-likelihood / adaptive HMCMC** for the continuous gimbal angle; (3) the **observability nullspace** toolkit, to be run first. The deepest cautionary analog is Kaasalainen's shape–albedo degeneracy — two physical re-orientations producing identical brightness — whose resolutions (low-DOF parameterization, large-phase-angle arcs, an independent channel) map directly onto our design choices.

**Sourcing caveat:** Tavily hit a usage limit, so the literature was gathered via WebSearch; several per-paper details (Hinks FIM rank, Bradley & Axelrad URL, Aerospace 2025 internals, the hyperspectral paper) are from abstracts/snippets, not full-text reads, and are flagged inline. Verify those against full text before they enter any published claims section.

---

## 10. Reporting conventions for this sub-project

- **Report five errors, not three:** `q0`-err, ω-dir-err, ω-mag-err, **`sp`-err (deg)**, **`ad`-err (deg)**.
- **ρ-bands unchanged** (`ρ = √(hifi_MSE)/0.05`, bands A/B/C/D); success = ≥1 Band-A∪B basin per seed, now annotated with its recovered `α`.
- **Every checkpoint saves `α`** alongside `(q0, ω)` and all errors (NPZ/JSON), so any later re-scoring can re-rank without re-running.
- **Multi-solution philosophy holds:** an articulation twin fitting to ρ < 2 is a valid, reportable solution — label it with its `α`; do not collapse it into the truth.

---

## Appendix — pseudocode

```python
# ---- D nested in B: profiled score for one (q0, omega) candidate ----
def profiled_cost(q0, omega, lc_obs, geom, SP_GRID, AD_RANGE):
    best = +inf
    for sp in SP_GRID:                              # outer: sp is DYNAMICAL
        I_sp   = compute_inertia_from_config(cfg, cm, masses, {'SP_North': sp, 'SP_South': sp})
        q_t, _ = propagate_jacobi_path2(q0, omega, I_sp.inertia_tensor, geom.times)
        k1b, k2b = to_body_frame(q_t, geom.sun_body, geom.obs_body)
        # inner: ad is OPTICAL-ONLY -> no re-propagation, ~10-50 surrogate calls
        ad_opt, c = minimize_scalar_bounded(
            lambda ad: full_lc_mse(predict(k1b, k2b, geom.dist, sp, ad), lc_obs),
            bounds=AD_RANGE)
        if c < best:
            best, sp_best, ad_best = c, sp, ad_opt
    return best, sp_best, ad_best                   # profiled score + recovered alpha

# ---- B: MMAE-style discrete classify over sp (parallel, Pool) ----
def classify_sp(lc_obs, geom, known_alpha_inverter, SP_GRID, sigma, log_prior):
    loglik = {}
    for sp in SP_GRID:                              # one polhode family per hypothesis
        (q, w), resid = known_alpha_inverter(lc_obs, geom, sp=sp)   # EXISTING pipeline, sp fixed
        loglik[sp] = -0.5 * np.nansum((resid / sigma) ** 2)
    weights = softmax(np.array([loglik[s] + log_prior[s] for s in SP_GRID]))
    return weights, SP_GRID[argmax(weights)]        # posterior over sp + MAP

# ---- A: bounded joint polish on a survivor ----
def joint_polish(q0, w, sp, ad, lc_obs, geom):
    th0 = pack(q0_chart(q0), w, sp, ad)             # 3 (q chart) + 3 (w) + 2 (alpha) = 8 DOF
    def resid(th):
        q0_, w_, sp_, ad_ = unpack(th)
        I = compute_inertia_from_config(cfg, cm, masses, {'SP_North': sp_, 'SP_South': sp_})
        q_t, _ = propagate_jacobi_path2(q0_, w_, I.inertia_tensor, geom.times)
        k1b, k2b = to_body_frame(q_t, geom.sun_body, geom.obs_body)
        return predict(k1b, k2b, geom.dist, sp_, ad_) - lc_obs
    sol = least_squares(resid, th0, method='trf',   # BOUNDED (alpha has hard limits); not 'lm'
                        bounds=physical_bounds())    # sp in [-180,180], ad in [0,90]
    return unpack(sol.x), rho_band(sol)
```

**Pipeline (one line):** `classify_sp` (B) → for top-weight `sp` cells, run the ω-grid/anchor search scored by `profiled_cost` (D) → `joint_polish` (A) the survivors → hi-fi render the basin winners for ρ-band classification.
