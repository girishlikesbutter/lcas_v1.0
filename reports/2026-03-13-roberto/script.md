# Talking Script — 13 March 2026

Experiments 1–6 are recap from last meeting. Experiments 7–10 are new work.

---

## Recap (Experiments 1–6) — keep brief

> "Quick recap of where we left off. Experiments 1 through 6 are in the report from last time. The key findings were: omega recovery from exact attitudes works trivially at short timescales, attitude error sensitivity is direction-dependent, lo-fi scoring is unreliable near glints, the staircase enumerates winding families but single-trough scoring can't pick the right one, and the L-conservation filter concept is sound but the staircase implementation was broken on leg 1 due to a near-pi singularity."
>
> "Since then I went in a different direction — instead of fixing the staircase immediately, I investigated what brightness peaks actually are physically."

---

## Experiment 7: PAB Alignment Diagnostic

> "I wanted to understand the physical mechanism behind brightness peaks. The Ashikhmin-Shirley BRDF has a specular term that peaks when a facet normal aligns with the Phase Angle Bisector — the halfway vector between the sun direction and observer direction in the body frame."
>
> "What the script does: it takes the 3840 mesh facets, groups them by unique normal direction — there are only 14 unique normals on IS-901 after articulation — and at every epoch computes how closely each normal aligns with the PAB. Then it re-runs the hi-fi forward model with per-facet flux decomposition to see which normal group is actually producing the brightness."
>
> "The result is the four-panel figure. At every bright peak — magnitude below 9 — a single normal group captures over 77% of total flux, with alignment above 0.99 to the PAB. These are unambiguous specular glints. The dim peaks between them are diffuse — dominated by the big bus broadside faces through sheer area, with no near-perfect alignment."
>
> "So brightness peaks aren't mysterious. Each one is a specific facet achieving near-perfect specular reflection toward the observer."

---

## Experiment 8: Multi-Trajectory Robustness

> "That was one trajectory. Is it universal? I ran 30 random trajectories — random starting attitude, random omega direction, magnitude uniformly between 0.5 and 5 degrees per second."
>
> "439 bright peaks total across 30 trajectories. 392 — 89% — pass the strict specular glint criteria. The other 47 are what I'm calling near-misses: they have alignment above 0.984 and dominant fraction above 0.54. They're the same phenomenon, just marginally below the strict thresholds. Zero genuine counterexamples."
>
> "The near-miss rate increases slightly with rotation speed — 2% below 1 degree per second, 14% between 3 and 5. And glint count scales linearly with omega magnitude — roughly 3 times omega plus 6 per 3600-second window. Faster tumblers give more anchor points."

---

## Experiment 9: PAB Filter on Existing Candidates

> "If peaks are specular glints, can we use that as a filter? I took the 5643 iso-brightness candidates from the existing pipeline at epoch 183 — a known glint — and checked whether any of the 14 body-frame normals aligns with the inertial PAB within some angular threshold."
>
> "The surprise: ALL 5643 candidates already fall within 4.4 to 7.5 degrees of some normal. Compare that to random SO(3) quaternions where the median misalignment is 28 degrees. Matching brightness at a glint epoch implicitly selects for PAB alignment — the two constraints are measuring nearly the same geometric property."
>
> "At a 5-degree threshold, the filter kills 89% of candidates — down to 600 from 5643 — while preserving the truth at 4.62 degrees. That's a 10x reduction for free. But it's partially redundant with the brightness matching we're already doing."
>
> "The practical value: 10x fewer candidates means 100x fewer pairs to bridge in the omega stage."

---

## Experiment 10: BRDF Glint Profile

> "The last experiment characterises the specular lobe itself. I set up a synthetic flat plate at GEO distance and swept the normal through the PAB, varying n-phong, r-s, and phase angle."
>
> "Three findings. First, n-phong controls the lobe width — FWHM goes as 1 over root n-phong. For IS-901 components at n-phong 200 to 267, the specular-dominated region extends to about 10 degrees. This matches exactly the 4-to-8 degree spread we see in the iso-brightness candidates."
>
> "Second, r-s controls the amplitude — about 4 magnitudes of range — but barely changes the width. Third, phase angle is completely irrelevant — less than 0.01 magnitudes of variation across 5 to 60 degrees. The glint profile is an intrinsic material property."
>
> "This tells us the angular scale for everything: PAB alignment thresholds, candidate spread, and filter settings all live in the same 5-to-10 degree window, set by the BRDF lobe width."

---

## Decisions

> "The big new insight: bright peaks are specular glints. That's universal — 30 trajectories, zero counterexamples. Each glint constrains the attitude to a 1-DOF circle on SO(3) — the set of rotations that align a known facet normal with the known inertial PAB direction."
>
> "The PAB filter gives 10x candidate reduction, partially redundant with iso-brightness."
>
> "Phase angle doesn't matter. The lobe width — about 9.5 degrees for IS-901 materials — is the natural angular scale for all alignment-based constraints."

---

## Next Steps

> "Four directions. First, component identification — can we tell which of the 14 normals is responsible for a given glint from its brightness, duration, or recurrence? If so, each glint becomes a single specific PAB circle instead of 14 candidates."
>
> "Second, multi-glint intersection. Two glints from different components at different epochs give two PAB circles at different orientations. Combined with the omega bridge, that could yield a discrete set of q-omega candidates directly."
>
> "Third, glint-based candidate generation — instead of sampling 10,000 random seeds and optimising for iso-brightness, construct the PAB circle analytically for each of the 14 normals. With a brightness check, this could produce a very small candidate set per glint with no optimisation at all."
>
> "Fourth, the integration test — run the full pipeline end-to-end with real iso-brightness candidates instead of oracle attitudes, and see if the L-conservation filter still works."
