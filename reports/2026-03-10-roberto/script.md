# Talking Script — 10 March 2026

Use alongside the report. Each section matches an experiment in the document.

---

## Context (opening)

> "IS-901 test case, fast tumbler at about 2.08 degrees per second — roughly 3 full rotations between brightness peaks. The core issue from last week: the graph pipeline failed because the bridge solver finds an omega that arrives at the correct attitude but takes a completely different path. This week I ran six experiments to understand why and to test Roberto's staircase idea."

---

## Experiment 1: Forward Propagation — Analytic ω, Exact Attitude, Hi-fi

> "Starting with the simplest possible case. I take the true quaternion trajectory, pick two consecutive observations near a brightness peak — separated by about 1.2 seconds — and compute omega analytically from the axis-angle rotation between them. Then I propagate forward from the exact attitude at that peak using Euler's equations and generate a hi-fi lightcurve."
>
> "There's no optimisation here. It's purely: does the analytic omega from two nearby exact quaternions reproduce the lightcurve? Yes — to less than one millionth of a degree per second error. At this short timescale the rotation between observations is only about 2.5 degrees, so there's no winding ambiguity. The axis-angle gives the right answer trivially."
>
> "The point: omega recovery is not the bottleneck when you have exact attitudes close together. The problem is when the gap is hundreds of seconds and multiple full rotations."

---

## Experiment 2: Forward Propagation — Analytic ω, 3° Attitude Nudge, Hi-fi

> "Same setup — same analytic omega from exact quaternions — but now I nudge the starting attitude by 3 degrees in 5 random directions before propagating. This simulates what happens if our attitude candidate has a few degrees of error."
>
> "The finding: sensitivity is completely direction-dependent. At the glint around t equals 107 seconds, some nudge directions produce nearly identical lightcurves, others shift the peak by more than 2 magnitudes. The same 3 degrees of error can be benign or catastrophic depending on where in SO(3) it points."
>
> "This matters for the pipeline because it means we can't just ask 'how many degrees off is our candidate' — the direction of the error matters as much as the size."

---

## Experiment 3: Forward Propagation — Analytic ω, Exact Attitude, Lo-fi Reconstruction

> "Now I test whether we can skip shadow computation. Same exact omega and exact starting attitude, but the reconstructed lightcurve is generated without self-shadowing — lo-fi. The observed lightcurve is still hi-fi."
>
> "At this particular orientation — 160 degrees about an arbitrary axis — lo-fi predicts a bright glint at t equals 60 seconds that simply doesn't exist in the observations. A facet is in self-shadow at that epoch. Hi-fi sees darkness, lo-fi sees a glint. And this is with zero attitude or omega error — the only difference is shadow computation. So the MSE between lo-fi and truth is large at the correct answer. You can't rank candidates with a metric that's wrong at truth."

---

## Experiment 4: Forward Propagation — Analytic ω, 3° Nudge, Lo-fi (Different Orientation)

> "But is that a universal failure? No. Same setup, different starting orientation — 30 degrees about x — and lo-fi agrees with hi-fi just fine. No phantom peak. The lo-fi failure only happens at orientations where self-shadowing is active."
>
> "The problem: we can't know a priori which candidate orientations will have active self-shadowing. So we can't safely use lo-fi scoring anywhere in the pipeline."

---

## Experiment 5: Staircase ω Enumeration

> "This is Roberto's idea from last meeting. The analytic axis-angle always gives the lowest winding number. For a 555-second leg, that's about 0.32 degrees per second — the true value is 2.08, so the axis-angle is completely wrong."
>
> "The staircase: I optimise omega with L-BFGS-B to minimise arrival attitude error, but with a soft lower barrier on the magnitude. Step 0 finds the lowest omega that arrives correctly. Then I raise the floor above that solution and optimise again — step 1 finds the next winding band. Repeat 8 times."
>
> "Result: 8 distinct omega solutions, all arriving at the correct attitude. The mechanism works — it enumerates the whole winding family."
>
> "Then I tried scoring each step by the lo-fi brightness at a single trough between the two peaks. It picks step 2 at 1.6 degrees per second. The truth is step 3 at 2.23. At the trough epoch — which is only partway through the 555-second leg — steps 2 and 3 haven't diverged enough. One checkpoint isn't enough to distinguish them."

---

## Experiment 6: Angular Momentum Conservation Filter

> "Instead of scoring brightness at one point, I tried a physics-based filter. In torque-free dynamics, angular momentum L equals R times I times omega is conserved. So if I run the staircase independently on two legs sharing a middle peak, the correct winding pair must produce matching L vectors at that shared peak."
>
> "I ran the staircase on leg 0 — peaks 183 to 260, 555 seconds — and leg 1 — peaks 260 to 360, 721 seconds — independently. 8 solutions each, 64 pairs, pick the minimum L-mismatch."
>
> "It picked the wrong pair. But look at the table — leg 0 steps uniformly at about 0.65 degrees per second per step. Leg 1 jumps from 0.25 straight to 3.28, skipping three winding bands entirely. The staircase is broken for leg 1."
>
> "The cause: the axis-angle initialisation for leg 1 lands near a pi-rotation. The quaternion double-cover creates a discontinuity and the solver jumps past multiple bands. The fix is simple — initialise step 0 from omega-zero instead of the axis-angle estimate, and tighten the upper barrier so each step stays in its band. The code already has the barrier logic; it's a parameter change."
>
> "The L-conservation idea is sound. The implementation just needs the staircase fixed first."

---

## Decisions

> "Three things ruled out. MIP — talked to Jack, problem isn't discrete enough. Lo-fi scoring — structurally unreliable near glints, can't predict which orientations fail. Single-trough scoring — not enough temporal leverage to distinguish adjacent winding bands."
>
> "Most promising direction: L-conservation filter, once the staircase is fixed."

---

## Next Steps

> "Fix the staircase initialisation — omega-zero plus tight upper barrier. Rerun the L-filter. If it works with oracle attitudes, degrade to 2-to-5 degree endpoint errors and see how it holds up. If one filter isn't enough, stack multiple trough epochs as a secondary discriminator."
