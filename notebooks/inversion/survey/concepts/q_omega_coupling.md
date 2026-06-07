---
title: "(q0, ω_dir) jointly determine the LC; |ω| is the only independent dimension"
type: concept
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# q-and-ω coupling

## What is independent in the inversion

The state to invert is `(q0, ω)` — six degrees of freedom:
- `q0` — initial attitude, 3 DOF (unit quaternion modulo sign)
- `ω` — initial body-frame angular velocity, 3 DOF (split as direction + magnitude)

Of these, only **|ω| (the angular speed) is independent** in the sense that varying it slides peaks in time without changing their pattern. The other 5 DOF — the 3 of `q0` and the 2 of ω̂ (ω direction) — couple jointly into the LC shape: changing one changes which sequence of body-frame sun/observer vectors the renderer sees.

Concretely: if you want to find truth `(q0_truth, ω̂_truth)` with `|ω| = |ω_truth|`, you cannot vary q0 with ω̂ fixed and expect to climb a smooth gradient toward truth — the LC is a joint function of both, and the basin in (q0, ω̂) space is generically narrow in BOTH coordinates.

## Why this matters for solver design

m103-era pipelines treated ω-direction and q0 as separable: grid + Nelder-Mead over ω-direction first, then phi-sweep over q0 anchors per ω-candidate. This works only when truth-near ω can be found independently of q0. Under correct truth, that assumption is questionable — m145 demonstrated that even with truth-near ω fed to m115's q-from-ω solver, the q0 search fails because the surrogate landscape at fixed truth-ω has a deceptive attractor at q0_err ≈ 135°.

**Implication for the survey:** when probing cost-at-truth, score at the FULL truth state `(q0_truth, ω_truth)`. When probing landscapes, vary BOTH q0 and ω̂ — varying only one risks measuring a 2-D slice through a basin that's narrow in the other dimension.

## Why |ω| is "only" the time-stretch axis

Increasing |ω| makes the satellite tumble faster: features in the LC come more frequently. The pattern of features is unchanged (same `(q0, ω̂)` orbit, just traversed faster). For inversion this means:
- |ω| is identifiable from peak spacing alone — a Lomb-Scargle-style frequency estimate often pins it within a few percent.
- Once |ω| is roughly correct, the residual `(q0, ω̂)` problem is the hard one.

This is why most pipelines treat |ω| as cheaply estimable (or even oracle-fed) and concentrate effort on `(q0, ω̂)`. The survey should keep this asymmetry in mind: a seed that is "constrained" in `|ω|` may still be wide open in `(q0, ω̂)`.

## What survey experiments should check

- When ranking candidates, always report all three errors: q0_err, ω_dir_err (in degrees), ω_mag_err (in % of truth). Hiding any one masks coupling.
- When diagnosing a cost surface failure, ask: is the failure in `q0` only? in ω̂ only? in both? An ω-only-correct candidate with random q0 is just as much a failure as the reverse, even though many parent-project pipelines log it as "ω-found, q0-pending."

## Cross-references

- Frozen-reference micro-experiment: `m114_3dof_basins.md` (parent wiki) — first quantification of joint basin geometry
- Auto-memory: `project_q_wdir_coupling.md`
- Survey concept: `omega_sign_degeneracy.md` (the structural degeneracy in ω-direction sign)
