---
title: "Attitude Level-Set Disconnection — components as competing hypotheses"
type: concept
sources:
  - "notebooks/inversion/score_kernel_consistency.py"
  - "data/results/inversion_diagnostics/kernel_consistency/seed_091/"
related:
  - "[[m136_kernel_consistency_failure]]"
  - "[[m118_cost_comparison]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[twin-degeneracy]]"
  - "[[alignment-cost]]"
  - "[[constraint-poor-regime]]"
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

# Attitude Level-Set Disconnection — components as competing hypotheses

The set of body attitudes that produce an observed brightness at a given epoch is **not connected** in SO(3). It is a disjoint union of small localised regions, each corresponding to a physically distinct configuration. This single geometric fact dictates the algorithmic shape of any cost that uses per-epoch attitude isoshells.

## Definitions

At epoch `t`, with known inertial sun/observer directions `k1_inertial(t)`, `k2_inertial(t)` (from SPICE), the forward operator at any candidate body attitude `q ∈ SO(3)` is

```
m(q; t) = surrogate( R(q) · k1_inertial(t),
                    R(q) · k2_inertial(t),
                    panel, dish, dist )
```

The **per-epoch attitude level set** is

```
L(t) = { q ∈ SO(3) : | m(q; t) − observed_mag(t) | < tol }
```

— a 3D thick tolerance tube around the (idealised, zero-tolerance) codimension-1 surface. With finite noise (σ ≈ 0.05 mag) plus surrogate residual, the tube has positive measure in SO(3).

## Why disconnected

Multiple physical mechanisms produce the same scalar brightness:

- **Different facets aligning to PAB.** A +X-panel specular glint, −X-panel glint, dish glint, and various face contributions can each yield identical observed brightness via distinct mechanisms. Each is a small region around its alignment q.
- **Different shadowing configurations.** Same net brightness from different sets of "lit / self-shadowed" face combinations. Hifi/surrogate shadows make these regions geometrically distinct.
- **Body-symmetry twins.** IS-901's ±X symmetry maps glint regions onto each other under 180° rotation about X. This guarantees at least two disconnected components related by the twin.
- **Diffuse-mix configurations.** Multiple partly-illuminated faces summing to the observed brightness — distinct attitude blobs from each mixing combinatorial choice.

[[m118_cost_comparison]] already encoded this on the lo-fi side: each epoch had a `loop_count` (number of connected components of the pab-contour level set). Many epochs in m118 had 2-4+ loops. With realistic phase angle and shadowing, the components multiply further (lo-fi merges what hi-fi separates).

## Components are competing physical hypotheses

Each component `C_i(t) ⊂ L(t)` is a *mutually-exclusive* theory about the satellite's configuration at time t. They are not "samples of the same locus" — they are alternative explanations for the same observed brightness.

**Therefore: no algorithm should pick one anchor q and propagate from it as if it were the right one.** Any cost shape that does an `argmin / min over q ∈ L(t)` collapses competing hypotheses into a single noisy choice, losing the structure that makes multi-epoch consistency informative.

## Implication: the cost shape that DOES respect this

Trajectory-level component-membership:

```
For each candidate (ω-direction, |ω|):
    propagate identity from t=0 forward/backward to each constraint epoch t
        → q_world(t)
    cost = Σ over constraint epochs t of  dist( q_world(t),  ⋃_i C_i(t) )
                                                              ^^^^^^^^^^
                                                  nearest component, no min over a chosen anchor
```

The candidate (q0, ω) trajectory either threads through the union of per-epoch components or it doesn't. No anchor enumeration, no min-collapse.

This naturally breaks the ±X twin: at glint epochs, the twin lands inside a component (it's a real physical configuration that produces that brightness); at mid-brightness epochs where the twin partner has different shadowing, it lands OUTSIDE every component. Summed over many epochs, the twin pays a cost the truth doesn't.

This is the proper formulation of the [[surrogate-attitude-isoshell]] branch's "intersect across epochs."

## What this rules out

- **`min over q_anchor ∈ Q_a` patterns** for large `|Q_a|` are doomed when Q_a samples across components (not just from a single component locus). [[m136_kernel_consistency_failure]] is the empirical demonstration: |Q_a|=54 + K=40 epochs → Spearman ρ(cost, ω-error) ≈ 0.
- **Treating L(t) as a sheet/blob/manifold to fit to.** It's a discrete component-set, conceptually closer to a set-valued constraint than to a smooth surface.
- **The pab-contour caching strategy.** That worked only because k1=k2=h made one shareable contour valid across all epochs. With realistic physics each epoch has its own L(t) with its own component count and locations.

## What still works

- **m118-style alignment costs with small per-epoch target sets** (2-4 IPL centroids per epoch). The "min" effect is small at this size, AND the centroids are physics-derived (one per component by construction) so component conflation is less severe. Just the physics of the targets is wrong (pab-contour offset), not the algorithmic shape.
- **m135 Finding 2-style unique-candidate re-ranking.** No anchor enumeration at all — each (q0, ω) in m103's lofi pool is its own row, scored independently against the full LC. Component disconnection is moot because no per-epoch anchor selection happens.

## Construction cost

Per-epoch L(t) build: 64k q-samples × surrogate forward × Pool(8) ≈ 5 sec / epoch. For ~100 constraint epochs per seed: 8-10 min Step-1 work / seed.

Connected-component clustering on a 64k-point set in SO(3) (use chord-distance proxy or quaternion DBSCAN with geodesic kernel): seconds per epoch.

This is the price of correct physics. The lo-fi pab-contour got it for free with broken physics; surrogate-attitude-isoshell must pay it but gets faithful per-epoch level sets in return.

## Cross-references

- [[m136_kernel_consistency_failure]] — empirical demonstration of the wrong shape (#dead-end this shape)
- [[m118_cost_comparison]] — `loop_count` was the lo-fi expression of this concept
- [[surrogate-attitude-isoshell]] — the proper successor branch (#open)
- [[twin-degeneracy]] — twin survives glint epochs, fails mid-brightness; component-membership is the discriminator
- [[constraint-poor-regime]] — when too few constraints exist, the per-epoch components don't intersect informatively
