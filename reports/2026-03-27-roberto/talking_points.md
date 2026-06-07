# Talking Points — 27 March 2026

## Opening (30s)

- First working blind inversion pipeline
- Single observed light curve in, attitude + angular velocity out
- 5-7 minutes runtime, tested on 6 trajectories

## Pipeline walkthrough (2-3 min)

- Show the pseudocode listing — walk through each step
- Emphasise two key ideas:
  - **Delta-q factorisation**: propagation is independent of anchor attitude. One ODE solve per omega candidate serves all 360 phi values. This is what makes the grid search tractable (40k omegas in 100s).
  - **Specular = +-X**: every bright glint (mag < 6) is a +-X face aligning with the PAB. 100% confirmed across 6000+ peaks. This gives us a hard geometric constraint at each glint epoch — attitude lies on a 1-DOF circle.

- Point out the two-phase structure:
  - Steps 1-4: purely geometric, no forward model needed (~175s)
  - Step 5: full ray-traced hi-fi, only for the cluster of survivors (~150s)

## Cluster detection (1 min)

- Show Figure 1 (hifi_cluster_seed93.pdf)
- Left panel: geometric cost has a clear cluster (6 candidates) then a 75x gap
- Right panel: within the cluster, hi-fi residual separates correct omega (4x margin)
- This two-stage filtering is why we can afford the expensive hi-fi — we only run it on a few candidates

## The degeneracy (1-2 min)

- Show Figure 2 (lc_twin_comparison.pdf)
- IS-901 has mirror symmetry about the XY body plane — N and S panels are identical
- For any valid attitude, rotating phi by 180 deg swaps N/S but looks the same
- The two light curves are nearly identical — specular peaks match by construction
- Differences only appear in dim epochs where shadow patterns diverge
- **This is not a pipeline failure** — it's a real physical degeneracy of this satellite geometry
- omega is unaffected by this ambiguity (both phi values give the same omega)
- Resolving it requires exploiting the front/back asymmetry (different dish positions) — future work

## Multi-seed results (1 min)

- Show Table 1 — 4/6 succeed with omega < 2.2 deg
- The two failures both have only 8 constraint epochs (specular + bright)
- All successes have >= 10 constraints
- Seed 14 is interesting: only 3 specular glints but 11 bright peaks compensate
- Takeaway: pipeline needs ~10 constraint epochs to work. This is an observability threshold, not a parameter tuning issue.

## Questions to anticipate

**"What about slower tumblers?"**
- Our test set is 0.85-1.45 deg/s. Slower means fewer peaks per hour, fewer constraints. May need longer observation windows.

**"What about satellites without +-X symmetry?"**
- The specular constraint (mag < 6 = +-X) is specific to IS-901's box shape. For other geometries, the bright-peak constraint (any normal) would be the primary signal. Needs testing.

**"Can you resolve the 180 deg ambiguity?"**
- In principle yes — the front and back of IS-901 are not identical (different dish geometry). The hi-fi residual already resolves it in 2/6 cases. Need to investigate what makes those 2 cases different.

**"How does noise affect this?"**
- Current tests use sigma = 0.05 mag. Not yet tested at higher noise. The specular peaks are very bright (mag 5) so they should be robust, but the bright peaks (mag 6-9) are closer to the noise floor.

**"Runtime at scale?"**
- 5-7 min per trajectory on 16 cores. For a survey of 100 trajectories: ~10 hours. The grid search is embarrassingly parallel — could distribute across nodes.
