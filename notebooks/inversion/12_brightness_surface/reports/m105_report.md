# m105 -- Pairwise Peak Alignment Diagnostic

## Hypothesis

Requiring a candidate omega0 to produce body-frame PAB near the correct lobe normal at TWO peaks simultaneously (from different lobe families) produces a dramatically tighter constraint than single-peak alignment. The psi (twist) degree of freedom that makes single-peak alignment loose is broken when a second peak from a different lobe family is added, because the same q0 must satisfy both constraints.

## Target

Seeds 28 and 44 -- grid-failure seeds where the standard pipeline fails to locate the true omega0 direction.

## Design

### Grid
- 2000 Fibonacci sphere directions x 20 magnitude bins = 40,000 candidates
- Magnitude range: true |omega| +/- 30% (simulates regression estimate uncertainty)

### Peak pair selection
- Both peaks must be bright (mag < 9)
- Peaks from different lobe families (X vs Y, X vs Z, Y vs Z, X vs WD/ED, etc.)
- Time separation > 50 epochs (360+ seconds) to ensure distinct geometry
- Top 3 pairs selected by brightness (lowest mag sum)

### Alignment check
For each candidate omega0:
1. Propagate delta-q from identity quaternion at the two peak epochs
2. Rotate inertial PAB at each peak through delta-q to get "rotated PAB" (v1, v2)
3. Compute R_align that maps v1 -> n1 (lobe normal at peak 1)
4. Sweep 72 twist angles psi about n1 (5-degree steps)
5. For each psi, apply R_twist @ R_align to v2 and measure angular distance to n2
6. Candidate survives if min angular distance < 10 degrees

### Two modes
- **Oracle**: Uses the true lobe assignments from the trajectory data
- **Blind**: Tests all cross-family lobe pairs (up to ~72 combos), candidate survives if ANY combo works

### Intersection
A candidate must survive ALL 3 peak pairs. This is the key metric: does the intersection of 3 pairwise constraints reduce the search space enough to make the grid feasible?

## Expected outcomes

### Seed 28
Bright peaks span Z, Y, WD/ED families. Expect:
- Oracle single-pair reduction: 5-20x per pair
- Oracle 3-pair intersection: 50-200x (leaving ~200-800 survivors from 40,000)
- Blind mode: 3-5x worse than oracle due to lobe ambiguity
- Truth should survive all pairs

### Seed 44
Has X, Y, Z, WD/ED peaks (good diversity). Expect:
- Similar or better reduction than seed 28
- The +X peak at epoch 209 (mag 5.0) is exceptionally bright and should anchor strongly

### Key question
If the intersection reduces 40,000 candidates to O(100-500), that is sufficient to feed into NM refinement. If it only reduces to O(5000+), the pairwise constraint is interesting but insufficient alone.

## Script

`notebooks/inversion/12_brightness_surface/m105_pairwise_ipl.py`

Run: `python3 notebooks/inversion/12_brightness_surface/m105_pairwise_ipl.py`

Output: `data/results/inversion_diagnostics/m105_pairwise/results.json`

## What to look for in results

1. **Truth survival**: Does the nearest-to-truth grid candidate survive all pairs in both oracle and blind modes?
2. **Reduction factor**: The ratio 40000 / n_survivors. Target: > 50x per pair, > 200x intersection.
3. **Oracle vs blind gap**: How much does lobe ambiguity cost? If < 2x, blind mode is viable.
4. **Truth rank**: Among oracle survivors, where does truth rank by peak-2 angular distance? Lower is better.
5. **Grid coverage**: Is truth within ~2 deg of the nearest grid direction? If not, grid is too coarse.
