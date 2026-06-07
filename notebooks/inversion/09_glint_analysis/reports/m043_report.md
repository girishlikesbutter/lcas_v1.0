# Micro-43: Focused Single-Normal PAB Seeding

## Question

If we identify the glinting normal from magnitude (using m039's rule-based
classifier), and concentrate ALL iso-brightness seeds on that one PAB circle,
how much better is candidate generation compared to m038's diluted 14-normal
PAB seeding and m010's random SO(3) seeding?

## Setup

- Epoch 183, observed magnitude = 7.17 (true = 7.14)
- Magnitude classifier predicts "z-faces" class (mag < 7.5)
- z-faces class maps to groups G6 [0,0,-1] and G7 [0,0,+1]
- Oracle dominant group: G7 [0,0,+1] with fractional flux = 1.0000
- Oracle IS in the predicted set (correct classification)

## Seed generation

| Method | Groups | Seeds per normal | Total seeds |
|--------|--------|-----------------|-------------|
| Predicted-PAB | G6 + G7 | 2160 | 4320 |
| Oracle-PAB | G7 only | 2160 | 2160 |
| Random SO(3) | N/A | N/A | 4320 |
| m038 (reference) | all 14 | 2160 | 30,240 |
| m010 (reference) | N/A | N/A | 10,000 |

## Results

| Method | Seeds | Converged | Unique | Nearest to truth | Time |
|--------|------:|----------:|-------:|-----------------:|-----:|
| Predicted-PAB (2 normals) | 4,320 | 3,267 | 1,993 | **12.44 deg** | 955s |
| Oracle-PAB (1 normal) | 2,160 | 1,632 | 1,020 | **20.16 deg** | 481s |
| Random SO(3) (this run) | 4,320 | 3,008 | 2,738 | **0.69 deg** | 1070s |
| m038 (14-normal PAB) | 30,240 | 21,305 | 10,860 | **5.00 deg** | 5433s |
| m010 (random SO(3), 10K) | 10,000 | 6,889 | 5,643 | **0.50 deg** | 1755s |

Note: m010 distance recomputed to truth at epoch 183 (the JSON reports
24.57 deg relative to q0, which is a different quaternion).

## Key findings

1. **Focused PAB seeding is WORSE than random SO(3)**, even when the correct
   normal is identified. The nearest focused-PAB candidate (12.44 deg for
   predicted, 20.16 deg for oracle) is far worse than random SO(3) at the
   same seed count (0.69 deg).

2. **The magnitude classifier works**: it correctly places the oracle group G7
   in the predicted class. The "z-faces" prediction is correct.

3. **Paradoxically, the oracle-only PAB (20.16 deg) is worse than the
   predicted 2-normal PAB (12.44 deg)**. This is because the predicted set
   includes G6 [0,0,-1] in addition to G7, and by chance some G6 seeds
   converge to closer candidates.

4. **PAB circle seeding explores a structurally constrained subspace of SO(3)**.
   The 1-DOF circle and its concentric rings sample a thin band around the
   PAB alignment direction. While this band is physically motivated (normals
   aligned with PAB produce glints), the iso-brightness manifold apparently
   has favorable basins in other parts of SO(3) that PAB seeding misses
   entirely.

5. **Convergence rates are comparable** across methods: ~70-76% for PAB,
   ~70% for random. So the seeds are not "bad" in the sense of failing to
   converge -- they converge to different (worse) basins.

6. **The 1st percentile tells the story**: Random SO(3) has 1st percentile
   at 21.6 deg, while focused PAB has 1st percentile at 68-68 deg. The best
   random candidates are dramatically closer to truth than the best PAB
   candidates.

## Interpretation

The hypothesis was that concentrating seeds on the correct PAB circle would
produce better candidates by avoiding dilution. The data decisively rejects
this hypothesis. The problem is not dilution across 14 normals but rather
that PAB-circle parameterization is too constrained. The iso-brightness
manifold near truth apparently requires seeds from diverse SO(3) regions,
not seeds concentrated near geometric alignment.

This suggests that the PAB alignment condition (normal aligned with the
phase-angle bisector) identifies a *necessary* condition for specular
glints, but the L-BFGS-B iso-brightness landscape has many local minima
that are NOT near PAB alignment. Random SO(3) seeding, by covering the
full rotation space, has a much higher probability of landing in a basin
near truth.

## Conclusion

Focused PAB seeding, even with correct normal identification, produces
worse candidates than random SO(3) seeding at the same seed count.
The PAB-circle approach should be abandoned in favor of random or
quasi-random SO(3) seeding for iso-brightness candidate generation.

## Output files

- Script: `notebooks/inversion/09_glint_analysis/m043_focused_pab_seeding.py`
- JSON: `data/results/inversion_diagnostics/m043_focused_pab_seeding.json`
- NPZ: `data/results/inversion_diagnostics/m043_focused_pab_seeding.npz`
- Plot: `data/results/inversion_diagnostics/m043_focused_pab_seeding.png`
- Log: `data/results/inversion_diagnostics/m043_stdout.txt`
- Runtime: 2599s (~43 min)
