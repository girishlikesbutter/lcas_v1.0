# m115 -- Surrogate multi-start pipeline replacement

## Hypothesis
Replacing the 1-DOF phi sweep with 10-start surrogate 3-DOF DE per omega
candidate will find valid solutions (hi-fi MSE < 1.0) for at least 7/10
baseline seeds (currently 2 OK + 4 PARTIAL in m102).

## Method
For each of 10 m102 baseline seeds (0, 6, 12, 14, 24, 27, 33, 36, 74, 93):
1. Load omega candidates from m103 geo_ckpt (3 best) or m102 result.npz (1)
2. Per omega: precompute delta_qs, run 10-start surrogate 3-DOF DE
3. Cluster solutions (>10 deg geodesic = separate basin)
4. Hi-fi validate top 3 basins
5. Classify: OK/PARTIAL/FAIL + has_valid_solution (hi-fi MSE < 1.0)

## m102 baseline

| Seed | m102 class | q0_err | w_dir_err | omega source |
|------|---------------|--------|-----------|--------------|
| 0    | OK            | 8.70   | 3.12      | geo_ckpt (3) |
| 6    | PARTIAL       | 176.85 | 3.08      | result_npz (1) |
| 12   | FAIL          | 169.60 | 8.01      | result_npz (1) |
| 14   | FAIL          | 90.72  | 56.27     | geo_ckpt (3) |
| 24   | FAIL          | 152.29 | 34.91     | geo_ckpt (3) |
| 27   | PARTIAL       | 175.39 | 38.38     | geo_ckpt (3) |
| 33   | FAIL          | 134.66 | 18.46     | result_npz (1) |
| 36   | PARTIAL       | 177.32 | 10.73     | result_npz (1) |
| 74   | OK            | 6.70   | 4.88      | result_npz (1) |
| 93   | PARTIAL       | 178.43 | 0.49      | result_npz (1) |

Seeds 0, 14, 24, 27 have m103 geo_ckpt with 26 candidates each (3 best selected).
Seeds 6, 12, 33, 36, 74, 93 have only m102 single winner omega.

## Results

### Population summary

| Seed | m102 | m115 | valid | hifi_MSE | q0_err | surr_MSE | improved |
|------|----------|----------|-------|----------|--------|----------|----------|
| 0    | OK       |          |       |          |        |          |          |
| 6    | PARTIAL  |          |       |          |        |          |          |
| 12   | FAIL     |          |       |          |        |          |          |
| 14   | FAIL     |          |       |          |        |          |          |
| 24   | FAIL     |          |       |          |        |          |          |
| 27   | PARTIAL  |          |       |          |        |          |          |
| 33   | FAIL     |          |       |          |        |          |          |
| 36   | PARTIAL  |          |       |          |        |          |          |
| 74   | OK       |          |       |          |        |          |          |
| 93   | PARTIAL  |          |       |          |        |          |          |

m115 classification: ? OK / ? PARTIAL / ? FAIL
Valid solutions: ?/10
Improved vs m102: ?/10

### Surrogate vs hi-fi tracking

| Seed | surr_MSE | hifi_MSE | diff  | ranking agrees? |
|------|----------|----------|-------|-----------------|
|      |          |          |       |                 |

### Timing

- DE per seed (3 omegas x 10 starts): ~? min
- DE per seed (1 omega x 10 starts):  ~? min
- Hi-fi per basin: ~50s
- Total wall time: ~? min

## Findings

1. **Hypothesis confirmed/refuted:** ?/10 seeds have valid solutions vs 7 target.
2. **Omega bottleneck:** Seeds with geo_ckpt (3 candidates) vs result_npz (1) -- which did better?
3. **Surrogate fidelity:** surr_MSE vs hifi_MSE correlation, ranking agreement.
4. **Basin enumeration:** How many distinct basins per seed? Twin degeneracy prevalence?
5. **Improvement over m102:** Which specific seeds flipped from FAIL -> OK/PARTIAL?

## Next steps

- If >= 7/10 valid: expand to full 100-seed population
- If < 7/10 valid: diagnose failure modes (omega error? surrogate error? insufficient starts?)
- If surrogate ranking disagrees with hi-fi: increase hi-fi validation budget
