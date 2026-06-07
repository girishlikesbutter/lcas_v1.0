---
title: "s044 — canonical(q0, ω) implementation + validation"
type: experiment
sources: [s043_twin_hifi_verify.md, concepts/twin_degeneracy.md]
related: [s042_basin_radius_cohort.md, s043_twin_hifi_verify.md]
created: 2026-05-06
updated: 2026-05-06
confidence: high
---

# TL;DR

Implemented `lib.twin.canonical(q0, ω)` plus a vectorised `canonical_batch`
following the s043 finding that the body-twin map `(q_180x ⊗ q0, R_180x · ω)`
is bit-exact in hi-fi (max |Δ|=2.29e-08 mag, ρ < 1e-6). All seven algebraic
checks pass with bit-exact output (idempotency, twin-pair-collapses,
involution, scalar↔batch, hemisphere membership, scipy cross-check, Q_180X²=−1).
Hi-fi LC equivalence on probe seed 89 between an arbitrary non-canonical
state and its canonicalised representative: max |Δ| = 1.68e-08 mag (ρ = 7.6e-08),
matching the s043 cohort signature. Canonicalisation is ready to apply at
the search stages.

# What

Translate s043's body-twin LC equivalence into a usable search-space-deduplication
primitive. Provide `canonical(q0, ω)` that maps any state to a unique
representative of its twin pair, and `canonical_batch(...)` for vectorised
use.

Canonical hemisphere convention:

```
keep iff  ω_y > 0
     or   (ω_y == 0 and ω_z > 0)
     or   (ω_y == 0 and ω_z == 0 and q_x < 0)
```

The third branch handles the measure-zero edge case where ω is along
body ±X (twin map fixes ω there, so the LC distinction lives in q0
alone). Quaternion sign is then normalised to `q[0] >= 0` so that
`canonical(canonical(x))` is bit-equal at the array level (not just
at the rotation level — `q_180x ⊗ q_180x = -1` means `twin(twin(q,ω)) = (-q, ω)`
without sign normalisation).

# How

`lib/twin.py` contains:

- `Q_180X`, `R_180X` — body-X 180° quaternion and rotation matrix.
- `quat_mul(q1, q2)` — Hamilton product, scalar-first.
- `twin(q0, ω)` — explicit twin map.
- `is_canonical(q0, ω)` — hemisphere membership test.
- `canonical(q0, ω)` — scalar canonicaliser (with sign-normalisation).
- `canonical_batch(q0_arr, omega_arr)` — vectorised over (N, 4) and (N, 3).
- `is_canonical_batch(...)` — batch hemisphere test.

`experiments/s044_canonical_validation.py` exercises seven algebraic
invariants on 10,000 random states each plus a hi-fi LC equivalence
smoke test on probe seed 89.

# Result

| Check | Outcome | Numbers |
|---|---|---|
| `Q_180X ⊗ Q_180X = -1`, `R_180X = R(Q_180X)` | PASS | err = 0.0 |
| `quat_mul` vs scipy `Rotation` product | PASS | max err = 3.6e-16 |
| `twin(twin(x))` = `(-q, ω)` | PASS | err = 0.0 (modulo q sign) |
| `canonical(canonical(x))` idempotency | PASS | err = 0.0 (bit-equal) |
| `canonical(twin(x))` = `canonical(x)` | PASS | err = 0.0 (bit-equal) |
| `is_canonical(canonical(x))` | PASS | 10000/10000 |
| scalar `canonical` vs `canonical_batch` | PASS | err = 0.0 (bit-equal) |
| Hi-fi LC: non-canonical vs canonical (seed 89) | PASS | max\|Δ\|=1.68e-08, ρ=7.6e-08 |

The hi-fi check matches s043's cohort signature (max |Δ|=2.29e-08 across
seeds 23/28/89). Canonical is operationally bit-exact under the post-fix
forward model.

# Why this matters

Until now the s042/s043 finding was an architectural insight without
infrastructure. With `lib.twin.canonical` available, every search stage
that enumerates (q0, ω) candidates can deduplicate via the canonical
hemisphere for a free ~2× compute reduction. The next experiment that
runs a cell-filter / hi-fi rerank / IC sampler should call this from
the start rather than retrofitting later.

Concrete application points (per wind-down 2026-05-06):

- ω-direction Fibonacci grid (s032-class cell filter): emit only canonical
  hemisphere → halve cell count.
- Sobol q0 sampler (s011-/s037b-class IC pool): canonicalise candidate
  pool post-hoc; for fixed canonical-ω cell, q0 sampling is unaffected,
  but candidates from a search that mixed both hemispheres should be
  deduplicated.
- Hi-fi rerank candidate set (s035-class): canonicalise + deduplicate
  before rendering — saves redundant hi-fi work on twin pairs.

# Numbers

Algebraic checks at N=10,000 random states (RNG seed 20260506):
all bit-exact. Hi-fi probe at seed 89, synthetic non-canonical state
`(q, ω)` with `ω_y < 0` enforced by construction:

```
q       = [-0.9075,  0.4017,  0.0779,  0.0780]
ω       = [-0.0072, -0.0058,  0.0014]  rad/s
canonical(q, ω) = ([-0.4017, -0.9075,  0.0780, -0.0779],
                    [-0.0072,  0.0058, -0.0014])  # twin applied + sign norm
```

Hi-fi render times: 8.5 s (orig) + 8.5 s (canonical), Pool=1, single
seed. Wall < 30 s including context build.

# Artefacts

- `lib/twin.py` — module
- `lib/__init__.py` — adds `twin` to `__all__`
- `experiments/s044_canonical_validation.py` — script
- `results/s044_canonical_validation/summary.json` — pass/fail per check
- `results/s044_canonical_validation/run.log` — full output

# Out of scope

- Modifying existing s032/s011/s035 scripts to use `canonical`. Those
  outputs are frozen artefacts; the next experiment chain (e.g., s039c
  hybrid pilot) should call `canonical` from the start rather than
  retrofitting the legacy runs.
- Hi-fi cohort-wide verification beyond s043's three seeds. The bit-
  exact result on seed 89 confirms the implementation; broader coverage
  is in the s043 study and not duplicated here.
- ω-magnitude axis behaviour. The twin map fixes |ω| (since R_180x is
  orthogonal); canonicalisation does not affect |ω|-magnitude bracketing.
  See s045 for the |ω|-aware bracket density projection.

# Cross-references

- Concept page: `concepts/twin_degeneracy.md`
- Hi-fi verification: `experiments/s043_twin_hifi_verify.md`
- Wind-down report: `report/wind_down_2026-05-06.tex`, §"The deduplication implementation"
