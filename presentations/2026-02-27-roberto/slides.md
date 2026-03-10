---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 36px;
    color: #2d3436;
  }
  h2 {
    font-size: 30px;
    color: #2d3436;
  }
  em {
    color: #6c5ce7;
  }
  strong {
    color: #d63031;
  }
  table { font-size: 20px; }
---

# Testing the Peak-Anchored Graph Pipeline
### Weekly Update — 27 February 2026

Girish Narayanan

---

## This Week's Goal

Implement and stress-test the dynamics-constrained inversion pipeline on IS-901.

**Test case:** 500 observations, $\boldsymbol{\omega} = (0.5, -0.3, 2.0)°/\text{s}$, known geometry + inertia.

**Strategy:** Systematic experiments — probe each component, identify failure modes, refine the approach.

---

## Building Block Results

| Question | Finding |
|----------|---------|
| How selective are peaks? | ~0.1% of SO(3) matches brightness (80× more selective than generic epochs) |
| Candidate quality from sampling? | 1M → ~1,400/peak, nearest ~5.95° from truth. 10M → nearest 2.14°. Scales as $N^{-1/3}$. |
| Bridge feasibility (arrival mismatch)? | At full scale, L-BFGS-B connects 98.5% of pairs to <0.02° mismatch. **No discrimination.** |
| Single-peak optimisation? | L-BFGS-B basins far from truth regardless of seed count. Random sampling better for coverage. |

---

## Full Pipeline Result (Steps 1–5)

50 candidates × 3 peaks. L-BFGS-B bridge optimisation. Lo-fi intermediate brightness scoring. Graph search.

| Stage | Result |
|-------|--------|
| Candidate generation | 1,400–2,987 hits/peak from 1M samples |
| Bridge optimisation (arrival mismatch) | 98.5% feasible — 3 free $\boldsymbol{\omega}$ DOF connects nearly anything |
| Lo-fi intermediate scoring | Truth bridge indistinguishable from random |
| Graph search | **Truth ranked #11,979 / 121,326** |

The pipeline failed. But the failure pointed to something important.

---

## Root Cause #1: Lo-fi Can't Rank

Lo-fi brightness (no shadow modelling) has attitude-dependent errors larger than the discriminating signal.

**Test:** Propagate from epoch 0 with true $\boldsymbol{\omega}$, truth vs 2°-nudged:
- Truth vs observed: RMS = **0.049 mag** (noise floor)
- 2°-nudged vs observed: RMS = **0.218 mag** (4.4× noise)

Hi-fi clearly sees the difference. Lo-fi cannot.

→ **Intermediate scoring must use hi-fi.** Lo-fi is fine for coarse candidate filtering, but bridge ranking requires ray-traced brightness.

---

## Root Cause #2: The $\boldsymbol{\omega}$ Degeneracy

Examined the truth bridge (nearest-to-truth candidates at peaks 1 and 2):

| | True $\boldsymbol{\omega}$ | Optimised $\boldsymbol{\omega}$ |
|---|---|---|
| Value (°/s) | (0.50, −0.30, 2.00) | (0.05, 0.03, −0.32) |
| Magnitude | 2.08 °/s | 0.33 °/s |
| Arrival mismatch | — | 0.0007° ✓ |
| Intermediate trajectory | Correct | **Completely wrong** |

The optimizer found a **different $\boldsymbol{\omega}$** that arrives at the right attitude (0.0007° mismatch) but takes a completely different path. It's 8× too slow and points in a different direction.

**Arrival mismatch is degenerate** — many $\boldsymbol{\omega}$ values connect the same pair via different routes.

---

## What This Means for the Pipeline

The original pipeline (Steps 4–5) assumes arrival mismatch + rate priors can select the right path, with brightness fitting deferred to global refinement (Step 6).

**This doesn't work.** By the time you reach Step 6, the graph search has already selected a wrong path based on degenerate arrival costs.

The fix: **$\boldsymbol{\omega}$ must be optimised for brightness, not arrival mismatch.**

The bridge cost needs to include intermediate lightcurve fit — the only signal that distinguishes the true $\boldsymbol{\omega}$ from degenerate alternatives.

---

## Proposed Fix: Brightness-Aware Bridge Optimisation

**Current (Steps 4–5):**
$$\min_{\boldsymbol{\omega}} \| q(t_2; q_1, \boldsymbol{\omega}) - q_2 \|^2$$
→ Finds any $\boldsymbol{\omega}$ that arrives correctly. Degenerate.

**Proposed:**
$$\min_{\boldsymbol{\omega}} \sum_{k \in \text{intermediate}} \| y_k - L(q(t_k; q_1, \boldsymbol{\omega})) \|^2 + \lambda \| q(t_2; q_1, \boldsymbol{\omega}) - q_2 \|^2$$
→ $\boldsymbol{\omega}$ must produce a trajectory that matches brightness **along the way**, not just at the endpoints.

**Computational cost:** Each $\boldsymbol{\omega}$ evaluation requires Euler propagation + hi-fi brightness at ~10 intermediate epochs. At ~130ms/eval, each bridge optimisation ≈ 1.3s. With multi-start (5 initial $\boldsymbol{\omega}$) × 5,000 bridges ÷ 8 workers ≈ **70 min.** Expensive but feasible.

---

## Hi-fi Feasibility (Confirmed)

- **Cost per hi-fi evaluation:** ~130ms (ray tracing + shadow casting)
- **Fork-based multiprocessing works:** SPICE kernels + satellite geometry survive fork
- **Scaling:** 8 workers give near-linear speedup on large batches

The bottleneck is not whether hi-fi is possible — it's where in the pipeline to spend the compute.

---

## Summary & Next Steps

**Established this week:**
- ✅ Peaks are excellent anchor points (80× selectivity)
- ✅ Full Euler dynamics working for bridging
- ✅ Hi-fi brightness can discriminate 2° errors (4.4× noise)
- ✅ Hi-fi parallelisation confirmed feasible
- ❌ Arrival mismatch is degenerate — cannot select the right $\boldsymbol{\omega}$
- ❌ Lo-fi brightness lacks the fidelity to distinguish true from false bridges

**Identified key modification to Steps 4–5:**
Optimise $\boldsymbol{\omega}$ for intermediate brightness fit, not arrival mismatch. This breaks the degeneracy.

**Next:** Implement brightness-aware bridge optimisation and rerun the graph pipeline.

### AMOS 2026 — Abstract deadline: **2 March**
