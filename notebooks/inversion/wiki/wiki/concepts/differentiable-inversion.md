> # ⚠️ PARTIAL RETRACTION 2026-04-15
> The conceptual argument (differentiable surrogate + cheap cost → gradient-based inversion) may still be worth considering on its own merits, but the **claim that [[m119_attitude_isoshell]] validated a useful cost function is FALSE** — m119 had a time-mismatch bug (see `notebooks/inversion/12_brightness_surface/m119_BUG.md`). The "cost function exists that puts truth at rank 0" motivation is therefore not established. Treat this page as a speculative horizon note, not a validated direction.

---
title: "Differentiable inversion — the modern reframe [RETRACTED MOTIVATION]"
type: concept
sources:
  - "raw/inversion_diagnostics/m119/seed_014/summary.json"
related:
  - "[[surrogate-model]]"
  - "[[m119_attitude_isoshell]]"
  - "[[surrogate-de-search]]"
  - "[[multi-solution-philosophy]]"
  - "[[gradient-based-inversion]]"
created: 2026-04-15
updated: 2026-04-15
confidence: medium
---

# Differentiable inversion — the modern reframe

## The realisation

The entire inversion pipeline we have built — grid search, Nelder-Mead, geodesic refinement, phi sweep, hi-fi rescoring — is machinery designed for the case where the forward model is **expensive and opaque** (a black-box ray tracer with Ashikhmin-Shirley BRDF). With an expensive black box, the classical playbook is:

- Minimise the number of forward-model calls
- Use grids and derivative-free optimisers (Nelder-Mead, DE)
- Pre-compute candidate pools and score them later
- Validate final candidates with the true model

That playbook was correct for its world. It is not correct for our world anymore.

## What changed

We now have **two** things that together invalidate the classical assumption:

1. **The surrogate is a neural network** — a differentiable function from `(k1_body, k2_body, panel, dish, dist)` to magnitude. Each evaluation is ~5 ms. See [[surrogate-model]].
2. **A cost function exists that empirically discriminates truth from wrong.** [[m119_attitude_isoshell]] showed that `mean_L1` of surrogate-residual over 255 constraint epochs places the truth trajectory at rank 0 out of 60,000 for seed 14.

Together these two facts mean the inversion problem is no longer black-box. It is a **cheap, differentiable cost landscape** on the 6-DOF parameter space `(q0, ω)` ∈ SO(3) × ℝ³.

## The modern playbook

For a cheap differentiable cost landscape, the textbook next step is **gradient-based optimisation or Bayesian sampling**:

1. **Define** `f(q0, ω) = mean_L1 of surrogate residual over constraint epochs`.
2. **Compute gradients** `∂f/∂(q0, ω)` by autograd — the surrogate is a numpy-MLP but it can be re-expressed in JAX/PyTorch or gradients can be computed analytically from the weight matrices.
3. **Optimise directly.** Choices include:
   - Gradient descent with Adam or L-BFGS from many random starts (enumerates basins)
   - Newton / Gauss-Newton if Hessian is available (quadratic convergence near minima)
   - Hamiltonian Monte Carlo on `p(q0, ω | observations) ∝ exp(-f/σ²)` — full Bayesian posterior
4. **Multi-start or MCMC gives basin enumeration for free**, aligned with [[multi-solution-philosophy]].

**No grids. No alignment cost. No phi sweep. No staged refinement. No hi-fi rescoring.** The cost function is the cost function; you optimise it.

## What this replaces

| Old stage | What it did | Modern replacement |
|-----------|-------------|---------------------|
| Alignment-cost grid | Find candidate omegas | Gradient starts from random (q0, ω) |
| Nelder-Mead | Local refinement w/o gradients | Gradient descent / L-BFGS |
| Geodesic refinement | L-BFGS-B on alignment cost | Same algorithm on the surrogate cost |
| Phi sweep | 1-DOF attitude search | Gradient directly in q0 |
| Hi-fi rescoring | Verify surrogate with ray tracer | Not needed (surrogate = cost) |

The current pipeline is a five-stage cascade with different cost functions at each stage, each cheap-but-approximate. The modern pipeline is one cost, one optimiser, multiple starts.

## Literatures to pull from

- **Differentiable inverse rendering** — computer graphics community, Mitsuba 3, NeRF-era work. Same mathematical structure: differentiable forward model + observations → optimise scene parameters.
- **Simulation-based inference** (SBI) — likelihood-free inference using trained neural simulators. Direct analogue.
- **Score-based generative models for inverse problems** — recent, uses diffusion model priors for under-determined inversions.
- **Hamiltonian Monte Carlo** — the classical Bayesian tool when likelihood is cheap and differentiable.

These are active research areas with mature tooling. We should not be reinventing them.

## Status

**Horizon direction.** Validated partially by [[m119_attitude_isoshell]] (cost function works for seed 14). Needs:
1. Multi-seed cost-function generalisation first (see [[surrogate-de-search]] + near-term m120 plan).
2. Then infrastructure work: port surrogate to JAX or PyTorch for autograd, or derive analytical gradients through the MLP.
3. Then a direct-optimisation POC to compare against the current multi-start DE (m115 pattern).

See [[gradient-based-inversion]] branch for the tracking node.

## Why this wasn't considered sooner

The surrogate was introduced in 2026-04-13. Work since then has used it as a drop-in replacement for the hi-fi ray tracer (same interface, faster evaluation). The differentiability was latent but unexploited; experiments treated the surrogate as a cheap scalar function, not as a gradient source.

[[m119_attitude_isoshell]] made the differentiability relevant by establishing that the surrogate-based cost function is useful — which is the precondition for "optimise it directly."

## Caveats

- **Twin symmetry is not removed** by this reframe. The cost is still bimodal (truth and twin score equally). Multi-start or MCMC will find both basins; the multi-solution philosophy still applies.
- **Surrogate fidelity is not infinite.** If the surrogate has training gaps (see [[surrogate-model]] dim-regime caveat, pending audit), gradients computed through those gaps may be misleading. Not a showstopper — just a reason to still hi-fi validate final candidates.
- **Differentiating through propagation.** The trajectory `R(t) = propagate(q0, ω, t)` involves an ODE solver. Differentiating requires either analytical gradients of the attitude propagator or treating it as a differentiable ODE (neural ODE-style adjoint). Manageable.
