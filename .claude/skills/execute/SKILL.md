---
name: execute
description: "Execution verb of the LCAS inversion loop — takes a FROZEN branch_contract and runs it: adapts the experiment script, runs the in-session multiprocessing Pool batch on the 24-core box, fires the trust gates, scores against the contract's confirm/refute criteria, and writes one run_record JSON per executed run (research_os/records/) stamped with contract_ref + substrate versions. Runs ONLY the uncertain seeds the contract named; kills at 2× the budgeted wall; hands the session rollup + commit to close. Absorbs the old orchestrate. Use when the user says execute the contract, run the batch/experiment/inversion, kick off the run, dispatch the contracted run — NOT for launching an app (that is the built-in run skill)."
---

# Execute — LCAS inversion loop (Research OS, branch forward_survey)

The **execution** verb (spine: orient → strategize → align → **execute** → close). Where
`align` freezes the agreed terms, `execute` *runs* them: it reads a frozen
`branch_contract`, adapts the script, runs the **in-session Pool batch** (PLAN §6 keeps
headless/queued execution in Phase 2 — Phase 1 runs on the 24-core box, in this session),
gates + scores the result, and writes one **`run_record` JSON per executed run**. It
absorbs the old `orchestrate` (which fanned out Claude workers against `EXPERIMENTS.md` —
that structure is dead). `orchestrate` stays live until this lands, then retires.

execute is Opus min thinking **high**; the **diagnosis sub-step** (a seed misbehaves, a
batch overruns) bumps to **xhigh** or hands to `diagnose` (PLAN §8).

> This is NOT the built-in `run` skill (launch/drive the app). execute runs a *contracted
> inversion batch* and writes trust-store records. If the user means "start the app", that's
> the other one.

## Step 1 — Load + gate the contract

```bash
python research_os/render/live_head.py            # confirm the branch + its contract
```
Read `research_os/contracts/contract_<slug>.json`. Refuse to run unless:
- **`status` is `frozen` or `amended`** — a `draft` contract hasn't been blessed; send the
  user to `align` first. Never run un-pre-registered work as a batch (probes ≤
  `probe_budget_s` are the only off-contract carve-out).
- **`analytical_probe` is filled** for an expensive contract (`align` already hard-gated
  this; execute double-checks — defense in depth). If the cheap path wasn't ruled out, stop.

The contract's **`predictions[]` uncertain cases ARE the run list** — run only the seeds
whose outcome is genuinely uncertain (CLAUDE.md "Run only uncertain seeds"; if re-scoring
already settled 8/10, run the other 2). Note the in-force amendment `version` (the frozen
original is `v0`); it goes in the record's `contract_ref`.

## Step 2 — Adapt the script (don't rebuild)

Write/adapt `notebooks/inversion/survey/experiments/sXXX_<slug>.py` on the **existing
substrate** — `lib.forward.propagate_to_body_frame`, the s059e residual, the Jacobi
propagator — do not re-implement quat math or rebuild what exists
([[feedback_use_existing_lib_forward]]). **Checkpoint-design-first**: write the NPZ/JSON
save blocks (every candidate's `q0`, `ω`, errors, predicted hi-fi LC) *before* the compute
code ([[feedback_checkpoint_design_first]], [[feedback_save_hifi_lcs]]). The id continues
the sequence (`ls research_os/records/ experiments/ | grep -oE '^s[0-9]+' | sort -V | tail -1`,
increment), lowercase slug.

## Step 3 — Run the in-session Pool batch

The established execution pattern, all rules load-bearing:
- **`Pool(24)`**; set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` *before* the
  Pool ([[feedback_blas_threads_for_pool]]); `gc.collect()` between hi-fi renders or Pool(24)
  OOMs on the 30 GB box ([[feedback_pool_size_with_trimesh_gc]]).
- **No stacking** CPU-saturating jobs — one batch at a time ([[feedback_no_parallel_cpu]]).
- **Surrogate-first**: search/rank/cluster/LM-polish on the surrogate; hi-fi is reserved for
  a single render per polished winner to classify its ρ-band — never an optimisation step
  ([[feedback_surrogate_first_hifi_last]], [[feedback_lm_cost_use_surrogate]]).
- **Parallelism gate** — seeds run in parallel by default; serial-where-independent is a flag.
- **Kill at 2× `budget.expected_wall_s`** — diagnose immediately, don't sleep-wait
  ([[feedback_kill_stuck_early]]). Use idle compute for analysis, never `sleep && check`.
- Run in the background (`run_in_background`) so you can analyse while it runs; poll the
  checkpoints, not the clock.

Expensive/`gated` run-types (`run_type_policy.expensive`) need the human go before launch.

## Step 4 — Deviation discipline (the contract is frozen)

If mid-run the result demands something off-contract (a seed the contract didn't name, a
different objective, a bigger grid): **stop and escalate** — except a probe ≤
`probe_budget_s`, which may run first to gather evidence, *then* escalate. Scope changes are
`align` **amendments** (versioned, approved), never silent goalpost moves. A contract you
quietly outgrew is the rework this system exists to prevent.

## Step 5 — Gate + score

- **Gates** — conservation/smoke-at-truth (L_J2000 & 2T drift < 1e-12; render-at-truth →
  ρ≈0), oracle-leak honesty, N-scope, parallelism. The `gate-check` hook fires automatically
  when you write the record (Step 6) and HARD-blocks N-scope / oracle-incoherence (exit 2) —
  but state the truth in the fields, don't fight the gate.
- **Score** — `q0_err`, `w_dir_err` (deg), `w_mag` in **deg/s** ([[feedback_omega_mag_units_degrees]]),
  ρ = √MSE/0.05 vs truth hi-fi LC → band (A<2 · B 2–4 · C 4–8 · D ≥8). Then **check the
  contract**: does each prediction's outcome match? Do the `confirm_criteria` /
  `refute_criteria` fire? That mapping sets the record's `status`.

## Step 6 — Write the run_record(s) — the canonical output

One record per executed run → `research_os/records/sXXX_<slug>.json`, to the run_record
schema. execute is the primary writer now; the **full template + honesty rules live in the
`close` skill Step 1** — mirror them. The fields execute owns specifically:
- **`contract_ref`** `{ "contract": "contract_<slug>", "version": "v0|v1|..." }` — the frozen
  contract + amendment in force. (Probes with no contract: `null`.)
- **`status`** — `confirmed | refuted | inconclusive`, from the Step-5 criteria check.
- **`substrate_versions`** — copy the live heads (`live_head.py --json` → `trust.substrate_heads`).
- **`oracle_clean`** — `true` ONLY if no `truth_*` value was read. Oracle-nearest-pair seeding,
  q-held-at-truth, scoring-against-truth → `false`. Honesty beats a clean-looking flag.
- **`gates_passed` / `gates_failed`**, **`metrics`**, **`artefacts`** (the contract's
  `required_artefacts` must all appear), **`narrative_md`** (1–3 sentences: headline + number
  + consequence).

Then gate the store:
```bash
python research_os/loop/validate.py        # exit 0 required; contract_ref must resolve
```

## Step 7 — Hand to close

execute writes the per-run records + artefacts; it does **not** commit. `close` does the
session rollup: goal-node `last_measured`/`state`, claim-card drafts (→ your confirmation),
PROGRESS printout, the commit, and the handoff — and *verifies* the records execute wrote
(the contract's `required_artefacts` exist, gates honest). Run `close` when the session ends.

## What NOT to do

- **Don't run a `draft` contract**, or any expensive work without one. Pre-registration before
  compute is the whole point — send the user to `align`.
- **Don't run seeds whose outcome you already know.** Re-score the cached `result.json` /
  checkpoints first; run only the contract's uncertain cases.
- **Don't use hi-fi as an optimisation step.** Surrogate for everything; hi-fi only to ρ-band
  a polished winner.
- **Don't sleep-wait or stack CPU jobs.** Kill at 2× wall and diagnose; analyse on the live
  checkpoints while the batch runs.
- **Don't quietly go off-contract.** Stop-and-escalate; scope changes are `align` amendments.
- **Don't lie in `oracle_clean` / `N` / `claim_scope`** to dodge the gate — the gate exists to
  keep the store honest; a false `true` is exactly the rot it guards against.
- **Don't commit.** That's `close`'s job (size gate, explicit paths, no Co-Authored-By).

## Where things live

- **Contracts (input):** `research_os/contracts/contract_<slug>.json`.
- **Run records (output):** `research_os/records/sXXX_*.json` · schema
  `research_os/schemas/run_record.schema.json` · gate `research_os/loop/validate.py`.
- **Scripts + results:** `notebooks/inversion/survey/experiments/` · `results/` · `lib/`.
- **Upstream / downstream:** `strategize` → `align` (freeze) → **execute** (run + record) →
  `close` (rollup + commit). The `gate-check` / `analytical-probe` hooks fire around it.
