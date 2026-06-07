---
name: align
description: "Alignment verb of the LCAS inversion loop — grills a branch of work into a FROZEN branch_contract JSON (research_os/contracts/) before any expensive compute. Specializes grill-me: one question at a time, recommend an answer, explore the store/code instead of asking. Fills question/hypothesis/uncertain-predictions/confirm+refute criteria/budget/artefacts, enforces the analytical-probe cheap-first gate (HARD block on expensive contracts), attaches the contract to its goal node, and validates the store stays closed. Use when the user says align, pre-register, freeze the plan, lock the branch, write/grill the contract, or bless a branch before running it. Pairs with strategize (picks the frontier item) upstream and execute (runs the frozen contract) downstream."
---

# Align — LCAS inversion loop (Research OS, branch forward_survey)

The **alignment** verb. Where `strategize` picks *what* to work on and `execute` *runs* it,
`align` freezes the **agreed terms** in between: it grills a branch of work into one
**`branch_contract` JSON** — a pre-registration that `execute` runs against and that can
only change via versioned amendments. This is human gate #2 of the four (PLAN §8): you
grill the contract, then it's frozen.

It specializes `grill-me`: **ask one question at a time, recommend an answer each time,
and explore the store/code instead of asking** when the answer is already on disk. The
difference from generic `grill-me` is the target — every question fills a field of the
`branch_contract` schema (`research_os/schemas/branch_contract.schema.json`), and the
session ends with a frozen object, not just shared understanding.

align is a high-reasoning skill (Opus, min thinking **xhigh** — PLAN §8). The *contract's*
own `model`/`reasoning` fields are what the **run** executes under (default `opus`/`high`,
override to `xhigh`/`max` for hard runs).

## Step 1 — Target the branch

Render the live head and locate the goal node this contract attaches to:
```bash
python research_os/render/live_head.py
```
- **Existing open/revivable branch** → read its node `research_os/goals/goal_<branch>.json`
  for the standing question, budget, and `last_measured`. The contract attaches here.
- **A genuinely new branch** → align is the place to open it. Mint a `node_kind: branch`
  goal node (or `question` for a single answerable question) under the right parent, e.g.:
  ```json
  {
    "schema_version": "1.0.0", "id": "goal_<kebab-slug>", "kind": "goal_node",
    "node_kind": "branch", "parent": "goal_<parent>",
    "title": "<the branch question>", "state": "open",
    "budget": { "expected_runs": 0, "expected_wall_s": 0 },
    "contract_refs": [], "is_trunk_artifact": false,
    "created_at": "YYYY-MM-DDT00:00:00Z"
  }
  ```
  Don't author `child_runs`/`spent` — they're derived (`refresh.py` owns them). Confirm the
  parent + title with the user before minting; a stray branch node is store litter.

## Step 2 — Grill to fill the contract (one question at a time)

Walk the schema fields, recommending an answer each time and reading the store/code rather
than asking when you can:

- **`question`** — the single uncertain question this branch answers.
- **`hypothesis`** — the standing hypothesis (what you expect, in one sentence).
- **`predictions[]`** — the load-bearing field. **Each names a specific UNCERTAIN
  seed/case + its expected outcome**; at least one must have `uncertain: true`. This is the
  experiment-discipline rule made mechanical: *if you can't name a seed whose outcome you
  genuinely don't know, the branch has no reason to run* (CLAUDE.md "State predictions
  before running"). Push back if every prediction is a foregone conclusion.
- **`confirm_criteria` / `refute_criteria`** — the agent/machine-checkable conditions that
  would CONFIRM vs REFUTE the hypothesis. Both required; they must be distinguishable and
  reference the predictions (e.g. "0 phantom Band A across ≥5 non-truth pairs" vs "≥1
  oracle-far pair reaches Band A").
- **`budget`** — `expected_runs` + `expected_wall_s` (seconds, compute only — never include
  writing/typing time, [[feedback_time_estimates_compute_only]]). Anchor it to the nearest
  analog run's actual wall.
- **`required_artefacts[]`** — what the run MUST produce (e.g. "ρ-band table over all tested
  pairs", "window-objective overlay plot"). `close` checks these exist; a visual artefact is
  a first-class validation gate ([[feedback_visual_artefacts_unlock_insight]]).
- **`model` / `reasoning`** — for the run: default `opus` / `high`; bump `reasoning` to
  `xhigh`/`max` if the run includes hard diagnosis.
- **`run_type_policy`** — `{cheap: auto, expensive: gated}` by default (cheap auto-runs;
  expensive/batch needs the human gate).
- **`probe_budget_s`** — the off-contract carve-out (default 180): probes up to this wall may
  run to gather evidence *before* escalating; anything longer stop-and-escalates first.

Units discipline while grilling: `|ω|` in **deg/s** ([[feedback_omega_mag_units_degrees]]),
wall in **seconds**, and the physical |ω| bracket [0.1, 1.5] deg/s if the contract sets one
([[feedback_omega_prior_physical_bracket]]).

## Step 3 — Analytical-probe gate (HARD block, cheap-first)

Before freezing a contract that authorizes **expensive** compute — operationally,
`budget.expected_wall_s > probe_budget_s` (i.e. not a sub-probe-budget probe) with
`run_type_policy.expensive: gated` — you MUST fill the `analytical_probe` object:
```json
"analytical_probe": {
  "cheap_path": "<the cheap analytical alternative considered — re-score existing seed_*/result.json, population stats on cached step1 checkpoints, a ranking-only change with no pipeline re-run>",
  "why_insufficient": "<why that cheap path cannot answer the question, justifying the compute spend>",
  "probe_run": null
}
```
This is the binding constraint of the whole system made executable (CLAUDE.md "Analytical
before computational"; PLAN §8). **Do not set `status: frozen` on an expensive contract
until `cheap_path` and `why_insufficient` are both genuinely answered** — not boilerplate.
If a 30-second re-score would answer the question, the right move is to run *that* (as a
`run_type: probe`, ≤ `probe_budget_s`), record it, and either drop the contract or set
`probe_run` to that probe's id once it shows the cheap path insufficient.

The PreToolUse `analytical-probe` hook (`ro_analytical_probe.sh`) is the mechanical backstop
at *batch-launch* time; this in-skill gate is the judgment gate at *freeze* time. For a
purely cheap/probe contract (wall ≤ `probe_budget_s`), `analytical_probe` is optional.

## Step 4 — Freeze the contract

Write `research_os/contracts/contract_<kebab-slug>.json` to the schema. `id` matches
`^contract_[a-z0-9]+(-[a-z0-9]+)*$` (kebab, lowercase). Flip `status: "draft" → "frozen"`,
set `frozen_at` to now, attach the id to the goal node's `contract_refs[]`, then gate:
```bash
python research_os/loop/validate.py
```
Exit 0 ("store is referentially closed") is required — it now checks the contract's
`goal_node` resolves and that the node's `contract_refs` resolve back. Fix the object you
just wrote if it errors (a dangling `goal_node`, an unattached `contract_refs`).

Once frozen, the original terms are **locked**. `execute` runs against this object; `close`
checks the `required_artefacts` exist and stamps the run record's `contract_ref`.

## Step 5 — Amendments (post-freeze, versioned only)

A frozen contract changes ONLY by appending to `amendments[]` — never edit a frozen term in
place (Q4: no silent goalpost moves). When the user wants to change scope/budget/criteria
mid-branch, re-grill just the delta and append:
```json
{ "version": "v1", "diff": "<human-readable change of terms>", "reason": "<why>",
  "approved_by": "girish", "at": "YYYY-MM-DDT00:00:00Z" }
```
Flip `status: "frozen" → "amended"`, bump `updated_at`, re-validate. The frozen original is
`v0`; amendments stack.

## What NOT to do

- **Don't freeze without an uncertain prediction.** A contract whose every prediction's
  outcome you already know does not justify a run — that's the discipline this gate exists
  to enforce, not a formality to fill.
- **Don't skip the analytical-probe gate on an expensive contract.** The cheap-first answer
  must be real. This is the single biggest lever against the rework that motivates the system.
- **Don't edit a frozen term in place.** Amendments are append-only and versioned.
- **Don't ask what's already on disk.** Read the goal node, the live-head, the nearest analog
  run record / writeup, and the code before asking the user — grill-me's core rule.
- **Don't author `child_runs`/`spent`** on a minted node — derived (`refresh.py` owns them).
- **Don't run the batch.** align freezes terms; `execute` runs them. Keep the seam clean.

## Where things live

- **Contracts (canonical):** `research_os/contracts/contract_<slug>.json` · schema +
  grammar `research_os/schemas/branch_contract.schema.json`.
- **Goal nodes:** `research_os/goals/goal_<branch>.json` (`contract_refs[]` points here).
- **Gate:** `research_os/loop/validate.py` (knows `branch_contract` + its 3 ref edges).
- **Upstream / downstream:** `strategize` (frontier pick) → **align** (freeze) → `execute`
  (run) → `close` (record + verify artefacts). Generic `grill-me` stays for non-contract
  stress-testing.
