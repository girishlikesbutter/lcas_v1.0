# Research OS — canonical schemas (Phase 0)

The five canonical objects of the trust machine, plus one supporting registry.
These are the **source of truth** the dashboard reflects (PLAN Q1: *reflect, not
own*). Everything here is a plain JSON **file**; the web app indexes the files into a
disposable, rebuildable cache. Delete the cache → lose nothing. Delete a file → lose
research state.

> Read `research_os/PLAN.md` first — it is the design source of truth. This README
> is the *implementation contract* for the objects PLAN §2 specifies.

---

## The objects

| File | Object | Mutability | One line |
|---|---|---|---|
| `run_record.schema.json` | Run record | **immutable, append-only** | One executed run; re-run ⇒ new record citing the old in `parents`. |
| `claim_card.schema.json` | Claim card | status/edges only | A unit of knowledge state. "What's true now" = `status == live`. |
| `goal_node.schema.json` | Goal node | mutable (state/spent) | One node of the goal tree (the *why*); runs attach here, budgets live here. |
| `pipeline.schema.json` | Pipeline node | mutable (state/edges) | **(ADR-0005)** A hypothesised *approach* (the *how-we're-trying*); `serves` goals, `composes` tools, runs `test` it. **(ADR-0007 §5.3)** an optional `steps[]` DAG makes it *runnable* (`compose`+`map`). |
| `glossary_term.schema.json` | Glossary term | mutable | Canon (or provisional) vocabulary; the lint enforces it. |
| `branch_contract.schema.json` | Branch contract | **frozen + versioned amendments** | Pre-registered terms for a branch; no silent goalpost moves. |
| `substrate_component.schema.json` | Substrate component | append-only `versions` | **(ADR-0007)** the **Tool** card — version registry + runnable binding (`entry_point`/`ports`/`default_params`) that makes blast-radius computable and the run-button possible. |
| `tool_run.schema.json` | Tool run | **immutable, append-only** | **(ADR-0007)** one invocation of one Tool by the run-button; value-neutral bench provenance (no hypothesis). |
| `pipeline_run.schema.json` | Pipeline run | **immutable, append-only** | **(ADR-0007 §5.3)** one execution of a Pipeline's `steps[]` DAG by `run_pipeline.py`; per-step drift/version stamp. |

All instances validate against JSON Schema **draft 2020-12**. Top-level
`additionalProperties` is `false` everywhere (typos fail fast); `metrics` on the run
record is the one deliberately-open bag.

---

## On-disk layout

```
research_os/
  PLAN.md                       # design source of truth
  schemas/
    *.schema.json               # the object schemas (this dir)
    README.md                   # this file
    examples/                   # worked instances, drawn from the real corpus
  records/        run_record   ⇒ records/s106.json
  claims/
    research/     claim_card   ⇒ claims/research/claim_<slug>.json   (deck=research)
    preferences/  claim_card   ⇒ claims/preferences/claim_<slug>.json (deck=preference)
  goals/          goal_node    ⇒ goals/goal_<slug>.json   (tree rebuilt from parent pointers)
  glossary/       glossary_term⇒ glossary/<term-slug>.json
  contracts/      branch_contract ⇒ contracts/contract_<slug>.json
  substrate/      substrate_component ⇒ substrate/<id>.json   (Tool cards, ADR-0007)
  pipelines/      pipeline     ⇒ pipelines/pipeline_<slug>.json
  tool_runs/      tool_run     ⇒ tool_runs/tr_<tool>_<stamp>.json   (canonical bench records)
  pipeline_runs/  pipeline_run ⇒ pipeline_runs/pr_<slug>_<stamp>.json (canonical DAG records)
```

One file per instance (git-friendly; node trees rebuild from `parent` pointers rather
than a monolithic tree file that would merge-conflict). The `records/`, `claims/`,
… directories are created by Phase 1 when the loop starts writing real objects; Phase
0 ships the schemas + examples only.

---

## ID & slug conventions

| Object | Pattern | Example |
|---|---|---|
| Run record | `s<NNN>[letter][.suffix]` — continues the experiment sequence | `s106`, `s073c`, `s116.de` |
| Claim card | `claim_<kebab-slug>` | `claim_windowed-photometry-breaks-hard-shoot-trap` |
| Goal node | `goal_<kebab-slug>` | `goal_attitude-inversion` |
| Glossary term | `<kebab-slug>` (also the filename) | `body-twin` |
| Branch contract | `contract_<kebab-slug>` | `contract_windowed-polish-discrimination` |
| Substrate component | `<terse_snake_id>` (LHS of `component@version`) | `propagator`, `surrogate` |
| Tool run | `tr_<tool>_<stamp>` | `tr_sample_so3_pool_20260607t174822` |
| Pipeline run | `pr_<pipeline-slug>_<stamp>` (hyphens → underscores) | `pr_so3_pool_dedup_20260607t193018` |

IDs are **stable forever** — they are the join keys. Never reuse or renumber.

---

## Substrate-ref grammar & blast-radius

The string `component@version` is the load-bearing edge (PLAN Q6/Q8).

```
substrate-ref := <substrate_component.id> "@" <version>
   e.g.  propagator@2.0.0   surrogate@v2   scorer@1.0.0
```

- **run_record.substrate_versions** — an object map `{component_id: version}` stamped
  at run time. "What did this run actually run against?"
- **claim_card.depends_on** — an array of `component@version` strings. "What does this
  claim's truth rest on?"
- **substrate_component.versions** — the append-only history; `current_version` is the
  head.

**Blast-radius (the one-line cure for the s001–s066 class).** When a component gains a
new `versions[]` entry (a bump), the `blast-radius` hook runs two queries:

1. every `run_record` whose `substrate_versions[id]` ≠ `current_version`  → in radius;
2. every `claim_card` whose `depends_on` contains `id@<any-older-version>` → set
   `status = needs_replication`.

A bug-fix bump (`is_bug_fix: true`) is the dangerous kind — it retroactively
invalidates evidence. The propagator example encodes exactly the ω-sign fix that cost
s001–s066; under this model that cost becomes one query.

> Discipline check: if a component's source file changes but `current_version` does
> not (the `current_hash` no longer matches), that is a *silent* change — the
> `gauge-sentinel` / promotion gate flags it as an undocumented bump.

---

## Controlled vocabularies

| Field | Values |
|---|---|
| `run_record.status` | `confirmed` · `refuted` · `inconclusive` |
| `run_record.run_type` | `probe` · `cheap` · `expensive` · `batch` |
| `run_record.claim_scope` | `N1` · `cohort` |
| `claim_card.deck` | `research` · `preference` |
| `claim_card.status` | `draft` · `live` · `needs_replication` · `superseded` · `retracted` |
| `claim_card.trust_stamp.confidence` | `low` · `medium` · `high` |
| `*.scope.kind` | `N1` · `cohort` |
| `goal_node.node_kind` | `thesis` · `chapter` · `branch` · `question` |
| `goal_node.state` | `open` · `blocked` · `closed` · `revivable` |
| `glossary_term.status` | `provisional` · `canon` |
| `branch_contract.status` | `draft` · `frozen` · `amended` · `closed` |
| `branch_contract.reasoning` | `high` · `xhigh` · `max` |
| `run_type_policy.{cheap,expensive}` | `auto` · `gated` |
| gate ids (`gates_passed`) | `oracle-leak` · `n-scope` · `validator-not-production` · `conservation-smoke` · `parallelism-check` · `glossary-clean` · `no-duplicate-symbol` · `deep-module-review` · `visible-test` |
| `artefacts[].kind` | `checkpoint` · `plot` · `anim` · `data` · `report` · `log` |
| `metrics.rho_band` (convention) | `A` · `B` · `C` · `D` |

Units convention: **`|ω|` is always deg/s** (working-preference card
`claim_omega-reported-in-degps`), wall in **seconds**, angular errors in **degrees**.

---

## Referential integrity (loader / linter checks)

The dashboard loader (and a Phase-0 `validate.py`) should reject a store that breaks:

- `run_record.goal_node` → resolves to a `goal_node`.
- `run_record.parents[]`, `claim_card.{supporting,refuting}_runs[]`,
  `goal_node.child_runs[]` → resolve to `run_record`s.
- `run_record.contract_ref.contract`, `goal_node.contract_refs[]` → `branch_contract`.
- `branch_contract.goal_node`, `goal_node.parent` → `goal_node` (parent null ⇔ thesis).
- `claim_card.{superseded_by,supersedes}` → `claim_card`.
- **(ADR-0005)** `pipeline.serves[]` → `goal_node`; `pipeline.composes[]` →
  `substrate_component`; `pipeline.{supersedes,superseded_by}` → `pipeline`
  (`state == superseded` ⟹ `superseded_by` set).
- **(ADR-0005)** `run_record.tests[]` → `pipeline`; `run_record.blocked_by[]` →
  `glossary_term` (the blocker is modelled as a named term, e.g. `finite-diff-omega-aliasing`).
- **(ADR-0005)** `substrate_component.promoted_from` → `run_record` (when set).
- **(ADR-0007)** `substrate_component.{variant_of}` → `substrate_component`;
  `substrate_component.ports.{input,output}[]` → `glossary_term` (the port type-system).
- **(ADR-0007)** `tool_run.tool` → `substrate_component` (and `tool_version` ∈ its `versions[]`).
- **(ADR-0007 §5.3)** `pipeline.steps[].tool` → `substrate_component`; when `steps[]` is
  present, `composes` must equal the set of step tools; step ids unique.
  `pipeline_run.pipeline` → `pipeline`; `pipeline_run.steps[].tool` → `substrate_component`.
- every key of `run_record.substrate_versions` and every LHS of
  `claim_card.depends_on` → a `substrate_component.id`, and the version exists in its
  `versions[]`.
- exactly one `goal_node` with `node_kind == thesis` and `parent == null`.
- `claim_card.status == superseded` ⟹ `superseded_by` is set.
- `branch_contract.predictions` has ≥1 entry with `uncertain == true`.

> The example instances under `examples/` are illustrative *fragments* — they do not
> form a referentially-closed graph (e.g. `goal_windowed-photometry-polish.parent`
> points at `goal_blind-fast-inversion`, which is not shipped as an example). They
> each validate against their schema; cross-object resolution is checked only on the
> real store.

---

## Validation

```bash
# one-off: validate every example against its schema (Python jsonschema)
python research_os/schemas/validate_examples.py
```

`validate_examples.py` maps each `examples/<schema-stem>.*.json` to
`<schema-stem>.schema.json` and asserts it validates. This is the Phase-0 "visible
test" for the schemas themselves. (The richer cross-object referential checks above
land with the Phase-1 loader.)

---

## Deviations from PLAN §2 (flagged — veto either)

Two places extend the plan. Both **implement already-decided behavior** rather than
reopen a decision; flagged here per the "flag explicitly before extending" rule.

1. **`substrate_component` is a 6th schema** (PLAN §2 names five). PLAN §3 (Q6)
   requires every shared function to "carry a version/hash and live behind a
   versioned, gated interface." Without a registry the `component@version` refs on run
   records and claim cards dangle and blast-radius can't resolve. This schema *is* that
   registry. → If you'd rather fold the registry into config/code rather than a
   canonical object, say so and I'll move it.

2. **`claim_card.status` adds `draft` and `retracted`** to PLAN §2's
   `live | superseded | needs_replication`. `draft` is required by Q8/§8 ("claim
   writes are auto-draft → human-confirm"); `retracted` is the terminal state for a
   card withdrawn on discovery of taint (the s058/s059 retraction had no home in the
   three named states). The three named states are unchanged. → If you want
   auto-draft modeled as a separate boolean instead of a status, say so.

A third, smaller choice (not a deviation, just a call): the two claim **decks**
(research / preference, PLAN §2 "two separate stores") are physically split by
subdirectory (`claims/research/`, `claims/preferences/`) and logically by the `deck`
field. Flat-dir + field would also work; subdirs honor "separate stores" on disk.

---

## Deliberately deferred (not Phase 0)

- **Meetings & literature decks** (PLAN §2 Q10) → Phase 3. They are deck-like but
  carry their own fields (action-items + deadlines; gap/method/novelty/risk). Separate
  schemas when External Decks lands; the existing Robinson & Frueh 2025 card is the
  first literature instance.
- **Thresholds** — N-scope minimum, budget-breach factor (PLAN §7.2) → live in config,
  not the schema, so they tune without a schema bump.
- **Queue & cache location** (PLAN §7.3) → Phase 1+. The schemas describe the
  canonical files only; the queue carries *intent*, not state.
- **`gate-check` / `blast-radius` / `glossary-lint` hook implementations** → Phase 0
  "cheap epistemic gates" / Phase 1. This dir defines the *data* they read and write.
```
