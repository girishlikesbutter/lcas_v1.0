# 0007 — The substrate is a laboratory; make Tools/Pipelines operational

Status: **accepted** — 2026-06-07 (Girish: "yes to both, scour first — write it all up").
Implementation **staged**: glossary terms + schema fields landed this session; the W-D scour
and the executor run-button are the next workstreams (see Consequences).

## Context — the laboratory reframe

Asked what felt *missing* from the Research OS, Girish's answer reframed the whole thing:
he doesn't want more features bolted onto the tracker — he wants a **laboratory**. *"Neatly
organised, reliably labelled boxes of things I can pick up, combine, test, log, analyse …
'run the IA Cloud block on seed 39' and KNOW one agreed thing is happening … results
visualised so I can decide … then take that block and decide if it needs to change. And if
there are 3 versions of the IA-Cloud generator, we can SEE that and fix it."*

The OS today reifies **outcomes** (`run_record`), **knowledge** (`claim_card`), and
**trajectory** (`goal_node`) well — but the **means of production** (the code blocks and the
data they act on) are only *described*, never *operational*:

- `substrate/` holds 3 hand-written *description* cards (propagator, scorer, surrogate) — prose
  interfaces and hand-typed line numbers. You cannot *run* one.
- The drift Girish wants to kill is already in the data: `scorer.json` records two functions as
  `"NOT YET PORTED — the working impl is filter_costs.py:196"`. The registry and the code have
  already diverged, and nothing forces reconciliation.
- The one block he most wants to run — the **IA-Cloud generator** — isn't even registered,
  though `IA Cloud` is canon glossary and pipelines reference it.

## Decision

### 1. Vocabulary (re-used, not invented)
Girish's three-level intuition — *base function → block → pipeline* — maps onto words the store
**already had** (ADR-0005's ontology: *Goal·Pipeline·Tool·Experiment·Claim·Term*). Coining a new
word ("component") would have created exactly the synonym-drift the glossary-lint exists to kill.
The mapping, now coined as canon glossary terms (`glossary/{tool,pipeline,primitive,artifact}.json`):

| Girish's word | Canon term | Store object |
|---|---|---|
| base function | **primitive** | code in `src/`/`lib/` (not individually carded) |
| **block** | **Tool** | `substrate_component` |
| **pipeline** | **Pipeline** | `pipeline` node |
| (the materials) | **artifact** | typed by a glossary term; instances staged later |

The distinction that justifies two words for "a composition of functions": a **Tool is a
capability** (value-neutral, no hypothesis); a **Pipeline is a bet** (has a `hypothesis`,
`serves` a Goal). "Caring to save & reuse a sequence" = the existing `Tool ←promoted-from←
Experiment` edge. The hierarchy is recursive ("all functions too"); the three are ONE
typed-operation concept in three ROLES, distinguished by lifecycle, not kind.

### 2. Tool anatomy (the runnable block)
A Tool binds to exactly **ONE entry-point at ONE version**; all variation lives in **params**,
never in copies. New optional `substrate_component` fields (this session):
`entry_point` (the `path.py:function` callable seam), `ports` (typed input/output in **artifact
types** — each a glossary_term id; the glossary doubles as the port type-system), `default_params`
(the parameterisation that collapses accidental copies into one Tool), `canon`, `variant_of`.

Worked example (the target the scour fills): `ia-cloud-generator` is `epoch-observation → IA Cloud`
— **per-epoch** (epoch is an input). "A cloud for the whole LC" is the *same* Tool **mapped over a
window** → a *set* of IA Clouds. So artifacts have **cardinality** (one-epoch vs set-over-window),
and a Pipeline has at least `map` (lift a per-epoch Tool across epochs) alongside `compose`.

### 3. Variant model (B)
One **canon** Tool per family (the default you get when you name it). **Meaningful** variants =
their own named Tool cards joined by a `variant_of` edge to the canon (mirrors Pipeline rivalry).
**Accidental** copies dissolve into `default_params`. The lab's job is to *surface the fork and
force the choice* — the same move `glossary-lint` makes for words.

### 4. Runnable architecture (Q1-safe)
Running a Tool does **not** violate Q1 ("reflect, don't own; code+git canonical; the store owns no
compute"):
- The Tool card **binds to canonical code** (`entry_point` + hash) — it *reflects* code, never
  replaces it. Verified-from-code, not hand-authored (which is why today's line numbers rot).
- A **drift-check / tool-lint** flags a Tool whose `entry_point` no longer resolves or whose hash
  moved, and **refuses to run** it — the mechanical "we can SEE there are 3 and fix it". Builds on
  the existing AST-grounded machinery map (`render/machinery_overlay.json`).
- **Execution is the executor (req #15), not the store.** "Run the IA Cloud Tool on seed 39" =
  enqueue intent → executor invokes the bound code → `run_record` + artifacts + plots flow *back*.
  Store owns definitions + records only.

### 5. Sequencing — scour first
1. **W-D scour** → an honest, visible, deduped *descriptive* registry (canon Tools + `variant_of`
   families + drift-check). Runnable-by-hand as today, but legible.
2. **Executor as a run-button** → makes the bench click-to-run; results auto-reflected. The
   keystone (#15) resurfaces here, but in its *smallest* form — a manual "run this one Tool on this
   one input," NOT an autonomous queue-draining daemon. The autonomy decision stays deferred.
3. **(later, Girish's call)** Pipelines as runnable DAGs (`compose` + `map`); artifact instances
   reified as addressable materials.

## Alternatives rejected
- **"component" as a new word for block** — synonym drift; the store already says Tool.
- **Variants in one card's `versions[]`** — `versions[]` is *temporal* (a fix superseding the old);
  it can't hold *parallel rivals* that both still exist.
- **Flat (every impl its own Tool, no canon)** — loses the "one agreed version" guarantee.
- **Code generated from the Tool card** — inverts Q1; too heavy.

## Consequences / Open
- **W-D — Tool/Pipeline backfill (the scour).** A W-B-sized, multi-agent audit of `src/`+`lib/`+
  `experiments/`: inventory every de-facto Tool & Pipeline, find the canon of each, classify the
  rest (accidental→params, meaningful→`variant_of`), register with bindings + ports. **Dozens+
  expected; run at SESSION-START for runway.** Plan: `backfill/W-D_tool_pipeline_scour.md`.
- **Schema landed:** `substrate_component` gains `entry_point/ports/default_params/canon/variant_of`
  (all optional — the 3 existing cards stay valid); `validate.py` now ref-checks `variant_of`→Tool
  and `ports`→glossary_term. Store re-validated closed.
- **Built since (2026-06-07):** the executor run-button (#15, `loop/run_tool.py`); the
  drift-check **tool-lint hook** — `loop/tool_lint.py --for-file` + `.claude/hooks/ro_tool_lint.sh`,
  wired into `settings.json` PostToolUse. Advisory (exit 0): it surfaces drift the moment an
  edit touches a Tool's bound code (the silent code-bump); the HARD refusal stays at execution
  in `run_tool.py`.
- **Built since (sequencing step 3a — Pipelines as runnable DAGs):** `pipeline.steps[]`
  schema (the `compose`+`map` DAG, optional — the 5 descriptive nodes stay valid); the
  `loop/run_pipeline.py` executor (per-step drift-check, `$in`/`$steps` ref-threading, both
  combinators, whole-DAG drift refusal); the canonical `pipeline_run` record
  (`schemas/pipeline_run.schema.json` + `validate.py` ref-checks + indexer feed); first
  runnable pipeline `pipeline_so3-pool-dedup` live-tested end-to-end.
- **Built since (sequencing step 3b — artifact instances, the shelf):** the
  `artifact_instance` primitive + `loop/artifacts.py` reify (Slice 1/1.5) — every port-typed
  output of a Tool run / pipeline step lands as an addressable, closure-counted material card +
  blob; and **re-feed (Slice 2):** `$artifact.<id>` / `{"$artifact": id}` pours a shelved
  material into a new run, type-checked against the consuming input port (keyed-map → exact,
  legacy-list → membership), wired into both executors. Live-tested: a standalone `twin_dedup`
  re-feeding shelved q-pool + ω-pool produced a result BIT-IDENTICAL to the pipeline's dedup step
  (re-feed == recompute); a type mismatch is REFUSED. The store is now the working pantry Girish
  asked for. Store closed @316. Plan: `plans/artifact-instances-and-lab-surface.md`.
- **Staged, not yet built:** the webapp surface (Slice 3 — `/pipelines/:id` run-detail +
  a `/materials` shelf; backend already loads pipeline_runs, boundary stays read-only/queue-only);
  then the Claude Design front-end beautify pass. Loose end: Slice 1.5b (3 SPICE kernel-FILE
  producers reify a written file, not a return value).
- **Re-prioritisation:** the laboratory gives the executor (#15) a concrete, non-scary first
  mission (a run-button), decoupled from the autonomy daemon.
