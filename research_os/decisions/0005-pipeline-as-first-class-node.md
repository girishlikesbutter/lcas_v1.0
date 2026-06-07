# 0005 — The store is a typed graph; promote `Pipeline` to a first-class node

Status: **built** — 2026-06-04 (accepted same day; Girish: "definitely correct"; "All four, full migrate").
Glossary shipped first (b96dc12). **Implementation:** `schemas/pipeline.schema.json` (new node type) +
the three off-pipeline edge fields (`run_record.tests`, `run_record.blocked_by`,
`substrate_component.promoted_from`); `loop/validate.py` wires all six edge checks + pulls
`glossary_term` into the closed store (blocked-by targets); 4 pipeline nodes
(`research_os/pipelines/`); `backfill/migrate_adr0005.py` repointed/tagged 15 runs, deleted the 2
pure-method goals, split the 2 live ones; `render/live_head.py` gained the TRIED projection.
`hard-shoot-trap` coined as a glossary term so the blocked-by payoff edge has a real target.
Store stays referentially closed (230 objects). `supersedes` + `promoted_from` are schema-wired
but left null — no grounded assertion exists yet; populate when one does (don't fabricate).

The goal tree (trunk → chapter → branch → leaf) is only *partly* the right abstraction.
Walking the system as a general graph, the blocks fall out as distinct layers:

> **Goal** = why · **Pipeline** = how-we're-trying · **Tool** = what-we-can-do ·
> **Experiment** = what-we-did · **Claim** = what-we-learned (+ Contract, Term)

Five of these already exist in the store (Goal = `goal_node`, Tool = `substrate_component`,
Experiment = `run_record`, Claim = `claim_card`, Contract = `branch_contract`,
Term = `glossary_term`). **The one genuinely missing primitive is `Pipeline`** — a
*hypothesised approach* to a goal (densify, joint-grid-pivot, cross-cloud, single-wind-window
polish). Today these live in the store **as `goal_node`s**, which is the recurring friction:
they aren't questions, they're **methods**. Splitting `Pipeline` out from `Goal` is what
unlocks the rest, because Pipeline is the hub the *lateral* edges attach to.

**Why a graph, not a tree.** A tree shows decomposition (`Goal → decomposes-into → Goal`).
It **cannot** show the edges where the research value hides:
- `Pipeline → serves → Goal`, `Pipeline → composes → Tool`, `Experiment → tests → Pipeline`,
  `Experiment → extends → Experiment` (lineage), `Pipeline → supersedes → Pipeline` (rivalry),
  `Tool → promoted-from → Experiment` (operationalisation), `Experiment → blocked-by → Blocker`.

**The payoff** (what Girish asked the graph to give a researcher):
- **Tool reuse / transfer** — a Tool used across goal-subtrees (e.g. the Jacobi propagator);
  invisible in a tree.
- **Pipeline rivalry / supersession** — `joint-grid-pivot` [closed] vs `densify` [open] are
  rivals for the same goal; the supersession edge stops re-walking dead ends.
- **Shared failure mode** (the big one) — the hard-shoot trap and finite-diff ω-aliasing block
  *multiple* pipelines; tagging experiments with their blocker surfaces "this obstacle keeps
  killing different regimes" → that's the real lever (it's how s100/s105 found the trap was
  binding, not coverage).
- **Concept bridges** — a glossary term in multiple pipelines is a bridge between regimes.

The three views Girish wants are then **projections** of one graph: *what we've tried* =
Pipeline nodes + status; *where we're going* = open Goals + under-exploration Pipelines;
*useful connections* = the reuse / blocker / rivalry edges.

**Trunk corollary (settled this session):** the trunk is the **most-general ancestor** (for
roll-up + leverage ranking), **not the most-important node**. Re-rooting *upward* (insert a
more-general parent, e.g. a future "General LC Inversion" above the current trunk for
maneuver-detection) is always clean. Re-rooting *onto a mid-tree node* (making
`blind-fast-invert` the trunk) orphans its sibling chapters — that's why it stays a branch.

**Decision (2026-06-04): adopt.** Glossary shipped first (b96dc12); Pipeline-as-first-class-node
is the **next OS-build work item**. **Scope (the minimal version):** add a `Pipeline` node type +
the six edges above (serves / composes / tests / supersedes / promoted-from / blocked-by);
migrate the existing `goal_node`s that are actually *methods* (densify, joint-grid-pivot,
cross-cloud, single-wind-window polish) onto `Pipeline`. Everything else already exists.

**Cost acknowledged:** a ~204-object referentially-closed store + the "reflect, don't own"
minimalism (PLAN Q1) means real schema + reconcile + render work — the build must keep the
lateral graph *earning its keep* (reuse / rivalry / shared-blocker edges that a tree can't show),
not merely conceptually nicer. The three views (tried / going / connections) are projections of
the one graph.
