# Research OS — Founding Plan

> A PhD research operating system. The surface is a dashboard; the engine is a
> trust machine. Built from a grill-me session on 2026-06-01.

---

## 0. Vision & reframe

This is **not** an inversion-experiment dashboard. It is a **PhD research OS** whose
job is to maximise **trustworthy conclusions per unit of compute**, and to keep the
human operating at the **strategy layer** while execution runs reliably and
autonomously underneath.

The human (Girish) thinks strategy and sets direction. The system captures that
direction as explicit, agreed terms (grill-me-style alignment), executes it
reliably, records results as structured data, and **at every moment keeps the trunk
goal and current trajectory visible** so neither human nor agent drifts.

### The binding constraint (decided)

Measured from the existing corpus, the dominant cost is **rework, not research**:

- **25%** of all log ingests carry reversal/correction language (38 / 152 entries).
- **53 of 100** experiments record a dead/refuted/broken result.
- **15%** of the knowledge base is corrections of itself (13 / 86 memory files).
- Three **foundation bugs** each silently poisoned dozens of downstream experiments
  before being caught: the ω-sign convention (invalidated s001–s066), the
  `times[0]==0` gauge bug (reversed s093/s094/s095 — it *manufactured* a wrong
  conclusion that looked solid enough to chase), and oracle injection (made the
  whole s058/s059 "PROVEN" architecture structurally false).

**Decision:** the system attacks **trust / foundation rot first.** The "we tangent
off the main thread" feeling is largely a *symptom* of this: a rotten foundation
produces false conclusions that look worth chasing for many sessions.

---

## 1. Core decisions (the agreed terms)

| # | Decision | Choice |
|---|----------|--------|
| Q1 | Source of truth | **Reflect, not own.** Text + git stay canonical; the dashboard is a derived read-model. Any DB is a disposable, rebuildable cache. |
| Q2 | Atomic unit | **Structured run record primary**; prose lives in a `narrative_md` field; the agent authors it, not the human. |
| Q3 | Trajectory | **Goal tree** (runs attach to nodes) + **drift governor** with per-branch budgets; budget breach → **auto-strategy-review** (a high-reasoning agent re-reads trunk+branch+last-N runs and hands a close/continue/pivot recommendation). |
| Q4 | Alignment | **Branch-level frozen contracts** (pre-registration: question, hypothesis, named uncertain predictions, confirm/refute criteria, budget, required artefacts, model/reasoning). Cheap run-types auto, expensive/batch gated. **Versioned amendments only** — no silent goalpost moves. Deviation rule: **stop-and-escalate before any off-contract action, except probes ≲3 min** (gather evidence first, then escalate). |
| Q5 | Execution | **Headless Claude Code** executor (reads the contract, adapts the script, runs the batch, scores, writes the record); **queued/async** driver (serialises CPU-heavy batches per the no-stacking rule); Workflow tool used *inside* a run for deterministic fan-out. Model/reasoning = **skill-default + contract-override**. |
| Q6 | Trust model | **Machine-recomputable trust stamp**, not human prose. **Substrate versioning + automatic blast-radius.** Gate suite (below). |
| Q7 | Code discipline | **Two-tier code: scratch vs substrate, with a gated promotion seam.** Glossary is two-tier the same way (provisional → canon) with a promotion gate + linter + **semantic synonym-block**. Enforcement = **mechanical hooks + reviewer agent + human sign-off**. **Parallelism gate = enforced-with-override.** |
| Q8 | Knowledge state | Findings are **first-class claim cards** (statement, supporting/refuting runs, trust stamp, scope N, **`depends_on: substrate@version`**, status, superseded_by). Prose narrative generated from cards. **Two separate decks:** research claims vs working-preferences. |
| Q9 | Promising/next | A **standing strategist pass** (proactive, session-start + on-demand) producing a **ranked frontier** with cited rationale; human sets the ranking lens per phase. |
| Q10 | External intent | **Unified — meetings & literature are more decks on the same goal tree + strategist.** The **gap / thesis contribution is a first-class trunk artifact** papers can threaten and meetings can steer. Meeting transcripts **auto-extracted into cards + proposed frontier reweights → human approval.** |
| Q11 | Surface | **Web app with an embedded terminal pane** (two views on one OS; focused work stays in-terminal). **Telegram = lite web app** (same invocable skill set, async). Dashboard is **read-model + control-plane only** (writes *intent* to a queue; owns no research state). |
| Q12 | Build order / migration | Phased 0→4 (engine before surface). **Lossy-pragmatic backfill** of the existing corpus into run records + claim cards + goal-tree skeleton, each linked to its original writeup. |

---

## 2. The object model

Everything is a file (Q1). The dashboard indexes these into a disposable cache.

### Run record (immutable, append-only) — Q2
```
id, parents[], goal_node, contract_ref,
question, hypothesis, status (confirmed|refuted|inconclusive),
seeds[], metrics{q0_err, w_dir_err, w_mag_err, rho_band, wall_s, ...},
artefacts[], commit, substrate_versions{propagator@v, scorer@v, ...},
gates_passed[], oracle_clean (bool), N, claim_scope (N1|cohort),
narrative_md
```

### Claim card (the knowledge state) — Q8
```
statement,
supporting_runs[], refuting_runs[],
trust_stamp{confidence, gates_passed, oracle_clean},
scope (N + N1|cohort),
depends_on[ substrate@version ],   # <-- powers automatic blast-radius
status (live|superseded|needs_replication),
superseded_by,
links_to_writeups[]
```
"What's true right now?" = `status == live`. A refutation flips a field and adds an
edge — **corrections are pointers, not rewrites.** A substrate version bump
auto-flips every dependent card to `needs_replication`.

### Goal node (the trajectory) — Q3
```
id, parent, title, kind (thesis|chapter|branch|question),
state (open|blocked|closed|revivable),
budget{expected_runs, expected_wall}, spent{runs, wall},
contract_refs[], child_runs[], last_measured,
is_trunk_artifact (bool)   # the gap / thesis contribution
```

### Branch contract (pre-registration) — Q4
```
goal_node, question, hypothesis,
predictions[ {named uncertain seed/case, expected outcome} ],
confirm_criteria, refute_criteria,
budget, required_artefacts[],
model, reasoning,
run_type_policy{ cheap: auto, expensive: gated },
amendments[ {version, diff, reason, approved_by} ]   # frozen + versioned
```

### Glossary term (canon) — Q7
```
term, definition, status (provisional|canon),
coined_in, promoted_on, synonyms_blocked[]
```

### Decks (Q8, Q10) — separate stores, same substrate
- **Research claims** (refuted by experiments)
- **Working preferences** (change when *you* change your mind)
- **Meetings** (Roberto / Jovan / Nick: decisions, steers, action-items+deadlines)
- **Literature** (papers: gap / method-to-transplant / novelty-relation / risk —
  e.g. the existing Robinson & Frueh 2025 card)

### Derived (never authored)
- **Live head** — ~30-line auto-rendered view (trunk node, current branch, current
  question, next-ranked). Replaces "read the first 80 lines of PROGRESS, but not the
  rest."
- **PROGRESS.md** — becomes a *printout* of the objects, not a thing anyone edits.

---

## 3. The trust machine (Q6) — the spine

### Gate suite (three families)

**Epistemic gates** (turn recurring traps into executable tripwires):
- **Oracle-leak assertion** — the scorer fails if any `truth_*` key is read during a
  blind run.
- **N-scope gate** — a claim tagged "cohort" must cite N ≥ threshold, else
  auto-downgrade to "N=1, provisional."
- **Validator == production cost** — refuse a validation grid containing truth at
  idx 0 (the s059i trap).
- **Conservation / smoke-at-truth** — L_J2000 & 2T drift < 1e-12; render-at-truth → ρ=0.

**Code-discipline gates** (at the promotion seam):
- **Glossary-clean** (linter + semantic synonym-block).
- **No-duplicate-symbol** (the ω-sign copy-paste killer).
- **Deep-module + API-doc** review.
- **Visible test** — a test the human can see as a plot.

**Compute-efficiency gate:**
- **Parallelism check** — flag/refuse serial-where-independent; executor
  parallelises seeds by default (enforced-with-override).

### Substrate versioning + automatic blast-radius
- Every shared function (`propagator`, `surrogate`, `scorer`, `cross`, `decimate`, …)
  carries a version/hash and lives behind a **versioned, gated interface**.
- Every run record is **stamped** with the substrate versions it ran against.
- A bug fix bumps a version → **every dependent run + claim auto-flips to
  `needs_replication`**, and the dashboard shows the blast radius as a query result.
  The thing that cost s001–s066 becomes one line.

### Two-tier code model (Q7)
- **Scratch** (`experiments/sXXX.py`): fast, loose, throwaway. Only epistemic gates.
- **Substrate** (`lib/`): versioned, deep modules, API docs, visible tests,
  glossary-clean, no duplicates.
- **Promotion gate**: the single seam where rot enters today. Nothing becomes
  substrate without passing the full code-discipline gate + human sign-off.

---

## 4. The control loop (Q3–Q5, Q9)

```
strategy (human, terminal)
   └─ grill → branch CONTRACT (frozen, agreed terms)
         └─ queued headless-CC RUN(s)  ──▶ batch on the 24-core box
                └─ gates + scoring  ──▶ RUN RECORD + CLAIM CARD updates
                      └─ goal-node state update
   ▲                                                  │
   │  governor: budget breach ─▶ auto-strategy-review ┘
   │  strategist: ranked frontier (proactive) ────────┘
   └──────── escalations / approvals (phone or web) ──────────
```

- **Governor** fires on budget breach → auto-strategy-review → recommendation to human.
- **Strategist** runs proactively → ranked **frontier** blending: open-node
  next-actions + **revived dead branches** (a claim that killed a branch got
  superseded → "its blocker is no longer true, reconsider?") + **advisor
  action-items** + **literature-driven openings/threats**. Human sets the lens.

---

## 5. The surface (Q11)

**Three surfaces, each doing what it is best at:**
- **Web app** = the map + control panel (decks, goal tree, frontier, claim set,
  meetings, literature, light curves, attitude anims) **with an embedded terminal
  pane** running a real Claude Code session. Panels deep-link into the terminal.
- **Interactive Claude Code** = the strategy conversation (grilling, blessing
  branches, writing contracts). The terminal stays a terminal.
- **Phone / Telegram** = async approvals + escalations + a lite invocable-skill set.

**Boundary discipline:** the web backend *watches the canonical files*, indexes them
into a disposable cache, and renders. It writes only **intent** (approved contracts,
blessings) into a queue the headless executor consumes. Delete the dashboard → lose
nothing but the cache.

---

## 6. Phasing (Q12)

- **Phase 0 — Schemas + foundations.** Define the canonical objects; stand up
  substrate versioning + cheap epistemic gates; establish the two-tier code model +
  promotion gate + glossary linter.
- **Phase 1 — The loop in the terminal (no web app yet).** Goal tree + derived
  live-head + run records + claim cards + branch contracts; rewrite `wind-down` /
  `resume` to read/write structured records instead of prose. Validate on a few real
  experiments. **Proves the engine before any UI.**
- **Phase 2 — Strategist + governor.** Drift governor, auto-strategy-review,
  promising-frontier, dead-branch revival, queued headless execution.
- **Phase 3 — External decks.** Meetings + literature ingestion; the gap-as-trunk
  artifact; auto-extract → approve flow.
- **Phase 4 — Web app.** Read-surface first (cheap), then control-plane + embedded
  terminal, then Telegram-lite.

### Cross-cutting workstreams
- **W-A — Ideal skill-set design (runs early).** Do *not* assume today's skills are
  the target. Audit how CC is actually used in the project + how Girish wants to work
  ideally, then design the ideal starting skill inventory (analysis, experiment-run,
  strategist, meeting-ingest, lit-review, viz, promotion-review, …) each with
  model/reasoning defaults. This defines what contracts can invoke.
- **W-B — Migration / backfill (lossy-pragmatic).** One-time agent pass parsing the
  100 experiments + 86 memory files + PROGRESS palimpsest into run records + claim
  cards + the goal-tree skeleton, each **linked back to its original writeup** for
  prose nuance. Don't transcribe every sentence; capture run metadata + claims +
  tree. This is what makes dead-branch revival and idea-exploration work on day one.
- **W-C — Claude Design prompt-pack (after structure is final).** Produce a set of
  prompts to give to Claude Design to generate + iterate cohesive UI mockups for the
  panels (the embedded terminal stays a terminal).

---

## 7. Open items to resolve before/along Phase 0

1. **Ideal skill-set design (W-A)** — needs its own grill-me session.
2. Exact thresholds for the N-scope gate and the budget-breach trigger.
3. Where the queue + cache live (local box; rebuildable).
4. Glossary seeding — extract the de-facto vocabulary already in PROGRESS
   ("body-twin", "anchor-baseline aliasing", "polhode diameter", "hard-shoot trap",
   "ρ-band", …) into canon definitions as part of W-B.
5. Claude Design prompt-pack scope (W-C).

---

---

## 8. Workstream A — Ideal skill inventory (designed 2026-06-01)

**Finding that motivated the redesign:** the current skill set is itself a
half-migrated palimpsest — `research-loop` and `orchestrate` point at the dead
`notebooks/inversion/` structure (CURRENT_STATE.md / wiki / EXPERIMENTS.md /
micro_*); `research-loop-workspace` and `resume-workspace` are empty stubs; `resume`
and `wind-down` straddle two workspaces. Redesign, don't patch.

### Structural decisions
- **Decompose the monolithic loop into distinct spine verbs** (each owns one
  structured artifact, model-tuned and updated in isolation).
- **Mechanical trust machinery = hooks** (fire automatically, every time); **judgment
  = invocable skills.**
- **Model policy: everything is Opus. Minimum thinking = `high`.** Harder skills →
  `xhigh` or `max`. No Sonnet/Haiku anywhere.
- **Two loops, not one:** the **convergent** loop exploits the frontier; the
  **divergent** loop (FAFO) explores off it. Both feed the same record/claim system.

### The inventory

**Spine (convergent loop verbs)**
| Skill | From | Model |
|---|---|---|
| `orient` | rewrite `resume` | Opus high |
| `strategize` (frontier) | new | Opus xhigh |
| `align` (grill → frozen contract) | specialize `grill-me` | Opus xhigh |
| `execute` (adapt → parallel batch → gate → score → record) — *planned as `run`; renamed to avoid the built-in `run` skill collision (bug #25209)* | new, absorbs `orchestrate` | Opus high (diagnosis sub-step xhigh) |
| `close` (run record + claim cards + node update) | rewrite `wind-down` | Opus high |

**Divergent loop**
| `fafo` | new | orchestrator brainstorms ~4 off-frontier ideas → gated mini-loops (oracle-leak, agreed artifact format) → max-reasoning strategist verdict (gold/promising/dross). **Gold → you decide promotion** (explore freely, don't auto-commit a branch). Dross → tried-and-shelved card (not blindly re-tried; revivable later). **On-demand first; standing idle-compute loop only after cost/benefit is measured.** Verdict agent: Opus max. |

**Governance / trust**
| `strategy-review` | new | governor escalation re-think | Opus max |
| `periodic-review` | new | standing "is the trunk still right?" pass | Opus max |
| `promote` | new (composes `tdd` + `code-review` + `improve-architecture`) | the promotion gate | Opus xhigh |
| `glossary` | new | define / promote / synonym-judgment | Opus high |

**Knowledge**
| `revive` | new (may fold into `strategize`) | dead-branch revival scan | Opus xhigh |

**External**
| `meeting-ingest` | new | transcript → cards + frontier reweights → your approval | Opus xhigh |
| `lit-card` / `stay-current` | new (composes `tavily-research`) | paper → literature card; monitor the field | Opus high |

**Instruments** (Opus high; all **emit to the web-app plot stream**)
`attitude-viz`, `attitude-anim`, `lc-compare`, `diagnose` (xhigh), `eli-noob`.

**Meta**
`skill-create` — keep `write-a-skill`; new skills born from grilling a repeated
manual pattern.

**Hooks (automatic — NOT skills)**
- `gate-check` — oracle-leak, N-scope, conservation/smoke, parallelism.
- `blast-radius` — substrate version bump → flip dependent cards to `needs_replication`.
- `glossary-lint` — mechanical canon-membership check.
- `analytical-probe` — cheap-first gate inside `align`/`execute` (can't authorize expensive
  compute until the cheap re-score is stated and shown insufficient).
- `visual-validation` — plots surface to the stream; **soft-surface by default,
  hard-block only on substrate promotion.**

**Retire:** `research-loop`, `research-loop-workspace`, `resume-workspace`,
`orchestrate`.

### Autonomy map
Four human gates: **(1)** pick a frontier item, **(2)** grill the contract (`align`),
**(3)** answer escalations (gate fail / off-contract deviation), **(4)** decide on
strategy-reviews. Everything else auto-chains. **Claim-card writes are auto-draft →
human-confirm.**

### Visual validation (first-class)
You spot issues visually that the agent misses → your eyes are a validation gate.
The web app has a **live plot/artifact stream** (auto-surfaced, tagged by run/branch,
newest-first — no path-hunting). Instruments are the emitters. Soft-surface except
substrate promotion (hard-block).

---

### `periodic-review` cadence (decided)
Fires on **(a) weekly**, **(b) auto-before each advisor meeting** (preps the meeting
*and* checks the trunk in one pass), **(c) on-demand**. Produces a "state of the
research" brief: trunk still right? stale/over-budget branches? claims gone stale via
blast-radius? current frontier; does the gap still hold vs recent literature?

### Candidate v1.x skills (create-later via `skill-create`, not in the v1 freeze)
- `devils-advocate` / red-team — adversarially attack a claim before it's trusted
  (serves the trust spine; strong candidate).
- `teach-back` — you explain your understanding, an agent finds the gaps (inverse of
  `eli-noob`).
- `paper-draft` — assemble live claims + figures into thesis/paper prose.

---

---

## 9. Skill backlog (brainstormed 2026-06-01)

Tiers: **[core]** = build early (attacks measured pain directly); **[v1]** = approved
for the first set; **[cand]** = candidate, unrated, create-later via `skill-create`.

**Trust & epistemic hygiene** (the binding constraint — invest here)
- `devils-advocate` **[core]** — adversarially attack a claim before trust.
- `replication-audit` **[core]** — re-derive a load-bearing claim by a *different* method; flag if it doesn't reproduce. Direct assault on the 25% reversal rate.
- `gauge-sentinel` **[core]** — auto finite-diff smoke + conservation + convention checks on any new propagation/scoring code. The ω-sign and `times[0]==0` bug class.
- `pre-mortem` **[core]** — assume the branch failed; what killed it? Before compute.
- `steelman` **[v1]** — strongest case for a shelved/dross idea.
- `oracle-hunt` **[v1]** — scan a "blind" pipeline for hidden truth leakage (s058/s059 disease).
- `assumption-archaeology` **[v1]** — excavate a branch's implicit assumptions + their substrate-version deps (pre-emptive blast radius).
- `calibration` **[v1]** — past contract predictions vs outcomes; tune trust stamps to real hit-rate.

**Strategy & divergent**
- `fafo` **[set]** — 4-idea exploration engine.
- `cross-pollinate` **[v1]** — transplant a method from an analogous solved problem (the RF25 NLL-loss move).
- `invert` **[v1]** — "what would make this impossible?"
- `north-star-recompute` **[v1]** — re-derive the trunk from first principles given current knowledge.
- `bet-sizing` **[cand]** — probability × payoff ranking of the frontier.

**Knowledge & synthesis**
- `claim-consolidate` **[v1]** — merge/dedupe claim cards, surface contradictions/orphans.
- `so-what` **[v1]** — force the real architectural consequence of a result.
- `gap-map` **[v1]** — live literature landscape: contribution, neighbors, threats, white space.

**PhD / external**
- `meeting-prep` **[core]** — pre-meeting: status, decisions-needed, demos, anticipated pushback.
- `advisor-sim` **[core]** — simulate Roberto/Jovan/Nick's questions from transcript history. Rehearsal.
- `reviewer-2` **[v1]** — hostile peer reviewer on a draft.
- `prior-art-guard` **[v1]** — novelty check before claiming (RF25/RF74 risk).
- `defense-prep` **[v1]** — mock thesis-defense Q&A.

**Comprehension & communication**
- `teach-back` **[v1]** — you explain, agent finds gaps.
- `paper-draft` **[v1]** — live claims + figures → prose.
- `figure-forge` **[v1]** — publication-quality figures.
- `rubber-duck` **[v1]** — agent listens + asks clarifying Qs; no grading.
- `elevator` **[cand]** — 30s/2m/5m pitch of current state.

**Execution & compute**
- `checkpoint-doctor` **[v1]** — verify a script saves all candidates/intermediate state *before* it runs.
- `profile-first` **[cand]** — profile before optimizing (the s099b move).
- `budget-forecast` **[cand]** — cost/wall vs nearest analog before a batch.

**Meta & morale**
- `friction-log` **[v1]** — capture friction the instant it happens; route to a fix.
- `momentum` **[v1]** — surface recent wins + trunk movement when morale dips.
- `retro` **[v1]** — post-session retrospective. **MUST read structured records + friction-log, NOT the chat transcript** (de-bias from end-of-chat fatigue/recency — user's catch).
- `time-capsule` **[cand]** — snapshot "what I believe & why" for later calibration.

---

## 10. Agent & context architecture

**Progressive disclosure:** only a skill's `name` + `description` is resident at
startup (~tens of tokens each); the body lazy-loads on invocation, in the invoker's
context. So a large library is cheap at rest. Real costs: triggering accuracy (cure:
tight non-overlapping descriptions) and body-bloat on invocation (cure: scope per
agent + run in subagents with their own context).

**The buckets ARE the agent boundaries** — and therefore the context strategy. Each
role-agent loads only its bucket; specialized work runs in subagents with **their own
context windows**, so heavy skill bodies never touch the main loop.

| Agent (own context) | Bucket |
|---|---|
| **Orchestrator** (main loop, thin) | `orient`, `strategize`, `align`, `execute`-dispatch, `close` |
| **Trust** | devils-advocate, steelman, replication-audit, oracle-hunt, gauge-sentinel, pre-mortem, assumption-archaeology, calibration |
| **Strategy** | fafo, cross-pollinate, invert, north-star-recompute, claim-consolidate, so-what, gap-map, revive |
| **Executor** (headless, per run) | run, profile-first, budget-forecast, checkpoint-doctor, diagnose, gauge-sentinel, instruments |
| **PhD/external** | meeting-ingest, meeting-prep, advisor-sim, lit-card, stay-current, prior-art-guard, gap-map |
| **Comms/writing** | paper-draft, reviewer-2, defense-prep, figure-forge, elevator, teach-back, eli-noob |
| **Meta** | retro, friction-log, momentum, skill-create, periodic-review, strategy-review |

- **Role-scoped standing agents, not a swarm for its own sake.** Transient swarms
  (Workflow tool) only for `fafo` and parallel experiments.
- **External backbones (Archon, MCP knowledge/task servers): components, never the
  spine.** They overlap with our claim-cards (knowledge) and goal-tree (tasks);
  adopting one as a *store* violates Q1 "reflect, don't own." Allowed only as a
  disposable index / RAG layer over the canonical files, or as instruments — evaluate
  per concrete need, later.

---

*End of founding plan + W-A skill design + skill backlog + agent architecture.*

---

## 11. Build log & current state (updated 2026-06-04)

**Done & committed:** design phase, Phase 0 schemas, W-B backfill (referentially-closed
store), Phase 1 spine (`orient → strategize → align → execute → close` + trust hooks).
**Phase 1 validated** by one real end-to-end cycle — the `slow-tumbler-generality`
contract → run **s113** (BLOCKED: weak-anchor cross intractable; the `execute` canary
caught an ~8.2-hr cost wall cheaply — the machinery working as designed).

**2026-06-04 session:** retro on cycle 1. Rather than jump to Phase 2, hardened the loop's
*ergonomics* (the retro's own recommendation — dogfood before building the governor). Shipped:
- `research_os/decisions/` — ADR-lite dir; this file's decisions now have a home. ADRs 0001–0004.
- **Verbosity** → persistent `concise` output-style (ADR-0004; lives in `~/.claude/output-styles/`, user-global). Activate with `/output-style concise`.
- **Visual surface** → local browser plot-stream `research_os/render/stream/` (ADR-0003); `stream_add.py` is the emit primitive (demo plots: s113 cost-wall, trust ledger, corpus status). Cache is gitignored.
- **Glossary** → decided RO-native store + harvest MP's *behaviors* (ADR-0002); skill NOT built — manual-first on the next real confusing term, then formalise.
- **Skill-integration policy** → harvest-and-refit installed skills, never adopt wholesale (ADR-0001).
- Fixed a backfill caveat: `status: confirmed` is *local*-hypothesis, not research-advanced (schema + backfill README) — surfaced by the corpus-status plot.

**2026-06-04 session 2:** built **`dynamic-viz`** (queue #2, DONE) + ran the glossary
exercise manually as the ADR-0002 dogfood. Shipped:
- **`dynamic-viz` skill** (`.claude/skills/dynamic-viz/`) — the visual instrument: decides
  *what* to draw for an RO artifact, draws the minimal glance, emits via `stream_add`. Defers
  LC/attitude to the domain instruments. Plus `research_os/render/ro_viz.py` (shared dark
  palette + `emit()` upsert-by-filename + live store readers, harvested from the 3 demos),
  `serve_stream.sh` (idempotent serve+open, :8137), and `viz_frontier_map.py` (proof plot).
- **Glossary manual sharpening** → `research_os/glossary/SEED_DRAFT.md`: ~22 atoms locked
  (IA Cloud, Filter vs Rank, spurious survivor↔degenerate solution, Single-Wind-LC-Window
  Polish, finite-diff ω-aliasing, multi-ωdir-start, ρ-band-with-generator, Basin
  characterisation, je-w-param, …). The method it taught: decompose a confusing *name* into
  atoms; the glossary stores atoms, composite goal/pipeline titles self-document.
- **ADR-0005 (proposed)** — the store is a typed graph; promote `Pipeline` to a first-class
  node distinct from `Goal`. Trunk = most-general ancestor, not most-important node.

**2026-06-04 session 3:** built the **`glossary`** skill + formalised the seed (queue #1, DONE).
Shipped:
- **`glossary` skill** (`.claude/skills/glossary/`) — the on-the-fly vocabulary instrument:
  decompose-a-confusing-name-into-atoms, lazy creation, in-conversation challenge,
  sharpen-overloaded-terms (the ADR-0002 behaviors). Coins `glossary_term` JSON; the lint arms
  itself the moment a `synonyms_blocked` exists.
- **22 canon terms** `research_os/glossary/*.json` (SEED_DRAFT → JSON, schema-valid, cross-linked,
  `synonyms_blocked` + `coined_in` provenance; **draft deleted**). The **`glossary-lint` hook is
  now LIVE** (was an inert no-op): improved `loop/glossary_lint.py` so a synonym can map to >1
  term (`discriminator` → Filter/Rank) and verified it fires on `discriminator`/`phantom`/
  `windowed photometry`.
- **Two goal retitles applied** (`goal_windowed-photometry-polish`, `goal_anchor-cap-slow-tumbler`)
  + `goals/_index.json` synced (also fixed a pre-existing open/blocked drift on the latter).
- **Frontier item filed** as `goal_basin-char-cheap-broad` (question under `goal_basin-polhode-twin`):
  basin characterisation cheap-and-broad across the cohort (the s003 result is N=3).
- Store re-validated: **referentially closed (204 objects)**.

**2026-06-04 session 4:** built **ADR-0005 (Pipeline-as-first-class-node)** — queue #1, DONE
(Girish: "All four, full migrate"). Shipped:
- **`schemas/pipeline.schema.json`** (new node type) + three off-pipeline edge fields
  (`run_record.tests`, `run_record.blocked_by`, `substrate_component.promoted_from`). All six
  ADR edges (serves / composes / tests / supersedes / promoted-from / blocked-by) now schema +
  ref-integrity wired in `loop/validate.py`, which also pulls `glossary_term` into the closed
  store (blocked-by targets a named term).
- **4 pipeline nodes** (`research_os/pipelines/`): densify [closed], joint-grid-pivot [closed],
  cross-cloud [open], single-wind-lc-window-polish [open]. `backfill/migrate_adr0005.py` (the
  audit trail) repointed/tagged **15 runs**, deleted the 2 pure-method goals (densify,
  joint-grid-pivot → runs repoint up to the served chapter), split the 2 live ones (cross-cloud
  chapter kept; windowed-polish retitled to the pure question). `hard-shoot-trap` coined as a
  glossary term so the blocked-by payoff edge is real (s105→hard-shoot-trap, s108→finite-diff-
  omega-aliasing — same edge, two regimes).
- **`render/live_head.py`** gained the **TRIED** projection (pipelines + state + tested_by× +
  serves + headline). `supersedes`/`promoted_from` are schema-wired but null (no grounded
  assertion yet — populate when one exists; don't fabricate).
- Store re-validated: **referentially closed (230 objects)**; 10/10 examples still valid; hooks
  import clean.

**2026-06-07 session (grill-with-docs — the laboratory reframe):** ADR-0007 **accepted**. Asked
what felt *missing*, Girish reframed: not features, a **laboratory** — named, versioned, runnable,
composable boxes ("run the IA Cloud block on seed 39", results auto-drawn, then edit the block;
"if there are 3 versions, SEE that and fix it"). Resolved:
- **Vocabulary** (re-used, not invented — the store already had the words via ADR-0005): coined 4
  canon glossary terms `glossary/{tool,pipeline,primitive,artifact}.json`. Girish's *base function
  → block → pipeline* = **primitive → Tool → Pipeline**; the "materials" = **artifact** (typed by a
  glossary term — the glossary doubles as the port type-system). Tool = capability (no hypothesis);
  Pipeline = bet (hypothesis + serves a Goal).
- **Tool anatomy:** one `entry_point` @ one version; variation in `default_params`, never copies.
  Schema fields added to `substrate_component` (`entry_point/ports/default_params/canon/variant_of`,
  all optional); `validate.py` ref-checks `variant_of`→Tool and `ports`→glossary_term. IA-Cloud
  generator is **per-epoch** (epoch is an input); a whole-LC cloud = the same Tool **mapped** over a
  window → Pipelines need `map` alongside `compose`.
- **Variant model (B):** one canon Tool/family; meaningful variants = own cards via `variant_of`;
  accidental copies collapse into `default_params`.
- **Runnable architecture (Q1-safe):** Tool card binds to canonical code; a drift-check/tool-lint
  (foundation = `render/machinery_overlay.json`) refuses to run a stale binding; execution is the
  **executor (#15)**, store reflects definitions+records only. Re-prioritises #15 to its smallest
  form: a manual **run-button**, NOT an autonomy daemon.
- **Sequencing: scour first.** W-D Tool/Pipeline backfill (`backfill/W-D_tool_pipeline_scour.md`) —
  W-B-sized, dozens+ Tools, **run at SESSION-START**. Then the run-button. Then DAG/artifact-instances.
- Store re-validated: **referentially closed (237 objects)**.

**Open queue (next agent):**
1. **W-D scour** (ADR-0007) — **kick off at session-start**; builds the honest Tool/Pipeline
   registry the laboratory needs. Plan: `backfill/W-D_tool_pipeline_scour.md`.
2. **Phase-2 fork** — governor / proactive-strategist vs more live-testing (the cycle-1 retro
   leaned live-test-first; Girish wants ≥1 more loop before calling Phase-1 done, and likely a
   caveman-level brevity pass beyond `concise`).
3. (Lower) Populate `supersedes` / `promoted_from` edges as grounded assertions appear; fuller
   `blocked_by` tagging across the corpus (finite-diff-aliasing also touches s100/s111/s113 —
   verify each before tagging). Exercise the glossary lint; render `CONTEXT.md` *from* the store.

**Next pointer:** `/orient` for the research state; this section for the OS-build state.
