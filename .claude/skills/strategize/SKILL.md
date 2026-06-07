---
name: strategize
description: "Strategy verb of the LCAS inversion loop — ranks the trust store's open/revivable frontier into a prioritised next-actions list with store-cited rationale, under a ranking lens you choose (leverage-to-trunk, cheapest-to-resolve, max-uncertainty-reduction, unblocks-most, trust-repair). Reads the live-head (which renders the frontier UNRANKED) and the goal nodes/claims; writes no CANONICAL store object — it may emit the ranking as a derived render artifact (render/frontier_ranking.json) so the dashboard shows it ranked. The top pick flows into align. Use when the user asks what's next, what should I work on, what's highest-leverage, rank/prioritise the branches, or strategise — NOT 'where are we / catch me up' (that's orient)."
---

# Strategize — LCAS inversion loop (Research OS, branch forward_survey)

The **strategy** verb (spine: orient → **strategize** → align → execute → close). `orient`
renders the frontier **UNRANKED**; `strategize` is where the judgment goes — it ranks the
open/revivable frontier into a prioritised next-actions list with **store-cited rationale**,
under a lens you choose. It writes **no canonical store object** (a ranking is lens-dependent
and transient) — but it **may emit the ranking as a *derived* render artifact**
(`render/frontier_ranking.json`, gitignored, alongside the live-head and plot-stream) so the
dashboard's Frontier view shows it ranked instead of recency-only (req #9). The top pick is
what you then take into `align` to freeze into a contract.

strategize is Opus min thinking **xhigh** (PLAN §8) — ranking leverage across the tree is
the highest-reasoning judgment in the loop.

> **Phase-1 scope.** strategize ranks the *explicit* frontier (open + already-`revivable`
> goal nodes). The proactive session-start pass, the governor-triggered re-rank, and the
> systematic **dead-branch revival scan** (superseded-claim → "its blocker is no longer
> true, reopen?") stay **Phase 2** (the `revive` skill + governor). Phase-1 rationale *may*
> still cite a trust signal — e.g. a branch whose blocker claim is now `needs_replication` —
> but strategize does not run the exhaustive revival detection itself.

## Step 1 — Pull the frontier (structured)

```bash
python research_os/render/live_head.py --json
```
Use the structured view, not the prose. The keys that matter:
- **`frontier[]`** — each `{id, state (open|revivable), title, last_run, last_at, chapter}`.
  This is the candidate set. `frontier_total` is the full count (the printout truncates).
- **`trust`** — `status_counts` (claim live/needs_replication/…), **`needs_replication[]`**
  (the blast-radius flags — load-bearing findings resting on stale substrate), `substrate_heads`.
- **`trunk`** — `{id, title, last_measured}`: the thesis goal the ranking serves.
- **`recent[]`** — last runs `{id, status, goal}`: momentum + what just moved.

For each frontier candidate you'll rank, read its goal node
`research_os/goals/<id>.json` for the standing `question`, `budget`, `spent`, and
`last_measured` — that's where the next concrete action and its cost come from. Cite it.

## Step 2 — Set the ranking lens (the human's call)

A frontier has no single objective ordering — the lens depends on the phase of the research
(Q9: *human sets the ranking lens per phase*). **Ask which lens this pass**, recommending the
default; the ranking changes meaningfully with the choice:
- **leverage-to-trunk** *(default)* — most directly advances the trunk question (recover
  `(q0, ω)` from one LC); discount branches orthogonal to the thesis contribution.
- **cheapest-to-resolve** — smallest `budget.expected_wall_s` / closest to a yes-no answer.
- **max-uncertainty-reduction** — the branch whose outcome most sharply splits the live
  hypotheses (the genuinely-uncertain prediction with the widest consequence either way).
- **unblocks-most** — answering it opens the most downstream branches.
- **trust-repair** — re-establish `needs_replication` claims that load-bearing work rests on
  (directly attacks foundation rot — the system's binding constraint).

If the user names a lens up front, skip the ask and use it.

## Step 3 — Rank with store-cited rationale

Produce a **ranked table** of frontier items (highest first). Each row:
- **goal node** (`id` + short title) and **the concrete next action** (from its `question` /
  `last_measured` — what one run would move it).
- **rationale** — one line under the chosen lens, **citing the store**: a `run` id, a claim
  status, a `needs_replication` flag, a budget. Every quantitative claim carries
  `(source: path/to/file[:line])` or is marked `unverified` (CLAUDE.md "Cite the file for
  every number"). Memory recall is not a source — ground it in the node/record/claim.
- **rough cost** — `budget.expected_wall_s` of the node or the nearest-analog run's actual wall.
- **uncertain crux** — the one thing whose outcome you can't predict (if you can't name one,
  the branch isn't ready to run — say so rather than ranking it high).

Be honest about ties and about thin evidence: a branch you can't cite is ranked on a guess —
flag it. Surface anything the frontier *should* contain but the store can't yet express — the
known gap is the **trajectory-class switching table** (which architecture suits which tumbler
class), which has no store object yet; cite class qualitatively from runs/claims and note the gap.

## Step 4 — Hand off + surface to the dashboard

Present the ranked frontier + your single top recommendation with the lens stated. The top
pick flows into **`align`** (grill → frozen contract) → `execute` → `close`. strategize
freezes nothing and runs nothing. If the user picks a different row than your top, that's the
lens working — proceed to `align` on their pick.

**Emit the derived ranking artifact** so the web dashboard shows the frontier ranked (req #9).
This is a *derived render artifact* (like the live-head / plot-stream), **not** a canonical
store object — write it, don't referentially-close it. Stamp it with the store revision you
ranked so the dashboard can flag it stale once a new run/claim lands:

```bash
python - <<'PY'
import json, sys, datetime
sys.path.insert(0, "research_os/webapp/backend")
import indexer  # reuse the one source of the store revision
ranking = {
    "lens": "leverage-to-trunk",                # the lens you actually used
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "generated_for_rev": indexer.store_revision(),
    "items": [
        # rank 1 = top pick. goal_id must resolve in goals/. rationale is store-cited.
        {"goal_id": "goal_XXX", "rank": 1, "rationale": "...", "action": "...",
         "cost": "~30 min", "crux": "..."},
        # … one row per ranked frontier node, in priority order
    ],
}
json.dump(ranking, open("research_os/render/frontier_ranking.json", "w"), indent=2)
print("wrote render/frontier_ranking.json")
PY
```

Schema: `research_os/schemas/frontier_ranking.schema.json`. The dashboard reads it via the
indexer and refreshes the Frontier view live (an SSE `ranking` event). If you do not emit it,
the dashboard simply stays recency-ordered — emitting is what turns req #9 on.

## What NOT to do

- **Don't write a store object.** The ranking is ephemeral and lens-dependent; re-derive it
  next time. Persisting a frontier snapshot is a Phase-2 governor concern, not this.
- **Don't rank without a lens.** The ordering is meaningless until the objective is named —
  ask, or use the one the user gave.
- **Don't cite from memory.** Ground every number in the goal node / run record / claim card;
  mark `unverified` if you can't.
- **Don't run the revival scan or freeze a contract.** Revival/governor are Phase 2; freezing
  is `align`. strategize only ranks and recommends.
- **Don't duplicate `orient`.** orient = "where are we" (renders state). strategize = "what
  next" (ranks the frontier). If the user just wants the lay of the land, that's orient.

## Where things live

- **Frontier source:** `research_os/render/live_head.py --json` (derived view) + the goal
  nodes `research_os/goals/*.json` it points at.
- **Trust signals:** `trust.needs_replication` in the live-head; the claim cards
  `research_os/claims/`.
- **Downstream:** `strategize` (rank) → `align` (freeze) → `execute` (run) → `close` (record).
