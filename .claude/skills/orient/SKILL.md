---
name: orient
description: "Orient at the start of an LCAS inversion session by rendering the live-head from the Research OS trust store (research_os/), not by reading the PROGRESS prose. Shows the trunk, the current branch, the open/revivable frontier, and the trust state (claims + blast-radius flags), checks store-vs-HEAD drift, then puts the current thread + the frontier pick to you. Replaces the older `resume` ritual. Triggers on: orient, resume, where are we, catch me up, what's the status, continue, what were we doing, pick up where we left off, start of session."
---

# Orient — LCAS inversion (Research OS, branch forward_survey)

Session-start. The canonical state lives in the **trust store** under `research_os/`
(run records, goal nodes, claim cards, substrate components). Orient *renders* that
store — it does not re-summarise prose. The active research workspace is still
`notebooks/inversion/survey/` on `forward_survey`; the older `notebooks/inversion/`
tree (m103/m115/m126, micro wiki) is frozen reference — do not import, do not cite its
buggy-era numbers.

## What to do

### 1. Render the live head
```bash
python research_os/render/live_head.py
```
This is a pure-read, deterministic ~30-line view: trunk + last_measured, current
branch (= goal node of the newest run record), the open/revivable **frontier**
(UNRANKED in Phase 1 — `/strategize` ranks it), the **trust block** (claim status +
blast-radius `needs_replication` flags), and the last 5 runs. Print it; it *is* the
orientation. Don't hand-roll a state summary on top of it.

### 2. Drift check (store vs HEAD) — the transition guard
Until `close` writes records automatically, a session may still land an experiment the
legacy way (a `experiments/sXXX_*.md` + PROGRESS edit) without a run record, leaving
the store behind. Detect it cheaply:
```bash
git log -3 --oneline
ls notebooks/inversion/survey/experiments/ | grep -oE '^s[0-9]+' | sort -V | tail -1
```
Compare the highest `experiments/sXXX` to the live-head's newest run id.
- **In sync** (equal) → the store is the source of truth; the PROGRESS top-banner is a
  legacy printout, **do not read it**.
- **Store behind** (a higher `sXXX` exists with no record) → say so, then read *only*
  that writeup's TL;DR + the PROGRESS top-banner (`head -12`) for the delta, and flag
  that the run needs porting into a record (a `close`/backfill job).

### 3. Present the thread + the pick
In one short paragraph: the current branch + its latest run's status, and — because the
frontier is usually >1 open branch — **name the genuine fork** (e.g. the newest branch
vs. a close-behind open branch) and ask: *continue this thread, or redirect?* Do not
silently assume the newest run is the intended thread; surface the choice.

If the live-head's "newest run" differs from the trunk's curated `last_measured`
(e.g. a report vs. the last real measurement), say both honestly.

## Standing frame (the little that isn't an object yet)

Most "what's true" now lives in the trust block (claim cards). Only carry what isn't
yet an object — and flag it as migrating:

- **Acceptance is multi-solution, not truth-recovery.** Headline cohort metric = **%
  seeds with ≥1 (A∪B) basin**, not "% truth recovered." Returning every `(q0, ω)` that
  explains the LC at noise fidelity is the goal; truth is a desirable subset.
- **ρ-bands** (ρ = √MSE / 0.05 vs truth hi-fi LC): **A** ρ<2 admit · **B** 2≤ρ<4 admit ·
  **C** 4≤ρ<8 keep-flagged · **D** ρ≥8 reject.  *(→ glossary, deferred to Phase 2.)*
- **The store's `needs_replication` flags are live epistemics, not history.** If the
  live-head flags a claim (e.g. `pol-diam-predicts-basin-width`,
  `lc-only-omega-priors-dead`), its evidence rests on stale substrate (`propagator@1.0.0`,
  pre-fix ≤ s066). Treat it as an open question, not a fact. This is the trust machine
  working — don't "restore" it to live out of deference to an old memory.

For anything deeper ("is the surrogate honest off-truth?", "which dead-ends are
closed?") read the relevant **claim card** in `research_os/claims/`, not skill prose.

## What NOT to do

- **Don't re-read the memory directory.** `MEMORY.md` auto-loads; re-reading entries is
  the biggest context-bloat culprit.
- **Don't read PROGRESS.md or log.md in full** (hundreds of KB). The live-head replaces
  the top-banner read; only `head -12` PROGRESS if the drift check says the store is behind.
- **Don't hand-summarise the store.** Render it. If the render is missing something a
  session needs, that's a renderer/backfill gap — say so, don't paper over it with prose.
- **Don't auto-load** `notebooks/inversion/CURRENT_STATE.md` or the `wiki/` (frozen pre-fix).
- **Don't print a "LOADED RULES" checklist.** Ceremony, not verification.

## Where things live

- **Trust store (canonical):** `research_os/{records,goals,claims,substrate}/` ·
  renderer `research_os/render/live_head.py` · schemas + grammar `research_os/schemas/`.
- **Research workspace:** `notebooks/inversion/survey/` — `experiments/` (sXXX scripts +
  writeups), `results/`, `lib/` (thin `src.*` wrappers), `concepts/`, `data/trajectories/`.
- **Legacy live state (transition only):** `notebooks/inversion/survey/PROGRESS.md`,
  `log.md` — being demoted to printouts as `close` takes over writing records.
- **Frozen reference (don't auto-load):** `notebooks/inversion/CURRENT_STATE.md`, `wiki/`.

## Target cost

≤1.5k tokens above auto-loaded memory: one renderer call + one drift check + the
paragraph. If you're reading wiki pages, full PROGRESS, or memory files at session
start, you've fallen back to the old ritual — stop and ask the user to redirect.
