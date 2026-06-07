# Research OS — web app (Phase 4 surface)

The dashboard surface from `research_os/PLAN.md` §5/§11. **A derived read-model + a
write-only control plane** over the canonical `research_os/` JSON store — *reflect,
not own* (Q1). The backend watches the files, indexes them into an in-memory cache,
and renders. It writes only **intent** (to `research_os/queue/`). Delete the whole
`webapp/` directory → you lose nothing but the cache.

## Run it

```bash
research_os/webapp/start.sh            # build frontend + serve everything on :8138
research_os/webapp/start.sh --dev      # vite hot-reload (:5180) + backend (:8138)
```

Then open <http://127.0.0.1:8138/>. Requires the repo `.venv` (FastAPI/uvicorn/
websockets/watchfiles installed there) and Node ≥ 20.

## What it gives you

- **Pinned trunk/branch header** — the trunk thesis, the current chapter→branch→
  question, and the latest run are *always visible*. The structural fix for "we
  tangent off the main thread" (PLAN §0).
- **Overview** — the live head: hero counts, blast-radius alert, trunk, current
  thread, frontier, what-we've-tried (pipelines), trust ledger, recent runs, latest
  plots. Replaces "read the first 80 lines of PROGRESS".
- **Goal tree** — the explorable thesis→chapter→branch→question tree, runs attached.
- **Frontier** — open/revivable branches & questions: what's next / promising.
- **Runs** — the run-record timeline + full per-run detail (metrics, narrative,
  artefacts, gates, substrate stamp, lineage, blast radius).
- **Claims** — the two decks (research / preference): *what's true right now*, trust
  stamps, supersession edges, live blast-radius flags.
- **Pipelines** — the ADR-0005 "how we're trying" method nodes (serves / composes /
  tested-by / supersession).
- **Substrate** — the version registry + **automatic blast radius** ("the thing that
  cost s001–s066 becomes one line").
- **Glossary** — the canon vocabulary + the lint's blocked synonyms.
- **Plot stream** — visualisation-as-we-go. Plots auto-surface, tagged by run/branch,
  newest-first, **no path-hunting**. New plots pop in live. Subsumes
  `render/stream/index.html`.
- **Terminals** — *multiple* persistent Claude Code / shell sessions in the browser.
  PTYs live server-side and survive reloads; the dock (⌃\`) is available on every view;
  panels deep-link in (e.g. **orient**, **align this branch**, **/strategize**).

Everything updates live: the backend SSE channel broadcasts a store revision whenever
a canonical file changes (a skill ran in a terminal, a record landed) and the UI
refetches. ⌘K opens a command palette over every object + the spine skills.

## Architecture

```
research_os/webapp/
├── backend/
│   ├── app.py        FastAPI: /api/* read-model · SSE /api/events · WS terminals · intent queue · serves dist/
│   ├── indexer.py    the read-model (reuses render/live_head.compute_head + richer projections)
│   ├── terminals.py  multi-PTY manager (stdlib pty), persistent, scrollback re-attach
│   └── intents.py    write-only control-plane queue -> research_os/queue/
├── frontend/         Vite + React + TS + Tailwind v4 + xterm.js (dark theme = ro_viz.py palette)
│   └── src/{lib,ui,shell,terminal,views}
├── shot.py           headless screenshot helper (verification)
└── start.sh          build + serve
```

### Boundary discipline (Q1/Q11)
The backend **never writes research state**. Reads: globs `records/ goals/ claims/
pipelines/ substrate/ contracts/ glossary/`. Writes: only `queue/*.json` intent files
for the (Phase-2) headless executor to consume. The canonical objects are authored
only by the spine skills (`orient → strategize → align → execute → close`) in a
terminal — the dashboard surfaces them, it does not own them.

### Endpoints
| | |
|---|---|
| `GET /api/snapshot` | the whole read-model |
| `GET /api/events` | SSE `{type: store\|stream\|intents, rev}` on change |
| `GET /api/stream/manifest`, `/stream/<file>` | plot stream |
| `GET/POST/DELETE /api/terminals`, `WS /ws/terminal/{id}` | terminals |
| `GET /api/intents`, `POST /api/intent` | control plane |

## Not yet (future)
Telegram-lite (same invocable skills, async); the headless executor consuming the
intent queue; Claude-Design polish pass (W-C). The structure here is what those build on.
