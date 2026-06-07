# Research OS — Phase-4 Web App: Compliance Review & Fix Backlog

**Reviewed:** 2026-06-06 · **Commit:** `0c7f085` (forward_survey) · **App:** running on `:8138` (rev verified live)
**Scope:** read-only review of the Phase-4 surface against `research_os/PLAN.md` (§0/§5/§6/§8, Q1/Q11), plus two issues reported from live use.
**Method:** read every backend + frontend file; screenshotted all 11 routes; exercised the API, SSE, the PTY-over-WebSocket round-trip, and the intent write-path; traced every backend filesystem write.

---

## 1. Verdict

The Phase-4 surface is **built to spec and genuinely works.** The boundary is clean (the only store-touching write in the entire backend is the intent queue), live SSE refresh fires on file change, the in-browser PTYs are real and persistent (`6*7` sent over the websocket returned `42`), and all 11 routes render in a cohesive dark, dense UI. **~10½ of 16 requirements fully met**; every *missing* requirement is a deliberately-deferred later-phase **engine** piece (frontier ranking, headless executor, meetings/literature decks, Telegram-lite, claim write-flow) — not a surface defect.

**The one correctness surprise** is from live use (Item 2 below): the spine-skill deep-links (orient / align / strategize / execute and the ⌘K skill actions) type `/orient` into a bare **bash**, which errors — they never launch Claude Code. The deep-link *plumbing* is sound; the *payload* is wrong. This downgrades the otherwise-met "panels deep-link into a terminal" feature to **partial/broken-for-purpose** until fixed. Nothing needs ripping out.

---

## 2. Compliance table

Each requirement → met / partial / missing + one-line evidence.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | At-a-glance landing head (trunk, branch+question, frontier, trust, recent) | **MET** | Overview = live-head: hero counts, blast alert, trust ledger, trunk, current thread, frontier, "tried", recent runs, latest plots. `indexer.build_snapshot` reuses `render/live_head.compute_head` (single source of truth) |
| 2 | Trunk/trajectory PINNED, always visible | **MET** | `TrunkBar.tsx` header (TRUNK ▸ CURRENT ▸ LAST-MEASURED) on every route — confirmed on Overview/Runs/Terminals/Substrate shots |
| 3 | Visualisation-as-we-go (live, auto-surfaced, tagged, newest-first, no path-hunting) | **MET** | Stream view + Overview "Latest plots"; manifest tags each plot by run + caption + ts; SSE `stream` channel wired to the `render/stream` watcher; emitters = instruments via `stream_add` |
| 4 | Multiple browser terminals; focused work in-terminal; panels deep-link in | **PARTIAL** | Multi-PTY verified (round-trip `6*7=42`, persists across reattach, resize works, `+new`); dock on every view. **BUT** deep-link payload is broken (Item 2) and terminals are *tabbed*, not tiled (Item 1) |
| 5 | Beautiful, cohesive, dark, dense (final polish = W-C, deferred) | **MET** | Visually confirmed across 10 routes; W-C Claude-Design pass correctly deferred per §6 |
| 6 | BOUNDARY: read-model + write-only control plane (never writes the store) | **MET** | Grep of all backend writes → only `intents.py:50` → `queue/`; `terminals.py:143` writes a PTY fd, not files. Delete `webapp/` → lose nothing but the cache |
| 7 | Decks: claims, working-preferences, meetings, literature | **PARTIAL** | research (13) + preferences (5) decks present & rendered; **meetings + literature absent** (no dir, not in `indexer`/`types`) — 2 of 4 |
| 8 | Goal tree explorable "for ideas" + history of everything | **MET** | GoalTree: full nested thesis→…→question tree, runs attached, roll-up counts, node detail panel; Runs = full timeline |
| 9 | Frontier = promising/next, RANKED by a strategist pass (Q9) | **PARTIAL** | Inventory + revivable nodes + deep-links present, but **explicitly UNRANKED** in the UI ("no /strategize pass yet · Rank it →"). No ranking artifact is produced or consumed |
| 10 | Claim cards: trust stamp + auto blast-radius + supersession edges + auto-draft→confirm | **PARTIAL** | trust stamp ✓, live-recomputed blast ✓, supersession edges ✓ (data populated + rendered `Claims.tsx:202-208`); **auto-draft→human-confirm WRITE flow MISSING** — UI is read-only |
| 11 | Substrate versioning + automatic blast-radius ("s001–s066 = one line") | **MET** | Substrate view: registry + version history + recomputed blast (propagator 2.0.0 → **104 runs / 4 claims**; surrogate → 10; scorer → 0) + affected-run chips |
| 12 | Glossary surfaced (canon vocab + blocked-synonym lint) | **MET** | Glossary view: canon cards, blocked synonyms (e.g. Filter blocks "discriminator"), coined-in provenance, related-terms, search. (Lint *enforcement* is a separate hook) |
| 13 | "Do whatever I need" control surface (command palette, actions) | **PARTIAL** | ⌘K palette over all objects + spine skills (deep-links to terminal) ✓; intent queue ✓; but UI-initiated intent actions not wired (`postIntent` is dead code) and skill deep-links are broken (Item 2) |
| 14 | Telegram-lite (same invocable skills, async) | **MISSING** | Not built (Phase-4-last, deferred per §6) |
| 15 | Control-plane intent queue actually CONSUMED by a headless executor | **MISSING** | Queue *write* works; **no consumer exists** anywhere (`grep` confirms none outside node_modules) — Phase-2/future |
| 16 | Live updates when underlying files change | **MET** | Verified: `touch` a record → SSE `store` frame with a new rev (`1756…→dac3…`), no git content change; `watchfiles` watches store+stream+queue; client refetches on rev change |

**Score:** 9 MET, 5 PARTIAL, 2 MISSING. The two MISSING and most PARTIALs are out-of-Phase-4-scope engine work (§6 phases 2–3); the in-scope defects are Items 1–2 and the fix list below.

---

## 3. Fix list — bugs/gaps in what was *built* (prioritized)

### Current bugs (will manifest in normal use)

| Sev | Issue | Location | Detail / fix |
|---|---|---|---|
| **HIGH** | **Skill deep-links type into bash, not Claude Code** | `TermSession.tsx:62` + all `openTerminal("/…\n")` callers | See **Item 2**. Breaks orient/align/strategize/execute + ⌘K skill actions. |
| **MED** | SSE "live" indicator never resets to false | `lib/store.tsx` (`connected` only set `true`) | Set `connected=false` in `EventSource.onerror`/on close. Cosmetic-misleading (data still recovers on reconnect). |
| **MED** | Empty duplicate-spawn guard | `terminal/TerminalsProvider.tsx:34-37` | `if (creating.current) {}` has an empty body — no early `return`. Rapid double-click on a deep-link spawns two PTYs. |
| **MED** | `fafo` palette entry is a non-existent skill | `shell/CommandPalette.tsx:41` | Selecting it types `/fafo` → errors (skill designed but not built). Remove until `fafo` exists. |
| **LOW** | Zombie on explicit terminal close | `backend/terminals.py:110-131` (`close()`) | `close()` removes the reader then `SIGHUP`+`os.close(fd)` but **never `waitpid`**. The EOF-reap fix (`_reap`, lines 175-196) covers shell self-exit but **not** the explicit-DELETE path → unreaped child until server exit. Add `os.waitpid(pid, WNOHANG)` in `close()`. |

### Latent (no crash on current data — harden *before* the auto-draft claim flow lands)

The indexer spreads raw JSON (`**c`/`**p`/`**s`), so optional schema fields reach the views verbatim. Current records all populate them (the claim schema even *requires* `trust_stamp`+`scope`), so nothing crashes today — but a future draft/partial record will throw "cannot read property of undefined":

| Sev | Unguarded access | Location | Field |
|---|---|---|---|
| LOW (schema-guarded) | `c.trust_stamp.confidence`, `c.scope.kind/.N` | `views/Claims.tsx:149-153` | `trust_stamp`, `scope` |
| LOW | `p.serves.length` / `.map` | `views/Overview.tsx:148`, `views/Pipelines.tsx:40,89` | `serves` (`serves_titles` is synthesized & safe) |
| LOW | `[...c.versions]` | `views/Substrate.tsx:71` | `versions` |
| LOW | `a.term.toLowerCase()` | `views/Glossary.tsx:31` | `term` |

Fix: add `?.` / `?? []` / `?? {}` defaults. Highest priority once auto-drafted claims (Item #10) start landing, since drafts are the records most likely to omit fields.

### Nits

- `App.tsx:182-187` `useKeyHandler` effect has no deps array → re-binds the keydown listener every render (churn, not a leak).
- `views/Overview.tsx:255` keys plot items by `file` alone (Stream.tsx correctly uses `file+ts`) → React key collision on re-emitted plots.
- Plot-stream fetches swallow errors silently (`Overview.tsx:242`, `Stream.tsx:27`) → manifest 500 shows "no plots" instead of an error.
- `app.py:41` `CORSMiddleware allow_origins=["*"]` — harmless (bound to `127.0.0.1`) but unnecessary.

### Confirmed clean (no action)

- **SPA route shadowing:** `/api/*`, `/stream/*`, `/ws/*` are all registered **before** the `/{full_path}` catch-all (`app.py:250`), so Starlette matches them first. All 11 routes incl. `/runs/:id` resolve.
- **The 3 previously-claimed fixes are all genuine:** intent-id collision (`intents.py:40-42`, ms-ts + process counter), dead glossary link (`App.tsx:33` + route 149 + in-app links), PTY EOF reap (`terminals.py:175-196`). *(The explicit-close zombie above is a fourth case the EOF fix didn't cover.)*
- **UI numbers compute correctly:** blast counts (104/10/0 runs, 4 claims) match the live snapshot; 148 total runs. (An earlier "154" was a screenshot misread.)

---

## 4. Issues reported from live use

### Item 1 — Terminals open as tabs; want **auto-tiling** (Hyprland-style), all visible at once

**Symptom (reported):** clicking to open a terminal adds a new **tab**; switching terminals means flicking between tabs. Desired: multiple terminals **tiled** and visible simultaneously, auto-tiling Hyprland-style, with a nice spawn/close animation.

**Root cause (current design — works as built, just not the desired model):**
- `TerminalDock.tsx` is a **single-pane tabbed** dock. The tab strip is `TerminalDock.tsx:25-57`; all sessions are mounted stacked (`absolute inset-0`, `TerminalDock.tsx:94-98`) but only the active one is shown — `TermSession.tsx:102` toggles `display: active ? "block" : "none"`.
- Layout control today is only **height** (`bar → half → full`, `TerminalDock.tsx:9-14,59-76`), not **split**.

**Why this is the easy version of the change:** the hard plumbing already exists. Each `TermSession` runs its own `FitAddon` + a `ResizeObserver` (`TermSession.tsx:43-49,76-79`) that re-fits xterm and pushes the new size to the PTY (`onResize → {t:"size"}`, lines 72-74). So **any** container geometry change already propagates correctly to the backend PTY — tiling is a layout/rendering change, not a protocol change.

**Proposed fix (size: M):**
1. Replace the "only-active-shown" rendering with a **tiling layout** of all sessions. Two viable layout engines:
   - *Simple grid (fast win):* CSS grid with cols/rows derived from session count (1→1×1, 2→2×1, 3→2×2 master+stack, 4→2×2, …). `grid-template` recompute on session add/remove.
   - *Hyprland feel (BSP/dwindle):* a binary-split tree where each new tile splits the focused tile along its longer axis (dwindle), with an optional master-stack mode and `mod+hjkl` focus movement.
2. **Animation:** wrap tiles in a layout-animation primitive (Framer Motion `layout` + `AnimatePresence`) or animate the grid with CSS transitions. **Caveat:** xterm's canvas does not reflow smoothly *during* a transition — animate the container, then call `fit()` on `transitionend`/animation-complete to avoid a stretched-glyph frame. (The `ResizeObserver` will also catch the final size.)
3. Each tile gets its own header strip (id/title + close + a "split" affordance). Keep the existing close-on-empty → `dock:"hidden"` behavior.
4. Keep the height toggle; optionally add a **layout toggle** (tiled ↔ tabbed) so the tab model remains available for narrow screens.
5. Focus model: clicking a tile focuses its xterm (`termRef.current?.focus()`, already in `TermSession.tsx:96`); the "active" concept becomes "focused tile" rather than "the only visible tab".

**Touch points:** `TerminalDock.tsx` (layout), `TerminalsProvider.tsx` (add a `layout`/split-tree to context state), `TermSession.tsx` (drop the `display:none` gate; keep fit-on-resize). Backend unchanged.

---

### Item 2 — The **orient** button (and all skill deep-links) type `/orient` into bash instead of starting Claude Code  ⟶ **HIGH-severity bug**

**Symptom (reported):** clicking **orient** (top bar, next to Terminal) opens a terminal and types `/orient`, which executes in the shell and returns `No such file or directory` — because Claude Code isn't running. Desired: the button should **start Claude Code via the `cc` function**, then run `/orient` inside it.

**Root cause (verified):**
- Deep-links call `openTerminal("/orient\n", "orient")` and friends — e.g. `TrunkBar.tsx:84`, `Overview.tsx:91`, `GoalTree.tsx:345`, `Frontier.tsx:273`, `Pipelines.tsx:202`, and the ⌘K palette `CommandPalette.tsx:43` (`action: () => openTerminal(\`/${s}\n\`, s)`).
- `openTerminal` stores that string as `initialInput` (`TerminalsProvider.tsx:40`), and `TermSession.tsx:59-63` sends it to the PTY **350 ms after `ws.onopen`**.
- The PTY is a bare login shell — `terminals.py:68` execs `bash -l`. So `/orient` is handed to **bash**, which tries to run the path `/orient` → `No such file or directory`. **This is true for every spine-skill deep-link and every ⌘K skill action**, so the headline "panels deep-link into the terminal" feature is currently non-functional for its main purpose.

**The fix the user wants:** launch Claude Code with the `cc` function, then issue the slash command. `cc` is a bash **function** (`~/.bashrc:36`): `cc() { claude --dangerously-skip-permissions "$@"; }` — available in the `bash -l` PTY, and it applies `--dangerously-skip-permissions` (the user's explicit preference).

**Verified:** `claude [options] [prompt]` "starts an interactive session by default" and resolves skills via `/skill-name` (per `claude --help`; `--disable-slash-commands` exists precisely to turn that off). So a prompt of `/orient` passed as the initial argument runs the skill *and* leaves the session interactive — exactly what a deep-link wants.

**Proposed fix (size: S) — three options, A recommended:**

- **A. Single combined command (recommended).** Change the deep-link payload from `/orient\n` to:
  ```
  cc "/orient"\n          # → claude --dangerously-skip-permissions "/orient"
  ```
  One send, no timing games; Claude Code boots and runs `/orient` as its first message. For parameterized links: `cc "/align <node>"\n`. *Verify once* that an initial `/skill` arg fires the skill in your build (help text and the `--disable-slash-commands` flag strongly imply yes).
- **B. Two-stage send (robust fallback).** Send `cc\n`; wait for a Claude-Code boot marker in the PTY byte-stream (more reliable than a fixed delay), then send `/orient\n`. Use this only if A turns out not to auto-run the initial slash command.
- **C. Backend-typed launch (cleanest separation).** Distinguish a plain "+ new" shell from a **skill terminal**: `openTerminal` passes a `skill` flag; the backend execs the child as `bash -lc 'cc "$@"' _ "/orient"` directly. Keeps the launch convention server-side and out of every call-site.

**Recommended concrete change (Option A, centralized):**
1. Give `openTerminal` a typed `skill`/`command` parameter (or a small helper `launchSkill(name, arg?)`) so the `cc "/…"` convention lives in **one** place instead of being re-spelled at every call-site.
2. Update the call-sites to pass the skill name only (`launchSkill("orient")`, `launchSkill("align", nodeId)`), and `CommandPalette.tsx:43` likewise.
3. Bump the `setTimeout` in `TermSession.tsx:62` from a fixed 350 ms to "send on first prompt seen" if cold-start latency proves racy (the box can be slow). Optional.
4. Remove the `fafo` palette entry (non-existent skill) while you're here.

**Why HIGH:** this is the difference between the dashboard's control surface (req #4, #13) working and not working. It's a small change with outsized impact.

---

## 5. Build list — remaining work, grouped by phase

Sizes are rough: **S** ≈ hours, **M** ≈ a day-ish, **L** ≈ multi-day.

### Phase 2 — engine (mostly out of Phase-4 *surface* scope, but these unblock the PARTIALs)
- **Headless executor consuming `queue/`** (req #15) — daemon reads intents → dispatches headless Claude Code → writes run records. **L.** Keystone: reqs #10 and #13 write-flows depend on it.
- **Strategist frontier RANKING** (req #9) — a `/strategize` pass that writes a ranked-frontier artifact the indexer reads. The UI already exposes the "Rank it →" seam. **M.**
- **Governor + auto-strategy-review** + dead-branch revival as a ranked pass (§4). **M.**

### Phase 3 — external decks (out of Phase-4 scope)
- **Meetings + literature decks** (reqs #7, #10) — schema + store dirs + 2 views + ingest skills + auto-extract→approve + gap-as-trunk-artifact. **M.**

### Phase 4 — surface completion (in scope, deferred by design)
- **Claim auto-draft→human-confirm write flow** (reqs #10, #13) — wire a confirm button → `POST /api/intent {confirm_claim}` (the intent kind + `postIntent` already exist as dead code); needs the executor (Phase 2) to apply. **S–M.**
- **Telegram-lite** (req #14). **M.**
- **W-C Claude-Design polish pass** (req #5 final). **M.**

### UX / bug batch (anytime, independent of phases)
- **Item 2** — skill deep-links launch `cc "/skill"`. **S.** *(do first — small, high impact)*
- **Item 1** — auto-tiling terminal layout + animation. **M.**
- The MED/LOW fixes from §3 (connected-flag, double-click guard, zombie-on-close, optional-field guards, `fafo` removal). **S.**

---

## 6. Consolidated priority order (suggested)

1. **Item 2** (deep-links launch `cc "/skill"`) — small, unbreaks the control surface. *(S)*
2. **§3 MED batch** (connected-flag reset, double-click guard, zombie-on-close, drop `fafo`). *(S)*
3. **Item 1** (terminal auto-tiling + animation). *(M)*
4. **Optional-field hardening** (§3 latent) — do alongside, and before #6 below. *(S)*
5. **Frontier ranking** (req #9 — the UI seam is ready). *(M)*
6. **Headless executor** (req #15) → then **claim auto-draft→confirm** (req #10). *(L → S–M)*
7. **Meetings/literature decks** (req #7), then **Telegram-lite** (#14), then **W-C polish** (#5). *(M each)*

Nothing in the build is mis-scoped: the previous agent's "known-incomplete" list is accurate, and the surface correctly stops at the read-model + write-primitive boundary that Phase 4 calls for.

---

# Part II — Implementation pass (2026-06-06)

> Executed the §6 priority order. Branch `forward_survey`, 4 webapp commits + this doc:
> `48690c4` (Item 2 + MED/latent batch) · `120a72b` (Item 1 tiling) · `463c2f2` (control plane) · `d82b8ee` (frontier ranking). Every change verified against the live app on `:8138` (PTY round-trip, screenshots of every touched route, intent-queue write trace).

## 7. Status update — what's now MET

| # | Requirement | Before | Now | What changed |
|---|---|---|---|---|
| 4 | Multiple terminals; panels deep-link in | PARTIAL | **MET** | Deep-links boot Claude Code via `launchSkill`→`cc "/skill"` (verified: `cc` resolves in the PTY, `cc "/orient"` boots CC); terminals now **auto-tile** (all visible) with a tiled⇆tabbed toggle + spawn/close animation. |
| 9 | Frontier RANKED by a strategist pass | PARTIAL | **MET (surface+producer)** | End-to-end ranking: `/strategize` emits a *derived* `render/frontier_ranking.json`; the indexer reads it (+ stale-vs-rev tag); a `ranked` lens renders #-rank + cited rationale/crux/cost, with a stale banner. Turns on the moment `/strategize` runs; honestly unranked until then. |
| 10 | Claim auto-draft → human-confirm write flow | PARTIAL | **MET (write side)** | Draft claims get a `confirm → live` action → `POST {confirm_claim}` with queued-feedback. The *apply* (draft→live) is the executor's job (#15, Phase 2) — by design the backend never flips the claim itself. |
| 13 | "Do whatever I need" control surface | PARTIAL | **MET** | `postIntent` is no longer dead: new **Control** view (queue + composer) + nav badge; confirm-claim, pin-frontier (`pick_frontier`), note, and run-skill intents all write to `queue/`. |
| 3 | Visualisation-as-we-go | MET | MET (hardened) | stream-manifest fetch errors now surface; Overview plot key is `file+ts` (no React key collision). |
| 6 | BOUNDARY (read-model + write-only queue) | MET | MET (re-verified) | Re-traced after every backend change: the only store-touching write is still `intents.py → queue/`. The new `frontier_ranking.json` is **read-only** to the backend (written by the skill). Removed the unused wildcard CORS. |

**Deferred (unchanged, Phase 2/3 engine — see §8):** #7 meetings/literature decks (PARTIAL), #14 Telegram-lite (MISSING), #15 headless executor (MISSING). #5 W-C design polish still deferred by design.

**New score:** **13 MET · 1 PARTIAL (#7) · 2 MISSING (#14, #15).** Every remaining gap is engine work that crosses the whole OS, not a surface defect.

### Bugs fixed from §3
All of §3 is done: deep-link payload (Item 2, HIGH); SSE `connected` resets on error; double-click guard early-returns; explicit-close zombie reaped via a non-blocking `waitpid(WNOHANG)` poll (DELETE made `async` so the reaper runs on the loop thread); `fafo` removed; all latent optional-field accesses guarded (Claims/Overview/Pipelines/Substrate/Glossary/CommandPalette); `useKeyHandler` binds once; plot-key collision; swallowed stream errors; wildcard CORS dropped.

### Improvements made on my own initiative (beyond the review)
- **Deep-link timing made robust:** instead of a fixed 350 ms, the deep-link payload flushes on the **first PTY output** (the prompt actually appeared) with a 1.5 s fallback — fixes the cold-start race on a slow box the review flagged as a risk.
- **Control plane made *visible*, not just writable:** the review only asked to wire `postIntent`; I added the whole Control view + nav badge so a queued intent is observable (you confirm a claim → you *see* it queued), which is what makes req #13 feel real.
- **`fit()` rAF-debounced:** the tiling animation resizes the container ~60×; coalescing to one fit per frame avoids a flood of PTY resizes + the stretched-glyph flicker the review warned about.
- **Frontier staleness:** the ranking self-reports when it predates the current store, so a ranked frontier can't quietly mislead after a run lands.

## 8. Ideas & recommendations (my own thinking + what's next)

The surface now does its §0/§5 job well: trunk pinned, eyes-as-validation (stream + ranked frontier), focused work in tiled terminals, control plane legible. The remaining value is in the **engine** behind it. Concrete proposals, in build order:

### 8.1 Headless executor (req #15) — the keystone, but an autonomy decision (surface first)
This is the consumer the whole control plane feeds. I deliberately did **not** build the auto-run daemon: a daemon that runs `claude --dangerously-skip-permissions` off a queue is the central autonomy commitment of the OS (PLAN §8's "everything else auto-chains") and is genuinely Girish's call. Proposed shape when blessed:
- `research_os/loop/executor.py`: watches `queue/`, dispatches per kind, **serialises** CPU-heavy `run_skill:execute`/`bless_contract` runs (the no-stacking rule, Q5), runs in a tmux session so the operator can watch.
- **Boundary stays intact:** the executor is a *spine-skill driver*, not a store writer — `confirm_claim` is applied by dispatching `/close` (which owns claim writes), never by the daemon mutating `claims/` directly. It marks intents `done` in `queue/` only.
- **Safety gate:** an allowlist of dispatchable skills + a dry-run mode; escalations (gate fail / off-contract) route back to the human (phone/web).
- *Decision to make:* auto-run with the 4 PLAN gates, vs human-confirm each dispatch, vs keep manual (run-in-terminal) for now. My lean: **human-confirm each dispatch** for the first version (the queue is already the inbox), then graduate to auto-run once calibrated.

### 8.2 Meetings + literature decks (req #7) — Phase 3, cheap to scaffold
`meeting_card` / `literature_card` schemas + `meetings/` `literature/` store dirs + two read-only views + nav, mirroring the claim decks. The existing Robinson & Frueh 2025 card is the first literature row. Auto-extract→approve (transcript → cards) needs the executor; the *read surface* does not. I left it out to avoid empty-deck nav clutter, but it's a half-day once there's a card to show.

### 8.3 Telegram-lite (req #14) — after the executor
A thin bot (writes only `queue/` intents + relays escalations and plot-stream images) reuses the exact boundary the web control plane already uses. It is essentially "the Control view as a bot," so it should follow the executor.

### 8.4 Smaller, high-leverage surface ideas (cheap; I'd do these next)
- **⌘K → enqueue, not just navigate:** let the palette post a `run_skill` intent (async) in addition to launching a terminal — one keystroke to queue work.
- **Keyboard tiling focus** (`mod+hjkl` between tiles, `mod+w` close) — the tiling is there; wiring focus movement makes it feel like the wm it resembles.
- **Blast-radius → one-click `needs_replication` triage:** from the Substrate blast panel, queue a re-replication intent per exposed claim (directly attacks the binding constraint, foundation rot).
- **Code-split the bundle:** it's 575 KB (one chunk); route-level `lazy()` would cut first paint. Cosmetic, not urgent.
- **A `friction-log` affordance** in the trunk bar: capture friction the instant it happens → `note` intent (PLAN §9 `friction-log` skill, but the capture point belongs in the surface).

## 9. Adversarial diff review (24 agents, 2026-06-06)

After the build, a multi-agent workflow reviewed the whole diff (`e180f4b..HEAD`) across 5 lenses — boundary, backend-async, react-ts, spec-ux, edge-cases — with **per-finding adversarial verification** (each candidate bug re-checked against the actual code, default-refute). **5 confirmed, 14 rejected** as false positives (e.g. "`Claim.statement` should be optional" — already non-optional; "`tileGeometry` n=0" — never called with 0; "double-close race on unmount" — the dock never unmounts; "`openTerminal` guard not reset on API failure" — it's in a `finally`). All 5 confirmed are fixed in `d9066ee`:

| Sev | Finding | Fix |
|---|---|---|
| MED | Zombie when a terminal is closed **before it's ever attached** (`self._loop` only set on attach) | `_reap_pid` falls back to `asyncio.get_running_loop()` (DELETE is async). Verified: create+DELETE-without-attach → no defunct child. |
| LOW | `store_revision()` (glob+stat+sha1 over the store) ran on the event-loop thread in the watcher | moved to `asyncio.to_thread` |
| LOW | A ranking missing `generated_for_rev` was shown as **fresh** | now treated as **stale** (can't prove freshness) |
| LOW | `ConfirmDraft` had no double-click guard → could post two `confirm_claim` intents | ref-guard (stale-closure-proof) |
| LOW (kept) | `store.tsx` intents-fetch swallows errors | **deliberately kept** keep-last-known; the suggested wipe-to-empty would falsely show "0 queued" on a transient blip |

**Boundary re-verified by the dedicated agent:** the only store-touching backend write remains `intents.py → queue/`; `frontier_ranking.json` is read-only to the backend. Q1/Q11 intact.

---

*End of implementation pass. Surface complete to the read-model + write-primitive boundary; the engine (executor / decks / telegram) is the next, separately-blessed, phase.*

