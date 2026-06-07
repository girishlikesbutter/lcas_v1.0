---
name: dynamic-viz
description: "Visual instrument of the Research OS — decides WHAT to draw for a research artifact (a result/run, a claim/trust state, the corpus, the frontier, a contract/proposal, a before-after), draws the minimal glance that makes it land, and emits it to the browser plot-stream via stream_add. Reads every number live from the store (records/claims/goals/substrate) — never hardcodes. Use when the user says draw/plot/visualise/chart/show me — the corpus, the frontier, the trust ledger, the claims, a run result, a cost-wall, the store status, this contract, this branch — or 'emit to the stream' / 'show me X as a glance'. NOT for light curves (use lc-compare) or attitude/omega states (use attitude-viz / attitude-anim) — those are domain instruments with their own pipelines."
---

# Dynamic-viz — the Research OS visual instrument (browser plot-stream)

The **emitter intelligence** on top of the `stream_add` primitive. Visual validation is
first-class for this operator (eyes are a validation gate, PLAN §8); the display is solved
(ADR-0003: PNGs in `research_os/render/stream/` + a polling `index.html`). What was left is
*judgment* — **deciding what to draw** for a given artifact and drawing the minimal version
that makes the point. That's this skill.

dynamic-viz is Opus high. The hard part is not matplotlib; it's choosing the **one thing the
artifact wants you to see** and refusing to draw a table when a glance will do.

## The boundary — what this is NOT

- **Light curves** → `lc-compare`. **Attitude / ω states** → `attitude-viz` / `attitude-anim`.
  Those are *domain* instruments with their own propagation pipelines. dynamic-viz is the
  *meta* instrument: it visualises **Research-OS state and research artifacts** (the store,
  the trust deck, a run's verdict, a cost-wall, a proposal's decision-space). If a domain
  instrument fits, hand off to it — don't re-draw it here.
- It does not author store objects. A plot is a *view*; the canonical state stays in the
  records/claims/goals. (Same stance as `strategize`: derived, never persisted as truth.)

## Step 1 — Name the one thing (the judgment)

Before any code, answer in one sentence: **what is this artifact trying to make the operator
see?** Then pick the smallest plot that delivers it. Heuristics:

- A **glance beats a table.** If the answer is a single number or a yes/no, one panel should
  *be* that number, big, with the framing under it (the demos all do this — see the
  binding-constraint panel in `viz_corpus_status.py`).
- **One artifact, one plot.** Don't cram four questions into a 2×2. Emit two plots instead;
  the stream is chronological and cheap.
- **The cheap-first gate applies to pictures too.** If a standing script already answers it
  (below), *run that* — don't write a new one.

## Step 2 — Pick the pattern (vocabulary → template)

| Artifact you're handed | What to draw | Start from |
|---|---|---|
| The corpus / "how's the research going" | outcome mix + the dead-work headline | `viz_corpus_status.py` |
| Claims / trust / blast-radius | status mix + which findings are stranded | `viz_trust_ledger.py` |
| The frontier / "what's on the table" | branches by staleness + trunk + blocked flag | `viz_frontier_map.py` |
| A run **result** (post-execute) | the data panel (bands/ρ/seeds/cost) + the verdict headline | `viz_s113_costwall.py` |
| A **contract / proposal** (pre-run) | the *decision space*: confirm-vs-refute outcomes, budget vs nearest-analog, the uncertain crux | (new — build it) |
| A **comparison** (A vs B, before/after) | side-by-side + a residual/delta panel | `viz_s113_costwall.py` layout |

The four `viz_*.py` scripts in `research_os/render/` are the living exemplars (the last two
rows reuse those layouts / are built fresh). For a recurring artifact,
**reuse or extend the existing `viz_*.py`**; for a genuinely new one, write a new
`viz_<slug>.py` in `research_os/render/` (reusable, re-runnable — not a throwaway snippet).

## Step 3 — Build it (conventions, all enforced by `ro_viz.py`)

Import the shared helper — it carries the dark palette that matches `index.html`, the status
colour map, and the one-call emit. A new viz is ~30 lines:

```python
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import ro_viz                          # same dir: research_os/render/

ro_viz.apply_dark_style()
recs = ro_viz.records()                # live store readers — never hardcode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
# ... ax1 = the data, ax2 = the glance/headline ...
ro_viz.emit(fig, "my_slug", "one-line caption stating the takeaway + Source: <path>")
```

- **Read every number live** from `ro_viz.records()` / `ro_viz.claims()` / `ro_viz.head_json()`
  or by reading the JSON directly. **Hardcoding a number is the one unforgivable sin** —
  CLAUDE.md "cite the file for every number"; the plot must regenerate to truth when the store
  moves. Put the source path in the caption.
- **Palette:** use `ro_viz.STATUS_COLORS[status]` for run/claim statuses; `ro_viz.BLUE/GREEN/
  AMBER/RED/MUTED` otherwise. Don't invent colours — consistency is what makes the stream
  scannable.
- **The headline panel:** `ax.axis("off")` + a big number + an italic muted framing line.
  This is the idiom that turns a chart into a *glance*.

## Step 4 — Emit and surface

`ro_viz.emit(fig, slug, caption, run="")` saves `stream/<slug>.png` and appends the manifest
(it wraps `stream_add.add`). Then make sure the operator can see it:

```bash
research_os/render/serve_stream.sh        # idempotent: starts http.server if needed, opens the tab
```

Print the saved path (CLAUDE.md output rule). Tag with the `run` id when the plot belongs to a
specific run record, so it's filterable in the stream. To start a clean stream for a new
session/branch, delete `research_os/render/stream/{*.png,manifest.json}` (gitignored cache,
ADR-0003) before emitting.

## What NOT to do

- **Don't hardcode numbers.** Read them live; cite the source in the caption. A plot that
  doesn't track the store is a lie waiting to happen.
- **Don't draw a table.** If you're plotting ten labelled bars no one will read, you've missed
  the one thing — go back to Step 1.
- **Don't rebuild a domain instrument.** Light curves → `lc-compare`; attitudes → `attitude-viz`.
- **Don't commit the cache.** `stream/*.png` + `manifest.json` are gitignored. The *scripts*
  (`viz_*.py`, `ro_viz.py`) are the durable artifact; commit those, not the PNGs.
- **Don't author a store object.** A view is not a record. Findings → claim cards (via `close`);
  pictures → the stream.

## Where things live

- **Emit primitive:** `research_os/render/stream_add.py` (`ro_viz.emit` wraps it).
- **Shared style + live readers:** `research_os/render/ro_viz.py`.
- **Exemplar scripts:** `research_os/render/viz_{corpus_status,trust_ledger,frontier_map,s113_costwall}.py`.
- **Display:** `research_os/render/stream/index.html` (served by `serve_stream.sh`).
- **Decision of record:** `research_os/decisions/0003-interim-visual-surface-browser-stream.md`.
