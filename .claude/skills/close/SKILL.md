---
name: close
description: "End-of-session skill for the LCAS inversion loop — the WRITE end. Writes the session's run_record JSON (canonical state) into research_os/records/, links the prose experiments/sXXX_*.md writeup, updates the goal node's authored fields + refreshes derived caches, auto-drafts any claim-card changes for your confirmation, validates the store stays referentially closed, demotes PROGRESS.md to an auto-printout, commits on forward_survey (explicit paths, size gate, no Co-Authored-By), then emits a ~100-word handoff. Use when the user says wind down, wrap up, close out, hand off, end the session, let's stop here, or any signal we're closing. Run BEFORE compaction. Replaces the older `wind-down` ritual; pairs with `orient`."
---

# Close — LCAS inversion loop (Research OS, branch forward_survey)

The write end of the loop. Where `orient` *renders* the trust store, `close` *writes*
to it. The canonical state is now the **run_record JSON** under `research_os/records/`
— NOT the PROGRESS prose. PROGRESS.md is demoted to an auto-generated printout. The
prose `experiments/sXXX_*.md` writeup stays, as linked nuance.

> **The shift from `wind-down`.** wind-down authored the PROGRESS top-banner as the
> canonical handoff. close makes the **run_record** canonical; PROGRESS becomes a
> printout of the live-head. Narrative nuance lives in the record's `narrative_md` +
> the linked `.md` writeup. Don't hand-author PROGRESS prose anymore.

Active research workspace: `notebooks/inversion/survey/` on `forward_survey`. If the
session ran no experiment (pure analysis/replanning), skip steps 1–2 and 4; still do
the node `last_measured` touch if state changed, the log line, and the commit.

When the `execute` skill runs (it writes a record per executed run), close mostly
*verifies* those records and does the session-level rollup (steps 3–8). For a session
that ran no `execute` (pure analysis / a legacy hand-run), close writes the record itself.

## Step 1 — Write the run record (the canonical object)

Pick the next id (continues the experiment sequence; never reuse):
```bash
ls research_os/records/ notebooks/inversion/survey/experiments/ \
  | grep -oE '^s[0-9]+' | sort -V | tail -1          # highest existing -> increment
python research_os/render/live_head.py --json | python3 -c \
  "import json,sys;print('substrate heads:',json.load(sys.stdin)['trust']['substrate_heads'])"
```
Write `research_os/records/sXXX_<slug>.json` to the run_record schema
(`research_os/schemas/run_record.schema.json` — validate.py enforces it). The `id` must
match `^s[0-9]+[a-z]*(_[a-z0-9]+)*$` — **lowercase** slug segments only (e.g.
`s113_anchor_cap_clean_sweep`, not `s113_Clean_Sweep`). Template:
```json
{
  "schema_version": "1.0.0",
  "id": "sXXX_<slug>",
  "kind": "run_record",
  "goal_node": "goal_<branch-this-attaches-to>",
  "question": "<the uncertain question this run answered>",
  "hypothesis": "<what you expected>",
  "status": "confirmed | refuted | inconclusive",
  "run_type": "probe | cheap | expensive | batch",
  "seeds": [119],
  "N": 1,
  "claim_scope": "N1 | cohort",
  "metrics": { "rho": 0.03, "rho_band": "A", "q0_err": 0.57, "w_dir_err": 0.57, "w_mag": 1.5025, "wall_s": 47 },
  "artefacts": [ { "path": "results/sXXX/...", "kind": "checkpoint|plot|anim|data|report|log" } ],
  "writeup_refs": ["notebooks/inversion/survey/experiments/sXXX_<slug>.md"],
  "parents": ["<prior run id(s) this builds on>"],
  "substrate_versions": { "propagator": "2.0.0", "surrogate": "v2", "scorer": "1.0.0" },
  "gates_passed": [], "gates_failed": [],
  "oracle_clean": false,
  "created_at": "YYYY-MM-DDT00:00:00Z",
  "narrative_md": "1–3 sentences: headline finding + load-bearing number + architectural consequence."
}
```
Honesty rules that the gates will check (state them truthfully now):
- **`oracle_clean: false`** if any `truth_*` value was read during the run (seeded from
  an oracle-nearest pair, q held at truth, scored against truth). Only genuinely-blind
  runs are `true`. Most diagnostic probes are `false` — that is fine, just honest.
- **`claim_scope`/`N`** — `cohort` requires N ≥ the configured threshold; one seed is `N1`.
- **`metrics`** — always include the three errors (`q0_err`, `w_dir_err`, `w_mag`) and
  `rho`/`rho_band` when a fit happened; `wall_s` in **seconds**; `|ω|` in **deg/s**.
- **`substrate_versions`** — copy the live heads from the command above. A run on a bumped
  component is what makes blast-radius resolve later.

## Step 2 — Keep the prose writeup (linked nuance)

Write `notebooks/inversion/survey/experiments/sXXX_<slug>.md` with the standard
frontmatter (`title / type / sources / related / created / updated / confidence`) and
sections (TL;DR · What · How · Result · Why this matters · Numbers · Artefacts · Out of
scope · Cross-references). All numbers cited `(source: path:line)`. The JSON is canonical
*state*; this `.md` is the prose nuance the record's `writeup_refs` points at and that
`narrative_md` distills. Don't duplicate the whole writeup into `narrative_md`.

## Step 3 — Update the goal node (AUTHORED fields only) + refresh derived caches

Edit ONLY the authored fields on `research_os/goals/goal_<branch>.json`:
- **`last_measured`** `{run, summary, at}` — set if this run is the branch's new headline.
- **`state`** — flip `open → closed` if the branch question is answered, `→ blocked` if
  it hit a dependency, `→ revivable` if shelved-but-reopenable.
- If this run is the **trunk** headline, also set `last_measured` on the thesis root
  `goal_attitude-inversion.json`.

Do **NOT** hand-edit `child_runs` or `spent` — they are derived. Recompute them:
```bash
python research_os/loop/refresh.py --date YYYY-MM-DD   # today; fills child_runs/spent/_index from records
```

## Step 4 — Auto-draft claim-card changes (→ your confirmation)

If the run bears on a claim, draft the change and **present it for explicit confirmation
before finalizing** (claim writes are auto-draft → human-confirm):
- **New finding** → new `research_os/claims/<deck>/claim_<slug>.json`, `status: "draft"`,
  `supporting_runs: [sXXX]`, `depends_on` = the substrate heads it rests on, scope/N.
- **Confirms an existing live claim** → append `sXXX` to its `supporting_runs`.
- **Refutes one** → append to `refuting_runs`; propose a `status` flip (`live →
  needs_replication` or a supersede).
- **Supersedes one** → new card + set `superseded_by` on the old (validate fills the
  reciprocal `supersedes`).

Show the diff; finalize only on a yes. Respect the standing blast-radius rule: a finding
whose evidence is pre-fix-only (`propagator@1.0.0`) is `needs_replication`, not `live`.
(Substrate version bumps themselves are the Stage-4 `blast-radius` hook's job — if you
bumped a component this session, flag it.)

## Step 5 — Validate (the gate — must pass before commit)
```bash
python research_os/loop/validate.py
```
Exit 0 ("store is referentially closed") is required. If it errors, fix the object you
just wrote — a dangling `goal_node`, `parents`, `depends_on`, or an unset `superseded_by`.

## Step 6 — Demote PROGRESS to a printout + append the log line

PROGRESS.md is no longer authored prose. Maintain a delimited auto-block at the very top
(insert it on first close, replace it thereafter); everything below stays frozen history:
```bash
{ echo '<!-- LIVE-HEAD · auto-generated by close from research_os/ · do not edit -->'; \
  echo '```'; python research_os/render/live_head.py; echo '```'; \
  echo '<!-- /LIVE-HEAD -->'; } > /tmp/livehead.md
# then replace the region between the markers in PROGRESS.md with /tmp/livehead.md
```
Append one line to `notebooks/inversion/survey/log.md` (append-only ledger; do NOT read
the whole file):
```
## [YYYY-MM-DD] ingest | sXXX | <one-line headline number + architectural consequence>
```
Op vocab: `ingest` (experiment landed) · `concept` · `lint` · `infra` · `audit`.

## Step 7 — Commit on forward_survey (explicit paths + size gate)

```bash
git status --short && git diff --stat
```
Three-filter gate (never `git add -A`):
- **Size** — any file >100MB (104857600 B) hard-skipped (`stat -c %s <f>`).
- **Type/location** — skip `results/**/*.npz|*.npy` (except small <10MB summary NPZs),
  `*_ckpt.npz`, anything under `data/results/inversion_diagnostics/`, `__pycache__`, scratch.
- **Stage explicitly** — the new objects + writeup + printout:
```bash
git add research_os/records/sXXX_<slug>.json \
        research_os/goals/goal_<branch>.json research_os/goals/_index.json \
        research_os/claims/<deck>/claim_<slug>.json \
        notebooks/inversion/survey/experiments/sXXX_<slug>.{py,md} \
        notebooks/inversion/survey/PROGRESS.md notebooks/inversion/survey/log.md
```
(Include only the goal/claim files you actually changed — `refresh.py` may have touched
several goal nodes' `spent`; stage those too if you want the cache committed.) Commit with
a HEREDOC, repo prefix style (`ingest:`/`infra:`/`docs:`/`validate:`), subject carrying the
load-bearing number, and **no `Co-Authored-By`** (hard rule):
```bash
git commit -m "$(cat <<'EOF'
ingest: sXXX — <short subject with the headline number>

<optional body, ≤6 lines: cite the writeup path + key numbers>
EOF
)"
git rev-parse --short HEAD     # feed the hash into last_measured if you cited it
```
Print: the hash + subject; what was staged (one path per line); what was skipped and why.

## Step 8 — Handoff prompt (~100 words, a delta on `orient`)

Emit a final `## Handoff prompt for next agent` block. Three parts: **where we are**
(branch + the run that landed + headline number), **what's open** (the current question),
**suggested next move, lightly hedged** ("natural next chunk: X, but open question Y might
be higher-leverage — run `/orient` and pick"). The next agent will run `/orient`; this is
the delta on top, not a replacement. A 60-word handoff is fine; skipping it is not.

## What NOT to do

- **Don't author PROGRESS prose.** It's a printout now (step 6). Nuance → `narrative_md` + the `.md` writeup.
- **Don't hand-edit `child_runs` / `spent`** — derived; `refresh.py` owns them.
- **Don't finalize a claim card without explicit confirmation.** Draft → show → yes → finalize.
- **Don't skip `validate.py`.** A broken store committed is the rot this system exists to prevent.
- **Don't `git add -A` / `git add .`** — explicit paths only; the size gate is void otherwise.
- **No `Co-Authored-By`.** Don't push (the user pushes; large-file blockers exist).
- **Don't read PROGRESS.md / log.md in full** — they're hundreds of KB. Edit small slices; append one log line.
- **Don't write an experiment record/writeup if no experiment ran.** Honest absence > ceremony — a `lint`/`audit` log line + the commit is enough.
