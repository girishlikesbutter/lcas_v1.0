# Run-record extraction contract (W-B backfill)

You convert experiment writeups into **lossy** run records. Read this whole file +
`research_os/schemas/run_record.schema.json` (the binding contract) before starting.

**Lossy-pragmatic rule:** capture metadata + outcome + a few key numbers + a 1-3
sentence narrative. Do NOT transcribe the writeup. Never invent a number — if a field
isn't in the writeup, omit it (every field except the required ones is optional).

## Inputs you are given
A list of experiment **ids** (= writeup filename stems). For each id:
- its metadata is in `research_os/backfill/run_manifest.json` (find the entry with that `id`):
  `goal_node`, `writeup_ref`, `title`, `created`, `updated`, `confidence`, `sources`, `related`.
- the prose is at `notebooks/inversion/survey/experiments/<id>.md`.

## What to read in each writeup (keep it cheap)
Read the **frontmatter + `# TL;DR` + the `# Result` / `## Numbers` section only.**
Skip `# How` unless the TL;DR is too thin to set status/metrics. That's enough for a
lossy record.

## Idempotent
If `research_os/records/<id>.json` already exists, SKIP it (do not overwrite).

## How to fill each field

- `schema_version`: `"1.0.0"`. `kind`: `"run_record"`. `id`: the stem. `goal_node`: from the manifest.
- `question`: the question the experiment answered (1 sentence — derive from title/TL;DR).
- `hypothesis`: the prediction under test, if the writeup states one; else omit.
- `status` (vs the experiment's OWN hypothesis):
  - `confirmed` — the experiment reached a decisive result for its hypothesis (incl. a decisive *negative* finding it set out to test).
  - `refuted` — the experiment's own hypothesis was falsified by the result.
  - `inconclusive` — blocked, ambiguous, broken tooling, or "needs more".
  (A run that *refutes a prior claim* but does so decisively is `confirmed` for ITS hypothesis. Use the writeup's framing.)
- `run_type`: `probe` (tiny), `cheap`, `expensive`, or `batch` — best guess from scale/wall. Omit if unclear.
- `seeds`: integer seeds exercised (from the text, e.g. [116, 119]); `[]` if cohort-wide/none named.
- `N`: sample size (len(seeds) or the cohort N stated). `claim_scope`: `cohort` if N≳20 or "cohort", else `N1`.
- `metrics`: an OPEN object — include 2-6 key numbers with the writeup's units. Common keys:
  `rho`, `rho_band` (A/B/C/D), `q0_err` (deg), `w_dir_err` (deg), `w_mag` (deg/s), `wall_s` (s).
  |ω| is ALWAYS deg/s. Copy numbers faithfully from the Result table / Numbers section.
- `artefacts`: from the manifest `sources`, keep only OUTPUT files (NOT `.py` code):
  `results/**.json`→kind `data`, `**.png`/`**.html`→`plot`/`anim`, `**.npz`→`checkpoint`.
  `{ "path": "...", "kind": "..." }`. Omit if none.
- `writeup_refs`: `[ <manifest.writeup_ref> ]`.
- `parents`: from the manifest `related` — keep entries that are `experiments/sXXX_*.md`, convert to the bare stem (strip `experiments/` and `.md`). Omit non-experiment relateds.
- `commit`: only if the writeup explicitly cites a git SHA for this run; else omit.
- `substrate_versions`: REQUIRED. Rule by era (the ω-sign fix landed 2026-05-12, commit d5705ff):
  - propagator: `"1.0.0"` if the experiment number is **≤ s066** (pre-fix, ω-sign-tainted era), else `"2.0.0"`.
    (s065/s066 diagnosed the bug → still `1.0.0`; s067+ validate/post-date the fix → `2.0.0`.)
  - surrogate: `"v2"` (default). Use `"v1"` only if the writeup is specifically about v1.
  - scorer: `"1.0.0"`.
  → `{ "propagator": "1.0.0|2.0.0", "surrogate": "v2", "scorer": "1.0.0" }`
- `oracle_clean`: REQUIRED boolean. `false` if the experiment **reads truth as an input** —
  cost/landscape *at truth*, render-at-truth, truth-ω held fixed, oracle-nearest/oracle pair seeding,
  perturbed-truth anchors, truth-centered bands. `true` for a genuinely blind/forward measurement
  that never consumes a `truth_*` value. When the writeup says "blind", lean `true`; when it says
  "at truth" / "oracle" / "perturbed truth", `false`. If unsure, `false` (conservative).
- `gates_failed`: if `oracle_clean` is false because it intentionally used truth, add `["oracle-leak"]`. Else omit.
- `created_at`: `<manifest.created>T00:00:00Z` (if created is empty, omit). `executed_at`: omit.
- `narrative_md`: 1-3 sentences, the lossy gist (compress the TL;DR; keep the key number + the verdict).

## Write
Write each record to `research_os/records/<id>.json` (pretty JSON, 2-space indent).
Create the `research_os/records/` dir if needed.

## Validate before finishing
Run: `python research_os/schemas/validate_one.py research_os/records/<id>.json run_record`
for at least your first record, fix any error, then write the rest. (If that helper
doesn't exist, validate mentally against the schema's `required` + enums.)

## Return (your final message — this is data, not prose for a human)
A compact JSON array, one object per record you wrote:
`[{"id":"...","goal_node":"...","status":"...","oracle_clean":true|false,"propagator":"1.0.0|2.0.0"}, ...]`
plus a final line `WROTE <n>, SKIPPED <m>`.
