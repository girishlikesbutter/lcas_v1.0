# Research OS — loop infra (Phase 1)

The ongoing read/write machinery the terminal loop runs against the trust store.
Distinct from `backfill/` (one-time migration, frozen `STAMP`, regenerates its own
README). These are **reusable, side-effect-disciplined** tools the spine skills and the
trust hooks call. Plain scripts, run by path from repo root; each also has importable
functions.

## Scripts

| Script | What | Mutates? | Used by |
|---|---|---|---|
| `validate.py` | schema + referential-integrity gate (knows `branch_contract`); exit 0/1 | no | `execute` + `close` (post-write gate), hooks |
| `refresh.py` | re-materialize derived caches (`child_runs`, `spent.runs/wall_s`, `_index`) from records, date-correct | derived fields only | `close` (after writing a record) |
| `blast_radius.py` | claims resting on a bug-fixed-older substrate version → `needs_replication` (safe direction) | claim `status` only | `blast-radius` hook |
| `gate_check.py` | epistemic gate on a written object. HARD: N-scope, oracle-coherent (exit 2). SOFT (`--all`/`--soft`): oracle-rests, stale-substrate | no | `gate-check` hook (hard), `close`/`periodic-review` (`--all` audit) |
| `glossary_lint.py` | blocked-synonym lint vs canon glossary. **Inert until seeded** (clean no-op now) | no | `glossary-lint` hook |
| `tool_lint.py` | **drift-check** (ADR-0007): re-resolve each Tool `entry_point` + re-hash vs `current_hash`; verdict `clean`/`missing`/`hash_moved`. `--check` exits 1 on any drift; `--for-file P` lints only the Tools an edited path touches | no | `tool-lint` hook (early warning), `run_tool.py` (pre-flight refusal gate) |
| `run_tool.py` | **the run-button** (ADR-0007 #15, smallest form): run ONE Tool on ONE input — drift-check → import `entry_point` → bind `default_params`+overrides → invoke → write a canonical `tool_run` record + gitignored artifacts + plot-stream emit | writes `tool_runs/*.json` (canonical bench record) | manual / webapp run-button |
| `run_pipeline.py` | **the DAG executor** (ADR-0007 §5.3): run a Pipeline's `steps[]` DAG — each step drift-checked + invoked via `run_tool`'s primitives, threaded by the ref language (`$in.*` / `$steps.<sid>[.key]`). Two combinators: `compose` (chain) + `map` (lift a per-epoch Tool over `over` → a set). Whole DAG refuses if any step drifts (`--force` overrides). `--dry-run` validates ref structure + drift without running | writes `pipeline_runs/*.json` (canonical DAG record) | manual / webapp (later) |

Live-head renderer is the sibling `../render/live_head.py` (read-only view).

## Trust hooks (wired in `.claude/settings.json`)

Mechanical, fire automatically (PLAN §8). Thin bash wrappers in `.claude/hooks/`
parse the hook stdin JSON, cwd-gate, path-filter (matchers are tool-name-only, so the
filter lives in-script), and call the script above.

| Hook | Event · matcher | Wrapper | Fires when | Effect |
|---|---|---|---|---|
| blast-radius | PostToolUse · `Write\|Edit` | `ro_blast_radius.sh` | path under `research_os/substrate/*.json` | flips dependent live/draft claims; reports |
| gate-check | PostToolUse · `Write\|Edit` | `ro_gate_check.sh` | path under `research_os/{records,claims}/*.json` | HARD violation → exit 2 (stderr → model) |
| glossary-lint | PostToolUse · `Write\|Edit` | `ro_glossary_lint.sh` | path under `survey/experiments/*.md` or a store `*.json` | warns on blocked synonyms (inert now) |
| tool-lint | PostToolUse · `Write\|Edit` | `ro_tool_lint.sh` | a `*.py` source file OR `research_os/substrate/*.json` | drift on a Tool the edit touches → stderr nudge (advisory, exit 0; `run_tool` is the hard gate) |
| analytical-probe | PreToolUse · `Bash` | `ro_analytical_probe.sh` | command matches a batch-launch signature | non-blocking `additionalContext` reminder (cheap-first) |

All matching hooks run in parallel. They never race: `gate-check` and `glossary-lint`
filter disjoint paths, and `tool-lint`/`blast-radius` (which both see substrate cards)
are read-only except for `blast-radius`'s claim-status flip — they touch disjoint
outputs. `gate-check` exit-2 surfaces to the model to fix the object it just wrote (the
write itself isn't undone — that's why `validate.py` is also a pre-commit gate in `close`).
`tool-lint` is advisory (exit 0): edits to bound code are legitimate; flip its final
`exit 0` → `exit 2` to make drift block-and-notify.

## Config

- N-scope threshold: env `RO_NSCOPE_MIN` (default 10).
- `blast_radius.py --check` for a dry run; `gate_check.py --all` for a full audit.

## Verified

`validate`/`refresh` idempotent on the reconciled store; a synthetic `s113` record
round-trip (write → validate → refresh → render → rollback) left 200 objects, zero
residue; the `gate-check` hook fired live in-session on a cohort/N=1 record and blocked
with `GATE-FAIL`. `tool_lint` lints 47/47 Tools clean (scour hashes still in sync) and
fires `hash_moved`/`missing` on synthetic drift; the `tool-lint` hook was live-tested via
its stdin payload — silent on clean/unbound, surfaced all 3 affected Tools on a synthetic
bump to a bound `lib/` file, exit 0 throughout; `run_tool sample_so3_pool` live-tested
end-to-end — dry-run, real run (canonical `tool_run` written, store closed @303),
missing-args usage error, and drift-refusal all exercised. `run_pipeline` (ADR-0007 §5.3)
live-tested on `pipeline_so3-pool-dedup` — **compose** threaded `sample_so3_pool`'s
`q_pool_wxyz`/`rotvec_pool` into `twin_dedup` (canonical `pipeline_run` written, store
closed @305); **map** lifted `sample_so3_pool` over a seed list (`n_mapped` correct);
the whole-DAG drift gate refused (exit 2) on a synthetic bound-file bump; `--dry-run`
validates ref structure without compute.
