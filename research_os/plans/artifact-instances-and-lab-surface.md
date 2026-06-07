# Plan — Artifact instances + the laboratory surface

**Implements** ADR-0007 §"Consequences / Open → Staged, not yet built" (the last two structural
items before the Claude Design front-end beautify pass):

> artifact *instances* as first-class addressable objects (steps thread outputs IN-MEMORY, not
> on-disk addressable materials); the webapp Pipeline-run view (engine feed exists, rich surface
> deferred).

Scoped 2026-06-07. **Locked decisions** (Girish, plain-English forks):

| Fork | Choice | Means |
|---|---|---|
| What gets shelved | **Recognised materials only** | An output becomes an artifact_instance ONLY if its Tool declares it via a `ports.output` type. Nameless scratch returns stay in-memory. |
| Shelf is album or pantry | **Working pantry** | Re-feeding (`$artifact.<id>`) is IN scope — a stored material can be poured into a new run, no recompute. |
| Webapp ambition | **Also a materials shelf** | Standalone browsable view of all materials by type + lineage, on top of the per-pipeline run detail. |

The analogy that drove the forks: every result a block makes becomes a **labelled jar on a shelf**
("IA Cloud · seed 39 · made 2026-06-07 · by ia-cloud-generator"). The `artifact_instance` is the
label. The shelf is browsable (materials view) and you can cook from it (re-feed).

## STATUS — 2026-06-07

- **Slice 1 (Reify) — DONE & gated.** New `artifact_instance` schema + store type; `validate.py`
  ref-checks (`artifact_type`→glossary, `produced_by.run`→tool_run/pipeline_run,
  `*.artifacts_produced`→artifact_instance); `ports.output` accepts the keyed map (legacy list still
  valid — transition); `loop/artifacts.py` `reify()` (dict per-key / whole-dict-as-material / tuple /
  single / map-set, with a **defensive per-material skip** so an unserialisable output never breaks a
  run) + `load()` (the Slice-2 re-feed seam, written but not yet wired); both executors emit materials
  + record `artifacts_produced`. **Live-test:** `pipeline_so3-pool-dedup` reified 2 materials; store
  re-validated closed **@308**. Backfill: **28 Tool cards re-keyed**, **17 deferred** (see below).
- **Slice 1.5 (result-dataclass + tuple-position reify) — DONE & gated (2026-06-08).** Of the 17
  deferred, **7 are now reified** by two small `loop/artifacts.py` mechanisms — no pickle, no new
  store type:
  - **Dataclass serialisation:** `_jsonable` + `_save_blob` detect a result dataclass (or a
    homogeneous list of them) and `dataclasses.asdict()` → json blob; the card `sample` holds the
    field *names*, never the arrays. Reifies `compute_inertia`→`inertia-result`,
    `global_optimize`/`local_refine`/`multi_start_optimize`→`candidate-set` (the last as one
    whole-list set), `invert_lightcurve`/`_multifidelity`→`inversion-result` (new glossary term
    coined). These are the actual research outputs — the shelf now holds *answers*, not just pools.
  - **Non-leading tuple position:** an output key written `name@<idx>` selects `rv[idx]`, so
    `propagate_to_body_frame`'s `quats` (return `k1, k2, quats`) reifies from position 2 as
    `attitude-trajectory`. The recorded `port` is the clean base name (`quats`).
  - **Gate:** `loop/test_artifacts.py` (temp-dir, store-safe) proves all 4 dataclass shapes + the
    `@2` tuple case — one card each, compact sample, `load()` round-trip. `run_tool`'s
    `summarise_return` survives a dataclass return (generic branch, no crash), so a future real
    inversion run is safe. Backfill re-keyed 7 (now 35/45 Tools reify); store closed **@309** (the
    +1 over @308 is the `inversion-result` glossary term). Re-keying makes them *reifiable*; the
    first real inversion/optimize run populates them as a natural live-test.
- **Slice 1.5b (file-reference artifact variant) — DONE & gated (2026-06-08).** The 3 SPICE
  kernel-file producers now reify: their material is a written *file*, not an in-memory value, so
  the artifact_instance is held by **reference** — the card's `path` points at the tool-written
  kernel, **no blob copy**. New `artifact_instance.storage` enum (`blob` default / `reference`;
  default stays implicit so the 35 blob cards are untouched). `loop/artifacts.py`: `_is_file_ref`
  (`os.PathLike`, or a `str` that points at an existing file) + `_reference_material` (repo-relative
  path when under the repo, else absolute; `sample` = `{file, ext, exists}`); `reify` routes
  path-like outputs to it and stamps `storage:"reference"`; `load()` returns the file PATH for a
  reference (you furnsh a kernel, you don't parse it). Re-keyed via the backfill script (moved from
  DEFER→REKEY): `spice_kernel_generate`/`spice_observer_kernel` → `{spk_path: spk-kernel}`,
  `spice_orientation_kernel` → `{ck_path: ck-kernel, sclk_path: sclk-kernel}` (the last FIXED — it
  was mis-keyed as spk-kernel; the glossary already had ck-/sclk-kernel). Now **38/45 Tools reify**,
  7 DECLINED. Webapp Materials card renders the file digest + a `ref` badge. **Gate:**
  `loop/test_artifacts.py` extended (single-Path → one reference card, no data/ copy, `load()`
  returns the path; `(ck,sclk)` tuple → two references) — all PASS; `run_tool.summarise_return`
  survives a Path / tuple-of-Paths (safe fallback, no double-storage); the 3 cards drift-clean +
  runnable; tsc/vite clean; store closed **@316** (no new objects — mechanism + re-key + schema
  field only). A real kernel run is out of scope (dormant `in_use:false` Tools needing mkspk/
  pinpoint/prediCkt); the first such run populates them as a natural live-test.
- **DECLINED (7, on purpose) — not produced materials:** `stl_load_satellite` (a fixture rebuilt
  from config), `spice_handler` (the class / infrastructure), `load_truth` (the oracle — re-feeding
  would taint), `lc_resolve` (a path resolver), and the 3 plot/anim tools (figures → plot-stream).
  Reasons live in `backfill/rekey_ports_adr0007.py` `DEFER` (now tagged STAGED vs DECLINED).
- **Slice 2 (re-feed / the pantry) — DONE & gated (2026-06-08).** `$artifact.<id>` (string form,
  `+.<subkey>`) and `{"$artifact": "<id>"}` (run_tool `--input` JSON form, `+"key"`) now pour a
  SHELVED material into a new run. The resolver lives in `loop/artifacts.py`
  (`is_artifact_ref`/`typecheck_ref`/`load_ref`/`resolve_supplied`, on top of the existing `load`),
  wired into BOTH executors: `run_tool` resolves `--input` refs before bind (REFUSED→exit 2, the
  ref kept verbatim in the record's `input` for provenance); `run_pipeline.resolve_ref` resolves a
  step `inputs` value, type-checked, dry-run-aware (validates id+type at plan time, loads the blob
  only on a real run). **Type-check** against the consuming port: keyed-map port → exact type;
  legacy-list port → membership; absent → unconstrained. **Enabler:** keyed INPUT ports landed on
  the so3-pool-dedup tools (`twin_dedup.input` `["ia-cloud"]`→`{q0_arr:ia-cloud, omega_arr:omega-cloud}`);
  `omega-cloud` glossary term coined (the ω-pool is a recognised material) and added to
  `sample_so3_pool.output` (`rotvec_pool`). **Gates:** `loop/test_refeed.py` (store-safe temp shelf —
  ref recognition, load+subkey round-trip, keyed/legacy/unconstrained type-check, mismatch+unknown
  refusal, `resolve_supplied`, the pipeline seam); plus a REAL live-test — pipeline shelves q-pool +
  ω-pool + dedup output, then standalone `twin_dedup` re-feeding both pools via `$artifact` produced a
  q0_canon **BIT-IDENTICAL** to the pipeline's dedup step (re-feed == recompute); CLI mismatch
  (omega-cloud→q0_arr) and unknown-id both REFUSED (exit 2). Store closed **@316**.
- **Slice 3 (webapp surface) — DONE & gated (2026-06-08).** The shelf is now SEEN. Backend
  (`indexer.py`): `artifact_instances` added to the load globs + the snapshot; each material
  enriched with its producer (`producer_kind`/`producer_name`) and the glossary label for its type
  (`artifact_type_term`); pipelines enriched with their `pipeline_runs` feed + count (the §5.3 DAG
  records grouped per-pipeline, mirroring per-Tool `bench_runs`). Frontend: `PipelineRun`/
  `PipelineStep`/`PipelineStepDef`/`ArtifactInstance` types defined (were referenced-but-undefined);
  **`/pipelines/:id`** (`PipelineDetail` — header + static DAG with `$steps`/`$artifact` input
  threading + the pipeline_run feed, each run expandable to per-step hash-sync verdict + metrics +
  the **threaded material jars** it shelved); **`/materials`** (the shelf — filter by `artifact_type`,
  producer lineage, sample digest, and a **"cook from this"** button that copies the `$artifact.<id>`
  ref). Run-button drops `run_pipeline.py <id> --source webapp` (+ a safe `--dry-run`) into a PTY.
  **Boundary intact**: indexer is pure-read; views only `openTerminal`/clipboard — no new write path
  (webapp still writes only `queue/`). **Gate:** `tsc --noEmit` clean + `vite build` OK; live-test —
  backend served the snapshot with 6 materials + 3 pipeline_runs, and headless screenshots confirmed
  the pipeline detail (DAG + feed + expanded per-step threaded jars) and the materials shelf render
  from live data. After this the structure is finalised → the Claude Design beautify pass.

---

## The gap, precisely (verified against code)

- **`run_tool.py` already persists outputs** — blob → `tool_runs/artifacts/<id>.npz`, recorded as
  `artefacts[]` (path/kind/caption) on the tool_run. But these are **untyped** (`kind` ∈
  checkpoint|plot|anim|data|report|log) and **not addressable** as a named, re-feedable material.
- **`run_pipeline.py` persists nothing** for step outputs — threads in-memory
  (`ctx["steps"][sid] = rv`, `run_pipeline.py:203`); the pipeline_run record stores only `metrics`
  digests per step (`run_pipeline.py:201`).

So both executors lose the materials. The fix is **one new primitive** they both emit.

---

## The new primitive: `artifact_instance`

A store **card** (counted in closure) that labels one produced material:

```jsonc
{
  "schema_version": "1.0.0",
  "id": "ai_ia_cloud_20260607t193018",      // ai_<type-slug>_<stamp>
  "kind": "artifact_instance",
  "artifact_type": "ia-cloud",               // -> glossary_term id (the "recognised material" filter)
  "cardinality": "one",                       // "one" (per-epoch/single) | "set" (map-over-window)
  "produced_by": { "run": "pr_so3_pool_dedup_20260607t193018", "step": "sample" },
                                              // run -> tool_run OR pipeline_run id; step null for a bare tool_run
  "port": "q_pool_wxyz",                      // which return-key of the Tool this came from
  "path": "research_os/artifact_instances/data/ai_ia_cloud_20260607t193018.npz",
  "caption": "IA Cloud · seed 39",
  "size_bytes": 412334,
  "sample": { "shape": [5000, 4], "dtype": "float64" },  // digest for the dashboard, never the blob
  "commit": "70b620a",
  "created_at": "2026-06-07T19:30:18"
}
```

**Storage layout** (mirrors tool_runs/pipeline_runs):
- `artifact_instances/*.json` — the cards (closure-counted; `validate.py` discovers them).
- `artifact_instances/data/<id>.npz|.json` — the blobs (NOT store objects, like figures).

**Closure**: `validate.py` adds `"artifact_instance": ["artifact_instances/*.json"]` to `DIRS`
(`validate.py:28-41`), indexes by id (`:71-84`), and ref-checks (`:180+`):
- `artifact_type` → must resolve to a glossary_term id,
- `produced_by.run` → must resolve to a tool_run OR pipeline_run id.
Object count grows from @305.

---

## The load-bearing refinement: `ports.output` must become a keyed map

Today `ports.output` is a **flat list** of type ids — e.g. `filter_evaluate_candidate` →
`output: ['filter','light-curve']` — but a Tool returns a **dict** (`q_pool_wxyz`, `R_cache`, …).
Nothing maps a return-key to its declared type, so "shelve recognised materials only" is
unresolvable as-is.

**Resolution** (Slice 1 sub-task): change `ports.output` from `["type", …]` to a map
`{ "<return-key>": "<artifact-type glossary id>", … }`:

```jsonc
"ports": {
  "input":  { "q0_arr": "q0-omega-state", "epoch": "epoch-observation" },
  "output": { "q_pool_wxyz": "so3-pool", "rotvec_pool": "omega-set" }
}
```

- Schema change to `substrate_component.schema.json` (`ports.input`/`output` → keyed object;
  values are glossary_term ids; `validate.py:175-178` ref-check adapts from list-iter to
  values-iter).
- **Backfill**: re-key the ~47 scoured Tool cards' ports (`substrate/*.json`). Bounded, mechanical;
  most have 1–2 outputs. A small migration script, same shape as `backfill/migrate_adr0005.py`.
- The executor then reifies a return value **iff its dict-key appears in `ports.output`**; the value
  there is the `artifact_type`. Everything else stays in-memory (the "recognised only" rule).
- Input ports keyed the same way pays off in Slice 2 (type-checked re-feed).

This is the single biggest piece of work and it's a prerequisite for the "recognised materials"
choice. (Alternative considered: positional/single-output inference — rejected, fragile for
dict/tuple returns and silent on multi-output Tools.)

---

## Slice 1 — Reify (write side)

**Goal:** every port-typed output of a tool run OR pipeline step lands as an artifact_instance card
+ blob. Retrofits run_tool's existing blob-saving into the one materials concept.

1. `schemas/artifact_instance.schema.json` (envelope above).
2. `substrate_component.schema.json` ports → keyed map; backfill the 47 cards (migration script).
3. `validate.py`: register the type in `DIRS`, index it, ref-check `artifact_type` + `produced_by.run`;
   adapt the ports ref-check to keyed maps.
4. **Shared helper** `loop/artifacts.py`: `reify(rv, ports_output, produced_by, dest_dir) -> [ai_id…]`
   — for each return-key in `ports.output`, save the blob, write the card, return the ids.
   `map` steps → `cardinality:"set"` (one card over the list, or one per element — default: one `set`
   card pointing at a stacked/listed blob).
5. `run_tool.py`: after a successful run, call `reify(...)`; record `artifacts_produced: [ai_id…]`
   on the tool_run. Keep `artefacts[]` for **plots/logs** (figures, not materials).
6. `run_pipeline.py`: in `run_step` after `ctx["steps"][sid] = rv` (`:203`), call `reify(...)` for
   that step's Tool ports; record `artifacts_produced` per step in the step-result.
7. Schema bumps to `tool_run`/`pipeline_run`: optional `artifacts_produced: [artifact_instance id]`
   per run / per step; `validate.py` ref-checks them.

**Gate:** `validate.py` PASS with new count; a live `run_pipeline pipeline_so3-pool-dedup` writes
artifact_instance cards for the typed step outputs; cards resolve closed.

---

## Slice 2 — Address / re-feed (the pantry)

**Goal:** `$artifact.<id>` in the ref-language loads a stored material and pours it into a new run —
type-checked against the consuming port.

1. Ref-language: extend `resolve_ref` (`run_pipeline.py:70-108`) and run_tool's `--input` parsing
   to recognise `"$artifact.<id>"` (or `{"$artifact": "<id>"}`):
   - load the artifact_instance card,
   - **type-check**: the card's `artifact_type` must equal the consuming Tool's `ports.input[kwarg]`
     type — else refuse (the port type-system's payoff: can't pour an IA-Cloud into a light-curve
     port),
   - `np.load`/`json.load` the blob and pass the value as the kwarg.
2. CLI ergonomics: `run_tool.py <tool> --input '{"q0_arr": {"$artifact": "ai_…"}}'`.

**Gate (live-test):** run A produces an artifact; run B consumes it via `$artifact.<id>`; B's result
matches the all-in-one pipeline's result for that step (proves re-feed == recompute). A deliberate
type-mismatch is refused.

---

## Slice 3 — Surface (webapp: pipeline-run detail + materials shelf)

Backend already loads `pipeline_runs` (`indexer.py:53`); boundary stays read-only + queue-only
(`intents.py`, writes only `queue/`). No new write path — "cook from shelf" launches a terminal with
the CLI command, exactly like today's run-button (`Substrate.tsx:202`).

**Backend (`indexer.py`):**
- Add `"artifact_instances": ["artifact_instances/*.json"]` to the load globs.
- Build `runs_by_pipeline` and enrich `pipe_rows` with their pipeline_run feed (mirrors the
  per-Tool `bench_runs` at `:331`).
- Expose `snapshot.artifact_instances[]` + lineage (each carries `produced_by`); enrich pipeline_run
  steps with `artifacts_produced`.

**Frontend (React/TSX):**
- Types (`lib/types.ts`): add `PipelineRun`, `PipelineStep`, `ArtifactInstance` (PipelineRun is
  currently referenced but undefined).
- `/pipelines/:id` → new `PipelineDetail`: header (hypothesis/serves/composes) + steps DAG +
  pipeline_run feed (newest-first) + per-step detail (tool, version, hash_ok, wall_s, metrics,
  error) + **threaded artifacts** (each step's `artifacts_produced` as jars).
- New `/materials` (the shelf): browse all artifact_instances, **filter by `artifact_type`**, show
  producer lineage (run + step), `cardinality`, sample digest, and a **"cook from this"** affordance
  that copies the `$artifact.<id>` ref / launches a terminal run.

**Gate:** dashboard renders pipeline-run detail + materials shelf from live data; boundary check
(webapp writes only `queue/`).

---

## Build order & why

1. **Slice 1 first** — nothing to address or render until materials exist on disk. The ports keyed-map
   refinement + backfill is the long pole; do it up front.
2. **Slice 2** — depends on instances existing (live-test needs real cards to re-feed).
3. **Slice 3** — depends on both (render what exists; surface the "cook from" of Slice 2).

This matches ADR-0007's own sequencing and the data dependency. After Slice 3, the structure is
finalised → the Claude Design beautify pass (the gate ADR-0007 names).

## Open risks
- **Blob format zoo**: returns are ndarray / dict-of-arrays / scalars / paths. `reify` needs a small
  type-dispatch (npz for arrays, json for scalars/dicts-of-scalars, passthrough for path strings).
- **`map` cardinality**: one `set` card over a stacked blob is simplest; per-element cards give finer
  lineage but multiply objects. Default to one `set` card; revisit if per-element lineage is wanted.
- **Backfill correctness**: re-keying 47 ports by hand-mapping return-key→type needs the actual
  return signatures; do it Tool-by-Tool, verify against `entry_point` source.
- **Retrofit double-storage**: ensure run_tool stops duplicating typed blobs under
  `tool_runs/artifacts/` once they live under `artifact_instances/data/` (plots/logs stay).
