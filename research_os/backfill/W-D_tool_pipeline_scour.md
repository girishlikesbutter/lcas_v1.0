# W-D — Tool/Pipeline scour (the laboratory backfill)

> **Status: DONE — 2026-06-07.** Registry authored, drift-checked, store validated closed (302 objects).
> Idempotent generator: `backfill/W-D_generate_cards.py` (reads `backfill/W-D_inventory.json`).

## Results (2026-06-07)
- **Inventory** (`W-D_inventory.json`): 7-agent fan-out → 97 candidate operations across `src/` + `survey/lib/`,
  24 functions marked in-use via `experiments/` call-sites, classified into 34 job-clusters + 6 human-judgment forks.
- **Decisions (Girish):** scope = EVERYTHING (src/inversion carded `in_use:false`); ρ-metric COLLAPSED to one Tool
  (`rho_band_classify`, `space` param); Fork 5 split `ls_bracket` to its own canon Tool; Forks 4/6 → `variant_of`, dormant.
- **Authored:** **47 Tool cards** in `substrate/` (35 canon + 12 variant; 3 existing repointed/enriched, 44 new) —
  23 `in_use:true` (live laboratory), 24 dormant/parallel. **20 glossary terms** (the artifact/port type-system).
  **5 Pipelines** (new `pipeline_blind-5step-invert` + 4 recomposed to the new Tool ids).
- **Drift-check:** all 47 `entry_point`s resolved in code + sha256-stamped; **zero duplicate entry_points** without a
  `variant_of`/`canon` relationship. Reconciled the 3 drifted cards (scorer repointed `cost_surfaces.py` shim →
  `surrogate_eval.py:full_lc_mse`; the two "NOT YET PORTED" stubs resolved — `alignment_cost` is a separate filter Tool,
  `lofi_peak_match` left uncovered). Tool ids are `underscore_case` (substrate id pattern; glossary/pipeline stay hyphen).
- **Next (per ADR-0007 sequencing):** the executor run-button (#15, smallest form) now has its registry.

---

> Authored 2026-06-07 alongside ADR-0007. W-B-sized (dozens+ Tools); idempotent (skip any card that already exists).

## Why
ADR-0007 makes the substrate a **laboratory**: Tools and Pipelines become first-class, visible,
versioned, runnable boxes. Today only 3 Tools are registered (propagator, scorer, surrogate), as
hand-written *descriptions* that have already drifted from the code (`scorer.json`: two functions
"NOT YET PORTED"). The block Girish most wants to run — the IA-Cloud generator — isn't registered
at all. The scour builds the honest, deduped registry the bench needs.

## Goal
Walk the real code (`src/`, `notebooks/inversion/survey/lib/`, `experiments/`) and produce, for
every de-facto operation actually in use:
- a **canon Tool card** (`substrate/<id>.json`) with `entry_point`, `ports` (artifact-type
  glossary ids), `default_params`, `canon: true`, bound to ONE code seam + hash;
- **`variant_of` Tool cards** for *meaningful* variants (distinct method, same job);
- **collapse** accidental copies into the canon Tool's `default_params` (record the collapse);
- **Pipeline cards** (`pipelines/<id>.json`) for the end-to-end compositions (LC → satellite info),
  with `composes: [tool ids]` + `serves: [goal]` + `hypothesis`.

## Method (mirror W-B's fan-out)
1. **Inventory pass** — one agent greps each code dir for def-sites + call-sites; emits a candidate
   operation list with file:function + rough role. (Use `Explore`/`general-purpose` subagents.)
2. **Dedup/classify pass** — cluster candidates by job; for each cluster decide
   **accidental-copy → collapse(params)** vs **meaningful-variant → variant_of** vs
   **distinct Tool**. This is the human-judgment seam — surface the fork, recommend, confirm.
3. **Card-authoring pass** — batch subagents write the Tool/Pipeline JSON (idempotent: skip
   existing ids). Each new `ports` artifact type that has no glossary term yet → **coin the term
   first** (the validator enforces `ports → glossary_term`).
4. **Drift-check** — for every `entry_point`, verify it resolves in the code and stamp the hash
   (foundation: the AST-grounded machinery map, `render/machinery_overlay.json`).
5. **Validate** — `python research_os/loop/validate.py` exits 0 (store stays referentially closed).

## Classification rule (from ADR-0007)
- Same job, differs only by a parameter value → **collapse** into `default_params`.
- Same job, genuinely different *method* → **separate Tool**, `variant_of` → canon.
- Different job → **separate canon Tool**.

## Output / done-criteria
- Every Tool actually used by a live Pipeline is registered + drift-checked.
- No two cards bind the same `entry_point` without a `variant_of`/`canon` relationship.
- `validate.py` clean. A `dynamic-viz` glance of the Tool bench (canon + variant families) renders.

## Notes
- Scope discipline: register Tools **actually in use** by live/open Pipelines first; dead-branch-only
  operations can wait (or get a `tried-and-shelved` treatment).
- The scour is **descriptive** — it does NOT need the executor. Runnability (the run-button, #15)
  comes after the registry exists.
