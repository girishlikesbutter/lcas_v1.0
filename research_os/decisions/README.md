# Research OS — decisions (ADR-lite)

Architecture & policy decisions for the Research OS **itself** — not research findings
(those are claim cards). A decision earns a file when it is **hard to reverse**,
**surprising without context**, or **the result of a real trade-off**. The value is
recording *that* a decision was made and *why* — a single paragraph is fine.

Format harvested from Matt Pocock's ADR practice (the method, refit to RO — see 0001).

## Format
`NNNN-slug.md`, sequential. Minimum: a title + 1–3 sentences (context, decision, why).
Optional `Status`, `Alternatives`, `Consequence`/`Open` lines only when they add value.

## Index
- [0001](0001-skill-integration-harvest-and-refit.md) — Harvest-and-refit installed skills; never adopt wholesale.
- [0002](0002-glossary-home-ro-native.md) — Glossary lives in the RO structured store; harvest MP's behaviors, not its `CONTEXT.md`.
- [0003](0003-interim-visual-surface-browser-stream.md) — Interim visual surface = local browser plot-stream, ahead of the Phase-4 web app.
- [0004](0004-verbosity-via-output-style.md) — Default verbosity governed by a persistent output-style, not in-session promises.
- [0005](0005-pipeline-as-first-class-node.md) — **(proposed)** The store is a typed graph; promote `Pipeline` to a first-class node, distinct from `Goal`.
- [0006](0006-oracle-clean-distinguish-labels-from-search.md) — **(proposed)** `oracle_clean` should distinguish truth-in-search (real leak) from truth-for-labels (harmless); fix the criterion, don't game it.
- [0007](0007-tool-laboratory-operational.md) — **(accepted)** The substrate is a laboratory; make Tools/Pipelines operational (one canonical binding + params, variant-of families, executor run-button, Q1-safe). Scour first (W-D).
