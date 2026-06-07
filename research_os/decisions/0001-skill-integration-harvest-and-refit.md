# 0001 — Harvest-and-refit installed skills; never adopt wholesale

Status: accepted — 2026-06-04

We have a large installed skill ecosystem (Matt Pocock's engineering skills:
`grill-with-docs`, `improve-codebase-architecture`, `to-issues`, `triage`, …). When one
of these does something the Research OS needs, we take its **method** and refit its
inputs/outputs to the RO objects (`substrate/`, `lib/`, run records, claim cards, the
plot-stream). We do **not** bolt the foreign skill onto the spine, and we do **not**
adopt its storage (`CONTEXT.md`, `docs/adr/`, `.scratch/`). This operationalises
PLAN §10 ("external backbones are components, never the spine") and Q1 ("reflect, don't
own").

Why: each foreign skill carries its own world-model (CONTEXT.md-as-glossary, ADRs in
docs/adr/, GitHub issues). Adopting it wholesale forks the source of truth and breaks the
trust store's single-home invariant. Its *method* (terminology sharpening, deep-module
review, tracer-bullet slicing) is the valuable, transplantable part.

Alternatives rejected: (a) adopt MP skills as-is — two sources of truth; (b) ignore them
and rebuild from scratch — discards proven method. Refit is the middle path.

Consequence: each MP capability we want becomes a small RO-native skill reusing the
method. First application: the glossary (0002).
