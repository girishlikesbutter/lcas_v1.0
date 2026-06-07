# 0002 — Glossary lives in the RO structured store; harvest MP's behaviors

Status: accepted — 2026-06-04

The glossary (project vocabulary) lives in the RO structured store
(`research_os/glossary/*.json` against `glossary_term.schema.json`: provisional→canon
tiering, `synonyms_blocked` vs accepted `aliases`, `coined_in` provenance,
`related_terms`), enforced by the `glossary-lint` hook — **not** in MP's flat-prose
`CONTEXT.md`. From the MP workflow we transplant the **behaviors**, not the storage:
**lazy creation** (the first word you flag creates the store + that one term, nothing
pre-built), **in-conversation challenge** (use the canon term and flag drift live; the
lint hook is the backstop, not the only line), and **sharpen-overloaded-terms** (on
add/fix, detect fuzz and propose a precise term).

Why: RO's term object is machine-queryable, lint-enforceable, version-stamped, and
cross-linked to the run corpus — strictly richer than prose. But MP's *practice* (teeth
during work, forcing precise splits) is what makes a glossary actually get used. Keep
RO's body, graft MP's reflexes. Instance of 0001.

Alternatives rejected: (a) adopt MP `CONTEXT.md` + `grill-with-docs` wholesale — loses
structure, forks the store; (b) build a fresh glossary skill ignoring MP — reinvents the
sharpen/challenge behaviors MP already proved.

Open: whether to render a `CONTEXT.md` *from* the RO store so MP engineering skills
(`diagnose`/`tdd`/`improve-codebase-architecture`) can read the vocab. Deferred.

Follow-up: build the thin on-the-fly `glossary` skill **after** exercising the practice
manually on one real confusing term (manual-first, then formalise).
