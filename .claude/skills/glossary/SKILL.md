---
name: glossary
description: "Vocabulary instrument of the Research OS — the on-the-fly term store. Use when a NAME confuses you mid-work (decompose it into atoms and coin the precise term), when you spot vocabulary drift (a blocked synonym used in flight), when asked to sharpen / formalise / define a term, or when asked what a project term means. Coins a glossary_term JSON against glossary_term.schema.json (provisional→canon, synonyms_blocked, coined_in, related_terms); the glossary-lint hook then enforces the synonym-block automatically. Triggers on: what does X mean, define X, that name is confusing, sharpen this term, is this the right word, formalise the vocabulary, add to the glossary, this is drift. NOT for drawing the corpus/trust state (that's dynamic-viz)."
---

# Glossary — the Research OS vocabulary instrument

The store's **term layer**. Vocabulary that fragments costs trust: the same thing under three
names reads as three findings; one name over two things hides a real split. This skill keeps
the project's words sharp — it **coins, sharpens, and enforces** terms as `glossary_term`
objects (`research_os/glossary/*.json` against `schemas/glossary_term.schema.json`), and the
`glossary-lint` hook is the mechanical backstop (ADR-0002: RO body, MP's reflexes).

It is **on-the-fly and lazy** — there is no pre-built dictionary to maintain. The first word
you flag creates its term; nothing else. Most of the work is *judgment about language*, not
file-writing.

## The method (what the seed session taught)

**When a NAME confuses you, decompose it into atoms — the glossary stores the *atoms*.**
A composite like "Anchor-Based IA Cloud Cross" is *self-documenting* once its atoms
(`anchor`, `ia-cloud`, `ia-cloud-cross`) are sharp; you don't store the composite, you store
the pieces. Likewise a goal/pipeline **title** documents itself via its own `title` field —
don't mint a glossary term for it. Store a term only when it is a **reusable atom** that will
recur across runs and whose fuzziness is a live hazard.

Two hazards the method catches (run these reflexes, ADR-0002):
- **sharpen-overloaded-terms** — one word doing two jobs. The cure is a *split*: "discriminator"
  → **Filter** (score + threshold, not truth-faithful) vs **Rank** (order, truth-faithful only).
  Coin both; block the fuzzy parent.
- **quote-with-its-qualifier** — a term that is meaningless without a companion. A ρ-band
  without its **generator** (HiFi / SurrV2 / LoFi) is noise. Encode the requirement in the
  definition; if it's a reporting rule, it's a `preference` claim card, not a term.

## The behaviors (transplanted from MP, ADR-0002)

- **Lazy creation.** Don't seed a dictionary. Coin the one term in front of you, now.
- **In-conversation challenge.** Use the canon term yourself and flag drift *live* when you or
  the operator writes a blocked synonym — the lint hook is the backstop, not the only line.
- **Sharpen on add.** Every new term is an invitation to ask "is this name doing one job?"
  If not, split before you save.

## Coin a term (the write)

One file per term, `research_os/glossary/<id>.json`. `id` = kebab slug = filename. Minimal:

```json
{
  "schema_version": "1.0.0",
  "id": "spurious-survivor",
  "kind": "glossary_term",
  "term": "Spurious survivor",
  "definition": "A candidate that scores well under an unfaithful/short-window score but is neither truth nor a degenerate solution. Cure: use that score as a Filter, + over-determine.",
  "status": "provisional",
  "coined_in": "s107_discrimination_test",
  "synonyms_blocked": ["phantom"],
  "related_terms": ["degenerate-solution", "filter", "rank"],
  "created_at": "2026-06-04T00:00:00Z"
}
```

Field discipline:
- **`status`** — `provisional` for anything coined in flight (the default). `canon` only after a
  bless: a deliberate operator decision, or a sharpening session that locks it. On promotion set
  `promoted_on`. The lint enforces blocked synonyms for **both** tiers, so provisional already
  has teeth — promotion is about *agreement*, not activation.
- **`synonyms_blocked`** — the forbidden words the lint replaces with `term`. Put the *retired*
  spellings here (`"phantom"`, `"cross-cloud"`). A synonym may map to **more than one** term
  (block `"discriminator"` on both `filter` and `rank`); the lint then says "use one of Filter or
  Rank." Do **not** block a word that is a legitimate *component* of the right phrase (e.g. bare
  "Band A" is a missing-qualifier problem, not a synonym — that's a preference card).
- **`aliases`** — accepted alternates that are NOT blocked (`"pol_diam"` for polhode diameter).
- **`coined_in`** — run_record id or writeup path where the term/concept first appeared. Cite a
  real one; omit rather than fabricate (CLAUDE.md: cite or mark unverified).
- **`related_terms`** — ids of sibling terms; wire the cross-links (Filter↔Rank↔Scoring).

Then validate and let the hook arm itself:

```bash
python research_os/schemas/validate_one.py research_os/glossary/<id>.json glossary_term
python research_os/loop/glossary_lint.py <any-writeup.md>   # confirm the new block fires
```

The `glossary-lint` hook (`.claude/hooks/ro_glossary_lint.sh` → `loop/glossary_lint.py`) fires
on every write to `survey/experiments/*.md`, soft (warns to stderr, exit 0). It activates the
moment a term with a `synonyms_blocked` exists — no wiring step. `--strict` makes a hit exit 2.

## What NOT to do

- **Don't pre-build a dictionary.** Lazy creation — one term, the one in front of you.
- **Don't mint a term for a self-documenting title.** Goal/pipeline names live in their node's
  `title`; the glossary is for reusable *atoms*. (Decompose, then store the pieces.)
- **Don't block a legitimate component word.** "Band A" isn't a synonym for "ρ-band" — it's a
  fragment missing its generator. Model that as a reporting `preference` claim card.
- **Don't promote to `canon` unilaterally.** Provisional already lints. `canon` is an
  agreement state — set it when the operator blesses or a sharpening session locks it.
- **Don't fabricate `coined_in`.** A real run/writeup or nothing.

## Where things live

- **Store:** `research_os/glossary/*.json` (one per term). **Schema:**
  `research_os/schemas/glossary_term.schema.json`.
- **Lint:** `research_os/loop/glossary_lint.py` (importable `lint(text)` → `[(syn, [terms])]`);
  hook wrapper `.claude/hooks/ro_glossary_lint.sh`.
- **Decision of record:** `research_os/decisions/0002-glossary-home-ro-native.md`.
