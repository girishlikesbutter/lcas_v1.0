# 0004 — Default verbosity governed by a persistent output-style

Status: accepted — 2026-06-04

Default response verbosity is controlled by a persistent Claude Code output-style
(`~/.claude/output-styles/concise.md`), **not** by in-session promises to "be terser."
Behaviour: answer-first, terse by default, full technical precision preserved (numbers,
paths, citations verbatim), expand only on an explicit signal ("more" / "why").

Why: "I'll do better" does not persist across turns or sessions; the operator asked for a
**structural** lever. An output-style is harness-level and reloaded every session — the
only durable mechanism, since nothing mechanically truncates prose.

Alternatives rejected: (a) in-session promises — proven not to hold (the prompt for this
decision); (b) caveman mode — compresses grammar but risks dropping the *why*/nuance, too
lossy as a default.

Consequence: activated via `"outputStyle"` in settings, revertable with `/output-style`.
The same brevity limit should later be echoed into the spine skills' own output sections.
