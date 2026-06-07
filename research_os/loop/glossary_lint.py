#!/usr/bin/env python3
"""Glossary-lint — mechanical canon-membership check (PLAN §3, Q7).

Reads the canon vocabulary from `research_os/glossary/*.json` (each a glossary_term
object: `term`, `status`, `synonyms_blocked[]`) and lints a target text: if a
BLOCKED synonym appears but its canon term does not, flag it. This is the
semantic-synonym-block that keeps the vocabulary from fragmenting.

PHASE-1 STATE: the glossary is intentionally UNSEEDED ("ship lint, defer seeding").
With no canon terms loaded this lints to a clean no-op — the script + wiring are in
place so the gate activates the moment terms are seeded, with zero further work.

Soft by default (warnings, exit 0); `--strict` makes a blocked-synonym hit exit 2.

Usage:
    python research_os/loop/glossary_lint.py path/to/writeup.md
    git diff --cached | python research_os/loop/glossary_lint.py -      # lint staged text
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_canon(root: str = ROOT):
    """Return {blocked_synonym_lower: [canon_term, ...]} from canon (or provisional) terms.

    A synonym may map to MORE than one canon term — e.g. 'discriminator' splits into
    both 'Filter' and 'Rank' (the split is exactly why the word was retired). The lint
    then nudges "use one of Filter or Rank" rather than silently picking one.
    """
    blocked: dict[str, list[str]] = {}
    canon_terms = []
    for f in glob.glob(os.path.join(root, "glossary/*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        d = json.load(open(f))
        term = d.get("term", "")
        canon_terms.append(term)
        for syn in d.get("synonyms_blocked", []) or []:
            terms = blocked.setdefault(syn.lower(), [])
            if term not in terms:
                terms.append(term)
    for terms in blocked.values():
        terms.sort()
    return blocked, canon_terms


# Human-prose fields in store objects. Only these are linted in a .json target —
# IDs, *_ref(s), child_runs, synonyms_blocked, aliases, timestamps, etc. are handles
# or declarations, NOT prose, and would false-positive on immutable spellings.
PROSE_KEYS = {
    "title", "summary", "definition", "description", "tldr", "hypothesis",
    "rationale", "note", "notes", "body", "text", "question", "finding",
    "conclusion", "why",
}


def extract_prose(obj) -> list[str]:
    """Collect human-prose strings from a parsed store object.

    Walks the structure and keeps only string values whose KEY is in PROSE_KEYS,
    so id handles (`goal_cross-cloud-…`), id-refs (`run`, `child_runs`), and the
    glossary's own `synonyms_blocked` declarations are never linted.
    """
    out: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in PROSE_KEYS and isinstance(v, str):
                    out.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(obj)
    return out


def text_for_path(path: str) -> str:
    """Return the lintable text for a target file.

    `.md` writeups lint whole; store `.json` objects lint only their prose fields
    (PROSE_KEYS). A glossary_term file declares synonyms by design — skip it whole.
    `_`-prefixed files (generated indices) are skipped too.
    """
    raw = open(path).read()
    if not path.endswith(".json"):
        return raw
    if os.path.basename(path).startswith("_"):
        return ""
    try:
        obj = json.loads(raw)
    except Exception:
        return raw  # malformed JSON -> fall back to raw text
    if isinstance(obj, dict) and obj.get("kind") == "glossary_term":
        return ""  # vocabulary declaration: self-referential, don't lint
    return "\n".join(extract_prose(obj))


def lint(text: str, root: str = ROOT):
    """Return a list of (blocked_synonym, [canon_term, ...]) hits.

    A hit fires when the synonym appears and NONE of its candidate canon terms is
    already present in the text (writing the precise term suppresses the warning).
    """
    blocked, canon_terms = load_canon(root)
    if not blocked:
        return []  # unseeded canon -> clean no-op
    low = text.lower()
    canon_present = {t.lower() for t in canon_terms if t.lower() in low}
    hits = []
    for syn, terms in blocked.items():
        if re.search(r"\b" + re.escape(syn) + r"\b", low) and not any(
            t.lower() in canon_present for t in terms
        ):
            hits.append((syn, terms))
    return hits


def main():
    ap = argparse.ArgumentParser(description="Lint text against the canon glossary.")
    ap.add_argument("path", help="file to lint, or '-' for stdin")
    ap.add_argument("--strict", action="store_true", help="exit 2 on a blocked-synonym hit")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    text = sys.stdin.read() if args.path == "-" else text_for_path(args.path)
    hits = lint(text, args.root)
    for syn, terms in hits:
        if len(terms) == 1:
            suggest = f"use the canon term '{terms[0]}'"
        else:
            suggest = "use one of " + " or ".join(f"'{t}'" for t in terms)
        print(f"glossary-lint: '{syn}' is a blocked synonym — {suggest}.", file=sys.stderr)
    if hits and args.strict:
        sys.exit(2)


if __name__ == "__main__":
    main()
