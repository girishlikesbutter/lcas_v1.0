#!/usr/bin/env python3
"""
cleanup_experiments_md.py — Phase D mechanical sweep of EXPERIMENTS.md.

Applies the same three regex transforms that `cleanup_wiki_content.py` applies
to wiki files, but targeted at the single top-level `notebooks/inversion/
EXPERIMENTS.md` doc (which Phase B explicitly skipped):

  1. [[microNN]] / [[microNN_suffix]] wiki-link tokens
     → full new page name from RENAMES.md wiki-page table.
     (Fallback to bare `m{NNN}{suffix}` if no mapping.)

  2. Script path references (`notebooks/inversion/**/microNN_*.py` or bare
     `microNN_*.py`) → new script name from RENAMES.md script tables.

  3. Remaining plain `\\bmicro(\\d+)([a-z]\\d*)?` tokens (prose, paths,
     print strings, comments) → `m{NNN}{suffix}`.

DOES NOT:
  - Replace the Section-1 top handoff block (the user does that by hand —
    it's a narrative rewrite, not a mechanical sweep).
  - Touch frontmatter (EXPERIMENTS.md has none).
  - Bump an `updated:` field (none to bump).

Usage: python3 scripts/cleanup_experiments_md.py [--dry-run]
Writes: scripts/cleanup_experiments_md.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVERSION = ROOT / "notebooks" / "inversion"
RENAMES = INVERSION / "RENAMES.md"
TARGET = INVERSION / "EXPERIMENTS.md"
LOG_PATH = ROOT / "scripts" / "cleanup_experiments_md.log"


SCRIPT_TABLE_ROW = re.compile(
    r"^\|\s*`([^`]+\.py)`\s*\|\s*`([^`]+\.py)`\s*\|"
)
WIKI_TABLE_ROW = re.compile(
    r"^\|\s*`([^`]+\.md)`\s*\|\s*`([^`]+\.md)`\s*\|"
)


def parse_renames() -> tuple[dict[str, str], dict[str, str]]:
    script_map: dict[str, str] = {}
    wiki_map: dict[str, str] = {}
    text = RENAMES.read_text()
    in_wiki = False
    for line in text.splitlines():
        if line.startswith("## Wiki Experiment Pages"):
            in_wiki = True
            continue
        if line.startswith("## ") and in_wiki:
            in_wiki = False
        if in_wiki:
            m = WIKI_TABLE_ROW.match(line)
            if m:
                wiki_map[m.group(1)] = m.group(2)
        else:
            m = SCRIPT_TABLE_ROW.match(line)
            if m:
                script_map[m.group(1)] = m.group(2)
    return script_map, wiki_map


MICRO_TOKEN_RE = re.compile(r"\bmicro(\d+)([a-z]\d*)?")


def simple_prefix_rewrite(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        num = int(m.group(1))
        suffix = m.group(2) or ""
        return f"m{num:03d}{suffix}"

    new_text = MICRO_TOKEN_RE.sub(repl, text)
    return new_text, count


def build_wiki_link_map(wiki_map: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for old_md, new_md in wiki_map.items():
        out[old_md.removesuffix(".md")] = new_md.removesuffix(".md")
    return out


def rewrite_wiki_links(text: str, link_map: dict[str, str]) -> tuple[str, int]:
    count = 0
    link_re = re.compile(r"\[\[(micro[^\]\|]+?)(\|[^\]]+)?\]\]")

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        target = m.group(1)
        alias = m.group(2) or ""
        if target in link_map:
            new_target = link_map[target]
        else:
            new_target, n = simple_prefix_rewrite(target)
            if n == 0:
                return m.group(0)
        count += 1
        return f"[[{new_target}{alias}]]"

    new_text = link_re.sub(repl, text)
    return new_text, count


def rewrite_script_paths(text: str, script_map: dict[str, str]) -> tuple[str, int]:
    count = 0
    keys_sorted = sorted(script_map.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(k) for k in keys_sorted) + r")(?![A-Za-z0-9_])"
    )

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return script_map[m.group(1)]

    new_text = pattern.sub(repl, text)
    return new_text, count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log: list[str] = []

    if not TARGET.exists():
        print(f"ERROR: {TARGET} does not exist", file=sys.stderr)
        return 2

    script_map, wiki_map = parse_renames()
    link_map = build_wiki_link_map(wiki_map)
    log.append(f"Parsed {len(script_map)} script mappings, "
               f"{len(wiki_map)} wiki page mappings, "
               f"{len(link_map)} link targets")

    original = TARGET.read_text(encoding="utf-8")
    text = original

    text, n_links = rewrite_wiki_links(text, link_map)
    text, n_paths = rewrite_script_paths(text, script_map)
    text, n_tokens = simple_prefix_rewrite(text)

    changed = (text != original)
    rel = TARGET.relative_to(ROOT)
    log.append(
        f"{rel}  links={n_links} paths={n_paths} tokens={n_tokens}"
        f"{'  [no change]' if not changed else ''}"
    )

    if changed and not args.dry_run:
        TARGET.write_text(text, encoding="utf-8")

    summary = (
        f"\nSummary: {rel}"
        f"\n  links rewritten:        {n_links}"
        f"\n  script paths rewritten: {n_paths}"
        f"\n  bare tokens rewritten:  {n_tokens}"
        f"\n  total regex rewrites:   {n_links + n_paths + n_tokens}"
        f"\n  dry-run:                {args.dry_run}"
    )
    log.append(summary)
    for ln in log:
        print(ln)

    LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
