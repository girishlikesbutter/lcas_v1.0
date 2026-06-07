#!/usr/bin/env python3
"""
cleanup_wiki_rename.py — Rename wiki/experiments/*.md + fix [[links]].

Parses the "## Wiki Experiment Pages" table in RENAMES.md and:
  1. git mv each wiki/experiments/microNN.md → mNNN_semantic.md
  2. Rewrite [[microNN]] → [[mNNN_semantic]] across every wiki/ file

Usage: python3 scripts/cleanup_wiki_rename.py [--dry-run]

Writes: scripts/cleanup_wiki_rename.log
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVERSION = ROOT / "notebooks" / "inversion"
RENAMES = INVERSION / "RENAMES.md"
WIKI = INVERSION / "wiki" / "wiki"
LOG = ROOT / "scripts" / "cleanup_wiki_rename.log"

WIKI_SECTION_RE = re.compile(r"^##\s+Wiki Experiment Pages")
TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+\.md)`\s*\|\s*`([^`]+\.md)`\s*\|")
END_SECTION_RE = re.compile(r"^##\s+[A-Z]")


def parse_wiki_renames() -> list[tuple[str, str]]:
    """Return list of (old_md, new_md) from the Wiki Experiment Pages table."""
    pairs: list[tuple[str, str]] = []
    in_section = False
    with open(RENAMES) as f:
        for line in f:
            if WIKI_SECTION_RE.match(line):
                in_section = True
                continue
            if in_section and END_SECTION_RE.match(line) and not WIKI_SECTION_RE.match(line):
                break
            if not in_section:
                continue
            m = TABLE_ROW_RE.match(line)
            if not m:
                continue
            old, new = m.group(1).strip(), m.group(2).strip()
            if old.lower() == "old":
                continue
            pairs.append((old, new))
    return pairs


def stem(md_filename: str) -> str:
    """Strip trailing .md."""
    return md_filename[:-3] if md_filename.endswith(".md") else md_filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log: list[str] = []

    pairs = parse_wiki_renames()
    log.append(f"Parsed {len(pairs)} wiki-page renames from RENAMES.md")

    # Build rename dict + [[link]] rewrite dict
    link_rewrites: dict[str, str] = {}
    for old, new in pairs:
        old_stem = stem(old)
        new_stem = stem(new)
        link_rewrites[old_stem] = new_stem

    # Also rewrite compound forms like [[micro107-108]] → [[m107_m108_ipl_cost]]
    # This is already handled by the table (old_stem "micro107-108" → new_stem "m107_m108_ipl_cost"), just ensure inclusion.

    # Step 1: rename files
    log.append("\n── Step 1: git mv wiki experiment pages ──")
    for old, new in pairs:
        src = WIKI / "experiments" / old
        dst = WIKI / "experiments" / new
        if not src.exists():
            log.append(f"SKIP (src missing): {src.relative_to(ROOT)}")
            continue
        if dst.exists():
            log.append(f"SKIP (dst exists): {dst.relative_to(ROOT)}")
            continue
        rel_src, rel_dst = str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))
        if args.dry_run:
            log.append(f"DRY: git mv {rel_src} {rel_dst}")
            continue
        r = subprocess.run(["git", "mv", rel_src, rel_dst], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            log.append(f"MV: {rel_src} → {rel_dst}")
        else:
            log.append(f"FAIL: {rel_src} → {rel_dst}  ({r.stderr.strip()})")

    # Step 2: rewrite [[links]] across all wiki/ markdown files
    log.append("\n── Step 2: rewrite [[microNN]] → [[mNNN_semantic]] ──")
    # Build one regex that matches any old_stem inside [[...]]
    # Longest-first to avoid partial overlaps (e.g. micro11 vs micro111)
    sorted_keys = sorted(link_rewrites.keys(), key=len, reverse=True)
    pattern = re.compile(r"\[\[(" + "|".join(re.escape(k) for k in sorted_keys) + r")(\|[^\]]*)?\]\]")

    def replace(m: re.Match) -> str:
        key = m.group(1)
        display = m.group(2) or ""
        return f"[[{link_rewrites[key]}{display}]]"

    touched = 0
    edits = 0
    for md in sorted(WIKI.rglob("*.md")):
        content = md.read_text()
        new_content, n = pattern.subn(replace, content)
        if n == 0:
            continue
        touched += 1
        edits += n
        rel = md.relative_to(ROOT)
        if args.dry_run:
            log.append(f"DRY edit {n} links in: {rel}")
        else:
            md.write_text(new_content)
            log.append(f"EDIT {n} links: {rel}")
    log.append(f"\nTotal: {edits} link rewrites across {touched} files")

    # Step 3: also rewrite in notebooks/inversion/*.md top-level (EXPERIMENTS.md, DEAD_ENDS.md, etc.)
    log.append("\n── Step 3: rewrite top-level inversion docs ──")
    for md in sorted(INVERSION.glob("*.md")):
        if md.name in ("RENAMES.md", "CONTRIBUTING.md"):
            continue  # leave RENAMES.md + CONTRIBUTING.md alone (they reference old names intentionally)
        content = md.read_text()
        new_content, n = pattern.subn(replace, content)
        if n == 0:
            continue
        rel = md.relative_to(ROOT)
        if args.dry_run:
            log.append(f"DRY edit {n} links in: {rel}")
        else:
            md.write_text(new_content)
            log.append(f"EDIT {n} links: {rel}")

    LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote log to {LOG}")
    counts: dict[str, int] = {}
    for line in log:
        tag = line.split(":", 1)[0].split(" ", 1)[0]
        if tag and not tag.startswith("──"):
            counts[tag] = counts.get(tag, 0) + 1
    print("Tag counts:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
