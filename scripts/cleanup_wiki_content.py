#!/usr/bin/env python3
"""
cleanup_wiki_content.py — Phase B mechanical sweep.

Rewrites stale micro\\d+ references across .md files under notebooks/inversion/
and prepares them for narrative content surgery. Does NOT touch prose SEMANTICS
(that's left for per-page human work); this is the mechanical layer.

Transforms, in order per file:

  1. [[microNN]] / [[microNN_suffix]] wiki-link tokens
     → full new page name from RENAMES.md wiki-page table.
     (Any link to a non-experiment page that happens to start with 'micro' is
     also handled by falling back to bare `m{NNN}{suffix}` if no entry matches.)

  2. Script path references (notebooks/inversion/**/microNN_*.py OR just
     microNN_*.py) → the new script name from RENAMES.md script tables.

  3. YAML frontmatter `raw/inversion_diagnostics/microNNN/...` paths
     → `raw/inversion_diagnostics/m{NNN}/...` (follows Phase A dir rename).

  4. Remaining plain `\\bmicro(\\d+)([a-z]\\d*)?` tokens (prose, print-strings,
     comments) → `m{NNN}{suffix}`. Mirrors the Phase A .py-file rule.

  5. `updated:` YAML field bumped to today's date IF the file's body content
     changed.

Files skipped:
  - RENAMES.md — it IS the mapping, must retain old names as keys.
  - CONTRIBUTING.md — references old names as historical examples.
  - progress_tracker.md / anything not under wiki/ or reports/ — caller decides.

Usage: python3 scripts/cleanup_wiki_content.py [--dry-run] [--bump-updated]
Writes: scripts/cleanup_wiki_content.log
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVERSION = ROOT / "notebooks" / "inversion"
RENAMES = INVERSION / "RENAMES.md"
LOG_PATH = ROOT / "scripts" / "cleanup_wiki_content.log"

TODAY = datetime.date.today().isoformat()

# Files we explicitly never touch (mapping source or out-of-scope).
SKIP_FILES = {
    INVERSION / "RENAMES.md",
    INVERSION / "CONTRIBUTING.md",
    INVERSION / "progress_tracker.md",
    # Top-level docs are Phase D, not B:
    INVERSION / "EXPERIMENTS.md",
    INVERSION / "DATA_INTEGRITY_BUG.md",
    INVERSION / "DEAD_ENDS.md",
}


# ── RENAMES.md parsing ───────────────────────────────────────────────────

SCRIPT_TABLE_ROW = re.compile(
    r"^\|\s*`([^`]+\.py)`\s*\|\s*`([^`]+\.py)`\s*\|"
)
WIKI_TABLE_ROW = re.compile(
    r"^\|\s*`([^`]+\.md)`\s*\|\s*`([^`]+\.md)`\s*\|"
)


def parse_renames() -> tuple[dict[str, str], dict[str, str]]:
    """Return (script_map, wiki_map) — old filename → new filename (no path)."""
    script_map: dict[str, str] = {}
    wiki_map: dict[str, str] = {}
    text = RENAMES.read_text()
    # Simple split — the Wiki table is easy to isolate by section header.
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


# ── Transform rules ──────────────────────────────────────────────────────

MICRO_TOKEN_RE = re.compile(r"\bmicro(\d+)([a-z]\d*)?")


def simple_prefix_rewrite(text: str) -> tuple[str, int]:
    """Apply the Phase A rule: \\bmicro(\\d+)([a-z]\\d*)? → m{NNN:03d}{suffix}."""
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
    """Return old-page-stem → new-page-stem (no .md extension) for [[links]].

    Handles the hyphenated compound keys like 'micro87-88' that map to
    'm087_m088_fast_grid'.
    """
    out: dict[str, str] = {}
    for old_md, new_md in wiki_map.items():
        old_stem = old_md.removesuffix(".md")
        new_stem = new_md.removesuffix(".md")
        out[old_stem] = new_stem
    return out


def rewrite_wiki_links(text: str, link_map: dict[str, str]) -> tuple[str, int]:
    """Rewrite [[microNN...]] and [[microNN...|alias]] using the explicit map.

    Unmapped micro-prefixed link targets fall back to bare m{NNN}{suffix}.
    """
    count = 0
    # Match [[target]] or [[target|alias]], target must start with 'micro'.
    link_re = re.compile(r"\[\[(micro[^\]\|]+?)(\|[^\]]+)?\]\]")

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        target = m.group(1)
        alias = m.group(2) or ""
        if target in link_map:
            new_target = link_map[target]
        else:
            # Fall back to simple prefix swap
            new_target, n = simple_prefix_rewrite(target)
            if n == 0:
                return m.group(0)  # no change
        count += 1
        return f"[[{new_target}{alias}]]"

    new_text = link_re.sub(repl, text)
    return new_text, count


def rewrite_script_paths(text: str, script_map: dict[str, str]) -> tuple[str, int]:
    """Rewrite micro*_*.py references using full old→new script map.

    Matches bare filenames (e.g. `micro122_hessian_at_truth.py`) anywhere in the
    text — in prose, in paths, in code spans. Path prefixes are preserved.
    """
    count = 0
    # Sort keys by length desc so longer matches win (e.g. micro119v2_ before micro119_).
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


# ── Frontmatter helpers ──────────────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
UPDATED_LINE_RE = re.compile(r"^(updated:\s*)(\S+)(\s*)$", re.MULTILINE)


def bump_updated_field(text: str) -> tuple[str, bool]:
    """If frontmatter has an `updated:` field, set it to today. Return (text, changed)."""
    fm = FRONTMATTER_RE.match(text)
    if not fm:
        return text, False
    fm_body = fm.group(1)
    new_fm_body, n = UPDATED_LINE_RE.subn(rf"\g<1>{TODAY}\g<3>", fm_body)
    if n == 0 or new_fm_body == fm_body:
        return text, False
    return text[: fm.start(1)] + new_fm_body + text[fm.end(1):], True


# ── File selector ────────────────────────────────────────────────────────

def target_files() -> list[Path]:
    out: list[Path] = []
    # All wiki pages
    wiki_dir = INVERSION / "wiki"
    if wiki_dir.exists():
        out.extend(sorted(wiki_dir.rglob("*.md")))
    # All series reports
    for series_dir in sorted(INVERSION.iterdir()):
        if not series_dir.is_dir():
            continue
        reports = series_dir / "reports"
        if reports.exists():
            out.extend(sorted(reports.rglob("*.md")))
        # Series-level FINDINGS.md / OTHER_NAME.md files
        for md in sorted(series_dir.glob("*.md")):
            out.append(md)
    # Deduplicate + filter skipped
    seen = set()
    filtered = []
    for p in out:
        if p in SKIP_FILES:
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        filtered.append(p)
    return filtered


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bump-updated", action="store_true",
                        help="Set `updated: YYYY-MM-DD` to today wherever the body changes")
    args = parser.parse_args()

    log: list[str] = []

    script_map, wiki_map = parse_renames()
    link_map = build_wiki_link_map(wiki_map)
    log.append(f"Parsed {len(script_map)} script mappings, "
               f"{len(wiki_map)} wiki page mappings, "
               f"{len(link_map)} link targets")

    files = target_files()
    log.append(f"Scanning {len(files)} .md files")

    total_files_changed = 0
    total_links = 0
    total_script_paths = 0
    total_tokens = 0
    total_updated_bumped = 0

    for md in files:
        try:
            original = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            log.append(f"SKIP: {md.relative_to(ROOT)}  ({e})")
            continue

        text = original

        text, n_links = rewrite_wiki_links(text, link_map)
        text, n_paths = rewrite_script_paths(text, script_map)
        text, n_tokens = simple_prefix_rewrite(text)

        body_changed = (text != original)
        if args.bump_updated and body_changed:
            text, updated_bumped = bump_updated_field(text)
            if updated_bumped:
                total_updated_bumped += 1
        else:
            updated_bumped = False

        if text == original:
            continue

        total_files_changed += 1
        total_links += n_links
        total_script_paths += n_paths
        total_tokens += n_tokens

        rel = md.relative_to(ROOT)
        tags = []
        if n_links:
            tags.append(f"links={n_links}")
        if n_paths:
            tags.append(f"scripts={n_paths}")
        if n_tokens:
            tags.append(f"tokens={n_tokens}")
        if updated_bumped:
            tags.append("updated")
        log.append(f"REWRITE ({', '.join(tags)}): {rel}")

        if not args.dry_run:
            md.write_text(text, encoding="utf-8")

    log.append("")
    log.append(f"Summary: {total_files_changed} files changed")
    log.append(f"  wiki links : {total_links}")
    log.append(f"  script paths: {total_script_paths}")
    log.append(f"  prose tokens: {total_tokens}")
    log.append(f"  updated: fields bumped: {total_updated_bumped}")

    LOG_PATH.write_text("\n".join(log) + "\n")
    print(f"Wrote log to {LOG_PATH}")
    print("\n".join(log[-5:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
