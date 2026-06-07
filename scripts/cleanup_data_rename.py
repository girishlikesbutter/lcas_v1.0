#!/usr/bin/env python3
"""
cleanup_data_rename.py — Phase A of the organizational cleanup.

Does two coupled operations atomically in one commit:

  A1: Rewrite ~1765 `micro\\d+` cross-refs in `notebooks/inversion/**/*.py`
      (OUT_BASE strings, data paths, docstrings, log messages).
      Rule: `\\bmicro(\\d+)([a-z]\\d*)?` → `m{NNN:03d}{suffix}`.

  A2: Rename ~286 entries under `data/results/inversion_diagnostics/`:
      - `microNN[_suffix]` (dir or file)   → `m{NNN}[_suffix]`
      - `harvester_lever1/`, `harvester_sanity/`, `inline_omega_selection/`
                                           → `analyses/{name}/`
      - `isoshell_viewer/`                 → `shared/isoshell_viewer/`
      - `plots/`, `unsorted/`              → `archive/{name}/`
      - `is901_brightness_table.npz`       → `shared/brightness_tables/...`
      - `wrappedbest_seedNNN/`             → `m126_wrapped/wrappedbest_seedNNN/`
      - `wrappedbest_seedNNN_lc_compare.png`
                                           → `m126_wrapped/wrappedbest_seedNNN_lc_compare.png`

The two operations are coupled: many cross-refs ARE data paths, so
rewriting refs before dirs exist would desync them. One script, one commit.

Prior-session gotcha: dry-run MUST NOT `mkdir` dest dirs (the first
version did, and left empty `archive/` that made the next `git mv
{src} {dst}` fail with CONFLICT). Every `mkdir` here is gated on
`if not args.dry_run`.

Usage: python3 scripts/cleanup_data_rename.py [--dry-run]
Writes: scripts/cleanup_data_rename.log
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVERSION = ROOT / "notebooks" / "inversion"
RESULTS = ROOT / "data" / "results" / "inversion_diagnostics"
LOG_PATH = ROOT / "scripts" / "cleanup_data_rename.log"


# ── Cross-ref rewrite rule (A1) ──────────────────────────────────────────

# `\bmicro(\d+)([a-z]\d*)?` captures:
#   group 1: digits to zero-pad
#   group 2: optional letter + optional digits (e.g. 'b', 'd', 'v2', 'a')
MICRO_TOKEN_RE = re.compile(r"\bmicro(\d+)([a-z]\d*)?")


def rewrite_micro_tokens(text: str) -> tuple[str, int]:
    """Return (rewritten_text, n_replacements)."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        num = int(m.group(1))
        suffix = m.group(2) or ""
        return f"m{num:03d}{suffix}"

    new_text = MICRO_TOKEN_RE.sub(repl, text)
    return new_text, count


# ── Orphan / explicit relocation map (A2) ────────────────────────────────

# Entries NOT starting with "micro" — these need explicit destinations.
# Key is entry name relative to RESULTS; value is relative destination path.
ORPHAN_MAP: dict[str, str] = {
    "harvester_lever1": "analyses/harvester_lever1",
    "harvester_sanity": "analyses/harvester_sanity",
    "inline_omega_selection": "analyses/inline_omega_selection",
    "isoshell_viewer": "shared/isoshell_viewer",
    "plots": "archive/plots",
    "unsorted": "archive/unsorted",
    "is901_brightness_table.npz": "shared/brightness_tables/is901_brightness_table.npz",
}

# wrappedbest_* entries all fold into the renamed m126 dir.
WRAPPEDBEST_DEST_PARENT = "m126_wrapped"


# ── git mv with shutil fallback ──────────────────────────────────────────

def move_entry(src: Path, dst: Path, dry_run: bool, log: list[str]) -> bool:
    """Move src → dst via git mv, falling back to shutil for untracked.

    Returns True if the move was performed (or would be, in dry-run)."""
    if not src.exists():
        log.append(f"SKIP (missing): {src.relative_to(ROOT)}")
        return False
    if dst.exists():
        if dst.resolve() == src.resolve():
            log.append(f"SKIP (same): {src.relative_to(ROOT)}")
            return False
        log.append(f"CONFLICT (dst exists): {dst.relative_to(ROOT)} (from {src.relative_to(ROOT)})")
        return False

    rel_src = str(src.relative_to(ROOT))
    rel_dst = str(dst.relative_to(ROOT))

    if dry_run:
        log.append(f"DRY: mv {rel_src} → {rel_dst}")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        ["git", "mv", rel_src, rel_dst],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode == 0:
        log.append(f"GIT-MV: {rel_src} → {rel_dst}")
        return True

    # Fall back to shutil for untracked entries.
    try:
        shutil.move(str(src), str(dst))
        log.append(f"FS-MV: {rel_src} → {rel_dst}  [git: {r.stderr.strip()}]")
        return True
    except Exception as e:
        log.append(f"FAIL: {rel_src} → {rel_dst}  ({e}; git: {r.stderr.strip()})")
        return False


# ── Dir rename planner (A2) ──────────────────────────────────────────────

# Same pattern as cross-refs, but anchored to start of entry name (no \b needed).
DIR_NAME_RE = re.compile(r"^micro(\d+)([a-z]\d*)?(.*)$")

# Dir-name renames that need explicit overrides (RENAMES.md D4 equivalents
# for directories). Empty for now — simple prefix+zero-pad handles everything.
# If later we want to disambiguate e.g. `micro89_diagnosis` → `m089d_diagnosis`,
# add entries here.
DIR_OVERRIDES: dict[str, str] = {}


def plan_entry_rename(name: str) -> str | None:
    """Return new name for a top-level entry in RESULTS, or None to skip."""
    if name in DIR_OVERRIDES:
        return DIR_OVERRIDES[name]

    m = DIR_NAME_RE.match(name)
    if m:
        num = int(m.group(1))
        sub_token = m.group(2) or ""    # 'b', 'd', 'v2', ...
        rest = m.group(3) or ""         # '_foo' or '.npz' or ''
        return f"m{num:03d}{sub_token}{rest}"

    return None  # Non-micro entries handled via ORPHAN_MAP / special cases.


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-a1", action="store_true",
                        help="Skip the .py cross-ref rewrite pass")
    parser.add_argument("--skip-a2", action="store_true",
                        help="Skip the data-dir rename pass")
    args = parser.parse_args()

    log: list[str] = []

    # ── Pass A2: data dir renames (do this FIRST so paths stabilise) ──
    if not args.skip_a2:
        log.append("\n══════════ Pass A2: data-dir renames ══════════")

        if not RESULTS.exists():
            log.append(f"FATAL: {RESULTS.relative_to(ROOT)} does not exist")
            LOG_PATH.write_text("\n".join(log) + "\n")
            return 1

        entries = sorted([p for p in RESULTS.iterdir() if p.name not in ("__pycache__",)])

        # A2.1: Handle wrappedbest_* entries (fold into m126_wrapped/)
        log.append(f"\n── A2.1: Fold wrappedbest_* into {WRAPPEDBEST_DEST_PARENT}/ ──")
        wb_parent = RESULTS / WRAPPEDBEST_DEST_PARENT
        # The m126 dir exists as `micro126_wrapped` before the main rename; but
        # we want wrappedbest_* to land in `m126_wrapped` AFTER m126 is renamed.
        # Strategy: rename m126 first, then move wrappedbest_* into it.
        src_m126 = RESULTS / "micro126_wrapped"
        if src_m126.exists() and not wb_parent.exists():
            move_entry(src_m126, wb_parent, args.dry_run, log)

        for entry in entries:
            if not entry.name.startswith("wrappedbest_"):
                continue
            dst = wb_parent / entry.name
            move_entry(entry, dst, args.dry_run, log)

        # A2.2: Orphan relocations (non-micro entries)
        log.append("\n── A2.2: Orphan dir/file relocations ──")
        for orphan_name, dst_rel in ORPHAN_MAP.items():
            src = RESULTS / orphan_name
            dst = RESULTS / dst_rel
            move_entry(src, dst, args.dry_run, log)

        # A2.3: Generic microNN* prefix rename for everything else
        log.append("\n── A2.3: Generic microNN → m{NNN} prefix rename ──")
        # Re-list: some entries were consumed above.
        for entry in sorted([p for p in RESULTS.iterdir() if p.name not in ("__pycache__",)]):
            # Skip already-handled patterns
            if entry.name.startswith("wrappedbest_"):
                continue
            if entry.name in ORPHAN_MAP:
                continue
            if entry.name in ("shared", "analyses", "archive"):
                continue
            if not entry.name.startswith("micro"):
                # Non-micro, non-orphan: leave alone but log
                log.append(f"SKIP (no rule): {entry.relative_to(ROOT)}")
                continue

            new_name = plan_entry_rename(entry.name)
            if new_name is None or new_name == entry.name:
                log.append(f"SKIP (no rename): {entry.relative_to(ROOT)}")
                continue

            dst = RESULTS / new_name
            move_entry(entry, dst, args.dry_run, log)

    # ── Pass A1: rewrite micro\d+ refs in .py files ──
    if not args.skip_a1:
        log.append("\n══════════ Pass A1: .py cross-ref rewrite ══════════")

        py_files = sorted(INVERSION.rglob("*.py"))
        total_replacements = 0
        total_files_changed = 0
        for py in py_files:
            try:
                original = py.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                log.append(f"SKIP (binary?): {py.relative_to(ROOT)}")
                continue

            rewritten, n = rewrite_micro_tokens(original)
            if n == 0:
                continue

            total_replacements += n
            total_files_changed += 1
            log.append(f"REWRITE ({n}): {py.relative_to(ROOT)}")
            if not args.dry_run:
                py.write_text(rewritten, encoding="utf-8")

        log.append(f"\nA1 summary: {total_replacements} replacements across "
                   f"{total_files_changed} files")

    # ── Write log and print summary ──
    LOG_PATH.write_text("\n".join(log) + "\n")
    print(f"Wrote log to {LOG_PATH}")
    print(f"Log lines: {len(log)}")

    # Tag-based summary
    counts: dict[str, int] = {}
    for line in log:
        if not line or line.startswith("─") or line.startswith("═") or line.startswith("\n"):
            continue
        tag = line.split(" ", 1)[0].rstrip(":")
        counts[tag] = counts.get(tag, 0) + 1
    print("Tag counts:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
