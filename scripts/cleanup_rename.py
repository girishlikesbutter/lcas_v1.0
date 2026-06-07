#!/usr/bin/env python3
"""
cleanup_rename.py — Execute the mass rename documented in RENAMES.md.

Parses notebooks/inversion/RENAMES.md script tables and executes `git mv`
per entry, honouring the strategist's D1–D6 overrides:

  D1: Three 07_* dirs merge into 07_omega_winding/
  D2: 00_pipeline_reference/ moves to notebooks/tutorials/ (outside inversion)
  D3: 06_omega_basin/ deleted; 06_omega_bridging_benchmarks/ → 06_omega_bridging/
      All its scripts go to archive/ (pre-systematic benchmark variants)
  D4: Letter-suffix convention for duplicate experiment numbers
  D5/D6: handled in Phase 5 (result-dir restructure) — not this script

Usage: python3 scripts/cleanup_rename.py [--dry-run]

Writes: scripts/cleanup_rename.log
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
RENAMES = INVERSION / "RENAMES.md"
LOG = ROOT / "scripts" / "cleanup_rename.log"


# D1: three 07_* dirs merge into one
SERIES_07_MERGED = "07_omega_winding"
SERIES_07_OLD = ["07_L_conservation", "07_multi_epoch_scoring", "07_winding_enumeration"]

# D2: 00_pipeline_reference relocates OUT of notebooks/inversion
# Handled as a special case in main(), not via the table.

# D3: 06 renames + all-archive
SERIES_06_OLD = "06_omega_bridging_benchmarks"
SERIES_06_NEW = "06_omega_bridging"
SERIES_06_EMPTY = "06_omega_basin"  # delete (if empty); if non-empty, archive all

# D4: duplicate-number resolutions (series 11 & 12 collisions) — explicit overrides
# Table in RENAMES.md resolution block takes precedence over generic table rows.
D4_OVERRIDES: dict[str, str] = {
    "11_casadi_formulation/micro89_diagnosis.py": "m089d_diagnostic_pipeline.py",
    "11_casadi_formulation/micro89_production.py": "m089_production_pipeline.py",
    "11_casadi_formulation/micro93b_peak_census.py": "m093c_peak_census.py",
    "12_brightness_surface/micro117_harvester.py": "m117_harvester.py",
    "12_brightness_surface/micro117_validate.py": "m117v_validate.py",
    "12_brightness_surface/micro118_kernel.py": "m118_kernel.py",
    "12_brightness_surface/micro118_costs.py": "m118c_costs.py",
    "12_brightness_surface/micro118_diag.py": "m118d_diag.py",
    "12_brightness_surface/micro119_attitude_isoshell.py": "m119_attitude_isoshell.py",
    "12_brightness_surface/micro119_multiseed_aggregate.py": "m119a_multiseed_aggregate.py",
}


# ── Table parser ─────────────────────────────────────────────────────────

SERIES_HEADER_RE = re.compile(r"^###\s+(\d+[a-z]?_[a-zA-Z0-9_]+)\s*\(")
SECTION_RESET_RE = re.compile(r"^##\s+[A-Z]")   # reset series on H2 (e.g. "## Wiki", "## Result")
TABLE_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*$"
)


def parse_renames() -> list[tuple[str, str, str]]:
    """Return list of (series_dir_old, old_file, new_file_spec)."""
    out: list[tuple[str, str, str]] = []
    series = None
    with open(RENAMES) as f:
        for line in f:
            m = SERIES_HEADER_RE.match(line)
            if m:
                series = m.group(1)
                continue
            if SECTION_RESET_RE.match(line):
                series = None  # leaving the script-series tables (e.g. entering "## Wiki" or "## Result")
                continue
            if not series:
                continue
            m = TABLE_ROW_RE.match(line)
            if not m:
                continue
            old, new = m.group(1).strip(), m.group(2).strip()
            # Skip header rows
            if old.lower() == "old" or new.lower() == "new":
                continue
            out.append((series, old, new))
    return out


# ── Destination resolver ────────────────────────────────────────────────

def resolve_destination(series_old: str, old_file: str, new_spec: str) -> Path | None:
    """Compute the final on-disk path for a rename, or None to skip."""

    # D4 overrides (check first)
    key = f"{series_old}/{old_file}"
    if key in D4_OVERRIDES:
        new_spec = D4_OVERRIDES[key]

    # Determine series dir at destination
    if series_old in SERIES_07_OLD:
        series_new = SERIES_07_MERGED
    elif series_old == SERIES_06_OLD:
        series_new = SERIES_06_NEW
    elif series_old == SERIES_06_EMPTY:
        # If anything sneaks in here, it goes to archive in the (now-deleted) dir's
        # successor. But this series is empty — skip.
        return None
    elif series_old == "00_pipeline_reference":
        # Moved outside inversion/ — handled separately. Skip here.
        return None
    else:
        series_new = series_old

    # Archive spec: "archive/foo.py" → series_new/archive/foo.py
    if new_spec.startswith("archive/"):
        return INVERSION / series_new / "archive" / new_spec.removeprefix("archive/")

    # D3: all scripts in 06 are archived regardless of table spec
    if series_old == SERIES_06_OLD:
        filename = new_spec.split("/")[-1]  # drop any subdir in spec
        return INVERSION / series_new / "archive" / filename

    # Normal rename
    return INVERSION / series_new / new_spec


def move_file(src: Path, dst: Path, dry_run: bool, log: list[str]) -> None:
    if not src.exists():
        log.append(f"SKIP (missing): {src.relative_to(ROOT)}")
        return
    if dst.exists() and dst.resolve() == src.resolve():
        log.append(f"SKIP (same): {src.relative_to(ROOT)}")
        return
    if dst.exists():
        log.append(f"CONFLICT (dst exists): {dst.relative_to(ROOT)} (from {src.relative_to(ROOT)})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = str(src.relative_to(ROOT))
    rel_dst = str(dst.relative_to(ROOT))
    if dry_run:
        log.append(f"DRY: git mv {rel_src} {rel_dst}")
        return
    result = subprocess.run(
        ["git", "mv", rel_src, rel_dst],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log.append(f"MV: {rel_src} → {rel_dst}")
    else:
        log.append(f"FAIL ({result.stderr.strip()}): git mv {rel_src} {rel_dst}")


def move_dir_contents(src_dir: Path, dst_dir: Path, dry_run: bool, log: list[str]) -> None:
    """Move every entry from src_dir to dst_dir, preserving git history via git mv."""
    if not src_dir.exists():
        log.append(f"SKIP dir (missing): {src_dir.relative_to(ROOT)}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_dir.iterdir()):
        dst = dst_dir / entry.name
        if entry.is_dir():
            continue  # handled recursively by entry-level moves elsewhere
        rel_src = str(entry.relative_to(ROOT))
        rel_dst = str(dst.relative_to(ROOT))
        if dst.exists():
            log.append(f"CONFLICT (dst exists): {rel_dst}")
            continue
        if dry_run:
            log.append(f"DRY: git mv {rel_src} {rel_dst}")
            continue
        result = subprocess.run(
            ["git", "mv", rel_src, rel_dst],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.append(f"MV: {rel_src} → {rel_dst}")
        else:
            log.append(f"FAIL ({result.stderr.strip()}): git mv {rel_src} {rel_dst}")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log: list[str] = []

    # D2: move 00_pipeline_reference OUT of notebooks/inversion
    src_00 = INVERSION / "00_pipeline_reference"
    dst_00 = ROOT / "notebooks" / "tutorials"
    if src_00.exists():
        log.append(f"\n── D2: Move {src_00.relative_to(ROOT)} → {dst_00.relative_to(ROOT)} ──")
        if not args.dry_run:
            dst_00.mkdir(parents=True, exist_ok=True)
        for entry in sorted(src_00.iterdir()):
            if entry.name == "__pycache__":
                continue
            dst = dst_00 / entry.name
            rel_src = str(entry.relative_to(ROOT))
            rel_dst = str(dst.relative_to(ROOT))
            if args.dry_run:
                log.append(f"DRY: git mv {rel_src} {rel_dst}")
                continue
            result = subprocess.run(
                ["git", "mv", rel_src, rel_dst],
                cwd=ROOT, capture_output=True, text=True,
            )
            if result.returncode == 0:
                log.append(f"MV: {rel_src} → {rel_dst}")
            else:
                # Untracked files: fall back to shutil (likely the .ipynb files)
                try:
                    if dst.exists():
                        log.append(f"CONFLICT: {rel_dst}")
                    else:
                        shutil.move(str(entry), str(dst))
                        log.append(f"FS-MV: {rel_src} → {rel_dst}  [{result.stderr.strip()}]")
                except Exception as e:
                    log.append(f"FAIL: {rel_src} → {rel_dst}  ({e})")
        # Attempt to remove now-empty src dir
        try:
            if not args.dry_run and src_00.exists() and not any(src_00.iterdir()):
                src_00.rmdir()
                log.append(f"RMDIR: {src_00.relative_to(ROOT)}")
        except Exception as e:
            log.append(f"RMDIR FAIL: {src_00.relative_to(ROOT)}  ({e})")

    # D3: delete empty 06_omega_basin/ if empty, else archive its scripts
    src_06e = INVERSION / "06_omega_basin"
    if src_06e.exists():
        non_meta = [p for p in src_06e.iterdir() if p.name not in ("__pycache__",)]
        if not non_meta:
            log.append(f"\n── D3: Delete empty {src_06e.relative_to(ROOT)} ──")
            if not args.dry_run:
                shutil.rmtree(src_06e)
                log.append(f"RMTREE: {src_06e.relative_to(ROOT)}")
            else:
                log.append(f"DRY: rmtree {src_06e.relative_to(ROOT)}")
        else:
            # Archive its .py contents into 06_omega_bridging/archive/ (since 06_omega_basin is decomissioned)
            log.append(f"\n── D3: Archive contents of {src_06e.relative_to(ROOT)} into 06_omega_bridging/archive ──")
            dst_archive = INVERSION / SERIES_06_NEW / "archive"
            if not args.dry_run:
                dst_archive.mkdir(parents=True, exist_ok=True)
            for entry in sorted(src_06e.iterdir()):
                if entry.name == "__pycache__":
                    continue
                dst = dst_archive / entry.name
                rel_src = str(entry.relative_to(ROOT))
                rel_dst = str(dst.relative_to(ROOT))
                if args.dry_run:
                    log.append(f"DRY: git mv {rel_src} {rel_dst}")
                    continue
                r = subprocess.run(["git", "mv", rel_src, rel_dst], cwd=ROOT, capture_output=True, text=True)
                if r.returncode == 0:
                    log.append(f"MV: {rel_src} → {rel_dst}")
                else:
                    try:
                        shutil.move(str(entry), str(dst))
                        log.append(f"FS-MV: {rel_src} → {rel_dst}  [{r.stderr.strip()}]")
                    except Exception as e:
                        log.append(f"FAIL: {rel_src} → {rel_dst}  ({e})")

    # D3: rename 06_omega_bridging_benchmarks → 06_omega_bridging
    src_06 = INVERSION / SERIES_06_OLD
    dst_06 = INVERSION / SERIES_06_NEW
    if src_06.exists() and not dst_06.exists():
        log.append(f"\n── D3: Rename series dir {src_06.relative_to(ROOT)} → {dst_06.relative_to(ROOT)} ──")
        if args.dry_run:
            log.append(f"DRY: git mv {src_06.relative_to(ROOT)} {dst_06.relative_to(ROOT)}")
        else:
            r = subprocess.run(["git", "mv", str(src_06.relative_to(ROOT)), str(dst_06.relative_to(ROOT))],
                               cwd=ROOT, capture_output=True, text=True)
            log.append(f"MV: {src_06.relative_to(ROOT)} → {dst_06.relative_to(ROOT)}  (rc={r.returncode}; {r.stderr.strip()})")
    elif dst_06.exists() and src_06.exists():
        log.append(f"CONFLICT: both {src_06.name} and {dst_06.name} exist")

    # D1: merge 07_* dirs into 07_omega_winding/
    # We do this by creating the target dir and git-mv'ing scripts/FINDINGS from each.
    log.append(f"\n── D1: Consolidate {SERIES_07_OLD} → {SERIES_07_MERGED} ──")
    dst_07 = INVERSION / SERIES_07_MERGED
    if not args.dry_run:
        dst_07.mkdir(parents=True, exist_ok=True)
    for old_series in SERIES_07_OLD:
        src = INVERSION / old_series
        if not src.exists():
            continue
        for entry in sorted(src.iterdir()):
            if entry.is_dir():
                continue
            if entry.name == "FINDINGS.md":
                # Rename to FINDINGS_{subtopic}.md so the three preserved
                # (will be merged into one FINDINGS.md in Phase 3 wiki surgery)
                subtopic = old_series.removeprefix("07_")
                dst_name = f"FINDINGS_{subtopic}.md"
            else:
                dst_name = entry.name  # File rename (micro → m) handled by table loop below
            dst = dst_07 / dst_name
            rel_src = str(entry.relative_to(ROOT))
            rel_dst = str(dst.relative_to(ROOT))
            if dst.exists():
                log.append(f"CONFLICT: {rel_dst}")
                continue
            if args.dry_run:
                log.append(f"DRY: git mv {rel_src} {rel_dst}")
                continue
            r = subprocess.run(["git", "mv", rel_src, rel_dst], cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                log.append(f"MV: {rel_src} → {rel_dst}")
            else:
                log.append(f"FAIL: {rel_src} → {rel_dst}  ({r.stderr.strip()})")

    # Attempt to remove now-empty 07_* old dirs
    for old_series in SERIES_07_OLD:
        src = INVERSION / old_series
        if src.exists():
            try:
                remaining = [p for p in src.iterdir() if p.name != "__pycache__"]
                if not remaining:
                    if not args.dry_run:
                        # Also remove __pycache__ if present
                        for p in list(src.iterdir()):
                            if p.is_dir():
                                shutil.rmtree(p)
                            else:
                                p.unlink()
                        src.rmdir()
                        log.append(f"RMDIR: {src.relative_to(ROOT)}")
                    else:
                        log.append(f"DRY: rmdir {src.relative_to(ROOT)}")
                else:
                    log.append(f"NOT EMPTY: {src.relative_to(ROOT)} — {[p.name for p in remaining]}")
            except Exception as e:
                log.append(f"RMDIR FAIL: {src.relative_to(ROOT)}  ({e})")

    # ── Phase 2 main: table-driven file renames ──
    log.append("\n── Phase 2 main: table-driven file renames ──")
    rows = parse_renames()
    log.append(f"Parsed {len(rows)} rename rows from RENAMES.md")

    skipped_series = {"00_pipeline_reference", SERIES_06_EMPTY}  # D2, D3
    for series, old, new in rows:
        if series in skipped_series:
            continue

        # Source path: the file lives in its original series dir (for 07 it's already been moved above)
        if series in SERIES_07_OLD:
            # Already moved above as part of D1 consolidation (with FINDINGS.md renamed,
            # but script files kept their micro* filename). Now we rename micro* → m* inside the new dir.
            src = INVERSION / SERIES_07_MERGED / old
        elif series == SERIES_06_OLD:
            # Dir was renamed; scripts still have micro* names there
            src = INVERSION / SERIES_06_NEW / old
        else:
            src = INVERSION / series / old

        dst = resolve_destination(series, old, new)
        if dst is None:
            continue
        move_file(src, dst, args.dry_run, log)

    # ── Write log ──
    LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote log to {LOG}")
    print(f"Lines: {len(log)}")
    # Summary
    counts: dict[str, int] = {}
    for line in log:
        tag = line.split(":", 1)[0].split(" ", 1)[0]
        if not tag or tag.startswith("──"):
            continue
        counts[tag] = counts.get(tag, 0) + 1
    print("Tag counts:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
