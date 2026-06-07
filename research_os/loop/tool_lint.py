#!/usr/bin/env python3
"""Tool drift-check — the Q1-safe gate in front of the run-button (ADR-0007 §4).

A Tool card *reflects* canonical code: it binds to ONE ``entry_point``
(``path.py:symbol``) at ONE version, and stamps the source file's content hash
(``current_hash``). Code moves; cards rot (the scour already found three drifted
cards). This module re-grounds a Tool against the live source and reports drift:

  - ``missing``    : the entry_point file or symbol no longer resolves.
  - ``hash_moved`` : the file changed but ``current_version`` / ``current_hash`` did
                     not — an UNDOCUMENTED bump (the dangerous, silent kind).
  - ``clean``      : resolves and the hash matches the card.

The executor (``run_tool.py``) calls :func:`lint_tool` BEFORE invoking a Tool and
refuses to run anything but ``clean`` (unless explicitly forced) — the mechanical
"we can SEE there are 3 versions and refuse the stale one" Girish asked for.

Q1 boundary: read-only. It never edits a card or the store — it only reports. A
human (or the close skill) repoints the card and bumps the version when drift is real.

Hash convention matches ``backfill/W-D_generate_cards.py`` (the scour that authored
the cards): ``sha256:`` + first 16 hex of sha256 over the file bytes.

    python research_os/loop/tool_lint.py              # lint every Tool, table + summary
    python research_os/loop/tool_lint.py --tool ID    # lint one Tool
    python research_os/loop/tool_lint.py --for-file P  # lint Tools an edited file affects
    python research_os/loop/tool_lint.py --check       # exit 1 if ANY Tool has drift
    python research_os/loop/tool_lint.py --json        # machine-readable

``--for-file`` is the drift-check hook's seam (``.claude/hooks/ro_tool_lint.sh``): given a
just-edited path it lints exactly the Tools that path touches — the card itself if the path
IS a substrate card, otherwise every Tool whose ``entry_point`` binds that source file (the
silent code-bump). Empty result (path no Tool reflects) → nothing to say.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)  # repo root


def file_hash(abspath: str) -> str:
    """sha256: + first 16 hex of the file bytes (the scour's convention)."""
    with open(abspath, "rb") as fh:
        return "sha256:" + hashlib.sha256(fh.read()).hexdigest()[:16]


def resolve_entry_point(ep: str) -> dict:
    """Ground a 'path.py:Symbol[.method]' entry_point against the live code.

    Returns {resolved, symbol_found, file_exists, hash, path, symbol}. ``hash`` is
    None when the file is absent. Symbol resolution is a deliberately permissive
    textual scan (def/class/assignment) — the same heuristic the scour used, so a
    card the scour blessed lints clean here.
    """
    out = {"resolved": False, "symbol_found": False, "file_exists": False,
           "hash": None, "path": ep or "", "symbol": None}
    if not ep or ":" not in ep:
        return out
    path, _, symbol = ep.partition(":")
    out["path"], out["symbol"] = path, symbol
    abspath = os.path.join(REPO, path)
    if not os.path.isfile(abspath):
        return out
    out["file_exists"] = True
    out["hash"] = file_hash(abspath)
    text = open(abspath, "rb").read().decode("utf-8", "replace")
    leaf = symbol.split(".")[-1]
    pats = [rf"\bdef\s+{re.escape(leaf)}\b",
            rf"\bclass\s+{re.escape(leaf)}\b",
            rf"(?m)^\s*{re.escape(leaf)}\s*="]
    out["symbol_found"] = any(re.search(p, text) for p in pats)
    out["resolved"] = out["symbol_found"]
    return out


def lint_tool(card: dict) -> dict:
    """Drift verdict for one Tool card.

    verdict ∈ {clean, missing, hash_moved, no_entry_point}; ``drift`` is True for
    anything the executor must refuse to run. ``runnable`` = card binds to live code.
    """
    ep = card.get("entry_point")
    res = resolve_entry_point(ep) if ep else None
    current_hash = card.get("current_hash") or None

    if not ep:
        verdict, drift, reason = "no_entry_point", True, "Tool card has no entry_point binding."
    elif not res["file_exists"]:
        verdict, drift, reason = "missing", True, f"entry_point file does not exist: {res['path']}"
    elif not res["symbol_found"]:
        verdict, drift, reason = "missing", True, f"symbol '{res['symbol']}' not found in {res['path']}"
    elif current_hash and res["hash"] != current_hash:
        verdict, drift, reason = ("hash_moved", True,
                                  f"file changed since the card was stamped "
                                  f"(card {current_hash} != live {res['hash']}) — undocumented bump")
    elif not current_hash:
        # Resolves, but the card never recorded a hash: runnable but un-pinned.
        verdict, drift, reason = "clean", False, "resolves; no current_hash on card (un-pinned)"
    else:
        verdict, drift, reason = "clean", False, "resolves; hash matches card"

    return {
        "id": card.get("id"),
        "entry_point": ep,
        "verdict": verdict,
        "drift": drift,
        "runnable": (not drift),
        "reason": reason,
        "card_hash": current_hash,
        "live_hash": (res or {}).get("hash"),
        "tool_version": card.get("current_version"),
    }


def _load_cards(root: str = ROOT) -> dict:
    out = {}
    for f in glob.glob(os.path.join(root, "substrate", "*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        d = json.load(open(f))
        out[d["id"]] = d
    return out


def lint_all(root: str = ROOT) -> list[dict]:
    return [lint_tool(c) for c in sorted(_load_cards(root).values(), key=lambda d: d["id"])]


def lint_one(tool_id: str, root: str = ROOT) -> dict:
    cards = _load_cards(root)
    if tool_id not in cards:
        return {"id": tool_id, "verdict": "unknown_tool", "drift": True, "runnable": False,
                "reason": f"no substrate_component with id '{tool_id}'"}
    return lint_tool(cards[tool_id])


def _rel(path: str) -> str:
    """Repo-relative, normalised — entry_point files are stored repo-relative."""
    return os.path.normpath(os.path.relpath(os.path.abspath(path), REPO))


def tools_binding_file(path: str, root: str = ROOT) -> list[dict]:
    """Tool cards whose entry_point source file == ``path`` (the silent code-bump set)."""
    rel = _rel(path)
    out = []
    for c in _load_cards(root).values():
        ep = c.get("entry_point") or ""
        epfile = ep.partition(":")[0]
        if epfile and os.path.normpath(epfile) == rel:
            out.append(c)
    return out


def lint_for_file(path: str, root: str = ROOT) -> list[dict]:
    """Lint the Tools a just-edited ``path`` affects.

    If ``path`` IS a substrate card → lint that one card (hand-edit drift). Otherwise →
    lint every Tool whose ``entry_point`` binds that source file (the silent code-bump the
    drift-check exists to catch, ADR-0007 §4). Empty list when no Tool reflects the path.
    """
    rel = _rel(path)
    substrate_dir = os.path.join("research_os", "substrate") + os.sep
    if rel.startswith(substrate_dir) and rel.endswith(".json") \
            and not os.path.basename(rel).startswith("_"):
        try:
            return [lint_tool(json.load(open(os.path.join(REPO, rel))))]
        except (OSError, ValueError):
            return []
    return [lint_tool(c) for c in tools_binding_file(path, root)]


def main():
    ap = argparse.ArgumentParser(description="Drift-check Tool entry_point bindings against live code.")
    ap.add_argument("--tool", help="lint a single Tool id")
    ap.add_argument("--for-file", dest="for_file", help="lint Tools a just-edited path affects")
    ap.add_argument("--check", action="store_true", help="exit 1 if any linted Tool has drift")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    if args.for_file:
        results = lint_for_file(args.for_file, args.root)
    elif args.tool:
        results = [lint_one(args.tool, args.root)]
    else:
        results = lint_all(args.root)

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.for_file:
        # Hook mode: silent unless this edit actually drifted a Tool it touches.
        drifted = [r for r in results if r["drift"]]
        for r in drifted:
            print(f"  tool-drift: '{r['id']}' ({r['verdict']}) — {r['reason']}")
        if drifted:
            print(f"  → repoint the entry_point and bump current_version/current_hash, "
                  f"or run_tool will refuse it. ({_rel(args.for_file)})")
    else:
        drifted = [r for r in results if r["drift"]]
        SYM = {"clean": "ok ", "missing": "MISS", "hash_moved": "HASH",
               "no_entry_point": "NONE", "unknown_tool": "????"}
        for r in results:
            tag = SYM.get(r["verdict"], "??? ")
            if r["drift"] or args.tool:
                print(f"  [{tag}] {r['id']:<32} {r['reason']}")
        print(f"\n{len(results)} Tools linted — {len(results) - len(drifted)} clean, "
              f"{len(drifted)} drifted.")
        if drifted and not args.tool:
            print("Drifted:", ", ".join(r["id"] for r in drifted))

    if args.check and any(r["drift"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
