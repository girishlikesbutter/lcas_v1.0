#!/usr/bin/env python3
"""Derive the machinery map — the "visualised pseudocode" of the inversion machinery.

This is the *auto-derive* half of an auto-derive-then-curate instrument (the curate
half is the hand-authored ``machinery_overlay.json``). It reads that overlay, then
walks the REAL code with the ``ast`` module to ground every curated claim in the
source: exact signature, docstring, line number, and the intra-graph call edges.
It then cross-checks the two and emits a ``drift`` report — overlay units that have
vanished from the code (``missing``) and load-bearing functions in scope that no
overlay unit covers (``uncovered``).

Boundary (Q1/Q11): this writes ONLY ``render/machinery_map.json`` — a DERIVED render
artifact alongside the live-head and the frontier ranking, never the canonical store.
The web backend only ever READS it (indexer.load_machinery_map). It is the code
analogue of ``/strategize`` writing ``frontier_ranking.json``: run it from a terminal,
the dashboard refetches.

Staleness is keyed on the CODE, not the store: the map records a content fingerprint
of every scanned file; the indexer recomputes it and flags the map stale the moment
the machinery source changes (so a stale map can never quietly mislead).

    python research_os/render/derive_machinery.py        # derive + write
    python research_os/render/derive_machinery.py --check # print drift, exit 1 if any

Schema: research_os/schemas/machinery_map.schema.json
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import sys

RENDER = os.path.dirname(os.path.abspath(__file__))
RESEARCH_OS = os.path.dirname(RENDER)
REPO_ROOT = os.path.dirname(RESEARCH_OS)
OVERLAY = os.path.join(RENDER, "machinery_overlay.json")
OUT = os.path.join(RENDER, "machinery_map.json")


def _file_sha(abspath: str) -> str | None:
    try:
        with open(abspath, "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:12]
    except OSError:
        return None


def _collect_defs(tree: ast.AST) -> tuple[dict[str, ast.AST], set[str]]:
    """Map every function/method to its node, keyed by BOTH its bare name and its
    qualified ``Class.method`` name, so an overlay unit can reference either.

    Also returns the set of bare names defined at MODULE or CLASS-METHOD level (not
    nested inside another function) — the only ones the drift scan should treat as
    'uncovered' candidates (a nested ODE-rhs helper is not a public surface)."""
    out: dict[str, ast.AST] = {}
    toplevel: set[str] = set()

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []
            self.func_depth = 0  # how many FunctionDefs we're nested inside

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _add(self, node):
            qual = ".".join(self.stack + [node.name])
            out.setdefault(node.name, node)   # bare name (first wins)
            out[qual] = node                  # qualified always
            if self.func_depth == 0:          # module-level or class method
                toplevel.add(node.name)
            saved, self.stack = self.stack, self.stack + [node.name]
            self.func_depth += 1
            self.generic_visit(node)
            self.func_depth -= 1
            self.stack = saved

        def visit_FunctionDef(self, node):
            self._add(node)

        def visit_AsyncFunctionDef(self, node):
            self._add(node)

    V().visit(tree)
    return out, toplevel


def _signature(node: ast.AST, src_lines: list[str]) -> str:
    """Reconstruct the def header from the AST itself — faithful to the real
    signature and immune to multi-line headers and stray ``):`` inside a string or
    annotation default (which a source-slice heuristic mis-terminates on)."""
    try:
        sig = f"{node.name}({ast.unparse(node.args)})"
        if getattr(node, "returns", None) is not None:
            sig += f" -> {ast.unparse(node.returns)}"
        return " ".join(sig.split())
    except Exception:
        # fallback: the first source line, trimmed of def/colon
        i = node.lineno - 1
        line = src_lines[i].strip() if 0 <= i < len(src_lines) else node.name
        return line.split("def ", 1)[-1].rstrip(":")


def _called_names(node: ast.AST) -> set[str]:
    """Names this function calls — ``foo(...)`` (Name) and ``x.foo(...)`` (Attribute).
    Best-effort, for the data-flow edges."""
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def derive() -> dict:
    if not os.path.exists(OVERLAY):
        raise SystemExit(f"no overlay at {OVERLAY} — author machinery_overlay.json first")
    overlay = json.load(open(OVERLAY))
    stages = overlay.get("stages", [])
    units = overlay.get("units", [])
    scope_files = overlay.get("scope_files", [])

    # parse every file referenced by a unit OR listed in scope, once
    referenced = sorted({u["file"] for u in units} | set(scope_files))
    parsed: dict[str, dict] = {}   # relpath -> {defs, src_lines, sha}
    fingerprint: dict[str, str | None] = {}   # None = referenced path absent (staleness sentinel)
    for rel in referenced:
        ab = os.path.join(REPO_ROOT, rel)
        sha = _file_sha(ab)
        # Record EVERY referenced path — including a None sentinel for one that does
        # not exist yet — so the indexer's symmetric recompute (missing → None) flips
        # stale the moment a scoped file is later created (no staleness blind spot).
        fingerprint[rel] = sha
        if sha is None:
            parsed[rel] = {"defs": {}, "toplevel": set(), "src_lines": [], "sha": None, "ok": False}
            continue
        try:
            src = open(ab, encoding="utf-8").read()
            tree = ast.parse(src, filename=rel)
            defs, toplevel = _collect_defs(tree)
            parsed[rel] = {"defs": defs, "toplevel": toplevel,
                           "src_lines": src.splitlines(), "sha": sha, "ok": True}
        except (OSError, SyntaxError) as e:
            parsed[rel] = {"defs": {}, "toplevel": set(), "src_lines": [],
                           "sha": sha, "ok": False, "err": str(e)}

    # resolve each unit against the AST
    node_names = {u["name"] for u in units}
    # (file, name) pairs a unit ACTUALLY covers — for the file-aware uncovered scan, so a
    # scope-file def is only 'covered' by a unit in THAT file (not a same-named one elsewhere).
    covered_pairs = {(u["file"], u["name"]) for u in units}
    nodes = []
    missing = []
    for u in units:
        p = parsed.get(u["file"], {"defs": {}, "src_lines": []})
        d = p["defs"].get(u["name"])
        if d is None:
            missing.append({"name": u["name"], "file": u["file"]})
            nodes.append({
                "name": u["name"], "file": u["file"], "stage": u.get("stage", ""),
                "role": u.get("role", ""), "substrate_component": u.get("substrate_component"),
                "exists": False, "signature": None, "doc": None, "line": None, "calls": [],
            })
            continue
        calls = _called_names(d)
        nodes.append({
            "name": u["name"], "file": u["file"], "stage": u.get("stage", ""),
            "role": u.get("role", ""), "substrate_component": u.get("substrate_component"),
            "exists": True,
            "signature": _signature(d, p["src_lines"]),
            "doc": (ast.get_docstring(d) or "").strip().split("\n")[0] or None,
            "line": d.lineno,
            "calls": sorted(c for c in calls if c in node_names and c != u["name"]),
        })

    # edges between nodes (both endpoints are curated nodes)
    edges = []
    seen_edge = set()
    for n in nodes:
        for c in n["calls"]:
            key = (n["name"], c)
            if key not in seen_edge:
                seen_edge.add(key)
                edges.append({"from": n["name"], "to": c})

    # uncovered: module-level / class-method public defs in scope_files no unit covers
    # (nested helpers like an ODE-rhs `dynamics` inside propagate_euler are excluded)
    uncovered = []
    for rel in scope_files:
        p = parsed.get(rel, {"defs": {}, "toplevel": set()})
        for name in p.get("toplevel", set()):
            if name.startswith("_"):   # skip private helpers
                continue
            if (rel, name) in covered_pairs:   # covered only by a unit IN THIS FILE
                continue
            d = p["defs"].get(name)
            uncovered.append({"name": name, "file": rel, "line": getattr(d, "lineno", None)})
    uncovered.sort(key=lambda x: (x["file"], x["line"] or 0))

    code_fp = hashlib.sha1(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()[:12]

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "generated_for_code": code_fp,
        "code_fingerprint": fingerprint,
        "stages": stages,
        "nodes": nodes,
        "edges": edges,
        "drift": {"missing": missing, "uncovered": uncovered},
    }


def main(argv: list[str]) -> int:
    m = derive()
    drift = m["drift"]
    if "--check" in argv:
        print(f"missing={len(drift['missing'])} uncovered={len(drift['uncovered'])}")
        for x in drift["missing"]:
            print(f"  MISSING  {x['file']}::{x['name']}  (overlay unit not found in code)")
        for x in drift["uncovered"]:
            print(f"  UNCOVERED {x['file']}::{x['name']}:{x['line']}")
        return 1 if (drift["missing"] or drift["uncovered"]) else 0
    json.dump(m, open(OUT, "w"), indent=2)
    rel = os.path.relpath(OUT, REPO_ROOT)
    print(f"wrote {rel}")
    print(f"  stages={len(m['stages'])} nodes={len(m['nodes'])} edges={len(m['edges'])} "
          f"missing={len(drift['missing'])} uncovered={len(drift['uncovered'])} "
          f"code_fp={m['generated_for_code']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
