#!/usr/bin/env python3
"""Read-model indexer for the Research OS web app.

Boundary discipline (PLAN §5, Q1/Q11): this is a DERIVED read-model. The canonical
state is the text + git under ``research_os/`` — this module only *reflects* it.
It globs the JSON object store, validates nothing it didn't author, and builds a
single in-memory snapshot the frontend renders. Delete the webapp → lose nothing
but this cache.

It reuses the canonical ``render/live_head.py`` for the ~30-line head projection
(single source of truth for trunk / current-branch / frontier / trust), then layers
richer projections the dashboard needs: the full goal tree, the run timeline, the
two claim decks with a *recomputed* blast-radius, the pipeline graph (ADR-0005),
the substrate registry with its blast radius, the glossary, and the contracts.

Everything here is pure-read and deterministic given the store (no clock, no RNG),
so the same store always yields the same snapshot — and a content hash over the
store files is a cheap revision id the SSE channel broadcasts on change.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from typing import Any

# research_os/ root (…/research_os)
RESEARCH_OS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER = os.path.join(RESEARCH_OS, "render")
sys.path.insert(0, RENDER)
sys.path.insert(0, os.path.join(RESEARCH_OS, "loop"))

import live_head  # noqa: E402  (canonical head renderer — single source of truth)
import tool_lint  # noqa: E402  (Tool drift-check — same code the run-button gates on)

# ---------------------------------------------------------------------------
# object loading
# ---------------------------------------------------------------------------

# stem -> glob patterns relative to research_os/. Mirrors loop/validate.py.
DIRS = {
    "runs": ["records/*.json"],
    "claims": ["claims/research/*.json", "claims/preferences/*.json"],
    "goals": ["goals/*.json"],
    "pipelines": ["pipelines/*.json"],
    "substrate": ["substrate/*.json"],
    "contracts": ["contracts/*.json"],
    "glossary": ["glossary/*.json"],
    "tool_runs": ["tool_runs/*.json"],  # ADR-0007 bench-run records
    "pipeline_runs": ["pipeline_runs/*.json"],  # ADR-0007 §5.3 DAG run records
    "artifact_instances": ["artifact_instances/*.json"],  # ADR-0007 Slice 1/2 the shelf
}

_ID_RE = re.compile(r"^s(\d+)([a-z]*)")


def _id_sortkey(run_id: str):
    m = _ID_RE.match(run_id or "")
    return (int(m.group(1)), m.group(2)) if m else (0, run_id or "")


def _run_sortkey(rec: dict):
    return (rec.get("created_at", ""), _id_sortkey(rec.get("id", "")))


def load_objects(root: str = RESEARCH_OS) -> dict[str, dict]:
    """Load every canonical object, keyed by id within each stem."""
    out: dict[str, dict] = {}
    for stem, pats in DIRS.items():
        bag: dict[str, dict] = {}
        for pat in pats:
            for f in glob.glob(os.path.join(root, pat)):
                base = os.path.basename(f)
                if base.startswith("_"):  # _index.json etc.
                    continue
                try:
                    d = json.load(open(f))
                except Exception:
                    continue
                if "id" in d:
                    d["_file"] = os.path.relpath(f, os.path.dirname(root))
                    bag[d["id"]] = d
        out[stem] = bag
    return out


def store_revision(root: str = RESEARCH_OS) -> str:
    """A cheap content hash over the store (names + mtimes). Changes iff the store
    changed; the SSE channel broadcasts it so the UI knows to refetch."""
    h = hashlib.sha1()
    files = []
    for pats in DIRS.values():
        for pat in pats:
            files.extend(glob.glob(os.path.join(root, pat)))
    for f in sorted(files):
        try:
            st = os.stat(f)
            h.update(f.encode())
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
        except OSError:
            continue
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# projections
# ---------------------------------------------------------------------------

def _resolve_blast(depends_on: list[str], substrate: dict) -> list[dict]:
    """Given a claim/run's substrate deps, return the refs whose component has since
    bumped past the cited version — the live blast radius (PLAN §3)."""
    stale = []
    for ref in depends_on or []:
        cid, _, ver = ref.partition("@")
        comp = substrate.get(cid)
        if comp and comp.get("current_version") and comp["current_version"] != ver:
            stale.append({
                "ref": ref, "component": cid, "cited": ver,
                "current": comp["current_version"],
            })
    return stale


def load_frontier_ranking(root: str, store_rev: str) -> dict | None:
    """Read the optional ranked-frontier artifact the `/strategize` skill emits
    (req #9). It is a DERIVED render artifact (render/frontier_ranking.json), not a
    canonical store object — the backend only ever reads it. Returns None when no
    ranking has been produced (the frontier is then shown unranked, honestly).

    The artifact carries `generated_for_rev` (the store revision it ranked); we tag
    it `stale` when the store has since moved, so the UI can warn that the ranking
    predates the current state and a re-rank is due."""
    p = os.path.join(root, "render", "frontier_ranking.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("items"), list):
        return None
    # A ranking that forgot to stamp its rev can't be proven fresh — treat as stale
    # so a half-written/legacy artifact never masquerades as current.
    gen_rev = d.get("generated_for_rev")
    d["stale"] = (not gen_rev) or (gen_rev != store_rev)
    return d


def load_machinery_map(root: str) -> dict | None:
    """Read the optional machinery map the `derive_machinery.py` script emits — the
    "visualised pseudocode" of the inversion machinery (stages → load-bearing
    functions → call edges, plus a drift report). Like the frontier ranking it is a
    DERIVED render artifact (render/machinery_map.json), never a canonical store
    object — the backend only READS it. Returns None when it hasn't been derived yet
    (the view then prompts to run the derive).

    Staleness is keyed on the CODE (not the store): the map records a content
    fingerprint per scanned file; we recompute it here and flag the map stale the
    moment the machinery source has changed since it was derived — so a stale map
    can't quietly misrepresent code that has since moved."""
    p = os.path.join(root, "render", "machinery_map.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("nodes"), list):
        return None
    repo_root = os.path.dirname(root)
    fp = d.get("code_fingerprint") or {}
    stale = not fp  # no fingerprint => can't prove freshness
    for rel, sha in fp.items():
        try:
            with open(os.path.join(repo_root, rel), "rb") as fh:
                cur = hashlib.sha1(fh.read()).hexdigest()[:12]
        except OSError:
            cur = None
        if cur != sha:
            stale = True
            break
    d["stale"] = stale
    return d


def build_goal_tree(goals: dict, runs: dict, pipelines: dict, contracts: dict) -> dict:
    """Nest goal nodes into the thesis tree. Each node carries its direct runs,
    serving pipelines, contracts and a rolled-up descendant run count."""
    # runs by goal
    runs_by_goal: dict[str, list] = {}
    for r in runs.values():
        runs_by_goal.setdefault(r.get("goal_node"), []).append(r["id"])
    # pipelines by served goal
    pipes_by_goal: dict[str, list] = {}
    for p in pipelines.values():
        for g in p.get("serves", []):
            pipes_by_goal.setdefault(g, []).append(p["id"])
    # children by parent
    children: dict[str, list] = {}
    for g in goals.values():
        children.setdefault(g.get("parent"), []).append(g["id"])

    def node(gid: str) -> dict:
        g = goals[gid]
        kids = sorted(
            children.get(gid, []),
            key=lambda k: (goals[k].get("node_kind", ""), goals[k].get("title", "")),
        )
        child_nodes = [node(k) for k in kids]
        direct = sorted(runs_by_goal.get(gid, []), key=_id_sortkey)
        total = len(direct) + sum(c["run_count_total"] for c in child_nodes)
        return {
            "id": gid,
            "node_kind": g.get("node_kind"),
            "title": g.get("title", ""),
            "state": g.get("state"),
            "is_trunk_artifact": g.get("is_trunk_artifact", False),
            "budget": g.get("budget"),
            "spent": g.get("spent"),
            "last_measured": g.get("last_measured"),
            "contract_refs": g.get("contract_refs", []),
            "direct_runs": direct,
            "pipelines": sorted(pipes_by_goal.get(gid, [])),
            "children": child_nodes,
            "run_count_direct": len(direct),
            "run_count_total": total,
        }

    roots = children.get(None, [])
    trunk = next((g for g in roots if goals[g].get("node_kind") == "thesis"), None)
    return node(trunk) if trunk else {}


def build_snapshot(root: str = RESEARCH_OS) -> dict[str, Any]:
    """The whole read-model in one object. The frontend fetches this; SSE tells it
    when to refetch."""
    objs = load_objects(root)
    runs, goals, claims = objs["runs"], objs["goals"], objs["claims"]
    pipelines, substrate = objs["pipelines"], objs["substrate"]
    contracts, glossary = objs["contracts"], objs["glossary"]

    # canonical head (reuse the renderer so the dashboard never drifts from /orient)
    head_store = {
        "runs": {k: v for k, v in runs.items()},
        "goals": goals, "claims": claims,
        "substrate": substrate, "pipelines": pipelines,
    }
    try:
        head = live_head.compute_head(head_store, frontier_n=50)
    except Exception as e:  # never let a head glitch take down the whole snapshot
        head = {"error": f"head render failed: {e}"}

    # --- runs: enrich with goal title + sort newest-first ---
    run_rows = []
    for r in sorted(runs.values(), key=_run_sortkey, reverse=True):
        g = goals.get(r.get("goal_node"))
        run_rows.append({
            **{k: v for k, v in r.items() if k != "narrative_md"},
            "narrative_md": r.get("narrative_md", ""),
            "goal_title": g.get("title") if g else None,
            "blast": _resolve_blast(
                [f"{c}@{v}" for c, v in (r.get("substrate_versions") or {}).items()],
                substrate,
            ),
        })

    # --- claims: split by deck, recompute live blast radius ---
    claim_rows = []
    for c in claims.values():
        stale = _resolve_blast(c.get("depends_on", []), substrate)
        claim_rows.append({
            **c,
            "blast": stale,
            "blast_stale": bool(stale),
        })
    claim_rows.sort(key=lambda c: (c.get("deck", ""), c.get("status", ""), c["id"]))

    # --- pipelines: tested_by + resolved serve titles + ADR-0007 §5.3 run feed ---
    tested_by: dict[str, int] = {}
    tested_by_runs: dict[str, list] = {}
    for r in runs.values():
        for pid in r.get("tests", []) or []:
            tested_by[pid] = tested_by.get(pid, 0) + 1
            tested_by_runs.setdefault(pid, []).append(r["id"])
    # group the pipeline_run DAG records by the pipeline they ran (the per-pipeline
    # "SEE the lab" feed, mirroring the per-Tool bench_runs on substrate)
    tr_sort = lambda t: t.get("executed_at") or t.get("created_at") or ""  # noqa: E731
    runs_by_pipeline: dict[str, list] = {}
    for pr in objs["pipeline_runs"].values():
        runs_by_pipeline.setdefault(pr.get("pipeline"), []).append(pr)
    pipe_rows = []
    for p in pipelines.values():
        pruns = sorted(runs_by_pipeline.get(p["id"], []), key=tr_sort, reverse=True)
        pipe_rows.append({
            **p,
            "tested_by": tested_by.get(p["id"], 0),
            "tested_by_runs": sorted(tested_by_runs.get(p["id"], []), key=_id_sortkey),
            "serves_titles": {g: goals[g]["title"] for g in p.get("serves", []) if g in goals},
            "pipeline_runs": pruns,
            "pipeline_run_count": len(pruns),
        })
    pipe_rows.sort(key=lambda p: (p.get("state", ""), p["id"]))

    # --- tool_runs (ADR-0007 bench records): global feed + per-Tool index ---
    tool_runs = objs["tool_runs"]
    tr_by_tool: dict[str, list] = {}
    for tr in tool_runs.values():
        tr_by_tool.setdefault(tr.get("tool"), []).append(tr)
    tool_run_rows = sorted(tool_runs.values(), key=tr_sort, reverse=True)

    # --- pipeline_runs (ADR-0007 §5.3 DAG records): global feed (also fed per-pipeline above) ---
    pipeline_run_rows = sorted(objs["pipeline_runs"].values(), key=tr_sort, reverse=True)

    # --- artifact_instances (ADR-0007 Slice 1/2 — the shelf): each produced material,
    # enriched with its producer's identity and the glossary label for its type, so the
    # materials view reads as "IA Cloud · from pipeline so3-pool-dedup · step sample". ---
    producer_index: dict[str, dict] = {}
    for tr in tool_runs.values():
        producer_index[tr["id"]] = {"kind": "tool_run", "name": tr.get("tool")}
    for pr in objs["pipeline_runs"].values():
        producer_index[pr["id"]] = {"kind": "pipeline_run", "name": pr.get("pipeline")}
    ai_rows = []
    for a in objs["artifact_instances"].values():
        prod = a.get("produced_by") or {}
        pinfo = producer_index.get(prod.get("run"), {})
        term = glossary.get(a.get("artifact_type"))
        ai_rows.append({
            **a,
            "producer_kind": pinfo.get("kind"),
            "producer_name": pinfo.get("name"),
            "artifact_type_term": term.get("term") if term else a.get("artifact_type"),
        })
    ai_rows.sort(key=lambda a: a.get("created_at", ""), reverse=True)

    # --- substrate: blast radius (which runs/claims rest on a non-head version) ---
    # plus the ADR-0007 laboratory view: each Tool's live drift status (same check
    # the run-button gates on) and its recent bench runs — "SEE the lab".
    sub_rows = []
    for cid, s in substrate.items():
        head_ver = s.get("current_version")
        stale_runs = [
            r["id"] for r in runs.values()
            if (r.get("substrate_versions") or {}).get(cid) not in (None, head_ver)
        ]
        stale_claims = [
            c["id"] for c in claims.values()
            if any(d.startswith(cid + "@") and not d.endswith("@" + str(head_ver))
                   for d in c.get("depends_on", []))
        ]
        drift = tool_lint.lint_tool(s)
        bench = sorted(tr_by_tool.get(cid, []), key=tr_sort, reverse=True)
        sub_rows.append({
            **s,
            "blast_runs": sorted(stale_runs, key=_id_sortkey),
            "blast_claims": sorted(stale_claims),
            "drift": {k: drift[k] for k in ("verdict", "drift", "runnable", "reason", "live_hash")},
            "bench_runs": bench[:8],
            "bench_run_count": len(bench),
        })
    sub_rows.sort(key=lambda s: s["id"])

    # --- glossary ---
    gloss_rows = sorted(glossary.values(), key=lambda t: t.get("term", t["id"]).lower())

    # --- trust summary (deck-aware) ---
    status_counts: dict[str, int] = {}
    for c in claims.values():
        status_counts[c.get("status", "?")] = status_counts.get(c.get("status", "?"), 0) + 1
    live_blast = [c["id"] for c in claim_rows if c["blast_stale"]]

    rev = store_revision(root)
    return {
        "rev": rev,
        "head": head,
        "frontier_ranking": load_frontier_ranking(root, rev),
        "machinery": load_machinery_map(root),
        "counts": {k: len(objs[k]) for k in objs},
        "goal_tree": build_goal_tree(goals, runs, pipelines, contracts),
        "goals": list(goals.values()),
        "runs": run_rows,
        "claims": claim_rows,
        "pipelines": pipe_rows,
        "substrate": sub_rows,
        "tool_runs": tool_run_rows,
        "pipeline_runs": pipeline_run_rows,
        "artifact_instances": ai_rows,
        "glossary": gloss_rows,
        "contracts": sorted(contracts.values(), key=lambda c: c["id"]),
        "trust": {
            "status_counts": status_counts,
            "live_blast_claims": live_blast,
            "substrate_heads": {cid: s.get("current_version") for cid, s in substrate.items()},
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_snapshot(), indent=2)[:4000])
