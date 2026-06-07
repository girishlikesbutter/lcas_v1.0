#!/usr/bin/env python3
"""Validate the trust store — schema conformance + referential integrity.

The side-effect-FREE gate (PLAN §3). Unlike `backfill/reconcile.py` (a one-time
migration tool that also rewrites goal nodes with a frozen STAMP and regenerates the
backfill README), this script ONLY reads and checks. `close` runs it after authoring
new objects to confirm the store stayed referentially closed; the Phase-4 hooks
(`gate-check`, `blast-radius`) import its checks. Exits non-zero on any break.

Usage:
    python research_os/loop/validate.py            # validate, exit 0/1
    python research_os/loop/validate.py --quiet     # only print on failure
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

try:
    from jsonschema import Draft202012Validator
except ImportError:  # schema check is best-effort; ref-integrity still runs
    Draft202012Validator = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/

# stem -> glob patterns (relative to ROOT). Mirrors the on-disk layout.
DIRS = {
    "run_record": ["records/*.json"],
    "claim_card": ["claims/research/*.json", "claims/preferences/*.json"],
    "goal_node": ["goals/*.json"],
    "pipeline": ["pipelines/*.json"],
    "substrate_component": ["substrate/*.json"],
    "branch_contract": ["contracts/*.json"],
    "glossary_term": ["glossary/*.json"],
    "tool_run": ["tool_runs/*.json"],  # ADR-0007 bench-run records (canonical)
    "pipeline_run": ["pipeline_runs/*.json"],  # ADR-0007 §5.3 DAG run records (canonical)
    "artifact_instance": ["artifact_instances/*.json"],  # ADR-0007 produced materials (canonical)
}
SKIP_BASENAMES = ("_",)  # _index.json etc.


def _load(root, stem):
    out = {}
    for pat in DIRS[stem]:
        for f in glob.glob(os.path.join(root, pat)):
            if os.path.basename(f).startswith(SKIP_BASENAMES):
                continue
            out[f] = json.load(open(f))
    return out


def validate(root: str = ROOT) -> list[str]:
    """Return a list of error strings; empty list == store is closed."""
    errors: list[str] = []
    objs = {s: _load(root, s) for s in DIRS}

    # (a) schema validation (skipped gracefully if jsonschema absent)
    if Draft202012Validator is not None:
        schemas_dir = os.path.join(root, "schemas")
        for stem, files in objs.items():
            schema = json.load(open(os.path.join(schemas_dir, f"{stem}.schema.json")))
            V = Draft202012Validator(schema)
            for f, inst in files.items():
                for e in V.iter_errors(inst):
                    loc = "/".join(str(p) for p in e.path) or "<root>"
                    errors.append(f"SCHEMA {os.path.relpath(f, root)} [{stem}/{loc}]: {e.message}")

    # index by id
    rec = {d["id"]: d for d in objs["run_record"].values()}
    claim = {d["id"]: d for d in objs["claim_card"].values()}
    goal = {d["id"]: d for d in objs["goal_node"].values()}
    pipe = {d["id"]: d for d in objs["pipeline"].values()}
    comp = {d["id"]: d for d in objs["substrate_component"].values()}
    comp_versions = {cid: {v["version"] for v in d["versions"]} for cid, d in comp.items()}
    contract = {d["id"]: d for d in objs["branch_contract"].values()}
    term = {d["id"]: d for d in objs["glossary_term"].values()}
    trun = {d["id"]: d for d in objs["tool_run"].values()}
    prun = {d["id"]: d for d in objs["pipeline_run"].values()}
    ainst = {d["id"]: d for d in objs["artifact_instance"].values()}
    rec_ids, goal_ids, claim_ids = set(rec), set(goal), set(claim)
    contract_ids = set(contract)
    pipe_ids, term_ids = set(pipe), set(term)

    # (e) referential integrity (schemas/README.md)
    for rid, d in rec.items():
        if d.get("goal_node") not in goal_ids:
            errors.append(f"REF run {rid}.goal_node -> missing goal {d.get('goal_node')}")
        for p in d.get("parents", []):
            if p not in rec_ids:
                errors.append(f"REF run {rid}.parents -> missing run {p}")
        cr = d.get("contract_ref")
        if cr and cr.get("contract") not in contract_ids:
            errors.append(f"REF run {rid}.contract_ref -> missing contract {cr.get('contract')}")
        for cid, ver in d.get("substrate_versions", {}).items():
            if cid not in comp:
                errors.append(f"REF run {rid}.substrate_versions -> missing component {cid}")
            elif ver not in comp_versions[cid]:
                errors.append(f"REF run {rid}.substrate_versions -> {cid}@{ver} not in versions")
        for p in d.get("tests", []):  # Experiment -> tests -> Pipeline (ADR-0005)
            if p not in pipe_ids:
                errors.append(f"REF run {rid}.tests -> missing pipeline {p}")
        for b in d.get("blocked_by", []):  # Experiment -> blocked-by -> Blocker (glossary term)
            if b not in term_ids:
                errors.append(f"REF run {rid}.blocked_by -> missing glossary_term {b}")

    for cid, d in claim.items():
        for r in d.get("supporting_runs", []) + d.get("refuting_runs", []):
            if r not in rec_ids:
                errors.append(f"REF claim {cid} run-ref -> missing run {r}")
        for edge in ("superseded_by", "supersedes"):
            t = d.get(edge)
            if t is not None and t not in claim_ids:
                errors.append(f"REF claim {cid}.{edge} -> missing claim {t}")
        for ref in d.get("depends_on", []):
            lhs, _, ver = ref.partition("@")
            if lhs not in comp:
                errors.append(f"REF claim {cid}.depends_on -> missing component {lhs}")
            elif ver not in comp_versions[lhs]:
                errors.append(f"REF claim {cid}.depends_on -> {ref} not in versions")
        if d.get("status") == "superseded" and not d.get("superseded_by"):
            errors.append(f"RULE claim {cid} status=superseded but superseded_by unset")

    for gid, d in goal.items():
        par = d.get("parent")
        if par is None:
            if d.get("node_kind") != "thesis":
                errors.append(f"RULE goal {gid} parent=null but node_kind={d.get('node_kind')}")
        elif par not in goal_ids:
            errors.append(f"REF goal {gid}.parent -> missing goal {par}")
        for r in d.get("child_runs", []):
            if r not in rec_ids:
                errors.append(f"REF goal {gid}.child_runs -> missing run {r}")
        for c in d.get("contract_refs", []):
            if c not in contract_ids:
                errors.append(f"REF goal {gid}.contract_refs -> missing contract {c}")
        lm = d.get("last_measured")
        if lm and lm.get("run") and lm["run"] not in rec_ids:
            errors.append(f"REF goal {gid}.last_measured.run -> missing run {lm['run']}")

    for pid, d in pipe.items():  # Pipeline edges (ADR-0005)
        for g in d.get("serves", []):  # Pipeline -> serves -> Goal
            if g not in goal_ids:
                errors.append(f"REF pipeline {pid}.serves -> missing goal {g}")
        for c in d.get("composes", []):  # Pipeline -> composes -> Tool
            if c not in comp:
                errors.append(f"REF pipeline {pid}.composes -> missing component {c}")
        step_ids = set()
        for st in d.get("steps", []):  # Pipeline steps -> run -> Tool (ADR-0007 §5.3)
            if st["tool"] not in comp:
                errors.append(f"REF pipeline {pid}.steps[{st['id']}].tool -> missing component {st['tool']}")
            if st["id"] in step_ids:
                errors.append(f"RULE pipeline {pid}.steps -> duplicate step id '{st['id']}'")
            step_ids.add(st["id"])
        if d.get("steps") and set(c for c in d.get("composes", [])) != {st["tool"] for st in d["steps"]}:
            errors.append(f"RULE pipeline {pid}.composes must equal the set of step tools when steps[] is present")
        for edge in ("supersedes", "superseded_by"):  # Pipeline -> supersedes -> Pipeline
            t = d.get(edge)
            if t is not None and t not in pipe_ids:
                errors.append(f"REF pipeline {pid}.{edge} -> missing pipeline {t}")
        if d.get("state") == "superseded" and not d.get("superseded_by"):
            errors.append(f"RULE pipeline {pid} state=superseded but superseded_by unset")
        lm = d.get("last_measured")
        if lm and lm.get("run") and lm["run"] not in rec_ids:
            errors.append(f"REF pipeline {pid}.last_measured.run -> missing run {lm['run']}")

    for cmid, d in comp.items():  # Tool -> promoted-from -> Experiment (ADR-0005)
        pf = d.get("promoted_from")
        if pf is not None and pf not in rec_ids:
            errors.append(f"REF component {cmid}.promoted_from -> missing run {pf}")
        vo = d.get("variant_of")  # Tool -> variant-of -> Tool (ADR-0007)
        if vo is not None and vo not in comp:
            errors.append(f"REF component {cmid}.variant_of -> missing component {vo}")
        ports = d.get("ports") or {}  # Tool ports typed in artifact types == glossary terms (ADR-0007)
        for side in ("input", "output"):  # each side is a flat list OR a keyed map (transition, ADR-0007)
            p = ports.get(side)
            for port in (list(p.values()) if isinstance(p, dict) else list(p or [])):
                if port not in term_ids:
                    errors.append(f"REF component {cmid}.ports.{side} -> missing glossary_term {port}")

    for cid, d in contract.items():
        if d.get("goal_node") not in goal_ids:
            errors.append(f"REF contract {cid}.goal_node -> missing goal {d.get('goal_node')}")

    for trid, d in trun.items():  # tool_run -> ran -> Tool (ADR-0007)
        tid = d.get("tool")
        if tid not in comp:
            errors.append(f"REF tool_run {trid}.tool -> missing component {tid}")
        elif d.get("tool_version") not in comp_versions[tid]:
            errors.append(f"REF tool_run {trid}.tool -> {tid}@{d.get('tool_version')} not in versions")

    run_provenance_ids = set(trun) | set(prun)
    for prid, d in prun.items():  # pipeline_run -> ran -> Pipeline + each step -> Tool (ADR-0007 §5.3)
        if d.get("pipeline") not in pipe_ids:
            errors.append(f"REF pipeline_run {prid}.pipeline -> missing pipeline {d.get('pipeline')}")
        for st in d.get("steps", []):
            if st["tool"] not in comp:
                errors.append(f"REF pipeline_run {prid}.steps[{st['step']}].tool -> missing component {st['tool']}")
            for aid in st.get("artifacts_produced", []):  # step -> produced -> artifact_instance
                if aid not in ainst:
                    errors.append(f"REF pipeline_run {prid}.steps[{st['step']}].artifacts_produced -> missing artifact_instance {aid}")

    for trid, d in trun.items():  # tool_run -> produced -> artifact_instance (ADR-0007)
        for aid in d.get("artifacts_produced", []):
            if aid not in ainst:
                errors.append(f"REF tool_run {trid}.artifacts_produced -> missing artifact_instance {aid}")

    for aid, d in ainst.items():  # artifact_instance -> typed-by -> glossary_term + produced-by -> run (ADR-0007)
        at = d.get("artifact_type")
        if at not in term_ids:
            errors.append(f"REF artifact_instance {aid}.artifact_type -> missing glossary_term {at}")
        src = (d.get("produced_by") or {}).get("run")
        if src not in run_provenance_ids:
            errors.append(f"REF artifact_instance {aid}.produced_by.run -> missing tool_run/pipeline_run {src}")

    thesis = [gid for gid, d in goal.items() if d.get("node_kind") == "thesis" and d.get("parent") is None]
    if len(thesis) != 1:
        errors.append(f"RULE expected exactly 1 thesis root, found {len(thesis)}: {thesis}")

    return errors


def main():
    ap = argparse.ArgumentParser(description="Validate the Research OS trust store.")
    ap.add_argument("--quiet", action="store_true", help="print only on failure")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    errs = validate(args.root)
    if errs:
        print(f"INTEGRITY ERRORS: {len(errs)}")
        for e in errs[:60]:
            print("  -", e)
        sys.exit(1)
    if not args.quiet:
        n = sum(len(_load(args.root, s)) for s in DIRS)
        print(f"ALL CHECKS PASS — store is referentially closed ({n} objects).")


if __name__ == "__main__":
    main()
