#!/usr/bin/env python3
"""W-D Tool/Pipeline scour — deterministic card generator (ADR-0007).

Reads backfill/W-D_inventory.json (the multi-agent inventory+classification) and
emits the laboratory registry: glossary terms (port type-system), substrate_component
Tool cards (canon + variant_of families), and pipeline cards. Idempotent: new ids are
skipped if their file already exists; the 3 pre-existing Tool cards (propagator/surrogate/
scorer) and the 4 pre-existing pipelines are UPDATED in place (versions[]/functions[]/
created_at preserved). Bakes the drift-check (step 4): every entry_point is resolved in
the code and its file is hashed.

Human decisions applied (this session):
  - Scope = EVERYTHING: src/inversion carded as in_use:false (parallel architecture).
  - rho COLLAPSED: one rho Tool (rho-band-classify) with a `space` param; scorer-rho dropped.
  - Fork 5: ls-bracket PROMOTED to its own canon Tool (split from lc-feature-extractor).
  - Forks 4/6: multianchor-shoot + geo-filter kept as variant_of, in_use:false.

Run:  python research_os/backfill/W-D_generate_cards.py
Then: python research_os/loop/validate.py
"""
from __future__ import annotations
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)  # repo root (research_os/ lives one level under it)
NOW = "2026-06-07T00:00:00Z"
TODAY = "2026-06-07"
COINED_IN = "research_os/backfill/W-D_tool_pipeline_scour.md"

inv = json.load(open(os.path.join(ROOT, "backfill", "W-D_inventory.json")))
synth = inv["synth"]

# substrate_component id pattern is underscore-separated (schema), unlike the
# hyphen-separated glossary/pipeline ids. Normalise all Tool ids to underscore_case.
def norm(s):
    return s.replace("-", "_")

# ---- helpers ---------------------------------------------------------------
def existing_glossary_ids():
    ids = set()
    for f in os.listdir(os.path.join(ROOT, "glossary")):
        if f.endswith(".json") and not f.startswith("_"):
            ids.add(f[:-5])
    return ids

def resolve_entry_point(ep):
    """Return (resolved: bool, hash: str|None, path: str) for 'path.py:Symbol[.method]'."""
    if not ep or ":" not in ep:
        return False, None, ep or ""
    path, _, symbol = ep.partition(":")
    abspath = os.path.join(REPO, path)
    if not os.path.isfile(abspath):
        return False, None, path
    src = open(abspath, "rb").read()
    h = "sha256:" + hashlib.sha256(src).hexdigest()[:16]
    text = src.decode("utf-8", "replace")
    leaf = symbol.split(".")[-1]
    pats = [rf"\bdef\s+{re.escape(leaf)}\b", rf"\bclass\s+{re.escape(leaf)}\b",
            rf"(?m)^\s*{re.escape(leaf)}\s*="]
    resolved = any(re.search(p, text) for p in pats)
    return resolved, h, path

def write_json(subdir, _id, obj, update=False):
    fp = os.path.join(ROOT, subdir, f"{_id}.json")
    if os.path.exists(fp) and not update:
        return "skip"
    json.dump(obj, open(fp, "w"), indent=2)
    open(fp, "a").write("\n")
    return "update" if (update and os.path.exists(fp)) else "write"

report = {"glossary": [], "tools": [], "pipelines": [], "unresolved": [], "orphan_ports": []}

# ---- 1. glossary terms (port type-system) ----------------------------------
GLOSSARY_RELATED = {
    "light-curve": ["hifi-lightcurve", "rho-band"],
    "hifi-lightcurve": ["light-curve", "rho-band"],
    "q0-omega-state": ["attitude-trajectory"],
    "attitude-trajectory": ["q0-omega-state", "epoch-observation"],
    "epoch-observation": ["attitude-trajectory", "satellite-model"],
    "candidate-set": ["q0-omega-state", "ia-cloud-cross"],
    "anchor-epoch": ["ia-cloud", "single-wind-lc-window"],
    "ck-kernel": ["spk-kernel", "sclk-kernel", "spice-setup-file"],
    "sclk-kernel": ["ck-kernel", "spk-kernel"],
    "spk-kernel": ["ck-kernel", "spice-setup-file"],
    "lc-feature-vector": ["light-curve", "multi-omegadir-start"],
}
valid_terms = existing_glossary_ids()
for g in synth["glossary_gaps"]:
    tid = g["term_id"]
    valid_terms.add(tid)
    obj = {
        "schema_version": "1.0.0", "id": tid, "kind": "glossary_term",
        "term": tid, "definition": g["definition"], "status": "provisional",
        "coined_in": COINED_IN, "synonyms_blocked": [], "aliases": [],
        "related_terms": [t for t in GLOSSARY_RELATED.get(tid, []) if t],
        "created_at": NOW,
    }
    r = write_json("glossary", tid, obj)
    report["glossary"].append((tid, r))

# ---- 2. build the canon + variant Tool spec list ---------------------------
# Apply the two human overrides to the inventory clusters.
canon_specs = {}   # id -> spec dict
variant_specs = {} # id -> (spec, canon_id)

for c in synth["clusters"]:
    cn = c["canon"]
    cid = norm(cn["id"])
    canon_specs[cid] = {
        "id": cid, "name": cn.get("name", cid),
        "path": cn.get("path") or (cn.get("entry_point", "").partition(":")[0]),
        "entry_point": cn.get("entry_point"), "interface": cn.get("interface", ""),
        "ports_in": cn.get("ports_in", []), "ports_out": cn.get("ports_out", []),
        "default_params": cn.get("default_params") or {}, "in_use": bool(c.get("in_use")),
    }
    for v in c.get("variants", []):
        variant_specs[norm(v["id"])] = (v, cid)

# Override A: drop scorer_rho (collapsed into rho_band_classify with a space param).
variant_specs.pop("scorer_rho", None)
# rho_band_classify becomes the single rho Tool: stamp the collapse into default_params.
if "rho_band_classify" in canon_specs:
    rb = canon_specs["rho_band_classify"]
    rb["default_params"] = {"space": "hifi", "_collapsed": "surrogate_eval.rho (space='surrogate') + hifi_render.rho_from_hifi (space='hifi') + rho_band classifier — one rho Tool, trust-level by `space` param (decision 2026-06-07)"}
    rb["ports_in"] = ["light-curve"]
    rb["ports_out"] = ["rho-band"]

# Override B: promote ls_bracket from variant -> its own canon Tool (Fork 5 split).
if "ls_bracket" in variant_specs:
    v, _ = variant_specs.pop("ls_bracket")
    canon_specs["ls_bracket"] = {
        "id": "ls_bracket", "name": "ls_bracket",
        "path": v["entry_point"].partition(":")[0], "entry_point": v["entry_point"],
        "interface": "Lomb-Scargle blind |omega| bracket: emits a geomspaced |omega| grid (the multi-omegadir-start prior) from LC periodogram peaks. The live blind |omega| prior block (s082+/s100).",
        "ports_in": ["light-curve"], "ports_out": ["multi-omegadir-start"],
        "default_params": {}, "in_use": True,
    }

# ---- 3. emit Tool cards ----------------------------------------------------
EXISTING_TOOLS = {"propagator", "surrogate", "scorer"}

def base_tool(spec, canon, variant_of):
    resolved, h, path = resolve_entry_point(spec.get("entry_point"))
    if not resolved:
        report["unresolved"].append((spec["id"], spec.get("entry_point")))
    ports_in = spec.get("ports_in") or []
    ports_out = spec.get("ports_out") or []
    for p in ports_in + ports_out:
        if p not in valid_terms:
            report["orphan_ports"].append((spec["id"], p))
    ports = None
    if ports_in or ports_out:
        ports = {"input": ports_in, "output": ports_out}
    dp = spec.get("default_params") or None
    obj = {
        "schema_version": "1.0.0", "id": spec["id"], "kind": "substrate_component",
        "name": spec.get("name", spec["id"]), "path": path or spec.get("path", ""),
        "tier": "substrate",
        "interface": spec.get("interface", "") + (f"  [in_use={spec.get('in_use')}]"),
        "current_version": "1.0.0",
        "current_hash": h or "",
        "canon": canon,
        "variant_of": variant_of,
        "promoted_from": None,
        "entry_point": spec.get("entry_point"),
        "ports": ports,
        "default_params": dp,
        "versions": [{
            "version": "1.0.0", "hash": h or "", "changed_on": TODAY,
            "change_reason": "Registered by the W-D laboratory scour (ADR-0007): de-facto Tool inventoried from code; entry_point resolved + hashed." + ("" if resolved else " [entry_point UNRESOLVED at scour time — needs reconciliation]"),
            "is_bug_fix": False,
        }],
        "created_at": NOW, "updated_at": NOW,
    }
    return obj

# canon (new ones; the 3 existing handled separately below)
for cid, spec in canon_specs.items():
    if cid in EXISTING_TOOLS:
        continue
    obj = base_tool(spec, canon=True, variant_of=None)
    report["tools"].append((cid, write_json("substrate", cid, obj), "canon", spec["in_use"]))

# variants
for vid, (v, canon_id) in variant_specs.items():
    canon = canon_specs.get(canon_id, {})
    spec = {
        "id": vid, "name": vid.replace("-", "_"),
        "path": v["entry_point"].partition(":")[0], "entry_point": v["entry_point"],
        "interface": "VARIANT: " + v.get("why", ""),
        "ports_in": canon.get("ports_in", []), "ports_out": canon.get("ports_out", []),
        "default_params": {}, "in_use": canon.get("in_use", False),
    }
    obj = base_tool(spec, canon=False, variant_of=canon_id)
    report["tools"].append((vid, write_json("substrate", vid, obj), f"variant_of {canon_id}", spec["in_use"]))

# ---- 3b. UPDATE the 3 existing Tool cards (preserve versions/functions/created_at) ----
def update_existing_tool(cid, new_path=None):
    fp = os.path.join(ROOT, "substrate", f"{cid}.json")
    card = json.load(open(fp))
    spec = canon_specs[cid]
    resolved, h, _ = resolve_entry_point(spec["entry_point"])
    if not resolved:
        report["unresolved"].append((cid, spec["entry_point"]))
    if new_path:  # scorer: repoint to the real impl file + rehash
        card["path"] = new_path
        card["current_hash"] = h or card.get("current_hash", "")
    ports_in = spec.get("ports_in") or []
    ports_out = spec.get("ports_out") or []
    for p in ports_in + ports_out:
        if p not in valid_terms:
            report["orphan_ports"].append((cid, p))
    card["entry_point"] = spec["entry_point"]
    card["ports"] = {"input": ports_in, "output": ports_out} if (ports_in or ports_out) else None
    card["default_params"] = spec.get("default_params") or None
    if "[in_use=" not in card.get("interface", ""):
        card["interface"] = card.get("interface", "") + f"  [in_use={spec.get('in_use')}]"
    card["canon"] = True
    card.setdefault("variant_of", None)
    card.setdefault("promoted_from", None)
    card["updated_at"] = NOW
    json.dump(card, open(fp, "w"), indent=2)
    open(fp, "a").write("\n")
    report["tools"].append((cid, "update-existing", "canon", spec["in_use"]))

update_existing_tool("propagator")
update_existing_tool("surrogate")
update_existing_tool("scorer", new_path="notebooks/inversion/survey/lib/surrogate_eval.py")

# ---- 4. pipelines ----------------------------------------------------------
# New pipeline: the blind 5-step chain (s100 era). Update the 4 existing composes[].
PIPE_UPDATES = {p["id"]: p for p in synth["pipelines"]}

# 4a. new blind-5step
new_p = PIPE_UPDATES.get("pipeline_blind-5step-invert")
if new_p:
    obj = {
        "schema_version": "1.0.0", "id": "pipeline_blind-5step-invert", "kind": "pipeline",
        "title": "Blind 5-step inversion (IA-Cloud anchors -> cross -> window-polish)",
        "hypothesis": new_p.get("hypothesis", ""),
        "state": "open",
        "serves": ["goal_blind-pipeline"],
        "composes": [norm(c) for c in new_p.get("composes", []) if os.path.exists(os.path.join(ROOT, "substrate", f"{norm(c)}.json"))],
        "supersedes": None, "superseded_by": None, "last_measured": None,
        "created_at": NOW,
    }
    report["pipelines"].append(("pipeline_blind-5step-invert", write_json("pipelines", "pipeline_blind-5step-invert", obj)))

# 4b. recompose the 4 existing
for pid in ("pipeline_densify", "pipeline_joint-grid-pivot", "pipeline_cross-cloud", "pipeline_single-wind-lc-window-polish"):
    fp = os.path.join(ROOT, "pipelines", f"{pid}.json")
    if not os.path.exists(fp) or pid not in PIPE_UPDATES:
        continue
    card = json.load(open(fp))
    composes = [norm(c) for c in PIPE_UPDATES[pid].get("composes", []) if os.path.exists(os.path.join(ROOT, "substrate", f"{norm(c)}.json"))]
    if composes:
        card["composes"] = composes
        card["updated_at"] = NOW
        json.dump(card, open(fp, "w"), indent=2)
        open(fp, "a").write("\n")
        report["pipelines"].append((pid, "recompose"))

# ---- report ----------------------------------------------------------------
def tally(items, idx):
    from collections import Counter
    return dict(Counter(x[idx] for x in items))

print("=== GLOSSARY ===", tally(report["glossary"], 1), f"({len(report['glossary'])} terms)")
nw = sum(1 for t in report["tools"] if t[1] == "write")
up = sum(1 for t in report["tools"] if t[1] in ("update", "update-existing"))
sk = sum(1 for t in report["tools"] if t[1] == "skip")
live = sum(1 for t in report["tools"] if t[3])
print(f"=== TOOLS === wrote={nw} updated={up} skipped={sk} | in_use=True:{live}/{len(report['tools'])}")
print("=== PIPELINES ===", [(p[0].replace('pipeline_',''), p[1]) for p in report["pipelines"]])
if report["orphan_ports"]:
    print("!!! ORPHAN PORTS (no glossary term):", report["orphan_ports"])
else:
    print("=== PORTS === all resolve to glossary terms")
if report["unresolved"]:
    print(f"!!! UNRESOLVED entry_points ({len(report['unresolved'])}) — drift-check could not find the symbol in code:")
    for cid, ep in report["unresolved"]:
        print("    -", cid, "->", ep)
else:
    print("=== DRIFT-CHECK === all entry_points resolved in code")
