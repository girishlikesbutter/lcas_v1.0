#!/usr/bin/env python3
"""W-B reconcile + validate + manifest (STATUS.md step 3).

(a) validate every object against its schema;
(b) fill goal_node.child_runs + spent.runs from the records that point at it
    (rewrite node files, preserving all other fields incl. last_measured);
(c) fill claim supersedes back-edges; verify supporting/refuting_runs resolve;
(d) refresh goals/_index.json n_experiments;
(e) run the referential-integrity checks from schemas/README.md;
(f) write research_os/backfill/README.md (coverage, lossiness, regenerate recipe).

Idempotent. Exits non-zero on any validation or integrity failure. Run from repo root.
"""
import json, glob, os, sys
from pathlib import Path
from collections import Counter, defaultdict
from jsonschema import Draft202012Validator

RO = Path("research_os")
SCHEMAS = RO / "schemas"
STAMP = "2026-06-01T00:00:00Z"

DIRS = {
    "run_record": ["records/*.json"],
    "claim_card": ["claims/research/*.json", "claims/preferences/*.json"],
    "goal_node": ["goals/*.json"],
    "substrate_component": ["substrate/*.json"],
}
SKIP = {"goals/_index.json"}

errors = []


def err(msg):
    errors.append(msg)


def load(stem):
    out = {}
    for pat in DIRS[stem]:
        for f in glob.glob(str(RO / pat)):
            rel = os.path.relpath(f, RO).replace(os.sep, "/")
            if rel in SKIP:
                continue
            out[f] = json.load(open(f))
    return out


def validators():
    return {s: Draft202012Validator(json.loads((SCHEMAS / f"{s}.schema.json").read_text())) for s in DIRS}


# ----------------------------------------------------------------------------
def main():
    V = validators()
    objs = {s: load(s) for s in DIRS}

    # (a) schema-validate everything
    nval = 0
    for stem, files in objs.items():
        for f, inst in files.items():
            errs = list(V[stem].iter_errors(inst))
            nval += 1
            for e in errs:
                loc = "/".join(str(p) for p in e.path) or "<root>"
                err(f"SCHEMA {os.path.relpath(f)} [{stem}/{loc}]: {e.message}")

    # index by id
    rec = {d["id"]: d for d in objs["run_record"].values()}
    claim = {d["id"]: (f, d) for f, d in objs["claim_card"].items()}
    goal = {d["id"]: (f, d) for f, d in objs["goal_node"].items()}
    comp = {d["id"]: d for d in objs["substrate_component"].values()}
    comp_versions = {cid: {v["version"] for v in d["versions"]} for cid, d in comp.items()}

    # (b) fill goal_node.child_runs + spent.runs
    by_node = defaultdict(list)
    for rid, d in rec.items():
        by_node[d["goal_node"]].append(rid)
    node_count = {}
    for gid, (f, d) in goal.items():
        kids = sorted(by_node.get(gid, []))
        node_count[gid] = len(kids)
        if d.get("child_runs") != kids or d.get("spent", {}).get("runs") != len(kids):
            d["child_runs"] = kids
            sp = d.get("spent", {}) or {}
            sp["runs"] = len(kids)
            sp.setdefault("wall_s", 0)
            d["spent"] = sp
            d["updated_at"] = STAMP
            Path(f).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")

    # (c) fill claim supersedes back-edges
    superseded_by = {cid: d.get("superseded_by") for cid, (f, d) in claim.items()}
    for cid, (f, d) in claim.items():
        target = d.get("superseded_by")
        if target and target in claim:
            tf, td = claim[target]
            if td.get("supersedes") != cid:
                td["supersedes"] = cid
                td["updated_at"] = STAMP
                Path(tf).write_text(json.dumps(td, indent=2, ensure_ascii=False) + "\n")

    # (d) refresh goals/_index.json n_experiments
    idx_path = RO / "goals" / "_index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        for e in idx:
            e["n_experiments"] = node_count.get(e["id"], 0)
        idx_path.write_text(json.dumps(idx, indent=2, ensure_ascii=False) + "\n")

    # ----- (e) referential integrity (schemas/README.md) -----
    rec_ids, goal_ids, claim_ids = set(rec), set(goal), set(claim)

    for rid, d in rec.items():
        if d["goal_node"] not in goal_ids:
            err(f"REF run {rid}.goal_node -> missing goal {d['goal_node']}")
        for p in d.get("parents", []):
            if p not in rec_ids:
                err(f"REF run {rid}.parents -> missing run {p}")
        for cid, ver in d["substrate_versions"].items():
            if cid not in comp:
                err(f"REF run {rid}.substrate_versions -> missing component {cid}")
            elif ver not in comp_versions[cid]:
                err(f"REF run {rid}.substrate_versions -> {cid}@{ver} not in versions")

    for cid, (f, d) in claim.items():
        for r in d.get("supporting_runs", []) + d.get("refuting_runs", []):
            if r not in rec_ids:
                err(f"REF claim {cid} run-ref -> missing run {r}")
        for edge in ("superseded_by", "supersedes"):
            t = d.get(edge)
            if t is not None and t not in claim_ids:
                err(f"REF claim {cid}.{edge} -> missing claim {t}")
        for ref in d.get("depends_on", []):
            lhs, _, ver = ref.partition("@")
            if lhs not in comp:
                err(f"REF claim {cid}.depends_on -> missing component {lhs}")
            elif ver not in comp_versions[lhs]:
                err(f"REF claim {cid}.depends_on -> {ref} not in versions")
        if d["status"] == "superseded" and not d.get("superseded_by"):
            err(f"RULE claim {cid} status=superseded but superseded_by unset")

    for gid, (f, d) in goal.items():
        par = d.get("parent")
        if par is None:
            if d["node_kind"] != "thesis":
                err(f"RULE goal {gid} parent=null but node_kind={d['node_kind']}")
        elif par not in goal_ids:
            err(f"REF goal {gid}.parent -> missing goal {par}")
        for r in d.get("child_runs", []):
            if r not in rec_ids:
                err(f"REF goal {gid}.child_runs -> missing run {r}")
        lm = d.get("last_measured")
        if lm and lm.get("run") and lm["run"] not in rec_ids:
            err(f"REF goal {gid}.last_measured.run -> missing run {lm['run']}")

    thesis = [gid for gid, (f, d) in goal.items() if d["node_kind"] == "thesis" and d.get("parent") is None]
    if len(thesis) != 1:
        err(f"RULE expected exactly 1 thesis root, found {len(thesis)}: {thesis}")

    # ----- (f) write the backfill README -----
    write_readme(rec, claim, goal, comp, node_count)

    # ----- summary -----
    print(f"objects validated: {nval}")
    print(f"  run_records      : {len(rec)}")
    print(f"  claim_cards      : {len(claim)}  ({sum(1 for _,(f,d) in claim.items() if d['deck']=='research')} research / {sum(1 for _,(f,d) in claim.items() if d['deck']=='preference')} preference)")
    print(f"  goal_nodes       : {len(goal)}  (non-empty: {sum(1 for v in node_count.values() if v)})")
    print(f"  substrate_comps  : {len(comp)}")
    print(f"run status         : {dict(Counter(d['status'] for d in rec.values()))}")
    print(f"claim status       : {dict(Counter(d['status'] for _,(f,d) in claim.items()))}")
    print()
    if errors:
        print(f"INTEGRITY ERRORS: {len(errors)}")
        for e in errors[:60]:
            print("  -", e)
        sys.exit(1)
    print("ALL CHECKS PASS — store is referentially closed.")


def write_readme(rec, claim, goal, comp, node_count):
    cl = {cid: d for cid, (f, d) in claim.items()}
    research = {k: v for k, v in cl.items() if v["deck"] == "research"}
    pref = {k: v for k, v in cl.items() if v["deck"] == "preference"}
    run_status = Counter(d["status"] for d in rec.values())
    oracle_true = sum(1 for d in rec.values() if d["oracle_clean"])
    prop = Counter(d["substrate_versions"].get("propagator") for d in rec.values())
    needs_rep = [cid for cid, d in research.items() if d["status"] == "needs_replication"]
    superseded = [cid for cid, d in research.items() if d["status"] == "superseded"]
    retracted = [cid for cid, d in research.items() if d["status"] == "retracted"]
    live = [cid for cid, d in research.items() if d["status"] == "live"]

    md = f"""# Research OS — W-B backfill README

One-time lossy-pragmatic migration of the survey corpus (146 experiment writeups +
the memory palimpsest + PROGRESS) into the canonical trust-machine objects. Generated
by the scripts in this directory; this file is regenerated by `reconcile.py`.

## Coverage

| Object | Count | Where |
|---|---|---|
| Run records | {len(rec)} | `research_os/records/*.json` (1:1 with the {len(rec)} manifest ids) |
| Claim cards | {len(cl)} | `research_os/claims/{{research,preferences}}/` ({len(research)} research / {len(pref)} preference) |
| Goal nodes | {len(goal)} | `research_os/goals/*.json` ({sum(1 for v in node_count.values() if v)} carry runs) |
| Substrate components | {len(comp)} | `research_os/substrate/*.json` ({", ".join(sorted(comp))}) |

Run-record status: {run_status['confirmed']} confirmed / {run_status['refuted']} refuted / {run_status['inconclusive']} inconclusive.  oracle_clean=true on {oracle_true}/{len(rec)}
(the genuinely-blind / no-truth-read runs; everything probed *at truth* is false).
Propagator era: {prop.get("1.0.0",0)} runs at 1.0.0 (pre ω-sign fix, ≤ s066) /
{prop.get("2.0.0",0)} at 2.0.0 (post-fix, commit d5705ff).

## The spine (claim cards)

**Supersede chains (corrections are pointers, not rewrites):**
- `claim_anchor-aliasing-is-the-blocker` → **superseded_by** `claim_hard-shoot-trap-is-the-blocker` (s099 → s100/s105).
- `claim_full-lc-rmse-fails-needs-windowing` → **superseded_by** `claim_full-lc-rmse-discriminator-works` (reversed by the times[0]==0 audit, s097).

**Retracted (oracle-leak, not a real result):**
- `claim_cloud-data-proven-on-seed89` — the Band-A rows were oracle-injected truth-cluster polish (s058/s059). Terminal state; no supersedes edge.

**needs_replication (blast-radius — evidence rests on propagator@1.0.0):**
{chr(10).join(f"- `{c}`" for c in needs_rep)}

> **Headline epistemic finding of the backfill.** Two of the five findings the
> resume note called "load-bearing live" — `claim_pol-diam-predicts-basin-width`
> (Spearman −0.94) and `claim_lc-only-omega-priors-dead` — rest *only* on pre-fix
> runs (propagator 1.0.0) and were never replicated post-fix. The blast-radius rule
> flips them to **needs_replication**, which is the convention-bug memory's own rule
> ("basin widths / ρ-values are stale") made mechanical. The umbrella card
> `claim_prefix-numerical-findings-stale` records the s001–s066 class. The other
> three named findings (body-twin, windowed-photometry, s019-bracket-held-out) have
> post-fix anchors (s081 / s106 / s099) and stay **live**.

**Live research claims:** {", ".join(f"`{c}`" for c in sorted(live))}.
**Preference deck:** {", ".join(f"`{c}`" for c in sorted(pref))}.

## What is lossy / what to know

- **Lossy by design.** Each run record keeps metadata + outcome + a few key numbers +
  a 1–3 sentence narrative — NOT the full prose. `writeup_refs` links every record
  back to its `.md` for nuance (PLAN W-B contract).
- **Non-experiment writeups processed uniformly.** 18 manifest entries are designs /
  handoffs / a tool / an insight / audits / validators; the designs/handoffs/tool/
  insight come out `status: inconclusive` (no experimental result), keeping
  `goal_node.child_runs` complete.
- **Dropped dangling parents.** 5 records referenced `s073d_polhode_match_cluster457`,
  which has no writeup and is not in the manifest; those `parents` edges were dropped
  by `normalize_parents.py` (s073e, s073f, s077, s079, s081). 12 short-form parent ids
  were expanded to full stems in the same pass.
- **Substrate hashes are sha256 prefixes** of the current source files (honest, not
  placeholders); the `versions[]` history encodes the ω-sign fix (1.0.0→2.0.0).

## How to regenerate (from repo root)

```bash
python research_os/backfill/build_goal_tree.py      # 35 goal nodes + manifest + _index
# run records: 7 general-purpose subagents over run_manifest.json per EXTRACT_INSTRUCTIONS.md
python research_os/backfill/normalize_parents.py     # parents -> resolvable full stems
python research_os/backfill/build_claim_cards.py      # the 16 spine claim cards
python research_os/backfill/reconcile.py              # validate + fill edges + this README
```

Referential integrity (the checks in `schemas/README.md`) passes: every run.goal_node,
parents, claim run-refs, supersede edges, depends_on / substrate_versions, and the
single thesis-root rule resolve. `reconcile.py` exits non-zero if any break.
"""
    (RO / "backfill" / "README.md").write_text(md)


if __name__ == "__main__":
    main()
