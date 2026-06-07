#!/usr/bin/env python3
"""Epistemic gate-check on a written object (PLAN §3 — turn recurring traps into
executable tripwires). Mechanical, object-level checks only:

  - N-scope        : a `cohort`-scoped run/claim must cite N >= RO_NSCOPE_MIN (default 10),
                     else it's really N1 — the s073 over-claim trap.
  - oracle-coherent: oracle_clean must agree with the oracle-leak gate fields; a run can't
                     claim blind (oracle_clean=true) while listing oracle-leak as failed.
  - oracle-rests   : a `live` claim resting ONLY on oracle_clean=false runs is suspect —
                     the s058/s059 disease. (soft warning)
  - stale-substrate: a `live` claim whose depends_on includes a bug-fixed-older version
                     should be needs_replication (overlaps blast-radius). (soft warning)

RUNTIME gates (conservation/smoke at truth, validator-not-at-idx-0, parallelism) cannot
be judged from a static object — they live inside `run`/the scorer, not this hook. This
checker only sees the file.

Severity: HARD violations -> exit 2 (the hook feeds stderr to the model to fix the
object it just wrote). SOFT warnings are heuristics that can false-positive (e.g.
`oracle-rests` flags forward-model property claims like body-twin, which legitimately
compare against truth without the blind-inversion concern) — so they surface only in
audit mode (`--all` / `--soft`), NOT on every per-write hook firing, keeping the hook
crisp and non-noisy.

Usage:
    python research_os/loop/gate_check.py research_os/records/sXXX.json   # one object, HARD only (the hook)
    python research_os/loop/gate_check.py --all                            # whole store, HARD + SOFT (audit)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSCOPE_MIN = int(os.environ.get("RO_NSCOPE_MIN", "10"))


def _records(root):
    return {json.load(open(f))["id"]: json.load(open(f)) for f in glob.glob(os.path.join(root, "records/*.json"))}


def _substrate(root):
    return {json.load(open(f))["id"]: json.load(open(f)) for f in glob.glob(os.path.join(root, "substrate/*.json"))}


def _bugfixed_older(subs, ref):
    cid, _, ver = ref.partition("@")
    comp = subs.get(cid)
    if not comp:
        return False
    vers = [v["version"] for v in comp["versions"]]
    if ver not in vers:
        return False
    idx = vers.index(ver)
    return any(v.get("is_bug_fix") for v in comp["versions"][idx + 1:])


def check_object(obj: dict, root: str = ROOT):
    """Return (hard, soft): lists of message strings."""
    hard, soft = [], []
    kind = obj.get("kind")

    if kind == "run_record":
        if obj.get("claim_scope") == "cohort" and (obj.get("N") or 0) < NSCOPE_MIN:
            hard.append(f"N-scope: claim_scope=cohort but N={obj.get('N')} < {NSCOPE_MIN} — downgrade to N1 or raise N.")
        oc = obj.get("oracle_clean")
        gf, gp = obj.get("gates_failed", []), obj.get("gates_passed", [])
        if oc is True and "oracle-leak" in gf:
            hard.append("oracle-coherent: oracle_clean=true but 'oracle-leak' is in gates_failed.")
        if oc is False and "oracle-leak" in gp:
            hard.append("oracle-coherent: oracle_clean=false but 'oracle-leak' is in gates_passed.")

    elif kind == "claim_card":
        sc = obj.get("scope", {})
        if sc.get("kind") == "cohort" and (sc.get("N") or 0) < NSCOPE_MIN:
            hard.append(f"N-scope: scope.kind=cohort but N={sc.get('N')} < {NSCOPE_MIN}.")
        if obj.get("status") == "live":
            subs = _substrate(root)
            for ref in obj.get("depends_on", []):
                if _bugfixed_older(subs, ref):
                    soft.append(f"stale-substrate: live claim depends_on {ref} (bug-fixed since) — likely needs_replication (run blast_radius.py).")
            recs = _records(root)
            sup = [recs[r] for r in obj.get("supporting_runs", []) if r in recs]
            if sup and all(r.get("oracle_clean") is False for r in sup):
                soft.append("oracle-rests: live claim's supporting_runs are ALL oracle_clean=false — the s058/s059 trap; needs a blind anchor.")

    return hard, soft


def main():
    ap = argparse.ArgumentParser(description="Epistemic gate-check on a store object.")
    ap.add_argument("path", nargs="?", help="object json to check")
    ap.add_argument("--all", action="store_true", help="check every record + claim (audit; includes soft warnings)")
    ap.add_argument("--soft", action="store_true", help="also emit soft heuristic warnings on a single path")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    show_soft = args.all or args.soft
    targets = []
    if args.all:
        targets = glob.glob(os.path.join(args.root, "records/*.json")) + glob.glob(os.path.join(args.root, "claims/*/*.json"))
    elif args.path:
        targets = [args.path]
    else:
        ap.error("give a path or --all")

    any_hard = False
    for p in targets:
        try:
            obj = json.load(open(p))
        except Exception:
            continue
        hard, soft = check_object(obj, args.root)
        for m in hard:
            print(f"GATE-FAIL {os.path.basename(p)}: {m}", file=sys.stderr)
            any_hard = True
        if show_soft:
            for m in soft:
                print(f"gate-warn {os.path.basename(p)}: {m}", file=sys.stderr)
    if any_hard:
        sys.exit(2)


if __name__ == "__main__":
    main()
