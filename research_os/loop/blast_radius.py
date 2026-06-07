#!/usr/bin/env python3
"""Blast-radius — the one-line cure for the s001–s066 class (PLAN §3, Q6).

When a substrate component gains a bug-fix version, every claim whose evidence rests
on an older version is no longer trustworthy. This script computes that radius and
flips the affected claims to `needs_replication` — the SAFE direction (it never
asserts a claim is true; it only marks evidence as needing re-measurement), so it is
allowed to act automatically (PLAN §8 lists blast-radius as a hook, distinct from
claim *authoring* which is human-confirm).

A claim `claim_X` is in the blast radius of component `c` iff:
  - some `depends_on` ref pins `c@vX`, and
  - a LATER version of `c` (after vX in its `versions[]`) has `is_bug_fix: true`.
Non-bug-fix bumps (features) do not invalidate evidence, so they don't flip.

The propagator example encodes exactly the ω-sign fix: claims on `propagator@1.0.0`
are in radius (2.0.0 is is_bug_fix); claims on `propagator@2.0.0` (the head) are not.

Usage:
    python research_os/loop/blast_radius.py            # report + APPLY flips
    python research_os/loop/blast_radius.py --check     # report only, no writes
    python research_os/loop/blast_radius.py --component propagator   # limit to one
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
# Claim statuses that blast-radius is allowed to move into needs_replication.
FLIPPABLE = {"live", "draft"}


def _load_substrate(root):
    out = {}
    for f in glob.glob(os.path.join(root, "substrate/*.json")):
        d = json.load(open(f))
        out[d["id"]] = d
    return out


def _load_claims(root):
    out = {}
    for f in glob.glob(os.path.join(root, "claims/*/*.json")):
        d = json.load(open(f))
        out[f] = d
    return out


def bugfix_after(comp: dict, version: str) -> list[str]:
    """Versions strictly after `version` (by list order) that are bug fixes."""
    vers = [v["version"] for v in comp["versions"]]
    if version not in vers:
        return []
    idx = vers.index(version)
    return [v["version"] for v in comp["versions"][idx + 1:] if v.get("is_bug_fix")]


def compute(root: str = ROOT, only: str | None = None):
    """Return (in_radius, to_flip): lists of (claim_file, claim, component, pinned_v, bugfixes)."""
    subs = _load_substrate(root)
    claims = _load_claims(root)
    in_radius, to_flip = [], []
    for f, c in claims.items():
        for ref in c.get("depends_on", []):
            cid, _, ver = ref.partition("@")
            if only and cid != only:
                continue
            comp = subs.get(cid)
            if not comp:
                continue
            fixes = bugfix_after(comp, ver)
            if fixes:
                rec = (f, c, cid, ver, fixes)
                in_radius.append(rec)
                if c.get("status") in FLIPPABLE:
                    to_flip.append(rec)
                break  # one in-radius dep is enough to flag the claim
    return in_radius, to_flip


def apply_flips(to_flip, stamp: str):
    for f, c, cid, ver, fixes in to_flip:
        c["status"] = "needs_replication"
        c["updated_at"] = stamp
        with open(f, "w") as fh:
            fh.write(json.dumps(c, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Compute + apply the substrate blast radius.")
    ap.add_argument("--check", action="store_true", help="report only; do not write")
    ap.add_argument("--component", help="limit to one component id")
    ap.add_argument("--date", help="YYYY-MM-DD stamp for flipped claims (default: today UTC)")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    stamp = f"{args.date}T00:00:00Z" if args.date else datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")

    in_radius, to_flip = compute(args.root, args.component)
    if not in_radius:
        print("blast-radius: no claims rest on a bug-fixed-older substrate version. Clean.")
        return

    print(f"blast-radius: {len(in_radius)} claim(s) in radius "
          f"({len(to_flip)} flippable live/draft, "
          f"{len(in_radius) - len(to_flip)} already needs_replication/superseded/retracted):")
    for f, c, cid, ver, fixes in in_radius:
        mark = "FLIP" if c.get("status") in FLIPPABLE else c.get("status")
        print(f"  [{mark:18s}] {c['id']:48s} {cid}@{ver} -> bug-fixed by {','.join(fixes)}")

    if to_flip and not args.check:
        apply_flips(to_flip, stamp)
        print(f"\napplied {len(to_flip)} flip(s) -> needs_replication.")
    elif to_flip:
        print(f"\n--check: {len(to_flip)} flip(s) NOT applied (dry run).")


if __name__ == "__main__":
    main()
