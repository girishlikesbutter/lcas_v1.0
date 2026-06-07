#!/usr/bin/env python3
"""Normalize run-record `parents` to resolvable full-stem run ids (W-B backfill).

The extraction agents copied the manifest `related` field verbatim, which uses
short forms (e.g. "s040", "s059j") while records are keyed by full stems
(e.g. "s040_basin_radius_3seed"). The run_record schema requires `parents` to
resolve to run-record ids. This pass:

  UNIQUE    short-form prefix-matches exactly one record  -> expand to it.
  AMBIGUOUS prefix matches a primary run + a _design/_handoff doc
                                                          -> resolve to the primary
                                                             (drop design/handoff candidate).
  DANGLING  no record matches (e.g. s073d_*, which has no writeup and is not in
            the manifest)                                 -> drop, log as a known
                                                             missing ancestor.

Idempotent: a second run is a no-op (every parent already resolves). Run from repo root.
"""
import json, glob, os, sys

REC_DIR = "research_os/records"
SECONDARY_SUFFIXES = ("_design", "_handoff")  # de-prioritised when disambiguating


def load_records():
    recs = {}
    for f in sorted(glob.glob(os.path.join(REC_DIR, "*.json"))):
        with open(f) as fh:
            recs[os.path.basename(f)[:-5]] = (json.load(fh), f)
    return recs


def resolve(p, rec_ids):
    """Return (resolved_id | None, kind)."""
    if p in rec_ids:
        return p, "EXACT"
    cands = sorted(r for r in rec_ids if r.startswith(p + "_"))
    if len(cands) == 1:
        return cands[0], "UNIQUE"
    if len(cands) > 1:
        primary = [c for c in cands if not c.endswith(SECONDARY_SUFFIXES)]
        if len(primary) == 1:
            return primary[0], "AMBIGUOUS->primary"
        return cands[0], "AMBIGUOUS->first"  # last-resort deterministic pick
    return None, "DANGLING"


def main():
    recs = load_records()
    rec_ids = set(recs)
    changed = 0
    dangling = []  # (record, dropped_parent)
    remaps = []    # (record, old, new)

    for stem, (d, path) in recs.items():
        parents = d.get("parents", [])
        if not parents:
            continue
        new_parents, seen = [], set()
        mutated = False
        for p in parents:
            rid, kind = resolve(p, rec_ids)
            if rid is None:
                dangling.append((stem, p))
                mutated = True
                continue
            if rid != p:
                remaps.append((stem, p, rid))
                mutated = True
            if rid not in seen:
                seen.add(rid)
                new_parents.append(rid)
        if mutated:
            d["parents"] = new_parents
            with open(path, "w") as fh:
                json.dump(d, fh, indent=2)
                fh.write("\n")
            changed += 1

    print(f"records rewritten: {changed}")
    print(f"\nremapped short-form parents ({len(remaps)}):")
    for stem, old, new in remaps:
        print(f"  {stem:42s} {old:38s} -> {new}")
    print(f"\ndropped DANGLING parents ({len(dangling)}):")
    for stem, p in dangling:
        print(f"  {stem:42s} -> {p}  [no such record]")

    # verify: every remaining parent now resolves
    recs = load_records()
    rec_ids = set(recs)
    bad = [(s, p) for s, (d, _) in recs.items() for p in d.get("parents", []) if p not in rec_ids]
    print(f"\nunresolved parents after normalize: {len(bad)}")
    if bad:
        for s, p in bad:
            print("  STILL BAD:", s, p)
        sys.exit(1)


if __name__ == "__main__":
    main()
