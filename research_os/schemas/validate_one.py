#!/usr/bin/env python3
"""Validate one instance file against a named schema.

Usage:  python research_os/schemas/validate_one.py <instance.json> <schema-stem>
   e.g.  python research_os/schemas/validate_one.py research_os/records/s106_hybrid_loss_polish.json run_record

Exit 0 = valid; exit 1 = invalid (errors printed).
"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit("usage: validate_one.py <instance.json> <schema-stem>")
    inst_path, stem = sys.argv[1], sys.argv[2]
    schema = json.loads((HERE / f"{stem}.schema.json").read_text())
    instance = json.loads(Path(inst_path).read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e: list(e.path))
    if not errors:
        print(f"ok  {inst_path}")
        return 0
    print(f"FAIL  {inst_path}  ({stem})")
    for e in errors:
        loc = "/".join(str(p) for p in e.path) or "<root>"
        print(f"  - {loc}: {e.message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
