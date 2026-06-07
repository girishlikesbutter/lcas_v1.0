#!/usr/bin/env python3
"""Phase-0 visible test: validate every example instance against its schema.

Maps examples/<schema-stem>.<label>.json  ->  <schema-stem>.schema.json
(e.g. examples/run_record.s106.json -> run_record.schema.json) and asserts each
instance validates under JSON Schema draft 2020-12.

Usage:  python research_os/schemas/validate_examples.py
Exit 0 = all valid; exit 1 = at least one failure (printed).
"""
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:
    sys.exit("jsonschema not installed: pip install jsonschema")

HERE = Path(__file__).resolve().parent
EX = HERE / "examples"

KNOWN_SCHEMAS = {p.name[: -len(".schema.json")] for p in HERE.glob("*.schema.json")}


def schema_stem_for(example_path: Path) -> str:
    """run_record.s106.json -> 'run_record' (longest known-schema prefix)."""
    name = example_path.name
    # strip .json, then take the leading dotted segment(s) that match a schema stem
    parts = name[: -len(".json")].split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in KNOWN_SCHEMAS:
            return candidate
    raise ValueError(f"no schema matches example {name} (known: {sorted(KNOWN_SCHEMAS)})")


def main() -> int:
    examples = sorted(EX.glob("*.json"))
    if not examples:
        print("no examples found")
        return 1

    failures = 0
    for ex in examples:
        stem = schema_stem_for(ex)
        schema = json.loads((HERE / f"{stem}.schema.json").read_text())
        instance = json.loads(ex.read_text())
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
        if errors:
            failures += 1
            print(f"FAIL  {ex.name}  (schema: {stem})")
            for e in errors:
                loc = "/".join(str(p) for p in e.path) or "<root>"
                print(f"      - {loc}: {e.message}")
        else:
            print(f"ok    {ex.name}  -> {stem}.schema.json")

    print(f"\n{len(examples) - failures}/{len(examples)} examples valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
