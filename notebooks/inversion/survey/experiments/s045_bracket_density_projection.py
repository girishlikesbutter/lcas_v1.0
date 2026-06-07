"""s045 — |ω|-aware bracket density projection.

Pure-math projection on cached data. Combines the s042 cohort scaling
rule (basin width vs ``|ω|``) with the s032 per-seed bracket coverage
(``nearest_cell_pct``) and the cohort ``|ω|`` distribution to project:

  1. Per-seed pass/fail of the current bracket against the s042 scaling
     rule under {const ∈ {1%, 2%}, exponent ∈ {0.5, 0.75, 1.0}}.
  2. Per-seed cell count needed to guarantee basin coverage assuming
     an LS-peak prior of ±50% (loose) and ±10% (tight) around truth-|ω|.
  3. Cohort-shared adaptive grid total cell count under power-law
     spacing ``δ|ω| = c × |ω|^a`` over [0.05, 10] dps.

No new compute beyond loading cached NPZs/JSONs. Wall < 5 s.

Outputs:
    results/s045_bracket_density_projection/projection.csv
    results/s045_bracket_density_projection/summary.json
    results/s045_bracket_density_projection/run.log
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))

from lib.traj_load import list_seeds, truth_state  # noqa: E402

S032_DIR = SURVEY_DIR / "results" / "s032_cohort_fast"
S042_PATH = SURVEY_DIR / "results" / "s042_basin_radius_cohort" / "summary.json"
OUT_DIR = SURVEY_DIR / "results" / "s045_bracket_density_projection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# s042 scaling rule: basin pct ≈ const / |ω|^exp.
# Wind-down (2026-05-06): const ∈ {1, 2} %, exp ∈ {0.5, 1}.
# We add exp = 0.75 as a midpoint and report all six combinations.
SCALING_RULE_CONSTS = [1.0, 2.0]  # percent
SCALING_RULE_EXPONENTS = [0.5, 0.75, 1.0]

# Search-range prior over |ω| in dps for the cohort-shared adaptive grid.
# Cohort empirical range is (0.106, 1.476) dps; we add ~30% margin on each
# end so the grid still works for outliers. The wider (0.05, 10.0) range is
# deliberately not used — it was a strawman baseline; the cohort doesn't
# populate it.
OMEGA_RANGE_DPS = (0.05, 2.0)


def basin_pct(omega_mag_dps: float, c_pct: float, a: float) -> float:
    """Required basin half-width as a percentage of |ω|, per the scaling rule.

    rule: δ|ω|/|ω| ≈ c_pct% / |ω|^a   →   basin_pct = c_pct / |ω|^a.
    """
    return c_pct / (omega_mag_dps ** a)


def adaptive_cells(
    omega_min_dps: float,
    omega_max_dps: float,
    c_pct: float,
    a: float,
) -> float:
    """Cell count for a power-law-spaced grid that hits basin everywhere in
    [omega_min, omega_max] dps under spacing δ|ω|/|ω| = (c_pct/100)/|ω|^a.

    Cell count N satisfies   ∫(d|ω| / spacing(|ω|)) = N,
    where spacing(|ω|) = (c_pct/100) × |ω|^(1-a).

    For a ≠ 1:   N = (omega_max^a - omega_min^a) / (a × (c_pct / 100)).
    For a == 1:  N = ln(omega_max / omega_min) / (c_pct / 100).
    """
    c = c_pct / 100.0
    if abs(a - 1.0) < 1e-12:
        return math.log(omega_max_dps / omega_min_dps) / c
    return (omega_max_dps ** a - omega_min_dps ** a) / (a * c)


def per_seed_cells(omega_dps: float, prior_pct: float, basin_pct_val: float) -> float:
    """Cells needed per seed if we know |ω| within ±prior_pct% of truth.

    Spacing must be ≤ basin_pct% × |ω|. Range = 2 × prior_pct% × |ω|.
    Cells = range / spacing = (2 × prior_pct) / basin_pct.
    """
    return (2.0 * prior_pct) / basin_pct_val


def load_cohort_omega_mag() -> dict[int, float]:
    """All 100 seeds → truth |ω| in dps (from cached trajectory NPZs)."""
    out = {}
    for s in list_seeds():
        out[s] = float(truth_state(s)["omega_mag_dps"])
    return out


def load_s032_nearest_pct() -> dict[int, float]:
    """Map seed → nearest_cell_pct from cohort_progress.csv. Only OK seeds."""
    csv_path = S032_DIR / "cohort_progress.csv"
    out = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "OK":
                continue
            try:
                s = int(row["seed"])
                pct = float(row["nearest_cell_pct"])
            except (TypeError, ValueError):
                continue
            out[s] = pct
    return out


def load_s042_measured_basins() -> dict[int, dict]:
    """Map seed → {omega_mag_dps, basin_pos_pct, basin_neg_pct, omega_dir_deg}.

    Uses the truth-attractor radii (twin radii are within reflection-symmetric
    by 12/13 of cohort per s042; for the projection we take the more
    conservative (smaller) of pos/neg per seed for ω-mag.
    """
    s = json.loads(S042_PATH.read_text())
    out = {}
    for k, rec in s["by_seed"].items():
        seed = int(k.replace("seed", ""))
        truth = rec["truth"]["basin_radii"]
        out[seed] = {
            "omega_mag_dps": float(rec["omega_mag_dps"]),
            "basin_pos_pct": float(truth["omega_mag_pct"]["radius_pos_pct"]),
            "basin_neg_pct": float(truth["omega_mag_pct"]["radius_neg_pct"]),
            "basin_pct_min": min(
                float(truth["omega_mag_pct"]["radius_pos_pct"]),
                float(truth["omega_mag_pct"]["radius_neg_pct"]),
            ),
            "omega_dir_basin_deg": float(truth["omega_dir_deg"]["radius"]),
        }
    return out


def main():
    print("=== s045 |ω|-aware bracket density projection ===", flush=True)

    cohort_omega = load_cohort_omega_mag()
    s032_nearest = load_s032_nearest_pct()
    s042_basins = load_s042_measured_basins()

    print(
        f"Loaded: 100-seed cohort |ω|; "
        f"s032 OK seeds: {len(s032_nearest)}; "
        f"s042 measured: {len(s042_basins)}",
        flush=True,
    )
    cohort_w = sorted(cohort_omega.values())
    print(
        f"Cohort |ω| (dps): min={cohort_w[0]:.4f}, max={cohort_w[-1]:.4f}, "
        f"median={cohort_w[len(cohort_w)//2]:.4f}, p90={cohort_w[int(0.9*len(cohort_w))]:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 1. Validate scaling rule against s042 measured basins.
    # ------------------------------------------------------------------ #
    print("\n[1] Scaling-rule validation against s042 measured basins:", flush=True)
    print(
        f"  {'seed':>4} {'|ω|_dps':>8} {'measured_pct':>12} "
        + "  ".join(
            f"c={c}/a={a}".ljust(10) for c in SCALING_RULE_CONSTS for a in SCALING_RULE_EXPONENTS
        ),
        flush=True,
    )
    rule_validation = []
    for seed in sorted(s042_basins.keys()):
        omega = s042_basins[seed]["omega_mag_dps"]
        measured = s042_basins[seed]["basin_pct_min"]
        row = {"seed": seed, "omega_mag_dps": omega, "measured_basin_pct": measured}
        cells = []
        for c in SCALING_RULE_CONSTS:
            for a in SCALING_RULE_EXPONENTS:
                pred = basin_pct(omega, c, a)
                row[f"pred_c{c}_a{a}"] = pred
                row[f"pass_c{c}_a{a}"] = pred <= measured  # rule predicts narrower than measured ⇒ rule is conservative
                cells.append(f"{pred:6.2f}{'P' if pred <= measured else 'F'}".ljust(10))
        rule_validation.append(row)
        print(
            f"  {seed:>4} {omega:>8.3f} {measured:>12.2f}  " + "  ".join(cells),
            flush=True,
        )
    # Pass count under each rule
    rule_pass = {}
    for c in SCALING_RULE_CONSTS:
        for a in SCALING_RULE_EXPONENTS:
            n = sum(1 for r in rule_validation if r[f"pass_c{c}_a{a}"])
            rule_pass[f"c{c}_a{a}"] = (n, len(rule_validation))
    print("\n  Rule pass counts (rule predicted basin <= measured ⇒ rule is conservative):", flush=True)
    for k, (n, total) in rule_pass.items():
        print(f"    {k}: {n}/{total}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Cohort coverage under the current s032 bracket.
    # ------------------------------------------------------------------ #
    print(
        "\n[2] Current s032 bracket coverage (nearest_cell_pct vs scaling-rule-required basin):",
        flush=True,
    )
    cohort_coverage = {}
    for c in SCALING_RULE_CONSTS:
        for a in SCALING_RULE_EXPONENTS:
            within = 0
            for seed, omega in cohort_omega.items():
                if seed not in s032_nearest:
                    continue
                req = basin_pct(omega, c, a)
                if s032_nearest[seed] <= req:
                    within += 1
            cohort_coverage[f"c{c}_a{a}"] = (within, len(s032_nearest))
            print(
                f"  rule c={c}/a={a}: {within}/{len(s032_nearest)} OK seeds within basin",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # 3. Per-seed cell count under |ω|-aware ADAPTIVE bracket.
    #    Assumption: LS-peak prior known within ±X% of truth.
    # ------------------------------------------------------------------ #
    print(
        "\n[3] Per-seed cell count under |ω|-aware adaptive bracket (ceil at 1):",
        flush=True,
    )
    print(
        "    Assume LS-peak prior gives truth ±50% (loose) or ±10% (tight). "
        "Spacing tuned to the cohort scaling rule. "
        "Scaling rule: c=1%, exponent=0.5 (the conservative tight choice that "
        "fits seed 14 within a factor of 2).",
        flush=True,
    )
    PRIOR_LOOSE_PCT = 50.0
    PRIOR_TIGHT_PCT = 10.0
    C_REPORT = 1.0
    A_REPORT = 0.5
    rows = []
    cohort_total_loose = 0.0
    cohort_total_tight = 0.0
    for seed in sorted(cohort_omega.keys()):
        omega = cohort_omega[seed]
        bp = basin_pct(omega, C_REPORT, A_REPORT)
        cells_loose = max(1.0, math.ceil(per_seed_cells(omega, PRIOR_LOOSE_PCT, bp)))
        cells_tight = max(1.0, math.ceil(per_seed_cells(omega, PRIOR_TIGHT_PCT, bp)))
        rows.append(
            {
                "seed": seed,
                "omega_mag_dps": omega,
                "basin_pct_required": bp,
                "cells_loose_prior": cells_loose,
                "cells_tight_prior": cells_tight,
                "current_nearest_cell_pct": s032_nearest.get(seed, None),
                "in_basin_current": (
                    s032_nearest[seed] <= bp if seed in s032_nearest else None
                ),
            }
        )
        cohort_total_loose += cells_loose
        cohort_total_tight += cells_tight

    n_in_basin = sum(1 for r in rows if r["in_basin_current"])
    print(
        f"  Cohort: |ω|-aware adaptive bracket totals — "
        f"loose prior (±{PRIOR_LOOSE_PCT:.0f}%) = {cohort_total_loose:,.0f} cells, "
        f"tight prior (±{PRIOR_TIGHT_PCT:.0f}%) = {cohort_total_tight:,.0f} cells.",
        flush=True,
    )
    print(
        f"  vs current uniform 5-cell-per-seed bracket: "
        f"{5 * len(s032_nearest)} cells (only "
        f"{n_in_basin}/{len(s032_nearest)} seeds within basin).",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 4. Cohort-SHARED adaptive grid: total cells across [0.05, 10] dps.
    # ------------------------------------------------------------------ #
    print(
        f"\n[4] Cohort-shared adaptive grid (covers |ω| ∈ {OMEGA_RANGE_DPS} dps), "
        "shared across all 100 seeds:",
        flush=True,
    )
    shared_grid = {}
    for c in SCALING_RULE_CONSTS:
        for a in SCALING_RULE_EXPONENTS:
            n = adaptive_cells(*OMEGA_RANGE_DPS, c, a)
            shared_grid[f"c{c}_a{a}"] = math.ceil(n)
            print(
                f"  rule c={c}/a={a}: {math.ceil(n):>5} cells across "
                f"|ω| ∈ {OMEGA_RANGE_DPS} dps",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # 5. Save artefacts.
    # ------------------------------------------------------------------ #
    out_csv = OUT_DIR / "projection.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_csv}", flush=True)

    summary = {
        "omega_range_dps": list(OMEGA_RANGE_DPS),
        "n_cohort_seeds": len(cohort_omega),
        "n_s032_ok_seeds": len(s032_nearest),
        "n_s042_measured_seeds": len(s042_basins),
        "scaling_rule_constants_pct": SCALING_RULE_CONSTS,
        "scaling_rule_exponents": SCALING_RULE_EXPONENTS,
        "rule_validation_pass_counts_against_s042": rule_pass,
        "cohort_coverage_under_current_bracket": cohort_coverage,
        "per_seed_adaptive_bracket": {
            "scaling_rule_used": {"const_pct": C_REPORT, "exponent": A_REPORT},
            "ls_prior_loose_pct": PRIOR_LOOSE_PCT,
            "ls_prior_tight_pct": PRIOR_TIGHT_PCT,
            "cohort_total_cells_loose": int(cohort_total_loose),
            "cohort_total_cells_tight": int(cohort_total_tight),
            "current_5cell_per_seed_total": 5 * len(s032_nearest),
            "n_in_basin_current": n_in_basin,
            "n_seeds_with_s032": len(s032_nearest),
        },
        "shared_adaptive_grid_cells": shared_grid,
    }
    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    # Headline
    print("\n=== HEADLINES ===", flush=True)
    print(
        f"  s032 within-basin under c=1%/a=0.5 rule: "
        f"{cohort_coverage['c1.0_a0.5'][0]}/{cohort_coverage['c1.0_a0.5'][1]}",
        flush=True,
    )
    print(
        f"  s032 within-basin under c=2%/a=1.0 rule: "
        f"{cohort_coverage['c2.0_a1.0'][0]}/{cohort_coverage['c2.0_a1.0'][1]}",
        flush=True,
    )
    print(
        f"  Per-seed adaptive (loose ±50% LS prior, c=1%/a=0.5): "
        f"{int(cohort_total_loose):,} cohort cells",
        flush=True,
    )
    print(
        f"  Per-seed adaptive (tight ±10% LS prior, c=1%/a=0.5): "
        f"{int(cohort_total_tight):,} cohort cells",
        flush=True,
    )
    print(
        f"  Shared grid (c=1%/a=0.5, |ω| ∈ {OMEGA_RANGE_DPS} dps): "
        f"{shared_grid['c1.0_a0.5']:,} cells",
        flush=True,
    )


if __name__ == "__main__":
    main()
