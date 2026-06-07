#!/usr/bin/env python3
"""
Extract checkpoint data from 25-seed m048 inversion batch and build failure-mode diagnostic table.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Cohort seeds (in order)
COHORT_SEEDS = [6, 7, 8, 11, 16, 17, 34, 35, 42, 45, 47, 48, 51, 57, 59, 64, 67, 69, 71, 78, 79, 84, 89, 91, 99]

PROJECT_ROOT = Path(__file__).parent.parent.parent
DIAG_ROOT = PROJECT_ROOT / "data/results/inversion_diagnostics"

def safe_load_json(path):
    """Load JSON, return empty dict if missing."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {}

def safe_load_npz(path):
    """Load NPZ, return empty dict if missing."""
    if not path.exists():
        return {}
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        return {}

def check_m103_status(seed_dir):
    """Check m103 status: ok / geo_timeout / crash"""
    timeout_flag = seed_dir / "geo_timeout.flag"
    result_json = seed_dir / "result.json"
    
    if timeout_flag.exists():
        return "geo_timeout"
    
    # No result.json = crash
    if not result_json.exists():
        return "crash"
    
    return "ok"

def extract_m103(seed):
    """Extract m103 data."""
    seed_dir = DIAG_ROOT / "m103_hybrid_m048" / f"seed_{seed:03d}"
    if not seed_dir.exists():
        return {}
    
    result = safe_load_json(seed_dir / "result.json")
    geo_ckpt = safe_load_npz(seed_dir / "geo_ckpt.npz")
    
    m103_data = {
        "m103_status": check_m103_status(seed_dir),
        "m103_winner_w_dir_err": result.get("winner", {}).get("w0_err", None),
        "m103_pool_size": 0,
        "m103_pool_min_w_dir_err": None,
    }
    
    # Extract pool data from geo_ckpt
    if "geo_costs" in geo_ckpt:
        m103_data["m103_pool_size"] = len(geo_ckpt["geo_costs"])
    
    if "w0_ref_errs" in geo_ckpt:
        errs = geo_ckpt["w0_ref_errs"]
        m103_data["m103_pool_min_w_dir_err"] = float(np.min(errs))
    
    return m103_data

def extract_m115(seed):
    """Extract m115 data."""
    # Try m115_surrogate_pipeline_m048 first, fall back to m115_surrogate_pipeline
    seed_dir = DIAG_ROOT / "m115_surrogate_pipeline_m048" / f"seed_{seed:03d}"
    if not seed_dir.exists():
        seed_dir = DIAG_ROOT / "m115_surrogate_pipeline" / f"seed_{seed:03d}"
    
    if not seed_dir.exists():
        return {}
    
    result = safe_load_json(seed_dir / "result.json")
    
    m115_data = {
        "m115_n_basins": None,
        "m115_best_q0_err": None,
        "m115_best_hifi_mse": None,
        "m115_winner_w_dir_err": None,
    }
    
    if "de_results" in result:
        m115_data["m115_n_basins"] = result["de_results"].get("n_basins")

    if "best_q0_err" in result:
        m115_data["m115_best_q0_err"] = result["best_q0_err"]

    if "best_hifi_mse" in result:
        m115_data["m115_best_hifi_mse"] = result["best_hifi_mse"]

    # Winner ω = ω of the basin with the lowest hifi_mse (what m115 actually picked)
    basins = result.get("de_results", {}).get("basins", [])
    if basins:
        best = min(basins, key=lambda b: b.get("hifi_mse", float("inf")))
        m115_data["m115_winner_w_dir_err"] = best.get("w_dir_err")

    # Also track ω pool that m115 tested (min w_dir_err across top-K ω candidates it received from m103)
    omegas = result.get("omegas", [])
    if omegas:
        errs = [o.get("w_dir_err") for o in omegas if o.get("w_dir_err") is not None]
        m115_data["m115_tested_pool_min_w_dir_err"] = min(errs) if errs else None
        m115_data["m115_n_omegas_tested"] = len(omegas)

    return m115_data

def extract_m126(seed):
    """Extract m126 data."""
    seed_dir = DIAG_ROOT / "m126_wrapped_m048" / f"seed_{seed:03d}"
    if not seed_dir.exists():
        return {"status": "no_m126"}
    
    result = safe_load_json(seed_dir / "result.json")
    
    best_hifi_wrapped = result.get("best_hifi_wrapped")
    best_hifi_m115 = result.get("best_hifi_m115")
    classification = result.get("classification")
    
    m126_data = {
        "status": "ok",
        "m126_best_hifi_wrapped": best_hifi_wrapped,
        "m126_best_hifi_m115": best_hifi_m115,
        "m126_classification": classification,
        "m126_rho": None,
    }
    
    if best_hifi_wrapped is not None:
        rho = np.sqrt(best_hifi_wrapped / 0.05)
        m126_data["m126_rho"] = rho
    
    return m126_data

def classify_failure_mode(seed, m103, m115, m126):
    """Classify failure mode."""
    # Check m103 status first
    if m103.get("m103_status") == "geo_timeout":
        return "geo_timeout"
    
    if m103.get("m103_status") == "crash":
        return "m103_crash"
    
    # Check m126 OK/PARTIAL
    if m126.get("status") == "ok":
        class_val = m126.get("m126_classification")
        if class_val == "OK":
            return "OK"
        elif class_val == "PARTIAL":
            return "PARTIAL"
    
    # The m115 "winner" is the ω of its best-hifi basin. This is what actually drives m126.
    # Pool miss = even m115's tested top-K lacks near-truth ω.
    tested_pool_min = m115.get("m115_tested_pool_min_w_dir_err")
    winner_err = m115.get("m115_winner_w_dir_err")
    q0_err = m115.get("m115_best_q0_err")

    # Subdivide m103 pool misses: full-pool sampling vs top-K ranking
    full_pool_min = m103.get("m103_pool_min_w_dir_err")
    if tested_pool_min is not None and tested_pool_min > 20.0:
        if full_pool_min is not None and full_pool_min <= 20.0:
            return "m103_topK_ranking_miss"   # surrogate-rescuable
        return "m103_full_pool_miss"          # needs denser m103 sampling

    # m115_ω_selection_miss: pool has a truth-close ω, but m115's winner basin uses a far ω
    if tested_pool_min is not None and tested_pool_min <= 20.0:
        if winner_err is not None and winner_err > 20.0:
            return "m115_ω_selection_miss"

    # q0_polish_miss: m115 picked a truth-close ω, but q0 is still wrong (twin or wrong basin)
    if winner_err is not None and winner_err <= 20.0:
        if q0_err is not None and q0_err > 20.0:
            return "q0_polish_miss"

    return "UNKNOWN"

def main():
    rows = []
    pool_miss_errors = []
    
    for seed in COHORT_SEEDS:
        m103 = extract_m103(seed)
        m115 = extract_m115(seed)
        m126 = extract_m126(seed)
        
        failure_mode = classify_failure_mode(seed, m103, m115, m126)
        
        # Collect pool_miss errors for histogram
        if failure_mode == "m103_ω_pool_miss":
            pool_min = m103.get("m103_pool_min_w_dir_err")
            if pool_min is not None:
                pool_miss_errors.append(pool_min)
        
        # Build table row
        band = m126.get("m126_classification", "—")
        if band is None:
            band = "—"
        
        pool_min = m103.get("m103_pool_min_w_dir_err")
        pool_min_str = f"{pool_min:.1f}" if pool_min is not None else "—"

        tested_pool_min = m115.get("m115_tested_pool_min_w_dir_err")
        tested_pool_str = f"{tested_pool_min:.1f}" if tested_pool_min is not None else "—"

        winner_err = m103.get("m103_winner_w_dir_err")
        m103_winner_str = f"{winner_err:.1f}" if winner_err is not None else "—"

        m115_winner_err = m115.get("m115_winner_w_dir_err")
        m115_winner_str = f"{m115_winner_err:.1f}" if m115_winner_err is not None else "—"
        
        m115_q0 = m115.get("m115_best_q0_err")
        m115_q0_str = f"{m115_q0:.1f}" if m115_q0 is not None else "—"
        
        rho = m126.get("m126_rho")
        rho_str = f"{rho:.2f}" if rho is not None else "—"
        
        rows.append({
            "seed": seed,
            "band": band,
            "m103_pool_min": pool_min_str,
            "m115_pool_min": tested_pool_str,
            "m115_winner_ω_err": m115_winner_str,
            "m115_best_q0_err": m115_q0_str,
            "m126_ρ": rho_str,
            "m126_class": band,
            "failure_mode": failure_mode,
        })

    # Print table
    print("| seed | m103_pool_min_ω | m115_topK_min_ω | m115_winner_ω | m115_q0_err | m126_ρ | class | failure_mode |")
    print("|------|-----------------|-----------------|---------------|-------------|--------|-------|--------------|")
    for row in rows:
        print(f"| {row['seed']} | {row['m103_pool_min']} | {row['m115_pool_min']} | {row['m115_winner_ω_err']} | {row['m115_best_q0_err']} | {row['m126_ρ']} | {row['m126_class']} | {row['failure_mode']} |")
    
    # Diagnostics
    print("\n## Count by failure_mode")
    from collections import Counter
    counts = Counter(row["failure_mode"] for row in rows)
    for mode, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"- {mode}: {count}")

    # Ranking vs sampling split for m103 pool misses
    print("\n## m103 pool-miss breakdown (when m115 top-K missed truth)")
    rank_seeds = [r["seed"] for r in rows if r["failure_mode"] == "m103_topK_ranking_miss"]
    samp_seeds = [r["seed"] for r in rows if r["failure_mode"] == "m103_full_pool_miss"]
    print(f"- ranking failure (truth in m103 full pool but not in top-3 by geo_cost) — {len(rank_seeds)} seeds: {rank_seeds}")
    print(f"- sampling failure (truth absent from m103 full pool entirely)           — {len(samp_seeds)} seeds: {samp_seeds}")
    
    print("\n## m103_ω_pool_miss distribution (by error bins)")
    if pool_miss_errors:
        print(f"Total seeds with m103_ω_pool_miss: {len(pool_miss_errors)}")
        bin_20_45 = sum(1 for e in pool_miss_errors if 20 <= e < 45)
        bin_45_90 = sum(1 for e in pool_miss_errors if 45 <= e < 90)
        bin_90_180 = sum(1 for e in pool_miss_errors if 90 <= e <= 180)
        if bin_20_45 > 0:
            print(f"- [20–45°]: {bin_20_45} seeds (mean: {np.mean([e for e in pool_miss_errors if 20 <= e < 45]):.1f}°)")
        else:
            print(f"- [20–45°]: 0 seeds")
        if bin_45_90 > 0:
            print(f"- [45–90°]: {bin_45_90} seeds (mean: {np.mean([e for e in pool_miss_errors if 45 <= e < 90]):.1f}°)")
        else:
            print(f"- [45–90°]: 0 seeds")
        if bin_90_180 > 0:
            print(f"- [90–180°]: {bin_90_180} seeds (mean: {np.mean([e for e in pool_miss_errors if 90 <= e <= 180]):.1f}°)")
        else:
            print(f"- [90–180°]: 0 seeds")
        print(f"Overall mean for m103_ω_pool_miss: {np.mean(pool_miss_errors):.1f}°")
    else:
        print("No m103_ω_pool_miss seeds found.")

if __name__ == "__main__":
    main()
