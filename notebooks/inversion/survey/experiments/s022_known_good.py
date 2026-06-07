"""s022 — Known-good validation of the filter framework.

Score the (alignment, geo) filters on candidates that we already know
are valid solutions or near-truth attractors:

  1. Truth (q0, ω) per seed — should be 1.0/1.0 (already done in s021).
  2. Body-twin (q_180x · q0, R_180x · ω) — should be 1.0/1.0 if BRDF
     is symmetric under the body-twin operation (already done in s021).
  3. s014 multi-solution candidates (15 from seeds 41/48/84, ρ < 4) —
     these are HI-FI-VALIDATED Band A∪B candidates outside the truth
     basin. Filters MUST keep these (else the necessary-condition
     framework is broken — would create false negatives against valid
     solutions).
  4. s011 in-basin LM landings (the density-recoverable cohort tail) —
     should also be 1.0/1.0 modulo numerical noise.

Output:
  results/s022/known_good.npz
  results/s022/summary.json

Wall: ~30 seconds.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib.traj_load import load_truth                  # noqa: E402
from lib import filter_costs as fc                    # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s022"
RESULTS.mkdir(parents=True, exist_ok=True)

ALIGN_BRIGHT_MAG = 11.0
ALIGN_WINDOW_EPOCHS = 3
GEO_THRESHOLD_DEG = 5.0
SPEC_THRESHOLD_DEG = 5.0


def _score(q0, omega, seed_data, static, tier):
    return fc.evaluate_candidate(
        q0, omega, seed_data,
        static["inertia_tensor"], static["face_normals"], tier["tier_face_idx"],
        align_window_epochs=ALIGN_WINDOW_EPOCHS,
        align_bright_mag=ALIGN_BRIGHT_MAG,
        geo_threshold_deg=GEO_THRESHOLD_DEG,
    )


def main():
    print("=" * 72)
    print("s022 — Known-good filter validation")
    print("=" * 72)

    static = fc.load_static_geometry()
    tier = fc.load_tier_table()

    # ---- Load s014 multi-solution + in-basin candidates ----
    s014_npz = SURVEY_DIR / "results" / "s014" / "rho_s011_nb.npz"
    s014 = np.load(s014_npz)
    seeds_s014 = s014["seed"]
    band_s014 = s014["band"]
    rho_s014 = s014["rho"]
    q0_err_s014 = s014["q0_err_deg"]
    q0_final = s014["q0_final_wxyz"]
    omega_final = s014["omega_final_rad"]

    rows = []

    # 3. Multi-solution (band A∪B, q0_err > 10° → outside truth-basin)
    ms_mask = (np.isin(band_s014, ["A", "B"])) & (q0_err_s014 > 10)
    ms_idx = np.where(ms_mask)[0]
    print(f"\nMulti-solution candidates (s014, ρ<4, q0_err>10°): {ms_idx.size}")

    seed_data_cache = {}
    t0 = time.time()
    for i in ms_idx:
        seed = int(seeds_s014[i])
        if seed not in seed_data_cache:
            truth = load_truth(seed)
            seed_data_cache[seed] = fc.precompute_seed_filter_data(
                truth, tier, spec_threshold_deg=SPEC_THRESHOLD_DEG,
                bright_mag_threshold=ALIGN_BRIGHT_MAG,
            )
        sd = seed_data_cache[seed]
        res = _score(q0_final[i], omega_final[i], sd, static, tier)
        rows.append({
            "kind": "multi_solution",
            "seed": seed,
            "rho": float(rho_s014[i]),
            "band": str(band_s014[i]),
            "q0_err_deg": float(q0_err_s014[i]),
            "score_align": float(res["score_alignment"]) if np.isfinite(res["score_alignment"]) else float("nan"),
            "score_geo": float(res["score_geo"]) if np.isfinite(res["score_geo"]) else float("nan"),
        })
        print(f"  seed {seed:3d} ρ={rho_s014[i]:.2f} q0_err={q0_err_s014[i]:.1f}° "
              f"→ align={res['score_alignment']:.3f} geo={res['score_geo']:.3f}")

    # 4. In-basin band A∪B from s014 (the density-recoverable lock-ons)
    inb_mask = (np.isin(band_s014, ["A", "B"])) & (q0_err_s014 <= 10)
    inb_idx = np.where(inb_mask)[0]
    print(f"\nIn-basin candidates (s014, ρ<4, q0_err<=10°): {inb_idx.size}")
    for i in inb_idx:
        seed = int(seeds_s014[i])
        if seed not in seed_data_cache:
            truth = load_truth(seed)
            seed_data_cache[seed] = fc.precompute_seed_filter_data(
                truth, tier, spec_threshold_deg=SPEC_THRESHOLD_DEG,
                bright_mag_threshold=ALIGN_BRIGHT_MAG,
            )
        sd = seed_data_cache[seed]
        res = _score(q0_final[i], omega_final[i], sd, static, tier)
        rows.append({
            "kind": "in_basin",
            "seed": seed,
            "rho": float(rho_s014[i]),
            "band": str(band_s014[i]),
            "q0_err_deg": float(q0_err_s014[i]),
            "score_align": float(res["score_alignment"]) if np.isfinite(res["score_alignment"]) else float("nan"),
            "score_geo": float(res["score_geo"]) if np.isfinite(res["score_geo"]) else float("nan"),
        })
        print(f"  seed {seed:3d} ρ={rho_s014[i]:.2f} q0_err={q0_err_s014[i]:.1f}° "
              f"→ align={res['score_alignment']:.3f} geo={res['score_geo']:.3f}")

    wall = time.time() - t0

    # Summary
    align_ms = np.array([r["score_align"] for r in rows if r["kind"] == "multi_solution"])
    geo_ms = np.array([r["score_geo"] for r in rows if r["kind"] == "multi_solution"])
    align_inb = np.array([r["score_align"] for r in rows if r["kind"] == "in_basin"])
    geo_inb = np.array([r["score_geo"] for r in rows if r["kind"] == "in_basin"])

    summary = {
        "wall_seconds": wall,
        "n_multi_solution": int(ms_idx.size),
        "n_in_basin": int(inb_idx.size),
        "params": {
            "align_bright_mag_threshold": ALIGN_BRIGHT_MAG,
            "align_window_epochs": ALIGN_WINDOW_EPOCHS,
            "geo_threshold_deg": GEO_THRESHOLD_DEG,
            "spec_threshold_deg": SPEC_THRESHOLD_DEG,
        },
        "multi_solution_filter_pass": {
            "align_median": float(np.nanmedian(align_ms)) if align_ms.size else None,
            "align_min": float(np.nanmin(align_ms)) if align_ms.size else None,
            "align_n_above_0.5": int(np.sum(align_ms >= 0.5)),
            "align_n_at_1.0": int(np.sum(align_ms >= 0.999)),
            "geo_median": float(np.nanmedian(geo_ms)) if geo_ms.size else None,
            "geo_n_finite": int(np.sum(np.isfinite(geo_ms))),
            "geo_n_above_0.5": int(np.sum(geo_ms >= 0.5)),
        },
        "in_basin_filter_pass": {
            "align_median": float(np.nanmedian(align_inb)) if align_inb.size else None,
            "align_n_at_1.0": int(np.sum(align_inb >= 0.999)),
            "geo_median": float(np.nanmedian(geo_inb)) if geo_inb.size else None,
            "geo_n_at_1.0": int(np.sum(geo_inb >= 0.999)),
        },
        "rows": rows,
    }

    # Save
    np.savez_compressed(
        RESULTS / "known_good.npz",
        kinds=np.array([r["kind"] for r in rows]),
        seeds=np.array([r["seed"] for r in rows]),
        rhos=np.array([r["rho"] for r in rows]),
        bands=np.array([r["band"] for r in rows]),
        q0_err_deg=np.array([r["q0_err_deg"] for r in rows]),
        score_align=np.array([r["score_align"] for r in rows]),
        score_geo=np.array([r["score_geo"] for r in rows]),
    )
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS / 'known_good.npz'}")
    print(f"Saved: {RESULTS / 'summary.json'}")
    print(f"Wall: {wall:.1f} s")
    print(f"\nSUMMARY:")
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == "__main__":
    main()
