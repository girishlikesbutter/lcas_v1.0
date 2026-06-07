#!/usr/bin/env python3
"""Backfill: re-key Tool cards' output ports from flat type-list -> keyed {return-key: type}.

ADR-0007 'recognised materials only'. The flat list `output:['ia-cloud']` describes a type but
doesn't bind it to a return value, so `loop/artifacts.reify` cannot shelve it. The keyed map
`output:{'q_pool_wxyz':'ia-cloud'}` binds each return-key (or leading tuple-position name) to its
glossary type. Idempotent: only sets ports.output for the cards in REKEY; leaves everything else.

Mappings were derived by reading each entry_point's `return` statement (5-agent scour, 2026-06-07)
and CORRECTED for direction + serialisability. Re-keyed = tools whose return is a (possibly
nested) ndarray / scalar / dict / tuple-position / RESULT DATACLASS — blob-serialisable by
reify._save_blob (Slice 1.5, 2026-06-08, added: asdict() for result dataclasses, 'name@<idx>'
keys for non-leading tuple positions, whole-list reify for homogeneous result lists).

Slice 1.5b (2026-06-08) added the file-reference variant: the 3 SPICE kernel-file producers now
reify by REFERENCE (the card's path points at the tool-written kernel; no blob copy) and moved
into REKEY. DEFERRED now holds only the DECLINED set (the return is not a produced material — a
Satellite fixture, the SpiceHandler class, the oracle load_truth, a path resolver, or a figure).

    python research_os/backfill/rekey_ports_adr0007.py            # apply
    python research_os/backfill/rekey_ports_adr0007.py --dry-run   # preview only
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
SUBSTRATE = os.path.join(ROOT, "substrate")

# id -> keyed output map {return-key / leading-tuple-position name: glossary_term id}
REKEY = {
    # --- survey/lib inversion bench (high-value; run on the bench soonest) ---
    "alignment_filter": {"return": "filter"},                       # return hits/float(...)  (scalar)
    "geo_filter": {"return": "filter"},                             # return hits/float(...)  (scalar)
    "glint_filter": {"f1": "filter"},                              # return {... 'f1': float ...}
    "filter_evaluate_candidate": {"score_alignment": "filter", "mag_pred": "light-curve"},
    "survive_at_epoch": {"pred": "light-curve", "keep": "filter"},  # return pred, keep
    "scorer": {"return": "scoring"},                               # full_lc_mse -> float (rho-band dropped: not produced here)
    "scorer_nll": {"return": "scoring"},                          # nll_cost -> float
    "objective_function": {"return": "scoring"},                  # ObjectiveFunction.evaluate -> float chi_sq
    "surrogate": {"return": "light-curve"},                       # predict -> (N,) mags
    "lc_feature_extractor": {"feats": "lc-feature-vector"},        # return feats, names
    "ls_bracket": {"return": "multi-omegadir-start"},             # return np.geomspace(...)
    "render_hifi": {"return": "hifi-lightcurve"},                 # return np.asarray(pred_mags)
    "rho_band_classify": {"return": "rho-band"},                  # rho_from_hifi -> float
    "omega_jacobi": {"omega_body_hist": "attitude-trajectory"},    # return omega_body_hist, info
    "propagator_jacobi_path2": {"q_body_hist": "attitude-trajectory", "omega_body_hist": "attitude-trajectory"},
    "shoot": {"omega": "q0-omega-state", "residual_norm": "hard-shoot-trap"},
    "shoot_multianchor": {"omega": "q0-omega-state", "residual_norm": "hard-shoot-trap"},
    "shoot_multianchor_freebase": {"omega": "q0-omega-state", "residual_norm": "hard-shoot-trap"},
    # --- src/ forward-model (serialisable array/dict returns) ---
    "propagator": {"quaternions": "attitude-trajectory", "omega_history": "attitude-trajectory"},
    "compute_observation_geometry": {"geometry_data": "attitude-trajectory"},   # whole dict is the material
    "compute_shadows": {"lit_status_dict": "lit-status"},                       # whole dict
    "interpolate_attitudes": {"sat_att_matrices": "attitude-trajectory"},       # (N,3,3) array
    "rotation_matrices_from_angles": {"component_matrices": "component-rotation-matrices"},  # whole dict
    "angle_interpolator": {"angles": "component-angles"},                       # array
    "articulation_angles_from_behaviors": {"component_angles": "component-angles"},  # whole dict
    "compute_fisher_uncertainty": {"covariance": "parameter-uncertainty", "std_devs": "parameter-uncertainty"},
    "compute_mcmc_uncertainty": {"return": "parameter-uncertainty"},           # whole dict (all arrays)
    "generate_lightcurves": {"magnitudes_shadowed": "hifi-lightcurve"},         # tuple position 0
    # --- Slice 1.5: result-dataclass returns (now reifiable via reify's asdict path) ---
    "compute_inertia": {"return": "inertia-result"},          # InertiaResult dataclass
    "global_optimize": {"return": "candidate-set"},           # single OptimizationResult
    "local_refine": {"return": "candidate-set"},              # single OptimizationResult
    "multi_start_optimize": {"return": "candidate-set"},      # list[OptimizationResult] -> one set
    "invert_lightcurve": {"return": "inversion-result"},      # InversionResult dataclass
    "invert_lightcurve_multifidelity": {"return": "inversion-result"},
    # --- Slice 1.5: non-leading tuple position (return k1, k2, quats) ---
    "propagate_to_body_frame": {"quats@2": "attitude-trajectory"},  # quats is rv[2], skips k1/k2
    # --- Slice 1.5b: file-reference materials (return is a Path to a tool-written kernel FILE,
    #     reified by REFERENCE not blob-copy — reify._reference_material) ---
    "spice_kernel_generate": {"spk_path": "spk-kernel"},        # create_ephemeris_from_tle -> Path (SPK)
    "spice_observer_kernel": {"spk_path": "spk-kernel"},        # create_observer_kernel -> Path (SPK)
    "spice_orientation_kernel": {"ck_path": "ck-kernel", "sclk_path": "sclk-kernel"},  # -> (CK, SCLK) tuple
}

# id -> reason. Two classes (Slice 1.5, 2026-06-08):
#   STAGED  = a clean material exists but reify needs a not-yet-built mechanism (the file-reference
#             artifact variant). Re-key these once that lands (Slice 1.5b / Mechanism 2).
#   DECLINED = the return is not a produced material (a fixture, infrastructure, the oracle, or a
#              figure) — reifying it would be wrong, not just unimplemented. Left legacy on purpose.
DEFER = {
    # STAGED set RESOLVED by Slice 1.5b (2026-06-08): the 3 SPICE kernel-file producers moved into
    # REKEY above — reify._reference_material shelves them as storage='reference' (path references
    # the tool-written file, no blob copy). DEFER now holds only the DECLINED set.
    # --- DECLINED: not a produced material ---
    "stl_load_satellite": "DECLINED — returns a Satellite FIXTURE rebuilt from config+STL, not a produced material",
    "spice_handler": "DECLINED — entry_point is the SpiceHandler CLASS (infrastructure), no material return",
    "load_truth": "DECLINED — the ORACLE; returns the truth NPZ (re-feeding it would taint a run), not shelf material",
    "lc_resolve": "DECLINED — a path RESOLVER (returns a Path to an existing LC), not a producer",
    "create_3d_animation": "DECLINED — writes an HTML FIGURE via a path arg; belongs to artefacts/plot-stream",
    "lc_compare_plot": "DECLINED — saves a PNG FIGURE, returns None; not a return material",
    "lightcurve_plot": "DECLINED — saves a PNG FIGURE, returns plot_time (a metric); not a material",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    changed = skipped = 0
    for cid, omap in REKEY.items():
        fp = os.path.join(SUBSTRATE, f"{cid}.json")
        if not os.path.isfile(fp):
            print(f"  MISSING {cid} (no {os.path.relpath(fp, ROOT)})")
            continue
        card = json.load(open(fp))
        ports = card.get("ports") or {}
        if ports.get("output") == omap:
            skipped += 1
            continue
        ports["output"] = omap
        card["ports"] = ports
        if not args.dry_run:
            with open(fp, "w") as f:
                json.dump(card, f, indent=2)
                f.write("\n")
        changed += 1
        print(f"  {'would re-key' if args.dry_run else 're-keyed'} {cid:<34} -> {omap}")
    print(f"\n{'DRY-RUN ' if args.dry_run else ''}re-keyed {changed}, already-current {skipped}, "
          f"deferred {len(DEFER)} (see DEFER in this file).")


if __name__ == "__main__":
    main()
