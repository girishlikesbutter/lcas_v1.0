#!/usr/bin/env python3
"""Single-entry inversion driver — runs the full pipeline for one seed.

Chains: m103 geo harvest (if geo_ckpt missing)  →  m115 surrogate-DE
        →  m126 wrapped polish  →  wrappedbest refresh  →  lc_compare plot.

Usage
-----
    python3 notebooks/inversion/invert.py --seed 42 --traj-source m048
    python3 notebooks/inversion/invert.py --seed 0  --traj-source m046

    # Skip stages already run (checkpoint-aware):
    python3 notebooks/inversion/invert.py --seed 42 --skip-lc-compare

Each stage runs as a subprocess so env-var plumbing stays self-contained.
Stages read TRAJ_SOURCE, MICRO103_SEED, and MICRO{115,126}_SEEDS from the
environment; this driver sets those before invoking the child scripts.

The m103 harvest stage runs with MICRO103_SKIP_HIFI=1 (emit geo_ckpt.npz
only; no Step 5 hi-fi validation), saving ~8 min/seed — m115 does its own
hi-fi validation on the basins it selects. The m103 stage auto-skips if a
valid geo_ckpt.npz already exists on disk; use --force-m103 to override.

Outputs land in the per-stage result dirs:
    m103_hybrid{_m048?}/seed_NNN/
    m115_surrogate_pipeline{_m048?}/seed_NNN/
    m126_wrapped{_m048?}/seed_NNN/
    wrappedbest_{source}_seed{NNN}/
    wrappedbest_{source}_seed{NNN}_lc_compare.png

A summary JSON is written at:
    invert_{source}_seed{NNN}/result.json
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.traj_source import VALID_SOURCES, load_truth  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
M103_SCRIPT = PROJECT_ROOT / "notebooks" / "inversion" / "11_casadi_formulation" / "m103_hybrid.py"
M115_SCRIPT = PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface" / "m115_surrogate_pipeline.py"
M126_SCRIPT = PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface" / "m126_wrapped_pipeline.py"
LC_COMPARE = PROJECT_ROOT / "notebooks" / "inversion" / "lib" / "lc_compare.py"


def m103_out_base(source):
    return RESULTS_DIR / ("m103_hybrid" if source == 'm046'
                          else f"m103_hybrid_{source}")


def m115_out_base(source):
    return RESULTS_DIR / ("m115_surrogate_pipeline" if source == 'm046'
                          else f"m115_surrogate_pipeline_{source}")


def m126_out_base(source):
    return RESULTS_DIR / ("m126_wrapped" if source == 'm046'
                          else f"m126_wrapped_{source}")


def wrappedbest_dir(seed, source):
    # Keep the legacy layout for m046 so existing tooling (m131) continues to
    # find wrappedbest_seed{NNN}/. Tag for non-m046 sources.
    if source == 'm046':
        return RESULTS_DIR / f"wrappedbest_seed{seed:03d}"
    return RESULTS_DIR / f"wrappedbest_{source}_seed{seed:03d}"


def wrappedbest_prefix(source):
    return "wrappedbest" if source == 'm046' else f"wrappedbest_{source}"


def invert_dir(seed, source):
    tag = "m046" if source == 'm046' else source
    return RESULTS_DIR / f"invert_{tag}_seed{seed:03d}"


def run_stage(name, script, env, check=True):
    print(f"\n{'=' * 72}\n[STAGE] {name}\n{'=' * 72}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    print(f"[STAGE] {name} -> rc={result.returncode} ({dt:.1f}s)")
    if check and result.returncode != 0:
        raise RuntimeError(f"stage {name} failed (rc={result.returncode})")
    return dt


def build_wrappedbest_single(seed, source):
    """Build a single-seed wrappedbest dir from the m126 output.

    Mirrors the logic in m131_refresh_wrappedbest.refresh_m126_seed, but for
    one seed and driven from the post-fix m126 output directly (no m124/m125
    fallback — m048 has no equivalent prior work).
    """
    seed_dir = m126_out_base(source) / f"seed_{seed:03d}"
    result_json = seed_dir / "result.json"
    hifi_ckpt = seed_dir / "hifi_ckpt.npz"
    if not result_json.exists():
        raise FileNotFoundError(f"m126 result missing: {result_json}")
    if not hifi_ckpt.exists():
        raise FileNotFoundError(f"m126 hifi_ckpt missing: {hifi_ckpt}")

    res = json.load(open(result_json))
    basins = res["basins"]
    winner_idx = int(np.argmin([b["hifi_wrapped"] for b in basins]))
    b = basins[winner_idx]
    use_after = b["hifi_after"] <= b["hifi_before"]
    q0_src = np.asarray(b["q0_after"] if use_after else b["q0_before"])
    w0_src = np.asarray(b["omega_after"] if use_after else b["omega_before"])

    ckpt = np.load(hifi_ckpt, allow_pickle=True)
    idx = int(np.where(ckpt["basin_idx"] == winner_idx)[0][0])
    key = "hifi_mags_after" if use_after else "hifi_mags_before"
    if key not in ckpt.files:
        # m126 only saves hifi_mags_after currently; fall back when polish worsened hifi.
        key = "hifi_mags_after"
    pred_lc = ckpt[key][idx]
    assert pred_lc.shape == (500,)

    # Compute error metrics vs truth
    truth = load_truth(seed, source)
    true_q0 = truth['q0_wxyz'] / np.linalg.norm(truth['q0_wxyz'])
    true_w0 = truth['omega0_rad']

    def quat_angle_deg(q1, q2):
        q1 = np.asarray(q1) / np.linalg.norm(q1)
        q2 = np.asarray(q2) / np.linalg.norm(q2)
        dot = min(1.0, max(-1.0, abs(float(np.dot(q1, q2)))))
        return 2.0 * np.degrees(np.arccos(dot))

    def vec_angle_deg(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-15 or n2 < 1e-15:
            return float('nan')
        cos = min(1.0, max(-1.0, float(np.dot(v1, v2) / (n1 * n2))))
        return np.degrees(np.arccos(cos))

    q0_err = quat_angle_deg(q0_src, true_q0)
    w_dir_err = vec_angle_deg(w0_src, true_w0)
    mag_est = float(np.linalg.norm(w0_src))
    mag_true = float(np.linalg.norm(true_w0))
    w_mag_err_pct = 100.0 * (mag_est - mag_true) / mag_true

    winner = {
        "q0_wxyz": q0_src.tolist(),
        "w0_rad": w0_src.tolist(),
        "w0_dps": np.rad2deg(w0_src).tolist(),
        "q0_err": q0_err,
        "w0_err": w_dir_err,
        "w_mag_err_pct": w_mag_err_pct,
        "hifi": float(b["hifi_wrapped"]),
        "source_label": f"m126_basin{winner_idx}_{'after' if use_after else 'before'}",
    }
    hifi_val = winner["hifi"]
    if hifi_val < 0.01:
        classification = "OK"
    elif hifi_val < 0.1:
        classification = "PARTIAL"
    else:
        classification = "FAIL"

    out_dir = wrappedbest_dir(seed, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": f"wrappedbest (invert.py {source})",
        "seed": seed,
        "traj_source": source,
        "classification": classification,
        "winner": winner,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(payload, f, indent=2)
    np.save(str(out_dir / "pred_lc.npy"), pred_lc.astype(np.float64))

    print(f"[wrappedbest] seed={seed} {classification} hifi={hifi_val:.5f} "
          f"q0={q0_err:.2f}° w_dir={w_dir_err:.2f}° "
          f"w_mag={w_mag_err_pct:+.2f}%")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True,
                    help="Trajectory seed (0..99)")
    ap.add_argument("--traj-source", default="m046", choices=VALID_SOURCES,
                    help="'m046' (legacy single-window) or 'm048' (per-seed)")
    ap.add_argument("--skip-m103", action="store_true",
                    help="Skip the upstream m103 harvest even if geo_ckpt is missing")
    ap.add_argument("--force-m103", action="store_true",
                    help="Force rerun of the m103 harvest even if geo_ckpt exists")
    ap.add_argument("--skip-m115", action="store_true",
                    help="Skip m115 (assume its checkpoints already exist)")
    ap.add_argument("--skip-m126", action="store_true",
                    help="Skip m126 (assume its checkpoints already exist)")
    ap.add_argument("--skip-lc-compare", action="store_true",
                    help="Skip the lc_compare plot at the end")
    args = ap.parse_args()

    seed = args.seed
    source = args.traj_source

    t_global = time.time()
    print(f"\n{'#' * 72}")
    print(f"# invert.py seed={seed} source={source}")
    print(f"{'#' * 72}")

    # Summary dir
    inv_dir = invert_dir(seed, source)
    inv_dir.mkdir(parents=True, exist_ok=True)

    # Env setup — child scripts read TRAJ_SOURCE + MICRO{103,115,126}_SEED{S}.
    env = os.environ.copy()
    env["TRAJ_SOURCE"] = source
    env["MICRO103_SEED"] = str(seed)
    env["MICRO115_SEEDS"] = str(seed)
    env["MICRO126_SEEDS"] = str(seed)

    timing = {}

    # m103: grid + NM + multi-phi + geo harvest (skips Step 5 hi-fi) ----------
    # Produces geo_ckpt.npz which m115.load_omega_candidates consumes. Auto-
    # skips if geo_ckpt already exists unless --force-m103.
    m103_seed_dir = m103_out_base(source) / f"seed_{seed:03d}"
    geo_ckpt = m103_seed_dir / "geo_ckpt.npz"
    geo_timeout_flag = m103_seed_dir / "geo_timeout.flag"
    if args.skip_m103:
        print(f"[stage] m103 skipped (--skip-m103)")
        timing["m103_s"] = 0.0
    elif geo_ckpt.exists() and not args.force_m103:
        print(f"[stage] m103 skipped (geo_ckpt present: {geo_ckpt})")
        timing["m103_s"] = 0.0
    else:
        m103_env = dict(env)
        m103_env["MICRO103_SKIP_HIFI"] = "1"  # harvest mode — emit geo_ckpt only
        timing["m103_s"] = run_stage("m103 hybrid (harvest, SKIP_HIFI=1)",
                                     M103_SCRIPT, m103_env)

        # Graceful m103 geo-timeout handoff: m103 writes geo_timeout.flag and
        # exits rc=0. We skip m115/m126/wrappedbest/lc_compare for this seed
        # and emit an invert-level summary tagged status=geo_timeout.
        if geo_timeout_flag.exists() and not geo_ckpt.exists():
            msg = geo_timeout_flag.read_text().strip()
            print(f"[stage] m103 geo TIMEOUT — skipping downstream for this seed")
            print(f"        {msg}")
            summary = {
                "driver": "invert.py",
                "seed": seed,
                "traj_source": source,
                "status": "geo_timeout",
                "geo_timeout_note": msg,
                "classification": "GEO_TIMEOUT",
                "winner": None,
                "paths": {
                    "m103_dir": str(m103_seed_dir),
                },
                "timing": timing,
            }
            summary_path = inv_dir / "result.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved: {summary_path}")
            print(f"\nRESULT seed={seed:03d} source={source}: GEO_TIMEOUT")
            print(f"Total wall: {time.time()-t_global:.1f}s")
            return

        if not geo_ckpt.exists():
            raise RuntimeError(
                f"m103 harvest finished but geo_ckpt missing: {geo_ckpt}")

    # m115: surrogate-DE basin search ------------------------------------------
    if args.skip_m115:
        print("[stage] m115 skipped (--skip-m115)")
        timing["m115_s"] = 0.0
    else:
        timing["m115_s"] = run_stage("m115 surrogate pipeline",
                                     M115_SCRIPT, env)

    # m126: wrapped polish + hi-fi --------------------------------------------
    if args.skip_m126:
        print("[stage] m126 skipped (--skip-m126)")
        timing["m126_s"] = 0.0
    else:
        timing["m126_s"] = run_stage("m126 wrapped pipeline",
                                     M126_SCRIPT, env)

    # Wrappedbest refresh (single-seed, in-process — no subprocess) ------------
    print(f"\n{'=' * 72}\n[STAGE] wrappedbest refresh (single-seed)\n{'=' * 72}",
          flush=True)
    t0 = time.time()
    wb_payload = build_wrappedbest_single(seed, source)
    timing["wrappedbest_s"] = time.time() - t0
    print(f"[STAGE] wrappedbest refresh -> ({timing['wrappedbest_s']:.1f}s)")

    # lc_compare ---------------------------------------------------------------
    if args.skip_lc_compare:
        print("[stage] lc_compare skipped (--skip-lc-compare)")
        timing["lc_compare_s"] = 0.0
    else:
        print(f"\n{'=' * 72}\n[STAGE] lc_compare\n{'=' * 72}", flush=True)
        t0 = time.time()
        rc = subprocess.run(
            [sys.executable, str(LC_COMPARE),
             "--prefix", wrappedbest_prefix(source),
             "--traj-source", source,
             str(seed)],
            env=env, cwd=str(PROJECT_ROOT)).returncode
        timing["lc_compare_s"] = time.time() - t0
        print(f"[STAGE] lc_compare -> rc={rc} ({timing['lc_compare_s']:.1f}s)")
        if rc != 0:
            print(f"WARNING: lc_compare returned rc={rc}; continuing")

    timing["total_s"] = time.time() - t_global

    # Save summary ------------------------------------------------------------
    summary = {
        "driver": "invert.py",
        "seed": seed,
        "traj_source": source,
        "winner": wb_payload["winner"],
        "classification": wb_payload["classification"],
        "paths": {
            "m103_dir": str(m103_out_base(source) / f"seed_{seed:03d}"),
            "m115_dir": str(m115_out_base(source) / f"seed_{seed:03d}"),
            "m126_dir": str(m126_out_base(source) / f"seed_{seed:03d}"),
            "wrappedbest_dir": str(wrappedbest_dir(seed, source)),
            "lc_compare_png": str(RESULTS_DIR / f"{wrappedbest_prefix(source)}_seed{seed:03d}_lc_compare.png"),
        },
        "timing": timing,
    }
    summary_path = inv_dir / "result.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Final line -- standard 3-error report (memory: report all three)
    w = wb_payload["winner"]
    cls = wb_payload["classification"]
    print(f"\nRESULT seed={seed:03d} source={source}: "
          f"q0={w['q0_err']:.2f}° w_dir={w['w0_err']:.2f}° "
          f"w_mag={w['w_mag_err_pct']:+.2f}% hifi={w['hifi']:.5f} [{cls}]")
    print(f"Total wall: {timing['total_s']:.1f}s ({timing['total_s']/60:.1f} min)")


if __name__ == "__main__":
    main()
