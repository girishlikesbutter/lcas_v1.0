#!/usr/bin/env python3
"""Graceful sequential batch driver for invert.py on m048 seeds.

Per seed:
  1. Skip if `wrappedbest_m048_seed{NNN}/.noise_fix_v1.done` marker present
     (resume-safe).
  2. Wipe `m126_wrapped_m048/seed_NNN/`, `wrappedbest_m048_seed{NNN}/`, and
     `invert_m048_seed{NNN}/` before running (noise-realisation-dependent;
     stale under any pre-fix run). Leave `m103_hybrid_m048/seed_NNN/` and
     `m115_surrogate_pipeline_m048/seed_NNN/` alone (noise-unchanged;
     invert.py reuses the checkpoints).
  3. Run `invert.py --seed N --traj-source m048 --skip-lc-compare` as a
     subprocess. Primary graceful timeout is INSIDE m103 (Step 4 geo
     refinement, env var `MICRO103_GEO_TIMEOUT_S`, default 480s = 8 min) —
     on hit, m103 writes `geo_timeout.flag` and exits 0, invert.py tags the
     seed `status=geo_timeout` and skips m115/m126.
  4. A secondary subprocess-level wall-clock cap (default 1500s = 25 min)
     exists only as a safety net for unexpected hangs outside the geo stage.
     If it fires, SIGKILL the process group and record 'hung'.
  5. Outcome routing:
        - wrappedbest result.json present → 'ok' (+ ρ-band)
        - invert summary with status=geo_timeout → 'geo_timeout'
        - rc!=0 → 'failed'
        - safety-net timeout → 'hung'

Usage:
    python3 notebooks/inversion/run_m048_batch.py --seeds 6,7,8,11 \
        --out-dir data/results/inversion_diagnostics/batch_m048_v1
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
INVERT = PROJECT_ROOT / "notebooks" / "inversion" / "invert.py"

NOISE_FIX_MARKER = ".noise_fix_v1.done"
NOISE_SIGMA = 0.05


def m126_dir(seed):
    return RESULTS_DIR / "m126_wrapped_m048" / f"seed_{seed:03d}"


def wb_dir(seed):
    return RESULTS_DIR / f"wrappedbest_m048_seed{seed:03d}"


def invert_summary_dir(seed):
    return RESULTS_DIR / f"invert_m048_seed{seed:03d}"


def rho_band(mse):
    if mse is None or not math.isfinite(mse):
        return None, None
    rho = math.sqrt(mse) / NOISE_SIGMA
    if rho < 2:
        band = "A"
    elif rho < 4:
        band = "B"
    elif rho < 8:
        band = "C"
    else:
        band = "D"
    return rho, band


def wipe_seed_dirs(seed):
    for p in (m126_dir(seed), wb_dir(seed), invert_summary_dir(seed)):
        if p.exists():
            shutil.rmtree(p)


def read_wb(seed):
    p = wb_dir(seed) / "result.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def run_one_seed(seed, timeout_s, log_f):
    """Return a dict record describing the seed outcome."""
    t0 = time.time()
    wb = wb_dir(seed)
    marker = wb / NOISE_FIX_MARKER

    # Resume-skip
    if marker.exists():
        res = read_wb(seed)
        w = (res or {}).get("winner", {})
        rho, band = rho_band(w.get("hifi"))
        return {
            "seed": seed,
            "status": "skipped_resume",
            "wall_s": 0.0,
            "classification": (res or {}).get("classification"),
            "band": band,
            "rho": rho,
            "hifi": w.get("hifi"),
            "q0_err": w.get("q0_err"),
            "w_dir_err": w.get("w0_err"),
            "w_mag_err_pct": w.get("w_mag_err_pct"),
        }

    wipe_seed_dirs(seed)

    cmd = [sys.executable, str(INVERT),
           "--seed", str(seed), "--traj-source", "m048", "--skip-lc-compare"]
    print(f"\n[batch] seed {seed}: launching, timeout {timeout_s}s", flush=True)
    log_f.write(f"[{time.strftime('%H:%M:%S')}] launching seed {seed}\n")
    log_f.flush()

    # setsid so we can SIGKILL the whole process group (m103 spawns Pool workers)
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), preexec_fn=os.setsid)
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        wall = time.time() - t0
        print(f"[batch] seed {seed}: HUNG after {wall:.1f}s (killed)", flush=True)
        return {
            "seed": seed, "status": "hung",
            "wall_s": wall, "rc": None,
            "reason": f"wall-clock exceeded {timeout_s}s",
        }

    wall = time.time() - t0
    if rc != 0:
        print(f"[batch] seed {seed}: FAILED rc={rc} after {wall:.1f}s", flush=True)
        return {"seed": seed, "status": "failed", "wall_s": wall, "rc": rc}

    # Check for a geo_timeout summary (m103 Step 4 graceful abort).
    inv_summary = invert_summary_dir(seed) / "result.json"
    if inv_summary.exists():
        inv = json.loads(inv_summary.read_text())
        if inv.get("status") == "geo_timeout":
            print(f"[batch] seed {seed}: GEO_TIMEOUT after {wall:.1f}s "
                  f"(graceful, move on)", flush=True)
            return {
                "seed": seed, "status": "geo_timeout",
                "wall_s": wall, "rc": rc,
                "classification": "GEO_TIMEOUT",
                "note": inv.get("geo_timeout_note"),
            }

    res = read_wb(seed)
    if res is None:
        print(f"[batch] seed {seed}: completed but wrappedbest missing", flush=True)
        return {"seed": seed, "status": "no_wrappedbest", "wall_s": wall, "rc": rc}

    # success — write the marker
    marker.write_text(
        f"noise_fix_v1 completed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    w = res.get("winner", {})
    rho, band = rho_band(w.get("hifi"))
    print(f"[batch] seed {seed}: OK [{band}] rho={rho:.2f} "
          f"hifi={w.get('hifi', 0):.5f} q0={w.get('q0_err', 0):.2f}° "
          f"w={w.get('w0_err', 0):.2f}° wall={wall:.1f}s", flush=True)
    return {
        "seed": seed,
        "status": "ok",
        "wall_s": wall,
        "rc": rc,
        "classification": res.get("classification"),
        "band": band,
        "rho": rho,
        "hifi": w.get("hifi"),
        "q0_err": w.get("q0_err"),
        "w_dir_err": w.get("w0_err"),
        "w_mag_err_pct": w.get("w_mag_err_pct"),
    }


def summarise(records):
    bands = {"A": [], "B": [], "C": [], "D": []}
    hung, failed, skipped, no_wb, geo_to = [], [], [], [], []
    for r in records:
        s = r["status"]
        if s == "hung":
            hung.append(r["seed"])
        elif s == "failed":
            failed.append(r["seed"])
        elif s == "no_wrappedbest":
            no_wb.append(r["seed"])
        elif s == "geo_timeout":
            geo_to.append(r["seed"])
        elif s == "skipped_resume":
            skipped.append(r["seed"])
            if r.get("band") in bands:
                bands[r["band"]].append(r["seed"])
        elif s == "ok":
            if r.get("band") in bands:
                bands[r["band"]].append(r["seed"])
    return {
        "n_total": len(records),
        "n_band_A": len(bands["A"]),
        "n_band_B": len(bands["B"]),
        "n_band_C": len(bands["C"]),
        "n_band_D": len(bands["D"]),
        "n_geo_timeout": len(geo_to),
        "n_hung": len(hung),
        "n_failed": len(failed),
        "n_no_wrappedbest": len(no_wb),
        "n_skipped_resume": len(skipped),
        "seeds_A": bands["A"],
        "seeds_B": bands["B"],
        "seeds_C": bands["C"],
        "seeds_D": bands["D"],
        "seeds_geo_timeout": geo_to,
        "seeds_hung": hung,
        "seeds_failed": failed,
        "seeds_no_wrappedbest": no_wb,
        "seeds_skipped_resume": skipped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True,
                    help="Comma-separated seed list, e.g. '28,49,69' or a path to "
                         "a newline-separated seeds file.")
    ap.add_argument("--timeout-s", type=int, default=1500,
                    help="Safety-net per-seed wall-clock cap in seconds "
                         "(default 1500 = 25 min). The real graceful timeout "
                         "is inside m103's Step 4 — env var "
                         "MICRO103_GEO_TIMEOUT_S, default 480s = 8 min.")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "batch_m048_v1"),
                    help="Directory for batch_log.jsonl and batch_summary.json")
    ap.add_argument("--tag", default=None,
                    help="Optional tag string written into summary metadata.")
    args = ap.parse_args()

    # Parse seeds
    if Path(args.seeds).exists():
        seeds = [int(x) for x in Path(args.seeds).read_text().split()
                 if x.strip() and not x.strip().startswith("#")]
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "batch_log.jsonl"
    txt_log_path = out_dir / "batch.log"
    summary_path = out_dir / "batch_summary.json"

    print(f"\n{'#' * 72}")
    print(f"# run_m048_batch: {len(seeds)} seeds, timeout {args.timeout_s}s each")
    print(f"# seeds = {seeds}")
    print(f"# out_dir = {out_dir}")
    print(f"{'#' * 72}", flush=True)

    records = []
    t_batch = time.time()
    with open(log_path, "a") as jsonl_f, open(txt_log_path, "a") as txt_f:
        txt_f.write(f"=== batch start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        txt_f.flush()
        for seed in seeds:
            rec = run_one_seed(seed, args.timeout_s, txt_f)
            records.append(rec)
            jsonl_f.write(json.dumps(rec) + "\n")
            jsonl_f.flush()
        txt_f.write(f"=== batch end {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    summary = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s_total": time.time() - t_batch,
        "timeout_s": args.timeout_s,
        "seeds_input": seeds,
        **summarise(records),
        "per_seed": records,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#' * 72}")
    print(f"# batch done  wall={summary['wall_s_total']:.0f}s "
          f"({summary['wall_s_total']/3600:.2f}h)")
    print(f"# A={summary['n_band_A']} B={summary['n_band_B']} "
          f"C={summary['n_band_C']} D={summary['n_band_D']} "
          f"geo_timeout={summary['n_geo_timeout']} "
          f"hung={summary['n_hung']} failed={summary['n_failed']} "
          f"resume-skipped={summary['n_skipped_resume']}")
    print(f"# summary: {summary_path}")
    print(f"# log:     {log_path}")
    print(f"{'#' * 72}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
