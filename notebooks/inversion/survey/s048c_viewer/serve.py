"""Flask app: localhost C_t-cloud config form + dispatcher + viewer.

Run with:
    python -m s048c_viewer.serve [--port 5048] [--host 127.0.0.1]

Routes:
    GET  /                       landing (form + cached-runs table)
    POST /run                    parse form, dispatch (or cache-hit redirect)
    GET  /progress/<jid>         HTML poll page
    GET  /progress/<jid>?json=1  JSON status
    GET  /viewer/<cfg_id>        send animation.html from results dir
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure BLAS doesn't oversubscribe the worker thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Make `survey/` and `lib/` importable when invoked as `python -m s048c_viewer.serve`.
_SURVEY_DIR = Path(__file__).resolve().parent.parent
if str(_SURVEY_DIR) not in sys.path:
    sys.path.insert(0, str(_SURVEY_DIR))

from flask import (  # noqa: E402
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from s048c_viewer import cache, compute, jobs, parser  # noqa: E402
from s048c_viewer.render import render_animation  # noqa: E402
from lib.traj_load import list_seeds, load_truth  # noqa: E402

VIEWER_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(VIEWER_DIR / "templates"),
    static_folder=str(VIEWER_DIR / "static"),
)

EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ct-compute")

# Guardrails (lift later if needed).
MAX_N_SAMPLES = 500_000
MAX_N_EPOCHS = 1000


def _run_compute_and_render(job_id: str, config: dict) -> None:
    """Worker: compute_c_t → render_animation. Updates JOBS in place."""
    jobs.mark_running(job_id)

    def cb(frac: float, message: str) -> None:
        # Reserve last 3% for render.
        scaled = min(0.97, frac * 0.97)
        jobs.update_progress(job_id, scaled, message)

    try:
        npz_path = compute.compute_c_t(progress_cb=cb, **config)
        meta_path = npz_path.parent / "meta.json"
        jobs.update_progress(job_id, 0.97, "rendering animation")
        render_animation(npz_path, meta_path)
        jobs.mark_done(job_id, config_id=npz_path.parent.name)
    except Exception as e:
        traceback.print_exc()
        jobs.mark_error(job_id, repr(e))


# ── Routes ──────────────────────────────────────────────────────────


@app.route("/")
def landing():
    return render_template(
        "landing.html",
        runs=cache.list_runs(),
        seeds=list_seeds(),
        defaults=dict(
            seed=89, surrogate="v1", epochs="all",
            n_samples=100_000, tolerance_mag=0.10,
        ),
    )


@app.route("/run", methods=["POST"])
def run():
    form = request.form
    try:
        seed = int(form["seed"])
        surrogate = form["surrogate"].strip().lower()
        epoch_spec = form.get("epochs", "all").strip() or "all"
        n_samples = int(form["n_samples"])
        tolerance_mag = float(form["tolerance_mag"])
        sample_seed = int(form.get("sample_seed", 42))

        if surrogate not in ("v1", "v2"):
            raise ValueError(f"surrogate must be v1 or v2 (got {surrogate!r})")
        if not (0 <= seed <= 999):
            raise ValueError(f"seed out of range (got {seed})")
        if n_samples < 1000 or n_samples > MAX_N_SAMPLES:
            raise ValueError(
                f"n_samples must be in [1000, {MAX_N_SAMPLES}] (got {n_samples})"
            )
        if not (1e-4 <= tolerance_mag <= 5.0):
            raise ValueError(
                f"tolerance_mag must be in [1e-4, 5.0] mag (got {tolerance_mag})"
            )

        if seed not in set(list_seeds()):
            raise ValueError(f"no trajectory NPZ for seed {seed}")

        traj = load_truth(seed)
        n_obs = int(traj["mag_hifi"].shape[0])
        epoch_indices = parser.parse_epoch_spec(epoch_spec, n_obs)
        if epoch_indices.size > MAX_N_EPOCHS:
            raise ValueError(
                f"too many epochs: {epoch_indices.size} > {MAX_N_EPOCHS}"
            )
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    config_id = cache.make_config_id(
        seed, surrogate, n_samples, sample_seed, tolerance_mag, epoch_indices
    )
    existing = cache.find_run(config_id)
    if existing is not None and (existing / "animation.html").exists():
        return jsonify({"cached": True, "redirect": f"/viewer/{config_id}"})

    job_id = uuid.uuid4().hex[:8]
    jobs.register(job_id, seed=seed, config_id=config_id)

    config = dict(
        seed=seed, surrogate=surrogate, n_samples=n_samples,
        sample_seed=sample_seed, tolerance_mag=tolerance_mag,
        epoch_indices=epoch_indices, epoch_spec_str=epoch_spec,
    )
    EXEC.submit(_run_compute_and_render, job_id, config)
    return jsonify(
        {"cached": False, "job_id": job_id, "redirect": f"/progress/{job_id}"}
    )


@app.route("/progress/<job_id>")
def progress(job_id: str):
    if job_id not in jobs.JOBS:
        if request.args.get("json") == "1":
            return jsonify({"error": "no such job"}), 404
        abort(404)
    if request.args.get("json") == "1":
        return jsonify(jobs.JOBS[job_id])
    return render_template(
        "progress.html", job_id=job_id, job=jobs.JOBS[job_id]
    )


@app.route("/viewer/<config_id>")
def viewer(config_id: str):
    run_dir = cache.find_run(config_id)
    if run_dir is None or not (run_dir / "animation.html").exists():
        abort(404)
    return send_from_directory(run_dir, "animation.html")


# ── Entrypoint ──────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5048)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--no-preload", action="store_true",
        help="skip preloading the v1 surrogate at startup",
    )
    args = ap.parse_args()

    if not args.no_preload:
        print("[startup] preloading v1 surrogate (slow init, ~10 s)...")
        compute.get_compute_model("v1")

    print(f"[startup] serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
