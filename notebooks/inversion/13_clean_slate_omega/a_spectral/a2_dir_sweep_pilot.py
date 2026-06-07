"""Stage 2: spectral-anchored ω_direction sweep pilot.

Pilot on 5 representative seeds spanning the phase/omega distribution:
    seed 0  (|ω|=1.449, phase 27.7°)  — fast, low-phase
    seed 49 (|ω|=1.110, phase 38.9°)  — mid
    seed 69 (|ω|=1.126, phase 67.7°)  — high-phase
    seed 23 (|ω|=0.965, phase 30.4°)  — mid
    seed 81 (|ω|=1.176, phase 45.6°)  — mid

Strategy:
    - Build a Fibonacci S² grid of 2562 candidate ω-direction unit vectors.
    - Assume |ω| = |ω_true| (this is the "direction-only" test).
    - Oracle q0 variant: use q0_true, score each direction by residual MSE
      against hi-fi LC.
    - Joint q0 diagnostic: run only on seed 0 with a 642-direction grid and
      a 24-point SO(3) quaternion grid; keep the best q0 per direction.
    - Save ALL candidates under RESIDUAL_MSE_GATE for multi-solution analysis.

Outputs (paths printed): per-seed NPZ + JSON + one S² globe HTML.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Keep BLAS single-threaded per-worker so Pool parallelism actually scales.
# Without this, each worker spawns 16 BLAS threads and they contend.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion" / "13_clean_slate_omega"))

from lib.data import load_seed  # noqa: E402
from lib.forward import predict_lc  # noqa: E402
from lib.scoring import (  # noqa: E402
    RESIDUAL_MSE_GATE,
    RESIDUAL_MSE_TIGHT,
    lc_mse,
    omega_errors,
)


OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "inversion_diagnostics"
    / "13_clean_slate_omega"
    / "a_spectral"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- fibonacci sphere ------------------------------------------------------


def fibonacci_sphere(n: int) -> np.ndarray:
    """n unit vectors approximately equidistributed on S²."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def so3_grid(n: int = 24) -> np.ndarray:
    """n quaternions approximately spanning SO(3) via Fibonacci S² directions
    × {0°, 120°, 240°} axis rotations is not ideal. Use a well-known 24-cell-like
    sampling: Fibonacci on unit 4-sphere. For n ~ 24 we use a random-seeded
    set chosen once."""
    # Hopf-like: pick ~sqrt(n) directions and ~sqrt(n) axis angles
    m_axis = int(np.ceil(np.sqrt(n)))
    m_angle = int(np.ceil(n / m_axis))
    axes = fibonacci_sphere(m_axis)
    angles = np.linspace(0.0, 2.0 * np.pi, m_angle, endpoint=False)
    quats = []
    for ax in axes:
        for a in angles:
            s = np.sin(a / 2.0)
            quats.append([np.cos(a / 2.0), s * ax[0], s * ax[1], s * ax[2]])
    q = np.asarray(quats)[:n]
    # normalize
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q


# -------- worker ----------------------------------------------------------------


_SEED_BUNDLE: dict | None = None


def _worker_init(seed: int) -> None:
    global _SEED_BUNDLE
    _SEED_BUNDLE = load_seed(seed)


def _score_direction_oracle_q0(args):
    """Score one ω direction using oracle q0. args = (idx, unit_vec, |ω|_rad)."""
    idx, uvec, omega_mag_rad = args
    b = _SEED_BUNDLE
    omega0 = uvec * omega_mag_rad
    mag = predict_lc(
        b["q0_true"], omega0, b["inertia_tensor"],
        b["observation_times"], b["sun_j2k"], b["obs_j2k"],
        b["sat_j2k"], b["obs_dist"],
    )
    return idx, lc_mse(mag, b["mag_hifi"])


def _score_direction_joint_q0(args):
    """Score one ω direction using a small SO(3) grid over q0."""
    idx, uvec, omega_mag_rad, q_grid = args
    b = _SEED_BUNDLE
    omega0 = uvec * omega_mag_rad
    best_mse = np.inf
    best_qi = -1
    for qi, q in enumerate(q_grid):
        mag = predict_lc(
            q, omega0, b["inertia_tensor"],
            b["observation_times"], b["sun_j2k"], b["obs_j2k"],
            b["sat_j2k"], b["obs_dist"],
        )
        e = lc_mse(mag, b["mag_hifi"])
        if e < best_mse:
            best_mse = e
            best_qi = qi
    return idx, best_mse, best_qi


# -------- per-seed sweep --------------------------------------------------------


def sweep_oracle_q0(seed: int, directions: np.ndarray, n_workers: int = 6) -> np.ndarray:
    b = load_seed(seed)
    omega_mag_rad = float(np.linalg.norm(b["omega0_true"]))
    args_iter = [(i, directions[i], omega_mag_rad) for i in range(len(directions))]
    mse = np.empty(len(directions))
    with Pool(n_workers, initializer=_worker_init, initargs=(seed,)) as p:
        for idx, m in p.imap_unordered(_score_direction_oracle_q0, args_iter, chunksize=32):
            mse[idx] = m
    return mse


def sweep_joint_q0(seed: int, directions: np.ndarray, q_grid: np.ndarray,
                    n_workers: int = 6) -> tuple[np.ndarray, np.ndarray]:
    b = load_seed(seed)
    omega_mag_rad = float(np.linalg.norm(b["omega0_true"]))
    args_iter = [(i, directions[i], omega_mag_rad, q_grid) for i in range(len(directions))]
    mse = np.empty(len(directions))
    best_qi = np.empty(len(directions), dtype=int)
    with Pool(n_workers, initializer=_worker_init, initargs=(seed,)) as p:
        for idx, m, qi in p.imap_unordered(_score_direction_joint_q0, args_iter, chunksize=8):
            mse[idx] = m
            best_qi[idx] = qi
    return mse, best_qi


# -------- main ------------------------------------------------------------------


def main() -> None:
    pilot_seeds = [0, 49, 69, 23, 81]
    n_dirs = 2562

    print(f"Building Fibonacci S² grid with {n_dirs} directions.")
    directions = fibonacci_sphere(n_dirs)

    # Oracle-q0 sweep across all pilot seeds
    per_seed_oracle: dict[int, dict] = {}
    t_total0 = time.perf_counter()
    for seed in pilot_seeds:
        b = load_seed(seed)
        print(f"\n=== seed {seed:03d}  |ω|={b['omega_mag_dps']:.3f} dps ===")
        t0 = time.perf_counter()
        mse = sweep_oracle_q0(seed, directions, n_workers=6)
        t_elapsed = time.perf_counter() - t0

        true_unit = b["omega0_true"] / np.linalg.norm(b["omega0_true"])
        dir_deg = np.degrees(np.arccos(np.clip(directions @ true_unit, -1.0, 1.0)))

        under_gate = mse < RESIDUAL_MSE_GATE
        under_tight = mse < RESIDUAL_MSE_TIGHT
        best_idx = int(np.argmin(mse))
        # what's the dir error of the best candidate?
        best_dir_deg = float(dir_deg[best_idx])
        # truth dir error within 5° count
        truth_near = dir_deg < 5.0
        sol_set_truth_in = int(np.sum(under_gate & truth_near))

        print(f"  elapsed {t_elapsed:.1f} s")
        print(f"  candidates under GATE:  {under_gate.sum():4d}/{n_dirs}")
        print(f"  candidates under TIGHT: {under_tight.sum():4d}/{n_dirs}")
        print(f"  best_idx={best_idx}  best_mse={mse[best_idx]:.6f}  best_dir_err={best_dir_deg:.2f}°")
        print(f"  does truth-nearby (dir<5°) candidate pass GATE? -> {sol_set_truth_in>0} (count={sol_set_truth_in})")

        # save
        seed_dir = OUT_DIR / f"seed{seed:03d}"
        seed_dir.mkdir(exist_ok=True)
        npz_path = seed_dir / "pilot_oracleq0.npz"
        np.savez_compressed(
            npz_path,
            directions=directions,
            mse=mse,
            omega_mag_rad=np.linalg.norm(b["omega0_true"]),
            true_unit=true_unit,
            dir_err_deg=dir_deg,
            under_gate_mask=under_gate,
            under_tight_mask=under_tight,
        )
        print(f"  Saved: {npz_path}")

        # top-K LCs
        topk = 10
        order = np.argsort(mse)[:topk]
        lcs = []
        for k in order:
            omega0 = directions[k] * np.linalg.norm(b["omega0_true"])
            mag = predict_lc(
                b["q0_true"], omega0, b["inertia_tensor"],
                b["observation_times"], b["sun_j2k"], b["obs_j2k"],
                b["sat_j2k"], b["obs_dist"],
            )
            lcs.append(mag)
        lc_path = seed_dir / "pilot_oracleq0_topK_lc.npz"
        np.savez_compressed(
            lc_path,
            top_indices=order,
            top_mse=mse[order],
            top_directions=directions[order],
            top_dir_err_deg=dir_deg[order],
            top_lcs=np.stack(lcs),
            mag_hifi=b["mag_hifi"],
            observation_times=b["observation_times"],
        )
        print(f"  Saved: {lc_path}")

        per_seed_oracle[seed] = {
            "seed": int(seed),
            "omega_mag_dps": float(b["omega_mag_dps"]),
            "phase_mean_deg": float(np.mean(b["phase_angle_3d"])),
            "n_dirs": int(n_dirs),
            "n_under_gate": int(under_gate.sum()),
            "n_under_tight": int(under_tight.sum()),
            "best_idx": best_idx,
            "best_mse": float(mse[best_idx]),
            "best_dir_err_deg": best_dir_deg,
            "truth_nearby_under_gate": int(sol_set_truth_in),
            "sweep_time_s": float(t_elapsed),
        }

    t_total1 = time.perf_counter()
    print(f"\nOracle-q0 sweep total: {t_total1 - t_total0:.1f} s "
          f"({(t_total1 - t_total0) / len(pilot_seeds):.1f} s/seed)")

    # Joint-q0 diagnostic on seed 0 with a smaller grid
    joint_summary = {}
    if True:
        seed_j = 0
        n_dirs_j = 642
        q_grid = so3_grid(24)
        print(f"\n=== JOINT-q0 diagnostic (seed {seed_j:03d}, {n_dirs_j} dirs × {len(q_grid)} q0) ===")
        dirs_j = fibonacci_sphere(n_dirs_j)
        t0 = time.perf_counter()
        mse_j, best_qi = sweep_joint_q0(seed_j, dirs_j, q_grid, n_workers=6)
        t_j = time.perf_counter() - t0

        b = load_seed(seed_j)
        true_unit = b["omega0_true"] / np.linalg.norm(b["omega0_true"])
        dir_deg_j = np.degrees(np.arccos(np.clip(dirs_j @ true_unit, -1.0, 1.0)))
        under_gate_j = mse_j < RESIDUAL_MSE_GATE
        best_idx_j = int(np.argmin(mse_j))

        # oracle-q0 comparison on the same dirs
        mse_o_same = sweep_oracle_q0(seed_j, dirs_j, n_workers=6)
        under_gate_o = mse_o_same < RESIDUAL_MSE_GATE

        print(f"  joint elapsed {t_j:.1f} s")
        print(f"  joint under GATE:  {under_gate_j.sum():4d}/{n_dirs_j}")
        print(f"  oracle under GATE: {under_gate_o.sum():4d}/{n_dirs_j} (same grid, same |ω|)")
        print(f"  joint best idx={best_idx_j}  mse={mse_j[best_idx_j]:.6f}  dir_err={dir_deg_j[best_idx_j]:.2f}°")

        seed_dir = OUT_DIR / f"seed{seed_j:03d}"
        seed_dir.mkdir(exist_ok=True)
        j_path = seed_dir / "pilot_jointq0.npz"
        np.savez_compressed(
            j_path,
            directions=dirs_j,
            mse_joint=mse_j,
            best_q_idx=best_qi,
            mse_oracle_same_grid=mse_o_same,
            q_grid=q_grid,
            true_unit=true_unit,
            dir_err_deg=dir_deg_j,
            under_gate_joint=under_gate_j,
            under_gate_oracle=under_gate_o,
        )
        print(f"  Saved: {j_path}")
        joint_summary = {
            "seed": int(seed_j),
            "n_dirs": int(n_dirs_j),
            "n_q0": int(len(q_grid)),
            "joint_under_gate": int(under_gate_j.sum()),
            "oracle_under_gate": int(under_gate_o.sum()),
            "joint_best_mse": float(mse_j[best_idx_j]),
            "joint_best_dir_err_deg": float(dir_deg_j[best_idx_j]),
            "joint_elapsed_s": float(t_j),
        }

    # JSON summary
    summary = {
        "pilot_seeds": pilot_seeds,
        "n_dirs_oracle": int(n_dirs),
        "sweep_oracle_q0": per_seed_oracle,
        "joint_q0_diagnostic": joint_summary,
    }
    s_path = OUT_DIR / "pilot_summary.json"
    with open(s_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {s_path}")

    # Globe HTML (one seed: seed 0)
    try:
        import plotly.graph_objects as go
        seed_g = 0
        data = np.load(OUT_DIR / f"seed{seed_g:03d}" / "pilot_oracleq0.npz")
        d = data["directions"]
        mse = data["mse"]
        tu = data["true_unit"]
        log_mse = np.log10(np.clip(mse, 1e-6, None))
        colorbar_title = "log10 MSE"
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=d[:, 0], y=d[:, 1], z=d[:, 2],
                    mode="markers",
                    marker=dict(size=3, color=log_mse, colorscale="Viridis",
                                 cmin=np.log10(1e-4), cmax=np.log10(1.0),
                                 colorbar=dict(title=colorbar_title)),
                    name="S² samples",
                ),
                go.Scatter3d(
                    x=[tu[0]], y=[tu[1]], z=[tu[2]],
                    mode="markers", marker=dict(size=12, color="red", symbol="x"),
                    name="truth ω̂",
                ),
            ]
        )
        fig.update_layout(
            title=f"seed {seed_g:03d} — oracle-q0 residual MSE on S² ({n_dirs} dirs, |ω|=true)",
            scene=dict(aspectmode="data"),
        )
        html_path = OUT_DIR / f"seed{seed_g:03d}_globe.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved: {html_path}")
    except Exception as e:
        print(f"[warn] globe render failed: {e}")


if __name__ == "__main__":
    main()
