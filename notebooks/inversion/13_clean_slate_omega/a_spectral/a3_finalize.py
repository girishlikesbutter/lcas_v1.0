"""a3_finalize.py — finish what a2_dir_sweep_pilot.py was doing when the PC hung.

Oracle-q0 sweeps on the 5 pilot seeds already landed as
    a_spectral/seed{NNN}/pilot_oracleq0.npz.
The joint-q0 diagnostic on seed 0, the pilot_summary.json, and the globe HTML
never ran. This script picks up exactly those three pieces — no re-runs of the
already-done work.

Thread-safety: sets OMP_NUM_THREADS=1 etc. BEFORE importing numpy, so Pool(4)
workers don't stampede BLAS. Runs one experiment at a time (no concurrent
c_learned_inverse/b_differentiable this time).
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# MUST precede numpy import — each Pool worker inherits the env.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion" / "13_clean_slate_omega"))

from lib.data import load_seed  # noqa: E402
from lib.forward import predict_lc  # noqa: E402
from lib.scoring import RESIDUAL_MSE_GATE, RESIDUAL_MSE_TIGHT, lc_mse  # noqa: E402

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "inversion_diagnostics"
    / "13_clean_slate_omega"
    / "a_spectral"
)


# -------- grid primitives (copied from a2, held stable) -----------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def so3_grid(n: int = 24) -> np.ndarray:
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
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q


# -------- workers -------------------------------------------------------------

_SEED_BUNDLE: dict | None = None


def _worker_init(seed: int) -> None:
    global _SEED_BUNDLE
    _SEED_BUNDLE = load_seed(seed)


def _score_direction_oracle_q0(args):
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


def sweep_oracle_q0(seed: int, directions: np.ndarray, n_workers: int = 4) -> np.ndarray:
    b = load_seed(seed)
    omega_mag_rad = float(np.linalg.norm(b["omega0_true"]))
    args_iter = [(i, directions[i], omega_mag_rad) for i in range(len(directions))]
    mse = np.empty(len(directions))
    with Pool(n_workers, initializer=_worker_init, initargs=(seed,)) as p:
        for idx, m in p.imap_unordered(_score_direction_oracle_q0, args_iter, chunksize=32):
            mse[idx] = m
    return mse


def sweep_joint_q0(seed: int, directions: np.ndarray, q_grid: np.ndarray,
                   n_workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
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


# -------- main ----------------------------------------------------------------

def reconstruct_oracle_summary(pilot_seeds: list[int]) -> tuple[dict, int]:
    """Read the already-saved per-seed oracle NPZs and rebuild the summary dict."""
    per_seed_oracle = {}
    n_dirs = None
    for seed in pilot_seeds:
        npz = OUT_DIR / f"seed{seed:03d}" / "pilot_oracleq0.npz"
        if not npz.exists():
            raise FileNotFoundError(f"Missing {npz}; re-run a2 before a3.")
        d = np.load(npz)
        mse = d["mse"]
        dir_deg = d["dir_err_deg"]
        under_gate = d["under_gate_mask"]
        under_tight = d["under_tight_mask"]
        best_idx = int(np.argmin(mse))
        sol_set_truth_in = int(np.sum(under_gate & (dir_deg < 5.0)))
        b = load_seed(seed)
        per_seed_oracle[seed] = {
            "seed": int(seed),
            "omega_mag_dps": float(b["omega_mag_dps"]),
            "phase_mean_deg": float(np.mean(b["phase_angle_3d"])),
            "n_dirs": int(len(mse)),
            "n_under_gate": int(under_gate.sum()),
            "n_under_tight": int(under_tight.sum()),
            "best_idx": best_idx,
            "best_mse": float(mse[best_idx]),
            "best_dir_err_deg": float(dir_deg[best_idx]),
            "truth_nearby_under_gate": sol_set_truth_in,
            "sweep_time_s": None,  # not tracked post-hoc
        }
        if n_dirs is None:
            n_dirs = int(len(mse))
    return per_seed_oracle, n_dirs


def run_joint_diagnostic(seed: int = 0, n_dirs: int = 642,
                         n_q0: int = 24, n_workers: int = 4) -> dict:
    q_grid = so3_grid(n_q0)
    print(f"\n=== JOINT-q0 diagnostic (seed {seed:03d}, {n_dirs} dirs × {len(q_grid)} q0) ===")
    dirs_j = fibonacci_sphere(n_dirs)
    t0 = time.perf_counter()
    mse_j, best_qi = sweep_joint_q0(seed, dirs_j, q_grid, n_workers=n_workers)
    t_j = time.perf_counter() - t0

    b = load_seed(seed)
    true_unit = b["omega0_true"] / np.linalg.norm(b["omega0_true"])
    dir_deg_j = np.degrees(np.arccos(np.clip(dirs_j @ true_unit, -1.0, 1.0)))
    under_gate_j = mse_j < RESIDUAL_MSE_GATE
    best_idx_j = int(np.argmin(mse_j))

    # oracle-q0 on the SAME (642-dir) grid for apples-to-apples
    mse_o_same = sweep_oracle_q0(seed, dirs_j, n_workers=n_workers)
    under_gate_o = mse_o_same < RESIDUAL_MSE_GATE

    print(f"  joint elapsed {t_j:.1f} s")
    print(f"  joint under GATE:  {int(under_gate_j.sum()):4d}/{n_dirs}")
    print(f"  oracle under GATE: {int(under_gate_o.sum()):4d}/{n_dirs} (same grid, same |ω|)")
    print(f"  joint best idx={best_idx_j}  mse={mse_j[best_idx_j]:.6f}  dir_err={dir_deg_j[best_idx_j]:.2f}°")

    seed_dir = OUT_DIR / f"seed{seed:03d}"
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
    return {
        "seed": int(seed),
        "n_dirs": int(n_dirs),
        "n_q0": int(len(q_grid)),
        "joint_under_gate": int(under_gate_j.sum()),
        "oracle_under_gate": int(under_gate_o.sum()),
        "joint_best_mse": float(mse_j[best_idx_j]),
        "joint_best_dir_err_deg": float(dir_deg_j[best_idx_j]),
        "joint_elapsed_s": float(t_j),
    }


def render_globe(seed: int = 0) -> None:
    try:
        import plotly.graph_objects as go
        data = np.load(OUT_DIR / f"seed{seed:03d}" / "pilot_oracleq0.npz")
        d = data["directions"]
        mse = data["mse"]
        tu = data["true_unit"]
        log_mse = np.log10(np.clip(mse, 1e-6, None))
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=d[:, 0], y=d[:, 1], z=d[:, 2],
                    mode="markers",
                    marker=dict(size=3, color=log_mse, colorscale="Viridis",
                                cmin=np.log10(1e-4), cmax=np.log10(1.0),
                                colorbar=dict(title="log10 MSE")),
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
            title=f"seed {seed:03d} — oracle-q0 residual MSE on S² (|ω|=true)",
            scene=dict(aspectmode="data"),
        )
        html_path = OUT_DIR / f"seed{seed:03d}_globe.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved: {html_path}")
    except Exception as e:
        print(f"[warn] globe render failed: {e}")


def main() -> None:
    pilot_seeds = [0, 49, 69, 23, 81]
    print("Reconstructing oracle-q0 summary from saved NPZs...")
    per_seed_oracle, n_dirs = reconstruct_oracle_summary(pilot_seeds)
    for seed in pilot_seeds:
        s = per_seed_oracle[seed]
        print(f"  seed{seed:03d}: |ω|={s['omega_mag_dps']:.3f} dps, "
              f"under_gate={s['n_under_gate']}/{s['n_dirs']}, "
              f"best_dir_err={s['best_dir_err_deg']:.2f}°, "
              f"truth_nearby_under_gate={s['truth_nearby_under_gate']}")

    joint_summary = run_joint_diagnostic(seed=0, n_dirs=642, n_q0=24, n_workers=4)

    summary = {
        "pilot_seeds": pilot_seeds,
        "n_dirs_oracle": n_dirs,
        "sweep_oracle_q0": per_seed_oracle,
        "joint_q0_diagnostic": joint_summary,
    }
    s_path = OUT_DIR / "pilot_summary.json"
    with open(s_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {s_path}")

    render_globe(seed=0)


if __name__ == "__main__":
    main()
