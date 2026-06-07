"""s074 — Path 2 phi(t) via scipy.special.elliprj (closed-form Pi).

Replicates s072's three-gate validation for the new closed-form phi(t)
implementation in `lib.jacobi_propagator.propagate_jacobi_path2(method='elliprj')`:

  Gate 1: m048-diagonal inertia (seeds 89/28/14, cached truth)
      - q_elliprj vs q_DOP853 (fresh propagate_attitude reference)
      - q_elliprj vs q_ode (s072's solve_ivp path on the same input)
      - 2T and L^2 conservation under the elliprj path
      - per-seed wall (single-process)

  Gate 2: perturbed inertia (m048 + 1% and 10% off-diagonal terms)
      - tests R_pa non-permutation
      - q_elliprj vs q_DOP853
      - q_elliprj vs q_ode
      - 2T and L^2 conservation

  Gate 3: asymmetric satellite (synthetic I, regimes A and B)
      - q_elliprj vs q_DOP853
      - q_elliprj vs q_ode
      - 2T and L^2 conservation

  Gate 4 (s074-specific): Pool(24) wall distribution over 1000 random
      (q0, omega) candidates, seed-89 inertia.

All q comparisons are antipode-aware (||q_a − q_b||_inf or ||q_a + q_b||_inf
whichever is smaller, per-epoch then max).
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Single-thread BLAS — single-process gates run cleanly without it but
# Gate 4 launches a Pool(24); we set in the main process AND in the
# pool worker initialiser to avoid 8x slowdown per MEMORY.md.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SURVEY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

from lib.jacobi_propagator import propagate_jacobi_path2  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402


RESULTS_DIR = SURVEY_ROOT / "results" / "s074"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers (mirror s072 conventions exactly)
# ---------------------------------------------------------------------------


def _antipode_inf(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Per-epoch min(||q_a-q_b||, ||q_a+q_b||), then max over epochs."""
    diff_pos = np.linalg.norm(q_a - q_b, axis=1)
    diff_neg = np.linalg.norm(q_a + q_b, axis=1)
    return float(np.minimum(diff_pos, diff_neg).max())


def _geodesic_deg_max(q_a: np.ndarray, q_b: np.ndarray) -> float:
    dots = np.clip(np.abs(np.einsum("ni,ni->n", q_a, q_b)), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dots)).max())


def _conservation_2T_L2(omega_hist: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    Iom = omega_hist @ I.T
    twoT = np.einsum("ni,ni->n", omega_hist, Iom)
    L2 = np.einsum("ni,ni->n", Iom, Iom)
    rel_2T = float(np.std(twoT) / np.abs(np.mean(twoT)))
    rel_L2 = float(np.std(L2) / np.abs(np.mean(L2)))
    return rel_2T, rel_L2


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------


def gate_1_m048_diagonal() -> dict:
    out = {"description": "Gate 1 — m048-diagonal inertia (cached truth)", "seeds": {}}
    for seed in (89, 28, 14):
        ctx = build_context(seed)
        I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)
        truth = load_truth(seed)
        q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
        omega0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
        times = np.asarray(truth["observation_times"], dtype=np.float64)
        q_cached = np.asarray(truth["quaternions"], dtype=np.float64)

        # Reference: tightened DOP853 attitude integration (matches s072).
        t0 = time.perf_counter()
        q_dop, om_dop = propagate_attitude(q0, omega0, times, "tumbling", I,
                                           rtol=1e-12, atol=1e-14)
        wall_dop = time.perf_counter() - t0

        # New path: elliprj-based closed-form phi.
        t0 = time.perf_counter()
        q_e, om_e = propagate_jacobi_path2(q0, omega0, I, times, method="elliprj")
        wall_e = time.perf_counter() - t0

        # Comparison path: s072's solve_ivp implementation (same function,
        # method='ode' selects the legacy path).
        t0 = time.perf_counter()
        q_o, om_o = propagate_jacobi_path2(q0, omega0, I, times, method="ode")
        wall_o = time.perf_counter() - t0

        gate_1a = _antipode_inf(q_e, q_dop)
        gate_1a_geo = _geodesic_deg_max(q_e, q_dop)
        gate_1a_cache = _antipode_inf(q_e, q_cached)
        gate_1_e_vs_ode = _antipode_inf(q_e, q_o)
        rel_2T_e, rel_L2_e = _conservation_2T_L2(om_e, I)

        dop_vs_cache = _antipode_inf(q_dop, q_cached)

        out["seeds"][f"seed_{seed:03d}"] = {
            "omega_mag_dps": float(truth["omega_mag_dps"]),
            "wall_elliprj_s": wall_e,
            "wall_ode_s": wall_o,
            "wall_dop853_s": wall_dop,
            "elliprj_vs_ode_speedup": wall_o / max(wall_e, 1e-12),
            "elliprj_vs_dop853_speedup": wall_dop / max(wall_e, 1e-12),
            "gate_1a_q_vs_dop853_inf": gate_1a,
            "gate_1a_q_vs_dop853_geo_deg_max": gate_1a_geo,
            "gate_1a_q_vs_cached_inf": gate_1a_cache,
            "dop853_vs_cached_inf": dop_vs_cache,
            "gate_1_elliprj_vs_ode_inf": gate_1_e_vs_ode,
            "gate_1b_2T_relstd": rel_2T_e,
            "gate_1b_L2_relstd": rel_L2_e,
        }
        print(f"  seed {seed}: |q_e - q_DOP|={gate_1a:.2e}  "
              f"|q_e - q_ode|={gate_1_e_vs_ode:.2e}  "
              f"2T_relstd={rel_2T_e:.2e}  L2_relstd={rel_L2_e:.2e}  "
              f"wall: elliprj={wall_e*1000:.2f}ms  ode={wall_o*1000:.2f}ms  "
              f"speedup={wall_o/max(wall_e,1e-12):.1f}x")
    return out


# ---------------------------------------------------------------------------
# Gate 2
# ---------------------------------------------------------------------------


def gate_2_perturbed_inertia() -> dict:
    truth = load_truth(89)
    q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    omega0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
    times = np.asarray(truth["observation_times"], dtype=np.float64)
    ctx = build_context(89)
    I_diag = np.asarray(ctx["inertia_tensor"], dtype=np.float64)

    out = {"description": "Gate 2 — perturbed m048 inertia (DOP853 reference)",
           "cases": {}}
    trace = float(np.trace(I_diag))
    for label, eps_frac in (("perturb_1pct", 0.01), ("perturb_10pct", 0.10)):
        off = trace * eps_frac
        delta = np.array([
            [0.0, 0.6 * off, 0.2 * off],
            [0.6 * off, 0.0, 0.4 * off],
            [0.2 * off, 0.4 * off, 0.0],
        ])
        I_pert = I_diag + delta
        eig = np.linalg.eigvalsh(I_pert)
        assert (eig > 0).all()

        t0 = time.perf_counter()
        q_dop, _ = propagate_attitude(q0, omega0, times, "tumbling", I_pert,
                                      rtol=1e-12, atol=1e-14)
        wall_dop = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_e, om_e = propagate_jacobi_path2(q0, omega0, I_pert, times, method="elliprj")
        wall_e = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_o, _ = propagate_jacobi_path2(q0, omega0, I_pert, times, method="ode")
        wall_o = time.perf_counter() - t0

        gate_a = _antipode_inf(q_e, q_dop)
        gate_a_geo = _geodesic_deg_max(q_e, q_dop)
        gate_e_vs_o = _antipode_inf(q_e, q_o)
        rel_2T, rel_L2 = _conservation_2T_L2(om_e, I_pert)

        out["cases"][label] = {
            "eps_frac_of_trace": eps_frac,
            "I_eigenvalues": [float(x) for x in eig],
            "wall_elliprj_s": wall_e,
            "wall_ode_s": wall_o,
            "wall_dop853_s": wall_dop,
            "gate_a_q_vs_dop853_inf": gate_a,
            "gate_a_q_vs_dop853_geo_deg_max": gate_a_geo,
            "gate_elliprj_vs_ode_inf": gate_e_vs_o,
            "gate_b_2T_relstd": rel_2T,
            "gate_b_L2_relstd": rel_L2,
        }
        print(f"  {label}: |q_e - q_DOP|={gate_a:.2e}  "
              f"|q_e - q_ode|={gate_e_vs_o:.2e}  "
              f"2T_relstd={rel_2T:.2e}  L2_relstd={rel_L2:.2e}  "
              f"wall: elliprj={wall_e*1000:.2f}ms  ode={wall_o*1000:.2f}ms")
    return out


# ---------------------------------------------------------------------------
# Gate 3
# ---------------------------------------------------------------------------


def gate_3_asymmetric_satellite() -> dict:
    I_async = np.array([
        [2.0, 0.4, 0.15],
        [0.4, 3.5, -0.25],
        [0.15, -0.25, 5.0],
    ])
    assert np.allclose(I_async, I_async.T)
    assert (np.linalg.eigvalsh(I_async) > 0).all()

    times = np.linspace(0.0, 3600.0, 500)

    from lib.jacobi_propagator import _eigendecompose_inertia, _build_omega_func
    I_pa, R_pa = _eigendecompose_inertia(I_async)
    I_1, I_2, I_3 = I_pa

    out = {"description": "Gate 3 — asymmetric satellite (synthetic q0, omega)",
           "I_pa_ascending": [float(x) for x in I_pa],
           "cases": {}}

    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(22.5)
    q0 = np.array([np.cos(half), np.sin(half) * axis[0],
                   np.sin(half) * axis[1], np.sin(half) * axis[2]])

    omega_pa_A = np.array([0.05, 0.012, 0.008])
    omega_body_A = R_pa @ omega_pa_A
    omega_pa_B = np.array([0.008, 0.012, 0.05])
    omega_body_B = R_pa @ omega_pa_B

    for label, omega0, regime_predicted in (
        ("regime_A_dominant_I1", omega_body_A, "A"),
        ("regime_B_dominant_I3", omega_body_B, "B"),
    ):
        t0 = time.perf_counter()
        q_dop, _ = propagate_attitude(q0, omega0, times, "tumbling", I_async,
                                      rtol=1e-12, atol=1e-14)
        wall_dop = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_e, om_e = propagate_jacobi_path2(q0, omega0, I_async, times, method="elliprj")
        wall_e = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_o, _ = propagate_jacobi_path2(q0, omega0, I_async, times, method="ode")
        wall_o = time.perf_counter() - t0

        gate_a = _antipode_inf(q_e, q_dop)
        gate_a_geo = _geodesic_deg_max(q_e, q_dop)
        gate_e_vs_o = _antipode_inf(q_e, q_o)
        rel_2T, rel_L2 = _conservation_2T_L2(om_e, I_async)

        omega0_pa = R_pa.T @ omega0
        _, info = _build_omega_func(I_pa, omega0_pa)
        assert info["regime"] == regime_predicted

        out["cases"][label] = {
            "regime": info["regime"],
            "omega_mag_dps": float(np.degrees(np.linalg.norm(omega0))),
            "wall_elliprj_s": wall_e,
            "wall_ode_s": wall_o,
            "wall_dop853_s": wall_dop,
            "gate_a_q_vs_dop853_inf": gate_a,
            "gate_a_q_vs_dop853_geo_deg_max": gate_a_geo,
            "gate_elliprj_vs_ode_inf": gate_e_vs_o,
            "gate_b_2T_relstd": rel_2T,
            "gate_b_L2_relstd": rel_L2,
        }
        print(f"  {label} ({info['regime']}): |q_e - q_DOP|={gate_a:.2e}  "
              f"|q_e - q_ode|={gate_e_vs_o:.2e}  "
              f"2T_relstd={rel_2T:.2e}  L2_relstd={rel_L2:.2e}  "
              f"wall: elliprj={wall_e*1000:.2f}ms  ode={wall_o*1000:.2f}ms")
    return out


# ---------------------------------------------------------------------------
# Gate 4 — Pool(24) wall distribution over 1000 random candidates
# ---------------------------------------------------------------------------


def _gate_4_worker_init():
    """Worker initialiser — runs once per worker, before any task.

    Sets BLAS threads to 1 (per MEMORY.md `feedback_blas_threads_for_pool.md`:
    missing the torch lines causes 8x slowdown), then pre-warms two
    propagate_jacobi_path2 calls so scipy.special caches are populated.
    Production inversion pipelines call propagate_jacobi_path2 many times
    per worker, so the realistic per-candidate cost is the steady-state
    wall not the cold-start wall.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch  # noqa: F401
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass

    sys.path.insert(0, str(SURVEY_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))
    from lib.traj_load import load_truth as _load
    from lib.hifi_render import build_context as _build
    from lib.jacobi_propagator import propagate_jacobi_path2 as _prop
    truth = _load(89)
    q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    omega0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
    times = np.asarray(truth["observation_times"], dtype=np.float64)
    ctx = _build(89)
    I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)
    _prop(q0, omega0, I, times, method="elliprj")
    _prop(q0, omega0, I, times, method="elliprj")


def _gate_4_worker(args):
    """Per-task worker: time a single propagate_jacobi_path2 call."""
    idx, q0, omega0, I, times = args
    from lib.jacobi_propagator import propagate_jacobi_path2 as _prop
    t0 = time.perf_counter()
    _prop(q0, omega0, I, times, method="elliprj")
    return idx, time.perf_counter() - t0


def gate_4_pool24_wall_distribution(n_candidates: int = 1000,
                                     pool_size: int = 24) -> dict:
    """Pool(24) timing over n random (q0, omega) candidates, seed-89 inertia.

    Each worker times a single call to propagate_jacobi_path2(method='elliprj').
    Worker initialiser pre-warms; per-task work is just the timed call.
    Reports median + percentiles of the per-candidate wall distribution.
    """
    from multiprocessing import Pool

    truth = load_truth(89)
    times = np.asarray(truth["observation_times"], dtype=np.float64)
    ctx = build_context(89)
    I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)

    rng = np.random.default_rng(seed=20260520)
    q_raw = rng.standard_normal((n_candidates, 4))
    q_raw /= np.linalg.norm(q_raw, axis=1, keepdims=True)

    omega_dirs = rng.standard_normal((n_candidates, 3))
    omega_dirs /= np.linalg.norm(omega_dirs, axis=1, keepdims=True)
    omega_mag_rad = rng.uniform(np.deg2rad(0.1), np.deg2rad(2.0), size=n_candidates)
    omegas = omega_dirs * omega_mag_rad[:, None]

    args_list = [(i, q_raw[i], omegas[i], I, times) for i in range(n_candidates)]

    t_pool_0 = time.perf_counter()
    with Pool(pool_size, initializer=_gate_4_worker_init) as pool:
        results = list(pool.imap_unordered(_gate_4_worker, args_list, chunksize=8))
    wall_total_s = time.perf_counter() - t_pool_0

    walls = np.array([r[1] for r in results])
    walls_ms = walls * 1000.0

    summary = {
        "description": (
            f"Gate 4 — Pool({pool_size}) per-candidate wall over "
            f"{n_candidates} random (q0, omega) candidates (seed-89 inertia)."
        ),
        "n_candidates": int(n_candidates),
        "pool_size": int(pool_size),
        "wall_total_s": wall_total_s,
        "per_candidate_wall_ms": {
            "median": float(np.median(walls_ms)),
            "mean": float(np.mean(walls_ms)),
            "p10": float(np.percentile(walls_ms, 10)),
            "p25": float(np.percentile(walls_ms, 25)),
            "p75": float(np.percentile(walls_ms, 75)),
            "p90": float(np.percentile(walls_ms, 90)),
            "p99": float(np.percentile(walls_ms, 99)),
            "min": float(np.min(walls_ms)),
            "max": float(np.max(walls_ms)),
        },
    }

    # Save raw distribution alongside summary for inspection.
    with open(RESULTS_DIR / "wall_distribution_1000_candidates.json", "w") as f:
        json.dump({
            **summary,
            "walls_ms": walls_ms.tolist(),
        }, f, indent=2)

    print(f"  Pool({pool_size}) wall median = {summary['per_candidate_wall_ms']['median']:.3f}ms "
          f"p90 = {summary['per_candidate_wall_ms']['p90']:.3f}ms "
          f"max = {summary['per_candidate_wall_ms']['max']:.3f}ms  "
          f"total = {wall_total_s:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# cProfile sanity check (single-process, same shape as s073b)
# ---------------------------------------------------------------------------


def cprofile_single_candidate() -> None:
    """Profile the elliprj path on seed 89 to confirm the bottleneck moved."""
    truth = load_truth(89)
    q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    omega0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
    times = np.asarray(truth["observation_times"], dtype=np.float64)
    ctx = build_context(89)
    I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)

    # Warm.
    propagate_jacobi_path2(q0, omega0, I, times, method="elliprj")
    propagate_jacobi_path2(q0, omega0, I, times, method="elliprj")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        propagate_jacobi_path2(q0, omega0, I, times, method="elliprj")
    pr.disable()

    out_path = RESULTS_DIR / "cprofile_post_elliprj.txt"
    with open(out_path, "w") as f:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(25)
        f.write("=== Sorted by cumulative time ===\n")
        f.write(s.getvalue())

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats(25)
        f.write("\n=== Sorted by self/tot time ===\n")
        f.write(s.getvalue())
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== s074 Gate 1: m048-diagonal inertia ===")
    g1 = gate_1_m048_diagonal()
    print()
    print("=== s074 Gate 2: perturbed inertia ===")
    g2 = gate_2_perturbed_inertia()
    print()
    print("=== s074 Gate 3: asymmetric satellite ===")
    g3 = gate_3_asymmetric_satellite()
    print()
    print("=== s074 cProfile sanity (single-process, 20 warm iters) ===")
    cprofile_single_candidate()

    print()
    print("=== s074 Gate 4: Pool(24) wall distribution over 1000 candidates ===")
    g4 = gate_4_pool24_wall_distribution(n_candidates=1000, pool_size=24)

    summary = {
        "experiment": "s074",
        "title": "Path 2 phi(t) via scipy.special.elliprj (closed-form Pi)",
        "gates": {
            "gate_1_m048_diagonal": g1,
            "gate_2_perturbed_inertia": g2,
            "gate_3_asymmetric_satellite": g3,
            "gate_4_pool24_wall_distribution": g4,
        },
        "thresholds": {
            "q_vs_dop853_inf": 1e-9,
            "conservation_relstd": 1e-12,
            "wall_pool24_median_ms_target": 5.0,
            "wall_pool24_median_ms_ideal": 3.0,
        },
    }
    out_path = RESULTS_DIR / "gate_validation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
