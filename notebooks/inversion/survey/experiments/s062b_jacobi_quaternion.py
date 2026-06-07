"""s062b — Jacobi q(t) — 3-gate validation on hybrid path.

Validates `lib.jacobi_propagator.propagate_jacobi` against cached post-fix
DOP853 truth on regime-spanning seeds.

NOTE (2026-05-12): The full closed-form Path 2 (precession+nutation Euler
decomposition) attempted in this session hit a convention puzzle — the
codebase's q ODE `dq/dt = +0.5 omega ⊗ q` (LEFT, Hamilton) does NOT match
the standard passive-J2000->body kinematic `dR/dt = -[omega]_x R`, and
L_J2000 = R.T @ L_body is NOT conserved by the propagator output on a toy
asymmetric tumbler. See `_propagate_jacobi_path2_incomplete` docstring in
`lib/jacobi_propagator.py` for the record.

Production `propagate_jacobi` is therefore the HYBRID path (closed-form
omega + DOP853 on q-quaternion ODE), which matches the propagator exactly
because they share the same q ODE and integrator family. Architecture-
speedup (~10x for closed-form q(t)) is deferred.

3-gate validation (per `experiments/s062_jacobi_design.md`):
- GATE 1: q vs fresh DOP853, max abs err < 1e-9 (element-wise antipode-aware).
- GATE 2: conserved 2T, L^2 along Jacobi q(t) to < 1e-12 (std/mean).
- GATE 3: renderer k1_body from Jacobi q matches cached truth to < 1e-12.

Test seeds (regime span):
- 89: slow tumbler, Case A, small polhode (|omega|=0.24 dps).
- 28: fast tumbler, Case B, large polhode (|omega|=1.44 dps).
- 14: near-prolate, Case A, near-separatrix-ish (|omega|=1.23 dps).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/inversion/survey"))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.hifi_render import _build_model
from lib.jacobi_propagator import (
    propagate_jacobi,
    omega_jacobi,
)
from src.dynamics.attitude_propagator import propagate_attitude


SEEDS = (89, 28, 14)


def antipode_max_diff(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Max element-wise antipode-aware diff: at each row, take min of
    ||q_a - q_b|| and ||q_a + q_b||, then take the max over rows.
    """
    diff_plus = np.linalg.norm(q_a - q_b, axis=1)
    diff_minus = np.linalg.norm(q_a + q_b, axis=1)
    return float(np.max(np.minimum(diff_plus, diff_minus)))


def render_k1_body(q_hist: np.ndarray, sun_pos: np.ndarray, sat_pos: np.ndarray) -> np.ndarray:
    """Apply renderer convention: k1_body = R_J2000_to_body @ (sun - sat) / ||...||."""
    sun_vec = sun_pos - sat_pos
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    N = q_hist.shape[0]
    out = np.empty_like(sun_unit)
    for i in range(N):
        q = q_hist[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        out[i] = R @ sun_unit[i]
    return out


def main():
    out_dir = PROJECT_ROOT / "notebooks/inversion/survey/results/s062b"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, I = _build_model()

    rows = []
    all_pass = True

    for seed in SEEDS:
        traj_path = PROJECT_ROOT / f"notebooks/inversion/survey/data/trajectories/traj_seed{seed:03d}.npz"
        d = np.load(traj_path)
        q0 = d["q0_wxyz"]
        w0 = d["omega0_rad"]
        times = d["observation_times"]
        q_truth = d["quaternions"]
        sun_pos = d["sun_pos"]
        sat_pos = d["sat_pos"]
        k1_body_truth = d["k1_body"]

        # Fresh DOP853 baseline
        t0 = time.time()
        q_dop, _ = propagate_attitude(q0, w0, times, "tumbling", I)
        wall_dop = time.time() - t0

        # Production Jacobi (hybrid: closed-form omega + DOP853 on q)
        t0 = time.time()
        q_jac, w_jac = propagate_jacobi(q0, w0, I, times)
        wall_jac = time.time() - t0

        # GATE 1: q_jac vs fresh DOP853
        gate1_diff = antipode_max_diff(q_jac, q_dop)
        gate1_pass = gate1_diff < 1e-9

        # GATE 2: conserved 2T, L^2 along Jacobi q(t)
        # Computed from w_jac (closed-form omega in body frame).
        # 2T = w^T I w; L^2 = (Iw)^T (Iw)
        Iw = w_jac @ I.T   # since I = I^T this is the same as I @ w_jac elementwise per row
        twoT_t = np.einsum("ni,ij,nj->n", w_jac, I, w_jac)
        L2_t = np.einsum("ni,ni->n", Iw, Iw)
        twoT_drift = float(np.std(twoT_t) / np.abs(np.mean(twoT_t)))
        L2_drift = float(np.std(L2_t) / np.abs(np.mean(L2_t)))
        gate2_pass = (twoT_drift < 1e-12) and (L2_drift < 1e-12)

        # GATE 3: renderer k1_body from Jacobi q matches cached truth
        k1_jac = render_k1_body(q_jac, sun_pos, sat_pos)
        gate3_diff = float(np.max(np.linalg.norm(k1_jac - k1_body_truth, axis=1)))
        gate3_pass = gate3_diff < 1e-12

        # Diagnostics — vs cached truth
        q_vs_truth = antipode_max_diff(q_jac, q_truth)
        q_dop_vs_truth = antipode_max_diff(q_dop, q_truth)

        seed_pass = gate1_pass and gate2_pass and gate3_pass
        all_pass = all_pass and seed_pass

        row = {
            "seed": seed,
            "gate1_q_vs_dop853": gate1_diff,
            "gate1_pass": bool(gate1_pass),
            "gate2_twoT_drift": twoT_drift,
            "gate2_L2_drift": L2_drift,
            "gate2_pass": bool(gate2_pass),
            "gate3_k1_body_vs_cached": gate3_diff,
            "gate3_pass": bool(gate3_pass),
            "q_vs_truth_max": q_vs_truth,
            "q_dop_vs_truth_max": q_dop_vs_truth,
            "wall_dop_s": wall_dop,
            "wall_jacobi_s": wall_jac,
            "all_pass": bool(seed_pass),
        }
        rows.append(row)

        print(f"\nseed {seed}:")
        print(f"  GATE 1 (q_jac vs DOP853):   {gate1_diff:.3e}  {'PASS' if gate1_pass else 'FAIL'}  (gate < 1e-9)")
        print(f"  GATE 2 (2T, L^2 drift):     2T={twoT_drift:.3e}  L2={L2_drift:.3e}  {'PASS' if gate2_pass else 'FAIL'}  (gate < 1e-12)")
        print(f"  GATE 3 (k1_body vs cached): {gate3_diff:.3e}  {'PASS' if gate3_pass else 'FAIL'}  (gate < 1e-12)")
        print(f"  diag: q_vs_truth={q_vs_truth:.3e}  q_dop_vs_truth={q_dop_vs_truth:.3e}")
        print(f"  wall: DOP={wall_dop*1000:.1f}ms  Jacobi={wall_jac*1000:.1f}ms")

    summary = {
        "seeds": list(SEEDS),
        "all_pass": bool(all_pass),
        "per_seed": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {out_dir / 'summary.json'}")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURE — see per-seed gates above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
