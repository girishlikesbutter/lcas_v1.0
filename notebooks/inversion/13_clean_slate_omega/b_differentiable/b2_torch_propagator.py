"""Stage 2: verify torch propagator matches the scipy DOP853 reference on seed 0.

We propagate the truth (q0_true, omega0_true, inertia_tensor) over
observation_times with both:
  - reference: src.dynamics.attitude_propagator.propagate_attitude(mode='tumbling')
  - torch RK4: propagate_euler_torch() with dt_internal=0.25

Then rotate sun/obs vectors into body frame and compare to k1_body_true /
k2_body_true on file.

Pass criterion: max |body_vec_diff| <= 1e-6 (unit-length vectors).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))  # project root for src.*

from lib.data import load_seed  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from torch_propagator import (  # noqa: E402
    propagate_euler_torch,
    body_vectors_from_quats,
)


OUT_DIR = (HERE.parents[3] / "data" / "results" / "inversion_diagnostics"
           / "13_clean_slate_omega" / "b_differentiable")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    seed = 0
    b = load_seed(seed)
    dtype = torch.float64
    device = torch.device("cpu")

    times = b["observation_times"]
    t_rel = (times - times[0]).astype(np.float64)

    # Reference path
    t0 = time.time()
    quats_ref, _ = propagate_attitude(
        q0=b["q0_true"].astype(np.float64),
        omega0=b["omega0_true"].astype(np.float64),
        times=t_rel,
        mode="tumbling",
        inertia_tensor=b["inertia_tensor"].astype(np.float64),
    )
    t_ref = time.time() - t0

    # Torch path
    q0_t = torch.as_tensor(b["q0_true"], dtype=dtype, device=device)
    w0_t = torch.as_tensor(b["omega0_true"], dtype=dtype, device=device)
    I_t = torch.as_tensor(b["inertia_tensor"], dtype=dtype, device=device)
    t_rel_t = torch.as_tensor(t_rel, dtype=dtype, device=device)

    t0 = time.time()
    quats_torch, _ = propagate_euler_torch(q0_t, w0_t, I_t, t_rel_t, substeps_per_obs=16)
    t_torch = time.time() - t0

    # Direct quaternion comparison (account for global sign)
    qref = quats_ref
    qt = quats_torch.detach().cpu().numpy()
    dots = np.abs(np.einsum("ij,ij->i", qref, qt))
    # global sign fix
    signs = np.sign(np.einsum("ij,ij->i", qref, qt))
    signs[signs == 0] = 1.0
    qt_aligned = qt * signs[:, None]
    q_abs_diff = np.abs(qt_aligned - qref).max()

    # Body-frame k1/k2 comparison vs truth NPZ arrays
    sun_t = torch.as_tensor(b["sun_j2k"], dtype=dtype, device=device)
    obs_t = torch.as_tensor(b["obs_j2k"], dtype=dtype, device=device)
    sat_t = torch.as_tensor(b["sat_j2k"], dtype=dtype, device=device)

    k1_t, k2_t = body_vectors_from_quats(quats_torch, sun_t, obs_t, sat_t)
    k1_np = k1_t.detach().cpu().numpy()
    k2_np = k2_t.detach().cpu().numpy()

    k1_diff = np.abs(k1_np - b["k1_body_true"]).max()
    k2_diff = np.abs(k2_np - b["k2_body_true"]).max()

    # Also sanity-check body vectors from the reference numpy propagator match
    # the truth file (to be sure the truth file is consistent with the
    # tumbling-mode propagator convention).
    from lib.forward import body_vectors_from_attitude
    k1_ref, k2_ref = body_vectors_from_attitude(quats_ref, b["sun_j2k"], b["obs_j2k"], b["sat_j2k"])
    k1_ref_diff = np.abs(k1_ref - b["k1_body_true"]).max()
    k2_ref_diff = np.abs(k2_ref - b["k2_body_true"]).max()

    # Gradient check
    q0_g = q0_t.clone().requires_grad_(True)
    w0_g = w0_t.clone().requires_grad_(True)
    quats_g, _ = propagate_euler_torch(q0_g, w0_g, I_t, t_rel_t, substeps_per_obs=8)
    loss = (quats_g ** 2).sum()
    loss.backward()
    grad_info = {
        "loss_value": float(loss.detach().cpu().item()),
        "q0_grad_finite": bool(torch.isfinite(q0_g.grad).all().item()),
        "w0_grad_finite": bool(torch.isfinite(w0_g.grad).all().item()),
        "q0_grad_norm": float(q0_g.grad.norm().item()),
        "w0_grad_norm": float(w0_g.grad.norm().item()),
    }

    report = {
        "seed": seed,
        "ref_time_s": float(t_ref),
        "torch_time_s": float(t_torch),
        "q_abs_diff_max_vs_ref": float(q_abs_diff),
        "k1_body_diff_max_vs_truth": float(k1_diff),
        "k2_body_diff_max_vs_truth": float(k2_diff),
        "k1_body_diff_max_ref_vs_truth": float(k1_ref_diff),
        "k2_body_diff_max_ref_vs_truth": float(k2_ref_diff),
        "pass_body_1e-6": bool(max(k1_diff, k2_diff) <= 1e-6),
        "grad_check": grad_info,
    }

    out_json = OUT_DIR / "b2_propagator_seed000.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    np.savez(
        OUT_DIR / "b2_propagator_seed000.npz",
        quats_ref=quats_ref,
        quats_torch=qt,
        k1_torch=k1_np,
        k2_torch=k2_np,
        k1_truth=b["k1_body_true"],
        k2_truth=b["k2_body_true"],
    )

    print(f"Saved: {out_json}")
    print(f"Saved: {OUT_DIR / 'b2_propagator_seed000.npz'}")
    print()
    print(f"q abs_diff (torch vs scipy): {q_abs_diff:.3e}")
    print(f"k1_body diff (torch vs truth): {k1_diff:.3e}")
    print(f"k2_body diff (torch vs truth): {k2_diff:.3e}")
    print(f"k1_body diff (scipy vs truth): {k1_ref_diff:.3e}")
    print(f"k2_body diff (scipy vs truth): {k2_ref_diff:.3e}")
    print(f"pass <=1e-6: {report['pass_body_1e-6']}")
    print(f"grad: q0 {grad_info['q0_grad_norm']:.3e}, w0 {grad_info['w0_grad_norm']:.3e}")
    print(f"timing: scipy {t_ref*1e3:.1f} ms | torch {t_torch*1e3:.1f} ms")

    if not report["pass_body_1e-6"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
