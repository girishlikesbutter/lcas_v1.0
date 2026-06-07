"""Stage 1: verify TorchSurrogateV2 matches numpy v2 SurrogateModel.

Checks on seed 0 (500 epochs, oracle k1_body / k2_body from truth):

  1. predict_magnitude(torch) vs predict_magnitude(numpy) — max abs diff <= 1e-5 mag
  2. predict_log_phi parity — max abs diff <= 1e-6 log10-units
  3. Forward is differentiable: build small loss and backward() without error.

Saves: b1_parity_seed000.json
Prints saved paths.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for lib/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "surrogate_model" / "surrogate_model"))

from lib.data import load_seed  # noqa: E402
from lib.forward import get_surrogate  # noqa: E402
from torch_surrogate import TorchSurrogateV2  # noqa: E402


OUT_DIR = (HERE.parents[3] / "data" / "results" / "inversion_diagnostics"
           / "13_clean_slate_omega" / "b_differentiable")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    seed = 0
    bundle = load_seed(seed)

    k1 = bundle["k1_body_true"]
    k2 = bundle["k2_body_true"]
    N = len(k1)
    panel = np.zeros(N)
    dish = np.full(N, 15.0)
    dist_km = bundle["obs_dist"]

    np_surr = get_surrogate("v2")

    # numpy baseline
    t0 = time.time()
    mag_np = np.asarray(np_surr.predict_magnitude(k1, k2, panel, dish, dist_km), dtype=np.float64)
    log_phi_np = np.asarray(np_surr.predict_log_phi(k1, k2, panel, dish), dtype=np.float64)
    t_np = time.time() - t0

    # torch port
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_surr = TorchSurrogateV2.load_default(dtype=torch.float64, device=device)
    torch_surr.eval()

    k1_t = torch.as_tensor(k1, dtype=torch.float64, device=device)
    k2_t = torch.as_tensor(k2, dtype=torch.float64, device=device)
    panel_t = torch.as_tensor(panel, dtype=torch.float64, device=device)
    dish_t = torch.as_tensor(dish, dtype=torch.float64, device=device)
    dist_t = torch.as_tensor(dist_km, dtype=torch.float64, device=device)

    t0 = time.time()
    with torch.no_grad():
        mag_torch = torch_surr.predict_magnitude(k1_t, k2_t, panel_t, dish_t, dist_t)
        log_phi_torch = torch_surr.predict_log_phi(k1_t, k2_t, panel_t, dish_t)
    t_torch = time.time() - t0

    mag_torch_np = mag_torch.cpu().numpy()
    log_phi_torch_np = log_phi_torch.cpu().numpy()

    diff_mag = np.abs(mag_torch_np - mag_np)
    diff_log = np.abs(log_phi_torch_np - log_phi_np)

    report = {
        "seed": seed,
        "N": int(N),
        "device": str(device),
        "numpy_time_s": float(t_np),
        "torch_time_s": float(t_torch),
        "mag_max_abs_diff": float(diff_mag.max()),
        "mag_mean_abs_diff": float(diff_mag.mean()),
        "log_phi_max_abs_diff": float(diff_log.max()),
        "log_phi_mean_abs_diff": float(diff_log.mean()),
        "mag_pass_1e-5": bool(diff_mag.max() <= 1e-5),
        "log_phi_pass_1e-6": bool(diff_log.max() <= 1e-6),
    }

    # Differentiability sanity check: gradient of mean magnitude wrt k1 should be finite.
    k1_g = torch.as_tensor(k1, dtype=torch.float64, device=device).clone().requires_grad_(True)
    k2_g = torch.as_tensor(k2, dtype=torch.float64, device=device).clone().requires_grad_(True)
    panel_g = torch.as_tensor(panel, dtype=torch.float64, device=device).clone().requires_grad_(True)
    dish_g = torch.as_tensor(dish, dtype=torch.float64, device=device).clone().requires_grad_(True)

    mag_pred = torch_surr.predict_magnitude(k1_g, k2_g, panel_g, dish_g, dist_t)
    loss = (mag_pred ** 2).mean()
    loss.backward()

    grad_info = {
        "loss_value": float(loss.detach().cpu().item()),
        "grad_k1_isfinite": bool(torch.isfinite(k1_g.grad).all().item()),
        "grad_k2_isfinite": bool(torch.isfinite(k2_g.grad).all().item()),
        "grad_panel_isfinite": bool(torch.isfinite(panel_g.grad).all().item()),
        "grad_dish_isfinite": bool(torch.isfinite(dish_g.grad).all().item()),
        "grad_k1_norm": float(k1_g.grad.norm().item()),
        "grad_panel_norm": float(panel_g.grad.norm().item()),
    }
    report["grad_check"] = grad_info

    out_json = OUT_DIR / "b1_parity_seed000.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    np.savez(
        OUT_DIR / "b1_parity_seed000.npz",
        mag_numpy=mag_np,
        mag_torch=mag_torch_np,
        log_phi_numpy=log_phi_np,
        log_phi_torch=log_phi_torch_np,
        mag_abs_diff=diff_mag,
    )

    print(f"Saved: {out_json}")
    print(f"Saved: {OUT_DIR / 'b1_parity_seed000.npz'}")
    print()
    print(f"mag  max_abs_diff = {diff_mag.max():.3e} (pass <= 1e-5: {report['mag_pass_1e-5']})")
    print(f"logphi max_abs_diff = {diff_log.max():.3e} (pass <= 1e-6: {report['log_phi_pass_1e-6']})")
    print(f"numpy fwd {t_np*1e3:.1f} ms  | torch fwd {t_torch*1e3:.1f} ms")
    print(f"grad check: k1 finite={grad_info['grad_k1_isfinite']}, "
          f"panel finite={grad_info['grad_panel_isfinite']}")

    if not (report["mag_pass_1e-5"] and grad_info["grad_k1_isfinite"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
