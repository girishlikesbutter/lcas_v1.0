"""b3_adam_multistart.py — Adam multi-start joint (q0, omega) inversion on a seed.

End-to-end differentiable: torch RK4 propagator + torch v2 surrogate -> mag MSE.
Random initial guesses are drawn from the prior:
    q0 ~ Uniform(S^3)   (4-vector then normalised)
    omega ~ mag log-uniform in [0.3, 20] dps, direction ~ Uniform(S^2)

We run all starts as a single batched gradient descent (torch propagate_euler_torch_batched
already supports B parallel trajectories). q0 is renormalised inside the loss so
the unit-norm constraint is enforced implicitly.

Multi-solution philosophy: keep ALL final candidates that land under GATE,
plus the full per-start history for debug. Saves:
  data/results/inversion_diagnostics/13_clean_slate_omega/b_differentiable/seed{NNN}/
    candidates.npz    — final (q0, omega, mse) for every start + mask
    trace.npz         — loss trace per start (for plot/diagnosis)
    result.json       — summary

CLI: --seed, --n-starts, --batch-size, --n-iter, --lr, --substeps
Defaults are memory-safe (B=8, substeps=8); see docstring at top of main.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep BLAS single-threaded so torch doesn't over-commit.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

from lib.data import load_seed  # noqa: E402
from lib.scoring import (  # noqa: E402
    RESIDUAL_MSE_GATE, RESIDUAL_MSE_TIGHT,
    lc_mse, omega_errors, quat_geodesic_deg, classify,
)
from torch_propagator import (  # noqa: E402
    propagate_euler_torch_batched, body_vectors_from_quats_batched,
)
from torch_surrogate import TorchSurrogateV2  # noqa: E402


OUT_ROOT = (HERE.parents[3] / "data" / "results" / "inversion_diagnostics"
            / "13_clean_slate_omega" / "b_differentiable")


def sample_random_starts(n: int, rng: np.random.Generator,
                         mag_lo_dps: float = 0.3, mag_hi_dps: float = 20.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Draw n random (q0, omega_rad) starts."""
    q = rng.standard_normal((n, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    lo, hi = np.log(mag_lo_dps), np.log(mag_hi_dps)
    mag = np.exp(rng.uniform(lo, hi, size=n))
    axis = rng.standard_normal((n, 3))
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    omega = axis * np.deg2rad(mag)[:, None]
    return q, omega


def run_batch_optimization(
    seed: int,
    q_init: np.ndarray,          # (B, 4)
    w_init: np.ndarray,          # (B, 3)
    n_iter: int,
    lr: float,
    substeps: int,
    model: TorchSurrogateV2,
    bundle: dict,
    dtype=torch.float64,
    device=torch.device("cpu"),
) -> dict:
    """Optimize B starts in parallel. Returns batched results + per-iter loss trace."""
    B = q_init.shape[0]

    # Bundle tensors (shared across batch)
    times_t = torch.as_tensor(bundle["observation_times"] - bundle["observation_times"][0],
                              dtype=dtype, device=device)
    sun_t = torch.as_tensor(bundle["sun_j2k"], dtype=dtype, device=device)
    obs_t = torch.as_tensor(bundle["obs_j2k"], dtype=dtype, device=device)
    sat_t = torch.as_tensor(bundle["sat_j2k"], dtype=dtype, device=device)
    dist_t = torch.as_tensor(bundle["obs_dist"], dtype=dtype, device=device)
    mag_true_t = torch.as_tensor(bundle["mag_hifi"], dtype=dtype, device=device)  # (N,)
    I_t = torch.as_tensor(bundle["inertia_tensor"], dtype=dtype, device=device)

    q_param = torch.as_tensor(q_init, dtype=dtype, device=device).clone().requires_grad_(True)
    w_param = torch.as_tensor(w_init, dtype=dtype, device=device).clone().requires_grad_(True)

    opt = torch.optim.Adam([q_param, w_param], lr=lr)

    loss_trace = np.empty((n_iter, B), dtype=np.float64)
    best_per_start = {
        "mse": np.full(B, np.inf),
        "q":   np.zeros((B, 4)),
        "w":   np.zeros((B, 3)),
        "iter": np.zeros(B, dtype=int),
    }

    for it in range(n_iter):
        opt.zero_grad()

        q_norm = q_param / torch.clamp(q_param.norm(dim=-1, keepdim=True), min=1e-12)
        # Snapshot the normalised PRE-step params so the loss and saved q/w
        # refer to the same state.
        q_pre = q_norm.detach().cpu().numpy().copy()
        w_pre = w_param.detach().cpu().numpy().copy()

        quats, _ = propagate_euler_torch_batched(
            q_norm, w_param, I_t, times_t,
            substeps_per_obs=substeps, renormalise=True,
        )
        k1, k2 = body_vectors_from_quats_batched(quats, sun_t, obs_t, sat_t)

        BN = B * k1.shape[1]
        k1_flat = k1.reshape(BN, 3)
        k2_flat = k2.reshape(BN, 3)
        panel_t = torch.zeros(BN, dtype=dtype, device=device)
        dish_t = torch.full((BN,), 15.0, dtype=dtype, device=device)
        dist_flat = dist_t.unsqueeze(0).expand(B, -1).reshape(BN)

        mag_flat = model.predict_magnitude(k1_flat, k2_flat, panel_t, dish_t, dist_flat)
        mag_pred = mag_flat.reshape(B, -1)

        residual = mag_pred - mag_true_t.unsqueeze(0)
        per_start_loss = (residual ** 2).mean(dim=1)
        total_loss = per_start_loss.sum()
        total_loss.backward()
        opt.step()

        with torch.no_grad():
            qn = q_param / torch.clamp(q_param.norm(dim=-1, keepdim=True), min=1e-12)
            q_param.copy_(qn)

            l_np = per_start_loss.detach().cpu().numpy()
            loss_trace[it] = l_np
            better = l_np < best_per_start["mse"]
            if better.any():
                idx = np.where(better)[0]
                best_per_start["mse"][idx] = l_np[idx]
                best_per_start["q"][idx] = q_pre[idx]
                best_per_start["w"][idx] = w_pre[idx]
                best_per_start["iter"][idx] = it

    return {
        "loss_trace": loss_trace,
        "best_mse": best_per_start["mse"],
        "best_q":   best_per_start["q"],
        "best_w":   best_per_start["w"],
        "best_iter": best_per_start["iter"],
    }


def run_seed(seed: int, n_starts: int, batch_size: int, n_iter: int, lr: float,
             substeps: int, seed_rng: int, include_truth: bool = True) -> dict:
    bundle = load_seed(seed)
    rng = np.random.default_rng(seed_rng + seed)

    # Optional: include the truth as one of the starts — sanity check
    q_rand, w_rand = sample_random_starts(n_starts, rng)
    if include_truth:
        q_rand[0] = bundle["q0_true"] / np.linalg.norm(bundle["q0_true"])
        w_rand[0] = bundle["omega0_true"]

    model = TorchSurrogateV2.load_default()

    all_best = {"mse": [], "q": [], "w": [], "iter": []}
    traces = []
    t0 = time.perf_counter()
    for start in range(0, n_starts, batch_size):
        q_init = q_rand[start:start + batch_size]
        w_init = w_rand[start:start + batch_size]
        print(f"  batch {start // batch_size + 1}: starts {start}..{start + q_init.shape[0] - 1}")
        out = run_batch_optimization(
            seed=seed, q_init=q_init, w_init=w_init,
            n_iter=n_iter, lr=lr, substeps=substeps,
            model=model, bundle=bundle,
        )
        all_best["mse"].append(out["best_mse"])
        all_best["q"].append(out["best_q"])
        all_best["w"].append(out["best_w"])
        all_best["iter"].append(out["best_iter"])
        traces.append(out["loss_trace"])
    total_time = time.perf_counter() - t0

    mse_all = np.concatenate(all_best["mse"])
    q_all = np.concatenate(all_best["q"])
    w_all = np.concatenate(all_best["w"])
    iter_all = np.concatenate(all_best["iter"])
    trace_all = np.concatenate(traces, axis=1)  # (n_iter, n_starts)

    # Error metrics per start
    q_err_deg = np.array([quat_geodesic_deg(q_all[i], bundle["q0_true"]) for i in range(n_starts)])
    w_err = [omega_errors(w_all[i], bundle["omega0_true"]) for i in range(n_starts)]
    w_dir_err_deg = np.array([e["dir_deg"] for e in w_err])
    w_mag_pct = np.array([e["mag_pct"] for e in w_err])

    under_gate = mse_all < RESIDUAL_MSE_GATE
    under_tight = mse_all < RESIDUAL_MSE_TIGHT

    # Save
    seed_dir = OUT_ROOT / f"seed{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cand_path = seed_dir / "candidates.npz"
    np.savez_compressed(
        cand_path,
        mse=mse_all, q=q_all, w=w_all, best_iter=iter_all,
        q_err_deg=q_err_deg, w_dir_err_deg=w_dir_err_deg, w_mag_pct=w_mag_pct,
        under_gate=under_gate, under_tight=under_tight,
        q0_true=bundle["q0_true"], omega0_true=bundle["omega0_true"],
    )
    print(f"  Saved: {cand_path}")

    trace_path = seed_dir / "trace.npz"
    np.savez_compressed(trace_path, loss_trace=trace_all.astype(np.float32))
    print(f"  Saved: {trace_path}")

    summary = {
        "seed": int(seed),
        "n_starts": int(n_starts),
        "batch_size": int(batch_size),
        "n_iter": int(n_iter),
        "lr": float(lr),
        "substeps": int(substeps),
        "include_truth_start": bool(include_truth),
        "elapsed_s": float(total_time),
        "best_mse_overall": float(mse_all.min()),
        "best_idx": int(mse_all.argmin()),
        "n_under_gate": int(under_gate.sum()),
        "n_under_tight": int(under_tight.sum()),
        "omega_mag_true_dps": float(bundle["omega_mag_dps"]),
        "best_w_dir_err_deg": float(w_dir_err_deg[mse_all.argmin()]),
        "best_w_mag_pct": float(w_mag_pct[mse_all.argmin()]),
        "best_q_err_deg": float(q_err_deg[mse_all.argmin()]),
        "classification_best": classify(float(mse_all.min())),
        # Truth-start (idx 0) as sanity anchor
        "truth_start_final_mse": float(mse_all[0]) if include_truth else None,
        "truth_start_w_dir_err_deg": float(w_dir_err_deg[0]) if include_truth else None,
    }
    json_path = seed_dir / "result.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"  Elapsed: {total_time:.1f} s")
    print(f"  Best MSE: {mse_all.min():.5f} (classify={classify(float(mse_all.min()))})")
    print(f"  Best w_dir_err: {w_dir_err_deg[mse_all.argmin()]:.2f}°  w_mag_err: {w_mag_pct[mse_all.argmin()]:.2f}%")
    print(f"  Under GATE: {under_gate.sum()}/{n_starts}  Under TIGHT: {under_tight.sum()}/{n_starts}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 23, 49, 69, 81])
    ap.add_argument("--n-starts", type=int, default=24,
                    help="total random starts per seed (plus 1 truth-init if --include-truth)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Adam batch (parallel starts in one graph); keep small for memory")
    ap.add_argument("--n-iter", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--master-seed", type=int, default=20260418)
    ap.add_argument("--no-truth-init", action="store_true",
                    help="skip the truth-initialized sanity start")
    args = ap.parse_args()

    include_truth = not args.no_truth_init
    print(f"Adam multi-start: seeds={args.seeds}, n_starts={args.n_starts}, "
          f"batch={args.batch_size}, n_iter={args.n_iter}, lr={args.lr}, substeps={args.substeps}")

    t_all0 = time.perf_counter()
    all_summaries = {}
    for seed in args.seeds:
        print(f"\n=== seed {seed:03d} ===")
        s = run_seed(
            seed=seed,
            n_starts=args.n_starts,
            batch_size=args.batch_size,
            n_iter=args.n_iter,
            lr=args.lr,
            substeps=args.substeps,
            seed_rng=args.master_seed,
            include_truth=include_truth,
        )
        all_summaries[seed] = s

    t_all = time.perf_counter() - t_all0
    super_summary = {
        "seeds": list(args.seeds),
        "params": {
            "n_starts": args.n_starts, "batch_size": args.batch_size,
            "n_iter": args.n_iter, "lr": args.lr, "substeps": args.substeps,
            "include_truth": include_truth,
        },
        "elapsed_s": float(t_all),
        "per_seed": all_summaries,
    }
    super_path = OUT_ROOT / "b3_adam_summary.json"
    with open(super_path, "w") as f:
        json.dump(super_summary, f, indent=2, default=str)
    print(f"\nSaved: {super_path}")
    print(f"TOTAL: {t_all:.1f} s ({t_all / max(len(args.seeds), 1):.1f} s/seed)")


if __name__ == "__main__":
    main()
