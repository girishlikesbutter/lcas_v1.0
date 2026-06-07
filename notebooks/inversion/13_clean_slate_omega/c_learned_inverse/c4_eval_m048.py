"""c4_eval_m048.py

Evaluate the trained MDN regressor on the 100 m048 seeds.

Procedure per seed:
  1. Load seed via lib.data.load_seed.
  2. Build the 8ch feature tensor from mag_hifi + geometry.
  3. Forward the MDN -> (log_w, mu, log_sigma).
  4. Sample N_SAMPLES candidate omegas from the mixture.
  5. Forward each candidate through lib.forward.predict_lc (using
     oracle q0_true for isolation — this evaluates the omega-recovery
     quality alone; a full pipeline would need a q0 search).
  6. Keep candidates under RESIDUAL_MSE_GATE. Compute omega dir/mag
     errors vs truth for each.
  7. Save candidates.npz, lc_fit.npz, result.json.

Finally emits a global summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SUB))

from c2_model import (
    LCInverseMDN, build_features, target_to_omega,
    LOGWMAG_MEAN, LOGWMAG_STD,
)
from lib.data import load_seed, all_seeds
from lib.forward import predict_lc
from lib.scoring import (
    lc_mse, omega_errors, classify, RESIDUAL_MSE_GATE, RESIDUAL_MSE_TIGHT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_ROOT = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
             / "13_clean_slate_omega" / "c_learned_inverse" / "eval_m048")


@torch.no_grad()
def predict_mixture(model, seed_bundle):
    feats = build_features(
        seed_bundle["mag_hifi"].astype(np.float32),
        (seed_bundle["sun_j2k"] - seed_bundle["sat_j2k"]).astype(np.float32),
        (seed_bundle["obs_j2k"] - seed_bundle["sat_j2k"]).astype(np.float32),
        seed_bundle["obs_dist"].astype(np.float32),
    )   # (1, 8, T)
    x = torch.from_numpy(feats)
    log_w, mu, log_sigma = model(x)
    return log_w[0], mu[0], log_sigma[0]


@torch.no_grad()
def sample_omegas(log_w, mu, log_sigma, n_samples: int, seed: int = 0):
    gen = torch.Generator(); gen.manual_seed(seed)
    weights = torch.softmax(log_w, dim=-1)
    K, D = mu.shape
    comp = torch.multinomial(weights, n_samples, replacement=True, generator=gen)
    mu_k = mu[comp]
    sigma_k = torch.exp(log_sigma[comp])
    eps = torch.randn(n_samples, D, generator=gen)
    samples = (mu_k + sigma_k * eps).cpu().numpy()
    # Also include the mixture component means directly (top-K candidates)
    mixture_means = mu.cpu().numpy()
    all_samples = np.concatenate([mixture_means, samples], axis=0)
    return target_to_omega(all_samples), comp.cpu().numpy()


def evaluate_seed(
    seed: int,
    model,
    out_dir: Path,
    n_samples: int = 512,
    use_oracle_q0: bool = True,
    top_k_lc_save: int = 10,
) -> dict:
    t0 = time.time()
    bundle = load_seed(seed)
    log_w, mu, log_sigma = predict_mixture(model, bundle)

    omegas_candidate, _ = sample_omegas(log_w, mu, log_sigma, n_samples, seed=seed)
    n_candidates = omegas_candidate.shape[0]

    # Score each candidate via forward model
    q0 = bundle["q0_true"] if use_oracle_q0 else bundle["q0_true"]
    # (We currently use oracle q0 per the design spec — "or use oracle q0 for
    #  isolation mode". Point-estimate q0 search is out-of-scope for Stage 4.)
    scored = np.full(n_candidates, np.nan, dtype=np.float64)
    dir_deg = np.full(n_candidates, np.nan, dtype=np.float64)
    mag_pct = np.full(n_candidates, np.nan, dtype=np.float64)
    lc_preds_top = []
    lc_mse_rank = []

    for i in range(n_candidates):
        try:
            lc = predict_lc(
                q0, omegas_candidate[i], bundle["inertia_tensor"],
                bundle["observation_times"], bundle["sun_j2k"],
                bundle["obs_j2k"], bundle["sat_j2k"], bundle["obs_dist"],
            )
            mse = lc_mse(lc, bundle["mag_hifi"])
        except Exception:
            lc = None
            mse = np.nan
        scored[i] = mse
        errs = omega_errors(omegas_candidate[i], bundle["omega0_true"])
        dir_deg[i] = errs["dir_deg"]
        mag_pct[i] = errs["mag_pct"]
        if lc is not None:
            lc_mse_rank.append((i, mse, lc))

    # Rank + pick best
    finite_mask = np.isfinite(scored)
    if finite_mask.sum() == 0:
        best_idx = -1
        best_mse = float("nan")
    else:
        best_idx = int(np.nanargmin(scored))
        best_mse = float(scored[best_idx])

    passed = (scored < RESIDUAL_MSE_GATE) & finite_mask
    tight = (scored < RESIDUAL_MSE_TIGHT) & finite_mask

    # Does any candidate hit 5°+5% of truth?
    within_5_5 = finite_mask & (np.abs(dir_deg) < 5.0) & (np.abs(mag_pct) < 5.0)

    # Save top-K LC predictions for plotting
    lc_mse_rank.sort(key=lambda r: (r[1] if np.isfinite(r[1]) else np.inf))
    lc_preds_top = [r[2] for r in lc_mse_rank[:top_k_lc_save]]
    lc_preds_idx = [r[0] for r in lc_mse_rank[:top_k_lc_save]]

    seed_out = out_dir / f"seed{seed:03d}"
    seed_out.mkdir(parents=True, exist_ok=True)

    # candidates.npz — ALL candidates under GATE (per feedback_multi_solution)
    cand_mask = passed
    np.savez(
        seed_out / "candidates.npz",
        omega_all=omegas_candidate.astype(np.float32),
        mse_all=scored.astype(np.float32),
        dir_deg_all=dir_deg.astype(np.float32),
        mag_pct_all=mag_pct.astype(np.float32),
        passed_mask=cand_mask,
        # convenience: just the passed subset
        omega_passed=omegas_candidate[cand_mask].astype(np.float32),
        mse_passed=scored[cand_mask].astype(np.float32),
        dir_deg_passed=dir_deg[cand_mask].astype(np.float32),
        mag_pct_passed=mag_pct[cand_mask].astype(np.float32),
    )

    if lc_preds_top:
        np.savez(
            seed_out / "lc_fit.npz",
            mag_hifi=bundle["mag_hifi"].astype(np.float32),
            lc_top=np.stack(lc_preds_top).astype(np.float32),
            lc_top_idx=np.asarray(lc_preds_idx, dtype=np.int64),
        )

    if best_idx >= 0:
        best_omega = omegas_candidate[best_idx]
        best_errs = omega_errors(best_omega, bundle["omega0_true"])
    else:
        best_omega = np.zeros(3)
        best_errs = {"dir_deg": np.nan, "mag_pct": np.nan,
                     "mag_dps_est": np.nan, "mag_dps_true": np.nan}

    result = {
        "seed": int(seed),
        "n_candidates_total": int(n_candidates),
        "n_passed_gate": int(cand_mask.sum()),
        "n_passed_tight": int(tight.sum()),
        "n_within_5_5": int(within_5_5.sum()),
        "truth_in_set": bool(within_5_5.any()),
        "best_mse": best_mse,
        "best_classification": classify(best_mse) if np.isfinite(best_mse) else "FAIL",
        "best_omega": best_omega.tolist(),
        "best_dir_deg": best_errs["dir_deg"],
        "best_mag_pct": best_errs["mag_pct"],
        "truth_omega": bundle["omega0_true"].tolist(),
        "truth_mag_dps": float(bundle["omega_mag_dps"]),
        "elapsed_s": time.time() - t0,
        "use_oracle_q0": bool(use_oracle_q0),
    }
    with open(seed_out / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to a .pt checkpoint")
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="",
                    help="comma list of seeds to run; default = all 100")
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()

    out_dir = EVAL_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eval output dir: {out_dir}")

    model = LCInverseMDN(K=args.K)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"Loaded ckpt: {args.ckpt}  (epoch={state.get('epoch','?')}, "
          f"val_metrics={state.get('val_metrics', {})})")

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = all_seeds()

    summary = []
    t0 = time.time()
    for s in seeds:
        r = evaluate_seed(s, model, out_dir, n_samples=args.n_samples)
        summary.append(r)
        print(f"seed {s:03d} | best_mse={r['best_mse']:.4f} ({r['best_classification']}) "
              f"| best dir {r['best_dir_deg']:.1f}° mag {r['best_mag_pct']:.1f}% "
              f"| passed_gate={r['n_passed_gate']}/{r['n_candidates_total']} "
              f"| truth_in_5_5={r['truth_in_set']} | {r['elapsed_s']:.1f}s")

    # Aggregate
    n = len(summary)
    n_ok = sum(1 for r in summary if r["best_classification"] == "OK")
    n_partial = sum(1 for r in summary if r["best_classification"] == "PARTIAL")
    n_fail = sum(1 for r in summary if r["best_classification"] == "FAIL")
    coverage_5_5 = sum(1 for r in summary if r["truth_in_set"])
    mean_passed = np.mean([r["n_passed_gate"] for r in summary])

    agg = {
        "ckpt": str(args.ckpt),
        "n_seeds": n,
        "n_OK": n_ok,
        "n_PARTIAL": n_partial,
        "n_FAIL": n_fail,
        "coverage_5deg_5pct": coverage_5_5,
        "mean_candidates_passing_gate": float(mean_passed),
        "total_elapsed_s": time.time() - t0,
        "seeds": summary,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved summary: {out_dir / 'summary.json'}")
    print(f"OK={n_ok} PARTIAL={n_partial} FAIL={n_fail} "
          f"| coverage(5°/5%)={coverage_5_5}/{n} "
          f"| avg gate-passing candidates={mean_passed:.1f}")


if __name__ == "__main__":
    main()
