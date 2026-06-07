"""s084 — is the "refine |ω| in 1-D" idea viable?

s083 showed the |ω| basin is razor-thin (±2% slow → <0.5% fast) but a clean,
steep, unique V *given truth q/dir*. The proposed architecture: anchor q, grid
ω-direction, then refine |ω| in 1-D against the full-LC surrogate-v2 residual.
Two unknowns decide whether that's a good idea:

  PART 1 (aliasing / global structure): scan |ω| over the FULL ls_bracket range
    (the actual production bracket, 8.6x-142x wide) at fine log resolution,
    holding q+dir at truth. Is truth-|ω| the GLOBAL min? Are there competing
    minima (aliases) a 1-D local search could fall into? If yes → "refine" must
    be "global 1-D scan", not local descent.

  PART 2 (realistic-anchor budget): repeat a narrow fine |ω| scan with q
    perturbed by 3/5/10/15 deg (a few random axes), dir at truth. Does a
    findable truth-|ω| minimum survive, and at what q-error does min-ρ leave
    Band B? Tells us whether |ω| can be refined SEPARATELY (holding q) or must
    be co-polished with q.

Truth is used only to centre the sweep and perturb q — this is a
basin/sensitivity probe (à la s073f/s083), NOT a search-yield claim.

Cheap because it is 1-D. Uses the elliprj Path 2 + surrogate-v2 (s081-validated
band agreement). Same 3 holdout seeds as s082/s083.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from lib.hifi_render import build_context  # noqa: E402
from lib.c_t_pipeline import compute_j2000_units  # noqa: E402
from lib.jacobi_propagator import propagate_jacobi_path2  # noqa: E402
from lib.surrogate_eval import get_model  # noqa: E402
from lib.lc_features import ls_bracket  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SEEDS = [(116, "LAM-slow"), (119, "LAM-fast"), (103, "SAM")]
N_FULL = 1500          # full-bracket scan points (resolves <0.5% spike over 142x)
N_NARROW = 301         # narrow scan points over ±15% of truth
Q_ERRORS_DEG = [3.0, 5.0, 10.0, 15.0]
N_AXES = 3             # random q-perturbation axes per error level
POOL_SIZE = 24
RNG = np.random.default_rng(7)

OUT_DIR = SURVEY_DIR / "results" / "s084"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CTX = {}


def init_worker(ctx_by_seed):
    global _CTX
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _CTX = ctx_by_seed
    _ = get_model()


def score_one(args):
    seed, q0, omega0 = args
    ctx = _CTX[seed]
    try:
        q_hist, _ = propagate_jacobi_path2(
            q0, omega0, ctx["inertia_tensor"], ctx["observation_times"]
        )
        R = Rotation.from_quat(q_hist[:, [1, 2, 3, 0]]).as_matrix()
        k1 = np.einsum("nij,nj->ni", R, ctx["sun_unit"])
        k2 = np.einsum("nij,nj->ni", R, ctx["obs_unit"])
        pred = get_model().predict_magnitude(k1, k2, 0.0, 15.0, ctx["obs_dist"])
        diff = pred - ctx["mag_observed"]
        m = np.isfinite(diff)
        if not m.any():
            return float("inf")
        return float(np.sqrt(np.mean(diff[m] ** 2)) / 0.05)  # ρ_surr
    except Exception:
        return float("inf")


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def perturb_q(q0, angle_deg, axis):
    """Rotate q0 by angle_deg about unit `axis` (geodesic error = angle_deg)."""
    a = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    q_delta = np.array([np.cos(a / 2.0), *(np.sin(a / 2.0) * axis)])
    out = quat_mul(q_delta, q0)
    return out / np.linalg.norm(out)


def local_minima(w, rho, max_rho=8.0):
    """Return list of (w, rho) local minima with rho < max_rho."""
    idx, _ = find_peaks(-rho, prominence=0.5)
    return [(float(w[i]), float(rho[i])) for i in idx if rho[i] < max_rho]


def main():
    t0 = time.time()
    # Build per-seed contexts + truth + scan grids
    ctx_by_seed = {}
    seed_meta = {}
    cands = []  # (seed, q0, omega0)
    index = {}  # tag -> (start, stop)

    for seed, label in SEEDS:
        ctx_raw = build_context(seed)
        su, ou = compute_j2000_units(
            ctx_raw["sun_pos"], ctx_raw["obs_pos"], ctx_raw["sat_pos"]
        )
        ctx_by_seed[seed] = {
            "inertia_tensor": ctx_raw["inertia_tensor"],
            "observation_times": ctx_raw["observation_times"],
            "sun_unit": su,
            "obs_unit": ou,
            "obs_dist": ctx_raw["obs_dist"],
            "mag_observed": ctx_raw["mag_hifi_truth"],
        }
        q0 = ctx_raw["q0_truth"]
        om = ctx_raw["omega0_truth_rad"]
        omg = float(np.linalg.norm(om))
        d = om / omg
        br = ls_bracket(ctx_raw["observation_times"], ctx_raw["mag_hifi_truth"], n_cells=5)
        lo, hi = float(br[0]), float(br[-1])

        # PART 1: full-bracket fine scan + exact truth point
        w_full = np.sort(np.append(np.geomspace(lo, hi, N_FULL), omg))
        s = len(cands)
        cands += [(seed, q0, d * w) for w in w_full]
        index[(seed, "P1")] = (s, len(cands))

        # PART 2: narrow ±15% scan at q-errors
        w_narrow = np.sort(np.append(np.linspace(0.85, 1.15, N_NARROW) * omg, omg))
        axes = RNG.normal(size=(N_AXES, 3))
        p2 = {}
        for qe in Q_ERRORS_DEG:
            for ai in range(N_AXES):
                qp = perturb_q(q0, qe, axes[ai])
                s = len(cands)
                cands += [(seed, qp, d * w) for w in w_narrow]
                p2[(qe, ai)] = (s, len(cands))
        index[(seed, "P2")] = p2

        seed_meta[seed] = {
            "label": label, "omega_mag_truth_rad_s": omg,
            "omega_mag_truth_dps": float(np.degrees(omg)),
            "bracket_lo": lo, "bracket_hi": hi, "bracket_span": hi / lo,
            "w_full": w_full, "w_narrow": w_narrow,
        }

    print(f"scoring {len(cands)} candidates on Pool({POOL_SIZE})...", flush=True)
    with Pool(POOL_SIZE, initializer=init_worker, initargs=(ctx_by_seed,)) as p:
        rho = np.array(list(p.imap(score_one, cands, chunksize=128)))
    print(f"  scored in {time.time()-t0:.1f}s", flush=True)

    # Analyse
    out = {"seeds": {}}
    for seed, label in SEEDS:
        meta = seed_meta[seed]
        omg = meta["omega_mag_truth_rad_s"]

        # PART 1
        a, b = index[(seed, "P1")]
        wf = meta["w_full"]
        rf = rho[a:b]
        gmin_i = int(np.argmin(rf))
        gmin_w = float(wf[gmin_i])
        gmin_err = (gmin_w - omg) / omg * 100.0
        # aliases: local minima with rho<8 that are NOT within 3% of truth
        mins = local_minima(wf, rf, max_rho=8.0)
        aliases = [(w, r, (w - omg) / omg * 100.0) for (w, r) in mins
                   if abs((w - omg) / omg) > 0.03]
        deep_aliases = [a_ for a_ in aliases if a_[1] < 4.0]

        # PART 2
        p2 = index[(seed, "P2")]
        wn = meta["w_narrow"]
        p2res = {}
        for qe in Q_ERRORS_DEG:
            rows = []
            for ai in range(N_AXES):
                aa, bb = p2[(qe, ai)]
                rn = rho[aa:bb]
                i = int(np.argmin(rn))
                rows.append({
                    "min_rho": float(rn[i]),
                    "min_w_err_pct": float((wn[i] - omg) / omg * 100.0),
                })
            p2res[str(qe)] = {
                "min_rho_median": float(np.median([r["min_rho"] for r in rows])),
                "min_rho_range": [float(min(r["min_rho"] for r in rows)),
                                  float(max(r["min_rho"] for r in rows))],
                "min_w_err_pct_median": float(np.median([r["min_w_err_pct"] for r in rows])),
                "per_axis": rows,
            }

        out["seeds"][str(seed)] = {
            "label": label,
            "omega_mag_truth_dps": meta["omega_mag_truth_dps"],
            "bracket_span": meta["bracket_span"],
            "part1_global_structure": {
                "global_min_w_err_pct": float(gmin_err),
                "global_min_rho": float(rf[gmin_i]),
                "truth_is_global_min_within_1pct": bool(abs(gmin_err) < 1.0),
                "n_local_minima_below_rho8": len(mins),
                "n_aliases_outside_3pct_below_rho8": len(aliases),
                "n_deep_aliases_below_rho4": len(deep_aliases),
                "aliases": [{"w_err_pct": e, "rho": r} for (w, r, e) in aliases][:10],
            },
            "part2_anchor_budget": p2res,
        }

        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax1.semilogx(wf, rf, color="tab:blue", lw=0.8)
        ax1.axvline(omg, color="k", ls="--", alpha=0.5, label="truth |ω|")
        ax1.axhline(4, color="red", ls=":", label="Band B")
        ax1.axhline(2, color="green", ls=":", label="Band A")
        for (w, r, e) in aliases:
            ax1.plot(w, r, "rx", ms=8)
        ax1.set_xlabel("|ω| (rad/s, log)")
        ax1.set_ylabel("ρ_surr")
        ax1.set_ylim(0, 12)
        ax1.set_title(f"s084 seed {seed} P1: full-bracket scan (span {meta['bracket_span']:.0f}×)")
        ax1.legend(fontsize=7)

        for qe in Q_ERRORS_DEG:
            aa, bb = p2[(qe, 0)]
            ax2.plot((wn / omg - 1) * 100, rho[aa:bb], lw=0.9, label=f"q err {qe:.0f}°")
        ax2.axhline(4, color="red", ls=":")
        ax2.axhline(2, color="green", ls=":")
        ax2.axvline(0, color="k", ls="--", alpha=0.4)
        ax2.set_xlabel("|ω| offset from truth (%)")
        ax2.set_ylabel("ρ_surr")
        ax2.set_ylim(0, 12)
        ax2.set_title(f"seed {seed} P2: |ω| scan vs q-error (axis 0)")
        ax2.legend(fontsize=7)
        fig.tight_layout()
        pp = OUT_DIR / f"seed_{seed:03d}_refine_viability.png"
        fig.savefig(pp, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {pp}", flush=True)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT_DIR/'summary.json'}", flush=True)
    print(f"Total wall {time.time()-t0:.1f}s", flush=True)

    # Console digest
    for seed, label in SEEDS:
        s = out["seeds"][str(seed)]
        p1 = s["part1_global_structure"]
        print(f"\nseed {seed} ({label}):")
        print(f"  P1 global min |ω|-err={p1['global_min_w_err_pct']:.2f}% rho={p1['global_min_rho']:.2f} "
              f"| local-minima<8: {p1['n_local_minima_below_rho8']} | "
              f"aliases(>3% off, rho<8): {p1['n_aliases_outside_3pct_below_rho8']} | "
              f"DEEP aliases(rho<4): {p1['n_deep_aliases_below_rho4']}")
        for qe in Q_ERRORS_DEG:
            r = s["part2_anchor_budget"][str(qe)]
            print(f"  P2 q-err {qe:>4.0f}°: min ρ med={r['min_rho_median']:.2f} "
                  f"(range {r['min_rho_range'][0]:.2f}-{r['min_rho_range'][1]:.2f}), "
                  f"|ω|-err med={r['min_w_err_pct_median']:.2f}%")


if __name__ == "__main__":
    main()
