"""s009 — cohort basin-radius probe (Q4b extension).

Hypothesis: extends s005's 5-seed basin-radius result to all 100 m048 seeds
at minimum density (T1 + T3 ICs only). Question: how seed-28-like is the
cohort tail? If <10% of seeds fail T1 (q0=2°/ωd=0.3°/ωm-1%) inside-tube
LM convergence, Q4c-on-bulk-cohort can target a 5°-basin density
(~10⁵-candidate per seed). If 25%+ of seeds fail T1, partial-cohort
acceptance is required.

Method: 100 seeds × 2 ICs = 200 LM runs. Each IC is a perturbation of
truth state (T1 inside-tube; T3 edge-of-tube), polished by joint (q0, ω)
LM with the surrogate full-LC residual. Identical kernel to s005.

Per-seed classification:
  - T1 ✅ + T3 ✅: WIDE basin (≥8° in q0)
  - T1 ✅ + T3 ❌: MID basin (2°-8°)
  - T1 ❌ + T3 ❌: TIGHT basin (<2°, seed-28-class)
  - T1 ❌ + T3 ✅: ANOMALOUS (rare; non-convex basin)

Convergence bar (matches s005 strict):
  q0_err < 5° AND ω_dir_err < 1° AND |ω_mag_err_pct| < 5%

Outputs:
  results/s009/runs.npz         - per-IC final state, errors, MSEs
  results/s009/summary.json     - cohort distribution scalars + tight-seed list
  results/s009/cohort_basins.png - per-seed bar chart sorted by tightness
  results/s009/error_panels.png - cohort histograms of q0_err / ωd_err / ωm_err
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib import surrogate_eval, traj_load  # noqa: E402
from lib.forward import propagate_to_body_frame, quat_geodesic_deg  # noqa: E402

PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s009"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = traj_load.list_seeds()  # all 100
N_WORKERS = 8
MAX_NFEV = 200

# Twin: 180° body-X rotation
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# IC ladder (same as s005 T1, T3)
DETERMINISTIC_TIERS = [
    ("T1_inside",  2.0, 0.3, 0.99),
    ("T3_edge",    8.0, 1.0, 0.95),
]


# ────────────────────────────────────────────────────────────────────────────
# Quaternion utilities (verbatim from s005)
# ────────────────────────────────────────────────────────────────────────────

def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_rotvec_wxyz(rotvec):
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0 - 0.125 * angle * angle,
                         0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    half = 0.5 * angle
    s = np.sin(half) / angle
    return np.array([np.cos(half), s * rotvec[0], s * rotvec[1], s * rotvec[2]])


# ────────────────────────────────────────────────────────────────────────────
# IC construction — deterministic axes only (no random ICs in s009)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class InitialCondition:
    seed: int
    ic_idx: int
    label: str
    tier: int
    q0_offset_deg: float
    omega_dir_offset_deg: float
    omega_mag_factor: float
    q0_seed_wxyz: np.ndarray
    omega_seed_rad: np.ndarray
    q0_truth_wxyz: np.ndarray
    omega_truth_rad: np.ndarray
    omega_truth_dir: np.ndarray
    omega_truth_mag: float


def deterministic_perpendicular_axis(direction):
    refs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for ref in refs:
        a = np.cross(direction, ref)
        n = np.linalg.norm(a)
        if n > 1e-6:
            return a / n
    raise RuntimeError("could not find perpendicular axis")


def build_ic(seed, ic_idx, label, tier, q0_offset_deg, omega_dir_offset_deg,
             omega_mag_factor, q0_truth, omega_truth):
    omega_mag = float(np.linalg.norm(omega_truth))
    omega_dir = omega_truth / omega_mag

    q0_axis = np.array([1.0, 0.0, 0.0])  # consistent across all det tiers
    q0_perturb_rotvec = q0_axis * np.radians(q0_offset_deg)
    delta_q = quat_from_rotvec_wxyz(q0_perturb_rotvec)
    q0_seed = quat_multiply_wxyz(delta_q, q0_truth)
    q0_seed = q0_seed / np.linalg.norm(q0_seed)

    if omega_dir_offset_deg > 0.0:
        omega_perp_axis = deterministic_perpendicular_axis(omega_dir)
        rot = Rotation.from_rotvec(np.radians(omega_dir_offset_deg) * omega_perp_axis)
        omega_dir_perturbed = rot.apply(omega_dir)
    else:
        omega_dir_perturbed = omega_dir.copy()
    omega_seed = omega_dir_perturbed * (omega_mag * omega_mag_factor)

    return InitialCondition(
        seed=seed, ic_idx=ic_idx, label=label, tier=tier,
        q0_offset_deg=q0_offset_deg,
        omega_dir_offset_deg=omega_dir_offset_deg,
        omega_mag_factor=omega_mag_factor,
        q0_seed_wxyz=q0_seed,
        omega_seed_rad=omega_seed,
        q0_truth_wxyz=q0_truth.copy(),
        omega_truth_rad=omega_truth.copy(),
        omega_truth_dir=omega_dir.copy(),
        omega_truth_mag=omega_mag,
    )


def build_ics_for_seed(seed, q0_truth, omega_truth):
    ics = []
    for tier_idx, (label, q0_d, dir_d, mag_f) in enumerate(DETERMINISTIC_TIERS):
        ic = build_ic(seed=seed, ic_idx=tier_idx, label=label,
                      tier=tier_idx + 1, q0_offset_deg=q0_d,
                      omega_dir_offset_deg=dir_d, omega_mag_factor=mag_f,
                      q0_truth=q0_truth, omega_truth=omega_truth)
        ics.append(ic)
    return ics


# ────────────────────────────────────────────────────────────────────────────
# Worker globals + LM
# ────────────────────────────────────────────────────────────────────────────

_W_TIMES_BY_SEED = {}
_W_SUN_BY_SEED = {}
_W_OBS_BY_SEED = {}
_W_SAT_BY_SEED = {}
_W_OBS_DIST_BY_SEED = {}
_W_MAG_HIFI_BY_SEED = {}
_W_INERTIA_TENSOR = None


def init_worker(seed_data, inertia_tensor):
    global _W_INERTIA_TENSOR
    _W_INERTIA_TENSOR = inertia_tensor
    for s, d in seed_data.items():
        _W_TIMES_BY_SEED[s] = d["observation_times"]
        _W_SUN_BY_SEED[s] = d["sun_pos"]
        _W_OBS_BY_SEED[s] = d["obs_pos"]
        _W_SAT_BY_SEED[s] = d["sat_pos"]
        _W_OBS_DIST_BY_SEED[s] = d["obs_dist"]
        _W_MAG_HIFI_BY_SEED[s] = d["mag_hifi"]
    surrogate_eval.get_model()


def make_residual_fn(seed, q0_seed_wxyz):
    times = _W_TIMES_BY_SEED[seed]
    sun = _W_SUN_BY_SEED[seed]
    obs = _W_OBS_BY_SEED[seed]
    sat = _W_SAT_BY_SEED[seed]
    obs_dist = _W_OBS_DIST_BY_SEED[seed]
    mag_truth = _W_MAG_HIFI_BY_SEED[seed]
    inertia = _W_INERTIA_TENSOR
    finite_truth = np.isfinite(mag_truth)

    def residuals(x):
        delta_theta = x[:3]
        omega = x[3:6]
        delta_q = quat_from_rotvec_wxyz(delta_theta)
        q0 = quat_multiply_wxyz(delta_q, q0_seed_wxyz)
        q0 = q0 / np.linalg.norm(q0)
        try:
            k1b, k2b, _ = propagate_to_body_frame(
                q0, omega, times, sun, obs, sat, inertia,
            )
            pred = surrogate_eval.predict(k1b, k2b, obs_dist)
        except Exception:
            return np.full_like(mag_truth, 1e3)
        r = pred - mag_truth
        bad = ~(finite_truth & np.isfinite(r))
        r = np.where(bad, 0.0, r)
        return r

    return residuals


def run_lm_for_ic(args):
    ic_dict, max_nfev = args
    seed = int(ic_dict["seed"])
    ic_idx = int(ic_dict["ic_idx"])
    label = str(ic_dict["label"])
    q0_seed_wxyz = np.asarray(ic_dict["q0_seed_wxyz"], dtype=float)
    omega_seed_rad = np.asarray(ic_dict["omega_seed_rad"], dtype=float)
    q0_truth = np.asarray(ic_dict["q0_truth_wxyz"], dtype=float)
    omega_truth = np.asarray(ic_dict["omega_truth_rad"], dtype=float)
    omega_truth_dir = np.asarray(ic_dict["omega_truth_dir"], dtype=float)
    omega_truth_mag = float(ic_dict["omega_truth_mag"])

    residual_fn = make_residual_fn(seed, q0_seed_wxyz)
    x0 = np.concatenate([np.zeros(3), omega_seed_rad])

    initial_q0_err = quat_geodesic_deg(q0_seed_wxyz, q0_truth)
    initial_omega_dir_err = float(np.degrees(np.arccos(
        float(np.clip(np.dot(omega_seed_rad / np.linalg.norm(omega_seed_rad),
                             omega_truth_dir), -1.0, 1.0))
    )))
    initial_omega_mag_err_pct = 100.0 * (
        float(np.linalg.norm(omega_seed_rad)) - omega_truth_mag
    ) / omega_truth_mag
    initial_residual = residual_fn(x0)
    initial_mse = float(np.mean(initial_residual ** 2))

    t0 = time.time()
    try:
        result = least_squares(
            residual_fn, x0, method="lm",
            max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )
        success = bool(result.success)
        n_fev = int(result.nfev)
        x_final = result.x.copy()
    except Exception as e:
        success = False
        n_fev = 0
        x_final = x0.copy()
    wall = time.time() - t0

    delta_theta_final = x_final[:3]
    omega_final = x_final[3:6]
    delta_q_final = quat_from_rotvec_wxyz(delta_theta_final)
    q0_final = quat_multiply_wxyz(delta_q_final, q0_seed_wxyz)
    q0_final = q0_final / np.linalg.norm(q0_final)

    final_residual = residual_fn(x_final)
    final_mse = float(np.mean(final_residual ** 2))

    q0_err = quat_geodesic_deg(q0_final, q0_truth)
    q_twin = quat_multiply_wxyz(Q_180X_WXYZ, q0_truth)
    twin_err = quat_geodesic_deg(q0_final, q_twin)
    omega_final_mag = float(np.linalg.norm(omega_final))
    if omega_final_mag < 1e-12:
        omega_dir_err = 180.0
    else:
        omega_dir_err = float(np.degrees(np.arccos(
            float(np.clip(np.dot(omega_final / omega_final_mag, omega_truth_dir),
                          -1.0, 1.0))
        )))
    omega_mag_err_pct = 100.0 * (omega_final_mag - omega_truth_mag) / omega_truth_mag

    truth_basin_strict = (q0_err < 5.0) and (omega_dir_err < 1.0) and (abs(omega_mag_err_pct) < 5.0)
    truth_basin_loose = (q0_err < 10.0) and (omega_dir_err < 2.0) and (abs(omega_mag_err_pct) < 10.0)
    twin_basin_strict = (twin_err < 5.0) and (omega_dir_err < 1.0) and (abs(omega_mag_err_pct) < 5.0)

    return {
        "seed": seed, "ic_idx": ic_idx, "label": label,
        "tier": int(ic_dict["tier"]),
        "initial_q0_err_deg": float(initial_q0_err),
        "initial_omega_dir_err_deg": float(initial_omega_dir_err),
        "initial_omega_mag_err_pct": float(initial_omega_mag_err_pct),
        "initial_mse": initial_mse,
        "q0_seed_wxyz": q0_seed_wxyz, "omega_seed_rad": omega_seed_rad,
        "x_final": x_final, "q0_final_wxyz": q0_final, "omega_final_rad": omega_final,
        "q0_err_deg": float(q0_err), "twin_err_deg": float(twin_err),
        "omega_dir_err_deg": float(omega_dir_err),
        "omega_mag_err_pct": float(omega_mag_err_pct),
        "final_mse": final_mse,
        "truth_basin_strict": bool(truth_basin_strict),
        "truth_basin_loose": bool(truth_basin_loose),
        "twin_basin_strict": bool(twin_basin_strict),
        "n_fev": n_fev, "wall_s": float(wall), "success": bool(success),
    }


def ic_to_dict(ic):
    return {
        "seed": ic.seed, "ic_idx": ic.ic_idx, "label": ic.label, "tier": ic.tier,
        "q0_offset_deg": ic.q0_offset_deg,
        "omega_dir_offset_deg": ic.omega_dir_offset_deg,
        "omega_mag_factor": ic.omega_mag_factor,
        "q0_seed_wxyz": ic.q0_seed_wxyz, "omega_seed_rad": ic.omega_seed_rad,
        "q0_truth_wxyz": ic.q0_truth_wxyz, "omega_truth_rad": ic.omega_truth_rad,
        "omega_truth_dir": ic.omega_truth_dir, "omega_truth_mag": ic.omega_truth_mag,
    }


def collect_seed_data():
    seed_data = {}
    for s in SEEDS:
        d = traj_load.load_truth(s)
        seed_data[s] = {
            "observation_times": d["observation_times"], "sun_pos": d["sun_pos"],
            "obs_pos": d["obs_pos"], "sat_pos": d["sat_pos"],
            "obs_dist": d["obs_dist"], "mag_hifi": d["mag_hifi"],
        }
    return seed_data


# ────────────────────────────────────────────────────────────────────────────
# Save / plot
# ────────────────────────────────────────────────────────────────────────────


def save_runs_npz(results):
    keys_scalar = [
        "seed", "ic_idx", "tier", "initial_q0_err_deg", "initial_omega_dir_err_deg",
        "initial_omega_mag_err_pct", "initial_mse", "q0_err_deg", "twin_err_deg",
        "omega_dir_err_deg", "omega_mag_err_pct", "final_mse", "truth_basin_strict",
        "truth_basin_loose", "twin_basin_strict", "n_fev", "wall_s", "success",
    ]
    payload = {k: np.array([r[k] for r in results]) for k in keys_scalar}
    payload["label"] = np.array([r["label"] for r in results])
    payload["q0_seed_wxyz"] = np.stack([r["q0_seed_wxyz"] for r in results])
    payload["omega_seed_rad"] = np.stack([r["omega_seed_rad"] for r in results])
    payload["q0_final_wxyz"] = np.stack([r["q0_final_wxyz"] for r in results])
    payload["omega_final_rad"] = np.stack([r["omega_final_rad"] for r in results])
    np.savez(OUT_DIR / "runs.npz", **payload)


def classify_seeds(results):
    """Per-seed: return dict {seed: 'wide'|'mid'|'tight'|'anomalous'}."""
    by_seed = {}
    for r in results:
        s = r["seed"]
        by_seed.setdefault(s, {})[r["label"]] = r["truth_basin_strict"]
    out = {}
    for s, d in by_seed.items():
        t1 = d.get("T1_inside", False)
        t3 = d.get("T3_edge", False)
        if t1 and t3:
            out[s] = "wide"  # ≥8°
        elif t1 and not t3:
            out[s] = "mid"   # 2°-8°
        elif (not t1) and t3:
            out[s] = "anomalous"
        else:
            out[s] = "tight"  # <2° (seed-28-class)
    return out


def save_summary(results):
    classification = classify_seeds(results)
    counts = {"wide": 0, "mid": 0, "tight": 0, "anomalous": 0}
    for s, c in classification.items():
        counts[c] += 1

    tight_seeds = sorted([s for s, c in classification.items() if c == "tight"])
    anomalous_seeds = sorted([s for s, c in classification.items() if c == "anomalous"])
    mid_seeds = sorted([s for s, c in classification.items() if c == "mid"])
    wide_seeds = sorted([s for s, c in classification.items() if c == "wide"])

    # tier-level success rates
    t1_results = [r for r in results if r["label"] == "T1_inside"]
    t3_results = [r for r in results if r["label"] == "T3_edge"]
    t1_pass = sum(1 for r in t1_results if r["truth_basin_strict"])
    t3_pass = sum(1 for r in t3_results if r["truth_basin_strict"])

    # per-tier error stats (final)
    def err_stats(rs, key):
        v = np.array([r[key] for r in rs])
        return {
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "max": float(np.max(v)),
            "min": float(np.min(v)),
        }

    twin_recoveries = sum(1 for r in results if r["twin_basin_strict"])

    summary = {
        "n_seeds": len(SEEDS),
        "n_ics": len(results),
        "n_workers": N_WORKERS,
        "tiers": [t[0] for t in DETERMINISTIC_TIERS],
        "tier_pass_rates": {
            "T1_inside": {
                "passed": t1_pass, "total": len(t1_results),
                "frac": t1_pass / max(len(t1_results), 1),
            },
            "T3_edge": {
                "passed": t3_pass, "total": len(t3_results),
                "frac": t3_pass / max(len(t3_results), 1),
            },
        },
        "cohort_basin_classification": {
            "wide_count": counts["wide"], "wide_frac": counts["wide"] / len(SEEDS),
            "mid_count": counts["mid"], "mid_frac": counts["mid"] / len(SEEDS),
            "tight_count": counts["tight"], "tight_frac": counts["tight"] / len(SEEDS),
            "anomalous_count": counts["anomalous"], "anomalous_frac": counts["anomalous"] / len(SEEDS),
        },
        "tight_seeds": tight_seeds,
        "anomalous_seeds": anomalous_seeds,
        "mid_seeds": mid_seeds,
        "wide_seeds_count": len(wide_seeds),
        "twin_basin_recoveries": twin_recoveries,
        "T1_q0_err_stats": err_stats(t1_results, "q0_err_deg"),
        "T1_omega_dir_err_stats": err_stats(t1_results, "omega_dir_err_deg"),
        "T1_omega_mag_err_pct_stats": err_stats(
            [r for r in t1_results],
            "omega_mag_err_pct",
        ),
        "T3_q0_err_stats": err_stats(t3_results, "q0_err_deg"),
        "T3_omega_dir_err_stats": err_stats(t3_results, "omega_dir_err_deg"),
        "T3_omega_mag_err_pct_stats": err_stats(t3_results, "omega_mag_err_pct"),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, classification


def save_plots(results, classification):
    seeds = sorted(set(r["seed"] for r in results))
    by_seed = {s: {r["label"]: r for r in results if r["seed"] == s} for s in seeds}

    # Bar chart of per-seed q0_err for T1 (sorted by tightness ascending)
    seeds_sorted = sorted(seeds, key=lambda s: by_seed[s]["T1_inside"]["q0_err_deg"], reverse=True)
    t1_q0_err = [by_seed[s]["T1_inside"]["q0_err_deg"] for s in seeds_sorted]
    t3_q0_err = [by_seed[s]["T3_edge"]["q0_err_deg"] for s in seeds_sorted]
    colors = []
    for s in seeds_sorted:
        c = classification[s]
        colors.append({"wide": "tab:green", "mid": "tab:olive",
                       "tight": "tab:red", "anomalous": "tab:purple"}[c])

    fig, axes = plt.subplots(2, 1, figsize=(20, 9), sharex=True)
    x = np.arange(len(seeds_sorted))
    axes[0].bar(x, t1_q0_err, color=colors)
    axes[0].axhline(5.0, ls="--", color="k", alpha=0.5, label="basin bar 5°")
    axes[0].set_ylabel("T1_inside q0_err_deg (final)")
    axes[0].set_yscale("log")
    axes[0].set_title("s009 — per-seed final q0 error after T1_inside LM (log scale)")
    axes[0].legend()

    axes[1].bar(x, t3_q0_err, color=colors)
    axes[1].axhline(5.0, ls="--", color="k", alpha=0.5, label="basin bar 5°")
    axes[1].set_ylabel("T3_edge q0_err_deg (final)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x[::5])
    axes[1].set_xticklabels([str(s) for s in seeds_sorted[::5]], rotation=90, fontsize=7)
    axes[1].set_xlabel("seed (sorted by T1 q0_err descending — tight basins on left)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cohort_basins.png", dpi=120)
    plt.close(fig)

    # Cohort histograms (T1 and T3 final errors)
    t1 = [r for r in results if r["label"] == "T1_inside"]
    t3 = [r for r in results if r["label"] == "T3_edge"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (key, xlabel, scale, threshold) in enumerate([
        ("q0_err_deg", "q0_err_deg (log)", "log", 5.0),
        ("omega_dir_err_deg", "omega_dir_err_deg (log)", "log", 1.0),
        ("omega_mag_err_pct", "omega_mag_err_pct (linear, abs)", "linear", 5.0),
    ]):
        for row, (label, rs) in enumerate([("T1_inside", t1), ("T3_edge", t3)]):
            ax = axes[row, col]
            v = np.array([r[key] for r in rs])
            if key == "omega_mag_err_pct":
                v = np.abs(v)
            ax.hist(v, bins=30 if scale == "linear" else np.logspace(np.log10(max(v.min(), 1e-6)), np.log10(max(v.max(), 1)), 30))
            if scale == "log":
                ax.set_xscale("log")
            ax.axvline(threshold, ls="--", color="r", alpha=0.5, label=f"bar {threshold}")
            ax.set_title(f"{label} {xlabel}")
            ax.set_ylabel("count")
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_panels.png", dpi=120)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    seed_data = collect_seed_data()

    all_ics = []
    for s in SEEDS:
        d = traj_load.load_truth(s)
        q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
        omega_truth = np.asarray(d["omega0_rad"], dtype=float)
        all_ics.extend(build_ics_for_seed(s, q0_truth, omega_truth))

    print(f"s009 — cohort basin probe on {len(SEEDS)} seeds × {len(DETERMINISTIC_TIERS)} ICs = {len(all_ics)} LM runs")
    print(f"Pool({N_WORKERS}), BLAS=1, MAX_NFEV={MAX_NFEV}", flush=True)

    args_list = [(ic_to_dict(ic), MAX_NFEV) for ic in all_ics]

    t_pool = time.time()
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia_tensor)) as pool:
        results = []
        for i, r in enumerate(pool.imap(run_lm_for_ic, args_list)):
            results.append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(args_list):
                done = i + 1
                elapsed = time.time() - t_pool
                rate = done / elapsed
                eta = (len(args_list) - done) / max(rate, 1e-6)
                print(f"  [{done}/{len(args_list)}] elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s",
                      flush=True)

    save_runs_npz(results)
    summary, classification = save_summary(results)
    save_plots(results, classification)

    wall = time.time() - t_main
    print(f"\n=== s009 summary (wall {wall:.1f} s) ===")
    print(f"T1_inside pass: {summary['tier_pass_rates']['T1_inside']['passed']}/{summary['tier_pass_rates']['T1_inside']['total']} ({summary['tier_pass_rates']['T1_inside']['frac']:.1%})")
    print(f"T3_edge pass:   {summary['tier_pass_rates']['T3_edge']['passed']}/{summary['tier_pass_rates']['T3_edge']['total']} ({summary['tier_pass_rates']['T3_edge']['frac']:.1%})")
    print(f"Cohort basin classification:")
    cb = summary["cohort_basin_classification"]
    print(f"  wide  (≥8° basin):  {cb['wide_count']:3d} ({cb['wide_frac']:.0%})")
    print(f"  mid   (2°-8°):      {cb['mid_count']:3d} ({cb['mid_frac']:.0%})")
    print(f"  tight (<2°):        {cb['tight_count']:3d} ({cb['tight_frac']:.0%})")
    print(f"  anomalous:          {cb['anomalous_count']:3d} ({cb['anomalous_frac']:.0%})")
    print(f"Tight seeds:     {summary['tight_seeds']}")
    print(f"Anomalous seeds: {summary['anomalous_seeds']}")
    print(f"Twin basin recoveries: {summary['twin_basin_recoveries']}/{summary['n_ics']}")
    print("Saved:")
    print(f"  {OUT_DIR / 'runs.npz'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {OUT_DIR / 'cohort_basins.png'}")
    print(f"  {OUT_DIR / 'error_panels.png'}")


if __name__ == "__main__":
    main()
