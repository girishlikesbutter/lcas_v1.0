"""s005 — joint (q0, ω) local-descent validation.

Survey question Q4. s003 + s004 closed every "ω-outer / q0-inner" architecture:
surrogate-MSE has a thin (~1° dir, ~2-5% mag) truth-ω tube; alignment cost is
strictly worse where defined and structurally undefined on 24/100 cohort seeds
(incl. seed 91). Joint (q0, ω) local descent in 6 DOF is the only architecture
not yet ruled out. This experiment validates it.

Method:
  - 5 PA-stratified seeds: 6 (m141), 18 (PA-low), 28 (s004 alignment-cost
    multi-basin pathology, PA-high), 41 (clean control), 91 (m115 anchor).
  - 10 initial conditions per seed:
      * 6 deterministic tiers spanning (deep inside tube → far outside):
        T1: q0=2°,  ω-dir=0.3°, ω-mag×0.99
        T2: q0=5°,  ω-dir=0.5°, ω-mag×0.97
        T3: q0=8°,  ω-dir=1°,   ω-mag×0.95
        T4: q0=15°, ω-dir=2°,   ω-mag×0.92
        T5: q0=30°, ω-dir=4°,   ω-mag×0.88
        T6: q0=60°, ω-dir=8°,   ω-mag×0.80
      * 4 random ICs inside tube: q0 ∈ U(0°, 5°), ω-dir ∈ U(0°, 1°),
        ω-mag × U(0.97, 1.03), random axes (per-seed RNG seeded by trajectory seed).
  - 6-DOF parameterization:
      x[0:3] = δθ — rotation vector applied as q0 = exp(δθ/2) · q0_seed
      x[3:6] = ω — body-frame angular velocity (rad/s)
    Initial x = (0, 0, 0, ω_seed) so LM starts at exactly the IC q0_seed.
  - Cost: surrogate full-LC residual vector r ∈ R^N where r[i] = pred[i] - truth_mag[i].
    LM minimises ||r||² via scipy.optimize.least_squares(method='lm').
  - max_nfev = 200; default ftol/xtol/gtol = 1e-8.

Reporting (per-IC tuple, all four metrics per concepts/q_omega_coupling.md +
concepts/rho_band.md):
  - q0_err: geodesic deg between final q0 and truth-q0 (antipode-aware)
  - twin_err: geodesic deg between final q0 and twin (q_180x · q0_truth)
  - ω_dir_err: deg between final ω̂ and truth-ω̂
  - ω_mag_err_pct: 100 · (|ω_final| − |ω_truth|) / |ω_truth|
  - surrogate_mse_final: surrogate full-LC MSE at final state
  - surrogate_mse_truth_ref: cached s001 surrogate-MSE-at-truth (per-seed reference)
  - n_iter, n_fev, wall_s, status

Convergence definition (truth basin):
  q0_err < 5° AND |ω_dir_err| < 1° AND |ω_mag_err_pct| < 5%
Looser bands also reported (10°, 2°, 10%).

Pool(8) over LM runs; each worker handles one full LM run sequentially (LM is
inherently sequential, so we parallelise across the 50 IC × seed runs not within
a single LM call). BLAS=1 in workers.

Outputs:
  - results/s005/runs.npz       per-run final state + initial state arrays
  - results/s005/summary.json   convergence rates per tier + per seed
  - results/s005/convergence.png  bar chart: per-tier truth-basin success %
  - results/s005/error_panels.png  4-panel: q0_err, ω_dir_err, ω_mag_err, MSE
                                   final vs initial offset (5 seeds × tiers)
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
OUT_DIR = SURVEY_DIR / "results" / "s005"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [6, 18, 28, 41, 91]
N_WORKERS = 8
MAX_NFEV = 200

# Per-seed cached surrogate-MSE-at-truth (from s001 per_seed.csv).
TRUTH_MSE_REF = {
    6:  2.92451e-3,
    18: 3.10499e-4,
    28: 5.84697e-4,
    41: 1.33258e-4,
    91: 2.03195e-4,
}

# Twin: 180° body-X rotation (concepts/twin_degeneracy.md, IS-901 ±X plane).
Q_180X_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# IC ladder: (label, q0_offset_deg, omega_dir_offset_deg, omega_mag_factor).
# T1-T3 are inside / at edge of the s003 tube. T4-T6 progressively outside.
DETERMINISTIC_TIERS = [
    ("T1_inside",   2.0,  0.3, 0.99),
    ("T2_inside",   5.0,  0.5, 0.97),
    ("T3_edge",     8.0,  1.0, 0.95),
    ("T4_outside", 15.0,  2.0, 0.92),
    ("T5_outside", 30.0,  4.0, 0.88),
    ("T6_outside", 60.0,  8.0, 0.80),
]
N_RANDOM_INSIDE = 4  # additional ICs per seed, sampled inside the tube
RANDOM_Q0_RANGE_DEG = (0.0, 5.0)
RANDOM_DIR_RANGE_DEG = (0.0, 1.0)
RANDOM_MAG_FACTOR_RANGE = (0.97, 1.03)


# ────────────────────────────────────────────────────────────────────────────
# Quaternion utilities
# ────────────────────────────────────────────────────────────────────────────


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (w, x, y, z) · (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_rotvec_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (rad) → unit quaternion (w, x, y, z).

    Stable at zero (uses Taylor expansion when |rotvec| < 1e-8).
    """
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        # Taylor: q ≈ (1 - angle²/8, 0.5·rotvec)
        return np.array([1.0 - 0.125 * angle * angle,
                         0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    half = 0.5 * angle
    s = np.sin(half) / angle  # = sin(half)/angle
    return np.array([np.cos(half), s * rotvec[0], s * rotvec[1], s * rotvec[2]])


def random_axis(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit vector on S²."""
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def perpendicular_random_axis(direction: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Random unit vector perpendicular to `direction` (uniform on the
    perpendicular circle)."""
    v = rng.normal(size=3)
    v = v - np.dot(v, direction) * direction
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        v = np.array([1.0, 0.0, 0.0])
        v = v - np.dot(v, direction) * direction
        n = float(np.linalg.norm(v))
    return v / n


def deterministic_perpendicular_axis(direction: np.ndarray) -> np.ndarray:
    """Same as s003 — cross with body-x then body-y fallback."""
    refs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for ref in refs:
        a = np.cross(direction, ref)
        n = np.linalg.norm(a)
        if n > 1e-6:
            return a / n
    raise RuntimeError("could not find perpendicular axis")


# ────────────────────────────────────────────────────────────────────────────
# Initial-condition construction
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class InitialCondition:
    seed: int
    ic_idx: int
    label: str
    kind: str  # "deterministic" or "random"
    tier: int  # 1..6 for det, 0 for random
    q0_offset_deg: float
    omega_dir_offset_deg: float
    omega_mag_factor: float
    q0_seed_wxyz: np.ndarray
    omega_seed_rad: np.ndarray
    q0_truth_wxyz: np.ndarray
    omega_truth_rad: np.ndarray
    omega_truth_dir: np.ndarray
    omega_truth_mag: float


def build_ic(
    seed: int,
    ic_idx: int,
    label: str,
    kind: str,
    tier: int,
    q0_offset_deg: float,
    omega_dir_offset_deg: float,
    omega_mag_factor: float,
    q0_truth: np.ndarray,
    omega_truth: np.ndarray,
    rng: np.random.Generator,
    deterministic_axes: bool,
) -> InitialCondition:
    """Construct a single (q0_seed, ω_seed) IC at the requested offsets."""
    omega_mag = float(np.linalg.norm(omega_truth))
    omega_dir = omega_truth / omega_mag

    # q0 perturbation: rotate truth-q0 by q0_offset_deg around an axis.
    if deterministic_axes:
        # Deterministic axis: rotate around a fixed perpendicular direction
        # in axis-angle space. Use cross of arbitrary fixed body-x with
        # truth-q0 axis as the perturbation axis. Differs per seed.
        q0_axis = np.array([1.0, 0.0, 0.0])  # consistent across all det tiers
    else:
        q0_axis = random_axis(rng)

    q0_perturb_rotvec = q0_axis * np.radians(q0_offset_deg)
    delta_q = quat_from_rotvec_wxyz(q0_perturb_rotvec)
    q0_seed = quat_multiply_wxyz(delta_q, q0_truth)
    q0_seed = q0_seed / np.linalg.norm(q0_seed)

    # ω-dir perturbation: rotate truth-ω̂ by omega_dir_offset_deg around a
    # perpendicular axis.
    if omega_dir_offset_deg > 0.0:
        if deterministic_axes:
            perp = deterministic_perpendicular_axis(omega_dir)
        else:
            perp = perpendicular_random_axis(omega_dir, rng)
        rot = Rotation.from_rotvec(perp * np.radians(omega_dir_offset_deg))
        omega_dir_perturbed = rot.apply(omega_dir)
    else:
        omega_dir_perturbed = omega_dir.copy()

    omega_seed = omega_dir_perturbed * (omega_mag * omega_mag_factor)

    return InitialCondition(
        seed=seed,
        ic_idx=ic_idx,
        label=label,
        kind=kind,
        tier=tier,
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


def build_ics_for_seed(seed: int, q0_truth: np.ndarray,
                       omega_truth: np.ndarray) -> list[InitialCondition]:
    """Build the 10 ICs for one seed (6 det tiers + 4 random)."""
    ics = []
    rng = np.random.default_rng(1000 + seed)
    for tier_idx, (label, q0_d, dir_d, mag_f) in enumerate(DETERMINISTIC_TIERS):
        ics.append(build_ic(
            seed=seed, ic_idx=tier_idx, label=label, kind="deterministic",
            tier=tier_idx + 1, q0_offset_deg=q0_d,
            omega_dir_offset_deg=dir_d, omega_mag_factor=mag_f,
            q0_truth=q0_truth, omega_truth=omega_truth, rng=rng,
            deterministic_axes=True,
        ))
    for j in range(N_RANDOM_INSIDE):
        q0_d = float(rng.uniform(*RANDOM_Q0_RANGE_DEG))
        dir_d = float(rng.uniform(*RANDOM_DIR_RANGE_DEG))
        mag_f = float(rng.uniform(*RANDOM_MAG_FACTOR_RANGE))
        ics.append(build_ic(
            seed=seed, ic_idx=len(DETERMINISTIC_TIERS) + j,
            label=f"R{j + 1}_random", kind="random", tier=0,
            q0_offset_deg=q0_d, omega_dir_offset_deg=dir_d,
            omega_mag_factor=mag_f, q0_truth=q0_truth,
            omega_truth=omega_truth, rng=rng, deterministic_axes=False,
        ))
    return ics


# ────────────────────────────────────────────────────────────────────────────
# Worker globals + LM run
# ────────────────────────────────────────────────────────────────────────────

_W_TIMES = None
_W_SUN = None
_W_OBS = None
_W_SAT = None
_W_INERTIA = None
_W_OBS_DIST = None
_W_MAG_HIFI_BY_SEED: dict[int, np.ndarray] = {}
_W_TIMES_BY_SEED: dict[int, np.ndarray] = {}
_W_SUN_BY_SEED: dict[int, np.ndarray] = {}
_W_OBS_BY_SEED: dict[int, np.ndarray] = {}
_W_SAT_BY_SEED: dict[int, np.ndarray] = {}
_W_OBS_DIST_BY_SEED: dict[int, np.ndarray] = {}
_W_INERTIA_TENSOR = None


def init_worker(seed_data: dict, inertia_tensor: np.ndarray):
    """Cache per-seed truth state on this worker; load surrogate model.

    seed_data maps seed → dict(times, sun, obs, sat, obs_dist, mag_hifi).
    """
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


def make_residual_fn(seed: int, q0_seed_wxyz: np.ndarray):
    """Closure: residual function for one IC (uses worker globals)."""
    times = _W_TIMES_BY_SEED[seed]
    sun = _W_SUN_BY_SEED[seed]
    obs = _W_OBS_BY_SEED[seed]
    sat = _W_SAT_BY_SEED[seed]
    obs_dist = _W_OBS_DIST_BY_SEED[seed]
    mag_truth = _W_MAG_HIFI_BY_SEED[seed]
    inertia = _W_INERTIA_TENSOR
    finite_truth = np.isfinite(mag_truth)

    def residuals(x: np.ndarray) -> np.ndarray:
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
        # Per-epoch residual; replace non-finite with 0 to keep output size.
        r = pred - mag_truth
        bad = ~(finite_truth & np.isfinite(r))
        r = np.where(bad, 0.0, r)
        return r

    return residuals


def run_lm_for_ic(args: tuple) -> dict:
    """Worker function — args is (ic_dict, max_nfev).

    ic_dict is a serialised InitialCondition (dataclass → dict).
    Returns a result dict ready for aggregation.
    """
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
        status = int(result.status)
        message = str(result.message)
        n_fev = int(result.nfev)
        n_iter = -1  # MINPACK does not report iter count separately
        x_final = result.x.copy()
    except Exception as e:
        success = False
        status = -99
        message = f"exception: {e}"
        n_fev = 0
        n_iter = 0
        x_final = x0.copy()
    wall = time.time() - t0

    # Decompose final state
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
        "seed": seed,
        "ic_idx": ic_idx,
        "label": label,
        "kind": str(ic_dict["kind"]),
        "tier": int(ic_dict["tier"]),
        "initial_q0_err_deg": float(initial_q0_err),
        "initial_omega_dir_err_deg": float(initial_omega_dir_err),
        "initial_omega_mag_err_pct": float(initial_omega_mag_err_pct),
        "initial_mse": initial_mse,
        "q0_seed_wxyz": q0_seed_wxyz,
        "omega_seed_rad": omega_seed_rad,
        "x_final": x_final,
        "q0_final_wxyz": q0_final,
        "omega_final_rad": omega_final,
        "q0_err_deg": float(q0_err),
        "twin_err_deg": float(twin_err),
        "omega_dir_err_deg": float(omega_dir_err),
        "omega_mag_err_pct": float(omega_mag_err_pct),
        "final_mse": final_mse,
        "truth_basin_strict": bool(truth_basin_strict),
        "truth_basin_loose": bool(truth_basin_loose),
        "twin_basin_strict": bool(twin_basin_strict),
        "n_iter": n_iter,
        "n_fev": n_fev,
        "wall_s": float(wall),
        "success": bool(success),
        "status": int(status),
        "message": message,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def smoke_test_one_ic(ic: InitialCondition,
                      seed_data: dict,
                      inertia_tensor: np.ndarray) -> dict:
    """Run one LM in the main process; useful before launching the Pool."""
    init_worker(seed_data, inertia_tensor)
    return run_lm_for_ic((ic_to_dict(ic), MAX_NFEV))


def ic_to_dict(ic: InitialCondition) -> dict:
    """Pickle-friendly dict for Pool dispatch."""
    return {
        "seed": ic.seed,
        "ic_idx": ic.ic_idx,
        "label": ic.label,
        "kind": ic.kind,
        "tier": ic.tier,
        "q0_offset_deg": ic.q0_offset_deg,
        "omega_dir_offset_deg": ic.omega_dir_offset_deg,
        "omega_mag_factor": ic.omega_mag_factor,
        "q0_seed_wxyz": ic.q0_seed_wxyz,
        "omega_seed_rad": ic.omega_seed_rad,
        "q0_truth_wxyz": ic.q0_truth_wxyz,
        "omega_truth_rad": ic.omega_truth_rad,
        "omega_truth_dir": ic.omega_truth_dir,
        "omega_truth_mag": ic.omega_truth_mag,
    }


def collect_seed_data() -> dict:
    """Per-seed traj_load → dict suitable for worker init."""
    seed_data = {}
    for s in SEEDS:
        d = traj_load.load_truth(s)
        seed_data[s] = {
            "observation_times": d["observation_times"],
            "sun_pos": d["sun_pos"],
            "obs_pos": d["obs_pos"],
            "sat_pos": d["sat_pos"],
            "obs_dist": d["obs_dist"],
            "mag_hifi": d["mag_hifi"],
        }
    return seed_data


def main(smoke_only: bool = False):
    t_main = time.time()
    master = np.load(str(M048_MASTER), allow_pickle=True)
    inertia_tensor = np.asarray(master["inertia_tensor"], dtype=float)

    seed_data = collect_seed_data()

    # Build all ICs.
    all_ics: list[InitialCondition] = []
    for s in SEEDS:
        d = traj_load.load_truth(s)
        q0_truth = np.asarray(d["q0_wxyz"], dtype=float)
        omega_truth = np.asarray(d["omega0_rad"], dtype=float)
        all_ics.extend(build_ics_for_seed(s, q0_truth, omega_truth))

    if smoke_only:
        # Smoke test: 1 IC (seed 41, T2 — moderate inside-tube).
        ic = next(ic for ic in all_ics if ic.seed == 41 and ic.label == "T2_inside")
        print(f"\n──── SMOKE TEST: seed 41 / T2 ────", flush=True)
        print(f"q0_seed offset {ic.q0_offset_deg:.2f}°, "
              f"ω-dir offset {ic.omega_dir_offset_deg:.2f}°, "
              f"ω-mag factor {ic.omega_mag_factor:.3f}", flush=True)
        r = smoke_test_one_ic(ic, seed_data, inertia_tensor)
        print(f"\nSmoke-test result:")
        print(f"  initial    q0_err={r['initial_q0_err_deg']:.2f}°  "
              f"ω_dir_err={r['initial_omega_dir_err_deg']:.2f}°  "
              f"ω_mag_err={r['initial_omega_mag_err_pct']:+.2f}%  "
              f"mse={r['initial_mse']:.3e}")
        print(f"  final      q0_err={r['q0_err_deg']:.4f}°  "
              f"ω_dir_err={r['omega_dir_err_deg']:.4f}°  "
              f"ω_mag_err={r['omega_mag_err_pct']:+.4f}%  "
              f"mse={r['final_mse']:.3e}")
        print(f"  truth_mse_ref={TRUTH_MSE_REF.get(41):.3e}")
        print(f"  basin_strict={r['truth_basin_strict']}  "
              f"basin_loose={r['truth_basin_loose']}")
        print(f"  n_fev={r['n_fev']}  wall={r['wall_s']:.1f}s  "
              f"success={r['success']}  status={r['status']}  "
              f"msg={r['message'][:80]}")
        return

    print(f"s005: {len(SEEDS)} seeds × 10 ICs = {len(all_ics)} LM runs, "
          f"Pool({N_WORKERS}), BLAS=1.\n", flush=True)

    args_list = [(ic_to_dict(ic), MAX_NFEV) for ic in all_ics]

    t0 = time.time()
    with Pool(processes=N_WORKERS, initializer=init_worker,
              initargs=(seed_data, inertia_tensor)) as pool:
        # Stream results so we get incremental progress. imap_unordered
        # would be fine; use ordered imap so the per-IC printing is in
        # IC submission order (per-seed, per-tier).
        results = []
        for i, r in enumerate(pool.imap(run_lm_for_ic, args_list)):
            results.append(r)
            print(
                f"  [{i + 1:3d}/{len(args_list)}] seed {r['seed']:3d} "
                f"{r['label']:14s}  init=({r['initial_q0_err_deg']:5.1f}°, "
                f"{r['initial_omega_dir_err_deg']:4.1f}°, "
                f"{r['initial_omega_mag_err_pct']:+5.1f}%)  →  "
                f"final=({r['q0_err_deg']:6.2f}°, "
                f"{r['omega_dir_err_deg']:5.2f}°, "
                f"{r['omega_mag_err_pct']:+5.2f}%)  "
                f"mse={r['final_mse']:.2e} "
                f"({'BASIN' if r['truth_basin_strict'] else ('twin' if r['twin_basin_strict'] else 'miss')})"
                f"  nfev={r['n_fev']} wall={r['wall_s']:.1f}s",
                flush=True,
            )
    wall = time.time() - t0
    print(f"\nAll runs done. Total wall: {wall:.1f}s\n", flush=True)

    # ── Save NPZ ──
    save_npz(results)
    save_summary(results)
    save_plots(results)

    print(f"\nTotal wall: {time.time() - t_main:.1f}s")


def save_npz(results: list[dict]):
    n = len(results)
    arrs = {
        "seed":              np.array([r["seed"] for r in results], dtype=int),
        "ic_idx":            np.array([r["ic_idx"] for r in results], dtype=int),
        "label":             np.array([r["label"] for r in results]),
        "kind":              np.array([r["kind"] for r in results]),
        "tier":              np.array([r["tier"] for r in results], dtype=int),
        "initial_q0_err_deg":          np.array([r["initial_q0_err_deg"] for r in results]),
        "initial_omega_dir_err_deg":   np.array([r["initial_omega_dir_err_deg"] for r in results]),
        "initial_omega_mag_err_pct":   np.array([r["initial_omega_mag_err_pct"] for r in results]),
        "initial_mse":       np.array([r["initial_mse"] for r in results]),
        "q0_seed_wxyz":      np.stack([r["q0_seed_wxyz"] for r in results]),
        "omega_seed_rad":    np.stack([r["omega_seed_rad"] for r in results]),
        "x_final":           np.stack([r["x_final"] for r in results]),
        "q0_final_wxyz":     np.stack([r["q0_final_wxyz"] for r in results]),
        "omega_final_rad":   np.stack([r["omega_final_rad"] for r in results]),
        "q0_err_deg":        np.array([r["q0_err_deg"] for r in results]),
        "twin_err_deg":      np.array([r["twin_err_deg"] for r in results]),
        "omega_dir_err_deg": np.array([r["omega_dir_err_deg"] for r in results]),
        "omega_mag_err_pct": np.array([r["omega_mag_err_pct"] for r in results]),
        "final_mse":         np.array([r["final_mse"] for r in results]),
        "truth_basin_strict":np.array([r["truth_basin_strict"] for r in results]),
        "truth_basin_loose": np.array([r["truth_basin_loose"] for r in results]),
        "twin_basin_strict": np.array([r["twin_basin_strict"] for r in results]),
        "n_fev":             np.array([r["n_fev"] for r in results], dtype=int),
        "wall_s":            np.array([r["wall_s"] for r in results]),
        "success":           np.array([r["success"] for r in results]),
        "status":            np.array([r["status"] for r in results], dtype=int),
    }
    out = OUT_DIR / "runs.npz"
    np.savez(out, **arrs)
    print(f"Saved: {out}")


def save_summary(results: list[dict]):
    n = len(results)
    rows = [
        {
            "seed":            r["seed"],
            "label":           r["label"],
            "tier":            r["tier"],
            "kind":            r["kind"],
            "initial_q0_err_deg": round(r["initial_q0_err_deg"], 3),
            "initial_omega_dir_err_deg": round(r["initial_omega_dir_err_deg"], 3),
            "initial_omega_mag_err_pct": round(r["initial_omega_mag_err_pct"], 3),
            "initial_mse":     float(f"{r['initial_mse']:.4e}"),
            "q0_err_deg":      round(r["q0_err_deg"], 4),
            "twin_err_deg":    round(r["twin_err_deg"], 4),
            "omega_dir_err_deg": round(r["omega_dir_err_deg"], 4),
            "omega_mag_err_pct": round(r["omega_mag_err_pct"], 4),
            "final_mse":       float(f"{r['final_mse']:.4e}"),
            "truth_basin_strict": r["truth_basin_strict"],
            "truth_basin_loose":  r["truth_basin_loose"],
            "twin_basin_strict":  r["twin_basin_strict"],
            "n_fev":           r["n_fev"],
            "wall_s":          round(r["wall_s"], 2),
            "success":         r["success"],
            "status":          r["status"],
        }
        for r in results
    ]

    # Per-tier success rates (across seeds × tier).
    by_tier: dict = {}
    for r in results:
        if r["kind"] != "deterministic":
            continue
        by_tier.setdefault(r["label"], []).append(r)
    tier_summary = {}
    for label, runs in by_tier.items():
        n_runs = len(runs)
        n_strict = sum(1 for r in runs if r["truth_basin_strict"])
        n_loose = sum(1 for r in runs if r["truth_basin_loose"])
        n_twin = sum(1 for r in runs if r["twin_basin_strict"])
        tier_summary[label] = {
            "n_runs": n_runs,
            "n_truth_basin_strict": n_strict,
            "n_truth_basin_loose": n_loose,
            "n_twin_basin_strict": n_twin,
            "median_q0_err_deg": float(np.median([r["q0_err_deg"] for r in runs])),
            "median_omega_dir_err_deg": float(np.median([r["omega_dir_err_deg"] for r in runs])),
            "median_omega_mag_err_pct": float(np.median([r["omega_mag_err_pct"] for r in runs])),
            "median_final_mse": float(np.median([r["final_mse"] for r in runs])),
            "median_wall_s": float(np.median([r["wall_s"] for r in runs])),
        }

    # Random-IC summary.
    random_runs = [r for r in results if r["kind"] == "random"]
    random_summary = {
        "n_runs": len(random_runs),
        "n_truth_basin_strict": sum(1 for r in random_runs if r["truth_basin_strict"]),
        "n_truth_basin_loose": sum(1 for r in random_runs if r["truth_basin_loose"]),
        "n_twin_basin_strict": sum(1 for r in random_runs if r["twin_basin_strict"]),
    }

    # Per-seed success rates.
    by_seed: dict = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r)
    seed_summary = {}
    for s, runs in by_seed.items():
        seed_summary[str(s)] = {
            "n_runs": len(runs),
            "n_truth_basin_strict": sum(1 for r in runs if r["truth_basin_strict"]),
            "n_truth_basin_loose": sum(1 for r in runs if r["truth_basin_loose"]),
            "truth_mse_ref": TRUTH_MSE_REF.get(s, None),
            "median_final_mse": float(np.median([r["final_mse"] for r in runs])),
            "min_final_mse": float(np.min([r["final_mse"] for r in runs])),
            "max_final_mse": float(np.max([r["final_mse"] for r in runs])),
        }

    summary = {
        "seeds": SEEDS,
        "n_runs_total": n,
        "n_truth_basin_strict": sum(1 for r in results if r["truth_basin_strict"]),
        "n_truth_basin_loose": sum(1 for r in results if r["truth_basin_loose"]),
        "n_twin_basin_strict": sum(1 for r in results if r["twin_basin_strict"]),
        "by_tier": tier_summary,
        "random_inside_tube": random_summary,
        "by_seed": seed_summary,
        "convergence_definition": {
            "strict": "q0_err<5° AND ω_dir_err<1° AND |ω_mag_err|<5%",
            "loose":  "q0_err<10° AND ω_dir_err<2° AND |ω_mag_err|<10%",
            "twin":   "twin_err<5° AND ω_dir_err<1° AND |ω_mag_err|<5%",
        },
        "rows": rows,
    }
    out = OUT_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


def save_plots(results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── convergence.png — per-tier truth-basin success rate ──
    fig, ax = plt.subplots(figsize=(9, 5))
    by_tier_label: dict = {}
    for r in results:
        if r["kind"] == "deterministic":
            by_tier_label.setdefault(r["label"], []).append(r)
    tier_labels = [t[0] for t in DETERMINISTIC_TIERS]
    pcts_strict = []
    pcts_loose = []
    pcts_twin = []
    for tl in tier_labels:
        runs = by_tier_label.get(tl, [])
        n = len(runs)
        if n == 0:
            pcts_strict.append(0.0)
            pcts_loose.append(0.0)
            pcts_twin.append(0.0)
        else:
            pcts_strict.append(100.0 * sum(1 for r in runs if r["truth_basin_strict"]) / n)
            pcts_loose.append(100.0 * sum(1 for r in runs if r["truth_basin_loose"]) / n)
            pcts_twin.append(100.0 * sum(1 for r in runs if r["twin_basin_strict"]) / n)
    x = np.arange(len(tier_labels))
    w = 0.27
    ax.bar(x - w, pcts_strict, w, label="strict (q0<5°,ωd<1°,|ωm|<5%)",
           color="tab:green")
    ax.bar(x,     pcts_loose,  w, label="loose  (q0<10°,ωd<2°,|ωm|<10%)",
           color="tab:olive")
    ax.bar(x + w, pcts_twin,   w, label="twin (q_180x · q0_truth)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("convergence rate across 5 seeds  [%]")
    ax.set_ylim(0, 105)
    ax.set_title("s005 — joint LM convergence vs initial-offset tier")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    p = OUT_DIR / "convergence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")

    # ── error_panels.png — final errors vs initial offset ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    seeds_present = sorted({r["seed"] for r in results})
    cmap = {s: plt.cm.tab10(i) for i, s in enumerate(seeds_present)}

    # Panel 1: q0_err final vs initial (det tiers + random).
    ax = axes[0, 0]
    for s in seeds_present:
        runs_s = [r for r in results if r["seed"] == s]
        x = [r["initial_q0_err_deg"] for r in runs_s]
        y = [r["q0_err_deg"] for r in runs_s]
        ax.scatter(x, y, color=cmap[s], label=f"seed {s}", s=30, alpha=0.85)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(5.0, color="green", linestyle="--", alpha=0.5, label="basin (5°)")
    ax.axhline(10.0, color="olive", linestyle="--", alpha=0.5)
    ax.set_xlabel("initial q0 offset  [deg]")
    ax.set_ylabel("final q0 error  [deg]")
    ax.set_title("q0 convergence")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 2: ω_dir_err final vs initial.
    ax = axes[0, 1]
    for s in seeds_present:
        runs_s = [r for r in results if r["seed"] == s]
        x = [r["initial_omega_dir_err_deg"] for r in runs_s]
        y = [r["omega_dir_err_deg"] for r in runs_s]
        ax.scatter(x, y, color=cmap[s], label=f"seed {s}", s=30, alpha=0.85)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="basin (1°)")
    ax.axhline(2.0, color="olive", linestyle="--", alpha=0.5)
    ax.set_xlabel("initial ω-dir offset  [deg]")
    ax.set_ylabel("final ω-dir error  [deg]")
    ax.set_title("ω-direction convergence")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 3: ω_mag_err final vs initial.
    ax = axes[1, 0]
    for s in seeds_present:
        runs_s = [r for r in results if r["seed"] == s]
        x = [r["initial_omega_mag_err_pct"] for r in runs_s]
        y = [r["omega_mag_err_pct"] for r in runs_s]
        ax.scatter(x, y, color=cmap[s], label=f"seed {s}", s=30, alpha=0.85)
    ax.axhline(5.0, color="green", linestyle="--", alpha=0.5, label="basin (5%)")
    ax.axhline(-5.0, color="green", linestyle="--", alpha=0.5)
    ax.axhline(10.0, color="olive", linestyle="--", alpha=0.5)
    ax.axhline(-10.0, color="olive", linestyle="--", alpha=0.5)
    ax.set_xlabel("initial ω-mag offset  [%]")
    ax.set_ylabel("final ω-mag error  [%]")
    ax.set_title("ω-magnitude convergence")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 4: final_mse vs initial_mse.
    ax = axes[1, 1]
    for s in seeds_present:
        runs_s = [r for r in results if r["seed"] == s]
        x = [r["initial_mse"] for r in runs_s]
        y = [r["final_mse"] for r in runs_s]
        ax.scatter(x, y, color=cmap[s], label=f"seed {s}", s=30, alpha=0.85)
        ref = TRUTH_MSE_REF.get(s, None)
        if ref is not None:
            ax.axhline(ref, color=cmap[s], linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initial surrogate MSE  [mag²]")
    ax.set_ylabel("final surrogate MSE  [mag²]")
    ax.set_title("surrogate-MSE convergence  (dotted = truth-MSE per seed from s001)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("s005 — joint (q0, ω) LM descent: per-IC initial vs final errors",
                 fontsize=12)
    fig.tight_layout()
    p = OUT_DIR / "error_panels.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    main(smoke_only=smoke)
