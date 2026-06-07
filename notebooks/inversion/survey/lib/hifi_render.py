"""Hi-fi forward model — non-truth (q0, ω) → predicted hi-fi LC.

Copy-adapted from `notebooks/inversion/lib/lc_compare.py:generate_hifi_lc`
(reference) using survey-local substrate: `lib.traj_load` for cached SPICE
state + `src.*` for the validated post-fix forward model. Per the workspace
contract (`survey/CLAUDE.md`), we do NOT import from
`notebooks/inversion/lib/`. Instead the assembly lives here.

Architecture decision: the satellite/BRDF/inertia don't depend on seed
(properties of the m048 forward model), so they're built once at module
level and reused. Per-seed context loads the cached SPICE state directly
from `traj_seedXXX.npz` — no SPICE roundtrip needed for rendering.

Cached vs computed:
- Cached in NPZ:  observation_times, sun_pos, obs_pos, sat_pos, obs_dist
- Computed:       satellite (STL+BRDF), inertia_tensor, art_matrices
                  (deterministic from config; matches m048 generator)

Smoke test: round-trip (q0_truth, ω0_truth) must reproduce
`traj_seedXXX.npz['mag_hifi']` to machine precision. Smoke-tested
on seeds 6/10/91 (run this file as `python -m lib.hifi_render` from
the survey workspace, or invoke `_smoke_test()` from a script).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SURVEY_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_DIR))

from src.config.rso_config_manager import RSO_ConfigManager  # noqa: E402
from src.computation.brdf import BRDFManager, BRDFCalculator  # noqa: E402
from src.computation.inertia_calculator import compute_inertia_from_config  # noqa: E402
from src.computation.shadow_engine import compute_shadows  # noqa: E402
from src.computation.lightcurve_generator import generate_lightcurves  # noqa: E402
from src.io.stl_loader import STLLoader  # noqa: E402
from src.articulation import compute_rotation_matrices_from_angles  # noqa: E402

from lib.forward import propagate_to_body_frame  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — must match the m048 generation pipeline exactly. Source:
# notebooks/inversion/09_glint_analysis/m048_generate_trajectories_v2.py:74-95
# and lines 152-154.
# ---------------------------------------------------------------------------

CONFIG_PATH = "intelsat_901/intelsat_901_config.yaml"

COMPONENT_MASSES = {
    "Bus": 1532.0,
    "SP_North": 170.0,
    "SP_South": 170.0,
    "AD_East": 50.0,
    "AD_West": 50.0,
}

INERTIA_ART_ANGLES_DEG = {"SP_North": 0.0, "SP_South": 0.0}

ART_ANGLES_DEG = {
    "SP_North": 0.0,
    "SP_South": 0.0,
    "AD_East": 15.0,
    "AD_West": 15.0,
}

# ---------------------------------------------------------------------------
# Module-level cache: satellite + BRDF + inertia. Heavy to build (~1 s for STL
# load + BRDF binding); reused across renders.
# ---------------------------------------------------------------------------

_MODEL_CACHE: Optional[tuple] = None


def _build_model() -> tuple:
    """Load satellite + BRDF + inertia. Cached at module level.

    Returns
    -------
    (satellite, inertia_tensor) tuple.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config(CONFIG_PATH)
    satellite = STLLoader.create_satellite_from_stl_config(
        config=config, config_manager=config_manager
    )
    brdf_manager = BRDFManager(config)
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

    inertia_result = compute_inertia_from_config(
        config=config,
        config_manager=config_manager,
        masses=COMPONENT_MASSES,
        articulation_angles=INERTIA_ART_ANGLES_DEG,
    )
    _MODEL_CACHE = (satellite, inertia_result.inertia_tensor)
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Per-seed context — light, holds cached SPICE state + per-n_obs art matrices.
# ---------------------------------------------------------------------------


def build_context(seed: int) -> dict:
    """Build a per-seed render context.

    Loads the satellite + inertia from config (module-cached) and the
    per-seed observation geometry from the cached truth NPZ. Articulation
    matrices are deterministic (fixed angles) and built per-call (cheap).

    Parameters
    ----------
    seed : int
        m048 seed index (0..99).

    Returns
    -------
    dict with keys:
        satellite, inertia_tensor, art_matrices,
        sun_pos, obs_pos, sat_pos, obs_dist, observation_times,
        q0_truth, omega0_truth_rad, mag_hifi_truth, seed.
    """
    satellite, inertia_tensor = _build_model()
    truth = load_truth(seed)
    n_obs = int(truth["observation_times"].shape[0])

    art_angles = {c: np.full(n_obs, ART_ANGLES_DEG[c]) for c in ART_ANGLES_DEG}
    art_matrices = compute_rotation_matrices_from_angles(art_angles, satellite)

    return {
        "satellite": satellite,
        "inertia_tensor": inertia_tensor,
        "art_matrices": art_matrices,
        "sun_pos": truth["sun_pos"],
        "obs_pos": truth["obs_pos"],
        "sat_pos": truth["sat_pos"],
        "obs_dist": truth["obs_dist"],
        "observation_times": truth["observation_times"],
        "q0_truth": truth["q0_wxyz"],
        "omega0_truth_rad": truth["omega0_rad"],
        "mag_hifi_truth": truth["mag_hifi"],
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Render: (q0, ω) → hi-fi LC magnitudes.
# ---------------------------------------------------------------------------


def render_hifi(
    q0_wxyz: np.ndarray,
    omega0_rad: np.ndarray,
    ctx: dict,
) -> np.ndarray:
    """Render the hi-fi LC for (q0, ω0) at this seed's geometry.

    Parameters
    ----------
    q0_wxyz : (4,)  initial quaternion (w, x, y, z).
    omega0_rad : (3,) initial body-frame ω in rad/s.
    ctx : dict from build_context(seed).

    Returns
    -------
    mag_hifi : (N,) magnitudes.
    """
    k1, k2, _ = propagate_to_body_frame(
        q0_wxyz=np.asarray(q0_wxyz, dtype=np.float64),
        omega0_rad=np.asarray(omega0_rad, dtype=np.float64),
        observation_times=ctx["observation_times"],
        sun_pos=ctx["sun_pos"],
        obs_pos=ctx["obs_pos"],
        sat_pos=ctx["sat_pos"],
        inertia_tensor=ctx["inertia_tensor"],
        mode="tumbling",
    )
    n_ep = k1.shape[0]
    lit = compute_shadows(
        satellite=ctx["satellite"],
        k1_vectors=k1,
        explicit_component_matrices=ctx["art_matrices"],
        show_progress=False,
    )
    pred_mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=k1,
        k2_vectors_array=k2,
        observer_distances=ctx["obs_dist"][:n_ep],
        satellite=ctx["satellite"],
        epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=ctx["art_matrices"],
        show_progress=False,
    )
    return np.asarray(pred_mags, dtype=np.float64)


# ---------------------------------------------------------------------------
# ρ-band classification helper (forward-model-only — does not import from
# the inversion side).
# ---------------------------------------------------------------------------


RHO_NOISE_SIGMA = 0.05  # canonical; matches concepts/rho_band.md and
# notebooks/inversion/lib/traj_source.py:CANONICAL_NOISE_SIGMA.


def rho_from_hifi(pred: np.ndarray, truth: np.ndarray) -> float:
    """ρ = √(MSE) / 0.05 between predicted and truth hi-fi LCs."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / RHO_NOISE_SIGMA)


def rho_band(rho: float) -> str:
    """Map ρ to A/B/C/D per concepts/rho_band.md.

    A: ρ < 2   — truth-grade fit, well below noise
    B: 2 ≤ ρ < 4 — acceptable (within ~2× noise)
    C: 4 ≤ ρ < 8 — marginal — visually presentable but not publishable
    D: ρ ≥ 8   — failure (LCs disagree)

    Acceptance bar for the survey: ρ < 4 (Band A∪B).
    """
    if rho < 2.0:
        return "A"
    if rho < 4.0:
        return "B"
    if rho < 8.0:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Smoke test — must reproduce cached mag_hifi to machine precision.
# ---------------------------------------------------------------------------


def _smoke_test(seeds=(6, 10, 91), tol_max_abs: float = 1e-9) -> dict:
    """Round-trip (q0_truth, ω0_truth) → mag_hifi vs cached mag_hifi.

    Returns a dict per seed with max_abs_diff and pass/fail flag.
    Raises AssertionError on first failure.
    """
    results = {}
    for seed in seeds:
        ctx = build_context(seed)
        pred = render_hifi(ctx["q0_truth"], ctx["omega0_truth_rad"], ctx)
        truth = ctx["mag_hifi_truth"]
        diff = pred - truth
        max_abs = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff ** 2)))
        ok = max_abs < tol_max_abs
        results[seed] = {"max_abs": max_abs, "rms": rms, "passed": ok}
        status = "PASS" if ok else "FAIL"
        print(
            f"  seed {seed:3d}: max|Δ|={max_abs:.3e} rms={rms:.3e} {status}"
        )
        if not ok:
            raise AssertionError(
                f"Smoke test FAILED on seed {seed}: max|Δ|={max_abs:.3e} > "
                f"tol={tol_max_abs:.0e}. Forward chain does not reproduce "
                f"cached mag_hifi to machine precision."
            )
    return results


if __name__ == "__main__":
    # Force single-threaded BLAS for reproducibility (avoid Pool issues if
    # imported in a multiprocessing context).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    print("Smoke testing hifi_render on seeds 6, 10, 91 ...")
    res = _smoke_test()
    print("All smoke tests passed.")
