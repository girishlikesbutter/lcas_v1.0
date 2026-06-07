"""
Shared experiment setup module.

Eliminates the ~100-line boilerplate duplicated across inversion experiment
scripts. Provides one-call setup and common brightness evaluation helpers.
"""

import sys
import os
import json
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader, Satellite
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion

OBSERVER_ID = 399999


@dataclass
class ExperimentContext:
    """Holds all shared experiment state."""
    # Satellite model + physics
    satellite: Satellite = None
    inertia_tensor: NDArray = None
    art_matrices: Dict = field(default_factory=dict)

    # SPICE / config references
    config: object = None
    config_manager: object = None
    spice_handler: object = None
    epochs: NDArray = None

    # Observation geometry
    observation_times: NDArray = None
    dt_sampling: float = 0.0
    n_observations: int = 0
    sun_pos: NDArray = None
    obs_pos: NDArray = None
    sat_pos: NDArray = None
    obs_dist: NDArray = None

    # Ground truth
    true_q0: NDArray = None
    true_omega0: NDArray = None
    true_quaternions: NDArray = None
    true_lc: NDArray = None

    # Observed data
    observed_lc: NDArray = None
    noise_sigma: float = 0.05


def setup_experiment(
    n_observations: int = 100,
    noise_sigma: float = 0.05,
    random_seed: int = 42,
    true_omega_deg: tuple = (0.005, -0.003, 0.05),
    end_time_utc: Optional[str] = None,
    skip_true_lc: bool = False,
    true_q0_wxyz: Optional[NDArray] = None,
    true_omega0_rad: Optional[NDArray] = None,
    start_et: Optional[float] = None,
    duration_s: float = 3600.0,
) -> ExperimentContext:
    """
    One-call setup: config, satellite, SPICE, geometry, true LC, noise.

    Parameters
    ----------
    skip_true_lc : bool
        If True, skip the expensive hi-fi LC generation (~50s). Use this
        when the observed LC comes from a pre-computed dataset (e.g. m046).
        The satellite model, SPICE positions, and articulation matrices are
        still loaded and available.
    true_q0_wxyz : Optional[NDArray]
        If provided, use this (4,) wxyz scalar-first quaternion verbatim as
        ctx.true_q0 instead of the hardcoded axis=(0.6,0.3,0.8) angle=45
        default. When None (default), preserves pre-patch behaviour.
    true_omega0_rad : Optional[NDArray]
        If provided, use this (3,) body-frame angular velocity in rad/s
        verbatim as ctx.true_omega0 instead of the `true_omega_deg` default.
        When None (default), preserves pre-patch behaviour.
    start_et : Optional[float]
        If provided (seconds past J2000), overrides the config
        simulation_defaults.start_time and `end_time_utc`. Epochs span
        [start_et, start_et + duration_s]. This is the m048 path —
        per-seed random start times loaded from traj_source.load_truth.
        When None (default), preserves m046 behaviour (config start +
        end_time_utc override).
    duration_s : float
        Only used when start_et is provided. Default 3600.0 (matches
        m046's 1-hour window and m048's per-seed duration).

    Returns an ExperimentContext with everything populated.
    """
    ctx = ExperimentContext()
    ctx.n_observations = n_observations
    ctx.noise_sigma = noise_sigma

    # Config + satellite
    ctx.config_manager = RSO_ConfigManager(PROJECT_ROOT)
    ctx.config = ctx.config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
    metakernel_path = ctx.config_manager.get_metakernel_path(ctx.config)
    ctx.satellite = STLLoader.create_satellite_from_stl_config(
        config=ctx.config, config_manager=ctx.config_manager)

    # BRDF
    brdf_manager = BRDFManager(ctx.config)
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(ctx.satellite, brdf_manager)

    # Inertia
    component_masses = {
        'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0,
        'AD_East': 50.0, 'AD_West': 50.0,
    }
    inertia_result = compute_inertia_from_config(
        config=ctx.config, config_manager=ctx.config_manager,
        masses=component_masses,
        articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
    ctx.inertia_tensor = inertia_result.inertia_tensor

    # SPICE
    ctx.spice_handler = SpiceHandler()
    ctx.spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Epochs
    if start_et is not None:
        # m048 path: per-seed start_et from trajectory npz, fixed-duration window.
        epoch_start_et = float(start_et)
        epoch_end_et = epoch_start_et + float(duration_s)
    else:
        # m046 / legacy path: config start + end_time_utc override.
        epoch_start_et = ctx.spice_handler.utc_to_et(ctx.config.simulation_defaults.start_time)
        if end_time_utc is not None:
            epoch_end_et = ctx.spice_handler.utc_to_et(end_time_utc)
        else:
            epoch_end_et = ctx.spice_handler.utc_to_et(ctx.config.simulation_defaults.end_time)
    ctx.epochs = np.linspace(epoch_start_et, epoch_end_et, n_observations)
    ctx.observation_times = ctx.epochs - ctx.epochs[0]
    ctx.dt_sampling = ctx.observation_times[-1] / (n_observations - 1) if n_observations > 1 else 0.0

    # Observation geometry
    geometry_data = compute_observation_geometry(
        epochs=ctx.epochs,
        satellite_id=ctx.config.spice_config.satellite_id,
        observer_id=OBSERVER_ID,
        spice_handler=ctx.spice_handler,
        config=ctx.config)
    ctx.sun_pos = geometry_data['sun_positions']
    ctx.obs_pos = geometry_data['obs_positions']
    ctx.sat_pos = geometry_data['sat_positions']
    ctx.obs_dist = geometry_data['observer_distances']

    # Articulation (fixed angles)
    art_angles = {
        'SP_North': np.full(n_observations, 0.0),
        'SP_South': np.full(n_observations, 0.0),
        'AD_East': np.full(n_observations, 15.0),
        'AD_West': np.full(n_observations, 15.0),
    }
    ctx.art_matrices = compute_rotation_matrices_from_angles(art_angles, ctx.satellite)

    if skip_true_lc:
        return ctx

    # True attitude
    if true_q0_wxyz is not None:
        ctx.true_q0 = np.asarray(true_q0_wxyz, dtype=np.float64).copy()
    else:
        true_axis = np.array([0.6, 0.3, 0.8])
        true_axis /= np.linalg.norm(true_axis)
        true_angle_rad = np.deg2rad(45.0)
        ctx.true_q0 = np.array([
            np.cos(true_angle_rad / 2),
            *(np.sin(true_angle_rad / 2) * true_axis)])
    if true_omega0_rad is not None:
        ctx.true_omega0 = np.asarray(true_omega0_rad, dtype=np.float64).copy()
    else:
        ctx.true_omega0 = np.deg2rad(np.array(true_omega_deg))

    ctx.true_quaternions, _ = propagate_attitude(
        q0=ctx.true_q0, omega0=ctx.true_omega0,
        times=ctx.observation_times,
        mode="tumbling", inertia_tensor=ctx.inertia_tensor)

    # Generate hi-fi observed lightcurve
    obj_temp = ObjectiveFunction(
        satellite=ctx.satellite,
        observation_times=ctx.observation_times,
        observed_lightcurve=np.zeros(n_observations),
        sun_positions_j2000=ctx.sun_pos,
        observer_positions_j2000=ctx.obs_pos,
        satellite_positions_j2000=ctx.sat_pos,
        observer_distances=ctx.obs_dist,
        compute_shadows_flag=True,
        articulation_matrices=ctx.art_matrices,
        mode="tumbling",
        inertia_tensor=ctx.inertia_tensor,
        show_progress=False)

    true_k1, true_k2 = obj_temp._compute_body_frame_vectors(ctx.true_quaternions)
    true_lit = compute_shadows(
        satellite=ctx.satellite, k1_vectors=true_k1,
        explicit_component_matrices=ctx.art_matrices, show_progress=False)
    ctx.true_lc, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=true_lit, k1_vectors_array=true_k1,
        k2_vectors_array=true_k2, observer_distances=ctx.obs_dist,
        satellite=ctx.satellite, epochs=ctx.epochs,
        pre_computed_matrices=ctx.art_matrices, show_progress=False)

    # Add noise
    np.random.seed(random_seed)
    ctx.observed_lc = ctx.true_lc + np.random.normal(0, noise_sigma, n_observations)

    return ctx


def brightness_single_epoch(
    q_wxyz: NDArray,
    epoch_idx: int,
    ctx: ExperimentContext,
    use_shadows: bool = False,
) -> float:
    """
    Evaluate brightness for one quaternion at one epoch.

    Parameters
    ----------
    q_wxyz : array (4,)
        Quaternion in scalar-first (w,x,y,z) convention.
    epoch_idx : int
        Index into ctx.epochs / ctx.observation_times.
    ctx : ExperimentContext
        Shared experiment state.
    use_shadows : bool
        False → lo-fi (no ray tracing), True → hi-fi (full ray tracing).

    Returns
    -------
    float
        Apparent magnitude at the given epoch.
    """
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (ctx.sun_pos[epoch_idx] - ctx.sat_pos[epoch_idx])
    s = s / np.linalg.norm(s)
    o = R @ (ctx.obs_pos[epoch_idx] - ctx.sat_pos[epoch_idx])
    o = o / np.linalg.norm(o)

    art_slice = {c: m[epoch_idx:epoch_idx + 1] for c, m in ctx.art_matrices.items()}

    if use_shadows:
        lit = compute_shadows(
            satellite=ctx.satellite, k1_vectors=s.reshape(1, 3),
            explicit_component_matrices=art_slice, show_progress=False)
    else:
        lit = create_no_shadow_lit_status(ctx.satellite, 1)

    mag, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=s.reshape(1, 3),
        k2_vectors_array=o.reshape(1, 3),
        observer_distances=np.array([ctx.obs_dist[epoch_idx]]),
        satellite=ctx.satellite,
        epochs=np.array([0.0]),
        pre_computed_matrices=art_slice,
        generate_no_shadow=False, animate=False, show_progress=False)
    return float(mag[0])


def attitude_error_deg(q_found: NDArray, q_true: NDArray) -> float:
    """Geodesic distance between two quaternions in degrees."""
    R_found = Rotation.from_quat([q_found[1], q_found[2], q_found[3], q_found[0]])
    R_true = Rotation.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]])
    return float(np.rad2deg((R_found.inv() * R_true).magnitude()))


def save_results(filepath, data):
    """Atomic JSON save (write to .tmp then rename)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(filepath)
