"""Worker module for parallel lo-fi objective evaluation."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# These get initialized once per worker process
_obj_lofi = None

def init_worker(config_path, n_obs, noise_sigma, observed_lc, obs_times,
                sun_pos, obs_pos, sat_pos, obs_dist, art_matrices, inertia, omega_bound_rad):
    """Initialize objective in each worker process."""
    global _obj_lofi, _bounds, _omega_bound_rad
    import os
    os.chdir(PROJECT_ROOT)
    
    from src.config.rso_config_manager import RSO_ConfigManager
    from src.io.stl_loader import STLLoader
    from src.computation.brdf import BRDFManager, BRDFCalculator
    from src.spice.spice_handler import SpiceHandler
    from src.computation import compute_inertia_from_config
    from src.articulation import compute_rotation_matrices_from_angles
    from src.inversion import ObjectiveFunction
    
    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config(config_path)
    metakernel_path = config_manager.get_metakernel_path(config)
    
    satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
    brdf_manager = BRDFManager(config)
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)
    
    spice_handler = SpiceHandler()
    spice_handler.load_metakernel_programmatically(str(metakernel_path))
    
    _obj_lofi = ObjectiveFunction(
        satellite=satellite,
        observation_times=obs_times,
        observed_lightcurve=observed_lc,
        sun_positions_j2000=sun_pos,
        observer_positions_j2000=obs_pos,
        satellite_positions_j2000=sat_pos,
        observer_distances=obs_dist,
        compute_shadows_flag=False,
        articulation_matrices=art_matrices,
        mode="tumbling",
        inertia_tensor=inertia,
    )
    _omega_bound_rad = omega_bound_rad


def evaluate_lofi(params):
    """Evaluate lo-fi objective — called by pool workers."""
    from src.inversion import axis_angle_to_quaternion, quaternion_to_axis_angle, normalize_quaternion
    
    aa = params[:3]
    omega = params[3:]
    q = axis_angle_to_quaternion(aa)
    q = normalize_quaternion(q)
    aa = quaternion_to_axis_angle(q)
    params_norm = np.concatenate([aa, omega])
    return _obj_lofi.evaluate(params_norm)
