"""Check surrogate-at-truth residuals across constraint epochs of seed 91."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model/surrogate_model")

from scipy.spatial.transform import Rotation
from m138_isoshell_h1 import propagate_identity_batch, quat_mul, quat_to_R
from lib.traj_source import load_truth, CANONICAL_NOISE_SIGMA
from lib.experiment_setup import setup_experiment
from surrogate import SurrogateModel

seed = 91
truth = load_truth(seed, "m048")
true_q0 = truth["q0_wxyz"]
true_omega = truth["omega0_rad"]
I_tensor = truth["inertia_tensor"]
I_diag = np.diag(I_tensor) if I_tensor.shape == (3, 3) else np.asarray(I_tensor)
obs_lc = truth["observed_lc"]
mag_hifi = truth["mag_hifi"]

ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                        random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                        start_et=truth["start_et"], skip_true_lc=True)

obs_times = ctx.observation_times
sun_vecs = ctx.sun_pos - ctx.sat_pos
sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
obs_vecs = ctx.obs_pos - ctx.sat_pos
obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
obs_dist_km = ctx.obs_dist

model = SurrogateModel.load_default()

# Propagate truth at all 500 obs_times
q_world = propagate_identity_batch(true_omega[None, :], I_diag, obs_times, dt=2.0)[0]
q_truth = quat_mul(np.broadcast_to(true_q0, q_world.shape).copy(), q_world)
R_truth = quat_to_R(q_truth)
k1_body = np.einsum('nij,nj->ni', R_truth, sun_dirs)
k2_body = np.einsum('nij,nj->ni', R_truth, obs_dirs)
pred = model.predict_magnitude(k1_body, k2_body, 0.0, 15.0, obs_dist_km)
print(f"surrogate at truth: range [{np.nanmin(pred):.2f}, {np.nanmax(pred):.2f}]")

residual_truth_minus_hifi = pred - mag_hifi
residual_truth_minus_obs = pred - obs_lc

print(f"\nsurrogate(truth) - mag_hifi (no noise) residual:")
print(f"  MAE = {np.nanmean(np.abs(residual_truth_minus_hifi)):.4f}")
print(f"  p50 = {np.nanmedian(np.abs(residual_truth_minus_hifi)):.4f}")
print(f"  p90 = {np.nanpercentile(np.abs(residual_truth_minus_hifi), 90):.4f}")
print(f"  p99 = {np.nanpercentile(np.abs(residual_truth_minus_hifi), 99):.4f}")
print(f"  max = {np.nanmax(np.abs(residual_truth_minus_hifi)):.4f}")

print(f"\nsurrogate(truth) - observed_lc (with noise) residual:")
print(f"  MAE = {np.nanmean(np.abs(residual_truth_minus_obs)):.4f}")
print(f"  p50 = {np.nanmedian(np.abs(residual_truth_minus_obs)):.4f}")
print(f"  p90 = {np.nanpercentile(np.abs(residual_truth_minus_obs), 90):.4f}")
print(f"  p99 = {np.nanpercentile(np.abs(residual_truth_minus_obs), 99):.4f}")
print(f"  max = {np.nanmax(np.abs(residual_truth_minus_obs)):.4f}")

# Bright epochs only
bright_idx = np.where((obs_lc < 11) & np.isfinite(obs_lc))[0]
abs_res_bright = np.abs(residual_truth_minus_obs[bright_idx])
print(f"\nBright epochs (mag<11), n={len(bright_idx)}:")
print(f"  MAE  = {np.nanmean(abs_res_bright):.4f}")
print(f"  p90  = {np.nanpercentile(abs_res_bright, 90):.4f}")
print(f"  max  = {np.nanmax(abs_res_bright):.4f}")
print(f"  frac > 0.12 (2σ tol)  = {(abs_res_bright > 0.12).mean():.3f}")
print(f"  frac > 0.20  = {(abs_res_bright > 0.20).mean():.3f}")
print(f"  frac > 0.30  = {(abs_res_bright > 0.30).mean():.3f}")

# Show specific epoch residuals
print("\nFirst 30 bright epochs:")
for i in bright_idx[:30]:
    print(f"  t={i}: obs={obs_lc[i]:.2f}, hifi={mag_hifi[i]:.2f}, "
          f"surr(truth)={pred[i]:.2f}, |surr-obs|={abs(residual_truth_minus_obs[i]):.3f}")
