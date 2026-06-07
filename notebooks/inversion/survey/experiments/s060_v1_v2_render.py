"""Render seed 28's truth trajectory under both v1 and v2 surrogates.

Uses the cached k1_body / k2_body from `data/trajectories/traj_seed028.npz` —
no propagation needed; the surrogate just maps body-frame coords to mag.

Output: two .npy files (v1_lc.npy, v2_lc.npy) under
    results/s060_v1_v2_compare/
which feed into two `/compare-lc` calls.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3].parent
SURVEY_ROOT = REPO_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_ROOT))

# v1 surrogate
sys.path.insert(0, "/home/girish/surrogate_model")
from surrogate import SurrogateModel as V1SurrogateModel  # noqa: E402

# v2 surrogate (via lib.surrogate_eval which already wraps it)
from lib.surrogate_eval import predict as v2_predict  # noqa: E402

SEED = 28
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0

traj_path = SURVEY_ROOT / "data" / "trajectories" / f"traj_seed{SEED:03d}.npz"
out_dir = SURVEY_ROOT / "results" / "s060_v1_v2_compare"
out_dir.mkdir(parents=True, exist_ok=True)

d = np.load(traj_path)
k1 = d["k1_body"]
k2 = d["k2_body"]
obs_dist = d["obs_dist"]
truth_mag = d["mag_hifi"]
N = len(truth_mag)
print(f"seed {SEED}: {N} epochs, |ω|_dps = {float(d['omega_mag_dps']):.3f}")
print(f"truth mag range: {truth_mag.min():.2f}..{truth_mag.max():.2f}")

# v1 surrogate prediction
v1_weights = "/home/girish/surrogate_model/s10_5M_weights.npz"
v1_norm = "/home/girish/surrogate_model/s10_5M_normalization.npz"
v1_model = V1SurrogateModel(v1_weights, v1_norm)
panel = np.full(N, SP_ANGLE_DEG)
dish = np.full(N, AD_ANGLE_DEG)
v1_mag = v1_model.predict_magnitude(k1, k2, panel, dish, obs_dist)
v1_resid = v1_mag - truth_mag
mask = np.isfinite(v1_resid)
v1_rms = float(np.sqrt(np.mean(v1_resid[mask] ** 2)))
print(f"v1 RMS = {v1_rms:.4f} mag,  ρ = {v1_rms/0.05:.3f}")

# v2 surrogate prediction
v2_mag = v2_predict(k1, k2, obs_dist, sp_angle_deg=SP_ANGLE_DEG, ad_angle_deg=AD_ANGLE_DEG)
v2_resid = v2_mag - truth_mag
mask = np.isfinite(v2_resid)
v2_rms = float(np.sqrt(np.mean(v2_resid[mask] ** 2)))
print(f"v2 RMS = {v2_rms:.4f} mag,  ρ = {v2_rms/0.05:.3f}")

np.save(out_dir / "v1_lc.npy", v1_mag)
np.save(out_dir / "v2_lc.npy", v2_mag)
print(f"Saved: {out_dir / 'v1_lc.npy'}")
print(f"Saved: {out_dir / 'v2_lc.npy'}")
