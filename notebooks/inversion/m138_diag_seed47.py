"""Combined orientation diagnostic for m138 seed 47.

Items (11) surrogate-residual at truth across seed 47 epochs and (12) cost-at-
truth-omega on the saved seed_047 levelset. Mirrors m138_check_residuals.py +
m138_debug_cost.py but pointed at seed 47 with multiple eps_cluster sweeps.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model/surrogate_model")

from sklearn.neighbors import BallTree
from m138_isoshell_h1 import (
    propagate_identity_batch, quat_mul, quat_conj, quat_canonicalise, quat_to_R,
)
from lib.traj_source import load_truth, CANONICAL_NOISE_SIGMA
from lib.experiment_setup import setup_experiment
from surrogate import SurrogateModel

SEED = 47
out_dir = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}"

print(f"=== Item 11: surrogate residual at truth, seed {SEED} ===")
truth = load_truth(SEED, "m048")
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
sun_dirs = (ctx.sun_pos - ctx.sat_pos)
sun_dirs = sun_dirs / np.linalg.norm(sun_dirs, axis=1, keepdims=True)
obs_dirs = (ctx.obs_pos - ctx.sat_pos)
obs_dirs = obs_dirs / np.linalg.norm(obs_dirs, axis=1, keepdims=True)
obs_dist_km = ctx.obs_dist
model = SurrogateModel.load_default()

q_world = propagate_identity_batch(true_omega[None, :], I_diag, obs_times, dt=2.0)[0]
q_truth = quat_mul(np.broadcast_to(true_q0, q_world.shape).copy(), q_world)
R_truth = quat_to_R(q_truth)
k1_body = np.einsum('nij,nj->ni', R_truth, sun_dirs)
k2_body = np.einsum('nij,nj->ni', R_truth, obs_dirs)
pred = model.predict_magnitude(k1_body, k2_body, 0.0, 15.0, obs_dist_km)
res_obs = pred - obs_lc
res_hifi = pred - mag_hifi

bright = np.where((obs_lc < 11) & np.isfinite(obs_lc))[0]
abs_res_b = np.abs(res_obs[bright])
print(f"  n bright<11 epochs: {len(bright)}")
print(f"  surrogate vs hifi (no noise): MAE={np.nanmean(np.abs(res_hifi)):.4f}, "
      f"p90={np.nanpercentile(np.abs(res_hifi),90):.4f}, max={np.nanmax(np.abs(res_hifi)):.4f}")
print(f"  bright residual vs obs: MAE={abs_res_b.mean():.4f}, p90={np.percentile(abs_res_b,90):.4f}, max={abs_res_b.max():.4f}")
print(f"  fraction bright > 0.12 (2sigma) = {(abs_res_b > 0.12).mean():.3f}")
print(f"  fraction bright > 0.20         = {(abs_res_b > 0.20).mean():.3f}")
print(f"  fraction bright > 0.30         = {(abs_res_b > 0.30).mean():.3f}")

# Also: relax to mag<13 cap to see how many more constraints become available
bright13 = np.where((obs_lc < 13) & np.isfinite(obs_lc))[0]
abs_res_b13 = np.abs(res_obs[bright13])
print(f"\n  AT MAG<13 CAP: n epochs = {len(bright13)} (vs {len(bright)} at mag<11)")
print(f"  bright<13 residual vs obs: MAE={abs_res_b13.mean():.4f}, p90={np.percentile(abs_res_b13,90):.4f}, max={abs_res_b13.max():.4f}")
print(f"  fraction bright<13 > 0.12 = {(abs_res_b13 > 0.12).mean():.3f}")

print(f"\n=== Item 12: cost at truth-omega on saved seed {SEED} levelset ===")
ls = np.load(out_dir / "levelset_ckpt.npz", allow_pickle=True)
constraint_idx = ls["constraint_idx"]
q_grid = ls["q_grid"]
kept_per_epoch = ls["kept_per_epoch"]
n_kept = np.array([len(k) for k in kept_per_epoch])
print(f"  loaded {len(constraint_idx)} epochs")
print(f"  n_kept median={int(np.median(n_kept))}, min={n_kept.min()}, max={n_kept.max()}, sum={n_kept.sum()}")

# Per-epoch oracle: distance from q_truth(t) to nearest kept-q
constraint_t_eval = obs_times[constraint_idx]
q_world_constr = propagate_identity_batch(true_omega[None, :], I_diag, constraint_t_eval, dt=2.0)[0]
q_truth_constr = quat_mul(np.broadcast_to(true_q0, q_world_constr.shape).copy(), q_world_constr)
q_truth_canon = quat_canonicalise(q_truth_constr)
truth_dist = []
for ti, kept_idx in enumerate(kept_per_epoch):
    if len(kept_idx) == 0:
        truth_dist.append(180.0); continue
    q_kept = quat_canonicalise(q_grid[kept_idx])
    dots = np.abs(q_kept @ q_truth_canon[ti]).max()
    truth_dist.append(2.0 * np.degrees(np.arccos(np.clip(dots, -1, 1))))
truth_dist = np.array(truth_dist)
print(f"  truth-trajectory distance to nearest kept-q per epoch (deg):")
print(f"    median={np.median(truth_dist):.2f}, p90={np.percentile(truth_dist,90):.2f}, max={truth_dist.max():.2f}")
print(f"    epochs ≤ 5° = {(truth_dist<=5).sum()}/{len(truth_dist)}")
print(f"    epochs ≤ 10° = {(truth_dist<=10).sum()}/{len(truth_dist)}")
print(f"    epochs ≤ 15° = {(truth_dist<=15).sum()}/{len(truth_dist)}")


def eval_cost(omega_test, label, eps_cluster_deg):
    """Evaluate H1 cost at omega_test for given eps_cluster_deg."""
    eps_chord = 2.0 * np.sin(np.deg2rad(eps_cluster_deg) / 2.0)
    q_world = propagate_identity_batch(omega_test[None, :], I_diag, constraint_t_eval, dt=2.0)[0]
    qw_inv = quat_conj(q_world)
    cloud_pts = []
    epoch_labels = []
    for ti, kept_idx in enumerate(kept_per_epoch):
        if len(kept_idx) == 0: continue
        q_kept_t = quat_canonicalise(q_grid[kept_idx])
        qw_inv_t = np.broadcast_to(qw_inv[ti], q_kept_t.shape).copy()
        q_hat = quat_mul(q_kept_t, qw_inv_t)
        cloud_pts.append(q_hat)
        epoch_labels.extend([ti] * q_hat.shape[0])
    cloud = quat_canonicalise(np.concatenate(cloud_pts, axis=0))
    epoch_labels = np.array(epoch_labels)
    tree = BallTree(cloud)
    neigh = tree.query_radius(cloud, r=eps_chord)
    unique_counts = np.array([len(np.unique(epoch_labels[ni])) for ni in neigh])
    raw_counts = np.array([len(ni) for ni in neigh])
    # Distance of cloud points to true_q0 (for diagnostic of whether top hit IS truth)
    dots = np.abs(cloud @ quat_canonicalise(true_q0[None, :])[0])
    geo_to_truth = 2.0 * np.degrees(np.arccos(dots.clip(-1, 1)))
    top_hits = np.argsort(unique_counts)[-3:][::-1]
    print(f"\n  {label} (eps={eps_cluster_deg}°):")
    print(f"    cloud pts: {len(cloud)}; max_unique={unique_counts.max()}; median={np.median(unique_counts):.1f}; p99={np.percentile(unique_counts,99):.1f}")
    for tk in top_hits:
        print(f"    top hit: unique={unique_counts[tk]}, dist_q0_truth={geo_to_truth[tk]:.2f}°, raw_density={raw_counts[tk]}")
    print(f"    closest cloud pt to q0_truth: {geo_to_truth.min():.2f}° (its unique-count={unique_counts[geo_to_truth.argmin()]})")
    return int(unique_counts.max())

print(f"\n=== Cost at TRUTH ω vs RANDOM ω vs |ω| sweeps ===")
rng = np.random.default_rng(123)
random_dir = rng.normal(size=3); random_dir /= np.linalg.norm(random_dir)
random_omega = random_dir * np.linalg.norm(true_omega)

for eps in [10.0, 8.0, 5.0, 3.0]:
    print(f"\n--- eps_cluster_deg = {eps}° ---")
    eval_cost(true_omega, "TRUTH ω", eps)
    eval_cost(random_omega, "RANDOM ω same |ω|", eps)
    # Truth dir, +5% mag, -5% mag
    eval_cost(true_omega * 1.05, "TRUTH dir, +5% mag", eps)
    eval_cost(true_omega * 0.95, "TRUTH dir, -5% mag", eps)

# Check |omega|-grid coverage at the production grid
print(f"\n=== |ω|-grid coverage check ===")
from m138_isoshell_h1 import estimate_omega_mag_grid
omega_mags = estimate_omega_mag_grid(obs_lc, ctx.observation_times)
truth_mag = np.linalg.norm(true_omega)
print(f"  truth |ω| = {truth_mag:.5f}")
print(f"  |ω|-grid range: [{omega_mags[0]:.5f}, {omega_mags[-1]:.5f}], n={len(omega_mags)}")
nearest_mag = omega_mags[np.argmin(np.abs(omega_mags - truth_mag))]
print(f"  nearest grid |ω| = {nearest_mag:.5f}, error = {(nearest_mag-truth_mag)/truth_mag*100:+.2f}%")
print(f"  grid step (median ratio): {np.median(omega_mags[1:]/omega_mags[:-1])-1:+.4f}")
