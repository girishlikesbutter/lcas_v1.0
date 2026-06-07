"""Debug: evaluate the H1 cost at the truth-ω and diagnose density structure."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model/surrogate_model")

from sklearn.neighbors import BallTree
from m138_isoshell_h1 import (
    propagate_identity_batch, quat_mul, quat_conj, quat_canonicalise,
)
from lib.traj_source import load_truth

seed = 91
out_dir = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{seed:03d}"

# Reload levelset from the prior run
ls = np.load(out_dir / "levelset_ckpt.npz", allow_pickle=True)
constraint_idx = ls["constraint_idx"]
q_grid = ls["q_grid"]
kept_per_epoch = ls["kept_per_epoch"]

print(f"loaded levelset: {len(constraint_idx)} epochs, q_grid shape {q_grid.shape}")
n_kept = np.array([len(k) for k in kept_per_epoch])
print(f"kept counts: median={int(np.median(n_kept))}, min={n_kept.min()}, max={n_kept.max()}")

# Load truth, set up obs_times
truth = load_truth(seed, "m048")
true_q0 = truth["q0_wxyz"]
true_omega = truth["omega0_rad"]
I_tensor = truth["inertia_tensor"]
I_diag = np.diag(I_tensor) if I_tensor.shape == (3, 3) else np.asarray(I_tensor)
obs_times = truth["observation_times"]
t_eval = obs_times[constraint_idx]

# Now: evaluate cost at truth-ω and at a random-ω, both with q0=I
def eval_cost(omega_test, label, n_kept_per_epoch=999999, eps_cluster_deg=8.0,
              rng_seed=0):
    """Use kept points per epoch; cost = max unique-epoch count within eps."""
    eps_chord = 2.0 * np.sin(np.deg2rad(eps_cluster_deg) / 2.0)
    q_world = propagate_identity_batch(omega_test[None, :], I_diag, t_eval, dt=2.0)[0]
    qw_inv = quat_conj(q_world)
    rng = np.random.default_rng(rng_seed)
    cloud_pts = []
    epoch_labels = []  # per q0_hat: which epoch did it come from
    for ti, kept_idx in enumerate(kept_per_epoch):
        if len(kept_idx) == 0:
            continue
        n_take = min(n_kept_per_epoch, len(kept_idx))
        sel = rng.choice(len(kept_idx), n_take, replace=False)
        q_kept_t = quat_canonicalise(q_grid[kept_idx[sel]])
        qw_inv_t = np.broadcast_to(qw_inv[ti], q_kept_t.shape).copy()
        q_hat = quat_mul(q_kept_t, qw_inv_t)
        cloud_pts.append(q_hat)
        epoch_labels.extend([ti] * q_hat.shape[0])
    cloud = np.concatenate(cloud_pts, axis=0)
    cloud = quat_canonicalise(cloud)
    epoch_labels = np.array(epoch_labels)
    n_total = cloud.shape[0]
    print(f"\n=== {label}: {n_total} q0_hat hypotheses ===")

    # Distance from each q0_hat to true_q0:
    true_q0_canon = quat_canonicalise(true_q0[None, :])[0]
    dots = np.abs(cloud @ true_q0_canon)
    geo_to_truth = 2.0 * np.degrees(np.arccos(dots.clip(-1, 1)))
    print(f"  geodesic q0_hat → q0_truth (deg): "
          f"min={geo_to_truth.min():.2f}, "
          f"<10°={(geo_to_truth<10).sum()}, "
          f"<5°={(geo_to_truth<5).sum()}, "
          f"<2°={(geo_to_truth<2).sum()}")

    tree = BallTree(cloud)
    # Get all neighbours of each point within eps
    neigh_idx_list = tree.query_radius(cloud, r=eps_chord)
    # For each query point, count unique epochs in the neighbourhood
    unique_epoch_counts = np.array([len(np.unique(epoch_labels[ni]))
                                     for ni in neigh_idx_list])
    raw_counts = np.array([len(ni) for ni in neigh_idx_list])
    print(f"  max raw density: {raw_counts.max()}, median: {np.median(raw_counts):.1f}")
    print(f"  max UNIQUE-EPOCH count (within {eps_cluster_deg}°): {unique_epoch_counts.max()}")
    print(f"  median unique-epochs: {np.median(unique_epoch_counts):.1f}")
    # Show top-3 query points
    top3 = np.argsort(unique_epoch_counts)[-3:][::-1]
    for tk in top3:
        d = 2.0 * np.degrees(np.arccos(abs(cloud[tk] @ quat_canonicalise(true_q0[None, :])[0]).clip(-1,1)))
        print(f"    top hit unique={unique_epoch_counts[tk]}, dist_to_q0_truth={d:.2f}°")
    return float(unique_epoch_counts.max())

# truth-ω
truth_max = eval_cost(true_omega, "TRUTH ω")

# random ω with similar magnitude
rng = np.random.default_rng(123)
random_dir = rng.normal(size=3)
random_dir /= np.linalg.norm(random_dir)
random_omega = random_dir * np.linalg.norm(true_omega)
rand_max = eval_cost(random_omega, "RANDOM ω (same |ω|)")

# Truth-ω with wrong magnitude
truth_wrong_mag = true_omega / np.linalg.norm(true_omega) * 0.05
wrong_mag_max = eval_cost(truth_wrong_mag, "TRUTH dir, 2x mag")

print(f"\nSummary: truth_max={truth_max}, random_max={rand_max}, wrong_mag_max={wrong_mag_max}")
