"""
micro01: Peak omega recovery from consecutive quaternion pairs.

For 4 starting quaternions, propagate true attitude, find the first
brightness peak, recover omega from consecutive q-pairs near the peak,
and compare reconstructed LCs to truth.
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from notebooks.inversion.lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

OUT = Path("data/results/inversion_diagnostics")

# ── Hardcoded starting quaternions (w,x,y,z) ────────────────────────
def aa_q(axis, angle_deg):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    h = np.deg2rad(angle_deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * a)])

Q0S = [
    np.array([1., 0., 0., 0.]),           # (a) identity
    aa_q([1, 0, 0], 30),                   # (b) 30° about x
    aa_q([0, 1, 1], 90),                   # (c) 90° about [0,1,1]/norm
    aa_q([0.5, -0.3, 0.7], 160),           # (d) 160° about arbitrary axis
]
LABELS = ["identity", "x30", "yz90", "arb160"]

# ── Setup (satellite, geometry, inertia) ─────────────────────────────
ctx = setup_experiment(
    n_observations=100,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc="2020-02-05T10:02:00",
)

def body_vecs(quats, idx):
    """k1, k2 body-frame vectors for quaternions at epoch indices idx."""
    n = len(idx)
    k1, k2 = np.empty((n, 3)), np.empty((n, 3))
    for i, j in enumerate(idx):
        R = Rotation.from_quat([quats[i,1], quats[i,2], quats[i,3], quats[i,0]]).as_matrix()
        s = R @ (ctx.sun_pos[j] - ctx.sat_pos[j]); k1[i] = s / np.linalg.norm(s)
        o = R @ (ctx.obs_pos[j] - ctx.sat_pos[j]); k2[i] = o / np.linalg.norm(o)
    return k1, k2

def hifi_lc(quats, idx):
    """Hi-fi lightcurve for quaternions at epoch indices idx."""
    k1, k2 = body_vecs(quats, idx)
    art = {c: m[idx] for c, m in ctx.art_matrices.items()}
    lit = compute_shadows(ctx.satellite, k1, explicit_component_matrices=art, show_progress=False)
    mag, *_ = generate_lightcurves(
        lit, k1, k2, ctx.obs_dist[idx], ctx.satellite,
        ctx.epochs[idx], pre_computed_matrices=art, show_progress=False)
    return mag

def recover_omega(q1, q2, dt):
    """Body-frame omega from two quaternions assuming constant rotation."""
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return (r1.inv() * r2).as_rotvec() / dt

# ── Main loop ────────────────────────────────────────────────────────
t = ctx.observation_times
all_idx = np.arange(ctx.n_observations)

for ti, (q0, label) in enumerate(zip(Q0S, LABELS)):
    print(f"\n{'='*60}\nTrajectory {ti}: {label}")

    # 1. Propagate true attitude and generate hi-fi LC
    quats, omegas = propagate_attitude(
        q0, ctx.true_omega0, t, mode="tumbling", inertia_tensor=ctx.inertia_tensor)
    true_lc = hifi_lc(quats, all_idx)

    # 1b. Add noise to the observed LC
    np.random.seed(42 + ti)
    observed_lc = true_lc + np.random.normal(0, ctx.noise_sigma, len(true_lc))

    # 2. Find first peak on the noisy observed LC (local min in mag = brightest)
    k = next((i for i in range(1, len(observed_lc) - 1)
              if observed_lc[i] < observed_lc[i-1] and observed_lc[i] < observed_lc[i+1]),
             int(np.argmin(observed_lc[1:-1])) + 1)
    print(f"  Peak k={k}, t={t[k]:.1f}s, mag_obs={observed_lc[k]:.2f} (true={true_lc[k]:.2f})")

    # 3. Time intervals around peak
    dt_L = t[k] - t[k-1]
    dt_R = t[k+1] - t[k]

    # 4. Recover omega from consecutive quaternion pairs
    om_L = recover_omega(quats[k-1], quats[k], dt_L)
    om_R = recover_omega(quats[k], quats[k+1], dt_R)

    # 5. Comparison table
    d = np.degrees
    print(f"\n  {'Pair':<14} {'Recovered (deg/s)':>30} {'True (deg/s)':>30} {'Err':>8}")
    for name, om, ri in [("left vs k-1", om_L, k-1), ("left vs k", om_L, k),
                          ("right vs k", om_R, k), ("right vs k+1", om_R, k+1)]:
        tr = omegas[ri]
        print(f"  {name:<14} [{d(om[0]):+8.5f},{d(om[1]):+8.5f},{d(om[2]):+8.5f}]"
              f"  [{d(tr[0]):+8.5f},{d(tr[1]):+8.5f},{d(tr[2]):+8.5f}]"
              f"  {d(np.linalg.norm(om - tr)):.6f}")

    # 6. Propagate 4 reconstructed LCs from peak neighborhood
    cases = [("a", k-1, om_L), ("b", k, om_L), ("c", k, om_R), ("d", k+1, om_R)]
    for cl, si, om in cases:
        t_fwd = t[si:] - t[si]
        idx_fwd = np.arange(si, ctx.n_observations)
        if len(t_fwd) < 2:
            q_rec = quats[si:si+1]
        else:
            q_rec, _ = propagate_attitude(
                quats[si], om, t_fwd, mode="tumbling", inertia_tensor=ctx.inertia_tensor)
        rec_lc = hifi_lc(q_rec, idx_fwd)

        # 7. Plot noisy observed vs clean reconstructed
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, observed_lc, 'k-', lw=0.8, alpha=0.6, label='Observed (noisy)')
        ax.plot(t, true_lc, 'k--', lw=0.5, alpha=0.3, label='True (clean)')
        ax.plot(t[si:], rec_lc, 'r-', lw=1, label=f'Reconstructed ({cl})')
        ax.axvline(t[k], color='blue', ls=':', alpha=0.5, label=f'Peak k={k}')
        ax.set(xlabel='Time (s)', ylabel='Magnitude')
        ax.set_title(f'Traj {ti} ({label}) — case {cl} [noisy, σ={ctx.noise_sigma}]')
        ax.legend(fontsize=8); ax.invert_yaxis()
        om_tag = f"{np.linalg.norm(ctx.true_omega0)*180/np.pi:.2f}dps"
        fig.savefig(OUT / f"micro01_peak_omega_{om_tag}_noisy_traj{ti}_{cl}.png",
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

print("\nDone.")
