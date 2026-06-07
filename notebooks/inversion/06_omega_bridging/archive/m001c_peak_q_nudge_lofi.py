"""
m001c: Peak omega recovery with quaternion nudge errors (lo-fi reconstructions).

Identical to m001b except the reconstructed LCs (step 6) are generated
in lo-fi (no shadows) instead of hi-fi.  The observed LC is still hi-fi.
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from notebooks.inversion.lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

OUT = Path("data/results/inversion_diagnostics")
N_DIRS = 5          # random nudge directions per magnitude
COLORS = ['r', 'orange', 'green', 'purple', 'deepskyblue']

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
NUDGE_DEGS = [0, 1, 2, 3, 4, 5]

# ── Pre-generate random nudge axes (deterministic) ───────────────────
rng = np.random.RandomState(123)
NUDGE_AXES = rng.randn(N_DIRS, 3)
NUDGE_AXES /= np.linalg.norm(NUDGE_AXES, axis=1, keepdims=True)

# ── Setup ────────────────────────────────────────────────────────────
ctx = setup_experiment(
    n_observations=100,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc="2020-02-05T10:02:00",
)

def body_vecs(quats, idx):
    n = len(idx)
    k1, k2 = np.empty((n, 3)), np.empty((n, 3))
    for i, j in enumerate(idx):
        R = Rotation.from_quat([quats[i,1], quats[i,2], quats[i,3], quats[i,0]]).as_matrix()
        s = R @ (ctx.sun_pos[j] - ctx.sat_pos[j]); k1[i] = s / np.linalg.norm(s)
        o = R @ (ctx.obs_pos[j] - ctx.sat_pos[j]); k2[i] = o / np.linalg.norm(o)
    return k1, k2

def hifi_lc(quats, idx):
    k1, k2 = body_vecs(quats, idx)
    art = {c: m[idx] for c, m in ctx.art_matrices.items()}
    lit = compute_shadows(ctx.satellite, k1, explicit_component_matrices=art, show_progress=False)
    mag, *_ = generate_lightcurves(
        lit, k1, k2, ctx.obs_dist[idx], ctx.satellite,
        ctx.epochs[idx], pre_computed_matrices=art, show_progress=False)
    return mag

def lofi_lc(quats, idx):
    k1, k2 = body_vecs(quats, idx)
    art = {c: m[idx] for c, m in ctx.art_matrices.items()}
    lit = create_no_shadow_lit_status(ctx.satellite, len(idx))
    mag, *_ = generate_lightcurves(
        lit, k1, k2, ctx.obs_dist[idx], ctx.satellite,
        ctx.epochs[idx], pre_computed_matrices=art, show_progress=False)
    return mag

def recover_omega(q1, q2, dt):
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return (r1.inv() * r2).as_rotvec() / dt

def nudge_quat(q_wxyz, axis, angle_deg):
    r_orig = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r_nudge = Rotation.from_rotvec(np.deg2rad(angle_deg) * axis)
    r_new = r_nudge * r_orig
    xyzw = r_new.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

# ── Main loop ────────────────────────────────────────────────────────
t = ctx.observation_times
all_idx = np.arange(ctx.n_observations)
om_tag = f"{np.linalg.norm(ctx.true_omega0)*180/np.pi:.2f}dps"

for ti, (q0, label) in enumerate(zip(Q0S, LABELS)):
    print(f"\n{'='*60}\nTrajectory {ti}: {label}")

    quats, omegas = propagate_attitude(
        q0, ctx.true_omega0, t, mode="tumbling", inertia_tensor=ctx.inertia_tensor)
    true_lc = hifi_lc(quats, all_idx)
    np.random.seed(42 + ti)
    observed_lc = true_lc + np.random.normal(0, ctx.noise_sigma, len(true_lc))

    k = next((i for i in range(1, len(observed_lc) - 1)
              if observed_lc[i] < observed_lc[i-1] and observed_lc[i] < observed_lc[i+1]),
             int(np.argmin(observed_lc[1:-1])) + 1)
    print(f"  Peak k={k}, t={t[k]:.1f}s")

    om_rec = recover_omega(quats[k-1], quats[k], t[k] - t[k-1])
    d = np.degrees
    print(f"  omega_rec: [{d(om_rec[0]):+.4f}, {d(om_rec[1]):+.4f}, {d(om_rec[2]):+.4f}] deg/s")

    idx_fwd = np.arange(k, ctx.n_observations)
    t_fwd = t[k:] - t[k]

    # Fix y-axis to the observed LC range so the hifi curve stays stationary
    obs_margin = 0.05 * (observed_lc.max() - observed_lc.min())
    ylim = (observed_lc.min() - obs_margin, observed_lc.max() + obs_margin)

    print(f"\n  {'Nudge°':<8} {'min MSE':>10} {'mean MSE':>10} {'max MSE':>10} {'spread':>10}")
    for nudge_deg in NUDGE_DEGS:
        n_dirs = 1 if nudge_deg == 0 else N_DIRS
        mses, rec_lcs = [], []

        for di in range(n_dirs):
            axis = NUDGE_AXES[di] if nudge_deg > 0 else np.array([1., 0., 0.])
            q_start = nudge_quat(quats[k], axis, nudge_deg)
            q_rec, _ = propagate_attitude(
                q_start, om_rec, t_fwd, mode="tumbling", inertia_tensor=ctx.inertia_tensor)
            lc = lofi_lc(q_rec, idx_fwd)
            rec_lcs.append(lc)
            mses.append(np.mean((lc - observed_lc[k:])**2))

        mses = np.array(mses)
        spread = mses.max() - mses.min() if len(mses) > 1 else 0.0
        print(f"  {nudge_deg:<8d} {mses.min():10.4f} {mses.mean():10.4f} "
              f"{mses.max():10.4f} {spread:10.4f}")

        # Plot: overlay all directions
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, observed_lc, 'k-', lw=0.8, alpha=0.6, label='Observed (noisy hifi)')
        ax.plot(t, true_lc, 'k--', lw=0.5, alpha=0.3, label='True (clean hifi)')
        for di, lc in enumerate(rec_lcs):
            lbl = f'dir {di+1}' if nudge_deg > 0 else 'nudge=0°'
            ax.plot(t[k:], lc, color=COLORS[di % len(COLORS)], lw=0.9, alpha=0.8, label=lbl)
        ax.axvline(t[k], color='blue', ls=':', alpha=0.5, label=f'Peak k={k}')
        ax.set(xlabel='Time (s)', ylabel='Magnitude')
        ax.set_ylim(ylim[1], ylim[0])  # inverted: bright (low mag) at top
        ax.set_title(f'Traj {ti} ({label}) — nudge {nudge_deg}° x{n_dirs}dirs [lofi recon, σ={ctx.noise_sigma}]')
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(OUT / f"m001c_lofi_qnudge_{om_tag}_traj{ti}_nudge{nudge_deg}.png",
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

print("\nDone.")
