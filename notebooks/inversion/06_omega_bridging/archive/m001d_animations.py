"""
m001d: 3D Plotly animations for quaternion nudge experiment.

Generates shadow-pattern animations (yellow=lit, dark blue=shadowed,
purple=back-culled) for every trajectory × nudge combination from m001c.
This verifies visually whether self-shadowing explains why lo-fi fails for
trajectories 2 & 3.

Output: data/results/inversion_diagnostics/m001d_animations/
  - traj{i}_{label}/true_hifi.html          (full window, shadows)
  - traj{i}_{label}/nudge{deg}deg_lofi.html (from peak, no shadows)
  - traj{i}_{label}/nudge{deg}deg_dir{d}_lofi.html
  - index.md
"""
import sys, time, copy, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from notebooks.inversion.lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.visualization.plotly_animation_generator import create_interactive_3d_animation

OUT = Path("data/results/inversion_diagnostics/m001d_animations")
OUT.mkdir(parents=True, exist_ok=True)

N_DIRS = 5
NUDGE_DEGS = [0, 1, 2, 3, 4, 5]

# ── Hardcoded starting quaternions (w,x,y,z) ────────────────────────
def aa_q(axis, angle_deg):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    h = np.deg2rad(angle_deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * a)])

Q0S = [
    np.array([1., 0., 0., 0.]),
    aa_q([1, 0, 0], 30),
    aa_q([0, 1, 1], 90),
    aa_q([0.5, -0.3, 0.7], 160),
]
LABELS = ["identity", "x30", "yz90", "arb160"]

# ── Pre-generate random nudge axes (deterministic, same as m001c) ─
rng = np.random.RandomState(123)
NUDGE_AXES = rng.randn(N_DIRS, 3)
NUDGE_AXES /= np.linalg.norm(NUDGE_AXES, axis=1, keepdims=True)

# ── Setup ────────────────────────────────────────────────────────────
ctx = setup_experiment(
    n_observations=100,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc="2020-02-05T10:02:00",
)

# ── Helpers ──────────────────────────────────────────────────────────
def body_vecs(quats, idx):
    n = len(idx)
    k1, k2 = np.empty((n, 3)), np.empty((n, 3))
    for i, j in enumerate(idx):
        R = Rotation.from_quat([quats[i,1], quats[i,2], quats[i,3], quats[i,0]]).as_matrix()
        s = R @ (ctx.sun_pos[j] - ctx.sat_pos[j]); k1[i] = s / np.linalg.norm(s)
        o = R @ (ctx.obs_pos[j] - ctx.sat_pos[j]); k2[i] = o / np.linalg.norm(o)
    return k1, k2

def nudge_quat(q_wxyz, axis, angle_deg):
    r_orig = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r_nudge = Rotation.from_rotvec(np.deg2rad(angle_deg) * axis)
    r_new = r_nudge * r_orig
    xyzw = r_new.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

def recover_omega(q1, q2, dt):
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return (r1.inv() * r2).as_rotvec() / dt

def quats_to_att_matrices(quats_wxyz):
    """Convert (N,4) wxyz quaternions to (N,3,3) rotation matrices."""
    n = len(quats_wxyz)
    mats = np.empty((n, 3, 3))
    for i in range(n):
        q = quats_wxyz[i]
        mats[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return mats

def make_animation(quats, idx, use_shadows, title, out_path):
    """Generate and save a 3D Plotly animation."""
    k1, k2 = body_vecs(quats, idx)
    art = {c: m[idx] for c, m in ctx.art_matrices.items()}

    if use_shadows:
        lit = compute_shadows(ctx.satellite, k1, explicit_component_matrices=art,
                              show_progress=False)
    else:
        lit = create_no_shadow_lit_status(ctx.satellite, len(idx))

    mag, flux, _, _, _, anim_data = generate_lightcurves(
        lit, k1, k2, ctx.obs_dist[idx], ctx.satellite, ctx.epochs[idx],
        pre_computed_matrices=art, animate=True, show_progress=False)

    t_hrs = (ctx.observation_times[idx] - ctx.observation_times[idx[0]]) / 3600.0
    geo = {'sat_att_matrices': quats_to_att_matrices(quats)}

    create_interactive_3d_animation(
        anim_data, mag, t_hrs, geo,
        satellite_name=title,
        show_j2000_frame=False,
        show_body_frame=True,
        show_sun_vector=True,
        show_observer_vector=True,
        color_mode='flux',
        output_path=out_path,
    )
    return mag

# ── Main loop ────────────────────────────────────────────────────────
t = ctx.observation_times
all_idx = np.arange(ctx.n_observations)
om_tag = f"{np.linalg.norm(ctx.true_omega0)*180/np.pi:.2f}dps"
index_rows = []
t0_total = time.time()

for ti, (q0, label) in enumerate(zip(Q0S, LABELS)):
    traj_dir = OUT / f"traj{ti}_{label}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nTrajectory {ti}: {label}")

    # Propagate true attitude
    quats, omegas = propagate_attitude(
        q0, ctx.true_omega0, t, mode="tumbling", inertia_tensor=ctx.inertia_tensor)

    # Observed LC (hi-fi + noise, same seed as m001c)
    k1_all, k2_all = body_vecs(quats, all_idx)
    art_all = {c: m[all_idx] for c, m in ctx.art_matrices.items()}
    lit_all = compute_shadows(ctx.satellite, k1_all,
                              explicit_component_matrices=art_all, show_progress=False)
    true_lc, *_ = generate_lightcurves(
        lit_all, k1_all, k2_all, ctx.obs_dist[all_idx], ctx.satellite,
        ctx.epochs[all_idx], pre_computed_matrices=art_all, show_progress=False)
    np.random.seed(42 + ti)
    observed_lc = true_lc + np.random.normal(0, ctx.noise_sigma, len(true_lc))

    # Find peak
    k = next((i for i in range(1, len(observed_lc) - 1)
              if observed_lc[i] < observed_lc[i-1] and observed_lc[i] < observed_lc[i+1]),
             int(np.argmin(observed_lc[1:-1])) + 1)
    print(f"  Peak k={k}, t={t[k]:.1f}s")

    om_rec = recover_omega(quats[k-1], quats[k], t[k] - t[k-1])

    # ── True hi-fi animation (full window) ───────────────────────────
    out_hifi = traj_dir / "true_hifi.html"
    print(f"  Generating true hi-fi animation...")
    t1 = time.time()
    make_animation(quats, all_idx, use_shadows=True,
                   title=f"Traj {ti} ({label}) — true hi-fi",
                   out_path=out_hifi)
    print(f"    Done in {time.time()-t1:.1f}s")
    index_rows.append((f"traj{ti}_{label}/true_hifi.html",
                       f"m001c_lofi_qnudge_{om_tag}_traj{ti}_nudge0.png",
                       label, "true", "hifi", "-"))

    # ── Nudge loop ───────────────────────────────────────────────────
    idx_fwd = np.arange(k, ctx.n_observations)
    t_fwd = t[k:] - t[k]

    for nudge_deg in NUDGE_DEGS:
        n_dirs = 1 if nudge_deg == 0 else N_DIRS
        for di in range(n_dirs):
            axis = NUDGE_AXES[di] if nudge_deg > 0 else np.array([1., 0., 0.])
            q_start = nudge_quat(quats[k], axis, nudge_deg)
            q_rec, _ = propagate_attitude(
                q_start, om_rec, t_fwd, mode="tumbling",
                inertia_tensor=ctx.inertia_tensor)

            if nudge_deg == 0:
                fname = "nudge0deg_lofi.html"
            else:
                fname = f"nudge{nudge_deg}deg_dir{di}_lofi.html"

            out_lofi = traj_dir / fname
            tag = f"nudge {nudge_deg}°" + (f" dir{di}" if nudge_deg > 0 else "")
            print(f"  {tag} ...", end=" ", flush=True)
            t1 = time.time()
            make_animation(q_rec, idx_fwd, use_shadows=False,
                           title=f"Traj {ti} ({label}) — {tag} lo-fi",
                           out_path=out_lofi)
            print(f"{time.time()-t1:.1f}s")

            m001c_plot = f"m001c_lofi_qnudge_{om_tag}_traj{ti}_nudge{nudge_deg}.png"
            index_rows.append((f"traj{ti}_{label}/{fname}", m001c_plot,
                               label, str(nudge_deg), "lofi",
                               str(di) if nudge_deg > 0 else "-"))

# ── Write index.md ───────────────────────────────────────────────────
with open(OUT / "index.md", "w") as f:
    f.write("# m001d — Animation Index\n\n")
    f.write("| Animation | m001c plot | Trajectory | Nudge° | Fidelity | Dir |\n")
    f.write("|-----------|-------------|------------|--------|----------|-----|\n")
    for row in index_rows:
        f.write("| " + " | ".join(row) + " |\n")

elapsed = time.time() - t0_total
print(f"\n{'='*60}")
print(f"Total: {len(index_rows)} animations in {elapsed/60:.1f} min")
print(f"Output: {OUT}")
print(f"Index:  {OUT / 'index.md'}")
