"""s052 — polhode overlay: cascade hypotheses vs truth polhode in body-frame ω-space.

Visual test of the s051 polhode-prior reframe. Plots, in body-frame ω-space:

  (1) Truth polhode for seed 14 — the closed curve traced by ω_truth(t).
  (2) The single point ω_truth(epoch=275) — what the cascade hypotheses are
      trying to estimate.
  (3) 153 truth-q_a cascade hypotheses (finite-diff'd from the truth-q
      sample-pair).
  (4) ~5k random subsample of the 141706 cascade hypotheses (the pool).

The visual question: does the 17% finite-diff noise on the truth-q_a
cluster spread PERPENDICULAR to the polhode tangent (good — projection
collapses noise) or ALONG the tangent (bad — projection useless)?

Outputs:
  results/s052_polhode_overlay/polhode.npz  — truth ω(t), idx of truth-qA
  results/s052_polhode_overlay/overlay.html — interactive 3D Plotly scene
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

from lib.hifi_render import build_context  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

SEED = 14
RESULTS = SURVEY_DIR / "results" / "s052_polhode_overlay"
RESULTS.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260507)
N_POOL_SUBSAMPLE = 5000

# ---------------------------------------------------------------------------
# 1. Build context, propagate truth, extract truth ω(t).
# ---------------------------------------------------------------------------
print("[1/4] Building seed-14 context + propagating truth ω(t) ...")
t0 = time.time()
ctx = build_context(SEED)
quats, om_history = propagate_attitude(
    q0=ctx["q0_truth"],
    omega0=ctx["omega0_truth_rad"],
    times=ctx["observation_times"],
    mode="tumbling",
    inertia_tensor=ctx["inertia_tensor"],
)
print(f"    truth-propagated in {time.time()-t0:.2f}s. om_history shape={om_history.shape}")

# Sanity: |ω| variation magnitude (polhode size proxy).
om_mag = np.linalg.norm(om_history, axis=1)
om_mag_dps = om_mag * 180.0 / np.pi
print(f"    |ω| dps: min={om_mag_dps.min():.4f} mean={om_mag_dps.mean():.4f} "
      f"max={om_mag_dps.max():.4f} std/mean={om_mag.std()/om_mag.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# 2. Load s049 cascade pool + identify truth-q_a hypotheses.
# ---------------------------------------------------------------------------
print("[2/4] Loading s049 cascade pool ...")
cas = np.load(SURVEY_DIR / "results" / "s049_cascade_seed14" / "cascade.npz")
qA_kept = cas["qA_kept"]            # (141706, 4)
om_kept = cas["om_kept"]            # (141706, 3) rad/s
truth_q_a = cas["truth_q_a"]        # (4,)
om_truth_at_t0 = cas["om_truth_at_t0"]   # (3,) rad/s -- the truth target
om_truth_cascade = cas["om_truth_cascade"]  # (3,) rad/s -- finite-diff truth estimate
t0_ep = int(cas["t0_ep"])
delta_t = float(cas["delta_t"])
print(f"    pool: N={qA_kept.shape[0]}, t0_ep={t0_ep}, Δt={delta_t:.4f}s")
print(f"    om_truth_at_t0  = {om_truth_at_t0}  |·|_dps={np.linalg.norm(om_truth_at_t0)*180/np.pi:.4f}")
print(f"    om_truth_cascade = {om_truth_cascade}  |·|_dps={np.linalg.norm(om_truth_cascade)*180/np.pi:.4f}")

# Identify rows where q_a == truth_q_a (antipode-aware — should be 153 per summary).
dots = np.abs(qA_kept @ truth_q_a)
truth_qa_mask = dots > (1.0 - 1e-9)
truth_qa_idx = np.where(truth_qa_mask)[0]
n_truth_qa = truth_qa_idx.size
print(f"    truth-q_a hypotheses: {n_truth_qa}")

# Subsample the rest of the pool for plotting.
non_truth_qa_idx = np.where(~truth_qa_mask)[0]
sample_idx = RNG.choice(non_truth_qa_idx, size=min(N_POOL_SUBSAMPLE, non_truth_qa_idx.size),
                        replace=False)
print(f"    pool subsample for plot: {sample_idx.size}")

# ---------------------------------------------------------------------------
# 3. Save NPZ checkpoint.
# ---------------------------------------------------------------------------
print("[3/4] Saving NPZ checkpoint ...")
np.savez(
    RESULTS / "polhode.npz",
    om_history=om_history,                           # (500, 3) truth polhode
    om_history_dps=om_history * 180.0 / np.pi,
    om_truth_at_t0=om_truth_at_t0,
    om_truth_cascade=om_truth_cascade,
    truth_qa_idx=truth_qa_idx,
    om_kept_truth_qa=om_kept[truth_qa_idx],          # (153, 3)
    om_kept_pool_subsample=om_kept[sample_idx],      # (5000, 3)
    pool_subsample_idx=sample_idx,
    t0_ep=t0_ep,
    seed=SEED,
)
print(f"    saved: {RESULTS / 'polhode.npz'}")

# ---------------------------------------------------------------------------
# 4. Build Plotly 3D scene.
# ---------------------------------------------------------------------------
print("[4/4] Rendering interactive HTML ...")

# Convert all to dps for human-readable axes.
om_traj_dps = om_history * 180.0 / np.pi
om_truth_t0_dps = om_truth_at_t0 * 180.0 / np.pi
om_truth_cas_dps = om_truth_cascade * 180.0 / np.pi
om_truth_qa_dps = om_kept[truth_qa_idx] * 180.0 / np.pi
om_pool_dps = om_kept[sample_idx] * 180.0 / np.pi

# Coloring: distance from each truth-qa hypothesis to its nearest polhode point.
def nearest_polhode_dist_deg(points_dps, polhode_dps):
    """Return nearest L2 distance (dps) and the polhode-frac index for each point."""
    diffs = points_dps[:, None, :] - polhode_dps[None, :, :]  # (M, T, 3)
    d2 = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(d2.shape[0]), nearest])
    return dist, nearest

dist_truth_qa, _ = nearest_polhode_dist_deg(om_truth_qa_dps, om_traj_dps)
dist_pool, _ = nearest_polhode_dist_deg(om_pool_dps, om_traj_dps)
print(f"    truth-qa polhode-distance dps:   p10={np.percentile(dist_truth_qa, 10):.4f} "
      f"p50={np.percentile(dist_truth_qa, 50):.4f} "
      f"p90={np.percentile(dist_truth_qa, 90):.4f}")
print(f"    pool polhode-distance dps:        p10={np.percentile(dist_pool, 10):.4f} "
      f"p50={np.percentile(dist_pool, 50):.4f} "
      f"p90={np.percentile(dist_pool, 90):.4f}")

fig = go.Figure()

# Polhode (truth ω trajectory).
fig.add_trace(go.Scatter3d(
    x=om_traj_dps[:, 0], y=om_traj_dps[:, 1], z=om_traj_dps[:, 2],
    mode="lines+markers",
    line=dict(color="rgb(20,90,200)", width=4),
    marker=dict(size=2, color="rgb(20,90,200)"),
    name=f"Truth polhode ω(t) [N=500, t∈[0,3600]s, ~12 loops]",
))

# Pool subsample (grey haze).
fig.add_trace(go.Scatter3d(
    x=om_pool_dps[:, 0], y=om_pool_dps[:, 1], z=om_pool_dps[:, 2],
    mode="markers",
    marker=dict(size=1.5, color="rgba(140,140,140,0.30)"),
    name=f"Cascade pool subsample (n={sample_idx.size})",
))

# Truth-q_a hypotheses (colored by polhode distance).
fig.add_trace(go.Scatter3d(
    x=om_truth_qa_dps[:, 0], y=om_truth_qa_dps[:, 1], z=om_truth_qa_dps[:, 2],
    mode="markers",
    marker=dict(
        size=4,
        color=dist_truth_qa,
        colorscale="Viridis",
        colorbar=dict(title="Δ to polhode (dps)", x=1.02),
        cmin=0.0,
        cmax=float(np.percentile(dist_truth_qa, 95)),
    ),
    name=f"Truth-q_a hypotheses (n={n_truth_qa})",
))

# Truth point at t_0 (the target).
fig.add_trace(go.Scatter3d(
    x=[om_truth_t0_dps[0]], y=[om_truth_t0_dps[1]], z=[om_truth_t0_dps[2]],
    mode="markers",
    marker=dict(size=10, color="rgb(255,160,0)", symbol="diamond"),
    name=f"ω_truth(epoch={t0_ep}) [target]",
))

# Truth-cascade derivation (the finite-diff applied to the actual truth pair).
fig.add_trace(go.Scatter3d(
    x=[om_truth_cas_dps[0]], y=[om_truth_cas_dps[1]], z=[om_truth_cas_dps[2]],
    mode="markers",
    marker=dict(size=8, color="rgb(220,40,40)", symbol="x"),
    name="ω_truth_cascade (finite-diff on truth pair)",
))

fig.update_layout(
    title=(f"Seed {SEED} — body-frame ω-space: truth polhode vs s049 cascade hypotheses<br>"
           f"<sub>Truth |ω|={om_mag_dps.mean():.3f} dps (std/mean {om_mag.std()/om_mag.mean()*100:.2f}%)</sub>"),
    scene=dict(
        xaxis_title="ω_x (dps)",
        yaxis_title="ω_y (dps)",
        zaxis_title="ω_z (dps)",
        aspectmode="data",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    ),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    width=1400,
    height=900,
    margin=dict(l=0, r=0, t=70, b=0),
)

html_path = RESULTS / "overlay.html"
fig.write_html(str(html_path), include_plotlyjs="cdn")
print(f"    saved: {html_path}")

# Save summary JSON.
summary = {
    "seed": int(SEED),
    "n_pool": int(qA_kept.shape[0]),
    "n_truth_qa": int(n_truth_qa),
    "n_pool_subsample": int(sample_idx.size),
    "om_mag_dps_mean": float(om_mag_dps.mean()),
    "om_mag_dps_std_over_mean": float(om_mag.std() / om_mag.mean()),
    "polhode_dist_dps": {
        "truth_qa": {"p10": float(np.percentile(dist_truth_qa, 10)),
                     "p50": float(np.percentile(dist_truth_qa, 50)),
                     "p90": float(np.percentile(dist_truth_qa, 90))},
        "pool":     {"p10": float(np.percentile(dist_pool, 10)),
                     "p50": float(np.percentile(dist_pool, 50)),
                     "p90": float(np.percentile(dist_pool, 90))},
    },
    "om_truth_at_t0_rad": list(map(float, om_truth_at_t0)),
    "om_truth_cascade_rad": list(map(float, om_truth_cascade)),
    "t0_ep": t0_ep,
    "delta_t_s": delta_t,
    "html": str(html_path),
    "npz": str(RESULTS / "polhode.npz"),
}
with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"    saved: {RESULTS / 'summary.json'}")
print(f"\nSaved: {html_path}")
print(f"Saved: {RESULTS / 'polhode.npz'}")
print(f"Saved: {RESULTS / 'summary.json'}")
