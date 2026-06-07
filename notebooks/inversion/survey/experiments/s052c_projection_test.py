"""s052c — does projecting onto the polhode actually reduce cascade noise?

For each truth-q_a hypothesis ω_h:
  1. Find nearest polhode point ω_p (its 'projection').
  2. Compute pre-projection error = ||ω_h - ω_truth_t0|| (target = the actual
     truth ω at epoch t_0).
  3. Compute post-projection error = ||ω_p - ω_truth_t0||.
  4. Decompose noise (ω_h - ω_p) into tangential (along polhode tangent at
     ω_p) vs perpendicular components.

If polhode tangent projection is high-leverage, post-projection error << pre-
projection error AND the perpendicular noise is large fraction of total.

Also produce a 4-panel matplotlib figure showing:
  (a) pre vs post error scatter
  (b) noise decomposition (tangent vs perpendicular magnitudes)
  (c) histogram of improvement factors
  (d) zoomed projection view: truth-qa points + nearest polhode points
      with arrow connecting each to its projection.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SURVEY_DIR = Path(__file__).resolve().parent.parent
RESULTS = SURVEY_DIR / "results" / "s052_polhode_overlay"

d = np.load(RESULTS / "polhode.npz")
om_traj_dps = d["om_history_dps"]                # (500, 3) — polhode samples
om_kept_truth_qa = d["om_kept_truth_qa"]          # (153, 3) rad/s
om_kept_pool = d["om_kept_pool_subsample"]        # (5000, 3) rad/s
om_truth_at_t0 = d["om_truth_at_t0"]              # (3,) rad/s

RAD2DEG = 180.0 / np.pi
om_truth_qa_dps = om_kept_truth_qa * RAD2DEG
om_pool_dps = om_kept_pool * RAD2DEG
om_truth_t0_dps = om_truth_at_t0 * RAD2DEG

# ---------------------------------------------------------------------------
# Find nearest polhode point + tangent for each truth-qa hypothesis.
# ---------------------------------------------------------------------------

def project_onto_polhode(points, polhode):
    """Returns nearest polhode point + tangent at that point + index.

    points: (M, 3), polhode: (T, 3) sampled trajectory.
    """
    diffs = points[:, None, :] - polhode[None, :, :]  # (M, T, 3)
    d2 = np.sum(diffs * diffs, axis=2)
    nearest_idx = np.argmin(d2, axis=1)
    nearest = polhode[nearest_idx]
    # Tangent: finite difference between adjacent polhode samples.
    T = polhode.shape[0]
    nxt_idx = (nearest_idx + 1) % T
    prv_idx = (nearest_idx - 1) % T
    tangents = polhode[nxt_idx] - polhode[prv_idx]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    return nearest, tangents, nearest_idx


near_qa, tang_qa, idx_qa = project_onto_polhode(om_truth_qa_dps, om_traj_dps)
near_pool, tang_pool, idx_pool = project_onto_polhode(om_pool_dps, om_traj_dps)

# Pre vs post projection error magnitudes (vs truth target).
pre_err_qa = np.linalg.norm(om_truth_qa_dps - om_truth_t0_dps, axis=1)
post_err_qa = np.linalg.norm(near_qa - om_truth_t0_dps, axis=1)
pre_err_pool = np.linalg.norm(om_pool_dps - om_truth_t0_dps, axis=1)
post_err_pool = np.linalg.norm(near_pool - om_truth_t0_dps, axis=1)

# Noise decomposition: ω_h - ω_p = tangential + perpendicular.
noise_qa = om_truth_qa_dps - near_qa  # (M, 3)
tang_comp_qa = np.sum(noise_qa * tang_qa, axis=1)        # (M,) signed scalar
perp_comp_qa = np.linalg.norm(
    noise_qa - tang_comp_qa[:, None] * tang_qa, axis=1   # (M,)
)

print("Seed 14 polhode-projection test (truth-q_a hypotheses, n=153):")
print(f"  Pre-projection error (vs ω_truth_t0):")
print(f"     p10/p50/p90 = {np.percentile(pre_err_qa, [10,50,90])} dps")
print(f"  Post-projection error:")
print(f"     p10/p50/p90 = {np.percentile(post_err_qa, [10,50,90])} dps")
ratio = post_err_qa / pre_err_qa
print(f"  Improvement factor (post/pre):")
print(f"     p10/p50/p90 = {np.percentile(ratio, [10,50,90])}")
print()
print(f"  Noise tangent component p50: {np.percentile(np.abs(tang_comp_qa), 50):.3f} dps")
print(f"  Noise perpendicular comp p50: {np.percentile(perp_comp_qa, 50):.3f} dps")
print(f"  Perp/total: {np.percentile(perp_comp_qa, 50) / np.percentile(np.linalg.norm(noise_qa, axis=1), 50):.2%}")

# Also pool stats.
print("\nPool subsample (n=5000):")
ratio_pool = post_err_pool / pre_err_pool
print(f"  Improvement factor p50 = {np.percentile(ratio_pool, 50):.3f}")
print(f"  Pre-error p50 / post-error p50 = "
      f"{np.percentile(pre_err_pool, 50):.3f} / {np.percentile(post_err_pool, 50):.3f}")

# ---------------------------------------------------------------------------
# 4-panel figure.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 11))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30)

# (a) pre vs post error scatter.
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(pre_err_pool, post_err_pool, s=2, c="lightgrey", alpha=0.4, label="pool subsample")
ax_a.scatter(pre_err_qa, post_err_qa, s=20, c="green", alpha=0.7, label="truth-q_a (n=153)")
maxv = float(max(pre_err_qa.max(), pre_err_pool.max()))
ax_a.plot([0, maxv], [0, maxv], "k--", lw=1, label="post = pre (no improvement)")
ax_a.plot([0, maxv], [0, 0.5 * maxv], "r:", lw=1, label="post = 0.5 × pre")
ax_a.set_xlabel("pre-projection error  ||ω_h − ω_truth|| (dps)")
ax_a.set_ylabel("post-projection error  ||ω_p − ω_truth|| (dps)")
ax_a.set_title(f"Pre vs post projection error — truth-q_a p50 ratio = {np.percentile(ratio, 50):.2f}")
ax_a.legend(fontsize=9)
ax_a.grid(alpha=0.3)

# (b) noise decomposition: tangent vs perpendicular magnitude.
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(np.abs(tang_comp_qa), perp_comp_qa, s=20, c=pre_err_qa, cmap="plasma")
maxv_b = float(max(np.abs(tang_comp_qa).max(), perp_comp_qa.max()))
ax_b.plot([0, maxv_b], [0, maxv_b], "k--", lw=1, label="tangent = perp")
ax_b.set_xlabel("|tangent component of (ω_h - ω_p)|  (dps)")
ax_b.set_ylabel("|perpendicular component|  (dps)")
ax_b.set_title("Noise decomposition: along-polhode vs off-polhode\n"
               f"perp/total p50 = "
               f"{np.percentile(perp_comp_qa, 50)/np.percentile(np.linalg.norm(noise_qa, axis=1), 50):.0%}")
plt.colorbar(ax_b.collections[0], ax=ax_b, label="pre-error (dps)")
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)

# (c) histogram of improvement factors.
ax_c = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, 2, 50)
ax_c.hist(ratio, bins=bins, alpha=0.6, color="green",
          label=f"truth-q_a (n={len(ratio)})", density=True)
ax_c.hist(ratio_pool, bins=bins, alpha=0.4, color="grey",
          label=f"pool (n={len(ratio_pool)})", density=True)
ax_c.axvline(1.0, color="black", ls="--", lw=1, label="no improvement")
ax_c.axvline(np.percentile(ratio, 50), color="green", lw=2,
             label=f"truth-q_a median = {np.percentile(ratio, 50):.2f}")
ax_c.set_xlabel("post-error / pre-error")
ax_c.set_ylabel("density")
ax_c.set_title("Improvement factor distribution")
ax_c.legend(fontsize=8)
ax_c.grid(alpha=0.3)

# (d) projection view in best 2D plane (PCA of polhode).
poly_centered = om_traj_dps - om_traj_dps.mean(axis=0)
U, S, Vt = np.linalg.svd(poly_centered, full_matrices=False)
basis = Vt[:2]  # principal 2D plane of polhode
poly_2d = poly_centered @ basis.T
qa_2d = (om_truth_qa_dps - om_traj_dps.mean(axis=0)) @ basis.T
near_qa_2d = (near_qa - om_traj_dps.mean(axis=0)) @ basis.T

ax_d = fig.add_subplot(gs[1, 1])
ax_d.plot(poly_2d[:, 0], poly_2d[:, 1], "-", color="navy", lw=2.5, label="polhode (PCA plane)")
ax_d.scatter(qa_2d[:, 0], qa_2d[:, 1], s=15, c="green", alpha=0.5, label="truth-q_a")
ax_d.scatter(near_qa_2d[:, 0], near_qa_2d[:, 1], s=15, c="red", alpha=0.5,
             label="projected truth-q_a")
# Draw projection arrows for a subsample.
for k in range(0, len(qa_2d), 8):
    ax_d.plot([qa_2d[k, 0], near_qa_2d[k, 0]],
              [qa_2d[k, 1], near_qa_2d[k, 1]],
              "k-", alpha=0.3, lw=0.5)
om_truth_t0_2d = (om_truth_t0_dps - om_traj_dps.mean(axis=0)) @ basis.T
ax_d.scatter(*om_truth_t0_2d, c="orange", s=200, marker="D", edgecolors="black",
             label="ω_truth_t0", zorder=5)
ax_d.set_xlabel("PC1 (dps)")
ax_d.set_ylabel("PC2 (dps)")
ax_d.set_aspect("equal")
ax_d.set_title("Projection view (polhode 2D PCA plane)\nblack lines = projection paths")
ax_d.legend(fontsize=8, loc="upper right")
ax_d.grid(alpha=0.3)

fig.suptitle("Seed 14 — does polhode-tangent projection reduce cascade noise?",
             fontsize=14, y=0.99)
out = RESULTS / "projection_test.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved: {out}")
