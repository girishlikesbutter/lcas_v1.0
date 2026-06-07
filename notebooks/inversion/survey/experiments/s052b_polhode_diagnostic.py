"""s052b — multi-panel diagnostic: is the polhode the load-bearing structure
on seed 14 cascade noise?

Reads s052 cached NPZ. Produces a matplotlib figure with:

  Top-left:  3D scatter, wide view (full noise extent).
  Top-right: 3D scatter, tight view (polhode + nearest truth-qa).
  Mid-row:   3 orthogonal 2D projections (ω_x-ω_y, ω_x-ω_z, ω_y-ω_z),
             tight view, polhode overplotted + truth-qa points colored.
  Bottom:    |ω|(t) line + truth-qa polhode-distance histogram.

Quantifies:
  - Polhode "diameter" = max pairwise dist on truth ω(t) (dps).
  - Truth-qa cluster extent (centroid-relative p90 dist).
  - Ratio: noise / polhode = (noise extent) / (polhode diameter).
  - Truth-qa polhode-distance: should be << noise-extent if projection helps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SURVEY_DIR = Path(__file__).resolve().parent.parent
RESULTS = SURVEY_DIR / "results" / "s052_polhode_overlay"

# Load cached NPZ.
d = np.load(RESULTS / "polhode.npz")
om_traj_dps = d["om_history_dps"]                       # (500, 3)
om_kept_truth_qa = d["om_kept_truth_qa"]                # (153, 3) rad/s
om_kept_pool = d["om_kept_pool_subsample"]              # (5000, 3) rad/s
om_truth_at_t0 = d["om_truth_at_t0"]                    # (3,) rad/s
om_truth_cascade = d["om_truth_cascade"]                # (3,) rad/s
seed = int(d["seed"])
t0_ep = int(d["t0_ep"])

# Convert to dps.
RAD2DEG = 180.0 / np.pi
om_truth_qa_dps = om_kept_truth_qa * RAD2DEG
om_pool_dps = om_kept_pool * RAD2DEG
om_truth_t0_dps = om_truth_at_t0 * RAD2DEG
om_truth_cas_dps = om_truth_cascade * RAD2DEG

# ---------------------------------------------------------------------------
# Quantitative diagnostics.
# ---------------------------------------------------------------------------

def pairwise_max_dist(X):
    """Max L2 distance between any two rows of X."""
    dXX = X[:, None, :] - X[None, :, :]
    d2 = np.sum(dXX * dXX, axis=2)
    return float(np.sqrt(d2.max()))

polhode_diameter = pairwise_max_dist(om_traj_dps)
truth_qa_centroid = om_truth_qa_dps.mean(axis=0)
truth_qa_dist_from_centroid = np.linalg.norm(om_truth_qa_dps - truth_qa_centroid, axis=1)
pool_centroid = om_pool_dps.mean(axis=0)
pool_dist_from_centroid = np.linalg.norm(om_pool_dps - pool_centroid, axis=1)

# Distance from each truth-qa point to nearest polhode point.
diffs = om_truth_qa_dps[:, None, :] - om_traj_dps[None, :, :]
truth_qa_polhode_dist = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)

# Same for pool.
diffs_p = om_pool_dps[:, None, :] - om_traj_dps[None, :, :]
pool_polhode_dist = np.sqrt(np.sum(diffs_p * diffs_p, axis=2)).min(axis=1)

# Ratio: noise / polhode.
truth_qa_extent_p90 = float(np.percentile(truth_qa_dist_from_centroid, 90))
pool_extent_p90 = float(np.percentile(pool_dist_from_centroid, 90))
ratio_truthqa = truth_qa_extent_p90 / polhode_diameter
ratio_pool = pool_extent_p90 / polhode_diameter

print(f"Seed {seed} polhode/noise diagnostic:")
print(f"  Polhode diameter (max pairwise dist on ω(t)):  {polhode_diameter:.4f} dps")
print(f"  Truth-qa cluster p90 from centroid:            {truth_qa_extent_p90:.4f} dps")
print(f"  Pool cluster      p90 from centroid:           {pool_extent_p90:.4f} dps")
print(f"  ratio truth-qa-noise / polhode:                {ratio_truthqa:.1f}×")
print(f"  ratio pool-noise / polhode:                    {ratio_pool:.1f}×")
print(f"  Truth-qa polhode-dist:  p10={np.percentile(truth_qa_polhode_dist,10):.3f}  "
      f"p50={np.percentile(truth_qa_polhode_dist,50):.3f}  "
      f"p90={np.percentile(truth_qa_polhode_dist,90):.3f} dps")
print(f"  Pool      polhode-dist:  p10={np.percentile(pool_polhode_dist,10):.3f}  "
      f"p50={np.percentile(pool_polhode_dist,50):.3f}  "
      f"p90={np.percentile(pool_polhode_dist,90):.3f} dps")

# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11))
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.4, 1.0, 0.7], hspace=0.35, wspace=0.35)

# (1) 3D wide view.
ax1 = fig.add_subplot(gs[0, 0], projection="3d")
ax1.scatter(om_pool_dps[:, 0], om_pool_dps[:, 1], om_pool_dps[:, 2],
            s=1, c="lightgrey", alpha=0.3, label="pool subsample")
ax1.scatter(om_truth_qa_dps[:, 0], om_truth_qa_dps[:, 1], om_truth_qa_dps[:, 2],
            s=20, c=truth_qa_polhode_dist, cmap="viridis",
            label="truth-q_a (n=153)")
ax1.plot(om_traj_dps[:, 0], om_traj_dps[:, 1], om_traj_dps[:, 2],
         "-", color="navy", lw=1.5, label="truth polhode")
ax1.scatter(*om_truth_t0_dps, c="orange", s=80, marker="D", label="ω_truth(t_0)")
ax1.scatter(*om_truth_cas_dps, c="red", s=80, marker="x", label="ω_truth_cascade")
ax1.set_xlabel("ω_x (dps)")
ax1.set_ylabel("ω_y (dps)")
ax1.set_zlabel("ω_z (dps)")
ax1.set_title(f"3D wide view\n(noise/polhode = {ratio_truthqa:.0f}× truth-qa, "
              f"{ratio_pool:.0f}× pool)")
ax1.legend(fontsize=7, loc="upper left")

# (2) 3D tight view (axes around polhode).
poly_center = om_traj_dps.mean(axis=0)
poly_span = polhode_diameter * 5  # show 5× polhode-diameter window
ax2 = fig.add_subplot(gs[0, 1], projection="3d")
ax2.plot(om_traj_dps[:, 0], om_traj_dps[:, 1], om_traj_dps[:, 2],
         "-", color="navy", lw=2.5, label="truth polhode")
# Only show truth-qa points within poly_span of polhode center.
near_mask = np.linalg.norm(om_truth_qa_dps - poly_center, axis=1) < poly_span
sc2 = ax2.scatter(om_truth_qa_dps[near_mask, 0], om_truth_qa_dps[near_mask, 1],
                  om_truth_qa_dps[near_mask, 2],
                  s=30, c=truth_qa_polhode_dist[near_mask], cmap="viridis",
                  label=f"truth-q_a near polhode (n={int(near_mask.sum())}/153)")
ax2.scatter(*om_truth_t0_dps, c="orange", s=120, marker="D", label="ω_truth(t_0)")
ax2.scatter(*om_truth_cas_dps, c="red", s=80, marker="x")
ax2.set_xlim(poly_center[0] - poly_span, poly_center[0] + poly_span)
ax2.set_ylim(poly_center[1] - poly_span, poly_center[1] + poly_span)
ax2.set_zlim(poly_center[2] - poly_span, poly_center[2] + poly_span)
ax2.set_xlabel("ω_x (dps)")
ax2.set_ylabel("ω_y (dps)")
ax2.set_zlabel("ω_z (dps)")
ax2.set_title(f"3D tight view\n(±{poly_span:.3f} dps around polhode center)")
ax2.legend(fontsize=7, loc="upper left")
plt.colorbar(sc2, ax=ax2, label="Δ to polhode (dps)", shrink=0.6)

# (3) Stats text panel (top-right).
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")
stats_txt = (
    f"Seed 14 — polhode-prior diagnostic\n"
    f"────────────────────────────────────\n\n"
    f"Truth-ω structure:\n"
    f"  |ω| mean:     1.235 dps\n"
    f"  |ω| std/mean: 0.37%\n"
    f"  Polhode shape: tiny loop, near principal axis\n\n"
    f"Polhode diameter:\n"
    f"  {polhode_diameter:.4f} dps\n\n"
    f"Cascade-noise extent (p90 from centroid):\n"
    f"  truth-q_a:    {truth_qa_extent_p90:.4f} dps\n"
    f"  pool subsample: {pool_extent_p90:.4f} dps\n\n"
    f"Noise / polhode ratio:\n"
    f"  truth-q_a:    {ratio_truthqa:.0f}×\n"
    f"  pool:         {ratio_pool:.0f}×\n\n"
    f"Truth-q_a → polhode distance (dps):\n"
    f"  p10 = {np.percentile(truth_qa_polhode_dist, 10):.3f}\n"
    f"  p50 = {np.percentile(truth_qa_polhode_dist, 50):.3f}\n"
    f"  p90 = {np.percentile(truth_qa_polhode_dist, 90):.3f}\n\n"
    f"Pool → polhode distance (dps):\n"
    f"  p10 = {np.percentile(pool_polhode_dist, 10):.3f}\n"
    f"  p50 = {np.percentile(pool_polhode_dist, 50):.3f}\n"
    f"  p90 = {np.percentile(pool_polhode_dist, 90):.3f}\n\n"
    f"Verdict for s051 polhode-tangent\n"
    f"projection on seed 14:\n\n"
    f"  Polhode is a near-point on the\n"
    f"  scale of cascade noise. Projection\n"
    f"  would snap all hypotheses to a\n"
    f"  ~0.013 dps loop — equivalent to a\n"
    f"  hard prior on ω being on this loop.\n"
    f"  Loses cascade discrimination but\n"
    f"  doesn't selectively help truth.\n"
)
ax3.text(0.02, 0.98, stats_txt, transform=ax3.transAxes,
         family="monospace", fontsize=9, va="top", ha="left",
         bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.5))

# Three orthogonal projections.
poly_min = om_traj_dps.min(axis=0) - polhode_diameter * 0.5
poly_max = om_traj_dps.max(axis=0) + polhode_diameter * 0.5
plot_min = poly_center - polhode_diameter * 5
plot_max = poly_center + polhode_diameter * 5

projection_specs = [
    ((0, 1), "ω_x", "ω_y"),
    ((0, 2), "ω_x", "ω_z"),
    ((1, 2), "ω_y", "ω_z"),
]
for col, ((i, j), xlabel, ylabel) in enumerate(projection_specs):
    ax = fig.add_subplot(gs[1, col])
    near_mask = np.linalg.norm(om_truth_qa_dps - poly_center, axis=1) < polhode_diameter * 5
    ax.scatter(om_truth_qa_dps[near_mask, i], om_truth_qa_dps[near_mask, j],
               s=20, c=truth_qa_polhode_dist[near_mask], cmap="viridis", alpha=0.85)
    ax.plot(om_traj_dps[:, i], om_traj_dps[:, j], "-", color="navy", lw=2)
    ax.scatter(om_truth_t0_dps[i], om_truth_t0_dps[j], c="orange", s=120, marker="D",
               edgecolors="black", label="ω_truth(t_0)")
    ax.scatter(om_truth_cas_dps[i], om_truth_cas_dps[j], c="red", s=80, marker="x",
               label="ω_truth_cascade")
    ax.set_xlim(plot_min[i], plot_max[i])
    ax.set_ylim(plot_min[j], plot_max[j])
    ax.set_xlabel(f"{xlabel} (dps)")
    ax.set_ylabel(f"{ylabel} (dps)")
    ax.set_title(f"{xlabel}-{ylabel} projection")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if col == 0:
        ax.legend(fontsize=7, loc="upper right")

# Bottom row.
# |ω|(t) line.
ax_om = fig.add_subplot(gs[2, 0:2])
om_mag_dps = np.linalg.norm(om_traj_dps, axis=1)
ax_om.plot(om_mag_dps, lw=1.5, color="navy")
ax_om.axhline(om_mag_dps.mean(), color="grey", ls=":", lw=1)
ax_om.axvline(t0_ep, color="orange", ls="--", lw=1, label=f"t_0=epoch {t0_ep}")
ax_om.set_xlabel("epoch")
ax_om.set_ylabel("|ω| (dps)")
ax_om.set_title(f"|ω|(t) — span {om_mag_dps.max() - om_mag_dps.min():.4f} dps "
                f"({(om_mag_dps.std() / om_mag_dps.mean()) * 100:.2f}% std/mean)")
ax_om.legend(fontsize=8)
ax_om.grid(alpha=0.3)

# Histogram of polhode-distances.
ax_hist = fig.add_subplot(gs[2, 2])
ax_hist.hist(truth_qa_polhode_dist, bins=30, alpha=0.6, color="green",
             label=f"truth-q_a (n={len(truth_qa_polhode_dist)})", density=True)
ax_hist.hist(pool_polhode_dist, bins=30, alpha=0.4, color="grey",
             label=f"pool (n={len(pool_polhode_dist)})", density=True)
ax_hist.axvline(polhode_diameter, color="navy", ls="--", lw=1.5,
                label=f"polhode diameter ({polhode_diameter:.3f})")
ax_hist.set_xlabel("polhode distance (dps)")
ax_hist.set_ylabel("density")
ax_hist.set_title("Distance to nearest polhode point")
ax_hist.legend(fontsize=7, loc="upper right")
ax_hist.grid(alpha=0.3)

fig.suptitle(f"Seed {seed} — does polhode constrain s049 cascade noise?", fontsize=14, y=0.99)
out = RESULTS / "diagnostic.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved: {out}")
