"""s052d — does cascade noise preserve polhode label?

The polhode is the intersection of:
  Energy ellipsoid: 2T = ω·I·ω
  Momentum sphere: |L|² = ω·I²·ω      (with L = I·ω in body frame)

Each polhode is uniquely identified by (|L|, 2T) up to the relation
ω·I·ω vs. (I·ω)·(I·ω). A cleaner 1-scalar label is

  D = 2T · I_b / |L|²

where I_a < I_b < I_c are the eigenvalues of I. D < 1 → polhode encloses
the I_a (low-inertia) principal axis; D > 1 → encloses I_c; D = 1 →
separatrix.

For each cascade hypothesis ω_h:
  - Compute |L|_h, 2T_h, D_h.
  - Compare to truth's |L|, 2T, D.

If truth-q_a hypotheses cluster sharply around (|L|_truth, 2T_truth)
while pool is broad, the polhode-LABEL prior is meaningful.

Plot:
  (a) Pool + truth-qa + truth in (|L|, 2T) plane.
  (b) Histogram of D for pool vs truth-qa vs truth.
  (c) Scatter (D, |L|) — labelled by perp distance to nearest polhode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

from lib.hifi_render import build_context  # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s052_polhode_overlay"

# Inertia tensor for m048 (seed-independent).
ctx = build_context(14)
I = ctx["inertia_tensor"]
eig_vals, eig_vecs = np.linalg.eigh(I)
print(f"I eigenvalues (sorted): {eig_vals}")
I_a, I_b, I_c = float(eig_vals[0]), float(eig_vals[1]), float(eig_vals[2])
print(f"Asymmetry: I_a/I_b = {I_a/I_b:.4f}, I_b/I_c = {I_b/I_c:.4f}, I_c/I_a = {I_c/I_a:.4f}")

# Load cached arrays.
d = np.load(RESULTS / "polhode.npz")
om_truth_qa_rad = d["om_kept_truth_qa"]                 # (153, 3) rad/s
om_pool_rad = d["om_kept_pool_subsample"]                # (5000, 3) rad/s
om_truth_t0 = d["om_truth_at_t0"]                       # (3,) rad/s
om_traj_rad = d["om_history"]                           # (500, 3) rad/s

# Compute (|L|, 2T, D) for any ω in body frame.
def lab_invariants(omega_rad, I):
    L = omega_rad @ I.T
    Lmag = np.linalg.norm(L, axis=-1)
    twoT = np.einsum("...i,ij,...j->...", omega_rad, I, omega_rad)
    return Lmag, twoT


L_truth, T_truth = lab_invariants(om_truth_t0, I)
L_qa, T_qa = lab_invariants(om_truth_qa_rad, I)
L_pool, T_pool = lab_invariants(om_pool_rad, I)
L_traj, T_traj = lab_invariants(om_traj_rad, I)
D_truth = float(T_truth * I_b / L_truth**2)
D_qa = T_qa * I_b / L_qa**2
D_pool = T_pool * I_b / L_pool**2
D_traj = T_traj * I_b / L_traj**2

print()
print(f"Truth: |L|={L_truth:.4f} kg m²/s, 2T={T_truth:.4e} J, D={D_truth:.4f}")
print(f"Polhode-trajectory invariants spread:")
print(f"  |L| min/max  : {L_traj.min():.4f} / {L_traj.max():.4f} (span {L_traj.max()-L_traj.min():.6f})")
print(f"  2T  min/max  : {T_traj.min():.4e} / {T_traj.max():.4e}")
print(f"  D   min/max  : {D_traj.min():.6f} / {D_traj.max():.6f}")
print()
print(f"Truth-qa hypotheses (n=153):")
print(f"  |L|: p10/p50/p90 = {np.percentile(L_qa, [10,50,90])}")
print(f"  D:   p10/p50/p90 = {np.percentile(D_qa, [10,50,90])}")
print(f"  D - D_truth: p10/p50/p90 = {np.percentile(D_qa - D_truth, [10,50,90])}")
print()
print(f"Pool (n=5000):")
print(f"  |L|: p10/p50/p90 = {np.percentile(L_pool, [10,50,90])}")
print(f"  D:   p10/p50/p90 = {np.percentile(D_pool, [10,50,90])}")

# Concentration test: how concentrated is truth-qa around D_truth vs pool?
def concentration(values, target, scale):
    """Fraction within ± scale of target."""
    return float(np.mean(np.abs(values - target) < scale))

for scale in [0.001, 0.01, 0.05, 0.1]:
    f_qa = concentration(D_qa, D_truth, scale)
    f_pool = concentration(D_pool, D_truth, scale)
    enrich = f_qa / f_pool if f_pool > 0 else float('inf')
    print(f"D within ±{scale} of truth: truth-qa {f_qa:.2%}, pool {f_pool:.2%}, enrich {enrich:.2f}×")

# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30)

# (a) (|L|, 2T) plane.
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(L_pool, T_pool, s=2, c="lightgrey", alpha=0.4, label="pool")
ax_a.scatter(L_qa, T_qa, s=20, c="green", alpha=0.7, label=f"truth-qa (n=153)")
ax_a.scatter(L_traj, T_traj, s=8, c="navy", alpha=0.9, label="truth polhode (n=500)")
ax_a.scatter(L_truth, T_truth, s=200, c="orange", marker="D", edgecolors="black",
             label="truth_t0", zorder=5)
ax_a.set_xlabel("|L|  (kg m²/s)")
ax_a.set_ylabel("2T  (J)")
ax_a.set_title("(|L|, 2T) plane — each polhode is a single point here")
ax_a.legend(fontsize=8)
ax_a.grid(alpha=0.3)

# (b) D = 2T·I_b / |L|² histogram.
ax_b = fig.add_subplot(gs[0, 1])
bins = np.linspace(min(D_pool.min(), D_qa.min()) * 0.95,
                   max(D_pool.max(), D_qa.max()) * 1.05, 80)
ax_b.hist(D_pool, bins=bins, alpha=0.4, color="grey",
          label=f"pool (n={len(D_pool)})", density=True)
ax_b.hist(D_qa, bins=bins, alpha=0.6, color="green",
          label=f"truth-qa (n={len(D_qa)})", density=True)
ax_b.axvline(D_truth, color="orange", lw=2, label=f"D_truth = {D_truth:.4f}")
ax_b.axvline(1.0, color="red", ls="--", lw=1, label="separatrix (D=1)")
ax_b.set_xlabel("polhode label  D = 2T·I_b / |L|²")
ax_b.set_ylabel("density")
ax_b.set_title("Polhode label distribution")
ax_b.legend(fontsize=8)
ax_b.grid(alpha=0.3)

# (c) D zoomed near truth.
ax_c = fig.add_subplot(gs[1, 0])
zoom_window = 0.05
mask_qa = np.abs(D_qa - D_truth) < zoom_window
mask_pool = np.abs(D_pool - D_truth) < zoom_window
zoom_bins = np.linspace(D_truth - zoom_window, D_truth + zoom_window, 60)
ax_c.hist(D_pool[mask_pool], bins=zoom_bins, alpha=0.4, color="grey",
          label=f"pool (in window n={int(mask_pool.sum())})", density=False)
ax_c.hist(D_qa[mask_qa], bins=zoom_bins, alpha=0.6, color="green",
          label=f"truth-qa (in window n={int(mask_qa.sum())})", density=False)
ax_c.axvline(D_truth, color="orange", lw=2, label="D_truth")
ax_c.set_xlabel("polhode label D (zoomed)")
ax_c.set_ylabel("count")
ax_c.set_title(f"Zoomed ±{zoom_window} around D_truth")
ax_c.legend(fontsize=8)
ax_c.grid(alpha=0.3)

# (d) Scatter D vs |L| with marker for truth.
ax_d = fig.add_subplot(gs[1, 1])
ax_d.scatter(L_pool, D_pool, s=2, c="lightgrey", alpha=0.4, label="pool")
ax_d.scatter(L_qa, D_qa, s=20, c="green", alpha=0.7, label="truth-qa")
ax_d.scatter(L_truth, D_truth, s=200, c="orange", marker="D", edgecolors="black",
             label="truth_t0", zorder=5)
ax_d.axhline(D_truth, color="orange", ls=":", lw=1, alpha=0.5)
ax_d.axvline(L_truth, color="orange", ls=":", lw=1, alpha=0.5)
ax_d.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5, label="separatrix")
ax_d.set_xlabel("|L|  (kg m²/s)")
ax_d.set_ylabel("D = 2T·I_b / |L|²")
ax_d.set_title("Polhode label vs scale")
ax_d.legend(fontsize=8)
ax_d.grid(alpha=0.3)

fig.suptitle(
    f"Seed 14 — does cascade noise preserve polhode label?  D_truth = {D_truth:.4f}",
    fontsize=14, y=0.99,
)
out = RESULTS / "polhode_label_test.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved: {out}")
