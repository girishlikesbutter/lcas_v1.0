"""Generate figures for the wind-down LaTeX report."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
FIG = SURVEY / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY))

# ---------------------------------------------------------------------------
# Cohort data (combine s040 + s042)
# ---------------------------------------------------------------------------
with open(SURVEY / "results/s040_basin_radius_3seed/summary.json") as f:
    S40 = json.load(f)
with open(SURVEY / "results/s042_basin_radius_cohort/summary.json") as f:
    S42 = json.load(f)

cohort = []
for k, v in S40["by_seed"].items():
    seed = int(k.replace("seed", ""))
    t = np.load(SURVEY / "data/trajectories" / f"traj_seed{seed:03d}.npz")
    cohort.append({
        "seed": seed,
        "om": float(t["omega_mag_dps"]),
        "src": "s040",
        "q0_along_t": v["truth"]["basin_radii"]["q0_along_omega"]["radius"],
        "q0_perp_t":  v["truth"]["basin_radii"]["q0_perp_omega"]["radius"],
        "om_pos_t":   v["truth"]["basin_radii"]["omega_mag_pct"]["radius_pos_pct"],
        "om_neg_t":   v["truth"]["basin_radii"]["omega_mag_pct"]["radius_neg_pct"],
        "om_dir_t":   v["truth"]["basin_radii"]["omega_dir_deg"]["radius"],
        "q0_along_w": v["twin"]["basin_radii"]["q0_along_omega"]["radius"],
        "q0_perp_w":  v["twin"]["basin_radii"]["q0_perp_omega"]["radius"],
        "om_pos_w":   v["twin"]["basin_radii"]["omega_mag_pct"]["radius_pos_pct"],
        "om_neg_w":   v["twin"]["basin_radii"]["omega_mag_pct"]["radius_neg_pct"],
        "om_dir_w":   v["twin"]["basin_radii"]["omega_dir_deg"]["radius"],
        "n_bA_t": v["truth"]["n_band_a"], "n_bB_t": v["truth"]["n_band_b"],
        "n_bC_t": v["truth"]["n_band_c"], "n_bD_t": v["truth"]["n_band_d"],
    })
for k, v in S42["by_seed"].items():
    seed = int(k.replace("seed", ""))
    cohort.append({
        "seed": seed,
        "om": v["omega_mag_dps"],
        "src": "s042",
        "q0_along_t": v["truth"]["basin_radii"]["q0_along_omega"]["radius"],
        "q0_perp_t":  v["truth"]["basin_radii"]["q0_perp_omega"]["radius"],
        "om_pos_t":   v["truth"]["basin_radii"]["omega_mag_pct"]["radius_pos_pct"],
        "om_neg_t":   v["truth"]["basin_radii"]["omega_mag_pct"]["radius_neg_pct"],
        "om_dir_t":   v["truth"]["basin_radii"]["omega_dir_deg"]["radius"],
        "q0_along_w": v["twin"]["basin_radii"]["q0_along_omega"]["radius"],
        "q0_perp_w":  v["twin"]["basin_radii"]["q0_perp_omega"]["radius"],
        "om_pos_w":   v["twin"]["basin_radii"]["omega_mag_pct"]["radius_pos_pct"],
        "om_neg_w":   v["twin"]["basin_radii"]["omega_mag_pct"]["radius_neg_pct"],
        "om_dir_w":   v["twin"]["basin_radii"]["omega_dir_deg"]["radius"],
        "n_bA_t": v["truth"]["n_band_a"], "n_bB_t": v["truth"]["n_band_b"],
        "n_bC_t": v["truth"]["n_band_c"], "n_bD_t": v["truth"]["n_band_d"],
    })
cohort.sort(key=lambda r: r["om"])
seeds = [r["seed"] for r in cohort]
oms = np.array([r["om"] for r in cohort])

# ===========================================================================
# Figure 1: ω-mag basin width vs |ω| (the cohort scaling)
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
om_pos_min = np.array([min(r["om_pos_t"], r["om_pos_w"]) for r in cohort])
om_neg_min = np.array([min(r["om_neg_t"], r["om_neg_w"]) for r in cohort])
om_minside = np.minimum(om_pos_min, om_neg_min)

# Saturated (basin = 10) vs measured
sat_mask = om_minside >= 10
meas_mask = ~sat_mask

ax.scatter(oms[meas_mask], om_minside[meas_mask], s=70, c="#2b6cb0",
           label="measured edge", zorder=3)
ax.scatter(oms[sat_mask], om_minside[sat_mask], s=70, c="#cbd5e0",
           edgecolors="#2b6cb0", linewidths=1.5, marker="^",
           label="saturated (≥10%; lower bound)", zorder=3)

# Annotate seeds
for r, om, basin in zip(cohort, oms, om_minside):
    color = "#2b6cb0" if basin < 10 else "#666666"
    weight = "bold" if r["seed"] == 14 else "normal"
    ax.annotate(f" {r['seed']}", (om, basin), fontsize=9, color=color,
                fontweight=weight, va="center")

# Cohort scaling rule: 1.5% / |ω|^0.7
om_grid = np.linspace(0.1, 1.5, 100)
rule = 1.5 / om_grid**0.7
ax.plot(om_grid, rule, "--", c="#e53e3e", lw=1.5, alpha=0.6,
        label=r"rule: $\delta|\omega|/|\omega|\;[\%] \approx 1.5/|\omega|_{\rm dps}^{0.7}$")

ax.set_xlabel(r"$|\omega|_{\rm truth}$  (dps)", fontsize=11)
ax.set_ylabel(r"min(\,$+$, $-$\,) $\omega$-magnitude basin width  (\%)", fontsize=11)
ax.set_title(r"$\omega$-magnitude basin width vs rotation rate (13-seed cohort)",
             fontsize=12)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.09, 1.6); ax.set_ylim(0.3, 13)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig(FIG / "fig1_omega_mag_vs_omega.pdf"); plt.close()
print(f"Saved: {FIG / 'fig1_omega_mag_vs_omega.pdf'}")

# ===========================================================================
# Figure 2: Cohort basin radii heatmap
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
axes_to_plot = [
    ("q0_along\n(°)",    [min(r["q0_along_t"] or 0, r["q0_along_w"] or 0) for r in cohort], 60.0),
    ("q0_perp\n(°)",     [min(r["q0_perp_t"] or 0, r["q0_perp_w"] or 0)  for r in cohort], 60.0),
    ("ω-mag +\n(\\%)",   om_pos_min, 10.0),
    ("ω-mag −\n(\\%)",   om_neg_min, 10.0),
    ("ω-dir\n(°)",       [min(r["om_dir_t"] or 0, r["om_dir_w"] or 0)    for r in cohort], 10.0),
]
data = np.array([row[1] for row in axes_to_plot], dtype=float)
labels = [row[0] for row in axes_to_plot]
norms  = np.array([row[2] for row in axes_to_plot])
data_norm = (data.T / norms).T
im = ax.imshow(data_norm, aspect="auto", cmap="RdYlGn",
               vmin=0, vmax=1.0, interpolation="nearest")
for i, _ in enumerate(axes_to_plot):
    for j, om_v in enumerate(oms):
        v = data[i, j]
        txt = f"{v:.1f}" if v < axes_to_plot[i][2] else f"≥{v:.0f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="black" if data_norm[i, j] > 0.4 else "white")

ax.set_xticks(range(len(seeds)))
ax.set_xticklabels([f"{s}\n{om:.2f}dps" for s, om in zip(seeds, oms)],
                   fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("seed (sorted by $|\\omega|$ ascending)", fontsize=11)
ax.set_title("Cohort basin radii (min over truth/twin) — saturation marked $\\geq$",
             fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
             label="basin / grid edge (1.0 = saturated)")
plt.tight_layout()
plt.savefig(FIG / "fig2_basin_heatmap.pdf"); plt.close()
print(f"Saved: {FIG / 'fig2_basin_heatmap.pdf'}")

# ===========================================================================
# Figure 3: Seed 14 cost surface profile (rho_final vs perturbation)
# ===========================================================================
with open(SURVEY / "results/s042_basin_radius_cohort/seed014/truth_basin.json") as f:
    s14 = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
axis_names = [("q0_along_omega", "$q_0$ along $\\omega$ axis  (deg)"),
              ("q0_perp_omega",  "$q_0$ perpendicular to $\\omega$  (deg)"),
              ("omega_mag_pct",  "$\\omega$-magnitude perturbation  (\\%)"),
              ("omega_dir_deg",  "$\\omega$-direction perturbation  (deg)")]
for ax, (name, label) in zip(axes.ravel(), axis_names):
    sub = sorted([r for r in s14["results"] if r["axis"] == name],
                 key=lambda r: r["mag"])
    mags  = [r["mag"] for r in sub]
    rhos  = [r["rho_final"] for r in sub]
    bands = [r["band"] for r in sub]
    color_map = {"A": "#2b6cb0", "B": "#dd6b20", "C": "#9f7aea", "D": "#e53e3e"}
    for r in sub:
        ax.scatter(r["mag"], r["rho_final"], s=80,
                   c=color_map[r["band"]], edgecolors="black", lw=0.5, zorder=3)
    ax.axhline(2, ls="--", c="#2b6cb0", alpha=0.5, lw=1, label="Band A/B")
    ax.axhline(4, ls="--", c="#dd6b20", alpha=0.5, lw=1, label="Band B/C")
    ax.axhline(8, ls="--", c="#9f7aea", alpha=0.5, lw=1, label="Band C/D")
    ax.set_yscale("log")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel(r"$\rho_{\rm final}$ (log)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(name.replace("_", " "), fontsize=10)
fig.suptitle("Seed 14 truth basin: per-axis $\\rho_{\\rm final}$ vs perturbation magnitude\n"
             "(blue = Band A, orange = Band B/multi-sol attractor, red = Band D escape)",
             fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "fig3_seed14_profile.pdf"); plt.close()
print(f"Saved: {FIG / 'fig3_seed14_profile.pdf'}")

# ===========================================================================
# Figure 4: Body-twin LC equivalence (s043) — show truth/twin differences
# ===========================================================================
with open(SURVEY / "results/s043_twin_hifi_verify/summary.json") as f:
    S43 = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, seed in zip(axes, [23, 28, 89]):
    d = np.load(SURVEY / "results/s043_twin_hifi_verify" / f"seed{seed:03d}_diff.npz")
    diff = d["truth_vs_twin"]
    valid = d["valid_mask"]
    epochs = np.arange(len(valid))[valid]
    ax.plot(epochs, np.abs(diff), c="#2b6cb0", lw=0.7, alpha=0.8)
    ax.axhline(0.05, ls="--", c="#e53e3e", lw=1.5, label="0.05 mag noise floor")
    ax.set_yscale("log")
    ax.set_xlabel("epoch index", fontsize=10)
    ax.set_ylabel(r"$|\Delta_{\rm mag}|$ (truth − twin)  (mag)", fontsize=10)
    ax.set_title(f"seed {seed}: max $|\\Delta|=$ "
                 f"{S43['seeds'][f'seed{seed:03d}']['max_truth_vs_twin_mag']:.2e} mag",
                 fontsize=10)
    ax.set_ylim(1e-12, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle(r"Body-twin LC equivalence: $|{\rm mag}_{\rm truth}-{\rm mag}_{\rm twin}|$ "
             r"per epoch (s043 hi-fi)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "fig4_twin_diff.pdf"); plt.close()
print(f"Saved: {FIG / 'fig4_twin_diff.pdf'}")

# ===========================================================================
# Figure 5: ω-direction basin vs body-frame ω geometry (negative result fig)
# ===========================================================================
from scipy.spatial.transform import Rotation
rows = []
for r in cohort:
    t = np.load(SURVEY / "data/trajectories" / f"traj_seed{r['seed']:03d}.npz")
    qxyzw = t["q0_wxyz"][[1, 2, 3, 0]]
    R_i2b = Rotation.from_quat(qxyzw).as_matrix()
    om_body = R_i2b @ t["omega0_rad"]
    om_norm = float(np.linalg.norm(om_body))
    om_xy = float(np.linalg.norm(om_body[[0, 1]]))
    om_xy_frac = om_xy / om_norm
    rows.append((r["seed"], r["om"], om_xy_frac,
                 min(r["om_dir_t"] or 0, r["om_dir_w"] or 0)))

fig, ax = plt.subplots(figsize=(8.5, 5))
xs = np.array([r[2] for r in rows])
ys = np.array([r[3] for r in rows])
sat = ys >= 10
ax.scatter(xs[~sat], ys[~sat], s=70, c="#2b6cb0", label="measured edge",
           zorder=3)
ax.scatter(xs[sat], ys[sat], s=70, c="#cbd5e0", marker="^",
           edgecolors="#2b6cb0", linewidths=1.5,
           label="saturated (≥10°)", zorder=3)
for r in rows:
    weight = "bold" if r[0] == 14 else "normal"
    ax.annotate(f" {r[0]}", (r[2], r[3]), fontsize=9, va="center",
                fontweight=weight)
ax.set_xlabel(r"$|\omega_{xy}^{\rm body}|/|\omega|$  (fraction in symmetric I-plane)",
              fontsize=11)
ax.set_ylabel(r"$\omega$-direction basin (deg)", fontsize=11)
ax.set_title(r"Negative result: body-frame $\omega$ geometry does NOT predict "
             r"$\omega$-dir basin width" + "\n"
             r"(Pearson $r = +0.04$, $p = 0.89$; methodologically underpowered — see report)",
             fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "fig5_omega_dir_geometry.pdf"); plt.close()
print(f"Saved: {FIG / 'fig5_omega_dir_geometry.pdf'}")

print(f"\nAll 5 figures written to {FIG}")
