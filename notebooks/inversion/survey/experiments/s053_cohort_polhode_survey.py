"""s053 — cohort polhode survey (all 100 m048 seeds).

For each seed compute:
  - |L|       = ||I·ω0||                (kg·m²/s) — angular momentum magnitude
  - 2T        = ω0·I·ω0                 (J)       — twice rotational kinetic energy
  - D         = 2T·I_b / |L|²           (-)       — polhode label
                  D < 1 → encloses I_a (low-inertia) axis
                  D = 1 → separatrix
                  D > 1 → encloses I_c (high-inertia) axis
  - Δ_polhode = max pairwise L2 distance across propagated ω(t) (dps)
                — polhode "diameter" in body-frame ω-space
  - |ω| std/mean (-) — magnitude variation amplitude
  - cone_angle = max angle between ω̂(t) and ω̂(t=0) (deg) — angular extent
                  of the polhode trace on the unit sphere

Cross-correlate D and polhode-diameter with s042's measured basin widths
on the 10 seeds tested there: [6, 14, 16, 19, 42, 44, 57, 62, 79, 84].

If high-|ω| seeds cluster near the separatrix (D ≈ 1) AND the s042 narrow
basins live in that subset → s051 mechanism is cohort-wide.

If D distributes broadly across |ω| with no pattern → s042's basin scaling
is mechanism-different from polhode topology; s051 reframe is structural
but not the load-bearing predictor of basin width.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

from lib.hifi_render import build_context, _build_model  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s053_cohort_polhode_survey"
RESULTS.mkdir(parents=True, exist_ok=True)
N_SEEDS = 100
S042_PATH = SURVEY_DIR / "results" / "s042_basin_radius_cohort" / "summary.json"

# Pre-load model once (cached).
_, I = _build_model()
eig_vals, _ = np.linalg.eigh(I)
I_a, I_b, I_c = float(eig_vals[0]), float(eig_vals[1]), float(eig_vals[2])
print(f"Inertia eigenvalues: I_a={I_a:.1f}  I_b={I_b:.1f}  I_c={I_c:.1f}  (kg·m²)")
print(f"Asymmetry: I_b/I_c={I_b/I_c:.4f}, I_c/I_a={I_c/I_a:.4f}\n")


def per_seed(seed: int) -> dict:
    """Compute polhode invariants and trajectory for one seed.

    Reuses the cached SPICE state from traj_seedXXX.npz; propagates
    (q0_truth, ω0_truth) over the full observation window.
    """
    truth = load_truth(seed)
    om0 = truth["omega0_rad"]
    q0 = truth["q0_wxyz"]
    obs_times = truth["observation_times"]

    L = I @ om0
    Lmag = float(np.linalg.norm(L))
    twoT = float(om0 @ I @ om0)
    D = twoT * I_b / Lmag**2
    om0_dps = float(np.linalg.norm(om0)) * 180.0 / np.pi

    quats, om_history = propagate_attitude(
        q0=q0, omega0=om0, times=obs_times,
        mode="tumbling", inertia_tensor=I,
    )
    om_history_dps = om_history * 180.0 / np.pi

    # Polhode diameter (max pairwise L2 distance across samples, dps).
    diff = om_history_dps[:, None, :] - om_history_dps[None, :, :]
    pol_diam = float(np.sqrt(np.sum(diff * diff, axis=2)).max())

    # |ω| std/mean.
    om_mag = np.linalg.norm(om_history, axis=1)
    om_std_over_mean = float(om_mag.std() / om_mag.mean())

    # Cone angle: max angle from ω̂(0) over the trajectory.
    om_hat = om_history / np.linalg.norm(om_history, axis=1, keepdims=True)
    cone_cos = np.clip(om_hat @ om_hat[0], -1.0, 1.0)
    cone_max_deg = float(np.degrees(np.arccos(cone_cos.min())))

    return {
        "seed": int(seed),
        "om_mag_dps": om0_dps,
        "Lmag": Lmag,
        "twoT": twoT,
        "D": float(D),
        "pol_diam_dps": pol_diam,
        "om_std_over_mean": om_std_over_mean,
        "cone_max_deg": cone_max_deg,
    }


# ---------------------------------------------------------------------------
# Run cohort.
# ---------------------------------------------------------------------------
print(f"[1/3] Computing polhode invariants for {N_SEEDS} seeds (sequential) ...")
t0 = time.time()
results = []
for seed in range(N_SEEDS):
    results.append(per_seed(seed))
    if (seed + 1) % 10 == 0:
        print(f"    {seed+1}/{N_SEEDS} done at {time.time()-t0:.1f}s")
print(f"    done in {time.time()-t0:.2f}s\n")

# Stack arrays.
seeds = np.array([r["seed"] for r in results])
om_mag_dps = np.array([r["om_mag_dps"] for r in results])
Lmag = np.array([r["Lmag"] for r in results])
twoT = np.array([r["twoT"] for r in results])
D = np.array([r["D"] for r in results])
pol_diam = np.array([r["pol_diam_dps"] for r in results])
om_std = np.array([r["om_std_over_mean"] for r in results])
cone_max = np.array([r["cone_max_deg"] for r in results])

print("Cohort-wide polhode summary (n=100):")
print(f"  |ω| dps   :  min={om_mag_dps.min():.4f} median={np.median(om_mag_dps):.4f} max={om_mag_dps.max():.4f}")
print(f"  D         :  min={D.min():.4f} median={np.median(D):.4f} max={D.max():.4f}")
print(f"  pol_diam  :  min={pol_diam.min():.4f} median={np.median(pol_diam):.4f} max={pol_diam.max():.4f} dps")
print(f"  cone_max  :  min={cone_max.min():.2f}° median={np.median(cone_max):.2f}° max={cone_max.max():.2f}°")
print(f"  |ω| std/mean: min={om_std.min()*100:.2f}% median={np.median(om_std)*100:.2f}% max={om_std.max()*100:.2f}%")
print()

# Classify by polhode topology.
near_sep = (np.abs(D - 1.0) < 0.05).sum()
encl_a = (D < 1.0).sum()  # encloses I_a (low-inertia, "minor axis")
encl_c = (D > 1.0).sum()  # encloses I_c (high-inertia, "major axis")
print(f"Polhode classification:")
print(f"  D < 1.0 (encloses I_a, low-inertia axis): {encl_a}/100")
print(f"  D > 1.0 (encloses I_c, high-inertia axis): {encl_c}/100")
print(f"  |D - 1.0| < 0.05 (near separatrix):       {near_sep}/100")
print()

# ---------------------------------------------------------------------------
# Cross-correlate with s042.
# ---------------------------------------------------------------------------
print("[2/3] Cross-correlating with s042 basin widths ...")
with open(S042_PATH) as f:
    s042 = json.load(f)
s042_seeds = []
s042_om_mag_pct = []  # ω-mag basin radius
s042_om_dir_deg = []
s042_q0_perp_deg = []
for key, ent in s042["by_seed"].items():
    seed_id = int(key.replace("seed", ""))
    br = ent["truth"]["basin_radii"]
    s042_seeds.append(seed_id)
    # ω-mag basin: take min(pos, neg) — the binding side.
    om_pct = min(br["omega_mag_pct"]["radius_neg_pct"],
                 br["omega_mag_pct"]["radius_pos_pct"])
    s042_om_mag_pct.append(om_pct)
    s042_om_dir_deg.append(br["omega_dir_deg"]["radius"])
    s042_q0_perp_deg.append(br["q0_perp_omega"]["radius"])

s042_seeds = np.array(s042_seeds)
s042_om_mag_pct = np.array(s042_om_mag_pct)
s042_om_dir_deg = np.array(s042_om_dir_deg)
s042_q0_perp_deg = np.array(s042_q0_perp_deg)

# Get the cohort polhode values for these 10 seeds.
order = np.array([int(np.where(seeds == s)[0][0]) for s in s042_seeds])
om_mag_at_s042 = om_mag_dps[order]
D_at_s042 = D[order]
pol_diam_at_s042 = pol_diam[order]
cone_at_s042 = cone_max[order]

print(f"  s042 seeds: {sorted(s042_seeds.tolist())}")
print(f"  Their polhode values:")
for i, s in enumerate(np.argsort(om_mag_at_s042)):
    print(f"    seed {s042_seeds[s]:3d}: |ω|={om_mag_at_s042[s]:.4f} dps  "
          f"D={D_at_s042[s]:.3f}  pol_diam={pol_diam_at_s042[s]:.3f} dps  "
          f"cone={cone_at_s042[s]:5.1f}°  ω-mag basin±{s042_om_mag_pct[s]:.1f}%")

from scipy.stats import spearmanr
corr_D, p_D = spearmanr(D_at_s042, s042_om_mag_pct)
corr_pol_diam, p_pd = spearmanr(pol_diam_at_s042, s042_om_mag_pct)
corr_cone, p_cn = spearmanr(cone_at_s042, s042_om_mag_pct)
corr_om, p_om = spearmanr(om_mag_at_s042, s042_om_mag_pct)
corr_dDsep, p_dd = spearmanr(np.abs(D_at_s042 - 1.0), s042_om_mag_pct)

print(f"\nSpearman ρ correlations vs s042 ω-mag basin width (n=10):")
print(f"  |ω|         : ρ={corr_om:+.3f}  p={p_om:.3f}")
print(f"  D           : ρ={corr_D:+.3f}  p={p_D:.3f}")
print(f"  |D - 1|     : ρ={corr_dDsep:+.3f}  p={p_dd:.3f}  (near-separatrix → narrow basin?)")
print(f"  pol_diam    : ρ={corr_pol_diam:+.3f}  p={p_pd:.3f}")
print(f"  cone_max    : ρ={corr_cone:+.3f}  p={p_cn:.3f}")

# ---------------------------------------------------------------------------
# Save NPZ + JSON.
# ---------------------------------------------------------------------------
np.savez(
    RESULTS / "cohort.npz",
    seeds=seeds, om_mag_dps=om_mag_dps, Lmag=Lmag, twoT=twoT, D=D,
    pol_diam_dps=pol_diam, om_std_over_mean=om_std, cone_max_deg=cone_max,
    s042_seeds=s042_seeds, s042_om_mag_pct=s042_om_mag_pct,
    s042_om_dir_deg=s042_om_dir_deg, s042_q0_perp_deg=s042_q0_perp_deg,
)
summary = {
    "n_seeds": int(N_SEEDS),
    "I_eigenvalues": [I_a, I_b, I_c],
    "polhode_classification": {
        "encloses_I_a_n": int(encl_a),
        "encloses_I_c_n": int(encl_c),
        "near_separatrix_n": int(near_sep),
    },
    "cohort_stats": {
        "om_mag_dps": {"min": float(om_mag_dps.min()), "median": float(np.median(om_mag_dps)),
                       "max": float(om_mag_dps.max())},
        "D": {"min": float(D.min()), "median": float(np.median(D)),
              "max": float(D.max())},
        "pol_diam_dps": {"min": float(pol_diam.min()), "median": float(np.median(pol_diam)),
                         "max": float(pol_diam.max())},
        "cone_max_deg": {"min": float(cone_max.min()), "median": float(np.median(cone_max)),
                         "max": float(cone_max.max())},
    },
    "s042_correlations_n10": {
        "om_mag":    {"rho": float(corr_om), "p": float(p_om)},
        "D":         {"rho": float(corr_D), "p": float(p_D)},
        "abs_D_m1":  {"rho": float(corr_dDsep), "p": float(p_dd)},
        "pol_diam":  {"rho": float(corr_pol_diam), "p": float(p_pd)},
        "cone_max":  {"rho": float(corr_cone), "p": float(p_cn)},
    },
}
with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------
print("\n[3/3] Rendering figure ...")
fig = plt.figure(figsize=(16, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# (a) D vs |ω|.
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(om_mag_dps, D, s=20, c="steelblue", alpha=0.7, label="cohort (n=100)")
ax_a.scatter(om_mag_at_s042, D_at_s042, s=80, c="orange", marker="o",
             edgecolors="black", label="s042 seeds (n=10)", zorder=5)
ax_a.axhline(1.0, color="red", ls="--", lw=1, label="separatrix (D=1)")
ax_a.set_xlabel("|ω| (dps)")
ax_a.set_ylabel("D = 2T·I_b / |L|²")
ax_a.set_title("Polhode label vs |ω| — cohort distribution")
ax_a.legend(fontsize=8)
ax_a.grid(alpha=0.3)

# (b) Polhode diameter vs |ω|.
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(om_mag_dps, pol_diam, s=20, c="steelblue", alpha=0.7)
ax_b.scatter(om_mag_at_s042, pol_diam_at_s042, s=80, c="orange",
             edgecolors="black", zorder=5)
ax_b.set_xlabel("|ω| (dps)")
ax_b.set_ylabel("polhode diameter (dps)")
ax_b.set_title("Polhode diameter vs |ω|")
ax_b.grid(alpha=0.3)

# (c) Cone angle vs |ω|.
ax_c = fig.add_subplot(gs[0, 2])
ax_c.scatter(om_mag_dps, cone_max, s=20, c="steelblue", alpha=0.7)
ax_c.scatter(om_mag_at_s042, cone_at_s042, s=80, c="orange",
             edgecolors="black", zorder=5)
ax_c.set_xlabel("|ω| (dps)")
ax_c.set_ylabel("max cone angle (deg)")
ax_c.set_title("Polhode angular extent (max ∠ from ω̂(0))")
ax_c.grid(alpha=0.3)

# (d) D distribution.
ax_d = fig.add_subplot(gs[1, 0])
ax_d.hist(D, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
for ds in D_at_s042:
    ax_d.axvline(ds, color="orange", lw=1, alpha=0.6)
ax_d.axvline(1.0, color="red", ls="--", lw=1.5, label="separatrix")
ax_d.set_xlabel("D")
ax_d.set_ylabel("# seeds")
ax_d.set_title(f"Cohort D distribution\n({encl_a}/100 enclose I_a, "
               f"{encl_c}/100 enclose I_c, {near_sep} near-separatrix)")
ax_d.legend(fontsize=8)
ax_d.grid(alpha=0.3)

# (e) Pol diameter distribution.
ax_e = fig.add_subplot(gs[1, 1])
ax_e.hist(pol_diam, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
for pd_v in pol_diam_at_s042:
    ax_e.axvline(pd_v, color="orange", lw=1, alpha=0.6)
ax_e.set_xlabel("polhode diameter (dps)")
ax_e.set_ylabel("# seeds")
ax_e.set_title("Polhode diameter distribution")
ax_e.grid(alpha=0.3)

# (f) Cone-angle distribution.
ax_f = fig.add_subplot(gs[1, 2])
ax_f.hist(cone_max, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
for cv in cone_at_s042:
    ax_f.axvline(cv, color="orange", lw=1, alpha=0.6)
ax_f.set_xlabel("max cone angle (deg)")
ax_f.set_ylabel("# seeds")
ax_f.set_title("Polhode angular extent distribution")
ax_f.grid(alpha=0.3)

# (g) ω-mag basin vs D — the predictor test.
ax_g = fig.add_subplot(gs[2, 0])
sc = ax_g.scatter(D_at_s042, s042_om_mag_pct, s=80,
                  c=om_mag_at_s042, cmap="plasma", edgecolors="black")
plt.colorbar(sc, ax=ax_g, label="|ω| (dps)")
for k, sk in enumerate(s042_seeds):
    ax_g.annotate(f"  {sk}", (D_at_s042[k], s042_om_mag_pct[k]), fontsize=9)
ax_g.axvline(1.0, color="red", ls="--", lw=1, alpha=0.5)
ax_g.set_xlabel("polhode label D")
ax_g.set_ylabel("ω-mag basin radius (% of |ω|)")
ax_g.set_title(f"s042 ω-mag basin vs D\nSpearman ρ={corr_D:+.2f}, p={p_D:.2f}")
ax_g.grid(alpha=0.3)

# (h) ω-mag basin vs |D - 1| (near-separatrix).
ax_h = fig.add_subplot(gs[2, 1])
absDm1 = np.abs(D_at_s042 - 1.0)
sc2 = ax_h.scatter(absDm1, s042_om_mag_pct, s=80,
                   c=om_mag_at_s042, cmap="plasma", edgecolors="black")
plt.colorbar(sc2, ax=ax_h, label="|ω| (dps)")
for k, sk in enumerate(s042_seeds):
    ax_h.annotate(f"  {sk}", (absDm1[k], s042_om_mag_pct[k]), fontsize=9)
ax_h.set_xlabel("|D - 1|  (distance to separatrix)")
ax_h.set_ylabel("ω-mag basin radius (% of |ω|)")
ax_h.set_title(f"s042 ω-mag basin vs distance-to-separatrix\nSpearman ρ={corr_dDsep:+.2f}, p={p_dd:.2f}")
ax_h.grid(alpha=0.3)

# (i) ω-mag basin vs polhode diameter (THE HEADLINE).
ax_i = fig.add_subplot(gs[2, 2])
sc3 = ax_i.scatter(pol_diam_at_s042, s042_om_mag_pct, s=80,
                   c=om_mag_at_s042, cmap="plasma", edgecolors="black")
plt.colorbar(sc3, ax=ax_i, label="|ω| (dps)")
for k, sk in enumerate(s042_seeds):
    ax_i.annotate(f"  {sk}", (pol_diam_at_s042[k], s042_om_mag_pct[k]), fontsize=9)
ax_i.set_xlabel("polhode diameter (dps)")
ax_i.set_ylabel("ω-mag basin radius (% of |ω|)")
ax_i.set_title(f"s042 ω-mag basin vs polhode diameter\nSpearman ρ={corr_pol_diam:+.2f}, p={p_pd:.4f}  ★HEADLINE")
ax_i.grid(alpha=0.3)

fig.suptitle(
    "s053 — cohort polhode survey (100 seeds) + s042 cross-correlation (10 seeds)",
    fontsize=14, y=0.995,
)
out = RESULTS / "cohort_polhode_survey.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved: {out}")
print(f"Saved: {RESULTS / 'cohort.npz'}")
print(f"Saved: {RESULTS / 'summary.json'}")
