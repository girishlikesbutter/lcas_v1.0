"""s054b — quick polhode-stats sanity check on holdout vs cohort.

Runs s053's per-seed polhode-invariant computation on the 20 holdout
seeds (100–119) and compares the (|L|, 2T, D, pol_diam) distribution
to the 100-seed cohort. Sanity check: does the holdout look like a
sample from the same generator?

If yes (KS-test high p-value or visual overlap): the generator is
stable and we can proceed with cohort-prior holdout-validation tests.

If no: the holdout corpus is anomalous (e.g., a code change shifted the
distribution); s055+ tests would need to wait for diagnosis.

Output: PNG with cohort vs holdout histograms for each polhode invariant.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

from lib.hifi_render import _build_model  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s054_holdout"
RESULTS.mkdir(parents=True, exist_ok=True)
COHORT_NPZ = SURVEY_DIR / "results" / "s053_cohort_polhode_survey" / "cohort.npz"

# Pre-load model.
_, I = _build_model()
eig_vals, _ = np.linalg.eigh(I)
I_a, I_b, I_c = float(eig_vals[0]), float(eig_vals[1]), float(eig_vals[2])

HOLDOUT_SEEDS = list(range(100, 120))


def per_seed(seed: int) -> dict:
    truth = load_truth(seed)
    om0 = truth["omega0_rad"]
    q0 = truth["q0_wxyz"]
    obs_times = truth["observation_times"]

    L = I @ om0
    Lmag = float(np.linalg.norm(L))
    twoT = float(om0 @ I @ om0)
    D = twoT * I_b / Lmag**2
    om0_dps = float(np.linalg.norm(om0)) * 180.0 / np.pi

    _, om_history = propagate_attitude(
        q0=q0, omega0=om0, times=obs_times,
        mode="tumbling", inertia_tensor=I,
    )
    om_history_dps = om_history * 180.0 / np.pi
    diff = om_history_dps[:, None, :] - om_history_dps[None, :, :]
    pol_diam = float(np.sqrt(np.sum(diff * diff, axis=2)).max())
    om_mag = np.linalg.norm(om_history, axis=1)
    om_std_over_mean = float(om_mag.std() / om_mag.mean())
    om_hat = om_history / np.linalg.norm(om_history, axis=1, keepdims=True)
    cone_cos = np.clip(om_hat @ om_hat[0], -1.0, 1.0)
    cone_max_deg = float(np.degrees(np.arccos(cone_cos.min())))

    return {
        "seed": int(seed),
        "om_mag_dps": om0_dps, "Lmag": Lmag, "twoT": twoT, "D": float(D),
        "pol_diam_dps": pol_diam, "om_std_over_mean": om_std_over_mean,
        "cone_max_deg": cone_max_deg,
    }


print(f"s054b — polhode stats on {len(HOLDOUT_SEEDS)} holdout seeds")
t0 = time.time()
results = [per_seed(s) for s in HOLDOUT_SEEDS]
print(f"  done in {time.time()-t0:.2f}s\n")

ho_om = np.array([r["om_mag_dps"] for r in results])
ho_D = np.array([r["D"] for r in results])
ho_pol = np.array([r["pol_diam_dps"] for r in results])
ho_cone = np.array([r["cone_max_deg"] for r in results])
ho_Lmag = np.array([r["Lmag"] for r in results])

# Cohort.
co = np.load(COHORT_NPZ)
co_om = co["om_mag_dps"]
co_D = co["D"]
co_pol = co["pol_diam_dps"]
co_cone = co["cone_max_deg"]
co_Lmag = co["Lmag"]

print("KS 2-sample tests (cohort vs holdout):")
for name, co_v, ho_v in [
    ("|ω|", co_om, ho_om),
    ("D", co_D, ho_D),
    ("pol_diam", co_pol, ho_pol),
    ("cone_max", co_cone, ho_cone),
    ("|L|", co_Lmag, ho_Lmag),
]:
    res = ks_2samp(co_v, ho_v)
    print(f"  {name:8s}: D={res.statistic:.3f}  p={res.pvalue:.3f}")

# Polhode classification on holdout.
ho_encl_a = (ho_D < 1.0).sum()
ho_encl_c = (ho_D > 1.0).sum()
ho_near_sep = (np.abs(ho_D - 1.0) < 0.05).sum()
print(f"\nHoldout topology:")
print(f"  D < 1.0 (encloses I_a):  {ho_encl_a}/20  (cohort: {(co_D < 1.0).sum()}/100)")
print(f"  D > 1.0 (encloses I_c):  {ho_encl_c}/20  (cohort: {(co_D > 1.0).sum()}/100)")
print(f"  near-separatrix:         {ho_near_sep}/20  (cohort: {(np.abs(co_D - 1.0) < 0.05).sum()}/100)")

print(f"\nHoldout (|ω|, pol_diam) ranges:")
print(f"  |ω|: {ho_om.min():.3f} → {ho_om.max():.3f} (cohort: {co_om.min():.3f} → {co_om.max():.3f})")
print(f"  pol_diam: {ho_pol.min():.3f} → {ho_pol.max():.3f} (cohort: {co_pol.min():.3f} → {co_pol.max():.3f})")

# Save NPZ.
np.savez(
    RESULTS / "holdout_polhode.npz",
    seeds=np.array([r["seed"] for r in results]),
    om_mag_dps=ho_om, Lmag=ho_Lmag, twoT=np.array([r["twoT"] for r in results]),
    D=ho_D, pol_diam_dps=ho_pol,
    cone_max_deg=ho_cone,
    om_std_over_mean=np.array([r["om_std_over_mean"] for r in results]),
)

# Figure: 4 panels comparing cohort (filled hist) vs holdout (overlay with markers).
fig = plt.figure(figsize=(15, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

panels = [
    ("|ω| (dps)", co_om, ho_om, np.linspace(0, 1.6, 30)),
    ("D = 2T·I_b / |L|²", co_D, ho_D, np.linspace(0.95, 5.0, 40)),
    ("polhode diameter (dps)", co_pol, ho_pol, np.linspace(0, 3.0, 30)),
    ("cone_max (deg)", co_cone, ho_cone, np.linspace(0, 180, 30)),
]
for k, (title, co_v, ho_v, bins) in enumerate(panels):
    ax = fig.add_subplot(gs[k // 2, k % 2])
    ax.hist(co_v, bins=bins, alpha=0.55, color="steelblue",
            density=True, edgecolor="black", linewidth=0.5,
            label=f"cohort (n=100)")
    ax.hist(ho_v, bins=bins, alpha=0.7, color="orange",
            density=True, edgecolor="black", linewidth=0.5,
            label=f"holdout (n=20)")
    res = ks_2samp(co_v, ho_v)
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    ax.set_title(f"{title} — KS p={res.pvalue:.2f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("s054b — holdout (seeds 100..119) vs cohort (seeds 0..99) polhode invariants",
             fontsize=13, y=0.995)
out = RESULTS / "holdout_vs_cohort.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved: {out}")
print(f"Saved: {RESULTS / 'holdout_polhode.npz'}")
