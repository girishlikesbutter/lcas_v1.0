"""Compare v1 vs v2 survival clouds at the absolute LC extrema of seed 14.

Reuses the cached 500k random-quaternion pool from
`results/s049_cascade_seed14/scan.npz` (v1-rendered there). At the global
brightest (argmin mag) and dimmest (argmax mag) epochs of seed 14's LC,
re-render both v1 and v2 clouds at the same pool and TOL=0.10 mag.

Output: a single PNG with four histograms (2 epochs × 2 surrogates) of
`pred - measured` residuals, with TOL=±0.10 marked, |C_t| labelled.

Quick task: ~10 sec compute Pool(1).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

SURVEY_ROOT = Path("/home/girish/projects/lcas_v1.0/notebooks/inversion/survey")
sys.path.insert(0, str(SURVEY_ROOT))

# v1 surrogate
sys.path.insert(0, "/home/girish/surrogate_model")
from surrogate import SurrogateModel as V1Model  # noqa: E402

# v2 surrogate via lib wrapper
from lib.surrogate_eval import get_model as get_v2_model  # noqa: E402

SEED = 14
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
TOL_MAG = 0.10

# ---------- load trajectory + cached pool ----------
traj = np.load(SURVEY_ROOT / "data" / "trajectories" / f"traj_seed{SEED:03d}.npz")
sun_vec = traj["sun_pos"] - traj["sat_pos"]
obs_vec = traj["obs_pos"] - traj["sat_pos"]
sun_unit_all = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
obs_unit_all = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
obs_dist_all = traj["obs_dist"]
mag_hifi = traj["mag_hifi"]

epoch_bright = int(np.argmin(mag_hifi))   # absolute LC brightest
epoch_dim = int(np.argmax(mag_hifi))      # absolute LC dimmest
mag_bright = float(mag_hifi[epoch_bright])
mag_dim = float(mag_hifi[epoch_dim])
print(f"Seed {SEED}: |ω|={float(traj['omega_mag_dps']):.3f} dps")
print(f"  Brightest epoch: t={epoch_bright}  mag={mag_bright:.3f}")
print(f"  Dimmest   epoch: t={epoch_dim}  mag={mag_dim:.3f}")

scan = np.load(SURVEY_ROOT / "results" / "s049_cascade_seed14" / "scan.npz")
q_pool_wxyz = scan["q_pool_wxyz"]  # (500_000, 4) wxyz
N_POOL = len(q_pool_wxyz)
print(f"Pool size: {N_POOL}")

# scipy uses xyzw — convert
q_pool_xyzw = q_pool_wxyz[:, [1, 2, 3, 0]]
R_cache = R.from_quat(q_pool_xyzw).as_matrix()  # (N, 3, 3)

# ---------- v1 model ----------
v1_w = "/home/girish/surrogate_model/s10_5M_weights.npz"
v1_n = "/home/girish/surrogate_model/s10_5M_normalization.npz"
v1 = V1Model(v1_w, v1_n)

# ---------- v2 model ----------
v2 = get_v2_model()


def render_cloud(ep, model, label):
    k1_body = np.einsum('nij,j->ni', R_cache, sun_unit_all[ep])
    k2_body = np.einsum('nij,j->ni', R_cache, obs_unit_all[ep])
    obs_dist = np.full(N_POOL, obs_dist_all[ep])
    panel = np.full(N_POOL, SP_ANGLE_DEG)
    dish = np.full(N_POOL, AD_ANGLE_DEG)
    pred = model.predict_magnitude(k1_body, k2_body, panel, dish, obs_dist)
    residual = pred - mag_hifi[ep]
    finite = np.isfinite(residual)
    survive = finite & (np.abs(residual) < TOL_MAG)
    print(f"  {label} ep={ep}: |C_t|={int(survive.sum()):>6d}  "
          f"finite={int(finite.sum())}/{N_POOL}  "
          f"residual_p1_p99={np.nanpercentile(residual[finite], [1, 99])}")
    return pred, residual, survive


print("\n[render brightest epoch]")
v1_bright_pred, v1_bright_res, v1_bright_keep = render_cloud(epoch_bright, v1, "v1")
v2_bright_pred, v2_bright_res, v2_bright_keep = render_cloud(epoch_bright, v2, "v2")
print("\n[render dimmest epoch]")
v1_dim_pred, v1_dim_res, v1_dim_keep = render_cloud(epoch_dim, v1, "v1")
v2_dim_pred, v2_dim_res, v2_dim_keep = render_cloud(epoch_dim, v2, "v2")


# ---------- overlap analysis ----------
def overlap(keep_a, keep_b, label_a, label_b):
    inter = (keep_a & keep_b).sum()
    only_a = (keep_a & ~keep_b).sum()
    only_b = (~keep_a & keep_b).sum()
    iou = inter / max(1, (keep_a | keep_b).sum())
    print(f"  {label_a} & {label_b}: |∩|={inter}, only-{label_a}={only_a}, "
          f"only-{label_b}={only_b}, IoU={iou:.4f}")


print("\n[overlap brightest]")
overlap(v1_bright_keep, v2_bright_keep, "v1", "v2")
print("[overlap dimmest]")
overlap(v1_dim_keep, v2_dim_keep, "v1", "v2")


# ---------- plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=False)
panels = [
    ("Brightest (t=%d, mag=%.2f)" % (epoch_bright, mag_bright), v1_bright_res,
     v1_bright_keep, v2_bright_res, v2_bright_keep),
    ("Dimmest   (t=%d, mag=%.2f)" % (epoch_dim, mag_dim), v1_dim_res,
     v1_dim_keep, v2_dim_res, v2_dim_keep),
]

x_lim_max = 2.5  # mag, for histogram x-axis
bins = np.linspace(-x_lim_max, x_lim_max, 200)

for row, (title, v1_res, v1_keep, v2_res, v2_keep) in enumerate(panels):
    for col, (res, keep, label, color) in enumerate([
        (v1_res, v1_keep, "v1", "tab:red"),
        (v2_res, v2_keep, "v2", "tab:blue"),
    ]):
        ax = axes[row, col]
        finite = np.isfinite(res)
        ax.hist(res[finite], bins=bins, color=color, alpha=0.7,
                label=f"{label}: |C_t|={int(keep.sum())}")
        ax.axvline(-TOL_MAG, color="k", linestyle=":", lw=0.8)
        ax.axvline(+TOL_MAG, color="k", linestyle=":", lw=0.8)
        ax.axvline(0, color="grey", linestyle="-", lw=0.4)
        ax.set_yscale("log")
        ax.set_title(f"{title} — {label}", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        if row == 1:
            ax.set_xlabel("pred − measured (mag)")
        if col == 0:
            ax.set_ylabel("count (log)")

fig.suptitle(
    f"Seed {SEED} (|ω|=1.23 dps) — v1 vs v2 cloud at LC extrema (N={N_POOL}, TOL=±{TOL_MAG} mag)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = SURVEY_ROOT / "results" / "s060_v1_v2_compare" / "seed14_extrema_v1_vs_v2_cloud.png"
fig.savefig(out, dpi=200)
print(f"\nSaved: {out}")
