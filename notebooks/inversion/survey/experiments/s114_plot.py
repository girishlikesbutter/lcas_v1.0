"""s114_plot — required artefacts for contract_anchor-direct-omega-multistart.

(1) multi-solution attractor plot: polished surr-ro vs omega-dir-err, colored by
    attitude geo-off (separates near-truth from body-twin ~180deg solutions), Band
    A/B lines, truth marked, hi-fi-confirmed points ringed.
(2) blind winner's predicted-vs-truth LC overlay (surrogate full-LC) + residual.

Output: results/s114/seed{SEED}/attractors.png, winner_lc.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.jacobi_propagator import propagate_jacobi_path2

SEED = int(os.environ.get("S114_SEED", 119))
OUT = SURVEY / "results" / "s114" / f"seed{SEED:03d}"
INERTIA = m048_inertia()


def main():
    ms = np.load(OUT / "multistart.npz", allow_pickle=True)
    pol = np.load(OUT / "polish.npz", allow_pickle=True)
    geo = ms["rep_geo_off"]; ep_a = int(ms["ep_a"])
    topk = pol["topk"].astype(int)
    rho = pol["pol_rmse"] / 0.05
    dir_off = pol["pol_dir"]
    pol_w = pol["pol_w"]
    geo_k = geo[topk]
    hifi = {}
    if (OUT / "hifi.json").exists():
        for r in json.loads((OUT / "hifi.json").read_text())["rows"]:
            if r["label"] != "truth":
                hifi[int(r["rep"])] = r

    # ---- (1) attractor scatter ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(dir_off, rho, c=geo_k, cmap="viridis", s=36, vmin=0, vmax=180,
                    edgecolor="k", linewidth=0.3, zorder=3)
    for thr, lab, col in [(2.0, "Band A (ro<2)", "tab:green"), (4.0, "Band B (ro<4)", "tab:orange")]:
        ax.axhline(thr, ls="--", lw=1, color=col, label=lab)
    # ring hi-fi-confirmed A/B
    for k, rep in enumerate(topk):
        if int(rep) in hifi and hifi[int(rep)]["hifi_band"] in ("A", "B"):
            ax.scatter([dir_off[k]], [rho[k]], s=160, facecolors="none",
                       edgecolors="red", linewidths=1.6, zorder=4)
    ax.set_xlabel("polished omega-direction error vs truth (deg)")
    ax.set_ylabel("polished surrogate ro  (= sqrt(MSE)/0.05)")
    ax.set_yscale("log")
    ax.set_title(f"s114 seed {SEED} anchor-direct multi-solutions (n={len(topk)} polished)\n"
                 f"red ring = hi-fi confirmed Band A/B")
    cb = fig.colorbar(sc); cb.set_label("attitude geo-off from truth (deg)  [~180=body-twin]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p1 = OUT / "attractors.png"; fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"Saved: results/s114/seed{SEED:03d}/attractors.png")

    # ---- (2) winner LC overlay (surrogate full-LC) ----
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    win_k = int(np.argmin(rho)); rep = int(topk[win_k]); w = pol_w[win_k]
    q_a = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                  allow_pickle=True)["repA"][rep] if not (OUT / "cloud.npz").exists() \
        else np.load(OUT / "cloud.npz", allow_pickle=True)["repA"][rep]
    surr = get_model()
    tf = times0[ep_a:] - times0[ep_a]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if ep_a:
        tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
        qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
        quats = np.vstack([qb[::-1][:-1], qf])
    else:
        quats = qf
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    pred = surr.predict_magnitude(np.einsum("nij,nj->ni", R, sun_u),
                                  np.einsum("nij,nj->ni", R, obs_u), 0.0, 15.0, od)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    a1.plot(times0, mag, "k.", ms=3, label="truth hi-fi LC")
    a1.plot(times0, pred, "r-", lw=1, label=f"winner rep {rep} (surr ro={rho[win_k]:.3f}, "
            f"geo {geo[rep]:.1f} deg, dir {dir_off[win_k]:.2f} deg)")
    a1.invert_yaxis(); a1.set_ylabel("magnitude"); a1.legend(loc="best")
    a1.set_title(f"s114 seed {SEED} blind winner vs truth")
    a2.plot(times0, pred - mag, "b-", lw=0.8); a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("resid"); a2.set_xlabel("time (s)")
    fig.tight_layout()
    p2 = OUT / "winner_lc.png"; fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"Saved: results/s114/seed{SEED:03d}/winner_lc.png")


if __name__ == "__main__":
    main()
