"""s114_plot_errors — joint (q0-error, omega-error) view of the s114 multi-solutions.

The attractors.png showed only omega-dir error (q error buried as raw geo-off, which
unfairly penalises valid body-twins). This recomputes errors TWIN-FOLDED: each polished
solution (q_a, omega_pol) is scored against whichever of {truth, twin(truth)} branch its
ATTITUDE is nearest, where twin = (q_180x (x) q0, R_180x . omega) [lib/twin.py: flips
omega_y, omega_z]. So a valid twin shows q0_err~0 AND wdir_err~0 against its branch.

Plots BOTH errors as primary axes (q0_err vs wdir_err), color = surr-ro band, |omega|
error annotated, hi-fi-confirmed ringed, truth at origin.

Output: results/s114/seed{SEED}/joint_errors.png
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

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2
from lib.twin import Q_180X, R_180X, quat_mul

SEED = int(os.environ.get("S114_SEED", 119))
OUT = SURVEY / "results" / "s114" / f"seed{SEED:03d}"
INERTIA = m048_inertia()
R2D = 57.29577951308232


def geo_deg(q1, q2):
    return 2.0 * np.degrees(np.arccos(np.clip(abs(float(q1 @ q2)), -1.0, 1.0)))


def dir_deg(a, b):
    a = a / (np.linalg.norm(a) + 1e-30); b = b / (np.linalg.norm(b) + 1e-30)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def main():
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)

    ms = np.load(OUT / "multistart.npz", allow_pickle=True)
    pol = np.load(OUT / "polish.npz", allow_pickle=True)
    ep_a = int(ms["ep_a"])
    repA = (np.load(OUT / "cloud.npz", allow_pickle=True)["repA"] if (OUT / "cloud.npz").exists()
            else np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                         allow_pickle=True)["repA"])
    topk = pol["topk"].astype(int)
    pol_w = pol["pol_w"]; rho = pol["pol_rmse"] / 0.05; band = pol["pol_band"]

    # truth + body-twin branches at anchor A
    qT, wT = q_hist[ep_a], w_hist[ep_a]
    qTw, wTw = quat_mul(Q_180X, qT), R_180X @ wT

    hifi_reps = set()
    if (OUT / "hifi.json").exists():
        for r in json.loads((OUT / "hifi.json").read_text())["rows"]:
            if r.get("hifi_band") in ("A", "B") and r["label"] != "truth":
                hifi_reps.add(int(r["rep"]))

    q0_err = np.empty(len(topk)); wdir_err = np.empty(len(topk)); wmag_err = np.empty(len(topk))
    branch = []
    for k, rep in enumerate(topk):
        q_a = repA[int(rep)]; w = pol_w[k]
        eT, eTw = geo_deg(q_a, qT), geo_deg(q_a, qTw)
        if eT <= eTw:
            q0_err[k] = eT; wdir_err[k] = dir_deg(w, wT); branch.append("truth")
            wref = wT
        else:
            q0_err[k] = eTw; wdir_err[k] = dir_deg(w, wTw); branch.append("twin")
            wref = wTw
        wmag_err[k] = abs(np.linalg.norm(w) - np.linalg.norm(wref)) * R2D

    # ---- joint scatter ----
    bandcol = {"A": "tab:green", "B": "tab:orange", "C": "tab:blue", "D": "0.6"}
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for b in ["D", "C", "B", "A"]:
        m = np.array([x == b for x in band])
        if m.any():
            ax.scatter(q0_err[m], wdir_err[m], s=44, c=bandcol[b], edgecolor="k", linewidth=0.3,
                       label=f"surr Band {b} (n={int(m.sum())})", zorder=3)
    for k, rep in enumerate(topk):
        if int(rep) in hifi_reps:
            ax.scatter([q0_err[k]], [wdir_err[k]], s=170, facecolors="none", edgecolors="red",
                       linewidths=1.7, zorder=4)
    ax.scatter([0], [0], marker="*", s=320, c="gold", edgecolor="k", zorder=5, label="truth / twin (0,0)")
    ax.axvline(2, ls=":", lw=0.8, color="0.5"); ax.axhline(2, ls=":", lw=0.8, color="0.5")
    ax.set_xlabel("q0 attitude error — twin-folded (deg)")
    ax.set_ylabel("omega-direction error vs matched branch (deg)")
    nt = sum(b == "truth" for b in branch); ntw = sum(b == "twin" for b in branch)
    ax.set_title(f"s114 seed {SEED}: JOINT (q0, omega-dir) error of {len(topk)} multi-solutions\n"
                 f"twin-folded — {nt} near-truth-branch, {ntw} near-twin-branch | red ring = hi-fi A/B")
    ax.legend(loc="upper right", fontsize=8)
    # annotate the hi-fi winners with |omega| error
    for k, rep in enumerate(topk):
        if int(rep) in hifi_reps and rho[k] < 4:
            ax.annotate(f"rep{int(rep)}\nd|w|={wmag_err[k]:.3f}dps", (q0_err[k], wdir_err[k]),
                        fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    fig.tight_layout()
    p = OUT / "joint_errors.png"; fig.savefig(p, dpi=135); plt.close(fig)
    print(f"Saved: results/s114/seed{SEED:03d}/joint_errors.png")

    # ---- corrected table for the hi-fi-confirmed solutions ----
    print(f"\n{'rep':>5} {'branch':>6} {'q0_err':>7} {'wdir_err':>9} {'d|w|dps':>8} {'surr-ro':>8} {'band':>4} {'hifi':>5}")
    ords = np.argsort(rho)
    for k in ords:
        rep = int(topk[k])
        if rep in hifi_reps or rho[k] < 4:
            print(f"{rep:>5} {branch[k]:>6} {q0_err[k]:7.2f} {wdir_err[k]:9.2f} {wmag_err[k]:8.3f} "
                  f"{rho[k]:8.3f} {band[k]:>4} {'Y' if rep in hifi_reps else '':>5}")


if __name__ == "__main__":
    main()
