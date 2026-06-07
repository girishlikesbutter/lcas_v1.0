"""s073f — cluster_457 local-geometry test: is L-direction the soft direction?

Background
----------
s073/s073d/s077 chain. On post-fix seed 89, cluster_457 is the lone Band A
multi-solution attractor (q0_err=59.58 deg vs truth, hi-fi rho=0.904). s073d
found it shares truth's *body-frame polhode* only approximately — 2T and
|L|^2 each differ ~1.1% — while its inertial L_J2000 direction is 128.6 deg
off truth. s073d left an ambiguity it could not resolve from N=1:

  (i)  STRICT     — the 1.1% Casimir mismatch is a *real geometric
                    distinction*; cluster_457 is genuinely off truth's
                    polhode and the cat-4 "same polhode, free L direction"
                    framing breaks for this seed.
  (ii) CHARITABLE — the s059k LM polish minimised LC residual, not polhode
                    distance; 1.1% could just be *polish residual*, i.e. the
                    LC is simply insensitive to a 1% polhode change.

s077 confirmed the *necessary half* cohort-wide ("found solutions preserve
|L| magnitude, spread L direction") but said nothing about the *generative
half* or about local LC sensitivity. This experiment is the local-geometry
test that distinguishes (i) from (ii).

Question (decisive)
-------------------
At cluster_457, perturb the 6-DOF state in two controlled directions and
measure how fast the hi-fi LC residual (vs seed-89 truth) grows:

  Direction (a) — L-DIRECTION (body polhode held fixed exactly).
    A global inertial-frame rotation applied to q0, with omega0 (body) left
    UNCHANGED. Because 2T = omega.I.omega and |L|^2 = |I omega|^2 depend only
    on omega0 and I, the body-frame polhode is preserved to machine
    precision; only the inertial direction of L_J2000 rotates. Three rotation
    axes:
      axis_1 = unit(L_c457 x L_truth)   — rotates L_c457 *toward* L_truth
      axis_2 = unit(L_c457 x axis_1)    — the orthogonal in-plane direction
      axis_3 = unit(L_c457)             — twist about L: L_J2000 does NOT
                                          move (control — the other
                                          fixed-(L,2T) DOF).

  Direction (b) — POLHODE / CASIMIRS (L_J2000 direction held fixed exactly).
    A uniform scaling omega0 -> (1+eps)*omega0 with q0 left UNCHANGED. This
    scales |L| by (1+eps) and 2T by (1+eps)^2 — i.e. it moves to a different
    (larger/smaller) polhode — while I.omega0 keeps its body-frame direction,
    so L_J2000 keeps its inertial direction exactly. cluster_457 differs from
    truth by ~1.1% in *both* Casimirs (a scaling-type offset), so this is the
    matched probe for the s073d offset.

Interpretation
--------------
  - (a) soft [axes 1,2 tolerate large L-direction change at <Band B] AND
    (b) stiff [eps of ~1% already exits Band A/B]
        => the LC IS insensitive to L direction but sensitive to the polhode.
           s073d's 1.1% Casimir mismatch is then a *real* distinction
           (cluster_457 off truth's polhode), but the L-direction softness is
           genuine — STRICT reading on the polhode, soft-L confirmed locally.
  - (a) and (b) BOTH stiff at comparable small perturbations
        => cluster_457 is an isolated point; no soft L-direction sheet;
           cat-4 weakens further.
  - (a) soft AND (b) also soft at the ~1% level
        => the LC is insensitive to a 1% polhode change; s073d's mismatch is
           polish residual — CHARITABLE reading.

Convention / self-checks
------------------------
Scalar-first quaternions (w,x,y,z). `_quat_to_matrix(q)` is the post-fix
conv-(a) passive J2000->body matrix; L_J2000 = R.T @ I @ omega (same formula
as s073/s073d/s077). Direction (a) is built purely in matrix algebra
(R_new = R_old @ R_in.T) then converted back with `_quat_from_matrix`; the
script asserts, for every perturbed state, that the recomputed L_J2000
matches the intended rotation of L_J2000 and that 2T/|L|^2 are preserved.
Direction (b) asserts L_J2000 direction is preserved and |L| scales by
(1+eps). Base-state hi-fi rho is cross-checked against the cached s069
value (0.904) as a pipeline-correctness gate.

Saves
-----
  results/s073f/summary.json   — base invariants, per-state rho/band, verdict
  results/s073f/states.npz     — every perturbed (q0, omega0) + rho + metrics
  results/s073f/s073f_sensitivity.png — rho vs perturbation, both directions

Cross-references
----------------
  experiments/s073_cluster457_l_vector_check.{py,md}
  experiments/s073d_polhode_match_cluster457.{py,md}
  experiments/s077_l_vector_basin_sweep.{py,md}
  lib/hifi_render.py  (render_hifi, rho_from_hifi, rho_band)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# BLAS threads = 1 before any heavy import (Pool discipline).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.jacobi_propagator import _quat_to_matrix, _quat_from_matrix  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402

RESULTS_DIR = SURVEY_ROOT / "results" / "s073f"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 89
S059K_SUMMARY = (
    SURVEY_ROOT / "results" / "s059k_nd800_seed89" / "seed089"
    / "full_lc_seeds" / "summary.json"
)

# Perturbation grids.
THETA_DEG_GRID = np.array(
    [-128.6, -90.0, -45.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0,
     1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0, 128.6]
)
EPS_PCT_GRID = np.array(
    [-10.0, -7.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0,
     0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
)

# Convention self-check tolerances.
TOL_INVARIANT = 1e-9   # relative; 2T/|L|^2 preservation, L-rotation match
TOL_BASE_RHO = 0.02    # absolute; base hi-fi rho vs cached s069 value (0.904)


def l_j2000(q_wxyz: np.ndarray, omega_rad: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Inertial angular momentum. Same formula as s073 / s073d / s077."""
    R = _quat_to_matrix(q_wxyz)          # passive J2000 -> body
    return R.T @ (inertia @ omega_rad)   # body -> J2000


def casimirs(omega_rad: np.ndarray, inertia: np.ndarray) -> tuple[float, float]:
    """(2T, |L|^2) — body-frame Casimir invariants."""
    Iw = inertia @ omega_rad
    return float(omega_rad @ Iw), float(Iw @ Iw)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    c = float(np.clip(u @ v / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def perturb_inertial_rotation(
    q0_base: np.ndarray, axis_unit: np.ndarray, theta_deg: float,
) -> np.ndarray:
    """Direction (a): global inertial rotation by theta about axis_unit.

    omega0 (body) is left unchanged, so 2T and |L|^2 are preserved exactly.
    L_J2000 rotates by R_in. Built in matrix algebra then converted back;
    the result is self-checked by the caller via l_j2000().
    """
    R_old = _quat_to_matrix(q0_base)                       # J2000 -> body
    R_in = Rotation.from_rotvec(
        np.radians(theta_deg) * axis_unit).as_matrix()      # active J2000 rot
    # Want L_new = R_in @ L_old with I.omega0 unchanged:
    #   L_new = R_new.T @ I omega0 = R_in @ R_old.T @ I omega0  =>  R_new = R_old @ R_in.T
    R_new = R_old @ R_in.T
    q_new = _quat_from_matrix(R_new)
    return q_new


def main() -> int:
    t_start = time.time()

    # ------------------------------------------------------------------ load
    truth = load_truth(SEED)
    q0_truth = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    om0_truth = np.asarray(truth["omega0_rad"], dtype=np.float64)

    ctx = build_context(SEED)
    inertia = np.asarray(ctx["inertia_tensor"], dtype=np.float64)
    mag_truth = np.asarray(ctx["mag_hifi_truth"], dtype=np.float64)

    s059k = json.load(open(S059K_SUMMARY))
    entry = next(e for e in s059k["polished"]
                 if e["cluster_id"] == 457 and e["mag_pct_offset"] == 0.0)
    q0_c457 = np.asarray(entry["q0_pol_wxyz"], dtype=np.float64)
    q0_c457 = q0_c457 / np.linalg.norm(q0_c457)
    om0_c457 = np.asarray(entry["om0_pol_rad"], dtype=np.float64)
    rho_c457_cached = float(entry["rho_polished_hifi"])

    # ------------------------------------------------------- base invariants
    L_c457 = l_j2000(q0_c457, om0_c457, inertia)
    L_truth = l_j2000(q0_truth, om0_truth, inertia)
    twoT_c457, L2_c457 = casimirs(om0_c457, inertia)
    L_c457_mag = float(np.linalg.norm(L_c457))
    L_c457_dir = L_c457 / L_c457_mag
    L_dir_angle_c457_vs_truth = angle_between(L_c457, L_truth)

    # ---------------------------------------------- direction (a) rotation axes
    cross = np.cross(L_c457_dir, L_truth / np.linalg.norm(L_truth))
    if np.linalg.norm(cross) < 1e-12:
        raise RuntimeError("L_c457 and L_truth are collinear; axis_1 undefined.")
    axis_1 = cross / np.linalg.norm(cross)               # _|_ L_c457, rotates toward L_truth
    axis_2 = np.cross(L_c457_dir, axis_1)
    axis_2 = axis_2 / np.linalg.norm(axis_2)             # _|_ L_c457 and axis_1
    axis_3 = L_c457_dir.copy()                           # twist about L (control)
    axes = {"axis_1_toward_truth": axis_1,
            "axis_2_orthogonal": axis_2,
            "axis_3_twist_about_L": axis_3}

    # ------------------------------------------------------ build state list
    # Each record: (tag, kind, label_value, q0, om0, L_dir_change_deg,
    #               d_twoT_rel, d_L2_rel)
    records = []

    # base state
    records.append(dict(tag="base", kind="base", label=0.0,
                        q0=q0_c457, om0=om0_c457,
                        L_dir_change_deg=0.0, d_twoT_rel=0.0, d_L2_rel=0.0))

    # direction (a): inertial rotations
    for ax_name, ax in axes.items():
        for theta in THETA_DEG_GRID:
            if theta == 0.0:
                continue  # base already covers theta=0
            q_new = perturb_inertial_rotation(q0_c457, ax, float(theta))
            # --- self-checks ---
            L_new = l_j2000(q_new, om0_c457, inertia)
            R_in = Rotation.from_rotvec(np.radians(theta) * ax).as_matrix()
            L_expect = R_in @ L_c457
            rel_L_err = np.linalg.norm(L_new - L_expect) / L_c457_mag
            twoT_new, L2_new = casimirs(om0_c457, inertia)  # omega unchanged
            d_twoT = abs(twoT_new - twoT_c457) / twoT_c457
            d_L2 = abs(L2_new - L2_c457) / L2_c457
            assert rel_L_err < TOL_INVARIANT, (
                f"(a) {ax_name} theta={theta}: L_J2000 rotation self-check "
                f"failed, rel err {rel_L_err:.2e}")
            assert d_twoT < TOL_INVARIANT and d_L2 < TOL_INVARIANT, (
                f"(a) {ax_name} theta={theta}: Casimirs not preserved "
                f"(d2T={d_twoT:.2e}, dL2={d_L2:.2e})")
            records.append(dict(
                tag=f"a:{ax_name}:{theta:+.1f}", kind=f"a:{ax_name}",
                label=float(theta), q0=q_new, om0=om0_c457,
                L_dir_change_deg=angle_between(L_new, L_c457),
                d_twoT_rel=0.0, d_L2_rel=0.0))

    # direction (b): uniform omega scaling
    for eps_pct in EPS_PCT_GRID:
        if eps_pct == 0.0:
            continue
        scale = 1.0 + eps_pct / 100.0
        om_new = scale * om0_c457
        # --- self-checks ---
        L_new = l_j2000(q0_c457, om_new, inertia)
        L_dir_change = angle_between(L_new, L_c457)
        L_mag_ratio = np.linalg.norm(L_new) / L_c457_mag
        twoT_new, L2_new = casimirs(om_new, inertia)
        d_twoT = abs(twoT_new - twoT_c457) / twoT_c457
        d_L2 = abs(L2_new - L2_c457) / L2_c457
        assert L_dir_change < 1e-6, (
            f"(b) eps={eps_pct}%: L_J2000 direction moved {L_dir_change:.2e} deg")
        assert abs(L_mag_ratio - abs(scale)) < TOL_INVARIANT, (
            f"(b) eps={eps_pct}%: |L| scaled by {L_mag_ratio} not {scale}")
        records.append(dict(
            tag=f"b:scale:{eps_pct:+.1f}", kind="b:omega_scale",
            label=float(eps_pct), q0=q0_c457, om0=om_new,
            L_dir_change_deg=0.0, d_twoT_rel=d_twoT, d_L2_rel=d_L2))

    print(f"=== s073f — cluster_457 local-geometry test (seed {SEED}) ===")
    print(f"  base cluster_457: hi-fi rho cached = {rho_c457_cached:.4f} (band {entry['band']})")
    print(f"  |L_c457| = {L_c457_mag:.6e}   L_c457 vs L_truth = "
          f"{L_dir_angle_c457_vs_truth:.2f} deg")
    print(f"  2T = {twoT_c457:.6e}   |L|^2 = {L2_c457:.6e}")
    print(f"  states to render: {len(records)} "
          f"(1 base + {len(THETA_DEG_GRID)-1}x3 rotations + {len(EPS_PCT_GRID)-1} scales)")
    print(f"  convention self-checks: PASS (all {len(records)-1} perturbed states)")
    print()

    # ------------------------------------------------------------ render Pool
    nproc = min(24, mp.cpu_count())
    args = [(i, rec["q0"], rec["om0"]) for i, rec in enumerate(records)]
    t_render = time.time()
    with mp.Pool(nproc, initializer=_init_worker, initargs=(SEED,)) as pool:
        out = pool.map(_render_one, args, chunksize=4)
    wall_render = time.time() - t_render

    rho_by_idx = {idx: r for idx, r in out}
    rho_arr = np.array([rho_by_idx[i] for i in range(len(records))])
    band_arr = np.array([rho_band(r) if np.isfinite(r) else "X" for r in rho_arr])

    print(f"  rendered {len(records)} states in {wall_render:.1f}s on Pool({nproc})")

    # -------------------------------------------------- base-render pipeline gate
    rho_base = rho_arr[0]
    base_ok = abs(rho_base - rho_c457_cached) < TOL_BASE_RHO
    print(f"  base hi-fi rho (this render) = {rho_base:.4f}  "
          f"(cached {rho_c457_cached:.4f}, |d|={abs(rho_base-rho_c457_cached):.4f})  "
          f"-> pipeline gate {'PASS' if base_ok else 'FAIL'}")
    if not base_ok:
        print("  WARNING: base render disagrees with cached s069 rho beyond "
              f"{TOL_BASE_RHO}; downstream numbers are suspect.", file=sys.stderr)
    print()

    # ----------------------------------------------------------------- arrays
    n = len(records)
    tags = np.array([r["tag"] for r in records])
    kinds = np.array([r["kind"] for r in records])
    labels = np.array([r["label"] for r in records])
    q0_all = np.array([r["q0"] for r in records])
    om0_all = np.array([r["om0"] for r in records])
    L_dir_change = np.array([r["L_dir_change_deg"] for r in records])
    d_twoT = np.array([r["d_twoT_rel"] for r in records])
    d_L2 = np.array([r["d_L2_rel"] for r in records])

    np.savez(
        RESULTS_DIR / "states.npz",
        tag=tags, kind=kinds, label=labels,
        q0=q0_all, om0=om0_all, rho=rho_arr, band=band_arr,
        L_dir_change_deg=L_dir_change, d_twoT_rel=d_twoT, d_L2_rel=d_L2,
        theta_deg_grid=THETA_DEG_GRID, eps_pct_grid=EPS_PCT_GRID,
        q0_c457=q0_c457, om0_c457=om0_c457,
        L_c457=L_c457, L_truth=L_truth, inertia=inertia,
        axis_1=axis_1, axis_2=axis_2, axis_3=axis_3,
    )

    # -------------------------------------------------------- band-exit metrics
    def first_exit(kind_prefix: str, band_thresh: float):
        """Smallest |perturbation| at which rho crosses band_thresh, scanning
        outward symmetrically. Returns (neg_exit, pos_exit) in label units, or
        None if rho stays below band_thresh across the grid."""
        m = np.array([k.startswith(kind_prefix) for k in kinds])
        idx = np.where(m)[0]
        lbl = labels[idx]
        rr = rho_arr[idx]
        pos = sorted([(l, r) for l, r in zip(lbl, rr) if l > 0])
        neg = sorted([(l, r) for l, r in zip(lbl, rr) if l < 0], reverse=True)
        pe = next((l for l, r in pos if not np.isfinite(r) or r >= band_thresh), None)
        ne = next((l for l, r in neg if not np.isfinite(r) or r >= band_thresh), None)
        return ne, pe

    exit_metrics = {}
    for kp in ["a:axis_1_toward_truth", "a:axis_2_orthogonal",
               "a:axis_3_twist_about_L", "b:omega_scale"]:
        ne2, pe2 = first_exit(kp, 2.0)
        ne4, pe4 = first_exit(kp, 4.0)
        exit_metrics[kp] = {
            "first_exit_bandA_neg": ne2, "first_exit_bandA_pos": pe2,
            "first_exit_bandAB_neg": ne4, "first_exit_bandAB_pos": pe4,
        }

    # ------------------------------------------------------------------ figure
    fig, axes_p = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes_p[0]
    colors = {"a:axis_1_toward_truth": "tab:red",
              "a:axis_2_orthogonal": "tab:orange",
              "a:axis_3_twist_about_L": "tab:gray"}
    for kp, c in colors.items():
        m = kinds == kp
        order = np.argsort(labels[m])
        ax.plot(labels[m][order], rho_arr[m][order], "o-", color=c,
                label=kp.replace("a:", ""), ms=4)
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.axhline(4.0, color="k", ls="--", lw=1)
    ax.axhline(rho_base, color="tab:green", ls="-", lw=0.8, alpha=0.6,
               label=f"base rho={rho_base:.2f}")
    ax.text(ax.get_xlim()[1], 2.0, " A|B", va="bottom", ha="right", fontsize=8)
    ax.text(ax.get_xlim()[1], 4.0, " B|C", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("inertial rotation angle theta (deg)")
    ax.set_ylabel("hi-fi rho vs seed-89 truth")
    ax.set_title("Direction (a): L-direction sweep (body polhode fixed)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes_p[1]
    m = kinds == "b:omega_scale"
    order = np.argsort(labels[m])
    ax.plot(labels[m][order], rho_arr[m][order], "o-", color="tab:blue",
            label="omega scale", ms=4)
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.axhline(4.0, color="k", ls="--", lw=1)
    ax.axhline(rho_base, color="tab:green", ls="-", lw=0.8, alpha=0.6,
               label=f"base rho={rho_base:.2f}")
    ax.text(ax.get_xlim()[1], 2.0, " A|B", va="bottom", ha="right", fontsize=8)
    ax.text(ax.get_xlim()[1], 4.0, " B|C", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("omega scale perturbation eps (%)")
    ax.set_ylabel("hi-fi rho vs seed-89 truth")
    ax.set_title("Direction (b): polhode/Casimir sweep (L_J2000 direction fixed)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"s073f — cluster_457 local geometry (seed {SEED}): is L-direction the soft direction?\n"
        f"base rho={rho_base:.3f}  |  L_c457 vs L_truth = {L_dir_angle_c457_vs_truth:.1f} deg",
        fontsize=11)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "s073f_sensitivity.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ----------------------------------------------------------------- verdict
    # Soft = stays Band A|B (rho < 4) over a wide range. Compare the L-direction
    # axes (a:axis_1, a:axis_2 — these actually move L direction) against the
    # polhode direction (b:omega_scale).
    def _max_inband_abs(kind_prefix: str, thresh: float) -> float:
        m = np.array([k.startswith(kind_prefix) for k in kinds])
        lbl = np.abs(labels[m])
        rr = rho_arr[m]
        good = lbl[np.isfinite(rr) & (rr < thresh)]
        return float(good.max()) if good.size else 0.0

    a1_AB = _max_inband_abs("a:axis_1_toward_truth", 4.0)
    a2_AB = _max_inband_abs("a:axis_2_orthogonal", 4.0)
    b_AB = _max_inband_abs("b:omega_scale", 4.0)

    summary = {
        "experiment": "s073f",
        "seed": SEED,
        "cluster_id": 457,
        "base": {
            "q0_c457_wxyz": q0_c457.tolist(),
            "om0_c457_rad": om0_c457.tolist(),
            "rho_hifi_rendered": float(rho_base),
            "rho_hifi_cached_s069": rho_c457_cached,
            "pipeline_gate_pass": bool(base_ok),
            "band": rho_band(rho_base),
            "L_c457_mag": L_c457_mag,
            "twoT_c457": twoT_c457,
            "L2_c457": L2_c457,
            "L_dir_angle_c457_vs_truth_deg": L_dir_angle_c457_vs_truth,
        },
        "grids": {
            "theta_deg": THETA_DEG_GRID.tolist(),
            "eps_pct": EPS_PCT_GRID.tolist(),
        },
        "n_states_rendered": n,
        "render_wall_s": wall_render,
        "render_nproc": nproc,
        "band_exit": exit_metrics,
        "max_inband_abs_perturbation_bandAB": {
            "a_axis_1_toward_truth_deg": a1_AB,
            "a_axis_2_orthogonal_deg": a2_AB,
            "b_omega_scale_pct": b_AB,
        },
        "per_state": [
            {"tag": records[i]["tag"], "kind": records[i]["kind"],
             "label": records[i]["label"], "rho": float(rho_arr[i]),
             "band": str(band_arr[i]),
             "L_dir_change_deg": float(L_dir_change[i]),
             "d_twoT_rel": float(d_twoT[i]), "d_L2_rel": float(d_L2[i])}
            for i in range(n)
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------- console
    print("  band-exit (smallest |perturbation| reaching rho >= 4, Band A|B exit):")
    for kp, em in exit_metrics.items():
        print(f"    {kp:28s}: neg={em['first_exit_bandAB_neg']}  "
              f"pos={em['first_exit_bandAB_pos']}")
    print()
    print(f"  widest in-Band-A|B perturbation:")
    print(f"    (a) axis_1 toward-truth : +/- {a1_AB:.1f} deg L-direction rotation")
    print(f"    (a) axis_2 orthogonal   : +/- {a2_AB:.1f} deg L-direction rotation")
    print(f"    (b) omega scale         : +/- {b_AB:.2f} % polhode scaling")
    print()
    print(f"Saved: {RESULTS_DIR / 'summary.json'}")
    print(f"Saved: {RESULTS_DIR / 'states.npz'}")
    print(f"Saved: {fig_path}")
    print(f"Total wall: {time.time() - t_start:.1f}s")
    return 0


# --------------------------------------------------------------------- worker
_W: dict = {}


def _init_worker(seed: int) -> None:
    _W["ctx"] = build_context(seed)
    _W["target"] = np.asarray(_W["ctx"]["mag_hifi_truth"], dtype=np.float64)


def _render_one(args):
    idx, q0, om0 = args
    try:
        pred = render_hifi(np.asarray(q0, dtype=np.float64),
                           np.asarray(om0, dtype=np.float64), _W["ctx"])
        rho = rho_from_hifi(pred, _W["target"])
    except Exception:
        rho = float("nan")
    return idx, float(rho)


if __name__ == "__main__":
    sys.exit(main())
