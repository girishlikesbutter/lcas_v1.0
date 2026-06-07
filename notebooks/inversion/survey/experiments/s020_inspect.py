"""s020 — checkpoint inspector.

Loads a seed checkpoint and prints a structured summary:
  • bracket vs truth
  • per-category counts
  • timing breakdown + would-have-saved
  • survivor q0 distribution near truth / twin
  • top-K survivors by combined score (geo + align + surrogate-MSE proxy)
  • per-class survivor count (truth-class < 30°, twin-class < 30°, multi-solution)

Usage:
  python notebooks/inversion/survey/experiments/s020_inspect.py 6
  python notebooks/inversion/survey/experiments/s020_inspect.py 6 --smoke
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    sub = "smoke" if args.smoke else f"seed{args.seed:03d}"
    cp_dir = SURVEY_DIR / "results" / "s020" / sub
    if not cp_dir.exists():
        raise FileNotFoundError(cp_dir)

    print(f"=== s020 checkpoint inspection — {cp_dir} ===\n")

    with open(cp_dir / "summary.json") as f:
        s = json.load(f)
    print(f"seed = {s['seed']}")
    print(f"config: N_dir={s['config']['N_OMEGA_DIR']}, "
          f"N_mag={s['config']['N_OMEGA_MAG_CELLS']}, "
          f"N_phi={s['config']['N_PHI_STEPS']}")
    print(f"scale: {s['scale']['N_total_candidates']:,} candidates "
          f"({s['scale']['N_cells']} cells × {s['scale']['M_q_target']} q_targets)\n")

    # Bracket
    print("--- Bracket ---")
    print(f"  truth ω-mag: {s['bracket']['truth_omega_mag_dps']:.4f} dps")
    print(f"  bracket cells (dps): "
          f"{[f'{c:.4f}' for c in s['bracket']['selected_cells_dps']]}")
    print(f"  nearest cell to truth: {s['bracket']['nearest_cell_pct']:.2f}%\n")

    # Categorisation
    print("--- Filter categorisation ---")
    cat = s["categorisation"]
    total = sum(cat.values())
    for k, v in cat.items():
        print(f"  {k:<22s}: {v:>10,d} ({100*v/total:>6.3f}%)")
    print()

    # Thresholds
    print("--- Thresholds (truth-calibrated) ---")
    th = s["thresholds"]
    print(f"  truth_geo_score   = {th['truth_geo_score']:.4f}")
    print(f"  truth_align_score = {th['truth_align_score']:.4f}")
    print(f"  geo_threshold     = {th['geo_threshold']:.4f}")
    print(f"  align_threshold   = {th['align_threshold']:.4f}\n")

    # Timing
    print("--- Timing ---")
    t = s["timing"]
    print(f"  bracket           : {t['bracket_wall_s']:>8.2f} s")
    print(f"  loop wall         : {t['loop_wall_s']:>8.2f} s "
          f"({t['loop_wall_s']/60:.1f} min)")
    print(f"  geo total         : {t['geo_total_s']:>8.2f} s")
    print(f"  align total       : {t['align_total_s']:>8.2f} s")
    print(f"  align on geo-PASS : {t['align_on_geo_pass_s']:>8.2f} s "
          f"({t['n_align_geo_pass']:,} cands)")
    print(f"  align on geo-FAIL : {t['align_on_geo_fail_s']:>8.2f} s "
          f"({t['n_align_geo_fail']:,} cands)")
    if t['align_total_s'] > 0:
        save_pct = 100 * t['align_on_geo_fail_s'] / t['align_total_s']
        print(f"  WOULD HAVE SAVED  : {t['would_have_saved_s']:>8.2f} s "
              f"({save_pct:.1f}% of align)")
    print(f"  TOTAL WALL        : {t['total_wall_s']:>8.2f} s "
          f"({t['total_wall_s']/60:.1f} min)\n")

    # Survivor diagnostics
    print("--- Survivor q0 distribution ---")
    sd = s["survivor_diagnostics"]
    n_surv = sd["n_survivors"]
    print(f"  n_survivors: {n_surv}")
    if n_surv > 0:
        print(f"  q0 geodesic to truth (deg): "
              f"min={sd['min_geodesic_to_truth_deg']:.2f}, "
              f"median={sd['median_geodesic_to_truth_deg']:.2f}")
        print(f"  q0 geodesic to twin  (deg): "
              f"min={sd['min_geodesic_to_twin_deg']:.2f}, "
              f"median={sd['median_geodesic_to_twin_deg']:.2f}")

    # Score distribution (post-hoc threshold analysis)
    print("--- Score distribution ---")
    meta_npz = np.load(cp_dir / "candidates_meta.npz")
    geo = meta_npz["geo_score"]
    al = meta_npz["align_score"]
    geo_finite = geo[np.isfinite(geo)]
    al_finite = al[np.isfinite(al)]
    print(f"  geo_score (n_finite={geo_finite.size}):")
    if geo_finite.size > 0:
        u, c = np.unique(np.round(geo_finite, 3), return_counts=True)
        for v, n in zip(u, c):
            bar = "█" * min(40, int(40 * n / geo_finite.size))
            print(f"    {v:.3f}: {n:>9,d}  {bar}")
    print(f"  align_score (n_finite={al_finite.size}):")
    if al_finite.size > 0:
        u, c = np.unique(np.round(al_finite, 3), return_counts=True)
        for v, n in zip(u, c):
            bar = "█" * min(40, int(40 * n / al_finite.size))
            print(f"    {v:.3f}: {n:>9,d}  {bar}")

    # Hypothetical survivor counts at relaxed thresholds
    print(f"\n--- Hypothetical survivors at relaxed thresholds ---")
    print(f"  {'geo_thresh':<11} {'align_thresh':<13} {'n_survivors':<12} "
          f"(% of total)")
    for gt in [1.0, 0.99, 0.5, 0.0]:
        for at in [1.0, 0.99, 0.85, 0.7, 0.5]:
            survs = (np.isfinite(geo) & (geo >= gt) &
                     np.isfinite(al) & (al >= at)).sum()
            pct = 100 * survs / max(geo.size, 1)
            print(f"  {gt:<11.3f} {at:<13.3f} {survs:>10,d}  ({pct:.4f}%)")

    # IC-pool truth / twin coverage (independent of thresholds)
    print("\n--- IC pool truth / twin coverage (every candidate, pre-threshold) ---")
    sd_full = np.load(cp_dir / "survivor_diagnostics.npz")
    truth_q0 = sd_full["truth_q0"].astype(float)
    truth_omega = sd_full["truth_omega"].astype(float)

    def _qmul(q1, q2):
        w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                         w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

    def _qgeo_batch(q_arr, q_ref):
        dots = np.abs(q_arr @ q_ref)
        return np.degrees(2.0 * np.arccos(np.clip(dots, -1, 1)))

    q_180x = np.array([0.0, 1.0, 0.0, 0.0])
    twin_q0 = _qmul(q_180x, truth_q0)
    R_180x = np.diag([1.0, -1.0, -1.0])
    truth_mag = float(np.linalg.norm(truth_omega))
    twin_omega = R_180x @ truth_omega
    twin_dir = twin_omega / max(truth_mag, 1e-12)
    truth_dir = truth_omega / max(truth_mag, 1e-12)

    cm = np.load(cp_dir / "candidates_meta.npz")
    og = np.load(cp_dir / "omega_grid.npz")
    q0_all = cm["q0"].astype(float)
    om_cell = cm["omega_cell_idx"]
    ovec = og["omega_vectors"]
    om = ovec[om_cell]
    om_mag = np.linalg.norm(om, axis=1)
    om_dir = om / np.maximum(om_mag[:, None], 1e-12)
    om_mag_pct = np.abs(om_mag - truth_mag) / max(truth_mag, 1e-12) * 100
    om_dir_truth = np.degrees(np.arccos(np.clip(om_dir @ truth_dir, -1, 1)))
    om_dir_twin = np.degrees(np.arccos(np.clip(om_dir @ twin_dir, -1, 1)))
    qd_truth = _qgeo_batch(q0_all, truth_q0)
    qd_twin = _qgeo_batch(q0_all, twin_q0)

    print(f"  closest q0 to truth (any ω): {qd_truth.min():.3f}°")
    print(f"  closest q0 to twin  (any ω): {qd_twin.min():.3f}°")
    print(f"  candidates with q<30° of truth (any ω): {(qd_truth < 30).sum()}")
    print(f"  candidates with q<30° of twin  (any ω): {(qd_twin < 30).sum()}")
    truth_basin = (qd_truth < 30) & (om_mag_pct < 10) & (om_dir_truth < 10)
    twin_basin = (qd_twin < 30) & (om_mag_pct < 10) & (om_dir_twin < 10)
    print(f"  TRUE truth-basin candidates  (q<30, |ω-mag|<10%, ω-dir<10°): {truth_basin.sum()}")
    print(f"  TRUE twin-basin candidates                                  : {twin_basin.sum()}")
    if truth_basin.sum() > 0:
        idxs = np.where(truth_basin)[0]
        # closest by q-truth
        order = idxs[np.argsort(qd_truth[idxs])[:5]]
        print(f"\n  Top 5 truth-basin candidates by q0-to-truth:")
        print(f"    {'idx':<8} {'q_truth':<8} {'ω-mag%':<8} {'ω-dir°':<8} "
              f"{'geo':<6} {'align':<6} {'survives?':<10}")
        for i in order:
            geo = float(cm["geo_score"][i])
            al = float(cm["align_score"][i])
            survives = bool(cm["cat_both"][i])
            print(f"    {int(i):<8d} {qd_truth[i]:<8.2f} {om_mag_pct[i]:<8.2f} "
                  f"{om_dir_truth[i]:<8.2f} {geo:<6.3f} {al:<6.3f} "
                  f"{'YES' if survives else 'no':<10}")

    # Detailed survivors
    if n_surv > 0:
        sd_npz = np.load(cp_dir / "survivor_diagnostics.npz")
        meta = np.load(cp_dir / "candidates_meta.npz")
        og = np.load(cp_dir / "omega_grid.npz")

        sq0 = sd_npz["survivor_q0"]
        sgeo = sd_npz["survivor_geo_score"]
        salign = sd_npz["survivor_align_score"]
        sg_truth = sd_npz["survivor_geodesic_to_truth_deg"]
        sg_twin = sd_npz["survivor_geodesic_to_twin_deg"]
        scell = sd_npz["survivor_omega_cell"]
        ovec = og["omega_vectors"]
        truth_omega = sd_npz["truth_omega"]

        # Per-class buckets
        truth_class = sg_truth < 30.0
        twin_class = (sg_twin < 30.0) & ~truth_class
        other_class = ~(truth_class | twin_class)
        print(f"\n--- Survivor classes (q0 angular dist) ---")
        print(f"  truth-class (< 30°): {truth_class.sum()}")
        print(f"  twin-class  (< 30°): {twin_class.sum()}")
        print(f"  other (multi-soln) : {other_class.sum()}")

        # Top-K survivors by composite score (geo * align — both 0..1)
        composite = sgeo * salign
        order = np.argsort(composite)[::-1][:args.top_k]
        print(f"\n--- Top-{args.top_k} survivors by (geo × align) ---")
        print(f"  {'rank':<5} {'geo':<7} {'align':<7} {'q0_geo_truth':<13} "
              f"{'q0_geo_twin':<13} {'ω_cell':<7} {'class':<7}")
        for rank, i in enumerate(order, 1):
            cls = ("truth" if truth_class[i] else
                   "twin" if twin_class[i] else "other")
            print(f"  {rank:<5d} {sgeo[i]:<7.3f} {salign[i]:<7.3f} "
                  f"{sg_truth[i]:<13.2f} {sg_twin[i]:<13.2f} "
                  f"{int(scell[i]):<7d} {cls:<7s}")

        # Truth-class summary
        if truth_class.sum() > 0:
            tcls_idx = np.where(truth_class)[0]
            best = tcls_idx[np.argmin(sg_truth[tcls_idx])]
            print(f"\n--- Best truth-class survivor ---")
            print(f"  q0_geo_truth = {sg_truth[best]:.3f}°")
            print(f"  ω-cell idx   = {int(scell[best])}")
            cell_omega = ovec[int(scell[best])]
            cell_mag = np.linalg.norm(cell_omega)
            truth_mag = np.linalg.norm(truth_omega)
            print(f"  ω-mag at cell vs truth: {cell_mag:.6f} vs {truth_mag:.6f} rad/s "
                  f"({(cell_mag-truth_mag)/truth_mag*100:+.2f}%)")
            cell_dir = cell_omega / max(cell_mag, 1e-12)
            truth_dir = truth_omega / max(truth_mag, 1e-12)
            ang = float(np.degrees(np.arccos(np.clip(np.dot(cell_dir, truth_dir),
                                                      -1, 1))))
            print(f"  ω-dir cell vs truth: {ang:.3f}°")


if __name__ == "__main__":
    main()
