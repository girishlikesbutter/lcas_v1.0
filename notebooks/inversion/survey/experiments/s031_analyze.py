"""s031 — post-hoc analysis on results/s031/seed006/.

Run AFTER s031_hifi_rerank_seed6.py finishes. Produces:
- Per-level "best 3 by ρ" with full (q0, ω) details + errors to truth/twin.
- Diagnostic table comparing surrogate-MSE rank vs hi-fi ρ rank for the
  union of hi-fi targets.
- Stage A predicted-Band coverage table.

Reads:
  results/s020/seed006/{candidates_meta.npz, survivor_diagnostics.npz, omega_grid.npz}
  results/s030/seed006/survivors_per_level.npz
  results/s031/seed006/{surrogate_mse_union.npz, hifi_rho_union.npz}
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parent.parent
S020 = SURVEY / "results" / "s020" / "seed006"
S030 = SURVEY / "results" / "s030" / "seed006"
S031 = SURVEY / "results" / "s031" / "seed006"


def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1n = q1 / (np.linalg.norm(q1) + 1e-30)
    q2n = q2 / (np.linalg.norm(q2) + 1e-30)
    d = float(np.abs(np.dot(q1n, q2n)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def rho_band(rho: float) -> str:
    if not np.isfinite(rho):
        return "INF"
    if rho < 2.0:
        return "A"
    if rho < 4.0:
        return "B"
    if rho < 8.0:
        return "C"
    return "D"


def main():
    cands = np.load(S020 / "candidates_meta.npz")
    omega_grid = np.load(S020 / "omega_grid.npz")
    diag = np.load(S020 / "survivor_diagnostics.npz")
    relax = np.load(S030 / "survivors_per_level.npz")
    surr = np.load(S031 / "surrogate_mse_union.npz")
    hifi = np.load(S031 / "hifi_rho_union.npz")

    q0_all = cands["q0"]
    cell_idx_all = cands["omega_cell_idx"]
    omega_vectors = omega_grid["omega_vectors"]
    truth_q0 = diag["truth_q0"]; twin_q0 = diag["twin_q0"]
    truth_omega = diag["truth_omega"]
    twin_omega = np.array([truth_omega[0], -truth_omega[1], -truth_omega[2]])
    truth_omega_mag = float(np.linalg.norm(truth_omega))
    truth_omega_dir = truth_omega / truth_omega_mag
    twin_omega_dir = twin_omega / np.linalg.norm(twin_omega)

    union_idx = surr["union_idx"]
    surrogate_mse = surr["surrogate_mse"]
    union_to_local = {int(i): k for k, i in enumerate(union_idx)}

    target_idx = hifi["hifi_target_idx"]
    target_rho = hifi["rho"]
    target_to_local = {int(i): k for k, i in enumerate(target_idx)}

    levels = [
        ("L0_strict_1p0_1p0", relax["L0_idx"]),
        ("L1_relaxed_0p5_6of7", relax["L1_idx"]),
        ("L2_relaxed_0p5_0p5", relax["L2_idx"]),
        ("L3_align_only_6of7", relax["L3_idx"]),
    ]
    truth_basin_idx = relax["truth_basin_idx"]
    twin_basin_idx = relax["twin_basin_idx"]

    print("=" * 78)
    print("s031 analysis — best 3 per level + truth/twin basin coverage")
    print("=" * 78)

    for name, ids in levels:
        if ids.size == 0:
            print(f"\n[{name}]  (empty)\n")
            continue

        # Filter to only those in the hifi target set (others were not hi-fi'd).
        in_hifi = np.array([
            i for i in ids if int(i) in target_to_local
        ], dtype=np.int64)
        if in_hifi.size == 0:
            print(f"\n[{name}]  no candidates hi-fi'd (cap left below this level's top)")
            continue

        rhos = np.array([target_rho[target_to_local[int(i)]] for i in in_hifi])
        finite = np.isfinite(rhos)
        order = np.argsort(np.where(finite, rhos, np.inf))

        nA = int(((rhos < 2.0) & finite).sum())
        nB = int(((rhos >= 2.0) & (rhos < 4.0) & finite).sum())
        nC = int(((rhos >= 4.0) & (rhos < 8.0) & finite).sum())
        nD = int(((rhos >= 8.0) & finite).sum())
        n_inf = int((~finite).sum())
        print(f"\n[{name}]  total_hifi={in_hifi.size}  "
              f"A={nA}  B={nB}  C={nC}  D={nD}  inf={n_inf}")

        for rank in range(min(3, in_hifi.size)):
            cand = int(in_hifi[order[rank]])
            rho_v = rhos[order[rank]]
            q0_c = q0_all[cand].astype(np.float64)
            omg_c = omega_vectors[cell_idx_all[cand]]
            mag_c = float(np.linalg.norm(omg_c))
            dir_c = omg_c / mag_c
            q_t = quat_geodesic_deg(q0_c, truth_q0)
            q_w = quat_geodesic_deg(q0_c, twin_q0)
            mag_rel = abs(mag_c - truth_omega_mag) / truth_omega_mag * 100
            d_t = float(np.degrees(np.arccos(np.clip(dir_c @ truth_omega_dir, -1, 1))))
            d_w = float(np.degrees(np.arccos(np.clip(dir_c @ twin_omega_dir, -1, 1))))
            local = union_to_local.get(cand)
            mse_v = surrogate_mse[local] if local is not None else float("nan")
            print(
                f"  rank{rank}  cand={cand:>7d}  ρ={rho_v:7.3f} ({rho_band(rho_v)})  "
                f"surr_MSE={mse_v:9.5f}  q→truth={q_t:6.2f}°  q→twin={q_w:6.2f}°  "
                f"|ω|={mag_c:.5f} ({mag_rel:+5.2f}%)  "
                f"ω→truth_dir={d_t:6.2f}°  ω→twin_dir={d_w:6.2f}°"
            )

    print("\n" + "=" * 78)
    print("Truth-basin and twin-basin candidates — hi-fi ρ if rendered")
    print("=" * 78)

    for tag, basin in [("TRUTH", truth_basin_idx), ("TWIN", twin_basin_idx)]:
        for cand in basin:
            cand = int(cand)
            in_hifi = cand in target_to_local
            in_union = cand in union_to_local
            mse = surrogate_mse[union_to_local[cand]] if in_union else float("nan")
            rho = target_rho[target_to_local[cand]] if in_hifi else float("nan")
            q0_c = q0_all[cand].astype(np.float64)
            omg_c = omega_vectors[cell_idx_all[cand]]
            mag_c = float(np.linalg.norm(omg_c))
            mag_rel = abs(mag_c - truth_omega_mag) / truth_omega_mag * 100
            q_to = (
                quat_geodesic_deg(q0_c, truth_q0) if tag == "TRUTH"
                else quat_geodesic_deg(q0_c, twin_q0)
            )
            print(
                f"  {tag} cand={cand:>7d}  q→{tag.lower()}={q_to:6.2f}°  "
                f"|dω-mag|={mag_rel:.2f}%  surr_MSE={mse:.5f}  "
                f"hi-fi_ρ={'%.3f'%rho if np.isfinite(rho) else 'NOT_RENDERED'}"
            )


if __name__ == "__main__":
    main()
