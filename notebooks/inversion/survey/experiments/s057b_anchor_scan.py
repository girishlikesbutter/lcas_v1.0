"""s057b — find a "valid" anchor where truth-q is in cloud at BOTH t_a and t_b.

s057 ran on the deepest |C_t| constriction (t_a=411) but truth_q_b was not in
cloud at t_b=421 — a structural failure of the architecture on that anchor.

This scans all 500 epochs to identify which have truth-q-in-cloud (`closest_idx[t]
∈ where(survive_all[t])`), then for each Δt ∈ {1, 2, 5, 10, 20} finds anchor
pairs where BOTH t_a and t_b are valid. Reports the smallest |C_{t_a}| ·
|C_{t_b}| pair available.

If no valid pair exists at any Δt, the architecture is structurally blocked
on seed 89 at the current pool density and surrogate tolerance — that's the
finding. Otherwise, surfaces the best anchor for s057's re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
OUT = SURVEY / "results" / "s057b_anchor_scan"
OUT.mkdir(parents=True, exist_ok=True)

DELTAS = [1, 2, 3, 5, 10, 15, 20, 30]


def main() -> dict:
    z = np.load(DENSE_RUN)
    n_surv = z["n_survivors"]
    survive_all = z["survive_all"]
    closest_idx = z["closest_idx_per_epoch"]
    closest_deg = z["closest_deg_per_epoch"]
    n_epochs = len(n_surv)

    # mask of epochs where the pool quaternion CLOSEST to truth survives
    truth_in_cloud = np.array([
        bool(survive_all[t, closest_idx[t]]) for t in range(n_epochs)
    ])
    n_valid = int(truth_in_cloud.sum())
    print(f"epochs with truth-q in cloud: {n_valid} / {n_epochs} ({n_valid/n_epochs*100:.1f}%)")

    summary = {
        "n_epochs": n_epochs,
        "n_valid_epochs": n_valid,
        "frac_valid": n_valid / n_epochs,
        "deltas_scanned": DELTAS,
        "by_delta": {},
    }

    valid_idx = np.where(truth_in_cloud)[0]
    print(f"valid epochs (first 30): {valid_idx[:30].tolist()}")
    print(f"valid |C_t| range: [{n_surv[valid_idx].min()}, {n_surv[valid_idx].max()}]")

    # for each Δt, find pairs (t_a, t_b=t_a+Δt) both valid
    for Δ in DELTAS:
        valid_pairs = []
        for t_a in valid_idx:
            t_b = t_a + Δ
            if t_b >= n_epochs:
                continue
            if not truth_in_cloud[t_b]:
                continue
            valid_pairs.append((int(t_a), int(t_b),
                                int(n_surv[t_a]), int(n_surv[t_b]),
                                float(closest_deg[t_a]), float(closest_deg[t_b])))
        if not valid_pairs:
            summary["by_delta"][str(Δ)] = {
                "n_pairs": 0, "best": None
            }
            print(f"  Δt={Δ:3d}: 0 valid pairs")
            continue
        # rank by |C_a| × |C_b| ascending (smallest pool of pairs to enumerate)
        valid_pairs.sort(key=lambda p: p[2] * p[3])
        best_pair = valid_pairs[0]
        # also rank by closest_deg sum (lowest discretisation error)
        valid_pairs_by_deg = sorted(valid_pairs, key=lambda p: p[4] + p[5])
        best_by_deg = valid_pairs_by_deg[0]
        summary["by_delta"][str(Δ)] = {
            "n_pairs": len(valid_pairs),
            "best_by_pair_count": {
                "t_a": best_pair[0], "t_b": best_pair[1],
                "n_C_a": best_pair[2], "n_C_b": best_pair[3],
                "closest_deg_a": best_pair[4], "closest_deg_b": best_pair[5],
                "n_pairs_total": best_pair[2] * best_pair[3],
            },
            "best_by_closest_deg": {
                "t_a": best_by_deg[0], "t_b": best_by_deg[1],
                "n_C_a": best_by_deg[2], "n_C_b": best_by_deg[3],
                "closest_deg_a": best_by_deg[4], "closest_deg_b": best_by_deg[5],
                "n_pairs_total": best_by_deg[2] * best_by_deg[3],
            },
        }
        print(f"  Δt={Δ:3d}: {len(valid_pairs):4d} valid pairs | "
              f"best by |C_a||C_b|: t_a={best_pair[0]} t_b={best_pair[1]} "
              f"|C_a|={best_pair[2]} |C_b|={best_pair[3]} "
              f"({best_pair[4]:.2f}°/{best_pair[5]:.2f}°)")

    out_p = OUT / "anchor_scan.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_p}")
    return summary


if __name__ == "__main__":
    main()
