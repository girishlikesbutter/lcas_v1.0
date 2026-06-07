"""s018b — Face-identity tier classifier from peak brightness, post-fix m048.

Builds the formal tiered face-identity classifier the buggy era was using
(recovered under correct truth from cached NPZ fields):

  for any LC peak with magnitude M, return a candidate face-group set
  {g : g is plausibly responsible for this peak's specular alignment}.

Spec-event criterion: `min_ang_dist < 5deg` AND `best_group = g`.
Per-face spec mag distributions are well-separated by face area; the
classifier is essentially a partition by area class.

DISTANCE NORMALISATION (load-bearing for portability):
  We classify on `mag_abs = mag_hifi - 5 * log10(obs_dist / D_REF_KM)`,
  not raw apparent magnitude. The cached `mag_hifi` is apparent magnitude
  with the inverse-square law baked in; for the tier classifier to be
  satellite-model-intrinsic (transferable across observer geometries) we
  reference everything to a fixed slant range D_REF_KM = 38649.2 km
  (cohort median for IS-901 at GEO observed from the m048 ground station).
  Within the m048 cohort the correction is empirically tiny (max
  |Delta| = 0.004 mag), but the absolute formulation lets the table be
  consumed downstream by inversion code on any future geometry simply by
  applying the inverse correction to incoming observed peaks:
      mag_abs_observed = mag_apparent_observed - 5 * log10(d_obs / D_REF_KM)

Outputs:
  results/s018b/face_tiers.npz        per-face mag distributions + tier table (abs)
  results/s018b/summary.json          tier definitions + cohort coverage
  results/s018b/per_face_mag_cdf.png  CDF of mag_abs at spec5 events, per face
  results/s018b/inverse_classifier_heatmap.png  P(face|mag_abs band, spec5)
  results/s018b/per_seed_tier_coverage.png      per-seed peak counts per tier
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.traj_load import list_seeds, load_truth  # noqa: E402

OUT_DIR = ROOT / "results" / "s018b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "+WD", "-WD", "+ED", "-ED"]
GROUP_AREAS_M2 = [97.3, 97.3, 16.93, 16.93, 22.75, 22.75, 9.8, 9.8, 9.8, 9.8]
SPEC_ANG_DEG = 5.0  # ground-truth criterion for "PAB-aligned spec event"

# Reference slant range for absolute-magnitude normalisation. Cohort median
# of all 50000 (100 seeds x 500 epochs) m048 obs_dist values. Stored
# alongside the tier table so downstream consumers can de-normalise to any
# observer geometry: mag_abs = mag_apparent - 5*log10(obs_dist / D_REF_KM).
D_REF_KM = 38649.2

# Tier definitions: ordered by face area class (brightest first).
# Magnitudes are ABSOLUTE (mag_abs), referenced to D_REF_KM.
# Each tier = (label, mag_abs_lo, mag_abs_hi, candidate_group_indices)
TIERS = [
    ("T1_X",   0.00, 6.00, [0, 1]),                # +-X (97.3 m^2)
    ("T2_YZ",  6.00, 7.00, [2, 3, 4, 5]),          # +-Y u +-Z
    ("T3_any", 7.00, 8.00, [2, 3, 4, 5, 6, 7, 8, 9]),  # any non-+-X
    ("T4_D",   8.00, 9.00, [6, 7, 8, 9]),          # dishes only
]
T_NULL = ("T0_none", 0.0, 99.0, [])  # mag_abs >= 9 is non-spec / unclassified


def assign_tier(mag_value: float) -> str:
    for label, lo, hi, _ in TIERS:
        if lo <= mag_value < hi:
            return label
    return T_NULL[0]


def main():
    t0 = time.perf_counter()
    seeds = list_seeds()
    print(f"s018b — face-identity tier classifier over {len(seeds)} post-fix m048 seeds")

    # Pool spec events
    pooled_mag_app = []  # apparent magnitude as cached in NPZ
    pooled_obs_d = []
    pooled_min_ang = []
    pooled_best_grp = []
    pooled_seed = []
    pooled_at_peak = []  # is this epoch in hifi_peak_epochs?

    # Per-seed: peak count per tier (using pre-detected hifi_peak_epochs)
    per_seed_tier_counts = {label: [] for label, *_ in TIERS}
    per_seed_tier_counts[T_NULL[0]] = []
    per_seed_n_peaks = []

    for seed in seeds:
        d = load_truth(seed)
        mag_app = d["mag_hifi"]
        obs_d = d["obs_dist"]
        mad = d["min_ang_dist"]
        bg = d["best_group"]
        peaks = d["hifi_peak_epochs"]

        # Convert apparent -> absolute mag at D_REF_KM for the tier classifier.
        # Within m048 the correction is empirically <0.004 mag, but stored
        # absolute so downstream consumers can transfer to other geometries.
        mag_abs = mag_app - 5.0 * np.log10(obs_d / D_REF_KM)

        n = mag_app.shape[0]
        at_peak = np.zeros(n, dtype=bool)
        at_peak[peaks] = True

        pooled_mag_app.append(mag_app)
        pooled_obs_d.append(obs_d)
        pooled_min_ang.append(mad)
        pooled_best_grp.append(bg)
        pooled_seed.append(np.full(n, seed, dtype=np.int32))
        pooled_at_peak.append(at_peak)

        # per-seed peak tier counts (banded on mag_abs at the peak epochs)
        peak_mag_abs = mag_abs[peaks]
        per_seed_n_peaks.append(int(len(peaks)))
        for label, lo, hi, _ in TIERS:
            per_seed_tier_counts[label].append(
                int(np.sum((peak_mag_abs >= lo) & (peak_mag_abs < hi)))
            )
        per_seed_tier_counts[T_NULL[0]].append(
            int(np.sum(peak_mag_abs >= 9.0))
        )

    pooled_mag_app = np.concatenate(pooled_mag_app)
    pooled_obs_d = np.concatenate(pooled_obs_d)
    pooled_min_ang = np.concatenate(pooled_min_ang)
    pooled_best_grp = np.concatenate(pooled_best_grp)
    pooled_seed = np.concatenate(pooled_seed)
    pooled_at_peak = np.concatenate(pooled_at_peak)
    pooled_mag = pooled_mag_app - 5.0 * np.log10(pooled_obs_d / D_REF_KM)  # absolute
    spec_mask = pooled_min_ang < SPEC_ANG_DEG
    n_spec = int(spec_mask.sum())

    # === Per-face mag distribution at spec events ===
    per_face_stats = {}
    per_face_mag_arrays = {}
    for g in range(10):
        m = spec_mask & (pooled_best_grp == g)
        if m.sum() == 0:
            continue
        mags = pooled_mag[m]
        per_face_mag_arrays[g] = mags
        per_face_stats[GROUP_NAMES[g]] = {
            "area_m2": GROUP_AREAS_M2[g],
            "n_spec_events": int(m.sum()),
            "mag_p5": float(np.percentile(mags, 5)),
            "mag_p10": float(np.percentile(mags, 10)),
            "mag_p25": float(np.percentile(mags, 25)),
            "mag_p50": float(np.percentile(mags, 50)),
            "mag_p75": float(np.percentile(mags, 75)),
            "mag_p90": float(np.percentile(mags, 90)),
            "mag_p95": float(np.percentile(mags, 95)),
        }

    # === Inverse classifier P(group=g | mag-band, spec5) ===
    mag_band_edges = [0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    inv_cls = np.zeros((len(mag_band_edges) - 1, 10))
    n_per_band = np.zeros(len(mag_band_edges) - 1, dtype=np.int64)
    for i in range(len(mag_band_edges) - 1):
        lo, hi = mag_band_edges[i], mag_band_edges[i + 1]
        in_band = spec_mask & (pooled_mag >= lo) & (pooled_mag < hi)
        n = int(in_band.sum())
        n_per_band[i] = n
        if n == 0:
            continue
        for g in range(10):
            inv_cls[i, g] = float((in_band & (pooled_best_grp == g)).sum()) / n

    # === Tier validation: P(group in tier_set | mag in tier_band, spec5) ===
    tier_validation = {}
    for label, lo, hi, candidate_grps in TIERS:
        in_band = spec_mask & (pooled_mag >= lo) & (pooled_mag < hi)
        n = int(in_band.sum())
        if n == 0:
            tier_validation[label] = {
                "n_spec_events_in_band": 0,
                "p_in_candidate_set": None,
                "candidate_group_names": [GROUP_NAMES[g] for g in candidate_grps],
                "shortlist_size": len(candidate_grps),
            }
            continue
        # number of spec events in this band where best_group is in the
        # candidate set
        in_set = in_band & np.isin(pooled_best_grp, candidate_grps)
        p = float(in_set.sum()) / n
        tier_validation[label] = {
            "n_spec_events_in_band": n,
            "p_in_candidate_set": p,
            "candidate_group_names": [GROUP_NAMES[g] for g in candidate_grps],
            "shortlist_size": len(candidate_grps),
        }

    # === Per-seed tier coverage at hi-fi peak epochs ===
    # peak count per tier (already gathered above)
    per_seed_summary = {
        "n_peaks_total": {
            "median": float(np.median(per_seed_n_peaks)),
            "p10": float(np.percentile(per_seed_n_peaks, 10)),
            "p90": float(np.percentile(per_seed_n_peaks, 90)),
            "min": int(min(per_seed_n_peaks)),
            "max": int(max(per_seed_n_peaks)),
        },
    }
    for label, *_ in TIERS:
        cnts = np.array(per_seed_tier_counts[label])
        per_seed_summary[label] = {
            "median": float(np.median(cnts)),
            "p10": float(np.percentile(cnts, 10)),
            "p90": float(np.percentile(cnts, 90)),
            "n_seeds_with_at_least_1": int(np.sum(cnts >= 1)),
            "n_seeds_with_at_least_2": int(np.sum(cnts >= 2)),
            "n_seeds_with_zero": int(np.sum(cnts == 0)),
        }
    null_cnts = np.array(per_seed_tier_counts[T_NULL[0]])
    per_seed_summary[T_NULL[0]] = {
        "median": float(np.median(null_cnts)),
        "p10": float(np.percentile(null_cnts, 10)),
        "p90": float(np.percentile(null_cnts, 90)),
        "n_seeds_with_zero": int(np.sum(null_cnts == 0)),
    }

    # Also: "informative coverage" — n seeds with >=1 T1 peak, and >=2 peaks
    # across {T1, T2, T3, T4} (so they have multi-peak constraints to intersect)
    t1_cnt = np.array(per_seed_tier_counts["T1_X"])
    t2_cnt = np.array(per_seed_tier_counts["T2_YZ"])
    t3_cnt = np.array(per_seed_tier_counts["T3_any"])
    t4_cnt = np.array(per_seed_tier_counts["T4_D"])
    multi_tier_cnt = (t1_cnt >= 1).astype(int) + (t2_cnt >= 1).astype(int) + \
                     (t3_cnt >= 1).astype(int) + (t4_cnt >= 1).astype(int)
    total_classifiable = t1_cnt + t2_cnt + t3_cnt + t4_cnt
    coverage = {
        "n_seeds_with_T1_peak": int(np.sum(t1_cnt >= 1)),
        "n_seeds_with_T1_only": int(np.sum((t1_cnt >= 1) & (multi_tier_cnt == 1))),
        "n_seeds_with_>=2_distinct_tiers_hit": int(np.sum(multi_tier_cnt >= 2)),
        "n_seeds_with_>=3_distinct_tiers_hit": int(np.sum(multi_tier_cnt >= 3)),
        "n_seeds_with_>=4_classifiable_peaks": int(np.sum(total_classifiable >= 4)),
        "n_seeds_with_zero_classifiable_peaks": int(np.sum(total_classifiable == 0)),
    }

    # === Plots ===
    # 1. Per-face mag CDF at spec5 events
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    for g in range(10):
        if g not in per_face_mag_arrays:
            continue
        mags = np.sort(per_face_mag_arrays[g])
        cdf = np.arange(1, len(mags) + 1) / len(mags)
        ax.plot(mags, cdf, "-", color=colors[g],
                label=f"{GROUP_NAMES[g]} ({GROUP_AREAS_M2[g]:.1f} m², n={len(mags)})")
    for label, lo, hi, _ in TIERS:
        ax.axvspan(lo, hi, alpha=0.05, color="gray")
    for x in [6.0, 7.0, 8.0, 9.0]:
        ax.axvline(x, ls=":", color="k", alpha=0.3)
    ax.set_xlabel("magnitude at spec event")
    ax.set_ylabel("cumulative P(mag <= M | best_group = g, spec5)")
    ax.set_title(
        f"s018b — per-face spec-event mag CDF, post-fix cohort\n"
        f"({n_spec:,} spec events across 100 seeds)"
    )
    ax.set_xlim(4.0, 10.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_face_mag_cdf.png", dpi=120)
    plt.close(fig)

    # 2. Inverse classifier heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(inv_cls, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels(GROUP_NAMES)
    ax.set_yticks(range(len(mag_band_edges) - 1))
    ax.set_yticklabels(
        [f"[{mag_band_edges[i]:g},{mag_band_edges[i+1]:g}) n={n_per_band[i]}"
         for i in range(len(mag_band_edges) - 1)]
    )
    ax.set_xlabel("face group")
    ax.set_ylabel("magnitude band (n=spec events)")
    ax.set_title(f"P(best_group = g | mag in band, min_ang_dist<{SPEC_ANG_DEG:g}°)")
    for i in range(len(mag_band_edges) - 1):
        for j in range(10):
            v = inv_cls[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "inverse_classifier_heatmap.png", dpi=120)
    plt.close(fig)

    # 3. Per-seed tier coverage stacked bar
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    seeds_arr = np.array(seeds)
    sort_order = np.argsort(-(t1_cnt + t2_cnt + t3_cnt + t4_cnt))
    bottom = np.zeros(len(seeds_arr))
    tier_colors = {"T1_X": "tab:red", "T2_YZ": "tab:orange",
                   "T3_any": "tab:olive", "T4_D": "tab:blue"}
    for label in ["T1_X", "T2_YZ", "T3_any", "T4_D"]:
        cnts = np.array(per_seed_tier_counts[label])[sort_order]
        ax.bar(range(len(seeds_arr)), cnts, bottom=bottom,
               color=tier_colors[label], label=label)
        bottom = bottom + cnts
    ax.set_xticks(range(0, len(seeds_arr), 5))
    ax.set_xticklabels(seeds_arr[sort_order][::5], rotation=90, fontsize=7)
    ax.set_xlabel("seed (sorted by total classifiable peak count)")
    ax.set_ylabel("# pre-detected hi-fi peaks in tier")
    ax.set_title(
        f"s018b — per-seed face-identity tier coverage on hi-fi peaks "
        f"(post-fix m048, 100 seeds)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_seed_tier_coverage.png", dpi=120)
    plt.close(fig)

    # === Save artefacts ===
    np.savez(
        OUT_DIR / "face_tiers.npz",
        # distance normalisation (load-bearing)
        d_ref_km=np.array(D_REF_KM),
        # tier definitions (parsable downstream); mag bounds are ABSOLUTE
        tier_labels=np.array([t[0] for t in TIERS]),
        tier_mag_abs_lo=np.array([t[1] for t in TIERS]),
        tier_mag_abs_hi=np.array([t[2] for t in TIERS]),
        tier_candidate_indices=np.array(
            [np.array(t[3] + [-1] * (8 - len(t[3]))) for t in TIERS]
        ),  # padded with -1
        tier_shortlist_sizes=np.array([len(t[3]) for t in TIERS]),
        group_names=np.array(GROUP_NAMES),
        group_areas_m2=np.array(GROUP_AREAS_M2),
        # per-face stats
        per_face_p5=np.array([per_face_stats.get(GROUP_NAMES[g], {}).get("mag_p5", np.nan) for g in range(10)]),
        per_face_p10=np.array([per_face_stats.get(GROUP_NAMES[g], {}).get("mag_p10", np.nan) for g in range(10)]),
        per_face_p50=np.array([per_face_stats.get(GROUP_NAMES[g], {}).get("mag_p50", np.nan) for g in range(10)]),
        per_face_p90=np.array([per_face_stats.get(GROUP_NAMES[g], {}).get("mag_p90", np.nan) for g in range(10)]),
        per_face_p95=np.array([per_face_stats.get(GROUP_NAMES[g], {}).get("mag_p95", np.nan) for g in range(10)]),
        per_face_n_spec=np.array([per_face_stats.get(GROUP_NAMES[g], {}).get("n_spec_events", 0) for g in range(10)]),
        # inverse classifier
        mag_band_edges=np.array(mag_band_edges),
        inv_classifier=inv_cls,
        n_spec_per_band=n_per_band,
        # per-seed tier counts
        seeds=seeds_arr,
        per_seed_n_peaks=np.array(per_seed_n_peaks),
        per_seed_t1_count=t1_cnt,
        per_seed_t2_count=t2_cnt,
        per_seed_t3_count=t3_cnt,
        per_seed_t4_count=t4_cnt,
        per_seed_null_count=null_cnts,
        per_seed_distinct_tiers_hit=multi_tier_cnt,
        per_seed_total_classifiable=total_classifiable,
    )

    summary = {
        "n_seeds": len(seeds),
        "n_spec_events_total": n_spec,
        "spec_angle_deg": float(SPEC_ANG_DEG),
        "d_ref_km": float(D_REF_KM),
        "distance_correction_max_abs_mag": float(
            np.max(np.abs(pooled_mag - pooled_mag_app))
        ),
        "obs_dist_cohort_min_km": float(pooled_obs_d.min()),
        "obs_dist_cohort_max_km": float(pooled_obs_d.max()),
        "magnitude_normalisation_formula": (
            "mag_abs = mag_apparent - 5*log10(obs_dist / D_REF_KM); tier "
            "thresholds below are in mag_abs. Downstream consumers on a "
            "different observer geometry must apply the same correction "
            "to incoming peaks before tier lookup."
        ),
        "tier_definitions": [
            {
                "label": label,
                "mag_lo": lo,
                "mag_hi": hi,
                "candidate_groups": [GROUP_NAMES[g] for g in cand],
                "shortlist_size": len(cand),
            }
            for label, lo, hi, cand in TIERS
        ],
        "tier_validation": tier_validation,
        "per_face_stats": per_face_stats,
        "per_seed_summary": per_seed_summary,
        "cohort_coverage": coverage,
        "wall_seconds": float(time.perf_counter() - t0),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # === Console report ===
    print("")
    print(f"=== s018b face-identity tier classifier (wall {summary['wall_seconds']:.1f} s) ===")
    print(f"spec events (min_ang_dist<{SPEC_ANG_DEG:g}°): {n_spec:,}")
    print(f"D_REF_KM = {D_REF_KM:.1f}  (distance correction max |Δ| = "
          f"{summary['distance_correction_max_abs_mag']:.4f} mag within m048)")
    print(f"cohort obs_dist range: [{summary['obs_dist_cohort_min_km']:.0f}, "
          f"{summary['obs_dist_cohort_max_km']:.0f}] km")
    print("")
    print("Per-face mag_abs distribution at spec events:")
    print(f"{'face':>4}  area    n     p10    p50    p90")
    for g in range(10):
        if GROUP_NAMES[g] not in per_face_stats:
            continue
        s = per_face_stats[GROUP_NAMES[g]]
        print(f"  {GROUP_NAMES[g]:>3s}  {s['area_m2']:>5.1f}  {s['n_spec_events']:>3d}  "
              f"{s['mag_p10']:>5.2f}  {s['mag_p50']:>5.2f}  {s['mag_p90']:>5.2f}")
    print("")
    print("Tier validation P(best_group in candidate_set | mag in band, spec5):")
    for label, lo, hi, cand in TIERS:
        v = tier_validation[label]
        cand_str = " ".join(GROUP_NAMES[g] for g in cand)
        p = v["p_in_candidate_set"]
        n = v["n_spec_events_in_band"]
        print(f"  {label:<8s} mag in [{lo:g},{hi:g})  n={n:>4d}  "
              f"shortlist={v['shortlist_size']}  P={p if p is not None else 'NA':.3f}  "
              f"candidates={cand_str}")
    print("")
    print("Cohort coverage (peaks classified by mag tier on pre-detected hifi peaks):")
    print(f"  seeds with >=1 T1 (mag<6, ±X) peak:           {coverage['n_seeds_with_T1_peak']:>3d}/100")
    print(f"  seeds with >=2 distinct tiers hit:           {coverage['n_seeds_with_>=2_distinct_tiers_hit']:>3d}/100")
    print(f"  seeds with >=3 distinct tiers hit:           {coverage['n_seeds_with_>=3_distinct_tiers_hit']:>3d}/100")
    print(f"  seeds with >=4 classifiable peaks:           {coverage['n_seeds_with_>=4_classifiable_peaks']:>3d}/100")
    print(f"  seeds with zero classifiable peaks (mag<9):   {coverage['n_seeds_with_zero_classifiable_peaks']:>3d}/100")
    print("")
    print("Saved:")
    print(f"  {OUT_DIR / 'face_tiers.npz'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {OUT_DIR / 'per_face_mag_cdf.png'}")
    print(f"  {OUT_DIR / 'inverse_classifier_heatmap.png'}")
    print(f"  {OUT_DIR / 'per_seed_tier_coverage.png'}")


if __name__ == "__main__":
    main()
