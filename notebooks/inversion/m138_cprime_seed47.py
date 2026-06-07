"""m138 Path C′ — re-rank B2's 50k candidates by per-|ω|-band best-K.

No new compute. Loads B2 isoshell_ckpt.npz, stratifies by |ω|-band, picks
top-K per band, then audits the combined pool against truth.

Idea: the cost is biased toward high-|ω| candidates because they sweep more
L(t) sets. Stratifying by |ω|-band prevents high-|ω| candidates from
crowding out truth-near low-|ω| ones in the top-K pool.
"""
import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from m138_isoshell_h1 import quat_mul, geodesic_deg
from lib.traj_source import load_truth

SEED = 47
B2_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "rescore_B2"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "rescore_Cprime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load truth + B2 candidates
truth = load_truth(SEED, "m048")
true_q0 = truth["q0_wxyz"]
true_omega = truth["omega0_rad"]
truth_dir = true_omega / np.linalg.norm(true_omega)
truth_mag = float(np.linalg.norm(true_omega))

z = np.load(B2_DIR / "isoshell_ckpt.npz", allow_pickle=True)
omegas = z["omega_batch"]
cost = z["cost"]
q0_estimate = z["q0_estimate"]
mags = np.linalg.norm(omegas, axis=1)
N = len(cost)
print(f"loaded B2: {N} candidates")
print(f"global ranking: rank-1 cost={cost.min():.0f}, dir_err of rank-1 candidate")

# Strategies to compare
def evaluate_pool(idx, label):
    """Audit a candidate pool and report pool_min, top-3 dir_errs, truth-near count."""
    o = omegas[idx]
    m = np.linalg.norm(o, axis=1)
    dirs = o / m[:, None]
    derr = np.degrees(np.arccos(np.clip(dirs @ truth_dir, -1, 1)))
    merr_pct = (m - truth_mag) / truth_mag * 100
    c = cost[idx]
    # rank within pool (best cost first)
    pool_order = np.argsort(c)
    print(f"\n=== {label} ({len(idx)} candidates) ===")
    print(f"  pool_min dir_err: {derr.min():.2f}°")
    print(f"  candidates with dir_err <5°: {(derr<5).sum()}; <10°: {(derr<10).sum()}; <15°: {(derr<15).sum()}")
    # Top-5 by cost in pool
    print(f"  Top-5 by cost in pool:")
    for r, i in enumerate(pool_order[:5]):
        print(f"    rank{r+1}: cost={c[i]:.0f}, dir={derr[i]:6.2f}°, |ω|-err={merr_pct[i]:+6.2f}%")
    # Best dir_err in pool's top-30
    if len(idx) >= 30:
        top30_pool_idx = pool_order[:30]
        print(f"  pool_min in pool's top-30: {derr[top30_pool_idx].min():.2f}°")
    return {
        "n": len(idx),
        "pool_min_dir_err": float(derr.min()),
        "n_within_5deg": int((derr<5).sum()),
        "n_within_10deg": int((derr<10).sum()),
        "n_within_15deg": int((derr<15).sum()),
        "top30_pool_dir_errs": derr[pool_order[:min(30, len(idx))]].tolist(),
        "top30_pool_costs": c[pool_order[:min(30, len(idx))]].tolist(),
        "top30_pool_mag_errs_pct": merr_pct[pool_order[:min(30, len(idx))]].tolist(),
    }

# Strategy A: global top-30 (baseline, B2's original)
order = np.argsort(cost)
A = evaluate_pool(order[:30], "A: global top-30 (baseline B2)")

# Strategy B: |ω|-band stratification — k_per_band best, M bands
def stratified(k_per_band, n_bands):
    """Sort all cands by |ω|, slice into n_bands equal-size bands, take k_per_band best per band by cost."""
    sort_by_mag = np.argsort(mags)
    band_size = N // n_bands
    out = []
    for b in range(n_bands):
        band_idx = sort_by_mag[b*band_size:(b+1)*band_size]
        # within band, sort by cost
        c_band = cost[band_idx]
        best = band_idx[np.argsort(c_band)[:k_per_band]]
        out.extend(best.tolist())
    return np.array(out)

B = evaluate_pool(stratified(3, 10), "B: 3-per-band × 10 bands = 30 cands")
C = evaluate_pool(stratified(2, 15), "C: 2-per-band × 15 bands = 30 cands")
D = evaluate_pool(stratified(1, 30), "D: 1-per-band × 30 bands = 30 cands")
E = evaluate_pool(stratified(5, 6), "E: 5-per-band × 6 bands = 30 cands")
F = evaluate_pool(stratified(10, 10), "F: 10-per-band × 10 bands = 100 cands")

# Strategy G: hybrid — global top-15 + 1-per-band × 15 bands
gl15 = order[:15]
band15 = stratified(1, 15)
G_idx = np.unique(np.concatenate([gl15, band15]))
G = evaluate_pool(G_idx, "G: global top-15 ∪ 1-per-15-bands")

# Strategy H: cost normalized by |ω|-band median
sort_by_mag = np.argsort(mags)
n_bands_norm = 20
band_size = N // n_bands_norm
norm_cost = cost.copy().astype(np.float64)
band_assign = np.zeros(N, dtype=int)
for b in range(n_bands_norm):
    band_idx = sort_by_mag[b*band_size:(b+1)*band_size]
    band_assign[band_idx] = b
    band_med = np.median(cost[band_idx])
    norm_cost[band_idx] = cost[band_idx] - band_med  # subtract band median; lower (more negative) = better
H = evaluate_pool(np.argsort(norm_cost)[:30], "H: top-30 by band-median-normalized cost")

# Save summary
summary = {
    "seed": SEED, "experiment": "m138_path_Cprime_rerank",
    "input": "rescore_B2/isoshell_ckpt.npz (50k cands, eps=5°, 50 mags)",
    "strategies": {
        "A_global_top30": A,
        "B_stratified_3x10": B,
        "C_stratified_2x15": C,
        "D_stratified_1x30": D,
        "E_stratified_5x6": E,
        "F_stratified_10x10_top100": F,
        "G_hybrid_global15_band15": G,
        "H_band_median_normalized": H,
    },
    "truth_omega_mag": truth_mag,
    "truth_omega_mag_deg_per_s": float(np.degrees(truth_mag)),
}
with open(OUT_DIR / "result.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
print("strategy            | n   | pool_min | <5° | <10° | <15°")
for k, v in summary["strategies"].items():
    print(f"  {k:30s} {v['n']:4d}  {v['pool_min_dir_err']:7.2f}°  {v['n_within_5deg']:3d}   {v['n_within_10deg']:4d}   {v['n_within_15deg']:4d}")
print(f"\nsaved: {OUT_DIR/'result.json'}")
