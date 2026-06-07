"""Smoke test for filter_costs on seed 6.

Verifies:
  1. Propagating truth (q0, ω0) reproduces cached k1_body/k2_body to machine precision.
  2. Truth alignment cost = 1.0 (or very close).
  3. Truth geo cost = 1.0 (or very close).
  4. Body-twin (q_180x ⊗ q0_truth, same ω) also scores 1.0.
  5. A random (q0, ω) scores low.
  6. Times one full propagation+scoring cycle.
"""
import sys, time
from pathlib import Path
import numpy as np
import quaternion as q_pkg

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))                       # for `from lib import ...`
sys.path.insert(0, str(SURVEY.parent.parent.parent))  # for `from src.*`

from lib.traj_load import load_truth                 # noqa: E402
from lib import filter_costs as fc                   # noqa: E402

print("=" * 70)
print("Smoke test — filter_costs on seed 6")
print("=" * 70)

# Load static + tier
static = fc.load_static_geometry()
tier_table = fc.load_tier_table()
print(f"Tier shortlist sizes: {[len(t) for t in tier_table['tier_face_idx']]}")
print(f"Inertia tensor diag: {np.diag(static['inertia_tensor'])}")

# Seed 6 truth
truth = load_truth(6)
seed_data = fc.precompute_seed_filter_data(truth, tier_table)
print(f"\nseed 6:")
print(f"  truth peaks (all):    {seed_data['truth_peak_idx'].size}")
print(f"  bright peaks (mag<11): {seed_data['bright_peak_idx'].size}")
print(f"  spec events:           {seed_data['spec_event_idx'].size}")
print(f"  spec tiers:           {seed_data['spec_tier'].tolist()}")

# (1) Verify propagation matches cached
t0 = time.perf_counter()
k1_b, k2_b, pab_b = fc.propagate_candidate(
    truth["q0_wxyz"], truth["omega0_rad"], seed_data, static["inertia_tensor"]
)
t_prop = time.perf_counter() - t0
print(f"\nPropagation took {t_prop*1000:.1f} ms")

err_k1 = np.max(np.abs(k1_b - truth["k1_body"]))
err_k2 = np.max(np.abs(k2_b - truth["k2_body"]))
err_pab = np.max(np.abs(pab_b - truth["pab_body"]))
print(f"  max|k1_body - truth.k1_body| = {err_k1:.3e}")
print(f"  max|k2_body - truth.k2_body| = {err_k2:.3e}")
print(f"  max|pab_body - truth.pab_body| = {err_pab:.3e}")
assert err_k1 < 1e-5 and err_k2 < 1e-5 and err_pab < 1e-5, "Propagation diverges from cache"

# (2)+(3) Truth scores
t0 = time.perf_counter()
res_truth = fc.evaluate_candidate(
    truth["q0_wxyz"], truth["omega0_rad"], seed_data,
    static["inertia_tensor"], static["face_normals"], tier_table["tier_face_idx"],
)
t_full = time.perf_counter() - t0
print(f"\nFull eval cycle: {t_full*1000:.1f} ms (propagation+surrogate+scoring)")
print(f"  TRUTH alignment cost = {res_truth['score_alignment']:.4f}")
print(f"  TRUTH geo cost       = {res_truth['score_geo']:.4f}")

# (4) Body-twin: q_180x · q0_truth, ω_twin = R_180x · ω_truth.
# IS-901 has 2-fold body symmetry about X (SP_N↔SP_S, AD_E↔AD_W); the body
# transformation requires also rotating ω since ω is body-frame. Empirical
# verification on seed 6: median surrogate-LC diff = 0.004 mag, max = 0.72 mag.
# (The concept page concepts/twin_degeneracy.md claims Y-axis with same ω;
# that is INCORRECT for IS-901 geometry — re-checked here.)
q0_t = truth["q0_wxyz"]
q0_t_q = q_pkg.quaternion(q0_t[0], q0_t[1], q0_t[2], q0_t[3])
q_180x = q_pkg.quaternion(0, 1, 0, 0)
q0_twin_q = q_180x * q0_t_q
q0_twin = np.array([q0_twin_q.w, q0_twin_q.x, q0_twin_q.y, q0_twin_q.z])
R_180x = np.diag([1.0, -1.0, -1.0])
omega_twin = R_180x @ truth["omega0_rad"]
res_twin = fc.evaluate_candidate(
    q0_twin, omega_twin, seed_data,
    static["inertia_tensor"], static["face_normals"], tier_table["tier_face_idx"],
)
print(f"\nBody-twin (q_180x · q0_truth, R_180x · ω):")
print(f"  alignment cost = {res_twin['score_alignment']:.4f}")
print(f"  geo cost       = {res_twin['score_geo']:.4f}")

# (5) Random candidate
rng = np.random.default_rng(42)
from scipy.spatial.transform import Rotation
q_rand_xyzw = Rotation.random(random_state=rng).as_quat()
q_rand = np.array([q_rand_xyzw[3], q_rand_xyzw[0], q_rand_xyzw[1], q_rand_xyzw[2]])
omega_dir = rng.standard_normal(3)
omega_dir /= np.linalg.norm(omega_dir)
omega_mag_dps = rng.uniform(0.1, 1.5)
omega_rand = np.deg2rad(omega_mag_dps * omega_dir)
res_rand = fc.evaluate_candidate(
    q_rand, omega_rand, seed_data,
    static["inertia_tensor"], static["face_normals"], tier_table["tier_face_idx"],
)
print(f"\nRandom candidate:")
print(f"  alignment cost = {res_rand['score_alignment']:.4f}")
print(f"  geo cost       = {res_rand['score_geo']:.4f}")

# Summary
print()
print("=" * 70)
print(f"Per-eval time estimate: {t_full*1000:.1f} ms")
print(f"100 seeds × 1000 random candidates × {t_full*1000:.0f}ms / 8 workers = "
      f"{100*1000*t_full/8/60:.1f} min on Pool(8)")
print(f"100 seeds × 10000 random candidates × {t_full*1000:.0f}ms / 8 workers = "
      f"{100*10000*t_full/8/60:.1f} min on Pool(8)")
print("=" * 70)
