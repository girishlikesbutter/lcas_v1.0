"""m015b_alpha_sweep.py — sweep alpha for min-|ω| bridge objective on leg 0."""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ── Load stage data ────────────────────────────────────────────────────────────
stage1 = np.load(RESULTS_DIR / "m013_stage1.npz", allow_pickle=True)
stage2 = np.load(RESULTS_DIR / "m013_stage2.npz", allow_pickle=True)
PEAKS = stage1["peaks"]          # [183, 260, 360]
CANDS = [stage1["c0"], stage1["c1"], stage1["c2"]]
TRUTH_IDX = stage1["truth_idx"]
BDT = stage2["bdt"]

CTX = setup_experiment(n_observations=500, true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
INERTIA = CTX.inertia_tensor

true_omega_mag_degs = float(np.degrees(np.linalg.norm(CTX.true_omega0)))
print(f"True ω magnitude (ground truth): {true_omega_mag_degs:.4f} deg/s")

# ── Bridge solver ──────────────────────────────────────────────────────────────
OMEGA_BOUND = 0.5  # rad/s — keeps ODE tractable; true |ω|~0.036 rad/s

def bridge_min_omega(q_start, q_end, dt, I, alpha):
    """Find ω minimising arrival_error² + alpha * ||ω||². Zero initial guess."""
    w0 = np.zeros(3)
    times = np.array([0.0, dt])
    bounds = [(-OMEGA_BOUND, OMEGA_BOUND)] * 3
    def obj(w):
        qp, _ = propagate_attitude(q_start, w, times, "tumbling", I)
        d = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        return (1.0 - d * d) + alpha * np.dot(w, w)
    result = minimize(obj, w0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-12})
    w_opt = result.x
    qp, _ = propagate_attitude(q_start, w_opt, times, "tumbling", I)
    err = attitude_error_deg(qp[-1], q_end)
    return w_opt, err

# ── Leg 0 pairs ────────────────────────────────────────────────────────────────
dt = float(BDT[0])
c0 = CANDS[0]   # shape (N, 4)
c1 = CANDS[1]
ti0, ti1 = int(TRUTH_IDX[0]), int(TRUTH_IDX[1])
N = len(c0)

# 5 random wrong pairs (neither index equals truth index)
rng = np.random.default_rng(42)
wrong_pairs = []
while len(wrong_pairs) < 5:
    i = rng.integers(0, N)
    j = rng.integers(0, N)
    if i != ti0 and j != ti1:
        wrong_pairs.append((i, j))

ALPHAS = [0.0, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
ALPHAS_PLOT = [max(a, 1e-5) for a in ALPHAS]  # for log axis

# ── Run sweep ──────────────────────────────────────────────────────────────────
true_results = []    # list of (omega_mag_dps, err_deg) per alpha
random_results = []  # list of list of (omega_mag_dps, err_deg) per alpha, per pair

t_sweep = time.time()
for k_a, alpha in enumerate(ALPHAS):
    w, err = bridge_min_omega(c0[ti0], c1[ti1], dt, INERTIA, alpha)
    true_results.append((float(np.degrees(np.linalg.norm(w))), float(err)))
    row = []
    for (i, j) in wrong_pairs:
        w2, err2 = bridge_min_omega(c0[i], c1[j], dt, INERTIA, alpha)
        row.append((float(np.degrees(np.linalg.norm(w2))), float(err2)))
    random_results.append(row)
    print(f"  alpha={alpha}: true |ω|={true_results[-1][0]:.3f} dps, err={true_results[-1][1]:.3f}° "
          f"[{time.time()-t_sweep:.1f}s]", flush=True)

# ── Console table ──────────────────────────────────────────────────────────────
hdr = f"{'alpha':<10} | {'true |ω| (dps)':<16} | {'true err (deg)':<15} | {'mean rnd |ω| (dps)':<23} | {'mean rnd err (deg)'}"
print("\n" + hdr)
print("-" * len(hdr))
for k, alpha in enumerate(ALPHAS):
    tw, te = true_results[k]
    rw_mean = np.mean([r[0] for r in random_results[k]])
    re_mean = np.mean([r[1] for r in random_results[k]])
    print(f"{alpha:<10} | {tw:<16.4f} | {te:<15.4f} | {rw_mean:<23.4f} | {re_mean:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Alpha sweep: zero-start min-|ω| bridge (leg 0: peak 183→260)")

true_omegas = [r[0] for r in true_results]
true_errs   = [r[1] for r in true_results]

for k_pair in range(5):
    rw = [random_results[k][k_pair][0] for k in range(len(ALPHAS))]
    re = [random_results[k][k_pair][1] for k in range(len(ALPHAS))]
    ax1.plot(ALPHAS_PLOT, rw, color='grey', linewidth=0.8, alpha=0.7)
    ax2.plot(ALPHAS_PLOT, re, color='grey', linewidth=0.8, alpha=0.7)

ax1.plot(ALPHAS_PLOT, true_omegas, color='red', linewidth=2.0, label='true pair')
ax1.axhline(true_omega_mag_degs, color='blue', linestyle='--', label=f'true |ω|={true_omega_mag_degs:.2f}')
ax1.set_xscale('log'); ax1.set_xlabel('alpha'); ax1.set_ylabel('|ω_min| (deg/s)')
ax1.set_title('Min-|ω| vs alpha'); ax1.legend()

ax2.plot(ALPHAS_PLOT, true_errs, color='red', linewidth=2.0, label='true pair')
ax2.axhline(1.0, color='orange', linestyle='--', label='1 deg threshold')
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_xlabel('alpha')
ax2.set_ylabel('arrival error (deg)'); ax2.set_title('Bridge arrival error vs alpha'); ax2.legend()

plt.tight_layout()
out_png = RESULTS_DIR / "m015b_alpha_sweep.png"
plt.savefig(out_png, dpi=150)
print(f"\nPlot saved: {out_png}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
save_results(RESULTS_DIR / "m015b_alpha_sweep.json", {
    'alphas': ALPHAS,
    'true_pair': {
        'i': ti0, 'j': ti1,
        'results': [{'alpha': ALPHAS[k], 'omega_mag_dps': true_results[k][0],
                     'arrival_err_deg': true_results[k][1]} for k in range(len(ALPHAS))]
    },
    'random_pairs': [
        {'i': int(wp[0]), 'j': int(wp[1]),
         'results': [{'alpha': ALPHAS[k], 'omega_mag_dps': random_results[k][p][0],
                      'arrival_err_deg': random_results[k][p][1]} for k in range(len(ALPHAS))]}
        for p, wp in enumerate(wrong_pairs)
    ],
    'true_omega_mag_degs': true_omega_mag_degs,
})
print(f"JSON saved: {RESULTS_DIR / 'm015b_alpha_sweep.json'}")
