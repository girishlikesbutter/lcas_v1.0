#!/usr/bin/env python3
"""
Mixed-Fidelity Pipeline v3 - Data-driven omega bounds + shadow fidelity.

Key innovations:
1. Pre-analysis: Lomb-Scargle + ACF on observed lightcurve → estimate dominant
   frequencies → set tight omega bounds with safety factor
2. Stage 1: Tumbling DE with NO shadows (correct physics, fast ~0.1s/eval)
3. Stage 2: Tumbling L-BFGS-B WITH shadows (full physics, ~3s/eval)

The omega bound tightening is the main speedup: at realistic GEO rates,
propagate_euler takes 0.09s vs 3.4s at the default 30 deg/s bounds.

Hard timeout: 45 min. Saves results throughout.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Project setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp3_v3_bounded_results.json"

WALL_CLOCK_LIMIT = 45 * 60
START_TIME = time.perf_counter()
all_results = {"timings": {}, "phases": {}, "period_analysis": {}}

def elapsed():
    return time.perf_counter() - START_TIME

def check_timeout(phase=""):
    if elapsed() > WALL_CLOCK_LIMIT:
        print(f"\n⏰ TIMEOUT after {elapsed():.0f}s during {phase}", flush=True)
        save_and_exit(timeout=True, phase=phase)

def save_and_exit(timeout=False, phase="", results=None):
    if results is None:
        results = all_results
    out = {"timestamp": datetime.now().isoformat(), "elapsed_s": elapsed(),
           "timeout": timeout, "phase": phase, "results": results}
    RESULTS_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)
    sys.exit(0 if not timeout else 1)

print("=" * 70, flush=True)
print("MIXED-FIDELITY v3: DATA-DRIVEN BOUNDS + SHADOW FIDELITY", flush=True)
print("=" * 70, flush=True)

# ── Imports ────────────────────────────────────────────────────────────
print("\n[1/7] Importing modules...", flush=True)
t0 = time.perf_counter()

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.dynamics import propagate_attitude
from src.inversion import (
    ObjectiveFunction,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    normalize_quaternion,
)
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from scipy.optimize import differential_evolution, minimize
from scipy.signal import lombscargle, find_peaks
from scipy.interpolate import interp1d

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── Load config & model ───────────────────────────────────────────────
print("\n[2/7] Loading Intelsat 901...", flush=True)
t0 = time.perf_counter()

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)
satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

n_observations = 50
OBSERVER_ID = 399999
noise_sigma = 0.05

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── SPICE + geometry ──────────────────────────────────────────────────
print("\n[3/7] Computing observation geometry...", flush=True)
t0 = time.perf_counter()

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=OBSERVER_ID, spice_handler=spice_handler, config=config,
)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0
fixed_articulation_angles = {
    'SP_North': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager, masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0},
)
inertia_tensor = inertia_result.inertia_tensor

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── Generate synthetic truth ──────────────────────────────────────────
print("\n[4/7] Generating synthetic lightcurve...", flush=True)
t0 = time.perf_counter()

true_axis = np.array([0.6, 0.3, 0.8])
true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([
    np.cos(true_angle_rad / 2),
    np.sin(true_angle_rad / 2) * true_axis[0],
    np.sin(true_angle_rad / 2) * true_axis[1],
    np.sin(true_angle_rad / 2) * true_axis[2],
])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

true_quaternions, _ = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)
k1_vectors, k2_vectors = obj_temp._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(
    satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False,
)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=observer_distances,
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False,
)

np.random.seed(42)
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
print(f"  Lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag", flush=True)

# ══════════════════════════════════════════════════════════════════════
# PERIOD PRE-ANALYSIS: Data-driven omega bounds
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("[5/7] PERIOD PRE-ANALYSIS", flush=True)
print("=" * 70, flush=True)

OMEGA_SAFETY_FACTOR = 5.0  # multiplier on highest detected LC frequency

lc_detrend = observed_lightcurve - np.mean(observed_lightcurve)

# Lomb-Scargle periodogram
f_min = 1.0 / observation_times[-1]
f_max = 0.5 / (observation_times[1] - observation_times[0])
freqs = np.linspace(f_min, f_max, 10000)
angular_freqs = 2 * np.pi * freqs
power = lombscargle(observation_times, lc_detrend, angular_freqs, normalize=True)

# Find peaks
peaks, props = find_peaks(power, height=0.05 * power.max(), distance=50)
peak_freqs = freqs[peaks]
peak_powers = power[peaks]
sort_idx = np.argsort(peak_powers)[::-1]

print(f"\n  Lomb-Scargle periodogram:", flush=True)
print(f"  Frequency range: [{f_min:.6f}, {f_max:.6f}] Hz", flush=True)
print(f"  Peaks found: {len(peaks)}", flush=True)

# Autocorrelation
t_uniform = np.linspace(observation_times[0], observation_times[-1], 500)
lc_interp = interp1d(observation_times, lc_detrend, kind='cubic')(t_uniform)
acf = np.correlate(lc_interp, lc_interp, mode='full')
acf = acf[len(acf)//2:]
acf = acf / acf[0]
dt_uniform = t_uniform[1] - t_uniform[0]
lags = np.arange(len(acf)) * dt_uniform
acf_peaks_idx, _ = find_peaks(acf, height=0.2, distance=10)

acf_period = None
if len(acf_peaks_idx) > 0:
    acf_period = lags[acf_peaks_idx[0]]
    print(f"\n  ACF first peak: {acf_period:.0f}s ({acf_period/3600:.2f} hrs)", flush=True)

# Determine highest significant LC frequency
# Use all peaks with power > 10% of max
significant_mask = peak_powers > 0.1 * power.max()
if np.any(significant_mask):
    sig_freqs = peak_freqs[significant_mask]
    f_max_significant = np.max(sig_freqs)
    f_dominant = peak_freqs[sort_idx[0]]
else:
    f_max_significant = peak_freqs[sort_idx[0]] if len(sort_idx) > 0 else f_min
    f_dominant = f_max_significant

print(f"\n  Top 5 peaks:", flush=True)
for i in sort_idx[:5]:
    f = peak_freqs[i]
    T = 1.0 / f
    print(f"    f={f:.6f} Hz, T={T:.0f}s ({T/3600:.2f}hrs), power={peak_powers[i]:.3f}", flush=True)

# The highest significant LC frequency bounds the fastest angular motion
# Any component of omega must satisfy: omega_i < 2*pi*f_max_LC
# (because faster rotation than this would produce higher-frequency LC variation)
# Apply safety factor for harmonics, non-convexity, and tumbling coupling
omega_bound_rad = 2 * np.pi * f_max_significant * OMEGA_SAFETY_FACTOR
omega_bound_deg = np.rad2deg(omega_bound_rad)

print(f"\n  Highest significant LC frequency: {f_max_significant:.6f} Hz", flush=True)
print(f"  → max angular rate from LC: {np.rad2deg(2*np.pi*f_max_significant):.4f} deg/s", flush=True)
print(f"  Safety factor: {OMEGA_SAFETY_FACTOR}x", flush=True)
print(f"  → Omega bound: ±{omega_bound_deg:.4f} deg/s (±{omega_bound_rad:.6f} rad/s)", flush=True)
print(f"\n  True |omega|: {np.rad2deg(np.linalg.norm(true_omega0)):.4f} deg/s", flush=True)
print(f"  True omega components: {np.rad2deg(true_omega0)} deg/s", flush=True)
print(f"  All components within bound: {np.all(np.abs(true_omega0) < omega_bound_rad)}", flush=True)

all_results["period_analysis"] = {
    "f_dominant_hz": float(f_dominant),
    "f_max_significant_hz": float(f_max_significant),
    "T_dominant_s": float(1.0 / f_dominant),
    "acf_period_s": float(acf_period) if acf_period else None,
    "safety_factor": OMEGA_SAFETY_FACTOR,
    "omega_bound_deg_s": omega_bound_deg,
    "omega_bound_rad_s": float(omega_bound_rad),
    "true_omega_within_bound": bool(np.all(np.abs(true_omega0) < omega_bound_rad)),
    "n_peaks_found": len(peaks),
}

# Save period analysis figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Lightcurve
axes[0].plot(observation_times / 3600, observed_lightcurve, 'ko-', markersize=3, label='Observed')
axes[0].plot(observation_times / 3600, true_lightcurve, 'r-', alpha=0.5, label='True')
axes[0].set_xlabel('Time (hours)')
axes[0].set_ylabel('Magnitude')
axes[0].set_title('Observed Lightcurve')
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Periodogram
axes[1].plot(freqs * 1000, power, 'b-', linewidth=0.5)
if len(peaks) > 0:
    axes[1].plot(peak_freqs * 1000, peak_powers, 'rv', markersize=6)
axes[1].axvline(f_max_significant * 1000, color='r', linestyle='--', alpha=0.5,
                label=f'f_max_sig={f_max_significant*1000:.3f} mHz')
true_omega_mag = np.linalg.norm(true_omega0)
axes[1].axvline(true_omega_mag / (2*np.pi) * 1000, color='g', linestyle='--', alpha=0.5,
                label=f'True rotation freq')
axes[1].set_xlabel('Frequency (mHz)')
axes[1].set_ylabel('Lomb-Scargle Power')
axes[1].set_title('Periodogram')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ACF
axes[2].plot(lags / 3600, acf, 'b-', linewidth=0.5)
if len(acf_peaks_idx) > 0:
    axes[2].plot(lags[acf_peaks_idx] / 3600, acf[acf_peaks_idx], 'rv', markersize=6)
axes[2].axhline(0, color='k', linewidth=0.5)
axes[2].set_xlabel('Lag (hours)')
axes[2].set_ylabel('Autocorrelation')
axes[2].set_title('Autocorrelation Function')
axes[2].set_xlim(0, observation_times[-1] / 3600 / 2)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS_DIR / 'period_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Figure saved: {RESULTS_DIR / 'period_analysis.png'}", flush=True)

check_timeout("period analysis")

# ══════════════════════════════════════════════════════════════════════
# CREATE OBJECTIVES WITH TIGHT BOUNDS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("[6/7] Creating objectives with data-driven bounds", flush=True)
print("=" * 70, flush=True)

bounds_list = [
    (-np.pi, np.pi),   # axis_angle_x
    (-np.pi, np.pi),   # axis_angle_y
    (-np.pi, np.pi),   # axis_angle_z
    (-omega_bound_rad, omega_bound_rad),  # omega_x
    (-omega_bound_rad, omega_bound_rad),  # omega_y
    (-omega_bound_rad, omega_bound_rad),  # omega_z
]

print(f"  Axis-angle bounds: ±{np.degrees(np.pi):.1f} deg", flush=True)
print(f"  Omega bounds:      ±{omega_bound_deg:.4f} deg/s", flush=True)

obj_kwargs = dict(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
)

obj_lofi = ObjectiveFunction(compute_shadows_flag=False, **obj_kwargs)
obj_hifi = ObjectiveFunction(compute_shadows_flag=True, **obj_kwargs)

# ── Timing calibration with tight bounds ──────────────────────────────
print(f"\n  Timing calibration (at data-driven bounds):", flush=True)
rng = np.random.default_rng(42)
bounds_arr = np.array(bounds_list)

# Lo-fi timing
times_lofi = []
for _ in range(10):
    rp = bounds_arr[:, 0] + rng.random(6) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    t0 = time.perf_counter()
    obj_lofi.evaluate(rp)
    times_lofi.append(time.perf_counter() - t0)
t_lofi_avg = np.mean(times_lofi)

# Hi-fi timing
times_hifi = []
for _ in range(3):
    rp = bounds_arr[:, 0] + rng.random(6) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    t0 = time.perf_counter()
    obj_hifi.evaluate(rp)
    times_hifi.append(time.perf_counter() - t0)
t_hifi_avg = np.mean(times_hifi)

speedup = t_hifi_avg / t_lofi_avg if t_lofi_avg > 0 else float('inf')

print(f"    Lo-fi (tumbling, no shadow): {t_lofi_avg:.4f}s/eval", flush=True)
print(f"    Hi-fi (tumbling, shadow):    {t_hifi_avg:.4f}s/eval", flush=True)
print(f"    Speedup:                     {speedup:.1f}x", flush=True)

all_results["timings"] = {
    "lofi_avg_s": t_lofi_avg,
    "hifi_avg_s": t_hifi_avg,
    "speedup": speedup,
    "omega_bound_deg_s": omega_bound_deg,
}

# Predictions
for label, lofi_budget, top_n, hifi_per in [
    ("tiny",   500,   1,  30),
    ("small",  2000,  2,  50),
    ("medium", 5000,  3,  50),
    ("large",  10000, 3, 100),
    ("full",   20000, 5, 200),
]:
    est_s1 = lofi_budget * t_lofi_avg
    est_s2 = top_n * hifi_per * t_hifi_avg
    print(f"    Predicted {label:6s}: S1={est_s1:.0f}s S2={est_s2:.0f}s Total={est_s1+est_s2:.0f}s ({(est_s1+est_s2)/60:.1f}min)", flush=True)

check_timeout("timing calibration")

# ── CountedObjective ──────────────────────────────────────────────────
class CountedObjective:
    def __init__(self, objective_fn, budget, penalty_value=1e10):
        self.objective_fn = objective_fn
        self.budget = budget
        self.penalty_value = penalty_value
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None
        self._last_print = 0

    def __call__(self, params):
        if self.n_evals >= self.budget:
            return self.penalty_value
        self.n_evals += 1
        aa = params[:3]; omega = params[3:]
        q = axis_angle_to_quaternion(aa)
        q = normalize_quaternion(q)
        aa = quaternion_to_axis_angle(q)
        params_norm = np.concatenate([aa, omega])
        value = self.objective_fn.evaluate(params_norm)
        if value < self.best_value:
            self.best_value = value
            self.best_params = params_norm.copy()
        interval = max(500, self.budget // 10)
        if self.n_evals - self._last_print >= interval:
            print(f"      [{self.n_evals}/{self.budget}] best={self.best_value:.4f} elapsed={elapsed():.0f}s", flush=True)
            self._last_print = self.n_evals
        return value

# ── Pipeline function ─────────────────────────────────────────────────
def run_bounded_mixed_fidelity(obj_lofi, obj_hifi, bounds, lofi_budget, top_n, hifi_evals_per, seed=42):
    """Two-stage: tumbling no-shadow DE → tumbling shadow L-BFGS-B."""
    n_params = len(bounds)

    # Stage 1: Lo-fi DE
    print(f"\n  Stage 1: Tumbling + no-shadow DE (budget={lofi_budget})", flush=True)
    counted_lofi = CountedObjective(obj_lofi, budget=lofi_budget)
    popsize = 15
    maxiter = max(1, int(lofi_budget / (popsize * n_params)) - 1)
    print(f"    popsize={popsize}, maxiter={maxiter}, n_params={n_params}", flush=True)

    t0 = time.perf_counter()
    de_result = differential_evolution(
        func=counted_lofi, bounds=bounds, seed=seed,
        maxiter=maxiter, tol=0.01, polish=False,
        strategy='best1bin', mutation=(0.5, 1.0), recombination=0.7,
        updating='deferred', workers=1,
    )
    stage1_time = time.perf_counter() - t0
    n_lofi = counted_lofi.n_evals
    print(f"  Stage 1 done: {n_lofi} evals, {stage1_time:.1f}s, best={de_result.fun:.4f}", flush=True)
    check_timeout("Stage 1 DE")

    # Extract top candidates from DE population
    sorted_idx = np.argsort(de_result.population_energies)[:top_n]
    top_candidates = de_result.population[sorted_idx].copy()
    top_energies = de_result.population_energies[sorted_idx].copy()

    # Stage 2: Hi-fi L-BFGS-B
    print(f"  Stage 2: Tumbling + shadow L-BFGS-B ({top_n} candidates, {hifi_evals_per} evals each)", flush=True)
    t0 = time.perf_counter()
    candidates = []
    n_hifi_total = 0

    for i in range(len(top_candidates)):
        print(f"    Candidate {i+1}/{top_n} (lofi_energy={top_energies[i]:.4f})...", flush=True)
        counted_hifi = CountedObjective(obj_hifi, budget=hifi_evals_per)
        result_i = minimize(
            counted_hifi, top_candidates[i], method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6},
        )
        x_opt = counted_hifi.best_params if counted_hifi.best_params is not None else result_i.x
        f_opt = counted_hifi.best_value if counted_hifi.best_params is not None else result_i.fun
        n_hifi_total += counted_hifi.n_evals
        candidates.append({
            "x_opt": x_opt, "f_opt": f_opt, "n_evals": counted_hifi.n_evals,
            "lofi_energy": float(top_energies[i]), "converged": bool(result_i.success),
        })
        print(f"      → f={f_opt:.4f}, evals={counted_hifi.n_evals}, converged={result_i.success}", flush=True)
        check_timeout(f"Stage 2 candidate {i+1}")

    stage2_time = time.perf_counter() - t0
    best_idx = int(np.argmin([c['f_opt'] for c in candidates]))

    return {
        "x_best": candidates[best_idx]["x_opt"],
        "f_best": candidates[best_idx]["f_opt"],
        "n_evals_lofi": n_lofi, "n_evals_hifi": n_hifi_total,
        "stage1_time": stage1_time, "stage2_time": stage2_time,
        "de_best": float(de_result.fun),
        "candidates": candidates,
    }

# ── Helper ────────────────────────────────────────────────────────────
OMEGA_ERROR_THRESHOLD = 0.1  # deg/s
RMS_THRESHOLD_FACTOR = 2.0

def evaluate_result(result, label):
    x = result["x_best"]
    aa_err = np.rad2deg(np.linalg.norm(x[:3] - true_params[:3]))
    omega_err = np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))
    rms = np.sqrt(result["f_best"] / n_observations)
    omega_ok = omega_err < OMEGA_ERROR_THRESHOLD
    rms_ok = rms < RMS_THRESHOLD_FACTOR * noise_sigma
    success = omega_ok and rms_ok
    total_time = result["stage1_time"] + result["stage2_time"]

    print(f"\n  {label} RESULTS:", flush=True)
    print(f"    Total time:     {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"    Stage 1:        {result['stage1_time']:.1f}s ({result['n_evals_lofi']} lo-fi evals)", flush=True)
    print(f"    Stage 2:        {result['stage2_time']:.1f}s ({result['n_evals_hifi']} hi-fi evals)", flush=True)
    print(f"    DE best (lofi): {result['de_best']:.4f}", flush=True)
    print(f"    Axis-angle err: {aa_err:.4f} deg", flush=True)
    print(f"    Omega err:      {omega_err:.4f} deg/s (threshold: {OMEGA_ERROR_THRESHOLD})", flush=True)
    print(f"    RMS residual:   {rms:.4f} mag (threshold: {RMS_THRESHOLD_FACTOR * noise_sigma:.4f})", flush=True)
    print(f"    {'✓ SUCCESS' if success else '✗ FAIL'}", flush=True)
    if not success:
        if not omega_ok:
            print(f"      omega_err {omega_err:.4f} >= {OMEGA_ERROR_THRESHOLD}", flush=True)
        if not rms_ok:
            print(f"      rms {rms:.4f} >= {RMS_THRESHOLD_FACTOR * noise_sigma:.4f}", flush=True)

    return {
        "total_time_s": total_time,
        "stage1_time_s": result["stage1_time"],
        "stage2_time_s": result["stage2_time"],
        "n_evals_lofi": result["n_evals_lofi"],
        "n_evals_hifi": result["n_evals_hifi"],
        "de_best_lofi": result["de_best"],
        "aa_error_deg": aa_err,
        "omega_error_deg_s": omega_err,
        "rms_residual": rms,
        "success": success,
        "x_best": x.tolist(),
        "true_params": true_params.tolist(),
    }

# ══════════════════════════════════════════════════════════════════════
# [7/7] RUN PIPELINE AT INCREASING SCALE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, flush=True)
print("[7/7] RUNNING PIPELINE", flush=True)
print("=" * 70, flush=True)

# Phase A: Tiny (validate code path)
print("\n--- Phase A: TINY (budget=500, top_n=1, hifi=30) ---", flush=True)
tiny = run_bounded_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 500, 1, 30)
all_results["phases"]["tiny"] = evaluate_result(tiny, "TINY")

# Save intermediate
RESULTS_FILE.write_text(json.dumps({"timestamp": datetime.now().isoformat(),
    "elapsed_s": elapsed(), "timeout": False, "phase": "tiny_done", "results": all_results},
    indent=2, default=str))
check_timeout("tiny pipeline")

# Phase B: Small
remaining = WALL_CLOCK_LIMIT - elapsed()
est_small = 2000 * t_lofi_avg + 2 * 50 * t_hifi_avg + 30
if remaining > est_small:
    print(f"\n--- Phase B: SMALL (budget=2000, top_n=2, hifi=50) ---", flush=True)
    print(f"  Estimated: {est_small:.0f}s, remaining: {remaining:.0f}s", flush=True)
    small = run_bounded_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 2000, 2, 50, seed=123)
    all_results["phases"]["small"] = evaluate_result(small, "SMALL")
    RESULTS_FILE.write_text(json.dumps({"timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed(), "timeout": False, "phase": "small_done", "results": all_results},
        indent=2, default=str))
else:
    print(f"\n  Skipping small: need {est_small:.0f}s, only {remaining:.0f}s left", flush=True)
check_timeout("small pipeline")

# Phase C: Medium
remaining = WALL_CLOCK_LIMIT - elapsed()
est_med = 5000 * t_lofi_avg + 3 * 50 * t_hifi_avg + 30
if remaining > est_med:
    print(f"\n--- Phase C: MEDIUM (budget=5000, top_n=3, hifi=50) ---", flush=True)
    print(f"  Estimated: {est_med:.0f}s, remaining: {remaining:.0f}s", flush=True)
    med = run_bounded_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 5000, 3, 50, seed=456)
    all_results["phases"]["medium"] = evaluate_result(med, "MEDIUM")
    RESULTS_FILE.write_text(json.dumps({"timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed(), "timeout": False, "phase": "medium_done", "results": all_results},
        indent=2, default=str))
else:
    print(f"\n  Skipping medium: need {est_med:.0f}s, only {remaining:.0f}s left", flush=True)
check_timeout("medium pipeline")

# Phase D: Large
remaining = WALL_CLOCK_LIMIT - elapsed()
est_large = 10000 * t_lofi_avg + 3 * 100 * t_hifi_avg + 60
if remaining > est_large:
    print(f"\n--- Phase D: LARGE (budget=10000, top_n=3, hifi=100) ---", flush=True)
    print(f"  Estimated: {est_large:.0f}s, remaining: {remaining:.0f}s", flush=True)
    large = run_bounded_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 10000, 3, 100, seed=789)
    all_results["phases"]["large"] = evaluate_result(large, "LARGE")
    RESULTS_FILE.write_text(json.dumps({"timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed(), "timeout": False, "phase": "large_done", "results": all_results},
        indent=2, default=str))
else:
    print(f"\n  Skipping large: need {est_large:.0f}s, only {remaining:.0f}s left", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"  Omega bound: ±{omega_bound_deg:.4f} deg/s (from LC period analysis, {OMEGA_SAFETY_FACTOR}x safety)", flush=True)
print(f"  Lo-fi eval: {t_lofi_avg:.4f}s | Hi-fi eval: {t_hifi_avg:.4f}s | Speedup: {speedup:.1f}x", flush=True)
print(f"  Total elapsed: {elapsed():.1f}s ({elapsed()/60:.1f} min)", flush=True)
print(flush=True)
for phase_name, phase_data in all_results["phases"].items():
    status = "✓ SUCCESS" if phase_data.get("success") else "✗ FAIL"
    print(f"  {phase_name:8s}: {status} | {phase_data['total_time_s']:.1f}s | ω_err={phase_data['omega_error_deg_s']:.4f}°/s | rms={phase_data['rms_residual']:.4f} | lofi={phase_data['n_evals_lofi']} hifi={phase_data['n_evals_hifi']}", flush=True)

save_and_exit(results=all_results)
