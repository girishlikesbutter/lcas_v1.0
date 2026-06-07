#!/usr/bin/env python3
"""
m096 — Stage 1: Constraint Identification on 30 Seeds.

Stage-by-stage validation approach. This script runs ONLY Step 1
(peak detection, anchor selection, constraint identification) on all
30 seeds and saves EVERYTHING computed — not just what the next stage
needs, but anything we might want to probe later.

Output: data/results/inversion_diagnostics/m096_stage1/
  - seed_{NNN}.npz: per-seed arrays (all computed data)
  - stage1_summary.json: human-readable summary table
  - analysis printed to stdout

Seeds: 10 diagnosed + 20 fresh (stratified by |w| and w_dir)
"""

import sys, os, time
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

DIAGNOSED = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]
FRESH = [s for s in range(100) if s not in DIAGNOSED]
ALL_SEEDS = list(range(100))

Z_NORMALS = {4, 5}

def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


CKPT_DIR = RESULTS_DIR / "m096_stage1"
CKPT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print(f"m096 — Stage 1: Constraint identification ({len(ALL_SEEDS)} seeds)")
print("=" * 70)

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
group_names = list(master['group_names'])

# Pipeline constants (saved so downstream stages use identical values)
N_PHI_COARSE = 36
CONSTRAINT_WEIGHT = 10.0

summary_rows = []

for seed in ALL_SEEDS:
    true_q0 = master['q0s'][seed]
    true_omega0 = master['omega0s'][seed]
    true_omega_mag_dps = float(master['omega_mags'][seed])
    true_lc = master['mag_hifi'][seed]

    # Noisy observed LC (deterministic: seed 42)
    rng = np.random.default_rng(42)
    observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

    # Smoothed LC for anchor ranking
    smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)

    # Peak detection
    peaks_idx, peak_properties = find_peaks(-observed_lc, distance=5, prominence=0.3)
    peak_prominences = peak_properties['prominences']
    peak_mags = observed_lc[peaks_idx]

    # Omega magnitude estimate
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = np.deg2rad(omega_est_dps)
    omega_est_err_pct = (omega_est_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

    # Specular peaks (mag < 9)
    spec_mask = peak_mags < 9.0
    spec_peaks = peaks_idx[spec_mask]
    n_spec = len(spec_peaks)

    # Omega body-frame direction
    w_hat = true_omega0 / np.linalg.norm(true_omega0)
    omega_body_dots = np.abs(w_hat)
    dominant_axis = ['X', 'Y', 'Z'][np.argmax(omega_body_dots)]
    is_diagnosed = seed in DIAGNOSED

    if n_spec < 2:
        # Save everything we have even for failed seeds
        np.savez(str(CKPT_DIR / f"seed_{seed:03d}.npz"),
                 seed=seed,
                 valid=False,
                 error='too_few_spec_peaks',
                 true_q0=true_q0,
                 true_omega0=true_omega0,
                 true_omega_mag_dps=true_omega_mag_dps,
                 true_lc=true_lc,
                 observed_lc=observed_lc,
                 smoothed_lc=smoothed_lc,
                 obs_times=obs_times,
                 peaks_idx=peaks_idx,
                 peak_mags=peak_mags,
                 peak_prominences=peak_prominences,
                 spec_peaks=spec_peaks,
                 omega_est_dps=omega_est_dps,
                 omega_est_rad=omega_est_rad,
                 omega_body_dots=omega_body_dots,
        )
        summary_rows.append({
            'seed': seed, 'group': 'diagnosed' if is_diagnosed else 'fresh',
            'n_peaks': int(len(peaks_idx)), 'n_spec': int(n_spec),
            'omega_est_dps': float(omega_est_dps),
            'omega_true_dps': float(true_omega_mag_dps),
            'omega_est_err_pct': float(omega_est_err_pct),
            'dominant_omega_axis': dominant_axis,
            'error': 'too_few_spec_peaks',
        })
        continue

    # ── Anchor selection ──────────────────────────────────────────────
    smooth_mags_at_spec = smoothed_lc[spec_peaks]
    smooth_ranking = np.argsort(smooth_mags_at_spec)  # ascending = brightest first
    if (len(smooth_ranking) >= 2 and
        abs(smooth_mags_at_spec[smooth_ranking[0]] - smooth_mags_at_spec[smooth_ranking[1]]) < 0.05):
        tied = smooth_ranking[:2]
        anchor_rank = tied[np.argmin(spec_peaks[tied])]  # earlier epoch
        anchor_tie_broken = True
    else:
        anchor_rank = smooth_ranking[0]
        anchor_tie_broken = False

    anchor_idx = int(spec_peaks[anchor_rank])
    anchor_time = obs_times[anchor_idx]
    anchor_mag = observed_lc[anchor_idx]
    anchor_smooth_mag = smoothed_lc[anchor_idx]
    anchor_allowed = get_allowed_normals(anchor_mag)
    anchor_pab = pab_j2000[anchor_idx]

    # ── Constraints ───────────────────────────────────────────────────
    non_anchor = spec_peaks[spec_peaks != anchor_idx]
    constraint_epochs = non_anchor
    constraint_mags = observed_lc[constraint_epochs]
    constraint_smooth_mags = smoothed_lc[constraint_epochs]
    constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
    # Flatten to padded array for NPZ (ragged lists can't be saved directly)
    max_allowed = 10
    constraint_allowed_padded = np.full((len(constraint_epochs), max_allowed), -1, dtype=int)
    constraint_allowed_counts = np.zeros(len(constraint_epochs), dtype=int)
    for ci, allowed in enumerate(constraint_allowed):
        constraint_allowed_padded[ci, :len(allowed)] = allowed
        constraint_allowed_counts[ci] = len(allowed)

    dt_constraints = obs_times[constraint_epochs] - anchor_time
    pab_at_constraints = pab_j2000[constraint_epochs]
    n_constraints = len(constraint_epochs)

    # Constraint classification
    n_bright = int(np.sum(constraint_mags < 5.9))
    n_medium = int(np.sum((constraint_mags >= 5.9) & (constraint_mags < 7.3)))
    n_dim = int(np.sum(constraint_mags >= 7.3))

    # ── PAB geometry ──────────────────────────────────────────────────
    # Full pairwise dot matrix including anchor
    all_pab_epochs = np.concatenate([[anchor_idx], constraint_epochs])
    all_pabs = pab_j2000[all_pab_epochs]
    n_pab = len(all_pabs)
    pab_dot_matrix = all_pabs @ all_pabs.T  # (n_pab, n_pab)
    # Extract off-diagonal for stats
    pab_offdiag = pab_dot_matrix[np.triu_indices(n_pab, k=1)]
    pab_min_dot = float(pab_offdiag.min()) if len(pab_offdiag) > 0 else 1.0
    pab_max_dot = float(pab_offdiag.max()) if len(pab_offdiag) > 0 else 1.0
    pab_mean_dot = float(pab_offdiag.mean()) if len(pab_offdiag) > 0 else 1.0

    # Time spread
    dt_range = float(dt_constraints.max() - dt_constraints.min()) if n_constraints > 0 else 0

    # ── Coarse phi anchor quaternions ─────────────────────────────────
    # Pre-compute for each allowed anchor normal (needed by Stage 2)
    phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
    phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_COARSE, endpoint=False)
    # Store as dict-like: per normal index, the phi array and quaternions
    qa_anchor_ni = []
    qa_anchor_phi = []
    qa_anchor_wxyz = []
    qa_anchor_xyzw = []
    for ni in anchor_allowed:
        phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
        qa = np.array([anchor_q_from_phi(p, unique_normals[ni], anchor_pab)
                        for p in phi_arr])
        qa_anchor_ni.append(ni)
        qa_anchor_phi.append(phi_arr)
        qa_anchor_wxyz.append(qa)
        qa_anchor_xyzw.append(qa[:, [1, 2, 3, 0]])

    # ── Oracle: true omega at anchor ──────────────────────────────────
    _, w_hist = propagate_attitude(true_q0, true_omega0,
        np.array([0.0, anchor_time]), "tumbling", I_tensor)
    true_omega_anchor = w_hist[1]

    # ── Save EVERYTHING per seed ──────────────────────────────────────
    save_dict = dict(
        # Metadata
        seed=seed,
        valid=True,
        group='diagnosed' if is_diagnosed else 'fresh',

        # Truth
        true_q0=true_q0,
        true_omega0=true_omega0,
        true_omega_mag_dps=true_omega_mag_dps,
        true_lc=true_lc,
        true_omega_anchor=true_omega_anchor,

        # Observed / smoothed LC
        observed_lc=observed_lc,
        smoothed_lc=smoothed_lc,
        obs_times=obs_times,

        # All peaks (not just specular)
        peaks_idx=peaks_idx,
        peak_mags=peak_mags,
        peak_prominences=peak_prominences,

        # Specular peaks
        spec_peaks=spec_peaks,

        # Omega estimate
        omega_est_dps=omega_est_dps,
        omega_est_rad=omega_est_rad,
        omega_est_err_pct=omega_est_err_pct,

        # Anchor selection
        anchor_idx=anchor_idx,
        anchor_time=anchor_time,
        anchor_mag=anchor_mag,
        anchor_smooth_mag=anchor_smooth_mag,
        anchor_rank=anchor_rank,
        anchor_tie_broken=anchor_tie_broken,
        anchor_allowed=np.array(anchor_allowed),
        anchor_pab=anchor_pab,
        smooth_mags_at_spec=smooth_mags_at_spec,
        smooth_ranking=smooth_ranking,

        # Constraints
        constraint_epochs=constraint_epochs,
        constraint_mags=constraint_mags,
        constraint_smooth_mags=constraint_smooth_mags,
        constraint_allowed_padded=constraint_allowed_padded,
        constraint_allowed_counts=constraint_allowed_counts,
        dt_constraints=dt_constraints,
        pab_at_constraints=pab_at_constraints,
        n_constraints=n_constraints,
        n_bright=n_bright,
        n_medium=n_medium,
        n_dim=n_dim,

        # PAB geometry
        pab_dot_matrix=pab_dot_matrix,
        pab_all_epochs=all_pab_epochs,
        pab_min_dot=pab_min_dot,
        pab_max_dot=pab_max_dot,
        pab_mean_dot=pab_mean_dot,
        dt_range_s=dt_range,

        # Coarse phi anchor quaternions (for Stage 2 grid search)
        qa_anchor_n_sets=len(qa_anchor_ni),
        qa_anchor_ni=np.array(qa_anchor_ni),
        phi_coarse_xy=phi_coarse_xy,
        phi_coarse_z=phi_coarse_z,

        # Omega body-frame direction
        omega_body_dots=omega_body_dots,
        dominant_omega_axis=dominant_axis,

        # Pipeline constants used
        n_phi_coarse=N_PHI_COARSE,
        constraint_weight=CONSTRAINT_WEIGHT,
    )
    # Add variable-length phi/quaternion arrays per anchor normal
    for i, ni in enumerate(qa_anchor_ni):
        save_dict[f'qa_phi_{i}'] = qa_anchor_phi[i]
        save_dict[f'qa_wxyz_{i}'] = qa_anchor_wxyz[i]
        save_dict[f'qa_xyzw_{i}'] = qa_anchor_xyzw[i]

    np.savez(str(CKPT_DIR / f"seed_{seed:03d}.npz"), **save_dict)

    # ── Summary row ───────────────────────────────────────────────────
    summary_rows.append({
        'seed': seed,
        'group': 'diagnosed' if is_diagnosed else 'fresh',
        'n_peaks': int(len(peaks_idx)),
        'n_spec': int(n_spec),
        'n_constraints': int(n_constraints),
        'n_bright': int(n_bright),
        'n_medium': int(n_medium),
        'n_dim': int(n_dim),
        'omega_est_dps': float(omega_est_dps),
        'omega_true_dps': float(true_omega_mag_dps),
        'omega_est_err_pct': float(omega_est_err_pct),
        'anchor_idx': int(anchor_idx),
        'anchor_time': float(anchor_time),
        'anchor_mag': float(anchor_mag),
        'anchor_allowed': [group_names[i] for i in anchor_allowed],
        'anchor_tie_broken': bool(anchor_tie_broken),
        'pab_min_dot': pab_min_dot,
        'pab_max_dot': pab_max_dot,
        'pab_mean_dot': pab_mean_dot,
        'dt_range_s': dt_range,
        'dominant_omega_axis': dominant_axis,
        'omega_body_dots': omega_body_dots.tolist(),
        'true_omega_anchor': true_omega_anchor.tolist(),
        'constraint_epochs': constraint_epochs.tolist(),
        'constraint_mags': constraint_mags.tolist(),
        'spec_peaks': spec_peaks.tolist(),
    })

save_results(str(CKPT_DIR / "stage1_summary.json"), summary_rows)
print(f"\nSaved {len(ALL_SEEDS)} seed checkpoints + summary to {CKPT_DIR}/")

# ══════════════════════════════════════════════════════════════════════
# VERIFICATION: load one seed back and check nothing is missing
# ══════════════════════════════════════════════════════════════════════
test_seed = 27
d = np.load(str(CKPT_DIR / f"seed_{test_seed:03d}.npz"), allow_pickle=True)
expected_keys = [
    'seed', 'valid', 'true_q0', 'true_omega0', 'true_omega_mag_dps',
    'true_lc', 'true_omega_anchor', 'observed_lc', 'smoothed_lc', 'obs_times',
    'peaks_idx', 'peak_mags', 'peak_prominences', 'spec_peaks',
    'omega_est_dps', 'omega_est_rad', 'omega_est_err_pct',
    'anchor_idx', 'anchor_time', 'anchor_mag', 'anchor_smooth_mag',
    'anchor_rank', 'anchor_tie_broken', 'anchor_allowed', 'anchor_pab',
    'smooth_mags_at_spec', 'smooth_ranking',
    'constraint_epochs', 'constraint_mags', 'constraint_smooth_mags',
    'constraint_allowed_padded', 'constraint_allowed_counts',
    'dt_constraints', 'pab_at_constraints',
    'n_constraints', 'n_bright', 'n_medium', 'n_dim',
    'pab_dot_matrix', 'pab_all_epochs', 'pab_min_dot', 'pab_max_dot', 'pab_mean_dot',
    'dt_range_s', 'qa_anchor_n_sets', 'qa_anchor_ni',
    'phi_coarse_xy', 'phi_coarse_z',
    'omega_body_dots', 'dominant_omega_axis',
    'n_phi_coarse', 'constraint_weight',
]
missing = [k for k in expected_keys if k not in d]
extra = [k for k in d.files if k not in expected_keys and not k.startswith('qa_')]
print(f"\nVerification (seed {test_seed}):")
print(f"  Keys in file: {len(d.files)}")
print(f"  Missing: {missing if missing else 'NONE'}")
print(f"  Extra (non-qa): {extra if extra else 'NONE'}")

# Quick shape checks
print(f"  observed_lc: {d['observed_lc'].shape}")
print(f"  constraint_epochs: {d['constraint_epochs'].shape}")
print(f"  pab_at_constraints: {d['pab_at_constraints'].shape}")
print(f"  pab_dot_matrix: {d['pab_dot_matrix'].shape}")
print(f"  dt_constraints: {d['dt_constraints'].shape}")
n_sets = int(d['qa_anchor_n_sets'])
for i in range(n_sets):
    print(f"  qa_xyzw_{i}: {d[f'qa_xyzw_{i}'].shape} (normal {d['qa_anchor_ni'][i]})")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS (same as before)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STAGE 1 ANALYSIS")
print(f"{'='*70}")

print(f"\n{'Seed':>4} {'Grp':>5} {'Pk':>3} {'Sp':>3} {'Cst':>4} {'B/M/D':>7} "
      f"{'|w|est':>7} {'|w|true':>7} {'err%':>6} "
      f"{'PABmin':>7} {'Anc':>6} {'wAxis':>5}")
print("-" * 90)

for r in sorted(summary_rows, key=lambda x: x['seed']):
    if 'error' in r:
        print(f"{r['seed']:4d} {'D' if r['seed'] in DIAGNOSED else 'F':>5} "
              f"{r['n_peaks']:3d} {r.get('n_spec',0):3d}   *** {r['error']} ***")
        continue
    grp = 'D' if r['seed'] in DIAGNOSED else 'F'
    bmd = f"{r['n_bright']}/{r['n_medium']}/{r['n_dim']}"
    print(f"{r['seed']:4d} {grp:>5} {r['n_peaks']:3d} {r['n_spec']:3d} {r['n_constraints']:4d} "
          f"{bmd:>7} {r['omega_est_dps']:7.3f} {r['omega_true_dps']:7.3f} "
          f"{r['omega_est_err_pct']:+5.1f}% {r['pab_min_dot']:7.4f} "
          f"{r['anchor_mag']:6.2f} {r['dominant_omega_axis']:>5}")

valid = [r for r in summary_rows if 'error' not in r]
print(f"\n--- Summary ---")
print(f"Seeds with <2 spec peaks: {sum(1 for r in summary_rows if 'error' in r)}/{len(ALL_SEEDS)}")
print(f"|w| estimation: median err {np.median([abs(r['omega_est_err_pct']) for r in valid]):.1f}%, "
      f"max {max(abs(r['omega_est_err_pct']) for r in valid):.1f}%")
print(f"Spec peaks: min={min(r['n_spec'] for r in valid)}, "
      f"median={int(np.median([r['n_spec'] for r in valid]))}, "
      f"max={max(r['n_spec'] for r in valid)}")
print(f"Constraints: min={min(r['n_constraints'] for r in valid)}, "
      f"median={int(np.median([r['n_constraints'] for r in valid]))}, "
      f"max={max(r['n_constraints'] for r in valid)}")

print(f"\n--- Potential Issues ---")
for r in valid:
    issues = []
    if r['n_constraints'] < 4: issues.append(f"few constraints ({r['n_constraints']})")
    if r['n_bright'] < 2: issues.append(f"few bright ({r['n_bright']})")
    if r['pab_min_dot'] > 0.999: issues.append(f"PAB degenerate ({r['pab_min_dot']:.4f})")
    if abs(r['omega_est_err_pct']) > 20: issues.append(f"|w| est off by {r['omega_est_err_pct']:+.1f}%")
    if r['n_spec'] < 3: issues.append(f"only {r['n_spec']} spec peaks")
    if issues:
        grp = 'D' if r['seed'] in DIAGNOSED else 'F'
        print(f"  seed {r['seed']:3d} ({grp}): {'; '.join(issues)}")

print(f"\nDone.")
