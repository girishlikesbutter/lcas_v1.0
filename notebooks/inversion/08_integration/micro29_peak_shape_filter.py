#!/usr/bin/env python3
"""Micro-29 — Peak-shape omega filter.

For each omega candidate from a band-sweep bridge solve, test whether the omega
produces the correct brightness peak shape at the anchor peaks. At a brightness
peak, magnitude is a local MINIMUM (brighter). Propagating ±1 epoch from the
peak with the correct omega should produce HIGHER magnitudes (dimmer) on both
sides. Wrong-winding candidates often break this shape.

Test at both the START peak and END peak of each leg (using the arriving omega
at the end). A candidate must pass both to survive the combined filter.
"""
import sys, os, time, json, numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.experiment_setup import setup_experiment, save_results, brightness_single_epoch
from src.dynamics.attitude_propagator import propagate_attitude

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_STARTS = 5          # reduced from 10 for budget
N_WORKERS = 8
ARRIVAL_THRESH = 1e-6
MAG_TOL = 0.05        # deg/s  (dedup tolerance)
DIR_TOL = 5.0         # degrees (dedup tolerance)
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ---------------------------------------------------------------------------
# Bridge-solve infrastructure (from micro26b, N_STARTS=5)
# ---------------------------------------------------------------------------
_SHARED = {}


def _ang_dist_deg(a, b):
    """Angular distance between two vectors in degrees."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1., 1.))))


def _init_worker(shared):
    global _SHARED
    _SHARED = shared


def _solve_band(args):
    """Solve for omega in a single |omega| band from a given initial guess."""
    w0, lb_rad, ub_rad = args
    qs, qe, dt, I = _SHARED['qs'], _SHARED['qe'], _SHARED['dt'], _SHARED['I']

    def obj(w):
        wn2 = float(np.dot(w, w))
        pen = max(0., lb_rad**2 - wn2)**2 * 1e4 + max(0., wn2 - ub_rad**2)**2 * 1e4
        qp, _ = propagate_attitude(qs, w, np.array([0., dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], qe), -1., 1.)
        return (1. - d * d) + pen

    res = minimize(obj, w0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-9})
    wk = res.x
    qp, _ = propagate_attitude(qs, wk, np.array([0., dt]), "tumbling", I)
    d = np.clip(np.dot(qp[-1], qe), -1., 1.)
    return {
        'omega': wk.tolist(),
        'mag_degs': float(np.rad2deg(np.linalg.norm(wk))),
        'arrival_err': float(1. - d * d),
    }


def run_leg(q_start, q_end, dt, I, rng_seed):
    """Run banded bridge solve for one leg, return deduped candidates."""
    bands = [(0.5 * i, 0.5 * (i + 1)) for i in range(13)]
    rng = np.random.RandomState(rng_seed)
    tasks = []
    for lb_d, ub_d in bands:
        lb_r = np.deg2rad(lb_d) + 1e-6
        ub_r = np.deg2rad(ub_d)
        for _ in range(N_STARTS):
            d = rng.randn(3)
            d /= np.linalg.norm(d)
            tasks.append(((lb_r + (ub_r - lb_r) * rng.rand()) * d, lb_r, ub_r))

    shared = {'qs': q_start, 'qe': q_end, 'dt': dt, 'I': I}
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(shared,)) as pool:
        raw = pool.map(_solve_band, tasks, chunksize=4)

    valid = sorted(
        [r for r in raw if r['arrival_err'] < ARRIVAL_THRESH],
        key=lambda r: r['mag_degs'])

    # Direction-aware dedup
    deduped = []
    for r in valid:
        w_r = np.array(r['omega'])
        is_dup = False
        for d in deduped:
            if abs(r['mag_degs'] - d['mag_degs']) < MAG_TOL:
                if _ang_dist_deg(w_r, np.array(d['omega'])) < DIR_TOL:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(r)
    return deduped


def _find_true_idx(candidates, true_omega):
    """Find candidate closest to truth (mag + direction)."""
    true_mag = float(np.rad2deg(np.linalg.norm(true_omega)))
    best_i, best_err = -1, 1e30
    for i, c in enumerate(candidates):
        mag_err = abs(c['mag_degs'] - true_mag)
        if mag_err < 0.1:
            vec_err = np.linalg.norm(np.array(c['omega']) - true_omega)
            if vec_err < best_err:
                best_err = vec_err
                best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Peak-shape filter
# ---------------------------------------------------------------------------
def peak_shape_test(q_peak, omega, peak_idx, dt_samp, ctx, I):
    """Test whether omega produces a valid brightness peak shape.

    At a brightness peak, magnitude is a local MINIMUM (brighter).
    Neighbors should have HIGHER magnitude (dimmer).

    Returns (pass_prev, pass_next, m_peak, m_prev, m_next).
    """
    m_peak = brightness_single_epoch(q_peak, peak_idx, ctx, use_shadows=False)

    # Forward 1 epoch
    q_fwd, _ = propagate_attitude(q_peak, omega, np.array([0.0, dt_samp]), "tumbling", I)
    q_next = q_fwd[-1]
    m_next = brightness_single_epoch(q_next, peak_idx + 1, ctx, use_shadows=False)

    # Backward 1 epoch (time-reversal: propagate -omega forward by dt)
    q_bwd, _ = propagate_attitude(q_peak, -omega, np.array([0.0, dt_samp]), "tumbling", I)
    q_prev = q_bwd[-1]
    m_prev = brightness_single_epoch(q_prev, peak_idx - 1, ctx, use_shadows=False)

    # At a brightness peak, magnitude is a local MINIMUM
    # So neighbors should have HIGHER magnitude (dimmer)
    pass_prev = bool(m_prev > m_peak)
    pass_next = bool(m_next > m_peak)

    return pass_prev, pass_next, float(m_peak), float(m_prev), float(m_next)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    t0 = time.time()

    # --- Setup ---
    CTX = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0), end_time_utc='2020-02-05T11:00:00')
    I = CTX.inertia_tensor
    OBS_T = CTX.observation_times
    dt_sampling = CTX.dt_sampling

    # --- Peak epochs ---
    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "micro13_stage1.npz")["peaks"]]
    print(f"Peaks: {PEAKS}", flush=True)

    # Propagate true attitude to peak epochs
    times_p = np.array([0.0] + [float(OBS_T[p]) for p in PEAKS])
    q_traj, om_traj = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_p, "tumbling", I)
    q_A, q_B, q_C = q_traj[1], q_traj[2], q_traj[3]
    om_A, om_B, om_C = om_traj[1], om_traj[2], om_traj[3]

    dt_0 = float(OBS_T[PEAKS[1]] - OBS_T[PEAKS[0]])
    dt_1 = float(OBS_T[PEAKS[2]] - OBS_T[PEAKS[1]])
    print(f"dt0={dt_0:.1f}s, dt1={dt_1:.1f}s, setup {time.time()-t0:.1f}s", flush=True)

    # --- Bridge solves ---
    print("\nRunning bridge solve Leg 0 (A->B)...", flush=True)
    leg0 = run_leg(q_A, q_B, dt_0, I, SEED)
    print(f"Leg 0: {len(leg0)} candidates ({time.time()-t0:.1f}s)", flush=True)

    print("Running bridge solve Leg 1 (B->C)...", flush=True)
    leg1 = run_leg(q_B, q_C, dt_1, I, SEED + 1000)
    print(f"Leg 1: {len(leg1)} candidates ({time.time()-t0:.1f}s)", flush=True)

    # Find true omega in each leg's candidates
    true_idx_0 = _find_true_idx(leg0, om_A)
    true_idx_1 = _find_true_idx(leg1, om_B)
    print(f"True omega index: leg0={true_idx_0}, leg1={true_idx_1}", flush=True)

    # --- Peak-shape filter ---
    print(f"\n--- Peak-shape filter ---", flush=True)

    # Pre-compute peak magnitudes (used for display but test recomputes internally)
    results_leg0 = []
    results_leg1 = []

    # ===== LEG 0 =====
    print(f"\nLeg 0: testing {len(leg0)} candidates at start peak A (epoch {PEAKS[0]}) "
          f"and end peak B (epoch {PEAKS[1]})...", flush=True)

    for k, cand in enumerate(leg0):
        omega_k = np.array(cand['omega'])
        mag_degs = cand['mag_degs']
        dir_err = _ang_dist_deg(omega_k, om_A)

        # Test at START peak A
        pass_prev_A, pass_next_A, m_peak_A, m_prev_A, m_next_A = \
            peak_shape_test(q_A, omega_k, PEAKS[0], dt_sampling, CTX, I)
        start_pass = pass_prev_A and pass_next_A

        # Test at END peak B: propagate full leg to get arriving omega
        q_arr, om_arr = propagate_attitude(
            q_A, omega_k, np.array([0.0, dt_0]), "tumbling", I)
        omega_arriving = om_arr[-1]

        pass_prev_B, pass_next_B, m_peak_B, m_prev_B, m_next_B = \
            peak_shape_test(q_B, omega_arriving, PEAKS[1], dt_sampling, CTX, I)
        end_pass = pass_prev_B and pass_next_B

        combined_pass = start_pass and end_pass
        is_true = (k == true_idx_0)

        results_leg0.append({
            'idx': k,
            'mag_degs': round(mag_degs, 4),
            'dir_err_vs_truth_deg': round(dir_err, 2),
            'is_true': is_true,
            'start_peak': {
                'pass_prev': pass_prev_A, 'pass_next': pass_next_A,
                'pass': start_pass,
                'm_peak': round(m_peak_A, 4), 'm_prev': round(m_prev_A, 4),
                'm_next': round(m_next_A, 4),
            },
            'end_peak': {
                'pass_prev': pass_prev_B, 'pass_next': pass_next_B,
                'pass': end_pass,
                'm_peak': round(m_peak_B, 4), 'm_prev': round(m_prev_B, 4),
                'm_next': round(m_next_B, 4),
            },
            'start_pass': start_pass,
            'end_pass': end_pass,
            'combined_pass': combined_pass,
        })

        label = "TRUE" if is_true else "    "
        status = "PASS" if combined_pass else "FAIL"
        print(f"  [{label}] k={k:2d}  |w|={mag_degs:5.2f} deg/s  "
              f"dir_err={dir_err:6.1f} deg  "
              f"start={'P' if start_pass else 'F'}  end={'P' if end_pass else 'F'}  "
              f"=> {status}", flush=True)

    print(f"Leg 0 filter done ({time.time()-t0:.1f}s)", flush=True)

    # ===== LEG 1 =====
    print(f"\nLeg 1: testing {len(leg1)} candidates at start peak B (epoch {PEAKS[1]}) "
          f"and end peak C (epoch {PEAKS[2]})...", flush=True)

    for j, cand in enumerate(leg1):
        omega_j = np.array(cand['omega'])
        mag_degs = cand['mag_degs']
        dir_err = _ang_dist_deg(omega_j, om_B)

        # Test at START peak B
        pass_prev_B, pass_next_B, m_peak_B, m_prev_B, m_next_B = \
            peak_shape_test(q_B, omega_j, PEAKS[1], dt_sampling, CTX, I)
        start_pass = pass_prev_B and pass_next_B

        # Test at END peak C: propagate full leg to get arriving omega
        q_arr, om_arr = propagate_attitude(
            q_B, omega_j, np.array([0.0, dt_1]), "tumbling", I)
        omega_arriving = om_arr[-1]

        pass_prev_C, pass_next_C, m_peak_C, m_prev_C, m_next_C = \
            peak_shape_test(q_C, omega_arriving, PEAKS[2], dt_sampling, CTX, I)
        end_pass = pass_prev_C and pass_next_C

        combined_pass = start_pass and end_pass
        is_true = (j == true_idx_1)

        results_leg1.append({
            'idx': j,
            'mag_degs': round(mag_degs, 4),
            'dir_err_vs_truth_deg': round(dir_err, 2),
            'is_true': is_true,
            'start_peak': {
                'pass_prev': pass_prev_B, 'pass_next': pass_next_B,
                'pass': start_pass,
                'm_peak': round(m_peak_B, 4), 'm_prev': round(m_prev_B, 4),
                'm_next': round(m_next_B, 4),
            },
            'end_peak': {
                'pass_prev': pass_prev_C, 'pass_next': pass_next_C,
                'pass': end_pass,
                'm_peak': round(m_peak_C, 4), 'm_prev': round(m_prev_C, 4),
                'm_next': round(m_next_C, 4),
            },
            'start_pass': start_pass,
            'end_pass': end_pass,
            'combined_pass': combined_pass,
        })

        label = "TRUE" if is_true else "    "
        status = "PASS" if combined_pass else "FAIL"
        print(f"  [{label}] j={j:2d}  |w|={mag_degs:5.2f} deg/s  "
              f"dir_err={dir_err:6.1f} deg  "
              f"start={'P' if start_pass else 'F'}  end={'P' if end_pass else 'F'}  "
              f"=> {status}", flush=True)

    print(f"Leg 1 filter done ({time.time()-t0:.1f}s)", flush=True)

    # --- Summary statistics ---
    def summarise_leg(results, leg_name):
        n = len(results)
        n_start = sum(1 for r in results if r['start_pass'])
        n_end = sum(1 for r in results if r['end_pass'])
        n_combined = sum(1 for r in results if r['combined_pass'])
        true_entry = [r for r in results if r['is_true']]
        true_survives_start = true_entry[0]['start_pass'] if true_entry else None
        true_survives_end = true_entry[0]['end_pass'] if true_entry else None
        true_survives_combined = true_entry[0]['combined_pass'] if true_entry else None

        print(f"\n  {leg_name}: {n} candidates")
        print(f"    Start peak: {n_start} pass, {n - n_start} killed  "
              f"(kill rate {(n - n_start)/n:.1%})")
        print(f"    End peak:   {n_end} pass, {n - n_end} killed  "
              f"(kill rate {(n - n_end)/n:.1%})")
        print(f"    Combined:   {n_combined} pass, {n - n_combined} killed  "
              f"(kill rate {(n - n_combined)/n:.1%})")
        if true_entry:
            print(f"    True omega survives: start={true_survives_start}, "
                  f"end={true_survives_end}, combined={true_survives_combined}")
        else:
            print(f"    True omega NOT FOUND in candidates!")

        return {
            'n_total': n,
            'n_pass_start': n_start,
            'n_pass_end': n_end,
            'n_pass_combined': n_combined,
            'kill_rate_start': round((n - n_start) / n, 4) if n > 0 else 0,
            'kill_rate_end': round((n - n_end) / n, 4) if n > 0 else 0,
            'kill_rate_combined': round((n - n_combined) / n, 4) if n > 0 else 0,
            'true_omega_survives_start': true_survives_start,
            'true_omega_survives_end': true_survives_end,
            'true_omega_survives_combined': true_survives_combined,
        }

    print(f"\n{'='*60}")
    print(f"=== MICRO-29 SUMMARY ===")
    summary_leg0 = summarise_leg(results_leg0, "Leg 0 (A->B)")
    summary_leg1 = summarise_leg(results_leg1, "Leg 1 (B->C)")

    # Per |omega| band analysis (0.5 deg/s bands)
    def band_analysis(results):
        bands = {}
        for r in results:
            band_idx = int(r['mag_degs'] / 0.5)
            band_key = f"{0.5*band_idx:.1f}-{0.5*(band_idx+1):.1f}"
            if band_key not in bands:
                bands[band_key] = {'pass': 0, 'fail': 0}
            if r['combined_pass']:
                bands[band_key]['pass'] += 1
            else:
                bands[band_key]['fail'] += 1
        return bands

    bands_leg0 = band_analysis(results_leg0)
    bands_leg1 = band_analysis(results_leg1)

    print(f"\n  Per-band (Leg 0): {json.dumps(bands_leg0, indent=4)}")
    print(f"  Per-band (Leg 1): {json.dumps(bands_leg1, indent=4)}")

    runtime = time.time() - t0
    print(f"\n  Runtime: {runtime:.1f}s")
    print(f"{'='*60}")

    # --- Save JSON ---
    output_json = {
        'peaks': PEAKS,
        'dt_0': round(dt_0, 2),
        'dt_1': round(dt_1, 2),
        'dt_sampling': round(dt_sampling, 4),
        'n_starts_per_band': N_STARTS,
        'n_bands': 13,
        'dedup': {'mag_tol_degs': MAG_TOL, 'dir_tol_deg': DIR_TOL},
        'leg0': {
            'summary': summary_leg0,
            'candidates': results_leg0,
            'bands': bands_leg0,
            'true_omega_idx': true_idx_0,
        },
        'leg1': {
            'summary': summary_leg1,
            'candidates': results_leg1,
            'bands': bands_leg1,
            'true_omega_idx': true_idx_1,
        },
        'runtime_s': round(runtime, 1),
    }
    out_json = RESULTS_DIR / "micro29_peak_shape_filter.json"
    save_results(out_json, output_json)
    print(f"\nSaved: {out_json}", flush=True)

    # --- Plot ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Micro-29: Peak-shape omega filter", fontsize=13)

    for ax, results, leg_label in [(ax0, results_leg0, "Leg 0 (A→B)"),
                                    (ax1, results_leg1, "Leg 1 (B→C)")]:
        # Sort by |omega| magnitude
        sorted_results = sorted(results, key=lambda r: r['mag_degs'])
        n = len(sorted_results)
        x = np.arange(n)
        mags = [r['mag_degs'] for r in sorted_results]

        colors = []
        for r in sorted_results:
            if r['combined_pass']:
                colors.append('green')
            elif r['start_pass'] or r['end_pass']:
                colors.append('orange')
            else:
                colors.append('red')

        ax.bar(x, mags, color=colors, edgecolor='k', linewidth=0.5, alpha=0.8)

        # Mark true omega with a star
        for i, r in enumerate(sorted_results):
            if r['is_true']:
                ax.plot(i, r['mag_degs'], '*', color='gold', markersize=18,
                        markeredgecolor='black', markeredgewidth=1.0, zorder=5)

        ax.set_ylabel('|omega| (deg/s)')
        ax.set_title(f"{leg_label}: {n} candidates  |  "
                     f"green=pass combined, orange=partial, red=fail")
        ax.grid(True, alpha=0.3, axis='y')

        if n > 30:
            ax.set_xticks(np.arange(0, n, max(1, n // 20)))

    ax1.set_xlabel('Candidate (sorted by |omega|)')
    plt.tight_layout()

    out_png = RESULTS_DIR / "micro29_peak_shape_filter.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}", flush=True)
