#!/usr/bin/env python3
"""Micro-37 -- BRDF specular glint profile characterization.

Question: How does the specular glint profile (brightness vs n.PAB
misalignment angle) depend on BRDF parameters (r_s, r_d, n_phong)
and phase angle?

Method: Pure BRDF math on a synthetic flat plate (1 m^2). No satellite,
no SPICE, no STL files. Sweep facet normal misalignment from PAB and
record magnitude for various parameter combinations.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.experiment_setup import save_results
from src.computation.brdf import calculate_brdf_vectorized, convert_flux_to_magnitude

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBSERVER_DISTANCE_KM = 36000.0   # GEO distance
PLATE_AREA = 1.0                 # m^2

# Misalignment sweep: 0 to 30 degrees in 0.1-degree steps
THETA_DEG = np.arange(0.0, 30.01, 0.1)   # 301 points
THETA_RAD = np.deg2rad(THETA_DEG)

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_JSON = RESULTS_DIR / "m037_brdf_glint_profile.json"
RESULTS_PNG  = RESULTS_DIR / "m037_brdf_glint_profile.png"


# ---------------------------------------------------------------------------
# Helper: compute magnitude profile for given BRDF params and phase angle
# ---------------------------------------------------------------------------
def compute_magnitude_profile(r_d, r_s, n_phong, phase_angle_deg, area=PLATE_AREA):
    """
    Compute apparent magnitude vs misalignment angle theta.

    Geometry:
        PAB = [0, 0, 1]
        k1 = [sin(alpha/2), 0, cos(alpha/2)]   (sun)
        k2 = [-sin(alpha/2), 0, cos(alpha/2)]   (observer)
        n(theta) = [sin(theta), 0, cos(theta)]  (facet normal)

    Returns:
        magnitudes: array of apparent magnitudes (inf where flux=0)
        flux_nums:  array of flux numerators (rho * area * n.k1 * n.k2)
    """
    alpha_rad = np.deg2rad(phase_angle_deg)
    half_alpha = alpha_rad / 2.0

    k1 = np.array([np.sin(half_alpha), 0.0, np.cos(half_alpha)])
    k2 = np.array([-np.sin(half_alpha), 0.0, np.cos(half_alpha)])

    # Halfway vector (should be [0, 0, 1] = PAB)
    h = k1 + k2
    h = h / np.linalg.norm(h)

    # Sweep facet normals
    normals = np.column_stack([np.sin(THETA_RAD),
                               np.zeros_like(THETA_RAD),
                               np.cos(THETA_RAD)])   # (N, 3)

    n_dot_k1 = normals @ k1        # (N,)
    n_dot_k2 = normals @ k2        # (N,)
    n_dot_h  = normals @ h         # (N,)
    h_dot_k1 = float(np.dot(h, k1))  # scalar, same for all

    # Visibility mask
    visible = (n_dot_k1 > 0) & (n_dot_k2 > 0)

    # Prepare BRDF arrays (broadcast scalars to arrays)
    N = len(THETA_RAD)
    r_d_arr    = np.full(N, r_d)
    r_s_arr    = np.full(N, r_s)
    n_phong_arr = np.full(N, float(n_phong))
    h_dot_k1_arr = np.full(N, h_dot_k1)

    # Compute BRDF
    rho = calculate_brdf_vectorized(
        n_dot_k1, n_dot_k2, h_dot_k1_arr, n_dot_h,
        r_d_arr, r_s_arr, n_phong_arr)

    # Flux numerator = rho * area * n.k1 * n.k2
    flux_num = rho * area * n_dot_k1 * n_dot_k2
    flux_num[~visible] = 0.0

    # Convert to magnitude
    magnitudes = np.full(N, np.inf)
    for i in range(N):
        if flux_num[i] > 1e-20:
            magnitudes[i] = convert_flux_to_magnitude(flux_num[i], OBSERVER_DISTANCE_KM)

    return magnitudes, flux_num


def compute_diffuse_only_profile(r_d, r_s, phase_angle_deg, area=PLATE_AREA):
    """Compute magnitude profile with r_s=0 (pure diffuse) for reference."""
    mags, flux = compute_magnitude_profile(r_d, 0.0, 1.0, phase_angle_deg, area)
    return mags, flux


# ---------------------------------------------------------------------------
# FWHM extraction
# ---------------------------------------------------------------------------
def extract_fwhm(flux_arr):
    """
    Extract FWHM of the flux peak in degrees.

    Finds theta where flux drops to half the peak (at theta=0).
    Returns FWHM = 2 * theta_half. Returns NaN if peak is zero.
    """
    peak_flux = flux_arr[0]
    if peak_flux <= 1e-20:
        return np.nan

    half_peak = peak_flux / 2.0
    # Find first index where flux drops below half-peak
    below_half = np.where(flux_arr < half_peak)[0]
    if len(below_half) == 0:
        return 2.0 * THETA_DEG[-1]  # wider than our sweep

    idx = below_half[0]
    # Linear interpolation for sub-step accuracy
    if idx > 0:
        f_above = flux_arr[idx - 1]
        f_below = flux_arr[idx]
        frac = (f_above - half_peak) / (f_above - f_below)
        theta_half = THETA_DEG[idx - 1] + frac * (THETA_DEG[idx] - THETA_DEG[idx - 1])
    else:
        theta_half = THETA_DEG[0]

    return 2.0 * theta_half


def extract_specular_equals_diffuse_angle(r_d, r_s, n_phong, phase_angle_deg):
    """
    Find theta where specular component equals diffuse component.

    Computes full BRDF and diffuse-only BRDF, finds where the specular
    excess drops to zero (i.e., total flux = 2 * diffuse flux).
    Returns angle in degrees, or NaN if not found.
    """
    _, flux_total = compute_magnitude_profile(r_d, r_s, n_phong, phase_angle_deg)
    _, flux_diffuse = compute_diffuse_only_profile(r_d, r_s, phase_angle_deg)

    # The diffuse-only profile uses r_s=0, so diffuse component is slightly
    # different (Ashikhmin-Shirley diffuse has a (1-r_s) factor).
    # Instead, compute specular contribution = total - diffuse_component
    # where diffuse_component uses the actual r_s in the (1-r_s) factor.
    # Simpler: find where total_flux / diffuse_baseline <= 2
    # But more precisely, compute with and without specular:

    # Recompute with r_s=0 but keep the (1-r_s) scaling:
    # Actually the cleanest approach: compute the ratio of specular to diffuse
    # by evaluating both separately.

    # For clarity, let's compute the "diffuse component" properly:
    # rho_diff = (28*r_d/(23*pi)) * (1-r_s) * (1 - alpha^5) * (1 - beta^5)
    # This is part of the full BRDF. The specular part is the remainder.
    # total = diffuse + specular, so specular = total - diffuse
    # We want theta where specular <= diffuse.

    # Compute diffuse-only BRDF (same params but n_phong doesn't matter for diffuse)
    alpha_rad = np.deg2rad(phase_angle_deg)
    half_alpha = alpha_rad / 2.0
    k1 = np.array([np.sin(half_alpha), 0.0, np.cos(half_alpha)])
    k2 = np.array([-np.sin(half_alpha), 0.0, np.cos(half_alpha)])
    h = k1 + k2
    h = h / np.linalg.norm(h)

    normals = np.column_stack([np.sin(THETA_RAD),
                               np.zeros_like(THETA_RAD),
                               np.cos(THETA_RAD)])
    n_dot_k1 = normals @ k1
    n_dot_k2 = normals @ k2

    # Diffuse component only (Ashikhmin-Shirley diffuse)
    alpha_term = 1.0 - n_dot_k1 / 2.0
    beta_term  = 1.0 - n_dot_k2 / 2.0
    rho_diff = (28.0 * r_d / (23.0 * np.pi)) * (1.0 - r_s) * \
               (1.0 - alpha_term**5) * (1.0 - beta_term**5)

    diffuse_flux = rho_diff * PLATE_AREA * n_dot_k1 * n_dot_k2
    diffuse_flux = np.maximum(diffuse_flux, 0.0)

    specular_flux = flux_total - diffuse_flux
    specular_flux = np.maximum(specular_flux, 0.0)

    # Find where specular drops below diffuse
    spec_dominant = specular_flux > diffuse_flux
    transitions = np.where(spec_dominant[:-1] & ~spec_dominant[1:])[0]

    if len(transitions) == 0:
        # Check if specular is always dominant or never dominant
        if spec_dominant[0]:
            return THETA_DEG[-1]  # wider than sweep
        else:
            return 0.0  # specular never dominates

    idx = transitions[0]
    # Linear interpolation
    s_above = specular_flux[idx] - diffuse_flux[idx]
    s_below = specular_flux[idx + 1] - diffuse_flux[idx + 1]
    frac = s_above / (s_above - s_below) if (s_above - s_below) != 0 else 0.5
    theta_cross = THETA_DEG[idx] + frac * (THETA_DEG[idx + 1] - THETA_DEG[idx])

    return theta_cross


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    results = {"experiment": "m037", "description": "BRDF specular glint profile characterization"}

    # -------------------------------------------------------------------
    # Part A: Flux profile vs n_phong
    # -------------------------------------------------------------------
    print("=== Part A: Magnitude vs theta, varying n_phong ===")
    n_phong_values = [50, 100, 200, 300, 500, 1000]
    r_s_A, r_d_A, phase_A = 0.40, 0.30, 20.0

    part_a_mags = {}
    part_a_flux = {}
    for np_val in n_phong_values:
        mags, flux = compute_magnitude_profile(r_d_A, r_s_A, np_val, phase_A)
        part_a_mags[np_val] = mags
        part_a_flux[np_val] = flux
        peak_mag = mags[0]
        print(f"  n_phong={np_val:4d}: peak mag = {peak_mag:.2f}")

    # -------------------------------------------------------------------
    # Part B: Flux profile vs r_s
    # -------------------------------------------------------------------
    print("\n=== Part B: Magnitude vs theta, varying r_s ===")
    r_s_values = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    n_phong_B, r_d_B, phase_B = 200, 0.30, 20.0

    part_b_mags = {}
    part_b_flux = {}
    for rs_val in r_s_values:
        mags, flux = compute_magnitude_profile(r_d_B, rs_val, n_phong_B, phase_B)
        part_b_mags[rs_val] = mags
        part_b_flux[rs_val] = flux
        peak_mag = mags[0]
        print(f"  r_s={rs_val:.2f}: peak mag = {peak_mag:.2f}")

    # -------------------------------------------------------------------
    # Part C: Flux profile vs phase angle
    # -------------------------------------------------------------------
    print("\n=== Part C: Magnitude vs theta, varying phase angle ===")
    phase_values = [5, 10, 20, 30, 45, 60]
    n_phong_C, r_s_C, r_d_C = 200, 0.40, 0.30

    part_c_mags = {}
    part_c_flux = {}
    for ph_val in phase_values:
        mags, flux = compute_magnitude_profile(r_d_C, r_s_C, n_phong_C, ph_val)
        part_c_mags[ph_val] = mags
        part_c_flux[ph_val] = flux
        peak_mag = mags[0]
        print(f"  phase={ph_val:2d} deg: peak mag = {peak_mag:.2f}")

    # -------------------------------------------------------------------
    # Part D: Lobe width extraction (FWHM + specular=diffuse crossing)
    # -------------------------------------------------------------------
    print("\n=== Part D: Lobe width extraction ===")

    # D.1 FWHM vs n_phong
    print("  FWHM vs n_phong:")
    fwhm_vs_nphong = {}
    spec_eq_diff_vs_nphong = {}
    for np_val in n_phong_values:
        fwhm = extract_fwhm(part_a_flux[np_val])
        sed = extract_specular_equals_diffuse_angle(r_d_A, r_s_A, np_val, phase_A)
        fwhm_vs_nphong[np_val] = fwhm
        spec_eq_diff_vs_nphong[np_val] = sed
        print(f"    n_phong={np_val:4d}: FWHM = {fwhm:.2f} deg, spec=diff @ {sed:.2f} deg")

    # D.2 FWHM vs r_s
    print("  FWHM vs r_s:")
    fwhm_vs_rs = {}
    spec_eq_diff_vs_rs = {}
    for rs_val in r_s_values:
        fwhm = extract_fwhm(part_b_flux[rs_val])
        sed = extract_specular_equals_diffuse_angle(r_d_B, rs_val, n_phong_B, phase_B)
        fwhm_vs_rs[rs_val] = fwhm
        spec_eq_diff_vs_rs[rs_val] = sed
        print(f"    r_s={rs_val:.2f}: FWHM = {fwhm:.2f} deg, spec=diff @ {sed:.2f} deg")

    # D.3 FWHM vs phase angle
    print("  FWHM vs phase angle:")
    fwhm_vs_phase = {}
    spec_eq_diff_vs_phase = {}
    for ph_val in phase_values:
        fwhm = extract_fwhm(part_c_flux[ph_val])
        sed = extract_specular_equals_diffuse_angle(r_d_C, r_s_C, n_phong_C, ph_val)
        fwhm_vs_phase[ph_val] = fwhm
        spec_eq_diff_vs_phase[ph_val] = sed
        print(f"    phase={ph_val:2d} deg: FWHM = {fwhm:.2f} deg, spec=diff @ {sed:.2f} deg")

    # -------------------------------------------------------------------
    # Part E: Area scaling verification
    # -------------------------------------------------------------------
    print("\n=== Part E: Area scaling verification ===")
    areas = [0.1, 1.0, 10.0, 100.0]
    n_phong_E, r_s_E, r_d_E, phase_E = 200, 0.40, 0.30, 20.0
    theta_E = 0  # perfect alignment

    area_mags = {}
    area_fluxes = {}
    for a in areas:
        mags, flux = compute_magnitude_profile(r_d_E, r_s_E, n_phong_E, phase_E, area=a)
        area_mags[a] = float(mags[theta_E])
        area_fluxes[a] = float(flux[theta_E])
        print(f"  area={a:6.1f} m^2: flux={flux[theta_E]:.6e}, mag={mags[theta_E]:.2f}")

    # Check linearity
    ref_flux = area_fluxes[1.0]
    print("  Linearity check (flux / area should be constant):")
    for a in areas:
        ratio = area_fluxes[a] / (a * ref_flux)
        print(f"    area={a:6.1f}: flux/(area*ref) = {ratio:.6f}")

    # Check magnitude shift
    ref_mag = area_mags[1.0]
    print("  Magnitude shift check (delta_m should be -2.5*log10(area_ratio)):")
    for a in areas:
        delta_m_actual = area_mags[a] - ref_mag
        delta_m_expected = -2.5 * np.log10(a / 1.0) if a > 0 else np.nan
        print(f"    area={a:6.1f}: delta_m = {delta_m_actual:+.4f} (expected {delta_m_expected:+.4f})")

    # -------------------------------------------------------------------
    # Save results JSON
    # -------------------------------------------------------------------
    results["part_a"] = {
        "description": "Magnitude vs theta, varying n_phong",
        "fixed": {"r_s": r_s_A, "r_d": r_d_A, "phase_angle_deg": phase_A},
        "n_phong_values": n_phong_values,
        "peak_magnitudes": {str(k): float(part_a_mags[k][0]) for k in n_phong_values},
    }
    results["part_b"] = {
        "description": "Magnitude vs theta, varying r_s",
        "fixed": {"n_phong": n_phong_B, "r_d": r_d_B, "phase_angle_deg": phase_B},
        "r_s_values": r_s_values,
        "peak_magnitudes": {str(k): float(part_b_mags[k][0]) for k in r_s_values},
    }
    results["part_c"] = {
        "description": "Magnitude vs theta, varying phase angle",
        "fixed": {"n_phong": n_phong_C, "r_s": r_s_C, "r_d": r_d_C},
        "phase_values": phase_values,
        "peak_magnitudes": {str(k): float(part_c_mags[k][0]) for k in phase_values},
    }
    results["part_d"] = {
        "description": "Lobe width extraction",
        "fwhm_vs_nphong": {str(k): float(v) for k, v in fwhm_vs_nphong.items()},
        "fwhm_vs_rs": {str(k): float(v) for k, v in fwhm_vs_rs.items()},
        "fwhm_vs_phase": {str(k): float(v) for k, v in fwhm_vs_phase.items()},
        "spec_eq_diff_vs_nphong": {str(k): float(v) for k, v in spec_eq_diff_vs_nphong.items()},
        "spec_eq_diff_vs_rs": {str(k): float(v) for k, v in spec_eq_diff_vs_rs.items()},
        "spec_eq_diff_vs_phase": {str(k): float(v) for k, v in spec_eq_diff_vs_phase.items()},
    }
    results["part_e"] = {
        "description": "Area scaling verification",
        "areas": areas,
        "magnitudes": {str(k): v for k, v in area_mags.items()},
        "fluxes": {str(k): v for k, v in area_fluxes.items()},
        "linearity_passed": all(
            abs(area_fluxes[a] / (a * ref_flux) - 1.0) < 1e-10 for a in areas
        ),
    }
    results["runtime_seconds"] = time.time() - t0

    save_results(str(RESULTS_JSON), results)
    print(f"\nResults saved to {RESULTS_JSON}")

    # -------------------------------------------------------------------
    # Plot: 2x2 figure
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Micro-37: BRDF Specular Glint Profile Characterization", fontsize=14, fontweight='bold')

    # --- Panel A: magnitude vs theta, varying n_phong ---
    ax = axes[0, 0]
    for np_val in n_phong_values:
        mags = part_a_mags[np_val]
        finite_mask = np.isfinite(mags)
        ax.plot(THETA_DEG[finite_mask], mags[finite_mask], label=f"n={np_val}")
    ax.set_xlabel("Misalignment angle (deg)")
    ax.set_ylabel("Apparent magnitude")
    ax.set_title(f"(A) Varying n_phong  (r_s={r_s_A}, r_d={r_d_A}, phase={phase_A} deg)")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)

    # --- Panel B: magnitude vs theta, varying r_s ---
    ax = axes[0, 1]
    for rs_val in r_s_values:
        mags = part_b_mags[rs_val]
        finite_mask = np.isfinite(mags)
        ax.plot(THETA_DEG[finite_mask], mags[finite_mask], label=f"r_s={rs_val:.2f}")
    ax.set_xlabel("Misalignment angle (deg)")
    ax.set_ylabel("Apparent magnitude")
    ax.set_title(f"(B) Varying r_s  (n_phong={n_phong_B}, r_d={r_d_B}, phase={phase_B} deg)")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)

    # --- Panel C: magnitude vs theta, varying phase angle ---
    ax = axes[1, 0]
    for ph_val in phase_values:
        mags = part_c_mags[ph_val]
        finite_mask = np.isfinite(mags)
        ax.plot(THETA_DEG[finite_mask], mags[finite_mask], label=f"phase={ph_val} deg")
    ax.set_xlabel("Misalignment angle (deg)")
    ax.set_ylabel("Apparent magnitude")
    ax.set_title(f"(C) Varying phase angle  (n_phong={n_phong_C}, r_s={r_s_C}, r_d={r_d_C})")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)

    # --- Panel D: FWHM summary ---
    ax = axes[1, 1]

    # Three lines on the same axes with different x-scales
    # Use twin axes for clarity
    color_nphong = '#1f77b4'
    color_rs = '#ff7f0e'
    color_phase = '#2ca02c'

    # Plot FWHM vs n_phong
    nphong_list = sorted(fwhm_vs_nphong.keys())
    fwhm_nphong_vals = [fwhm_vs_nphong[k] for k in nphong_list]
    ax.plot(nphong_list, fwhm_nphong_vals, 'o-', color=color_nphong, label='FWHM vs n_phong')

    # Also plot spec=diff crossing angle
    sed_nphong_vals = [spec_eq_diff_vs_nphong[k] for k in nphong_list]
    ax.plot(nphong_list, sed_nphong_vals, 's--', color=color_nphong, alpha=0.5,
            label='spec=diff vs n_phong')

    ax.set_xlabel("n_phong", color=color_nphong)
    ax.set_ylabel("Angular width (deg)")
    ax.set_title("(D) Lobe widths")
    ax.tick_params(axis='x', labelcolor=color_nphong)
    ax.grid(True, alpha=0.3)

    # Secondary x-axis for r_s
    ax2 = ax.twiny()
    rs_list = sorted(fwhm_vs_rs.keys())
    fwhm_rs_vals = [fwhm_vs_rs[k] for k in rs_list]
    ax2.plot(rs_list, fwhm_rs_vals, 'o-', color=color_rs, label='FWHM vs r_s')
    sed_rs_vals = [spec_eq_diff_vs_rs[k] for k in rs_list]
    ax2.plot(rs_list, sed_rs_vals, 's--', color=color_rs, alpha=0.5, label='spec=diff vs r_s')
    ax2.set_xlabel("r_s", color=color_rs)
    ax2.tick_params(axis='x', labelcolor=color_rs)

    # Tertiary x-axis for phase angle
    ax3 = ax.twiny()
    ax3.spines['top'].set_position(('outward', 40))
    phase_list = sorted(fwhm_vs_phase.keys())
    fwhm_phase_vals = [fwhm_vs_phase[k] for k in phase_list]
    ax3.plot(phase_list, fwhm_phase_vals, 'o-', color=color_phase, label='FWHM vs phase')
    sed_phase_vals = [spec_eq_diff_vs_phase[k] for k in phase_list]
    ax3.plot(phase_list, sed_phase_vals, 's--', color=color_phase, alpha=0.5,
             label='spec=diff vs phase')
    ax3.set_xlabel("Phase angle (deg)", color=color_phase)
    ax3.tick_params(axis='x', labelcolor=color_phase)

    # Combined legend
    lines_labels = []
    for a in [ax, ax2, ax3]:
        h, l = a.get_legend_handles_labels()
        lines_labels.extend(zip(h, l))
    handles, labels = zip(*lines_labels)
    ax.legend(handles, labels, fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(str(RESULTS_PNG), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {RESULTS_PNG}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Micro-37 complete in {elapsed:.2f}s")
    print(f"{'='*60}")

    print("\nKey findings:")
    print(f"  Part A (n_phong effect on FWHM):")
    for np_val in n_phong_values:
        print(f"    n_phong={np_val:4d}: FWHM={fwhm_vs_nphong[np_val]:.2f} deg, "
              f"spec=diff @ {spec_eq_diff_vs_nphong[np_val]:.2f} deg")

    print(f"  Part B (r_s effect on FWHM):")
    for rs_val in r_s_values:
        print(f"    r_s={rs_val:.2f}: FWHM={fwhm_vs_rs[rs_val]:.2f} deg, "
              f"spec=diff @ {spec_eq_diff_vs_rs[rs_val]:.2f} deg")

    print(f"  Part C (phase angle effect on FWHM):")
    for ph_val in phase_values:
        print(f"    phase={ph_val:2d} deg: FWHM={fwhm_vs_phase[ph_val]:.2f} deg, "
              f"spec=diff @ {spec_eq_diff_vs_phase[ph_val]:.2f} deg")

    print(f"  Part E: Area scaling linearity = {results['part_e']['linearity_passed']}")


if __name__ == "__main__":
    main()
