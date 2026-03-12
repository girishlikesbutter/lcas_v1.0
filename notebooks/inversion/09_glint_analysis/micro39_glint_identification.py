#!/usr/bin/env python3
"""Micro-39 -- Blind glint component identification from LC shape.

Question: Can we identify which normal group is responsible for a glint purely
from the observed lightcurve shape, without knowing the attitude?

Method:
  1. Detect all bright peaks (mag < 9) from the true (hi-fi) LC.
  2. Extract feature vectors from the OBSERVED (noisy) LC around each peak:
     peak magnitude, FWHM, rise slope, decay slope, recurrence interval.
  3. Apply hierarchical clustering on standardised features (k=3,4,5).
  4. Compare blind clusters to oracle labels (dominant normal group from micro34)
     using adjusted Rand score and confusion matrix.
  5. Also test a simple rule-based classifier on peak magnitude thresholds.
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
from scipy.signal import argrelmin
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import comb

from lib.experiment_setup import setup_experiment, save_results


# ---------------------------------------------------------------------------
# Lightweight implementations of adjusted_rand_score and confusion_matrix
# to avoid a scikit-learn dependency.
# ---------------------------------------------------------------------------

def confusion_matrix(labels_true, labels_pred):
    """Build a confusion matrix (rows=true, cols=pred).

    Returns a 2-D numpy array of shape (n_true_classes, n_pred_classes).
    Classes are sorted in ascending order.
    """
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    true_map = {c: i for i, c in enumerate(classes_true)}
    pred_map = {c: i for i, c in enumerate(classes_pred)}
    cm = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        cm[true_map[t], pred_map[p]] += 1
    return cm


def adjusted_rand_score(labels_true, labels_pred):
    """Compute the Adjusted Rand Index between two clusterings.

    Implements the standard formula using the contingency table.
    Returns a float in [-1, 1]; 1 = perfect agreement, 0 = random.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n < 2:
        return 0.0

    # Build contingency table
    cm = confusion_matrix(labels_true, labels_pred)

    # Sum of C(n_ij, 2) over all cells
    sum_comb_ij = sum(comb(int(v), 2) for v in cm.ravel())

    # Row sums and column sums
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    sum_comb_rows = sum(comb(int(a), 2) for a in row_sums)
    sum_comb_cols = sum(comb(int(b), 2) for b in col_sums)

    total_comb = comb(n, 2)
    if total_comb == 0:
        return 0.0

    expected = sum_comb_rows * sum_comb_cols / total_comb
    max_index = (sum_comb_rows + sum_comb_cols) / 2.0
    denom = max_index - expected

    if abs(denom) < 1e-15:
        return 0.0 if abs(sum_comb_ij - expected) < 1e-15 else 1.0

    return (sum_comb_ij - expected) / denom

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

LOG_PATH = RESULTS_DIR / "micro39_stdout.txt"
_log_file = open(LOG_PATH, 'w')
sys.stdout = Tee(sys.__stdout__, _log_file)


# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("micro39 -- Blind glint component identification from LC shape")
print("=" * 70)

t_global = time.time()

CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)

n_obs = CTX.n_observations
dt_sampling = CTX.dt_sampling
print(f"Setup complete: {n_obs} observations, dt_sampling = {dt_sampling:.2f}s")


# ===========================================================================
# Load micro34 data for oracle labels
# ===========================================================================
print("\n--- Loading micro34 oracle data ---")

micro34_npz = np.load(str(RESULTS_DIR / "micro34_pab_alignment.npz"))
true_lc = micro34_npz['true_lc']            # (500,) hi-fi noiseless LC
frac_flux = micro34_npz['frac_flux']         # (14, 500) fractional flux per group per epoch
unique_normals = micro34_npz['unique_normals']  # (14, 3)

with open(RESULTS_DIR / "micro34_pab_alignment.json") as f:
    micro34_json = json.load(f)
group_info = micro34_json['all_groups']  # list of dicts with 'group_id', 'components', 'total_area', etc.

print(f"  Loaded true_lc: shape={true_lc.shape}")
print(f"  Loaded frac_flux: shape={frac_flux.shape}")
print(f"  Number of normal groups: {len(group_info)}")


# ===========================================================================
# Verify true_lc matches CTX.true_lc
# ===========================================================================
lc_diff = np.max(np.abs(true_lc - CTX.true_lc))
print(f"  Max |micro34_true_lc - CTX.true_lc| = {lc_diff:.6f}")
if lc_diff > 0.01:
    print("  WARNING: LCs differ significantly — oracle labels may not match current trajectory!")


# ===========================================================================
# Detect bright peaks from the true (noiseless) LC
# ===========================================================================
print("\n--- Peak detection (from noiseless true LC) ---")

# LC convention: magnitudes, so bright peaks are LOCAL MINIMA
peak_indices = argrelmin(true_lc, order=5)[0]
print(f"Total peaks detected (order=5): {len(peak_indices)}")

# Filter to bright peaks only (mag < 9)
bright_mask = true_lc[peak_indices] < 9.0
bright_peaks = peak_indices[bright_mask]
print(f"Bright peaks (mag < 9): {len(bright_peaks)}")

# Print bright peak summary
print(f"\nBright peak epochs and magnitudes:")
for i, pidx in enumerate(bright_peaks):
    print(f"  [{i:2d}] epoch={pidx:4d}, true_mag={true_lc[pidx]:.3f}, "
          f"observed_mag={CTX.observed_lc[pidx]:.3f}")


# ===========================================================================
# Oracle labels: dominant normal group at each bright peak
# ===========================================================================
print("\n--- Oracle labels (from micro34 fractional flux) ---")

oracle_labels = []
for pidx in bright_peaks:
    dominant_group = int(np.argmax(frac_flux[:, pidx]))
    oracle_labels.append(dominant_group)
oracle_labels = np.array(oracle_labels)

# Summarise oracle label distribution
unique_oracle_labels, oracle_counts = np.unique(oracle_labels, return_counts=True)
print(f"Oracle label distribution:")
for label, count in zip(unique_oracle_labels, oracle_counts):
    info = group_info[label]
    print(f"  Group {label:2d}: {count:2d} peaks, "
          f"components={info['components']}, "
          f"area={info['total_area']:.2f}, "
          f"normal={info['normal']}")


# ===========================================================================
# Feature extraction from the OBSERVED (noisy) LC
# ===========================================================================
print("\n--- Feature extraction from observed (noisy) LC ---")

observed_lc = CTX.observed_lc  # (500,) with noise sigma=0.05


def extract_glint_features(peak_idx, lc, dt_samp):
    """Extract feature vector for one glint peak.

    Parameters
    ----------
    peak_idx : int
        Index of the peak in the lightcurve array.
    lc : ndarray (N,)
        The lightcurve (magnitudes).
    dt_samp : float
        Sampling interval in seconds.

    Returns
    -------
    dict
        Feature vector with keys: peak_mag, fwhm_epochs, fwhm_seconds,
        rise_slope, decay_slope.
    """
    n = len(lc)

    # Peak magnitude (min in +/-2 window to handle noise)
    lo = max(0, peak_idx - 2)
    hi = min(n, peak_idx + 3)
    peak_mag = float(np.min(lc[lo:hi]))

    # FWHM: find half-max level in +/-15 epoch window
    window_lo = max(0, peak_idx - 15)
    window_hi = min(n, peak_idx + 16)
    window = lc[window_lo:window_hi]
    peak_val = np.min(window)
    # "half max" in magnitude space: midpoint between peak and baseline
    baseline = np.max(window)  # dimmest point in window
    half_level = (peak_val + baseline) / 2.0
    # Count epochs below half_level
    below_half = np.where(window < half_level)[0]
    if len(below_half) > 1:
        fwhm_epochs = float(below_half[-1] - below_half[0])
    else:
        fwhm_epochs = 1.0
    fwhm_seconds = fwhm_epochs * dt_samp

    # Rise and decay slopes (mag/epoch in +/-5 epoch window)
    rise_lo = max(0, peak_idx - 5)
    rise_slope = (lc[peak_idx] - lc[rise_lo]) / max(1, peak_idx - rise_lo)

    decay_hi = min(n - 1, peak_idx + 5)
    decay_slope = (lc[decay_hi] - lc[peak_idx]) / max(1, decay_hi - peak_idx)

    return {
        'peak_mag': peak_mag,
        'fwhm_epochs': fwhm_epochs,
        'fwhm_seconds': fwhm_seconds,
        'rise_slope': float(rise_slope),
        'decay_slope': float(decay_slope),
    }


# Extract features for all bright peaks
feature_dicts = []
for pidx in bright_peaks:
    feat = extract_glint_features(pidx, observed_lc, dt_sampling)
    feature_dicts.append(feat)

# Add recurrence (time since previous bright peak, 0 for first)
for i in range(len(feature_dicts)):
    if i == 0:
        feature_dicts[i]['recurrence_epochs'] = 0.0
        feature_dicts[i]['recurrence_seconds'] = 0.0
    else:
        delta = float(bright_peaks[i] - bright_peaks[i - 1])
        feature_dicts[i]['recurrence_epochs'] = delta
        feature_dicts[i]['recurrence_seconds'] = delta * dt_sampling

# Print feature table
feature_names = ['peak_mag', 'fwhm_epochs', 'rise_slope', 'decay_slope', 'recurrence_epochs']
print(f"\nFeature table ({len(feature_dicts)} bright peaks):")
header = f"{'idx':>4s}  {'epoch':>5s}  {'oracle':>6s}  "
header += "  ".join(f"{fn:>14s}" for fn in feature_names)
print(header)
print("-" * len(header))
for i, feat in enumerate(feature_dicts):
    row = f"{i:4d}  {bright_peaks[i]:5d}  G{oracle_labels[i]:4d}  "
    row += "  ".join(f"{feat[fn]:14.4f}" for fn in feature_names)
    print(row)


# ===========================================================================
# Build feature matrix and standardise
# ===========================================================================
print("\n--- Building feature matrix ---")

n_peaks = len(feature_dicts)
feature_matrix = np.zeros((n_peaks, len(feature_names)))
for i, feat in enumerate(feature_dicts):
    for j, fn in enumerate(feature_names):
        feature_matrix[i, j] = feat[fn]

print(f"Feature matrix shape: {feature_matrix.shape}")

# Standardise (zero mean, unit variance per feature)
feat_mean = feature_matrix.mean(axis=0)
feat_std = feature_matrix.std(axis=0)
# Protect against zero-std features
feat_std[feat_std < 1e-10] = 1.0
feature_matrix_std = (feature_matrix - feat_mean) / feat_std

print(f"Feature means:  {feat_mean}")
print(f"Feature stds:   {feat_std}")


# ===========================================================================
# Hierarchical clustering (Ward linkage)
# ===========================================================================
print("\n--- Hierarchical clustering ---")

if n_peaks < 2:
    print("ERROR: fewer than 2 bright peaks — cannot cluster.")
    sys.exit(1)

Z = linkage(feature_matrix_std, method='ward', metric='euclidean')

k_values = [3, 4, 5]
cluster_results = {}

for k in k_values:
    cluster_labels = fcluster(Z, t=k, criterion='maxclust')
    # fcluster returns 1-based labels; convert to 0-based
    cluster_labels_0 = cluster_labels - 1

    ari = adjusted_rand_score(oracle_labels, cluster_labels_0)

    # Confusion matrix: rows = oracle groups, cols = clusters
    cm = confusion_matrix(oracle_labels, cluster_labels_0)

    cluster_results[k] = {
        'cluster_labels': cluster_labels_0,
        'ari': float(ari),
        'confusion_matrix': cm,
    }

    print(f"\n  k={k}: Adjusted Rand Index = {ari:.4f}")
    print(f"  Confusion matrix (rows=oracle, cols=cluster):")
    # Print with group labels
    oracle_groups_in_cm = sorted(set(oracle_labels))
    cluster_ids_in_cm = sorted(set(cluster_labels_0))
    print(f"    {'':>10s}  " + "  ".join(f"C{c:d}" for c in range(k)))
    for row_idx, og in enumerate(oracle_groups_in_cm):
        if row_idx < cm.shape[0]:
            row_str = "  ".join(f"{cm[row_idx, c]:3d}" for c in range(min(k, cm.shape[1])))
            info = group_info[og]
            print(f"    G{og:2d} ({','.join(info['components'])[:15]:>15s}): {row_str}")


# ===========================================================================
# Pick the best k (highest ARI)
# ===========================================================================
best_k = max(cluster_results, key=lambda k: cluster_results[k]['ari'])
best_ari = cluster_results[best_k]['ari']
best_cluster_labels = cluster_results[best_k]['cluster_labels']

print(f"\nBest k = {best_k} with ARI = {best_ari:.4f}")


# ===========================================================================
# Rule-based classifier
# ===========================================================================
print("\n--- Rule-based classifier ---")

rule_labels = np.zeros(n_peaks, dtype=int)
for i, feat in enumerate(feature_dicts):
    mag = feat['peak_mag']
    if mag < 7.5:
        rule_labels[i] = 0  # z-faces (large area)
    elif mag < 8.5:
        rule_labels[i] = 1  # antenna dishes
    else:
        rule_labels[i] = 2  # bus/other

rule_ari = adjusted_rand_score(oracle_labels, rule_labels)
rule_cm = confusion_matrix(oracle_labels, rule_labels)

print(f"Rule-based ARI = {rule_ari:.4f}")
print(f"Rule-based confusion matrix (rows=oracle, cols=rule):")
oracle_groups_for_rule = sorted(set(oracle_labels))
print(f"  {'':>10s}  mag<7.5  7.5-8.5  8.5-9.0")
for row_idx, og in enumerate(oracle_groups_for_rule):
    if row_idx < rule_cm.shape[0]:
        ncols = min(3, rule_cm.shape[1])
        row_str = "  ".join(f"{rule_cm[row_idx, c]:6d}" for c in range(ncols))
        info = group_info[og]
        print(f"  G{og:2d} ({','.join(info['components'])[:15]:>15s}): {row_str}")

# Per-peak classification table
print(f"\nPer-peak classification comparison:")
print(f"  {'idx':>4s}  {'epoch':>5s}  {'mag':>8s}  {'oracle':>8s}  {'cluster':>8s}  {'rule':>8s}")
for i in range(n_peaks):
    print(f"  {i:4d}  {bright_peaks[i]:5d}  {feature_dicts[i]['peak_mag']:8.3f}  "
          f"G{oracle_labels[i]:5d}  C{best_cluster_labels[i]:5d}  R{rule_labels[i]:5d}")


# ===========================================================================
# Plot: 2-panel side-by-side scatter
# ===========================================================================
print("\n--- Generating plot ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Micro-39: Blind Glint Component Identification from LC Shape",
             fontsize=14, fontweight='bold')

# Colour maps
cmap_cluster = plt.cm.tab10
cmap_oracle = plt.cm.tab10

# --- Panel 1: peaks coloured by blind cluster assignment (best k) ---
unique_clusters = sorted(set(best_cluster_labels))
for c in unique_clusters:
    mask = best_cluster_labels == c
    ax1.scatter(bright_peaks[mask], feature_matrix[mask, 0],
                color=cmap_cluster(c % 10), s=60,
                edgecolors='black', linewidth=0.5, zorder=5,
                label=f"Cluster {c} (n={mask.sum()})")

ax1.invert_yaxis()  # brighter = up
ax1.set_xlabel("Peak epoch index")
ax1.set_ylabel("Peak magnitude (brighter up)")
ax1.set_title(f"Panel 1: Blind clustering (k={best_k}, ARI={best_ari:.3f})")
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- Panel 2: peaks coloured by oracle group ---
for og in sorted(set(oracle_labels)):
    mask = oracle_labels == og
    info = group_info[og]
    label_str = f"G{og} {','.join(info['components'][:2])} (n={mask.sum()})"
    ax2.scatter(bright_peaks[mask], feature_matrix[mask, 0],
                color=cmap_oracle(og % 10), s=60,
                edgecolors='black', linewidth=0.5, zorder=5,
                label=label_str)

ax2.invert_yaxis()  # brighter = up
ax2.set_xlabel("Peak epoch index")
ax2.set_ylabel("Peak magnitude (brighter up)")
ax2.set_title("Panel 2: Oracle grouping (micro34 dominant normal group)")
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = RESULTS_DIR / "micro39_glint_identification.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save JSON results
# ===========================================================================
print("\n--- Saving results ---")

results = {
    'experiment': 'micro39_glint_identification',
    'description': 'Blind glint component identification from LC shape',
    'n_observations': n_obs,
    'dt_sampling_s': float(dt_sampling),
    'noise_sigma': CTX.noise_sigma,
    'n_bright_peaks': int(n_peaks),
    'bright_peak_epochs': bright_peaks.tolist(),
    'oracle_labels': oracle_labels.tolist(),
    'oracle_label_distribution': {
        str(int(label)): {
            'count': int(count),
            'components': group_info[int(label)]['components'],
            'area': group_info[int(label)]['total_area'],
            'normal': group_info[int(label)]['normal'],
        }
        for label, count in zip(unique_oracle_labels, oracle_counts)
    },
    'features': [
        {
            'epoch': int(bright_peaks[i]),
            'oracle_group': int(oracle_labels[i]),
            **feature_dicts[i],
        }
        for i in range(n_peaks)
    ],
    'feature_names': feature_names,
    'feature_means': feat_mean.tolist(),
    'feature_stds': feat_std.tolist(),
    'clustering': {},
    'rule_based': {
        'ari': float(rule_ari),
        'confusion_matrix': rule_cm.tolist(),
        'thresholds': {'bright': 7.5, 'medium': 8.5, 'dim': 9.0},
        'labels': rule_labels.tolist(),
    },
    'best_k': int(best_k),
    'best_ari': float(best_ari),
}

# Add clustering results for each k
for k in k_values:
    cr = cluster_results[k]
    results['clustering'][str(k)] = {
        'ari': cr['ari'],
        'confusion_matrix': cr['confusion_matrix'].tolist(),
        'cluster_labels': cr['cluster_labels'].tolist(),
    }

json_path = RESULTS_DIR / "micro39_glint_identification.json"
save_results(str(json_path), results)
print(f"JSON saved: {json_path}")


# ===========================================================================
# Summary
# ===========================================================================
elapsed = time.time() - t_global
print(f"\n{'=' * 70}")
print(f"Micro-39 complete in {elapsed:.1f}s")
print(f"{'=' * 70}")

print(f"\nKey findings:")
print(f"  Bright peaks detected (mag < 9): {n_peaks}")
print(f"  Oracle groups present: {len(unique_oracle_labels)} distinct groups")
for label, count in zip(unique_oracle_labels, oracle_counts):
    info = group_info[int(label)]
    print(f"    G{label}: {count} peaks, components={info['components']}")

print(f"\n  Hierarchical clustering results:")
for k in k_values:
    cr = cluster_results[k]
    print(f"    k={k}: ARI = {cr['ari']:.4f}")
print(f"  Best: k={best_k}, ARI={best_ari:.4f}")

print(f"\n  Rule-based classifier: ARI = {rule_ari:.4f}")

print(f"\n  Interpretation:")
if best_ari > 0.5:
    print(f"    Clustering recovers oracle grouping well (ARI > 0.5).")
    print(f"    LC shape alone can identify the source normal group.")
elif best_ari > 0.2:
    print(f"    Partial agreement (0.2 < ARI < 0.5): shape captures some structure.")
else:
    print(f"    Poor agreement (ARI < 0.2): shape features alone are insufficient")
    print(f"    to identify the source normal group without attitude knowledge.")

if rule_ari > best_ari:
    print(f"    Rule-based classifier outperforms hierarchical clustering.")
elif rule_ari < best_ari:
    print(f"    Hierarchical clustering outperforms simple rule-based classifier.")
else:
    print(f"    Both methods perform similarly.")

# Restore stdout before closing log to avoid flush-after-close error
sys.stdout = sys.__stdout__
_log_file.close()
