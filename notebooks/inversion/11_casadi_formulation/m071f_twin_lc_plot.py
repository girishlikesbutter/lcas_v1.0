#!/usr/bin/env python3
"""
m071f — Plot true vs phi+180 twin LCs.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

data = np.load(str(RESULTS_DIR / "m071e_correct_twin.npz"))
obs_times = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                     allow_pickle=True)['observation_times']

lc_true = data['lc_true']
lc_twin = data['lc_twin']
diff = data['diff']

fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                          gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

axes[0].plot(obs_times, lc_true, 'g-', alpha=0.7, lw=0.9, label='True')
axes[0].plot(obs_times, lc_twin, 'r--', alpha=0.7, lw=0.9, label='Phi+180 twin')
axes[0].set_ylabel('Apparent Magnitude')
axes[0].set_title(f'True vs phi+180 twin | RMS = {np.sqrt(np.nanmean(diff**2)):.2f} mag')
axes[0].legend(fontsize=9)
axes[0].invert_yaxis()

valid = np.isfinite(diff)
axes[1].plot(obs_times[valid], diff[valid], 'b-', alpha=0.5, lw=0.6)
axes[1].axhline(0, color='k', lw=0.5, ls='--')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('True − Twin (mag)')
axes[1].set_ylim(-10, 10)

plt.tight_layout()
out = RESULTS_DIR / "m071f_twin_lc_plot.png"
plt.savefig(str(out), dpi=150)
print(f"Saved: {out}")
