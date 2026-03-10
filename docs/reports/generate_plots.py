"""Generate all plots for the Series 07 findings report."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
DATA = Path(__file__).parents[2] / "data" / "results" / "inversion_diagnostics"

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})

# ── Colour palette ──────────────────────────────────────────────────────────
C_TRUE = '#2ca02c'      # green for truth
C_WRONG = '#d62728'     # red for wrong/dead-end
C_GOOD = '#1f77b4'      # blue for positive findings
C_NEUTRAL = '#7f7f7f'   # grey
C_HIGHLIGHT = '#ff7f0e'  # orange for highlights


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Winding Landscape (micro19) — arrival error vs |ω| for both legs
# ═══════════════════════════════════════════════════════════════════════════════
def plot_winding_landscape():
    with open(DATA / "micro19_winding_landscape.json") as f:
        d = json.load(f)

    mags = np.linspace(d['mag_range_degs'][0], d['mag_range_degs'][1], d['n_bins'])
    leg0 = np.array(d['leg0_errs'])
    leg1 = np.array(d['leg1_errs'])
    true_mag = d['true_omega_mag_degs']
    threshold = d['valley_threshold']

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Leg 0
    ax0.semilogy(mags, leg0, color=C_GOOD, lw=0.8, alpha=0.8)
    ax0.axhline(threshold, color=C_NEUTRAL, ls='--', lw=0.8, label=f'threshold = {threshold}')
    ax0.axvline(true_mag, color=C_TRUE, ls='-', lw=2, alpha=0.7, label=f'true |ω| = {true_mag:.2f} deg/s')
    # Mark valleys
    valley_mask0 = leg0 < threshold
    ax0.fill_between(mags, 1e-12, leg0, where=valley_mask0, alpha=0.15, color=C_GOOD)
    ax0.set_ylabel('Arrival error')
    ax0.set_title(f'Leg 0 — peaks 183→260, dt = {d["dt_leg0_s"]:.0f}s  (10 valleys)')
    ax0.set_ylim(1e-7, 2)
    ax0.legend(loc='upper right')

    # Leg 1
    ax1.semilogy(mags, leg1, color=C_HIGHLIGHT, lw=0.8, alpha=0.8)
    ax1.axhline(threshold, color=C_NEUTRAL, ls='--', lw=0.8, label=f'threshold = {threshold}')
    ax1.axvline(true_mag, color=C_TRUE, ls='-', lw=2, alpha=0.7, label=f'true |ω| = {true_mag:.2f} deg/s')
    valley_mask1 = leg1 < threshold
    ax1.fill_between(mags, 1e-12, leg1, where=valley_mask1, alpha=0.15, color=C_HIGHLIGHT)
    ax1.set_ylabel('Arrival error')
    ax1.set_xlabel('|ω| (deg/s)')
    ax1.set_title(f'Leg 1 — peaks 260→360, dt = {d["dt_leg1_s"]:.0f}s  (6 valleys, wider)')
    ax1.set_ylim(1e-7, 2)
    ax1.legend(loc='upper right')

    fig.suptitle('micro19: Winding Landscape — Valid ω Solutions Exist at Multiple Magnitudes',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ASSETS / "01_winding_landscape.png", bbox_inches='tight')
    plt.close(fig)
    print("  [1/7] winding_landscape")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Staircase gap on leg 1 — micro20 band-sweep vs original staircase
# ═══════════════════════════════════════════════════════════════════════════════
def plot_staircase_gap():
    with open(DATA / "micro20_multistart_staircase.json") as f:
        d = json.load(f)

    true_mag = d['true_omega_mag_degs']
    staircase_mags = d['micro18_staircase_mags']

    fig, ax = plt.subplots(figsize=(10, 5))

    # Band sweep results
    for band in d['bands']:
        lo, hi = band['band_degs']
        mid = (lo + hi) / 2
        n = band['n_valid']
        colour = C_TRUE if lo <= true_mag <= hi else C_GOOD
        ax.bar(mid, n, width=0.45, color=colour, alpha=0.7, edgecolor='white', lw=0.5)
        ax.text(mid, n + 0.3, str(n), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Staircase mags (red X markers)
    for sm in staircase_mags:
        ax.plot(sm, -0.6, 'x', color=C_WRONG, ms=10, mew=2.5)

    # True omega line
    ax.axvline(true_mag, color=C_TRUE, ls='-', lw=2.5, alpha=0.8, label=f'true |ω| = {true_mag:.2f} deg/s')

    # Gap annotation
    ax.annotate('', xy=(0.249, -1.4), xytext=(3.279, -1.4),
                arrowprops=dict(arrowstyle='<->', color=C_WRONG, lw=2))
    ax.text(1.76, -1.8, 'staircase gap: 0.25 → 3.28 deg/s\n(5 families missed)',
            ha='center', fontsize=9, color=C_WRONG, fontweight='bold')

    ax.set_xlabel('|ω| (deg/s)')
    ax.set_ylabel('Valid solutions found (out of 10 starts)')
    ax.set_title('micro20: Multi-Start Band Sweep Recovers ALL Missing Winding Families (Leg 1)',
                 fontweight='bold')
    ax.set_ylim(-2.5, 12)
    ax.set_xlim(-0.3, 6.5)
    ax.legend(loc='upper right')

    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color=C_WRONG, lw=0, ms=10, mew=2.5,
               label='Original staircase (micro18)'),
        plt.Rectangle((0, 0), 1, 1, fc=C_GOOD, alpha=0.7, label='Band-sweep valid solutions'),
        plt.Rectangle((0, 0), 1, 1, fc=C_TRUE, alpha=0.7, label='True ω band'),
        Line2D([0], [0], color=C_TRUE, lw=2.5, label=f'true |ω| = {true_mag:.2f} deg/s'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "02_staircase_gap.png", bbox_inches='tight')
    plt.close(fig)
    print("  [2/7] staircase_gap")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Multi-epoch scoring failure (micro21)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_multi_epoch_scoring():
    with open(DATA / "micro21_multi_epoch_winding_score.json") as f:
        d = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: MSE ranking vs observed
    ranking = d['vs_observed']['ranking']
    steps = [r['step'] for r in ranking]
    mses = [r['mse'] for r in ranking]
    dir_errs = [r['dir_err_deg'] for r in ranking]
    mags = [r['mag_degs'] for r in ranking]

    colours = [C_TRUE if s == 3 else C_NEUTRAL for s in steps]
    bars = ax1.barh(range(len(steps)), mses, color=colours, edgecolor='white', height=0.7)

    for i, (s, mse, de, m) in enumerate(zip(steps, mses, dir_errs, mags)):
        label = f'step {s}  |ω|={m:.1f}  dir_err={de:.0f}°'
        ax1.text(mse + 0.05, i, label, va='center', fontsize=8)

    ax1.set_yticks(range(len(steps)))
    ax1.set_yticklabels([f'Rank {i+1}' for i in range(len(steps))])
    ax1.set_xlabel('Multi-epoch MSE (vs observed hi-fi LC)')
    ax1.set_title('Ranking vs Observed LC', fontweight='bold')
    ax1.set_xlim(0, 5.5)
    ax1.invert_yaxis()

    # Annotate correct step
    correct_idx = next(i for i, s in enumerate(steps) if s == 3)
    ax1.annotate('correct winding', xy=(mses[correct_idx], correct_idx),
                xytext=(mses[correct_idx] + 0.8, correct_idx - 0.8),
                arrowprops=dict(arrowstyle='->', color=C_TRUE, lw=1.5),
                fontsize=9, color=C_TRUE, fontweight='bold')

    # Panel B: Direction error for all 8 steps
    all_dir_errs = d['omega_direction_errors_deg']
    step_nums = list(range(len(all_dir_errs)))
    colours2 = [C_TRUE if s == 3 else C_WRONG for s in step_nums]
    ax2.bar(step_nums, all_dir_errs, color=colours2, alpha=0.8, edgecolor='white')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_xlabel('Staircase step')
    ax2.set_ylabel('ω direction error (deg)')
    ax2.set_title('All Steps Have Wrong Direction (13-25°)', fontweight='bold')
    ax2.set_ylim(0, 30)

    for i, (de, ) in enumerate(zip(all_dir_errs,)):
        ax2.text(i, de + 0.5, f'{de:.0f}°', ha='center', fontsize=8, fontweight='bold')

    fig.suptitle('micro21: Multi-Epoch LC Scoring Cannot Discriminate — Direction Error is the Problem',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "03_multi_epoch_scoring_failure.png", bbox_inches='tight')
    plt.close(fig)
    print("  [3/7] multi_epoch_scoring")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: L-conservation heatmap (micro23)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L_heatmap():
    with open(DATA / "micro23_L_oracle_test.json") as f:
        d = json.load(f)

    L_err = np.array(d['L_err_matrix'])
    leg0_mags = d['leg0_mags_degs']
    leg1_mags = d['leg1_mags_degs']
    true_k, true_j = d['true_pair']['k'], d['true_pair']['j']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel A: Heatmap
    im = ax1.imshow(np.log10(L_err + 1e-10), cmap='RdYlGn_r', aspect='auto',
                    interpolation='nearest')
    ax1.set_xticks(range(8))
    ax1.set_xticklabels([f'{m:.1f}' for m in leg1_mags], fontsize=8)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels([f'{m:.1f}' for m in leg0_mags], fontsize=8)
    ax1.set_xlabel('Leg 1 |ω| (deg/s)')
    ax1.set_ylabel('Leg 0 |ω| (deg/s)')
    ax1.set_title('||ΔL|| Matrix (log10 scale)', fontweight='bold')

    # Mark true pair
    rect = plt.Rectangle((true_j - 0.5, true_k - 0.5), 1, 1,
                          fill=False, edgecolor=C_TRUE, lw=3)
    ax1.add_patch(rect)
    ax1.text(true_j, true_k, f'TRUE\n{d["true_L_err"]:.1e}',
             ha='center', va='center', fontsize=7, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.2', facecolor=C_TRUE, alpha=0.8))

    cb = fig.colorbar(im, ax=ax1, shrink=0.8)
    cb.set_label('log₁₀(||ΔL|| + ε)')

    # Panel B: Gap visualisation
    L_flat = L_err.flatten()
    L_sorted = np.sort(L_flat)
    ax2.semilogy(range(len(L_sorted)), L_sorted + 1e-10, 'o-', color=C_GOOD, ms=4)
    ax2.axhline(L_sorted[0] + 1e-10, color=C_TRUE, ls='--', lw=1,
                label=f'True pair: {L_sorted[0]:.1e}')
    ax2.axhline(L_sorted[1], color=C_WRONG, ls='--', lw=1,
                label=f'Next best: {L_sorted[1]:.1f}')

    # Gap annotation
    ax2.annotate('', xy=(0.5, L_sorted[0] + 1e-8), xytext=(0.5, L_sorted[1]),
                arrowprops=dict(arrowstyle='<->', color=C_HIGHLIGHT, lw=2.5))
    ax2.text(2, 0.5, f'gap = {d["gap_L"]:.0f}\nkg·m²/s\n(9 orders\nof magnitude)',
             fontsize=10, color=C_HIGHLIGHT, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax2.set_xlabel('Winding pair (sorted)')
    ax2.set_ylabel('||ΔL|| (kg·m²/s)')
    ax2.set_title('Gap: True Pair vs All Others', fontweight='bold')
    ax2.legend(loc='lower right')

    fig.suptitle('micro23: L-Conservation Uniquely Identifies Correct Winding Pair (Rank 1/64)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "04_L_conservation_heatmap.png", bbox_inches='tight')
    plt.close(fig)
    print("  [4/7] L_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: L-conservation nudge robustness (micro24)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L_nudge_robustness():
    with open(DATA / "micro24_L_nudge_sensitivity.json") as f:
        d = json.load(f)

    nudges = d['nudge_degs']
    results = d['part_B_results']

    gaps = [results[str(n)]['avg_gap'] for n in nudges]
    true_Lerrs = [results[str(n)]['avg_true_Lerr'] for n in nudges]
    p_corrects = [results[str(n)]['p_correct'] for n in nudges]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Gap vs nudge
    ax1.fill_between(nudges, true_Lerrs, gaps, alpha=0.15, color=C_GOOD)
    ax1.plot(nudges, gaps, 'o-', color=C_GOOD, lw=2, ms=7, label='Gap to next-best')
    ax1.plot(nudges, true_Lerrs, 's-', color=C_HIGHLIGHT, lw=2, ms=7, label='True pair ||ΔL||')
    ax1.set_xlabel('Endpoint attitude error (deg)')
    ax1.set_ylabel('||ΔL|| (kg·m²/s)')
    ax1.set_title('Signal (Gap) vs Noise (True Pair Error)', fontweight='bold')
    ax1.legend(loc='center right')
    ax1.set_xlim(0, 11)

    # SNR annotation
    for i, n in enumerate(nudges):
        snr = gaps[i] / max(true_Lerrs[i], 0.01)
        if n in [1.0, 5.0, 10.0]:
            ax1.annotate(f'SNR={snr:.0f}x', xy=(n, gaps[i]),
                        xytext=(n + 0.5, gaps[i] + 3),
                        fontsize=8, color=C_GOOD, fontweight='bold')

    # Panel B: P(correct) — all 100%
    ax2.bar(range(len(nudges)), [pc * 100 for pc in p_corrects],
            color=C_TRUE, alpha=0.8, edgecolor='white')
    ax2.set_xticks(range(len(nudges)))
    ax2.set_xticklabels([f'{n}°' for n in nudges])
    ax2.set_xlabel('Endpoint attitude error')
    ax2.set_ylabel('P(correct pair selected) %')
    ax2.set_title('100% Correct at All Nudge Levels', fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.axhline(100, color=C_NEUTRAL, ls='--', lw=0.5)
    for i in range(len(nudges)):
        ax2.text(i, 103, '100%', ha='center', fontsize=9, fontweight='bold', color=C_TRUE)

    fig.suptitle('micro24: L-Conservation Robust to Endpoint Attitude Error (30 trials per nudge)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "05_L_nudge_robustness.png", bbox_inches='tight')
    plt.close(fig)
    print("  [5/7] L_nudge_robustness")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Staircase overview (micro17) — magnitudes and trough scores
# ═══════════════════════════════════════════════════════════════════════════════
def plot_staircase_overview():
    with open(DATA / "micro17_staircase_omega.json") as f:
        d = json.load(f)

    steps = d['steps']
    mags = [s['mag_degs'] for s in steps]
    trough_errs = [s['trough_err'] for s in steps]
    true_mag = d['true_omega_mag_degs']
    correct = d['correct_step_idx']
    best_trough = d['best_step_by_trough']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Panel A: Magnitude staircase
    colours = [C_TRUE if i == correct else C_GOOD for i in range(len(mags))]
    bars = ax1.bar(range(len(mags)), mags, color=colours, alpha=0.8, edgecolor='white')
    ax1.axhline(true_mag, color=C_TRUE, ls='--', lw=2, label=f'true |ω| = {true_mag:.2f} deg/s')
    ax1.set_ylabel('|ω| (deg/s)')
    ax1.set_title('Staircase: 8 Winding Solutions (Each Adds ~1 Revolution)', fontweight='bold')
    ax1.legend(loc='upper left')

    for i, m in enumerate(mags):
        mark = ' ← correct' if i == correct else ''
        ax1.text(i, m + 0.1, f'{m:.2f}{mark}', ha='center', fontsize=8, fontweight='bold')

    # Panel B: Trough scores
    colours2 = [C_HIGHLIGHT if i == best_trough else (C_TRUE if i == correct else C_NEUTRAL)
                for i in range(len(trough_errs))]
    ax2.bar(range(len(trough_errs)), trough_errs, color=colours2, alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Staircase step')
    ax2.set_ylabel('Trough brightness error')
    ax2.set_title('Single-Trough Scoring Picks Wrong Winding (step 2, not step 3)', fontweight='bold')
    ax2.set_xticks(range(len(mags)))

    for i, te in enumerate(trough_errs):
        mark = ''
        if i == best_trough:
            mark = ' (best)'
        elif i == correct:
            mark = ' (true)'
        ax2.text(i, te + 0.03, f'{te:.2f}{mark}', ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS / "06_staircase_overview.png", bbox_inches='tight')
    plt.close(fig)
    print("  [6/7] staircase_overview")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Pipeline architecture summary
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    boxes = [
        (0.5, 4.0, 2.5, 1.4, 'Step 1\nIso-Brightness\nCandidates',
         'At each peak, find\nattitudes matching\nobserved brightness\n(L-BFGS-B, 10K seeds)',
         '#E8F4FD', C_GOOD),
        (3.5, 4.0, 2.5, 1.4, 'Step 2\nBand-Sweep\nω Enumeration',
         'Per leg: 0.5 deg/s bands\n10 random starts each\n→ all winding families\n(~240 bridge solves)',
         '#FFF3E0', C_HIGHLIGHT),
        (6.5, 4.0, 2.5, 1.4, 'Step 3\nL-Conservation\nWinding Filter',
         'At shared peak nodes:\n||ΔL|| = ||I·Δω||\nTrue pair: rank 1/64\ngap = 113 kg·m²/s',
         '#E8F5E9', C_TRUE),
        (9.5, 4.0, 2.5, 1.4, 'Step 4\nLocal Joint\nRefinement',
         'L-BFGS-B on 6D\n(q₀, ω₀) jointly\nBasin: ~5° att, ~0.02 dps',
         '#F3E5F5', '#9C27B0'),
    ]

    for x, y, w, h, title, desc, fc, ec in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                              lw=2, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.15, title, ha='center', va='top',
                fontsize=10, fontweight='bold', zorder=3)
        ax.text(x + w/2, y + 0.15, desc, ha='center', va='bottom',
                fontsize=7.5, color='#333', zorder=3)

    # Arrows
    for i in range(3):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3] / 2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Status badges below boxes
    statuses = [
        (1.75, 3.5, 'Series 02/05\nValidated', C_GOOD),
        (4.75, 3.5, 'Series 07c\nValidated', C_TRUE),
        (7.75, 3.5, 'Series 07b\nValidated', C_TRUE),
        (10.75, 3.5, 'Series 04\nBasin known', C_HIGHLIGHT),
    ]
    for x, y, text, colour in statuses:
        ax.text(x, y, text, ha='center', va='top', fontsize=8,
                color=colour, fontweight='bold')

    # Dead end annotations
    dead_ends = [
        (4.75, 2.2, 'Series 07a: Multi-epoch\nLC scoring (DEAD END)\nAll ω directions wrong,\nscoring can\'t compensate', C_WRONG),
        (1.75, 2.2, 'Series 07: Staircase\nheuristic (DEAD END)\nMisses 5/12 families\non long legs', C_WRONG),
    ]
    for x, y, text, colour in dead_ends:
        ax.text(x, y, text, ha='center', va='top', fontsize=7.5,
                color=colour, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=C_WRONG,
                          alpha=0.8, lw=1))

    ax.set_title('Proposed Integration Pipeline (All Components Validated Independently)',
                 fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(ASSETS / "07_pipeline_architecture.png", bbox_inches='tight')
    plt.close(fig)
    print("  [7/7] pipeline_architecture")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating report plots...")
    plot_winding_landscape()
    plot_staircase_gap()
    plot_multi_epoch_scoring()
    plot_L_heatmap()
    plot_L_nudge_robustness()
    plot_staircase_overview()
    plot_pipeline_architecture()
    print("Done. All plots saved to", ASSETS)
