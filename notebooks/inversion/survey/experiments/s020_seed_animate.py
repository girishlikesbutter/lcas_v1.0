"""s020 — animation for the seed-pipeline checkpoint.

Reads the checkpoint NPZs from `results/s020/seedXXX/` and produces a
self-contained HTML animation (Plotly) showing the funnel:

  Stage 0 — ALL CANDIDATES
  Stage 1 — GEO PRUNE       (geo_score >= geo_threshold)
  Stage 2 — ALIGN PRUNE     (also align_score >= align_threshold)

Layout (4 panels, master stage slider):
  ┌─────────────────────────┬──────────────────────────┐
  │  ω-space scatter (3D)   │  Bracket spectrum        │
  ├─────────────────────────┼──────────────────────────┤
  │  Score landscape (2D)   │  LC overlay              │
  └─────────────────────────┴──────────────────────────┘

Usage:
  python notebooks/inversion/survey/experiments/s020_seed_animate.py 6
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))


N_DISPLAY_POINTS = 50000   # subsample for browser performance
RNG_SEED = 42

STAGE_LABELS = ["ALL CANDIDATES", "GEO PRUNE", "ALIGN PRUNE"]


def load_checkpoint(seed_dir: Path) -> dict:
    cp = {}
    for name in ["bracket", "omega_grid", "candidates_meta", "thresholds",
                 "spec_geometry", "survivor_lcs", "survivor_diagnostics",
                 "q_target_pool"]:
        path = seed_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        d = np.load(path, allow_pickle=True)
        cp[name] = {k: d[k] for k in d.files}
    with open(seed_dir / "summary.json") as f:
        cp["summary"] = json.load(f)
    return cp


def stratified_subsample(N_total, geo_pass, align_pass, n_target=N_DISPLAY_POINTS,
                         rng=None):
    """Subsample candidates respecting a display budget while preserving the
    funnel structure.

    Priority order:
      1. ALL cat_both (filter survivors) — never sample these out.
      2. Up to 30% of budget on cat_geo_only.
      3. Up to 30% of budget on cat_align_only.
      4. Remaining budget on cat_neither (the rejected mass).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    cat_both = geo_pass & align_pass
    cat_geo_only = geo_pass & ~align_pass
    cat_align_only = ~geo_pass & align_pass
    cat_neither = ~geo_pass & ~align_pass

    keep = np.zeros(N_total, dtype=bool)
    keep[cat_both] = True
    n_used = int(cat_both.sum())

    def _sample(mask, max_count):
        nonlocal n_used
        idx = np.where(mask)[0]
        budget = max(0, min(max_count, n_target - n_used))
        if idx.size <= budget:
            keep[idx] = True
            n_used += idx.size
        elif budget > 0:
            chosen = rng.choice(idx, size=budget, replace=False)
            keep[chosen] = True
            n_used += budget

    geo_budget = int(0.30 * n_target)
    align_budget = int(0.30 * n_target)
    _sample(cat_geo_only, geo_budget)
    _sample(cat_align_only, align_budget)
    _sample(cat_neither, n_target - n_used)
    return keep


def build_omega_3d(cp, sub_idx):
    """3D scatter of candidates in ω-space, coloured by stage."""
    omega_vectors = cp["omega_grid"]["omega_vectors"]
    cand_omega_cell = cp["candidates_meta"]["omega_cell_idx"]
    cand_omega = omega_vectors[cand_omega_cell[sub_idx]]      # (n_disp, 3)
    truth_omega = cp["survivor_diagnostics"]["truth_omega"]
    return cand_omega, truth_omega


def color_by_stage(geo_pass, align_pass, sub_idx, stage):
    """Per-point alpha (opacity) for a stage."""
    g = geo_pass[sub_idx]
    a = align_pass[sub_idx]
    n = sub_idx.size
    opacity = np.full(n, 0.06, dtype=float)
    if stage == 0:           # ALL
        opacity[:] = 0.06
        opacity[g | a] = 0.10  # mild bias up for would-survivors so they peek through
    elif stage == 1:         # GEO PRUNE
        opacity[~g] = 0.02
        opacity[g] = 0.40
    elif stage == 2:         # ALIGN PRUNE
        opacity[~(g & a)] = 0.02
        opacity[g & a] = 0.95
    return opacity


def color_value(geo_pass, align_pass, sub_idx, stage):
    """RGBA-style color tags per point.

    Encoding:
      0 = rejected_both, 1 = passed_geo_only, 2 = passed_align_only, 3 = passed_both
    """
    g = geo_pass[sub_idx]
    a = align_pass[sub_idx]
    code = np.zeros(sub_idx.size, dtype=int)
    code[g & ~a] = 1
    code[~g & a] = 2
    code[g & a] = 3
    return code


def build_figure(cp, seed: int) -> go.Figure:
    cand_meta = cp["candidates_meta"]
    geo_pass = cand_meta["cat_both"] | cand_meta["cat_geo_only"]
    align_pass = cand_meta["cat_both"] | cand_meta["cat_align_only"]
    geo_score = cand_meta["geo_score"]
    align_score = cand_meta["align_score"]

    N_total = geo_pass.size
    sub_keep = stratified_subsample(N_total, geo_pass, align_pass)
    sub_idx = np.where(sub_keep)[0]
    n_disp = sub_idx.size

    cand_omega, truth_omega = build_omega_3d(cp, sub_idx)

    # ---- Build figure ----
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "ω-space (rad/s) — candidates colored by category",
            "LS bracket spectrum — bracket cells in red",
            "Score landscape — geo vs alignment",
            "LC overlay — truth (white) + survivors",
        ),
        column_widths=[0.55, 0.45],
        row_heights=[0.55, 0.45],
        horizontal_spacing=0.08, vertical_spacing=0.10,
    )

    # Color codes
    cat_code = color_value(geo_pass, align_pass, sub_idx, stage=0)
    color_map = {
        0: "rgba(80,80,80,1)",       # rejected_both — grey
        1: "rgba(180,140,40,1)",     # geo_only      — amber
        2: "rgba(60,160,200,1)",     # align_only    — cyan
        3: "rgba(255,80,80,1)",      # both          — red-ish
    }
    cat_labels = {
        0: "rejected_both",
        1: "geo_pass / align_fail",
        2: "geo_fail / align_pass",
        3: "passed both",
    }

    # ---- Pre-build per-stage opacity arrays ----
    stage_opacities = {s: color_by_stage(geo_pass, align_pass, sub_idx, s)
                       for s in range(3)}

    # ---- Panel A: 3D ω-space ----
    # We'll add four separate scatters (one per category) so the legend is meaningful,
    # plus a truth marker.
    omega_x = cand_omega[:, 0]
    omega_y = cand_omega[:, 1]
    omega_z = cand_omega[:, 2]

    base_traces_per_cat = {}
    for cat_val in [0, 1, 2, 3]:
        m = cat_code == cat_val
        if m.sum() == 0:
            continue
        cat_data = {
            "x": omega_x[m].tolist(), "y": omega_y[m].tolist(),
            "z": omega_z[m].tolist(),
            "opacity_by_stage": [stage_opacities[s][m].mean() for s in range(3)],
            "n": int(m.sum()),
        }
        base_traces_per_cat[cat_val] = cat_data
        fig.add_trace(
            go.Scatter3d(
                x=cat_data["x"], y=cat_data["y"], z=cat_data["z"],
                mode='markers',
                marker=dict(
                    size=2.0,
                    color=color_map[cat_val],
                    opacity=cat_data["opacity_by_stage"][0],
                ),
                name=f"{cat_labels[cat_val]} (n={cat_data['n']})",
                hoverinfo='skip',
                legendgroup="cands",
            ),
            row=1, col=1,
        )

    # Truth marker
    fig.add_trace(
        go.Scatter3d(
            x=[truth_omega[0]], y=[truth_omega[1]], z=[truth_omega[2]],
            mode='markers',
            marker=dict(size=10, color='gold', symbol='diamond',
                        line=dict(color='black', width=1)),
            name='TRUTH ω',
            legendgroup="truth",
        ),
        row=1, col=1,
    )

    # ---- Panel B: bracket spectrum ----
    bracket = cp["bracket"]
    ls_freqs = bracket["ls_freqs"]
    ls_power = bracket["ls_power"]
    ls_omegas = 2 * np.pi * ls_freqs
    bracket_cells = bracket["bracket_cells"]
    truth_omega_mag = float(bracket["truth_omega_mag_rad"])

    fig.add_trace(
        go.Scatter(
            x=ls_omegas, y=ls_power,
            mode='lines',
            line=dict(color='steelblue', width=1.5),
            name='LS power',
            legendgroup='bracket',
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=bracket_cells,
            y=[ls_power.max() * 0.9] * bracket_cells.size,
            mode='markers',
            marker=dict(symbol='line-ns-open', size=20, color='red',
                        line=dict(color='red', width=2)),
            name=f'bracket cells ({bracket_cells.size})',
            legendgroup='bracket',
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[truth_omega_mag, truth_omega_mag],
            y=[0, ls_power.max()],
            mode='lines',
            line=dict(color='gold', width=2, dash='dash'),
            name='truth |ω|',
            legendgroup='truth',
        ),
        row=1, col=2,
    )

    # ---- Panel C: score landscape (2D) ----
    geo_threshold = float(cp["thresholds"]["geo_threshold"])
    align_threshold = float(cp["thresholds"]["align_threshold"])

    for cat_val in [0, 1, 2, 3]:
        m = cat_code == cat_val
        if m.sum() == 0:
            continue
        gs = geo_score[sub_idx][m]
        als = align_score[sub_idx][m]
        # Add tiny jitter so points at identical scores don't all overlap
        rng = np.random.default_rng(seed * 7 + cat_val)
        jx = rng.uniform(-0.005, 0.005, size=m.sum())
        jy = rng.uniform(-0.005, 0.005, size=m.sum())
        fig.add_trace(
            go.Scatter(
                x=gs + jx, y=als + jy,
                mode='markers',
                marker=dict(size=4,
                            color=color_map[cat_val],
                            opacity=stage_opacities[0][m].mean()),
                name=f'{cat_labels[cat_val]}',
                legendgroup='scores',
                showlegend=False,
            ),
            row=2, col=1,
        )
    # Threshold crosshairs (Scatter lines — add_hline doesn't compose well with
    # a 3D scene subplot in the same figure)
    fig.add_trace(
        go.Scatter(x=[-0.05, 1.05], y=[align_threshold, align_threshold],
                   mode='lines',
                   line=dict(color='gold', dash='dash', width=1.5),
                   name=f'align thresh = {align_threshold:.3f}',
                   legendgroup='thresh', showlegend=True),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[geo_threshold, geo_threshold], y=[-0.05, 1.05],
                   mode='lines',
                   line=dict(color='gold', dash='dash', width=1.5),
                   name=f'geo thresh = {geo_threshold:.3f}',
                   legendgroup='thresh', showlegend=True),
        row=2, col=1,
    )

    # ---- Panel D: LC overlay ----
    surv_lcs = cp["survivor_lcs"]
    obs_times = surv_lcs["observation_times"]
    truth_mag_hifi = surv_lcs["truth_mag_hifi"]
    survivor_mag_pred = surv_lcs["survivor_mag_pred"]
    survivor_cand_idx = surv_lcs["survivor_cand_idx"]

    fig.add_trace(
        go.Scatter(
            x=obs_times, y=truth_mag_hifi,
            mode='lines',
            line=dict(color='white', width=2),
            name='TRUTH hi-fi',
            legendgroup='lc',
        ),
        row=2, col=2,
    )
    if survivor_mag_pred.shape[0] > 0:
        # Plot up to 50 random survivors to avoid overcrowding
        n_show = min(survivor_mag_pred.shape[0], 50)
        rng = np.random.default_rng(0)
        show_idx = rng.choice(survivor_mag_pred.shape[0], size=n_show, replace=False)
        for k, si in enumerate(show_idx):
            fig.add_trace(
                go.Scatter(
                    x=obs_times, y=survivor_mag_pred[si],
                    mode='lines',
                    line=dict(color='rgba(255,80,80,0.35)', width=0.8),
                    name=f'survivor #{si}',
                    legendgroup='survivors',
                    showlegend=(k == 0),
                ),
                row=2, col=2,
            )

    # ---- Layout + axis ----
    fig.update_layout(
        title=dict(
            text=(f"<b>s020 funnel — seed {seed}</b>  |  "
                  f"thresholds geo={geo_threshold:.3f}, align={align_threshold:.3f}  |  "
                  f"survivors: {int(cand_meta['cat_both'].sum())} of "
                  f"{int(geo_pass.size)} candidates"),
            font=dict(size=14),
        ),
        template='plotly_dark',
        height=900,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top', y=0.98,
            xanchor='left', x=1.02,
            font=dict(size=10),
        ),
    )
    fig.update_scenes(
        xaxis_title='ωx (rad/s)',
        yaxis_title='ωy (rad/s)',
        zaxis_title='ωz (rad/s)',
        aspectmode='cube',
        row=1, col=1,
    )
    fig.update_xaxes(title_text='|ω| (rad/s)', row=1, col=2)
    fig.update_yaxes(title_text='LS power', row=1, col=2)
    fig.update_xaxes(title_text='geo_score', range=[-0.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text='align_score', range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text='time (s)', row=2, col=2)
    fig.update_yaxes(title_text='magnitude', autorange='reversed', row=2, col=2)

    # ---- Stage slider ----
    # Each stage updates the candidate-marker opacities in panel A and panel C.
    # We rebuild the trace updates: for the 4 candidate Scatter3d traces
    # (panel A) and the 4 candidate Scatter traces (panel C, indices known by
    # construction order).
    sliders_steps = []
    for s in range(3):
        # Trace order added: per category in {0,1,2,3} (panel A scatter3d) +
        # 1 truth marker (panel A) + 3 bracket traces + per category in {0,1,2,3}
        # (panel C) + truth/survivor traces (panel D).
        # Build opacity update for panel A and C.
        marker_opacities = []
        for trace in fig.data:
            if isinstance(trace, go.Scatter3d) and trace.legendgroup == 'cands':
                # find which category this is
                # parse from name "<label> (n=...)"
                name = trace.name.split(' (')[0]
                cat_val = next(k for k, v in cat_labels.items() if v == name)
                m = cat_code == cat_val
                if m.sum() == 0:
                    marker_opacities.append(None)
                    continue
                # Convert per-point to a single mean opacity for this stage.
                # Plotly Scatter3d only supports scalar opacity, so we use the mean.
                # For better fidelity, surviving categories get high opacity,
                # rejected get low.
                if cat_val == 0:
                    marker_opacities.append(0.06 if s == 0 else 0.015)
                elif cat_val == 1:
                    marker_opacities.append(0.08 if s == 0 else (0.50 if s == 1 else 0.10))
                elif cat_val == 2:
                    marker_opacities.append(0.08 if s == 0 else (0.05 if s == 1 else 0.05))
                elif cat_val == 3:
                    marker_opacities.append(0.10 if s == 0 else (0.50 if s == 1 else 0.95))
            elif isinstance(trace, go.Scatter) and trace.legendgroup == 'scores':
                name = trace.name
                cat_val = next(k for k, v in cat_labels.items() if v == name)
                if cat_val == 0:
                    marker_opacities.append(0.30 if s == 0 else 0.05)
                elif cat_val == 1:
                    marker_opacities.append(0.30 if s == 0 else (0.7 if s == 1 else 0.20))
                elif cat_val == 2:
                    marker_opacities.append(0.30 if s == 0 else (0.10 if s == 1 else 0.20))
                elif cat_val == 3:
                    marker_opacities.append(0.40 if s == 0 else (0.7 if s == 1 else 1.0))
            else:
                marker_opacities.append(None)

        sliders_steps.append({
            "method": "restyle",
            "label": STAGE_LABELS[s],
            "args": [
                {"marker.opacity": marker_opacities},
            ],
        })

    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Stage: ', font=dict(size=14, color='gold')),
            pad=dict(t=40, b=10),
            steps=sliders_steps,
        )],
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Read from results/s020/smoke/ instead of seedXXX/")
    args = parser.parse_args()

    root = Path(args.checkpoint_root) if args.checkpoint_root else \
        SURVEY_DIR / "results" / "s020"
    sub = "smoke" if args.smoke else f"seed{args.seed:03d}"
    seed_dir = root / sub
    if not seed_dir.exists():
        raise FileNotFoundError(f"No checkpoint at {seed_dir}")

    print(f"Loading checkpoint from {seed_dir}", flush=True)
    cp = load_checkpoint(seed_dir)

    print("Building figure …", flush=True)
    fig = build_figure(cp, args.seed)

    out = Path(args.out) if args.out else \
        seed_dir / f"s020_funnel_seed{args.seed:03d}.html"
    fig.write_html(str(out), include_plotlyjs='cdn')
    size_mb = out.stat().st_size / 1e6
    print(f"Saved: {out} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
