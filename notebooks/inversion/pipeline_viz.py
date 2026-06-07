#!/usr/bin/env python3
"""Stage-by-stage pipeline visualisation for one seed.

Reads the instrumented checkpoints (m103 grid/lofi/NM/phi-sweep, m115 DE
history + hi-fi LCs, m126 polish trajectory + before/after LCs) plus the
m048 trajectory truth and the m133 surr_q0polish_mse cache, and renders
a single self-contained HTML report with one Plotly figure per pipeline
stage.

Usage:
    python3 notebooks/inversion/pipeline_viz.py --seed 91 --traj-source m048

Output:
    data/results/inversion_diagnostics/pipeline_viz/seed_NNN.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.traj_source import load_truth  # noqa: E402

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


# ── Data loading ─────────────────────────────────────────────────────────

def m103_dir(seed, source):
    sub = "m103_hybrid" if source == "m046" else f"m103_hybrid_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def m115_dir(seed, source):
    sub = "m115_surrogate_pipeline" if source == "m046" else f"m115_surrogate_pipeline_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def m126_dir(seed, source):
    sub = "m126_wrapped" if source == "m046" else f"m126_wrapped_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def load_all(seed, source):
    truth = load_truth(seed, source)
    m103 = m103_dir(seed, source)
    m115 = m115_dir(seed, source)
    m126 = m126_dir(seed, source)
    data = {
        "seed": seed, "source": source, "truth": truth,
        "grid": np.load(m103 / "grid_top500_ckpt.npz", allow_pickle=True),
        "lofi": np.load(m103 / "lofi_ckpt.npz", allow_pickle=True),
        "nm": np.load(m103 / "nm_prededup_ckpt.npz", allow_pickle=True),
        "phi_sweeps": np.load(m103 / "phi_sweeps_ckpt.npz", allow_pickle=True),
        "multi_phi": np.load(m103 / "multi_phi_ckpt.npz", allow_pickle=True),
        "geo": np.load(m103 / "geo_ckpt.npz", allow_pickle=True),
        "m103_result": json.load(open(m103 / "result.json")),
        "de_history": np.load(m115 / "de_history.npz", allow_pickle=True),
        "step1_de": np.load(m115 / "step1_de.npz", allow_pickle=True),
        "step2_hifi": np.load(m115 / "step2_hifi.npz", allow_pickle=True),
        "m115_result": json.load(open(m115 / "result.json")),
        "polish": np.load(m126 / "polish_ckpt.npz", allow_pickle=True),
        "hifi": np.load(m126 / "hifi_ckpt.npz", allow_pickle=True),
        "m126_result": json.load(open(m126 / "result.json")),
    }
    q0p_path = DIAG / "rerank_experiment" / f"seed_{seed:03d}_q0polish.json"
    data["q0polish"] = json.load(open(q0p_path)) if q0p_path.exists() else None
    surr_path = m103_dir(seed, source) / "lofi_surr_ckpt.npz"
    data["lofi_surr"] = np.load(surr_path, allow_pickle=True) if surr_path.exists() else None
    grid_surr_path = m103_dir(seed, source) / "grid_surr_ckpt.npz"
    data["grid_surr"] = np.load(grid_surr_path, allow_pickle=True) if grid_surr_path.exists() else None
    return data


# ── Helpers ──────────────────────────────────────────────────────────────

def omega_dir_err_deg(w1, w2):
    a = np.asarray(w1) / np.linalg.norm(w1)
    b = np.asarray(w2) / np.linalg.norm(w2)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def stage_title(stage_name, n_in, n_out, extra=""):
    return f"<b>{stage_name}</b> &nbsp;|&nbsp; in: {n_in} &nbsp;→&nbsp; out: {n_out} {extra}"


# ── Panel 0 — Header / verdict ───────────────────────────────────────────

def best_basin_idx(d):
    """Index (in m126 records & hifi_ckpt order) of the lowest hifi_wrapped basin."""
    basins = d["m126_result"]["basins"]
    return int(min(range(len(basins)), key=lambda i: basins[i]["hifi_wrapped"]))


def panel_header(d):
    truth_lc = d["truth"]["mag_hifi"]
    obs_lc = d["truth"]["observed_lc"]
    bi = best_basin_idx(d)
    final_lc = d["hifi"]["hifi_mags_after"][bi]
    t_axis = np.arange(len(truth_lc))
    res = d["m126_result"]
    winner = res["basins"][bi]
    err_q0 = winner["q0_err_after"]
    err_w_dir = winner["w_dir_err_after"]
    err_w_mag = winner["w_mag_err_pct_after"]
    rho = float(np.sqrt(res["best_hifi_wrapped"] / 0.05))

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3],
        subplot_titles=("Light curve — truth (black) vs final winner (orange)", "Final errors"),
    )
    fig.add_trace(go.Scatter(x=t_axis, y=truth_lc, name="truth", mode="lines",
                             line=dict(color="black", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_axis, y=obs_lc, name="observed (noisy)", mode="lines",
                             line=dict(color="gray", width=0.5, dash="dot"),
                             opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_axis, y=final_lc, name="m126 winner", mode="lines",
                             line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1, title="mag")
    fig.update_xaxes(title="epoch", row=1, col=1)

    err_labels = ["q0_err (deg)", "w_dir_err (deg)", "|w_mag_err| (%)"]
    err_vals = [err_q0, err_w_dir, abs(err_w_mag)]
    err_colors = ["#2ca02c" if v < 5 else "#ff7f0e" if v < 30 else "#d62728" for v in err_vals]
    fig.add_trace(go.Bar(x=err_labels, y=err_vals, marker_color=err_colors,
                         text=[f"{v:.2f}" for v in err_vals], textposition="auto",
                         showlegend=False), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2, title="(log scale)")

    funnel = ("Pipeline funnel: 2000 grid &rarr; 500 top &rarr; 300 lofi &rarr; "
              "300 NM &rarr; 26 deduped &rarr; 3 omegas &rarr; 3 basins &rarr; 1 winner")
    fig.update_layout(
        title=(f"<b>Seed {d['seed']} ({d['source']})</b> &nbsp;|&nbsp; "
               f"&rho;={rho:.2f} ({['A','B','C','D'][min(int(rho/2),3)]}) "
               f"&nbsp;|&nbsp; q0={err_q0:.2f}&deg; "
               f"w_dir={err_w_dir:.2f}&deg; w_mag={err_w_mag:+.2f}%<br>"
               f"<sub>{funnel}</sub>"),
        height=400, margin=dict(t=90, b=40, l=60, r=20),
    )
    return fig


# ── Panel 1 — m103 Step 2 grid sweep ─────────────────────────────────────

def _fibonacci_sphere(n):
    """Same fibonacci sampler as m103.fibonacci_sphere — replicated to avoid
    importing the m103 script (which runs main code at import time)."""
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])


def panel_grid_sweep(d):
    g = d["grid"]
    omegas = g["top_omegas"]  # (500, 3)
    costs = g["top_costs"]
    n_dirs = int(g["n_dirs"])
    true_w = d["truth"]["omega0_rad"]
    true_w_hat = true_w / np.linalg.norm(true_w)

    # Reconstruct full Fibonacci grid (deterministic) so we can show the
    # underlying uniform sampling vs the cost-selected top-500.
    full_grid = _fibonacci_sphere(n_dirs)
    selected_dirs = omegas / np.linalg.norm(omegas, axis=1, keepdims=True)
    # mark which fibonacci points were selected by nearest-neighbour match
    nn_idx = set()
    for sd in selected_dirs:
        nn = int(np.argmax(full_grid @ sd))
        nn_idx.add(nn)
    not_selected = np.array([i for i in range(n_dirs) if i not in nn_idx])

    # Spherical coords for 2D Mollweide-like projection (lon, lat).
    def to_lonlat(vecs):
        lon = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))  # -180..180
        lat = np.degrees(np.arcsin(np.clip(vecs[:, 2], -1, 1)))  # -90..90
        return lon, lat

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.4, 0.4, 0.2],
        specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(f"3D sphere ({n_dirs} fibonacci dirs)",
                        "2D equirectangular projection (lon, lat)",
                        "Cost ranking"),
        horizontal_spacing=0.06,
    )

    # ── 3D panel ───────────────────────────────────────────────────────
    # Background: rejected fibonacci grid points (visible darker gray)
    fig.add_trace(go.Scatter3d(
        x=full_grid[not_selected, 0], y=full_grid[not_selected, 1], z=full_grid[not_selected, 2],
        mode="markers",
        marker=dict(size=2.5, color="#888", opacity=0.7),
        hoverinfo="skip", name="grid (rejected)", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=selected_dirs[:, 0], y=selected_dirs[:, 1], z=selected_dirs[:, 2],
        mode="markers",
        marker=dict(size=4, color=costs, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="cost", x=0.34, len=0.85)),
        text=[f"rank {i}<br>|w|={np.linalg.norm(o):.4f} rad/s<br>cost={c:.3e}"
              for i, (o, c) in enumerate(zip(omegas, costs))],
        hoverinfo="text", name="top-500", showlegend=False,
    ), row=1, col=1)
    # Truth marker: hollow circle with crosshair so any underlying top-500
    # point at (or near) truth remains visible.
    fig.add_trace(go.Scatter3d(
        x=[true_w_hat[0]], y=[true_w_hat[1]], z=[true_w_hat[2]],
        mode="markers",
        marker=dict(size=18, color="rgba(0,0,0,0)",
                    line=dict(color="red", width=4)),
        name="truth", text=[f"truth |w|={np.linalg.norm(true_w):.4f} rad/s"],
        hoverinfo="text",
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[true_w_hat[0]], y=[true_w_hat[1]], z=[true_w_hat[2]],
        mode="markers", marker=dict(size=4, color="red", symbol="cross"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── 2D projection panel ───────────────────────────────────────────
    lon_g, lat_g = to_lonlat(full_grid[not_selected])
    lon_s, lat_s = to_lonlat(selected_dirs)
    lon_t, lat_t = to_lonlat(true_w_hat[None, :])
    fig.add_trace(go.Scatter(
        x=lon_g, y=lat_g, mode="markers",
        marker=dict(size=4, color="#888", opacity=0.55),
        hoverinfo="skip", name="grid (rejected)", showlegend=True,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=lon_s, y=lat_s, mode="markers",
        marker=dict(size=7, color=costs, colorscale="Viridis",
                    line=dict(width=0.5, color="black"), showscale=False),
        text=[f"rank {i}<br>cost={c:.3e}" for i, c in enumerate(costs)],
        hoverinfo="text", name="top-500", showlegend=True,
    ), row=1, col=2)
    # Truth: hollow ring so an underlying selected dot remains visible.
    fig.add_trace(go.Scatter(
        x=lon_t, y=lat_t, mode="markers",
        marker=dict(size=24, color="rgba(0,0,0,0)",
                    line=dict(color="red", width=3), symbol="circle"),
        name="truth", showlegend=True,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=lon_t, y=lat_t, mode="markers",
        marker=dict(size=8, color="red", symbol="cross"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)
    fig.update_xaxes(title="longitude (deg)", range=[-185, 185], row=1, col=2,
                     dtick=60, gridcolor="#eee")
    fig.update_yaxes(title="latitude (deg)", range=[-92, 92], row=1, col=2,
                     dtick=30, gridcolor="#eee")

    # ── Cost ranking panel ────────────────────────────────────────────
    sorted_costs = np.sort(g["all_costs"])
    fig.add_trace(go.Scatter(x=np.arange(n_dirs), y=sorted_costs, mode="markers",
                             marker=dict(size=2, color=sorted_costs, colorscale="Viridis"),
                             showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[500, 500],
                             y=[float(sorted_costs.min()), float(sorted_costs.max())],
                             mode="lines", line=dict(dash="dash", color="red"),
                             name="top-500 cutoff", showlegend=False),
                  row=1, col=3)
    fig.update_xaxes(title="rank", row=1, col=3)
    fig.update_yaxes(title="alignment cost (log)", type="log", row=1, col=3)

    fig.update_layout(
        title=stage_title("m103 Step 2 — Grid sweep over &omega;-directions",
                          n_dirs, 500,
                          extra="(grid: 2000 directions &times; 20 magnitudes = 40k tested) "
                                "&mdash; gray = uniform Fibonacci grid; colored = top-500 by cost"),
        height=520, margin=dict(t=90, b=40, l=10, r=10),
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            aspectmode="cube",
        ),
        legend=dict(x=0.42, y=1.0, font=dict(size=10)),
    )
    return fig


# ── Panel 2 — m103 Step 2b lofi pool ─────────────────────────────────────

def panel_lofi(d):
    lofi = d["lofi"]
    nm = d["nm"]
    n = int(lofi["n"])
    n_matched = lofi["n_matched"]
    lofi_mse = lofi["lofi_mse"]
    align_cost = lofi["align_cost"]
    # Lofi candidates and NM-pre-dedup are 1:1 in original order? NO — lofi
    # gets RE-SORTED by (-n_matched, lofi_mse) before NM, so nm_prededup is in
    # post-sort order. lofi_ckpt is saved AFTER the sort. So they ARE 1:1.
    post_nm_q0_err = nm["q0_err"]
    post_nm_w0_err = nm["w0_err"]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        subplot_titles=("(n_matched, lofi_mse) for 300 lofi candidates &mdash; color = post-NM q0_err",
                        "Lofi alignment cost vs lofi_mse"),
    )
    fig.add_trace(go.Scatter(
        x=n_matched + np.random.uniform(-0.15, 0.15, n),  # jitter for visibility
        y=lofi_mse, mode="markers",
        marker=dict(
            size=6, color=post_nm_q0_err, colorscale="Plasma_r", cmin=0, cmax=180,
            showscale=True, colorbar=dict(title="post-NM<br>q0_err (deg)", x=0.46, len=0.85),
        ),
        text=[f"idx {i}<br>n_matched={int(n_matched[i])}<br>lofi_mse={lofi_mse[i]:.3f}<br>"
              f"post-NM q0_err={post_nm_q0_err[i]:.1f}&deg;<br>w0_err={post_nm_w0_err[i]:.1f}&deg;"
              for i in range(n)],
        hoverinfo="text", showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(title="n_matched (peaks)", row=1, col=1)
    fig.update_yaxes(title="lofi MSE", row=1, col=1, type="log")

    fig.add_trace(go.Scatter(
        x=align_cost, y=lofi_mse, mode="markers",
        marker=dict(size=6, color=post_nm_q0_err, colorscale="Plasma_r", cmin=0, cmax=180),
        text=[f"idx {i}<br>align={align_cost[i]:.3e}<br>lofi_mse={lofi_mse[i]:.3f}"
              for i in range(n)], hoverinfo="text", showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title="alignment cost (log)", row=1, col=2, type="log")
    fig.update_yaxes(title="lofi MSE", row=1, col=2, type="log")

    fig.update_layout(
        title=stage_title("m103 Step 2b &mdash; Lofi peak-matching",
                          int(d["grid"]["top_k"]), n,
                          extra="(re-rank by n_matched then lofi_mse; truncate to NM_TOP=300)"),
        height=480, margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


# ── Panel 2.0b — Surrogate-cost grid (full ω-direction sphere) ───────────

def panel_surrogate_grid(d):
    """If a grid_surr_ckpt is present, plot 2000 fibonacci ω-direction
    candidates colored by min surrogate-MSE (over phi & |ω|). Side-by-side
    with the same Mollweide projection of the alignment-cost top-500 for
    direct comparison."""
    g = d["grid"]
    gs = d["grid_surr"]
    if gs is None:
        return None
    omega_dirs = np.asarray(gs["omega_dirs"])
    min_mse = np.asarray(gs["min_surr_mse"])
    finite = np.isfinite(min_mse)
    n_dirs = len(omega_dirs)
    log_mse = np.log10(np.maximum(min_mse, 1e-30))

    true_w = d["truth"]["omega0_rad"]
    true_w_hat = true_w / np.linalg.norm(true_w)

    def to_lonlat(vecs):
        lon = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(vecs[:, 2], -1, 1)))
        return lon, lat

    # rank-1 by surrogate
    rank1 = int(np.argmin(np.where(finite, min_mse, np.inf)))
    offset = float(np.degrees(np.arccos(np.clip(
        float(omega_dirs[rank1] @ true_w_hat), -1, 1))))

    # alignment-cost equivalent: m103's grid_top500 (top-500 only)
    align_dirs = g["top_omegas"] / np.linalg.norm(g["top_omegas"], axis=1, keepdims=True)
    align_costs = np.asarray(g["top_costs"])
    log_align = np.log10(np.maximum(align_costs, 1e-30))

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        subplot_titles=(
            "<b>m103 alignment cost</b> &mdash; top-500 of 2000 grid (color = log10 cost)",
            "<b>Surrogate full-LC MSE</b> &mdash; ALL 2000 grid directions (color = log10 cost)",
        ),
        horizontal_spacing=0.06,
    )
    # Left: alignment cost top-500 (gray bg = rejected from top-500)
    full_grid = _fibonacci_sphere(2000)
    nn_idx = set()
    for sd in align_dirs:
        nn = int(np.argmax(full_grid @ sd))
        nn_idx.add(nn)
    not_selected = np.array([i for i in range(2000) if i not in nn_idx])
    lon_g, lat_g = to_lonlat(full_grid[not_selected])
    lon_a, lat_a = to_lonlat(align_dirs)
    lon_t, lat_t = to_lonlat(true_w_hat[None, :])
    fig.add_trace(go.Scatter(x=lon_g, y=lat_g, mode="markers",
                             marker=dict(size=3, color="#888", opacity=0.4),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lon_a, y=lat_a, mode="markers",
        marker=dict(size=7, color=log_align, colorscale="Viridis",
                    showscale=False, line=dict(color="black", width=0.3)),
        hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=lon_t, y=lat_t, mode="markers",
                             marker=dict(size=24, color="rgba(0,0,0,0)",
                                         line=dict(color="red", width=3)),
                             name="truth", showlegend=False), row=1, col=1)

    # Right: surrogate cost — all 2000 dirs, colored
    lon_s, lat_s = to_lonlat(omega_dirs)
    fig.add_trace(go.Scatter(
        x=lon_s, y=lat_s, mode="markers",
        marker=dict(size=6, color=log_mse, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="log10<br>surr MSE",
                                                  x=1.0, len=0.85),
                    line=dict(color="black", width=0.2)),
        text=[f"dir {i}<br>min surr MSE={min_mse[i]:.3f}" for i in range(n_dirs)],
        hoverinfo="text", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=lon_t, y=lat_t, mode="markers",
                             marker=dict(size=24, color="rgba(0,0,0,0)",
                                         line=dict(color="red", width=3)),
                             name="truth", showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[float(np.degrees(np.arctan2(omega_dirs[rank1, 1], omega_dirs[rank1, 0])))],
        y=[float(np.degrees(np.arcsin(np.clip(omega_dirs[rank1, 2], -1, 1))))],
        mode="markers",
        marker=dict(size=22, color="rgba(0,0,0,0)",
                    line=dict(color="lime", width=3)),
        name=f"rank-1 (offset {offset:.2f}°)", showlegend=True,
    ), row=1, col=2)

    for col in (1, 2):
        fig.update_xaxes(title="lon (deg)", range=[-185, 185], dtick=60,
                         row=1, col=col, gridcolor="#eee")
        fig.update_yaxes(title="lat (deg)" if col == 1 else "",
                         range=[-92, 92], dtick=30, row=1, col=col,
                         gridcolor="#eee")

    fig.update_layout(
        title=stage_title("Surrogate-cost grid sweep &mdash; 2000 fibonacci &omega; directions",
                          n_dirs, n_dirs,
                          extra=f"&nbsp;|&nbsp; rank-1 surr offset from truth: <b>{offset:.2f}&deg;</b>"
                                f"&nbsp;|&nbsp; min surr MSE: {np.nanmin(min_mse):.3f}"
                                f"&nbsp;|&nbsp; |&omega;| sweep: {len(gs['omega_mags_searched'])} mags "
                                f"(truth-mag included: {bool(gs['include_truth_mag'])})"),
        height=480, margin=dict(t=100, b=40, l=60, r=10),
        legend=dict(x=0.55, y=1.02, orientation="h"),
    )
    return fig


# ── Panel 2.5 — Cost-landscape comparison ────────────────────────────────

def panel_cost_landscape(d):
    """Same 300 lofi candidates, colored under 4 different cost functions, on
    a 2D Mollweide-style projection of their omega-direction. Lets you read
    off which cost localizes truth vs which cost has spurious low-cost
    basins."""
    lofi = d["lofi"]
    nm = d["nm"]
    n = int(lofi["n"])
    w0 = lofi["w0"]
    w0_hat = w0 / np.linalg.norm(w0, axis=1, keepdims=True)
    align = np.asarray(lofi["align_cost"])
    lofi_mse = np.asarray(lofi["lofi_mse"])
    post_nm_q0_err = np.asarray(nm["q0_err"])
    surr = d["lofi_surr"]
    has_surr = surr is not None
    surr_mse = np.asarray(surr["surr_mse"]) if has_surr else None

    true_w = d["truth"]["omega0_rad"]
    true_w_hat = true_w / np.linalg.norm(true_w)

    def to_lonlat(vecs):
        lon = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(vecs[:, 2], -1, 1)))
        return lon, lat

    lon, lat = to_lonlat(w0_hat)
    lon_t, lat_t = to_lonlat(true_w_hat[None, :])

    # Build column list dynamically
    panels = [
        ("alignment cost (m103 grid score)", np.log10(np.maximum(align, 1e-30))),
        ("full-LC MSE (lofi_mse — m103 has it but does NOT rank by it!)",
         np.log10(np.maximum(lofi_mse, 1e-30))),
    ]
    if has_surr:
        panels.append(("surrogate full-LC MSE",
                       np.log10(np.maximum(surr_mse, 1e-30))))
    panels.append(("post-NM q0_err (oracle reference)", post_nm_q0_err))

    n_cols = len(panels)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"<b>{name}</b>" for name, _ in panels],
        horizontal_spacing=0.05,
    )

    # Identify rank-1 (lowest-cost / lowest-error) candidate per cost
    summaries = []
    for col, (name, vals) in enumerate(panels, start=1):
        rank1 = int(np.argmin(vals))
        offset_truth_deg = float(np.degrees(np.arccos(np.clip(
            float(w0_hat[rank1] @ true_w_hat), -1, 1))))
        # plot
        fig.add_trace(go.Scatter(
            x=lon, y=lat, mode="markers",
            marker=dict(
                size=8,
                color=vals,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="log10(cost)" if "q0_err" not in name else "deg",
                    x=col / n_cols - 0.04, len=0.85,
                ),
                line=dict(width=0.3, color="black"),
            ),
            text=[f"idx {i}<br>cost/val={vals[i]:.3e}<br>"
                  f"post-NM q0_err={post_nm_q0_err[i]:.1f}&deg;"
                  for i in range(n)],
            hoverinfo="text", showlegend=False,
        ), row=1, col=col)
        # rank-1 marker
        fig.add_trace(go.Scatter(
            x=[lon[rank1]], y=[lat[rank1]], mode="markers",
            marker=dict(size=22, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color="lime", width=3)),
            name="rank-1 (best by this cost)",
            showlegend=(col == 1), hoverinfo="text",
            text=[f"rank-1: idx {rank1}<br>offset from truth = {offset_truth_deg:.2f}&deg;"],
        ), row=1, col=col)
        # truth ring
        fig.add_trace(go.Scatter(
            x=lon_t, y=lat_t, mode="markers",
            marker=dict(size=24, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color="red", width=3)),
            name="truth", showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=lon_t, y=lat_t, mode="markers",
            marker=dict(size=8, color="red", symbol="cross"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)
        fig.update_xaxes(range=[-185, 185], row=1, col=col, dtick=60,
                         gridcolor="#eee")
        fig.update_yaxes(range=[-92, 92], row=1, col=col, dtick=30,
                         gridcolor="#eee")
        if col == 1:
            fig.update_yaxes(title="lat (deg)", row=1, col=col)
        fig.update_xaxes(title="lon (deg)", row=1, col=col)
        summaries.append((name, rank1, offset_truth_deg, vals[rank1]))

    summary_html = "<br>".join(
        f"<b>{name}</b>: rank-1 is candidate {r1}, "
        f"<b>{off:.1f}&deg; offset from truth</b> "
        f"(value={val:.3e})"
        for name, r1, off, val in summaries
    )
    fig.update_layout(
        title=("<b>Cost-landscape comparison &mdash; same 300 lofi (q0, &omega;) candidates, four cost functions</b>"
               f"<br><sub>{summary_html}</sub>"),
        height=520, margin=dict(t=140, b=40, l=60, r=10),
        legend=dict(x=0.5, y=1.02, orientation="h"),
    )
    return fig


# ── Panel 3 — m103 Step 3 NM polish ──────────────────────────────────────

def panel_nm(d):
    nm = d["nm"]
    n = int(nm["n"])
    q0_err = nm["q0_err"]
    w0_err = nm["w0_err"]
    refined_costs = nm["refined_costs"]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=("(q0_err, w_dir_err) post-NM &mdash; color = refined_cost",
                        "Refined cost ranking"),
    )
    fig.add_trace(go.Scatter(
        x=q0_err, y=w0_err, mode="markers",
        marker=dict(
            size=7, color=np.log10(np.maximum(refined_costs, 1e-30)),
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="log10<br>cost", x=0.5, len=0.85),
        ),
        text=[f"idx {i}<br>q0_err={q0_err[i]:.1f}&deg;<br>w_dir_err={w0_err[i]:.1f}&deg;<br>"
              f"cost={refined_costs[i]:.3e}" for i in range(n)],
        hoverinfo="text", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", marker=dict(size=18, color="red", symbol="x"),
        name="truth", showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(title="q0_err (deg)", row=1, col=1)
    fig.update_yaxes(title="w_dir_err (deg)", row=1, col=1)

    sorted_idx = np.argsort(refined_costs)
    fig.add_trace(go.Scatter(
        x=np.arange(n), y=refined_costs[sorted_idx], mode="markers",
        marker=dict(size=4, color=q0_err[sorted_idx], colorscale="Plasma_r",
                    cmin=0, cmax=180,
                    colorbar=dict(title="q0_err", x=1.0, len=0.85)),
        text=[f"rank {r}<br>orig idx {int(sorted_idx[r])}<br>cost={refined_costs[sorted_idx[r]]:.3e}"
              f"<br>q0_err={q0_err[sorted_idx[r]]:.1f}&deg;<br>"
              f"w0_err={w0_err[sorted_idx[r]]:.1f}&deg;" for r in range(n)],
        hoverinfo="text", showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title="rank by refined_cost", row=1, col=2)
    fig.update_yaxes(title="cost (log)", row=1, col=2, type="log")

    truth_close = int(np.sum((q0_err < 30) & (w0_err < 10)))
    fig.update_layout(
        title=stage_title(
            "m103 Step 3 &mdash; Nelder-Mead polish &amp; pre-dedup",
            n, n,
            extra=f"&nbsp;|&nbsp; truth-close (q0&lt;30&deg;, w0&lt;10&deg;): {truth_close}/{n}",
        ),
        height=480, margin=dict(t=80, b=40, l=60, r=70),
    )
    return fig


# ── Panel 4 — m103 Step 3.5 phi sweep ────────────────────────────────────

def panel_phi_sweeps(d):
    ps = d["phi_sweeps"]
    n_sweeps = int(ps["n"])
    fig = make_subplots(
        rows=1, cols=n_sweeps,
        subplot_titles=[f"&omega;-rank {int(ps['omega_rank'][i])} (normal_ni={int(ps['normal_ni'][i])})"
                        for i in range(n_sweeps)],
    )
    for i in range(n_sweeps):
        phi_arr = np.asarray(ps["phi_arr_rad"][i], dtype=np.float64)
        phi_costs = np.asarray(ps["phi_costs"][i], dtype=np.float64)
        orig_idx = int(ps["original_phi_idx"][i])
        sel_idx = np.asarray(ps["selected_phi_indices"][i], dtype=int)
        phi_deg = np.degrees(phi_arr)
        fig.add_trace(go.Scatter(
            x=phi_deg, y=phi_costs, mode="lines+markers",
            line=dict(color="steelblue", width=1),
            marker=dict(size=3, color="steelblue"),
            name="cost", showlegend=False,
        ), row=1, col=i+1)
        fig.add_trace(go.Scatter(
            x=[phi_deg[orig_idx]], y=[phi_costs[orig_idx]], mode="markers",
            marker=dict(size=14, color="orange", symbol="circle-open", line=dict(width=3)),
            name="original (NM)", showlegend=(i == 0),
        ), row=1, col=i+1)
        fig.add_trace(go.Scatter(
            x=phi_deg[sel_idx], y=phi_costs[sel_idx], mode="markers",
            marker=dict(size=10, color="red", symbol="diamond"),
            name="selected", showlegend=(i == 0),
        ), row=1, col=i+1)
        fig.update_xaxes(title="phi (deg)", row=1, col=i+1)
        fig.update_yaxes(title="alignment cost (log)" if i == 0 else "",
                         type="log", row=1, col=i+1)

    fig.update_layout(
        title=stage_title("m103 Step 3.5 &mdash; Multi-phi expansion (top-2 &omega; candidates)",
                          2, "2&times;~3",
                          extra="(N_PHI_PER_OMEGA=4 with 20&deg; min separation)"),
        height=380, margin=dict(t=80, b=40, l=60, r=20), legend=dict(x=0.5, y=1.08),
    )
    return fig


# ── Panel 5 — m103 Step 4 geo ranking + reranking ────────────────────────

def panel_geo_ranking(d):
    geo = d["geo"]
    n = int(geo["n_candidates"])
    geo_costs = geo["geo_costs"]
    q0_err = geo["q0_ref_errs"]
    w0_err = geo["w0_ref_errs"]
    truth_dist = q0_err + w0_err  # composite for color

    has_q0p = d["q0polish"] is not None
    if has_q0p:
        q0p_mse = np.asarray(d["q0polish"]["surr_q0polish_mse"])

    fig = make_subplots(
        rows=1, cols=2 if has_q0p else 1,
        subplot_titles=(["Sorted by <b>geo_cost</b> (legacy ranker)",
                         "Sorted by <b>surr_q0polish_mse</b> (m133 / latest)"] if has_q0p else
                        ["Sorted by geo_cost"]),
    )

    def _draw(col, vals, title):
        order = np.argsort(vals)
        ranks = np.arange(n)
        colors = ["#1f77b4"] * n
        for r in range(min(3, n)):
            colors[r] = "#d62728"
        bars_q = q0_err[order]
        bars_w = w0_err[order]
        fig.add_trace(go.Bar(
            x=ranks, y=vals[order], marker_color=colors,
            text=[f"q0={bars_q[r]:.0f}&deg;<br>w_dir={bars_w[r]:.1f}&deg;"
                  for r in range(n)],
            textposition="outside",
            hovertext=[f"orig idx {int(order[r])}<br>{title}={vals[order[r]]:.3e}<br>"
                       f"q0_err={bars_q[r]:.1f}&deg;<br>w_dir_err={bars_w[r]:.1f}&deg;"
                       for r in range(n)],
            hoverinfo="text", showlegend=False,
        ), row=1, col=col)
        fig.update_xaxes(title="rank", row=1, col=col)
        fig.update_yaxes(title=title, row=1, col=col, type="log")

    _draw(1, geo_costs, "geo_cost")
    if has_q0p:
        _draw(2, q0p_mse, "surr_q0polish_mse")

    n_truth_close = int(np.sum((q0_err < 30) & (w0_err < 10)))
    fig.update_layout(
        title=stage_title(
            "m103 Step 4 &mdash; Geo refinement (alignment-cost L-BFGS-B) + m133 reranking",
            "26 deduped", "top-3 &rarr; m115",
            extra=f"&nbsp;|&nbsp; truth-close (q0&lt;30&deg;, w0&lt;10&deg;): {n_truth_close}/{n}",
        ),
        height=480, margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


# ── Panel 6 — m115 DE convergence ────────────────────────────────────────

def panel_de_convergence(d):
    de = d["de_history"]
    omega_idx = de["omega_idx"]
    start_idx = de["start_idx"]
    trace_cost = de["trace_cost"]
    final_cost = de["final_cost"]
    n_omega = int(omega_idx.max() + 1) if len(omega_idx) else 0

    fig = make_subplots(
        rows=1, cols=n_omega,
        subplot_titles=[f"&omega; seed {ic}" for ic in range(n_omega)],
        shared_yaxes=True,
    )
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for run in range(len(omega_idx)):
        ic = int(omega_idx[run]); si = int(start_idx[run])
        tc = np.asarray(trace_cost[run])
        if len(tc) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=np.arange(len(tc)), y=tc, mode="lines",
            line=dict(color=palette[si % len(palette)], width=1),
            name=f"start {si}" if ic == 0 else None,
            showlegend=(ic == 0),
            hovertext=[f"&omega; {ic}, start {si}, gen {g}, cost={c:.3e}"
                       for g, c in enumerate(tc)],
            hoverinfo="text",
        ), row=1, col=ic+1)
        fig.add_trace(go.Scatter(
            x=[len(tc)-1], y=[final_cost[run]], mode="markers",
            marker=dict(size=8, color=palette[si % len(palette)],
                        symbol="circle", line=dict(color="black", width=1)),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=ic+1)
    for ic in range(n_omega):
        fig.update_xaxes(title="DE generation", row=1, col=ic+1)
        if ic == 0:
            fig.update_yaxes(title="surrogate MSE (log)", type="log", row=1, col=ic+1)
        else:
            fig.update_yaxes(type="log", row=1, col=ic+1)

    fig.update_layout(
        title=stage_title("m115 Step 1 &mdash; Multi-start DE per &omega; seed",
                          f"{n_omega} omegas &times; 10 starts",
                          f"{int(de['n_runs'])} runs",
                          extra=f"(maxiter={int(de['de_maxiter'])}, popsize={int(de['de_popsize'])})"),
        height=420, margin=dict(t=80, b=40, l=60, r=20),
        legend=dict(x=1.02, y=1.0),
    )
    return fig


# ── Panel 7 — m115 hi-fi validation LCs ──────────────────────────────────

def panel_hifi_validation(d):
    truth_lc = d["truth"]["mag_hifi"]
    obs_lc = d["truth"]["observed_lc"]
    s2 = d["step2_hifi"]
    hifi_mags = s2["hifi_mags"]  # (3, 500)
    n_basins = hifi_mags.shape[0]
    info = json.loads(str(s2["hifi_json"]))
    t = np.arange(len(truth_lc))

    fig = make_subplots(
        rows=1, cols=n_basins,
        subplot_titles=[f"Basin {ib}: hifi={info[ib]['hifi_mse']:.4f}, "
                        f"q0_err={info[ib]['q0_err']:.1f}&deg;, "
                        f"w_dir={info[ib]['w_dir_err']:.1f}&deg;"
                        for ib in range(n_basins)],
        shared_yaxes=True,
    )
    for ib in range(n_basins):
        fig.add_trace(go.Scatter(x=t, y=truth_lc, mode="lines",
                                 line=dict(color="black", width=1.5),
                                 name="truth", showlegend=(ib == 0)),
                      row=1, col=ib+1)
        fig.add_trace(go.Scatter(x=t, y=obs_lc, mode="lines",
                                 line=dict(color="gray", width=0.5, dash="dot"),
                                 name="observed", showlegend=(ib == 0), opacity=0.5),
                      row=1, col=ib+1)
        fig.add_trace(go.Scatter(x=t, y=hifi_mags[ib], mode="lines",
                                 line=dict(color="#d62728", width=1),
                                 name=f"basin {ib} pred", showlegend=(ib == 0)),
                      row=1, col=ib+1)
        fig.update_xaxes(title="epoch", row=1, col=ib+1)
        fig.update_yaxes(autorange="reversed",
                         title="mag" if ib == 0 else "", row=1, col=ib+1)
    fig.update_layout(
        title=stage_title("m115 Step 2 &mdash; Hi-fi validation of top-3 basins",
                          "3 basins", "3 LCs"),
        height=380, margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


# ── Panel 8 — m126 polish ────────────────────────────────────────────────

def panel_polish(d):
    pol = d["polish"]
    traj_x = pol["traj_x"]
    traj_cost = pol["traj_cost"]
    n_basins = int(pol["n_basins"])

    fig = make_subplots(
        rows=2, cols=n_basins,
        row_heights=[0.6, 0.4],
        subplot_titles=([f"Basin {bi} cost vs L-BFGS-B iter" for bi in range(n_basins)] +
                        [f"Basin {bi} errors before/after" for bi in range(n_basins)]),
        vertical_spacing=0.18,
    )
    for bi in range(n_basins):
        tc = np.asarray(traj_cost[bi])
        fig.add_trace(go.Scatter(
            x=np.arange(len(tc)), y=tc, mode="lines+markers",
            line=dict(color="#1f77b4", width=1.5), marker=dict(size=4),
            showlegend=False,
        ), row=1, col=bi+1)
        fig.update_xaxes(title="iter", row=1, col=bi+1)
        fig.update_yaxes(title="surr MSE (log)" if bi == 0 else "",
                         type="log", row=1, col=bi+1)
        labels = ["q0_err", "w_dir_err", "|w_mag_err|"]
        before = [pol["q0_err_before"][bi], pol["w_dir_err_before"][bi],
                  abs(pol["w_mag_err_pct_before"][bi])]
        after = [pol["q0_err_after"][bi], pol["w_dir_err_after"][bi],
                 abs(pol["w_mag_err_pct_after"][bi])]
        fig.add_trace(go.Bar(x=labels, y=before, name="before",
                             marker_color="#7f7f7f",
                             showlegend=(bi == 0)), row=2, col=bi+1)
        fig.add_trace(go.Bar(x=labels, y=after, name="after",
                             marker_color="#2ca02c",
                             showlegend=(bi == 0)), row=2, col=bi+1)
        fig.update_yaxes(type="log", row=2, col=bi+1)

    fig.update_layout(
        title=stage_title("m126 &mdash; L-BFGS-B polish (6-DOF tangent-space) per basin",
                          "3 basins", "3 polished states"),
        height=560, margin=dict(t=80, b=40, l=60, r=20), barmode="group",
    )
    return fig


# ── Panel 9 — Final winner LC overlay ────────────────────────────────────

def panel_final(d):
    truth_lc = d["truth"]["mag_hifi"]
    obs_lc = d["truth"]["observed_lc"]
    hifi = d["hifi"]
    bi = best_basin_idx(d)
    before = hifi["hifi_mags_before"][bi]
    after = hifi["hifi_mags_after"][bi]
    t = np.arange(len(truth_lc))

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.65, 0.35],
        subplot_titles=("Best basin LC: truth vs m115 (before polish) vs m126 (after polish)",
                        "Residuals (predicted &minus; truth)"),
        vertical_spacing=0.15,
    )
    fig.add_trace(go.Scatter(x=t, y=truth_lc, mode="lines",
                             line=dict(color="black", width=1.5), name="truth"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=obs_lc, mode="lines",
                             line=dict(color="gray", width=0.5, dash="dot"),
                             name="observed (noisy)", opacity=0.5),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=before, mode="lines",
                             line=dict(color="#7f7f7f", width=1.0, dash="dash"),
                             name=f"m115 before (hifi={hifi['hifi_before'][bi]:.4f})"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=after, mode="lines",
                             line=dict(color="orange", width=1.5),
                             name=f"m126 after (hifi={hifi['hifi_after'][bi]:.4f})"),
                  row=1, col=1)
    fig.update_yaxes(autorange="reversed", title="mag", row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=before - truth_lc, mode="lines",
                             line=dict(color="#7f7f7f", width=1.0),
                             name="m115 res", showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=after - truth_lc, mode="lines",
                             line=dict(color="orange", width=1.5),
                             name="m126 res", showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)
    fig.update_xaxes(title="epoch", row=2, col=1)
    fig.update_yaxes(title="residual (mag)", row=2, col=1)

    fig.update_layout(
        title=stage_title("Final &mdash; m126 winner LC overlay",
                          1, 1, extra=f"(basin {bi})"),
        height=560, margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


# ── Dashboard assembly ───────────────────────────────────────────────────

def build_html(d, panels, out_path):
    seed = d["seed"]
    source = d["source"]
    title = f"Pipeline visualisation — seed {seed} ({source})"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "       margin: 12px 24px; color: #222; }",
        "h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 4px; }",
        ".panel { margin: 8px 0 24px 0; padding: 6px; border-top: 1px solid #ddd; }",
        ".panel h2 { font-size: 14px; color: #555; margin: 4px 0; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p style='color:#888;font-size:12px'>Generated by notebooks/inversion/pipeline_viz.py</p>",
    ]
    for i, fig in enumerate(panels):
        parts.append(f"<div class='panel' id='panel_{i}'>")
        parts.append(pio.to_html(fig, include_plotlyjs=False,
                                 full_html=False, div_id=f"plot_{i}"))
        parts.append("</div>")
    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default: data/.../pipeline_viz/seed_NNN.html)")
    args = ap.parse_args()

    d = load_all(args.seed, args.traj_source)
    panels = [
        panel_header(d),
        panel_grid_sweep(d),
    ]
    p_sg = panel_surrogate_grid(d)
    if p_sg is not None:
        panels.append(p_sg)
    panels += [
        panel_cost_landscape(d),
        panel_lofi(d),
        panel_nm(d),
        panel_phi_sweeps(d),
        panel_geo_ranking(d),
        panel_de_convergence(d),
        panel_hifi_validation(d),
        panel_polish(d),
        panel_final(d),
    ]
    out = (Path(args.out) if args.out else
           DIAG / "pipeline_viz" / f"seed_{args.seed:03d}.html")
    build_html(d, panels, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
