---
name: lc-compare
description: "Plot N comparison light curves against a single reference light curve, with one residual subplot per comparison. Each LC source can be a cached truth, an .npy/.npz array, or an on-the-spot propagation of a (q0, ω) state. Use when the user asks to compare, overlay, or plot light curves. Triggers on: compare light curves, show me the lc, plot lc, lc comparison, light curve comparison, show the fit."
---

# Light Curve Comparison

Plot N comparison light curves against a single reference, with one residual subplot per comparison (`cmp_i − ref`). Each LC is supplied via a SOURCE spec — either loaded from disk if it exists, or propagated and rendered on the spot via the post-fix forward chain (`lib.hifi_render`).

## Where it lives

`notebooks/inversion/survey/lib/lc_compare.py`. The post-fix-validated forward model (`src.dynamics.attitude_propagator`) is reached through `lib.hifi_render`. **Do not** call the legacy `notebooks/inversion/lib/lc_compare.py` — that one threads through `notebooks/inversion/lib/experiment_setup.py` which is buggy-era-tainted.

## Source specs

A SOURCE spec resolves to a 1-D magnitude array. Schemes:

| spec | meaning |
|---|---|
| `traj:SEED` | cached truth `mag_hifi` from `survey/data/trajectories/traj_seedSEED.npz` |
| `traj:SEED:lofi` | cached truth `mag_lofi` |
| `npy:PATH` | 1-D array on disk |
| `npz:PATH:KEY` | 1-D array under `KEY` in an `.npz` |
| `npz:PATH:KEY:I` | row `I` of a 2-D array under `KEY` |
| `state:SEED:Qw,Qx,Qy,Qz:Wx,Wy,Wz` | propagate `(q0, ω_rad)` at SEED's cached SPICE/inertia geometry, render hi-fi |

`state:` LCs are optionally cached under `--cache-dir` keyed by spec hash so re-runs are free.

## CLI

```bash
python -m lib.lc_compare \
    --ref      "LABEL=SPEC" \
    --cmp      "LABEL=SPEC" \
    [--cmp     "LABEL=SPEC" ...] \
    --output   PATH/to/out.png \
    [--title   "Plot title"] \
    [--cache-dir PATH/to/cache]
```

Run from the survey workspace (`cd notebooks/inversion/survey`) so `lib.*` imports resolve.

## Plot layout

- Top panel (height ratio 3): reference + all comparisons overlaid. **Reference is rendered as small black dots (`markersize=1.0`, no line).** **Comparisons are thin lines (`lw=0.7`) cycling through `tab:blue, tab:red, tab:green, tab:orange, tab:purple, tab:brown, tab:pink, tab:olive, tab:cyan, tab:gray`** — in that order. Lines are thin enough that overlapping comparisons remain individually visible; dots are small enough not to mask comparison lines that match well. Magnitude axis inverted. Each comparison's legend label includes `RMS` and `ρ = RMS / 0.05`.
- N residual subplots beneath (height ratio 1 each): `cmp_i − ref`, drawn as a thin line in the same colour as the comparison's overlay; `±0.05 mag` noise floor shown as dotted grey lines.
- Shared x-axis on epoch index.
- **PNG saved at 300 DPI** so fine-grained mismatches between dots and lines remain visible at zoom.

The dots-vs-lines convention is **load-bearing**: it means the reference and comparisons are never confusable even when they overlap exactly. Don't change the convention without a reason.

## Examples

**Truth vs two on-the-spot propagations**

```bash
python -m lib.lc_compare \
    --ref "Truth seed 89=traj:89" \
    --cmp "Polished truth-cluster=state:89:-0.145,-0.960,-0.001,-0.241:-4.05e-3,5.37e-4,-9.06e-4" \
    --cmp "Body-twin candidate=state:89:0.241,-0.001,0.960,-0.145:-4.05e-3,-5.37e-4,9.06e-4" \
    --output results/lc_compare_demo.png \
    --title "Demo"
```

**Truth vs cached polished LCs from an NPZ**

```bash
python -m lib.lc_compare \
    --ref "Truth seed 89=traj:89" \
    --cmp "s058 truth basin (ρ=0.18)=npz:results/s058_lm_polish_clusters/polished_states.npz:pred_hifi:5" \
    --cmp "s058 multi-sol id=32 (ρ=0.17)=npz:results/s058_lm_polish_clusters/polished_states.npz:pred_hifi:4" \
    --output results/s058_lm_polish_clusters/lc_compare_band_A.png \
    --title "s058 Band-A candidates vs truth, seed 89"
```

## What to do

1. Identify the reference and comparison LCs the user wants. If they reference an experiment's polished states, the cached `pred_hifi` rows in the result NPZ are usually the right comparisons.
2. Build one `--cmp` arg per comparison; build a single `--ref`. Each label should be human-readable (`"Polished truth basin (ρ=0.18)"` — not `"cmp1"`).
3. Run from the survey workspace. Show the user the saved PNG path and the image.
