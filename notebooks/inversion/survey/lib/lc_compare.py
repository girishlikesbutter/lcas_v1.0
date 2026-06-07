"""LC comparison — N comparison light curves vs 1 reference, with N residuals.

Survey-local replacement for `notebooks/inversion/lib/lc_compare.py`. Uses
the post-fix forward model exclusively (`lib.hifi_render` → `src.dynamics.
attitude_propagator` post-fix). Per the workspace contract (CLAUDE.md), no
imports from `notebooks/inversion/lib/`.

Each LC source is one of:
    traj:SEED              cached truth `mag_hifi` from data/trajectories/
    traj:SEED:lofi         cached truth `mag_lofi`
    npy:PATH               1-D numpy array on disk
    npz:PATH:KEY           1-D array stored under KEY in an .npz
    npz:PATH:KEY:I         row I of a 2-D array under KEY
    state:SEED:Qw,Qx,Qy,Qz:Wx,Wy,Wz
                           propagate (q0_wxyz, ω_rad) at SEED's cached
                           geometry, render hi-fi (post-fix). Cached at
                           caller-supplied --cache-dir if given.

Output: PNG with reference + N comparisons overlaid on top, then N
residual subplots (cmp_i − ref).

CLI:
    python -m lib.lc_compare \
        --ref      "TRUTH=traj:89" \
        --cmp      "Polish 1=npz:results/s058.../polished_states.npz:pred_hifi:5" \
        --cmp      "Polish 2=state:89:0.1,0.2,0.3,0.9:1e-3,2e-3,3e-3" \
        --output   /tmp/lc_compare.png \
        --title    "Demo"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

_SCHEME_RE = re.compile(r"=(traj|npy|npz|state):")

# Fixed comparison-line colour cycle: blue, red, green, then extras. Reference
# is always rendered as black dots so it can't be confused with a comparison.
_CMP_COLORS = (
    "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray",
)
_REF_COLOR = "black"
_REF_MARKERSIZE = 1.0
_CMP_LW = 0.7
_DPI = 300

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SURVEY = Path(__file__).resolve().parent.parent
if str(SURVEY) not in sys.path:
    sys.path.insert(0, str(SURVEY))

from lib.traj_load import load_truth as _load_traj
from lib.hifi_render import build_context, render_hifi


def _parse_state_spec(parts: list[str]) -> tuple[int, np.ndarray, np.ndarray]:
    """Parse `SEED:Qw,Qx,Qy,Qz:Wx,Wy,Wz` form."""
    if len(parts) != 3:
        raise ValueError(
            "state spec needs SEED:Qw,Qx,Qy,Qz:Wx,Wy,Wz (got "
            f"{len(parts)} colon-separated fields)"
        )
    seed = int(parts[0])
    q = np.array([float(v) for v in parts[1].split(",")], dtype=np.float64)
    om = np.array([float(v) for v in parts[2].split(",")], dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"q0 must be length 4 (wxyz); got {q.shape}")
    if om.shape != (3,):
        raise ValueError(f"omega must be length 3 (rad/s); got {om.shape}")
    return seed, q, om


def _resolve_state(
    seed: int, q0: np.ndarray, om: np.ndarray,
    spec: str, cache_dir: Optional[Path],
    ctx_cache: dict,
) -> np.ndarray:
    """Render hi-fi for (q0, ω). Caches by spec hash if cache_dir given."""
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        h = abs(hash(spec)) % (10 ** 12)
        cache_path = cache_dir / f"state_seed{seed:03d}_{h}.npy"
        if cache_path.exists():
            print(f"  cache hit: {cache_path.name}")
            return np.load(cache_path)
    if seed not in ctx_cache:
        print(f"  building hi-fi context for seed {seed}…")
        ctx_cache[seed] = build_context(seed=seed)
    print(f"  rendering hi-fi (q0={q0.tolist()}, ω={om.tolist()})…")
    pred = render_hifi(q0, om, ctx_cache[seed])
    if cache_path is not None:
        np.save(cache_path, pred)
        print(f"  cached: {cache_path.name}")
    return pred


def resolve_lc(
    spec: str, ctx_cache: dict, cache_dir: Optional[Path]
) -> np.ndarray:
    """Resolve an LC spec to a 1-D numpy array of magnitudes."""
    if spec.startswith("traj:"):
        parts = spec.split(":")
        seed = int(parts[1])
        kind = parts[2] if len(parts) > 2 else "hifi"
        d = _load_traj(seed)
        key = "mag_hifi" if kind == "hifi" else "mag_lofi"
        return np.asarray(d[key], dtype=np.float64)
    if spec.startswith("npy:"):
        path = Path(spec[len("npy:"):]).expanduser()
        return np.asarray(np.load(path), dtype=np.float64)
    if spec.startswith("npz:"):
        rest = spec[len("npz:"):]
        parts = rest.split(":")
        if len(parts) < 2:
            raise ValueError("npz spec needs npz:PATH:KEY[:I]")
        path = Path(parts[0]).expanduser()
        key = parts[1]
        z = np.load(path, allow_pickle=True)
        arr = np.asarray(z[key])
        if len(parts) >= 3:
            arr = arr[int(parts[2])]
        return np.asarray(arr, dtype=np.float64)
    if spec.startswith("state:"):
        rest = spec[len("state:"):]
        seed, q0, om = _parse_state_spec(rest.split(":"))
        return _resolve_state(seed, q0, om, spec, cache_dir, ctx_cache)
    raise ValueError(
        f"unknown spec scheme in {spec!r}; use traj:/npy:/npz:/state:"
    )


def parse_label_spec(arg: str) -> tuple[str, str]:
    """Split 'LABEL=SCHEME:...'.

    Splits on the first '=' immediately followed by a known scheme prefix
    (`traj:`, `npy:`, `npz:`, `state:`). This lets labels contain literal
    '=' characters (e.g. 'cluster id=44') without escaping.
    """
    m = _SCHEME_RE.search(arg)
    if m is None:
        raise ValueError(
            f"argument must be LABEL=SCHEME:... where SCHEME ∈ "
            "{traj, npy, npz, state}. Got: " + repr(arg)
        )
    label = arg[: m.start()].strip()
    spec = arg[m.start() + 1:].strip()
    return label, spec


def make_plot(
    ref_label: str, ref_lc: np.ndarray,
    cmps: list[tuple[str, np.ndarray]],
    output: Path, title: Optional[str] = None,
    rho_noise_sigma: float = 0.05,
) -> None:
    """N+1 stacked panels: overlay row + N residual rows."""
    n = len(cmps)
    n_ep = len(ref_lc)
    epoch = np.arange(n_ep)

    fig_h = 3.0 + 1.6 * n
    fig, axes = plt.subplots(
        n + 1, 1, figsize=(13, fig_h),
        gridspec_kw={"height_ratios": [3] + [1] * n},
        sharex=True,
    )
    if n + 1 == 1:
        axes = [axes]

    def _cmp_color(i: int) -> str:
        return _CMP_COLORS[i % len(_CMP_COLORS)]

    ax = axes[0]
    # Reference: dots only, no line. Black so it never collides with a
    # comparison colour.
    ax.plot(epoch, ref_lc, ".", color=_REF_COLOR, markersize=_REF_MARKERSIZE,
            label=ref_label, alpha=0.9, zorder=5)
    for i, (lab, lc) in enumerate(cmps):
        if len(lc) != n_ep:
            print(f"  WARNING: '{lab}' length {len(lc)} != ref {n_ep}; truncating")
            m = min(n_ep, len(lc))
            lc_used = lc[:m]
            ax.plot(np.arange(m), lc_used, "-", color=_cmp_color(i),
                    lw=_CMP_LW, alpha=0.9, label=lab, zorder=3)
            cmps[i] = (lab, lc_used)
            continue
        rms = float(np.sqrt(np.mean((lc - ref_lc) ** 2)))
        rho = rms / rho_noise_sigma
        ax.plot(epoch, lc, "-", color=_cmp_color(i), lw=_CMP_LW, alpha=0.9,
                label=f"{lab}  (RMS={rms:.3f} mag, ρ={rho:.2f})", zorder=3)
    ax.invert_yaxis()
    ax.set_ylabel("magnitude")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    if title is not None:
        ax.set_title(title, fontsize=10)

    for i, (lab, lc) in enumerate(cmps):
        m = min(len(lc), len(ref_lc))
        residual = lc[:m] - ref_lc[:m]
        ax = axes[i + 1]
        ax.axhline(0, color="black", lw=0.5)
        ax.plot(np.arange(m), residual, "-", color=_cmp_color(i), lw=_CMP_LW)
        peak = max(2 * rho_noise_sigma, 1.2 * float(np.max(np.abs(residual))))
        ax.set_ylim(-peak, peak)
        ax.axhline(rho_noise_sigma, ls=":", color="grey", lw=0.5)
        ax.axhline(-rho_noise_sigma, ls=":", color="grey", lw=0.5)
        ax.set_ylabel(f"{lab}\n − ref (mag)", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("epoch index")
    plt.tight_layout()
    plt.savefig(output, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main(argv: Optional[list[str]] = None) -> Path:
    p = argparse.ArgumentParser(
        description="Plot N comparison LCs vs 1 reference LC, with N residuals.",
    )
    p.add_argument(
        "--ref", required=True, metavar="LABEL=SPEC",
        help="Reference LC. SPEC: traj:SEED | npy:PATH | npz:PATH:KEY[:I] "
             "| state:SEED:Qw,Qx,Qy,Qz:Wx,Wy,Wz",
    )
    p.add_argument(
        "--cmp", action="append", default=[], metavar="LABEL=SPEC",
        help="Comparison LC (repeatable). Same SPEC scheme as --ref.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output PNG path (parent directory created if missing).",
    )
    p.add_argument(
        "--title", default=None, help="Plot title.",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Optional dir for caching freshly propagated state:* LCs.",
    )
    args = p.parse_args(argv)

    if not args.cmp:
        p.error("at least one --cmp is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ctx_cache: dict[int, dict] = {}

    print(f"Resolving reference: {args.ref}")
    ref_label, ref_spec = parse_label_spec(args.ref)
    ref_lc = resolve_lc(ref_spec, ctx_cache, args.cache_dir)
    print(f"  → '{ref_label}'  shape={ref_lc.shape}")

    cmps: list[tuple[str, np.ndarray]] = []
    for c in args.cmp:
        print(f"\nResolving comparison: {c}")
        lab, spec = parse_label_spec(c)
        lc = resolve_lc(spec, ctx_cache, args.cache_dir)
        print(f"  → '{lab}'  shape={lc.shape}")
        cmps.append((lab, lc))

    print(f"\nMaking plot: ref + {len(cmps)} comparisons + {len(cmps)} residuals")
    make_plot(
        ref_label, ref_lc, cmps,
        output=args.output, title=args.title,
    )
    return args.output


if __name__ == "__main__":
    main()
