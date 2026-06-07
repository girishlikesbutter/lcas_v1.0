"""Render a self-contained animation HTML from a spread.npz + meta.json.

The template lives at `viewer_template.html`; this module embeds the data
payload as inline JSON (with binary arrays base64-encoded) and writes
`animation.html` next to the NPZ.

Encoding strategy:
- `rotvec_pool` is quantized to int16 over [-π, π] and base64-encoded
  (~600 KB for N=100k). Lossy at ~0.0001 rad ≈ 0.006° — far below
  the visual scale of the markers.
- `haze_indices`, `survivor_indices` are uint32 base64. Survivor index
  list is concatenated across epochs with an offset table.
- All other arrays (truth/closest/LC/etc.) ship as JSON floats (small).

Continuity correction (rotvec wraparound at |r|=π):
- truth_rotvec_per_epoch is **temporally unwrapped** so truth's trail is
  continuous past the boundary (can have |r| up to 2π).
- closest_pos_per_epoch is **anchored** per-frame to the unwrapped truth
  so the yellow marker sticks to truth.
- survivors are anchored per-frame in JS (to keep payload size bounded;
  see `viewer_template.html`).
- haze stays in canonical rotvec coords as a fixed background reference.

Side panels added 2026-05-07 (in-frame with the cloud animation):
- Articulated rest-frame mesh (body coords) — shipped once; rotated in JS
  per frame via R(q[t]). Component-color facecolors; mesh3d trace.
- ω(t) body-frame: |ω|(t) magnitude line plot + ω-direction point on a
  body-frame unit sphere with fading trail.
- Sun/observer unit vectors per sampled epoch in the inertial frame for
  the satellite mini's reference arrows.

The total HTML is typically ~3-15 MB depending on n_epochs and survivor
density (≤+200 KB after the side panels were added).
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
for _p in (str(PROJECT_ROOT), str(SURVEY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.c_t_pipeline import (  # noqa: E402
    anchor_rotvec_to,
    compute_j2000_units,
    temporal_unwrap_rotvec,
)
from lib.traj_load import load_truth  # noqa: E402

TEMPLATE_PATH = Path(__file__).resolve().parent / "viewer_template.html"
HAZE_SIZE = 10_000

# Articulation defaults — must match m048 generator + surrogate baseline.
# Cross-ref: lib.surrogate_eval (DEFAULT_SP_ANGLE_DEG=0, DEFAULT_AD_ANGLE_DEG=15)
# and lib.hifi_render.ART_ANGLES_DEG.
DEFAULT_SP_ANGLE_DEG = 0.0
DEFAULT_AD_ANGLE_DEG = 15.0

# Body-frame bounding-box half-extent (m) for vertex int16 quantization.
# Articulated mesh max |vertex| ≈ 15.55 (component body_pos translation
# moves SP / AD components out from origin). Pick 18 for headroom.
# Quantization step ≈ 18/32767 ≈ 5.5e-4 m, ~0.5 mm — invisible at the
# satellite mini's render scale.
MESH_BBOX_HALF = 18.0

# Component coloring. Indices into COMPONENT_COLORS in JS template.
# Order MUST stay in sync with viewer_template.html#COMPONENT_COLORS.
_COMPONENT_ORDER = ("Bus", "SP_North", "SP_South", "AD_East", "AD_West")


def _b64_int16_quantize_pi(arr: np.ndarray) -> str:
    """Quantize float arr in [-π, π] to int16 and return base64 string."""
    scaled = np.clip(np.asarray(arr, np.float64) / np.pi, -1.0, 1.0)
    q = np.rint(scaled * 32767.0).astype(np.int16)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _b64_int16_unit(arr: np.ndarray) -> str:
    """Quantize float arr in [-1, 1] to int16 (for unit vectors / quaternions)."""
    scaled = np.clip(np.asarray(arr, np.float64), -1.0, 1.0)
    q = np.rint(scaled * 32767.0).astype(np.int16)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _b64_int16_range(arr: np.ndarray, half_range: float) -> str:
    """Quantize float arr in [-half_range, +half_range] to int16."""
    scaled = np.clip(np.asarray(arr, np.float64) / float(half_range), -1.0, 1.0)
    q = np.rint(scaled * 32767.0).astype(np.int16)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _b64_uint32(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, np.uint32).tobytes()).decode("ascii")


def _b64_uint16(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, np.uint16).tobytes()).decode("ascii")


def _b64_uint8(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, np.uint8).tobytes()).decode("ascii")


def _round_array(arr: np.ndarray, ndigits: int = 4) -> list:
    return np.round(np.asarray(arr, float), ndigits).tolist()


# --------------------------------------------------------------------------
# Quaternion helpers (local — wxyz convention, scalar-first).
# --------------------------------------------------------------------------


def _qmul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    wb, xb, yb, zb = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    return np.stack(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        axis=-1,
    )


def _qconj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., 1:] *= -1
    return out


def compute_omega_body(
    quats_wxyz: np.ndarray, observation_times: np.ndarray
) -> np.ndarray:
    """Body-frame ω over the full trajectory via finite-difference.

    ω is in rad/s. For step t→t+1: q_rel = q[t]^-1 ⊗ q[t+1] (body frame).
    Length is matched to the input by repeating the final ω.
    """
    n = int(len(quats_wxyz))
    if n < 2:
        return np.zeros((n, 3), dtype=np.float64)
    qa = np.asarray(quats_wxyz[:-1], np.float64)
    qb = np.asarray(quats_wxyz[1:], np.float64)
    q_rel = _qmul(_qconj(qa), qb)
    flip = q_rel[:, 0] < 0.0
    q_rel[flip] = -q_rel[flip]
    w = np.clip(q_rel[:, 0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    sin_half = np.sin(theta * 0.5)
    safe = np.where(sin_half > 1e-12, sin_half, 1.0)
    axis = q_rel[:, 1:] / safe[:, None]
    dt = np.diff(np.asarray(observation_times, np.float64))
    dt = np.where(np.abs(dt) > 1e-12, dt, 1.0)
    omega = axis * (theta / dt)[:, None]  # (n-1, 3) rad/s
    return np.concatenate([omega, omega[-1:]], axis=0)  # (n, 3)


# --------------------------------------------------------------------------
# Articulated rest-frame mesh (module-cached, expensive to build).
# --------------------------------------------------------------------------


_MESH_CACHE: tuple | None = None


def _build_articulated_mesh() -> tuple:
    """Build the IS-901 mesh in body-frame coords with articulation pre-baked.

    Returns
    -------
    vertices_flat (M*3, 3) float64
        Flattened triangle vertices in body frame, rest pose. Each
        consecutive 3-tuple is one face: [v0, v1, v2, v3, v4, v5, ...].
    face_indices (M, 3) uint16
        Trivially [[0,1,2], [3,4,5], ...] — matches the flat layout.
    face_comp_id (M,) uint8
        Component id per face: 0=Bus, 1=SP_North, 2=SP_South, 3=AD_East,
        4=AD_West. Indexes COMPONENT_ORDER and JS COMPONENT_COLORS.
    component_order: tuple of component names corresponding to face_comp_id.
    """
    global _MESH_CACHE
    if _MESH_CACHE is not None:
        return _MESH_CACHE

    from src.config.rso_config_manager import RSO_ConfigManager
    from src.io.stl_loader import STLLoader
    from src.articulation import compute_rotation_matrices_from_angles
    from src.computation.facet_data_extractor import (
        extract_facet_arrays,
        apply_articulation_to_vertices,
    )

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
    satellite = STLLoader.create_satellite_from_stl_config(
        config=config, config_manager=config_manager
    )
    art_angles = {
        "SP_North": np.array([DEFAULT_SP_ANGLE_DEG]),
        "SP_South": np.array([DEFAULT_SP_ANGLE_DEG]),
        "AD_East": np.array([DEFAULT_AD_ANGLE_DEG]),
        "AD_West": np.array([DEFAULT_AD_ANGLE_DEG]),
    }
    art_matrices = compute_rotation_matrices_from_angles(art_angles, satellite)
    fa = extract_facet_arrays(satellite)
    vertices_flat = apply_articulation_to_vertices(fa, art_matrices, 0, satellite)
    M = int(fa.total_facets)
    face_indices = np.arange(M * 3, dtype=np.uint16).reshape(M, 3)
    face_comp_id = np.zeros(M, dtype=np.uint8)
    for cid, cname in enumerate(_COMPONENT_ORDER):
        sl = fa.component_slices.get(cname)
        if sl is not None:
            face_comp_id[sl] = cid
    _MESH_CACHE = (
        vertices_flat.astype(np.float64),
        face_indices,
        face_comp_id,
        _COMPONENT_ORDER,
    )
    return _MESH_CACHE


# --------------------------------------------------------------------------
# Main render entry point.
# --------------------------------------------------------------------------


def render_animation(npz_path: Path, meta_path: Path) -> Path:
    """Read NPZ + meta.json, emit animation.html alongside. Returns HTML path."""
    t0 = time.time()
    npz_path = Path(npz_path)
    meta_path = Path(meta_path)
    out_path = npz_path.parent / "animation.html"

    d = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    n_samples = int(d["q_pool_wxyz"].shape[0])
    n_ep = int(d["epoch_indices"].shape[0])

    rotvec_pool = np.asarray(d["rotvec_pool"], np.float32)  # (N, 3)
    survive_all = np.asarray(d["survive_all"], bool)        # (n_ep, N)
    pred_all = np.asarray(d["pred_all"], np.float32)        # (n_ep, N)
    epoch_indices = np.asarray(d["epoch_indices"], np.int64)
    measured_at = np.asarray(d["measured_at"], np.float64)
    n_survivors = np.asarray(d["n_survivors"], np.int64)
    truth_rv_canonical = np.asarray(d["rotvec_truth_at"], np.float64)
    closest_pos_canonical = np.asarray(d["closest_pos_per_epoch"], np.float64)
    closest_deg = np.asarray(d["closest_deg_per_epoch"], np.float64)
    mag_hifi = np.asarray(d["mag_hifi"], np.float64)

    # ── Continuity correction ──────────────────────────────────────────
    # 1. Temporally unwrap truth so its trail is continuous past |r|=π.
    # 2. Anchor closest to the unwrapped truth so the yellow marker sticks.
    # Survivors are anchored per-frame in JS (cheaper than shipping
    # per-frame positions; see viewer_template.html anchorTo helper).
    truth_rv = temporal_unwrap_rotvec(truth_rv_canonical).astype(np.float32)
    closest_pos = anchor_rotvec_to(closest_pos_canonical, truth_rv).astype(np.float32)

    # haze: deterministic 10k subsample of pool
    rng = np.random.default_rng(meta.get("haze_subsample_seed", 0))
    haze_n = min(HAZE_SIZE, n_samples)
    haze_idx = rng.choice(n_samples, haze_n, replace=False).astype(np.uint32)
    haze_idx.sort()  # sorted for slightly better cache locality on read

    # per-epoch survivor indices into pool, concatenated
    surv_idx_list = [np.where(survive_all[j])[0].astype(np.uint32) for j in range(n_ep)]
    surv_concat = np.concatenate(surv_idx_list) if surv_idx_list else np.empty(0, np.uint32)
    surv_offsets = np.zeros(n_ep + 1, dtype=np.int64)
    np.cumsum([len(a) for a in surv_idx_list], out=surv_offsets[1:])

    # ── Brightness colour gradient ─────────────────────────────────────
    finite = np.isfinite(mag_hifi)
    mag_cmin = float(mag_hifi[finite].min()) if finite.any() else 0.0
    mag_cmax = float(mag_hifi[finite].max()) if finite.any() else 20.0
    mag_span = max(mag_cmax - mag_cmin, 1e-6)

    haze_mags_per_ep = pred_all[:, haze_idx]                       # (n_ep, n_haze)
    surv_mags_concat = (
        np.concatenate(
            [pred_all[j, np.where(survive_all[j])[0]] for j in range(n_ep)]
        )
        if n_ep
        else np.empty(0, np.float32)
    )

    def _quantize_mag(mags: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(mags, np.float64) - mag_cmin) / mag_span * 255.0,
            0,
            255,
        ).astype(np.uint8)

    haze_mags_u8 = _quantize_mag(haze_mags_per_ep.ravel())  # row-major: epoch-major
    surv_mags_u8 = _quantize_mag(surv_mags_concat)

    # ── Side-panel data: full trajectory + per-epoch ───────────────────
    seed = int(meta["seed"])
    traj = load_truth(seed)
    quaternions_full = np.asarray(traj["quaternions"], np.float64)  # (n_obs, 4) wxyz
    obs_times = np.asarray(traj["observation_times"], np.float64)
    sun_pos = np.asarray(traj["sun_pos"], np.float64)
    obs_pos = np.asarray(traj["obs_pos"], np.float64)
    sat_pos = np.asarray(traj["sat_pos"], np.float64)
    n_obs = int(quaternions_full.shape[0])

    # Body-frame ω over full trajectory.
    omega_body_full = compute_omega_body(quaternions_full, obs_times)  # (n_obs, 3) rad/s
    omega_mag_dps_full = np.linalg.norm(omega_body_full, axis=1) * (180.0 / np.pi)  # (n_obs,)
    safe_mag = np.where(omega_mag_dps_full > 1e-12, omega_mag_dps_full, 1.0)
    omega_dir_full = omega_body_full / (
        safe_mag[:, None] * (np.pi / 180.0)
    )  # unit vectors (n_obs, 3) — divide by mag in rad/s
    # Re-canonicalize (numerical safety):
    omega_dir_full = omega_dir_full / np.maximum(
        np.linalg.norm(omega_dir_full, axis=1, keepdims=True), 1e-12
    )

    # Quaternions at sampled epochs (for satellite-mini rotation per frame).
    quaternions_at = quaternions_full[epoch_indices]  # (n_ep, 4)
    # Canonicalize w >= 0 for compact int16 quantization (no sign-flip drift).
    flip = quaternions_at[:, 0] < 0
    quaternions_at = np.where(
        flip[:, None], -quaternions_at, quaternions_at
    ).astype(np.float64)

    # Sun / observer unit vectors at sampled epochs (inertial frame).
    sun_unit_full, obs_unit_full = compute_j2000_units(sun_pos, obs_pos, sat_pos)
    sun_unit_at = sun_unit_full[epoch_indices].astype(np.float64)
    obs_unit_at = obs_unit_full[epoch_indices].astype(np.float64)

    # Articulated rest-frame mesh (body coords).
    mesh_verts, mesh_faces, face_comp_id, comp_order = _build_articulated_mesh()

    payload = {
        "config_id": meta["config_id"],
        "seed": meta["seed"],
        "surrogate": meta["surrogate"],
        "n_samples": n_samples,
        "tolerance_mag": meta["tolerance_mag"],
        "omega_mag_dps": meta["omega_mag_dps"],
        "n_obs": n_obs,
        "n_epochs": n_ep,
        "epoch_indices": epoch_indices.tolist(),
        "measured_at": _round_array(measured_at, 4),
        "n_survivors_per_epoch": n_survivors.tolist(),
        "closest_deg_per_epoch": _round_array(closest_deg, 4),
        "truth_rotvec_per_epoch": _round_array(truth_rv, 5),
        "closest_pos_per_epoch": _round_array(closest_pos, 5),
        "mag_hifi_full": _round_array(np.where(np.isfinite(mag_hifi), mag_hifi, np.nan), 4),
        "rotvec_pool_i16_b64": _b64_int16_quantize_pi(rotvec_pool),
        "haze_indices_u32_b64": _b64_uint32(haze_idx),
        "survivor_indices_u32_b64": _b64_uint32(surv_concat),
        "survivor_offsets": surv_offsets.tolist(),
        "haze_mags_u8_b64": base64.b64encode(haze_mags_u8.tobytes()).decode("ascii"),
        "survivor_mags_u8_b64": base64.b64encode(surv_mags_u8.tobytes()).decode("ascii"),
        "mag_cmin": mag_cmin,
        "mag_cmax": mag_cmax,
        "haze_subsample_seed": int(meta.get("haze_subsample_seed", 0)),
        "epoch_spec_str": meta.get("epoch_spec_str"),
        # ── Satellite-orientation mini panel (rest-frame articulated mesh) ──
        "mesh_verts_i16_b64": _b64_int16_range(mesh_verts.ravel(), MESH_BBOX_HALF),
        "mesh_verts_n": int(mesh_verts.shape[0]),
        "mesh_bbox_half": MESH_BBOX_HALF,
        "mesh_faces_u16_b64": _b64_uint16(mesh_faces.ravel()),
        "mesh_faces_n": int(mesh_faces.shape[0]),
        "face_comp_id_u8_b64": _b64_uint8(face_comp_id),
        "component_order": list(comp_order),
        # ── Per-sampled-epoch quaternions (body→inertial), wxyz ──
        "quaternions_at_i16_b64": _b64_int16_unit(quaternions_at),
        # ── Per-sampled-epoch inertial sun/observer unit vectors ──
        "sun_unit_at_i16_b64": _b64_int16_unit(sun_unit_at),
        "obs_unit_at_i16_b64": _b64_int16_unit(obs_unit_at),
        # ── Full-trajectory ω body-frame data ──
        "omega_mag_dps_full": _round_array(omega_mag_dps_full, 5),
        "omega_dir_full_i16_b64": _b64_int16_unit(omega_dir_full),
        "obs_times_full": _round_array(obs_times, 4),
    }

    template = TEMPLATE_PATH.read_text()
    json_blob = json.dumps(payload, separators=(",", ":"))
    html = template.replace("__PAYLOAD__", json_blob)

    out_path.write_text(html)
    wall = time.time() - t0

    # update meta.json with render wall time
    meta["wall_render_s"] = wall
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=float)

    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="path to a results/.../{config_id}/ directory")
    args = ap.parse_args()
    npz = args.run_dir / "spread.npz"
    meta = args.run_dir / "meta.json"
    out = render_animation(npz, meta)
    print(f"[saved] {out} ({out.stat().st_size / 1e6:.1f} MB)")
