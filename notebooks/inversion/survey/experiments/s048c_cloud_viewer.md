---
title: s048c — cloud-evolution browser viewer (implementation)
type: tool
sources: [s048, s048b, s048c_handoff, project_pab_manifold_viewer]
related: [s048, s048b, s048c_handoff]
created: 2026-05-07
updated: 2026-05-07b
confidence: high
---

# s048c — cloud-evolution browser viewer

## TL;DR

Built a single-user local Flask tool (`s048c_viewer/`) that turns the
s048b per-epoch C_t survival sweep into a polished, browseable, animated
3D viewer. User picks `(seed, surrogate, epochs, n_samples,
tolerance_mag)` from a form; if cached, redirects to viewer; otherwise
dispatches a single-worker compute job, polls progress in-browser,
redirects to viewer on completion. The viewer renders survivors blue,
killed haze grey, truth-q green, closest-to-truth-among-all-candidates
yellow with a connecting line and a degree readout — playable,
scrubbable, and rotatable concurrently (camera persists across frames).

## What

A self-contained workflow:

1. **`lib/c_t_pipeline.py`** — reusable primitives factored from
   s048b: `sample_so3_pool`, `compute_j2000_units`, `project_directions`,
   `survive_at_epoch`, `nearest_in_pool_to_truth(s)`,
   `quats_to_rotvec_wxyz`. Byte-equivalent to s048b on every survival
   bit across all 30 dimmest epochs.
2. **`s048c_viewer/compute.py`** — `compute_c_t(seed, surrogate, ...)`
   wrapper that runs the sweep and saves NPZ + meta.json under
   `results/s048c_cloud_viewer/seed{NNN}/{config_id}/`.
   Cache-aware (returns cached path on hit). Supports v1 and v2
   surrogates; v1 preloaded at server startup.
3. **`s048c_viewer/render.py`** — emits a self-contained HTML animation
   from the NPZ. Plotly + procedural JS (`setInterval` + `Plotly.restyle`),
   modeled after the design pattern in
   `notebooks/inversion/lib/hifi_isoshell_viewer.py`. Camera baked in
   initial layout, never touched on update — that's what enables
   rotate/zoom-while-playing without resetting.
4. **`s048c_viewer/serve.py`** — Flask app: form + cached-runs table
   landing page, async dispatch via single-worker `ThreadPoolExecutor`,
   XHR-polled progress page, viewer route.
5. **`s048c_viewer/cache.py`, `parser.py`, `jobs.py`** — config_id
   hashing (sha256-12 over canonical JSON), epoch free-text parser
   (`all` / `::N` / `M-N` / `M,N,P,...`), in-memory job registry.

## Continuity correction (rotvec wrap-around at |r|=π)

The canonical rotvec chart `|r| ≤ π` has a discontinuity on the boundary
sphere — antipodal boundary points represent the same physical rotation
(180° flip around `n̂` ≡ 180° flip around `−n̂`). When truth's rotation
angle θ smoothly crosses π, scipy's canonical pick flips truth's rotvec
to the antipodal side; same artifact for the closest pool entry whenever
two q-space-close candidates straddle the boundary.

The viewer fixes this in three coordinated steps:

1. **Truth temporally unwrapped.** `lib/c_t_pipeline.temporal_unwrap_rotvec`
   sequentially picks at each frame whichever of the two equivalent
   rotvec reps (`r` canonical vs `r' = (1 − 2π/|r|)·r` alt) is closer to
   the previous frame's chosen rep. Truth's trail walks past `|r|=π` into
   the outer shell instead of teleporting.
2. **Closest anchored to (unwrapped) truth.** For each frame,
   `closest_pos_per_epoch` is replaced with whichever rep of itself is
   closer in 3D to the unwrapped `truth_rotvec[f]`. Yellow marker stays
   glued to the green diamond.
3. **Survivors anchored per-frame in JS.** Same anchoring done in the
   browser inside `survivorXYZ(f)` (a few flops per survivor; cost
   negligible vs the restyle itself). Cloud follows truth coherently.

Haze stays canonical as the fixed background (10k subsample of the pool's
canonical rotvecs). Bounding box is `[-2π, 2π]` per axis to accommodate
truth + cloud walking into the outer shell.

For seed 89, this puts ~219/500 frames in the outer shell (truth crosses
the boundary multiple times); frame-to-frame jump max drops from "across
the ball" to ≤ 0.11 rad. For *fast* tumblers (multi-revolution rotations),
truth's unwrapped trail walks linearly outward; the bounding box would
need to grow without bound. For those seeds the cleaner option is a
**truth-relative chart** (truth at origin, every entity plotted as
`log(R_truth⁻¹·R_entity)` — guaranteed `|r|≤π` because that's geodesic
distance from truth). Not implemented; deferred until we hit a fast seed
and want to look at it.

## How

Run the server:

```bash
cd notebooks/inversion/survey
python -m s048c_viewer.serve [--port 5048] [--no-preload]
```

Browse to `http://127.0.0.1:5048`. Submit a config, watch the progress
page redirect to the viewer when compute finishes.

In the viewer:
- Space toggles play/pause.
- ←/→ step ±1 frame.
- Slider scrubs frames; epoch text input jumps to nearest sampled epoch.
- T toggles truth-q trail (last 30 frames).
- H toggles the killed haze.
- Mouse rotate / scroll-zoom works during playback — the camera persists
  across frames because `Plotly.restyle` never touches `scene.camera`.

Click on the LC sidebar to jump frames. Closest-to-truth distance (deg)
is shown in the readout panel.

## Result

End-to-end smoke (Flask test client):
- New 3-epoch / 50k-sample config: dispatch + compute + render + redirect
  in **1.5 seconds** wall.
- Re-dispatch same config: instant cache hit (`cached: true`).
- All bad-input paths return 400 with clear messages (out-of-range seed,
  epoch garbage, oversize `n_samples`).

The s048b 30-epoch / 500k-sample reference matches BIT-EXACTLY on
survival masks: `mismatches: 0/30`. The lifted primitive is a faithful
factorization, not a re-implementation.

## Why this matters

Three things this tool enables that the static 30-panel plots couldn't:

1. **Visual access to the user's "breathing cloud" hypothesis.** Watching
   |C_t| expand/contract across the LC with truth-q's smooth motion as a
   reference frame is the cheapest way to see if there's periodic
   structure. Whether that structure is ω-related is a separate
   measurement (FFT of `|C_t|(t)` is a one-liner once we want to look),
   but the animation tells you whether the question is even worth asking.

2. **Browseable cohort exploration.** The cached-runs table accumulates
   every (seed, surrogate, tolerance) combo we've inspected. Comparing
   seed 89 (|ω|=0.24 dps) vs seed 14 (|ω|=1.23 dps) is now a click each,
   not two CLI invocations + two file paths.

3. **Foundation for joint multi-epoch survivor filters.** The
   `lib/c_t_pipeline.py` primitives are the substrate the next-tier
   experiments will build on (joint k-epoch consistency, peak-cascade
   variants on higher-|ω| seeds). The byte-equivalence test pins them
   down before higher-order experiments diverge.

## Numbers

| metric | value |
|---|---|
| pipeline byte-equivalence to s048b | 0 mismatches / 30 epochs / 500k samples |
| 5-epoch / 100k-sample compute wall | 3.8 s (v1) |
| 3-epoch / 50k-sample wall (test client) | 1.5 s (compute + render) |
| **500-epoch / 100k-sample compute wall** | **369.7 s = 6:10 (v1, seed 89)** |
| 500-epoch render wall | 0.1 s |
| 500-epoch NPZ size | 254 MB |
| 500-epoch animation HTML size | 15.5 MB |
| 500-epoch \|C_t\| min / max (seed 89) | 44 / 17 085 survivors (388× spread — the "breathing") |
| 500-epoch closest-to-truth min / max | 0.59° / 6.04° |
| HAZE_SIZE (deterministic subsample) | 10 000 (seed=0) |
| guardrails | n_samples ∈ [1k, 500k]; n_epochs ≤ 1000; tol ∈ [1e-4, 5] mag |

## Artefacts

- `lib/c_t_pipeline.py`
- `s048c_viewer/__init__.py`
- `s048c_viewer/compute.py`
- `s048c_viewer/render.py`
- `s048c_viewer/cache.py`
- `s048c_viewer/parser.py`
- `s048c_viewer/jobs.py`
- `s048c_viewer/serve.py`
- `s048c_viewer/viewer_template.html`
- `s048c_viewer/templates/{base,landing,progress}.html`
- `s048c_viewer/static/style.css`
- `results/s048c_cloud_viewer/seed089/1fe2346148eb/` — 5-epoch smoke
- `results/s048c_cloud_viewer/seed089/9b7842a96e63/` — 3-epoch test-client smoke
- `results/s048c_cloud_viewer/seed089/<TBD>/` — full 500-epoch run

Plan file: `~/.claude/plans/yeah-i-wanna-build-velvet-neumann.md`.

## Out of scope

- Multi-user, auth, deployment beyond `127.0.0.1`.
- Streaming partial cloud during compute (only progress %).
- Joint multi-epoch survivor filters (Tier 1+ from s048).
- WebSocket progress (XHR poll at 500 ms is fine).
- Auto-prune cached runs.
- Re-rendering with different visual configs from same NPZ via the web
  UI (the `python -m s048c_viewer.render <run_dir>` CLI hook exists but
  isn't exposed through the form).
- Cloud-breathing periodicity quantification (separate analysis once
  visual confirms the hypothesis).

## Cross-references

- `experiments/s048_peak_cascade_smoke.md` — Tier 0 confirmation.
- `experiments/s048b_per_epoch_spread_v1.py` — pipeline source of truth
  (lifted into `lib/c_t_pipeline.py`).
- `experiments/s048c_handoff.md` — original brief.
- `notebooks/inversion/lib/hifi_isoshell_viewer.py` — design polish
  reference (not imported; pattern only).
- `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/project_pab_manifold_viewer.md` —
  load-bearing viewer design rules (global scaling / show killed / no
  mesh3d for deforming geometry).

## 2026-05-07 update — three synchronised side panels

The viewer now sports three additional panels driven by the same
`setInterval + Plotly.restyle` frame loop, with the camera-persistence
pattern (initial layout's `scene.camera` baked, never restyled) preserved
on every 3D plot — rotate/zoom-while-playing still works.

### What's new

1. **Satellite mini 3D (top-left of sidebar)** — IS-901 articulated mesh
   in **inertial frame**. Vertices = `R(q[t]) · v_body_articulated`,
   where `R(q[t])` is recomputed per frame in JS from a 4-int16 quaternion
   via `quatToMat`. Mesh shipped once as int16-quantized vertices over a
   ±18 m bbox (3.8k facets, 11.5k vertices). Body axes
   (X red / Y green / Z blue) tumble with the body. Sun (gold) and
   observer (cyan) arrows are shown in inertial frame and update per
   sampled epoch. Initial camera baked at the observer's POV at frame 0;
   user can rotate freely.
2. **ω-direction sphere (top-right of sidebar)** — wireframe unit sphere
   (3 great-circle rings on principal planes) + RGB body axes. Pink
   ω̂_body(t) point updates per frame; fading 30-frame trail traces the
   precession of the rotation axis through body frame.
3. **|ω|(t) line strip (above LC)** — full-trajectory line + sampled
   markers + current-epoch cursor, with click-to-jump matching the LC
   strip. Visually flat for seed 89 (energy conservation, std/mean = 0.5%);
   the dynamic content lives in the ω-direction sphere.

### Pre-render probe finding

Seed 89: `|ω|(t)` is **constant** to ~0.5% (rad/s std/mean) — energy
conservation in the propagator. But body-frame components vary
substantially (ω_bx range 0.022 dps, ω_by ±0.087 dps, ω_bz ±0.086 dps):
classical asymmetric-rigid-body precession. The user's chosen split
(magnitude as a line plot + direction as a moving point on a sphere)
surfaces both — energy conservation as the flat line, precession as the
sphere trail.

### Architecture

**Render-time only.** No compute regen, no NPZ schema changes, no
cache invalidation. `render.py` loads the trajectory NPZ at render time
and packs the new arrays into the inline payload. The articulated
rest-frame mesh is built once via a module-level cache by reusing
`src.computation.facet_data_extractor.{extract_facet_arrays,
apply_articulation_to_vertices}` and the SP=0°/AD=15° constants from
`lib.hifi_render`. Body-frame ω is finite-diffed across the full
trajectory (not just sampled epochs) so the ω-direction sphere is
correct under any subsampling scheme.

### Mesh3d caching gotcha — observed not to apply here

The `project_pab_manifold_viewer` memory documents that `Plotly.restyle`
on `x/y/z` of a *uniform* mesh3d is unreliable — the rendered mesh
stays stuck at first-drawn size. With explicit `i/j/k` face indices
**and** per-face `facecolor` variation (5-component coloring), the
restyle works correctly: confirmed live, the satellite tumbles
faithfully across frames.

### Sizing

| asset | shipped | inflation |
|---|---|---|
| articulated mesh vertices (11520 × 3 int16) | once | +69 KB |
| mesh face indices (3840 × 3 uint16) | once | +23 KB |
| face-component ids (3840 × uint8) | once | +5 KB |
| quaternions per sampled epoch (n_ep × 4 int16) | per run | +4 KB at 500 ep |
| ω̂_body per full-trajectory epoch (500 × 3 int16) | once | +3 KB |
| ω_mag per full-trajectory epoch (500 × float) | once | +12 KB JSON |
| sun/obs unit vectors per sampled epoch (n_ep × 3 × 2 int16) | per run | +6 KB at 500 ep |
| **500-ep showcase HTML** | | **25.83 → 26.00 MB (+0.7%)** |

### Layout rebalance (CSS)

| element | before | after |
|---|---|---|
| `#plot3d` | flex 3 | flex 2 |
| `#sidebar` | flex 1, minWidth 320, maxWidth 460 | flex 1, minWidth 600, maxWidth 820 |
| sidebar contents | LC plot + readouts | sat mini + ω-dir (top row) → \|ω\| strip → LC plot → readouts |

### Verification

- Headless Chrome with software WebGL (`--use-angle=swiftshader`) on the
  500-epoch run, frame 0 vs frame 250: satellite has visibly tumbled
  (~1.2 full rotations consistent with |ω|=0.24 dps × 1800 s = 432°);
  ω-axis sphere shows precessed pink point with trail; |ω|(t) and LC
  cursors both advanced to epoch 250.
- 5-epoch run (epoch_indices [0, 100, 200, 412, 499]) at frame 4
  (epoch 499) renders cleanly with all panels populated.
- HTML payload validated programmatically: every new field present,
  shapes correct (mesh_verts_n=11520, mesh_faces_n=3840, n_obs=500,
  ω_mag range 0.2398–0.2435 dps matches the probe).

### Artefacts (delta)

- `s048c_viewer/render.py` — extended payload (mesh + ω + sampled epoch arrays)
- `s048c_viewer/viewer_template.html` — CSS rebalance, three new Plotly
  inits, `quatToMat` helper, extended `updateFrame()`
- `results/s048c_cloud_viewer/seed089/{8bb9b81f1602, 1fe2346148eb,
  9b7842a96e63}/animation.html` — all three cached runs re-rendered
- Plan file: `~/.claude/plans/resilient-foraging-alpaca.md`
