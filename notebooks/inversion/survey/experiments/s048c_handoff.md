---
title: s048c — handoff for cloud-evolution animation (next agent)
type: handoff
sources: [s048, s048b, project_pab_manifold_viewer]
related: [s048, s048b, lib/hifi_isoshell_viewer.py]
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# s048c — handoff: per-epoch C_t cloud-evolution animation

## Mission

Produce a beautiful, scientifically-truthful animation of how the per-epoch C_t pre-image cloud evolves across the full LC for one seed (start with seed 89). The user has watched the static 30-panel plots from s048b and wants to see the cloud "breathe" — expand/contract, densify/rarefy — and where truth-q sits relative to the cloud at every frame.

The user's intuition: cloud breathing has periodic structure related to ω. The animation is for visual exploration of this hypothesis (a separate FFT analysis was offered but the user chose the animation as primary).

**Visual quality benchmark:** match the polish of `notebooks/inversion/lib/hifi_isoshell_viewer.py` and its HTML output (`hifi_isoshell_seed000.html`). That viewer's design invariants are documented in `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/project_pab_manifold_viewer.md`. Read it before designing.

## What it should show

Per frame (one frame per epoch, 500 frames total for seed 89):

1. **The C_t cloud** for that epoch — hundreds-to-tens-of-thousands of surviving q's as a 3D scatter (rotvec or some other geometric projection of SO(3)).
2. **Killed candidates** — semi-transparent grey background to give the cloud visual context (subsample to ~5–10k for rendering speed).
3. **Truth-q at this epoch** — bright marker (red star or similar), prominent.
4. **Truth-q's recent trajectory** — fading trail of the last ~10–30 frames so you can see truth move smoothly through SO(3) while the cloud changes shape around it.
5. **Sidebar / overlay**:
   - The full LC with the current epoch marked.
   - `|C_t|(t)` curve with current point marked (so you can see the cloud-size oscillation).
   - Current `mag_measured` and `|C_t|` numerical readout.
   - Optionally: phase-angle (`pab` from NPZ), |ω| (constant for this seed = 0.24 dps).

## Data already produced (don't recompute)

`results/s048_peak_cascade_smoke/seed089_v1_spread_n30_dimmest/spread.npz` and the n30/n15 spread NPZs ARE NOT enough — they only have 30 epochs. **You need a fresh 500-epoch sweep** at sufficient sample count for visualization.

Suggested sweep parameters:
- N samples: **100k** (not 500k — 5× faster, plot-quality unaffected once you subsample for visual).
- Epochs: **all 500** (one frame per epoch).
- Surrogate: **v1** (4.5 μs/sample) — at 100k × 500 = 50M evals = ~225 sec compute = ~4 min wall.
- Tolerance: 0.10 mag (matches s048b).
- Cache R(q) once across all 500 epochs (the smart way per s048b).

Total wall: ~5–8 min for the sweep, then render the animation on the saved data.

## v1 vs v2 — do BOTH work?

YES. Both v1 and v2 surrogates produce body-frame mag predictions with identical APIs:

```python
# v1 (faster, 0.044 mag MAE):
from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1
v1 = SurrogateV1('/home/girish/surrogate_model/surrogate_model/s10_5M_weights.npz',
                 '/home/girish/surrogate_model/surrogate_model/s10_5M_normalization.npz')

# v2 (slower, 0.008 mag MAE; default in lib/surrogate_eval.py):
from lib.surrogate_eval import predict, get_model
# get_model().predict_magnitude(...)
```

Both signatures: `predict_magnitude(k1, k2, sp_angle_deg, ad_angle_deg, observer_distance_km) → mag`.

For the animation: **v1 is the right choice** — 12× faster, accuracy difference (~36 mmag) is well below the 100 mmag filtering tolerance, no benefit to v2 at this stage. v2 should be reserved for hi-fi-adjacent confirmation (which is not what the animation is for).

## Implementation pattern (recommended)

Mirror `notebooks/inversion/lib/hifi_isoshell_viewer.py`:
- Plotly-based, HTML output.
- Per-epoch quantized data shipped in the HTML (efficient for 500 frames).
- Slider + autoplay control.
- Static camera (don't auto-rotate; user controls camera).

**Critical design decisions** (per the PAB manifold viewer memory — these are LOAD-BEARING):

1. **Global radial/scale normalization, never per-epoch.** If you scale the cloud display per-frame to "fill the panel," the animation lies about absolute geometry. Compute the global (rotvec) extent once, hold the axes constant.

2. **Don't hide killed candidates.** Show them as a grey background haze (subsample heavily, e.g. 5k of 100k). The contrast between dense killed-haze and bright cloud is what makes the visualization read.

3. **Plotly mesh3d caching gotcha:** if you use mesh3d for any "shell" or "surface" element, restyle won't update its geometry reliably. Use scatter3d with great-circle line traces instead.

4. **One radius = one brightness across all frames** (analogue to the isoshell rule). The C_t cloud lives in q-space (rotvec), not mag-space, so this rule needs translation: a 3D rotvec point at angle θ from origin is unambiguous regardless of epoch — don't rescale rotvec axes per frame.

## Choice of projection / coordinates

SO(3) is 3D and curved. Options:

- **rotvec (axis × angle, rad)** — what s048b's static plots use. Bounded by a ball of radius π. Distorts near the antipode but reasonable for visualization. The static plots used `rotvec[:, 0:2]` (2D); the animation should use full 3D rotvec scatter3d.
- **Quaternion (x, y, z) imaginary parts with `w ≥ 0` half-sphere** — different distortion profile.
- **Euler angles** — bad, gimbal-lock causes apparent jumps.

Recommend **rotvec 3D** for the animation. Truth-q's trajectory in rotvec space is smooth (it's the integrated ω-trajectory) and visually intuitive.

## Truth-q computation

Per epoch, truth-q is `quaternions[epoch]` from the trajectory NPZ (already cached, post-fix-validated). Convert to rotvec via:

```python
truth_rotvec_at = Rotation.from_quat(quats_truth[:, [1, 2, 3, 0]]).as_rotvec()
```

(Reshuffle wxyz → xyzw for scipy.)

## Sanity tests before scaling

Before producing the full 500-frame animation:
1. Generate frames for ~5 specific epochs (e.g. 0, 100, 208, 412, 499) and confirm:
   - The cloud size matches s048b's measurements at those epochs (e.g. ep 208 ≈ 384 survivors at 500k samples; should scale to ~80 at 100k samples).
   - Truth-q is visually inside or very close to the cloud at every checked epoch.
   - The killed-haze gives readable contrast.
2. Run the full 500-epoch sweep, save NPZ, then render the animation from the saved data (so it's cheap to iterate on visual style without re-sweeping).

## What to checkpoint

NPZ (one file, ~1.5 GB at 100k × 500):
- `q_pool_wxyz` (100k, 4)
- `rotvec_pool` (100k, 3)
- `survive_all` (500, 100k) — bool
- `pred_all` (500, 100k) — float32
- `obs_times` (500,)
- `mag_hifi` (500,)
- `q_truth_per_epoch` (500, 4)
- `rotvec_truth_per_epoch` (500, 3)
- `n_survivors_per_epoch` (500,)
- `nearest_truth_deg_per_epoch` (500,)
- `omega_mag_dps`, `tolerance_mag` — scalars

Then render the animation HTML from this NPZ — re-render is fast and cheap.

## Pitfalls to avoid

- **Don't re-propagate the truth attitude per epoch** — the trajectory NPZ already has `quaternions` cached, validated at machine precision in s044/s043.
- **Don't try to rotate the camera** — kills user's spatial intuition. Static camera with manual mouse control.
- **Don't subsample the cloud's survivors** — show all of them. Subsample only the killed (grey haze).
- **Don't filter peaks by prominence** — for the animation, every epoch is a frame regardless of LC structure.
- **Don't switch to v2 for any reason** without flagging — there's no quality benefit at this scale and 12× wall hit.
- **Don't claim the animation answers the ω-frequency question** by itself. It's a visualization. If the user wants quantitative ω-periodicity confirmation, that's a separate FFT/autocorrelation analysis on `n_survivors_per_epoch`.

## Acceptance criteria

- 500-frame HTML animation with slider + autoplay.
- Cloud + killed-haze + truth-q + truth-trajectory visible per frame.
- LC + |C_t|(t) sidebar with current-epoch marker.
- Loads in <10 sec, plays smoothly at ~10 fps.
- Visually as polished as `hifi_isoshell_seed000.html`.

## Files to read first

1. `experiments/s048b_per_epoch_spread_v1.py` — the sweep script and data layout.
2. `notebooks/inversion/lib/hifi_isoshell_viewer.py` + `hifi_isoshell_template.html` — the visual quality benchmark.
3. `experiments/s048_peak_cascade_smoke.md` and `experiments/s048b_per_epoch_spread_v1.md` — the science context.
4. `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/project_pab_manifold_viewer.md` — the load-bearing design rules.
