---
title: "Shadow Asymmetry — The Only Phi Discriminator"
type: concept
sources: ["data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz"]
related: ["[[att-fail-diagnosis]]", "[[isoshell-phi-limits]]", "[[hi-fi-scoring]]", "[[sparse-hifi-phi]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# Shadow Asymmetry — The Only Phi Discriminator

## The Symmetry Problem

The IS-901 brightness surface (zero phase angle, no shadows) is locally symmetric around each lobe normal. At ±X lobes (solar panels), the specular BRDF is cos^n(angle_from_normal) where n=250. Rotating phi (twist about the normal) barely changes brightness: cos^250 varies by <0.01% for a 5° twist.

This symmetry makes lo-fi (no-shadow) phi evaluation useless for ATT_FAIL seeds:
- Lo-fi MSE: truth ranks correctly but with vanishing margin (0.0001 mag^2 over 500 epochs)
- IPL centroid proximity: disc ratio 0.997 (essentially random)
- Brightness derivative: disc ratio 1.021 (barely above noise)

## What Breaks the Symmetry: Shadows

Self-occlusion shadows are NOT symmetric about lobe normals. The solar panels cast shadows on the bus when the body-frame sun direction approaches ±Y.

Measured shadow effects (hi-fi - lo-fi):
| Seed | Max |shadow| | Lobe | % epochs > 0.1 mag |
|------|---------------|------|---------------------|
| 27   | 4.56 mag      | +Y   | 44.0%               |
| 46   | 3.43 mag      | +Y   | 40.4%               |
| 58   | 4.34 mag      | -Y   | 34.2%               |
| 93   | 1.89 mag      | +Y   | 40.0%               |

Shadows are concentrated at ±Y lobes (solar panel occlusion geometry). At ±X and ±Z lobes, shadows are weak.

## Why Shadows Discriminate Phi

Different phis produce different body-frame lobe-visiting schedules. The correct phi visits ±Y at specific epochs where the observation shows shadow-dimmed magnitudes. A wrong phi visits ±Y at different epochs where the observation doesn't show dimming (or doesn't visit ±Y at all).

For the correct phi: hi-fi evaluation correctly predicts shadows → matches observation.
For the wrong phi: hi-fi evaluation produces different shadow pattern → doesn't match.

## Implications for Pipeline

1. **Lo-fi phi sweep is fundamentally insufficient** for seeds with symmetric lobe geometry
2. **Hi-fi phi sweep at shadow-rich epochs** (±Y visits) is the path forward
3. **Sparse hi-fi** (5-10 epochs) should suffice if the epochs are shadow-critical
4. **Epoch selection** should prioritize epochs where the PAB approaches ±Y normals

## Open Question

Does sparse hi-fi (10 peaks × 72 phis) actually discriminate truth phi? This is the subject of the [[sparse-hifi-phi]] branch (m110).
