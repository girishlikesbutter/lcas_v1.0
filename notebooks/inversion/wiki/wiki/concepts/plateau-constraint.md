---
title: "Plateau Constraint — PAB Circling a Lobe"
type: concept
sources: []
related: ["[[pab-contour-isoshell]]", "[[att-fail-diagnosis]]", "[[ipl-candidate-generation]]"]
created: 2026-04-13
updated: 2026-04-13
confidence: medium
---

# Plateau Constraint — PAB Circling a Lobe

## Mechanism

When the light curve magnitude is constant for an extended period (10-15+ epochs), the PAB is circling a single lobe at constant radial distance on the brightness surface. Only one IPL loop is active during a plateau -- the PAB can't escape the lobe without changing magnitude.

## What It Provides

1. **Lobe identity** -- which body-frame lobe is being orbited
2. **Radial distance** from lobe center (from magnitude level)
3. **Angular velocity** of the circling (from subtle magnitude variations)

## Discrimination Power

Plateau lo-fi gives 125x MSE discrimination range (much better than peaks' 3.7x) -- the signal is strong but lo-fi picks the wrong phi (8.9x worse than best).

## Applicability

Potentially powerful for slow tumblers (the 19 seeds without enough tight IPL minima) which may have long plateau phases.

## Status

UNTESTED with hi-fi -- a key next step.
