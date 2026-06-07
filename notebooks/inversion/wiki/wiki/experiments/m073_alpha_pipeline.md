---
title: "m073 — Multi-seed Alpha Pipeline"
type: experiment
sources:
  - "raw/inversion_diagnostics/m073/"
related:
  - "[[m070_full_pipeline]]"
  - "[[m090_robust_peak_selection]]"
  - "[[twin-degeneracy]]"
  - "[[phi-sweep]]"
created: 2026-03-26
updated: 2026-04-16
confidence: high
---

# m073 — Multi-seed Alpha Pipeline

First multi-seed validation of the [[m070_full_pipeline]] pipeline.

## Setup

6 seeds tested with the full pipeline from [[m070_full_pipeline]].

## Results

- **4/6** recover omega direction <5deg
- **1/6** full success (attitude + omega)
- **1/6** failure
- **Runtime**: 4.5-7.4 min/seed
- **180deg attitude twin** observed in 3/6 seeds

## Key Takeaway

The pipeline works on multiple seeds but the [[twin-degeneracy]] is pervasive — half the seeds converge to a ~180deg-rotated attitude. This motivated the twin-axis investigation in [[m091_twin_state_test]] and [[m092_twin_axis_visualization]].
