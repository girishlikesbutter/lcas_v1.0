# micro26 / micro26b Worker Report

## What was attempted

**Goal:** micro26 — end-to-end integration pipeline (oracle attitudes). Wire band-sweep winding enumeration + L-conservation filter using oracle attitudes at 3 peaks.

### Scripts written

1. **`micro26_integration_oracle.py`** (146 lines) — written and ran on branch `exp/integration-nudged`. Band-sweep (13 bands × 10 starts per leg), magnitude-only dedup (0.05 deg/s), L-conservation scoring at peak B.

2. **`micro26b_direction_aware_dedup.py`** (175 lines) — written on branch `inversion_q_w`. Same pipeline but with direction-aware dedup: two omegas are duplicates only if |omega| within 0.05 deg/s AND angular distance < 5 deg.

## What results exist

### micro26 (on branch `exp/integration-nudged`, committed)
- Script: `notebooks/inversion/08_integration/micro26_integration_oracle.py`
- JSON: `data/results/inversion_diagnostics/micro26_integration_oracle.json`
- PNG: `data/results/inversion_diagnostics/micro26_integration_oracle.png`
- FINDINGS.md: `notebooks/inversion/08_integration/FINDINGS.md`
- Commits: `b08b721`, `471469c`, `06787cc`

**micro26 results summary:**
- Leg 0: 34 candidates, Leg 1: 32 candidates
- True pair rank: **423/1088** — L-conservation completely failed
- Root cause: magnitude-only dedup keeps one arbitrary direction per magnitude; the kept direction at the true |omega| was wrong, so L doesn't match between legs
- Runtime: 2291s (~38 min)

### micro26b (on branch `inversion_q_w`, NOT committed)
- Script: `notebooks/inversion/08_integration/micro26b_direction_aware_dedup.py` — exists on disk
- **No results yet** — script was running in background (task `ba6cbihz4`), had completed setup (120s) and was in the bridge-solve phase when interrupted

## What went wrong / took long

1. **Runtime:** Each leg takes ~19 min for 130 bridge solves with Pool(8). Total ~38 min for micro26, well over the 15-min target. The bottleneck is `propagate_attitude` ODE integration over dt=555s and dt=721s.

2. **Branch confusion:** The task said to create branch `exp/integration-oracle`, which was done. But micro26 ended up committed on `exp/integration-nudged` (the branch had been switched between sessions). micro26b was then written on `inversion_q_w`. The micro26 results files (JSON, PNG) only exist on `exp/integration-nudged`.

3. **micro26b still running:** The background process was interrupted before completion. It may still be running as a system process.

## Current state

- **Branch:** `inversion_q_w`
- **Running:** micro26b background task `ba6cbihz4` — may still be running, was in bridge-solve phase
- **Uncommitted on `inversion_q_w`:**
  - `notebooks/inversion/08_integration/micro26b_direction_aware_dedup.py`
  - `notebooks/inversion/08_integration/WORKER1_REPORT.md`
- **Committed on `exp/integration-nudged`:**
  - micro26 script + results + FINDINGS.md (3 commits)
- **micro26 results NOT on `inversion_q_w`** — they live only on `exp/integration-nudged`
