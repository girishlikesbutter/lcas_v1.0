#!/usr/bin/env bash
# Phase 2 pilot: run invert.py end-to-end on 3 m048 seeds at low/mid/high phase angle.
# Seeds picked in data/results/inversion_diagnostics/phase_B_pilot_seeds.json.
#
# Runs SEQUENTIALLY — each invert.py stage already uses Pool(24) internally,
# so stacking seeds would thrash. Exits on first failure so we can investigate
# without wasting downstream compute.
#
# Expected total wall: ~40 min (3 seeds × ~13.5 min each).

set -euo pipefail

cd "$(dirname "$0")/../.."
INVERT="python3 notebooks/inversion/invert.py"
LOG_DIR="data/results/inversion_diagnostics/phase_B_pilot_logs"
mkdir -p "$LOG_DIR"

echo "=== Phase 2 pilot started $(date -Is) ==="

for SEED in 24 91 28; do
  echo
  echo "=== seed $SEED (start $(date -Is)) ==="
  $INVERT --seed "$SEED" --traj-source m048 2>&1 \
    | tee "$LOG_DIR/invert_m048_seed$(printf %03d "$SEED").log"
  echo "=== seed $SEED (done $(date -Is)) ==="
done

echo
echo "=== Phase 2 pilot finished $(date -Is) ==="
