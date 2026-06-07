#!/usr/bin/env bash
# micro119 multi-seed runner
# For each seed: harvest micro118 kernel if absent, then run micro119 if absent.
# Skip-if-exists at both stages. Sequential — do NOT parallelise (CPU-saturating).
#
# Usage: bash notebooks/inversion/12_brightness_surface/micro119_multiseed_runner.sh

# No `set -e`: per-seed skip-if-exists handles restart. A transient failure
# in one seed (e.g., stderr-at-exit Tee bug) shouldn't kill the whole batch.
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
echo "[multiseed] project root: $ROOT"

SEEDS="0 6 12 24 27 33 36 74 93"

for s in $SEEDS; do
  sdir=$(printf "seed_%03d" $s)
  kernel="$ROOT/data/results/inversion_diagnostics/micro118/$sdir/kernel.npz"
  summary="$ROOT/data/results/inversion_diagnostics/micro119/$sdir/summary.json"

  echo "============================================================"
  echo "[multiseed] seed $s"
  echo "============================================================"

  if [ -f "$kernel" ]; then
    echo "[multiseed] kernel exists for seed $s, skip harvest"
  else
    echo "[multiseed] harvesting kernel for seed $s ..."
    t0=$(date +%s)
    MICRO118_SEED=$s MICRO118_POOL_SIZE=24 python3 notebooks/inversion/12_brightness_surface/micro118_kernel.py
    t1=$(date +%s)
    echo "[multiseed] kernel $s done in $((t1-t0)) s"
  fi

  if [ -f "$summary" ]; then
    echo "[multiseed] micro119 summary exists for seed $s, skip"
  else
    echo "[multiseed] running micro119 for seed $s ..."
    t0=$(date +%s)
    MICRO119_SEED=$s MICRO119_POOL=8 python3 notebooks/inversion/12_brightness_surface/micro119_attitude_isoshell.py
    t1=$(date +%s)
    echo "[multiseed] micro119 $s done in $((t1-t0)) s"
  fi
done

echo "[multiseed] ALL DONE"
