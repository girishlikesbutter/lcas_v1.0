#!/usr/bin/env bash
# Pre-wipe the 21 m048 cohort seeds (excluding 91, already done) for the
# union_3cost K=7 batch. Archives m115 dir (preserves baseline) and removes
# stale m126/wrappedbest/invert dirs so the batch driver runs them fresh.
#
# Seed 91 is already on disk under the patched union_3cost run; its
# .noise_fix_v1.done marker will cause the batch driver to skip it.
set -euo pipefail

cd "$(dirname "$0")/../../.."
ROOT="data/results/inversion_diagnostics"

SEEDS=(6 7 8 11 16 17 34 45 47 48 51 57 59 64 67 71 78 79 84 89 99)
ARCHIVE_TAG="prebatch_union3_K7"

n_archived=0
n_wiped=0
for s in "${SEEDS[@]}"; do
  S=$(printf '%03d' "$s")
  M115_DIR="$ROOT/m115_surrogate_pipeline_m048/seed_$S"
  M115_BAK="$ROOT/m115_surrogate_pipeline_m048/seed_$S.${ARCHIVE_TAG}.bak"
  M126_DIR="$ROOT/m126_wrapped_m048/seed_$S"
  WB_DIR="$ROOT/wrappedbest_m048_seed$S"
  INV_DIR="$ROOT/invert_m048_seed$S"

  if [ -d "$M115_DIR" ]; then
    if [ -d "$M115_BAK" ]; then
      echo "seed $s: archive already exists ($M115_BAK), removing live dir"
      rm -rf "$M115_DIR"
    else
      mv "$M115_DIR" "$M115_BAK"
      echo "seed $s: archived m115 -> $M115_BAK"
    fi
    n_archived=$((n_archived + 1))
  fi
  for d in "$M126_DIR" "$WB_DIR" "$INV_DIR"; do
    if [ -d "$d" ]; then
      rm -rf "$d"
      n_wiped=$((n_wiped + 1))
    fi
  done
done

echo "---"
echo "archived $n_archived m115 dirs"
echo "wiped $n_wiped downstream dirs (m126/wb/invert)"
echo "21 seeds now ready for fresh union_3cost K=7 batch"
