#!/usr/bin/env bash
# Honest-mode constrained-anchor on the 5 m048 failure seeds (skip 79 — 0 constraints).
# NO oracle |omega| injection. n-mags=5 spans peak-count estimate ±30%.
#
# Run after stage 2 (surrogate scoring) finishes — depends on no shared resources.

set -e
cd "$(dirname "$0")/../.."

SEEDS_FOR_ANCHOR=(47 51 84 89)
LOG_ROOT="data/results/inversion_diagnostics/failure_seed_battery_2026_04_28"
mkdir -p "$LOG_ROOT"

echo "==== STAGE 3: constrained-anchor — HONEST (n-mags=5, no truth injection) ===="
python3 notebooks/inversion/score_constrained_anchor.py \
    --seeds "${SEEDS_FOR_ANCHOR[@]}" --traj-source m048 \
    --n-pab 800 --n-phi 80 --tolerance 0.15 --middle-frac 0.3 \
    --n-dirs 2000 --n-mags 5 --max-obs-mag 11.0 --workers 8 \
    --no-include-truth-mag \
    2>&1 | tee "$LOG_ROOT/anchor_honest.log"

# Tag honest copy explicitly
for seed in "${SEEDS_FOR_ANCHOR[@]}"; do
    sd="data/results/inversion_diagnostics/m103_hybrid_m048/seed_$(printf %03d $seed)"
    [ -f "$sd/constrained_anchor_ckpt.npz" ] && \
        cp "$sd/constrained_anchor_ckpt.npz" "$sd/constrained_anchor_ckpt_honest.npz"
done

echo "==== AUDIT ===="
python3 notebooks/inversion/audit_failure_seed_battery.py \
    2>&1 | tee "$LOG_ROOT/audit.log"

echo "==== DONE ===="
