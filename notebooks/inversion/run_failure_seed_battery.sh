#!/usr/bin/env bash
# Battery on the 5 m048 sampling-failure seeds (47, 51, 79, 84, 89).
# Stage 1: ensure each seed has fresh lofi_ckpt.npz (only re-run m103 if missing).
# Stage 2: score lofi pool with surrogate full-LC MSE (~40s/seed).
# Stage 3a: constrained-anchor with truth-mag injection (oracle reference).
# Stage 3b: constrained-anchor without truth-mag (honest, n-mags=5).
#
# We don't care about geo step (it's the alignment-cost ranking we're trying
# to replace). Stage 1 here exists only to populate lofi_ckpt.
#
# Skips seed 79 from stage 3 (0 constraints — anchor selection undefined).

set -e
cd "$(dirname "$0")/../.."

SEEDS=(47 51 79 84 89)
SEEDS_FOR_ANCHOR=(47 51 84 89)
LOG_ROOT="data/results/inversion_diagnostics/failure_seed_battery_2026_04_28"
mkdir -p "$LOG_ROOT"

echo "==== STAGE 1: ensure lofi_ckpt.npz exists for each seed ===="
for seed in "${SEEDS[@]}"; do
    d="data/results/inversion_diagnostics/m103_hybrid_m048/seed_$(printf %03d $seed)"
    if [ -f "$d/lofi_ckpt.npz" ]; then
        echo ">>> seed $seed: lofi_ckpt.npz already present, skipping m103"
        continue
    fi
    echo ">>> seed $seed: re-running m103 (will incur unnecessary geo step ~480s)"
    python3 notebooks/inversion/invert.py --seed "$seed" --traj-source m048 \
        --force-m103 --skip-m115 --skip-m126 --skip-lc-compare \
        2>&1 | tee "$LOG_ROOT/m103_seed_${seed}.log"
done

echo "==== STAGE 2: surrogate-LC scoring of lofi-300 pool ===="
python3 notebooks/inversion/score_lofi_surrogate.py \
    --seeds "${SEEDS[@]}" --traj-source m048 \
    2>&1 | tee "$LOG_ROOT/score_lofi_surrogate.log"

echo "==== STAGE 3a: constrained-anchor — ORACLE |omega| ===="
python3 notebooks/inversion/score_constrained_anchor.py \
    --seeds "${SEEDS_FOR_ANCHOR[@]}" --traj-source m048 \
    --n-pab 800 --n-phi 80 --tolerance 0.15 --middle-frac 0.3 \
    --n-dirs 2000 --n-mags 1 --max-obs-mag 11.0 --workers 8 \
    --include-truth-mag \
    2>&1 | tee "$LOG_ROOT/anchor_oracle.log"

# Tag oracle results before honest run overwrites them
for seed in "${SEEDS_FOR_ANCHOR[@]}"; do
    sd="data/results/inversion_diagnostics/m103_hybrid_m048/seed_$(printf %03d $seed)"
    [ -f "$sd/constrained_anchor_ckpt.npz" ] && \
        mv "$sd/constrained_anchor_ckpt.npz" "$sd/constrained_anchor_ckpt_oracle.npz"
done

echo "==== STAGE 3b: constrained-anchor — HONEST (n-mags=5) ===="
python3 notebooks/inversion/score_constrained_anchor.py \
    --seeds "${SEEDS_FOR_ANCHOR[@]}" --traj-source m048 \
    --n-pab 800 --n-phi 80 --tolerance 0.15 --middle-frac 0.3 \
    --n-dirs 2000 --n-mags 5 --max-obs-mag 11.0 --workers 8 \
    --no-include-truth-mag \
    2>&1 | tee "$LOG_ROOT/anchor_honest.log"

# Tag honest copy too
for seed in "${SEEDS_FOR_ANCHOR[@]}"; do
    sd="data/results/inversion_diagnostics/m103_hybrid_m048/seed_$(printf %03d $seed)"
    [ -f "$sd/constrained_anchor_ckpt.npz" ] && \
        cp "$sd/constrained_anchor_ckpt.npz" "$sd/constrained_anchor_ckpt_honest.npz"
done

echo "==== AUDIT ===="
python3 notebooks/inversion/audit_failure_seed_battery.py \
    2>&1 | tee "$LOG_ROOT/audit.log"

echo "==== DONE ===="
