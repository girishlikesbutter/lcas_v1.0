#!/usr/bin/env bash
# run_phase_B_descending.sh — sequential invert.py for 3 m048 seeds at
# decreasing phase angles (90, 49, 23) to verify consistency in the safe band.
#
# Each seed saturates Pool(24) so these MUST run one at a time
# (feedback_no_parallel_cpu.md).
#
# Per-seed logs land in data/results/inversion_diagnostics/phase_B_gap_logs/
# for easy tail + monitor. Top-level log records RESULT lines for all 3.

set -u  # abort on undefined var; NOT -e because we want to continue on per-seed failure
cd "$(dirname "$0")/../.."

LOG_DIR="data/results/inversion_diagnostics/phase_B_gap_logs"
TOP_LOG="$LOG_DIR/descending_run.log"
mkdir -p "$LOG_DIR"

SEEDS=(90 49 23)
PHASES=("50.8 deg" "38.9 deg" "30.4 deg")

echo "=== Phase-B descending-phase run ===" | tee "$TOP_LOG"
echo "Start: $(date -Iseconds)" | tee -a "$TOP_LOG"
echo "Seeds: ${SEEDS[*]}" | tee -a "$TOP_LOG"
echo | tee -a "$TOP_LOG"

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    PHASE="${PHASES[$i]}"
    SEED_LOG="$LOG_DIR/invert_seed$(printf '%03d' "$SEED").log"

    echo "---" | tee -a "$TOP_LOG"
    echo "[$(date -Iseconds)] Seed $SEED  (phase $PHASE)" | tee -a "$TOP_LOG"
    echo "  Log: $SEED_LOG" | tee -a "$TOP_LOG"

    python3 -u notebooks/inversion/invert.py --seed "$SEED" --traj-source m048 \
        2>&1 | tee "$SEED_LOG"
    RC="${PIPESTATUS[0]}"

    # Extract RESULT line for the top-level summary
    RESULT_LINE=$(grep -E "^RESULT seed=" "$SEED_LOG" | tail -1 || true)
    echo "  rc=$RC  ${RESULT_LINE:-(no RESULT line found)}" | tee -a "$TOP_LOG"
done

echo | tee -a "$TOP_LOG"
echo "=== Descending run complete: $(date -Iseconds) ===" | tee -a "$TOP_LOG"
echo | tee -a "$TOP_LOG"
echo "Summary:" | tee -a "$TOP_LOG"
grep -E "^RESULT seed=" "$TOP_LOG" | tee -a "$TOP_LOG" || true
