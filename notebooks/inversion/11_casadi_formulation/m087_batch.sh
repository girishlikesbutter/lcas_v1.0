#!/bin/bash
# Batch runner for micro87 pipeline across multiple seeds.
# Usage: bash micro87_batch.sh [seed1 seed2 ...]
# Default: all micro77 working seeds

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/micro87_fast_grid.py"
RESULTS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/data/results/inversion_diagnostics"

# Default seeds: micro77 working seeds first, then failed seeds
if [ $# -eq 0 ]; then
    SEEDS=(0 6 12 14 24 27 33 36 74 93)
else
    SEEDS=("$@")
fi

echo "========================================"
echo "micro87 batch — ${#SEEDS[@]} seeds"
echo "========================================"

SUMMARY_FILE="$RESULTS_DIR/micro87_batch_summary.txt"
echo "seed  w_dir  q0_err  w_mag%  time_s  status" > "$SUMMARY_FILE"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Running seed $SEED ---"
    t_start=$SECONDS
    MICRO77_SEED=$SEED python3 "$SCRIPT" 2>&1 | tail -1
    t_elapsed=$((SECONDS - t_start))

    # Extract results from JSON
    RESULT_FILE="$RESULTS_DIR/micro87_fast_seed$(printf '%03d' $SEED)/result.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
w = d['winner']
status = 'OK' if w['w0_err'] < 5 else ('PARTIAL' if w['w0_err'] < 10 else 'FAIL')
print(f\"{d['traj_seed']:4d}  {w['w0_err']:5.1f}  {w['q0_err']:6.1f}  {w['w_mag_err_pct']:+5.1f}  {d['timing']['total_s']:6.0f}  {status}\")
" >> "$SUMMARY_FILE"
        python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
w = d['winner']
status = 'OK' if w['w0_err'] < 5 else ('PARTIAL' if w['w0_err'] < 10 else 'FAIL')
print(f\"  seed {d['traj_seed']:3d}: w_dir={w['w0_err']:5.1f}° q0={w['q0_err']:6.1f}° w_mag={w['w_mag_err_pct']:+5.1f}% t={d['timing']['total_s']:.0f}s {status}\")
"
    else
        echo "  seed $SEED: RESULT FILE MISSING"
        echo "$SEED  -1  -1  -1  -1  MISSING" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "========================================"
echo "BATCH SUMMARY"
echo "========================================"
cat "$SUMMARY_FILE"
