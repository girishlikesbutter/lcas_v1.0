#!/usr/bin/env bash
# Run invert.py serially for seeds 59/64/67 with M115_SORT_BY=surr_q0polish_mse.
# m103 is skipped (geo_ckpt already on disk); only m115 + m126 + wrappedbest run.
# Logs land in rerank_experiment/pipeline_test_2026_04_22/logs/.
set -u

cd /home/girish/projects/lcas_v1.0

export M115_SORT_BY=surr_q0polish_mse
LOG_DIR=data/results/inversion_diagnostics/rerank_experiment/pipeline_test_2026_04_22/logs
mkdir -p "$LOG_DIR"

for SEED in 59 64 67; do
    echo "=== $(date +%H:%M:%S) starting seed $SEED (M115_SORT_BY=$M115_SORT_BY) ==="
    python3 notebooks/inversion/invert.py \
        --seed "$SEED" --traj-source m048 --skip-m103 \
        >"$LOG_DIR/seed_${SEED}.log" 2>&1
    RC=$?
    echo "=== $(date +%H:%M:%S) seed $SEED done rc=$RC ==="
done
echo "ALL DONE $(date +%H:%M:%S)"
