#!/usr/bin/env bash
# s032 — full 100-seed cohort run of s020 pipeline with the geo-fail gate OFF
# (production / fast mode; ~75 s/seed wall = ~125 min total).
#
# Each seed already saturates Pool(8); seeds run in serial.
set -u

PROJECT_ROOT="/home/girish/projects/lcas_v1.0"
SCRIPT="${PROJECT_ROOT}/notebooks/inversion/survey/experiments/s020_seed_pipeline.py"
OUT_ROOT="${PROJECT_ROOT}/notebooks/inversion/survey/results/s032_cohort_fast"
LOG_DIR="${OUT_ROOT}/logs"
SUMMARY="${OUT_ROOT}/cohort_progress.csv"

mkdir -p "${LOG_DIR}"
echo "seed,wall_s,n_survivors,n_passed_geo_only,n_rejected_both,nearest_cell_pct,status" > "${SUMMARY}"

t0=$(date +%s)
for seed in $(seq 0 99); do
    seed_padded=$(printf "%03d" "$seed")
    seed_dir="${OUT_ROOT}/seed${seed_padded}"
    log_file="${LOG_DIR}/seed${seed_padded}.log"

    if [[ -f "${seed_dir}/summary.json" ]]; then
        wall_s=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['timing']['total_wall_s'])" 2>/dev/null || echo "0")
        n_surv=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['categorisation']['passed_both'])" 2>/dev/null || echo "0")
        n_geo=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['categorisation']['passed_geo_only'])" 2>/dev/null || echo "0")
        n_rej=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['categorisation']['rejected_both'])" 2>/dev/null || echo "0")
        nearest=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['bracket']['nearest_cell_pct'])" 2>/dev/null || echo "")
        echo "${seed},${wall_s},${n_surv},${n_geo},${n_rej},${nearest},CACHED" >> "${SUMMARY}"
        echo "[$(date +%H:%M:%S)] seed ${seed_padded} CACHED — skip" >&2
        continue
    fi

    seed_t0=$(date +%s)
    if python "${SCRIPT}" "${seed}" --out-root "${OUT_ROOT}" > "${log_file}" 2>&1; then
        wall_s=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['timing']['total_wall_s'])" 2>/dev/null || echo "0")
        n_surv=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['categorisation']['passed_both'])" 2>/dev/null || echo "0")
        n_geo=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['categorisation']['passed_geo_only'])" 2>/dev/null || echo "0")
        n_rej=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['categorisation']['rejected_both'])" 2>/dev/null || echo "0")
        nearest=$(python -c "import json; print(json.load(open('${seed_dir}/summary.json'))['bracket']['nearest_cell_pct'])" 2>/dev/null || echo "")
        echo "${seed},${wall_s},${n_surv},${n_geo},${n_rej},${nearest},OK" >> "${SUMMARY}"
        seed_dt=$(( $(date +%s) - seed_t0 ))
        elapsed=$(( $(date +%s) - t0 ))
        echo "[$(date +%H:%M:%S)] seed ${seed_padded} OK — wall ${seed_dt}s, surv ${n_surv}, nearest_cell ${nearest}% | cohort elapsed $((elapsed/60))m" >&2
    else
        echo "${seed},,,,,,FAIL" >> "${SUMMARY}"
        echo "[$(date +%H:%M:%S)] seed ${seed_padded} FAIL — see ${log_file}" >&2
    fi
done

t1=$(date +%s)
echo "=== Cohort complete in $(( (t1 - t0) / 60 )) min ===" >&2
echo "Summary: ${SUMMARY}" >&2
