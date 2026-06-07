#!/usr/bin/env bash
# s100(clamp+anchor-cap) -> s110(surrogate polish) for the slow-tumbler sweep.
# Sequential: one Pool(24) job at a time (no CPU oversubscription).
set -u
cd /home/girish/projects/lcas_v1.0/notebooks/inversion/survey
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/s100 results/s110

# s113/contract_slow-tumbler-generality: 116 FIRST as the regression canary (the decimation
# fix touches its branch); then the uncertain slow seeds. fix #3: padded output-dir check.
for SEED in 116 10 42 31; do
  PAD=$(printf '%03d' $SEED)
  echo "===== SWEEP seed ${SEED}: s100 start $(date +%H:%M:%S) ====="
  S100_SEED=$SEED S100_CLAMP=1 S100_ANCHOR_CAP=1 \
    python experiments/s100_5step_proto.py > results/s100/seed${SEED}_capped_run.log 2>&1
  echo "  s100 seed ${SEED} exit=$? $(date +%H:%M:%S)"
  if [ -f results/s100/seed${PAD}/invert.npz ]; then
    echo "  s110 polish seed ${SEED} start $(date +%H:%M:%S)"
    S110_SEED=$SEED python experiments/s110_polish_116.py > results/s110/polish_${SEED}_run.log 2>&1
    echo "  s110 seed ${SEED} exit=$? $(date +%H:%M:%S)"
  else
    echo "  !! s100 seed ${SEED} produced no invert.npz (aborted/no survivors) — skipping polish"
  fi
done
echo "===== SWEEP DONE $(date +%H:%M:%S) ====="
