#!/bin/zsh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=scratchpad/audit5/eval
mkdir -p "$OUT"
for arm in a5ctl a5k15 a5k30; do
 for seed in 90 91; do
  for ep in 5000 10000 20000; do
    f="checkpoints/benedict_task4/q_table_${arm}_s${seed}__ep${ep}.npy"
    [[ -f $f ]] || { echo "missing $f"; continue; }
    BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="_${arm}_s${seed}__ep${ep}" \
      uv run python tools/evaluate.py --agents benedict_task4 --opponents rule_based \
        --n-rounds 300 --seed 550731 --label "${arm}_s${seed}__ep${ep}" --out-dir "$OUT" --quiet &
  done
 done
done
wait
echo done
