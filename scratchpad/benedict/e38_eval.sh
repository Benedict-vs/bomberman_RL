#!/bin/zsh
# E38: wait for the 300 000-episode sweep, then evaluate EVERY checkpoint.
#
# 25 evaluations (5 seeds x 5 checkpoints) at the pre-registered settings: 1000
# rounds, eps = 0, validation seed 550731, 3 x rule_based_agent.
#
# "Every checkpoint" is not thoroughness for its own sake. E21 left five
# 300 000-episode tables on disk unevaluated; when they were finally measured
# they were the best in the arm and reversed a scored prediction. The cheap half
# of that experiment was the training.
#
# No BM_D8 to export: the digit-8 map is the default since the E37 cleanup, and
# these tables were trained under it.
#
# Run from the repo root:   ./scratchpad/benedict/e38_eval.sh
#   skip the wait:          WAIT=0 ./scratchpad/benedict/e38_eval.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

BATCH=10
SEEDS=(120 121 122 123 124)
EPS=(20000 40000 80000 160000 300000)
WAIT=${WAIT:-1}
OUT=results/eval/task4_tournament
CKPT=checkpoints/benedict_task4
mkdir -p "$OUT"

if [[ "$WAIT" == "1" ]]; then
    echo "=== waiting for E38 training, $(date) ==="
    while pgrep -f "main.py play .* --train 1" > /dev/null; do sleep 120; done
    echo "=== training finished, $(date) ==="
fi

echo "checkpoints on disk: $(ls "$CKPT"/q_table_e38_*__ep*.npy(N) | wc -l | tr -d ' ') (expect 25)"
grep -l Traceback results/train/task4_tournament/e38_*.out(N) || echo "no training tracebacks"

jobs_list=()
for e in $EPS; do
  for s in $SEEDS; do
    [[ -f "$CKPT/q_table_e38_s${s}__ep${e}.npy" ]] || { echo "MISSING: e38_s${s}__ep${e}"; continue }
    jobs_list+=("_e38_s${s}__ep${e} benedict_q_e38_s${s}__ep${e}__task4_rb_val550731")
  done
done

echo "=== ${#jobs_list} evaluations, batches of $BATCH, $(date) ==="

i=0
for job in $jobs_list; do
    suffix="${job%% *}"
    label="${job##* }"
    i=$((i + 1))
    if [[ -f "$OUT/$label.csv" ]]; then echo "skip (exists): $label"; continue; fi

    BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="$suffix" \
    uv run python tools/evaluate.py \
        --agents benedict_task4 --opponents rule_based \
        --n-rounds 1000 --seed 550731 \
        --label "$label" --out-dir "$OUT" \
        --log-dir "logs/e38_$label" --quiet \
        > "$OUT/$label.log" 2>&1 &

    if (( i % BATCH == 0 )); then wait; echo "  ... $i/${#jobs_list} done, $(date +%H:%M:%S)"; fi
done
wait

echo "=== E38 evaluation finished, $(date) ==="
grep -l Traceback "$OUT"/benedict_q_e38_*.log(N) || echo "no tracebacks"
