#!/bin/zsh
# E30: wait for the training sweep to finish, then evaluate every checkpoint.
#
# 32 evaluations, all at the pre-registered settings: 1000 rounds, eps = 0,
# BM_TIE_TOL=0.0 (the default), validation seed 550731, 3 x rule_based_agent.
#
#   F        the frozen table -- the control. The committed val550731 run is
#            only 300 rounds, and E30 pre-registered 1000, so it is re-measured.
#   S        the D4-folded table, untrained. Not part of E30's design: it
#            separates arm TD's *initialisation* from its update rule, which the
#            training curve suggested were confounded. E29 skipped this arm when
#            its gate failed; it is cheap and it makes P3 readable.
#   T/TD     5 seeds x 3 checkpoints each.
#
# Batches of 8 with a plain `wait` -- deliberately NOT `jobs -rp`, which returns
# nothing inside $( ) because the subshell has no job table. That is why the
# training sweep launched all ten runs at once instead of five.

set -u
cd "$(dirname "$0")/../.." || exit 1

BATCH=8
SEEDS=(60 61 62 63 64)
EPS=(5000 10000 20000)
OUT=results/eval/task4_tournament
CKPT=checkpoints/benedict_task4
mkdir -p "$OUT"

echo "=== waiting for E30 training, $(date) ==="
while pgrep -f "main.py play .* --train 1" > /dev/null; do sleep 60; done
echo "=== training finished, $(date) ==="

n_ckpt=$(ls "$CKPT"/q_table_e30_*__ep*.npy(N) | wc -l | tr -d ' ')
echo "checkpoints: $n_ckpt (expect 30)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e30_*.out(N) || echo "  none"
if [[ "$n_ckpt" != "30" ]]; then
    echo "WARNING: proceeding anyway with what exists"
fi

# Build the job list: "suffix label"
jobs_list=()
jobs_list+=("__NONE__ benedict_q_e30_F__task4_rb_val550731")
jobs_list+=("_d4 benedict_q_e30_S_folded__task4_rb_val550731")
for arm in T TD; do
  for s in $SEEDS; do
    for e in $EPS; do
      [[ -f "$CKPT/q_table_e30_${arm}_s${s}__ep${e}.npy" ]] || continue
      jobs_list+=("_e30_${arm}_s${s}__ep${e} benedict_q_e30_${arm}_s${s}__ep${e}__task4_rb_val550731")
    done
  done
done

echo "=== ${#jobs_list} evaluations, batches of $BATCH, $(date) ==="

i=0
for job in $jobs_list; do
    suffix="${job%% *}"
    label="${job##* }"
    i=$((i + 1))

    if [[ -f "$OUT/$label.csv" ]]; then
        echo "skip (exists): $label"
        continue
    fi

    if [[ "$suffix" == "__NONE__" ]]; then
        env_suffix=""
    else
        env_suffix="$suffix"
    fi

    BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="$env_suffix" \
    uv run python tools/evaluate.py \
        --agents benedict_task4 --opponents rule_based \
        --n-rounds 1000 --seed 550731 \
        --label "$label" --out-dir "$OUT" \
        --log-dir "logs/e30_$label" --quiet \
        > "$OUT/$label.log" 2>&1 &

    if (( i % BATCH == 0 )); then
        wait
        echo "  ... $i/${#jobs_list} done, $(date +%H:%M:%S)"
    fi
done
wait

echo "=== E30 evaluation finished, $(date) ==="
ls "$OUT"/benedict_q_e30_*.csv(N) | wc -l
grep -l Traceback "$OUT"/*.log(N) || echo "no tracebacks"
