#!/bin/zsh
# E37: wait for the training sweep, then evaluate every checkpoint.
#
# 180 evaluations (4 arms x 15 seeds x 3 checkpoints) at the pre-registered
# settings: 1000 rounds, eps = 0, BM_TIE_TOL=0.0 (default), validation seed
# 550731, 3 x rule_based_agent.
#
# THE ONE THING THAT MUST NOT BE FORGOTTEN
# ----------------------------------------
# BM_D8 changes what digit 8 *means*, so it changes which row of the table a
# state maps to. A table trained with BM_D8=parity and evaluated without it is
# read at the wrong indices -- the agent would look up ctl2's rows in PAR's
# table and the arm would silently measure noise. Every job below therefore
# exports the same BM_D8 it trained with. `callbacks.py` raises on an unknown
# value, so a typo fails loudly; an *omitted* value does not, which is exactly
# why it is set from the same case block as the training launcher.
#
# @20000 runs first, for all four arms, because that is the pre-registered
# reporting checkpoint -- H1/H2/H3 are readable after the first 60 evaluations.
# 5000 and 10000 follow and only feed the monotonicity argument (the strongest
# piece of evidence E36's PLB had, so worth having, but not primary).
#
# Run from the repo root:   ./scratchpad/benedict/e37_eval.sh
#   primary only (60 evals):     EPS="20000" ./scratchpad/benedict/e37_eval.sh
#   skip the wait (sweep done):  WAIT=0 ./scratchpad/benedict/e37_eval.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

BATCH=10
SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114)
EPS=${EPS:-"20000 5000 10000"}
ARMS=${ARMS:-"ctl2 PLB2 PAR SHF"}
WAIT=${WAIT:-1}
OUT=results/eval/task4_tournament
CKPT=checkpoints/benedict_task4
mkdir -p "$OUT"

if [[ "$WAIT" == "1" ]]; then
    echo "=== waiting for E37 training, $(date) ==="
    while pgrep -f "main.py play .* --train 1" > /dev/null; do sleep 60; done
    echo "=== training finished, $(date) ==="
fi

n_ckpt=$(ls "$CKPT"/q_table_e37_*__ep*.npy(N) | wc -l | tr -d ' ')
echo "checkpoints: $n_ckpt (expect 180)"
echo "tracebacks in training:"
grep -l Traceback results/train/task4_tournament/e37_*.out(N) || echo "  none"

# Build the job list: "suffix d8mode label"
jobs_list=()
for e in ${=EPS}; do
  for arm in ${=ARMS}; do
    case $arm in
      ctl2) mode="__NONE__" ;;
      PLB2) mode="stripe"   ;;
      PAR)  mode="parity"   ;;
      SHF)  mode="shuffle"  ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    for s in $SEEDS; do
      [[ -f "$CKPT/q_table_e37_${arm}_s${s}__ep${e}.npy" ]] || { echo "MISSING: e37_${arm}_s${s}__ep${e}"; continue }
      jobs_list+=("_e37_${arm}_s${s}__ep${e} $mode benedict_q_e37_${arm}_s${s}__ep${e}__task4_rb_val550731")
    done
  done
done

echo "=== ${#jobs_list} evaluations, batches of $BATCH, $(date) ==="

i=0
for job in $jobs_list; do
    suffix="${${(z)job}[1]}"
    mode="${${(z)job}[2]}"
    label="${${(z)job}[3]}"
    i=$((i + 1))

    if [[ -f "$OUT/$label.csv" ]]; then
        echo "skip (exists): $label"
        continue
    fi

    [[ "$mode" == "__NONE__" ]] && d8="" || d8="$mode"

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="$suffix" \
    BM_D8="$d8" \
    uv run python tools/evaluate.py \
        --agents benedict_task4 --opponents rule_based \
        --n-rounds 1000 --seed 550731 \
        --label "$label" --out-dir "$OUT" \
        --log-dir "logs/e37_$label" --quiet \
        > "$OUT/$label.log" 2>&1 &

    if (( i % BATCH == 0 )); then
        wait
        echo "  ... $i/${#jobs_list} done, $(date +%H:%M:%S)"
    fi
done
wait

echo "=== E37 evaluation finished, $(date) ==="
ls "$OUT"/benedict_q_e37_*.csv(N) | wc -l
grep -l Traceback "$OUT"/*.log(N) || echo "no tracebacks"
