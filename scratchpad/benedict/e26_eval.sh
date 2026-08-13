#!/usr/bin/env bash
# E26 evaluation batch.
#
#   arms S, H  x 5 seeds x {20000 (primary), 40000} in the training field (cc)  = 20
#   arms S, H  x 5 seeds x {20000} in the transfer field (rule_based)           = 10
#   arm F      -- the FROZEN rung-2 table with BM_HUNT=1, no training at all --
#                in cc / rule_based / peaceful                                  =  3
#                plus the same table with HUNT off in cc, as F's own control    =  1
#
# Arm F carries no BM_MODEL_SUFFIX: benedict_task3/q_table.npy is the untouched
# rung-2 table (verified byte-identical to q_table_rung2ship.npy), and every
# training run wrote to checkpoints/ instead.
#
# think_ms from this batch is VOID -- 5 concurrent evaluations inflate per-step
# timing. `e26_thinktime.sh` re-measures it serially afterwards.
#
# Each run gets its own --log-dir: agents.py:226-228 opens the agent log "w" from
# a hardcoded path, so same-agent runs otherwise clobber each other.
set -u

MAXJOBS=${MAXJOBS:-5}
OUT=results/eval/task3_opponents
SEED=550731
N=300
mkdir -p "$OUT"

cmds=$(mktemp)
emit() {  # arm suffix hunt field label_field
  local hunt_env=""
  [ "$3" = "1" ] && hunt_env="BM_HUNT=1 "
  local sfx=""
  [ -n "$2" ] && sfx="BM_MODEL_SUFFIX=$2 "
  echo "${hunt_env}${sfx}uv run python tools/evaluate.py --agents benedict_task3 \
    --opponents $4 --n-rounds $N --seed $SEED --label benedict_q_e26_$5 \
    --out-dir $OUT --log-dir logs/eval_e26_$5 --quiet" >> "$cmds"
}

for arm in S H; do
  hunt=0; [ "$arm" = "H" ] && hunt=1
  for i in 20 21 22 23 24; do
    for ep in 20000 40000; do
      emit "$arm" "_e26_${arm}_s${i}__ep${ep}" "$hunt" coin_collector \
           "${arm}_s${i}__ep${ep}__task3_cc_val550731"
    done
    emit "$arm" "_e26_${arm}_s${i}__ep20000" "$hunt" rule_based \
         "${arm}_s${i}__ep20000__task3_rb_val550731"
  done
done

# Arm F: the feature change with no training, plus its HUNT-off control.
emit F "" 1 coin_collector "F_frozen__task3_cc_val550731"
emit F "" 1 rule_based     "F_frozen__task3_rb_val550731"
emit F "" 1 peaceful       "F_frozen__task3_peaceful_val550731"
emit F "" 0 coin_collector "F_control_huntoff__task3_cc_val550731"

echo "$(wc -l < "$cmds") evaluations, ${MAXJOBS} at a time"
while IFS= read -r cmd; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 2; done
  bash -c "$cmd" &
done < "$cmds"
wait
rm -f "$cmds"

echo "=== done ==="
echo "CSVs: $(ls "$OUT"/benedict_q_e26_*.csv 2>/dev/null | wc -l | tr -d ' ') (expect 34)"
