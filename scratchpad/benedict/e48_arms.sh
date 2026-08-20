#!/usr/bin/env bash
# E48 -- rung-4 reward calibration. Three new arms; the control is free (E37 PLB2 s100-107
# IS the current reward table). bash, not zsh: word-splitting of $FIELD is required.
set -u
cd "$(dirname "$0")/../.."
FIELD="rule_based_agent rule_based_agent rule_based_agent"
CONC=5
launch () {  # launch <tag> <idx> <env assignments...>
  tag=$1; idx=$2; shift 2
  env "$@" BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="_e48${tag}_s${idx}" BM_RUN_INDEX="$idx" \
    uv run python main.py play --agents benedict_task4 $FIELD \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731 \
    > "scratchpad/benedict/e48/train_${tag}_s${idx}.log" 2>&1
  echo "trained ${tag} s${idx}"
}
for i in $(seq 400 407); do launch C  "$i" BM_CRATE=0.25 &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done; done
for i in $(seq 410 417); do launch D  "$i" BM_GOT_KILLED=0 &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done; done
for i in $(seq 420 427); do launch CD "$i" BM_CRATE=0.25 BM_GOT_KILLED=0 &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done; done
wait
echo "E48: all 24 runs done"
