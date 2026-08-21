#!/usr/bin/env bash
# E48 -- rung-4 reward calibration. Three new arms; the control is free (E37 PLB2 s100-107
# IS the current reward table). bash, not zsh: word-splitting of $FIELD is required.
set -u
cd "$(dirname "$0")/../.."
FIELD="rule_based_agent rule_based_agent rule_based_agent"
CONC=5
launch () {  # launch <tag> <idx> <env assignments...>
  tag=$1; idx=$2; shift 2
  env "$@" BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="_e49${tag}_s${idx}" BM_RUN_INDEX="$idx" \
    uv run python main.py play --agents benedict_task4 $FIELD \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731 \
    > "scratchpad/benedict/e49/train_${tag}_s${idx}.log" 2>&1
  echo "trained ${tag} s${idx}"
}
for i in $(seq 500 507); do launch K  "$i" BM_COIN=10 &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done; done
for i in $(seq 510 517); do launch R  "$i" BM_CRATE=2.0 &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done; done
for i in $(seq 520 527); do launch KR "$i" BM_COIN=10 BM_CRATE=2.0 &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done; done
wait
echo "E48: all 24 runs done"
