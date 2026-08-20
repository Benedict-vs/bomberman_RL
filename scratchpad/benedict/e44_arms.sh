#!/usr/bin/env bash
# E44 -- the retrain E42 claimed to be. BOTH arms from scratch (BM_WARM=""), so the field is
# isolated instead of being confounded with a rule_based-trained warm parent at alpha=0.0398.
# The warm row of the 2x2 already exists: E37 PLB2 s100-107 and E42 s200-207.
#
# NOTE bash, not zsh: this script relies on word-splitting of $FIELD. Run directly in zsh and
# the agent list arrives as ONE argument named "a b c" -- which is exactly how an earlier pilot
# silently failed with ModuleNotFoundError.
set -u
cd "$(dirname "$0")/../.."
RB="rule_based_agent rule_based_agent rule_based_agent"
MIX="ext_xiaoxiae_binary_v6 ext_aielka_ql_atom rule_based_agent"
CONC=5

launch () {   # launch <suffix-tag> <run-index> <field...>
  tag=$1; idx=$2; shift 2
  BM_QUIET_LOGS=1 BM_WARM="" BM_MODEL_SUFFIX="_e44scr_${tag}_s${idx}" BM_RUN_INDEX="$idx" \
    uv run python main.py play --agents benedict_task4 "$@" \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731 \
    > "scratchpad/benedict/e44/train_${tag}_s${idx}.log" 2>&1
  echo "trained ${tag} s${idx}"
}

for i in $(seq 300 307); do
  launch rb  "$i" $RB &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done
done
for i in $(seq 310 317); do
  launch mix "$i" $MIX &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done
done
wait
echo "E44: all 16 scratch runs done"
