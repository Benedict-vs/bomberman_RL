#!/bin/zsh
# E36: give the idle digit 8 an opponent-distance bucket in danger rows.
#
#   OPP  BM_OPPDIST=1      digit 8 = bucketed BFS distance to nearest opponent
#   PLB  BM_OPPDIST_PLB=1  digit 8 = (x+y) % 4 -- PLACEBO, matched arity, no
#                          opponent information. Not optional: filling 40 960
#                          previously-unreachable rows is itself a perturbation.
#
# ctl is E33 ctl, already measured (score 3.719 / won 0.372). RUN_INDEX 100-104
# matches it, so with --seed 810731 arenas and the exploration stream match and
# the comparison is paired at the run level.
#
# FEATURE_SIZES is UNCHANGED -- 64 000 rows either way -- so the shipped table is
# a factor-1 warm-start parent. But a factor-1 copy would leave every d8 > 0
# danger row empty, so the parent is broadcast first (see broadcast_oppdist.py).
# Both arms share that parent: the placebo populates the same rows, or it is not
# a placebo for the thing being tested.
#
# Requires callbacks.py at E36 (BM_OPPDIST, BM_OPPDIST_PLB, opponent_bucket) and
# EXPERIMENT = "e36" in train.py.
#
# Run from the repo root:   ./scratchpad/benedict/e36_arms.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
BATCH=10
WORLD_SEED=810731
SEEDS=(100 101 102 103 104)
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

PARENT="$CKPT/q_table_e36parent.npy"
echo "=== building the broadcast warm-start parent, $(date) ==="
uv run python scratchpad/benedict/broadcast_oppdist.py \
    agent_code/benedict_task4/q_table.npy "$PARENT" || exit 1

echo "=== E36 started at $(date) ==="

i=0
for arm in OPP PLB; do
  if [[ $arm == OPP ]]; then od=1; pl=0; else od=0; pl=1; fi
  for seed in $SEEDS; do
    i=$((i + 1))

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e36_${arm}_s${seed}" \
    BM_ARM="$arm" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_OPPDIST="$od" \
    BM_OPPDIST_PLB="$pl" \
    BM_WARM=_e36parent \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        --seed "$WORLD_SEED" \
        > "results/train/task4_tournament/e36_${arm}_s${seed}.out" 2>&1 &

    echo "launched ${arm} s${seed}  (BM_OPPDIST=${od} BM_OPPDIST_PLB=${pl})"
    if (( i % BATCH == 0 )); then wait; echo "  ... $i/10 $(date +%H:%M:%S)"; fi
  done
done
wait

echo "=== E36 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e36_*__ep*.npy(N) | wc -l | tr -d ' ')  (expect 30)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e36_*.out(N) || echo "  none"
