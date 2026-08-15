#!/bin/zsh
# E35: price KILLED_OPPONENT, which has been 0.0 in every rung-4 run.
#
#   K5   BM_KILL=5              one coin's worth at this table's scale
#   K25  BM_KILL=25             the game's own 5:1 kill:coin ratio, given BM_COIN=5
#   PLB  BM_COIN=7, BM_KILL=0   PLACEBO -- comparable Q-magnitude, no kill information
#
# The placebo is E32's carried-forward requirement #4, never honoured: without it
# "pricing kills" and "any perturbation of a few Q-units" are not separable. The
# middle dose is deliberately omitted -- in E32's probe the middle dose was the
# one that collapsed.
#
# Control is E33 ctl, already measured. RUN_INDEX 100-104 deliberately matches
# it, so with --seed 810731 the arenas and the exploration stream match and the
# comparison is paired at the run level.
#
# Requires EXPERIMENT = "e35" and CHECKPOINTS = (5_000, 10_000, 20_000) in
# train.py. (E34 set CHECKPOINTS to include 500/2000 -- put it back, or the
# sweep writes ten checkpoints per run for no reason.)
#
# Run from the repo root:   ./scratchpad/benedict/e35_arms.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
BATCH=8
WORLD_SEED=810731
SEEDS=(100 101 102 103 104)
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

if [[ ! -f "$CKPT/q_table_rung2ship.npy" ]]; then
    cp agent_code/benedict_task4/q_table.npy "$CKPT/q_table_rung2ship.npy"
fi

echo "=== E35 started at $(date) ==="

i=0
for arm in K5 K25 PLB; do
  case $arm in
    K5)  kill=5;  coin=5 ;;
    K25) kill=25; coin=5 ;;
    PLB) kill=0;  coin=7 ;;
  esac
  for seed in $SEEDS; do
    i=$((i + 1))

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e35_${arm}_s${seed}" \
    BM_ARM="$arm" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_KILL="$kill" \
    BM_COIN="$coin" \
    BM_WARM=_rung2ship \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        --seed "$WORLD_SEED" \
        > "results/train/task4_tournament/e35_${arm}_s${seed}.out" 2>&1 &

    echo "launched ${arm} s${seed}  (BM_KILL=${kill} BM_COIN=${coin})"
    if (( i % BATCH == 0 )); then wait; echo "  ... $i/15 $(date +%H:%M:%S)"; fi
  done
done
wait

echo "=== E35 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e35_*__ep*.npy(N) | wc -l | tr -d ' ')  (expect 45)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e35_*.out(N) || echo "  none"
