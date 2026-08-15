#!/bin/zsh
# E34: is the ceiling a reachable fixed point?
#
# Each run warm-starts from its OWN E33 control table with the argmax re-pointed
# to digit 6 in every danger row, then trains the E33 control configuration
# verbatim. The objective is untouched -- only the initial condition changes.
#
# Why this and not another reward: the table sits at its own Bellman fixed point
# on the decisive row (residual -0.087 over 1457 transitions, death correctly
# priced on the 5.1 % of visits where it fires), and audit 5's DOWN/UP
# bimodality shows a second fixed point exists at identical hyperparameters. A
# reward change moves where the fixed points are; only an initial condition
# selects among them.
#
# RUN_INDEX 100-104 deliberately matches the E33 control, because run s100 starts
# from E33 ctl s100's forced table. With --seed 810731 all runs share one arena
# sequence and the exploration stream is matched, so the comparison is paired at
# the run level.
#
# Requires EXPERIMENT = "e34" in train.py.
#
# Run from the repo root:   ./scratchpad/benedict/e34_arms.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
WORLD_SEED=810731
SEEDS=(100 101 102 103 104)
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

echo "=== building forced warm-start tables, $(date) ==="
for seed in $SEEDS; do
    src="$CKPT/q_table_e33_ctl_s${seed}__ep20000.npy"
    if [[ ! -f "$src" ]]; then echo "MISSING $src -- E33 control not on disk"; exit 1; fi
    uv run python scratchpad/benedict/force_escape.py "$src" "$CKPT/q_table_a6esc_s${seed}.npy" || exit 1
done

echo "=== E34 started at $(date) ==="

for seed in $SEEDS; do
    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e34_W_s${seed}" \
    BM_ARM=W \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_WARM="_a6esc_s${seed}" \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        --seed "$WORLD_SEED" \
        > "results/train/task4_tournament/e34_W_s${seed}.out" 2>&1 &
    echo "launched W s${seed}  (warm start _a6esc_s${seed})"
done
wait

echo "=== E34 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e34_*__ep*.npy(N) | wc -l | tr -d ' ')  (expect 25)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e34_*.out(N) || echo "  none"
