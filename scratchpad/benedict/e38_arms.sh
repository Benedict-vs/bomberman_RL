#!/bin/zsh
# E38: is 20 000 episodes the right budget on rung 4, or an inherited one?
#
# 20k was chosen on rung 2 (E23) and never re-validated here. The rung-2 episode
# response is NON-MONOTONE and the mechanism is named: 40k lost ~7 crates, 200k
# turned 97.31 into 62.85 by "re-rolling near-tie rows" (E18), and 300k was the
# best in its arm (97.59 dev / 97.45 held-out, E21). So the curve dips and
# recovers, and no extrapolation from three early checkpoints is valid.
#
# One arm, five seeds, one long run each. The control is the SAME run's @20 000
# checkpoint, so the comparison is within-run and paired by seed -- no separate
# control arm is needed and none could be afforded at this horizon.
#
# BATCH=5, not 10: with five concurrent runs each lands on a performance core
# instead of contending, so wall-clock per run drops enough to fit one night.
#
# Run from the repo root:   ./scratchpad/benedict/e38_arms.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=300000
BATCH=5
WORLD_SEED=810731
SEEDS=(120 121 122 123 124)
CKPTS="20000,40000,80000,160000,300000"
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

if [[ ! -f "$CKPT/q_table_parent.npy" ]]; then
    echo "missing warm parent $CKPT/q_table_parent.npy"; exit 1
fi

echo "=== E38 started at $(date) ==="
echo "${#SEEDS} runs x $EPISODES episodes, $BATCH concurrent, checkpoints $CKPTS"

i=0
for seed in $SEEDS; do
    i=$((i + 1))

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e38_s${seed}" \
    BM_ARM="e38" \
    BM_RUN_INDEX="$seed" \
    BM_CHECKPOINTS="$CKPTS" \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        --seed "$WORLD_SEED" \
        > "results/train/task4_tournament/e38_s${seed}.out" 2>&1 &

    echo "launched s${seed}"
    if (( i % BATCH == 0 )); then wait; echo "  ... $i done $(date +%H:%M:%S)"; fi
done
wait

echo "=== E38 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e38_*__ep*.npy(N) | wc -l | tr -d ' ') (expect 25)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e38_*.out(N) || echo "  none"
