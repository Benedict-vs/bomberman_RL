#!/bin/zsh
# E31: BM_STEP_COST=0. One switch, everything else E30 arm T verbatim.
#
# Audit 4 established that the -0.1 per-step cost is action-independent and so
# contributes nothing to the argmax, while acting as a constant downward ratchet
# on whichever action is currently greedy: the visit-weighted action gap falls
# 1.788 -> 0.668 and 46 % of steps end up inside a period-2 cycle. At
# STEP_COST = 0 the drive is proportional rather than additive.
#
# Run from the repo root:   ./scratchpad/benedict/e31_arms.sh
#
# Seeds 80-84: NOT 60-64, which audit 3 showed were contaminated by E27's
# disclosed pilot (seed 60), and not 70/71, which audit 4's own probe used.
#
# CHECKPOINTS in train.py must be (5_000, 10_000, 20_000) -- unchanged from E30.
# Concurrency uses fixed batches with a plain `wait`; `jobs -rp` returns nothing
# inside $( ) because the subshell has no job table, which is why E30's launcher
# started all ten runs at once.

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
SEEDS=(80 81 82 83 84)
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

if [[ ! -f "$CKPT/q_table_rung2ship.npy" ]]; then
    cp agent_code/benedict_task4/q_table.npy "$CKPT/q_table_rung2ship.npy"
fi

echo "=== E31 started at $(date) ==="

for seed in $SEEDS; do
    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e31_S0_s${seed}" \
    BM_ARM=S0 \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_WARM=_rung2ship \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        > "results/train/task4_tournament/e31_S0_s${seed}.out" 2>&1 &
    echo "launched S0 s${seed}"
done

wait
echo "=== E31 finished at $(date) ==="

echo "checkpoints: $(ls "$CKPT"/q_table_e31_*__ep*.npy(N) | wc -l | tr -d ' ')  (expect 15)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e31_*.out(N) || echo "  none"
