#!/bin/zsh
# E32: price a suicide separately from being killed. One switch, two levels.
#
# A suicide fires KILLED_SELF *and* GOT_KILLED (environment.py:251 and :264), so
# the two add. Leaving GOT_KILLED at -5 and setting KILLED_SELF prices own-bomb
# death without touching the price of an opponent's kill:
#
#   K15  BM_KILLED_SELF=-10  -> suicide 15, enemy kill 5
#   K30  BM_KILLED_SELF=-25  -> suicide 30, enemy kill 5
#
# Control is E31 arm S0 (KILLED_SELF 0 -> suicide 5), already measured.
# Everything else is E31 verbatim: BM_STEP_COST=0, BM_CRATE=1.0, warm start from
# the frozen table.
#
# Run from the repo root:   ./scratchpad/benedict/e32_arms.sh
#
# Seeds 90-94: unused. 60-64 were contaminated by E27's pilot (audit 3), 70/71
# by audit 4's probe, 80-84 by E31.
#
# Before running, train.py needs EXPERIMENT = "e32" -- it only labels the
# training CSVs (checkpoint names come from BM_MODEL_SUFFIX), but mislabelled
# runs are how E30's logs ended up saying e27.
#
# Concurrency: fixed batches with a plain `wait`. `jobs -rp` returns nothing
# inside $( ) because the subshell has no job table -- that is why E30's
# launcher started all ten runs at once instead of five.

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
BATCH=5
SEEDS=(90 91 92 93 94)
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

if [[ ! -f "$CKPT/q_table_rung2ship.npy" ]]; then
    cp agent_code/benedict_task4/q_table.npy "$CKPT/q_table_rung2ship.npy"
fi

echo "=== E32 started at $(date) ==="

i=0
for arm in K15 K30; do
  if [[ $arm == K15 ]]; then ks=-10; else ks=-25; fi
  for seed in $SEEDS; do
    i=$((i + 1))

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e32_${arm}_s${seed}" \
    BM_ARM="$arm" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_KILLED_SELF="$ks" \
    BM_WARM=_rung2ship \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        > "results/train/task4_tournament/e32_${arm}_s${seed}.out" 2>&1 &

    echo "launched ${arm} s${seed} (KILLED_SELF=${ks})"
    if (( i % BATCH == 0 )); then wait; echo "  ... batch done $(date +%H:%M:%S)"; fi
  done
done
wait

echo "=== E32 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e32_*__ep*.npy(N) | wc -l | tr -d ' ')  (expect 30)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e32_*.out(N) || echo "  none"
