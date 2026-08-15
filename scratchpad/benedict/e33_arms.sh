#!/bin/zsh
# E33: how much of audit 5's ceiling is tie-breaking?
#
# Pay the agent for taking the escape step digit 6 already found. The shaping
# differential is 2r, so r selects which decisions get overridden:
#
#   F005  r=0.05  flips gaps < 0.10  -- near-ties only (row 55060's is 0.055)
#   F020  r=0.20  flips gaps < 0.40
#   F080  r=0.80  flips gaps < 1.60  -- the mean danger-row gap, i.e. the rule
#
# ctl is CONTEMPORANEOUS. E31's numbers are not a usable control: audit 5 showed
# its suicide rate is bimodal on one row's argmax (0.690 there, 0.500/0.533 on
# two replications of the identical configuration).
#
# --seed is passed for the first time. Audit 5 found no training run in E30-E32
# was ever arena-seeded, contrary to what those entries claim.
#
# Requires train.py at E33 (BM_ESCAPE, tag_escape) and EXPERIMENT = "e33".
#
# Run from the repo root:   ./scratchpad/benedict/e33_arms.sh
#
# Concurrency: fixed batches with a plain `wait`. `jobs -rp` returns nothing
# inside $( ) because the subshell has no job table -- that is why E30's
# launcher started all ten runs at once instead of five.

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
BATCH=10
WORLD_SEED=810731
SEEDS=(100 101 102 103 104)
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

if [[ ! -f "$CKPT/q_table_rung2ship.npy" ]]; then
    cp agent_code/benedict_task4/q_table.npy "$CKPT/q_table_rung2ship.npy"
fi

echo "=== E33 started at $(date) ==="

i=0
for arm in ctl F005 F020 F080; do
  case $arm in
    ctl)  r=0    ;;
    F005) r=0.05 ;;
    F020) r=0.20 ;;
    F080) r=0.80 ;;
  esac
  for seed in $SEEDS; do
    i=$((i + 1))

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e33_${arm}_s${seed}" \
    BM_ARM="$arm" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_ESCAPE="$r" \
    BM_WARM=_rung2ship \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        --seed "$WORLD_SEED" \
        > "results/train/task4_tournament/e33_${arm}_s${seed}.out" 2>&1 &

    echo "launched ${arm} s${seed} (BM_ESCAPE=${r})"
    if (( i % BATCH == 0 )); then wait; echo "  ... $i/20 done $(date +%H:%M:%S)"; fi
  done
done
wait

echo "=== E33 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e33_*__ep*.npy(N) | wc -l | tr -d ' ')  (expect 60)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e33_*.out(N) || echo "  none"
