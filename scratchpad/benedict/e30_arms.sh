#!/bin/zsh
# E30: train in the rung-4 field. Two arms x five seeds.
#
#   T   train vs 3x rule_based_agent, BM_CRATE=1.0, warm start from the frozen table
#   TD  same + every TD update shared across the state's D4 orbit, warm start
#       from the folded table so the initialisation is already self-consistent
#
# Run from the repo root:   ./scratchpad/benedict/e30_arms.sh
#
# Notes that cost time on previous sweeps and are handled here:
#   - agents.py:226-228 hardcodes the agent log path in mode "w", so parallel
#     runs of the same agent overwrite each other's log. The training CSV is the
#     only usable record; the log dir is pre-created to dodge the TOCTOU race in
#     makedirs that lost a run in E25.
#   - BSD xargs caps -I-constructed arguments at 255 bytes and these commands are
#     longer, so this uses a plain `jobs -rp` job-control loop instead.
#   - zsh aborts the whole command when any glob matches nothing, hence (N).

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
MAXJOBS=5           # 10 cores; 5 leaves headroom and keeps per-run wall clock honest
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

# Warm-start parents. T starts from the shipped table; TD from its D4 fold, so
# arm TD does not spend its first thousand episodes symmetrising an asymmetric
# initialisation that its own update rule then enforces.
if [[ ! -f "$CKPT/q_table_rung2ship.npy" ]]; then
    cp agent_code/benedict_task4/q_table.npy "$CKPT/q_table_rung2ship.npy"
fi
if [[ ! -f "$CKPT/q_table_d4.npy" ]]; then
    uv run python scratchpad/benedict/d4.py --fold \
        agent_code/benedict_task4/q_table.npy "$CKPT/q_table_d4.npy" || exit 1
fi

echo "=== E30 started at $(date) ==="

for arm in T TD; do
  for seed in 60 61 62 63 64; do
    while (( $(jobs -rp | wc -l) >= MAXJOBS )); do sleep 5; done

    if [[ $arm == TD ]]; then d4=1; warm=_d4; else d4=0; warm=_rung2ship; fi

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e30_${arm}_s${seed}" \
    BM_ARM="$arm" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_D4="$d4" \
    BM_WARM="$warm" \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        > "results/train/task4_tournament/e30_${arm}_s${seed}.out" 2>&1 &

    echo "launched ${arm} s${seed}"
  done
done

wait
echo "=== E30 finished at $(date) ==="

echo "checkpoints: $(ls "$CKPT"/q_table_e30_*__ep*.npy(N) | wc -l)  (expect 30)"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e30_*.out(N) || echo "  none"
