#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

case "${1:-}" in
    control) arm="full_symmetry_control_v1" ;;
    candidate) arm="full_symmetry_v1" ;;
    *)
        print -u2 "Usage: $0 {control|candidate}"
        exit 2
        ;;
esac

episodes=1000
seed=11
model="agent_code/ben_task4/ben_task4_${arm}_${episodes}ep_seed${seed}.pt"
stem="results/train/ben_task4/ben_task4__task4_${arm}_${episodes}ep_seed${seed}"
checkpoint_glob="results/train/ben_task4/task4_${arm}_${episodes}ep_seed${seed}__episode_*.pt"

if [[ -e "$model" || -e "${stem}.csv" || -e "${stem}.meta.json" ]] \
    || (( ${#${~checkpoint_glob}(N)} > 0 )); then
    print -u2 "Refusing to overwrite existing artifacts for $arm"
    exit 1
fi

BM_QUIET_LOGS=1 \
BM_TASK4_TRAINING_ARM="$arm" \
BM_TASK4_TOTAL_EPISODES="$episodes" \
BM_TASK4_TRAINING_SEED="$seed" \
UV_CACHE_DIR=/tmp/ben_uv_cache \
caffeinate -i uv run python main.py play \
    --agents ben_task4 peaceful_agent rule_based_agent rule_based_agent \
    --train 1 \
    --scenario classic \
    --seed "$seed" \
    --n-rounds "$episodes" \
    --no-gui
