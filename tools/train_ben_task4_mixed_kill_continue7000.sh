#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

arm="mixed_kill_continue7000_v1"
model="agent_code/ben_task4/ben_task4_${arm}_7000ep_seed11.pt"
stem="results/train/ben_task4/ben_task4__task4_${arm}_7000ep_seed11"
checkpoint_glob="results/train/ben_task4/task4_${arm}_7000ep_seed11__episode_*.pt"
source_model="agent_code/ben_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt"

if [[ ! -f "$source_model" ]]; then
    print -u2 "Missing source model: $source_model"
    exit 1
fi
if [[ -e "$model" || -e "$stem.csv" || -e "$stem.meta.json" ]] \
    || (( ${#${~checkpoint_glob}(N)} > 0 )); then
    print -u2 "Refusing to overwrite existing artifacts for $arm"
    exit 1
fi

BM_QUIET_LOGS=1 \
BM_TASK4_TRAINING_ARM="$arm" \
BM_TASK4_TOTAL_EPISODES=7000 \
BM_TASK4_TRAINING_SEED=11 \
caffeinate -i uv run python main.py play \
    --agents ben_task4 peaceful_agent rule_based_agent rule_based_agent \
    --train 1 \
    --scenario classic \
    --seed 11 \
    --n-rounds 5000 \
    --no-gui
