#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

run_arm() {
    local arm="$1"
    local model="agent_code/ben_task4/ben_task4_${arm}_1000ep_seed11.pt"
    local stem="results/train/ben_task4/ben_task4__task4_${arm}_1000ep_seed11"
    local checkpoint_glob="results/train/ben_task4/task4_${arm}_1000ep_seed11__episode_*.pt"

    if [[ -e "$model" || -e "$stem.csv" || -e "$stem.meta.json" ]] \
        || (( ${#${~checkpoint_glob}(N)} > 0 )); then
        print -u2 "Refusing to overwrite existing artifacts for $arm"
        return 1
    fi

    BM_QUIET_LOGS=1 \
    BM_TASK4_TRAINING_ARM="$arm" \
    BM_TASK4_TOTAL_EPISODES=1000 \
    BM_TASK4_TRAINING_SEED=11 \
    caffeinate -i uv run python main.py play \
        --agents ben_task4 peaceful_agent rule_based_agent rule_based_agent \
        --train 1 \
        --scenario classic \
        --seed 11 \
        --n-rounds 1000 \
        --no-gui
}

case "${1:-}" in
    control)
        run_arm survival_penalty_control_v1
        ;;
    candidate)
        run_arm survival_penalty10_v1
        ;;
    *)
        print -u2 "Usage: $0 {control|candidate}"
        exit 2
        ;;
esac
