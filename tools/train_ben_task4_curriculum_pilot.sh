#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

run_phase() {
    local arm="$1"
    local source_model="$2"
    local metadata_lineup="$3"
    shift 3
    local -a lineup=("$@")
    local model="agent_code/ben_task4/ben_task4_${arm}_250ep_seed11.pt"
    local stem="results/train/ben_task4/ben_task4__task4_${arm}_250ep_seed11"
    local checkpoint_glob="results/train/ben_task4/task4_${arm}_250ep_seed11__episode_*.pt"

    if [[ ! -f "agent_code/ben_task4/${source_model}" ]]; then
        print -u2 "Missing curriculum source: ${source_model}"
        return 1
    fi
    if [[ -e "$model" || -e "$stem.csv" || -e "$stem.meta.json" ]] \
        || (( ${#${~checkpoint_glob}(N)} > 0 )); then
        print -u2 "Refusing to overwrite existing artifacts for $arm"
        return 1
    fi

    BM_QUIET_LOGS=1 \
    BM_TASK4_TRAINING_ARM="$arm" \
    BM_TASK4_TOTAL_EPISODES=250 \
    BM_TASK4_TRAINING_SEED=11 \
    BM_TASK4_LOAD_MODEL_FILE="$source_model" \
    BM_TASK4_TRAINING_OPPONENTS="$metadata_lineup" \
    caffeinate -i uv run python main.py play \
        --agents ben_task4 "${lineup[@]}" \
        --train 1 \
        --scenario classic \
        --seed 11 \
        --n-rounds 250 \
        --no-gui
}

MIXED="peaceful_agent,rule_based_agent,rule_based_agent"
EXTERNAL="ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,ext_xiaoxiae_binary_v6"

run_control() {
    local source="ben_task4_mixed_kill_v1_2000ep_seed11.pt"
    for phase in 1 2 3 4; do
        local arm="curriculum_control_p${phase}_v1"
        run_phase "$arm" "$source" "$MIXED" \
            peaceful_agent rule_based_agent rule_based_agent
        source="ben_task4_${arm}_250ep_seed11.pt"
    done
}

run_candidate() {
    local source="ben_task4_mixed_kill_v1_2000ep_seed11.pt"
    for phase in 1 2 3 4; do
        local arm="curriculum_candidate_p${phase}_v1"
        if (( phase % 2 == 0 )); then
            run_phase "$arm" "$source" "$EXTERNAL" \
                ext_lijesse_featureeverything ext_xiaoxiae_bindist_v2 \
                ext_xiaoxiae_binary_v6
        else
            run_phase "$arm" "$source" "$MIXED" \
                peaceful_agent rule_based_agent rule_based_agent
        fi
        source="ben_task4_${arm}_250ep_seed11.pt"
    done
}

case "${1:-}" in
    control) run_control ;;
    candidate) run_candidate ;;
    *)
        print -u2 "Usage: $0 {control|candidate}"
        exit 2
        ;;
esac
