#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

MIXED="peaceful_agent,rule_based_agent,rule_based_agent"
EXTERNAL="ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,ext_xiaoxiae_binary_v6"
stop_on_plateau="${BM_PLATEAU_STOP:-0}"
eval_files=()

run_phase() {
    local arm="$1"
    local source_model="$2"
    local metadata_lineup="$3"
    shift 3
    local -a lineup=("$@")
    local model="agent_code/ben_task4/ben_task4_${arm}_250ep_seed11.pt"
    local stem="results/train/ben_task4/ben_task4__task4_${arm}_250ep_seed11"
    local checkpoint_glob="results/train/ben_task4/task4_${arm}_250ep_seed11__episode_*.pt"
    local eval_label="ben_dqn_${arm}__task4_external_top3_quick100"
    local eval_csv="results/eval/task4_kill_reward_plateau/${eval_label}.csv"

    if [[ ! -f "agent_code/ben_task4/${source_model}" ]]; then
        print -u2 "Missing source model: ${source_model}"
        return 1
    fi
    if [[ -e "$model" || -e "$stem.csv" || -e "$stem.meta.json" ]] \
        || (( ${#${~checkpoint_glob}(N)} > 0 )); then
        print -u2 "Refusing to overwrite existing artifacts for $arm"
        return 1
    fi

    mkdir -p results/eval/task4_kill_reward_plateau
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

    BM_QUIET_LOGS=1 \
    BM_TASK4_TRAINING_ARM="$arm" \
    BM_TASK4_TOTAL_EPISODES=250 \
    BM_TASK4_TRAINING_SEED=11 \
    BM_TASK4_MODEL_VARIANT=trained \
    UV_CACHE_DIR=/tmp/ben_uv_cache \
    uv run python tools/evaluate.py \
        --agents ben_task4 \
            ext_lijesse_featureeverything \
            ext_xiaoxiae_bindist_v2 \
            ext_xiaoxiae_binary_v6 \
        --opponents none \
        --n-rounds 100 \
        --scenario classic \
        --seed 20260731 \
        --label "$eval_label" \
        --out-dir results/eval/task4_kill_reward_plateau
    eval_files+=("$eval_csv")
}

run_schedule() {
    local prefix="$1"
    local source="ben_task4_mixed_kill_v1_2000ep_seed11.pt"
    for phase in 1 2 3 4; do
        local arm="${prefix}_p${phase}_v1"
        if (( phase % 2 == 0 )); then
            run_phase "$arm" "$source" "$EXTERNAL" \
                ext_lijesse_featureeverything ext_xiaoxiae_bindist_v2 \
                ext_xiaoxiae_binary_v6
        else
            run_phase "$arm" "$source" "$MIXED" \
                peaceful_agent rule_based_agent rule_based_agent
        fi
        source="ben_task4_${arm}_250ep_seed11.pt"

        if (( ${#eval_files[@]} >= 3 )); then
            set +e
            UV_CACHE_DIR=/tmp/ben_uv_cache uv run python tools/plateau_check.py \
                --agent ben_task4 --patience 2 "${eval_files[@]}"
            plateau_status=$?
            set -e
            if (( plateau_status == 0 )); then
                print "Plateau detected after phase ${phase}."
                if [[ "$stop_on_plateau" == "1" ]]; then
                    print "BM_PLATEAU_STOP=1: stopping schedule."
                    break
                fi
            fi
        fi
    done
}

case "${1:-}" in
    control) run_schedule kill_reward_control ;;
    candidate) run_schedule kill_reward75 ;;
    *)
        print -u2 "Usage: $0 {control|candidate}"
        exit 2
        ;;
esac
