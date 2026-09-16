#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

out_dir="results/eval/task4_tournament"

run_arm() {
    local arm="$1"
    local label="ben_dqn_task4_${arm}_1000ep_seed11__task4_rule_based_eval1000"

    if [[ -e "$out_dir/$label.csv" || -e "$out_dir/$label.meta.json" ]]; then
        print -u2 "Refusing to overwrite existing output for $label"
        return 1
    fi

    BM_QUIET_LOGS=1 \
    BM_TASK4_TRAINING_ARM="$arm" \
    BM_TASK4_TOTAL_EPISODES=1000 \
    BM_TASK4_TRAINING_SEED=11 \
    BM_TASK4_MODEL_VARIANT=trained \
    uv run python tools/evaluate.py \
        --agents ben_task4 \
        --opponents rule_based \
        --n-rounds 1000 \
        --label "$label" \
        --out-dir "$out_dir"
}

case "${1:-}" in
    control)
        run_arm temporal_safety_zero_v1
        ;;
    candidate)
        run_arm temporal_safety_v1
        ;;
    *)
        print -u2 "Usage: $0 {control|candidate}"
        exit 2
        ;;
esac
