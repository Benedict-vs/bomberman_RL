#!/bin/zsh

set -eu

script_directory="${0:A:h}"
cd "${script_directory}/.."

for training_seed in 11 12 13; do
    label="ben_dqn_task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}__task3_coin_collector_eval1000"
    csv="results/eval/task3_opponents/${label}.csv"
    meta="results/eval/task3_opponents/${label}.meta.json"
    log_dir="logs/${label}"

    for target in "$csv" "$meta" "$log_dir"; do
        if [[ -e "$target" ]]; then
            print -u2 "ABBRUCH: Ziel existiert bereits: $target"
            exit 1
        fi
    done
done

for training_seed in 11 12 13; do
    label="ben_dqn_task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}__task3_coin_collector_eval1000"
    log_dir="logs/${label}"
    mkdir "$log_dir"
    print "Evaluiere Trainingsseed ${training_seed} über 1.000 Runden ..."

    BM_QUIET_LOGS=1 \
    BM_TASK3_TRAINING_ARM=mixed_safety_multiseed_v1 \
    BM_TASK3_TRAINING_SEED="$training_seed" \
    BM_TASK3_TOTAL_EPISODES=5000 \
    BM_TASK3_MODEL_VARIANT=trained \
    uv run python tools/evaluate.py \
        --agents ben_task3 \
        --opponents coin_collector \
        --scenario classic \
        --n-rounds 1000 \
        --label "$label" \
        --out-dir results/eval/task3_opponents \
        --log-dir "$log_dir"
done

print "Alle drei 1.000-Runden-Evaluationen sind abgeschlossen."
