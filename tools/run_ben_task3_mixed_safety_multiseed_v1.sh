#!/bin/zsh

set -eu

script_directory="${0:A:h}"
cd "${script_directory}/.."

for training_seed in 11 12 13; do
    model="agent_code/ben_task3/ben_task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}.pt"
    csv="results/train/ben_task3/ben_task3__task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}.csv"
    meta="results/train/ben_task3/ben_task3__task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}.meta.json"
    log_dir="logs/task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}_train"

    for target in "$model" "$csv" "$meta" "$log_dir"; do
        if [[ -e "$target" ]]; then
            print -u2 "ABBRUCH: Ziel existiert bereits: $target"
            exit 1
        fi
    done

    for episode in {100..5000..100}; do
        checkpoint="results/train/ben_task3/task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}__episode_${episode}.pt"
        if [[ -e "$checkpoint" ]]; then
            print -u2 "ABBRUCH: Checkpoint existiert bereits: $checkpoint"
            exit 1
        fi
    done
done

for training_seed in 11 12 13; do
    log_dir="logs/task3_mixed_safety_multiseed_v1_5000ep_seed${training_seed}_train"
    mkdir "$log_dir"
    print "Starte Trainingsseed ${training_seed} ..."

    BM_QUIET_LOGS=1 \
    BM_TASK3_TRAINING_ARM=mixed_safety_multiseed_v1 \
    BM_TASK3_TRAINING_SEED="$training_seed" \
    BM_TASK3_TOTAL_EPISODES=5000 \
    caffeinate -i uv run python main.py play \
        --agents ben_task3 coin_collector_agent coin_collector_agent coin_collector_agent \
        --train 1 \
        --scenario classic \
        --n-rounds 5000 \
        --seed "$training_seed" \
        --no-gui \
        --log-dir "$log_dir"
done

print "Alle drei Trainingsseeds sind abgeschlossen."
