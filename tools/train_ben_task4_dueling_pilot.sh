#!/bin/zsh
set -euo pipefail
repo_dir="${0:A:h:h}"
cd "$repo_dir"
case "${1:-}" in
  control) training_arm="dueling_control_v1" ;;
  candidate)
    training_arm="dueling_v1"
    if [[ ! -f agent_code/ben_task4/ben_task4_dueling_mixed_kill_source.pt ]]; then
      UV_CACHE_DIR=/tmp/ben_uv_cache uv run python tools/convert_ben_task4_dueling_source.py
    fi
    ;;
  *) print -u2 "Usage: $0 {control|candidate}"; exit 2 ;;
esac
model="agent_code/ben_task4/ben_task4_${training_arm}_2000ep_seed11.pt"
stem="results/train/ben_task4/ben_task4__task4_${training_arm}_2000ep_seed11"
if [[ -e "$model" || -e "$stem.csv" || -e "$stem.meta.json" ]]; then
  print -u2 "Refusing to overwrite existing artifacts for ${training_arm}"
  exit 1
fi
BM_QUIET_LOGS=1 BM_TASK4_TRAINING_ARM="$training_arm" \
BM_TASK4_TOTAL_EPISODES=2000 BM_TASK4_TRAINING_SEED=11 \
caffeinate -i uv run python main.py play \
  --agents ben_task4 peaceful_agent rule_based_agent rule_based_agent \
  --train 1 --scenario classic --seed 11 --n-rounds 2000 --no-gui
BM_QUIET_LOGS=1 BM_TASK4_TRAINING_ARM="$training_arm" \
BM_TASK4_TOTAL_EPISODES=2000 BM_TASK4_TRAINING_SEED=11 \
BM_TASK4_MODEL_VARIANT=trained UV_CACHE_DIR=/tmp/ben_uv_cache \
uv run python tools/evaluate.py \
  --agents ben_task4 ext_lijesse_featureeverything ext_xiaoxiae_bindist_v2 ext_xiaoxiae_binary_v6 \
  --opponents none --n-rounds 100 --scenario classic --seed 20260731 \
  --label "ben_dqn_${training_arm}__task4_external_top3_quick100" \
  --out-dir results/eval/task4_dueling
