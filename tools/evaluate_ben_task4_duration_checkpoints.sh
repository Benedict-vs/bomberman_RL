#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
cd "$repo_dir"

training_arm="${1:-}"
case "$training_arm" in
  learning_rate_control_v1|learning_rate5e5_v1) ;;
  *)
    print -u2 "Usage: $0 {learning_rate_control_v1|learning_rate5e5_v1}"
    exit 2
    ;;
esac

eval_files=()
for checkpoint in 200 500 800 1000; do
  label="ben_dqn_${training_arm}_checkpoint${checkpoint}__task4_external_top3_quick100"
  csv="results/eval/task4_duration/${label}.csv"
  if [[ -e "$csv" || -e "${csv%.csv}.meta.json" ]]; then
    print -u2 "Refusing to overwrite existing evaluation: $csv"
    exit 1
  fi

  BM_QUIET_LOGS=1 \
  BM_TASK4_TRAINING_ARM="$training_arm" \
  BM_TASK4_TOTAL_EPISODES=1000 \
  BM_TASK4_TRAINING_SEED=11 \
  BM_TASK4_MODEL_VARIANT=trained \
  BM_TASK4_CHECKPOINT_EPISODE="$checkpoint" \
  UV_CACHE_DIR=/tmp/ben_uv_cache \
  uv run python tools/evaluate.py \
    --agents ben_task4 ext_lijesse_featureeverything ext_xiaoxiae_bindist_v2 ext_xiaoxiae_binary_v6 \
    --opponents none --n-rounds 100 --scenario classic --seed 20260731 \
    --label "$label" --out-dir results/eval/task4_duration
  eval_files+=("$csv")
done

UV_CACHE_DIR=/tmp/ben_uv_cache uv run python tools/plateau_check.py \
  --agent ben_task4 "${eval_files[@]}"
