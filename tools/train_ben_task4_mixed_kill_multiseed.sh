#!/bin/zsh
set -euo pipefail
repo_dir="${0:A:h:h}"
cd "$repo_dir"
seed="${1:-}"
case "$seed" in 12|13) ;; *) print -u2 "Usage: $0 {12|13}"; exit 2;; esac
training_arm="mixed_kill_seed${seed}_v1"
model="agent_code/ben_task4/ben_task4_${training_arm}_2000ep_seed${seed}.pt"
stem="results/train/ben_task4/ben_task4__task4_${training_arm}_2000ep_seed${seed}"
if [[ -e "$model" || -e "$stem.csv" || -e "$stem.meta.json" ]]; then
  print -u2 "Refusing to overwrite existing artifacts for ${training_arm}"
  exit 1
fi
BM_QUIET_LOGS=1 BM_TASK4_TRAINING_ARM="$training_arm" BM_TASK4_TOTAL_EPISODES=2000 BM_TASK4_TRAINING_SEED="$seed" caffeinate -i uv run python main.py play --agents ben_task4 peaceful_agent rule_based_agent rule_based_agent --train 1 --scenario classic --seed "$seed" --n-rounds 2000 --no-gui
for field in 3rb external; do
  if [[ "$field" == "3rb" ]]; then
    eval_agents=(ben_task4)
    opponents=(--opponents rule_based)
  else
    eval_agents=(ben_task4 ext_lijesse_featureeverything ext_xiaoxiae_bindist_v2 ext_xiaoxiae_binary_v6)
    opponents=(--opponents none)
  fi
  BM_QUIET_LOGS=1 BM_TASK4_TRAINING_ARM="$training_arm" BM_TASK4_TOTAL_EPISODES=2000 BM_TASK4_TRAINING_SEED="$seed" BM_TASK4_MODEL_VARIANT=trained UV_CACHE_DIR=/tmp/ben_uv_cache uv run python tools/evaluate.py --agents "${eval_agents[@]}" "${opponents[@]}" --n-rounds 100 --scenario classic --seed 20260731 --label "ben_dqn_${training_arm}__task4_${field}_quick100" --out-dir results/eval/task4_multiseed
done
