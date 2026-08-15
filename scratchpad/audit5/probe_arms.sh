#!/bin/zsh
# Audit-5 probe: does BM_KILLED_SELF move the policy, and by how much?
# 3 arms x 2 seeds x 20000 episodes, everything else identical to e31_arms.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
EPISODES=20000
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament scratchpad/audit5/out
echo "=== audit5 probe started $(date) ==="
for spec in "a5ctl:0" "a5k15:-10" "a5k30:-25"; do
  arm=${spec%%:*}; ks=${spec##*:}
  for seed in 90 91; do
    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_${arm}_s${seed}" \
    BM_ARM="${arm}" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_WARM=_rung2ship \
    BM_KILLED_SELF="$ks" \
    uv run python main.py play \
      --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
      --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
      > "scratchpad/audit5/out/${arm}_s${seed}.out" 2>&1 &
    echo "launched ${arm} s${seed} (KILLED_SELF=${ks})"
  done
done
wait
echo "=== audit5 probe finished $(date) ==="
