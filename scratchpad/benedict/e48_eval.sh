#!/usr/bin/env bash
# E48 evaluation: 3 arms x 8 seeds x 2 fields, 300 rounds, VALIDATION seed 550731.
# Control evals already exist from E42 (e42_ctl_s1NN__task4_{guard,heldout}_val550731).
# Rewards affect training only, so no switch needs exporting here -- BM_MODEL_SUFFIX picks
# the table and callbacks.py reads the default feature map for every arm.
set -u
cd "$(dirname "$0")/../.."
jobs_file=$(mktemp)
for spec in C:400 D:410 CD:420; do
  arm=${spec%%:*}; base=${spec##*:}
  for i in $(seq $base $((base+7))); do
    for f in guard heldout; do echo "$arm|$i|$f"; done
  done
done > "$jobs_file"
run_one () {
  IFS='|' read -r arm seed field <<< "$1"
  case $field in
    guard)   AG="rule_based_agent rule_based_agent rule_based_agent" ;;
    heldout) AG="ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2" ;;
  esac
  BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="_e48${arm}_s${seed}" \
  uv run python tools/evaluate.py --agents benedict_task4 $AG \
      --n-rounds 300 --seed 550731 \
      --label "e48_${arm}_s${seed}__task4_${field}_val550731" \
      --out-dir results/eval/task4_tournament \
      > "scratchpad/benedict/e48/eval_${arm}_${seed}_${field}.log" 2>&1
  echo "evaluated $arm s$seed $field"
}
export -f run_one
xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
echo "E48: all 48 evaluations done"
