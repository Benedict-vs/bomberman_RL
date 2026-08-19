#!/usr/bin/env bash
# E42 evaluation. Both arms, 8 seeds each, three fields, 300 rounds, VALIDATION seed 550731.
# The ship seed 990731 is never used for selection -- only to confirm a table that already won
# on 550731 (pre-registered ship rule, experiments/benedict.md E42).
#
# Table selection is via BM_MODEL_SUFFIX (callbacks.py:56-61), which points MODEL_FILE at
# checkpoints/benedict_task4/q_table<suffix>.npy. Both arms' tables already live there, so no
# file is copied and agent_code/ is not touched.
#
# There is no feature-map switch to export: E42 changes only --agents at TRAINING time, so both
# arms read the default map at HEAD. (Contrast benedict_task4.md section 5.6, where an unexported
# digit-8 switch silently turned suicides 0.450 into 0.751 while score barely moved.)
set -u
cd "$(dirname "$0")/../.."
HELD_OUT="ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2"
INDIST="ext_xiaoxiae_binary_v6 ext_aielka_ql_atom rule_based_agent"
GUARD="rule_based_agent rule_based_agent rule_based_agent"

jobs_file=$(mktemp)
for i in 200 201 202 203 204 205 206 207; do
  for fld in heldout indist guard; do echo "mix|$i|_e42mix_s$i|$fld"; done
done > "$jobs_file"
for i in 100 101 102 103 104 105 106 107; do
  for fld in heldout indist guard; do echo "ctl|$i|_e37_PLB2_s${i}__ep20000|$fld"; done
done >> "$jobs_file"

run_one () {
  IFS='|' read -r arm seed suffix field <<< "$1"
  case $field in
    heldout) AG="ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2" ;;
    indist)  AG="ext_xiaoxiae_binary_v6 ext_aielka_ql_atom rule_based_agent" ;;
    guard)   AG="rule_based_agent rule_based_agent rule_based_agent" ;;
  esac
  BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="$suffix" \
  uv run python tools/evaluate.py --agents benedict_task4 $AG \
      --n-rounds 300 --seed 550731 \
      --label "e42_${arm}_s${seed}__task4_${field}_val550731" \
      --out-dir results/eval/task4_tournament \
      > "scratchpad/benedict/e42/eval_${arm}_${seed}_${field}.log" 2>&1
  echo "evaluated $arm s$seed $field"
}
export -f run_one
xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
echo "E42: all 48 evaluations done"
