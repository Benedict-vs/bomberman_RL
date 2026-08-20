#!/usr/bin/env bash
# E44 evaluation: 16 from-scratch tables x 3 fields, 300 rounds, VALIDATION seed 550731.
# bash (not zsh) for word-splitting; BM_MODEL_SUFFIX selects the table (callbacks.py:56-61),
# verified live: a bogus suffix raises FileNotFoundError rather than falling back to q_table.npy.
# BM_WARM is irrelevant here -- nothing trains.
set -u
cd "$(dirname "$0")/../.."
jobs_file=$(mktemp)
for i in $(seq 300 307); do for f in heldout indist guard; do echo "rb|$i|$f"; done; done >  "$jobs_file"
for i in $(seq 310 317); do for f in heldout indist guard; do echo "mix|$i|$f"; done; done >> "$jobs_file"

run_one () {
  IFS='|' read -r arm seed field <<< "$1"
  case $field in
    heldout) AG="ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2 ext_xiaoxiae_bindist_v2" ;;
    indist)  AG="ext_xiaoxiae_binary_v6 ext_aielka_ql_atom rule_based_agent" ;;
    guard)   AG="rule_based_agent rule_based_agent rule_based_agent" ;;
  esac
  BM_QUIET_LOGS=1 BM_MODEL_SUFFIX="_e44scr_${arm}_s${seed}" \
  uv run python tools/evaluate.py --agents benedict_task4 $AG \
      --n-rounds 300 --seed 550731 \
      --label "e44_scr${arm}_s${seed}__task4_${field}_val550731" \
      --out-dir results/eval/task4_tournament \
      > "scratchpad/benedict/e44/eval_${arm}_${seed}_${field}.log" 2>&1
  echo "evaluated $arm s$seed $field"
}
export -f run_one
xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
echo "E44: all 48 evaluations done"
