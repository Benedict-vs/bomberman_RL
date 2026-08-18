#!/usr/bin/env bash
# E41 -- the shipped table against four third-party agents. 1000 rounds, ship seed 990731.
# Three line-ups per external (see experiments/benedict.md E41):
#   C  external + 3x rule_based   -- the calibration constant, their agent in OUR slot
#   A  ours + 3x external         -- head-to-head, the primary
#   B  4x external                -- the symmetric bar, prices the field itself
# Four lanes only: the 0.5 s think-time guard is pre-registered, and CPU contention
# inflates think_ms, so a pass under load is a pass a fortiori -- but 8 lanes on 4
# performance cores would make the number meaningless rather than conservative.
set -u
cd "$(dirname "$0")/../.."
OUT=results/eval/task4_tournament
OURS=benedict_task4
EXT=(ext_xiaoxiae_binary_v6 ext_xiaoxiae_bindist_v2 ext_aielka_ql_atom ext_lijesse_featureeverything)

jobs_file=$(mktemp)
# fast agents first so the slow one (lijesse, ~5.3 s/round) does not hold a lane early
for e in "${EXT[@]}"; do
  echo "C|$e|ext_${e#ext_}_calib__task4_rb_ship990731|$e rule_based_agent rule_based_agent rule_based_agent"
  echo "A|$e|${OURS}_shipped_e37__task4_${e}_ship990731|$OURS $e $e $e"
  echo "B|$e|ref_${e}__task4_field_${e}_ship990731|$e $e $e $e"
done > "$jobs_file"

run_one () {
  IFS='|' read -r lineup ext label agents <<< "$1"
  BM_QUIET_LOGS=1 uv run python tools/evaluate.py \
      --agents $agents --n-rounds 1000 --seed 990731 \
      --label "$label" --out-dir results/eval/task4_tournament \
      > "scratchpad/benedict/e41/${lineup}_${ext}.log" 2>&1
  echo "done $lineup $ext"
}
export -f run_one
xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
echo "E41: all 12 runs done"
