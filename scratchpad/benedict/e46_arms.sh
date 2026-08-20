#!/usr/bin/env bash
# E46 -- the zero-slack bomb gate. 4 arms x 2 fields x 4000 rounds, ship seed 990731.
set -u
cd "$(dirname "$0")/../.."
jobs_file=$(mktemp)
for fld in ext_xiaoxiae_binary_v6 rule_based_agent; do
  for v in 0 5 4 3; do echo "$v|$fld"; done
done > "$jobs_file"
run_one () {
  IFS='|' read -r v fld <<< "$1"
  uv run python scratchpad/benedict/e46_bombgate.py --veto-at "$v" --field "$fld" \
      --n-rounds 4000 --seed 990731 --label "e46_v${v}__${fld}_ship990731" \
      --out-dir scratchpad/benedict/e46 \
      > "scratchpad/benedict/e46/run_v${v}_${fld}.log" 2>&1
  echo "done v$v $fld"
}
export -f run_one
xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
echo "E46: all 8 runs done"
