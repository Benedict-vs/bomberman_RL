#!/usr/bin/env bash
# E25 evaluation batch: 10 tables x 2 checkpoints in the training field, plus the
# 40 000 checkpoint in the transfer field. 30 runs of 300 rounds on the held-out
# validation seed 550731.
#
# Concurrency is deliberately modest. `think_ms` is measured per step and every
# concurrent evaluation inflates it, so this batch's timing columns are NOT a
# valid latency measurement -- the 0.5 s tournament check has to be re-run
# serially before anything ships. Survival, score and kills are unaffected.
#
# Each run gets its own --log-dir: agents.py:226-228 opens the agent log in "w"
# mode from a hardcoded path, so runs of the same agent otherwise overwrite each
# other's logs.
set -u

MAXJOBS=${MAXJOBS:-5}
OUT=results/eval/task3_opponents
mkdir -p "$OUT"

cmds=$(mktemp)
for arm in A B; do
  for i in 10 11 12 13 14; do
    for ep in 20000 40000; do
      echo "BM_MODEL_SUFFIX=_e25_${arm}_s${i}__ep${ep} uv run python tools/evaluate.py \
        --agents benedict_task3 --opponents coin_collector --n-rounds 300 --seed 550731 \
        --label benedict_q_e25_${arm}_s${i}__ep${ep}__task3_cc_val550731 \
        --out-dir $OUT --log-dir logs/eval_e25_${arm}_${i}_cc${ep} --quiet" >> "$cmds"
    done
    echo "BM_MODEL_SUFFIX=_e25_${arm}_s${i}__ep40000 uv run python tools/evaluate.py \
      --agents benedict_task3 --opponents rule_based --n-rounds 300 --seed 550731 \
      --label benedict_q_e25_${arm}_s${i}__ep40000__task3_rb_val550731 \
      --out-dir $OUT --log-dir logs/eval_e25_${arm}_${i}_rb --quiet" >> "$cmds"
  done
done

echo "$(wc -l < "$cmds") evaluations, ${MAXJOBS} at a time"

# Not xargs: BSD xargs caps a -I constructed argument at 255 bytes and these
# commands are ~320, which fails as "command line cannot be assembled, too long"
# *before* running anything.
# Each job's exit status is not collected: `cmd &` runs in a subshell, so a
# status variable set there would not survive. The CSV count below is the check.
while IFS= read -r cmd; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 2; done
  bash -c "$cmd" &
done < "$cmds"
wait
rm -f "$cmds"
status=0

echo "=== done, exit $status ==="
echo "CSVs written: $(ls "$OUT"/benedict_q_e25_*.csv 2>/dev/null | wc -l | tr -d ' ') (expect 30)"
exit $status
