#!/usr/bin/env bash
# E27: 3 arms x 5 seeds x {20000 (primary), 40000} in the cc field = 30
#      plus the 20000 checkpoint in the rule_based field (P2)      = 15
# TIE_TOL stays at the shipped 0.0 -- pre-registered, because tuning it per arm
# would flatter precisely the arm that fails (E26 correction).
set -u
MAXJOBS=${MAXJOBS:-5}
OUT=results/eval/task3_opponents
mkdir -p "$OUT"
cmds=$(mktemp)
for arm in C03 C10 S03; do
  for i in 50 51 52 53 54; do
    for ep in 20000 40000; do
      echo "BM_HUNT=1 BM_MODEL_SUFFIX=_e27_${arm}_s${i}__ep${ep} uv run python tools/evaluate.py \
--agents benedict_task3 --opponents coin_collector --n-rounds 300 --seed 550731 \
--label benedict_q_e27_${arm}_s${i}__ep${ep}__task3_cc_val550731 --out-dir $OUT \
--log-dir logs/ev27_${arm}_${i}_cc${ep} --quiet" >> "$cmds"
    done
    echo "BM_HUNT=1 BM_MODEL_SUFFIX=_e27_${arm}_s${i}__ep20000 uv run python tools/evaluate.py \
--agents benedict_task3 --opponents rule_based --n-rounds 300 --seed 550731 \
--label benedict_q_e27_${arm}_s${i}__ep20000__task3_rb_val550731 --out-dir $OUT \
--log-dir logs/ev27_${arm}_${i}_rb --quiet" >> "$cmds"
  done
done
echo "$(wc -l < "$cmds") evaluations, ${MAXJOBS} at a time"
while IFS= read -r cmd; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 2; done
  bash -c "$cmd" &
done < "$cmds"
wait; rm -f "$cmds"
echo "=== done: $(ls $OUT/benedict_q_e27_*.csv 2>/dev/null | wc -l | tr -d ' ') CSVs (expect 45) ==="
