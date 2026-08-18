#!/usr/bin/env bash
# E42 -- train against a field that hunts back. TREATMENT ARM ONLY.
# The control arm needs no compute: checkpoints/benedict_task4/q_table_e37_PLB2_s10N__ep20000.npy
# are already 8 seeds of the current recipe (3x rule_based), same configuration, which became
# the default. Only --agents differs between the arms.
set -u
cd "$(dirname "$0")/../.."
FIELD="ext_xiaoxiae_binary_v6 ext_aielka_ql_atom rule_based_agent"
CONC=5                       # 5 lanes so each holds a performance core (E38's setting)

for i in $(seq 200 207); do
  (
    BM_QUIET_LOGS=1 BM_MODEL_SUFFIX=_e42mix_s$i BM_RUN_INDEX=$i \
      uv run python main.py play --agents benedict_task4 $FIELD \
      --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731 \
      > scratchpad/benedict/e42/train_s$i.log 2>&1
    echo "trained s$i"
  ) &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done
done
wait
echo "E42: all 8 treatment runs done"
