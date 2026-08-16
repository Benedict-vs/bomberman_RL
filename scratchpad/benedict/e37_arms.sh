#!/bin/zsh
# E37: is E36's placebo effect real, and is its mechanism the wall lattice?
#
#   ctl2  BM_D8 unset      digit 8 stays DIST_NONE -- the matched control E36 lacked
#   PLB2  BM_D8=stripe     (x+y) % 4 -- E36's PLB verbatim, replication
#   PAR   BM_D8=parity     (x+y) % 2 -- the lattice class alone
#   SHF   BM_D8=shuffle    fixed relabelling balanced within each lattice class:
#                          position WITHOUT the lattice, the true null
#
# ALL FOUR share the _e36parent warm start. That is the defect this sweep exists
# to fix: E36's arms used _e36parent (7879 valued rows) while its control used
# _rung2ship (2364), so "E33 ctl config verbatim" was false and PLB - ctl was
# confounded by the initial condition.
#
# n = 15, chosen from the measured paired SD of score differences (0.1825):
# 80%-power MDE is 0.352 at n=5, 0.210 at n=10, 0.164 at n=15, against an
# expected effect of ~0.20. n=10 would be ~50% powered -- the exact mistake
# E33-E36 made and that this entry corrects.
#
# Requires callbacks.py at E37 (BM_D8) and EXPERIMENT = "e37" in train.py.
#
# Run from the repo root:   ./scratchpad/benedict/e37_arms.sh
#   fallback (drop PAR, 45 runs):  ARMS="ctl2 PLB2 SHF" ./scratchpad/benedict/e37_arms.sh

set -u
cd "$(dirname "$0")/../.." || exit 1

EPISODES=20000
BATCH=10
WORLD_SEED=810731
SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114)
ARMS=${ARMS:-"ctl2 PLB2 PAR SHF"}
CKPT=checkpoints/benedict_task4
mkdir -p "$CKPT" agent_code/benedict_task4/logs results/train/task4_tournament

PARENT="$CKPT/q_table_e36parent.npy"
if [[ ! -f "$PARENT" ]]; then
    uv run python scratchpad/benedict/broadcast_oppdist.py \
        agent_code/benedict_task4/q_table.npy "$PARENT" || exit 1
fi

echo "=== E37 started at $(date) ==="
echo "arms: $ARMS   seeds: ${#SEEDS}   -> $(( $(echo $ARMS | wc -w) * ${#SEEDS} )) runs"

i=0
for arm in ${=ARMS}; do
  case $arm in
    ctl2) mode=""        ;;
    PLB2) mode="stripe"  ;;
    PAR)  mode="parity"  ;;
    SHF)  mode="shuffle" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  for seed in $SEEDS; do
    i=$((i + 1))

    BM_QUIET_LOGS=1 \
    BM_MODEL_SUFFIX="_e37_${arm}_s${seed}" \
    BM_ARM="$arm" \
    BM_RUN_INDEX="$seed" \
    BM_CRATE=1.0 \
    BM_STEP_COST=0 \
    BM_D8="$mode" \
    BM_WARM=_e36parent \
    uv run python main.py play \
        --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --train 1 --n-rounds "$EPISODES" --no-gui \
        --seed "$WORLD_SEED" \
        > "results/train/task4_tournament/e37_${arm}_s${seed}.out" 2>&1 &

    echo "launched ${arm} s${seed}  (BM_D8='${mode}')"
    if (( i % BATCH == 0 )); then wait; echo "  ... $i done $(date +%H:%M:%S)"; fi
  done
done
wait

echo "=== E37 finished at $(date) ==="
echo "checkpoints: $(ls "$CKPT"/q_table_e37_*__ep*.npy(N) | wc -l | tr -d ' ')"
echo "tracebacks:"
grep -l Traceback results/train/task4_tournament/e37_*.out(N) || echo "  none"
