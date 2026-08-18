#!/usr/bin/env bash
# E40 -- hunt ceiling remeasured with a simultaneous-move trap test.
# Four arms x 8000 paired arenas, held-out ship seed 990731, run concurrently.
# No training: one fixed table, so the only variance is evaluation noise.
set -u
cd "$(dirname "$0")/../.."
OUT=scratchpad/benedict/e40
mkdir -p "$OUT"

run () {  # run <k> <trap-model> <tag>
  BM_QUIET_LOGS=1 uv run python scratchpad/benedict/hunt_ceiling_v2.py \
      --k "$1" --trap-model "$2" --n-rounds 8000 --seed 990731 \
      --label "e40_$3__task4_rb_ship990731" --out-dir "$OUT" \
      > "$OUT/$3.log" 2>&1 &
}

run -1 sim   ctl        # override disabled: trap-model is irrelevant on this arm
run  4 stale k4stale    # reproduces E39's headline arm from this binary
run  4 sim   k4sim      # the corrected oracle -- primary
run  8 sim   k8sim      # does a stricter trap test want a longer reach?

wait
echo "all four arms done"
