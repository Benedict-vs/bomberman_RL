# benedict_task4 — tabular Q-learning

Submitted agent for the `classic` scenario against three opponents.

## Dependencies

**numpy only.** `callbacks.py` imports `os`, `numpy` and the framework's own
`settings`; it never imports `tools/`, uses no absolute paths, and does not need
`train.py` to be present. Nothing beyond the provided `Dockerfile` is required.

## Files

| file | role |
|---|---|
| `callbacks.py` | `setup` / `act` — the feature map and the greedy policy. Loaded always. |
| `train.py` | `setup_training` / `game_events_occurred` / `end_of_round`. Loaded only with `--train`. |
| `q_table.npy` | the trained table, 64 000 rows × 6 actions |
| `q_table.npy.layout.json` | the `FEATURE_SIZES` the table was trained on, checked on load |

## The model

Tabular Q-learning over a hand-built state encoding. The state is one mixed-radix
row index over eight digits — four neighbour statuses, the grace left on the
agent's own tile, a BFS direction to the current objective, whether a bomb here
would pay off, and a distance-or-structure digit. `callbacks.py` documents each.

The table is dense storage over a state space that is mostly unreachable by
construction: the greedy policy visits a few hundred rows per round. Loading it
checks the layout sidecar, so a table trained on a different feature map fails
loudly instead of being read at the wrong indices.

**Timing.** Mean decision time 0.137 ms, worst single step observed 53.4 ms
across 180 000 evaluated rounds, against the 0.5 s per-step limit — 0 breaches.
The cost is dominated by two BFS traversals over a 17×17 board.

## Performance

1000 rounds against 3 × `rule_based_agent`, held-out seed 990731:

| | score | `won` | kills | suicides |
|---|---|---|---|---|
| **this agent** | **3.949** | **0.406** | 0.226 | 0.488 |
| `rule_based_agent` in the same slot | 3.254 | 0.286 | 0.196 | 0.533 |

## Training

Defaults in `train.py` are the shipped configuration:

```bash
BM_QUIET_LOGS=1 BM_MODEL_SUFFIX=_run BM_RUN_INDEX=106 \
uv run python main.py play --agents benedict_task4 \
    rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731
```

`BM_MODEL_SUFFIX` writes to `checkpoints/` instead of overwriting `q_table.npy`,
and warm-starts from `checkpoints/benedict_task4/q_table_parent.npy`. Set
`BM_WARM=""` to train from zero.

**This reproduces the configuration, not the bytes.** `main.py` does not seed the
provided opponents, so two runs of the same command give different tables. The
shipped table is one seed of a fifteen-seed sweep, selected on a validation world
seed and confirmed on a held-out one. Full protocol and the experiments behind
every hyperparameter: `experiments/benedict_task4.md`.
