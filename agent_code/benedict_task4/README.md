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
| `q_table.npy` | the trained table, 64 000 rows × 6 actions. **This is the submitted model** — byte-identical to `checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy` |
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

**Timing.** Mean **0.110 ms per decision**, step-weighted over **125.6 million
decisions** in 472 600 evaluated agent-rounds; worst single decision **54.4 ms**;
**0** steps over the 0.5 s limit. The distribution is tight — the 99th-percentile
*worst step within a round* is 2.67 ms, so the 54.4 ms outlier is one decision in
125 million. Cost is dominated by two BFS traversals over a 17×17 board.

Measured on the development machine. In the provided Docker image the worst step
was 9.33 ms; the tournament reference (one thread of a Ryzen 5 2600) is slower
again, so budget the *maximum*, not the mean — it still clears 0.5 s by an order
of magnitude. (An earlier figure of 0.137 ms / 53.4 ms across 180 000 rounds was
the E37 sweep alone and was not reproducible; the numbers above are
step-weighted across every committed evaluation.)

## Performance

1000 rounds against 3 × `rule_based_agent`, held-out seed 990731:

| | score | `won` | kills | suicides |
|---|---|---|---|---|
| **this agent** | **3.949** | **0.406** | 0.226 | 0.488 |
| `rule_based_agent` in the same slot | 3.254 | 0.286 | 0.196 | 0.533 |

A 1000-round evaluation carries a **±0.12 noise floor on `score`** from the
opponents' unseeded stdlib RNG: the same table re-evaluated gives 3.828.

**Against other students' agents it is weaker, and that is measured, not
assumed.** Four SS2024 agents from public repositories, same slot, same field
(E41): all four beat this agent's 3.949 — 5.572 / 5.336 / 5.143 / 4.690 — and
head-to-head the paired within-round margin is negative against three of them.
The deficit is 65–70 % kills, and the mechanism is survival under pressure
rather than bomb siting. `experiments/benedict_task4.md` §6 has the full account.

## Training

Defaults in `train.py` are the shipped configuration:

```bash
uv run python main.py play --agents benedict_task4 \
    rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731
```

**No environment variables are needed** — every default in `train.py` is the
shipped value, including the exploration seed (`BM_RUN_INDEX=106`). The run
writes `q_table_trained.npy` beside the agent and warm-starts from
`checkpoints/benedict_task4/q_table_parent.npy`; `q_table.npy` is never written
by training, so a stray `--train` cannot destroy the submitted model. Promote a
retrained table by copying it over `q_table.npy` deliberately.

Set `BM_WARM=""` to train from zero — but note that 20 000 rung-4 episodes
rebuild only ~44 % of the warm table's learned rows (E44), so from-scratch runs
at this budget are substantially weaker.

`BM_MODEL_SUFFIX=<name>` redirects both the output and the checkpoints into
`checkpoints/benedict_task4/`, which is what the sweep scripts use to run many
seeds concurrently.

**This reproduces the configuration, not the bytes.** `main.py` does not seed the
provided opponents, so two runs of the same command give different tables. The
shipped table is one seed of a fifteen-seed sweep, selected on a validation world
seed and confirmed on a held-out one. Full protocol and the experiments behind
every hyperparameter: `experiments/benedict_task4.md`.
