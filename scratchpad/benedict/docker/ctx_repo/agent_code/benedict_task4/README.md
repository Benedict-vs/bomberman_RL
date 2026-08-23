# benedict_task4 — tabular Q-learning

Submitted agent for the `classic` scenario against three opponents.

## Dependencies

**numpy only.** `callbacks.py` imports `os`, `numpy` and the framework's own
`settings`; it never imports `tools/`, uses no absolute paths, and does not need
`train.py` to be present. Nothing beyond the provided `Dockerfile` is required.
The death filter below adds no import and no data file.

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

## The certain-death move filter

The policy is the greedy Q-table **plus one inference-time filter** (E51): before
the argmax, any *movement* action after which no continuation survives the bombs
already on the board is masked out. It runs the same time-aware BFS the escape
feature uses, from the tile the action lands on.

Three properties, each of them a decision:

- **BOMB is never vetoed.** Gating bomb placement on escape slack was measured
  and cost **−0.283 score** (E46): the tightest bombs are also the most
  productive, because a bomb in a dense pocket has a contained blast and a tight
  escape for the same geometric reason. This filter fixes the escape instead.
- **Only bombs and fire visible now are modelled** — no assumption that an
  opponent may bomb. Enemy bombs placed after the agent committed account for
  2.3 % of its deaths (E43), so the pessimistic version would buy 2.3 % and pay
  for it in conservatism on every step.
- **When every filterable action is fatal the filter stands aside** and the
  unfiltered row decides, so the agent is never worse off than without it.

`BM_DEATH_FILTER=0` restores the pre-E51 policy exactly (verified identical over
60 deterministic solo rounds); `step` is the shallow variant. **The tournament
sets no environment variables and therefore gets the filter.**

It changes the decision on **0.09 % of steps** — the table already picks a
non-vetoed action 99.91 % of the time — and pays through deaths to *opponents'*
bombs (−0.022), not through suicides (−0.008, not demonstrated). The agent lives
longer, so it bombs more and banks more coins and kills.

**Timing.** Mean **0.110 ms per decision**, step-weighted over **125.6 million
decisions** in 472 600 evaluated agent-rounds; worst single decision **54.4 ms**;
**0** steps over the 0.5 s limit. The distribution is tight — the 99th-percentile
*worst step within a round* is 2.67 ms, so the 54.4 ms outlier is one decision in
125 million. Cost is dominated by two BFS traversals over a 17×17 board.

That figure is the table alone. **With the filter — the shipped configuration —
the mean is 0.128 ms**, step-weighted over 1 969 116 decisions in the 8 000
evaluated agent-rounds of E51's two shipped arms, worst single decision
**2.41 ms**, **0** steps over the limit. The filter costs ~0.02 ms of mean and
runs at all only on the 33.6 % of steps where a bomb or fire is on the board.

Measured on the development machine. In the provided Docker image the worst step
was 9.33 ms; the tournament reference (one thread of a Ryzen 5 2600) is slower
again, so budget the *maximum*, not the mean — it still clears 0.5 s by an order
of magnitude. (An earlier figure of 0.137 ms / 53.4 ms across 180 000 rounds was
the E37 sweep alone and was not reproducible; the numbers above are
step-weighted across every committed evaluation.)

## Performance

**4000** rounds against 3 × `rule_based_agent`, held-out seed 990731 (E51):

| | score | `won` | kills | suicides | killed by opp. |
|---|---|---|---|---|---|
| **this agent, as shipped** | **4.050** | **0.409** | 0.237 | 0.471 | 0.052 |
| the same table, filter off | 3.928 | 0.390 | 0.228 | 0.475 | 0.065 |
| `rule_based_agent` in the same slot | 3.254 | 0.286 | 0.196 | 0.533 | — |

The filter is worth **+0.121 [+0.007, +0.236]** here and **+0.099
[+0.067, +0.132]** against 3 × `xiaoxiae/binary_agent_v6` (2.877 vs 2.778), on
the same frozen table with no retraining.

A `rule_based` evaluation carries a **±0.12 noise floor on `score`** from those
opponents' unseeded stdlib RNG — the same table re-evaluated gives 3.828. **That
noise floor is a property of that opponent, not of the harness**: against the
external agents, which do not use the stdlib RNG, re-evaluation is bit-exact
(0 of 1000 rounds differ, against 999 of 1000 for `rule_based`).

**Against other students' agents it is weaker, and that is measured, not
assumed.** Four SS2024 agents from public repositories, same slot, same field
(E41): all four beat this agent — 5.572 / 5.336 / 5.143 / 4.690 against its
3.949 at the time, and the filter closes about a tenth of a point of that — and
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
