# Arminator: tabular Q-learning

Submitted agent for the `classic` scenario against three opponents.

During development the agent was called `benedict_task4`, and the results ledger
(`experiments/benedict_task4.md`) and the checkpoint directory still use that
name. The tournament name is the folder name, `Arminator`.

## Dependencies

Only numpy. `callbacks.py` imports `os`, `numpy` and the framework's `settings`.
It does not import anything from `tools/`, uses no absolute paths and works
without `train.py`. Nothing beyond the provided `Dockerfile` is required, and the
death filter described below needs no extra import or data file.

## Files

| file | role |
|---|---|
| `callbacks.py` | `setup` / `act`: the feature map and the greedy policy. Always loaded. |
| `train.py` | `setup_training` / `game_events_occurred` / `end_of_round`. Loaded only with `--train`. |
| `q_table.npy` | the trained table, 64 000 rows × 6 actions. This is the submitted model, byte-identical to `checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy` |
| `q_table.npy.layout.json` | the `FEATURE_SIZES` the table was trained on, checked on load |

## Model

Tabular Q-learning on a hand-built state encoding. The state is a single
mixed-radix row index over eight digits: four neighbour statuses, the grace left
on the agent's own tile, a BFS direction to the current objective, whether a bomb
here would pay off, and a digit that holds either the target distance or the
local wall structure. Each digit is documented in `callbacks.py`.

The table is stored densely, but most of the state space is never reached: the
greedy policy visits a few hundred rows per round. On load the layout sidecar is
checked, so a table trained on a different feature map raises an error instead
of being read at the wrong indices.

## Death filter

The policy is the greedy Q-table plus one filter applied at inference time
(E51). Before the argmax, every movement action after which no continuation
survives the bombs already on the board is masked out. The check uses the same
time-aware BFS as the escape feature, started from the tile the action leads to.

The filter has three properties:

- **BOMB is never vetoed.** We tested gating bomb placement on escape slack, and
  it cost −0.283 score (E46). The tightest bombs are also the most productive,
  because a bomb in a dense pocket has a small blast and a tight escape for the
  same geometric reason. The filter therefore leaves bomb placement alone and
  corrects the escape.
- **Only bombs and fire that are visible now are modelled.** The filter does not
  assume that an opponent might bomb. Enemy bombs placed after the agent
  committed to a path cause 2.3 % of its deaths (E43), so a pessimistic version
  would gain at most 2.3 % and make the agent more conservative on every step.
- **If every filterable action is fatal, the filter is skipped** and the
  unfiltered row decides. The agent is therefore never worse off than without it.

`BM_DEATH_FILTER=0` restores the pre-E51 policy (verified identical over 60
deterministic solo rounds); `step` selects the shallow variant. The tournament
sets no environment variables, so it runs with the filter.

The filter changes the decision on 0.09 % of steps; on the other 99.91 % the
table already picks a non-vetoed action. The gain comes from fewer deaths to
opponents' bombs (−0.022), while the change in suicides (−0.008) is not
demonstrated. Because the agent lives longer, it places more bombs and collects
more coins and kills.

### Timing

Without the filter the mean is 0.110 ms per decision, step-weighted over 125.6
million decisions in 472 600 evaluated agent-rounds. The worst single decision
took 54.4 ms, and no step exceeded the 0.5 s limit. The distribution is narrow:
the 99th percentile of the worst step within a round is 2.67 ms, so the 54.4 ms
outlier is one decision in 125 million. Most of the cost is two BFS traversals
of the 17×17 board.

With the filter, which is the submitted configuration, the mean is **0.128 ms**,
step-weighted over 1 969 116 decisions in the 8 000 evaluated agent-rounds of
E51's two shipped arms. The worst single decision took 2.41 ms, and no step
exceeded the limit. The filter adds about 0.02 ms to the mean and only runs on
the 33.6 % of steps where a bomb or fire is on the board.

These numbers are from the development machine. In the provided Docker image the
worst step was 9.33 ms. The tournament reference machine (one thread of a Ryzen 5
2600) is slower still, so the maximum is the relevant figure, and it stays below
0.5 s by more than an order of magnitude. An earlier figure of 0.137 ms / 53.4 ms
over 180 000 rounds came from the E37 sweep alone and could not be reproduced;
the numbers above are step-weighted over all committed evaluations.

## Performance

4000 rounds against 3 × `rule_based_agent`, held-out seed 990731 (E51):

| | score | `won` | kills | suicides | killed by opp. |
|---|---|---|---|---|---|
| **this agent, as shipped** | **4.050** | **0.409** | 0.237 | 0.471 | 0.052 |
| the same table, filter off | 3.928 | 0.390 | 0.228 | 0.475 | 0.065 |
| `rule_based_agent` in the same slot | 3.254 | 0.286 | 0.196 | 0.533 | n/a |

On the same frozen table without retraining, the filter is worth +0.121
[+0.007, +0.236] here and +0.099 [+0.067, +0.132] against 3 ×
`xiaoxiae/binary_agent_v6` (2.877 vs 2.778).

Evaluations against `rule_based` have a noise floor of about ±0.12 on `score`,
because those opponents use the unseeded stdlib RNG. Re-evaluating the same
table gave 3.828. This noise comes from that opponent and not from the
evaluation harness: against the external agents, which do not use the stdlib RNG,
a re-evaluation is bit-exact (0 of 1000 rounds differ, compared with 999 of 1000
for `rule_based`).

Against other students' agents the agent is weaker. We measured four SS2024
agents from public repositories in the same slot and field (E41). All four beat
this agent: 5.572 / 5.336 / 5.143 / 4.690 against its 3.949 at the time. The
filter closes about a tenth of a point of that gap. In head-to-head games the
paired within-round margin is negative against three of them. Kills account for
65–70 % of the deficit, and the cause is survival under pressure rather than
where the bombs are placed. `experiments/benedict_task4.md` §6 has the details.

## Training

The defaults in `train.py` are the submitted configuration:

```bash
uv run python main.py play --agents Arminator \
    rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731
```

No environment variables are needed, since every default in `train.py` is the
shipped value, including the exploration seed (`BM_RUN_INDEX=106`). The run
warm-starts from `checkpoints/benedict_task4/q_table_parent.npy` and writes
`q_table_trained.npy` next to the agent. Training never writes `q_table.npy`, so
an accidental `--train` cannot overwrite the submitted model. To use a retrained
table, copy it over `q_table.npy` by hand.

Set `BM_WARM=""` to train from zero. With 20 000 rung-4 episodes this rebuilds
only ~44 % of the warm table's learned rows (E44), so from-scratch runs at this
budget are much weaker.

`BM_MODEL_SUFFIX=<name>` moves both the output and the checkpoints into
`checkpoints/<this folder's name>/`. The sweep scripts use this to run many seeds
in parallel. Warm-start parents are always read from `checkpoints/benedict_task4/`,
because `train.py` ties them to the lineage name and not to the folder name.

The command reproduces the configuration, but not the exact bytes of the table.
`main.py` does not seed the provided opponents, so two runs of the same command
give different tables. The shipped table is one seed of a fifteen-seed sweep,
selected on a validation world seed and confirmed on a held-out one. The full
protocol and the experiments behind each hyperparameter are in
`experiments/benedict_task4.md`.
