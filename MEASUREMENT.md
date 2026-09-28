# Measurement: rationale and evidence

The top-level `README.md` lists the measurement rules in short form. This file gives the
reasoning and the measured evidence behind them: why each rule exists and which experiment
established it. It is meant to be read once when starting to work with the measurement tools,
and again before breaking one of the rules.

---

## Why `tools/evaluate.py` instead of `main.py --save-stats`

`main.py --save-stats` only writes lifetime totals per agent and per-round totals summed over
all agents. There is no per-agent, per-round record, so neither a per-agent confidence interval
nor a paired comparison can be computed. `tools/evaluate.py` drives `BombeRLeWorld` directly and
does not modify any framework file, so it keeps working after the tournament reset.

## Training curves are not results

Measured 2026-08-05 (`experiments/maxi.md` E01): 48.2 coins/episode at the end of training,
1.45 in the evaluation of the same model. During training, ε-exploration and the table, which
is still changing, keep breaking the policy out of cycles. A policy that is broken at ε = 0 can
therefore look healthy in the training log. We claim nothing until it has been measured with
`tools/evaluate.py` at ε = 0.

## Completion rate and `survived`

`steps` only measures path length in rounds that were completed. The round ends when the last
coin is collected, so 400 means that the agent never finished, and a mean over finished and
unfinished rounds has no clear meaning.

From task 2 on this gets worse, because a dead agent also ends the round. Measured on `classic`
(`experiments/benedict.md` E08): `random_agent` "completes" 100 % of rounds at 19.0 steps
because it kills itself, while `rule_based_agent` completes 8 % at 399.1. On its own, completion
rate ranks the worst agent in the field first.

`tools/plot_task1_versions.py` restricts `steps` to completed rounds, which is the other valid
way to read it.

## Seeding in training and in measurement

`BombeRLeWorld.__init__` (`environment.py:335`) seeds the world RNG once, and `build_arena` keeps
drawing from it. `main.py --seed S` therefore gives a deterministic sequence of distinct arenas.
This is what we want for training: variety, but reproducible as a whole.

`tools/evaluate.py` instead reseeds with `base_seed + round_index` before every round, so round
*i* is the same arena for every agent ever measured at that seed. This makes `--compare` paired
and the CIs tight. `main.py --seed` alone is not enough for a measurement: the world draws from
the same RNG every step, so the arenas drift apart from round 2 on.

## Exploration RNG: reproducible since 2026-08-08

Amended 2026-08-07, revised 2026-08-08. This section used to say that a lost training log
"cannot be reproduced, only replaced by a different one", because `--seed` fixes the arenas but
not the agent's exploration RNG. That changed when the exploration RNG was seeded in
`setup_training` (`np.random.default_rng(TRAIN_SEED + RUN_INDEX)`). `experiments/benedict.md` E12
confirmed it by rerunning a 40 000-round sweep, which gave round-for-round identical evaluation
results.

## Training with opponents is not reproducible

Amended 2026-08-11 (E24). `main.py` does not seed the provided agents, and each of them reseeds
the global RNG from OS entropy in `setup`. Their `setup` runs after ours, so nothing in
`setup_training` can come before it. A rung-3 training run is therefore reproducible in its
arenas and in our own exploration, but not in the opponents, and two runs of the same command
give different tables.

We accept this instead of fixing it. Opponent variety during training is useful for the same
reason arena variety is. The only hook that runs before an opponent's first action is our own
`act()`, and seeding the global RNG there would interfere with other agents' behaviour, which
the submitted agent should never do. The numbers themselves stay reproducible because
`evaluate.py` seeds the opponents per round (next section). In practice this means that rung-3
arms must be compared across several training seeds and never on a single run.

## Seeding the opponents in evaluation (task 3 on)

Added 2026-08-11 (`experiments/benedict.md` E24). All three provided agents call
`np.random.seed()` without an argument in `setup` (`peaceful_agent:5`,
`coin_collector_agent:68`, `rule_based_agent:69`), which reseeds the global legacy RNG from OS
entropy. With opponents on the board, an evaluation at a fixed seed was therefore not
reproducible; only its arenas were. Rerunning one diagnostic gave 12 vs 7 deaths against
`peaceful_agent` on identical seeds.

`evaluate.py` now calls `np.random.seed(base_seed + round_index)` next to the `world.rng` line.
Nothing in the framework draws from that RNG (`environment.py:335` is the only generator) and
our agents only use seeded `default_rng` objects. The change affects opponent behaviour and
nothing else: arenas are unchanged, and evaluations without opponents are bit-identical to
before.

**Numbers recorded with opponents before this date are single draws, not constants.**

### The fix is incomplete

Corrected 2026-08-13, found by the E27 audit (`scratchpad/audit2/`, in the git history).
`coin_collector_agent:1` and `rule_based_agent:2` also do `from random import shuffle`. This is
the stdlib RNG, a separate generator that `np.random.seed()` does not touch, and both agents use
it to shuffle their candidate moves on every step (`coin_collector_agent:44,116`,
`rule_based_agent:45,140`). Seeding `np.random` alone therefore fixes only part of their
behaviour.

Measured: rerunning arm F on the committed configuration reproduces the mean (score 4.1600 vs
4.1600, crates 26.63 vs 26.66), but only 22.7 % of rounds are identical. Every claim that a
rung-3 evaluation is reproducible round for round is therefore false. This includes the first
version of this section, our earlier project notes and the write-up of E24. Rung-3 "paired" CIs
are paired on arenas only, so they are wider than fully paired CIs would be. Directions and
magnitudes still hold (E26's +1.595 is far too large to be affected), but the pairing claim
does not.

The missing line is `random.seed(base_seed + round_index)` next to the existing
`np.random.seed(...)`. It has not been applied yet. It changes the opponents' actual behaviour,
so it breaks comparability with every committed rung-3 CSV and would need its own entry with
the baselines re-measured. `peaceful_agent` only draws from `np.random` and is not affected
either way.

## What is committed

Measurements are versioned (decided 2026-08-04; this was the open item in `KONZEPT.md` §6.7).
`.gitignore` excludes only `results/figures/`. `results/eval/` and `results/train/` are
committed, since they are the evidence for the Experiments chapter of the report. Figures are
not committed because they are generated from the CSVs.

Exception: `results/train/task2_crates/` is no longer committed (`.gitignore`, from 2026-08-08).
It had grown to 282 MB. An evaluation CSV is ~35 KB, but a 100 000-episode training log is
~7 MB and a 20-run sweep is 140 MB, which every collaborator would have to clone. The logs can
be reproduced from the commit plus the `BM_*` arm variables recorded in each run's `.meta.json`.
Logs written before the change remain in the history. Removing them from the index stops the
growth but does not shrink an existing clone.

`results/train/task1_coin_collectors/` is unchanged: it belongs to Maxi and Ben and is 21 MB.
`results/eval/` stays committed in full. It is small, and every number in the report is computed
from it.

## Sweep hygiene

- `BM_QUIET_LOGS=1` lowers all three log levels to WARNING. The engine logs every step, so a
  40 000-round `classic` run produces ~10 M INFO lines. Setting the switch saved 25 % of wall
  clock time in our measurement, and `logs/game.log` went from several GB to empty. Since it is
  an environment switch and not an edited constant, `settings.py` keeps the upstream values
  whenever it is unset (normal games, evaluations, the tournament), and nothing has to be
  restored before submitting.
- `agents.py:226-228` hardcodes the agent log path in mode `"w"`, so parallel runs of the same
  agent overwrite each other's log. During a sweep the training CSV is the only usable record.
- `np.save` frequency: saving a 614 KB table once per round adds up to ~25 GB of writes per run,
  and with five concurrent runs this becomes the dominant I/O.

## What `.meta.json` records

`evaluate.py` stores a snapshot of the rules in `settings.py` (board size, bomb, timeout,
rewards, scenario config) together with the git commit and the seed. It does not store the whole
file. An edited rule is therefore visible in the metadata, but other edits are not. Log levels,
for example, are not recorded. For this reason, training-only changes should be environment
switches whose default is the upstream value.

Since 2026-08-16 both writers also record `bm_env`, i.e. every `BM_*` variable in the launching
environment. Before that, the statement above that a run can be reproduced "from the commit plus
the `BM_*` arm variables recorded in each run's `.meta.json`" was false, because nothing
recorded them. Audit 8 found this, and E37 showed the cost. Runs before 2026-08-16 have no
`bm_env` key; for them the arm can only be recovered from the `run` name and the committed
`*_arms.sh` launcher. The E37 sweep spans the fix: `ctl2` seeds 100-109 were launched at 06:22
and still ran the old module, so they are the last records without the key.

The variables matter for a concrete reason. A `BM_*` switch in `train.py` (`BM_ESCAPE`, `BM_D4`,
`BM_KILL`) changes rewards or updates. It matters at training time and is irrelevant afterwards.
A switch in `callbacks.py` (`BM_OPPDIST`, `BM_D8`, `BM_ABLATE`) changes what a digit of the state
encoding means, and therefore which row of the table a state indexes. **Such a switch must also
be set at evaluation time, with the same value the table was trained with.** A table trained
under `BM_D8=parity` and evaluated without it is read at the wrong indices, and the arm measures
noise without any error being raised. E36 is the only rung-4 entry where this applied, and it
was done correctly, but this could only be checked from a script in `/tmp`.

To check it afterwards, re-evaluate the table both ways at n = 1000 and compare against the
committed CSV. At 1000 rounds the means reproduce to four decimals; at n = 100 they do not
(27/100 vs 19/100 exact round matches, both inside the documented 22.7 % reproduction rate, so
the test cannot decide). Result for E36 arm `OPP` s100 on 2026-08-16:

| | score | crates | suicides | survived |
|---|---|---|---|---|
| committed | 3.644 | 32.15 | **0.450** | **0.502** |
| re-run `BM_OPPDIST=1` | 3.627 | 32.12 | **0.423** | **0.501** |
| re-run `BM_OPPDIST=0` | 3.593 | 32.20 | **0.751** | **0.193** |

The behavioural metrics decide this, not the headline metric. `score` separates the two
readings by 0.017 vs 0.051, which is too small to tell them apart. `suicides` and `survived`
separate them by a factor of 3-4×: read through the wrong feature map, the same table behaves
like a different agent that dies to its own bombs and survives a fifth as often. This is how
the failure typically shows up. Nothing raises an error, the array shapes match, and only the
behaviour reveals it.

Sweep watchers and evaluation launchers therefore belong in the repository next to the
`*_arms.sh` files (during development `scratchpad/benedict/`, now in the git history) and are
committed with the results, never kept in `/tmp`. A launcher that is deleted on reboot cannot
serve as evidence, and the question of which environment a number came from tends to come up
months later, when writing the report.
