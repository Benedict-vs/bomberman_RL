# Measurement — rationale, history and the evidence behind the rules

The top-level `README.md` states the measurement *rules* in short form. This file holds
the reasoning and the measured evidence behind them: why each rule exists, what it cost to learn,
and which experiment established it. Read it once when you join the measurement chain, or when
you are tempted to break one of the rules.

---

## Why `tools/evaluate.py` and not `main.py --save-stats`

`main.py --save-stats` only writes lifetime totals per agent and per-round totals summed over
*all* agents. There is no per-agent, per-round record, so no per-agent confidence interval is
possible and no paired comparison can be built. `tools/evaluate.py` drives `BombeRLeWorld`
directly and touches no framework file, so it also survives the tournament reset.

## A training curve is not a result

Measured 2026-08-05 (`experiments/maxi.md` E01): **48.2 coins/episode** at the end of training,
**1.45** in the evaluation of the same model. While training, ε-exploration and the
still-changing table keep breaking the policy out of cycles, so a policy that is broken at ε = 0
looks healthy in the training log. Nothing is claimed until it has been measured with
`tools/evaluate.py` at ε = 0.

## Why completion rate must be read next to `survived`

`steps` only measures path length in rounds that were *completed*. The round ends the moment the
last coin is collected, so 400 means "never finished", not "slow", and a mean over both kinds of
round is meaningless.

From task 2 on it gets worse, because a dead agent also ends the round. Measured on `classic`
(`experiments/benedict.md` E08): `random_agent` "completes" **100 %** of rounds at **19.0** steps
because it kills itself, while `rule_based_agent` completes **8 %** at **399.1**. Taken alone,
completion rate ranks the worst agent in the field first.

`tools/plot_task1_versions.py` restricts `steps` to completed rounds, which is the other valid
way to read it.

## Seeding differs between training and measurement, on purpose

`BombeRLeWorld.__init__` (`environment.py:335`) seeds the world RNG **once**, and `build_arena`
keeps drawing from it. So `main.py --seed S` gives a deterministic *sequence* of distinct arenas:
variety for training, reproducible as a whole.

`tools/evaluate.py` instead reseeds with `base_seed + round_index` **before every round**, so
round *i* is the same arena for every agent ever measured at that seed. That is what makes
`--compare` paired and the CIs tight. `main.py --seed` alone is *not* sufficient for a
measurement — the world draws from the same RNG every step, so arenas drift apart from round 2
onwards.

## Exploration RNG: reproducible since 2026-08-08

*Amended 2026-08-07, revised 2026-08-08.* This used to say a lost training log "cannot be
reproduced, only replaced by a different one", because `--seed` fixes the arenas but not the
agent's exploration RNG. That stopped being true when the exploration RNG was seeded in
`setup_training` (`np.random.default_rng(TRAIN_SEED + RUN_INDEX)`); `experiments/benedict.md` E12
confirmed it by rerunning a 40 000-round sweep and getting round-for-round identical evaluation
results.

## …but that guarantee ends when opponents train alongside us

*Amended 2026-08-11 (E24).* `main.py` does not seed the provided agents, and each of them reseeds
the global RNG from OS entropy in `setup` — which runs *after* ours, so nothing in
`setup_training` can precede it. A rung-3 training run is therefore reproducible in its arenas and
in our own exploration, but **not** in the opponents, and two runs of the same command give
different tables.

This is **accepted rather than fixed**. Opponent variety during training is desirable for the same
reason arena variety is, and the only hook that runs before an opponent's first action is our own
`act()`, where seeding the global RNG would reach into other agents' behaviour — not something the
submitted agent should ever do. Reproducibility is preserved where the numbers come from:
`evaluate.py` seeds the opponents per round (below). The practical consequence is the rule in
`README.md`: rung-3 arms must be compared **across several training seeds**, never on a single run.

## From task 3 on, the opponents are seeded in evaluation

Added 2026-08-11 (`experiments/benedict.md` E24). All three provided agents call
`np.random.seed()` **with no argument** in `setup` (`peaceful_agent:5`,
`coin_collector_agent:68`, `rule_based_agent:69`), reseeding the global legacy RNG from OS
entropy. With opponents on the board an evaluation was therefore *not* reproducible at a fixed
seed — only its arenas were. Rerunning one diagnostic gave **12 vs 7** deaths against
`peaceful_agent` on identical seeds.

`evaluate.py` now calls `np.random.seed(base_seed + round_index)` beside the `world.rng` line.
Nothing in the framework draws from that RNG (`environment.py:335` is the only generator) and our
agents use seeded `default_rng` objects only, so this changes opponent behaviour and nothing else:
arenas are untouched and evaluations without opponents are bit-identical to before.

**Numbers recorded with opponents before this date are single draws, not constants.**

### That fix is INCOMPLETE, and this section claimed more than it delivered

Corrected 2026-08-13, found by the E27 audit (`scratchpad/audit2/`). `coin_collector_agent:1`
and `rule_based_agent:2` also do **`from random import shuffle`** — the *stdlib* RNG, a
different generator that `np.random.seed()` does not touch — and shuffle their candidate moves
with it on every step (`coin_collector_agent:44,116`, `rule_based_agent:45,140`). So seeding
`np.random` alone fixes only part of their behaviour.

Measured: re-running arm F on the committed configuration reproduces the **mean** (score 4.1600
vs 4.1600, crates 26.63 vs 26.66) but only **22.7 % of rounds are identical**. Every claim that
a rung-3 evaluation is reproducible round-for-round is therefore **false** — that includes this
section as first written, the sentence in the project rules at the time, and E24's write-up. Rung-3 "paired"
CIs are paired on **arenas only**, so they are wider than a fully-paired CI would be. Directions
and magnitudes stand — E26's +1.595 is far too large to be affected — but the pairing claim does
not.

The one-line completion is `random.seed(base_seed + round_index)` beside the existing
`np.random.seed(...)`. **Not applied yet**: it changes the opponents' realised behaviour, so it
breaks comparability with every committed rung-3 CSV and needs its own entry with the baselines
re-measured under it. `peaceful_agent` draws only from `np.random` and is unaffected either way.

## What is committed, and why

**Measurements are versioned** (settled 2026-08-04, was the open item in `KONZEPT.md` §6.7).
`.gitignore` excludes only `results/figures/`; `results/eval/` and `results/train/` are committed —
they are the evidence for the report's most important chapter. Figures stay ignored: they are pure
functions of the CSVs.

**Exception — `results/train/task2_crates/` is no longer committed** (`.gitignore`, from
2026-08-08). It had reached 282 MB: an evaluation CSV is ~35 KB, but a 100 000-episode training
log is ~7 MB and a 20-run sweep is 140 MB, which every collaborator would clone. They are
reproducible from the commit plus the `BM_*` arm variables recorded in each run's `.meta.json`.
Logs written before the change stay in the history — removing them from the index stops the
growth, it does not shrink an existing clone.

`results/train/task1_coin_collectors/` is untouched: it is Maxi's and Ben's and it is 21 MB.
**`results/eval/` stays committed in full** — it is small and it is what every number in the
report is computed from.

## Sweep hygiene (the numbers behind the README warnings)

- **`BM_QUIET_LOGS=1`** drops all three log levels to WARNING. The engine logs per step, so a
  40 000-round `classic` run is ~10 M INFO lines; measured saving **25 % of wall clock**, and
  `logs/game.log` goes from GB-scale to empty. It is an environment switch, not an edited
  constant, so unset — every normal game, every evaluation, the tournament — `settings.py` holds
  exactly the upstream values and there is nothing to restore before submitting.
- **`agents.py:226-228`** hardcodes the agent log path in mode `"w"`, so parallel runs of the same
  agent overwrite each other's log. During a sweep the training CSV is the only usable record.
- **`np.save` frequency**: once per round on a 614 KB table is ~25 GB of writes per run, and five
  concurrent runs make that the dominant I/O.

## What `.meta.json` does and does not catch

`evaluate.py` snapshots the *rules* in `settings.py` — board size, bomb, timeout, rewards,
scenario config — plus the git commit and seed. It does **not** snapshot the whole file. An edited
rule is therefore visible in the metadata; anything else is not. Log levels, for instance, are not
snapshotted, so an edit there is invisible. This is the reason training-only changes should be
environment switches with the upstream value as default.

**`bm_env` — every `BM_*` variable in the launching environment — is recorded by both writers from
2026-08-16.** Until then the sentence above about reproducing a run "from the commit plus the
`BM_*` arm variables recorded in each run's `.meta.json`" was simply false: nothing recorded them.
Audit 8 found it, and E37 found what it costs. **Runs before 2026-08-16 have no `bm_env` key**, so
for those the arm is recoverable only from the `run` name and the committed `*_arms.sh` launcher.
The E37 sweep straddles the fix — `ctl2` seeds 100-109 launched at 06:22 and hold the old module in
memory, so they are the last records without it.

**Why it is not cosmetic, stated once so nobody has to rediscover it.** A `BM_*` switch that lives
in `train.py` (`BM_ESCAPE`, `BM_D4`, `BM_KILL`) changes rewards or updates, so it matters at
training time and is irrelevant afterwards. A switch that lives in `callbacks.py` (`BM_OPPDIST`,
`BM_D8`, `BM_ABLATE`) changes what a *digit means*, hence which row of the table a state indexes —
so **it must be exported at evaluation time too, with exactly the value the table was trained
with.** A table trained under `BM_D8=parity` and evaluated without it is read at the wrong indices
and the arm silently measures noise. E36 is the only rung-4 entry where this applied, and it was
done correctly; the point is that it was only checkable from a script in `/tmp`.

**How to check it after the fact, and which metric to check.** Re-evaluate the table both ways at
n = 1000 and compare against the committed CSV — at 1000 rounds the means reproduce to four
decimals, which n = 100 does not (27/100 vs 19/100 exact round matches, both inside the documented
22.7 % reproduction rate, i.e. undecidable). Settled for E36 arm `OPP` s100 on 2026-08-16:

| | score | crates | suicides | survived |
|---|---|---|---|---|
| committed | 3.644 | 32.15 | **0.450** | **0.502** |
| re-run `BM_OPPDIST=1` | 3.627 | 32.12 | **0.423** | **0.501** |
| re-run `BM_OPPDIST=0` | 3.593 | 32.20 | **0.751** | **0.193** |

**Read the behavioural metrics, not the headline.** `score` separates the two readings by 0.017 vs
0.051 — useless. `suicides` and `survived` separate them by 3-4×: through the wrong map the same
table is a different agent, one that dies to its own bombs and survives a fifth as often. This is
the general shape of the failure — nothing raises, the array shapes match, and only behaviour
gives it away.

Hence: **sweep-watchers and eval launchers belong in `scratchpad/benedict/` beside the
`*_arms.sh` files and get committed with the results, never in `/tmp`.** A launcher that is
deleted on reboot is not evidence, and "which environment did that number come from" is a question
that gets asked months later, in the report.
