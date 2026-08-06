# Task 1 — `coin-heaven`: joint findings, and the agent task 2 starts from

Consolidated account of rung 1, drawn from **two agents developed independently** and then
merged: `benedict_coin_collector` (ledger: `experiments/benedict.md`, E01–E07) and
`maxi_coin_collector` (ledger: `experiments/maxi.md`, E00–E02). Both ledgers carry the
per-experiment predictions written before each run; this file is the synthesis.

The most valuable material here is **§4**, where the two agents hit the same failure from
two different causes. Neither of us could have written that section alone.

---

## 1 · The rung, and what "solved" actually means

`coin-heaven`: `CRATE_DENSITY = 0`, `COIN_COUNT = 50`, one agent, no crates, no opponents,
`MAX_STEPS = 400`. All coins visible from step 1 — pure navigation.

**Reference measurement (Maxi, E00)** — the provided agents, each alone, 300 rounds:

| Agent | coins | steps | invalid |
|---|---|---|---|
| `random_agent` | 1.77 | 22.4 | 9.56 |
| `peaceful_agent` | 18.70 | 400.0 | 148.10 |
| `coin_collector_agent` | **50.00** [50.00, 50.00] | **125.3** | 0.00 |
| `rule_based_agent` | **50.00** [50.00, 50.00] | 124.8 | 0.00 |

This reframes the whole rung, and it is the single most useful thing either ledger contains
about *what to measure*:

1. **`coins` is a pass criterion, not an optimisation target.** Both reference agents collect
   all 50 in *every* round with zero variance. Once our agent reaches 50, `coins` cannot
   distinguish anything — the real target is **`steps`, benchmark ≈ 125**.
2. `coin_collector_agent` and `rule_based_agent` are indistinguishable here (125.3 vs 124.8);
   without crates the bomb logic never runs. For rung 1, `coin_collector_agent` is the only
   meaningful reference.
3. The 125 steps are beatable: the reference walks greedily to the *nearest* coin, which is
   not the optimal tour.
4. `peaceful_agent` is the more informative floor — never dies, 18.7 coins, **148 invalid
   actions**: the profile "moves, but does not navigate".

Two measurement traps, both discovered on this rung and now in `AGENTS.md`:

- The round ends the moment the last coin is collected (`environment.py:289`). So `steps` is
  **completion time**, and 400 means "never finished", not "slow". A mean over completed and
  unfinished rounds is meaningless — **always report the completion rate beside it**
  (Maxi, E02 addendum; `tools/plot_task1_versions.py` restricts `steps` to completed rounds).
- `analyze.py` assumes higher is better and therefore prints `WORSE` for a *falling* `steps`.
  On this rung that verdict is inverted.

## 2 · Two independent development paths

### Benedict — `experiments/benedict.md`

| # | Change | Result | Verdict |
|---|---|---|---|
| E01 | 4 wall bits only, 16 rows | 1.353 coins · 70/300 rounds never moved | Baseline (below `random_agent`) |
| E02 | + `sgn(Δx), sgn(Δy)`, 144 rows | 12.823 · 43 % suicides · invalid 1.85 | BETTER |
| E03 | `KILLED_SELF = −5` | 16.703 · suicides **0.000** | BETTER |
| E04 | Δ clipped to ±3, 784 rows | 46.107 · 242/300 full sweeps | BETTER |
| E05 | ε decay | 49.930 (n = 1) | **Later refuted** |
| E05b | control: constant ε, step-matched | 29.477 · invalid **163** | Exposed run-to-run variance |
| E06 | **α = 1/N(s,a)^0.7**, n = 5 per arm | const α **20.78 ± 6.98** · per-cell **49.15 ± 1.23** | BETTER, decisively |
| E07 | constant ε, per-cell α, n = 5 | 48.68 ± 0.33 | No effect shown |

### Maxi — `experiments/maxi.md`

| # | Change | Result | Verdict |
|---|---|---|---|
| E00 | reference measurement | see §1 | Scale established |
| E01-v1 | BFS coin direction + 4 wall bits, 80 rows, `BOMB` in action set | training: **100 % `KILLED_SELF`**, 18 of 400 steps alive | Broken |
| E01-v1a | *control:* action mask only | suicides → 0.00, but 78 invalid/round at ε = 0.62 | Insufficient alone |
| E01-v1b | *control:* faster ε decay only | invalid 4.2 → 2.6, but 62 % still suicide | Insufficient alone |
| E01-v2 | both | training 48.16 coins — **evaluation 1.45** | **WORSE** |
| E02 | + step cost −0.1 | **50.00 coins · 123.8 steps · invalid 0.06** | BETTER than v2 *and* than the reference |

Maxi's E01 is a model of how to handle two simultaneous changes: rather than confounding
them, he ran both single-change controls alongside, which separated the effects cleanly —
the mask fixes dying, the decay makes the policy actually get exercised, and neither alone
is enough.

His E02 result beats the reference: paired against `coin_collector_agent`,
**`steps` −1.45 [−2.48, −0.41]** with `coins` unchanged at 50.00. By our CI rule that is a
genuine improvement over the provided agent.

## 3 · Where each agent ended up

| | coins | steps | invalid | n |
|---|---|---|---|---|
| `coin_collector_agent` (reference) | 50.00 | 125.3 | 0.00 | — |
| Benedict, E06 | 50.00 (shipped) · 49.15 ± 1.23 (5 seeds) | **130.3** | 0.00 | 5 |
| Maxi, E02-v3 | 50.00 | **123.8** | 0.06 | 1 |
| **Merged agent** (§7) | 50.00 (shipped) · 49.62 ± 0.46 (5 seeds) | **123.7** | 0.20 | 5 |

Read against §1, this says something the coin column hides: **Benedict's agent solved the
pass criterion but was 5 steps *slower than the provided reference*.** Maxi's beat it. The
merged agent beats it too. On the metric that actually discriminates on this rung, the BFS
direction was the deciding component.

## 4 · Convergent findings — the same failure, two different causes

Both agents, independently, reached a state where **the training log looked healthy and the
frozen policy was broken**, and in both cases the mechanism was an absorbing cycle. The
causes were different, and that is exactly what makes the pair informative.

**The shared failure signature.** A deterministic, memoryless policy has an eventually
periodic trajectory. If the cycle touches no coin, the coin set never changes, so the state
never changes, so the cycle is absorbing. During training ε keeps breaking it and the table
keeps moving, so the log never shows it.

- **Maxi (E01-v2):** at ε = 0 the `invalid` distribution is bimodal, not noisy —
  **119 of 300 rounds have exactly 399 invalid actions** (stuck against a wall from step 1),
  **156 rounds have exactly 0** and instead oscillate between two tiles. Training: 48.16
  coins/episode. Evaluation: **1.45**.
- **Benedict (E06 arm A):** training 44.8 coins/episode, evaluation **11.1**. Replaying the
  greedy policy showed loops starting at step 25–69 with 10–30 coins collected.

**Cause A — the reward flattens the table (Maxi).** With `COIN_COLLECTED` +5 and no step
cost, every path eventually collects every coin, so all four moves have nearly the same
return. The *difference* that encodes "go toward the coin" is swamped by the common level.
Measured on his own tables: spread over the four moves 5.78 → 3.05, and the fraction of
states whose argmax equals the BFS direction fell **96 % → 82 %**. Adding the step cost
restored it: spread 5.34, argmax = coin direction **100 %**, and the agent jumped from 1.45
to 50.00 coins.

**Cause B — the learning rate never lets cells settle (Benedict).** With constant α = 0.1
every visit moves a cell by a tenth of the TD error, so busy cells keep jittering. In E05b
the wrong action won a high-traffic row by **0.022** — noise-level — and because an invalid
move leaves the state unchanged, that one cell out of 1770 cost 17 coins and 163 wasted
actions per round. A per-cell α = 1/N^0.7 (satisfying L26's `Σα = ∞`, `Σα² < ∞`, which a
constant α satisfies neither of) took the agent from 20.78 ± 6.98 to 49.15 ± 1.23.

**The unifying statement, which neither ledger reaches alone:**

> The argmax of a Q-table is decided by the *spread* between actions, not by their level.
> Anything that shrinks the spread relative to the noise in the estimate — a reward scheme
> with no cost term, or a learning rate that never converges — turns a perfectly good feature
> set into an absorbing deadlock. And because an invalid or non-moving action leaves the
> state unchanged, the failure is not graceful: it is a fixed point.

**Both also found that more training made things worse.** Maxi's v2 trained longer, in
400-step episodes, saturated *more*, and thereby lost the coin-direction signal (96 % → 82 %).
Benedict's E05b control raised the round count from 10 000 to 17 800 and the agent fell from
46.1 to 29.5 coins. Two different routes to "more data, worse frozen policy".

**Why Maxi's constant α was nevertheless fine — and why it will not stay fine.** His state
space is 39 reachable rows × 4 allowed actions = 156 cells; Benedict's was 295 × 6 = 1770,
an 11× difference on comparable training budgets. More visits per cell means better
convergence, so a constant α survives at his scale. That predicts his agent becomes fragile
exactly when the state space grows — which is what task 2 does. It is the main reason the
merged agent keeps the per-cell learning rate even though Maxi's rung-1 agent did not need it.
*(Hypothesis with a stated mechanism, not a measurement — his rung-1 result is n = 1.)*

## 5 · What we learned about method

**5.1 A single training run is not a result** (Benedict, E05b/E06). Two runs of identical
code differed by 17 coins. Five seeds of the configuration that once produced 49.93 give
20.78 ± 6.98. Two variance sources exist and a 300-round CI sees only one: *within-run*
(arenas — covered) and *between-run* (which cells exploration visited — invisible). E06 arm
A's five seeds were 11.1, 20.1, 30.0, 18.7, 24.0: five precise, mutually contradictory
numbers. **Everything from E06 on uses n = 5 and reports the spread and the worst seed.**
Maxi's rung-1 results are n = 1 and should be read as such.

**5.2 A training curve is not a result either** (both, independently). 48.16 → 1.45 for
Maxi; 44.8 → 11.1 for Benedict. Nothing is claimed until `tools/evaluate.py` has run at
ε = 0.

**5.3 Aliasing and undertraining are indistinguishable in a Q-table** (Benedict, E04→E05).
Both leave a row whose values do not separate. E04 blamed a coarse feature; E05 disproved it
without touching the feature — those states had never been visited, because training episodes
ended at step 61. Check visitation before blaming the representation.

**5.4 Run controls when you change two things at once** (Maxi, E01). Two changes, two
single-change control runs, effects cleanly separated. Cheaper than untangling it afterwards.

**5.5 Record what would refute the prediction, not just what would confirm it** (Benedict,
E07). The finding came from a branch written before the run: *"if training steps stay near 61
while evaluation still reaches ~49, the mechanism story is wrong."* It did, and it was.

**5.6 Replaying the frozen policy beats a 300-round evaluation for diagnosis** (Benedict).
The rung-1 arena is deterministic, so the policy can be replayed straight from the table in
seconds (`scratchpad/benedict/loop_check.py`). Caveat: the simplified start and coin placement
do not reproduce the game exactly — it reports 0 invalid actions where the real evaluation
finds 0.20 per round.

**5.7 Reproducibility needs two seeds** (Benedict). The exploration RNG *and* `main.py --seed`
(which places the coins). Either alone leaves runs incomparable. Train on a world seed that
is **not** the evaluation seed, or the agent is measured on arenas it trained on.

**5.8 Select the shipped model on a validation seed** (Benedict). With five trained models,
picking the best on the reported metric is cherry-picking, and picking index 0 blindly can
ship the worst — it did (s0 was 48.89 of five). Select on a held-out world seed (550731),
report on the standard one (20260731), ties to the lowest index.

**5.9 `steps` needs the completion rate beside it** (Maxi, E02 addendum). His v1 looked slow
at 293.2 steps; restricted to the 109 rounds it completed it ran at **124.1**, already as fast
as the reference. Its problem was never path length — it was getting stuck in 64 % of rounds.
Without the split, "navigates badly" and "navigates fine but stalls" are indistinguishable.

## 6 · Withdrawn claims

The report is stronger for showing these. All were stated, measured, and retracted.

| Claim | Whose | Status |
|---|---|---|
| ε decay improves task 1 (+3.8 coins) | Benedict, E05 | **Refuted.** Lucky draw; no demonstrated difference over 5 seeds (E07) |
| …because training never reaches the late round | Benedict, E05 | **Refuted.** Constant ε trains at 68–72 steps and still evaluates at 48.7 |
| Unfinished rounds are the ±3 clip saturating | Benedict, E04 | **Refuted.** Those states were unvisited, not unlearnable |
| More training data helps | Benedict, E05b | **Inconclusive** — swamped by run-to-run variance |
| The deadlock is always a blocked-direction argmax | Benedict, E05b | **Too narrow.** Usually a 2-cycle between two tiles |
| Step costs are path-length fine-tuning | Maxi, E02 pred. 4 | **Wrong, instructively.** They are the *precondition for the features to work at all* — they create the Q-value spread. Predicted `steps` would stay above 125; measured 123.8, below the reference |
| `coins` > 45 but not 50 after step costs | Maxi, E02 pred. 3 | Too cautious — 50.00 immediately |
| Maxi's higher decision time is caused by BFS | Benedict | **Too simple.** BFS is *faster* than the offset features when coins are dense; see §8 |

## 7 · The merged agent — `agent_code/tabular_q_task1/`

Built jointly after both agents solved the rung, on the evidence above.

| Component | From | Why |
|---|---|---|
| BFS direction to nearest coin (5 values) | **Maxi** | The component that beats the reference on `steps`. Routes along the actual path, and **can never point into a wall** — the origin of Benedict's row-93/409 deadlocks. Legitimate from task 2 on, where it stops being the optimal policy |
| Random tie-breaking among equal Q-values | **Maxi** | Insurance against a near-tie becoming an absorbing loop |
| Step cost −0.1 | **both** | Maxi E02 showed it is what creates the Q-value spread; Benedict had it from E01 |
| Per-cell α = 1/N^0.7 | **Benedict** | Largest measured effect; and see §4 on why it matters more as the state space grows |
| Full six-action set, `BOMB` handled by `KILLED_SELF = −5` | **Benedict** | Masking must be undone on rung 2 anyway; the reward already solves it |
| Dense array + mixed-radix `encode()` | **Benedict** | Keeps the reachable-row count checkable, which caught two real bugs. Storage sits behind `encode()`, so a dict is a two-line swap |
| Seeded RNGs, `BM_MODEL_SUFFIX`, `TrainLogger`, ablation switches | **Benedict** | The measurement harness |
| **Dropped:** clipped ±3 offset | Benedict | BFS direction subsumes it |

**State:** 4 wall bits + BFS direction → 16 × 5 = **80 rows, 39 reachable** (enumerated over
all 176 × 176 tile/coin pairs — the same 39 Maxi's agent converged to). BFS points into a
wall in **0** of those pairs. The trained table occupies exactly 39.

**Result (5 seeds × 300 rounds):**

| | coins | std | worst seed | steps | invalid |
|---|---|---|---|---|---|
| Reference | 50.00 | — | — | 125.3 | 0.00 |
| Benedict E06 | 49.15 | 1.23 | 47.16 | 130.3 | 0.06 |
| Maxi E02-v3 | 50.00 | — (n=1) | — | 123.8 | 0.06 |
| **Merged** | **49.62** | **0.46** | **48.89** | **122.9** | 0.11 |

Shipped model **s1**, selected on validation seed 550731: **50.000 coins, 123.7 steps,
100 % survival, 0.20 invalid** on the reported seed. No row has `BOMB` as argmax; the replay
diagnostic finds **0 absorbing loops in 300 simulated rounds**.

The merge keeps Maxi's routing (122.9 vs his 123.8, both under the reference), halves the
seed-to-seed spread of either parent, and raises the worst seed. The tie-break never fires —
the converged table has **no ties at all** among its 39 rows — so it is pure insurance for
task 2, where per-cell convergence will be thinner.

## 8 · Head to head (300 paired rounds, 2026-08-06)

**Solo, identical arenas** — both at 50.000 ± 0.000 coins, so the rung's pass criterion
separates nothing:

| Metric | Maxi | Benedict | Reading |
|---|---|---|---|
| steps | **123.8** | 130.3 | **Maxi**, +6.4 [+5.2, +7.6]. `analyze.py` labels this backwards |
| invalid | 0.06 | **0.00** | Benedict |
| `think_mean_ms` | 0.056 | **0.017** | Benedict |

**Head to head on one board:** paired coin difference **−0.577, 95 % CI [−1.327, +0.173]** —
no demonstrated difference (114 ahead, 33 tied, 153 behind).

**The finding neither of us would have made alone:** in the duel, invalid actions jump from
0.00/0.06 solo to **11.5 (Benedict) and 155.8 (Maxi)** per round. `environment.py:121` counts
active agents and bombs as blocking, and **neither feature map contains `others`** — both
agents repeatedly try to walk through their opponent. Free on rung 1, expensive from rung 3.
(Why one degrades 14× worse is a hypothesis — masking `WAIT` may leave no legal way to yield —
and needs a controlled run before it is claimed.)

**Decision cost** (both far under the 500 ms budget):

| coins on board | Maxi `state_to_features` | Benedict |
|---|---|---|
| 50 | **0.0020 ms** | 0.0127 ms |
| 5 | 0.0216 ms | **0.0093 ms** |
| 1 | 0.1158 ms | **0.0102 ms** |

BFS scales *inversely* with coin density — it stops at the first coin reached. Crossover
around 5–10 coins. Maxi's higher round average comes from the tail plus ~0.009 ms from two
`logger.debug` calls in `act` (`settings.py:71` sets agent code to DEBUG). His
`think_max_ms` of 22.3 is **not** BFS — the worst measured BFS call is 0.19 ms and the spikes
are scattered across rounds, so they are GC or scheduler noise.

## 9 · Evidence map

| Claim | File |
|---|---|
| Reference agents (E00) | `results/eval/baselines/baseline_*__task1.csv` |
| Benedict E01–E05 progression | `results/eval/benedict_q_v{1..5}__task1.csv` |
| Benedict E05b (the 17-coin swing) | `results/eval/benedict_q_v4b__task1.csv` |
| Benedict E06 both arms, 5 seeds | `results/eval/benedict_q_e06{a,b}_s{0..4}__task1.csv` |
| Benedict E07 ε ablation, 5 seeds | `results/eval/benedict_q_e07c_s{0..4}__task1.csv` |
| Maxi E01/E02 | `results/eval/maxi_q_v{1,2,3}__task1.csv` |
| Cross-agent solo / duel | `results/eval/{benedict,maxi}_final__task1.csv`, `duell_benedict_vs_maxi__task1.csv` |
| Merged agent, 5 seeds | `results/eval/benedict_merged_s{0..4}__task1.csv` |
| Merged agent, model selection | `results/eval/benedict_merged_s{0..4}__task1_val.csv` (seed 550731) |
| Merged agent, shipped model | `results/eval/benedict_merged__task1.csv` |
| Learning curves | `results/train/*.csv` + `.meta.json` (hyperparameters per run) |
| Figures | `tools/plot_task1_versions.py` (steps split by completion), `analyze.py --plot` |

Every `.meta.json` records the git commit, seed, scenario and a snapshot of `settings.py`.

## 10 · Carried into task 2

**Method:** per-cell α · n = 5 seeds · prediction *and its refutation condition* before the
run · single-change controls when two things move · validation-seed model selection · the
replay diagnostic · both RNGs seeded · completion rate always beside `steps`.

**Kept without rung-1 evidence, deliberately:** the ε schedule. Not justified by the numbers
(E07). Retained because §4 explains why its mechanism should apply on task 2 and not here.
If it earns nothing there either, it goes.

**Required next:**

1. A **danger feature** — in blast radius / steps to detonation / is an escape available.
   Both ledgers converge on this: Maxi's E01 measured 100 % `KILLED_SELF` with `BOMB` in the
   action set and no danger feature, and Benedict's E03 showed that the reward penalty only
   worked because the agent *stood still* rather than fleeing. On task 2 it must flee, so the
   max-bootstrap argument applies and a penalty alone will not be enough.
2. An **opponent feature** — §8. Free on rung 1, expensive from rung 3.
3. `CRATE_DESTROYED` in the reward table.
4. The **`COIN_COLLECTED` +5 vs the game's actual +1 ablation**, open since Benedict's E01.
   Maxi's E02 makes this sharper, not less urgent: the *ratio* between the coin reward and
   the step cost is what sets the Q-value spread, so the two must be tuned together.
5. Re-check the **spread and argmax-follows-feature diagnostic** (Maxi's E01/E02 table) after
   every feature change. It caught a broken policy that every other number called healthy.

## 11 · Reproduction

```bash
# five seeds in parallel, each with its own table
for i in 0 1 2 3 4; do
  ( export BM_MODEL_SUFFIX="_s${i}"
    BM_RUN_INDEX=$i uv run python main.py play --no-gui --agents tabular_q_task1 \
      --train 1 --scenario coin-heaven --n-rounds 10000 --seed 810731 >/dev/null 2>&1
    uv run python tools/evaluate.py --agents tabular_q_task1 --opponents none \
      --scenario coin-heaven --n-rounds 300 --label benedict_merged_s${i}__task1 --quiet ) &
done
wait

# model selection on the held-out seed, then ship
for i in 0 1 2 3 4; do
  ( export BM_MODEL_SUFFIX="_s${i}"
    uv run python tools/evaluate.py --agents tabular_q_task1 --opponents none \
      --scenario coin-heaven --n-rounds 300 --seed 550731 \
      --label benedict_merged_s${i}__task1_val --quiet ) &
done
wait
cp agent_code/tabular_q_task1/q_table_s1.npy agent_code/tabular_q_task1/q_table.npy
rm -f agent_code/tabular_q_task1/q_table_s*.npy
```

Defaults in `train.py` are the settings rung 1 ended on (`BM_ALPHA=visit`, `BM_EPS=decay`).
Training world seed 810731 is deliberately not the evaluation seed 20260731.
