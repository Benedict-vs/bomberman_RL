# Versuchsprotokoll — Benedict

Ergebnisbuch, ein Eintrag pro Experiment. Bewusst getrennt von `BENEDICT.md`:
das Logbuch hält fest, *warum* ich etwas entschieden habe, hier stehen die *Zahlen*.
Die Zeilen hier werden im Bericht fast wörtlich zu Tabellenzeilen.

Regeln, an die ich mich halte:

- **Vorhersage vor der Messung.** Steht sie nicht vorher da, war es kein Experiment,
  sondern eine Beobachtung. Ich committe die Vorhersage, bevor ich messe — dann
  belegt die Git-Historie die Reihenfolge.
- **Eine Änderung pro Eintrag.** Sonst ist nicht zuzuordnen, was gewirkt hat.
- **Commit-Hash mitschreiben.** Steht in `results/eval/<label>.meta.json`. Mit festem
  Seed ist ein Lauf damit exakt reproduzierbar.
- **Negative Ergebnisse bleiben stehen.** Sie kommen so in den Bericht.
- Fester Seed `20260731`, 300 Runden für jede berichtete Zahl.
- **Neueste Einträge oben**, wie im Logbuch. Für den Bericht wird von unten nach oben
  gelesen — E01, E02, … ist die Reihenfolge, in der die Argumentation aufgebaut ist.

Urteil: **BESSER** · **SCHLECHTER** · **nicht gezeigt** (KI enthält die Null).

---

## E06 — A per-cell learning rate, and 5 seeds per configuration

- **Question:** E05b showed that two training runs of *identical code* can differ by 17
  coins. Does a learning rate that decays with the number of visits to a cell remove that
  fragility — and how large is the run-to-run spread once it is measured properly?

- **Why α.** In E05b the wrong action won row 409 by **0.022**. With a constant α = 0.1
  every visit moves a cell by a tenth of the TD error, so a high-traffic cell never settles;
  it keeps jittering at exactly the scale that decides the argmax. L26 gives the condition
  for convergence — `sum α = ∞` and `sum α² < ∞` — which a constant α does not satisfy.
  A per-cell schedule does:

  `α(s,a) = 1 / N(s,a)^0.7`, with `N(s,a)` the visit count. Exponent 0.7 lies in the (0.5, 1]
  window where both conditions hold. High-traffic cells like row 409 become nearly frozen;
  rarely visited ones keep learning fast. That is the exact shape of the problem.

  Deliberately *not* doing action masking, which was the other candidate. The agent should
  learn that walking into a wall is bad — it already does, `invalid` is 0.15 in v5 — and
  masking would hide the instability rather than fix it.

- **Change from E05:** `self.alpha` becomes `1 / N(s,a)^0.7` instead of the constant 0.1.
  φ, rewards, γ and the ε schedule are untouched.

- **Design — two arms, five seeds each.** This is the first experiment with n > 1, because
  E05b proved n = 1 is not interpretable:

  | arm | configuration | runs |
  |---|---|---|
  | **A (baseline)** | v5 exactly: constant α = 0.1, ε decay | seeds 0–4 |
  | **B** | as A but per-cell α | seeds 0–4 |

  Arm A doubles as the **replication of E05**, which is still an n = 1 claim.

- **Reproducibility, fixed before this experiment (infrastructure, not a version).**
  Two independent RNGs had to be pinned, which is why v4 and v4b were incomparable:
  1. the agent's exploration — `self.rng = np.random.default_rng(TRAIN_SEED + RUN_INDEX)`
     in `setup_training`, drawn from in `act`. Seeded here rather than in `callbacks.py` so
     the tournament path, where `train.py` is never imported, cannot touch it;
  2. the world — `main.py --seed`, which places the coins. Without it every run sees
     different arenas.

  Verified: same index + same world seed → **bit-identical** Q-tables; different index →
  different tables. The training world seed is **810731**, deliberately *not* the evaluation
  seed 20260731, so the agent is never trained on the arenas it is measured on.

  Seeding buys reproducibility, not reliability — a bad draw returns identically. Reliability
  is what the five runs per arm are for.

### Prediction (written before the run)

1. **The means barely move.** Arm A ≈ arm B ≈ 49–50 coins. Task 1 is at the ceiling; there
   is under half a coin of headroom. Anyone reading only the means will conclude "no effect".
2. **The spread collapses, and that is the whole point.** Standard deviation of `coins`
   across the five seeds: arm A above 5 (it must contain runs like v4b), arm B below 1.
   The metric that decides this experiment is **the worst run**, not the mean:
   min over seeds ≥ 48 for arm B, and I expect arm A to produce at least one run below 40.
3. **Rows whose argmax is a blocked direction:** arm B has zero in high-traffic rows
   (offset ≠ (0,0)). Arm A has at least one such row in at least one of its five runs.
4. **Arm A replicates E05 in the mean** — around 49–50 coins, confirming the ε-decay result
   is real and not the same kind of lucky draw that v4 turned out to be.
5. **Risk:** with α = 1 on the first visit, early estimates swing violently, and a cell
   visited once early can sit on a wild value for a long time before enough visits pull it
   back. If arm B is *worse* on the mean, that is the mechanism, and the fix is a floor
   (`α = max(0.01, 1/N^0.7)`) rather than abandoning the schedule.

**What a null result would mean.** If both arms have small spread, then E05b's v4b was not
learning-rate instability but something specific to that draw, and the report's claim
becomes "single training runs are not comparable, therefore we report n = 5" — which is
worth stating regardless, since it changes how every earlier number should be read.

### Result

| Metric | arm A (const α) | arm B (per-cell α) | |
|---|---|---|---|
| `coins` mean over 5 seeds | | | |
| `coins` std over 5 seeds | | | |
| `coins` worst seed | | | |
| `invalid` mean / worst | | | |
| runs with a blocked-argmax high-traffic row | | | |

### Verdict

*(filled in after the measurement)*

### What I do next

*(filled in after the measurement)*

---

## E05b — Control: is E05 the distribution or just more data?

- **Question:** E05 trained on 1.8× the transitions of E04 because ε decay lengthens
  episodes. Was the gain the *distribution* of the data or merely its *volume*? Control:
  v4's configuration (constant ε = 0.2) with the round count raised from 10 000 to 17 800
  so total steps match.
- **Agent:** v4b · commit `28f32ba` · `eps_decay = 1`, everything else identical to v5.
- **Prediction:** no improvement, `coins` stays near 46. Under constant ε the agent dies at
  step 61 regardless of round count, so late-round states stay unvisited; more data cannot
  reach states the policy never enters.

### Result

| Metric | v4 (10 000 rounds) | v4b (17 800 rounds) | Paired difference | 95 % CI |
|---|---|---|---|---|
| `coins` | 46.107 | **29.477** | −16.630 | [−18.770, −14.537] |
| `steps` | 176.5 | 299.6 | +123.117 | [+102.036, +143.584] |
| `invalid` | 0.12 | **163.15** | +163.030 | [+143.586, +182.914] |
| `suicides` | 0.010 | 0.167 | +0.157 | [+0.117, +0.200] |

Training dynamics were indistinguishable from v4 (ε 0.200 throughout, 55–65 steps per
episode, `KILLED_SELF` 0.95, ~4 invalid actions per episode). The configuration was correct.
The *agent* was not.

### Cause: one cell

Row 409 — `UP` blocked, coin one tile to the left. A very common state.

| | UP (blocked) | RIGHT | DOWN | LEFT (correct) | argmax |
|---|---|---|---|---|---|
| v4 | 15.730 | 15.457 | 15.752 | **18.935** | LEFT |
| v5 | 12.169 | 12.609 | 12.792 | **15.878** | LEFT |
| v4b | **15.514** | 13.479 | 15.492 | 14.267 | **UP** |

In v4b the correct action ranks fourth and `UP` wins by **0.022**. And an invalid action
**does not move the agent**, so the next state is identical, so it chooses `UP` again — a
deterministic, absorbing deadlock. 163 wasted actions per round out of one wrong cell in
1770.

That is the amplifier worth remembering: elsewhere a 0.022 error costs 0.022. Here it costs
17 coins, because the failure mode is a fixed point rather than a detour.

### Verdict

**The control does not answer the question it was built to answer, and reveals a bigger
problem.** v4 and v4b differ in round count *and* in the unseeded exploration RNG, so this
is n = 1 against n = 1. A 17-coin swing traced to a single cell is run-to-run variance, not
evidence about data volume.

The uncomfortable consequence: **E05's +3.823 is also n = 1 against n = 1** and is not yet a
result. Its mechanism evidence is much stronger — training episodes 61 → 119 steps,
completions 242 → 299, the entire training distribution shifted — but the effect size cannot
be claimed until it is replicated. That is arm A of E06.

`tools/README.md` has said "3–5 runs per configuration with different seeds" from the start.
Every experiment up to here ignored it, and E01–E05 should be read as single draws.

### What I do next

1. **Seed both RNGs** (agent exploration and world) so runs are reproducible at all. Done as
   infrastructure ahead of E06; verified bit-identical.
2. **E06 with five seeds per arm**, arm A replicating E05.
3. **Not** action masking. The deadlock is real, but the honest fix is a learning rate that
   converges, not removing the agent's ability to make the mistake.

---

## E05 — Decay ε so training sees the round the agent actually plays

- **Question:** ε has been fixed at 0.2 since E01. With six actions that is a 3.3 % chance
  of `BOMB` per step, so the training agent bombs itself roughly every 30 steps and never
  survives long enough to see a whole round. Does closing the gap between the training and
  evaluation distributions help — and if not, what does that tell me?

- **The gap, measured on the v4 logs:**

  | | training, ε = 0.2 | evaluation, ε = 0 |
  |---|---|---|
  | steps per episode | **61.0** | 176.5 (128 to completion) |
  | coins per episode | 20.6 | **46.1** |
  | bombs dropped | **1.97** | 0.00 |
  | `KILLED_SELF` | **0.94** | 0.01 |

  These are two different agents. The one that generates the training data drops two bombs
  per episode and dies almost every time; the one that gets measured never bombs at all.
  The numbers are flat across all 10 000 episodes, so this is not a warm-up effect — the
  agent **never** experiences a round past step ~61, which is precisely the late-round
  regime (few coins left, all of them far away) where the 58 unfinished evaluation rounds
  break down.

- **Change from E04:** exactly one — ε becomes a schedule instead of a constant.
  `EPS_START = 0.2`, `EPS_END = 0.02`, exponential decay per episode
  (`self.eps = max(EPS_END, self.eps * 0.99967)`), reaching the floor at ~70 % of training.
  φ is untouched (784 rows), all rewards untouched, α and γ untouched.

- **Agent:** `benedict_coin_collector` v5 · commit `<to be filled in>`
- **Training:** 10 000 rounds, `coin-heaven`, no opponents, `run = q_v5_task1`.
- **Measurement:** `results/eval/benedict_q_v5__task1.csv`, 300 rounds, seed 20260731,
  paired against `benedict_q_v4__task1.csv`.

- **Known confound, stated up front.** Longer episodes mean more transitions per round, so
  v5 trains on substantially more data than v4 at the same round count. An improvement could
  therefore come from *more data* rather than from *better-distributed data*. The control is
  a second v4-configuration run with the round count raised until total steps match
  (`q_v4b_task1`). I will only run it if E05 shows an effect worth attributing — if the
  result is null, the confound does not matter, because more data did not help either.

### Prediction (written before the run)

1. **Training dynamics change sharply, and this is near-certain.** By the last 1000
   episodes: `steps` above 120 (from 61.0), `KILLED_SELF` below 0.15 (from 0.94),
   `BOMB_DROPPED` below 0.3 (from 1.97). If this does *not* happen the schedule is not
   wired up correctly and nothing else in the experiment means anything.
2. **`coins` barely moves — I predict 46 to 49, quite possibly "no effect shown".**
   The 58 unfinished rounds fail because the clipped offset saturates beyond ±3, which is
   a *representational* limit. Extra visits to far-field states add data, not information.
   A tabular method cannot separate two states that encode identically no matter how often
   it sees them.
3. **`suicides` stays around 0.01 or improves slightly.** Row 171's `BOMB` argmax comes
   from the `(0,0)` collision at round start, which occurs at a rate independent of ε.
   Lower ε means fewer `(171, BOMB)` samples *and* fewer terminal penalties on them, so the
   sign of the effect is genuinely unclear to me.
4. **Real risk in the other direction:** with ε at 0.02 for the last 3000 episodes, rarely
   visited rows stop being refreshed and can freeze at noisy values. If `invalid` or
   `suicides` rise, that is the mechanism, and the answer is a higher floor rather than
   abandoning the schedule.
5. **Occupied rows stay at 295** — or drop slightly, if the reduced exploration means some
   far rows are never entered at all. A drop would itself be evidence for point 4.

**What a null result would mean, and why it is still worth running.** If prediction 2 holds,
the conclusion is that on this rung the binding constraint is the feature map and not the
data, which is a much stronger statement than "clipping is the limit" made from E04 alone —
it survives a deliberate attempt to fix the problem by other means. It also matters for
task 2 regardless of the outcome here: there, dying early is not an exploration artefact but
the thing to be learned, and an agent that never survives its own bomb during training cannot
learn what comes after it.

### Result

| Metric | v4 | v5 | Paired difference | 95 % CI | Verdict |
|---|---|---|---|---|---|
| `coins` | 46.107 | **49.930** | +3.823 | [+2.720, +4.993] | BETTER |
| `steps` (completion time) | 176.5 | **130.4** | −46.100 | [−58.707, −34.327] | faster, see note |
| `invalid` | 0.12 | 0.15 | +0.023 | [−0.020, +0.067] | no effect shown |
| `suicides` | 0.010 | **0.000** | −0.010 | [−0.023, +0.000] | no effect shown |
| `survived` | 0.990 | **1.000** | +0.010 | [+0.000, +0.023] | no effect shown |
| rounds collecting all 50 | 242 / 300 | **299 / 300** | — | — | — |
| occupied rows | 295 | 295 | — | — | — |
| rows with `BOMB` as argmax | 1 | **0** | — | — | — |

**49.93 of 50 coins. 299 of 300 rounds are a clean sweep**, median completion 129 steps
(105–160). The single failure collected 29. `steps` falls because completion is faster, not
because survival is worse — the inversion `AGENTS.md` now documents.

Training dynamics, first vs last 1000 episodes:

| | ε | steps | `KILLED_SELF` | `BOMB_DROPPED` | coins |
|---|---|---|---|---|---|
| first 1000 | 0.157 | 72.2 | 0.922 | 1.83 | 22.93 |
| last 1000 | 0.020 | **119.3** | **0.193** | **0.52** | **45.48** |

Total training steps 1 086 665 against v4's 610 208.

### Predictions, scored

1. **Training dynamics — held directionally, thresholds slightly optimistic.** `steps`
   61.0 → 119.3 (predicted > 120, near enough), `KILLED_SELF` 0.94 → 0.193 (predicted
   < 0.15, missed), `BOMB_DROPPED` 1.97 → 0.52 (predicted < 0.3, missed). The mechanism is
   confirmed; my numbers were a little too generous.
2. **`coins` 46–49, "quite possibly no effect shown" — wrong, and wrongly reasoned.**
   49.930, +3.823 with a CI far from zero. See below; this is the important one.
3. **`suicides` — better than predicted.** Exactly 0.000, and the `BOMB`-argmax row is gone
   entirely. I had called the sign of this effect genuinely unclear.
4. **Frozen rare rows — did not happen.** `invalid` unchanged, occupancy still 295, despite
   ε sitting at the 0.02 floor for over half the run.
5. **295 occupied rows — held exactly.**

### This refutes E04's conclusion, not just this prediction

E04 concluded that the 58 unfinished rounds were caused by the clipped offset saturating
beyond ±3 — a *representational* limit — and I wrote that extra visits to far-field states
would "add data, not information". That was wrong, and E05 is the counter-example: nothing
about φ changed, and 57 of those 58 rounds now finish.

Why the reasoning failed: a saturated state such as `(dx=+3, dy=+3)` means "the coin is at
least 3 right and at least 3 down". That is not information-free — the *direction* is still
there, only the distance is gone. Moving right or down is correct and perfectly learnable in
such a state. The states were not unlearnable, they were **unvisited**: under constant
ε = 0.2 the agent died at step 61 and the late-round regime never appeared in its training
data at all.

The general lesson, and it is the more useful one for the report: *before concluding that a
feature is too coarse, check whether the states in question were ever visited.* Aliasing and
undertraining look identical in a Q-table — both leave a row whose values do not separate —
and I diagnosed one as the other. The check is cheap: the training log already carries
episode length, and 61 versus 176 was visible in E04's own data.

### The confound is now worth resolving

Stated before the run: longer episodes mean more transitions, so v5 saw 1.8× the data at
the same round count. The improvement could be *more* data rather than *better-distributed*
data. E05 showed a clear effect, so by the rule I set myself the control is now due.

**E05b:** v4 configuration (constant ε = 0.2), round count raised until total steps match —
610 208 steps came from 10 000 rounds at 61 steps, so ≈ 17 800 rounds reaches 1.09 M.
Prediction: **no improvement over v4, coins stays near 46.** Under constant ε the agent dies
at step 61 no matter how many rounds are run, so the late-round states remain unvisited;
more data cannot reach states the policy never enters. If v4b *does* improve, then volume
was the driver and the distribution argument above is wrong.

### Verdict

**BETTER, and task 1 is finished.** 49.93 of 50 coins, 299/300 perfect rounds, 100 %
survival, zero suicides, 0.15 invalid actions per round, 0.1 ms per decision against a
500 ms budget. There is no headroom left worth pursuing on this rung.

### What I do next

1. **E05b, the matched-step control.** Cheap, and it decides whether the report claims
   "training distribution" or only "more data".
2. **Then task 2** (`classic`, no opponents): crates, bombs that must be used, and escape.
   Everything above is a navigation agent that survives by never bombing — on task 2 it must
   bomb deliberately and then run, which is the case where E03's max-bootstrap argument
   genuinely applies and where a danger feature becomes mandatory.
3. **Carry the ε schedule forward.** On task 2 it matters more, not less: dying early is the
   thing to be learned rather than an exploration artefact, and an agent that never survives
   its own bomb during training cannot learn what comes after it.
4. **Still open and now cheap to settle:** the `COIN_COLLECTED` +5 vs the game's +1 ablation
   (open since E01), and the `(0,0)` collision between "no coins left" and "coin on my own
   tile". The latter no longer costs anything measurable, but it will confuse task 2, where
   coins can genuinely be absent for long stretches.

---

## E04 — Give the coin feature a sense of distance

- **Question:** E03 removed every death, so the agent now has all 400 steps available —
  and uses almost none of them. Replaying the trained policy shows it walking back and
  forth between two tiles. Does the agent need to know *how far* the coin is, not just
  which way?
- **The observation that prompted this.** Rebuilding the greedy policy from the v3 table
  and simulating 300 rounds (the `coin-heaven` field is deterministic, so only coin
  placement varies):

  | | |
  |---|---|
  | rounds entering an **absorbing** loop | **300 / 300** |
  | median step at which the loop starts | **37** |
  | coins collected before the loop | 18.4 |
  | coins collected after it | **0** |
  | simulated coins/round | 18.4 (measured 16.7 — the model reproduces reality) |

  The agent is productive for roughly 40 of its 400 steps. The remaining 90 % of every
  round is spent oscillating. That is a far larger effect than anything left in the
  reward function.

- **Why the loop is absorbing.** A deterministic memoryless policy has an eventually
  periodic trajectory — that much is unavoidable. What makes it *permanent* is that the
  loop passes over no coin, so the coin set never changes, so the state never changes.
  And the two tiles are indistinguishable because **`sgn(Δ)` carries no distance
  information**: it is scale-free, identical whether the coin is 2 tiles away or 12.
  Neither cell of the 2-cycle knows it is the closer one.

- **Change from E03:** exactly one. `sgn(Δx), sgn(Δy)` → **Δx, Δy clipped to [−3, +3]**,
  i.e. 7 × 7 = 49 values instead of 9. `FEATURE_SIZES = (2,2,2,2,7,7)`, 784 rows.
  Everything else identical: α = 0.1 · γ = 0.9 · ε = 0.2 · coin +5 · `KILLED_SELF` −5 ·
  `INVALID_ACTION` −1 · `WAITED` −0.1 · step cost −0.1.

  Why this encoding and not a BFS first step: the BFS direction would be the stronger fix,
  but on `coin-heaven` it is essentially *the optimal policy*, and the task description
  forbids a feature that returns the best action. The clipped offset hands the model the
  information it needs to learn routing without handing it the answer. It also **subsumes**
  the sign feature — outside the ±3 box it degrades to the same coarse direction — so it
  cannot carry less information than what it replaces.

- **Agent:** `benedict_coin_collector` v4 · commit `<to be filled in>`
- **Training:** 10 000 rounds, `coin-heaven`, no opponents, `run = q_v4_task1`.
  295 reachable rows × 6 = 1770 live cells; training episodes still end at ≈ 51 steps
  (ε forces a bomb every ~30 steps even though the greedy policy never bombs), so the
  sample rule gives 50 · 1770 / (51 · 0.2) ≈ 8700. Round up to 10 000.
- **Measurement:** `results/eval/benedict_q_v4__task1.csv`, 300 rounds, seed 20260731,
  paired against `benedict_q_v3__task1.csv`.

### Prediction (written before the run)

1. **Loops stop being universal.** Re-running the simulation on the v4 table: fewer than
   100 of 300 rounds enter an absorbing loop, and the median entry step moves past 150.
   This is the primary prediction — it is the mechanism the change targets, and it is
   checkable without a 300-round evaluation.
2. **`coins` between 25 and 35** (from 16.703). If loops largely disappear the agent has
   ~10× the productive steps but faces the diminishing-return effect noted in E03: the
   later the coin, the further away it is. I do not expect anything near 50.
3. **295 of 784 rows occupied.** Counted in advance over all (tile, coin) pairs *including
   a coin on the agent's own tile* — the case whose omission made the E02 count wrong.
4. **Regression guards: `suicides` stays 0.000 and `steps` stays ≈ 400.** The feature
   change touches nothing about bombs. If suicides return, the extra rows have diluted the
   data enough that `BOMB` wins some row by noise again, which would mean 10 000 rounds is
   still too few.
5. **`invalid` stays around 0.08.**
6. **Not predicted to be fixed: the training/evaluation mismatch.** Training episodes end
   at ~51 steps, so the agent still learns almost exclusively from the opening of a round
   and barely sees the late-round regime (few coins, all far away) in which it spends most
   of its evaluation time. That is E05 (ε decay), deliberately not bundled here.

### Result

| Metric | v3 | v4 | Paired difference | 95 % CI | Verdict |
|---|---|---|---|---|---|
| `coins` | 16.703 | **46.107** | +29.403 | [+27.627, +31.200] | BETTER |
| `steps` | 399.1 | 176.5 | −222.620 | [−234.790, −209.647] | see below |
| `invalid` | 0.08 | 0.12 | +0.040 | [−0.010, +0.090] | no effect shown |
| `suicides` | 0.000 | 0.010 | +0.010 | [+0.000, +0.023] | no effect shown |
| `survived` | 1.000 | 0.990 | −0.010 | [−0.023, +0.000] | no effect shown |
| occupied rows | 66 / 67 | **295 / 295** | — | — | — |

**46.1 of 50 coins.** Nearly triple E03, and the CI is nowhere near zero.

### `steps` did not get worse — the metric inverts here

`environment.py:289`: with one agent left, no crates, no collectable coins and no bombs,
the round is wrapped up. On `coin-heaven` with a single agent that means **the round ends
the moment the last coin is collected**. So `steps` is not survival time, it is completion
time, and it fell because the agent now finishes the task instead of wandering to step 400.
`analyze.py` labels it WORSE because the metric table declares "higher is better", which is
correct on every other rung and wrong on this one.

The breakdown makes it unambiguous:

| | rounds | coins | steps |
|---|---|---|---|
| collected **all 50** | **242 / 300** | 50 | median **128** (106–154) |
| did not finish | 58 / 300 | 29.9 | 379.6 |
| died | 3 / 300 | 2–3 | 5 |

81 % of rounds are a clean sweep in ~128 steps. The distribution is bimodal, not spread:
the agent either solves the arena or gets stuck in the far field.

### Predictions, scored

1. **"Fewer than 100 of 300 loop, median entry past 150" — badly framed, not just wrong.**
   Still 300/300, median entry step 129. But the median number of coins *remaining* at loop
   entry is **0**: in most rounds the "loop" is the agent standing around after collecting
   everything, which is not a failure. I picked a proxy metric without checking that it
   measured what I cared about. The honest version of this prediction is "coins collected
   before the loop", which went 18.4 → 46.8.
2. **"`coins` 25–35" — wrong, too pessimistic.** 46.107. I assumed the diminishing-return
   effect from E03 would bite much harder than it does. Simulation predicted 46.8 against
   46.1 measured, so the simulator is trustworthy to ~1.5 %.
3. **"295 of 784 rows occupied" — exactly right.** The counting method finally works,
   including the coin-on-own-tile case that made E02's count wrong.
4. **Regression guards — half held.** `suicides` 0.010 and `invalid` 0.12, neither
   demonstrated as an effect. But one row *does* have `BOMB` as argmax, where I predicted
   zero. See below; the caveat I attached to this prediction ("the extra rows have diluted
   the data") was the right worry pointing at the wrong row.
5. `invalid` unchanged — held.

### The 3 deaths are the E02 collision coming due

```
row 171  walls(U,R,D,L)=(0,0,1,1)   offset(dx,dy)=(0,0)
         q = [19.26 18.97 18.80 18.70 19.48 21.14]   argmax BOMB
```

Walls down and left = the bottom-left corner. Offset `(0,0)` is the ambiguous code flagged
in E02: it means **either** "no coins left" **or** "a coin is on my own tile". A coin
spawning under the agent at round start puts it in exactly this row, `BOMB` wins, and it
dies at step 5 — which is precisely the three rounds that died with 2–3 coins after 5 steps.

In E02 I wrote that the collision's "practical harm is small". It is small — 1 % of rounds —
but it is no longer hypothetical, and it is the only remaining source of death.

### Why the residual loops happen

100 % of genuine loops (coins still on the board) occur with the nearest coin **further than
the clip radius**: mean distance 5.0, median 4, against a clip of ±3. Beyond ±3 the feature
saturates and degrades to exactly what v3 had — coarse direction, no distance. The v3
failure mode was not removed, it was pushed into the far field, where it now costs 3.9
coins instead of 30.

### Verdict

**Task 1 is solved for practical purposes.** 46.1/50 coins, 81 % perfect rounds at a median
of 128 steps, 99 % survival, 0.12 invalid actions per round, 0.1 ms per decision against a
500 ms budget. The two remaining defects both have named causes rather than being noise.

### What I do next

1. **Do not chase the last 3.9 coins.** The fix is known — non-linear binning
   (`{0, ±1, ±2, ±3, ±(4–6), ±(7+)}`, 11 values per axis, 1936 rows) would give far-field
   discrimination without a ±6 clip's 2704 rows. It goes in the report as the identified
   next step. The effort belongs on task 2.
2. **Fix the `(0,0)` collision** whenever φ is next touched: "no coins left" and "coin on my
   own tile" need separate codes. Cheapest as a separate flag rather than an eighth offset
   value.
3. **E05: ε decay.** Unaffected by any of this and now the largest structural problem.
   Training episodes end at ~51 steps because ε forces a bomb every ~30, while evaluation
   runs 128–400. The agent learns almost entirely from the opening of a round. This will get
   worse on task 2, where surviving longer *is* the task.
4. **Then task 2 with a danger feature** — the case where E03's max-bootstrap argument
   actually applies, because there the agent must bomb a crate and genuinely run away.
5. **Measurement-chain note for the team:** `analyze.py` treats `steps` as
   higher-is-better. On task 1 with a single agent that is backwards, because the round
   ends on completion. Worth a footnote in `tools/README.md` rather than a code change —
   the direction is right for rungs 2–4.

---

## E03 — Penalise `KILLED_SELF`

- **Question:** E02 left the agent dead for 40 % of the round, 42.7 % of it by its own
  bomb. Nothing in the reward function says dying is bad — the only pressure against
  `BOMB` is structural (the episode ends, so the terminal update carries no bootstrap
  term), and in the aliased row 93 that pressure was worth 0.02. Does saying it out loud
  fix it?
- **Change from E02:** exactly one entry, `e.KILLED_SELF: -5`. Everything else identical:
  φ unchanged (144 rows), α = 0.1 · γ = 0.9 · ε = 0.2 · coin +5 · `INVALID_ACTION` −1 ·
  `WAITED` −0.1 · step cost −0.1.
  Magnitude chosen deliberately equal to one coin: dying should cost about what the agent
  is chasing. Not tuned — if the sign of the effect is right and the size is not, that is
  a separate, later question.
- **Agent:** `benedict_coin_collector` v3 · commit `<to be filled in>`
- **Training:** 5000 rounds, `coin-heaven`, no opponents, `run = q_v3_task1`.
- **Measurement:** `results/eval/benedict_q_v3__task1.csv`, 300 rounds, seed 20260731,
  paired against `benedict_q_v2__task1.csv`.

### Prediction (written before the run)

**I expect this to fail, and the reason is the point of the experiment.**

The bomb kills four steps after it is dropped:
`s0 --BOMB--> s1 --a1--> s2 --a2--> s3 --a3--> dead`.

- The update for the `BOMB` action is `Q(s0,BOMB) <- r + γ·max_a Q(s1,a)`, where `r` is
  only the step cost — `BOMB_DROPPED` is not in the reward table. And **φ contains no
  bomb information**: `s1` is four wall bits plus a coin direction, indistinguishable from
  a state with no bomb anywhere. So `max Q(s1,·)` stays around 9 and `Q(s0,BOMB)` stays
  around 8. The new penalty never touches the action that caused the death.
- The −5 lands on the terminal update instead, `Q(s3,a3) <- −5.1`, where `a3` is some
  ordinary move. It does not propagate backwards, because Q-learning bootstraps with
  **max**: only `a3` was depressed, the other five actions in `s3` are untouched, and `s3`
  is aliased with safe states in which those actions really are good. So `max Q(s3,·)`
  hardly moves.
- Net: −5 smeared over whichever arbitrary state the agent happened to die in, once per
  episode. That is close to a uniform downward offset on Q, and uniform offsets do not
  change argmaxes.

Concretely:

1. **`suicides` stays high — I predict above 0.30**, against 0.427 in E02. A drop to near
   zero would refute the whole argument above, and I would want to understand why before
   trusting it.
2. **`steps` and `coins` therefore change little.** If `coins` jumps to ~21 (the E02
   extrapolation), the mechanism reasoning is wrong.
3. **All Q-values shift downward roughly uniformly.** Visible by comparing row means
   against the v2 table. That is the offset, not learning.
4. **Row 93 stays flat.** Whether `BOMB` remains its argmax is a coin flip — its margin
   was 0.02, i.e. noise. Either outcome is consistent with the prediction; what matters
   is that the *spread* in that row stays small, because no information was added.
5. `invalid` stays around 1.85 — untouched by this change.

**If the prediction holds**, the conclusion for the report is the general one:
*an agent cannot learn to avoid a hazard its state representation cannot see.* With
optimistic max-bootstrapping, credit for a delayed death cannot flow back through states
that look safe. That is an argument for a danger feature, not for a bigger penalty, and
task 2 needs one regardless.

**If it fails to hold**, the alternative deliberately not tried here is `BOMB_DROPPED: −X`,
which lands directly on `(s, BOMB)` and needs no propagation at all — but which makes
bombing unconditionally bad and would have to be undone from task 2 on.

### Result

| Metric | v2 | v3 | Paired difference | 95 % CI | Verdict |
|---|---|---|---|---|---|
| `coins` | 12.823 | **16.703** | +3.880 | [+2.457, +5.253] | BETTER |
| `steps` | 241.9 | **399.1** | +157.163 | [+137.120, +177.540] | BETTER |
| `invalid` | 1.85 | **0.08** | −1.770 | [−1.993, −1.550] | BETTER |
| `suicides` | 0.427 | **0.000** | −0.427 | [−0.480, −0.373] | BETTER |
| `survived` | 0.573 | **1.000** | +0.427 | [+0.373, +0.480] | BETTER |
| rows with `BOMB` as argmax | 2 | **0** | — | — | — |

**Every prediction I made was wrong.** One reward constant fixed all four problems at once.

### Why the prediction failed

The mechanism argument assumed the agent *walks away* after dropping a bomb, so that the
death happens four steps later in a state aliased with safe ones, and the penalty cannot
propagate back through a max-bootstrap. The agent does not walk away.

In v2, `BOMB` was the argmax of row 93. Bombing does not change the wall bits and barely
changes the coin direction, so the agent is still in row 93 on the next step — and picks
`BOMB` again. It has no bomb left, so that is an `INVALID_ACTION`, which costs −1 but does
not move it. It repeats this until its own blast arrives. The evidence is unambiguous:

| | v2 |
|---|---|
| rounds that dropped a bomb | 128 of 300 |
| of those, rounds that died | **128 of 128** |
| mean `invalid` in those rounds | **4.11** ≈ `BOMB_TIMER` = 4 |
| `bombs` per round | 0.427 = `suicides` per round, exactly |

So the last action before death was `BOMB` itself, and `end_of_round` wrote the −5 straight
into `Q(93, BOMB)`. Direct credit assignment — no propagation required. Row 93 went from
a flat band of 0.85 to a spread of 3.04, `BOMB` fell 9.15 → 7.73, and `LEFT` took over at
10.34. No row in the table has `BOMB` as its argmax any more, and the agent drops zero
bombs in 300 evaluation rounds.

Also wrong in detail:

- **Prediction 3 (uniform offset).** The row-mean shift is −1.03 with a standard deviation
  of 0.80. Not uniform — the change is structured, which is exactly why it worked.
- **The 1.85 invalid actions in E02 were not mainly the `(0,0)`-direction rows.** They were
  the repeated `BOMB` attempts: 4.11 per bombing round × 0.427 bombing rounds ≈ 1.76 of the
  1.85. Rows 85 and 112 account for the 0.08 that remains in v3 — my E02 diagnosis had the
  right rows but the wrong order of magnitude.
- **`coins` reached 16.703, not the ~21 I extrapolated.** The extrapolation assumed a
  constant collection rate. It is not constant: v2 managed 0.053 coins/step over 242 steps,
  v3 only 0.042 over 399. The more coins are collected, the further away the next one is.
  Linear extrapolation over a round is optimistic and I should stop using it.

### What survives of the argument

The general claim is still true, but it applies to a case this experiment did not contain:

> With optimistic max-bootstrapping, credit for a *delayed* hazard cannot flow back through
> states the feature map cannot distinguish from safe ones.

A policy that stands still and repeats the fatal action turns a delayed hazard into an
immediate one, and then a plain terminal penalty is enough. That is what happened here.
**On task 2 it will not happen**, because there the agent must bomb a crate and then
genuinely run away — at which point the death really is four steps and several tiles
removed from its cause, the intervening states really are aliased with safe ones, and the
propagation argument applies as written. So a danger feature is still required; E03 simply
did not test it.

Lesson worth keeping: I reasoned about the mechanism without checking what the policy
actually does. `bombs` = `suicides` = 0.427 and `invalid` = 4.11 per bombing round were in
the E02 CSV the whole time and would have refuted the prediction before the run.

### Verdict

**BETTER on every metric, and the strongest single change so far.** Task 1 is effectively
solved for this feature set: the agent survives all 300 rounds, wastes almost no actions
(0.08 invalid), and collects 16.7 of 50 coins.

### What I do next

1. **E04: the direction encoding.** With deaths gone, the only remaining limit on task 1 is
   navigation quality — the aliased states where `sgn(Δx) = 0` and the wanted direction is
   blocked. Candidate: four bits for "does this neighbour reduce the BFS distance to the
   nearest coin". Still a learned choice rather than a returned action.
2. **Re-check `steps` = 399.1, not 400.0.** With zero deaths it should be exactly 400.
   Small, but E01 and E02 both had exactly-400 rounds, so the 0.9 wants an explanation
   rather than a shrug.
3. **The reward balance is now two constants, coin +5 against death −5.** The ablation
   against the game's real +1 (open since E01) has become more interesting, not less:
   the ratio is what matters, and neither number is tuned.
4. **Task 2 needs the danger feature.** Noted above — do not read E03 as evidence that a
   penalty alone is enough there.

---

## E02 — Münzrichtung im Zustand

- **Frage:** E01 hat gezeigt, dass ein ortsblinder Agent 1,35 von 50 Münzen holt und in
  einer von vier Startecken sogar stehen bleibt. Reicht **eine** zusätzliche Merkmals-
  komponente — die grobe Richtung zur nächsten Münze — damit daraus Navigation wird?
- **Änderung ggü. E01:** genau eine. `FEATURE_SIZES` wird von `(2,2,2,2)` zu
  `(2,2,2,2,3,3)`; die beiden neuen Ziffern sind `sgn(Δx)` und `sgn(Δy)` zur nächsten
  Münze, auf `{0,1,2}` verschoben. **Alles andere bleibt gleich:** α = 0,1 · γ = 0,9 ·
  ε = 0,2 · Münze +5 · `INVALID_ACTION` −1 · `WAITED` −0,1 · Schrittkosten −0,1.
- **Agent:** `benedict_coin_collector` v2 · Code-Commit `8ea6204`
  (Der `.meta.json`-Stempel des Messlaufs lautet `8ea6204-dirty`: Das Training schreibt
  `q_table.npy` neu, und die Datei ist versioniert. Ab E03 erst das trainierte Modell
  committen, dann messen — sonst ist jeder Stempel nach einem Training „dirty".)
  - Q-Tabelle 144 × 6
  - „Nächste" Münze über **Manhattan-Distanz**, nicht über den tatsächlichen Weg.
    Bewusst so: Auf `coin-heaven` sind die einzigen Hindernisse die festen Säulen auf
    (gerade, gerade), der Umweg ist also ein bis zwei Schritte und kippt die Rangfolge
    selten. Damit bleiben *Distanzmaß* und *Richtungskodierung* zwei unabhängig
    austauschbare Dinge — „Manhattan vs. BFS" wird so später eine saubere Ablation
    auf unverändertem Merkmalslayout (Kandidat für E03).
  - Sonderfall „keine Münze mehr im Spiel": Richtung `(0,0)` → Ziffern `(1,1)`. Diese
    Kombination ist sonst unerreichbar, weil eine Münze auf dem eigenen Feld beim
    Betreten eingesammelt wird und im nächsten Zustand nicht mehr in `coins` steht.
    Vor dem Lauf per Assert auf dem Testbrett geprüft, nicht angenommen.
- **Training:** 5000 Runden (statt 1000), `coin-heaven`, keine Gegner, `run = q_v2_task1`.
  Begründung: 67 erreichbare Zeilen × 6 Aktionen = 402 lebende Zellen, ~42 Schritte pro
  Episode, Abdeckung ~0,2 bei ε = 0,2, also 50 · 402 / (42 · 0,2) ≈ 2400 Runden. Ich nehme
  bewusst 5000: Der Lauf dauert unter einer Minute, und die Reserve geht an die selten
  besuchten Randzeilen — genau die, die in E01 die Warte-Falle erzeugt haben. Mit den
  1000 Runden aus E01 würde ich „zu wenig Daten" messen und es für „Merkmal hilft nicht"
  halten.
- **Messung:** `results/eval/benedict_q_v2__task1.csv` · coin-heaven · 300 Runden ·
  Seed 20260731 · ε = 0. Gepaart gegen `benedict_q_v1__task1.csv`.

### Vorhersage (vor dem Lauf notiert)

1. **Die 70 Runden mit `moves` = 0 verschwinden.** Das ist die schärfste Vorhersage.
   In der Ecke unten rechts zeigt die Münzrichtung jetzt nach oben-links, `WAIT` und `UP`
   sind damit nicht mehr ununterscheidbar. Bleiben die 70 Runden bestehen, ist das
   Merkmal nicht in der Politik angekommen — dann ist es ein **Bug**, keine schwache
   Idee, und ich suche im Code statt an den Hyperparametern.
2. **`coins` deutlich zweistellig, geschätzt 15–25 von 50.** Ich lege mich absichtlich
   auf eine Zahl fest. Danebenliegen ist informativ; sich nicht festlegen nicht.
3. **`invalid` bleibt 0,00.** Das ist hier ein Regressionswächter: Das neue Merkmal
   spaltet jedes Wandmuster in neun Zeilen auf, jede bekommt also ein Neuntel der Daten.
   Steigt `invalid`, heißt das „zu wenig trainiert", nicht „kaputt".
4. **67 der 144 Zeilen werden belegt sein.** Dieselbe Art Test wie die 11 von 16 in E01:
   vorher abgezählt, hinterher nachgeschaut.
   *Korrektur vor dem Lauf:* Mein erster Wert war 11 × 9 = 99 und war falsch. Er
   unterstellt, dass Wandmuster und Münzrichtung unabhängig sind — das sind sie nicht.
   Das Wandmuster eines Feldes ist eine Funktion seiner Position, und die Position
   schränkt die möglichen Münzrichtungen ein: Auf dem Feld (1,1) sind oben und links
   Wand, und jede Münze liegt zwangsläufig bei x ≥ 1, y ≥ 1 — negative Vorzeichen kommen
   dort nie vor. Rand- und Eckmuster verlieren so den Großteil ihrer neun Richtungen,
   nur das offene Innenmuster behält alle acht. Ausgezählt über alle 176 × 176 Paare
   (Feld, Münze): **56 Zeilen aus echten Münzrichtungen + 11 für „keine Münze mehr" = 67**.
   Die Zählung ist exakt, nicht geschätzt: Die nächste Münze liegt immer auf *irgendeinem*
   freien Feld, also deckt der Durchlauf über alle Einzelmünzen jede erreichbare Richtung
   ab und keine darüber hinaus.
   Dass ich die Zahl *vor* der Messung korrigiere, ist kein Verschieben des Zielpfostens:
   Es ist eine Eigenschaft des Codes und der Arena, nachprüfbar ohne einen einzigen
   Trainingsschritt. Was geprüft wird, bleibt dasselbe — ob die trainierte Tabelle
   genau die Zeilen belegt, die sie belegen kann.
   Beim selben Durchlauf mitgeprüft: Die Richtung `(0,0)` tritt bei nicht-leerer
   Münzliste **null Mal** auf. Die Doppelbelegung für „keine Münze mehr" ist damit
   belegt und nicht angenommen.
5. **`KILLED_SELF` bleibt im Training bei ~1,0 pro Episode.** An den Bomben hat sich
   nichts geändert, ε legt weiterhin alle ~30 Schritte eine. Steigt oder fällt das
   deutlich, hat die Merkmalsänderung etwas beeinflusst, das sie nicht beeinflussen sollte.
6. **Vorhergesagter Verlustkanal: Vorzeichen-Merkmale sind an Säulen unterbestimmt.**
   Steht der Agent auf einem Feld mit gerader x-Koordinate und liegt die Münze genau
   über ihm, sagt das Merkmal „hoch, keine x-Präferenz" — während `UP` von der Säule
   blockiert ist. Die Tabelle muss sich dann blind für links oder rechts entscheiden und
   wird sich auf eine Seite festlegen. Das ist genau das Argument für die BFS-Richtung
   als nächstes Experiment, und ich will es als Zahl sehen, bevor ich es behebe.

### Result

*(From here on this ledger is written in English.)*

Paired over 300 identical arenas, `results/eval/benedict_q_v2__task1.csv`:

| Metric | v1 | v2 | Paired difference | 95 % CI | Verdict |
|---|---|---|---|---|---|
| `coins` | 1.353 | **12.823** | **+11.470** | [+10.280, +12.707] | BETTER |
| `steps` | 400.0 | 241.9 | −158.060 | [−178.353, −137.620] | WORSE |
| `invalid` | 0.00 | 1.85 | +1.853 | [+1.633, +2.077] | WORSE |
| rounds with `moves` = 0 | 70 | **0** | −70 | — | — |
| `suicides` | 0.000 | 0.427 | +0.427 | [0.373, 0.480] | WORSE |
| occupied rows | 11 / 11 | 66 / 67 | — | — | — |

**Coins up by a factor of 9.5, CI nowhere near zero.** On the primary task-1 metric the
feature is a demonstrated improvement. The two WORSE rows are not a regression of
something v1 did well — they are new failure modes that only became *possible* once the
agent started moving. v1 scored 400 steps and 0 invalid actions by standing still or
walking in a fixed cycle, which is a degenerate way to look perfect.

### Predictions, scored

1. **The 70 zero-move rounds vanish — confirmed exactly, 70 → 0.** The `WAIT` trap was
   a symptom of the missing feature, as argued, and it disappeared without being patched.
   Deciding in advance *not* to fix it directly is what makes this a measurement.
2. **`coins` 15–25 — nominally wrong (12.8), in substance right.** 12.823 coins in
   241.9 steps is 0.053 coins/step; over a full 400-step round that is ≈ 21 coins, inside
   the predicted band. The navigation estimate was fine. What I failed to predict is that
   the agent would be dead for 40 % of the round. The prediction was not too optimistic
   about steering — it was blind to a second failure mode.
3. **`invalid` stays at 0.00 — wrong.** 1.85 per round. See the diagnosis below; the
   cause is not undertraining, which is what I had assumed the risk was.
4. **67 occupied rows — 66.** Row 13 (walls (0,0,0,1), direction (0,0)) was never entered
   in 5000 training episodes. Everything visited was reachable; nothing unreachable was
   visited. Off by one, and the one is explained.
5. **`KILLED_SELF` ≈ 1.0 per training episode — held.**
6. **Sign features under-specify at pillars — confirmed, and worse than predicted.**
   I expected wasted steps. It causes deaths. See below.

### Diagnosis: where the two WORSE columns come from

**The 42.7 % suicide rate is one row.**

```
row 93  walls(U,R,D,L)=(1,0,1,0)  dir=(0,-1)   q = [8.42 8.63 8.85 8.30 9.13 9.15]
                                                    UP  RIGHT DOWN LEFT WAIT BOMB
```

An east-west corridor with the coin straight overhead. The feature says "up", `UP` is a
wall, and `sgn(dx) = 0` expresses no left/right preference — so the agent has literally
no information about which way to go around. All six values lie inside a 0.85 band: the
state is aliased, no action is reliably better, and `BOMB` won the tie by **0.02**. This
is exactly prediction 6, but the consequence is death rather than a detour, because
nothing in the reward function says that dying is bad. The only pressure against `BOMB`
is structural — the episode ends, so the terminal update carries no bootstrap term.
0.02 is all that structural pressure has left after the coin reward inflates the row.

**The 1.85 invalid actions are the `(0,0)`-direction rows**, 85 and 112, both with `UP`
as argmax while `UP` is blocked. Rarely visited, so noise decides — not undertraining in
the sense I predicted, but a genuinely information-free state.

**Correction to my own pre-run check.** In the prediction I recorded that direction
`(0,0)` cannot occur with a non-empty coin list, and called that "verified, not assumed".
The verification was circular: the enumeration filtered with `c != p`, which excludes the
one case at issue — a coin sitting on the agent's own tile. Re-run without the filter, all
11 `(0,0)` rows are reachable **both** via a real coin and via an empty coin list. So the
digit pair `(1,1)` is a **collision**, not a free slot. Practical harm is small (both cases
mean "no direction information") and the row count is unaffected (both routes reach the
same rows, so 67 stands). But it is an ambiguity in the feature map, it is now documented,
and it is a candidate cause if those rows keep misbehaving.
Lesson worth keeping: a check that filters out the case it is meant to test always passes.

### Verdict

**BETTER on the primary metric, and the feature works as designed.** Coins ×9.5, the
`WAIT` trap gone, the state space populated as counted. The agent navigates.

Both regressions trace to states in which the sign feature carries no usable information,
and they split cleanly into two independent causes:

- **a reward problem** — nothing penalises `KILLED_SELF`, so in a flat row `BOMB` is only
  0.02 away from winning;
- **a feature problem** — `sgn(dx) = 0` with the wanted direction blocked is an
  information-free state by construction.

### What I do next

1. **E03: penalise `KILLED_SELF`.** One constant, no feature change — the cheapest possible
   controlled experiment, and it targets the larger of the two effects directly. Prediction
   to write beforehand: `suicides` collapses toward 0, `steps` returns toward 400, and
   `coins` lands near 21 (the extrapolation above). If `coins` does *not* reach ~21, my
   model of what is limiting the agent is wrong.
2. **E04: better direction encoding.** Only after E03, so the two causes stay separable.
   The obvious candidate is a BFS first step, but that is close to "a feature that returns
   the best action", which the task description forbids — on `coin-heaven` it would
   essentially *be* the optimal policy. Better options to weigh: which of the four
   neighbours reduces the BFS distance (4 bits, still a learned choice), or keeping the
   sign feature and adding the tie-break information it is missing.
3. **Watch row 13.** Reachable but never visited. Harmless now; worth rechecking after
   E03 changes how long episodes last.
4. **Still open from E01:** ablate `COIN_COLLECTED` +5 against the game's actual +1.
   Now that there is behaviour for the reward to act on, this has become meaningful —
   and after E03 there will be a second reward constant whose balance against it matters.

---

## E01 — Baseline: nur Wandbits

- **Frage:** Läuft die Messkette von Ende zu Ende, und wie weit kommt ein Agent,
  dessen Zustand *keine* Münzinformation enthält? Der Eintrag ist kein Versuch,
  gut zu spielen — er legt den Bezugspunkt fest, gegen den alles Weitere gemessen wird.
- **Änderung ggü. vorher:** — (Ausgangspunkt)
- **Agent:** `benedict_coin_collector` v1 · Commit `ef030a8`
  - Merkmale: 4 Bits „Nachbarfeld blockiert?" (U/R/D/L) → Q-Tabelle 16 × 6,
    davon 11 Zeilen erreichbar
- **Training:** 1000 Runden, `coin-heaven`, keine Gegner
  - α = 0,1 · γ = 0,9 · ε = 0,2 (konstant)
  - Rewards: Münze +5 · `INVALID_ACTION` −1 · `WAITED` −0,1 · Schrittkosten −0,1
  - `KILLED_SELF` ist **nicht** in der Reward-Tabelle. Der Druck gegen `BOMB` entsteht
    allein daraus, dass die Episode endet und im Terminal-Update kein γ·max Q steht.
- **Messung:** `results/eval/benedict_q_v1__task1.csv` · coin-heaven · 300 Runden ·
  Seed 20260731 · ε = 0 (nicht im Trainingsmodus)

### Vorhersage (vor dem Lauf notiert)

1. **`invalid` ≈ 0.** Das ist das eigentliche Bestehenskriterium dieser Stufe. Die
   Wandbits stehen genau dafür im Zustand; sind die ungültigen Aktionen nicht nahe null,
   ist die Pipeline noch kaputt und alles Weitere wertlos.
2. **`steps` = 400, also die volle Runde.** Mit ε = 0 ist die Politik das Argmax der
   Tabelle, und `BOMB` war in keiner Zeile die beste Aktion. Der Agent legt also nie
   eine Bombe und kann folglich nicht sterben. (Im *Training* ist das anders: dort
   sprengt ε sich regelmäßig selbst, die Episoden brechen früh ab.)
3. **`coins` einstellig.** Die Politik ist deterministisch *und* ortsblind: gleiches
   Wandmuster → immer derselbe Zug. Damit ist die Trajektorie letztlich periodisch —
   der Agent läuft in einen kurzen Zyklus und besucht für den Rest der 400 Schritte
   dieselben paar Felder. Die Münzen, die er bekommt, sammelt er im Wesentlichen
   zufällig auf dem Weg in diesen Zyklus ein.
4. Fällt `coins` deutlich höher aus, liegt der Fehler in meinem Verständnis des
   Aufbaus, nicht im Agenten. Dann nachsehen, nicht freuen.

### Ergebnis

| Metrik | Mittelwert | 95-%-KI |
|---|---|---|
| `coins` | 1,353 | [1,153 · 1,553] |
| `steps` | 400,0 | [400,0 · 400,0] |
| `invalid` | 0,00 | [0,00 · 0,00] |
| `suicides` | 0,000 | [0,000 · 0,000] |
| `survived` | 1,000 | [1,000 · 1,000] |
| `think_max_ms` | 0,0 | — |

Alle drei Vorhersagen bestätigt, und zwar exakt: die KIs von `steps` und `invalid` sind
entartet, weil *jede einzelne* der 300 Runden 400 Schritte lief und *keine* ungültige
Aktion enthielt. Die Q-Tabelle erfüllt in allen 10 erreichbaren Zeilen mit blockierten
Richtungen das Kriterium „blockierte Richtung < jeder legale Zug".

**Reproduzierbarkeit geprüft:** Der Lauf wurde zweimal ausgeführt (einmal auf schmutzigem
Baum, einmal sauber unter `ef030a8`). Alle Spalten außer `time`, `think_mean_ms` und
`think_max_ms` sind zeilenweise identisch — 0 von 300 Runden weichen ab. Bei ε = 0 ist die
Politik das Argmax einer festen Tabelle, und der Seed legt die Arenen fest; nur die
Laufzeitmessung streut. Ein Messlauf ist also aus Commit + Seed exakt wiederherstellbar.
Für das *Training* gilt das ausdrücklich **nicht** — dessen RNG ist ungeseedet.

### Der eigentliche Befund: die Zugzahl ist bimodal

`moves` ist nicht gestreut, sondern hat genau zwei Werte:

| `moves` | Runden | mittlere Münzen |
|---|---|---|
| 0 | **70** (23 %) | 0,26 |
| 400 | 230 (77 %) | 1,69 |

In 70 Runden bewegt der Agent sich **kein einziges Mal**. Die Ursache steht in der
Tabelle: Der Agent startet in einer der vier Ecken. Für die Ecke unten rechts (15,15)
sind rechts und unten Außenwand, das Merkmal ist also (U,R,D,L) = (0,1,1,0) = Zeile 6 —
und dort ist das Argmax `WAIT` (2,636) und nicht `UP` (2,455). Mit ε = 0 ist die Politik
deterministisch, das Feld ändert sich durch Warten nicht, also bleibt der Agent bis
Schritt 400 stehen. 70/300 ≈ ¼ passt genau zu „eine von vier Startecken".

Das ist kein Bug in der Implementierung, sondern die Zustandsabstraktion, die genau das
tut, was sie soll: Zeile 6 hat wenige Felder und wird selten besucht, die Schätzung ist
verrauscht, und `WAIT` liegt zufällig 0,2 über `UP`. Für ein ortsblindes Merkmal *sind*
Warten und Laufen ununterscheidbar — ohne Münzinformation gibt es keinen Grund,
das eine dem anderen vorzuziehen. Die Schrittkosten −0,1 gelten für beide gleich.

### Trainingsverlauf

`results/train/benedict_coin_collector__q_v1_task1.csv`, Kurven in `results/figures/`.

| | erste 100 Episoden | letzte 100 |
|---|---|---|
| `steps` | 39,6 | 42,1 |
| `KILLED_SELF` | 1,00 | 1,00 |
| `COIN_COLLECTED` | 3,07 | 4,08 |
| `td_error` | 1,05 | 1,33 |

Zwei Dinge, die ich ohne das Log nicht gesehen hätte:

- **Jede Trainingsepisode endet im Selbstmord.** ε = 0,2 heißt 3,3 % Chance auf `BOMB`
  pro Schritt, also im Mittel nach ~30 Schritten eine Bombe — und der Agent hat kein
  Merkmal, das ihm sagt, wo sie liegt. 1000 Trainingsrunden sind damit nur ~42 000
  Schritte, nicht 400 000. Beim *Messen* passiert das nicht, weil `BOMB` in keiner Zeile
  Argmax ist und ε = 0 gilt — daher `suicides` = 0. Der Unterschied zwischen Trainings-
  und Messverhalten ist hier extrem, und ohne die Kurve hätte ich ihn übersehen.
- **`td_error` fällt nicht, er steigt leicht.** Als Konvergenzmaß taugt der *absolute*
  TD-Fehler hier nicht: er wächst mit den Q-Werten mit, und die Belohnung ist mit ±5
  von Natur aus stark verrauscht. Für die nächste Version relativ messen oder auf
  die Änderung der Q-Tabelle zwischen Episoden umstellen.

### Urteil

**Messkette verifiziert.** `invalid` = 0 exakt, `steps` = 400 exakt, Training läuft,
Lernkurve wird geschrieben, Auswertung liefert KIs. Der Bezugspunkt steht:
**1,353 Münzen von 50**. Das ist die Zahl, die die Münzrichtung schlagen muss.

### Was ich daraus mache

1. **Nächstes Experiment (E02): Münzrichtung ins Merkmal.** Vorzeichen des Offsets zur
   nächsten Münze, 3 × 3 Werte → `|Ŝ|` = 16 × 9 = 144 Zeilen. Das ist die eine Änderung.
2. **Die Warte-Falle nicht separat reparieren.** Es ist verlockend, `WAITED` härter zu
   bestrafen oder `WAIT` aus dem Aktionsraum zu nehmen. Beides würde das Symptom
   verdecken: Sobald der Zustand eine Münzrichtung enthält, ist Laufen *nachweislich*
   besser als Warten, und die Falle verschwindet von selbst. Wenn sie das nicht tut,
   ist das eine interessante Information, die ich mir nicht wegpatchen will.
   → Für den Bericht ist der Vorher/Nachher-Vergleich von `moves` die schönere Abbildung.
3. **Trainingsrunden erhöhen.** 144 statt 16 Zeilen bei ~42 Schritten pro Episode:
   nach der Stichprobenregel (~50 Besuche je Zelle, Abdeckung ~0,2) sind das
   50 · 144 · 6 / (42 · 0,2) ≈ 5000 Runden. Sonst messe ich „zu wenig Daten" und halte
   es für „Merkmal hilft nicht".
4. Offen für später: `COIN_COLLECTED` +5 gegen die tatsächlichen +1 des Spiels ablatieren.
   Jetzt noch nicht — erst muss es überhaupt etwas geben, worauf die Belohnung wirkt.
