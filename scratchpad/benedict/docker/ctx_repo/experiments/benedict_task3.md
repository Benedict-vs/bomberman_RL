# Task 3 — `classic` with opponents: what was tried, and why the agent did not change

Consolidated account of rung 3, from `experiments/benedict.md` **E24–E27**. Short on purpose:
**the shipped table is byte-identical to the rung-2 table.** Rung 3 changed one *feature* and
nothing else, and the interesting content is the four training attempts that failed and the
reason they failed.

Audited twice by independent sessions given the raw data and told to form their own numbers
before reading the ledger (§5). Both changed verdicts. That is the most transferable thing here.

---

## 1 · What ships

`agent_code/benedict_task3/` — the rung-2 Q-table, unchanged, read through one changed digit.

| `coin_collector` field, 1000 rounds, ship seed 990731 | HUNT off | **HUNT on** |
|---|---|---|
| score | 2.593 | **4.188** |
| kills | 0.088 | **0.420** |
| invalid | 24.21 | **1.43** |
| won | 0.228 | **0.367** |
| think_max_ms | 0.252 | **0.321** |

**Paired over 1000 identical arenas: +1.595 [+1.342, +1.849]** (E26). It clears
`rule_based_agent`'s 2.753 in the same slot; against `peaceful_agent` it scores 17.93 and wins
**100 %** of rounds; against `rule_based_agent` 2.757 with `won` 0.170.

**The change, in one sentence:** digit 6 (the BFS objective) falls through to the nearest
*opponent* when no coin and no crate is reachable, and digit 7 counts an opponent in blast range
as a reason to bomb. `FEATURE_SIZES` is untouched, so the rung-2 table remains a valid parent.

---

## 2 · Why that one digit was worth 1.6 score

Not because the agent got better at anything. Coins and crates are **unchanged** (2.15 → 2.09,
27.2 → 26.2). The entire gain is stopping a leak:

- Four agents strip all 122 crates by **~step 140**. For the remaining two thirds of the round
  the rung-2 state had **no objective at all** — digit 6 = `NO_TARGET` on **26.3 %** of safe
  steps in the `coin_collector` field, against 0.43 % solo.
- The table's learned answer in that row is `BOMB`, which it usually cannot do. **`INVALID_ACTION`
  cost −24.25 per round — more than coins and crates earned together.** 96.3 % of it was `BOMB`
  with no bomb available, and **99 % of it lived in exactly two Q-rows**.
- Giving those rows an objective removed the leak (24.21 → 1.43) and the freed steps became
  kills (0.088 → 0.420).

This sat in E24's own results table, one row below `crates`, for two entries before an audit
priced it.

---

## 3 · The four training attempts, and the one cause

| entry | what was trained | score (cc field) |
|---|---|---|
| — | frozen rung-2 table + HUNT (**the ship**) | **4.160** |
| E25 A | rung-2 rewards, no feature | 0.081 |
| E25 B | + `GOT_KILLED` (mispriced, see §5) | 0.118 |
| E26 S | corrected death semantics | 0.177 |
| E26 H | + the working feature + `KILLED_OPPONENT` = 25 | 0.829 |
| E27 C10 | + `CRATE_DESTROYED` 0.3 → 1.0 | 1.777 |
| E27 S03 | + `STEP_COST` −0.1 → −0.03 | 2.986 [0.36, 5.62] |

**Twenty-five training runs, four reward configurations, two feature maps, five seeds each — not
one beat the untrained table.**

**The cause (E27, found by audit).** The value function's dynamic range is gross earnings against
an **action-independent** step cost:

| | earnings | costs | earn/cost |
|---|---|---|---|
| rung 2, solo | +77.3 | −40.1 | **1.93** |
| rung 3, *same reward table* | +18.3 | −26.1 | **0.70** |
| rung 3, crate = 1.0 | +36.7 | −26.1 | **1.41** |

Nine coins shared four ways cuts earnings ~4×; steps alive fall only 1.85×. Once a constant
−0.1/step dominates the return, the fixed point is action-independent too — and the decision
margin collapses. Measured medians: **0.0003** (rung-2 rewards) → 0.005–0.06 (crate 1.0) →
**0.639** (the frozen table). E27 confirmed the mechanism decisively — paired, raising only the
crate reward is worth **+0.93 score and +11.5 crates** — and still finished 1.2 points short.

**Conclusion:** the reward table was calibrated on a board the agent had to itself. Fixing the
scale is necessary and not sufficient. Rung 3 ships untrained.

---

## 4 · What did *not* explain it

Each of these was proposed as the cause, and each is refuted with a number:

| hypothesis | refutation |
|---|---|
| state coverage — deaths land in never-visited rows | true (66 % vs 1.4 % base rate) but E25 drove it to 0.00 % and the agent got **20× worse** |
| ε = 0.2 erases the warm start in ~1 000 episodes | ε = 0.2 kills the frozen table in 44 steps **solo**; crates/step *rises* to 0.25 through episode 1 000 |
| the crate→coin chain is starved by opponents | `COIN_FOUND` 2.08 vs `COIN_COLLECTED` 2.23 — it collects *more* than its own bombs reveal |
| the agent is behaving rationally | bombing is **+0.67 EV**; E25's policy is worth −11.19 under its own reward table vs −0.589 for its initialisation — worse than standing still |
| α = 1/N^0.7 anneals cells shut | α = 1/N^0.55 is *worse* (0.987 vs 1.393) |
| the warm start is poison | cold start 1.013 vs warm 1.393, identical margin collapse |
| the double terminal update in `train.py` | real bug, fires in **100 %** of rung-2 episodes and 6–38 % of rung-3 ones — wrong direction |

---

## 5 · Method failures worth more than the agent

1. **A pre-registered metric set must name which number decides.** E25 got **four of six
   predictions "correct"** on an agent 20× worse on `score` — every one a *guard* metric, and two
   were satisfied by the agent abandoning the behaviour they existed to protect.
2. **`GOT_KILLED` means "died", not "killed by an opponent"** (`environment.py:264`; `:251` adds
   `KILLED_SELF` *on top*). An intended −5/−5 actually paid −10/−5 and voided E25's arm contrast.
3. **A training log and an ε = 0 evaluation can disagree completely.** E26 arm H logged 18.8
   crates/episode, flat, while evaluating at 3.74.
4. **Part of that gap is a readout artefact.** `TIE_TOL = 0.001` on the same frozen table recovers
   3.74 → **19.11** crates; the ordering survives below the resolution of a deterministic argmax.
   A diagnostic, never a fix — it triples a broken table and does nothing for a healthy one.
5. **Evaluations are not round-reproducible from rung 3 on.** `coin_collector_agent` and
   `rule_based_agent` shuffle with the **stdlib** `random`, which `np.random.seed()` does not
   touch: 22.7 % of rounds repeat, means repeat to four decimals. Rung-3 pairing is on arenas
   only. Documented as fully fixed for two entries before an audit caught it.
6. **A mean of five can be one seed.** E27's S03 mean 2.986 is 6.707, 2.673, 1.630, 1.783, 2.137.
7. **Is it still machine learning?** The strongest purely-feature policy the same digits allow —
   escape, bomb if the digit fires, else walk digit 6 — **dies in every round** and scores 0.030
   against the learned 4.377. The features are necessary and nowhere near sufficient; what the
   table contributes is *when not to bomb*.

---

## 6 · Carried into task 4

1. **Start from the frozen table**, not from a rung-3 training run. Rung-4 reference: arm F scores
   2.757 with `won` 0.170 against 3× `rule_based_agent`.
2. **Untested hypothesis, pre-register before claiming.** E27's S03 beat the frozen table on
   `won` against `rule_based_agent` on **all five seeds** (0.347/0.277/0.213/0.217/0.297 vs 0.167)
   at equal score. `won` is not what E27 pre-registered, so this is an idea, not a result — and
   `MEASUREMENT.md` says `won` matters more than mean score on rung 4.
3. **Two known bugs, both deferred deliberately**, each needing its own entry with baselines
   re-measured: the double terminal update (`end_of_round` replays an uncleared event list, so a
   surviving round updates its terminal cell twice and drops the bootstrap on what is a
   *truncation*, not a terminal state), and the stdlib-RNG seeding in `tools/evaluate.py`.
4. **Split the deaths on rung 4** — `suicides` (own bomb) vs `killed_by` (opponent's). The ship is
   at 0.377 / 0.553 against `rule_based_agent`, i.e. it dies to itself almost as often as to them.

---

## 7 · Reproduction

```bash
# the shipped table IS the rung-2 table -- rebuild it there, not here
cd agent_code/benedict_task2 && see experiments/benedict_task2.md §9

# the rung-3 agent = that table + the feature, which is the default
uv run python tools/evaluate.py --agents benedict_task3 --opponents coin_collector \
    --n-rounds 1000 --seed 990731 --label benedict_q_e26_F_frozen__task3_cc_ship990731 \
    --out-dir results/eval/task3_opponents

# BM_HUNT=0 reproduces the pre-E26 map, required for any rung-2 lineage table
```

Evidence: `results/eval/task3_opponents/` (committed), `scratchpad/audit/`, `scratchpad/audit2/`,
`scratchpad/benedict/`. Training logs are reproducible only up to opponent RNG (§5.5) and are
not committed.
