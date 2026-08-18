# Task B — should the agent hunt? The framing argument, before any design

Every number here comes from a probe in this directory over committed data or a greedy rollout of
the shipped table. Nothing was trained.

---

## 0 · The claim

**Hunting is the wrong target, but so is survival, and so is `won`.** The round is two different
games and only the first one pays. `won` is a linear function of `score` at **+0.088 per point**,
`score` is a linear function of `crates` at **≈ +0.14 coins per crate**, and the whole crate/coin
economy is consumed by **step ~200 of 400**. Kills are worth more per unit than anything else in
the game — and there are almost none available to take.

---

## 1 · The round is two games, and the second one is empty

`round_economy.py`, 150 rounds, shipped table vs 3 × `rule_based_agent`. Columns are cumulative
means over rounds still running at that step:

| step | our coins | opp coins | coins on board | crates left | our kills | opp kills |
|---|---|---|---|---|---|---|
| 40 | 0.873 | 2.013 | 1.17 | 68.6 | 0.000 | 0.000 |
| 100 | 2.107 | 5.007 | 0.32 | 23.2 | 0.100 | 0.160 |
| **200** | **2.709** | **6.054** | **0.04** | **3.4** | **0.203** | 0.399 |
| 300 | 2.837 | 6.114 | 0.00 | 0.8 | 0.293 | 0.528 |
| 400 | 2.857 | 6.133 | 0.00 | 0.2 | 0.276 | 0.571 |

- All 9 coins are gone by step ~220. All 122 crates by step ~250, and 97 % of them by step 200.
- **Our score at step 200 is 2.709 + 5 × 0.203 = 3.72, against a final 3.949 — ≈ 94 % of the score
  exists by the round's halfway point**, and part of the remaining 6 % is survivorship in the
  denominator (only 105 of 150 rounds are still running at step 400).
- **70 % of rounds run the full 400 steps** (`won_anatomy.py`). So the median round spends its
  second half on a stripped board with 1.3 opponents and nothing to collect.

### The consequence, and it closes an open item in `benedict_task4.md` §3

`phase_split.py` splits every rung-4 arm into **P1** (share of rounds we are alive at step 200) and
**P2** (share of those that reach 400):

| arm | score | crates | **P1** | **P2** | survived | suicides |
|---|---|---|---|---|---|---|
| E33 ctl @20k | 3.719 | 32.13 | 0.641 | 0.520 | 0.333 | 0.616 |
| E33 F080 | 3.523 | 30.81 | 0.654 | **0.691** | 0.451 | 0.496 |
| **E36 OPP @20k** | 3.726 | 32.10 | 0.684 | **0.765** | **0.525** | 0.422 |
| E36 PLB @20k | 3.930 | 33.11 | 0.640 | 0.678 | 0.435 | 0.516 |
| **E37 PLB2 (ship)** | **3.949** | **33.55** | 0.631 | 0.742 | 0.464 | 0.488 |
| E37 ctl2 | 3.667 | 32.30 | 0.658 | 0.611 | 0.402 | 0.550 |
| `rule_based_agent` | 3.255 | 30.55 | 0.501 | 0.496 | 0.372 | 0.535 |

**E36's celebrated survival gain (0.333 → 0.525) is +0.043 on P1 and +0.245 on P2.** It bought a
quarter more of the empty half. E33 F080 is the same shape and *lost* 0.196 score. The ledger says
"survival is not the lever" and leaves the mechanism open; **this is the mechanism.** Six
interventions bought phase-2 survival, which is worth ≈ 0.15 coins and ≈ 0.07 kills.

Note also that we already survive phase 1 far better than `rule_based_agent` (0.63–0.68 vs 0.501),
so there was never much headroom there either.

### What *does* track score

Across those 8 arms (n = 8, so read as direction not magnitude):

| against `score` | corr |
|---|---|
| coins | **+0.948** |
| **crates** | **+0.937** |
| crates/bomb | +0.854 |
| kills | +0.606 |
| survived | +0.162 |
| **P1 (phase-1 survival)** | **−0.377** |
| **bombs dropped** | **−0.612** |

and `corr(coins, crates) = +0.996`. **Coins are not a separate currency from crates; they are a
linear readout of them.** `denial.py`: our crates 33.41 + opponents' 89.27 = **122.68 of 122** — the
crate pool is exactly, exclusively consumed. The board is a zero-sum race, and we take 27.4 % of it.

---

## 2 · `won` is score, at a measured exchange rate

`survival_value.py`, 2 000 rounds pooled, recounting rather than fitting: take each round's actual
margin against the *best* of three opponents and ask how many flip if our score were `d` higher.

| Δscore | resulting `won` | per point |
|---|---|---|
| baseline | 0.393 | — |
| +1 | 0.474 | **+0.088 [+0.076, +0.101]** |
| +2 | 0.568 | +0.090 |
| +5 | 0.832 | +0.088 |

Flat over the whole range. Two implications:

- **Stop treating `won` as a separate objective.** `benedict_task4.md` §6 is right that score
  carries the claim, and this puts a number on it. (§6's `won ≈ 0.113 × score` is a ratio of means;
  the marginal is 0.088, so §6's MDE arguments get slightly harder, not easier.)
- **The margin distribution is dense near zero**: 8.1 % of rounds are lost by exactly 1 point,
  17.2 % by ≤ 2, 42.9 % by ≤ 5. There is no cliff to exploit and no cheap tie-breaking trick.

---

## 3 · Kills are the most valuable event in the game and there are almost none to take

### The value side — kills are worth *more* than their 5 points

`denial.py`, restricted to rounds where we ourselves live past step 200 (so this is not our own
death driving it), by how many opponents are already dead at step 100:

| opponents gone by step 100 | n | our score | our coins | our crates | `won` | best opponent |
|---|---|---|---|---|---|---|
| 0 | 570 | 4.539 | 2.977 | 36.15 | 0.447 | 5.554 |
| 1 | 546 | 4.573 | 3.255 | 37.83 | 0.529 | 4.614 |
| 2 | 121 | 5.058 | 3.446 | 42.03 | **0.595** | 4.620 |

Regression form: **−0.256 of our coins per 100 opponent phase-1 steps**, so removing one opponent
at step 60 is worth ≈ +0.36 coins and ≈ +3 crates *on top of* the +5. And it strikes the order
statistic `won` is measured against from both sides. A kill at step ~100 is worth roughly
**+0.5 `won` in that round.** Nothing else in this game is worth that.

(Mostly-suicides drive this table, so it is not a clean instrument — but rule-based suicides are
exogenous to us, and the regression slope agrees with the categorical split.)

### The supply side — this is where it dies

`trap_opportunity.py`, 100 greedy rounds, 27 857 alive steps, 10 293 of them with a bomb available.
For the hypothetical bomb on our own tile, "trapped" means an opponent in its blast has **no
survivable escape**, evaluated with the same time-aware BFS the agent runs on itself:

| | count | of armed steps |
|---|---|---|
| opponent standing in the blast | 2 604 | **25.30 %** |
| **opponent in the blast *and trapped*** | **43** | **0.42 %** |
| trapped and safe for us | 41 | 0.40 % |
| crate in the blast | 2 795 | 27.15 % |
| an escape from our own bomb exists | 10 198 | **99.08 %** |

- **0.43 trapped-opponent steps per round; only 32 % of rounds contain even one**, and a trapped
  opponent stays trapped for several consecutive steps, so the number of distinct *chances* is
  lower still.
- **The greedy policy already picks `BOMB` in 51.2 % of them.**
- `BOMB_TIMER = 4` is the whole story: an opponent in a blast has four moves to walk out, and
  **98.3 % of them do have somewhere to go.**

**This refutes the obvious next experiment before it runs.** E35's diagnosis — digit 7 is one bit
shared between "opens a crate" and "catches an opponent", so the price cannot reach the decision —
is right about the mechanism and would lead straight to "split digit 7". But the opponent half of
that bit is **98.3 % noise**: it fires on a quarter of all armed steps and is lethal in 1.7 % of
them. Splitting it gives the learner a cleaner channel to a signal that is not there.

The bit that *would* carry signal is "a trapped opponent is in my blast", at 0.42 % of armed steps.
Ceiling if we took **every** currently-existing opportunity instead of half, and every one
converted: **+0.2 kills/round = +1.0 score = +0.088 `won`.** Realistically +0.05–0.10 kills, i.e.
**+0.25–0.50 score**, before paying for whatever bombs get diverted from crates.

**Caveat I cannot close cheaply, and it is the strongest counter-argument:** this bounds
*opportunistic* trapping only — a bomb at the tile we are standing on, right now. A real hunting
policy *creates* traps by approaching and cutting off exits over several steps. A one-step tabular
feature cannot express that, and I have not measured how many constructible traps exist. If you
want to fund one measurement to break my conclusion, that is the one.

Supporting context: `rule_based_agent` gets **0.184 kills/round** each; we get **0.226**. We are
already the best killer on the board, and our kills are overwhelmingly incidental to crate bombing
(74 % of them land before step 200, in the dense phase). And **C in the survey is the one source
that tried aggression: their "bomb on sight" variant stopped collecting coins and they shipped the
conservative one.**

---

## 4 · Answers to your four questions

**1. Is kill-hunting the right target?** No — and neither is outlasting. `won` is decided in the
first 200 steps by crate throughput. Kills are the most valuable *event* but the supply is ~0.4
opportunities/round of which we already take half. Hunting as a *behaviour* is a bad trade against
bombs diverted from crates; kills as a *by-product of bombing better* is the same lever as crates
and E37 already showed it moves.

**2. Minimum representational change for kills.** Splitting digit 7 into crate-bomb and
opponent-bomb: refuted (§3). The only bit worth encoding is *trapped* opponent, and there is a
zero-row home for it — redefine digit 7's opponent branch to require trapped, so
`digit7 = have_bomb AND (crate in blast OR trapped opponent in blast)`. That *removes* 25 % of
armed steps of noise from an existing bit and adds 0.4 % of signal, at zero new rows, keeping the
warm-start parent. It is cheap and honest, and its expected effect is small.

**3. Can it be pre-registered honestly?** Yes, but not on `won`. Measured planning SDs (`mde.py`,
between-seed within-arm at ep20000; run-level pairing buys nothing per §5.3, so these are the right
quantities). MDE = 2.8 · SD · √(2/n), 80 % power, α = 0.05 two-sided:

| metric | between-seed SD | MDE n=5 | **n=15** | n=25 |
|---|---|---|---|---|
| score | 0.249 | 0.440 | **0.254** | 0.197 |
| kills | 0.031 | 0.055 | **0.032** | 0.025 |
| `won` | 0.029 | 0.050 | **0.029** | 0.023 |
| crates | 0.988 | 1.749 | **1.010** | 0.782 |
| suicides | 0.067 | 0.119 | 0.069 | 0.053 |

A trapped-opponent experiment: **primary = kills** (expect +0.05–0.10 against MDE 0.032 — powered);
**gate = score** (expect +0.25–0.50 against MDE 0.254 — powered only at the top of the range);
**guard = crates must not fall** (MDE 1.01 crates ≈ 0.14 score, and a crate loss of that size eats
the entire kill gain — this is the real risk and it is C's measured failure mode). `won` must be a
reported secondary, never the primary; expected +0.02–0.04 against MDE 0.029 is exactly the §5.1
error repeated.

**Free 2× in power, before any of this.** That 0.249 score SD is dominated by one collapsed
training run (E33 F080's seed at 2.508 against its own 3.74–3.86). With collapsed runs screened out
the between-seed SD is ≈ 0.11 and the n = 15 MDE falls to ≈ 0.11. `benedict_task4.md` §5.4 already
found this once. **Pre-registering a collapse screen — a stated rule on the training log, fixed
before the arms are evaluated — costs nothing and roughly doubles what every remaining sweep can
read.** Do this before spending any of the 25.

**4. Better expected value than either.** Ranked:

1. **Target type as a digit** (`TASK_A_survey_vs_ours.md` §1.1). The safe-row objective is 39 %
   opponent / 37 % crate / 21 % coin and the table cannot tell them apart. It is the corpus's only
   clean drop-two-bits ablation (K), it acts on phase 1 where 94 % of the score lives, and
   `TASK_A` §0 removes the row-cost objection: our *effective* state count is 290, not 64 000.
2. **The collapse screen** (above). Free.
3. **Rung-4 reward recalibration.** E27 showed the reward table was calibrated on a solo board and
   that raising only the crate reward was worth **+0.93 score and +11.5 crates** on rung 3. Nobody
   has redone that arithmetic for rung 4, where `crates` is now the variable that tracks score
   (corr +0.937). Hyperparameter optimisation is explicitly graded.
4. **Crate count instead of a crate bit.** M's feature: how many crates a bomb here would open
   (0/1/2/3+) rather than one bit. `corr(score, bombs) = −0.612` and we drop 28.6 bombs/round to
   `rule_based`'s 20.0 at 1.168 crates/bomb; E37 and E38 both act on exactly this quantity in
   opposite directions. Costs one digit widening (2 → 4).
5. **The truncation bug.** 70 % of rounds hit `MAX_STEPS` and `end_of_round` treats that as
   termination with no bootstrap, worth ≈ 8.8 Q units — known since rung 3, never fixed in
   isolation, and it is a plausible candidate for E38's unexplained "why does more training degrade
   bomb siting".
6. Trapped-opponent bit. Cheap, honest, small.

---

## 5 · What would falsify the argument

- **§1**: a round where the economy stays open past step 200 — e.g. if the tournament field is
  weaker at crate clearing than `rule_based_agent`, the crate phase lengthens and phase-2 survival
  starts to pay. Our whole picture is calibrated on one opponent type (`benedict_task4.md` §6 says
  so). Measurable against `coin_collector` or `peaceful` fields with the probes already written.
- **§3**: a measurement of *constructible* traps (multi-step approach) rather than the
  standing-here traps I counted. If constructible traps are an order of magnitude more common, the
  supply argument collapses and hunting becomes the top lever.
- **§2**: the +0.088 conversion is measured on this field and would change against opponents with a
  different score distribution.
- **§4.1**: if a target-type arm moves nothing, the "aliasing" reading of the E gap is wrong and §2
  of `TASK_A` should fall back to the reward table or the curriculum.

---

## 6 · Benedict's hunt hypothesis, tested (added after the first round)

> "The agent almost always stays alive until all crates and coins are gone and there is at least
> one other agent on the board. At that point it is harder to get a kill than when there are still
> crates — it is so much easier to just outrun the bomb and hide. With crates it is easier to lock
> the opponent in a corner. So: while destroying crates and collecting coins, work one's way toward
> the nearest opponent, rather than only collecting until no direct path to an opponent is left."

Probes: `hunt_window.py` (60 rounds, sampled every 3 steps), `trap_persistence.py` (40 rounds,
every step). Both roll out the shipped table at ε = 0 against 3 × `rule_based_agent`.

### 6.1 · The premise holds

| | |
|---|---|
| rounds where all 122 crates are eventually cleared | 55 / 60 |
| median step at which that happens | **210** (quartiles 176–250) |
| **we are still alive at that moment** | **69.1 %** |
| we die before step 200 | 35.0 % |

"Almost always" is nearer *seven times in ten*, but the picture is right: the modal round ends with
us alive on a stripped board with an opponent still on it.

### 6.2 · The mechanism holds — crates do make opponents trappable

A "trappable" step is one where **some** free tile exists whose bomb would leave the nearest
reachable opponent with no survivable escape, whether or not we are standing on it:

| crates standing | steps | dist. to nearest opponent | **trappable** | trap site ≤ 4 steps away | mean walk |
|---|---|---|---|---|---|
| 90+ | 507 | 13.7 | 1.97 % | 0.00 % | 11.9 |
| 60–89 | 496 | 12.9 | 24.4 % | 1.6 % | 12.4 |
| **30–59** | 674 | 9.8 | **27.2 %** | 5.8 % | 9.7 |
| 10–29 | 805 | 6.9 | 21.1 % | 5.7 % | 7.7 |
| 3–9 | 536 | 6.2 | 18.5 % | 6.5 % | 6.0 |
| **0** | 2174 | **4.7** | **16.7 %** | 6.4 % | 5.2 |

**Confirmed: a trappable configuration is ~1.6× more common mid-crate-phase than on the stripped
board** (27.2 % vs 16.7 %), exactly as the hypothesis says. My earlier §3 count missed this because
it only counted bombs at the tile we already occupy.

**But the actionable supply is flat.** With crates up the nearest opponent is 9.8 tiles away and the
nearest trap site 9.7 steps; on the empty board, 4.7 and 5.2. "A trap site within 4 steps" is
5.8 % with crates against 6.4 % without. **The configuration advantage of the crate phase is
cancelled, almost exactly, by the distance you have to cover to use it.**

### 6.3 · And the walk destroys the trap

`trap_persistence.py`. At every step, find the nearest reachable opponent, the closest tile from
which a bomb traps it *and which we could survive*, and our walking distance *w*. Schedule a
re-check at step *t + w*: is a bomb on that same tile still lethal to that same opponent?

19.1 % of our alive steps offer such a site somewhere (median walk 6). Of those:

| walk distance | n | **still lethal on arrival** |
|---|---|---|
| 0 (checked one step later) | 31 | **22.6 %** |
| 1–2 | 219 | 16.0 % |
| 3–4 | 452 | 8.4 % |
| 5–8 | 846 | **4.0 %** |
| 9–14 | 395 | 4.1 % |
| 15+ | 163 | 3.7 % |
| all | 2106 | 6.5 % |

The tail is flat at ~4 % from five steps out to fifteen — that is the memoryless base rate, i.e.
**knowing where a trap was carries no information beyond about three steps.** Even standing on the
site, only 22.6 % of traps survive a *single* step: `rule_based_agent` flees actively and a trap's
half-life is roughly one move.

And splitting persistence by crates standing at the time the plan was made:

| crates | n | still lethal |
|---|---|---|
| 30+ | 622 | 5.3 % |
| 10–29 | 386 | 6.5 % |
| 1–9 | 261 | 4.2 % |
| **0** | 837 | **8.0 %** |

Persistence is if anything *better* on the stripped board. Confounded with walk distance (crate-phase
plans are longer), but it does not go the hypothesis's way at any reading.

**Verdict: the mechanism is real and the strategy built on it is not — because the range at which
the crate phase offers traps (≈10 tiles) is three times the range over which trap information
survives (≈3 steps).**

### 6.4 · The part of the hypothesis that is right and that I under-weighted

`callbacks.py:244-255` — `target_direction` falls through to the nearest opponent **only when no
coin and no crate is reachable**. By construction, digit 6 can never point at an opponent while the
board still has crates. `target_type.py` confirms it: the objective is an opponent on 39.1 % of safe
steps, and those are the stripped-board steps.

**The agent cannot express this strategy at all.** That is a genuine representational fact, found by
watching play rather than by reading the ledger, and it belongs in the write-up whatever the payoff
turns out to be. It is also the reason no experiment on this rung has ever tested hunting: E35
priced kills that the state could not aim at, and E36 put opponent *distance* in the danger rows
only.

### 6.5 · What I would run to settle it — a ceiling agent, not a feature

Do not spend a training sweep on this. Measure the **ceiling** first, the way E33 did
(`benedict_q_e33ceil_*`):

- Drive the provided `agent_code/user_agent/` (it just returns `game_state['user_input']`, so
  **no file under `agent_code/` is touched**) from a scratchpad script.
- Policy: the shipped table's greedy action, **except** when a self-survivable trap site for a
  reachable opponent is within *k* steps — then walk the BFS first step toward it, and `BOMB` on
  arrival.
- Arms *k* ∈ {2, 4, 8}, plus *k* = 0 as an exact reproduction of the shipped agent.
- 1000 rounds each at the ship seed, paired over identical arenas, no training at all.

This plays the hypothesis *perfectly*, with an oracle no tabular feature could ever supply. It costs
one evaluation per arm (~15 min each) instead of a 15-seed sweep, and it is decisive in one
direction: **if the oracle hunter does not beat 3.949, no digit encoding this will.** If it does
beat it, we know the size of the prize before designing the digit, and §6.3 says the digit only
needs a 3-step horizon — which is cheap.

Power: no training noise, so the only variance is evaluation noise (±0.12 on `score` from the
opponents' unseeded RNG, `benedict_task4.md` §5.9), and pairing is on arenas. A ceiling worth
chasing should be ≥ +0.5; anything under ~+0.25 is not readable and, being a ceiling, not worth
pursuing anyway.

### 6.6 · What these probes cannot see

- They only find traps in the **existing** geometry. They never use our own bomb to close an exit.
  With one bomb at a time you can rarely build a trap, but you can herd — unmodelled.
- Opponents are modelled as moving freely (they may cross each other) and as never dropping their
  own bomb, so "trappable" is an upper bound on configurations.
- Opponents move **unconditioned on our approach**, because we never actually walk to the site.
  Against a fleeing `rule_based_agent` that is optimistic; against one that would have blocked us it
  is pessimistic. The ceiling agent in §6.5 removes this caveat entirely, which is the main reason
  to run it.

---

## 7 · The hunt ceiling, measured

`hunt_ceiling.py`. The shipped Q-table's greedy action, overridden only when a **self-survivable
trap site** for a reachable opponent lies within *k* steps: walk the BFS first step toward it and
`BOMB` on arrival. A trap site is a free tile whose bomb would leave that opponent with no
survivable escape, re-verified every step with the same time-aware BFS `callbacks.escape_direction`
runs on ourselves. `k = -1` disables the override.

Nothing under `agent_code/` was written: the line-up uses the provided `agent_code/user_agent/`
(its `act` returns `game_state['user_input']`) and the script supplies that input. The evaluation
protocol mirrors `tools/evaluate.py` exactly — per-round `world.rng` reseed *and* the
`np.random.seed` that reaches the provided opponents — so the CSVs are paired arena-for-arena and
go straight into `tools/analyze.py --compare`.

**Control validation.** `k = -1` over 4 000 rounds gives score **3.958**, `won` 0.382, kills 0.229,
suicides 0.482, crates 33.49 — against the shipped agent's published 3.949 / 0.406 / 0.226 / 0.488 /
33.55. The harness reproduces the agent.

### 7.1 · First pass, 1000 rounds, five arms

| k | score | coins | kills | won | suicides | crates | override fires |
|---|---|---|---|---|---|---|---|
| −1 (control) | 3.886 | 2.781 | 0.221 | 0.403 | 0.465 | 33.40 | — |
| 0 | 4.015 | 2.820 | 0.239 | 0.381 | 0.468 | 33.37 | 0.13 % of steps |
| 2 | 4.023 | 2.748 | 0.255 | 0.394 | 0.450 | 33.59 | 0.48 % |
| **4** | **4.086** | 2.841 | 0.249 | 0.410 | 0.448 | 33.29 | 1.54 % |
| 8 | 3.952 | 2.727 | 0.245 | 0.403 | 0.459 | **32.51** | 4.24 % |

Every arm "no effect shown" on score; k = 8 is significantly **worse on crates**
(−0.890 [−1.576, −0.199]) — the diversion cost showing up exactly where §3 predicted it. Kills rise
in all four arms. n = 1000 cannot separate any of it, so the two arms that matter were rerun at
n = 4000.

### 7.2 · 4000 paired arenas — the answer

**k = 4 against the control:**

| metric | control | k = 4 | paired difference | verdict |
|---|---|---|---|---|
| **score** | 3.958 | 4.074 | **+0.116 [+0.002, +0.233]** | BETTER |
| **`won`** | 0.382 | 0.419 | **+0.037 [+0.017, +0.057]** | BETTER |
| **kills** | 0.229 | 0.260 | **+0.032 [+0.011, +0.052]** | BETTER |
| coins | 2.816 | 2.772 | −0.043 [−0.091, +0.005] | — |
| **crates** | 33.49 | 33.07 | **−0.422 [−0.750, −0.081]** | WORSE |
| suicides | 0.482 | 0.463 | −0.019 [−0.041, +0.002] | — |

**k = 0** (bomb only when already standing on a verified trap) moves **nothing**:
score +0.039 [−0.078, +0.155], kills +0.006 [−0.014, +0.026].

### 7.3 · Verdict

**The strategy is real and it is smaller than the instrument that would have to measure it.**

- The effect exists: +0.032 kills, +0.037 `won`, +0.116 score, all with CIs excluding zero at
  n = 4000. The `won` gain is, as far as I can tell, the first `won` effect anything on this project
  has demonstrated. The predicted cost is there too: **−0.422 crates**, so hunting does take bombs
  away from the economy, exactly as C measured in the survey.
- But **+0.116 score is less than half of E37's shipped +0.255**, and this is an *oracle*: it knows
  every step which tile traps which opponent, with a perfect BFS over their escape options,
  re-verified continuously. A tabular digit can offer at most "a trap is within k steps" plus a
  direction, and the table would still have to *learn* to follow it.
- **The ceiling sits at or below the MDE of the experiment that would validate a feature version**
  — 0.254 on score at n = 15 seeds, ≈ 0.11 with the collapse screen (§4.3). An implementation
  capturing, say, 60 % of an oracle worth +0.116 lands at +0.07, which this project cannot read.
- `k = 0` moving nothing kills the cheap version too: the zero-row digit-7 redefinition I proposed
  in §4.2 is exactly the `k = 0` policy, and it is worth **+0.039 [−0.078, +0.155]**.

**So §3's conclusion survives, with its reason corrected.** I said the supply was not there; §6.2
showed the configurations *are* there and §7 shows they convert — just barely. The right statement
is not "there are no kills to take" but **"the kills that positioning can take are worth about a
tenth of a point, and we cannot measure a tenth of a point."**

**What could still overturn this:** the ceiling never uses our own bomb to herd or close an exit,
and it never chases an opponent that is not already trappable. The k = 8 arm bounds the
"chase harder" direction empirically — it was worse. Herding is untested and would need a
two-bomb or body-blocking model that a one-step feature cannot express anyway.
