# What to do next on rung 4 — ranked, with the number each rests on

Written after the session's measurements (`TASK_A_survey_vs_ours.md`, `TASK_B_argument.md`, and
the hunt ceiling in `TASK_B_argument.md` §7). Deadlines, the report and the second model are
deliberately out of scope here.

---

## 0 · Two corrections to project conventions, found in `final_project.pdf`

Both are load-bearing and neither is in `AGENTS.md`. I have not edited any shared file.

### 0.1 · The tournament is decided by **total score**, not by win rate

`final_project.pdf` §3, page 3, verbatim:

> "Because of the random elements, multiple episodes of the game will be played **to determine a
> winner by total score**."

`AGENTS.md` says "`won`/`rank` = standing within the round; on task 4 that matters more than mean
score", and `tools/evaluate.py:264-267` states it as fact: *"an agent on 5.0 that leads 60 % of
rounds beats one on 5.5 that is reliably second."* **Under total score that is exactly backwards.**

This matters because it has been steering metric choice for the whole rung, and because §6 of
`benedict_task4.md` treats "`won` was never demonstrated" as the shipped agent's headline
limitation. If the spec means what it says, `won` was never the target — and everything measured
today points the same way anyway (`won` is a linear readout of score at +0.088/point,
`TASK_B_argument.md` §2).

The PDF's sentence sits in "General setting and rules of the game" and the tournament format is not
spelled out beyond "a tournament between all trained agents". **Ask on `#final-project-questions`.**
It costs one message and it changes which number the remaining five weeks optimise.

### 0.2 · Downloading other teams' agents is explicitly sanctioned

`final_project.pdf`, page 2, verbatim:

> "Discussions about the final project with other teams are very much encouraged. You can share
> your trained agents (*without training code*) on **#final-project-beat-my-agent** and **download
> other teams' agents to test your approach.** Just keep in mind that in the tournament you will
> compete for prizes, so you may want to keep your best ideas to yourself :)"

So the tournament-field idea is not merely allowed — the course runs a channel for it, with
**this cohort's agents on this framework version**, which is strictly better than the SS2024
GitHub repos in the survey. Sharing ours is optional and the PDF hints at not doing it.

---

## 1 · Run a real opponent field. This is the top item and it is your idea.

**Why it outranks every feature change.** All **412** committed rung-4 evaluation CSVs are against
3 × `rule_based_agent` — `ls results/eval/task4_tournament/*.csv | grep -v _rb_` is empty. The
shipped table had never met another opponent type until today. Four fields, 300 rounds each, same
table, no retraining (`scratchpad/strategy/fields/`):

| field | score | coins | kills | suicides | survived | crates | bombs | crates/bomb | best opp |
|---|---|---|---|---|---|---|---|---|---|
| 3 × `peaceful` | **14.49** | 6.74 | **1.550** | **0.087** | 0.913 | 93.9 | 33.9 | 2.77 | 0.06 |
| 3 × `coin_collector` | 6.03 | 2.37 | 0.733 | 0.160 | 0.817 | 29.5 | 33.5 | 0.88 | 4.18 |
| mixed | 5.80 | 3.45 | 0.470 | 0.403 | 0.577 | 41.4 | 30.9 | 1.34 | 7.41 |
| 3 × `rule_based` | 3.95 | 2.82 | **0.226** | **0.488** | 0.464 | 33.6 | 28.9 | 1.16 | 5.32 |

**Two things fall out of that table and both are new.**

**(a) The agent is not bad at killing — it is bad at killing `rule_based_agent`.** Kills run
1.550 → 0.733 → 0.470 → 0.226 strictly inversely with how well the field evades, with *the same
table and no change of any kind*. It kills 1.55 of 3 opponents a round when they cannot escape.
That is the same finding as `TASK_B_argument.md` §6.3 from the other side: the bottleneck is the
opponent's evasion, not our aiming. **So our tournament kill count will be set by the field's
evasion quality, and we cannot know it until we test against real agents.** It is also why building
a hunting feature against `rule_based` would be optimising against the hardest possible evader.

**(b) 82 % of our own-bomb deaths are opponent-induced, and nothing has ever addressed this.**
Suicides run 0.087 → 0.160 → 0.403 → 0.488 across the same four fields. Against `peaceful_agent` —
opponents that never bomb and move randomly — we kill ourselves 0.087 times a round while placing
*more* bombs (33.9 vs 28.9) and living longer (384 vs 275 steps): a per-step own-bomb death rate
**7.8× lower**. So the escape logic is close to sound in isolation, and what kills us is opponents
blocking escape tiles, their blasts overlapping ours, or both (`evaluate.py:258` cannot separate the
overlap case — `benedict_task4.md` §5.8). This is precisely the hole the survey names as unfilled by
the entire published corpus: *"a digit answering 'could an opponent's bomb, placed right now, kill
me here, and do I have an exit that survives it' is genuinely novel against this corpus."* Caveat:
the nearest prior test, E36's opponent-BFS-distance-in-the-danger-rows, was refuted — but it
measured `won` at n = 5 and did cut suicides 0.616 → 0.422. Ceiling-test it with the
`hunt_ceiling.py` harness before designing a digit.

And every strategic conclusion from today is *conditional on rule_based's specific behaviour*:

| conclusion | the rule_based behaviour it rests on |
|---|---|
| the economy closes at step ~200 | rule_based clears crates fast — opponents take **89.3 of 122** crates |
| traps decay to 4 % over a 5-step walk | rule_based **flees actively**; a trap's half-life is ~1 move |
| the kill pool is not free | rule_based suicides **1.50 times/round** of 1.82 deaths |
| `won` = 0.088 × score | the margin distribution against *the best of three rule_based agents* |

A learned student agent plausibly breaks all four: slower crate clearing (longer phase 1, so
phase-2 survival starts to pay), worse fleeing (traps persist, so hunting revives), fewer suicides
(a smaller free-death pool but a larger *takeable* one). **The hunt question you asked may have a
different answer against a different field, and this is the only way to find out.**

There is also a direct payoff on the open Task-A question. Source E (`Bomberman_RL_2024`, lukevoss)
claims **5.04** against 3 × `rule_based` with a 335-state table and no CI. We cannot tell how much
of the 1.09-point gap to our 3.949 is real, because we have never had **their `rule_based`
baseline measured in their framework**. Running *their agent* in *our* harness gives that
calibration constant directly, with our paired CI, in one evaluation and no training.

**Design, in order of value:**

1. `#final-project-beat-my-agent` first — same cohort, same framework, no API drift.
2. Then the survey repos as a fallback: `lukevoss/Bomberman_RL_2024` (E, the 5.04 claim),
   `Li-Jesse-Jiaze/MLE_project_bomberman` (3rd place SS24), `nickstr15/bomberman` (M, 3.2),
   `KunkelAlexander` (K, 6.3 in a different field).
3. Line-ups, 1000 rounds each at the ship seed: ours + 3 × external; ours + 1 external + 2 ×
   rule_based; **and the symmetric bar** — 4 × external, which is the only way to price the field
   itself, exactly as `benedict_task4.md` §1 does with 4 × rule_based.
4. Report **score** primary (§0.1), with `won`/`rank` alongside until the Discord answer lands.

**Cautions, all cheap:**

- `AGENTS.md` forbids copy-pasting existing solutions. Using an agent as an **opponent** is
  measurement, not copying — it is exactly what `rule_based_agent` is for. Keep their code out of
  `agent_code/benedict_task4/` and out of the submission zip, and put third-party folders somewhere
  the zip cannot reach.
- Check licences before committing anything of theirs; prefer *not* committing it and recording the
  commit hash you ran instead.
- API drift: `final_project.pdf` §5 pins the `game_state` keys and the `setup`/`act` signature; an
  SS2024 agent should load, but expect absolute paths in their model loading (the classic crash),
  missing dependencies, and think-time over 0.5 s. `evaluate.py` already records
  `think_over_limit`, so a slow external opponent shows up rather than silently distorting.
- Their agents may be *stronger*. That is information, not failure, and it is the whole point of
  measuring before the tournament rather than after.

**A prediction worth writing down before you run it**, since this is the entry's whole value:
because the crate pool is exactly zero-sum (ours 33.41 + theirs 89.27 = **122.68 of 122**) and
`rule_based_agent` is a strong hand-tuned crate clearer, our crate share and therefore our score
should go **up** against most learned agents. If it does not, the agent is more overfitted to
rule_based than anything in the ledger suggests.

---

## 2 · Fix the measurement before spending sweeps on features

Two changes, both free, both raising what the remaining ~25 sweeps can read.

**2.1 · Pre-register a collapse screen.** The between-seed score SD of 0.249 (`mde.py`) is dominated
by a single collapsed training run — E33 F080's seed at 2.508 against its own 3.74–3.86. Screened
out, the SD is ≈ 0.11 and the n = 15 MDE falls from **0.254 to ≈ 0.11**, i.e. roughly a doubling of
resolution for zero compute. `benedict_task4.md` §5.4 found this once and did not turn it into a
rule. The screen must be a stated criterion on the *training* log (e.g. final-1000-episode mean
reward more than k SDs below the arm median), fixed before any arm is evaluated, so it cannot be a
post-hoc exclusion.

**2.2 · Stop pre-registering `won` as a primary.** MDE on `won` is 0.029 at n = 15 against a
conversion of +0.088 per score point, so any effect below +0.33 score is unreadable on `won` by
construction. Four rung-4 entries already made this error (`benedict_task4.md` §5.1). Combined with
§0.1, `won` should be a reported secondary and nothing more.

---

## 3 · The feature work, ranked

**3.1 · Target type — the one substantial digit worth adding.**
`target_type.py`: the safe-row objective is an **opponent 39.1 %**, a **crate 36.6 %**, a
**coin 21.4 %** of steps, and digit 6 cannot tell them apart, so one row pools three situations
that demand three different actions. It is the corpus's only clean drop-two-bits ablation (K: the
agent fell into period-2 loops without it), and it acts on phase 1 where 94 % of the score lives.
The old objection — "our feature map is the binding constraint" — does not survive `state_visits.py`:
the *effective* state count is **290**, not 64 000, so a 4× larger table is affordable. Two shapes,
one free (re-partition digit 8 in the safe rows) and one costing a size-4 digit with a warm-start
remap; run the free one first because it keeps the parent and is one sweep pair.

**3.2 · Recalibrate the rewards for rung 4.** Never done on this rung. E27 showed the table was
calibrated on a solo board and that raising *only* `CRATE_DESTROYED` was worth **+0.93 score and
+11.5 crates** on rung 3. We now know `crates` is the variable that tracks score
(`corr(score, crates) = +0.937`, `corr(coins, crates) = +0.996`). Hyperparameter optimisation is
explicitly graded and this is the cheapest sweep on the list.

**3.3 · Bomb siting — measure the ceiling before building a digit.** `bomb_siting.py`, 4 936 armed
steps: 71.5 % of armed steps have **zero** crates in range and the policy still bombs on 16 % of
them, so **≈ 45 % of our 28.6 bombs/round clear nothing**. A strictly better tile exists within one
step on 22.5 % of armed steps, within three on 44.2 %. *But* the table already grades its bombing
monotonically by crate count despite digit 7 being a single bit — P(BOMB) runs 0.16 / 0.36 / 0.56 /
0.64 / 0.59 / 0.69 for 0/1/2/3/4/5 crates, because digits 1–4 leak crate density. So a crate-*count*
digit may be buying something the table already has. **Use the `hunt_ceiling.py` harness to test it
without training**: an arm that suppresses `BOMB` below N crates, and an arm that walks one step to
a better tile. Same cost as today's ceiling run, same decisiveness.

**3.4 · Opponent-induced suicide — ceiling-test it before building anything.** §1(b): 0.488 own-bomb
deaths against `rule_based` vs 0.087 against `peaceful`, a 7.8× per-step gap that no rung-4 entry has
targeted. It is the largest single behavioural deficit in the field table and the survey says it is
unaddressed in the entire published corpus. The prior is mixed (E36 refuted), so spend one
`hunt_ceiling.py`-style ceiling arm first — e.g. suppress `BOMB` whenever an armed opponent is
within blast range of any tile on our escape route — and only design a digit if the ceiling clears
+0.25 score.

**3.5 · The truncation bug.** 70 % of rounds hit `MAX_STEPS` and `end_of_round` treats truncation as
termination with no bootstrap, worth ≈ 8.8 Q units — known since rung 3, never fixed in isolation.
It is also the most plausible untested candidate for E38's open question (why 15× more training
degrades bomb siting while sharpening the rows it has seen).

**3.6 · Ceiling-test before every feature from now on.** Today's hunt question was settled in an
afternoon by `hunt_ceiling.py` — an oracle policy over the shipped table, driven through the
provided `user_agent`, no training, paired arenas, `analyze.py --compare` straight out. It answered
in 4 000 rounds what a 15-seed sweep could not have read at all. **This is the cheapest
methodological upgrade available after the collapse screen**: for any proposed digit, write the
oracle version first. If the oracle does not clear the sweep's MDE, the digit cannot.

---

## 4 · Do not spend a sweep on these — each is closed by a measurement

| don't | the number that closes it |
|---|---|
| train longer or shorter | E38: inverted U, 20 000 at the peak; 300 000 costs −0.770 [−1.355, −0.185], 0/5 seeds |
| price kills | E35: `KILLED_OPPONENT` at 5 and 25 moved kills −0.002 |
| split digit 7 into crate-bomb / opponent-bomb | an opponent is in blast range on 25.3 % of armed steps and **trapped on 0.42 %** — the opponent half of that bit is 98.3 % noise |
| add "is `BOMB` safe here" (K/E/H all have it) | an escape from our own bomb exists on **99.08 %** of armed steps — the bit is a constant |
| buy survival | six replications, plus the mechanism: every one of them bought **phase-2** survival, and the economy is over by step 200 |
| build a hunting digit | the *oracle* is worth +0.116 [+0.002, +0.233] score at n = 4000, less than half of E37's shipped effect and at or below the n = 15 sweep MDE — `TASK_B_argument.md` §7 |
| symmetry canonicalisation | E30 not demonstrated; X: 81 → 15 states, "performance didn't improve noticeably" |
| opponents-alive / "am I leading" | E measured their 29-dim vector containing it as *worse* after training |
| coins-per-quadrant, continuous per-direction danger, dead-end coordinates | H, E, C — all measured failures in the corpus |
