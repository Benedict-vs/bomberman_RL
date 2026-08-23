# Task 4 — `classic` against three `rule_based_agent`: one feature, fourteen failures, and the opponent nobody had measured

Consolidated account of rung 4, from `experiments/benedict.md` **E28–E50**. The shape of the rung
is the opposite of rung 3's: there, training never beat the frozen table and a *feature* won; here
training works, **one intervention in fifteen paid**, and the most important result was not an
intervention at all — it was discovering, at E41, that every conclusion the rung had reached was
conditional on the only opponent it had ever been measured against.

Audited six times by independent sessions given the raw data and briefed to break the claim rather
than check it (§7). **All six overturned something**, twice including a previous audit's finding.
That remains the most transferable content.

---

## 1 · What ships

`agent_code/benedict_task4/` — a table trained by `train.py` at its defaults, read through the
feature map in `callbacks.py`. Unlike rung 3, **this table is a product of training.**

| 1000 rounds, held-out ship seed 990731, 3× `rule_based_agent` | previous ship | **shipped** | paired difference |
|---|---|---|---|
| score | 3.694 | **3.949** | **+0.255 [+0.039, +0.475]** |
| `won` | 0.372 | **0.406** | +0.034 [−0.005, +0.073] |
| kills | 0.196 | 0.226 | +0.030 [−0.007, +0.069] |
| **suicides** | 0.749 | **0.488** | **−0.261 [−0.302, −0.219]** |
| killed by opponent | 0.050 | 0.048 | −0.002 [−0.022, +0.017] |
| crates | — | 33.55 | — |
| think_max (ms) | 0.2 | 0.2 | +0.032 [+0.021, +0.043] |

The reference in the same slot is `rule_based_agent` at **3.254 / `won` 0.286**, and the symmetric
bar — four rule-based agents in one field — is `won` **0.282**, so the agent beats both by a clear
margin. `q_table.npy` is byte-identical to
`checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy` (md5 `54d63bc7…`, verified).

**Read that score against the noise floor (§7.9).** Re-evaluating the *same shipped table* at the
same seed returns **3.828**, not 3.949 — a 0.121 swing from the opponents' unseeded RNG alone. The
claim rests on the fifteen-seed sweep; the held-out evaluation confirms it, and a single evaluation
could never have established it.

**And read it against §6.** Against four other students' agents the same table scores 2.2–4.3, and
loses to three of the four.

---

## 2 · The one change that worked

**Digit 8 carries the wall lattice in the danger rows.** While a bomb covers the agent's tile,
digit 6 holds the escape direction, so digit 8 has no "target distance" to encode and previously
sat pinned at a constant. That left **40 960 of 64 000 rows structurally unreachable** — 43.9 % of
all steps, and essentially 100 % of deaths, spent in rows carrying no structural information.

`(x + y) % 4` fills it at **zero new rows** — `FEATURE_SIZES` is unchanged, so the previous table
remains a valid factor-1 warm-start parent. Its low bit *is* the lattice class exactly: stone
pillars sit at (even, even), so a free tile with `x + y` even has both coordinates odd and is a
**crossing** — four structural exits, own bomb clears twelve tiles — while `x + y` odd is a
**corridor**, two exits and six tiles. Verified 11 477/11 477 and 6 427/6 427 over 17 904 danger
steps, zero exceptions.

Digits 1–4 nearly carry this already, but not quite: `H(lattice | digits 1-4)` = 0.192 bits of
0.942, because a blocked neighbour merges wall with crate, so a corridor's two permanent walls look
exactly like two crates.

**Measured in the Q-tables, not inferred from outcomes:**

- Restricted to rows training actually *changed*, `max_a Q`(crossing) − `max_a Q`(corridor) is
  **−0.060 (14/15 seeds)**; the information-free control arm gives +0.013.
- **The informative bit does ~3× the work of the arbitrary one**: within the shipped map the mean
  split *across* lattice classes is **1.123** against **0.361** *within* them, ratio **3.12, 15/15
  seeds**.
- **And it is conditional on the fuse:** a crossing is worth **more** with one move of grace left
  (+0.246, 15/15) and **less** at two or more (−0.189 / −0.092 / −0.108). That is board physics —
  one move needs exits, two or more needs to clear a blast covering twice as many tiles. Without
  the digit the table learns the average of the two, which is an aliased row, not an optimum.

**It pays through bomb siting, not survival.** BOMB attractiveness in the safe rows is identical
across all four arms (0.1810 / 0.1790 / 0.1789 / 0.1790), yet the shipped arm drops **2.26 fewer
bombs and destroys 1.76 more crates** — crates per bomb **+0.130 [+0.061, +0.198]**. The score
gain closes entirely on that route: +0.224 coins + 5 × 0.011 kills = +0.279 of a +0.280 total.

**E41 later tested whether this is `rule_based`-specific and it is not** — crates/bomb *rises*
against every external field (1.25–1.96 against 1.18), so bomb siting is the one component of the
agent that transfers.

---

## 3 · Fourteen interventions, one pattern

| entry | intervention | result |
|---|---|---|
| E30 | passive play (D4-shared updates) | not demonstrated |
| E31 | reckless play | not demonstrated |
| E33 | pay for taking the escape step the map already found | not demonstrated |
| E34 | force the escape step, untrained | not demonstrated |
| E35 | price `KILLED_OPPONENT` at 5 and at 25 | not demonstrated |
| E36 | opponent BFS distance in the danger rows | refuted |
| **E37** | **the wall lattice in the danger rows** | **+0.255 score** |
| E38 | train 15× longer | **−0.770 [−1.355, −0.185]**, 0/5 seeds |
| E39/E40 | hunt: oracle trap-seeking | +0.075 [−0.009, +0.157] — below every bar |
| E42 | fine-tune against a mixed field | −0.043 held out; behaviourally worse |
| E44 | train from scratch against a mixed field | −0.042 held out |
| E46 | gate bombs on escape slack | **−0.283 [−0.349, −0.212]** |
| E48 | crate reward 1.0 → 0.25 | **−1.789 [−2.061, −1.501]** |
| E48 | remove the −5 death penalty | **inert**: +0.097, suicides +0.010 |
| E49 | coin reward 5 → 10 / crate 1.0 → 2.0 / both | +0.084 / −0.085 / +0.013 |

**Eight independent replications that survival does not convert into points.** E36's arm cut
suicides 0.616 → 0.422 and raised survival 0.333 → 0.525 with `won` *unchanged*. E33 and E34
installed the escape behaviour and lost the benefit. E44's from-scratch mixed arm raised held-out
survival +0.079 and lost 0.102 score. The one intervention that paid did so by placing bombs
better, not by dying less.

**E46 is the first entry that priced the exchange.** Vetoing bombs with zero escape slack cut
suicides **−0.131** and raised survival **+0.050** exactly as designed — and cost **−0.283 score**,
entirely through coins (−0.233; kills did not move). **+0.050 survival costs −0.283 score.**

**E48 then showed the price of death is not doing the work either.** Removing `GOT_KILLED = −5`
outright moves suicides by **+0.010 [−0.049, +0.068]** and score by +0.097 — on both fields, at
n = 8 per arm. The single largest negative term in the reward table is **inert**. What *does* move
suicides is what the agent is paid to *do*: E49's crate-reward arm raises them +0.132.

**Why the kill price could not work (E35).** `score = coins + 5·kills`, and kills are 30 % of our
score, so pricing them looks like the obvious lever. It moved kills by −0.002 while training reward
rose 26.82 → 31.31 with 17 % of it kill income: **the reward reached the learner and the policy
ignored it.** Digit 7 is one bit shared between "a bomb here opens a crate" and "a bomb here
catches an opponent", and crates outnumber kills heavily, so the price cannot reach the decision.

---

## 4 · The training horizon, and the parent that outweighs everything

**Longer training is measured and it is worse** (E38). 5 seeds × 300 000 episodes, every checkpoint
evaluated at ε = 0:

| episodes | 20 000 | 40 000 | 80 000 | 160 000 | 300 000 |
|---|---|---|---|---|---|
| score | **3.976** | 3.810 | 3.848 | 3.637 | **3.206** |
| crates/bomb | 1.182 | 1.155 | 1.165 | 1.104 | **0.988** |

Paired against @20 000, 300 000 costs **−0.770 [−1.355, −0.185], 0/5 seeds improving.** The decline
is **monotone**, where rung 2's was a dip that recovered by 300 000 — the two rungs fail
differently. It is **not margin erosion**: rows already updated at 20 000 get *sharper*
(thin-margin share 0.37×), and the pooled statistic suggesting otherwise was confounded by a 50 %
growth in updated rows. The loss runs through **bomb siting** — crates/bomb −0.194, against the
+0.130 the shipped feature bought.

**And the largest single effect on this rung was never an intervention.** E44 trained both arms of a
warm/scratch × `rule_based`/mixed 2×2 from zero. The rung-2 warm-start parent is worth
**−1.661 / −1.936 / −2.019 score** across three fields (all p ≤ 0.0001) — **eight times E37's
shipped +0.255**, and more than every feature, reward and horizon change E28–E49 tested combined.
The mechanism is coverage: 20 000 rung-4 episodes rebuild only **44 %** of the warm table's learned
rows (3 595 against 8 168). **The shipped agent is mostly the rung-2 table plus a lattice bit**, and
preserving that parent is a hard constraint on every future change.

---

## 5 · What the deficit actually is

Three diagnostics, run late, that between them explain why so many interventions failed.

**E43 — 93.5 % of our deaths are our own bomb, in states the features already describe.** For every
death, find the last step at which some action still survived, then trace which bomb's blast
covered the tile we died on:

| at that last survivable moment | 3 × `binary_v6` | 3 × `rule_based` |
|---|---|---|
| **own bomb** | 78.6 % | 79.1 % |
| own + enemy blast overlapping | 14.9 % | 16.5 % |
| enemy bomb, already visible | 4.2 % | 3.3 % |
| **enemy bomb placed after we committed** | **2.3 %** | **0.5 %** |

The last row is the entire case for an opponent-danger digit — the feature the survey calls
unaddressed in the whole published corpus — and it is **2.3 % of deaths**, about 0.017 per round.
**The digit was refuted before it was built.** And in **76.7 %** of deaths two or more surviving
actions existed: this is a *policy* failure inside states the map already sees, not a
representation gap. The composition is nearly field-independent; what changes with a stronger
opponent is the frequency (215 vs 182 deaths per 300 rounds) and the tightness — being down to a
single surviving action more than doubles, 9.3 % → 23.3 %. **Strong opponents do not kill us; they
compress our space until our own bombs do.**

**E45 — the bombs that clear nothing are the *safe* ones.** Against `rule_based`, 72.9 % of armed
steps have zero crates in blast range and the policy bombs on 18.8 % of them, so **48.8 % of its
bombs clear no crate** (`bomb_siting.py` measured 71.5 % / 16 % on the same field). Against the
external field that falls to 32.2 %. Either way, ranked by what the *killing* bomb reached, they are
not what kills us: 0 crates is 32.2 % of bombs and 13.0 % of own-bomb deaths (**lift 0.40×**), while
3+ crates is 33.7 % of bombs and **52.0 %** of deaths (**lift 1.54×**) — external field, and the
ranking holds on `rule_based`.
Bombing "only when needed" would delete the harmless half and keep the killers. What discriminates
is **escape slack**: bombs whose nearest safe tile is 4+ steps away (the fuse is 4) are **4.2 % of
bombs and 28.5 % of own-bomb deaths, a ~7× lift** — and that is *not* the near-constant "is BOMB
safe" bit, which fires on 0.92 % of armed steps. E46 gated on it and lost 0.283 score: the
zero-slack bombs are simultaneously the most lethal and the most productive, because a bomb in a
dense pocket has a contained blast and a tight escape *for the same geometric reason*.

**E47 — the target-type digit is real but smaller than the corpus claims.** `H(type | full row)` is
**0.782–1.042 bits**, four to five times E37's lattice bit. But `target_direction` falls through to
an opponent *only when no crate is reachable*, so "opponent" is structurally the stripped board —
0.0 % at every nonzero crate level — and that phase is worth **2.6 % of score** against a strong
field, 10.0 % against `rule_based`. Restricted to the crate phase, where ~95 % of the score is, the
question is only coin-vs-crate and the residual halves to **0.250–0.311 bits**. Still 2.5–3× the
lattice bit; the free re-partition of digit 8's safe rows remains the one untested feature worth
building.

---

## 6 · External validity — the finding of the rung

**All 412 committed rung-4 evaluations before E41 were against 3 × `rule_based_agent`.** Every
strategic conclusion above was conditional on one hand-written opponent. `final_project.pdf` p. 2
explicitly sanctions testing against other teams' agents; four were taken from public SS2024
repositories (24 surveyed, `scratchpad/external/FINDINGS.md`), vetted, and run in our harness.

**The calibration constant — their agent in *our* slot, against *our* reference field:**

| agent | score | vs our 3.949 |
|---|---|---|
| `xiaoxiae/binary_agent_v6` | **5.572** | +1.623 |
| `xiaoxiae/binary_distance_agent_v2` | **5.336** | +1.387 |
| `Li-Jesse/feature_is_everything` | **5.143** | +1.194 |
| `AI-ELka/ql` (= lukevoss "Atom") | **4.690** | +0.741 |

**All four beat this agent.** Atom lands at 4.690 against its author's claimed 5.04, inside the
±0.5 pre-registered band — so the 1.09-point gap the survey reported is **real**, not a framework
or measurement artefact, and three other agents are further ahead than Atom was.

**Head-to-head, paired within-round margin** (ours minus the mean opponent, n = 1000, all
p < 0.0001): **−1.008** vs `binary_v6`, **−0.797** vs `bindist_v2`, **−0.544** vs
`feature_is_everything`, **+0.912** vs Atom.

**What collapses is our survival, not our bomb siting** — note this is a statement about *our*
deficit, not about their advantage: E47 found their suicides are only halved (0.377 against our
0.505), so **their** edge is not survival either. Against a strong field our crates/bomb *rises*
(1.18 → 1.80) while bombs fall (28.3 → 19.6), because survival collapses **0.440 → 0.223** and
deaths to opponents' bombs rise **2.5×**. And the kill deficit splits two ways: they *enlarge* the
takeable pool 57 % (`rule_based` dies 2.146 times a round against them, 1.822 against us) **and**
convert **91.3 %** of it against our **69.5 %**.

**This reverses a conclusion the rung held for eight entries.** "Buying survival is closed" rested
on six replications — all measured against `rule_based`, against whom we reach step 200 anyway, so
survival was never the binding constraint there. Against these agents we survive the round 22–31 %
of the time and die *inside* the economy. **Survival was not worthless; it was non-binding against
the only opponent ever measured.** E42 and E44 then tested whether training against a mixed field
fixes it — at two learning rates 25× apart — and it does not: **−0.043 and −0.042** held out.

Caveats, stated: every field is three copies of *one* agent, which is harsher than a tournament
round of one each, so these margins are a **lower bound**; and all four agents are SS2024, not this
cohort — the only two SS2026 forks found are bare framework with no trained agent.

---

## 7 · Method failures worth more than the agent

1. **Five "pre-registered negatives" partly measured the design, not the interventions.** The
   paired SD of `won` differences is 0.023, so **n = 5 has an 80 %-power MDE of 0.045** — and E33,
   E34, E35 and E36 all pre-registered `won` targets *below* it.
2. **And E37, the entry written to fix that, was underpowered too.** Its realised paired SD on
   `score` is **0.358**, not the 0.1825 it assumed, so at n = 15 it ran at ~60 % power.
3. **Six entries pre-registered a bar their own design could not reach.** E40's +0.25 needed ≥25 %
   trap conversion at the 0.200 bombs/round it had *already measured* — above its own P3 bar of
   20 %, so **passing P3 guaranteed failing P1**; the bar was carried over "unchanged" from E39,
   where the same +0.25 needed only 11.7 % because there were twice as many shots. E48's crate
   guard was breached by the arm's own failure. Audit 11 caught the first; the arithmetic is now
   done *before* the run.
4. **Run-level pairing bought nothing.** corr(arm, control) at matched seed runs −0.47 to +0.44;
   `main.py` does not seed the provided opponents, so the pairing every rung-4 power calculation
   assumed does not exist.
5. **One collapsed seed can be a third of an effect.** E37's headline +0.280 becomes **+0.198** once
   a single control run that collapsed in *training* is removed — and gets *more* significant, not
   less, because the SD halves.
6. **An accept-the-null test passes more easily the worse your data is.** E37's specificity
   hypothesis "holds" with a CI tolerating **96 % of the treatment effect**, and it was a leg of a
   pre-committed continuation rule.
7. **A switch that changes what a digit *means* must be exported at evaluation time too.** Reading a
   trained table under the wrong digit-8 map does not crash — shapes match — it silently produces a
   different agent: suicides 0.450 → 0.751, survived 0.502 → 0.193. `score` barely moves, so **the
   behavioural metrics are the discriminator, not the headline.** E42 repeated the lesson from the
   other side: its regression guard passed on score (−0.049) while suicides rose 0.183 and survival
   fell 0.170.
8. **A row that "carries value" is not a row training touched.** Measuring the lattice split over
   rows that merely hold a warm-start value returns **+0.45 on a control whose digit 8 is pinned** —
   a table that cannot encode the lattice at all.
9. **`tools/evaluate.py:259` undercounts `killed_by_opponent`.** `died − suicides` misses deaths
   where own and enemy blasts overlap, and the undercount scales with bombs placed. Audit 10
   measured the excess directly: **16 % of deaths in the `rule_based` field are double-credited, and
   exactly 0 % against `peaceful`** — so `suicides` is not the same statistic across fields and must
   never be differenced across them.
10. **A 1000-round evaluation has a ±0.12 noise floor on `score`, and `AGENTS.md` said it had
    none.** "22.7 % of rounds repeat exactly, the means repeat to four decimals" was believed for
    nine entries. The first half is right; the second is false — the same table twice gives 3.949
    and 3.828.
11. **A fixed bootstrap seed makes a coin flip look like a verdict.** `analyze.py` seeded its
    percentile bootstrap with a hard-coded `default_rng(12345)`, which is reproducible but not
    *stable*. E39's headline "+0.116 [+0.002, +0.233]" excluded zero on a **minority** of seeds and
    did not replicate (+0.075 [−0.009, +0.157] at n = 8000). `--compare` now prints a `t` and a
    sign-flip permutation p and marks a row `(fragile)` when they disagree; **a fragile row is not
    demonstrated regardless of its CI**. E42's own headline fails that rule at p = 0.064.
12. **A pilot's sign can be wrong.** E46's arm measured **+0.187** at n = 150 and **−0.283** at
    n = 4000 — a 0.47 swing on the same arm and seed family. The pre-registration disclosed that the
    bar had been set knowing the pilot's direction; the pilot pointed the wrong way.
13. **An audit can be wrong, and the fix can be worse than the bug.** Audit 10 reported that the
    hunt oracle's trap test ignored the game's simultaneous move. It does not — `escapable()`'s
    depth-1 BFS nodes *are* the tiles the target can reach when the bomb lands, and a correctly
    timed test returns the identical site set over 20 022 steps. The E40 "fix" written from that
    report deleted **31 %** of trap sites through two new bugs. Audit 11 caught it. **The verdict
    survived on the arm the entry called broken.**
14. **A within-agent correlation is not a between-agent one.** `corr(score, crates) = +0.937` across
    rounds nearly sent a sweep after crate throughput; between agents it inverts — the agent that
    beats us by 1.744 opens **fewer** crates (25.09 vs 33.28). Caught before the run, by checking.
15. **Estimators break on the arms that make them interesting.** E48's `coins_per_crate` computed as
    a mean of per-round ratios returned ~10⁶, because the arm being tested leaves many rounds with
    zero crates. The correct estimator is a ratio of totals.

---

## 8 · Limitations, stated rather than hidden

- **The agent loses to this cohort's published agents** (§6), by margins 4–8× larger than any
  effect this rung's instruments can resolve. Every remaining lever on the strategy list has been
  measured and none closes it: hunting (E40), an opponent-danger digit (E43, 2.3 % ceiling),
  training distribution (E42/E44, at two learning rates), bombing discipline (E45/E46), the reward
  table in four directions (E48/E49). **What remains untested is the free target-type re-partition
  (§5) and a jointly re-derived reward table, which is not affordable in the time left.**
- **`won` was never demonstrated — and was never the target.** `final_project.pdf` §3 says the
  tournament winner is determined *"by total score"*; `won`/`rank` are constructs of
  `tools/evaluate.py`. `AGENTS.md` asserted the opposite for the whole rung and steered four entries
  into pre-registering `won` at a sample size where its MDE made it unreadable. It matters less than
  it looks because `won` is a linear readout of score: **+0.088 [+0.076, +0.101] per point**.
  *The tournament format beyond "a tournament between all trained agents" is still not spelled out
  in the PDF; the question is open with the course.*
- **The truncation defect is real, characterised, and deliberately not fixed** (E50). It is not what
  the strategy list described — not a missing bootstrap but a **duplicate, un-bootstrapped update**
  on a cell the step update already handled correctly, because `agents.py:168` sets
  `last_game_state` before `act` and never updates it. The list's "70 % of rounds" is an
  *evaluation* figure; training survival is **10–22 %**, so it touches **≈0.1 % of Q updates**, on
  rows in a phase worth 2.6–10 % of the score. Below any instrument this project has. The correct
  fix, for the record, is to skip the `end_of_round` update when the agent survived.
- **Why the four-way split beats the lattice bit alone is not settled.** The leading explanation is
  a learning-rate artifact — α = 1/visits^0.7 per cell, so a 4-way split holds α 2^0.7 = 1.62×
  higher than a 2-way one — and it is untested. E49 produced a weak echo of the same mechanism:
  doubling every reward at a fixed ratio cost **−0.202 held out** and raised invalid actions,
  which is an optimiser interaction rather than a reward effect.
- Reward magnitudes were probed one dimension at a time from one operating point. A jointly
  re-derived table is the thing none of this rules out.

---

## 9 · Reproduction

```bash
# the shipped table: one seed of a fifteen-seed sweep, selected on validation seed
# 550731 and confirmed on held-out 990731. main.py does not seed the provided
# opponents, so this reproduces the CONFIGURATION, not the bytes -- the sweep is
# the unit of evidence, never a single run. No environment variables are needed:
# every default in train.py is the shipped value, BM_RUN_INDEX=106 included.
uv run python main.py play --agents benedict_task4 \
    rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731
# -> writes q_table_trained.npy beside the agent; q_table.npy is never written by
#    training, so a stray --train cannot destroy the submitted model.

# the warm-start parent, committed:  checkpoints/benedict_task4/q_table_parent.npy
# the exact shipped checkpoint:      checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy
#   (byte-identical to agent_code/benedict_task4/q_table.npy, md5 54d63bc7...)

# evaluation, held-out seed
uv run python tools/evaluate.py --agents benedict_task4 --opponents rule_based \
    --n-rounds 1000 --seed 990731 --label benedict_task4_shipped_e37__task4_rb_ship990731 \
    --out-dir results/eval/task4_tournament

# the external field (E41). The agents are NOT committed -- 20 of 24 source repos
# carry no licence. scratchpad/external/FINDINGS.md records URL + commit hash for
# each, and scratchpad/external/install/INSTALL.md the fixes needed to load them.
```

Reproducing **E24–E38** requires checking out the commit that ran them. Their arm switches
(`BM_ABLATE`, `BM_HUNT`, `BM_OPPDIST`, `BM_D8`, …) were removed from `callbacks.py` when the winning
configuration became the default, and the submission cleanup removed the rest from `train.py`:
`BM_SHAPE` (E19), `BM_D4` (E30), `BM_ESCAPE` (E33), the `ALPHA`/`EPS` mode switches E05 and E06
settled, and **`BM_ARM`** — which the nine `scratchpad/benedict/e3*_arms.sh` launchers set, so the
range extends past E36 to E38. One consequence to know before re-running any sweep at HEAD:
`RUN_NAME` no longer carries the arm, so **two arms at the same `BM_RUN_INDEX` now append into one
training CSV** rather than two. Every one of those runs is recoverable from its
`.meta.json`, which records the commit and — from 2026-08-16 — the full `BM_*` environment.

Evidence: `results/eval/task4_tournament/` and `results/train/task4_tournament/` (committed),
`scratchpad/audit7/` … `scratchpad/audit12/`, `scratchpad/benedict/`, `scratchpad/strategy/`,
`scratchpad/external/`.
