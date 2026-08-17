# Task 4 — `classic` against three `rule_based_agent`: one feature, eight failures, and a power problem

Consolidated account of rung 4, from `experiments/benedict.md` **E28–E37**. The shape of the rung
is the opposite of rung 3's: there, training never beat the frozen table and a *feature* won;
here, training works, and **eight of nine interventions still failed** — seven of them because
they bought survival, which does not convert into points on this board.

Audited three times by independent sessions given the raw data and briefed to break the claim,
not check it (§5). All three overturned something. That remains the most transferable content.

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

Paired over 1000 identical arenas. The reference in the same slot is `rule_based_agent` at
**3.254 / `won` 0.286**, and the symmetric bar — four rule-based agents in one field — is `won`
**0.282**, so the agent beats both by a clear margin.

**Note the suicide column against §3.** Every intervention that set out to buy survival failed to
convert it into points; this one bought survival *as a by-product* of siting bombs better, and the
score moved. The distinction is the whole content of the rung: survival is not the lever, but it
is correlated with the thing that is.

**And read that score against the noise floor (§5.9).** Re-evaluating the *same shipped table* at
the same seed returns **3.828**, not 3.949 — a 0.121 swing from the opponents' unseeded RNG alone.
The claim rests on the fifteen-seed sweep; the held-out evaluation confirms it, and a single
evaluation could never have established it.

The table is `q_table_e37_PLB2_s106__ep20000.npy`, one seed of a fifteen-seed sweep, **selected on
validation seed 550731 and confirmed on the held-out 990731.** The confirmation matters: the
validation-rank-2 seed reached only +0.136 held-out, and the *control* arm's best seed reached
−0.027. Single-seed differences on this rung are mostly noise (§5.1).

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

---

## 3 · Eight interventions, one pattern

| entry | intervention | result on `won` |
|---|---|---|
| E30 | passive play (D4-shared updates) | not demonstrated |
| E31 | reckless play | not demonstrated |
| E33 | pay for taking the escape step the map already found | not demonstrated |
| E34 | force the escape step, untrained | not demonstrated |
| E35 | price `KILLED_OPPONENT` at 5 and at 25 | not demonstrated |
| E36 | opponent BFS distance in the danger rows | refuted |
| **E37** | **the wall lattice in the danger rows** | **+0.024, score +0.280** |

**Six independent replications that survival does not convert into points.** E36's arm cut suicides
0.616 → 0.422 and raised survival 0.333 → 0.525 with `won` *unchanged*. E33 and E34 installed the
escape behaviour and lost the benefit. The one intervention that paid did so by placing bombs
better, not by dying less — and its own survival gain (+0.020) is not significant.

**Why the kill price could not work (E35).** `score = coins + 5·kills`, and kills are 30 % of our
score, so pricing them looks like the obvious lever. It moved kills by −0.002 while training reward
rose 26.82 → 31.31 with 17 % of it kill income: **the reward reached the learner and the policy
ignored it.** Digit 7 is one bit shared between "a bomb here opens a crate" and "a bomb here
catches an opponent", and crates outnumber kills heavily, so the price cannot reach the decision.
The answer to a kill deficit is a feature, not a price — which is exactly what E37 then was.

---

## 4 · What did *not* explain it

| hypothesis | refutation |
|---|---|
| deaths land in never-visited rows (coverage) | all-zero-row share is 0.00006 across every arm, against a 0.01 guard — coverage has not bound since E30 |
| opponent proximity is the missing information | verified 6.1× death lift at BFS ≤ 2, and it converts to **nothing**: `won` −0.007 [−0.031, +0.017] |
| row 55060 is a second Bellman fixed point | it is one aliased row — 0 % fatal across 655 far visits, 36.5 % across 148 near ones |
| the agent is too reckless / too passive | tested in both directions (E30, E31); neither moved `won` |
| a 4-way split of the idle digit is enough on its own | the information-free 4-way control gives +0.079 [−0.111, +0.269] and −0.136 crates |

---

## 5 · Method failures worth more than the agent

1. **Five "pre-registered negatives" partly measured the design, not the interventions.** The
   paired SD of `won` differences is 0.023, so **n = 5 has an 80 %-power MDE of 0.045** — and E33,
   E34, E35 and E36 all pre-registered `won` targets *below* it. Four entries were underpowered on
   their own primary metric before they ran.
2. **And E37, the entry written to fix that, was underpowered too.** Its realised paired SD on
   `score` is **0.358**, not the 0.1825 it assumed, so at n = 15 it ran at ~60 % power. The
   pre-registered MDE table was also internally inconsistent — 0.352 / 0.210 / 0.164 back-solves to
   SD = 0.211 at every n, not the 0.1825 printed beside it.
3. **Run-level pairing bought nothing.** corr(arm, control) at matched seed runs −0.47 to +0.44, and
   `SD_paired/SD_unpaired` exceeds 1 in 5 of 9 cells. `main.py` does not seed the provided
   opponents, so the pairing every rung-4 power calculation assumed does not exist.
4. **One collapsed seed can be a third of an effect.** E37's headline +0.280 becomes **+0.198** once
   a single control run that collapsed in *training* is removed — and the result gets *more*
   significant, not less (p = 0.0091 → 0.0010), because the SD halves. The differences are
   non-normal (Shapiro p = 0.0009, skew +2.16), so the t-CI's lower bound was a fit artifact;
   bootstrap gives [+0.133, +0.482].
5. **An accept-the-null test passes more easily the worse your data is.** E37's specificity
   hypothesis "holds" with a CI tolerating **96 % of the treatment effect** — and it is a leg of the
   pre-committed continuation rule, so the rule fired `CONTINUE` on evidence that cannot distinguish
   its null from the full effect. The same error as pricing a placebo ratio against a zero
   denominator, moved onto the go/no-go decision.
6. **A switch that changes what a digit *means* must be exported at evaluation time too.** Reading
   a trained table under the wrong digit-8 map does not crash — shapes match — it silently produces
   a different agent: suicides 0.450 → 0.751, survived 0.502 → 0.193. `score` barely moves (0.017
   vs 0.051), so **the behavioural metrics are the discriminator, not the headline.**
7. **A row that "carries value" is not a row training touched.** Measuring the lattice split over
   rows that merely hold a warm-start value returns **+0.45 on a control whose digit 8 is pinned**
   — a table that cannot encode the lattice at all. That estimator measures visit coverage
   (crossings updated on 75 % of danger bases, corridors on 21 %) multiplied by uplift.
8. **`tools/evaluate.py:258` undercounts `killed_by_opponent`.** `died − suicides` misses deaths
   where own and enemy blasts overlap, and the undercount scales with bombs placed, so every such
   magnitude quoted since E28 is inflated roughly 2×.
9. **A 1000-round evaluation has a ±0.12 noise floor on `score`, and `AGENTS.md` said it had
   none.** "22.7 % of rounds repeat exactly, the means repeat to four decimals" was believed for
   nine entries. The first half is right; the second is false. Evaluating the *same table* twice
   at the same seed gives 3.949 and 3.828 — 20.5 % of rounds identical, paired difference
   −0.121 [−0.346, +0.104]. Found while regression-testing the shipped agent after a code cleanup,
   where it first looked like the cleanup had broken something. Any single-run difference below
   ~0.12 on this rung was never readable, which is a second, independent reason the E30–E36
   negatives measured the design as much as the interventions.

---

## 6 · Limitations, stated rather than hidden

- **`won` was never demonstrated.** The shipped agent's +0.034 on the held-out seed has a CI
  including zero, and at any n this project can afford it always would: the MDE on `won` is ~0.021
  at n = 15 against a conversion rate of `won ≈ 0.113 × score`. Score is what carries the claim.
- **`end_of_round` treats truncation at `MAX_STEPS` as termination** (no bootstrap) on ~33 % of
  rounds, worth ≈8.8 Q units. Known since rung 3, never fixed in isolation.
- **Why the four-way split beats the lattice bit alone is not settled.** The leading explanation is
  a learning-rate artifact — α = 1/visits^0.7 per cell, so a 4-way split holds α 2^0.7 = 1.62×
  higher than a 2-way one for the whole run — and it is untested. A 2-way information-free control
  would test it.
- The `rule_based_agent` field is a *proxy* for the tournament. Nothing here measures play against
  other students' learned agents.

---

## 7 · Reproduction

```bash
# the shipped table: one seed of a 15-seed sweep, selected on 550731, confirmed on 990731.
# main.py does not seed the provided opponents, so this reproduces the configuration,
# not the bytes -- the sweep is the unit of evidence, never a single run.
BM_QUIET_LOGS=1 BM_MODEL_SUFFIX=_run BM_RUN_INDEX=106 \
uv run python main.py play --agents benedict_task4 \
    rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731

# the warm-start parent, committed: checkpoints/benedict_task4/q_table_parent.npy
# the exact shipped checkpoint, committed: q_table_e37_PLB2_s106__ep20000.npy

# evaluation, held-out seed
uv run python tools/evaluate.py --agents benedict_task4 --opponents rule_based \
    --n-rounds 1000 --seed 990731 --label benedict_task4_shipped_e37__task4_rb_ship990731 \
    --out-dir results/eval/task4_tournament
```

Reproducing **E24–E36** requires checking out the commit that ran them: their arm switches
(`BM_ABLATE`, `BM_HUNT`, `BM_OPPDIST`, `BM_D8`, …) were removed from `callbacks.py` when the
winning configuration became the default. Every one of those runs is recoverable from its
`.meta.json`, which records the commit and — from 2026-08-16 — the full `BM_*` environment.

Evidence: `results/eval/task4_tournament/` and `results/train/task4_tournament/` (committed),
`scratchpad/audit7/`, `scratchpad/audit8/`, `scratchpad/audit9/`, `scratchpad/benedict/`.
