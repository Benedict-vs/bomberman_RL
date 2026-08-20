# Audit 12 — E42 (train against a field that hunts back)

Brief: break the *interpretation* and the *forward decision*, not the null result.
Everything below was produced by scripts in `scratchpad/audit12/` (outputs in `*.out`).
Nothing outside this directory was modified. No games were played; every number is a
re-analysis of committed CSVs, committed training logs, committed Q-tables, or read-only
git history.

---

## 1. Verdicts

| claim | verdict | the number that decides it |
|---|---|---|
| **A — the mechanism is credit assignment** ("the agent learns that dying is not its fault and stops paying to avoid it") | **OVERTURNED** | The suicide rate **before step 200** is unchanged on all three fields: guard **+0.010 [−0.013, +0.032]**, in-dist −0.014, held-out −0.012. The whole of the reported +0.183 is the post-step-200 hazard, 0.292 → 0.570, in a phase where a suicide gives nobody points. |
| **B — the control arm (E37 PLB2 checkpoints) is valid** | **SURVIVES** | Three independent measurements of the same recipe give guard-field suicides **0.515 / 0.490 / 0.478** against the mixed arm's **0.687**; warm parent is byte-identical (md5 `367ddf4f…`) and the feature map is provably unchanged. I attacked this and could not break it. |
| **C — "the deficit cannot be trained away while the feature map cannot see the pressure", so the digit must come first** | **WEAKENED** (direction survives its pre-registration; the stated justification does not) | `killed_by_opponent` — the exact quantity the proposed digit addresses — **fell** under mixed training in all three fields, significantly pre-200 on the held-out field: **−0.013 [−0.024, −0.004]**. The lever E42 declares inert moved the target quantity in the right direction, by ~8 %. |
| **scorer** — `scratchpad/benedict/e42_analyze.py` | **one row overturned** | The n = 8 percentile bootstrap is 15–20 % narrower than a Welch t interval. Exactly one row flips, and it is the entry's declared headline: in-dist score, bootstrap **[−0.305, −0.014]** vs t **[−0.334, +0.012]**, permutation p = 0.064. |

---

## 2. Findings, most damaging first

### F1 — Claim A is refuted by E42's own data: the extra deaths are entirely post-economy, and they cost nothing

**The claim.** "Against strong agents a large share of deaths are opponent-induced and
essentially unattributable … **The agent learns that dying is not its fault, and stops
paying to avoid it** — which is precisely what suicides +0.183 and survival −0.170
describe."

**What is wrong.** That story is a claim about the agent's willingness to pay to stay
alive. It must show up when staying alive is worth something. Split each round at step
200 (the point `scratchpad/strategy/` establishes the round's economy closes) and the
effect is absent before the cut and doubles after it, on every field
(`scratchpad/audit12/a3_hazard.py`, `a3_hazard.out`):

| field | suicide rate < 200 | P(alive at 200) | suicide hazard ≥ 200 given alive |
|---|---|---|---|
| guard (3× `rule_based`) | 0.319 → 0.328 **[−0.013, +0.032]** | 0.633 → 0.628 [−0.026, +0.017] | 0.292 → **0.570** [+0.194, +0.358] |
| in-dist (training field) | 0.359 → 0.345 [−0.044, +0.012] | 0.509 → 0.535 [+0.003, +0.049] | 0.169 → 0.235 [+0.028, +0.101] |
| held out (3× `bindist_v2`) | 0.478 → 0.467 [−0.030, +0.006] | 0.370 → 0.395 [+0.007, +0.042] | 0.212 → 0.259 [+0.003, +0.093] |

Three things follow, and each is fatal to the mechanism as written:

1. **During the phase where death is expensive, the mixed arm dies no more often.** In
   two of the three fields it survives to step 200 *more* often, with a CI excluding 0.
   An agent that had "stopped paying to avoid death" would die more when death is
   expensive; this one does not.
2. **The extra deaths cost nothing, by the rules of the game.** `environment.py:249-251`
   awards no points for a suicide — only `explosion.owner.update_score(REWARD_KILL)` on
   the else branch. After the crates and coins are gone there is nothing left to forfeit,
   which is why score moved −0.049 (ns) and guard `margin_mean` moved **+0.009**, i.e. in
   the mixed arm's favour: a late suicide denies an opponent the +5 it would otherwise
   collect (`killed_by_opponent` −0.014 [−0.024, −0.004]).
3. **The direction of the guard-field death shift is the opposite of the story.**
   Opponent-caused death — the "unattributable" channel — went **down**; own-bomb death in
   a zero-stakes phase went up.

**Corollary that inverts the entry's own lesson.** E42 calls P2's score-only guard "the
entry's most useful mistake" and concludes the behavioural metrics should have been the
discriminator. The split above says the opposite: score was *right*, and the behavioural
row is what misled. `suicides` on task 4 is a regression guard for escape logic
(`AGENTS.md`), and pre-200 escape logic is intact.

### F2 — There is no learned policy difference to attribute a mechanism to

Loading the 16 tables (`a5_tables.py`, `a6_flips.py`):

- **Death is not priced lower in the mixed arm.** Over danger rows (digit 5 > 0) the two
  arms updated, `min_a Q` is **−1.996 (ctl) vs −2.015 (mix)** — the mixed arm's worst
  action is *slightly more negative*. What shrank is the optimistic side:
  `max_a Q` 7.126 → 6.942, mean |Q| 3.123 → 2.905. That is a lower-return field, not a
  cheaper death. The credit-assignment story predicts the death side; the death side did
  not move.
- **Zero systematic argmax flips.** Over the **2 888 rows all 16 tables updated**, there is
  not one row where all 8 control seeds agree on an action and all 8 mixed seeds agree on
  a different one (`a6_flips.out`). Between-arm argmax agreement is **0.986**; within-ctl
  and within-mix agreement are both **0.987**. The two arms' greedy policies differ no
  more than two seeds of the same arm do.
- Mixed-arm coverage is *broader*, not narrower: 4 692 vs 3 995 rows touched (danger rows
  2 784 vs 2 312). The "less experience" alternative E42 dismissed is not a coverage
  deficit in the naive direction — see F6 for where the deficit really is.

So the mechanism section is inferring a change in what the agent believes about death from
two summary statistics, while the object those beliefs live in shows no such change.

### F3 — P3's second leg ("the loss did arrive through bomb siting") is a denominator artefact, and crates cannot carry a score loss anyway

On the held-out field (`a1_full_table.out`):

| | ctl | mix | diff | 95 % CI |
|---|---|---|---|---|
| crates destroyed | 39.096 | 39.727 | **+0.632** | [+0.239, +0.974] |
| bombs placed | 20.793 | 22.793 | **+2.000** | [+1.464, +2.545] |
| steps alive | 196.6 | 204.1 | +7.410 | [+2.582, +11.960] |
| crates/bomb (mean of per-round ratios) | 2.214 | 2.053 | −0.160 | [−0.185, −0.137] |

The mixed arm destroys **more** crates, from more bombs, while alive longer. The ratio
falls because the denominator grew faster than the numerator. Reading that as "the loss
arrived through bomb siting" reverses the sign of the productive quantity. (I checked and
cleared the obvious ratio bug: `bombs == 0` rounds are 0.000 of the sample on every field,
so `max(bombs, 1e-9)` is never exercised.)

More basic: **score = coins + 5 × kills**; crates contribute nothing. The score movement
decomposes exactly:

| field | Δcoins | 5 × Δkills | sum | Δscore |
|---|---|---|---|---|
| held out | −0.013 | −0.100 | −0.113 | −0.115 |
| in-dist | −0.073 | −0.085 | −0.158 | −0.161 |
| guard | −0.047 | −0.000 | −0.047 | −0.049 |

Every field closes to within 0.003. The held-out and in-dist losses are ~85 % the kills
term; the guard loss is 100 % coins. Bomb siting is not a channel in any of them.

### F4 — The declared headline finding does not clear the project's own bar

E42: "**P4 REFUTED, and this is the finding** … On the field it trained against for
20 000 episodes, the mixed arm scores 3.181 against the control's 3.342 — *the control,
which never saw that field, plays it better.*" The commit message repeats it as the
entry's lead result.

That row is in-dist score, −0.161, bootstrap [−0.305, −0.014], permutation p = 0.064
(`a9_stats.out`). Two independent reasons it is not demonstrated:

1. **`AGENTS.md`'s own fragility rule**: a row is fragile when the interval and the
   sign-flip p disagree, and "a fragile row is *not demonstrated* regardless of what its
   CI says". CI excludes 0, p = 0.064. This is the exact pattern the rule was written for
   after E39/E40.
2. **The interval is an artefact of the estimator.** `boot_ci` resamples 8 seeds per arm
   and takes raw percentiles — no `(n−1)/n` correction, z rather than t. Measured over all
   ten reported rows, the bootstrap width is **0.80–0.85×** the Welch t width. On nine
   rows that changes nothing. On this one it flips the verdict: Welch t gives
   **[−0.334, +0.012]**, which contains zero.

`e42_analyze.py` never runs the fragility check `analyze.py` was extended to run in
c40a902, and would have printed "nicht gezeigt" for this row had anyone read the p column
next to it. My re-scoring also flags `opp_score` on the guard and held-out fields as
fragile in the same way.

The pre-registered P4 (in-dist gain exceeds held-out gain) is genuinely refuted — that
much survives. The *interpretation* built on top of it is not.

### F5 — E42's "the training curve and the policy point in opposite directions" is competing risks, and E42's own guard P5 forbids the comparison

From the training logs (`a10_trainlogs.out`, 20 000 episodes × 8 seeds per arm):

| | ctl (3× rule_based) | mix | diff |
|---|---|---|---|
| `KILLED_SELF` / episode | 0.818 | 0.740 | **−0.077** |
| opponent-caused (`GOT_KILLED − KILLED_SELF`) | 0.070 | 0.161 | **+0.091** |
| `GOT_KILLED` / episode | 0.888 | 0.902 | +0.013 |

The fall in training suicides is the rise in opponent kills, to within 0.014: total deaths
are unchanged. The mixed arm does not suicide less; its suicides are **pre-empted**. E42
presents this as a paradox ("the sixth time on this project that a training log has looked
healthy over a worse agent") and uses it as evidence for the mechanism. It is the
competing-risks artefact of differencing `suicides` across fields — which E42's own guard
**P5** pre-registers as forbidden ("`suicides` reported per field and **never differenced
across fields** (audit 10 F5)"). The entry differences it across fields in the sentence
that carries its mechanism.

The same table also sizes the premise: **82 % of the mixed arm's training deaths are still
its own bomb.** "A large share of deaths are opponent-induced and essentially
unattributable" describes 17.9 % of deaths against 7.9 %. The attributable self-inflicted
signal is still 4.6× as frequent and only 9.5 % less frequent per episode than in the
control.

### F6 — The alternative that does fit: endgame under-exposure plus loss of a `rule_based`-specific specialisation

I owe a positive account, and E42 dismissed the experience hypothesis on an aggregate
("only 8 % fewer training steps") that averages over the phase where the damage lives.
Measured in the right phase (`a11_exposure.out`):

| | ctl | mix | ratio |
|---|---|---|---|
| P(training episode reaches step 200) | 0.322 | 0.253 | **0.79** |
| P(reaches step 300) | 0.186 | 0.145 | 0.78 |
| total training steps | 3.20 M | 2.87 M | 0.90 |
| steps inside episodes ≥ 200 | 2.07 M | 1.63 M | **0.79** |

Endgame exposure is down **21 %**, not 8 %. And the *content* of the endgame differs: on a
`rule_based` board past step 200 the crates are gone, so digit 6 falls through to the
nearest opponent and digit 7 fires only on opponents — a distinct region of the table that
the mixed arm both visited less and visited against different agents. That predicts what is
observed: the late-phase hazard rises most on the field the control is specialised to
(+0.278) and least on the two it is not (+0.066, +0.047).

I have **not** separated "less endgame experience" from "a `rule_based`-specific endgame
specialisation the mixed arm gave up"; both are consistent with the data, and both are
ordinary distribution shift rather than a credit-assignment failure.

### F7 — Claim C: the evidence chain to "the digit must come first" is broken at both ends

The forward decision is pre-registered (P1's refutation clause names the feature map), so I
am not attacking the *direction*. I am attacking the two load-bearing sentences.

**"The deficit cannot be trained away while the feature map cannot see the pressure."**
Under mixed training, `killed_by_opponent` — precisely what an opponent-bomb-danger digit
would reduce — fell on **every** field: held-out −0.011, in-dist −0.012, guard −0.014
[−0.024, −0.004]; restricted to the pre-step-200 phase on the held-out field, −0.013
[−0.024, −0.004]. Small — about 8 % of 0.175 per round, against a ~1.0 margin deficit —
but the sign is wrong for a claim of impossibility. The honest statement is "the training
lever moves the target quantity by ~8 % and does not close a 1.0-point gap", not "cannot
be trained away".

**"Training against a field that hunts back."** It was not trained against that field; it
was **fine-tuned**. `train.py:515` sets `visits = WARM_N = 100` on every row the parent
valued, so α = 1/100^0.7 = **0.0398** at the first update and falls from there, and the
parent (`q_table_parent.npy`, md5 `367ddf4f…`) is the shipped table trained entirely
against `rule_based`. **84 % of the mixed arm's updated rows carried that pseudo-count**
(93 % for the control). A from-scratch run on the mixed field — `BM_WARM=""`, which the
code already supports — was never done, and it is the run that would actually test whether
the mixed field can be learned. E42's own limitation list mentions the mixture and the
budget but not this.

**Magnitude, for whichever direction is chosen.** On the held-out field the arm's deaths
are 0.557 self-inflicted against 0.175 opponent-caused: **76 % of deaths are still our own
bomb**, and 0.478 per round are suicides before step 200. E41's attribution of the
*increment* over the `rule_based` field to opponent bombs survives (killed_by 0.062 →
0.175 against suicides 0.503 → 0.557), but the largest absolute loss channel against
strong agents is unchanged and is not what the proposed digit addresses.

**On the "is a tabular agent at its ceiling" reading**, which I was asked to weigh: E42
does not license that either. It tested one mixture, one budget, one warm start, and its
own held-out numbers are flat rather than collapsed. Neither "further feature work is
unjustified" nor "the feature map is the blocker" is demonstrated by this entry; what is
demonstrated is that a 20 k-episode fine-tune on a mixed field does not close the gap.

### F8 — Claim B: attacked, survived (with two caveats that change nothing)

I expected this to be the soft target and it is not.

- **Code identity.** The control tables were trained 2026-08-16 07:48–11:12, between
  `efe16ed` and `a76269b`, with `BM_D8=stripe`; the treatment at HEAD, where the hardwired
  `lattice_class(x, y)` is `(x + y) % 4`. `git diff efe16ed HEAD -- agent_code/benedict_task4/`
  shows the danger branch changing from `elif D8_MODE == "stripe": target_dist = (x+y)%4`
  to `target_dist = lattice_class(x, y)` — the same function. Everything else removed was
  a switch whose default was off (`BM_ABLATE`, `BM_HUNT=1`, `BM_TIEBREAK=0`,
  `BM_TIE_TOL=0.0`, `BM_OPPDIST`). The update rule, `learning_rate`, and `warm_start` are
  byte-identical. `e37_arms.sh` exports `BM_CRATE=1.0 BM_STEP_COST=0`, which are HEAD's
  defaults. `EPS_DECAY` is applied per episode (`train.py:439`) and both arms ran 20 000
  episodes, so the ε schedule is identical — the annealing alternative is dead.
- **Warm parent.** `md5 q_table_parent.npy` = `md5 q_table_e36parent.npy` =
  `367ddf4ffbac96898680c193c200d8b7`. Same initial condition.
- **Replication.** Three independent groups of the control recipe on the guard field
  (`a4_control_history.out`):

  | group | n | rounds | score | suicides | survived | late hazard |
  |---|---|---|---|---|---|---|
  | E37 PLB2 s100–107, as evaluated in E37 | 8 | 1000 | 3.897 | 0.515 | 0.431 | 0.301 |
  | E37 PLB2 s108–114 (the seeds E42 did not use) | 7 | 1000 | 3.892 | 0.490 | 0.451 | 0.268 |
  | **E38 s120–124 @ep20000, trained after the cleanup** | 5 | 1000 | 3.976 | 0.478 | 0.457 | 0.235 |
  | E42 ctl s100–107, re-evaluated 08-19 | 8 | 300 | 3.832 | 0.503 | 0.434 | 0.292 |
  | **E42 mix** | 8 | 300 | 3.782 | **0.687** | **0.265** | **0.570** |

  The E38 row is the contemporaneous control E42 says it does not have: same field, same
  20 k budget, same parent, trained on the code the treatment ran on. It sits with the
  others. Re-evaluating the *same eight tables* three days later moves suicides by −0.011.
- **Caveat 1 (immaterial):** PLB2 is the arm E37 *selected* out of four on score against
  `rule_based`, so the control carries a winner's curse on the guard field. With
  between-seed SD 0.249 at n = 15 the expected inflation is ≈ 0.07 score — comparable to
  the −0.049 guard score difference, and irrelevant to the suicide/survival rows that
  carry the entry's conclusions.
- **Caveat 2 (immaterial):** across E37's four arms × 15 seeds — 60 tables, all trained on
  the guard field — guard suicides span 0.372–0.664 with arm means 0.503–0.534
  (`a7_e37_arms.out`). The mixed arm's 0.687 and hazard 0.570 sit outside that entire
  range. **The effect E42 measured is real.** Only its mechanism and its interpretation are
  not.

---

## 3. Smaller notes

- `perm_p` cannot return a p below 2/C(16,8) = 1.55e-4 under an exact test; the script
  prints `0.0000` where it should print `< 0.0002`. Cosmetic.
- `boot_ci` and `perm_p` are both fine as *tests*; the seed-level unit of inference is the
  right one and matches the pre-registered MDE. I found nothing wrong with that choice.
- The scorer's `crates_per_bomb` is a mean of per-round ratios. The ratio-of-sums moves the
  same way here (1.881 → 1.744 held out), so the estimator choice is not the problem —
  F3's problem is what the metric is being asked to mean.
- `zero_bomb_round` is 0.000 on all three fields; the `max(bombs, 1e-9)` guard never fires.

## 4. What I could not test

- **Whether a from-scratch (`BM_WARM=""`) mixed-field run behaves differently.** That is a
  training sweep and out of scope for this audit. It is the single experiment that would
  decide F7's first half.
- **Which rows the extra late-game suicides actually occur in.** Q-tables carry no visit
  counts, and attributing deaths to rows needs an instrumented rollout I judged not worth
  the CPU while another probe is running. F6 is therefore an inference from exposure and
  from the hazard's field pattern, not a direct measurement.
- **Whether the mixed arm's 1 330 mix-only rows are ever visited on a `rule_based` board.**
  Same reason. If they are not, the entire behavioural difference must live in the ~150
  rows where the argmax is not unanimous within each arm — which would make it a near-tie
  phenomenon of the kind audit 5 and E32 already documented (one row's argmax moving
  suicides 0.500 → 0.690).
- I did not re-derive E41's numbers; F7's magnitude arguments take E41's −1.0 margin as
  given.
