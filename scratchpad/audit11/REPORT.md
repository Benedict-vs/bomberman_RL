# Audit 11 — E40, "the hunt ceiling remeasured with a trap test that matches the game's move rule"

Target: `experiments/benedict.md` E40 (commit `5819f0f`). Harness
`scratchpad/benedict/hunt_ceiling_v2.py`, scoring `scratchpad/benedict/e40_analyze.py`,
conversion trace `scratchpad/benedict/e40_conversion.py`, data `scratchpad/benedict/e40/`.
Audit code and raw output: `scratchpad/audit11/*.py`, `scratchpad/audit11/*.out`.

---

## 1 · Verdicts

| # | claim | verdict | the number that decides it |
|---|---|---|---|
| C1 | The v2 trap test corrects a mis-specification against the game's simultaneous-move rule | **OVERTURNED** | The correctly-timed simultaneous-move test returns the **identical site set** to the E39 "stale" test: **0 disagreements over 20 022 searched steps / 1 949 (target, site) pairs** (`equiv.out`) |
| C2 | `k4 sim` is "the corrected oracle — the primary arm" | **OVERTURNED** | v2 loses **31 % of trap sites and 44 % of bomb opportunities** to two new errors; the arm the entry calls broken is the correctly specified one |
| C3 | "fixing it changed no conclusion … an instrument can be provably wrong and still return the right answer" | **OVERTURNED** | P2's null is not a break-even trade; the fix is a **no-op plus two bugs**, and P2's MDE (0.119) is 2x the largest difference the mechanism can produce (0.06) |
| C4 | "P1 REFUTED, and not marginally" (+0.053 against a +0.25 bar) | **OVERTURNED as evidence** | at the firing rate measured **before** the sweep, the arm's arithmetic maximum at its own P3 bar is **+0.200 < +0.25** |
| C5 | "hunting is closed … on an instrument entitled to close it" | **SURVIVES, re-attributed** | on the correct instrument (= the `k4stale` arm) score is **+0.075 [−0.009, +0.157]** and `margin_best` **+0.185 [+0.055, +0.314]** — still both under +0.25 |
| C6 | "Prediction (written before the run)" | **OVERTURNED as verifiable** | E39's *and* E40's entries both first appear in commit `5819f0f`, written after the sweep |
| C7 | Control-arm guard: "must reproduce the shipped agent … within ±0.12" | **WEAKENED** | paired on identical arenas the control is **+0.175 (contains 0)** against the 3.949 run and **+0.296 [+0.063, +0.529]** against the 3.828 run of the *same table* |
| C8 | P3's diagnosis: the residual mis-specification is the **static board** over the fuse | **WEAKENED** | at least two other causes are in the code and are not ranked (F6); untested |
| C9 | "Positioning to trap opponents is worth a real +0.163 on the between-agent margin" | **OVERTURNED as attribution** | the best-powered field effect is **opponents' suicides +0.046, t=+4.17** — twice our kill gain — and **79 %** of the oracle's bombs involve no walking at all |

**Bottom line.** The E40 sweep did not measure what the entry says it measured. Its "corrected"
primary arm is the only mis-specified one of the three trap models; the E39 test it replaced was
already correct with respect to audit 10's objection. The NO-GO on hunting survives — but it is
carried by the `k4stale` arm, and the entry's central methodological claim ("an instrument can be
provably wrong and still return the right answer") is exactly backwards. Separately, what survives
of the effect is mis-attributed: it is four-fifths opportunism rather than positioning (F10), and
its best-powered channel is opponents killing *themselves* more, not us killing them (F11).

**Note on this directory.** `a_pool.*`, `b_trapmodels.*`, `c_attrib.*`, `d_ctlfidelity.*`,
`e_arms.*`, `f_gap.*`, `g_k0.*` were written 09:24–09:37, before this session started — they are
the earlier audit-11 attempt, which got much further than "one tool call". I reviewed their code
and independently recomputed every number of theirs I cite (`oppside.py`, `attrib_id.py`,
`ctl2.py`, and a re-run of `g_k0.py`). The equivalence result in F1 is new to this session; the
earlier attempt tried a clock-only fix (`b_trapmodels.out`, `sim_clk`), saw no change, and stopped
before the lethal-destination filter that reveals the identity.

---

## 2 · Findings, most damaging first

### F1 — The simultaneous-move "fix" corrects a non-problem, and the two things it actually changes are both errors. Verified exactly: 0 / 20 022.

**The argument.** `escapable(p, hyp)` — the E39 test — is a BFS whose start node is depth 0 and
whose depth-`d` node is *the target's position at the end of step `T+d−1`*. Its depth-1 nodes are
therefore precisely the tiles the target can move to during step `T`, admitted under
`danger[n] > 0` (so a tile that burns at the end of step `T` is refused, correctly). "The target
moves to `c` and then escapes" is the path `p → c → …` that the BFS from `p` already walks.
The stale test **already models the target's step-`T` move**; audit 10's move-order objection does
not imply the trap test was wrong.

If that is right, the *correctly implemented* simultaneous-move test — escape-less from every cell
the target can **survivably** occupy after the step, with the clock **advanced by one** — must
return exactly the stale site set. I built four variants and compared the full site sets, per step,
against the shipped table driving the k=4 sim policy (so the state distribution is E40's primary
arm's):

```
uv run python scratchpad/audit11/equiv.py 300          # scratchpad/audit11/equiv.out
rounds=300 steps=86218 searched=20022
model     steps w/ site   %steps  steps w/ d=0  BOMBs/round  total (tgt,site) pairs
stale              1515    1.76%           124        0.413                    1949
sim                1058    1.23%            70        0.233                    1343
simfix             1077    1.25%            74        0.247                    1377
simfilt            1498    1.74%           120        0.400                    1915
simfix2            1515    1.76%           124        0.413                    1949

searched steps where stale != correct(simfix2): 0 of 20022
  pairs only in stale  : 0
  pairs only in correct: 0
searched steps where sim is NOT a subset of stale: 0
```

- `stale` = E39: `cells = [p]`, danger `hyp`
- `sim` = **E40's shipped "correction"**: `cells = target_cells(p)`, danger `hyp` (clock *not* advanced)
- `simfix` = clock advanced (`hyp − 1`)
- `simfilt` = destinations that are lethal at the end of step `T` dropped
- `simfix2` = **both, i.e. the correct test** — and it is **bit-identical to `stale`**

The probe reproduces the sweep: my `stale` 0.413 and `sim` 0.233 override bombs/round against the
arms' own `.meta.json` values of **0.420** and **0.228** (`e40_ctl/k4stale/k4sim…meta.json`).

**The two new errors, decomposed.**

1. **The clock is not advanced.** `trap_sites(simultaneous=True)` evaluates `escapable` from the
   target's post-move cell but passes the *pre-move* danger map. The start node is labelled
   depth 0 ("before the step-`T` move") while it *is* the position after that move, so the target
   is handed **one free extra move**. Timing from `environment.py` (`do_step` → `poll_and_run_agents`
   → `update_bombs` → `evaluate_explosions`) and `callbacks.danger_map`'s own docstring: a bomb
   placed in step `T` kills at the end of step `T+4`, and `danger` value `v` means "deadly at the
   end of step `T+v`". Cost: 34 of 1 949 pairs (`sim`→`simfix`).
2. **A step into a live explosion is counted as an escape.** `target_cells` filters only
   `field == 0` and bombs — not `danger == 0` — and `escapable` never checks the danger of its own
   start cell. So the test rejects a real trap because the target could "escape" by walking into a
   fire and out again. Cost: **572 of 1 949 pairs** — the dominant error.

Together: **v2 discards 31 % of trap sites and 44 % of the distance-0 bomb opportunities** relative
to the correct test. `sim ⊆ stale` strictly (0 violations of the subset relation), so the E40
primary arm is the E39 arm *minus* the sites those two bugs delete.

**How much this moves the conclusion.** It moves the *entire premise and the choice of primary
arm*, and about 30 % of the headline effect. Under the correct test the oracle is the arm labelled
`k4stale`: score **+0.075 [−0.009, +0.157] p=.074**, `margin_best` **+0.185 [+0.055, +0.314] p=.005**,
kills **+0.022 p=.002**, crates **−0.296 p=.014** (`scratchpad/benedict/e40/RESULTS.md`). The
NO-GO verdict is unchanged — both still miss +0.25 — but every sentence about "an instrument
entitled to close it", "the corrected oracle", and "a floor, not a ceiling" is attached to the
wrong arm.

**The consequence for P2 and for the entry's methodological headline.** The entry reads
`k4sim − k4stale = −0.022 [−0.104, +0.063]` as *"the fix trades a ~2x lower firing rate for a higher
conversion rate and those cancel — an instrument can be provably wrong and still return the right
answer."* What actually happened: the correction is a no-op, so all that remains is two bugs that
delete a third of the sites. The 12.0 % vs 8.4 % conversion is not the fix working, it is
**over-strictness acting as an accidental confidence filter** — and it is a bad trade in absolute
terms, because credited kills/round fall **0.036 → 0.024** (`e40/conv_stale.log`, `conv_sim.log`).
The right report sentence is the opposite of the entry's: *the instrument was already right, the
"correction" broke it, and the only reason the conclusion held is that the effect is small enough
that a 44 % loss of opportunity does not move it past the noise floor.*

### F2 — P1's +0.25 bar and P3's 20 % bar are arithmetically incompatible. Passing P3 at its bar *guarantees* failing P1.

`score` here is **exactly** `coins + 5 × kills`. Verified on all 32 000 agent-rows of the four arms,
max absolute residual **0.000000** (`uv run python scratchpad/audit11/arith.py`, `arith.out`):

```
ctl      n=8000 score=4.0086 coins=2.8274 kills=0.2362 coins+5k=4.0086 max|resid|=0.000000
k4stale  n=8000 score=4.0840 coins=2.7915 kills=0.2585 coins+5k=4.0840 max|resid|=0.000000
k4sim    n=8000 score=4.0616 coins=2.7872 kills=0.2549 coins+5k=4.0616 max|resid|=0.000000
k8sim    n=8000 score=3.9854 coins=2.7260 kills=0.2519 coins+5k=3.9854 max|resid|=0.000000
```

So the oracle's only score channel is the override bomb (×5) net of the coins it costs (×1).
The author measured that channel **before the sweep** — `e40/conv_sim.log` is timestamped 07:40,
the arm CSVs 08:33–08:37:

| arm | override bombs/round | conversion needed for a +0.25 score gain |
|---|---|---|
| `k4stale` | 0.429 | **11.7 %** |
| `k4sim` (the primary) | **0.200** | **25.0 %** |

P3's own pre-registered conversion bar is **20 %**, which at 0.200 bombs/round buys
**+0.200 < +0.25**. With the measured coin cost (−0.040, itself significant at p=.023) the
requirement rises to **≥ 29 %**. **P1 could not be passed by any oracle that merely met P3.**
The sweep re-derived what the 1000-round trace had already fixed an hour earlier —
`0.200 × 0.12 × 5 = +0.120` — and the realised kill delta (+0.019/round → +0.095 score, observed
+0.053 after the coin loss) matches it.

"P1 REFUTED, and not marginally" is therefore not a measurement of hunting. It is a measurement of
how often `trap_sites` fires, which F1 shows is 44 % too low.

### F3 — The bar was "unchanged from E39", but the intervention's opportunity count halved. Holding the bar fixed doubled the difficulty.

Same arithmetic (`arith.out`): E39's instrument needed **11.7 %** conversion to clear +0.25;
E40's needs **25.0 %**. The entry states the bar is "the same bar E39 pre-registered and failed"
and carries it unchanged into E41's P4 reopener. An absolute score bar is not a fixed standard when
the number of shots halves. The "correction" made the NO-GO strictly easier to reach and the entry
reads that as the instrument having been "wrong but harmless".

### F4 — The ledger's own proof of pre-registration does not exist for E40 (or E39).

`experiments/benedict.md` states the rule: *"Ich committe die Vorhersage, bevor ich messe — dann
belegt die Git-Historie die Reihenfolge."*

```
$ git show fd18450:experiments/benedict.md | grep -c "E39\|E40"     ->  0
$ git log --oneline --diff-filter=M -3 -- experiments/benedict.md
5819f0f  E40: hunting is closed ...      # 2026-08-18 09:12
fda0c98  E38: ...                        # 2026-08-17 16:27
```

E39's and E40's entries — predictions included — both first enter version control in `5819f0f`,
after the conversion trace (07:40) and after the arm CSVs (08:33–08:37).

This lands hardest on the one argument the entry singles out as its methodological win:

> "**Had margin not been pre-registered here, +0.270 would have looked like it cleared +0.25 and a
> digit would have been designed on it.**"

That argument's whole force is that the +0.25 bar on `margin_best` was fixed before `+0.163` was
known, and no artefact establishes it. `scratchpad/benedict/e40_field_design.md` (mtime 08-17 20:40,
before the sweep) does carry a +0.25 bar, but for the **E41** external-field rerun and on **score**.
The cited provenance, `NEXT_STEPS.md` §3.4, sets +0.25 **on score, for a different intervention**
(opponent-induced suicide) — so "matching §3.4's ceiling-test threshold" imports a score bar onto a
differently-scaled metric by analogy, not by derivation. `margin_best`'s realised SE (0.065) is 1.5x
`score`'s (0.042), so the same nominal bar is a different standard.

I cannot show the prediction was written after the fact. I can show the entry's own stated proof of
ordering is absent. Treat E40 as un-preregistered and say so in the report.

### F5 — P2 had no power to answer the question it was built for, and this was computable in advance.

P2 asks whether the corrected oracle is worth more than the broken one, with the "interesting
outcome" being sim > stale with a CI excluding 0. From the author's own pre-sweep trace, the
mechanism's entire budget for that difference is
`5 × (0.024 − 0.036) = −0.06` score/round. E40's own pre-registered MDE on score is **0.119** —
twice the largest difference the mechanism could produce, in either direction. The observed
−0.022 [−0.104, +0.063] was the only possible outcome.

Combined with F1, the reading "P2 PASSES as predicted, and it is the most interesting line in the
entry" should be: *P2 was a null test between two nested policies that agree on 98.9 % of steps,
run at an n whose MDE exceeds the mechanism's ceiling by 2x.* Nothing was learned.

### F6 — The residual mis-specification is attributed to one cause out of at least three, and the ranking is asserted, not measured.

The entry concludes: *"there is a third mis-specification … `trap_sites` evaluates a static board
while other agents' blasts destroy crates and open escape routes across the four-step fuse."*
Two other candidates are in the code and are not mentioned:

- **`danger_map` makes a blast permanently lethal.** `items.py:89-97` + `environment.py:203-240`:
  an `Explosion` is created with `timer = EXPLOSION_TIMER = 2, stage = 0`, kills in
  `evaluate_explosions` at the end of the step it is created and again the following step, then
  `next_stage()` moves it to `stage 1` and `is_dangerous()` goes False. **A blast is lethal for
  exactly two steps.** `danger_map` encodes only the *onset* (`v` = "deadly at the end of step
  `T+v`") and `escapable` treats every tile with `danger < SAFE` as unusable forever, so a tile
  that another bomb clears long before our fuse runs is refused as an escape. This **manufactures
  traps** that are not traps — the opposite sign to the crate story, and it fires in exactly the
  situations where trap sites are found (a board with bombs already on it).
- **`escapable` cannot WAIT.** It only models moving one tile per step, so an escape that requires
  standing still for a step while a blast clears is invisible. Also manufactures traps.

Both push the conversion rate *down* and both are shared with the shipped agent's own escape logic.
The entry's forward-looking sentence — *"A trap test that projected the board forward four steps …
would find rarer but truer traps"* — may well be right, but the entry has not established that
crates-over-the-fuse is the dominant term rather than a permanent-deadliness artefact. I did not
have the budget to run the conversion trace under a time-windowed danger model; see §3.

### F7 — The control-arm guard is not a test: it can pass or fail on the choice of reference, and the entry used the most favourable one.

The shipped E37 table has **two** committed 1000-round evaluations at seed 990731 —
`benedict_q_e37_PLB2_s106__ep20000` (**3.949**, the number the guard cites) and
`benedict_task4_shipped_e37` (**3.828**, the re-run `benedict_task4.md`:38 documents as the origin
of the ±0.12 noise floor). A third CSV, `benedict_task4_shipped` (3.718), predates the E37 ship
commit `a76269b` and is the *old* table — not a valid reference.

The guard was scored **unpaired**, 4.009 (n=8000) against 3.949 (n=1000), when the paired data
exist. Paired on the identical 1000 arenas (`uv run python scratchpad/audit11/ctl2.py`, `ctl2.out`):

```
e40 ctl n=8000 mean=4.0086  first-1000 mean=4.1240
ctl per-1000 block means: 4.124 4.080 3.910 4.054 3.804 3.976 4.140 3.981   (block SD 0.114)

PLB2_s106 (=the 3.949)     paired ctl-ref = +0.1750  t=+1.45  boot95 [-0.0610, +0.4140]  contains 0
shipped_e37 (=the 3.828)   paired ctl-ref = +0.2960  t=+2.51  boot95 [+0.0630, +0.5290]  EXCLUDES 0
```

So by the project's own criterion the control **fails** against one published run of the same table
and **passes** against the other. I could not demonstrate a real harness artefact: the control's own
1000-round block SD is **0.114**, i.e. the documented noise floor exactly, and the pooled gap
(4.009 vs the two references' mean 3.889) is +0.12 with SE ≈ 0.09, t ≈ 1.3. I checked the obvious
structural suspects and found none — `tools/evaluate.py:201-226` and `hunt_ceiling_v2.py:300-317`
run the identical loop (`world.rng = default_rng(base+r)`, `np.random.seed(base+r)`, `new_round`,
`while world.running: do_step`), `cb.act`'s greedy branch is reproduced byte-for-byte including
`TIE_TOL` and `POLICY_SEED`, and `state_to_features` never reads `game_state['step']` (so the
harness's one-step-earlier snapshot cannot shift the feature).

**Verdict: WEAKENED, not overturned.** The control is probably the shipped agent. But "inside the
±0.12 noise floor, guard passes" overstates what was done, and the same guard applied to the other
reference would have failed. The wider point for E41: **3.949 is the higher of two draws of the same
table**, and E41's P2 pre-registers external agents against it as "our 3.949". The correct reference
is the mean of the runs on record (**3.889**), or better, a fresh multi-thousand-round measurement.

### F8 — Two significant rows the entry does not report

`e40/RESULTS.md`, `k4sim − ctl`: **`coins` −0.040 [−0.077, −0.006] p=.023 (SCHLECHTER)** and
**`won` +0.018 [+0.004, +0.033] p=.015**. The entry's "Guards pass" paragraph lists suicides and
crates only. The coin loss is not cosmetic: it is 43 % of the gross kill gain (+0.095) and is the
reason score moves only +0.053. Reporting the guard set without it makes the oracle look free.

### F10 — E40 has no arm that isolates the strategy it is named after. E39 had one; dropping it was a regression.

The strategy under test, in E39's and E40's own words, is *"work one's way toward the nearest
opponent and try to kill it."* The oracle bundles that with a second thing: **bomb a trap site you
are already standing on**, which needs no hunting, no walking and no target feature.

E39 separated them with a `k = 0` arm and E40 dropped it. Re-run from the committed E39 CSVs in
`scratchpad/strategy/ceil/` (`uv run python scratchpad/audit11/g_k0.py`, `g_k0_rerun.out`,
n = 4000 paired arenas, stale trap model):

```
metric            k=-1      k=0      k=4              k0-ctl                  k4-ctl                   k0-k4
score            3.958    3.997    4.074   +0.039[-0.076,+0.154]   +0.116[-0.002,+0.233]   -0.077[-0.195,+0.041]
margin_best     -1.455   -1.369   -1.185   +0.086[-0.092,+0.263]   +0.270[+0.091,+0.449]   -0.184[-0.362,-0.006]
kills            0.229    0.234    0.260   +0.006[-0.014,+0.026]   +0.032[+0.011,+0.052]   -0.026[-0.046,-0.005]

bombs/round: k0 = 0.319 with ZERO walk steps ; k4 = 0.406 with 3.93 walks/round
```

**79 % of the oracle's bombs (0.319 of 0.406) are placed without taking a single step toward
anyone.** The walking half — the only half a `target_type`/hunt digit would have to learn — is
worth `k4 − k0` = +0.077 [−0.041, +0.195] on score and +0.184 [+0.006, +0.362] on `margin_best`,
neither of which E40 measures.

**Consequence.** E40's headline "positioning to trap opponents is worth +0.163 margin" is a
statement about a bundle whose larger component is opportunism, not positioning. If a digit were
ever designed here, the cheap version — "an adjacent/in-range opponent is escape-less" as a bit,
no navigation — is the part carrying most of the effect, and E40's design cannot see that. Adding
a `k = 0` arm to the sweep would have cost one of four lanes.

### F11 — The oracle's best-powered effect on the field is not our kills. It is that opponents blow themselves up more, and E40 does not report it.

Recomputed from the E40 CSVs, paired, n = 8000
(`uv run python scratchpad/audit11/oppside.py`, `oppside.out`):

| quantity | ctl level | k4stale − ctl | k4sim − ctl |
|---|---|---|---|
| our `kills` | 0.236 | +0.0222 [+0.008, +0.037] **t=+3.01** | +0.0186 **t=+2.53** |
| **opponents' `suicides`** | 1.487 | **+0.0456 [+0.024, +0.067] t=+4.17** | **+0.0348 t=+3.21** |
| opponents' `kills` (on anyone) | 0.559 | −0.0360 [−0.057, −0.015] t=−3.32 | −0.0339 t=−3.16 |
| opponents' total deaths | 1.832 | +0.0360 t=+4.23 | +0.0257 t=+3.02 |
| opponents' mean `score` | 2.984 | −0.0479 [−0.086, −0.010] t=−2.48 | −0.0438 t=−2.30 |

The single most significant thing the oracle does is raise the opponents' **own-bomb** death rate
by twice the amount it raises our kill count, while their kill credit *falls*. `margin_best`'s
+0.185 is therefore roughly half a denial effect on the field (−0.048 on their score) and half a
gain for us (+0.075, not significant on its own).

Two consequences the entry does not draw:

1. **"Positioning to trap opponents is worth a real +0.163 on the margin" is mis-attributed.**
   Most of the margin comes from opponents dying to themselves more often when we crowd them —
   a *pressure* effect on `rule_based_agent`'s escape routine, not a *trapping* effect.
2. **It is the most opponent-specific result in the entry.** A feature that pays by making
   `rule_based_agent` panic is exactly what E41's external-validity worry is about, and E41's P4
   reopener does not test for it.

### F12 — `kills` is a noisy proxy for "we caused this death" at the scale E40 works at.

`environment.py:238-263` credits **every** explosion owner whose blast covers the victim, and adds
`KILLED_SELF` on top when one of those owners is the victim. So one death can be the victim's
suicide *and* a +5 kill for us. Level check from the same CSVs
(`uv run python scratchpad/audit11/attrib_id.py`, `attrib_id.out`): in the control,

```
opp_deaths 1.8324   opp_suicides 1.4869   our_kills 0.2362   opp_on_opp_kills 0.5008
X (opponent deaths carrying more than one attribution) = 0.3915  -> 21 % of opponent deaths
```

A prior-attempt probe in this directory (`c_attrib.py`, code reviewed, wraps `evaluate_explosions`
and records the exact owner set per death; 1000 rounds) puts the share directly: **33.3 % (74/222)
of the opponent deaths we are credited for in the control had the victim standing in its own blast
as well.** Roughly 0.08 of our 0.236 kills/round is co-credit on a suicide.

**What I will *not* claim.** The same probe reads 38.5 % (101/262) under `k4sim`, i.e. the share
*rises*. That is ~1.8 SE and the inclusion-exclusion check above shows the multi-attributed count
`X` does **not** move significantly under any arm (k4stale −0.005, t = −0.53). So the oracle's
extra credit is mostly extra deaths, not extra double-counting. The finding is a **level** caveat
on the metric, not an explanation of the delta — but at an effect size of +0.019 kills against a
baseline where 21 % of deaths are multiply attributed, `kills` is not clean enough to carry a
mechanism story on its own.

### F13 — Things I attacked and could not break

Stated plainly, because a clean bill on these is useful:

- **`e40_analyze.py`'s statistics are sound.** The bootstrap is a straight percentile CI on the
  paired differences; the sign-flip permutation is correctly implemented (sign then mean, 5 000
  draws); the verdict requires both, and the seed-stability line is an honest disclosure of the
  2 000-resample Monte-Carlo error. I re-derived `score = coins + 5·kills` per round from the raw
  CSVs and every arm mean in `RESULTS.md` reproduces. Pairing keys are the round index, `common` is
  the full 8 000 in every comparison. I found no bug beyond the one the entry already fixed.
- **The margin arithmetic is right.** `margin_best` uses `max(opponents)` per round, `margin_mean`
  the mean; both recompute from the CSVs.
- **The claim that audit 10's +0.270 does not replicate holds** — the same stale arm at n=8000
  gives +0.185, and the shrinkage is the ordinary winner's curse.
- **The `k=8` crate cost is real** and much better powered than E39's (t = −7.28).
- **The oracle's override does not silently break the rest of the policy**: suicides fall, survival
  rises, and the walk step refuses lethal and illegal tiles.

---

## 3 · What I could not test, and why

- **Whether a time-windowed escape model (F6) raises or lowers the trap count and the conversion
  rate.** This is the interesting open question and it needs a conversion trace (~1000 rounds,
  following every override bomb to its fuse) plus a rewrite of `escapable`'s fatality rule. A
  12-run sweep was on four lanes; I kept to two short probes (300 + 150 rounds, ~5 min total) and
  did not run it. Prediction, recorded here so it can be scored later: a windowed model will find
  **fewer** traps than `stale`, not more, and the "floor, not ceiling" reading will not survive it
  either.
- **Whether the control's +0.18 gap against the shipped table's mean is a harness artefact.** The
  1000-round block SD is 0.114, so nothing under ~0.25 is readable at that n and I would need
  several thousand rounds of the *shipped agent through `tools/evaluate.py`* to settle it. That is
  a sweep, not an audit probe.
- **Whether the prediction text pre-dated the data.** Only the author knows; git does not.
- **Anything about external opponents (E41).** Out of scope and still running.

---

## 4 · What I would do with this

0. **Re-run the arms with a `k = 0` control** (F10) before any of the below. It is one lane, it
   splits opportunism from positioning, and E39 already had it — E40's design cannot answer the
   question it asks without it.
1. **Re-designate `k4stale` as E40's primary arm** and rewrite the entry around it: the E39 test was
   correct, `hunt_ceiling_v2 --trap-model sim` is the mis-specified one, and the NO-GO stands at
   score +0.075 [−0.009, +0.157] / `margin_best` +0.185 [+0.055, +0.314]. Delete the "provably wrong
   but harmless instrument" paragraph — it is the strongest-sounding sentence in the entry and it is
   false. The genuine methodological result is better: *a correction that is a no-op in theory can
   still be a 44 % regression in practice, and only a set-level equivalence check catches it.*
2. **Fix or retire `trap_sites(simultaneous=True)`.** If it is kept for E41's P4 reopener, it must
   advance the clock and drop lethal destinations — at which point it is `simultaneous=False` and
   the flag can go. As written, E41's P4 would run the hunt reopener on the broken model at 44 %
   of its true firing rate, against the same +0.25 bar.
3. **State bars as conversion rates or as effect sizes relative to the mechanism's budget**, not as
   absolute score. `0.25 score` means "≥ 11.7 % conversion" for one instrument and "≥ 25 %" for
   another; that is not a pre-registration, it is a moving target.
4. **Replace "our 3.949" with the mean of the runs on record (3.889)** everywhere it is used as a
   calibration constant, E41's P2 included.
5. **Report the opponent side of the table** (F11). `opp_suicides` and `opp_score` are in every CSV
   already, they are better powered than our own `kills`, and on this rung they are what
   `margin_best` is actually made of.
