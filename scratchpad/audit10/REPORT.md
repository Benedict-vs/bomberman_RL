# Audit 10 — the 2026-08-17 strategy session, attacked

Ninth audit on this project; the eighth to overturn something. Briefed to **break** the claims, not
check them, and not told the author's hypothesis on any diagnostic question.

**Provenance of this file.** The audit ran as a subagent whose harness blocked it from writing
`.md` report files, so it returned its narrative in-channel and I (the main session) transcribed it
here unaltered in substance, then appended §5 — my own independent verification. Every `*.py` /
`*.out` / `*.csv` in this directory is the auditor's own work, written by it during the run.
Compute spent: 1300 rounds of rollout, no training.

---

## 1 · Verdicts

| claim | verdict | the number that decides it |
|---|---|---|
| A — the ceiling is +0.116 [+0.002, +0.233], CI excludes 0 | **OVERTURNED** | the interval excludes 0 for only 15.5 % of bootstrap seeds; sign-flip p = **0.054** |
| A — the oracle is an upper bound on hunting | **OVERTURNED** | its "inescapable trap" bombs earn us the kill **8.1 % [3.3, 16.1]** of the time |
| A — the direction is closed because +0.116 < MDE | **OVERTURNED** | on a between-agent margin the oracle is worth **+0.270 [+0.091, +0.452]** |
| A — the control reproduces the shipped agent | **SURVIVES** | paired score −0.017 [−0.256, +0.220]; `invalid` 4.771→4.734, `moves` 213.5→213.4 |
| B1 — kills scale inversely with the field's evasion | **OVERTURNED** | the takeable pool collapses 4.65× while our share of it falls only 1.47×; ordering non-monotone |
| B2 — 82 % of own-bomb deaths are opponent-induced | **SURVIVES, strengthened, cause re-attributed** | true isolation 0.118/100 bombs → **93 %, 14.3×**; but blocking is 8.8 %, opponent *bombs* 91.2 % |
| C1 — effective state count 290, so a 4× table is affordable | **OVERTURNED** | the shipped table has **8 186** rows with non-zero Q — 28× the "effective" count |
| C3 — "is BOMB safe" is a constant (99.08 %) | **WEAKENED** (conclusion survives, reason wrong) | not constant but **aliased**: 95.2 % of deadly-to-bomb steps share rows with safe ones |
| C4 — splitting digit 7 is 98.3 % noise | **WEAKENED** | trapped-safe conversion is worth +0.032 kills against 0.226 earned — most kills come from the "noise" half |
| C2 — `won` is a linear readout of score at +0.088/pt | **WEAKENED** | frozen-opponent partial derivative; mispredicts §7's own result by **3.7×** |

---

## 2 · Findings, most damaging first

### F1 · "+0.116 [+0.002, +0.233]" excludes zero only because of `analyze.py`'s hard-coded seed

`tools/analyze.py:220` seeds the percentile bootstrap with `default_rng(12345)`. That makes the
interval reproducible, not *stable* — the lower bound carries its own Monte-Carlo error and the
truth sits inside it (`d_ci_fragility.py` → `d_ci_fragility.out`):

```
n=4000  mean diff = +0.11575  se = 0.05988
analyze.py's own bootstrap (rng 12345, 10k):  [+0.0018, +0.2325]  -> EXCLUDES 0
200 other bootstrap seeds: lower bound mean -0.00153, sd 0.00158
   fraction of bootstrap seeds where the CI excludes 0: 15.5%
normal-theory 95% CI: [-0.0016, +0.2331]  t = 1.933
two-sided p (normal approx): 0.0532      sign-flip permutation p: 0.0543
```

The published table was reproduced exactly first (`c_analyze_repro.out`), so this is the same data
read with a different random seed. **By the project's own rule, `score` on k=4 is "not
demonstrated".** `kills` (p = 0.0025) and `won` (t = 3.6) survive. `TASK_B_argument.md` §7.3's
"+0.032 kills, +0.037 won, +0.116 score, all with CIs excluding zero" is wrong on its primary.

**Process fix worth more than the finding:** the fixed seed makes every borderline verdict on this
project look deterministic when it is a coin flip. Printing `t` or a permutation p beside the
interval would have caught this in one line.

### F2 · The oracle is not an oracle — it converts at 8 %, and that 8 % *is* the measured effect

`hunt_ceiling.trap_sites` asks whether the target can escape **from the tile it currently
occupies**. But `environment.py:421-432` hands every agent the same pre-action snapshot and only
then executes actions in a random permutation — so when our BOMB appears, the target has already
made a move the trap test never modelled. The trap is verified against a position being vacated.

`l_trap_followup.py` (imports the policy from `hunt_ceiling` so the trap definition cannot drift),
k=4, 200 rounds → `l_trap_followup.out`:

```
oracle 'inescapable trap' bombs placed : 86  (0.430 per round)
  target dead within the fuse          : 16 (18.6%)   95% CP [11.0%, 28.4%]
  ... and WE were credited with a kill : 7  (8.1%)    95% CP [ 3.3%, 16.1%]
```

The arithmetic closes: 0.406 override bombs/round × 8.1 % × 5 = **+0.16**, against +0.116 measured
after the −0.42 crate cost. Replacing *only* `trap_sites` with a simultaneous-move version
(`n_robust_trap.py` → `n_robust_trap.out`) gives 13 bombs, **61.5 % [31.6, 86.1]** target-death —
non-overlapping CIs, a 3.3× better class of site, 6.6× rarer. That is a **diagnostic**, not a
better policy.

> §7's +0.116 is the value of a one-step-stale heuristic with an 8 % hit rate. It is a **lower**
> bound on hunting, not the upper bound §7.3 asserts.

**Same primitive, three claims.** `escapable()` also underpins C3 (99.08 %) and C4 (0.42 %
trapped). One mis-calibrated function carries three conclusions.

### F3 · The ceiling was scored on our own score; on a between-agent margin it is 2.3× larger and robust

`NEXT_STEPS.md` §0.1 establishes the tournament is decided by **total score** — a ranking of four
agents' totals. A kill scores us +5 *and* removes a rival's future earnings; §7 measured only the
first half. All four slots are in the ceiling CSVs (`e_margin.py` → `e_margin.out`, n=4000 paired):

```
  mine         d= +0.1158 [-0.0027,+0.2335]  t=+1.93
  opp_best     d= -0.1542 [-0.2675,-0.0425]  t=-2.67
  opp_sum      d= -0.2037 [-0.3625,-0.0447]  t=-2.51
  margin_mean  d= +0.1837 [+0.0430,+0.3258]  t=+2.56   sign-flip p=0.0099
  margin_best  d= +0.2700 [+0.0895,+0.4502]  t=+2.95   sign-flip p=0.0031
```

Both margins exclude zero in **100 %** of 200 bootstrap seeds (`f_extras.out`) — unlike the
own-score number. The mechanism is a clean credit transfer: total opponent deaths unchanged
(+0.0057, t = +0.48), opponents' kills −0.049, our kills +0.032. **The oracle does not kill more
opponents; it takes deaths that were happening anyway and puts our name on them.**

`NEXT_STEPS.md` §3.4 sets the ceiling-test bar at +0.25. On the criterion the same document declared
correct twelve pages earlier, the hunt ceiling clears it.

Two caveats the auditor tested rather than asserted:
- **Post-hoc metric risk** — margin was chosen after seeing the score result; exactly two variants
  were looked at and both move together.
- **Selection** — k=4 was the pilot's best of five and rounds 0–999 are shared
  (`p_margin_freshsplit.out`): fresh arenas give score +0.102 t=1.47, `margin_best`
  +0.253 [+0.044, +0.462] t=2.38. The winner's curse is ~0.05 on score and the margin finding
  survives the test that finishes off the score one.

**Related over-read:** §7.3's "the k=8 arm bounds the 'chase harder' direction empirically — it was
worse". k8 − k4 on score is −0.134, t = −1.09; on margin k8 is +0.120 vs k4's +0.111. Nothing is
bounded (`f_extras.out`).

### F4 · B1 measures the size of the kill pool, not our aim

Every field CSV contains the opponents' own rows (`g_fields.py` → `g_fields.out`); `takeable` =
opponent deaths that were not self-inflicted:

```
field       n   ourkills  oppdeath  oppsuic  takeable  ourshare
peaceful   300     1.550     1.550    0.000     1.550   100.0%
coin_coll  300     0.733     1.530    0.723     0.807    90.9%
mixed      300     0.470     2.020    0.777     1.243    37.8%
rule_based 1000    0.226     1.820    1.487     0.333    67.9%
```

1. Kills fall 6.86× peaceful→rule_based. The **pool** falls 4.65×; our **conversion** falls 1.47×.
   `rule_based_agent` suicides 1.487 of its 1.820 deaths — 82 % of the field's mortality is consumed
   before we can reach it. "Bad at killing rule_based" is a 1.47× effect wearing a 6.86× number.
2. **Non-monotone inside its own table**: our share is 37.8 % in mixed vs 67.9 % against three
   rule_based. If kills tracked evasion quality that could not happen.
3. Against `peaceful_agent` the only way an opponent can die is our bomb, so 100 % is an identity.
   Crates/bomb there is 2.77 — the bombs are sited for the economy and the kills are random walkers
   stepping into a crate-clearing blast.

**What dies:** "we are not bad at killing", and "building a hunting feature against rule_based means
optimising against the hardest evader" — we already take two thirds of what is available.
**What survives:** §1's ranking of the field run, which F4 actually sharpens — what you need from an
external agent is its **suicide rate**, since that sets your takeable pool.

### F5 · B2 survives and is bigger than claimed; its causal breakdown is wrong

The claim asserts an "in isolation" baseline never measured. The auditor measured it (300 rounds,
`tools/evaluate.py`, `audit10_solo_noopp.csv`; `m_solo_ladder.py` → `m_solo_ladder.out`):

```
field                       n  suic/rd  /1e3 step  /100 bomb  oursteps  ourbombs  oppbombs
no opponents              300    0.040      0.102      0.118     390.3     33.96      0.00
3x peaceful (no bombs)    300    0.087      0.226      0.255     384.1     33.93      0.00
3x coin_collector         300    0.160      0.464      0.478     344.7     33.48     32.30
mixed (1 rule_based)      300    0.403      1.357      1.307     297.2     30.86     35.41
3x rule_based            1000    0.488      1.772      1.687     275.3     28.92     59.65

  our escape logic alone (empty board)         0.118   7.0% of the rule_based rate
  + bodies that block but never bomb           0.255  (+0.138,  8.8 % of the gap)
  + those bodies also bombing                  1.687  (+1.432, 91.2 % of the gap)
```

- **Conclusion right and understated**: 14.3× and 93 % opponent-induced, not 7.8× / 82 %.
- **Breakdown wrong where it matters**: §1(b) leads with "opponents blocking escape tiles" — worth
  **8.8 %**. Opponent *bombs* are **91.2 %**. §3.4's proposed ceiling arm happens to aim at the
  right 91 %, but the stated reasoning would equally have justified an opponent-as-obstacle digit.
- **Metric artefact**: `environment.py:239-262` loops over all explosions, so a death in overlapping
  own+enemy blasts fires `KILLED_SELF` *and* credits the opponent. Excess (Σkills + Σsuicides −
  Σdied) is +0.381/round against rule_based (32.0 % of rounds) and **exactly 0** against peaceful.
  **16 % of deaths in the rule_based field are double-counted, so `suicides` is not the same
  statistic across the two arms being compared.** Real; does not move the verdict (still 11.7×).

### F6 · "290 effective states" is the wrong statistic, and the shipped table says so

Probe reproduced (`i_state_visits.out`: 1385 rows visited, effective 298.8 vs the quoted 289.5 —
the stdlib-RNG floor, not a discrepancy). The problem is what it is asked to do: `NEXT_STEPS.md`
§3.1 uses it to remove the row-cost objection to a size-4 digit. The probe's own output refutes it:

```
rows with any nonzero Q in shipped table: 8186
distinct rows visited (eps=0):            1385
effective rows (exp entropy):              298.8
```

1. exp-entropy is a **concentration** measure, not a capacity measure — 60 rows carry half the
   steps, 99.9 % coverage needs 1333, and it weights the tail at ~0. The tail is exactly where a
   trap or hunting digit fires (trapped-safe = 0.57 % of armed steps).
2. It is measured at **ε = 0 on the converged policy**; the table is paid for during ε-greedy
   training. 8 186 rows carry learned values — 28× the "effective" 290.
3. At 20 000 × 156.7 steps that is 383 updates/row, ~64 per (row, action) at the mean. A 4× table
   takes it to ~96/row, and **E38 closes the usual remedy** (−0.770 at 300 k, 0/5 seeds).

**Internal tension worth naming:** §3.1 argues rare rows are cheap to add; §4 rejects the BOMB-safety
bit *because* it is rare. Both are "the tail is small". F2 settles it — an override firing on 1.56 %
of steps moved `margin_best` by +0.27.

### F7 · C3 — the BOMB-safety bit is aliased, not constant; conclusion holds for a different reason

`k_alias.py` → `k_alias.out`, 200 rounds:

```
'bombing here is certain death' on 189 armed steps (0.9107%) = 0.94 per round
rows that are ALWAYS deadly-to-bomb: 7 (9 steps)
rows carrying BOTH values (aliased): 20 (180 deadly steps, 285 safe steps)
-> 95.2% of deadly-to-bomb steps sit in a row that ALSO contains safe-to-bomb steps
deadly-to-bomb steps where the greedy action is BOMB (or an untrained tie):
  8 = 0.040 per round, against a measured suicide rate of 0.488 per round
```

**"The bit is a constant" is false** — the digit carries genuine new information. What kills it is
behavioural: the table already avoids BOMB in 97 % of those rows, so its addressable market is
0.040 of 0.488 suicides ≈ **8 %**. The right §4 entry is *"the table already avoids BOMB there —
0.040/round of 0.488"*, and that phrasing also tells you the 92 % a BOMB-safety bit would not touch.

### F8 · C4 — "98.3 % noise" presumes `escapable()` is ground truth for a kill

Ratio reproduced (`j_trap_opportunity.out`: in-range 24.70 %, trapped 0.578 %). But trapped-safe
opportunities are 0.59/round, the policy already bombs in 45.4 % of them, and converting *all* of
them is worth +0.032 kills — against 0.226 kills/round actually earned. **Even at 100 % conversion
the trapped states cannot account for most of our kills.** Most come from the in-range-not-trapped
states the claim calls noise (policy bombs there 57.4 %). Whether splitting digit 7 helps is
untested; the number offered to close it does not close it.

### F9 · C2 — the `won` slope is a frozen-opponent partial derivative

`survival_value.slope()` adds `d` to our score and leaves `best` alone — correct as written and
caveated in the script. The problem is the framing now in `AGENTS.md` (`o_won_slope.py` →
`o_won_slope.out`):

```
won == (our score >= best opponent)?  100.0% of rounds
slope applied to control margins, d=+0.116: dP(won) = +0.0000
actually measured for k=4:  dscore +0.1158, dwon +0.0373
predicted from '0.088 per score point':     dwon +0.0102 -> understates by 3.7x
the missing term: the best opponent's score fell by 0.1543
```

Scores are integers, so `won` is a step function and the honest prediction for +0.116 is 0.000. And
the slope is blind to anything that moves the opponents — i.e. every kill. **`NEXT_STEPS.md` §2.2
("any effect below +0.33 score is unreadable on `won` by construction") is directly falsified by
§7**, which read a +0.116-score effect on `won` at +0.037, t = 3.6 — a *stronger* signal than on
score. This does **not** revive `won` as primary; §0.1's reading of the PDF is untouched. It means
the stated reason is wrong, and that `won` responds to the **margin** — the same point as F3.

---

## 3 · Attacked and could not break

- **Control fidelity.** A first comparison looked catastrophic (2.822 vs 3.932) until the auditor
  noticed it had loaded `benedict_q_e28_F4`, an obsolete table. Against the real shipped E37 the
  k=−1 arm matches on every marginal *and* on two behavioural fingerprints that would expose a
  policy difference: `invalid` 4.771→4.734, `moves` 213.543→213.441, paired score
  −0.017 [−0.256, +0.220] (`b_control_fidelity.out`). §7's control validation is sound.
- **"Reproduces the mean" vs "reproduces the agent".** Per-round agreement is low (22.9 % identical,
  r = +0.152) but that is the known stdlib-RNG floor: an A/A test inside the harness gives
  +0.046 [−0.169, +0.262], the same spread (`a_ceiling.out` §2).
- **Pairing.** Real but nearly worthless — r = +0.125, `se_paired` 0.0599 vs `se_unpaired` 0.0640, a
  6 % narrower interval, not the large factor `evaluate.py`'s docstring promises. Not an error; the
  CI is valid.

## 4 · Not tested

Whether a *learned* hunting digit captures any of the margin gain (needs a sweep); a properly
specified oracle (the simultaneous-move model is a diagnostic at n=13); whether the margin effect
transfers to a non-`rule_based` field; arena-identity between the ceiling harness and `evaluate.py`
(seeding code paths verified identical instead, `hunt_ceiling.py:261-262` vs
`tools/evaluate.py:209-224`); the remaining §4 entries and `round_economy.py` / `phase_split.py` /
`denial.py` / `bomb_siting.py`.

The auditor stakes the audit on **F1, F2's 8.1 %, F3's margin, F4's pool decomposition, F6's 8 186**,
and explicitly declines to defend to three digits the 61.5 % robust-trap conversion (n=13) and the
8.8 %/91.2 % blocking split.

---

## 5 · Independent verification by the main session

Written from the CSVs directly rather than by re-running the auditor's code, so an error in its
harness cannot propagate into the check: `scratchpad/benedict/e39_audit_verify.py`.

| claim | status |
|---|---|
| **F1** — bootstrap-seed fragility | **CONFIRMED.** score t = +1.93, sign-flip p = 0.0531; CI excludes 0 in a minority of 200 seeds. Mechanism confirmed at `tools/analyze.py:220` (`default_rng(12345)`) |
| **F3** — the margin | **CONFIRMED exactly.** margin_mean +0.1837 t=2.56 p=0.0097; margin_best +0.2700 t=2.95 p=0.0034; both 100 % of 200 seeds. Fresh arenas: score +0.102 t=1.47, margin_best +0.253 t=2.38. Credit transfer: our kills +0.0318 t=+3.03, opp kills −0.0490 t=−3.21, opp deaths +0.0057 t=+0.48 |
| **F6** — 8 186 rows | **CONFIRMED exactly** from `agent_code/benedict_task4/q_table.npy` |
| **F4** — pool decomposition | **CONFIRMED.** peaceful/coin_coll/mixed reproduce to three decimals; rule_based row 0.214 kills / 69.5 % share vs the auditor's 0.226 / 67.9 % (different rule_based CSV), decomposition and conclusion identical |
| **F2** — the 8.1 % conversion | **NOT independently reproduced** — needs the instrumented rollout. Its *mechanism* is confirmed in framework code: `environment.py:421-432` polls all agents on the same pre-action snapshot, then executes in a random permutation |

Consequences applied in `experiments/benedict.md` E39.
