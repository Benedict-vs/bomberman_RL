# Audit 6 — what to run next, and what to stop running

Commissioned to decide the next experiment independently, after four hypotheses for the
escape-ceiling gap were knocked down in a single session. Read: `AGENTS.md`, `MEASUREMENT.md`,
`experiments/benedict.md` lines 1-430 (E30-E33) **and E19 (3006-3175)**,
`experiments/benedict_task3.md`, `scratchpad/survey/REPORT.md`.

The agent's own file write was blocked by its harness; this report is its returned deliverable.
Its analysis script is on disk at `scratchpad/audit6/launder.py` and is re-runnable. **§0.1 and the
E19 finding were verified independently before any action was taken** — see the ledger.

---

## 0 · Corrections to the brief (measured)

### 0.1 The submittable agent was the rung-2 table, and it loses to `rule_based_agent`

```
md5 agent_code/benedict_task4/q_table.npy            = 412b8117c246ce9d19668ca246edbcf8
md5 agent_code/benedict_task3/q_table.npy            = 412b8117c246ce9d19668ca246edbcf8
md5 agent_code/benedict_task2/q_table.npy            = 412b8117c246ce9d19668ca246edbcf8
md5 checkpoints/benedict_task4/q_table_rung2ship.npy = 412b8117c246ce9d19668ca246edbcf8
```

| | score | won | kills | suicides | survived |
|---|---|---|---|---|---|
| **what was in `agent_code/`** | **2.890** | **0.178** | 0.119 | 0.403 | 0.078 |
| `rule_based_agent`, slot 0 of its own field | 3.254 | 0.286 | 0.196 | 0.533 | 0.376 |
| E33 control @20 000, 5 seeds | 3.719 | 0.372 | 0.221 | 0.616 | 0.333 |

The rung-4 result — **+0.83 score, +0.194 won** — existed only as `.npy` under `checkpoints/`,
which `.gitignore` excluded wholesale. Its stated justification ("reproducible from the commit
plus the `BM_*` variables") **is false from rung 3 on**: `main.py` does not seed the provided
opponents, and audit 5 established no E30-E33 run passed `--seed` either.

**Acted on.** `q_table_e33_ctl_s104__ep20000.npy` selected on validation seed 550731
(`won` 0.382), confirmed on ship seed 990731 (score 3.694, `won` 0.372 [0.343, 0.402]), installed
as `agent_code/benedict_task4/q_table.npy`, and the installed artifact re-evaluated with no `BM_*`
switches at 3.718 / 0.353 — the spread is opponent-RNG non-reproducibility, both inside the CI.
`.gitignore` gained a `!` exception for the parent with the reproducibility argument stated
correctly.

### 0.2 A DQN already exists — Ben's, on task 1, and it works

`agent_code/ben_coin_collector_dqn/`: full DQN with `dqn.py`, `model.py`, `replay_buffer.py`,
`action_mask.py`, `augmentation.py`, 8 test files, **31 hyperparameter arms**, **74 committed
evaluation CSVs**. Best: `v29_soft_target` @10 000, n = 1000 → **49.03/50 coins at 201.8 steps**
(tabular task-1 baseline 50.00 at 123.7).

"Model B has not been started" is wrong at the repo level. What is true: **no DQN has run on task
2/3/4**, and `BEN.md` has no entries. The two-model rule is a *team* requirement.
**Confirm with Ben before relying on it.**

### 0.3 Compute is not the constraint

20 000 episodes = **4 134 s (69 min)** single-process; five fit concurrently. A 3-arm × 5-seed
sweep is ~3-4 h plus evaluation. Five weeks buys ~30 sweeps. **Nothing should be ranked on cost.**

---

## 1 · What was measured

### 1.1 The ceiling is a *second fixed point* — not under-convergence, not a price

**(a) The table is at its own Bellman fixed point on the decisive row** (E31 s80 @20 000, 300 rounds):

| row, action | n transitions | terminal | Q | empirical 1-step target under π | residual |
|---|---|---|---|---|---|
| **55060, DOWN** (fatal; digit 6 says UP) | 1 457 | 74 | 7.867 | 7.780 | **−0.087** |
| 59160, DOWN | 1 486 | 0 | 8.261 | 8.277 | +0.016 |
| 35032, UP | 279 | 0 | 8.868 | 8.466 | −0.402 |

Death fires on 74/1457 = **5.1 %** of visits, priced correctly at ≈ −0.2, and DOWN still wins.
**Q-learning converged — to a worse policy.**

**(b) A second fixed point demonstrably exists at these hyperparameters.** Audit 5's own datum,
whose significance was missed: identical configuration, greedy-`DOWN` on E31's five seeds and
greedy-`UP` on two replications, **zero overlap** in suicides (0.587-0.747 vs 0.500/0.533),
indistinguishable training curves.

**This explains why every reward-side attack failed and will keep failing** — E32 (saturates,
middle dose collapses), E33 (follow rate moved exactly as designed, `won` fell), E19 (PBS, −9.17
crates). **A reward change moves where the fixed points are; it cannot move which one you land in.
Only an initial-condition change can.**

### 1.2 How far ahead the bootstrap sees (`launder.py`)

Pre-registered in the script docstring: *"mean `max_a Q(row_t)` is flat at ~5.2 for every k ≥ 2
steps before death, |Δ| < 0.5; falsifier: it falls monotonically over the last 4 steps."*

E31 s80 @20 000, 300 rounds, 79 543 alive steps, 209 deaths, 91 rounds ending alive. Baseline
`E[max_a Q]` = **8.800** (sd 0.991).

| k = steps before end | died (n=209) | survived (n=91) |
|---|---|---|
| 0 | **2.238** | 8.677 |
| 1 | 6.537 | 8.661 |
| 2 | 7.635 | 8.490 |
| 3 | 8.163 | 8.360 |
| 4 | 8.478 | 8.502 |

**Prediction REFUTED at k = 1 and k = 2** (Δ = −2.26, −1.17), confirmed from k = 4. The value
function *does* carry a warning, attenuated ≈0.54× per backward step — recorded as a refutation
rather than reinterpreted.

The consequence is sharper than the hypothesis it killed: **three steps before a death — where
78 % of deaths are still avoidable in 1-2 actions — the whole signal is −0.64, against a true
death cost of ≈ 11** (−4.33 realised plus a forfeited continuation this table values at 8.8).
Attenuation ≈ **17×**. Not a bug: the row is ~95 % survivable, so −0.64 *is* the right
expectation. **The information is not in the row, and no reward re-pricing puts it there.**

### 1.3 Where the remaining points are (n = 5000 rounds)

`score = coins + 5·kills` (2.613 + 5×0.221 = 3.719, exact).

- `COIN_COUNT = 9` → fair share 2.25. **We take 2.613, already above it.** Coin headroom is thin.
- **Kills are 30 % of our score and are priced at 0.0.**
- 1.847 opponent deaths/round occur in our field; we are credited with 0.221 (**12 %**); ~82 % are
  their own suicides. The pool is not free, but it is the only channel with 5× leverage.
- Ceiling arm: coins +0.22, kills +0.09, bombs 29.9 → 35.9. The +0.68 is ⅓ coins, ⅔ kills,
  arriving via **more bombs placed and survived**, not better bombs.

---

## 2 · The ranked plan

**Rank 0 — ship what was already measured.** Done; see §0.1.

**Rank 1 — E34: is the ceiling a reachable fixed point?** Warm-start from the forced table
(`scratchpad/audit5/mkesc.py` construction applied to each E33 *control* table), then train the
E33 control configuration **verbatim** — objective unchanged. The rule runs once, offline, on
initial conditions. Only instrument that can test §1.1. 5 seeds (110-114), checkpoints
**500 / 2 000 / 5 000 / 10 000 / 20 000** (the decay curve *is* the measurement).

- **P1 primary:** `won` @20 000 beats 0.372, t-CI excluding 0. **Magnitude 0.390-0.425.**
  *Falsifier: CI includes 0 → the forced policy is not a fixed point and **the +0.68 is
  unreachable by value learning on these eight digits**. That closes the line and is the most
  valuable negative available.*
- **P2 mechanism:** follow rate starts at 1.000 (ctl 0.634); **monotone decay settling in
  [0.70, 0.85]**. *Falsifier: within 0.02 of 0.634 → uniquely attracting fixed point.*
- **P3 interpretation split, pre-committed:** with *f* = follow rate @20 000 and *m* = fraction of
  re-pointed rows whose argmax moved back off digit 6 — *f* ∈ [0.70, 0.90] and *m* ≥ 0.30 →
  **learned**, report the initialisation and cite the survey's precedents; *f* > 0.97 and
  *m* < 0.05 → **training did nothing, the rule shipped by hand** → report as such and **do not
  ship it**, because that is the `AGENTS.md` prohibition wearing a `.npy` extension.
- **P4 guard:** `crates` ≥ 31.0 (E33's measured failure mode: the agent stops bombing).
  **The suicide guard is not reinstated** — E33 falsified it by intervention.
- **P5:** suicides @20 000 in [0.25, 0.50]. If P1 passes with suicides high, the mechanism is not
  the claimed one and the entry is inconclusive regardless of score.

**Rank 2 — E35: price the kill.** 30 % of score comes from an event priced 0.0 since E30, never
tested under a healthy configuration (E26 used `BM_KILL=25` with the step cost E31 showed
ratchets the argmax). 3 arms × 5 seeds (120-124): **K5** (`BM_KILL=5`), **K25** (`BM_KILL=25`, the
game's own 5:1 ratio), **PLB** (`BM_COIN` 5→7, `BM_KILL=0` — **placebo**, E32's carried-forward
requirement #4, never honoured). Middle dose deliberately omitted.

- **P1 primary:** `score` @20 000 paired vs ctl. **K25: 3.95-4.45; K5: 3.75-4.00.** *Falsifier:
  neither raises `kills` by ≥ 0.05 with CI excluding 0 → the agent cannot convert a kill price
  into kills on these digits, and the answer is a feature.*
- **P2:** `kills` monotone in `BM_KILL`; **K25 0.30-0.42** (pool bound from §1.3 caps cheap
  conversion near 0.35-0.45, not 1.8).
- **P3:** if PLB moves `score` by more than **half** K25's move, the entry is **inconclusive by
  construction** — declared before the run.
- **P4 guard:** `crates` ≥ 30.0 and `won` ≥ 0.34. **Not `suicides`.**

**Order: E34 first.** If P1 fails, that reframes E35 — a reward change on a table stuck in the
worse fixed point tells you about the fixed point, not about kills.

**Rank 3 — E36: the contested-escape digit, but resolve a contradiction first.** "49.3 % of deaths
are 'chose a safe tile, an opponent took it'" versus measured `killed_by_opponent` = 0.051/0.667 =
**7.6 %**. Both hold only if the 49.3 % are *own-bomb* deaths where an opponent **blocked** the
escape square — movement contention, not enemy bombs. Then the right digit is **"digit 6's next
tile is contestable — an opponent within one step of it"**, one bit, factor-2 `warm_start`.
**Gate (free, from `scratchpad/audit5/deaths_e31_s80_ep20000.pkl`): classify all 209 deaths; if
the middle bucket is < 20 %, drop E36.** Ranked third because three survey ablations say reduction
beats richness (M 72→12, C 10⁵→256, E 2²⁰→335 at the 5.04 benchmark) against our 64 000 rows.

**Rank 4 — E37: truncation is not termination.** §1.2 measures the cost: survivors' `max_a Q` sits
at 8.4-8.7 through their last eight steps, so a surviving round's terminal cell is targeted at ≈ 0
when its continuation is worth ≈ **8.8**, on **33.3 %** of rounds. E31 bounded this at ≲0.2 Q
units — but against the *action-gap collapse*, a different question. Primary `won`, paired,
5 seeds. *Falsifier: CI includes 0 → record as fixed-and-neutral and stop citing it as an open
bug.* Ranked last because it moves every baseline and must not land between E34 and E35.

---

## 3 · What NOT to do

1. **Potential-based shaping with Φ = −distance to safety** (the E34 that was about to be written).
   Three reasons, the second free and fatal:
   - **E19 already measured PBS on this exact table**: `shape02` = −9.17 crates, `shape10` =
     −85.83 (t = −13.87). Its finding 3: *"shaping injects within-row noise of exactly the size it
     was meant to remove."* `train.py:260-261` carries the conclusion **in the source**: *"a row of
     this table is a bucket of states with different Phi, so the offset does not cancel between
     actions the way the theorem needs."*
   - **The proposed Φ is, up to bucketing, already digit 8** ("how far digit 6's target is").
     Where Φ is a function of the row, PBS is provably inert; where it isn't, it is E19's noise.
     **Neither branch delivers +0.68.**
   - *Gate if anyone insists:* measure `sd(BFS distance to safety | row)` over danger rows. **If
     < 0.5 the arm is a no-op and must not be run.**
2. **Any further death-price or suicide sweep.** Saturation measured (flip threshold 1.08; −10 and
   −25 flip the same 90.4 % of cells), middle dose collapses, `won ~ suicides` slope +0.038
   [−0.054, +0.143], and E33 **intervened**: suicides 0.616 → 0.496 and `won` fell.
   `rule_based_agent` suicides 0.533 in its own field. **Guard stays retired.**
3. **`BM_TIE_TOL > 0`** anywhere. E31: recovers no score, destroys it monotonically.
4. **A raw-board CNN DQN from scratch.** §0.2 — one exists and is measured; the survey records
   that *every* unscaffolded raw-board DQN in the corpus failed. **What is missing is the write-up
   and a task-2 evaluation of the existing one — a report task, not an experiment.** Conditional
   on confirming with Ben.
5. **Targeting the published 5.04.** Different field compositions, and no source in that corpus
   reports a confidence interval. Our measured bar is `won` = 0.283 and we are at 0.372.
6. **More rung-4 arms after E34 + E35.** If both fail their primaries, tabular Q on these eight
   digits is at its ceiling at 3.719 / 0.372 — **which already beats the reference** — and the
   remaining weeks belong to the report. **Two clean pre-registered negatives plus the fixed-point
   diagnosis is a better chapter than a third 0.02 of `won`.**

---

## 4 · Defects

| | defect | severity | status |
|---|---|---|---|
| (a) | `agent_code/benedict_task4/q_table.npy` was the rung-2 table; rung-4 tables gitignored and not reproducible | **blocking** | **fixed** |
| (b) | `tools/evaluate.py:258` `killed_by_opponent = max(0, died − suicides)` charges an overlapping own+enemy blast wholly to suicide | must land **before E36**, whose primary metric it is | open |
| (c) | `KILLED_OPPONENT = 0.0` | not a fix — it is **E35's arm** | — |
| (d) | truncation-as-termination in `end_of_round` | **E37**; §1.2 puts it at ≈8.8 Q on 33.3 % of rounds | open |

(b) is additive: read the `killers` list rather than differencing two counters. It changes no CSV
already written, but every `killed_by` magnitude quoted since E28 needs a correction note.

---

## 5 · The decision

The table sits at its own Bellman solution on the decisive row (residual −0.087 over 1 457
transitions) while a second solution demonstrably exists at identical hyperparameters. Reward
changes relocate fixed points and have failed four times running; **only an initial-condition
change selects among them.** So: run **E34**, a warm start from the forced table under an
unmodified objective, with P3 pre-committing how the report must describe every outcome. Run
**E35** beside it, because a third of the score comes from an event priced at zero and the placebo
finally makes that a real experiment. **And if both fail their primaries, stop** — 3.719 / 0.372
against a 0.283 bar is a result, and the remaining weeks are worth more spent on the chapter than
on the fourth decimal.
