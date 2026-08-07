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

## E12 — A learning curve measured at ε = 0, and what the round count should be

- **Question:** the E11 training curve rises to ~34 crates by episode 10 000 and then oscillates
  around it for 30 000 more. Is that convergence, or is the training curve hiding progress the
  greedy policy is still making? And what round count should every experiment after this one use?
- **Change:** none to the agent, the features or the rewards. `train.py` additionally writes the
  table at five checkpoints. Learning is untouched — the same updates in the same order — so this
  entry's answer transfers to E13 onward.
- **Agent:** `benedict_task2`, the E11 configuration exactly · commit `<fill in>` · labels
  `benedict_q_e12_s{0..4}__ep{2500,5000,10000,20000,40000}__task2`
- **Training:** one 40 000-round sweep, five seeds, world seed 810731. Checkpoints at
  **2 500 / 5 000 / 10 000 / 20 000 / 40 000** episodes.
- **Measurement:** 25 evaluations, 300 rounds each, ε = 0, seed 20260731, `--preset task2`, plus
  the loop probe at every checkpoint.

### Why this is worth a full entry rather than a quick check

Every convergence claim in this ledger rests on a *training* log, and `AGENTS.md` has said since
2026-08-05 that a training curve is not a result — with rung-1 evidence where the gap was
catastrophic (48.2 coins in training, 1.45 at evaluation, `experiments/maxi.md` E01). This
produces **the first learning curve in the project measured at ε = 0**, which turns that rule
from a warning into a quantity: how wrong is the training curve, as a function of training time?

It also happens to answer the round-count question, but that is the by-product. A 10 000-round
run would have been the cheap way to ask it, and it is strictly worse: because both RNGs are
seeded and no update depends on the total round count, a 10 000-round run *is* the first 10 000
episodes of the 40 000-round run. Checkpointing gets five answers from one sweep instead of one.

### Why the plateau is expected, and why that is not the same as "converged well"

`α = 1/N(s,a)^0.7`. Seed 0 ran 9.8 M steps over 432 occupied rows, so a busy cell is visited
~22 700 times:

| visits | α |
|---|---|
| 3 000 (≈ episode 5 000) | 0.0037 |
| 23 000 (episode 40 000) | 0.00088 |

The high-traffic cells are frozen by episode ~5 000, and ε reaches its 0.02 floor at 4 605 — two
schedules landing in the same place by coincidence. So a plateau is what the hyperparameters
*predict*. What that does not tell me is whether the frozen policy is the best one seen: E06 arm A
established that a table can get worse with more training, and every E11 seed ends in a 2-cycle,
which is a property of a hardening argmax.

### Prediction (written before the run)

1. **`crates` at 10 000 episodes is within 15 % of its value at 40 000** — so ≥ 22 against E11's
   26.19 five-seed mean. **Refutation:** below ~18 while the training curve already reads 34 at
   episode 5–10 k means the training curve *overstates* convergence badly and the greedy policy
   improves long after it flattens — the rung-1 failure in the opposite direction, and the round
   count stays at 40 000.
2. **`crates` at 2 500 is much lower, 10–18.** ε is still 0.06 there and the table is half-formed.
3. **The ε = 0 curve is not monotone.** At least one seed scores lower at 40 000 than at its own
   20 000 checkpoint. This is the prediction only an ε = 0 curve can test, and the mechanism is
   E06 arm A's: as α freezes, an argmax that happened to settle wrong stays wrong.
4. **The training/evaluation gap is roughly constant at 0.8–0.9** (E11 s0: 33 in training against
   28.8 measured). If it holds across all five checkpoints, then on *this* rung the training
   curve is a biased but usable convergence signal — a qualification of the `AGENTS.md` rule
   rather than a contradiction of it, and worth writing down as such.
5. **Seed spread does not shrink with training.** `crates` std across seeds ≥ 5 at every
   checkpoint from 10 000 on (E11 at 40 000: 8.78). The seeds are not converging to one policy,
   they are converging to different ones — s2 collapses into its cycle at step 11, s0 at 396.
6. **The 2-cycle gets *worse*, not better, with training.** Loop-probe cycle entry step at 40 000
   is no later than at 10 000 for at least three of five seeds. A near-tie broken by exploration
   early becomes a hardened argmax late.

**Refutation condition for the whole entry:** `crates` still rising by more than 5 between the
20 000 and 40 000 checkpoints, with a paired CI excluding 0. That would mean the plateau I read
off the training log is an artefact of ε-driven variance masking real progress, and both the
round-count conclusion and prediction 4 fall with it.

### What I do with the answer

If prediction 1 holds, **10 000 rounds becomes the standard for E13 onward** — 9 minutes per
sweep instead of 36, which is the difference between three experiments a day and one. If
prediction 3 or 6 holds, the ship-the-final-table convention needs revisiting: "train to 40 000
and ship" would be shipping a knowingly worse table than a checkpoint that was already measured.

### Result

25 evaluations, 300 rounds each, ε = 0, seed 20260731. Five-seed means:

| checkpoint | `crates` | std | `score` | `suicides` | `survived` | crates/bomb |
|---|---|---|---|---|---|---|
| 2 500 | 19.73 | 8.43 | 1.178 | 0.022 | 0.978 | 0.86 |
| 5 000 | 20.61 | 6.18 | 1.283 | 0.039 | 0.961 | 1.07 |
| 10 000 | 21.70 | 12.44 | 1.363 | 0.002 | 0.998 | 1.37 |
| **20 000** | **33.32** | **4.90** | **2.287** | 0.001 | 0.999 | **1.91** |
| 40 000 | 26.19 | 8.78 | 1.723 | **0.000** | **1.000** | 1.62 |

Per seed, `crates`:

| ep | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| 20 000 | 36.07 | 38.28 | 34.24 | 25.41 | 32.61 |
| 40 000 | 28.82 | 32.10 | **10.67** | 28.89 | 30.45 |

Paired, 20 000 → 40 000: s0 `crates` −7.247 [−12.133, −2.383] **WORSE**, `score` −0.697
[−1.073, −0.317] **WORSE**; s2 `crates` −23.563 [−27.800, −19.507] **WORSE**, `score` −1.857
[−2.200, −1.527] **WORSE**.

**The best table is the one at 20 000 episodes. Training to 40 000 costs 21 % of the crates and
25 % of the score, and four seeds of five get worse.** The reference is still 116.26 / 8.45, so
the peak is 29 % of it.

### The curve does not converge — it peaks and declines

Nothing in this ledger predicted that. The refutation condition I wrote was that `crates` might
still be *rising* between 20 000 and 40 000; instead it falls, with a CI far from zero on both
seeds tested. That is the E06 arm A finding in a new form: a table can get worse with more
training, and the only way to see it is to measure at ε = 0.

**Mechanism, and it is the defect E13 already targets.** Seed 2 is the clean case:

| | ep 20 000 | ep 40 000 |
|---|---|---|
| `crates` | 34.24 | 10.67 |
| `bombs` | 17.42 | 5.27 |
| crates per bomb | 1.97 | 2.02 |
| loop-probe cycle entry (median step) | 111 | **11** |

It does not bomb *worse* — the crates-per-bomb ratio is unchanged. **It stops playing sooner.**
As α shrinks (0.0037 at ~3 000 visits, 0.00088 at ~23 000) and ε sits at its floor, near-ties in
the safe-branch rows harden into a fixed argmax, and when that argmax closes a 2-cycle the round
is effectively over. More training does not make the policy wrong; it makes an already-wrong
tie permanent. The thing that degrades with training is exactly the constant digit 6 that E13
exists to fix.

### The training curve is not a usable proxy, and the error is not constant

Seed 0, training mean over the 1000 episodes before each checkpoint, against the ε = 0 value:

| checkpoint | training | ε = 0 | ratio |
|---|---|---|---|
| 2 500 | 16.46 | 12.69 | 0.77 |
| 5 000 | 34.32 | 20.38 | **0.59** |
| 10 000 | 32.59 | 19.25 | **0.59** |
| 20 000 | 39.74 | 36.07 | 0.91 |
| 40 000 | 34.36 | 28.82 | 0.84 |

I predicted a roughly constant 0.8–0.9. It ranges 0.59 to 0.91, and it is *worst exactly where
the policy is least settled* — at 5 000 and 10 000 the training log overstates the greedy policy
by 40 %. So the training curve is not a biased-but-usable convergence signal on this rung either;
it is a signal whose bias depends on the thing being measured. `AGENTS.md`'s rule survives
without the qualification I was hoping to add to it.

It is also why I misread the plateau in the first place. The training curve looked flat from
10 000 onward; the ε = 0 curve rises by 50 % between 10 000 and 20 000 and then falls.

### Predictions, scored

1. **"`crates` at 10 000 within 15 % of 40 000 (≥ 22)" — narrowly wrong (21.70 vs 26.19, 17 %
   off), and wrong at the premise.** I framed 40 000 as the endpoint worth matching; it is not
   the best table, so the comparison was the wrong one to make.
2. **"`crates` at 2 500 much lower, 10–18" — wrong.** 19.73, statistically indistinguishable
   from 5 000 (20.61) and 10 000 (21.70). **The first 2 500 episodes buy 59 % of the peak and the
   next 7 500 buy nothing.** All the remaining progress happens between 10 000 and 20 000.
3. **"The ε = 0 curve is not monotone; at least one seed lower at 40 000 than at its own
   20 000" — right, and far stronger than predicted.** Four of five.
4. **"Training/evaluation gap roughly constant at 0.8–0.9" — wrong.** 0.59–0.91, see above.
5. **"Seed spread ≥ 5 at every checkpoint from 10 000" — narrowly wrong**, and the miss is the
   interesting part: the spread is *smallest* at the best checkpoint (4.90 at 20 000 against
   12.44 at 10 000 and 8.78 at 40 000). The seeds agree when the policy is good and diverge when
   it is not, which makes std a cheap secondary signal.
6. **"Cycle entry at 40 000 no later than at 10 000 for ≥ 3 of 5 seeds" — wrong.** Only s2. The
   degradation is not a uniform hardening; it is one seed of five falling off a cliff.

### Reproducibility confirmed as a by-product

E12's 40 000 checkpoint is **round-for-round identical** to E11's final table (per-round `crates`
compared across all 300 evaluation rounds). Two independent 40 000-round training runs, same
world seed and same `BM_RUN_INDEX`, produced the same table — so the seeding regime introduced in
E06 holds on rung 2, and a checkpoint really is what a shorter run would have produced.

### Tooling note

`analyze.py --ablation` was the wrong tool for plotting this curve. It labels every non-base file
as a "component removed" and prints **HARMFUL** for anything better than the base, which is
meaningless here and must not reach the report. The numbers are correct paired differences; only
the framing is wrong. A `--series` mode that plots one metric against an ordered axis with CIs is
worth adding before the report needs this figure.

### Verdict

**20 000 rounds is the standard from E13 on** — not the 10 000 I expected (21.70 crates is far
short of 33.32) and not 40 000 (which is actively harmful). Sweeps drop from ~27 min to ~13.

Second finding, and the one for the report: **on this rung, "train longer" is not free and the
last checkpoint is not the best one.** The convention of shipping the final table is wrong here;
what should ship is the best *measured* checkpoint, chosen the way E06 fixed the seed convention
— by a rule declared in advance, not by picking the winner afterwards. Proposed rule, declared
now: **ship the checkpoint with the highest five-seed mean on the rung's leading indicator, using
run index 0 within that checkpoint.**

### What I do next

1. **E13 (digit 6's safe branch) at 20 000 rounds.** Its mechanism and E12's degradation
   mechanism are the same 2-cycle, so E13 should shrink the 20 000 → 40 000 gap as a side effect.
   Prediction to be written before the run, with that as a secondary check.
2. **Add the checkpoint rule to `AGENTS.md`**: measure at ε = 0 at several checkpoints, ship the
   best by a pre-declared rule. This is a team-wide methodology point, not mine alone.
3. **`--series` mode for `analyze.py`**, before the report needs the curve figure.

---

## E11 — Point digit 6 at the exit when the agent is in a blast

- **Question:** E10 established that the agent declines to bomb because, under its own escape
  policy, bombing is negative-EV. Does giving it a *direction out* flip that calculation — and by
  how much does the leading indicator move?
- **Change from E10:** exactly one. Digit 6 gains a danger branch: when the agent's own
  tile is covered by a live bomb (digit 5 > 0), digit 6 is the first step of a shortest path to a
  tile no bomb reaches, found by a time-aware BFS (at depth *k* a tile may only be entered if it
  is still survivable after *k* moves). The coin and crate branches are untouched, `FEATURE_SIZES`
  is untouched, the rewards are untouched, and the table stays at 12 800 rows.
- **Agent:** `benedict_task2` · commit `<fill in from .meta.json>` · labels
  `benedict_q_e11_s{0..4}__task2`
- **Training:** 40 000 rounds, `classic`, world seed 810731, `BM_RUN_INDEX` 0–4. Identical to E10
  in every other respect, so this is a paired comparison against it.
- **Measurement:** `results/eval/task2_crates/benedict_q_e11_s{0..4}__task2.csv`, 300 rounds,
  seed 20260731, `--preset task2`.

### Why this is the one change, and why it is defensible

The E10 arithmetic: a bomb pays ≈ +0.50 after γ⁴ discounting, against ≈ −1.50 from a 27–33 %
escape-failure rate at `KILLED_SELF` = −5. Only the second term is worth attacking, because the
first is bounded by the crate count and the third — navigation — is nearly free on a board where
99.7 % of tiles are already bombing spots.

**On the "feature returns the best action" rule.** `final_project.pdf` forbids *"a feature that
deterministically returns the action which results in the best move"*, and lists as *sanctioned*
examples "pathfinding features, e.g. the direction to move which brings you closest to the
nearest coin" and "life-saving features, e.g. whether or not the agent is in the path of a bomb".
An escape direction is the same construction as the coin direction already shipped in
`tabular_q_task1`, and it does not return the best move: standing still, collecting a coin on the
way, or bombing again can all beat running, and the agent still has to weigh them. It is stated
here so the report argues it rather than hoping nobody asks.

### Prediction (written before the run)

1. **`crates` rises to 20–60 per round**, five-seed mean, from 2.41. Mechanism: with a reliable
   exit, P(die | bomb) falls below ~0.1, the −1.50 term becomes ≈ −0.5, and `BOMB` turns
   positive-EV. **Refutation:** `crates` < 5 means the EV story is wrong and the binding
   constraint is the reward scale, not the escape — in which case E12 becomes `CRATE_DESTROYED`
   rather than the crate target.
2. **`suicides` 0.1–0.5, not 0.** The new digit helps *after* the bomb is down. Nothing tells the
   agent, at the moment it presses `BOMB`, whether an escape will exist once its own bomb is on
   the board. It has to infer that from the wall pattern in digits 1–4, which is learnable — a
   dead end has three blocked neighbours — but only from experience. **I expect the remaining
   suicides to concentrate in rows with three blocked neighbours**, and that is checkable in the
   table rather than by another 300-round run.
3. **`score` 1.5–5.0**, against 0.109 and a reference of 8.45. Crates open, coins become visible,
   and digit 6 switches to the coin branch on its own.
4. **The 2-cycle is reduced but not gone.** Loop probe: median distinct tiles per round **> 15**
   (from 2), rounds entering a 2-tile cycle **< 50 %** (from 92 %). It cannot go to zero, because
   while the agent is safe with no bomb available and no coin visible, digit 6 is still pinned at
   0 by the crate branch — that is E12's job, and the residual measured here is the size of the
   prize for doing it.
5. **`table_check` finding 3 (in a blast, a way out exists, the argmax does not take it): < 10 %**
   of trained danger rows, from 26.6–33.2 %. This is the direct static signature of the change; if
   it does not move, the branch is not wired up and nothing else in the entry means anything.
6. **Seed spread stays wide: `crates` std < 15.** E10's std was 2.27 only because all five seeds
   failed; a real effect should be large and variable before it is large and stable.

**Refutation condition for the whole entry:** `crates` < 5 *and* finding 3 below 10 %. That
combination means the escape digit works, the agent still will not use it, and the problem was
never the escape — it is the reward, and I have been optimising the wrong term.

### Gate order, before any 300-round evaluation

1. `table_check` finding 5 (BOMB rows) — must exceed E10's 17–34.
2. `table_check` finding 3 — prediction 5.
3. The loop probe — prediction 4.

E10's lesson is that a static row count is not a visitation distribution, so gate 3 outranks the
other two: it is the only one that rolls out the policy.

### Result

40 000 rounds per seed. Evaluation at ε = 0, 300 rounds, seed 20260731:

| seed | `score` | `crates` | `bombs` | crates/bomb | `suicides` | `survived` | `invalid` |
|---|---|---|---|---|---|---|---|
| s0 | 1.777 | 28.82 | 27.20 | 1.06 | 0.000 | 1.000 | 0.20 |
| s1 | 2.137 | 32.10 | 16.85 | 1.90 | 0.000 | 1.000 | 0.24 |
| s2 | 0.527 | 10.67 | 5.27 | 2.02 | 0.000 | 1.000 | 0.01 |
| s3 | 2.000 | 28.89 | 14.46 | 2.00 | 0.000 | 1.000 | 0.03 |
| s4 | 2.173 | 30.45 | 17.13 | 1.78 | 0.000 | 1.000 | 0.13 |
| **mean** | **1.723** ± 0.686 | **26.19** ± 8.78 | 16.18 ± 7.82 | 1.75 | **0.000** | **1.000** | — |

`think_max_ms` 0.147–0.194. Paired against E10 (s0 vs s0, 300 rounds):

| Metric | E10 | E11 | diff | 95 % CI | verdict |
|---|---|---|---|---|---|
| `score` | 0.213 | **1.777** | +1.563 | [+1.320, +1.817] | **BETTER** |
| `crates` | 4.69 | **28.82** | +24.130 | [+20.760, +27.593] | **BETTER** |
| `bombs` | 2.91 | **27.20** | +24.287 | [+21.667, +26.943] | **BETTER** |
| `suicides` | 0.063 | **0.000** | −0.063 | [−0.093, −0.037] | **BETTER** |
| `survived` | 0.937 | **1.000** | +0.063 | [+0.037, +0.093] | **BETTER** |

Against the reference (`coin_collector_agent`): `score` −6.723 [−6.973, −6.463] WORSE, `crates`
−87.44 [−90.52, −84.19] WORSE, `suicides` and `survival` tied at 0.000 / 1.000.

**One change, 10.9× the crates and 15.8× the score, and every death removed.** The EV argument
from E10 was right: the binding constraint was the escape, and moving it flipped `BOMB` from
negative- to positive-EV without touching the reward table.

### Predictions, scored

1. **`crates` 20–60 — right.** 26.19.
2. **`suicides` 0.1–0.5 — wrong, and better than predicted.** 0.000 across 1500 rounds, with
   100 % survival. See below; the reasoning behind the miss is the interesting part.
3. **`score` 1.5–5.0 — right**, at the bottom of the range. 1.723.
4. **"2-cycle reduced but not gone: median distinct tiles > 15, cycles in < 50 % of rounds" —
   half right, and the second half badly wrong.** Distinct tiles per round median 30 · 9 · 4 ·
   25 · 6 (E10: 2), so two seeds of five clear the threshold. But **20/20 rounds still end
   confined to two tiles**, in every seed. What changed is *when*: entry step median 396 · 46 ·
   11 · 110 · 23, against E10's 0–23.
5. **`table_check` finding 3 < 10 % — wrong.** 19.3–22.9 %, down from 26.6–33.2 %. It moved, so
   the branch is wired up, but nowhere near the prediction.
6. **`crates` std < 15 — right.** 8.78.

### Finding 3 predicted 20 % deaths and there were none

The static check says one danger row in five has an argmax that does not lead out. The agent died
zero times in 1500 rounds. **The rows are real and the agent does not enter them** — the same
error as E09's 78.5 %-of-tiles figure and E10's "17–34 rows prefer `BOMB`" while dropping none.
Three experiments, three disguises, one lesson: *a count of rows in a table is not a visitation
distribution.* The loop probe, which rolls the policy out, was right about E10 and right here;
`table_check` was wrong about both, in opposite directions.

Finding 3 should therefore be read as an *upper bound on rows that could kill*, never as a
predicted death rate. Weighting it by visitation would require a rollout, at which point the loop
probe is the cheaper tool.

### The cycle is now the binding constraint, and it is measurable

Cycle entry step against bombs dropped, across the five seeds:

| seed | cycle entry (median step) | `bombs` | `crates` |
|---|---|---|---|
| s0 | 396 | 27.20 | 28.82 |
| s3 | 110 | 14.46 | 28.89 |
| s1 | 46 | 16.85 | 32.10 |
| s4 | 23 | 17.13 | 30.45 |
| s2 | **11** | **5.27** | **10.67** |

The agent is productive until it enters the cycle and does essentially nothing afterwards, so
`crates` is a function of how long it lasts. s2 gives up at step 11 and collects a quarter of what
the others do. The cause is unchanged from E10 and now isolated: while the agent is *safe*, digit
6 still collapses to `NO_TARGET` at 99.7 % of tiles, so the safe branch of the state carries no
gradient and two mirror rows point at each other. E11 fixed the danger branch only, which is
exactly what it set out to do — and the residual is the measured size of E12's prize.

### Second gap: it bombs in poor positions

**1.75 crates per bomb, against the reference's 3.07** (116.26 crates from 37.83 bombs). That is
a separate defect from the cycle and it has a specific cause: digit 7 is binary, and by the same
99.7 % figure it is *true almost everywhere*. It has therefore degenerated into "do I have a
bomb" and cannot distinguish a spot that opens one crate from one that opens four. Closing this
gap alone would take 26.19 crates to ~46 at the same bomb count.

### Verdict

**BETTER, decisively, and the first rung-2 agent that plays.** 1.723 score / 26.19 crates /
0.000 suicides / 100 % survival, against a floor of 0.000 (E09) and a reference of 8.45 / 116.26
(E08). Roughly 20 % of the reference on score and 23 % on crates, from one change of one digit.

Both remaining gaps are now quantified rather than guessed, and they are independent:

| gap | evidence | size |
|---|---|---|
| the safe branch of digit 6 is constant | 20/20 rounds end in a 2-cycle; `crates` tracks cycle entry | s2 (step 11) collects 10.67, s0 (step 396) collects 28.82 |
| digit 7 cannot rank bombing spots | 1.75 crates/bomb vs 3.07 | ~46 crates at the same bomb count |

### What I do next

1. **E12: digit 6's safe branch stops collapsing to `NO_TARGET`.** Point at the nearest crate
   instead of going silent on top of a bombing spot. One change, and the loop probe measures it
   directly — prediction: cycle entry moves past step 300 in the median seed.
2. **E13: digit 7 becomes a count.** Crates in blast bucketed 0 / 1 / 2 / 3+ (2 → 4 values, table
   12 800 → 25 600). Predict `crates`/bomb toward 2.5–3.0.
3. **Then the ablation panel E10 owes**, from an agent that works: remove the escape branch, the
   crate branch, digit 7's granularity, and `CRATE_DESTROYED`, one at a time, five seeds each.
4. **`scratchpad/benedict/loop_probe.py` is now the standing gate** and is committed alongside
   this entry. It reproduced E10's failure as a control (median 2 tiles, 20/20 rounds, entry at
   step 0) before being trusted on E11.

---

## E10 — A rung-2 feature map: danger, crates as crates, and a target that exists

- **Question:** E09 measured the floor at 0.000 and named three requirements for the rung-2
  state. Do they, together, get an agent off that floor — and if it still dies, *which* of the
  two jobs (bombing, escaping) is the one the features fail to support?
- **Agent:** `benedict_task2` · commit `<fill in from .meta.json>` · labels
  `benedict_q_e10_s{0..4}__task2`
- **Training:** 40 000 rounds, `classic`, no opponents, world seed **810731**, five runs at
  `BM_RUN_INDEX` 0–4. Per-cell α, ε decay, γ = 0.9 — all unchanged from rung 1.
- **Measurement:** `results/eval/task2_crates/benedict_q_e10_s{0..4}__task2.csv`, 300 rounds
  each, seed 20260731, `--preset task2`. Compared against E09 (floor) and E08 (reference).

### Why this entry changes more than one thing

Every previous entry moved one knob. This one cannot, and the reason is worth recording rather
than apologising for: **on this rung the components are not individually measurable.** A danger
feature with `BOMB` unrewarded is never exercised — the agent behaves exactly as in E09. A
rewarded `BOMB` without danger features is the 100 % `KILLED_SELF` result already in `AGENTS.md`.
A crate-target digit without bombing walks the agent to a crate, where it stands. Each of those
is a five-seed training run whose result I can write down in advance, and all three are zero.

So E10 is declared as a rung transition, and the obligation it takes on is to **decompose the
bundle by ablation afterwards** (E11), from a working agent rather than from a broken one. That
ordering is not a convenience: E09 is the proof that a component's contribution depends entirely
on the context it sits in — the same table scored 49.15 coins on `coin-heaven` and 0.000 on
`classic`.

What buys back the attribution in the meantime are three gates that are *not* 300-round
evaluations, so a broken bundle is diagnosed before it costs a measurement: `table_check.py` on
each trained table, the training curve on `CRATE_DESTROYED` and `KILLED_SELF`, and one replay.

### The change

| digit | values | meaning |
|---|---|---|
| 1–4 | 4 each | per direction: **0** blocked (wall, crate, bomb, agent) · **1** lethal at the end of this step · **2** covered by a live bomb · **3** clear |
| 5 | 5 | own tile: **0** safe, else moves left *including this one* (a bomb seen at timer `t` leaves `t+1`) |
| 6 | 5 | BFS first step to the nearest **coin**, falling back to the nearest **crate-bombing position** |
| 7 | 2 | a bomb dropped here opens ≥ 1 crate **and** I have one to drop |

4⁴ × 5 × 5 × 2 = **12 800 rows**, from 80. Rewards gain `CRATE_DESTROYED = +0.3`; everything
else is untouched.

Three notes on the design, because each answers one of E09's requirements:

1. **Digits 1–4 are quaternary, not the ternary I first drew up.** Collapsing "lethal now" and
   "in a blast" into one "unsafe" value fails precisely where the rung lives: while escaping its
   own bomb the agent is *surrounded* by blast tiles and must cross them, and the difference
   between "three moves on the clock" and "walking into fire" is the whole decision.
2. **Digit 7 folds in `bomb_possible`.** Without it, "crate in range, no bomb left" is the same
   row as "crate in range, bomb ready"; the argmax is `BOMB`; the action is invalid; the state
   does not change. That is E09's absorbing row rebuilt in a new place.
3. **No escape feature, deliberately.** "Is there a route out of the blast within the grace
   period" is the natural next digit and the one that comes closest to handing the model the
   answer. Leaving it out is what makes prediction 2 below falsifiable, and it is E11's single
   change if the prediction holds.

### Evidence going in

A 3000-round pilot (`BM_RUN_INDEX=98`, same world seed), and a probe that maps
`rule_based_agent`'s own trajectories through this feature map:

| | first 500 episodes | last 500 |
|---|---|---|
| ε | 0.177 | 0.051 |
| steps | 15.5 | **71.2** |
| `CRATE_DESTROYED` | 4.36 | **6.73** |
| `KILLED_SELF` | 1.00 | **0.99** |
| `INVALID_ACTION` | 2.78 | 1.76 |

Rows a competent agent visits: **451 of 12 800**, with 146 covering 90 % of its steps. The table
is large but cheap — its size is not its sample cost.

The pilot already separates the two jobs: it is learning to bomb (crates rising, episodes
lengthening 4.6×) and it is **not** learning to survive (0.99 suicides at ε = 0.05).

### Prediction (written before the run)

1. **`crates` comes off the floor decisively: 15–45 per round** at ε = 0 — above `random_agent`'s
   3.01, far below the reference's 116. **Refutation:** `crates` < 5 with `bombs` ≈ 0 means the
   agent learned *not* to bomb, and the cause would be the reward scale rather than the features:
   a bomb opens ~2.5 crates for +0.75, discounted by γ⁴ = 0.66 across the four steps to
   detonation, i.e. **≈ +0.5 against a −5 death** — which only pays if the agent believes it dies
   less than one time in ten. At the pilot's 0.99 it does not. If that is what the numbers say,
   E11 is the `CRATE_DESTROYED` scale, not the escape feature.
2. **Central prediction: `suicides` stays high — 0.5–1.0 per round, `survived` < 0.5.** The
   mechanism is specific and follows from what was left out: digits 1–4 describe the *adjacent*
   tile, and say nothing about whether the corridor beyond it leads out within the grace period.
   A three-tile dead end and a genuine escape route are the same row at the moment of decision.
   **Refutation, and the outcome I would rather have:** `suicides` < 0.2 means escape *is*
   learnable from local danger alone, E11's escape feature is unnecessary, and multi-step credit
   assignment is doing far more work than I credit it with.
3. **`score` 0.3–2.5** — non-zero, unlike E09, and nowhere near 8.45. The agent opens crates and
   reveals coins, then dies before collecting them. Per E08 `score` is the outcome metric and not
   the progress signal on this rung; I expect it to lag both other numbers.
4. **The static gates, all checkable in one second per seed and *before* any evaluation:**
   - `table_check` finding 2: **0 frozen spawns, exit 0, in all five seeds.** The rung-1 table
     fails this at 100 %. If E10 also fails it, nothing else in this entry is worth reading.
   - finding 5 (rows preferring `BOMB`): **> 20.** Zero means prediction 1 is already dead.
   - finding 3 (in a blast, a way out exists, the argmax does not take it): **> 20 % of trained
     danger rows.** This is prediction 2's static signature. If the two disagree, the diagnostic
     is wrong rather than the agent, and that is worth knowing before I trust it again.
5. **Spread over the five seeds: `score` std < 1.0, `crates` std < 8.** Per-cell α removed the
   6.98-coin spread on rung 1 (E06); a 160× larger table is exactly where that would stop
   holding. **Refutation:** larger spreads mean 40 000 rounds is too few for 12 800 rows, and the
   round count — not the feature map — is what the next experiment changes.
6. **`think_max_ms` < 5.** The crate BFS floods once the arena is nearly cleared, which is the
   mirror image of rung 1's coin BFS (cheap late, expensive early).

**Refutation condition for the whole entry:** `crates` < 5 *and* fewer than 5 rows preferring
`BOMB`. That combination means the bundle failed at the reward rather than at the state, and no
amount of feature work is the fix.

### What I do with the answer

Prediction 2 is the branch point. If it holds, E11 adds the escape feature as a genuine single
change with a paired CI, and the ablation panel follows it. If it fails — the agent survives
without one — then E11 *is* the ablation panel, and the finding is that local danger digits are
sufficient on this rung, which is the more interesting result and the cheaper agent.

### Result

Commit `c188bb9`. 40 000 rounds per seed, 36 min wall clock for all five in parallel.
Evaluation at ε = 0, 300 rounds, seed 20260731:

| seed | `score` | `crates` | `bombs` | `suicides` | `survived` | `steps` | `invalid` |
|---|---|---|---|---|---|---|---|
| s0 | 0.213 | 4.69 | 2.91 | 0.063 | 0.937 | 376.9 | 0.08 |
| s1 | 0.217 | 4.99 | **24.95** | 0.180 | 0.820 | 332.0 | 0.27 |
| s2 | 0.053 | 1.21 | 0.59 | 0.020 | 0.980 | 392.5 | 0.00 |
| s3 | **0.000** | **0.00** | **0.00** | 0.000 | 1.000 | 400.0 | 0.00 |
| s4 | 0.060 | 1.18 | 0.50 | 0.003 | 0.997 | 398.7 | 0.06 |
| **mean** | **0.109** ± 0.100 | **2.41** ± 2.27 | 5.79 ± **10.77** | 0.053 | 0.947 | — | — |

`think_max_ms` 0.108–0.146 against a 500 ms budget. Target was 8.45 score / ~116 crates (E08).

Paired against the E09 floor (s0, 300 rounds): `score` +0.213 [+0.157, +0.273] BETTER ·
`crates` +4.693 [+3.947, +5.480] BETTER · `suicides` +0.063 WORSE · `survived` −0.063 WORSE.
Against `random_agent`, s0 wins everything (`crates` +1.687 [+0.940, +2.477]) — but the
**five-seed mean of 2.41 crates is below `random_agent`'s 3.01.** Tier 1 at best, and only on
two seeds of five.

### The mechanism: all five seeds park in an absorbing 2-tile cycle

| seed | distinct tiles per round (median) | enters a 2-tile cycle at step | rounds affected |
|---|---|---|---|
| s0 | 2 | 4 | 20/20 |
| s1 | 6 | 23 | 13/20 |
| s2 | 2 | 0 | 19/20 |
| s3 | **2** | **0** | 20/20 |
| s4 | 2 | 0 | 20/20 |

Seed 3 is the pure case: **10 distinct rows over 8000 steps**, action mix LEFT 32.5 % /
RIGHT 32.5 % / UP 17.5 % / DOWN 17.5 % — a period-2 oscillation, in one of two orientations,
from step 0 of every round. And in **100 %** of those steps a bomb dropped where it stands would
have opened a crate. It never bombs.

This is the **fourth** appearance of the same failure and the third distinct route to it: E04
(loops from a scale-free feature), E05b/E06 arm A (loops from an unconverged learning rate),
E09 (a fixed point from distribution shift), now E10.

### Root cause: the crate fallback is dead for exactly the reason `NO_COIN` was dead

`target_direction` returns `NO_TARGET` when the agent is *standing on* a target, mirroring the
coin case. Measured over 200 generated arenas:

| | share of free tiles from which a bomb opens ≥ 1 crate |
|---|---|
| fresh `classic` arena | **99.7 %** |
| after 25 % of crates are cleared | 98.3 % |

So digit 6 reads 0 at **99.7 %** of positions. The fallback I added to fix E09's frozen digit
reproduces E09's frozen digit. The agent is running on digits 1–5 and 7, and in an open corridor
with no live bomb nothing in those digits distinguishes left from right — so two mirror-image
rows whose argmaxes point at each other are an absorbing 2-cycle, and the state never changes to
break it.

I wrote E09's verdict as *"`NO_COIN` must stop being a single state — it needs a target that is
defined when no coin exists"*. I then defined that target so that it, too, is undefined almost
everywhere. The requirement was met in letter and inverted in substance.

### Why it will not bomb, even where BOMB is the obvious move

`Q(best) − Q(BOMB)` in rows where a bomb would open a crate: median **0.617** (s0), **0.104**
(s1), **0.610** (s3). The margin is small, and it is on the right side of zero for a reason
that is not a bug:

- a bomb opens ~2.5 crates → **+0.75**, discounted by γ⁴ = 0.66 across the four steps to
  detonation → **≈ +0.50**;
- `table_check` finding 3 says the escape policy fails in **27–33 %** of danger rows that have a
  way out; at `KILLED_SELF` = −5 that is **≈ −1.5** in expectation.

**The agent is correctly learning that bombing is negative-EV under its own escape policy.**
That is the trap on this rung, and it is circular: it cannot learn to escape without bombing, and
it will not bomb because it cannot escape. ε-exploration does not break it either — the training
log shows s3 dropping 1.70 bombs per episode while its greedy policy drops **zero**.

### Predictions, scored

1. **`crates` 15–45 — wrong, and the refutation clause fired.** 2.41. I wrote that
   `crates` < 5 with `bombs` ≈ 0 would mean "the agent learned not to bomb, and the cause is the
   reward scale". Half right: it did learn not to bomb, and the +0.5-against-−5 arithmetic is
   real. But the reward scale is the *second* cause; the first is a dead target digit, which I
   did not consider at all.
2. **`suicides` 0.5–1.0 — wrong.** 0.053. **My refutation condition for this prediction was
   badly built**: I wrote that `suicides` < 0.2 would mean "escape is learnable from local danger
   alone". It means nothing of the sort — the agent survives by never bombing. The condition
   should have been *conditioned on* `crates` > 15. An unconditional survival threshold cannot
   distinguish a competent agent from an inert one, which is the exact trap E08 documented for
   completion rate and which I then walked into one experiment later.
3. **`score` 0.3–2.5 — wrong.** 0.109.
4. **Static gates: (a) right, (b) half, (c) right-but-meaningless.**
   0 frozen spawns in all five, exit 0 (a). BOMB rows 34/18/17/17/27 against a predicted > 20 —
   two of five (b). Finding 3 at 26.6–33.2 % against a predicted > 20 % (c) — but this
   "confirmation" measures a situation the agent almost never enters.
5. **Seed spread: `score` std 0.100 < 1.0 and `crates` std 2.27 < 8 — technically right, and
   worthless.** The seeds agree because they all fail. Where it matters the spread is enormous:
   `bombs` ranges 0.00 to 24.95, std 10.77.
6. **`think_max_ms` < 5 — right.** 0.146.

**The formal refutation condition did not fire, and that is the most useful thing in this
entry.** I wrote: *"`crates` < 5 **and** fewer than 5 rows preferring `BOMB`"*. `crates` is 2.41,
but every seed has 17–34 rows whose argmax is `BOMB`. The table can bomb; the greedy trajectory
never arrives at those rows. **A count of rows in a table is not a visitation distribution** —
which is precisely the error E09 caught me making with its 78.5 %-of-tiles figure, and I built it
straight into the gate that was supposed to prevent a repeat.

`table_check` also missed the cycle itself, for a concrete and fixable reason: it detects
period-1 absorbing rows (an argmax into a wall). This cycle is **period 2** and made of perfectly
legal moves. E06 already named 2-cycles as the failure class; I built a detector for the wrong
period.

### Verdict

**Off the floor, and nowhere near a result.** 0.109 score against a reference of 8.45, 2.41
crates against 116, and below `random_agent` on the leading indicator. The bundle is not refuted
as a whole — the danger digits work (`suicides` 0.053, `survived` 0.947, and `invalid` fell to
0.08 from E09's 290.7) — but the two components meant to make the agent *act* both fail:

1. **Digit 6 is constant.** Fix: stop returning `NO_TARGET` when standing on a bombing spot.
   Digit 7 already carries "bombing here pays off", so digit 6 is free to keep pointing at the
   nearest crate and give the agent a gradient to follow. This is E11 and it is one change.
2. **Bombing is negative-EV under the current escape policy.** Two candidate levers — the
   `CRATE_DESTROYED` scale, and the escape feature that would move the 27–33 % failure rate.
   These are E12 and E13, and they must not be bundled with E11: unlike the rung transition,
   each of them *is* measurable on its own once the agent moves purposefully.

### What I do next

*(Revised after the post-mortem: my first instinct was to fix digit 6's crate branch in E11. The
99.7 % figure argues against it. If nearly every tile is a bombing spot, then **navigating to
crates is nearly unnecessary on this rung** — a bomb-and-escape cycle is ~7 steps, so 400 steps
allow ~57 bombs at ~2.5 crates each ≈ 140, against the reference's 116 from 37.8 bombs. An agent
that only bombs where it stands and survives is already at reference scale. The crate gradient
buys the last stretch, not the first, so it goes second.)*

1. **E11: digit 6 gains a danger branch.** When digit 5 > 0 it points along a shortest path to a
   safe tile; the coin and crate branches are untouched and the table does not change size. This
   attacks the binding constraint — the −1.50 escape-failure term in the EV above.
2. **E12: the crate branch stops collapsing to `NO_TARGET`.** Once the agent bombs at all, the
   difference between "clears its own neighbourhood then stalls" and "travels to fresh crates"
   is directly visible in `crates`, so this becomes measurable on its own.
3. **The loop probe is the new gate.** Median distinct tiles per round separated all five seeds
   in 20 rounds where `table_check` saw nothing, and it would have flagged this before the
   36-minute training run finished. `table_check` looks for period-1 fixed points; the recurring
   failure on this project is period 2.
4. **Do not touch the danger digits.** They are the part of the bundle that worked, and E11 must
   not confound its own measurement.
5. **Later, and separately: D₄ canonicalization.** `final_project.pdf` p. 8 names the board's
   rotational and mirror symmetries explicitly. Our state is direction-indexed throughout
   (digits 1–4 and 6 permute with the group, 5 and 7 are invariant, and the four movement actions
   permute with it), so folding the table by its 8-element symmetry group is exact rather than
   approximate and multiplies the samples per cell by up to 8. It changes convergence speed, not
   what is representable, so it is worthless until the feature map carries information — which is
   exactly why it is not the answer to E10.

---

## E09 — Transfer floor: the rung-1 agent, unchanged, on `classic`

- **Question:** what does the task-1 agent actually *do* when crates appear? This is the
  "before" column of every table in the task-2 chapter, and it decides whether the rung-2
  work starts from something partially useful or from scratch.
- **Change:** none to the agent. `agent_code/benedict_task2/` is byte-identical to
  `tabular_q_task1/` apart from identity strings, and it loads the rung-1 table
  (49.15 ± 1.23 coins on `coin-heaven`). Only the scenario changes.
- **Agent:** `benedict_task2` · commit `<fill in from .meta.json>` · label
  `benedict_task1model__task2_floor`
- **Training:** none. This is a pure transfer measurement at ε = 0.
- **Measurement:** `results/eval/task2_crates/benedict_task1model__task2_floor.csv`,
  300 rounds, `classic`, `--opponents none`, seed 20260731.

### What the state space does when crates appear (established before the run)

Two facts about the rung-1 feature map, both checked against the code and the shipped table
rather than assumed:

1. **The coin digit goes dead.** `environment.py:377–386` places coins *preferentially under
   crates*: the 9 coins are drawn from a shuffled list of crate tiles first, free tiles only
   as a fallback. `classic` has ≈ 90 crates and 9 coins, so all 9 start non-collectable and
   `game_state['coins']` is **empty at step 1**. `coin_direction` returns `NO_COIN` at every
   step until the agent blows a crate open. Digit 5 is therefore frozen at 0, and the agent
   runs on four wall bits alone — a 16-row table.
2. **Most of those 16 rows were never trained.** `blocked` uses `field != 0`, so on `classic`
   crates count as walls. Simulated over 300 generated arenas: the blocked-neighbour count of
   a free tile is distributed **[0, 1, 2, 3, 4] → 0.001, 0.012, 0.202, 0.421, 0.364**, i.e.
   79 % of free tiles have three or four blocked neighbours. On `coin-heaven` (no crates)
   those configurations essentially do not occur, and the shipped table confirms it: rows
   `0111`, `1011`, `1101`, `1110`, `1111` at `NO_COIN` are **exactly zero**, 5 of the 16.
   **78.5 % of free tiles on a `classic` arena map to an all-zero row.**

An all-zero row means `q_row == q_row.max()` is true for all six actions, so `act` tie-breaks
uniformly — **including `BOMB`**. The agent that scored 49/50 coins is, on this rung, a random
agent for roughly four steps in five.

### Prediction (written before the run)

1. **This is not a "score 0" floor, it is a `random_agent` floor.** Expect the row to look
   like E08's `random_agent`, not like its `coin_collector_agent`: `suicides` **> 0.5 per
   round**, `survived` **< 0.5**, `bombs` on the order of 1–3 per round, and `crates`
   **> 0** — it will destroy crates by accident. `score` **0.0–0.4**, non-zero only because a
   blind bomb occasionally frees a coin the agent then stumbles onto.
2. **`score` will be within noise of `random_agent`'s.** If the paired CI against
   `random_agent` excludes 0 in *either* direction, one of the two arguments above is wrong
   and I need to know which before building on the table.
3. **The four trained `NO_COIN` rows with large positive values** (rows 15, 30, 45, 60, max
   ≈ +18.6 to +19.6, one and two blocked neighbours) **will not help.** They cover 21 % of
   tiles and their argmax is a movement direction learned from a coin gradient that no longer
   exists. Concretely: `invalid` should stay low (< 1 per round) while `score` stays at zero —
   the agent moves legally and pointlessly.
4. **`steps` will be 400.00 in essentially every round**, because the round only ends when
   crates, coins and bombs are all gone. If the completion rate is above ~2 %, my reading of
   `environment.py:289` is wrong for this rung.

**Refutation condition for the whole entry:** any of `score` > 1.0, `suicides` < 0.1, or
`bombs` ≈ 0 means the all-zero-row argument is wrong — most likely because the agent's
*visitation* distribution differs sharply from the static tile distribution I simulated
(it dies early, so it never reaches the interior). That would be worth knowing on its own,
and the fix is to log the visited rows rather than count tiles.

### Why run it at all, given the prediction

Two reasons. It is the honest "before" number for the report, and — more usefully — it turns
"the rung-1 features are insufficient" from an assertion into a measurement with a CI. The
78.5 % figure above is the quantitative version of the argument that the danger and crate
features are *necessary*, and it is worth having in the Experiments chapter as the reason the
next two experiments exist.

### Result

| Metric | value (300 rounds) |
|---|---|
| `score` | **0.000** |
| `coins` | 0.000 |
| `crates` | **0.000** |
| `bombs` | **0.000** |
| **`moves`** | **0.000** |
| `invalid` | 290.667 |
| `suicides` | 0.000 |
| `survived` | **1.000** |
| `round_steps` | 400.0 in every round |
| `think_mean_ms` | 0.006 |

Paired against `random_agent` (E08), 300 rounds: `score` −0.003 [−0.010, +0.000] *no effect
shown* · `crates` −3.007 [−3.153, −2.867] WORSE · `bombs` −1.080 WORSE · `suicides` −1.000
BETTER · `survival` +1.000 BETTER.

**The agent never takes a single step in any of the 300 rounds.** `moves = 0` with
`invalid = 290.7` is not a bad policy, it is no policy at all.

### Diagnosis: it freezes on its starting tile

`invalid` is not spread out — it is bimodal, and the two modes are exactly 3 : 1.

| `invalid` | rounds | share | behaviour |
|---|---|---|---|
| 400 | 218 | 0.727 | pushes into a wall every step, 400 times |
| 0 | 82 | 0.273 | waits every step, 400 times |

0.727 and 0.273 are 3/4 and 1/4. The four starting corners are cleared of crates
(`environment.py:369–374`), so each has exactly two blocked neighbours — both board walls —
and the coin digit is `NO_COIN` from step 1. That fixes the row on step 1 of every round:

| corner | blocked (U,R,D,L) | row | argmax | outcome |
|---|---|---|---|---|
| (1,1) | 1,0,0,1 | 45 | `UP` | blocked → INVALID, state unchanged, forever |
| (1,15) | 0,0,1,1 | 15 | `DOWN` | blocked → INVALID, forever |
| (15,1) | 1,1,0,0 | 60 | `UP` | blocked → INVALID, forever |
| (15,15) | 0,1,1,0 | 30 | `WAIT` | waits, forever |

Three corners of four freeze on an invalid action, one on `WAIT` — 0.75 / 0.25, measured
0.727 / 0.273. The agent is stuck on the tile it spawned on, in every round, from step 1.

### Why those four rows are the worst in the table

They carry the **highest values in the whole model**:

```
NO_COIN rows   max 19.60   <- global argmax of the entire table is row 30
coin rows      max 17.70
```

`NO_COIN` on `coin-heaven` only occurs at the instant the last coin is collected. Those rows
were therefore updated a handful of times each, always from `end_of_round`, always
bootstrapping off one another — enough to inflate them above every legitimately trained row
and not nearly enough to order the six actions sensibly. Three of the four resulting argmaxes
point into a wall.

The semantic mismatch is the whole story. In the training distribution `NO_COIN` meant *the
round is over and it went well*. On `classic` it means *no coin has been visible for 400
steps*. Transfer carried a "success" state onto a "no information" state, and because an
invalid action leaves the state unchanged, the arbitrary argmax of a barely-trained row is
not a small error — it is an absorbing fixed point.

This is the third distinct route to the same failure. E01 reached it through an unconverged
learning rate, E04 through deterministic tie-breaking, and E09 through distribution shift.
**On this state representation, any row the agent can enter and not leave is a round-ending
bug, and the number of such rows is a property of the deployment distribution, not of
training.**

### Predictions, scored

1. **"A `random_agent` floor, not a zero floor: suicides > 0.5, bombs 1–3, crates > 0" —
   wrong, and wrong in the direction that mattered.** Measured: 0 suicides, 0 bombs, 0 crates,
   100 % survival. My argument was that 78.5 % of *free tiles* map to an all-zero row where
   `act` tie-breaks uniformly over six actions including `BOMB`. The tile statistic is
   correct; it is simply irrelevant, because the agent never reaches a second tile.
2. **"`score` within noise of `random_agent`" — right by accident.** −0.003 [−0.010, +0.000].
   Both agents score ~0, for completely different reasons; the null hides the fact that
   `random_agent` at least destroys 3 crates a round and mine destroys none.
3. **"The four high-value `NO_COIN` rows will not help" — right, but I got the consequence
   backwards.** I predicted `invalid < 1 per round` with the agent "moving legally and
   pointlessly". It is 290.7, and those four rows are not merely unhelpful — they are the
   entire failure.
4. **"`round_steps` = 400 in essentially every round" — right.** 400.0 in all 300, completion
   rate 0.000.

**The refutation condition fired exactly as written.** I wrote: *"any of `score` > 1.0,
`suicides` < 0.1, or `bombs` ≈ 0 means the all-zero-row argument is wrong — most likely
because the agent's visitation distribution differs sharply from the static tile distribution
I simulated."* Two of the three triggered, and the stated cause was the actual cause. Writing
down *why* a prediction might fail, and not just what it predicts, is what turned a wrong
guess into a diagnosis in one run instead of an afternoon.

### Verdict

**Floor established: 0.000 score, 0.000 crates, 0.000 moves.** The rung-1 model transfers
*nothing* to `classic` — it is strictly worse than `random_agent`, which at least opens 3.0
crates per round. Nothing in the rung-1 table is worth keeping as an initialisation; task 2
starts from zeros.

Three requirements for the task-2 feature map follow directly, and they are now measured
rather than argued:

1. **`NO_COIN` must stop being a single state.** "No coin is visible" is the *normal*
   condition on this rung, not a terminal one. It needs a target that is defined when no coin
   exists — the direction to the nearest crate — so the digit carries information for the
   399 steps out of 400 where the coin list is empty.
2. **Crates must not be encoded as walls.** `field != 0` collapses "impassable forever" and
   "impassable until I bomb it", and the second is the entire point of the rung. The corner
   rows above are not even the worst case of this — they are just the first one reached.
3. **An unreachable-argmax row must be detectable before a 300-round run.** The four rows
   that broke this were visible in the shipped table by inspection: three of them have an
   argmax pointing at a blocked neighbour, which is knowable from the row index alone. That
   check belongs in the standing diagnostic next to the loop detector, and it would have
   predicted this result in under a second.

Requirement 3 is the cheapest and goes in first, because it is a guard on every experiment
after this one, not a feature.

---

## E08 — Reference measurement on `classic`, no opponents

- **Question:** what is the scale on rung 2, and **which metric discriminates on it**? On
  rung 1 the equivalent measurement (Maxi's E00) reframed the whole rung: `coins` turned out
  to be a pass criterion with zero variance, and `steps` was the real target. Running the
  provided agents first is cheap and stops me optimising the wrong column for a week.
- **Change:** none. Measurement of the four provided agents, each alone.
- **Agent:** `random_agent`, `peaceful_agent`, `coin_collector_agent`, `rule_based_agent` ·
  commit `<fill in from .meta.json>` · labels `ref_<agent>__task2`
- **Measurement:** `results/eval/baselines/ref_<agent>__task2.csv`, 300 rounds each,
  `classic`, `--opponents none`, seed 20260731, `--preset task2`.

### Prediction (written before the run)

1. **`coin_collector_agent` scores exactly 0.00**, with `bombs` 0.00 and `crates` 0.00, and
   survives every round. It has no bomb logic, and by the coin-placement argument in E09 all
   9 coins start under crates, so nothing ever becomes collectable. **Refutation:** any
   non-zero score means my reading of `environment.py:377–386` is wrong, and E09's central
   argument goes with it.
2. **`peaceful_agent` also scores 0.00** and also survives every round, for the same reason
   plus its own refusal to bomb. So **three of the four reference agents are tied at zero on
   the primary metric** — which is the actual finding of this experiment: on rung 2, `score`
   does not order the reference field, it only separates `rule_based_agent` from everything
   else.
3. **`random_agent`: `crates` 1–5 per round, `score` < 0.3, `suicides` > 0.7, `survived`
   < 0.3.** It is the only cheap agent that bombs, so it is the honest floor for anything
   that bombs — and, per E09, the right comparison for my own transferred agent.
4. **`rule_based_agent` is the only meaningful reference on this rung.** Expect `score` 4–8
   of 9, `crates` 25–50, `suicides` < 0.15, `survived` > 0.80.
5. **The discriminating metric is `crates`, with `score` second and `steps` uninformative.**
   `crates` orders all four agents; `score` orders only one against three; `steps` should be
   400.00 for every agent that survives, because the round cannot end while crates remain.
   **Refutation:** if `rule_based_agent` completes more than ~10 % of rounds inside 400 steps,
   `steps` is informative after all and belongs beside `crates` as a rung-2 efficiency metric.

**What I do with the answer.** Prediction 5 sets the progress metric for E10 and E11. If it
holds, the task-2 ladder is *crates destroyed → coins scored → suicides down*, in that order,
and `score` is not usable as a progress signal until the agent reliably opens crates at all.

### Result

300 rounds each, `classic`, solo, seed 20260731, commit `7799000`:

| Agent | score | crates | bombs | suicides | survived | `round_steps` | invalid |
|---|---|---|---|---|---|---|---|
| `peaceful_agent` | 0.000 | 0.00 | 0.00 | 0.000 | 1.000 | 400.0 | 256.21 |
| **`benedict_task2` (E09)** | 0.000 | 0.00 | 0.00 | 0.000 | 1.000 | 400.0 | 290.67 |
| `random_agent` | 0.003 | 3.01 | 1.08 | **1.000** | 0.000 | 19.0 | 11.84 |
| `coin_collector_agent` | **8.500** [8.420, 8.577] | **116.26** | 37.83 | 0.000 | 1.000 | 399.5 | 1.74 |
| `rule_based_agent` | **8.410** [8.313, 8.503] | **116.43** | 37.87 | 0.000 | 1.000 | 399.1 | 1.76 |

Completion rate (round ended before step 400): `coin_collector` 0.053, `rule_based` 0.080,
`peaceful` 0.000, `random` **1.000**.

### Prediction 1 was wrong at the premise

I predicted `coin_collector_agent` would score exactly 0.00 with 0 bombs, on the grounds that
"it has no bomb logic". It scores 8.500 and drops 37.8 bombs a round. **I asserted the
contents of an agent I had not read.**

`diff agent_code/coin_collector_agent/callbacks.py agent_code/rule_based_agent/callbacks.py`
is 36 lines: `coin_collector_agent` is `rule_based_agent` minus the `coordinate_history`
loop-breaker, minus opponent hunting, minus the per-round reset. The crate-bombing, dead-end
bombing and blast-escape logic is **identical**. On rung 1 it looked like a pure coin walker
because without crates that code never fires — and I carried that impression onto a rung
where it is false.

The coin-placement fact underneath the prediction was correct and is still correct (all 9
coins start under crates, `environment.py:377–386`). The error was the inference from it: I
checked the environment and not the agent.

### Predictions, scored

1. **`coin_collector_agent` scores 0.00 — wrong**, see above. 8.500, 116.26 crates.
2. **`peaceful_agent` 0.00, survives every round — right.** 0.000 / 1.000, and 256.21 invalid
   actions, the "moves but does not navigate" profile from rung 1.
3. **`random_agent`: crates 1–5, score < 0.3, suicides > 0.7, survived < 0.3 — right on all
   four.** 3.01 · 0.003 · **1.000** · 0.000. It kills itself in *every single round*.
4. **`rule_based_agent`: score 4–8, crates 25–50, suicides < 0.15, survived > 0.80 — half
   right.** Suicides and survival right (0.000, 1.000). Score 8.410, just above my range.
   Crates 116.43, **more than double the top of my range** — I underestimated how much of the
   board a competent agent clears in 400 steps.
5. **"`crates` discriminates, `score` orders only one against three, `steps` uninformative" —
   the ordering claim is wrong, the `steps` claim is right.** Completion is 0.053 and 0.080,
   below the 10 % refutation threshold I set, so `steps` is not a rung-2 metric. But `crates`
   does **not** order the field: 116.26 [115.55, 116.97] against 116.43 [115.71, 117.16] is a
   tie. And `score` orders *three tiers*, not one against three.

### The actual structure: three tiers, and the reference is one of them

| Tier | agents | score | crates | suicides |
|---|---|---|---|---|
| 0 — inert | `peaceful`, **E09** | 0.000 | 0.00 | 0.000 |
| 1 — bombs, dies | `random` | 0.003 | 3.01 | 1.000 |
| 2 — reference | `coin_collector` ≈ `rule_based` | ~8.45 | ~116 | 0.000 |

**`coin_collector_agent` and `rule_based_agent` are indistinguishable on this rung**, exactly
as they were on rung 1 (125.3 vs 124.8 steps) and for the mirror-image reason: there the bomb
code never ran, here the opponent code never runs. Solo `classic` does not exercise anything
that separates them. So rung 2 again has **one** reference, not two — and the difference in
score (8.500 vs 8.410) is small enough and in the *unexpected* direction that it should not be
claimed without a paired test.

**The ceiling is 8.45 of 9 coins**, i.e. 94 %. That is the number E11 gets measured against,
and it costs ~38 bombs across 400 steps with **zero** suicides.

### What discriminates, and in what order

Revised from prediction 5, and this sets the progress metric for E10 and E11:

- **`crates` is the leading indicator.** It moves first and it separates tier 0 from tier 1
  from tier 2 (0.00 / 3.01 / 116). An agent that learns to bomb crates but not to find the
  revealed coins registers on `crates` while `score` is still 0 — which is exactly the state
  E10 is expected to reach.
- **`suicides` is the gate between tier 1 and tier 2, and it is binary in the reference
  field**: 1.000 for `random`, 0.000 for both competent agents. There is no middle ground in
  the references. That is the whole rung in one number.
- **`score` is the outcome metric** and is what gets reported, but it is useless as a progress
  signal until the agent both bombs *and* survives — it stays pinned at 0 through the entire
  first half of the work.
- **`steps` is not a metric on this rung** (completion 5–8 %), and it is actively misleading:
  `random_agent` "completes" 100 % of rounds at 19.0 steps because it *dies*. **Completion
  rate must be read together with survival**, or the worst agent in the field looks like the
  fastest. This is the rung-1 `steps` trap in a new disguise and belongs in `AGENTS.md`.

### Verdict

**Scale established.** Target 8.45 score / ~116 crates / 0.000 suicides, one meaningful
reference (`coin_collector_agent` and `rule_based_agent` are interchangeable solo). Progress
metric order for task 2: **`crates` → `suicides` → `score`**.

Cost of the E08 error: nothing, because it was measured before anything was built on it. That
is the argument for running the reference measurement first, and it is now the second rung in
a row where doing so overturned an assumption I would otherwise have optimised against for a
week.

**Method note.** E08 prediction 1 and E09 prediction 1 failed the same way: I reasoned
confidently from the environment code and did not check the *other* half of the system — the
agent in one case, the agent's visitation distribution in the other. Reading
`coin_collector_agent/callbacks.py` would have taken 30 seconds.

---

## E07 — Is the ε schedule still doing anything?

- **Question:** E06 arm B changed *two* things relative to E04 — the per-cell learning rate
  **and** the ε schedule — and they were never separated. E05, which introduced the schedule,
  turned out to be an unreplicated lucky draw. So: with a converging α, does decaying ε still
  contribute anything at all?
- **Change from E06 arm B:** exactly one. `EPS_DECAY = 1.0`, i.e. ε stays at 0.2 for the whole
  run. Per-cell α = 1/N^0.7 kept, φ kept, rewards kept, γ kept.
- **Agent:** `benedict_coin_collector` · commit `<to be filled in>` · arm label `e07c`
- **Training:** 10 000 rounds, `coin-heaven`, world seed **810731** (never the evaluation
  seed), five runs at `BM_RUN_INDEX` 0–4.
- **Measurement:** `results/eval/benedict_q_e07c_s{0..4}__task1.csv`, 300 rounds each,
  seed 20260731. Compared against E06 arm B (49.15 ± 1.23, worst 47.16) as distributions,
  not as single runs.

### Prediction (written before the run)

1. **No demonstrated difference. E07 lands at 47–50 mean with std below 2**, overlapping
   arm B's 49.15 ± 1.23. Reasoning: E05's mechanism was "training never reaches the late
   round because the agent dies at step 61". Per-cell α already removes the deadlocks that
   truncate those episodes, so long episodes should now appear *without* the schedule.
2. **The decisive diagnostic is training `steps` over the last 1000 episodes.** If it lands
   near 120 (arm B's value) at a constant ε = 0.2, the schedule was never the cause — α was.
   If it stays near 61 (the v4/E04 value) while evaluation coins still reach ~49, then the
   whole "training must see the late round" story is wrong, because the agent would be
   playing well at ε = 0 having never trained there.
3. **`invalid` and `suicides` stay at arm B's levels** (0.15 and 0.000). Constant ε = 0.2
   means the training agent keeps bombing itself — `KILLED_SELF` near 0.9 rather than arm B's
   0.10 — but that is a property of the *behaviour* policy and should not reach the greedy one.
4. **If E07 wins outright** (mean above 49.15 with non-overlapping spread), the reading is
   that constant exploration keeps rare rows refreshed, which the ε floor of 0.02 does not.
   That would be an argument for a *higher floor*, not for abandoning schedules.

**Why it is worth running even though I expect a null.** Carrying an untested knob into
task 2 is how a configuration becomes folklore. If ε decay does nothing here, the task-2
agent starts with one fewer thing to reason about; if it does something, I learn that before
the rung where it is expensive to discover.

### Result

| Metric | E06 arm B (ε decay) | E07c (constant ε) |
|---|---|---|
| `coins`, mean over 5 seeds | 49.15 | 48.68 |
| `coins`, std | 1.23 | **0.33** |
| `coins`, worst seed | 47.16 | **48.10** |
| per seed | 50.0 · 47.2 · 50.0 · 49.8 · 48.7 | 48.9 · 48.1 · 48.7 · 48.9 · 48.8 |
| full-sweep rate (all 50 coins) | 0.981 (worst 0.940) | 0.966 (worst **0.950**) |
| `invalid` | 0.055 | 0.043 |
| `suicides` | 0.0173 | 0.0247 |
| **training `steps`, last 1000** | 120–128 | **68–72** |
| **training `KILLED_SELF`, last 1000** | 0.10–0.15 | **0.88–0.90** |

Mean difference −0.47 with a pooled standard error of ≈ 0.57 — **not demonstrated**.
Constant ε is markedly *more consistent* (std 0.33 against 1.23) and has the better worst
seed, which was not predicted.

### E05's mechanism is refuted, not just its number

E05 argued that ε decay helps because *training never reaches the late round*: at constant
ε the agent bombs itself and dies at step ~61, while evaluation runs to 130–400.

E07 reproduces that training regime exactly — **68–72 steps per episode, dying in 89 % of
them** — and still evaluates at **48.68 coins over ~130 steps**. The agent plays well in a
regime it has essentially never trained in. The distribution mismatch is real; it simply
does not matter.

Why: **φ is position-relative and contains no notion of time or of how many coins are left.**
"The late round" is not a distinct region of the state space. A coin five tiles away produces
exactly the same row at step 3 as at step 300, and the agent meets that row constantly in the
first 70 steps. Only the *frequency* of rows shifts as the round progresses, not their
identity — which is what a good state abstraction is supposed to do.

So E05 was wrong twice: the +3.823 was a lucky draw (E06), and the story explaining it was
also wrong (E07). What actually fixed task 1 was the learning rate, and nothing else.

### Predictions, scored

1. **"No demonstrated difference, 47–50, std below 2" — right.** 48.68 ± 0.33.
2. **Prediction 2's second branch fired, and it was the decisive one.** I wrote: *"if training
   steps stay near 61 while evaluation coins still reach ~49, then the whole 'training must
   see the late round' story is wrong."* Training steps 68–72, evaluation 48.68. Recording
   both branches in advance is what makes this a refutation rather than a shrug.
3. **`KILLED_SELF` near 0.9 in training, without reaching the greedy policy — right.**
   0.88–0.90 while training, `suicides` 0.0247 at evaluation.
4. **"If E07 wins outright" — it did not win on the mean, but it won on spread and worst
   case**, which I had not considered. Lower variance is the more useful property when only
   one model ships.

### Verdict

**No demonstrated difference on task 1 — the ε schedule earns nothing here.** Task 1 is
settled either way: ~49 coins of 50, ~97 % full sweeps, both configurations.

**Decision: keep the schedule going into task 2 anyway**, and be explicit that this is *not*
justified by the task-1 numbers. The reason E05's mechanism failed is specific to task 1's
feature map: nothing in φ marks a hazard or a deadline, so no part of the state space depends
on surviving. **On task 2 that changes completely.** Danger states — in a blast radius, bomb
timer running — are a genuinely distinct region, reachable only by surviving the four steps
after dropping a bomb. An agent dying in 89 % of training episodes never experiences the
escape it must learn. The argument that failed here is the argument that should hold there,
and E07 is the reason I will be able to say why.

If the schedule turns out to earn nothing on task 2 either, it goes.

### What I do next

1. **Ship `q_e06b_s0.npy`** as the task-1 model, following the pre-declared "index 0" rule.
   Choosing the ε-decay arm is a decision about task 2, not about its 50.0 on seed 0 — the two
   arms are statistically indistinguishable here.
2. **Task 2** (`classic`, no opponents). Everything so far is a navigator that survives by
   never bombing; there it must bomb deliberately and escape. Required:
   - a **danger feature** (in blast radius / steps until detonation / is there an escape),
   - crates in the wall bits are already handled (`field != 0`),
   - the reward table revisited: `CRATE_DESTROYED` added, and `COIN_COLLECTED` +5 finally
     ablated against the game's +1 — an ablation now three experiments overdue.
3. **Carry forward: per-cell α, n = 5 seeds, prediction before the run, and the
   `loop_check` replay** — the last of which caught more than the 300-round evaluations did.

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

| Metric | arm A (const α = 0.1) | arm B (α = 1/N^0.7) |
|---|---|---|
| `coins`, mean over 5 seeds | 20.78 | **49.15** |
| `coins`, std over 5 seeds | **6.98** | **1.23** |
| `coins`, worst seed | 11.07 | **47.16** |
| `coins`, best seed | 30.01 | 50.00 |
| per seed | 11.1 · 20.1 · 30.0 · 18.7 · 24.0 | 50.0 · 47.2 · 50.0 · 49.8 · 48.7 |
| high-traffic blocked-argmax rows | 0 in all 5 | 0 in all 5 |
| simulated loop entry (step) | 25 · 42 · 69 · 54 · 56 | 136 · 133 · 137 · 136 · 135 |
| simulated coins before the loop | 10.7 · 18.4 · 29.5 · 22.8 · 23.8 | **50.0 in all five** |

Every arm-B run sweeps the whole board; its "loop" is the benign idle after the last coin.
Every arm-A run deadlocks between step 25 and step 69.

### The result that hurts: E05 does not replicate

Arm A **is** the v5 configuration — same φ, same rewards, same ε schedule, only the RNG
pinned. v5 measured **49.93**. Five seeded runs of the same configuration give
**20.78 ± 6.98, best 30.01.** v5 was not a result, it was a lucky draw, and its 49.93 lies
far outside the distribution its own configuration produces.

The same doubt now attaches to v4's 46.107, also a single draw of a constant-α
configuration. The two numbers that E04 and E05 were built on are both unreplicated.

**What still stands:** the large effects. E02 (+29.4 coins from the coin direction) and E03
(`suicides` 0.427 → 0.000, with the mechanism read directly out of the table) are far too
big to be draw noise, and E03's cause was verified cell by cell rather than inferred from a
mean. The small ones — E04's +3.9 and E05's +3.8 — are **not established**, and the report
must say so.

### Why the training logs showed nothing

| last 1000 episodes | v5 | arm A s0 | arm B s0 |
|---|---|---|---|
| steps | 119.3 | 118.5 | 125.6 |
| coins | 45.48 | 44.77 | 48.40 |
| `KILLED_SELF` | 0.193 | 0.239 | 0.104 |

Arm A's training is indistinguishable from v5's and looks healthy — 44.8 coins per episode —
while the same table scores 11.1 at evaluation. The cause is ε: at 0.02 a random action
arrives every ~50 steps and knocks the policy out of the cycle it is stuck in, so a
deadlock-prone table never reveals itself in training. At ε = 0 nothing rescues it.

This is exactly the rule Maxi added to `AGENTS.md` on 2026-08-05 ("a training curve is not a
result"), arrived at independently from a different agent and a different failure. It is now
supported by two unrelated pieces of evidence and belongs in the report as a methodological
finding rather than a footnote.

### Predictions, scored

1. **"Both arms 49–50, the means barely move" — badly wrong.** Arm A is 20.78. The error was
   assuming arm A would reproduce v5; that assumption was the very thing under test.
2. **"The spread collapses" — right, and the thresholds nearly exact.** Arm A std 6.98
   (predicted > 5), arm B std 1.23 (predicted < 1, marginally missed), arm B worst 47.16
   (predicted ≥ 48, marginally missed), arm A produced runs below 40 (predicted at least one;
   all five were below 31).
3. **"Arm A has a blocked-argmax high-traffic row" — wrong.** Zero, in all ten tables. The
   failure mode here is an early absorbing **2-cycle** between two tiles, not the row-409
   deadlock from E05b. Row 409 was one instance of a broader class, and I generalised from a
   sample of one.
4. **"Arm A replicates E05" — wrong, and the most important miss.** See above.
5. **α = 1 on the first visit destabilising early learning — did not happen.** No floor needed.

### Verdict

**Arm B wins decisively, and the experiment's real finding is about method.** A per-cell
learning rate takes task 1 from 20.78 ± 6.98 to 49.15 ± 1.23 with a worst case of 47.16 —
it does not merely raise the mean, it removes the failure mode. `sum α = ∞`, `sum α² < ∞`
from L26 turns out not to be a formality: with a constant α the high-traffic cells never
settle, and on this problem an unsettled cell is not a small error but an absorbing
deadlock.

Second finding, equally important: **every result in E01–E05 is a single draw.** From here
on, no claim without n = 5.

### What I do next

1. **E07: is ε decay still needed?** Arm B is per-cell α *plus* ε decay, and the two were
   never separated. Ablate it — per-cell α with constant ε = 0.2, five seeds. If it makes no
   difference, E05's mechanism story was wrong as well as its number, and the schedule can be
   dropped before task 2 rather than carried along untested.
2. **Ship arm B seed 0 as the task-1 model**, and fix the convention now: *the submitted
   table is always run index 0*, chosen before seeing the results. Picking the best of five
   would be cherry-picking, and the difference (50.0 vs 49.15) is not worth the dishonesty.
3. **Re-run E04's comparison under n = 5** if the report needs the clipped-offset claim.
   Cheaper alternative: state it as unreplicated and let the E02/E03/E06 chain carry the
   argument.
4. **Then task 2.** Per-cell α goes along; on a rung where a single wrong cell means death
   rather than a wasted step, the stability matters more, not less.

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
