# Versuchsprotokoll — Benedict

Ergebnisbuch, ein Eintrag pro Experiment. Bewusst getrennt von `BENEDICT.md`:
das Logbuch hält fest, *warum* ich etwas entschieden habe, hier stehen die *Zahlen*.
Die Zeilen hier werden im Bericht fast wörtlich zu Tabellenzeilen.

Regeln, an die ich mich halte:

- **Vorhersage vor der Messung.** Steht sie nicht vorher da, war es kein Experiment,
  sondern eine Beobachtung. Ich committe die Vorhersage, bevor ich messe — dann
  belegt die Git-Historie die Reihenfolge.
- **Eine Änderung pro Eintrag.** Sonst ist nicht zuzuordnen, was gewirkt hat.
- **Commit-Hash mitschreiben.** Steht in `results/eval/<label>.meta.json`, seit
  2026-08-16 zusammen mit der vollständigen `BM_*`-Umgebung. **Ein fester Seed macht
  einen Lauf nicht exakt reproduzierbar** — `main.py` seedet die mitgelieferten Gegner
  nicht, und `evaluate.py` erreicht deren stdlib-`random` nicht: ~21 % der Runden
  wiederholen sich exakt, die Mittelwerte nicht. Eine 1000-Runden-Auswertung trägt ein
  **Rauschniveau von ±0.12 auf `score`** allein durch die Gegner (E37 §5.9).
- **Negative Ergebnisse bleiben stehen.** Sie kommen so in den Bericht.
- Seeds ab Rung 4: Training `810731`, Validierung `550731`, Held-out/Ship `990731`.
  **1000 Runden** für jede berichtete Zahl, 300 für einen Sweep-Arm, 100 für einen
  Schnelltest. Rung 1–3 nutzten `20260731` und 300 Runden.
- **Eine Zeile zählt nur, wenn das gepaarte 95-%-KI die Null ausschließt *und*
  `analyze.py` sie nicht `(fragile)` markiert** — der fixe Bootstrap-Seed macht sonst
  aus einem Münzwurf ein Urteil (E39 → E40, Audit 10).
- **Neueste Einträge oben**, wie im Logbuch. Für den Bericht wird von unten nach oben
  gelesen — E01, E02, … ist die Reihenfolge, in der die Argumentation aufgebaut ist.

Urteil: **BESSER** · **SCHLECHTER** · **nicht gezeigt** (KI enthält die Null).

---

## E51 — veto the blunder, not the bomb: a certain-death move filter

- **Question.** E43 traced every death to the last step at which some action still survived and
  found that in **76.7 %** of them **two or more surviving actions existed**. That is a *policy*
  failure inside states the feature map already describes, not a representation gap — and fourteen
  of the fifteen interventions on this rung tried to fix it by changing what the table *learns*.
  This one does not touch the table at all: it removes provably fatal moves from the argmax at
  inference, on the shipped E37 table, unchanged.

  **It is the counterpart to E46, and deliberately its mirror image.** E46 vetoed the *bomb* when
  the escape had no slack: it cut suicides −0.131 and survival +0.050 exactly as designed and cost
  **−0.283 score**, entirely through coins, because the zero-slack bombs are simultaneously the
  most lethal and the most productive — a bomb in a dense pocket has a contained blast and a tight
  escape for the same geometric reason. **This filter never vetoes BOMB.** It keeps the productive
  bomb and forbids the fumbled escape instead, which is the one route to the same survival that
  cannot pay for it in crates.

- **The contradiction this is built to resolve.** Two slopes relating survival to score are on
  record and they have opposite signs:

  | source | slope | measured how |
  |---|---|---|
  | E46, causal, within field | **−5.7** score per unit survival | +0.050 survival bought, −0.283 score paid |
  | E41 §6, correlational, across fields | **+9.4** score per unit survival | 3.828 @ 0.440 vs 2.611 @ 0.310 |

  Eight replications say survival does not convert into points — **and all eight were run against
  `rule_based`, the field E41 showed to be the one where survival is not binding** (we reach step
  200 there anyway). Against `binary_v6` survival collapses 0.440 → 0.223 and E41 reversed the
  conclusion. Because this intervention buys survival with the crate route closed *by construction*,
  it separates the two slopes more cleanly than anything run on this rung.

- **Change.** `agent_code/benedict_task4/callbacks.py` only; `train.py` is untouched and nothing is
  retrained. One environment switch, `BM_DEATH_FILTER`, default off:

  | mode | behaviour |
  |---|---|
  | unset / `0` | the shipped policy, byte for byte — the control |
  | `step` | veto moves that are lethal at the **end of this step** only |
  | `1` | veto moves after which **no continuation survives** the bombs already visible |

  Mode `1` runs the same time-aware BFS as `escape_direction`, from the tile the action lands on,
  and asks whether any tile outside every blast is reachable in time. Three details that are
  decisions, not implementation:

  1. **BOMB is never in the mask.** This is the whole difference from E46 and is an invariant of the
     file, not an accident of the code path.
  2. **Only bombs and fire visible *now* are modelled** — no opponent pessimism. E43 puts "enemy
     bomb placed after we committed" at **2.3 %** of deaths, so modelling it would buy 2.3 % and pay
     for it in conservatism on every step.
  3. **A blocked direction is scored as WAIT**, because `environment.py:121-126` leaves the agent
     where it is; scoring it on a tile it never reaches would veto the wrong action.

  When every action is vetoed the filter stands aside and the unfiltered row decides, so the agent
  is never worse off than the control in a hopeless position.

- **Design.** 3 arms × 2 fields, **n = 4000** paired arenas each, held-out ship seed **990731**.
  Primary field **3 × `ext_xiaoxiae_binary_v6`** (external, per E41 — the strongest agent measured
  and the one against which survival is binding); secondary **3 × `rule_based_agent`**, the field
  every earlier conclusion on this rung was drawn on.

- **Power and reachability, stated before the bar.** At n = 4000 the realised SE on `score` is
  ~0.06 (E40), so the 80 %-power MDE is **~0.17**. The control dies by its own bomb 0.552 times a
  round on the external field, and E45 puts bombs with *no escape at all* at 8.0 % of own-bomb
  deaths — the filter cannot save those, since it never vetoes BOMB. So **~0.51 suicides/round are
  addressable.** Converting even half of that is ±1.4 score at E46's slope or +2.4 at E41's. **Both
  hypotheses predict effects far above the MDE, in opposite directions**, so this design cannot
  return an uninformative null for want of power — which is the failure mode audit 11 caught in E40.

- **Disclosure — what I looked at before writing this.** Eight hand-checked cases on synthetic
  boards (all pass, including two that caught *my* errors rather than the code's: the arena has free
  tiles at (odd, even) so a bare row is not a corridor, and at `t = 4` a wasted step is still
  survivable — the filter correctly vetoes nothing there). A 60-round solo-board regression
  confirming the filter-off path is **identical on all 18 behavioural columns, 60/60 rounds**, so
  the control is the shipped agent and not a re-implementation of it. A 30-round smoke run of all
  three modes on the primary field in which **only the think-time columns were read** — mean 0.064 →
  0.097 ms, worst round-max 0.59 ms, zero steps over the limit. No outcome column has been read in
  any mode. The direction of this prediction is therefore not informed by data, unlike E46's.

### Prediction (written and committed before the n = 4000 run)

1. **P1, primary.** On the **external** field the `1` arm beats the control on **`score`** by
   **≥ +0.25**, CI excluding 0 and a non-fragile permutation p.
   **Refutation:** below +0.25 → survival bought this cheaply still does not pay, E46's slope
   generalises beyond bomb-gating, and the pre-committed consequence is that **the action filter is
   demoted in the CNN plan from a load-bearing component to a training-time safety rail** — it would
   then be justified only by what it does for exploration, and must be re-argued on that ground.
2. **P2, mechanism — this must hold or P1 is a coincidence.** `suicides` on the external field fall
   by **≥ 0.25** (from 0.552) and `survived` rises. The filter is *designed* to make own-bomb death
   nearly impossible; if suicides move by less, something defeats it systematically — opponents
   blocking the escape, which the static-world model does not see — and that, not P1, is the finding.
3. **P3, the internal control.** The `step` arm is **near-inert**: |Δscore| < 0.17 and
   |Δsuicides| < 0.05. Digits 1-4 already carry exactly this as `NB_LETHAL`, so a trained table
   should almost never walk into a blast that lands the same step. **If the shallow arm moves
   substantially, the table is far worse at using its own features than E37 claims, and P1 can no
   longer be attributed to the lookahead** — the whole effect would then be one digit the policy
   ignores.
4. **P4, the cost — E46's failure route, closed by construction and checked anyway.** `bombs` per
   round stay within **±0.5** of the control and `crates` do not fall by more than 1.0. BOMB is never
   vetoed, so a drop means the filter is suppressing bombing *indirectly* by keeping the agent out
   of the pockets where bombs pay, and E46's cost has reopened by another door.
5. **P5, the field split.** Δscore is **larger on the external field than on `rule_based`**. E41 says
   survival is non-binding against `rule_based`; this is the direct test. **If the ordering reverses,
   "survival is non-binding against `rule_based`" is wrong and the eight survival nulls need
   revisiting** — which would be a bigger result than P1.
6. **Guards.** The control arm must reproduce the known held-out numbers within the ±0.12 noise
   floor: **2.611** external, **3.949** `rule_based`. `think_over_limit` must be 0 in all six runs.
   Any arm failing a guard voids the comparison rather than being explained.

### Run

```bash
# three arms x two fields, n = 4000, held-out ship seed 990731
for MODE in 0 step 1; do
  BM_DEATH_FILTER=$MODE BM_QUIET_LOGS=1 uv run python tools/evaluate.py \
    --agents benedict_task4 ext_xiaoxiae_binary_v6 ext_xiaoxiae_binary_v6 ext_xiaoxiae_binary_v6 \
    --n-rounds 4000 --seed 990731 --out-dir results/eval/task4_tournament \
    --label benedict_q_e51_${MODE}__task4_ext_xiaoxiae_binary_v6_ship990731
  BM_DEATH_FILTER=$MODE BM_QUIET_LOGS=1 uv run python tools/evaluate.py \
    --agents benedict_task4 --opponents rule_based \
    --n-rounds 4000 --seed 990731 --out-dir results/eval/task4_tournament \
    --label benedict_q_e51_${MODE}__task4_rb_ship990731
done
```

### Method failure — the first run was void, and the bug inverted the intervention

Run at `b0bf2c6`, 6 arms × n = 4000, discarded. Kept as `benedict_q_e51bug_*` because the numbers
are the evidence for what follows.

**The bug.** `act()` tested `veto.all()` for "no action survives". **BOMB is never vetoed, so
`veto.all()` is permanently False** and the stand-aside path never ran. In a position where every
*movement* action is fatal, the mask therefore left BOMB as the only finite entry in the row and the
filter **forced the agent to bomb** — a suicide with one in hand, an `INVALID_ACTION` that stands
still and dies without. A filter written to prevent own-bomb death was compelling it.

**It is visible in the metrics, on the primary field, and only in the arm that has the lookahead:**

| 3 × `binary_v6`, n = 4000 | control | `step` | **`1` (void)** |
|---|---|---|---|
| score | 2.778 | 2.794 | 2.806 |
| **suicides** | 0.540 | 0.535 | **0.558** — +0.019 [+0.008, +0.030], p = 0.0008, *worse* |
| **invalid** | 2.838 | 3.489 | **4.505** |
| killed by opp. | 0.158 | 0.159 | 0.138 |

`invalid` is the tell: +1.667 actions per round that the control never wasted, which is the
no-bomb-in-hand branch of the same forced BOMB. The one genuine signal underneath it —
killed-by-opponent −0.021 — was cancelled by the pathology the filter itself introduced.

**Why the tests passed anyway, which is the transferable part.** Eight hand-checked cases passed,
including one named *"corridor sealed → every action fatal (`act()` falls back)"*. It asserted on
the **mask** and never called `act()`. The name claimed a behaviour the test did not exercise, so it
certified the exact path that was broken. **A mask is not a decision.** Three `act()`-level cases
added: hopeless × bomb-in-hand, hopeless × no bomb, and escapable — each asserting on the action
returned, not on the veto array.

**The repair, not a redesign.** The fallback now tests `veto[FILTERABLE].all()` over the five
vetoable actions, which is what §Change already said it did: *"when every action is vetoed the filter
stands aside and the unfiltered row decides."* The pre-registration is unchanged and still precedes
the measurement. Filter-off remains identical on 60/60 deterministic solo rounds.

**Two guards survived the void run and are worth more than it was.**

1. **The control reproduced 2.611 exactly** — Δ +0.000 — on the 1000 arenas the E41 run used.
   (Its own 4000-round mean is 2.778; rounds 1000–3999 are simply easier, at 2.834. Compare arenas,
   never headline means.)
2. **The external field is fully deterministic.** Re-evaluating the same table on the same arenas:
   **0 of 1000 rounds differ** against `binary_v6`, **999 of 1000** against `rule_based`. So the
   **±0.12 noise floor is a property of the `rule_based` opponents' unseeded stdlib `random`, not of
   the harness** — this file's header and `AGENTS.md` both state it as though it were universal.
   Paired comparisons on the external field carry **no opponent noise at all**, which makes the
   primary field of this experiment considerably more powerful than §Power assumed and means a
   single external-field evaluation *is* reproducible, unlike a `rule_based` one.

### Result

*(to be filled in after the repaired run — commit hash from the `.meta.json`)*

---

## E50 — the truncation bug is real, is not what the list said it was, and is not worth fixing

- **Question:** `NEXT_STEPS.md` §3.5 has carried this since rung 3: *"70 % of rounds hit `MAX_STEPS`
  and `end_of_round` treats truncation as termination with no bootstrap, worth ≈ 8.8 Q units —
  known since rung 3, never fixed in isolation."* It is the last non-reward item on the list. Before
  spending a sweep, characterise it.

- **What it actually is.** Traced through the framework rather than inferred:

  1. `agents.py:168` sets `last_game_state` inside `store_game_state`, which runs **before** `act`.
  2. `agents.py:155-159` `process_game_events` sends
     `game_events_occurred(last_game_state, last_action, current_state, events)` and **does not
     update `last_game_state`**.
  3. `agents.py:183-184` `round_ended` therefore sends `end_of_round(last_game_state, last_action,
     events)` with **the same state and action the step update just saw**.
  4. `environment.py:158-177` `do_step` calls `send_game_events()` and *then* `end_round()`, and
     `environment.py:467` skips dead agents in `send_game_events`.

  So the two paths differ:

  | round ends by | `game_events_occurred` fires? | `end_of_round` target | correct? |
  |---|---|---|---|
  | **death** | **no** — the agent is dead and is skipped | `reward`, no bootstrap | **yes** — it really is terminal |
  | **truncation / survival** | **yes**, with a correct bootstrapped target | `reward`, no bootstrap, **on the same (s, a)** | **no** |

  **It is not a missing bootstrap. It is a duplicate, un-bootstrapped update applied to a cell the
  step update had already updated correctly.** §3.5's description is wrong about the mechanism, and
  a fix written from that description — "add a bootstrap term to `end_of_round`" — would have made
  it *worse*, double-counting the transition with two bootstrapped targets instead of one.

- **How often it fires, and §3.5's other error.** The "70 % of rounds" figure is an **evaluation**
  number (ε = 0, survival 0.44). Training runs at ε start 0.2, where the agent almost always dies.
  Measured over full 20 000-episode runs (`SURVIVED_ROUND` per episode):

  | run | overall | last 2000 episodes |
  |---|---|---|
  | E48 arm C (s400) | 0.224 | 0.311 |
  | E48 arm D (s410) | 0.115 | 0.152 |
  | E37 `PLB2` (s100) | 0.104 | 0.158 |

  A 60-episode instrumented probe on the shipped configuration returned **0 survivals in 60**, which
  is what first exposed the discrepancy.

  So the bug fires on **10-22 % of episodes**, once each, against ~155 step updates per episode:
  **≈ 0.08-0.14 % of all Q updates.** And the affected cells are by construction the states the
  agent occupies at `MAX_STEPS` — the stripped board, which E47 measured as worth **2.6 % of score
  against a strong field and 10.0 % against `rule_based`**.

### Verdict — closed by diagnosis, not by a sweep

**The defect is real and the fix is one line, but no experiment this project can run could detect
its effect.** 0.1 % of updates, on rows in a phase worth a few percent of the score, against an
n = 8 MDE of 0.35. Pre-registering a sweep against it would be the fifth entry on this rung to
target something below its own detection threshold (§5.1, E33-E36, E40's unreachable +0.25, E48's
crate floor) — and this time the arithmetic is available *before* the run rather than from an audit
afterwards.

**Recommendation: do not change the shipped agent's training code.** The current table is the
product of this code path; altering it invalidates the warm-start parent that E44 measured at ~2
score, and buys an effect we have bounded at roughly nothing. **It goes in the report as a
characterised known defect with a measured bound**, which is a better outcome than a null sweep.

**What would change that:** if a future configuration raised training survival substantially — arm C
already sits at 0.311 in its last 2000 episodes — the share grows and the calculation should be
redone. The correct fix, for the record, is to **skip the `end_of_round` update when the agent
survived** (`e.GOT_KILLED not in events`), not to add a bootstrap term to it.

---

## E49 — the other side of the cliff: value coins more, rather than crates less

- **Question:** E48 established that `CRATE_DESTROYED` is a steep **throughput control** — cutting
  it 1.0 → 0.25 collapsed crates 32.98 → 15.25 and score 3.83 → 2.04 — and that `GOT_KILLED` is
  inert. It tested one knob in one direction. Two moves remain: raise the crate reward (E27's
  rung-3 direction, worth +0.93 there), or raise the **coin** reward, which has never been varied
  on this rung at all.

- **A correction to my own reasoning, recorded before the design.** I first proposed raising
  `BM_CRATE` on the strength of `corr(score, crates) = +0.937`. That correlation is **within-agent,
  across rounds**, and the between-agent comparison says the opposite:

  | vs 3 x `rule_based`, n = 1000 | our crates | opponents' | our share | score |
  |---|---|---|---|---|
  | shipped (3.828) | 33.28 | 89.38 | **27.1 %** | 3.828 |
  | `binary_v6` (5.572) | **25.09** | 96.90 | **20.6 %** | **5.572** |

  **The agent that beats us by 1.744 opens fewer crates than we do.** Reading a within-agent
  correlation across agents is the ecological-correlation trap `AGENTS.md` warns about, and I nearly
  spent a sweep on it. With E47's harvest figures (0.134 coins/crate against our 0.083) the
  mechanism is that **they collect coins revealed by the 89 crates the *field* opens** — free-riding
  our crate reward cannot reach. So crate throughput is not the winning axis, and the arm that
  targets the measured deficit is the **coin** price, not the crate price.

- **Change:** reward constants only, all existing environment switches.

  | arm | change | ratio | tests |
  |---|---|---|---|
  | **ctl** | `COIN=5`, `CRATE=1.0` | 5:1 | **free** — E37 `PLB2` is this table |
  | **K** (coin) | `BM_COIN=10` | 10:1 | harvest emphasis **without** touching the throughput control E48 showed is on a cliff |
  | **R** (crate) | `BM_CRATE=2.0` | 2.5:1 | E27's rung-3 direction, the mirror of E48's arm C |
  | **KR** | both | 5:1 | pure **scale** — the ratio is unchanged, so this isolates magnitude from balance |

- **Design.** 8 seeds per arm (`BM_RUN_INDEX` 500-507 / 510-517 / 520-527), 20 000 episodes,
  `--seed 810731`, warm start at default. Evaluation 300 rounds at **validation seed 550731** on
  the `rule_based` guard field and the held-out `bindist_v2` field; the control's evaluations on
  both already exist from E42. Identical infrastructure to E48, whose per-run cost was 1 h 23 m.

- **Power.** n = 8, between-seed score SD 0.249 → 80 %-power MDE **0.35**. E48's arms moved by
  1.5-2.2, so this design reads anything of that character easily; its risk is a *small* true effect,
  not a large one. Bootstrap CI **and** permutation p, `(fragile)` counts as not demonstrated.

### Prediction (written and committed before training starts)

1. **P1, primary — arm K.** Raising the coin price beats the control on `score` on the guard field
   by **≥ +0.35**. **My prediction: it does NOT clear.** `COIN_COLLECTED` already sits at 5 against
   the game's own +1, so coins are over-weighted relative to true score five times over, and the
   binding constraint on harvesting is *reaching* revealed coins, which is a pathfinding property
   (digit 6 already prioritises coins over crates, `callbacks.py:240-245`) rather than a price.
   **Refutation:** it clears → the coin price was the mis-set constant all along and E48 tested the
   wrong knob.
2. **P2 — arm R, and this one is a genuine two-sided test.** E48's arm C (crate 0.25) lost 1.789.
   If the response is monotone in the crate reward, arm R (crate 2.0) should **gain**. If it is a
   ridge with the current value near the peak, R should lose too. **Prediction: R loses less than
   0.35 but does not gain** — a ridge, not a slope. **Refutation either way is informative**, which
   is why the arm is here despite the between-agent evidence above pointing away from crates.
3. **P3, mechanism — required for any P1 claim.** If arm K gains, **coins per crate must rise by
   ≥ 0.010** (E48's arm C achieved exactly that while losing on throughput). A score gain in arm K
   without a harvest-efficiency gain means the effect came from somewhere I have not identified.
4. **P4 — scale versus balance.** Arm KR holds the 5:1 ratio and doubles the magnitude. With
   `GAMMA = 0.99` and `STEP_COST = 0` a pure rescale should be close to a no-op for the greedy
   policy. **Prediction: |Δscore| < 0.35.** **If KR moves, reward *magnitude* interacts with the
   visit-count learning rate** (`α = 1/visits^0.7`), which would be a finding about the optimiser
   rather than the reward, and would apply to every entry on this rung.
5. **Guards.** `crates` must stay above 28.0 in arms K and KR (they do not touch the crate reward,
   so a fall would indicate an unintended interaction); `invalid` must not rise; no collapsed seed
   by the training-band screen.

**Ship rule, pre-committed.** Unchanged from E48: an arm ships only if it beats the current ship on
the guard field with a CI excluding 0 and a non-fragile p, does not regress by more than the MDE on
the held-out field, and is then confirmed at 1000 rounds on the held-out ship seed 990731.

**This is the last reward lever.** `KILLED_OPPONENT` was closed by E35 (5 and 25, kills moved
−0.002), `GOT_KILLED` by E48 (inert), `CRATE_DESTROYED` downward by E48 and upward here, and
`COIN_COLLECTED` here. Custom-event shaping is closed three separate ways: E19 (potential-based —
and the failure is **structural**, since a row of this table buckets states with different Φ so the
offset cannot cancel between actions), E33 and E34 (action-based, paying for the escape step the map
already found). **If E49 does not move it, the reward table is calibrated and the remaining time
belongs to the truncation bug, the submission and the report.**

### Result — 3 arms x 8 seeds x 2 fields, 300 rounds, validation seed 550731

24 runs, ~1 h 20 m each. Arms verified from live metadata (K `COIN=10/CRATE=1.0`, R
`COIN=5/CRATE=2.0`, KR `COIN=10/CRATE=2.0`). No collapsed seeds — and unlike E48, no arm's
training band left the control's neighbourhood (K 2.66-3.12, R 2.63-2.82, KR 2.73-2.91, ctl ~2.99),
so the crate cliff E48 found is **one-sided**: cutting the reward is catastrophic, doubling it is
not.

**Guard field, 3 x `rule_based`** (cell means: ctl **3.832**, K **3.916**, R **3.746**, KR **3.845**):

| vs control | **K** (coin 10) | **R** (crate 2.0) | **KR** (both) |
|---|---|---|---|
| **score** | **+0.084 [−0.067, +0.230]** | −0.085 [−0.219, +0.044] | +0.013 [−0.140, +0.164] |
| coins | +0.053 | −0.113 | −0.077 |
| crates | −0.522 | −0.820 | −0.610 |
| coins/crate | **+0.003 [+0.001, +0.005]** | −0.001 | −0.001 |
| **suicides** | −0.050 | **+0.132** | **+0.198** |
| **survived** | +0.051 *(fragile)* | **−0.117** | **−0.192** |
| invalid | +0.342 | +0.452 *(fragile)* | **+0.873** |

Held out, `bindist_v2`: K **−0.186 [−0.377, +0.004]**, R **−0.163 [−0.289, −0.043]**,
KR **−0.202 [−0.324, −0.089]**.

**P1 REFUTED, as predicted.** Arm K moves `score` **+0.084** against a +0.35 bar, CI spanning zero,
p = 0.32 — and **−0.186 on the held-out field**. The prediction written before the run was that it
would not clear, because `COIN_COLLECTED` already sits at 5 against the game's own +1 and the
binding constraint on harvesting is *reaching* revealed coins (a pathfinding property of digit 6),
not their price. Nothing here contradicts that reading.

**P2 — the ridge is confirmed, and E48's cliff is one-sided.** Arm R loses **0.085** on the guard
field, well inside the 0.35 bar and not significant: as predicted, *"R loses less than 0.35 but does
not gain."* Against E48's arm C at **−1.789** for the same-sized move downward, the response is
strongly asymmetric — **the current 5:1 sits on a plateau with a cliff below it and flat ground
above.** Doubling the crate reward does not buy crates either: crates *fell* 0.820 on the guard
field. The reward is not a throughput dial in the upward direction.

**P3 — not triggered, and its one positive is worth recording.** P3 required a coins/crate rise of
≥ 0.010 for any arm-K score claim; there is no score claim to support. Arm K did raise
coins/crate **+0.003 [+0.001, +0.005]**, real but a third of the bar and a third of what E48's arm C
achieved while losing 1.789. **Harvest efficiency responds to price; score does not follow.**

**P4 PASSES on the guard field and FAILS held out.** KR holds the 5:1 ratio at double magnitude, so
a pure rescale should be a near no-op: guard **+0.013**, comfortably inside |0.35|. But held out it
is **−0.202 [−0.324, −0.089]**, CI excluding zero. The pre-registered consequence — *"if KR moves,
reward magnitude interacts with the visit-count learning rate"* — is therefore live in weak form.
With `α = 1/visits^0.7`, doubling every reward doubles every TD error while α is unchanged, so early
updates take proportionally larger steps. **This is a finding about the optimiser, not the reward
table**, and it applies to every entry on this rung that changed a reward magnitude. It is small
(−0.202 against a 0.35 MDE, on one field of two) and I am recording it as a lead, not a result.

**Guards: the informative failure is `invalid`.** It was pre-registered to not rise; it rises in
every arm and significantly in KR (**+0.873** guard, **+1.028** held out), and in R held out
(**+1.638**). Bigger rewards make the table pick illegal actions more often — consistent with the
α interaction above. `crates` stayed above 28.0 everywhere (32.2-40.0), so that guard passes.

**And the suicide result inverts E48's.** Arms R and KR *raise* suicides (+0.132, +0.198) and lower
survival (−0.117, −0.192) on the guard field. E48 showed removing the −5 death penalty changes
suicides by +0.010; here, raising the **crate** reward raises them by 0.132. **The agent's survival
behaviour is controlled by what it is paid to do, not by what it is paid to avoid** — which is a
cleaner statement of the eight-replication survival null than any entry has managed, and it comes
from the reward table rather than from a feature.

### Verdict — the reward table is calibrated; the last lever is closed

**Ship rule: nothing ships.** No arm beats the control with a CI excluding 0 on the guard field, and
all three are worse held out.

Across E48 and E49 the table has now been probed in four directions from its operating point:

| knob | direction | result |
|---|---|---|
| `CRATE_DESTROYED` | 1.0 → 0.25 | **−1.789** (E48) — a cliff |
| `CRATE_DESTROYED` | 1.0 → 2.0 | −0.085 — flat, and costs survival |
| `COIN_COLLECTED` | 5 → 10 | +0.084 guard, −0.186 held out |
| `GOT_KILLED` | −5 → 0 | **inert** (E48) — +0.097, suicides +0.010 |
| both, scale | x2 | +0.013 guard, **−0.202 held out** — an optimiser artefact |

**`COIN_COLLECTED = 5`, `CRATE_DESTROYED = 1.0`, `GOT_KILLED = −5`, `KILLED_OPPONENT = 0` is at or
adjacent to a local optimum**, and the one direction that is not flat is the one that falls off a
cliff. Combined with E35 (kill price, closed), E19/E33/E34 (shaping, closed three ways) and E50 (the
truncation defect, bounded at ~0.1 % of updates), **the reward table is done.**

### Limitations

- **Local, not global.** Four one-dimensional moves from one point. A jointly re-derived table
  (Bayesian optimisation over four knobs, ~100 runs) is the thing this cannot rule out, and it is
  not affordable in the remaining time.
- **One step size per direction**, and E48 showed step size matters enormously downward. A 1.0 → 0.5
  crate arm might sit between the cliff and the plateau; nothing here locates the edge.
- The P4 optimiser interaction is a lead on one field with n = 8, not a result. Testing it properly
  means varying `ALPHA_EXP` or `WARM_N` against a fixed reward scale, which is an experiment about
  the learner, not the rewards.

---

## E48 — the reward table has never been calibrated for rung 4

- **Question:** the reward scale was derived on **rung 3** (E27, worth **+0.93 score and +11.5
  crates** when `CRATE_DESTROYED` alone was corrected) and has been inherited unchanged through the
  whole of rung 4. Two of this rung's own measurements now say it is mis-set, in opposite
  directions from the ones anybody guessed:

  **(a) We over-pay for opening crates relative to harvesting them.** E47, same slot and field,
  n = 1000: we destroy **33.28** crates and collect **2.758** coins; `binary_v6` destroys **25.09**
  and collects **3.372**. Coins per crate **0.083 vs 0.134, +62 % for them**. `CRATE_DESTROYED` is
  pure shaping toward a quantity that **scores nothing** — `score = coins + 5·kills`, verified to
  within 0.003 on every field in E46 — so a crate reward that is too high buys crate-opening at the
  expense of the coin it reveals.

  **(b) We pay 5 for avoiding something measured to be worth less than nothing.** `GOT_KILLED` is
  −5, and this rung has now replicated **seven times** that survival does not convert into points —
  E46 being the first to price it: **+0.050 survival cost −0.283 score.** A penalty that buys
  survival is buying a negative.

- **What this entry deliberately does NOT do.** `xiaoxiae/BombermanML`'s table (read in E47) prices
  **no game outcome at all** — no `COIN_COLLECTED`, `CRATE_DESTROYED`, `GOT_KILLED` or
  `KILLED_OPPONENT`. Every term is a hand-authored per-step judgement (`MOVED_TOWARD_COIN: 50`,
  `PLACED_USEFUL_BOMB: 50`, `DID_NOT_MOVE_TOWARD_SAFETY: -500`) with `EPS_START = EPS_END = 0.0`.
  **That is a hand-written heuristic expressed as a reward function**, which means their 5.572 is
  not evidence that a *learned* policy reaches 5.572. We are not copying it: E33 and E34 already
  tested paying for the step our own map had found, and both failed; `AGENTS.md` further notes that
  shaping should be potential-based (state-dependent), not action-dependent. **Both arms below come
  from our own measurements.**

- **Change:** two reward constants, already environment switches, nothing else.

  | arm | change | motivated by |
  |---|---|---|
  | **ctl** | current: `BM_COIN=5`, `BM_CRATE=1.0`, `BM_GOT_KILLED=-5` | **free** — E37 `PLB2` s100-107 *is* this table |
  | **C** (harvest) | `BM_CRATE=0.25` → coin:crate 20:1 | (a): coins/crate 0.083 vs 0.134 |
  | **D** (no death price) | `BM_GOT_KILLED=0` | (b): seven replications, priced at −0.283 |
  | **CD** | both | the interaction |

- **Design.** 8 seeds per new arm (`BM_RUN_INDEX` 400-407 / 410-417 / 420-427), 20 000 episodes,
  `--seed 810731`, warm start at its default (E44 measured the parent at ~2 score — removing it is
  not an option, and E44's scratch arms reached only 44 % of the warm coverage). Evaluation 300
  rounds at **validation seed 550731** on the `rule_based` guard field and the held-out
  `bindist_v2` field; **the control's evaluations on both already exist from E42**, so only the
  three new arms need evaluating.

- **Power.** Between-seed score SD 0.249 → n = 8 gives an 80 %-power MDE of **0.35**. E27's
  rung-3 effect was +0.93, comfortably above it; a rung-4 effect of a third that size would still
  be readable. Every row scored on the bootstrap CI **and** the permutation p, `(fragile)` counting
  as not demonstrated.

### Prediction (written and committed before training starts)

1. **P1, primary — the harvest arm.** Arm **C** beats the control on `score` on the `rule_based`
   guard field by **≥ +0.35**, CI excluding 0 and non-fragile. **Prediction: it clears.** The crate
   reward is shaping toward a non-scoring quantity and our own harvest efficiency is 38 % below the
   agents that beat us. **Refutation:** below +0.35 → the 5:1 ratio was not the binding constraint
   and the E47 coins/crate gap is a *consequence* of their better play, not a cause of ours.
2. **P2, mechanism — required, or P1 is a coincidence.** Arm C's **coins per crate** rises by
   **≥ 0.02** (from 0.083, i.e. ≥ 25 % of the way to their 0.134). If score moves without
   coins/crate moving, the effect came from somewhere I have not identified and P1 does not count.
3. **P3 — the death penalty is not load-bearing.** Arm **D** does **not lose** more than the MDE:
   the paired CI's lower bound sits above **−0.35**. **Prediction: it holds, and suicides rise
   substantially while score does not fall.** That is the direct test of seven replications'
   worth of accumulated null. **Refutation:** a real loss → survival *does* have instrumental value
   that the seven nulls missed, and E46's −0.283 pricing is wrong or field-specific.
4. **P4 — interaction.** `CD − C − D + ctl`. No directional prediction; it is reported because a
   2x2 costs one extra arm and settles whether the two knobs are separable.
5. **Guards.** `crates` will fall in arm C (that is the point) and must not fall below **28.0**;
   `invalid` must not rise above the control's; `think_max_ms` untouched (rewards do not affect
   inference). All-zero-row share among visited rows < 0.01.

**Ship rule, pre-committed.** The shipped table changes **only if** an arm beats the current ship on
the guard field with a CI excluding 0 and a non-fragile p at 300 rounds on validation seed 550731,
**and** does not regress on the held-out `bindist_v2` field by more than the MDE, **and** is then
confirmed at **1000 rounds on the held-out ship seed 990731**. Selection never touches 990731.

**Why this is the right last big spend.** Hyperparameter optimisation is explicitly a graded
criterion, this is the cheapest untested lever on the `NEXT_STEPS.md` list, and it is the only one
whose prior effect size on this project (+0.93, E27) exceeds the gap we are trying to close.

### Result — 3 arms x 8 seeds x 2 fields, 300 rounds, validation seed 550731

24 runs, ~7 h at 5 lanes. Arm assignments verified from each run's live `hyperparams.rewards`,
not from the launcher. No collapsed seeds — training bands are tight within every arm
(C 2.13-2.31, D 2.79-3.12, CD 2.11-2.40). Tables in `scratchpad/benedict/e48/RESULTS.md`.

**Guard field, 3 x `rule_based`** (cell means on `score`: ctl **3.832**, C **2.042**, D **3.929**,
CD **2.273**):

| vs control | **C** (crate 0.25) | **D** (no death price) | CD |
|---|---|---|---|
| **score** | **−1.789 [−2.061, −1.501]** | **+0.097 [−0.030, +0.221]** | −1.558 |
| coins | −1.333 | +0.050 | −1.165 |
| kills | −0.091 | +0.010 | −0.079 |
| **crates** | **−17.735** | −0.035 | −16.235 |
| coins/crate | +0.010 | +0.002 | +0.011 |
| suicides | −0.178 | **+0.010** | −0.150 |
| survived | +0.160 | −0.020 | +0.122 |

Held-out `bindist_v2` reproduces it: C −2.228, CD −2.162, **D +0.000 [−0.162, +0.155]**.

**P1 REFUTED, and by five times the bar in the wrong direction.** I predicted arm C would clear
+0.35; it delivers **−1.789**. The pre-registered refutation clause was that the E47 coins/crate gap
would then be *"a consequence of their better play, not a cause of ours"* — and that is what the
mechanism says.

**Why it failed, and my reasoning was wrong in a specific way.** I argued `CRATE_DESTROYED` is
shaping toward a quantity that scores nothing, so paying less for it should redirect effort to
coins. But crates are not a proxy for coins — **they are the causal step that produces them.** Cut
the reward 4x and the agent stops bombing: crates **32.98 → 15.25**, and coins fall with them
**2.740 → 1.407**. The shaping was not mis-weighted; it was load-bearing.

**P2 FAILS as written, and the way it fails is the point.** The bar was coins/crate rising ≥ 0.020;
it rose **+0.010 [+0.004, +0.017]** on the guard field. So arm C *did* make each crate more
productive — half the pre-registered improvement — **and still lost 1.789 score, because it opened
less than half as many.** Efficiency rose, throughput collapsed. **A guard I wrote to keep P1 honest
ended up explaining P1's failure**, which is the most useful thing it could have done.

*(Estimator bug, caught before the entry was written: `coins_per_crate` was first computed as the
mean of per-round ratios, and arm C leaves many rounds with **zero** crates, so the `max(crates,
1e-9)` guard produced values of order 1e6. The correct estimator is the ratio of totals. The
figures above are the corrected ones.)*

**P3 PASSES, emphatically, and it is the finding.** Removing the **−5 `GOT_KILLED` penalty
entirely** changes **nothing**:

| arm D vs control | guard field | held out |
|---|---|---|
| score | +0.097 [−0.030, +0.221] | +0.000 [−0.162, +0.155] |
| **suicides** | **+0.010 [−0.049, +0.068]** | +0.008 [−0.014, +0.031] |
| survived | −0.020 [−0.075, +0.035] | −0.000 [−0.019, +0.017] |
| crates | −0.035 | −0.498 |
| kills | +0.010 | −0.003 |

The pre-registered bar was a CI lower bound above −0.35; it is **−0.030**. And it is stronger than
"survival does not convert": the penalty **does not even buy survival**. Suicides move by 0.010 with
a CI spanning zero, survival by −0.020, on both fields, at n = 8 per arm. Training suicides agree —
0.82/episode for arm D against the control's 0.82-0.84.

**The single largest negative term in the reward table is inert.** This is the eighth replication of
the survival null on this rung and the first to show the *price* is not doing the work either: E46
priced the exchange (+0.050 survival = −0.283 score); E48 shows we were not even buying the
survival we were paying for.

**P4 — the knobs are near-additive.** Interaction `(CD − C) − (D − ctl)` is **+0.133** on the guard
field and **+0.065** held out, both small against a 0.35 MDE. Arm C dominates and arm D contributes
nothing, in combination as in isolation.

**Guards.** `crates` fell to 15.25 in arm C against a pre-registered floor of 28.0 — **breached, as
the arm's own failure implies**. `invalid` did not rise in any arm. No table ships.

### Verdict — the reward table is better calibrated than it looked, in both directions

**Ship rule: nothing ships.** No arm beats the control with a CI excluding 0; C and CD are
catastrophically worse and D is indistinguishable.

Two results, and neither is the one this entry was designed to find:

1. **The 5:1 coin:crate ratio is not mis-set — it is close to a cliff.** Moving it to 20:1 costs
   1.789 score by suppressing bombing outright. E27 raised `CRATE_DESTROYED` on rung 3 and gained
   +0.93; this entry lowers it on rung 4 and loses 1.789. **The reward is a throughput control, and
   the current value is on the right side of it.** Whether a *higher* crate reward would help on
   rung 4 is now the obvious open question and this design did not test it.
2. **`GOT_KILLED = −5` can be set to 0 with no measurable effect on anything.** That is not an
   argument for changing it — a null is not a reason to move a shipped constant — but it retires
   the last version of "the agent needs to be taught to survive", which has motivated interventions
   on this rung since E30.

### Limitations

- **One step size per knob.** 1.0 → 0.25 is a 4x cut and it fell off a cliff; 1.0 → 0.5, or
  1.0 → 2.0, are untested and the second is the direction E27 found on rung 3.
- The death-penalty null is measured at the *current* crate reward. Arm CD shows the two are
  near-additive, so a joint effect is unlikely, but it is not excluded at other settings.
- `KILLED_OPPONENT` (0) and `COIN_COLLECTED` (5) were held fixed. E35 already closed the kill price
  at 5 and 25; the coin price has never been varied on this rung.

---

## E47 — target type, and where the 1.7-point gap actually lives

Two diagnostics, run together because they answer the same question from opposite ends: a probe of
our own aliasing (`scratchpad/benedict/e47_target_type.py`) and a targeted read of the two agents
that beat us (`scratchpad/xiaoxiae_read/FINDINGS.md`, GPL-3.0, `xiaoxiae/BombermanML` @ `50b682f`).
**No third-party code was copied**; the read produced mechanisms and `file:line` citations, which is
literature review, not the copy-pasting `AGENTS.md` forbids. 20 of 24 surveyed repos carry no
licence, so vendoring is also not legally available.

### Part 1 — is a target-type digit worth building?

`TASK_A_survey_vs_ours.md` §5.1 ranks this the corpus's best-evidenced feature and our largest
measured aliasing: digit 6 gives a *direction* with no *type*. Two things must hold, and the same
instrument E37 used for the lattice bit answers the first.

**(A) The type does carry real information.** 200 rounds at ε = 0, safe rows only:

| | 3 x `binary_v6` | 3 x `rule_based` |
|---|---|---|
| H(type) | 1.467 | 1.698 |
| **H(type \| full row)** | **0.782** | **1.042** |
| mutual information | 0.685 | 0.656 |

For scale, E37's lattice bit had H(lattice | digits 1-4) = 0.192 bits of 0.942 and was worth
+0.255 score. **The type residual is four to five times larger.**

**(B) But most of it sits where the score does not.** `callbacks.py:244-255` falls through to the
opponent *only when no crate is reachable*, so "opponent" is structurally the stripped board — and
the probe confirms it with no exceptions:

| crates left | coin | crate | opponent |
|---|---|---|---|
| 60+ | 25.5 % | 74.5 % | **0.0 %** |
| 30-59 | 22.8 % | 77.2 % | **0.0 %** |
| 10-29 | 19.4 % | 80.6 % | **0.0 %** |
| 1-9 | 7.0 % | 93.0 % | **0.0 %** |
| **0** | 0.7 % | 0.0 % | **73.0 %** |

And that phase is worth almost nothing: **2.6 %** of all score is earned after the board strips
against `binary_v6` (42/200 rounds reach it), **10.0 %** against `rule_based`. This corroborates
`TASK_B_argument.md` §1 on fresh data and from a different direction.

**(A2) Restricted to the crate phase, where 80 % of safe steps and ~95 % of the score are, the
question is only coin-vs-crate — and the residual halves:**

| crate phase only | 3 x `binary_v6` | 3 x `rule_based` |
|---|---|---|
| H(coin\|crate) | 0.730 | 0.943 |
| **H(coin\|crate \| full row)** | **0.480** | **0.633** |
| mutual information | 0.250 | 0.311 |

**Verdict: the free re-partition is still justified, on a residual 2.5-3x E37's.** But the headline
"39.1 % of steps point at an opponent and the table cannot tell" overstates the prize by folding in
a phase worth 2.6-10 % of the score. **The honest target is coin-vs-crate in phase 1**, not
three-way type disambiguation.

*A measurement bug, recorded because it nearly shipped a number.* The first run reported "68.3 % of
score is earned after the board strips", which contradicted TASK_B and was false:
`environment.py:276` writes `note_stat("score", ...)` only inside `end_round`, so
`agent.statistics["score"]` is **0 for the entire round** and my mid-round baseline was always zero.
The live value is `agent.score` (`agents.py:120,152`). Corrected figures are 2.6 % / 10.0 %.

### Part 2 — where the 1.744-point gap actually lives

Same slot, same field, n = 1000, from our own committed CSVs:

| | score | coins | kills | crates | coins/crate | suicides | survived |
|---|---|---|---|---|---|---|---|
| ours | 3.828 | 2.758 | 0.214 | 33.28 | 0.083 | 0.505 | 0.440 |
| `binary_v6` | **5.572** | 3.372 | **0.440** | 25.09 | 0.134 | 0.377 | 0.546 |
| `bindist_v2` | 5.336 | 3.216 | 0.424 | 25.10 | 0.128 | 0.276 | 0.641 |

**Kills are 65-70 % of the gap; coins are 30-35 %.** (The read that produced this comparison called
the deficit "harvesting"; its own numbers say otherwise, and the correction is recorded here rather
than repeated.) Decomposing the kills:

| | opp deaths | opp suicides | **takeable pool** | kills | **conversion** |
|---|---|---|---|---|---|
| ours | 1.822 | 1.514 | 0.308 | 0.214 | **69.5 %** |
| `binary_v6` | **2.146** | 1.664 | **0.482** | 0.440 | **91.3 %** |
| `bindist_v2` | 2.147 | 1.670 | 0.477 | 0.424 | 88.9 % |

**They do two separable things.** They *enlarge* the pool 57 % — `rule_based` dies 2.146 times a
round against them versus 1.822 against us — and they *convert* 91.3 % of it against our 69.5 %.

**This partially reopens a question E40 closed.** E40 ruled out hunting-by-trap-positioning, but
audit 11 established that E40's real measured effect was **opponents suiciding more when
approached** (+0.046, t = +4.17) — pool enlargement, at a third the scale these agents reach, from
an oracle firing on 1.4 % of steps. **What E40 refuted is trap-seeking; sustained pressure is a
different intervention and has never been measured.** E31 tested "reckless play", but as a reward
arm at n = 5 on `won`, which §5.1 shows was unreadable by construction.

**What they do NOT have, checked directly:** no action mask, no bomb veto — `act` is a bare argmax
(`binary_agent_v6/callbacks.py:33-38`). Their suicides are 0.377 against our 0.505: halved, not
eliminated, and E46 already priced that exchange at −0.283 score. **Their edge is not survival.**

Three representational differences worth recording: four *separate typed* objective channels rather
than one overloaded direction digit; objective BFS run over *simulated future* states rather than a
static board (ours is time-aware only in `escape_direction`, only in danger rows); and crate goals
defined as *firing positions* (crate in an adjacent cell) rather than the crate itself — note E10
rejected the radius-3 version of that, theirs is radius-1.

### What this changes

**Ranked, and neither item is a new feature:**

1. **The conversion gap, 69.5 % → 91.3 %, is worth ~+0.34 score inside our existing pool** and needs
   no change in aggression. It is above the n = 15 MDE and has never been diagnosed.
2. **The free target-type re-partition** remains justified at a 0.250-0.311 bit residual, but the
   prize is coin-vs-crate in phase 1, not the three-way split the survey framed.
3. **Rung-4 reward recalibration (§3.2) moves up.** Their scheme is roughly 100:1 coin:crate with
   *zero* death penalty and *zero* kill reward; ours is 5:1 with −5 on death. E27 showed on rung 3
   that this scale was mis-set and worth +0.93 when fixed, and it has never been re-derived here.
   Every knob is already an environment variable, so this is the cheapest sweep on the list.

**Still open and unaffected:** §3.5 the truncation bug; the mixed-agent line-up; the tournament
format question; `docker build`.

---

## E46 — don't bomb without escape *room*: the zero-slack gate

- **Question:** Benedict, from watching play: *"the agent still bombs when nothing is on the
  board — maybe it should only place one when really needed."* The observation is right and the
  inference is not, and E45 measured both.

  **Right:** 72.9 % of armed steps have zero crates in blast range and the policy bombs on 18.8 %
  of them (`bomb_siting.py` reported 71.5 % / 16 %; independently reproduced).

  **Wrong:** those bombs are not what kills us. By the crate count of the **killing** bomb, against
  3 x `binary_v6`: 0 crates is 32.2 % of bombs but 13.0 % of own-bomb deaths (**lift 0.40x**), while
  3+ crates is 33.7 % of bombs and **52.0 %** of deaths (**lift 1.54x**). Truly worthless bombs
  (0 crates *and* 0 opponents) cause **0.0 % / 1.2 %** of own-bomb deaths. **Bombing "only when
  needed" would delete the safe half and keep the killers**, and pay crates for it. The mechanism is
  structural: a bomb reaching 3+ crates is by definition in a dense pocket, where the blast is
  contained and the escape routes are the crates being destroyed.

  **What does discriminate is escape *slack*.** The fuse is `BOMB_TIMER = 4`, so a bomb whose
  nearest safe tile is 4 steps away has zero room for interference:

  | escape distance at bomb time | share of bombs | share of own-bomb deaths | lift |
  |---|---|---|---|
  | 2 | 50.5 % | 22.0 % | 0.44x |
  | 3 | 45.4 % | 49.5 % | 1.09x |
  | **4 (zero slack)** | **2.9 %** | **20.5 %** | **7.10x** |
  | **5+ (no escape at all)** | 1.3 % | 8.0 % | 6.32x |

  **4.2 % of bombs cause 28.5 % of own-bomb deaths.** And the `d = 4` row is *not* the feature
  `TASK_A_survey_vs_ours.md` §1.2 dismissed: that one is "does an escape exist", which is the
  `d >= 5` row alone and is 99.08 % constant. **"An escape exists but with zero slack" has never
  been measured by anyone in this project or in the surveyed corpus.**

- **Change:** an oracle veto. When the shipped greedy policy says `BOMB`, compute the post-bomb
  escape distance with the same time-aware BFS `callbacks.escape_direction` uses; if it is >= the
  arm's threshold, take the best non-`BOMB` action from the *same* Q-row instead, so the veto
  changes only the bomb decision. `--veto-at 0` disables. No training; nothing under `agent_code/`
  is written; the line-up drives the provided `user_agent`.

- **Design.** Four arms x two fields, **n = 4000** paired arenas, held-out ship seed 990731.
  Primary field is **3 x `ext_xiaoxiae_binary_v6`** — external, per E41, because every previous
  ceiling test on this project was run against the one opponent E41 showed to be unrepresentative.
  Secondary field 3 x `rule_based_agent`.

  | arm | role |
  |---|---|
  | `veto 0` | control — the shipped agent |
  | `veto 5` | **placebo / internal control**: vetoes only bombs with no escape at all, i.e. TASK_A's near-constant bit. Should reproduce the control |
  | **`veto 4`** | **the arm** — zero-slack bombs vetoed |
  | `veto 3` | the "gate harder" bound, powered this time rather than read off noise (E40's over-read) |

- **Power and reachability, stated before the bar.** At n = 4000 the standard error on `score` is
  ~0.06 (E40's realised value), so the 80 %-power MDE is **~0.17**. **Is +0.25 reachable by this
  design?** E45 bounds the mechanism at 28.5 % of 0.667 own-bomb deaths/round = **0.19 deaths/round
  addressable**, and E41's cross-field data puts roughly 9 score per unit of survival
  (3.828 at survived 0.440 vs 2.611 at 0.310). So the bar is reachable if the intervention converts
  even a fraction of the addressable deaths. **This check is here because E40 pre-registered +0.25
  on a design whose own arithmetic could not reach it, and that was only caught by audit 11.**

- **Disclosure — this pre-registration is not blind.** A 150-round pilot was run to fix the design,
  because the first version of this gate (require >= 2 post-bomb escape *routes*) vetoed **99.6 %**
  of bombs and collapsed score to 0.140 — a bomb covering your tile and all four arms usually leaves
  exactly one way out, so the route count cannot be a gate. That pilot also showed `veto 4` at
  score 2.807 vs 2.620 and suicides 0.400 vs 0.593. **n = 150 is far below the +/-0.12 noise floor,
  so it is a lead, not a result** — but I saw it, and the bar below is therefore set with knowledge
  of the direction. Recorded rather than hidden.

### Prediction (written and committed before the n = 4000 run)

1. **P1, primary.** `veto 4` beats the control on **`score`** by **>= +0.25** on the external field,
   CI excluding 0 **and** a non-fragile permutation p. **Refutation:** below +0.25 → the zero-slack
   gate is real but too small to build, and the pre-committed consequence is that no digit is
   designed for it.
2. **P2, mechanism — this must hold or P1 is a coincidence.** `suicides` fall by **>= 0.10** and
   `survived` rises. E45 says the gate can only work by removing 7x-lift bombs; if score moves
   without suicides moving, the effect is coming from somewhere else and P1 does not count.
3. **P3, the cost.** `crates` fall (the pilot lost 3.6). The gain must survive that — score is the
   primary, not crates, and `score = coins + 5·kills` exactly, so crates are only an input.
4. **P4, the placebo.** `veto 5` moves **nothing** (|Δscore| < 0.17, the MDE). It vetoes only the
   0.92 % of bombs with no escape at all — TASK_A's constant. **If `veto 5` shows a large effect,
   the harness is wrong and P1 must be discarded**, because that arm should be a no-op.
5. **P5, the direction bound.** `veto 3` is **worse** than the control, with a CI excluding 0.
   The pilot has it at 0.593 vs 2.620, so this should be unmissable; it exists to bound "gate
   harder" empirically rather than by assertion.
6. **Guards.** `think_max_ms` irrelevant (offline harness), but the control arm must reproduce the
   E41 head-to-head number (2.611 score, −1.008 margin) within the noise floor.

**Ship rule, pre-committed.** Nothing ships from a ceiling. What this decides: **if P1 and P2 both
clear, a digit is designed — and it is cheap, one bit, "bombing here leaves zero slack"**, which can
go into digit 7 (currently a single bit) as a 2-bit split at a factor-2 table cost, keeping the
current table as a warm-start parent. E44 measured that parent at ~2 score, so preserving it is not
optional. **If P1 fails, this closes the last untested lever on the `NEXT_STEPS.md` list and the
tabular line is done.**

### Result — 4 arms x 2 fields, n = 4000 paired arenas, held-out ship seed 990731

**Control validates:** 2.778 score / −0.949 margin against E41's head-to-head 2.611 / −1.008, and
3.944 / +0.945 on `rule_based` against the shipped 3.949. Guard passes.

**3 x `ext_xiaoxiae_binary_v6` (primary, external):**

| arm | score | margin_mean | suicides | survived | crates | bombs |
|---|---|---|---|---|---|---|
| control | 2.778 | −0.949 | 0.540 | 0.302 | 39.12 | 21.59 |
| veto 5 (placebo) | 2.748 | −1.016 | 0.536 | 0.290 | 38.77 | 21.19 |
| **veto 4** | **2.495** | **−1.308** | **0.409** | **0.352** | 34.71 | 19.18 |
| veto 3 | 0.523 | −3.608 | 0.117 | 0.509 | 5.82 | 3.82 |

| veto 4 − control | external field | `rule_based` |
|---|---|---|
| **score** | **−0.283 [−0.349, −0.212]** | **−0.156 [−0.271, −0.042]** |
| margin_mean | −0.359 [−0.453, −0.265] | −0.232 [−0.373, −0.092] |
| **suicides** | **−0.131 [−0.147, −0.115]** | −0.023 [−0.043, −0.001] |
| **survived** | **+0.050 [+0.035, +0.065]** | +0.017 [−0.005, +0.037] |
| coins | −0.233 | −0.162 |
| crates | −4.412 | −2.110 |

**P1 REFUTED — and in the wrong direction.** The bar was **+0.25**; the arm delivers **−0.283**,
significantly *worse*, on both fields. The pre-committed consequence applies: **no digit is
designed for this.**

**P2 PASSES — the mechanism worked exactly as designed.** Suicides fell **0.131** (bar was 0.10)
and survival rose **+0.050**, both with CIs excluding zero and p < 0.0001. **The gate prevented the
deaths E45 said it would prevent, and the agent still lost points.**

**P3 is where it dies.** The gate costs **−4.412 crates** and **−0.233 coins**. `score = coins +
5·kills` exactly, and kills did not move (−0.010, ns), so the whole loss is coins: **−0.233 coins ≈
−0.283 score**, closing to within 0.05. Survival bought nothing; the forgone economy cost
everything.

**P4 PASSES — the placebo is a no-op on the pre-registered metric.** `veto 5` moves score by
−0.029 (external, ns) and +0.017 (`rule_based`, ns), both far inside the 0.17 MDE, at veto rates of
0.3 % and 1.6 %. The harness is not manufacturing effects. *(Its `margin_mean` −0.067 and
`survived` −0.012 do reach significance on the external field; at n = 4000 a 0.3 % intervention can
show a real sliver, and it is small enough not to threaten P1 — which failed in the opposite
direction anyway.)*

**P5 PASSES, unmissably.** `veto 3` is −2.255 score, and this time it is bounded rather than
asserted: E40 claimed its k = 8 arm "bounds the chase-harder direction" on a t of 1.09, which
audit 11 correctly called an over-read. This one is −2.255 [−2.334, −2.179].

### Verdict — the seventh replication, and it explains the previous six

**The zero-slack bombs are both the most lethal and the most productive, and on net they are worth
placing.** E45 found 3+ crate bombs carry a 1.54x death lift; this entry shows why that is not a
defect to fix. A bomb in a dense pocket has a contained blast and a tight escape *because* it is
surrounded by crates — the risk and the reward are the same geometric fact. Removing 11 % of bombs
removed 4.4 crates and 0.23 coins to buy 0.13 fewer suicides, and the trade is losing.

This is now the **seventh independent replication that survival does not convert into points on
this board** (E30, E31, E33, E34, E36, E44's `scratch x mix` survival gain, and this) — and the
first one that isolates the exchange rate: **+0.050 survival cost −0.283 score.** Previous entries
observed the null; this one prices it.

**Benedict's hypothesis is refuted in both of its forms.** "Bomb only when needed" fails because the
worthless bombs are the *safe* ones (E45: 0.40x death lift). "Bomb only with escape room" fails
because the tight bombs are the *productive* ones. **The agent's bombing policy is not the deficit.**

### What this closes

The pre-registration committed to the consequence, and I then **overstated it when writing this
verdict up. Corrected 2026-08-20:** P1 failing closes **§3.3 (bomb siting)**, not "the last untested
lever". Checked against the list rather than from memory:

| `NEXT_STEPS.md` §3 | status |
|---|---|
| 3.1 target type — **free** re-partition of digit 8 in the safe rows | **UNTESTED.** Audit 10 F6 killed the *size-4* variant's cost justification and said so explicitly; the zero-row variant was never touched |
| 3.2 recalibrate rewards for rung 4 | **UNTESTED.** E27 did this on rung 3, never on this one |
| 3.3 bomb siting | closed by E45/E46 |
| 3.4 opponent-induced suicide | closed by E43 (2.3 % ceiling) |
| 3.5 the truncation bug | **UNTESTED.** A known correctness defect since rung 3 |

So the honest statement is narrower: **E40 (hunting), E43 (opponent-danger digit), E44 (training
distribution) and E46 (bombing discipline) are closed. Three levers remain**, and one of them —
target type — is the corpus's single best-evidenced feature (`TASK_A_survey_vs_ours.md` §5.1).

That is a bounded, mechanised negative rather than an absence of results: the deficit is a
policy-quality problem inside states the features already describe (E43: 76.7 % of deaths had two or
more surviving actions available), and this rung has now tested representation, training
distribution, aggression and bombing discipline against it.

### Limitations

- **The pilot's sign was wrong.** At n = 150 `veto 4` measured +0.187 score; at n = 4000 it is
  −0.283. A 0.47 swing between a pilot and its confirmation, on the same arm and the same seed
  family. **This is the cleanest demonstration in the ledger of why the +/-0.12 noise floor matters**
  — and the disclosure written into this entry's pre-registration, that the bar was set knowing the
  pilot's direction, turns out to have protected a bar the pilot pointed the wrong way.
- One external field. The `rule_based` field replicates the sign and the mechanism at roughly half
  the magnitude, which is consistent with its lower death rate, but two fields is two fields.
- The oracle recomputes true escape distance every step; a digit could offer only a bucketed
  version. Since the oracle *loses*, that gap does not matter here.

---

## E45 — what did the bomb that killed us actually destroy?

- **Question:** Benedict, from watching play: *"the agent still bombs when nothing is on the board —
  maybe it should only place one when really needed."* `scratchpad/strategy/bomb_siting.py` had
  measured the setup (71.5 % of armed steps have zero crates in range, the policy bombs on 16 % of
  them) and E43 had measured the deaths (93.5 % are our own bomb), but nobody had **joined them**:
  what were the bombs that killed us worth?

- **Method.** `scratchpad/benedict/e45_bomb_value.py`. Roll out the shipped table at ε = 0. Tag every
  bomb at placement with what its blast could reach — crates, opponents. On death, trace which of
  our bombs' blasts covered the tile we died on, restricted to bombs whose fuse could still be live,
  and report that bomb's tag. 300 rounds per field, against an external field and `rule_based`.

  **Two instrument bugs, both found before the numbers were believed.** The first version blamed any
  bomb whose blast geometry covered the death tile regardless of when it was placed; the second
  compared placement time against `world.step` *at round end* rather than at our own death step,
  which made every bomb look long expired and returned an absurd 0.5 % own-bomb death rate. The
  corrected run reproduces E43's independent 93.5 % from a separate instrument, which is the check
  that the fix is right.

### Result — 300 rounds per field, ε = 0, validation seed 550731

| | 3 × `binary_v6` | 3 × `rule_based` |
|---|---|---|
| armed steps with **0 crates in range** | 53.7 % | **72.9 %** (`bomb_siting.py`: 71.5 %) |
| P(BOMB \| 0 crates) | 18.3 % | **18.8 %** (`bomb_siting.py`: 16 %) |
| bombs placed per round | 21.6 | 28.0 |
| **killed by one of our own bombs** | **93.0 %** | **93.7 %** |

**The observation is confirmed and the inference from it is refuted.** Ranked by what the *killing*
bomb reached:

| crates the killing bomb reached | share of deaths | share of bombs placed | **lift** |
|---|---|---|---|
| **0** | 13.0 % | 32.2 % | **0.40×** |
| 1 | 16.5 % | 14.6 % | 1.13× |
| 2 | 18.5 % | 19.5 % | 0.95× |
| **3+** | **52.0 %** | 33.7 % | **1.54×** |

*(external field; against `rule_based` the same ranking gives 0.95× / 1.40× / 1.04× / 0.81× on a
board where 48.8 % of bombs clear no crate at all)*

**Bombs that clear nothing are the *safe* ones.** Truly worthless bombs — 0 crates *and* 0 opponents
— cause **0.0 % / 1.2 %** of own-bomb deaths; suppressing them entirely would save ~0.007 deaths per
round. "Bomb only when needed" would delete the harmless half and keep the killers, and pay crates
for it.

**What discriminates is escape slack.** The fuse is `BOMB_TIMER = 4`, so a bomb whose nearest safe
tile is 4 steps away has no room for interference:

| escape distance at bomb time | share of bombs | share of own-bomb deaths | lift |
|---|---|---|---|
| 2 | 50.5 % | 22.0 % | 0.44× |
| 3 | 45.4 % | 49.5 % | 1.09× |
| **4 (zero slack)** | **2.9 %** | **20.5 %** | **7.10×** |
| **5+ (no escape at all)** | 1.3 % | 8.0 % | 6.32× |

**4.2 % of bombs cause 28.5 % of own-bomb deaths.** And the `d = 4` row is *not* the feature
`TASK_A_survey_vs_ours.md` §1.2 dismissed as a near-constant: that one is "does an escape exist",
the `d ≥ 5` row alone, which fires on 0.92 % of armed steps. **"An escape exists but with zero
slack" had never been measured by this project or the surveyed corpus.**

### Verdict

The bombing *rate* is not the deficit; the bombing *geometry* might be. E46 gates on the zero-slack
row and settles it.

---

---

## E44 — the retrain E42 claimed to be: from scratch, both arms, a 2x2

- **Question:** audit 12 established that E42 never ran the experiment its title claims.
  `train.py:242` defaults `BM_WARM="_parent"` with `WARM_N = 100`, and `ALPHA_EXP = 0.7`, so the
  first update on any warm row runs at **α = 1/100^0.7 = 0.0398** — on a parent trained entirely
  against `rule_based_agent`, covering 84 % of the arm's updated rows. E42 was a **fine-tune of a
  rule_based-specialised table**, not training against a mixed field. Whether the training field
  matters is therefore still open.

- **Change:** `BM_WARM=""` (verified: `.meta.json` records `"warm": ""` and `train.py:263` skips
  the warm-start branch entirely), and the training opponents. Nothing else.

- **Design — and the two expensive cells already exist.** The warm row of the 2x2 is already on
  disk, so only the scratch row needs compute:

  | | 3 x `rule_based` | mixed field |
  |---|---|---|
  | **warm** (α starts 0.0398) | E37 `PLB2` s100-107 — **free** | E42 s200-207 — **free** |
  | **scratch** (α starts 1.0) | **new**, `BM_RUN_INDEX` 300-307 | **new**, 310-317 |

  Mixed field is `ext_xiaoxiae_binary_v6` + `ext_aielka_ql_atom` + `rule_based_agent`, unchanged
  from E42 so the two rows are comparable. 8 seeds per new arm, 20 000 episodes, `--seed 810731`.
  Evaluation identical to E42's: 300 rounds, **validation seed 550731**, on the **held-out**
  `bindist_v2` field, the in-distribution mixed field, and the `rule_based` regression guard.
  Pairing on evaluation arenas only; the unit of inference is the **seed**.

- **Power.** Between-seed score SD 0.249 → n = 8 gives an 80 %-power MDE of **0.35**, same as E42.
  Coarse on purpose: the deficit is −1.0 and anything under 0.35 does not change the project.
  Every row scored on the bootstrap CI **and** the permutation p, with `(fragile)` counting as not
  demonstrated — the rule `e42_analyze.py` failed to apply and now does.

- **Measured cost, not guessed.** Pilot (`scratchpad/benedict/e44/pilot_*.log`, `BM_WARM=""`):
  from scratch reaches score 0.550 at episode 2000 against `rule_based` and 0.165 at episode 1000
  against the mixed field, versus 0.975 for the warm arm at episode ~1100. **From scratch learns,
  and it learns slower** — which is what P2 guards.

### Prediction (written and committed before training starts)

1. **P1, primary — the field effect, tested properly at last.** On the **held-out** `bindist_v2`
   field, `scratch x mixed` beats `scratch x rule_based` on `margin_mean` by **≥ +0.35**.
   **My prediction is that it FAILS**, and E43 is why: the *composition* of our deaths is
   field-independent (78.6/14.9/4.2/2.3 against a strong DQN vs 79.1/16.5/3.3/0.5 against
   `rule_based`), 93.5 % are our own bombs, and in 76.7 % of them a surviving action existed. If
   how we die does not depend on the opponent, the training opponent should not fix it.
   **Refutation:** it clears +0.35 → the field *does* matter once the learning rate lets it, E42's
   null was an artefact of α = 0.04, and mixed-field training becomes the main line.
2. **P2, the guard that decides whether P1 is readable at all.** From-scratch at 20 000 episodes
   may simply be undertrained. **Both scratch arms must reach at least 80 % of their warm
   counterpart's score on the `rule_based` guard field** (warm x rb scores 3.832, so the bar is
   **≥ 3.07**), and the ep5000/ep10000/ep20000 checkpoints must not still be climbing steeply at
   the end. **If P2 fails, P1 is uninterpretable** — a null between two undertrained arms says
   nothing about the field — and the entry reports that rather than a field result.
3. **P3, the interaction — the actual scientific content of the 2x2.** Is the field effect
   different at α = 1.0 than at α = 0.04? Formally: `(scratch_mix − scratch_rb)` vs
   `(warm_mix − warm_rb)` on the held-out field. E42 measured the warm difference at
   **−0.043 [−0.174, +0.082]**. **Prediction: the two differences agree within the MDE**, i.e. the
   warm start was not what suppressed the field effect.
4. **P4, coverage guard, specific to training from zero.** All-zero-row share among *visited* rows
   must stay under 0.01 at ε = 0 (the guard E37 used, where it ran 0.00006). A from-scratch table
   with unvisited rows falls back to the tie-break and would look like a policy failure that is
   really a coverage failure. Also `suicides` reported per field, never differenced across fields
   (audit 10 F5), and split by death step (E42's correction) rather than pooled.

**Ship rule, pre-committed.** Nothing ships unless a scratch seed beats the current ship on the
held-out field with a CI excluding 0 **and** a non-fragile permutation p, survives the
`rule_based` regression guard, and is then confirmed at 1000 rounds on the held-out ship seed
990731. Selection on 550731 only.

**What this entry cannot settle.** E43 already refuted the opponent-danger digit at a 2.3 % ceiling,
and E42's correction withdrew the "digit first" recommendation. **If P1 also fails, the honest
reading is that neither the training distribution nor an opponent-danger feature closes the −1.0
gap, and the tabular line is at its ceiling.** That is a legitimate result and it is written here
before the run so it cannot be softened afterwards.

### Result — 16 new runs, 48 evaluations, 300 rounds, validation seed 550731

Trained 2026-08-19/20. All 16 reached 20 000 episodes, `"warm": ""` confirmed in the live runs'
metadata (not just the pilot's). Tables in `scratchpad/benedict/e44/RESULTS.md`.

**The completed 2x2** (score / `margin_mean`):

| | held-out `bindist_v2` | in-distribution | guard `rule_based` |
|---|---|---|---|
| **warm x rb** (E37 `PLB2`) | 2.965 / −1.000 | 3.342 / −0.045 | **3.832 / +0.795** |
| **warm x mix** (E42) | 2.850 / −1.043 | 3.181 / −0.164 | 3.782 / +0.804 |
| **scratch x rb** | 1.305 / −2.642 | 1.406 / −2.206 | **1.813 / −1.522** |
| **scratch x mix** | 1.203 / −2.684 | 1.271 / −2.312 | 1.432 / −1.966 |

**P2 FAILS, decisively, and it was written to be the gate.** The bar was 3.07 on the guard field
(80 % of warm x rb's 3.832). `scratch x rb` reaches **1.813**, `scratch x mix` **1.432** — 47 % and
37 %. **Both scratch arms are badly undertrained, so P1 cannot be read as an absolute answer**, and
the pre-registered consequence applies: this entry reports that rather than dressing a tie up as
evidence.

**P4 gives the mechanism, and it is coverage, not episodes.** Rows carrying any learned value:

| arm | nonzero rows of 64 000 |
|---|---|
| warm x rb | 8 168 (8 145–8 186) |
| warm x mix | 8 631 |
| **scratch x rb** | **3 595** (3 389–3 772) |
| **scratch x mix** | 4 095 |

The scratch tables reach **44 % of the warm tables' coverage**. 20 000 rung-4 episodes cannot
rebuild what the rung-2 parent supplies, which is exactly why `WARM_N` exists.

**P1 — not demonstrated, as predicted.** `scratch x mix − scratch x rb` on the held-out field:
`margin_mean` **−0.042 [−0.251, +0.181]**, p = 0.73, against a +0.35 bar. E43's reason for the
prediction stands: if the composition of our deaths is field-independent, the training field should
not fix it. But P2 means this is a null between two weak agents, and I am not claiming it as the
answer.

**P3 CONFIRMED — and it is what makes the entry worth its compute.** The field effect on the
held-out field is **−0.043 at α = 0.0398 (warm) and −0.042 at α = 1.0 (scratch): a difference of
+0.001.** In-distribution, −0.119 vs −0.106. **The warm start was not what suppressed the field
effect in E42.** This is a difference-of-differences between two equally-trained arms *within* each
row, so it survives P2's failure — the absolute level is uninterpretable, the contrast is not.

It also closes the cheaper option I passed over. I had considered keeping the warm start with
`BM_WARM_N=1` so α starts near 1.0 with coverage intact; **P3 says the learning rate is not the
variable**, so that arm would measure the same nothing.

*(The guard-field cell is the exception, −0.453, and it should be: `scratch x mix` is the only arm
that never trains on a pure `rule_based` field, so it is the only one being tested out of
distribution there.)*

**One positive row, reported because it is the only one.** On the held-out field `scratch x mix`
survives more than `scratch x rb`: **+0.079 [+0.017, +0.147]**, and it is *stable* — the CI excludes
0 on **60/60** bootstrap seeds, permutation p 0.044–0.049 across five seeds. It is real, it is
borderline, it is a secondary metric on an undertrained arm, and it does not convert: score −0.102,
`margin_mean` −0.042. **Consistent with six previous replications that survival does not become
points on this board** — now extended from `rule_based` to a strong DQN field.

**And the largest effect on this rung was never an intervention.** The rung-2 warm parent is worth
**−1.661 / −1.936 / −2.019 score** across the three fields (all p ≤ 0.0001). That is **eight times
E37's shipped +0.255** and dwarfs every feature, reward and horizon change E28–E42 tested. The
agent that ships is mostly the rung-2 table plus a lattice bit; rung-4 training refines it.

### Verdict — the training distribution is not the lever, measured two ways

`margin_mean` on the held-out field moves **−0.043** when a well-trained table is fine-tuned on a
mixed field, and **−0.042** when a table is trained on one from zero. Two learning rates spanning a
factor of 25, the same answer.

**The pre-registered reading applies, and it was written before the run so it cannot be softened
now:** E43 refuted the opponent-danger digit at a 2.3 % ceiling, and E44 finds no field effect at
either learning rate. **Neither the training distribution nor an opponent-danger feature closes the
−1.0 gap to the published SS2024 agents. On both axes tested, this tabular line is at its ceiling.**

That is not a failed experiment; it is a bounded negative with a mechanism (E43: 93.5 % of deaths
are our own bomb, in states the features already describe, with 2+ surviving actions available
76.7 % of the time). **The remaining deficit is a policy-quality problem inside states the agent can
already see** — and E33 and E34 both already failed to fix exactly that.

### Limitations

- **P2's failure is the dominant one.** A from-scratch arm at 20 000 episodes is not a fair test of
  from-scratch training; it is a test of a 20 000-episode budget without a parent. A longer
  from-scratch budget is untested and, given E38's inverted U *with* a parent, not obviously safe.
- The interaction in P3 rests on two cells whose absolute level is low. It says the *field* effect
  is invariant to α; it does not say a well-trained from-scratch mixed agent would behave the same.
- One held-out field, one mixed composition, `bindist_v2` sharing an author with a training
  opponent (stated before the run).
- Coverage is measured as nonzero rows in the table, not as all-zero rows among *visited* rows at
  ε = 0, which is what P4 actually pre-registered. The stronger version needs a visitation rollout
  and was not run; the 44 % figure is sufficient to explain P2 and is not sufficient to score P4
  exactly as written. **Scored: not measured as specified.**

---

## E43 — when we die, could any feature have saved us? The digit is refuted before it was built

- **Question:** E42 concluded the mixed-field retrain failed because opponent-induced deaths are
  *"unattributable given our eight digits"*, and I turned that into a recommendation: **build an
  opponent-bomb-danger digit first, retrain second.** That is a claim about the *representation*
  and it makes a testable prediction. `AGENTS.md` §3.6 and E39/E40 both say the same thing —
  **ceiling-test before building** — so this probe runs before any digit is designed.

- **Method.** `scratchpad/benedict/e43_attributability.py`. Roll out the shipped table at ε = 0
  (`HuntCeiling(q, -1)`, byte-identical to the shipped agent). At every step record whether *any*
  action still survives, using only the bombs visible on the board at that moment. For each death
  find **`t_doom`** = the last step at which a surviving action existed, then trace which bomb's
  blast actually covered the tile we died on, and ask whether that bomb was on the board at
  `t_doom`:

  | class | meaning |
  |---|---|
  | **OWN_BOMB** | the killing blast is ours → escape-logic failure; already in digits 1–5 |
  | **OVERLAP** | own + enemy blast on the same tile → the `evaluate.py:258` blind spot (audit 10 F5) |
  | **VISIBLE** | enemy bomb **already placed** at `t_doom` → the information was in `danger_map`, so it is already in digits 1–5 and **a new digit cannot help** |
  | **UNSEEABLE** | enemy bomb placed **after** we committed → no feature of the current `game_state` could have shown it. **This, and only this, is the case for the digit.** |

  Run against an **external** field, never `rule_based` alone — E41 showed `rule_based` is
  unrepresentative (it suicides 1.487 of its 1.820 deaths/round, so it barely hunts).

  **One instrument bug, found and fixed before the numbers were believed.** The first version
  flagged a death as OWN_BOMB whenever *any* of our bombs was on the board — which for this agent
  is nearly always — and returned a meaningless 100 %. The fix traces `blast_coords` from the tile
  we actually died on. The pilot that produced the 100 % is the reason this is stated rather than
  quietly corrected.

### Result — 300 rounds per field, ε = 0, validation seed 550731

| at `t_doom`, what killed us? | 3 × `binary_v6` | 3 × `rule_based` |
|---|---|---|
| **OWN_BOMB** | **78.6 %** | **79.1 %** |
| **OVERLAP** (own + enemy) | 14.9 % | 16.5 % |
| VISIBLE enemy bomb | 4.2 % | 3.3 % |
| **UNSEEABLE enemy bomb** | **2.3 %** | **0.5 %** |
| rounds died / 300 | 215 | 182 |

**The digit is refuted.** The entire case for an opponent-bomb-danger feature is the UNSEEABLE row,
and it is **2.3 %** of deaths — about **0.017 deaths per round**, two orders of magnitude below the
−1.0 margin deficit it was meant to address. **93.5 % of our deaths involve a bomb we placed
ourselves**, whose blast is fully described by `danger_map` and therefore already sits in digits
1–5. **These deaths are not unattributable. They are maximally attributable.**

**And the failure is a policy failure, not a representation gap.** Safe actions available at the
last survivable moment:

| safe actions at `t_doom` | 3 × `binary_v6` | 3 × `rule_based` |
|---|---|---|
| exactly 1 (a needle) | 23.3 % | 9.3 % |
| 2 | 55.8 % | 59.9 % |
| 3+ | 20.9 % | 30.8 % |

**In 76.7 % of deaths the agent had two or more surviving actions and chose a losing one**, in
states its own features already describe.

**The composition is field-independent; only the frequency changes.** 78.6/14.9/4.2/2.3 against a
strong DQN versus 79.1/16.5/3.3/0.5 against `rule_based` — nearly identical. What changes is how
often we die (215 vs 182 of 300) and how tight the escape is: **being down to a single surviving
action more than doubles, 9.3 % → 23.3 %**. **Strong opponents do not kill us. They compress our
space until our own bombs do.**

This corroborates `TASK_A_survey_vs_ours.md` §1.2 from the other side — *"our suicides are not
caused by bombing when trapped, they are caused by not walking the escape that exists"* — and it is
uncomfortable, because **E33 (pay for the escape step) and E34 (force it) both already tried to fix
exactly this and both failed.** This is a known-hard problem, not a fresh lead.

### Limitations

- Two fields at 300 rounds; the external field is one agent.
- `survivable_actions` assumes **no new bombs are placed**, which is generous to the agent and
  would over-count "a safe action existed". Since 93.5 % of killers are our own bombs, that
  generosity barely bites — but the 76.7 % figure is an upper bound.
- `t_doom` is the last step at which escape was possible *given bombs then on the board*. An
  opponent who would have bombed our escape route regardless is scored as OWN_BOMB. So the split
  understates opponent influence on *causation* while correctly measuring what a **feature over the
  current `game_state`** could have seen — which is the question asked.

---

## E42 — CORRECTION, added 2026-08-19 after audit 12 and E43

**Three of this entry's conclusions were wrong. The verdict (no gain from the mixed field) stands;
the explanation, the headline statistic and the forward recommendation do not.** Audit 12's report
is `scratchpad/audit12/REPORT.md`; I verified its central claim myself before writing this.

**1. The credit-assignment mechanism is REFUTED.** I wrote that the arm "learns that dying is not
its fault and stops paying to avoid it". Splitting every round at our agent's own death step
(`steps`, not `round_steps` — the round-length version of this split shows nothing, since 97–100 %
of rounds run past 200):

| field | suicides, died by step 200 | suicides, alive past 200 | P(reach step 200) |
|---|---|---|---|
| guard | +0.015 [−0.002, +0.034] | **+0.279 [+0.196, +0.358]** | −0.004 |
| in-dist | +0.010 [−0.034, +0.047] | **+0.065 [+0.027, +0.100]** | **+0.025** |
| held-out | +0.013 [−0.001, +0.029] | **+0.047 [+0.003, +0.093]** | **+0.027** |

**The entire +0.183 is post-step-200**, after the economy closes (`TASK_B_argument.md` §1), where
`environment.py:249` awards nothing for a suicide — which is exactly why score moved only −0.049
(ns). Early suicides are unchanged on all three fields, and the mixed arm **reaches** step 200 more
often on two of them with CIs excluding zero. An agent that stopped valuing its life would die more
when death is expensive; this one dies more only when death is free. Audit 12 adds two
corroborations from the tables: danger-row `min Q` is *not* less negative (−1.996 → −2.015; what
fell is `max Q`, 7.13 → 6.94), and across the 2 888 rows all 16 tables updated there are **zero**
systematic argmax flips (between-arm agreement 0.986 vs within-arm 0.987).

**2. The declared headline is fragile and therefore not demonstrated.** In-distribution score
−0.161 has a bootstrap CI of [−0.305, −0.014] but a **permutation p of 0.064**. `AGENTS.md` has said
since c40a902 that a fragile row is not demonstrated regardless of its CI — and
`scratchpad/benedict/e42_analyze.py`, which I wrote after adding that rule, never checked it. The
rule is now applied there. **"The control plays the mixed field better than the arm trained on it"
is not a demonstrated claim**; the point estimate stands, the significance does not.

**3. P3's "the loss arrived through bomb siting" is a denominator artefact.** Held-out crates went
**up** (+0.632 [+0.239, +0.974]) on more bombs (+2.000) while alive longer. crates/bomb fell because
the denominator grew. And crates are not a scoring channel at all: `score = coins + 5·kills` closes
to within 0.003 on every field (held-out: −0.013 + 5 × −0.020 = −0.113 against a measured −0.115).

**4. The entry was a fine-tune, not a retrain — so its title claim was never tested.**
`train.py:242` defaults `BM_WARM="_parent"` and `WARM_N = 100`, so with `ALPHA_EXP = 0.7` the first
update runs at **α = 1/100^0.7 = 0.0398** on a parent trained entirely against `rule_based`. "Train
against a field that hunts back" was never run. E44 runs it, from scratch, both arms.

**5. What survives.** Audit 12 attacked the control-arm reuse — the shortcut I flagged as the soft
target — and it **holds**: the warm parent is byte-identical, ε decays per episode, both arms ran
20 000, and three independent measurements of the recipe give guard-field suicides 0.515 / 0.490 /
0.478 (the last being E38's ep20000 checkpoints, a contemporaneous control this entry claimed not to
have) against the mixed arm's 0.687, outside the 0.372–0.664 range of all 60 E37 same-field tables.
**The effect is real. Only my explanation of it was wrong.**

**6. The recommendation this entry produced is withdrawn.** "The digit has to come first" rested
entirely on the refuted mechanism, and E43 independently measures the digit's ceiling at **2.3 % of
deaths**. Two separate lines now close it: the deaths are our own bombs, and `killed_by_opponent` —
precisely what the digit targets — *fell* in all three fields (−0.013 [−0.024, −0.004] pre-200,
held out). **No opponent-danger digit will be built.**

---

## E42 — training against a field that hunts back

- **Question:** E41 established that the shipped table loses to three of four third-party agents,
  and that the mechanism is **survival, not bomb siting**: crates/bomb *rises* (1.18 → 1.80) while
  bombs fall (28.3 → 19.6) because survival collapses (0.440 → **0.223**) and deaths to opponents'
  bombs rise 2.5×. **The table has never trained against an opponent that hunts it.** Every one of
  its 20 000 episodes was played against three `rule_based_agent`s, whose 1.487 suicides/round mean
  they mostly kill themselves. That is not a feature-map problem and no digit on the
  `NEXT_STEPS.md` list addresses it — it is a *training distribution* problem, and it has never
  been tested.

- **Change:** the training opponents, and nothing else. Same feature map, same reward table, same
  hyperparameters, same 20 000-episode budget (E38 established that as the optimum and that longer
  is worse), same warm-start parent. Only `--agents` differs.

  | arm | training field |
  |---|---|
  | **control** | 3 × `rule_based_agent` — the current recipe. **Free: the E37 `PLB2` sweep tables at ep20000 already are this arm**, same configuration, which became the default |
  | **treatment** | `ext_xiaoxiae_binary_v6` + `ext_aielka_ql_atom` + `rule_based_agent` — deliberately heterogeneous: a strong DQN, a tabular agent, and the incumbent |

  Measured cost, not estimated (`scratchpad/benedict/e42_timing.out`): 200 episodes take 10 s against
  3 × `rule_based` and 22–25 s against a strong external field, so a 20 000-episode run goes from
  ~0.3 h to ~0.7 h. Episodes lengthen as training proceeds (E38 learned this the expensive way, its
  8–10 h estimate becoming 19 h), so budget ~2 h per run.

- **Design.** **8 seeds per arm.** Treatment seeds are trained (`BM_RUN_INDEX` 200–207); control
  seeds are the existing `q_table_e37_PLB2_s100..s107__ep20000.npy`. Evaluation at ε = 0 through
  `evaluate.py`, **300 rounds, validation seed 550731**, on three fields:

  | field | role |
  |---|---|
  | **3 × `ext_xiaoxiae_bindist_v2`** | **HELD OUT — the primary.** Never seen in training |
  | the mixed training field | in-distribution, to size the generalisation gap |
  | 3 × `rule_based_agent` | **regression guard** — did we destroy what already works |

  Run-level pairing between arms is *not* assumed: `benedict_task4.md` §5.3 measured
  corr(arm, control) at matched seed between −0.47 and +0.44, because `main.py` does not seed the
  provided opponents. Pairing is on **evaluation arenas only**.

- **Power, stated in advance.** Between-seed score SD is 0.249, so n = 8 gives an 80 %-power MDE of
  **0.35** on score. **That is coarse, and it is deliberate**: the deficit this entry attacks is
  **−1.008** on margin against `binary_v6` and −0.797 against `bindist_v2`, three times the MDE.
  This design can see a third of the gap closing and cannot see a tenth of it. An effect too small
  for n = 8 is an effect too small to matter at this point in the project.

  Every comparison is scored on the bootstrap CI **and** the sign-flip p, and a row `analyze.py`
  marks `(fragile)` counts as not demonstrated regardless (`AGENTS.md`, added after E39/E40).

### Prediction (written and committed before training starts)

1. **P1, primary — generalisation.** On the **held-out** `bindist_v2` field, the treatment arm's
   `margin_mean` beats the control arm's by **≥ +0.35** (the design's own MDE), CI excluding 0 and
   sign-flip p < 0.05. **My prediction: it clears.** The control sits at −0.797 there; I expect the
   treatment to roughly halve that, not to close it. **Refutation:** no gain, or a gain only
   in-distribution → the deficit is not a training-distribution artefact and the next move is the
   feature map after all.
2. **P2, the regression guard — and it is the one I am least sure of.** On 3 × `rule_based`, the
   treatment arm loses **no more than 0.35 score** against the control. The shipped table scores
   3.949 there and that is the only number this project has ever optimised. **Refutation:** a larger
   loss means the mixed field trades the incumbent away, and any ship decision becomes a bet on
   which field the tournament actually resembles — which we do not know.
3. **P3, mechanism.** The gain, if any, shows up in **survival and `killed_by`**, not in
   crates/bomb. Specifically: `survived` rises on the held-out field, and crates/bomb moves by less
   than 0.15. **Refutation:** the gain arrives through crates/bomb → E41's diagnosis was wrong and
   the story is about the economy, not about dying.
4. **P4, the generalisation gap — pre-registered so it cannot be discovered later.** The
   in-distribution gain (mixed field) exceeds the held-out gain (`bindist_v2`). If in-distribution
   improves and held-out does not, the arm has learned three specific opponents rather than "how to
   survive aggression", which is the same overfitting failure one level up.
5. **Guards.** `think_max_ms` < 500 for every agent; `suicides` reported per field and never
   differenced across fields (audit 10 F5).

**Ship rule, pre-committed.** The shipped table changes **only if** a treatment seed beats the
current ship on the held-out field with a CI excluding 0 at 300 rounds, **and** survives P2's
regression guard, **and** is then confirmed at **1000 rounds on the held-out ship seed 990731**
against all four E41 fields including `feature_is_everything`. Selection is on validation seed
550731 only; the ship seed is never used to choose.

**Limitation, stated before the run rather than after.** The held-out agent `bindist_v2` shares an
author and a code lineage with the training opponent `binary_v6`. A genuinely independent held-out
agent would be `feature_is_everything`, but it runs at ~14 ms/step — 4.6 h per 1000-round
evaluation — which makes it unaffordable inside a sweep. **It is therefore used only in the final
1000-round confirmation**, where it is the strongest available test of generalisation.

### Result — 8 seeds/arm, 48 evaluations, 300 rounds, validation seed 550731

Trained 2026-08-18/19, ~8 h at 5 lanes (my 2 h estimate was 4x wrong, the same way E38's was — I
budgeted the per-run lengthening and not the wall clock). All 8 runs reached 20 000 episodes, no
tracebacks. **Collapse screen passes cleanly:** the last-1000-episode training band is 2.276-2.467
across the eight seeds, a spread of 0.19 with no collapsed run, so the n = 8 MDE of 0.35 is not
inflated by one bad seed the way E33's and E37's were.

Unit of inference is the **seed**, not the round, because that is what the pre-registered MDE was
computed on. Two-sample bootstrap CI and permutation p; tables in `scratchpad/benedict/e42/RESULTS.md`.

| field | metric | control | mixed | difference | p |
|---|---|---|---|---|---|
| **HELD OUT** 3 x `bindist_v2` | **margin_mean** | −1.000 | −1.043 | **−0.043 [−0.174, +0.082]** | 0.57 |
| | score | 2.965 | 2.850 | −0.115 [−0.243, +0.007] | 0.12 |
| | survived | 0.268 | 0.268 | −0.000 [−0.027, +0.025] | 0.97 |
| | crates/bomb | 2.214 | 2.053 | **−0.160 [−0.185, −0.137]** | 0.0000 |
| in-distribution (training field) | score | 3.342 | 3.181 | −0.161 [−0.305, −0.014] | 0.064 |
| | margin_mean | −0.045 | −0.164 | −0.119 [−0.292, +0.058] | 0.24 |
| **guard** 3 x `rule_based` | score | 3.832 | 3.782 | −0.049 [−0.182, +0.075] | 0.49 |
| | **suicides** | 0.503 | 0.687 | **+0.183 [+0.126, +0.236]** | 0.0002 |
| | **survived** | 0.434 | 0.265 | **−0.170 [−0.218, −0.117]** | 0.0001 |
| | killed_by_opponent | 0.062 | 0.049 | −0.014 [−0.024, −0.004] | 0.026 |

**P1 REFUTED.** The bar was a **+0.35** gain in held-out `margin_mean`. Measured: **−0.043**, with a
CI containing zero and a point estimate on the wrong side. Training against a field that hunts back
does not transfer to an unseen aggressive opponent. **The pre-committed consequence applies: the
deficit is not a training-distribution artefact, and E41's -1.0 gap is not closed this way.**

**P2 PASSES as written — and the guard was too narrow, which is the entry's most useful mistake.**
I registered "loses no more than 0.35 score" on `rule_based`; the loss is 0.049, comfortably inside.
But **score held while the behaviour underneath it fell apart**: suicides **+0.183** and survival
**−0.170**, both with p < 0.0003. A score-only regression guard passed an arm that kills itself 36 %
more often. This is exactly `benedict_task4.md` §5.6's lesson — *the behavioural metrics are the
discriminator, not the headline* — and I wrote the guard on the headline anyway.

**P3 REFUTED on both legs.** Predicted `survived` would rise on the held-out field: it moved
−0.000. Predicted crates/bomb would move less than 0.15: it fell **0.160**, significantly. The gain
did not arrive through survival, and the loss *did* arrive through bomb siting — the mirror image
of the mechanism E41 diagnosed.

**P4 REFUTED, and this is the finding.** P4 asked whether the in-distribution gain would exceed the
held-out gain, expecting overfitting to the training opponents. **There is no in-distribution gain
to overfit with.** On the field it trained against for 20 000 episodes, the mixed arm scores
**3.181 against the control's 3.342** — *the control, which never saw that field, plays it better.*

### Verdict — training against stronger opponents made the agent worse, everywhere

Not "no effect": the mixed arm is worse on the held-out field, worse in-distribution, and
behaviourally much worse on the incumbent field. **Ship rule fires: no table changes.** No treatment
seed beats the current ship on the held-out field, so the 1000-round confirmation against the E41
fields is not run.

**The mechanism is not lack of experience.** Total training steps are 2.89 M (mixed) against 3.15 M
(control) — 8 % fewer, nowhere near enough to explain it. And the mixed arm suicides *less* during
training (0.74 vs 0.82 per episode) while suiciding *more* at ε = 0 (0.687 vs 0.503). The training
curve and the policy point in opposite directions, which is the sixth time on this project that a
training log has looked healthy over a worse agent.

**The explanation that fits the numbers is credit assignment.** Against `rule_based_agent`, most of
our deaths are our own doing — it suicides 1.487 of its 1.820 deaths per round (E41) and rarely
hunts. Against strong agents a large share of deaths are opponent-induced and, given our eight
digits, essentially unattributable: the state cannot represent "an opponent is about to bomb my
escape route", so those deaths arrive as noise on whatever action was taken. **The agent learns
that dying is not its fault, and stops paying to avoid it** — which is precisely what suicides
+0.183 and survival −0.170 describe. Training on a harder distribution did not teach a better
policy; it taught a *less attributable* one.

**That reframes the problem E41 opened, and it does not close it.** E41 showed the deficit is
survival under pressure. E42 shows the deficit **cannot be trained away while the feature map
cannot see the pressure.** The order of operations is the opposite of what I proposed: the digit has
to come first, and the mixed-field retrain is what tests it afterwards. `NEXT_STEPS.md` §3.4's
opponent-bomb-danger digit — the one audit 10 F5 re-attributed to **91 % opponent bombs, 9 %
opponent bodies**, and which the survey says the entire published corpus leaves unaddressed — is now
the only candidate on the list that addresses a measured, replicated, 1.0-point deficit.

### Limitations

- **The held-out agent shares an author with a training opponent** (`bindist_v2` / `binary_v6`,
  both `xiaoxiae/BombermanML`), as stated before the run. The genuinely independent test,
  `feature_is_everything`, was reserved for a confirmation that the ship rule never triggered.
- **One field, one mixed composition, one budget.** A different mixture (more `rule_based`, or a
  curriculum that starts easy) is untested, and the credit-assignment story above predicts a
  curriculum would do better. That is a hypothesis this entry does not test.
- The control arm is E37's `PLB2` checkpoints rather than freshly trained seeds. Same configuration
  and same commit-era defaults, but they were trained on a different day, and `main.py` does not
  seed the provided opponents, so arm-level differences carry that too.

---

## E41 — the first opponent that is not `rule_based_agent`

- **Question:** **all 412 committed rung-4 evaluation CSVs are against 3 x `rule_based_agent`**
  (`ls results/eval/task4_tournament/*.csv | grep -v _rb_` returns nothing). Every strategic
  conclusion this project holds is conditional on one hand-written opponent's behaviour:

  | conclusion | the `rule_based` behaviour it rests on |
  |---|---|
  | the economy closes at step ~200 | it clears crates fast — opponents take **89.3 of 122** |
  | traps decay to 4 % over a 5-step walk (E39/E40) | it **flees actively**; a trap's half-life is ~1 move |
  | the kill pool is not free | it suicides **1.487 of its 1.820 deaths/round** (audit 10 F4) |
  | `won` = 0.088 x score | the margin distribution against *the best of three* `rule_based` |

  Audit 10's F4 sharpened this into the reason it matters: our kill count is set mostly by **the
  field's own suicide rate**, not by our aim. `rule_based` consumes 82 % of its own mortality before
  we can reach it. Against a field that dies less to itself, the takeable pool — and therefore
  everything E39/E40 measured about hunting — could be materially different.

  `final_project.pdf` p. 2 explicitly sanctions the fix: *"You can share your trained agents
  (without training code) ... and **download other teams' agents to test your approach.**"*

- **Change:** none to the agent. The shipped table plays unmodified against **four third-party
  agents pulled from GitHub** (24 repos surveyed, `scratchpad/external/FINDINGS.md`), chosen as the
  ones that load against our framework with no fix:

  | slot name | origin | kind |
  |---|---|---|
  | `ext_xiaoxiae_bindist_v2` | `xiaoxiae/BombermanML` @ `50b682f`, GPL-3.0 | DQN |
  | `ext_xiaoxiae_binary_v6` | `xiaoxiae/BombermanML` @ `50b682f`, GPL-3.0 | DQN |
  | `ext_aielka_ql_atom` | `AI-ELka/BombermanRLAgents` @ `8d85731` | tabular, 329 states |
  | `ext_lijesse_featureeverything` | `Li-Jesse-Jiaze/MLE_project_bomberman` @ `a7fe504` | 3rd place SS2024 |

  **`ext_aielka_ql_atom` is source E of the survey** — lukevoss's "Atom", the agent whose claimed
  **5.04** against 3 x `rule_based` is the 1.09-point gap `TASK_A_survey_vs_ours.md` §2 could never
  price. lukevoss's own repo ships the code without the weights; this is a byte-identical copy of it
  *with* `q_table.pkl`. Running it in our harness gives that calibration constant directly.

- **Design.** 1000 rounds each at the held-out ship seed **990731**, `--out-dir
  results/eval/task4_tournament`. Three line-ups per external agent, and the second and third are
  the ones previous field work has always omitted:

  | line-up | what only this one answers |
  |---|---|
  | **A**: ours + 3 x external | our head-to-head standing against that team |
  | **B**: 4 x external | **the symmetric bar** — prices the field itself, as `benedict_task4.md` §1 does with 4 x `rule_based` (`won` 0.282). Without it a weak opponent and an easy arena are indistinguishable |
  | **C**: external + 3 x `rule_based` | **the calibration constant** — puts *their* agent in *our* slot against *our* reference field, the only apples-to-apples comparison with our 3.949 |

- **Power, stated in advance — and for once it is not the binding constraint.** At n = 1000 the
  standard error on `score` is ~0.085 (scaled from E40's realised 0.042 at n = 8000), so the
  80 %-power MDE is **~0.24**; on the margin it is **~0.40**. That is coarse by this project's
  standards. **It does not matter here, because this is the first rung-4 experiment whose expected
  effect is measured in points rather than hundredths of one:** the smoke tests put three externals
  1.6–2.0 points above our 3.949. Any result small enough to be unreadable at n = 1000 is a result
  that says the field is a wash, which is itself the answer to P1.

  The external agents carry their own unseeded RNG on top of the opponents', so the +/-0.12 noise
  floor is a **lower** bound here. No single-run difference under ~0.2 will be claimed.

### Prediction (written before the run)

**Primary is the paired within-round margin, not our absolute score.** Our absolute score swings
3.95 -> 14.49 purely on who else is on the board (`scratchpad/strategy/fields/`), and the crate pool
is exactly zero-sum, so *any* field that clears crates worse than `rule_based` hands us points for
free. "Our score went up" would be near-unfalsifiable. `margin_mean` = our score minus the mean
opponent score in the same round is what a total-score ranking actually sums.

1. **P1, primary — and I expect to lose.** `margin_mean` in line-up A, per external agent.
   **My prediction is that it is negative with a CI excluding 0 against at least two of the four.**
   The smoke tests (n = 15, far below the noise floor, so a shortlist not a result) put
   `binary_distance_agent_v2` at 6.00, `binary_agent_v6` at 5.60 and `ql`/Atom at 5.60 against the
   same 3 x `rule_based` field where we score 3.949. **Refutation:** we win or draw against three or
   four of them, and the agent is more robust than this entry assumes.
2. **P2, the calibration constant — the number the survey could never get.** In line-up C, each
   external's `score` in *our* slot against 3 x `rule_based`, compared to our **3.949**.
   **Prediction: at least two exceed it, and `ext_aielka_ql_atom` lands within +/-0.5 of the
   claimed 5.04.** **Refutation:** Atom scores near 3.9 -> the survey's 5.04 was selection, a
   different framework version, or a different measurement convention, and the 1.09-point gap this
   project has treated as real never existed. *That outcome would be worth more than winning.*
3. **P3, the overfitting guard — the most consequential possible negative.** Our **crates/bomb**
   against the external fields stays within **+/-0.15** of the **1.16** measured against
   `rule_based`. This is the quantity E37 bought (+0.130) and the one E38 showed the training
   horizon destroys. **Refutation:** it falls outside that band -> the bomb-siting policy E37
   installed is tuned to one opponent's movement, the shipped agent is overfitted in the way nothing
   in the ledger has ever tested, and the remaining weeks go to robustness rather than features.
4. **P4, the hunt reopener — pre-committed now so it cannot be chosen later.** Audit 10 F4:
   `rule_based` self-consumes 1.487 of 1.820 deaths/round, leaving a takeable pool of 0.333.
   **Prediction: the externals suicide less, so the takeable pool is larger.** If any field's
   takeable pool exceeds **0.5/round**, rerun `hunt_ceiling_v2.py --k 4 --trap-model sim` against
   that field. **Hunting reopens only if that oracle clears +0.25** — E40's bar, unchanged, and it
   is the one condition under which E40's NO-GO does not bind.
5. **Guards, and one is an exclusion rather than a flag.** `think_over_limit` must be **< 1 % of
   steps for every agent in the line-up**. An agent over the 0.5 s limit gets `WAIT` with the
   overrun billed to its next step, so it plays *crippled* and our margin against it is inflated.
   **Any agent breaching 1 % is reported as a compatibility result only and no strength claim is
   drawn from it.** Also: `suicides` is not comparable across fields (audit 10 F5 — 16 % of deaths
   against `rule_based` are double-credited when own and enemy blasts overlap, 0 % against
   `peaceful`), so it is reported per field and never differenced across them.

**Ship rule, pre-committed.** Nothing ships from this entry either; it is a measurement of external
validity. What it decides: **P3 failing sends the next sweep to a mixed-field retrain rather than to
any feature on the `NEXT_STEPS.md` list. P3 holding and P4 not firing leaves `target_type` (§3.1,
free variant only — the size-4 version lost its cost justification to audit 10 F6) as the next
feature.** P1 is reported whatever it says: **an external agent beating us is information, not
failure, and finding it out five weeks before the deadline is the entire point of running this.**

**Licence and hygiene.** 20 of the 24 repos carry no LICENSE, so all rights reserved: the clones
stay untracked (`.gitignore`: `scratchpad/external/*/`), nothing of theirs enters
`agent_code/benedict_task4/`, and nothing of theirs reaches the submission zip. Their code is cited
by URL and commit hash, never vendored. Using an agent as an *opponent* is measurement, not copying
— it is exactly what `rule_based_agent` is for.

### Result — 11 of 12 runs, n = 1000, held-out ship seed 990731

Ran 2026-08-18, four lanes, ~5 h. Scoring `scratchpad/benedict/e41_analyze.py`, tables
`scratchpad/benedict/e41/RESULTS.md`, install record `scratchpad/external/install/INSTALL.md`.
**One run was dropped:** the symmetric bar for `ext_lijesse_featureeverything` (4 x a 14 ms/step
agent, ~3.8 h remaining) was killed for cost after the other three agreed to within 0.34. Recorded
here rather than omitted; it is the only planned measurement missing.

**P1 CONFIRMED — we lose to three of the four.** Paired within-round margin, all p < 0.0001:

| field | our score | their mean | **margin_mean** | margin_best |
|---|---|---|---|---|
| 3 x `binary_v6` | 2.611 | 3.619 | **−1.008 [−1.197, −0.813]** | −4.263 |
| 3 x `bindist_v2` | 2.986 | 3.783 | **−0.797 [−1.012, −0.570]** | −4.308 |
| 3 x `feature_is_everything` | 2.226 | 2.770 | **−0.544 [−0.684, −0.399]** | −2.284 |
| 3 x `ql`/Atom | 4.283 | 3.371 | **+0.912 [+0.661, +1.165]** | −1.872 |

I pre-registered "negative with a CI excluding 0 against at least two of the four." It is negative
against **three**. This is the largest effect ever measured on this rung and it points the wrong way.

**P2 CONFIRMED on both legs — and the survey's 5.04 was real.** Their agent in *our* slot against
*our* reference field, where our shipped table scores **3.949**:

| agent | score | vs ours | coins | kills | suicides | crates |
|---|---|---|---|---|---|---|
| `binary_v6` | **5.572** | +1.623 | 3.372 | 0.440 | 0.377 | 25.09 |
| `bindist_v2` | **5.336** | +1.387 | 3.216 | 0.424 | 0.276 | 25.10 |
| `feature_is_everything` | **5.143** | +1.194 | 3.728 | 0.283 | 0.108 | 39.72 |
| `ql`/Atom | **4.690** | +0.741 | 2.770 | 0.384 | 0.127 | 30.40 |

**All four beat us**, and Atom lands at 4.690 against its claimed 5.04 — inside the +/-0.5 bar.
**So the 1.09-point gap `TASK_A_survey_vs_ours.md` §2 could never price is genuine**, it is not a
framework or measurement artefact, and three other agents are further ahead than Atom was.

**P3 REFUTED as written — and the guard was the wrong instrument.** crates/bomb was 1.797 / 1.960 /
1.429 / 1.249 against a 1.16 +/- 0.15 band: out of band in three of four. But it moved **up**, the
opposite direction to the overfitting hypothesis it was built to detect, and the cause is
composition: crate availability depends on how fast the *opponents* clear crates. They take 25.1 /
25.1 / 30.4 / 39.7 in the calibration runs, and our crates/bomb falls as theirs rises. A guard that
assumes crates/bomb is opponent-independent cannot test opponent-specific overfitting.
**This is the sixth pre-registration on this rung whose bar did not match its design** (E33-E36 on
`won`, E40's arithmetically unreachable +0.25, now this). Scored as written; the inference does not
follow, and the honest reading is that **E37's bomb siting is not overfitted to `rule_based`** — it
is the one thing that holds up.

**P4 — the suicide leg CONFIRMED, the pool leg fires once.** Every external suicides less than
`rule_based`'s 1.487/round: 1.463 / 1.220 / 1.049 / 0.239. But a smaller suicide rate does not imply
a larger takeable pool, because the strong agents also *die less overall*:

| field | opp deaths | opp suicides | **takeable** | our kills | our share |
|---|---|---|---|---|---|
| 3 x `rule_based` | 1.822 | 1.514 | 0.308 | 0.214 | 69.5 % |
| 3 x `binary_v6` | 1.819 | 1.463 | 0.356 | 0.091 | 25.6 % |
| 3 x `bindist_v2` | 1.675 | 1.220 | 0.455 | 0.173 | 38.0 % |
| **3 x `ql`/Atom** | 1.717 | 1.049 | **0.668** | 0.301 | 45.1 % |
| 3 x `feature_is_everything` | **0.400** | 0.239 | 0.161 | 0.092 | 57.1 % |

**The reopen bar fires on the Atom field only** (0.668 > 0.5), so the pre-committed rerun of the
oracle against that field is triggered — with `--trap-model stale` per the amendment recorded in
E40 before this ran. Note `feature_is_everything`: its agents die **0.400** times a round against
`rule_based`'s 1.822. Against a field like that there is essentially no kill pool at all, and every
kill-side conclusion this project holds evaporates.

**Guards pass.** `think_over_limit` is **0.00 % of steps for every agent in every line-up**, even
under four-lane CPU contention — a pass a fortiori, since contention inflates think time (max
170 ms for xiaoxiae under load vs 36.9 ms clean; ours 10.2 ms). Atom's invalid actions are reported
as a **median of 3** rather than a mean: its 20-bit encoding aliases "no recommendation anywhere"
onto an untied `LEFT`, and since an invalid action does not change the state it wedges against a
wall — longest observed run 227 consecutive invalid steps (`INSTALL.md`). Its `steps` and `survived`
are inflated for a reason unrelated to skill.

**The symmetric bar says the fields are not richer — the agents are better.** Four copies of one
external, no us: mean score **3.377 / 3.428 / 3.089**, against four `rule_based` agents' 3.254.
Best-of-four runs **7.744 / 7.678 / 6.550** against `rule_based`'s 5.32. Same pool, concentrated
into stronger hands.

### Verdict — the agent is competitive with `rule_based_agent` and not with this cohort

Three of four beat us head-to-head; four of four beat us in our own slot. **Our score falls from
3.949 to 2.2-3.0 against a strong field**, and the mechanism is not the one eight rung-4 entries
have been chasing:

| field | score | coins | kills | suicides | killed_by | crates/bomb | bombs | **survived** |
|---|---|---|---|---|---|---|---|---|
| 3 x `rule_based` | 3.828 | 2.758 | 0.214 | 0.505 | 0.055 | 1.18 | 28.3 | **0.440** |
| 3 x `binary_v6` | 2.611 | 2.156 | 0.091 | 0.552 | **0.138** | 1.80 | 21.9 | **0.310** |
| 3 x `bindist_v2` | 2.986 | 2.121 | 0.173 | 0.578 | **0.149** | 1.96 | 19.6 | **0.273** |
| 3 x `feature_is_everything` | 2.226 | 1.766 | 0.092 | 0.704 | 0.073 | 1.25 | 22.5 | **0.223** |

**Bomb siting is fine — crates per bomb goes up.** What collapses is that we *get fewer bombs off*
(28.3 -> 19.6-22.5) because **we are dead**: survival 0.440 -> 0.223-0.310, own-bomb suicides up,
and deaths to opponents' bombs up **2.5-2.7x** (0.055 -> 0.138/0.149, and `evaluate.py:258`
undercounts that by ~2x per audit 10 F5, so read it as ~0.3).

**This reverses the rung's central finding, and the reversal is conditional, not a contradiction.**
`NEXT_STEPS.md` §4 says "buy survival" is closed by six replications, and
`TASK_B_argument.md` §1 gives the mechanism: the economy closes by step ~200 and every survival
intervention bought *phase-2* survival, which is worth nothing. **That holds against
`rule_based_agent`, where we survive to step 200 anyway.** Against these agents we survive to the
end of the round in **22-31 %** of rounds and we are dying *inside* the economy. **Survival was
never worthless; it was non-binding against the only opponent we ever measured.** Six replications
of a null measured one field.

### What this changes

1. **The do-not-do list was collected against one opponent and at least one entry on it is now
   wrong.** Every item in `NEXT_STEPS.md` §4 was justified by an effect of +/-0.1 measured against
   `rule_based`. We are **1.0 points** behind a student DQN — eight times the largest effect this
   project has ever spent a sweep on.
2. **The next work is survival-under-pressure, not the feature map.** Specifically deaths to
   opponents' bombs, which audit 10 F5 already re-attributed to **91 % opponent bombs, 9 % opponent
   bodies**, and which is the one thing the survey says the entire published corpus leaves
   unaddressed. It is now measured as 2.5-2.7x worse against real agents than against `rule_based`.
3. **Training against `rule_based_agent` alone is itself the confound.** The shipped table has never
   seen an opponent that hunts it. A mixed-field retrain is the obvious arm and has never been run.
4. **`won` and `rank` are now meaningless for us** in these fields (margin_best −1.9 to −4.3): we
   are not competing for the round, we are competing for second place.

### Limitations

- **The mixed line-up was not run.** Every field here is three copies of *one* agent, which is
  harsher than a tournament round containing one of each. **These margins are a lower bound on our
  standing**, and the mixed field is the measurement that should come next.
- **Four agents, all SS2024.** The only two SS2026 forks found are bare framework with no trained
  agent, so **this is not our cohort** — it is the previous one, sampled by whoever published to
  GitHub, which selects for people who were pleased with their result.
- The symmetric bar for `feature_is_everything` is missing (killed for cost).
- `suicides` is not comparable across fields (audit 10 F5: 16 % of deaths against `rule_based` are
  double-credited when own and enemy blasts overlap). Reported per field, never differenced.

---

## E40 — the hunt ceiling remeasured with a trap test that matches the game's move rule

- **Question:** E39 returned an oracle ceiling of +0.116 on score and concluded the hunting
  direction was closed. **Audit 10 broke the instrument.** `hunt_ceiling.trap_sites` asks whether
  the target can escape **from the tile it currently occupies**, but `environment.py:421-432` polls
  every agent on the same pre-action snapshot and only then executes the actions in a random
  permutation — so by the instant our BOMB exists, the target has already taken a move the trap test
  never modelled. Followed to the fuse, those "verified inescapable" bombs earn us the kill
  **8.1 % [3.3, 16.1]** of the time (`scratchpad/audit10/l_trap_followup.out`), and
  0.406 bombs/round × 8.1 % × 5 ≈ +0.16 accounts for the whole measured effect.

  **So +0.116 is a floor on hunting, not the ceiling E39 called it.** The direction is not closed;
  it is unmeasured. This entry measures it with a trap test whose world model matches the
  environment's.

- **Change:** one function. `scratchpad/benedict/hunt_ceiling_v2.py` is `hunt_ceiling.py` with
  `trap_sites` requiring the target to be escape-less **from every cell it could occupy once the
  step resolves** — its current tile plus every free neighbour — instead of from its current tile
  alone. `--trap-model stale` reproduces the E39 behaviour exactly from the same binary, so the
  stale/sim contrast is not confounded by anything else. Other agents are *not* removed from the
  target's options: granting it more freedom makes the test stricter, which is the safe direction
  for an upper bound. The new bomb's own tile is excluded as a destination — walking onto the bomb
  is not an escape.

  Nothing under `agent_code/` is touched; the line-up still drives the provided `user_agent`.

- **Design.** Four arms, **n = 8000** paired arenas each, held-out ship seed 990731, 3 ×
  `rule_based_agent`, run concurrently:

  | arm | what it is for |
  |---|---|
  | `k = -1` | control — the shipped agent, re-run under v2 so every arm comes from one binary |
  | `k = 4, stale` | reproduces E39's headline arm; the bridge between the two entries |
  | **`k = 4, sim`** | **the corrected oracle — the primary arm** |
  | `k = 8, sim` | a stricter trap test finds rarer sites; does it want a longer reach? |

  n was doubled from E39's 4000 precisely because that entry died at the significance boundary.
  The harness now records the git commit in its `.meta.json`, which E39 flagged as missing.

- **Power, stated in advance.** From E39's realised paired SDs, at n = 8000 the standard errors are
  **score 0.042, margin_mean 0.051, margin_best 0.065**, so the 80 %-power MDEs are
  **score 0.119, margin_mean 0.142, margin_best 0.181**. E39's +0.116 sits *just* below the score
  MDE — this design can separate it from zero only if it grows.

  **And the CI is not read off `analyze.py` alone.** Audit 10 showed `analyze.py:220`'s hard-coded
  `default_rng(12345)` makes a borderline bound look deterministic when it is a coin flip; every
  verdict below is scored on the **sign-flip permutation p** as well, and a claim counts only if
  both agree.

### Prediction (written before the run)

**Both score and margin are pre-registered, because the criterion is genuinely unresolved.** The
tournament-format question is out with the course and unanswered; `final_project.pdf` §3 says the
winner is decided "by total score", which is a *ranking of four totals*, and audit 10 showed the
same data gives +0.116 on our own score and +0.270 on the margin. Registering both now is the only
way to avoid E39's problem, where the interesting metric was chosen after the fact and had to be
recorded as post-hoc. **No metric here gets swapped after the numbers land.**

1. **P1, primary — score.** `k = 4 sim` beats the control by **≥ +0.25**, the same bar E39
   pre-registered and failed. **Refutation:** below +0.25, and the pre-committed consequence is that
   hunting stays closed *on score* — this time on an instrument that is actually an upper bound,
   which is the conclusion E39 was not entitled to.
2. **P2 — the reason this entry exists: is the corrected oracle worth more than the broken one?**
   `k = 4 sim` − `k = 4 stale` on score. **My prediction is that it is NOT significantly different
   (CI contains 0),** because the fix trades a ~2× lower firing rate for a higher conversion rate
   and I expect those to roughly cancel. **Refutation, and it would be the interesting outcome:**
   sim beats stale with a CI excluding 0 → the audit's "floor, not ceiling" reading is confirmed
   quantitatively and the ceiling is genuinely higher than anything measured so far.
3. **P3, mechanism — conversion per bomb.** The whole premise is that sim's bombs are *better*
   bombs. Sim's override bombs must convert to a credited kill at a **materially higher rate** than
   stale's 8.1 %, and I pre-register **≥ 20 %** as "materially". Measured the same way audit 10 did,
   by following each override bomb to its fuse. **Refutation:** sim converts no better → the trap
   test is not what was wrong, and both oracles are measuring something other than trapping.
4. **P4, pre-registered secondary — margin.** `margin_best` = our score − the best opponent's score,
   per round. Bar **≥ +0.25**, matching `NEXT_STEPS.md` §3.4's ceiling-test threshold. Reported
   whatever P1 does, and *not* substituted for P1.
5. **P5 — reach.** `k = 8 sim` ≥ `k = 4 sim` on score. E39's k = 8 was worse under the stale test,
   but rarer real traps plausibly justify walking further. **This is exploratory** and no decision
   hangs on it; E39's "k = 8 bounds the chase-harder direction" was itself an over-read at t = 1.09.
6. **Guards.** `suicides` must not rise above the control's ~0.48; `crates` must not fall more than
   the −0.42 E39 measured at k = 4; `think_max` irrelevant (offline harness) but the control arm
   must reproduce the shipped agent's score to within the ±0.12 noise floor.

**Ship rule, pre-committed.** Nothing ships from this entry — it is a ceiling, not an agent. What it
decides is *whether a hunting digit gets designed at all*: **design one only if P1 or P4 clears its
+0.25 bar with both a CI excluding 0 and a sign-flip p < 0.05.** If neither clears, hunting is
closed on a correct instrument and `NEXT_STEPS.md` §3.4's remaining candidate — the opponent-bomb
suicide arm, which audit 10 re-attributed to 91 % opponent *bombs* rather than 9 % opponent bodies —
becomes the next ceiling test.

### Result — 4 arms x 8000 paired arenas, eps = 0, held-out seed 990731

Sweep ran 2026-08-18, four arms concurrently, ~70 min. Raw tables in
`scratchpad/benedict/e40/RESULTS.md`, scoring code `scratchpad/benedict/e40_analyze.py`.
**Control validation:** score 4.009 / kills 0.236 / suicides 0.477 / crates 33.51 against the
shipped agent's published 3.949 / 0.226 / 0.488 / 33.55 — inside the +/-0.12 noise floor, guard passes.

Every row carries a bootstrap CI **and** a sign-flip p, and counts only if both agree:

| comparison | score | margin_mean | margin_best | kills | crates |
|---|---|---|---|---|---|
| **k4 sim − control** | +0.053 [−0.028, +0.132] p=.22 | +0.097 [−0.003, +0.193] p=.065 | **+0.163 [+0.032, +0.290] p=.014** | **+0.019 p=.015** | −0.060 p=.62 |
| **k4 sim − k4 stale** | −0.022 [−0.104, +0.063] p=.60 | −0.027 p=.62 | −0.022 p=.75 | −0.004 p=.64 | +0.236 p=.050 |
| k4 stale − control | +0.075 [−0.009, +0.157] p=.074 | **+0.123 p=.017** | **+0.185 [+0.055, +0.314] p=.005** | **+0.022 p=.002** | **−0.296 p=.014** |
| k8 sim − k4 sim | −0.076 p=.082 | −0.095 p=.070 | −0.122 p=.063 | −0.003 p=.71 | **−0.907 p<.0001** |

**P1 REFUTED, and not marginally.** The bar was +0.25 on score; the corrected oracle delivers
**+0.053 [−0.028, +0.132]**, not distinguishable from zero at an n whose MDE is 0.119. The
pre-committed consequence applies: **no hunting digit gets designed.**

**P2 PASSES as predicted, and it is the most interesting line in the entry.** I predicted the
corrected oracle would *not* significantly beat the broken one, because the fix trades a ~2× lower
firing rate for a higher conversion rate. Measured: **−0.022 [−0.104, +0.063]** on score, and
nothing on any other metric. The firing rate fell 1.57 % → 1.14 % of steps and the bombs fell
3364 → 1826, exactly as designed — **and the outcome did not move.**

So: audit 10 was right that the instrument was mis-specified, and **fixing it changed no
conclusion.** The stale test was wrong in a way that happened not to matter, because trading
accuracy against opportunity was almost exactly break-even. That is worth stating in the report as
a methodological result in its own right: *an instrument can be provably wrong and still return the
right answer, and you only learn which by fixing it.*

**P3 FAILED** (measured before the sweep, `scratchpad/benedict/e40_conversion.py`, 1000 rounds
each). I pre-registered ≥ 20 % credited-kill conversion for the corrected oracle:

| | override bombs/round | target dies | **we get the credit** | credited kills/round |
|---|---|---|---|---|
| stale | 0.429 | 22.1 % | **8.4 % [6.1, 11.4]** | 0.036 |
| sim | 0.200 | 31.0 % | **12.0 % [8.2, 17.2]** | **0.024** |

Two things. **Audit 10's F2 is independently reproduced** — their 8.1 % [3.3, 16.1] against my
8.4 % [6.1, 11.4], from a separate harness; that was the one claim of theirs I could not verify at
the time. And the corrected oracle still converts at only 12 %, so **there is a third
mis-specification underneath the one this entry fixed.** Even when the target is provably
escape-less at the instant the bomb lands, it survives 69 % of the time — because `trap_sites`
evaluates a **static** board while other agents' blasts destroy crates and open escape routes across
the four-step fuse. Move-order was one error; treating the board as frozen over the fuse is a larger
one, and it is not fixed here.

**P4 FAILED as written, and the pre-registration is what makes that readable.** The bar was
`margin_best` ≥ +0.25; measured **+0.163 [+0.032, +0.290]**. The effect is *real* — CI excludes 0,
p = 0.014, and unlike E39's score number it is stable across bootstrap seeds — but it does not clear
the bar. **Audit 10's post-hoc +0.270 does not replicate: on 8000 clean arenas the same stale arm
gives +0.185.** The ~0.09 shrinkage is the winner's curse the audit itself estimated at ~0.05, plus
noise. **Had margin not been pre-registered here, +0.270 would have looked like it cleared +0.25
and a digit would have been designed on it.** That is exactly the failure mode §5.1 of
`benedict_task4.md` catalogues four times, caught prospectively for once.

**P5 — no.** k = 8 is worse than k = 4 on every headline (score −0.076, margin_best −0.122, none
significant) and **decisively worse on crates: −0.907 [−1.150, −0.663], t = −7.28.** E39 called the
k = 8 crate cost a bound on "chase harder" and audit 10 correctly flagged that as an over-read at
t = 1.09; at n = 8000 the crate cost is real and large even though the score cost is not. **Hunting
does take bombs away from the economy — that part of E39 survives, now properly powered.**

**Guards pass.** Suicides −0.011 (never rises in any arm), crates −0.060 at k = 4 sim, control
validated above.

### Verdict — hunting is closed, this time on an instrument entitled to close it

**Ship rule fires NO-GO.** It required P1 or P4 to clear +0.25 with both a CI excluding 0 and a
sign-flip p < 0.05. P1 gives +0.053 (p = 0.22); P4 gives +0.163 (p = 0.014) — significant but under
the bar. No hunting digit is designed.

The difference from E39 is what this entry is for. E39 concluded the same thing from a broken
instrument and a number (+0.116 [+0.002, +0.233]) that audit 10 showed was a bootstrap-seed
artefact. **That number does not replicate:** the same stale arm at n = 8000 gives
**+0.075 [−0.009, +0.157], p = 0.074**. The conclusion held; the evidence for it did not, and it has
now been replaced.

**What is left standing, and it is not nothing.** Positioning to trap opponents is worth a real
+0.163 on the between-agent margin and +0.019 kills, reproducibly, from an oracle. It is simply
smaller than +0.25 — the threshold below which `NEXT_STEPS.md` §3.4 says a ceiling is not worth
pursuing, and far below the MDE of the 15-seed sweep that would validate a learned version. **The
honest report sentence is "measured, real, and too small to build", not "there are no kills to
take".**

**What could still overturn it:** the fuse-window mis-specification in P3. A trap test that
projected the board forward four steps — other agents' bombs included — would find rarer but truer
traps. The evidence that this would not help is indirect but consistent: k = 8, which searches
harder in the same static way, is worse; and the correction that *did* get made moved nothing.

### CORRECTION, added 2026-08-18 after audit 11 — the premise of this entry was wrong

**Audit 11 overturned the reason E40 exists.** Full report `scratchpad/audit11/REPORT.md`; I verified
the load-bearing claims myself before writing this.

**`escapable()` already modelled the simultaneous move.** It is a time-aware BFS whose depth-0 node
is the target's current tile and whose **depth-1 nodes are exactly the tiles it can move to during
the step our bomb lands**. Audit 10's F2 — "the trap is verified against a position the opponent is
leaving" — is a misreading of `hunt_ceiling.py:80-106`. A correctly-timed test returns the
**identical site set** to E39's original: audit 11 measured 0 disagreements over 20 022 steps.

**So `k4sim` is not "the corrected oracle". It is E39's test minus 31 % of its sites**, and the
deletions come from two bugs I introduced in `target_cells`:

1. `escapable(neighbour, ...)` restarts the depth counter at 0, so the target gets a free move
   *plus* a full fresh escape budget — a phantom extra move it does not have.
2. `escapable` only checks its entry tile against `>= SAFE`, never against `danger <= depth`, so
   **stepping into a live explosion and back out counts as an escape.**

Both make the test over-strict, so `sim` ⊂ `stale` strictly. My own sweep metadata says the same
thing and I did not read it that way at the time: override bombs 3364 → 1826, a 46 % cut.

**Consequences, in order:**

- **The verdict does not change.** The correct instrument is the **`k4stale`** arm, and on it
  hunting still fails: score **+0.075 [−0.009, +0.157]** (ns), `margin_best` **+0.185** against a
  +0.25 bar. The ship rule still fires NO-GO. What changes is which arm carries it.
- **P2's interpretation was backwards.** I wrote that "an instrument can be provably wrong and still
  return the right answer". The truth is the instrument was *right* and my fix was a no-op plus two
  bugs. `sim ≈ stale` not because accuracy and opportunity traded off at break-even, but because
  `sim` is `stale` with 31 % of its sites deleted and those sites were not converting anyway. **The
  sentence must not go into the report as written.**
- **The pre-registered bars were arithmetically unreachable.** `score = coins + 5·kills` exactly
  (audit 11: max residual 0.000000 over 32 000 rows). At the 0.200 override bombs/round I had
  *already measured* before the sweep, +0.25 score needs ≥ 25 % conversion — **above P3's own 20 %
  bar, which buys only +0.200.** Passing P3 guaranteed failing P1. And I carried +0.25 over from
  E39 "unchanged" as if that were a virtue: there it needed 11.7 % conversion, because there were
  0.429 bombs/round. **Halving the shots while holding the standard is not the same experiment.**
  This is the fifth entry on this rung to pre-register a target its own design could not reach.
- **The attribution is wrong, and this one matters for E41.** I read the margin gain as "positioning
  to trap opponents". Verified on the `k4stale` arm at n = 8000:

  | | paired difference | t |
  |---|---|---|
  | our kills | +0.0222 | +3.01 |
  | **opponents' suicides** | **+0.0456** | **+4.17** |
  | opponents' deaths | +0.0360 | +4.23 |
  | opponents' score | −0.1437 | −2.48 |

  **The best-powered effect is opponents killing *themselves* more — twice our kill gain.** The
  oracle's value is substantially that walking at `rule_based_agent` makes it panic into its own
  blast, not that we trap it. Audit 11 adds that **79 % of oracle bombs involve no walking at all**,
  and E39's `k = 0` arm — which separates opportunism from positioning and which E40 dropped —
  puts the walking half at +0.077 (ns).
- **This is a `rule_based_agent` behaviour, not a game mechanic**, so it is precisely what E41 was
  built to test. **Amendment, recorded before the rerun it governs:** E41's P4 says to rerun
  `hunt_ceiling_v2.py --k 4 --trap-model sim`. Per this correction the correct instrument is
  **`--trap-model stale`**, and the +0.25 bar must be restated as a **conversion** bar against that
  field's own bombs/round rather than copied across designs. Nothing about E41's P1–P3 changes.
- **Two significant results this entry left unreported:** `coins` −0.040 (t = −2.25) and `won`
  +0.018 (t = +2.47) on k4 sim − control.
- **The ledger's own order guarantee does not hold for E39 or E40.** The file's rules say "Ich
  committe die Vorhersage, bevor ich messe — dann belegt die Git-Historie die Reihenfolge." Both
  entries first appear in commit `5819f0f`, *after* their sweeps. The predictions genuinely were
  written first, but **the git history cannot prove it**, which is the whole point of the rule.
  Fixed going forward by committing the pre-registration before launching.

**What survives untouched:** the NO-GO itself; P3's conversion measurement (8.4 % stale / 12.0 %
sim, and audit 10's F2 conversion rate reproduces even though its *explanation* does not); the
static-board-over-the-fuse problem, which remains the real reason a trap test mispredicts and is
still unfixed; and E39's headline +0.116 failing to replicate.

### Limitations

- **This is still an oracle over a fixed table, against `rule_based_agent` only.** Audit 10's F4
  showed our kill count is set mostly by the *field's* suicide rate — `rule_based` consumes 82 % of
  its own mortality before we can reach it. **Hunting could be worth materially more against a field
  that dies less to itself**, and `scratchpad/benedict/e40_field_design.md` P3 pre-registers the
  rerun against external agents at the same +0.25 bar.
- **`margin` vs `score` is still unresolved** pending the course's answer on the tournament format.
  Both were pre-registered here so the entry is scored either way, and **both fail their bar**, so
  the answer does not change this verdict — only how the number is reported.
- The conversion trace credits us if our kill count rose during the fuse window, which would
  false-positive on a simultaneous kill of a *different* opponent. That biases the 12 % **upward**,
  making the P3 failure stronger.

---

## E39 — the hunt ceiling: an oracle that plays the strategy perfectly is worth +0.116 score

- **Question:** does it pay to work toward the nearest opponent *while* crates still stand, rather
  than only after the board is stripped? My hypothesis, written down before any probe ran:

  > "The agent almost always stays alive until all crates and coins are gone and there is at least
  > one other agent on the board. At that point it is harder to get a kill than when there are still
  > crates — it is so much easier to just outrun the bomb and hide. With crates it is easier to lock
  > the opponent in a corner. So: while destroying crates and collecting coins, work one's way
  > toward the nearest opponent, rather than only collecting until no direct path to an opponent is
  > left."

  **No rung-4 experiment has ever tested this**, and the reason is representational, not
  accidental. `callbacks.py:244-255` — `target_direction` falls through to the nearest opponent
  **only when no coin and no crate is reachable**. By construction digit 6 can never point at an
  opponent while the board still has crates, so *the agent cannot express the strategy at all*.
  E35 priced kills the state could not aim at (`KILLED_OPPONENT` at 5 and at 25 moved kills
  −0.002); E36 put opponent BFS distance in the *danger* rows only, and was refuted.

  Three probes established the premise before this ran (`scratchpad/strategy/`,
  `TASK_B_argument.md` §6). `hunt_window.py`: we are alive when the last crate falls in **69.1 %**
  of rounds (median step 210), and a trappable configuration is **1.6× more common** mid-crate-phase
  than on the stripped board (27.2 % at 30–59 crates vs 16.7 % at 0) — **the mechanism is real.**
  But the nearest trap site is 9.7 steps away with crates against 5.2 without, and
  `trap_persistence.py` measures trap information dead beyond ~3 steps: of traps re-checked on
  arrival, 22.6 % survive a *single* step, 4.0 % survive a 5–8 step walk, and the tail is flat at
  the memoryless base rate from five steps out to fifteen. `rule_based_agent` flees actively and a
  trap's half-life is roughly one move.

  So the configuration advantage of the crate phase is cancelled by the distance needed to use it —
  on paper. **A ceiling agent settles it without a sweep**, and this rung cannot afford to spend a
  15-seed sweep on a question a one-afternoon oracle can answer.

- **Change:** none to the agent. **No file under `agent_code/` was written.** The line-up drives the
  provided `agent_code/user_agent/` (its `act` returns `game_state['user_input']`) from
  `scratchpad/strategy/hunt_ceiling.py`, which supplies that input. The policy is the shipped
  Q-table's greedy action, **overridden only** when a *self-survivable trap site* for a reachable
  opponent lies within *k* steps: walk the BFS first step toward it, `BOMB` on arrival. A trap site
  is a free tile whose bomb would leave that opponent with no survivable escape; it is re-verified
  every step with the same time-aware BFS that `callbacks.escape_direction` runs on ourselves.
  `k = -1` disables the override entirely.

  This plays the hypothesis **perfectly**, with an oracle no tabular feature could ever supply: it
  knows every step which tile traps which opponent, with a full BFS over that opponent's escape
  options. That is the point — it bounds the prize before anyone designs a digit.

- **Design.** Arms *k* ∈ {−1, 0, 2, 4, 8} at n = 1000, then the two that matter rerun at
  **n = 4000**. Held-out ship seed **990731**, scenario `classic`, 3 × `rule_based_agent`. The
  protocol mirrors `tools/evaluate.py` exactly — per-round `world.rng` reseed *and* the
  `np.random.seed` that reaches the provided opponents — so the CSVs are paired arena-for-arena and
  feed `analyze.py --compare` directly. The table is `agent_code/benedict_task4/q_table.npy`,
  md5 `54d63bc79179fdf80d3b9bfb80f21461`, **byte-identical to the shipped
  `checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy`** — verified, not assumed.
  Commit `5b0b0c4` (dirty). Data: `scratchpad/strategy/ceil/`.

- **Power, stated in advance.** There is **no training variance here** — one fixed table, so the
  only noise is evaluation noise. That is the ±0.12 floor on `score` at n = 1000 from the
  opponents' unseeded stdlib RNG (E37 §5.9); at n = 4000 it is ~±0.06, and pairing is on arenas.
  **This is the most precise instrument on the project**, which is exactly why the question was
  routed here instead of to a sweep whose n = 15 MDE is 0.254.

### Prediction

**Provenance, stated plainly: these bars were written in `scratchpad/strategy/TASK_B_argument.md`
§6.5 before the arms in §7 were run, and this ledger entry was written afterwards.** The
predictions below are quoted from §6.5, not reconstructed; the entry is late, the pre-registration
is not. Scored as written.

1. **P1, primary — the go/no-go.** *"A ceiling worth chasing should be ≥ +0.5; anything under
   ~+0.25 is not readable and, being a ceiling, not worth pursuing anyway."* So: **GO** if the best
   arm beats the control by ≥ +0.50 score with a CI excluding 0; **NO-GO** if it is under +0.25.
   Between the two is an undecided band and would need a wider design.
2. **P2, decisiveness.** *"If the oracle hunter does not beat 3.949, no digit encoding this will."*
   The oracle is an upper bound: a tabular digit can offer at most "a trap is within *k* steps" plus
   a direction, and the table would still have to *learn* to follow it.
3. **P3, the horizon.** §6.3 measured trap information dead beyond ~3 steps, so the optimum should
   sit at small *k*, and **k = 8 should be no better than k = 4** — with the diversion cost landing
   on `crates`, which is where §3 said hunting is paid for.
4. **P4, the free version.** *k* = 0 — bomb only when already standing on a verified trap site — is
   exactly the zero-row digit-7 redefinition proposed in §4.2. If the whole effect sits in *k* = 0,
   the feature costs nothing and ships.
5. **Guards.** `suicides` must not rise (hunting is exactly when an agent forgets to run from its
   own bomb) and the control arm must reproduce the shipped agent's published numbers.

### Result — 4000 paired arenas, ε = 0, held-out seed 990731

**Control validation first.** *k* = −1 over 4000 rounds gives score **3.958**, `won` 0.382, kills
0.229, suicides 0.482, crates 33.49, against the shipped agent's published 3.949 / 0.406 / 0.226 /
0.488 / 33.55. The harness reproduces the agent. (`won` 0.382 vs 0.406 is the largest gap and sits
inside the ±0.12-score noise band via the +0.088/point conversion.)

First pass, n = 1000, five arms. "override" is diverted walk steps as a share of policy steps, plus
the count of bombs the override fired:

| k | score | coins | kills | `won` | suicides | crates | override |
|---|---|---|---|---|---|---|---|
| −1 (control) | 3.886 | 2.781 | 0.221 | 0.403 | 0.465 | 33.40 | — |
| 0 | 4.015 | 2.820 | 0.239 | 0.381 | 0.468 | 33.37 | 0.00 % / 370 bombs |
| 2 | 4.023 | 2.748 | 0.255 | 0.394 | 0.450 | 33.59 | 0.36 % / 348 |
| **4** | **4.086** | 2.841 | 0.249 | 0.410 | 0.448 | 33.29 | 1.39 % / 414 |
| 8 | 3.952 | 2.727 | 0.245 | 0.403 | 0.459 | **32.51** | 4.07 % / 469 |

No arm separates on score at n = 1000; *k* = 8 is significantly worse on **crates**
(−0.890 [−1.576, −0.199]). Kills rise in all four. **k = 4 and k = 0 were rerun at n = 4000.**

**k = 4 against the control, n = 4000:**

| metric | control | k = 4 | paired difference | verdict |
|---|---|---|---|---|
| **score** | 3.958 | 4.074 | **+0.116 [+0.002, +0.233]** | BESSER |
| `won` | 0.382 | 0.419 | **+0.037 [+0.017, +0.057]** | BESSER |
| kills | 0.229 | 0.260 | **+0.032 [+0.011, +0.052]** | BESSER |
| coins | 2.816 | 2.772 | −0.043 [−0.091, +0.005] | nicht gezeigt |
| **crates** | 33.49 | 33.07 | **−0.422 [−0.750, −0.081]** | SCHLECHTER |
| suicides | 0.482 | 0.463 | −0.019 [−0.041, +0.002] | nicht gezeigt |
| survived | 0.460 | 0.475 | +0.016 [−0.006, +0.038] | nicht gezeigt |

**k = 0 against the control, n = 4000:** score +0.039 [−0.078, +0.155], `won` +0.011 [−0.009,
+0.032], kills +0.006 [−0.014, +0.026], crates +0.046 [−0.285, +0.367]. **Nothing.**

**P1 — NO-GO, as written, and audit 10 makes it stronger rather than weaker.** The bar for GO was
+0.50 and for NO-GO +0.25. The best arm delivers **+0.116**, below the lower bar, so the
pre-committed consequence applies: **do not design a hunting digit on this evidence.**

And the +0.116 is weaker than the table above makes it look. **`analyze.py:220` seeds its bootstrap
with a hard-coded `default_rng(12345)`**, which makes the printed interval reproducible but not
*stable* — the lower bound has its own Monte-Carlo error and here the truth sits inside it:

| | score, k = 4 − control, n = 4000 |
|---|---|
| mean, standard error | +0.1158, 0.0599 |
| `analyze.py` interval (seed 12345) | [+0.002, +0.233] → excludes 0 |
| lower bound over 200 other bootstrap seeds | straddles 0; the interval excludes 0 in a **minority** of seeds |
| t | **+1.93** |
| sign-flip permutation p | **0.053** |

So on the primary metric, by this project's own rule, the ceiling is **nicht gezeigt**. `kills`
(+0.032, t = +3.03) and `won` (+0.037) do survive. Verified independently of `analyze.py` in
`scratchpad/benedict/e39_audit_verify.py`.

**And the selection contamination flagged above is real but small.** k = 4 was the pilot's best of
five arms at n = 1000, and rounds 0–999 of the confirmation reuse those arenas. On the 3000 **fresh**
arenas alone: score **+0.102 [−0.041, +0.241], t = 1.47**. The winner's curse costs ~0.014; the
metric was already not significant without it.

**P2 — the premise is FALSE, and this is the finding that matters most.** P2 assumed the oracle is
an *upper* bound. It is not. `hunt_ceiling.trap_sites` asks whether the target can escape **from the
tile it currently occupies** — but `environment.py:421-432` hands every agent the same pre-action
snapshot and only then executes the actions in a random permutation, so by the instant our BOMB
exists the target has already taken a move the trap test never modelled. **The "verified inescapable
trap" is verified against a position the opponent is leaving.** Audit 10 followed every override
bomb to its fuse (k = 4, 200 rounds): the target dies **18.6 % [11.0, 28.4]** of the time and *we*
are credited **8.1 % [3.3, 16.1]**. The arithmetic closes — 0.406 override bombs/round × 8.1 % × 5
points ≈ +0.16, against +0.116 after the crate cost. **The entire measured ceiling is the conversion
rate of a one-step-stale heuristic.** So +0.116 is a **lower** bound on "work toward the nearest
opponent", not the upper bound §6.5 promised, and the sentence "it knows every step which tile traps
which opponent" is false. I verified the framework mechanism; the 8.1 % is audit 10's rollout and I
have not independently reproduced it.

**P3 — the crate leg PASSES, the ordering leg is not demonstrated.** The predicted diversion cost
lands exactly where §3 said it would: k = 8 loses **−0.890 [−1.576, −0.199] crates** against
control, CI excluding 0. **Hunting does take bombs away from the economy**, as C measured in the
survey (the aggressive variant "stopped collecting coins"). But "k = 8 is no better than k = 4" is
a point estimate only — k8 − k4 on score is −0.134 at t = −1.09. **The claim in
`TASK_B_argument.md` §7.3 that "the k = 8 arm bounds the 'chase harder' direction empirically" is an
over-read of noise and should not be repeated.**

**P4 FAILS — the free version buys nothing.** k = 0 moves score +0.039 [−0.078, +0.155] on 4000
paired arenas. The zero-row digit-7 redefinition is dead, and the whole of the (small) effect lives
in the *diversion*, not the bombing rule: k = 4 fires 1.39 % of steps as diverted walks and 44 extra
bombs per 1000 rounds over k = 0.

**Guards pass.** Suicides do not rise in any arm (−0.019 at k = 4). Control validation above, and
audit 10 confirmed it on two behavioural fingerprints the marginals would not have caught
(`invalid` 4.771 → 4.734, `moves` 213.54 → 213.44, paired score −0.017 [−0.256, +0.220]).

### Verdict — **nicht gezeigt**, and the pre-registered NO-GO stands

On the metric this entry pre-registered, the hunt ceiling is **not demonstrated** (t = 1.93,
p = 0.053), and the pre-committed consequence — do not design a hunting digit — applies unchanged.

**But the reason recorded in `TASK_B_argument.md` §7.3 is wrong and must not go into the report as
written.** §7.3 says the direction is closed because a *perfect* oracle is worth only a tenth of a
point. What was actually measured is a *broken* oracle with an 8 % hit rate. The honest statement is:

> **The hunting direction is not closed. It is unmeasured.** The instrument built to bound it was
> mis-specified against the game's simultaneous-move rule, and the number it returned is a floor,
> not a ceiling.

**Method note, and it is the transferable part.** The fixed-table oracle *is* the right instrument —
it removed training variance and answered in an afternoon what fifteen seeds could not read. What
this entry adds is that **an oracle is only an upper bound if its world model matches the
environment's**, and ours did not. Write the oracle before the digit, then check the oracle against
`environment.py` before believing it.

### What audit 10 found that this entry does not resolve

Nine audits have now run on this project and eight overturned something; this is the eighth. Full
report and evidence in `scratchpad/audit10/`. Verified independently by me in
`scratchpad/benedict/e39_audit_verify.py`: F1, F3, F6 and F4 reproduce exactly; F2's 8.1 % does not
(it needs their instrumented rollout), though its mechanism is confirmed in `environment.py`.

- **The ceiling was scored on our own score; on a *between-agent* margin it is 2.3× larger and
  robust.** `score_ours − score_best_opponent`, same 4000 paired arenas: **+0.270 [+0.088, +0.444],
  t = 2.95**, sign-flip p = 0.0034, and the interval excludes 0 on **100 %** of 200 bootstrap seeds
  — unlike the own-score number. On fresh arenas only, +0.253 [+0.046, +0.465]. The mechanism is a
  clean credit transfer, not extra killing: our kills **+0.032 (t = +3.03)**, opponents' kills
  **−0.049 (t = −3.21)**, total opponent deaths **+0.006 (t = +0.48)**. The oracle takes deaths that
  were going to happen anyway and puts our name on them — worth +5 to us and −5 to a rival.
  **This is post-hoc and is deliberately NOT used to rescore P1**; a statistic that failed does not
  get redefined after the fact. It is recorded here because it defines the next question, not
  because it changes this one. Two caveats of my own: the metric was chosen after seeing the result,
  and the "rivals" here are `rule_based_agent`s, who are not in the tournament — denial is only
  worth points against agents whose score is actually being ranked against ours.
- **Whether margin or own score is the right criterion depends on the unanswered Discord question.**
  If the tournament sums our score across games, own score is right and this entry is finished. If
  it ranks agents by total score, margin is right and the hunt question reopens above
  `NEXT_STEPS.md` §3.4's own +0.25 bar. **That message is no longer a formality — it is worth
  +0.15 of measured effect on a live decision.**
- **`analyze.py`'s fixed bootstrap seed makes every borderline verdict on this project look
  deterministic when it is a coin flip.** Any shipped verdict whose CI bound sits within ~±0.01 of
  zero needs re-reading with a t or a permutation p. That includes checking E37's +0.255.
- **The 290-effective-states argument does not license a bigger table.** The shipped table has
  **8 186 rows with non-zero Q** — 28× the "effective" 290 — because the cost is paid during
  ε-greedy training, not at ε = 0 on the converged policy. At 20 000 episodes that is ~383
  updates/row, and E38 forbids buying more. **`NEXT_STEPS.md` §3.1's size-4 target-type digit loses
  its cost justification; the free re-partition option is untouched.** Verified: 8 186 exactly.

### Limitations of this entry

- **The oracle never uses our own bomb to herd or close an exit**, and never chases an opponent that
  is not already trappable — and per P2 it does not even correctly identify the ones that are.
- **The arms drifted from the plan.** §6.5 described k = 0 as "an exact reproduction of the shipped
  agent"; it is not — it fires 370 bombs the shipped agent does not. The real control, k = −1, was
  added at execution time.
- **Pairing buys almost nothing here.** `se_paired` 0.0599 vs `se_unpaired` 0.0640 — a 6 % narrower
  interval, because the arms desynchronise the opponents' stdlib RNG from the first divergent step
  and only the arena is shared. The CI is valid; `evaluate.py`'s docstring oversells the design.
- `hunt_ceiling.py` writes its own `.meta.json` and, unlike `tools/evaluate.py`, **does not record
  the git commit**. The commit above is inferred from the same hour's `evaluate.py` artefacts
  (`scratchpad/strategy/fields/*.meta.json`: `5b0b0c4-dirty`). Any future scratchpad harness should
  snapshot the commit itself.

---

## E38 — 20 000 episodes was inherited from rung 2, and rung 2's own curve is non-monotone

- **Question:** every rung-4 run has used 20 000 episodes. That number was chosen on **rung 2**
  (E23: "longer is worse here, 40 000 loses ~7 crates") and **never re-validated on rung 4.** E37
  makes the omission live: its `PLB2` arm is the first that is still learning at the horizon.

  | | @5 000 | @10 000 | @20 000 |
  |---|---|---|---|
  | `ctl2` score | 3.625 | 3.636 | 3.615 |
  | `SHF` score | 3.727 | 3.731 | 3.694 |
  | **`PLB2` score** | 3.722 | 3.834 | **3.895** |

  The control is **flat from 5 000 on** — which is exactly why nobody questioned the budget: for
  every arm before E37 the last 15 000 episodes bought nothing. `PLB2`'s training stream rises
  monotonically into the final band (2.304 → 2.854 → 2.924 → **2.966**) where the control's falls
  (2.224 → 2.730 → 2.710 → **2.686**).

  **But "train longer" cannot be extrapolated, and rung 2 is why.** The episode response there is
  non-monotone with a named mechanism: 40 000 lost ~7 crates, **200 000 turned the incumbent from
  97.31 into 62.85 by re-rolling near-tie rows** (E18), and **300 000 was the best in its arm**
  (97.59 dev / 97.45 held-out, E21). The ledger's own verdict: *"the fine map dips and recovers by
  300 000, so 'train longer' is not reliably anything."* E22 corroborates the mechanism from the
  other side — α → 1.0 tripled the thin-margin fraction and cost the best seed 88 crates.

  So this is not a budget sweep. It is a test of whether **margin erosion** applies to a feature
  map that splits its danger rows harder than rung 2's did.

- **Change:** episodes only. 300 000 instead of 20 000, checkpoints at 20 000 / 40 000 / 80 000 /
  160 000 / 300 000. Everything else is the shipped configuration at its defaults.

- **Design.** 5 seeds (`BM_RUN_INDEX` 120-124), `--seed 810731`, **5 concurrent** rather than 10 so
  each run holds a performance core. **The control is the same run's @20 000 checkpoint**, so every
  comparison is within-run and paired by seed. No separate control arm exists at this horizon and
  none is affordable — which is a real limitation, since it means the margin-erosion prediction is
  tested against rung 2's recorded curve rather than against a contemporaneous `ctl2`.
  1000 rounds at ε = 0, validation seed 550731, every checkpoint evaluated.

- **Power, stated in advance.** n = 5 against the measured paired SD of `score` differences (0.358)
  gives an 80 %-power MDE of **0.482**. **This design can only detect a rung-2-scale collapse, not
  a small gain** — and that is deliberate, because the question is "does the horizon hurt", not
  "is 300 000 slightly better". Any *positive* result here is a lead requiring confirmation at
  n ≈ 15, never a claim.

### Prediction (written before the run)

1. **P1, primary — the guard.** `score` @300 000 is **not worse** than @20 000 by more than the
   MDE: the paired difference has a CI whose lower bound is above **−0.48**. **Refutation:** a
   drop with a CI excluding 0 → rung 2's re-rolling failure reproduces on rung 4, 20 000 is
   vindicated as a budget, and the shipped table stays.
2. **P2, mechanism — the shape, which is what makes this worth a night.** The curve is
   **monotone or flat**, not U-shaped. Specifically `score` @40 000 ≥ @20 000 − 0.20.
   **Refutation:** a dip at 40 000-80 000 that recovers by 300 000 reproduces rung 2's shape, and
   the conclusion is that margin erosion is a property of the *algorithm* (ε floor + α annealing),
   not of the feature map.
3. **P3, the margin measurement that discriminates them.** Thin-margin share among updated danger
   rows — |best − second| < 0.01 — **stays below 1.5× its @20 000 value at every checkpoint.**
   This is measured on the tables, not inferred from score, and it is the quantity E18/E22 named.
   **Refutation:** it grows past 1.5× while score holds → the erosion is real but not yet
   expressed, and 300 000 is a cliff edge rather than a safe horizon.
4. **P4, guards.** `suicides` ≤ 0.60 · `crates` ≥ 31.0 · all-zero-row share < 0.01 ·
   `think_over_limit` = 0.
5. **Ship rule, pre-committed.** The shipped table changes **only if** a checkpoint beats @20 000
   with a paired CI excluding 0 on `score` at n = 5 *and* survives confirmation on the held-out
   ship seed 990731 against the current ship. Given P1's MDE that requires an effect ≥ 0.48, which
   I do not expect. **The expected outcome of this experiment is "20 000 was fine", and its value
   is that this stops being an assumption.**

**Cost:** 5 runs × 300 000 episodes ≈ 13-16 h at 5 concurrent, one night. E21's lesson is carried:
**every checkpoint gets evaluated.** Its five 300 000-episode tables sat on disk unevaluated and
reversed a scored prediction when they were finally measured.

### Result — 25 checkpoints, 1000 rounds at ε = 0, validation seed 550731, n = 5

Sweep ran 2026-08-16 20:38 to 2026-08-17 15:40 (19.0 h; my 8-10 h estimate was wrong because
episodes lengthen as the horizon grows). 25/25 checkpoints, no tracebacks, all 25 evaluated.

| episodes | score | `won` | crates | coins | suicides | survived | crates/bomb | paired vs @20 000 |
|---|---|---|---|---|---|---|---|---|
| **20 000** | **3.976** | **0.389** | 32.32 | 2.762 | 0.478 | 0.457 | 1.182 | — |
| 40 000 | 3.810 | 0.377 | 32.63 | 2.687 | 0.510 | 0.430 | 1.155 | −0.166 [−0.502, +0.170] 2/5 |
| 80 000 | 3.848 | 0.374 | 32.79 | 2.627 | 0.550 | 0.401 | 1.165 | −0.128 [−0.473, +0.217] 2/5 |
| 160 000 | 3.637 | 0.348 | 29.87 | 2.393 | 0.493 | 0.459 | 1.104 | −0.339 [−0.912, +0.234] 1/5 |
| **300 000** | **3.206** | **0.304** | 26.23 | 2.020 | 0.487 | 0.474 | **0.988** | **−0.770 [−1.355, −0.185] 0/5** |

**P1 REFUTED.** The pre-registered guard was a CI lower bound above −0.48; it is **−1.355**, the
point estimate is −0.770, and **0 of 5 seeds improved**. Training past 20 000 episodes makes this
agent *worse*, and by more than the design's own MDE of 0.482. **The pre-committed consequence
applies: 20 000 is vindicated as a budget on evidence rather than inheritance, and the shipped
table stays.** The ship rule required a checkpoint beating @20 000 with a CI excluding 0; none does.

**P2 PASSES as written, and the shape refutes rung 2's.** The bar was `score` @40 000 ≥ @20 000 −
0.20, i.e. ≥ 3.776; it is 3.810. There is **no dip-and-recover** — the curve declines steadily and
never comes back, where rung 2's fell to 62.85 at 200 000 and returned to its best at 300 000. So
the two rungs fail differently, and "the fine map dips and recovers" does not generalise.

**P3 FAILS as written, and the mechanism it was built to detect is contradicted.** Pooled thin
margin ran 1.00 → 1.34 → 1.54 → 1.59 → **1.68×**, breaching the 1.5× threshold at 80 000. But the
pooled statistic is confounded by cohort composition: the number of updated danger rows grows
2 314 → 3 465 (+50 %), and newly reached rows are thin by nature (share ≈ 0.22-0.24) because they
are barely trained. **Fixing the cohort to the rows already updated at 20 000, the thin-margin
share *falls* 0.0466 → 0.0171 — 0.37×.** Established rows get **sharper**, not thinner. E18's
re-rolling mechanism does not reproduce here.

**That is my error, made one day after audit 9 caught its sibling.** Audit 9 showed that measuring
over rows which merely *carry* value rather than rows training *changed* returns +0.45 on a control
that cannot encode the signal at all. P3's definition repeats the same class of mistake in a new
place: a pooled statistic over a growing population measures the population, not the quantity.

**P4: one guard fails.** `crates` @300 000 is **26.23 against a 31.0 bar — FAIL**, which is the
substantive finding rather than a technicality. `suicides` 0.487 ≤ 0.60 passes; `think_over_limit`
is 0 across all 25 checkpoints (worst single step 7.86 ms against 500); coverage passes with room —
0 all-zero rows in 28 818 ε = 0 alive steps, and rows carrying value *rise* 8 167 → 9 413, so this
is not a coverage failure either.

**The channel is bomb siting, and it is the one E37 won on.** Crates per bomb falls
**1.182 → 0.988 (−0.194)** while bombs dropped barely move. E37's shipped feature bought **+0.130**
crates per bomb; the horizon gives back **1.5× that**, through the identical quantity. Coins track
it (2.762 → 2.020) because coins come from opened crates.

**A reading I had to withdraw.** From the training stream I described the agent as becoming
*passive* — survival rose 31 % and episodes lengthened 14 % between 20 000 and 250 000. **At ε = 0
that does not hold**: survival is 0.457 → 0.474, flat, and suicides are flat too. The passivity is
a property of the ε = 0.02 exploring policy, not of the greedy one. The greedy policy is not
safer; it is simply worse at placing bombs. **A training curve is not a result, and this is the
sixth time that rule has bitten on this project.**

- **Verdict: SCHLECHTER for longer training. 20 000 episodes is correct on rung 4, now measured.**
  The shipped table is unchanged.
- **P1 refuted · P2 passed · P3 failed as written, with its mechanism contradicted · P4 one guard
  failed (`crates`), three passed.**
- **What this is worth for the report.** The last inherited hyperparameter on this rung is now
  measured rather than assumed, and the answer is not the one rung 2 would have predicted: the
  degradation is monotone rather than U-shaped, it is not margin erosion, and it runs through the
  same bomb-siting channel that the winning feature ran through in the opposite direction. That
  makes "how long to train" a statement about *what the extra episodes do to bomb placement*, not
  a budget line.
- **Open, and deliberately not chased:** why more training degrades bomb siting while sharpening
  the rows it has already seen. The obvious candidate is that the target is non-stationary — the
  policy trains against a distribution it is itself changing — but nothing here tests that, and
  rung 4 is closed.

---

## E37 — Is the danger-row lattice class worth points, and is it the lattice or the label?

- **Question:** E36's intended placebo beat its treatment. `(x+y) mod 4` in the danger rows is
  worth **+0.980 crates** and **+0.211 score** against the control (5/5 seeds, monotone across
  checkpoints, replicated in the training stream at +0.94), while opponent distance is worth
  nothing. But E36 **cannot go into the report as it stands** — audit 8 found three defects: its
  arms and control had different warm-start parents, its verdict label was wrong, and its primary
  metric was untestable at its own predicted magnitude. One sweep removes all three.

  And there is a real question underneath. `(x+y) mod 4` is **not** "position parity" as the survey
  uses the term. Its low bit *is* the lattice class exactly — pillars sit at (even, even), so a
  free tile has `x+y` even ⟺ both coordinates odd ⟺ a **crossing** (4 structural exits, own bomb
  clears 12 tiles); `x+y` odd ⟺ a **corridor** (2 exits, 6 tiles). Verified on 17 904 danger steps:
  11 477/11 477 and 6 427/6 427, zero exceptions. Its high bit is an arbitrary diagonal stripe.

  Three measurements say the lattice bit is the part that matters. **It is 94.6 % but not 100 %
  predictable from digits 1-4** — H(lattice) = 0.942 bits, H(lattice | digits 1-4) = 0.192 —
  because `NB_BLOCKED` merges wall with crate, so a corridor's two permanent walls look like two
  crates. **The trained policy splits on it 4.8× harder than on the arbitrary component**: moving
  digit 8 *across* parity changes the greedy action on 5.75 % of danger steps, moving it *within*
  parity 1.20 % (null calibration on the control table: 18.29 % vs 18.29 %, exactly symmetric).
  **And E36's own P2 points the same way** — the only bucket where the opponent digit changed
  behaviour was bucket 0, which is 73.4 % "opponents alive but walled off", i.e. structural.

  So the hypothesis is specific: **the missing information is about the board, not the opponents**,
  and it is the wall lattice the neighbour code throws away.

- **Change:** digit 8's meaning in the `own_danger` branch only. `FEATURE_SIZES` untouched.
  Four arms, **all warm-started from `_e36parent`** — which is the matched control E36 never had.

  | arm | digit 8 in the danger branch | role |
  |---|---|---|
  | **ctl2** | `DIST_NONE` (unchanged) | **the control E36 lacked** |
  | **PLB2** | `(x + y) % 4` | direct replication |
  | **PAR** | `(x + y) % 2` | the lattice class alone, no positional component |
  | **SHF** | fixed relabelling of free tiles into 4 groups, **balanced within each lattice class** | position *without* the lattice — the true null |

  `SHF` is the control E36 should have had: same arity, same "fills 40 960 dead rows" perturbation,
  and by construction it carries **no** crossing/corridor information.

- **Design.** **n = 15 seeds** (`BM_RUN_INDEX` 100-114), 20 000 episodes, checkpoints
  5 000 / 10 000 / 20 000, otherwise E33 ctl verbatim, `--seed 810731`, paired at run level.
  1000 rounds at ε = 0, validation seed 550731.

  **n = 15 is chosen from the measured variance, not from habit.** Paired SD of `score`
  differences over eight arm sets is **0.1825**:

  | n | 80 %-power MDE on `score` | on `won` |
  |---|---|---|
  | 5 | 0.352 | 0.045 |
  | 10 | 0.210 | 0.027 |
  | **15** | **0.164** | **0.021** |

  The expected effect is ~0.20 on `score`. **At n = 10 the MDE is 0.210 — the design would be
  ~50 % powered, which is the exact mistake this entry exists to correct.** (Audit 8 proposed
  n = 10 quoting an MDE of 0.165; recomputed from the committed CSVs it is 0.210, so its own
  recommendation was underpowered.)

- **Primary metric: `score`, committed now with the reason.** `won` at n = 15 has MDE 0.021
  against an expected +0.024 — a coin flip by construction. **This is not metric-shopping only
  because the question under test is "is E36's effect real, and is its mechanism the lattice?",
  not "do we have a better tournament agent".** `won` is reported alongside with its CI and the
  MDE quoted next to it, and **no tournament-ranking claim is made from `score`.**

### Prediction (written before the run)

1. **H1, replication, primary.** `PLB2 − ctl2` on `score` ≥ **+0.10**, CI excluding 0.
   **Magnitude 0.15-0.30.** **Refutation:** CI includes 0 → E36's PLB was the warm parent or noise.
   **Rung-4 optimisation then stops for good** and E36 is written as a clean negative.
2. **H2, mechanism.** `PAR − ctl2` on `score` ≥ **0.5 ×** `(PLB2 − ctl2)`, CI excluding 0.
   **Refutation:** `PAR` ≈ 0 while `PLB2` > 0 → the gain rides on the arbitrary positional
   component, i.e. it is overfitting to the training arenas. **Stop, and report it as such.**
3. **H3, specificity — the control E36 should have had.** `SHF − ctl2` on `score` CI **includes 0**
   *and* `SHF < PAR`. **Refutation:** `SHF ≈ PLB2` → any 4-way positional split works, so the gain
   is capacity rather than information. **Stop.**
4. **Guards.** `won` ≥ 0.36 · `crates` ≥ 31.0 · all-zero-row share < 0.01 · `think_max_ms` < 5.
5. **Secondary, reported not tested.** `won` per arm with CI, **with the n = 15 MDE of 0.021
   printed beside it**, so that a null is not read as a refutation the way E33-E36's were.

**Continuation rule, pre-committed.** Only if **H1 and (H2 or H3)** hold does a `won`-powered
confirmation at n = 20-30 follow. Otherwise **rung 4 closes** and the remaining weeks go to the
report. This is one terminal experiment, not a new programme.

**Cost:** 60 runs ≈ 11 h unattended, ~2 of the ~30 sweeps the remaining five weeks allow.
**Fallback if that is too long:** drop `PAR` (45 runs, ≈ 9 h); H1 and H3 still resolve and H2
becomes an inference from `SHF` alone.

### Result — 1000 rounds at validation seed 550731, n = 15 runs, paired at run level

Sweep ran 2026-08-16, 06:22-15:12; 180/180 checkpoints, no tracebacks. Two structural checks that
E36 failed both pass, and the second was **impossible to run before this morning**: all 60 runs
share `warm = _e36parent` with identical reward tables, and **180/180 evaluations read their table
under the correct `BM_D8`**, verified from the new `bm_env` field in each `.meta.json`.

| @20 000 | score | **won** | crates | coins | suicides | survived | kills |
|---|---|---|---|---|---|---|---|
| **ctl2** | 3.615 | 0.3595 | 31.38 | 2.539 | 0.530 | 0.420 | 0.216 |
| **PLB2** `(x+y)%4` | **3.895** | 0.3833 | **33.13** | 2.763 | 0.503 | 0.440 | 0.228 |
| **PAR** `(x+y)%2` | 3.770 | 0.3672 | 32.43 | 2.650 | 0.534 | 0.414 | 0.225 |
| **SHF** null | 3.694 | 0.3607 | 31.24 | 2.596 | 0.522 | 0.423 | 0.221 |

Paired differences vs `ctl2`, 95 % t-CI, seeds positive:

| | score | crates | coins | won |
|---|---|---|---|---|
| **PLB2** | **+0.280 [+0.081, +0.478]** 13/15 | **+1.757 [+0.892, +2.622]** | **+0.224 [+0.129, +0.318]** | +0.024 [−0.006, +0.054] |
| **PAR** | +0.155 [−0.018, +0.328] 13/15 | **+1.057 [+0.233, +1.881]** | **+0.111 [+0.038, +0.184]** | +0.008 [−0.016, +0.031] |
| **SHF** | +0.079 [−0.111, +0.269] 9/15 | −0.136 [−1.442, +1.170] | +0.057 [−0.057, +0.172] | +0.001 [−0.025, +0.027] |

**H1 HOLDS.** +0.280, p = 0.0091, inside the pre-registered 0.15-0.30 band, monotone across
checkpoints (+0.097 → +0.198 → +0.280; crates +0.512 → +0.889 → +1.757). **E36's PLB effect
replicates with the warm-start confound removed, and larger** (E36's confounded estimate: +0.211).

**H2 REFUTED as written** — `PAR` needed ≥ +0.140 *and* a CI excluding 0; it reaches +0.155 but
p = 0.076. **The refutation reading pre-registered for it does not apply.** That text says "`PAR`
≈ 0 while `PLB2` > 0 → the gain rides on the arbitrary positional component". `PAR` is not ≈ 0: it
moves crates and coins with CIs excluding zero, and the table measurements below show the opposite
of what the refutation asserts. H2 failed on power, not on its mechanism.

**H3 HOLDS as written but is UNDEFINED at this power, and this is the entry's methodological
finding.** `SHF − ctl2` = +0.079 [−0.111, **+0.269**]: an accept-the-null test whose CI tolerates
**96 % of `PLB2`'s own effect**. Low power makes such a test *easier* to pass. **H3 is a leg of the
continuation rule**, so the rule fires `CONTINUE` on evidence that cannot distinguish "the null
does nothing" from "the null does most of what the treatment does". **This is E36's P5 error —
scoring a ratio against a denominator indistinguishable from zero — relocated onto the go/no-go
decision.** `e37_analyse.py` printed the underpower caveat for `won` and not for H3; fixed.

**Guards (P4).** `crates` ≥ 31.0 passes everywhere (31.24-33.13). `won` ≥ 0.36 passes in all three
arms; `ctl2` misses at 0.3595, which is the control, not an arm regressing. **Coverage passes with
a ≥ 125× margin** — all-zero-row share `ctl2` 0.00000, `PLB2` 0.00008, `PAR` 0.00004, `SHF` 0.00006
over ~66 000 ε = 0 alive steps per arm (`scratchpad/benedict/e37_coverage.py`, which calls the
agent's own `state_to_features` rather than reimplementing it). E28's deferral condition stays
expired for the third measurement running: 0.026 → 0.0001 → 0.00006.
**`think_max_ms` < 5 ms FAILS in all four arms** (14.5-41.5, global max 53.4). It carries no
information: the control fails identically, mean think time is 0.134-0.138 ms in every arm, and the
constraint it proxies — `TIMEOUT = 0.5 s` per step — is met with **0 breaches in 180 000 rounds**.
It is a max over ~2.7 M steps taken under ten concurrent evaluations, i.e. an OS-scheduling
outlier. **Future entries should guard on `think_over_limit > 0` or a p99, not on a max.**

**P5, `won` reported not tested.** `PLB2` +0.024 [−0.006, +0.054] against an observed MDE of 0.042.
Null, and **underpowered — not evidence of absence**, exactly as point 5 required it be stated.

### Audit 9 — the effect survives; the design and my analysis do not

Briefed to break H1, analysis-only. It could not kill the effect. Everything below was re-derived
independently before being written here.

**The headline is inflated by one collapsed control run.** `ctl2` s105 evaluates at 2.596 against
3.688 for the other fourteen (z = −8.4 against them), and it is a genuine *training* collapse. Its
paired difference is +1.42, 3.2× the next largest. **Drop it: +0.198 [+0.097, +0.299], p = 0.0010**
— the effect becomes *more* significant and the SD halves. Leave-one-out never leaves
[+0.198, +0.312]. **The honest magnitude is ≈ +0.20, not +0.28**, and H1 clears +0.10 on every
estimator.

**The t-CI is not valid.** Shapiro p = 0.0009, skew +2.16. Bootstrap over seeds gives
**[+0.133, +0.482]**; Wilcoxon p = 0.0015; sign test p = 0.0074. The +0.081 lower bound was a
normal-fit artifact.

**"Paired at run level" bought nothing.** corr(arm, ctl2) at matched seed runs −0.47 to +0.44, and
`SD_paired/SD_unpaired` exceeds 1 in 5 of 9 cells — the unseeded training opponents destroy the
pairing that the design's power calculation assumed.

**So E37 was underpowered on its own primary, which is what it existed to fix.** Realised paired SD
is **0.358**, not the 0.1825 assumed → 80 %-power MDE **0.256**, not 0.164; the design was ~60 %
powered. The pre-registered MDE table was also internally inconsistent before the sweep ran: 0.352
/ 0.210 / 0.164 back-solves to SD = 0.211 at every n, not the 0.1825 quoted beside it. And
`e37_analyse.py` used the large-sample constant 2.8 for (t₀.₉₇₅ + t₀.₈₀), understating every MDE by
7.6 %. Both fixed.

**The clean contrast is one this entry never pre-registered.** `PLB2 − SHF`: **+0.2005
[+0.0437, +0.3572]**, 11/15, Shapiro 0.718, and +0.183 without s105. Same arity, same parent, same
learning-rate dilution, differing only in whether the label tracks the lattice. It is immune to the
outlier, to the α confound, and to "any 4-way split works" — and it is the number this entry should
be built on.

**`SHF` is not inert either.** In the training stream it beats `ctl2` by +0.146 (p = 0.030, 13/15
over ep 15-20 k), indistinguishable from `PAR`. **Changing digit 8 at all buys something**, which is
the perturbation effect the entry predicted and which is visible at 5 000, where all three arms sit
at ~+0.10 and are indistinguishable. Only `PLB2` keeps building.

### Mechanism — the lattice bit does the work, conditional on the fuse

Measured on the Q-tables themselves, which no previous entry did. **A row counts only if training
*changed* it** (|q − parent| > 1e-12); "the row carries value" is not the same test, because the
warm parent is a factor-1 broadcast, so every warm-valued base is non-zero in all five siblings.

**My first derivation used the wrong filter and was falsified on the control.** Pooling over rows
that merely carry value returns **+0.4522 on `ctl2`** — a table whose digit 8 is pinned and which
can carry no lattice information at all. That estimator measures visit coverage × uplift, not
value: crossings are updated on 75 % of danger bases and corridors on 21 %, because a corridor has
two permanently blocked neighbours. The coverage gap alone manufactures the number.

Under the corrected filter, on rows training actually touched:

- **`max_a Q`(crossing) − `max_a Q`(corridor) = −0.060 (14/15 seeds) for `PLB2`, −0.056 (13/15) for
  `PAR`, +0.013 for `SHF`.** The arms carrying the lattice split it; the null does not.
- **The informative bit does ~3× the work of the arbitrary one.** Within `PLB2`, the mean split
  *across* lattice classes is **1.123** against **0.361** *within* them — **ratio 3.12, 15/15
  seeds**. This is the direct refutation of H2's canned reading: the gain does **not** ride on the
  arbitrary positional component.
- **And the information is conditional on the fuse:** crossings are worth **more** with one move
  left (+0.246, 15/15) and **less** at 2/3/4 (−0.189, −0.092, −0.108, 15/15 or 14/15). That is
  board physics — one move needs exits (4 at a crossing vs 2), two or more needs to clear a blast
  covering 12 tiles at a crossing and 6 in a corridor. `ctl2` learns the average of the two.
  **The same aliasing diagnosis as E36's row 55060, in a new place.**

**Pathway: bomb siting, not survival.** BOMB attractiveness in non-danger rows is identical across
arms (0.1810/0.1790/0.1789/0.1790), yet `PLB2` drops 2.26 *fewer* bombs and destroys 1.76 *more*
crates — crates per bomb **+0.130 [+0.061, +0.198]**. The score closes entirely on that route:
+0.224 coins + 5 × 0.011 kills = +0.279 of the +0.280. Survival moves +0.020, not significant.
**Sixth replication that survival does not convert into points on this board.**

**Why `PLB2` beats `PAR` is not settled.** The leading explanation is a learning-rate artifact:
α = 1/visits^0.7 per cell, so a 4-way split keeps α 2^0.7 = 1.62× higher than a 2-way one for the
whole run, and `PAR` is still climbing at 20 000. It is untested. **The cheapest test is a 2-way
`SHF`-style null arm**, and it belongs in any follow-up. Note also that `SHF` is not only "position
without the lattice" — it is spatially *white*, where `(x+y) mod 4` changes by ±1 along a
trajectory, so it nulls two things at once.

- **Verdict: BESSER for `PLB2` on `score`** — +0.20 by the robust estimators, CI excluding 0 under
  t, bootstrap, Wilcoxon and sign test, monotone across checkpoints, replicated in the independent
  training stream, and with a mechanism measured in the tables. **The first feature-map gain on
  rung 4, and the first positive result since E26.**
- **H1 holds · H2 refuted (but not for its stated reason) · H3 holds as written but is undefined at
  this power · P4 mixed · P5 null and underpowered.**
- **The continuation rule fires `CONTINUE`, and it should not be obeyed as written.** It fires on
  H1 ∧ (H2 ∨ H3), and the leg supplying it is H3 — the one that cannot distinguish its null from
  the full treatment effect. Any confirmation sweep must be sized from the **realised** SD of 0.358
  (n ≈ 30-40 for `score` at +0.20, and `won` is out of reach at any n this project can afford),
  must report **`PLB2` − `SHF`** as primary rather than `PLB2` − `ctl2`, and should add the 2-way
  null that tests the α explanation. **Otherwise rung 4 closes here on a positive result** — which
  is a better place to stop than the five negatives that preceded it.

---

## E36 — The map is blind exactly where it dies, and the fix costs zero rows

- **Question:** rung-4 optimisation stopped after E33, E34 and E35 all failed. An audit
  commissioned to **break** that decision (`scratchpad/audit7/`) broke it, and the hole is one I
  should have seen: **none of the five failed interventions changed the feature map.** E19, E32,
  E33 moved the reward; E34 the initial condition; E35 a price. The last feature change was E26,
  on rung 3, and it was worth **+1.595 score**.

  E28 ranked the one surviving feature candidate — opponent BFS distance — and deferred it
  explicitly: *"comes **after** coverage, never alone… every new digit makes coverage worse."*
  **That condition expired at E30 and nobody re-checked it.** Measured share of ε = 0 alive steps
  landing in an all-zero row: **0.026** on E28's solo-trained table, **0.0001** on the shipped
  rung-4 table (4 steps in 46 424). Coverage is solved.

- **The structural finding, verified directly in `callbacks.py` and by row count.** In
  `state_to_features`, when `own_danger > 0` the escape branch overwrites digit 6 **and pins
  digit 8 to `DIST_NONE`**:

  ```python
  if own_danger and ABLATE != "escape":
      target = escape_direction(x, y, field, danger, occupied)
      target_dist = DIST_NONE
  ```

  So digit 8 is 0 on **every** danger step — 43.9 % of all steps, and where essentially 100 % of
  deaths occur — and **40 960 of 64 000 rows (64.0 %) are structurally unreachable.** In the rows
  where the agent dies, the state carries **no opponent information at all**, and a whole digit
  sits idle.

- **The variable that is missing is opponent proximity, measured three ways** (audit 7, and the
  first two re-verified here):

  1. **Deaths.** At the last step before death, the nearest opponent is at BFS ≤ 2 in **93.3 %** of
     cases, against a base rate of 15.3 % — a **6.1× lift**. 94 % of those are *own-bomb* deaths.
  2. **The decisive row is a mixture, not a fixed point.** Row 55060 — the row audit 6's whole
     "second Bellman fixed point" diagnosis rested on, and which I built E34 on — takes 809 visits
     that split **148 / 354 / 301** across opponent bands, with **zero deaths in the 655 far
     visits** and 36.5 % fatality in the near ones (permutation p < 1e-4). Pooled over 41 rows:
     observed 39.81 against a null of 3.68 ± 0.79, p < 0.0005, and **58.5 % of the death mass sits
     in rows the digit separates at p < 0.05.**
  3. **It changes what the escape is worth.** Audit 7 re-ran the ceiling rule restricted to each
     half (n = 1000, validation seed, paired). Score gained per death avoided: **far 1.10, near
     0.28.** `far − ctl` score **+0.418 [+0.178, +0.660]**; `near − ctl` +0.102, not demonstrated.

  **This refines E33's headline negative rather than contradicting it.** E33 concluded "survival is
  not worth points on this board" — measured *conditionally*, survival bought in the far sub-state
  is worth four times what it is worth in the near one. **E33's conclusion was an average over an
  interaction the feature map hides**, which is exactly why paying for the escape everywhere (E33)
  and forcing it everywhere (E34) both installed the behaviour and lost the benefit.

  **So "a second Bellman fixed point" was the wrong diagnosis** — mine, carried from audit 6 and
  quoted as the headline of E34. The row is not one state Q-learning priced correctly at 5.1 %
  death; it is a 0 %-fatal sub-state glued to a 36 %-fatal one, and the fixed point is the average
  of two states that want different actions. **Aliasing, not a competing optimum.**

- **Change: one line, and it costs zero new rows.** `BM_OPPDIST` (default 0). When on, the
  `own_danger` branch sets `target_dist` to a bucketed opponent BFS distance
  {0 none/unreachable, 1 ≤ 2, 2 3-5, 3 ≥ 6} instead of `DIST_NONE`. **`FEATURE_SIZES` is
  untouched** — 64 000 rows either way — because the digit is already there and idle. The shipped
  table stays a valid **factor-1** warm-start parent; the 1 131 valued danger rows are broadcast
  into their four siblings, and the 820 danger rows holding stale pre-E20 `d8 > 0` values are
  overwritten, not preserved. Measured cost of the extra BFS: **0.028 ms mean, 0.220 ms max**
  against a 500 ms budget.

  **This is precisely the move that won rung 3.** E26 gave an idle digit a meaning and bought
  +1.595. It also answers the survey's "reduction beats richness" caution head-on: information
  added at **zero** state-space cost.

- **Design.** 3 arms × 5 seeds, `BM_RUN_INDEX` **100-104** (paired at run level with the E33
  control; `--seed 810731`), 20 000 episodes, checkpoints 5 000 / 10 000 / 20 000, E33 ctl config
  verbatim. `ctl` is E33 ctl, already measured. **`PLB` is a placebo carrying an information-free
  bucket of matched marginals** (`(x+y) mod 4`) — not optional here, because filling 40 960
  previously-dead rows is itself a perturbation.
- **Measurement:** 1000 rounds, ε = 0, validation seed 550731, n = 5, reported at @20 000.

### Prediction (written before the run)

1. **P1, primary.** `OPP`'s `won` at 20 000 beats the control's **0.372**, paired t-CI over the
   five runs excluding 0. **Magnitude 0.395-0.430** (score 3.95-4.25), derived from the far-rule's
   +0.418 score / +0.040 `won` measured on a table that never trained with the split.
   **Refutation:** CI includes 0 → the conditional policy is not learnable even once it is
   *expressible*, the representation premise is then properly tested for the first time, and the
   stop is justified on evidence rather than on an untested inference.
2. **P2, mechanism.** Escape-follow rate measured **separately in the near and far sub-row
   families** (control: 0.634 in both, by construction). **Prediction: follow(far) − follow(near)
   ≥ 0.15.** **Refutation:** |difference| < 0.05 → the split is unused and P1, if positive, is
   something else.
3. **P3, coverage guard — the condition that deferred this feature in the first place.** Share of
   ε = 0 alive steps in an all-zero row stays **< 0.01** (shipped 0.0001; E28's solo table 0.026).
   **Refutation:** ≥ 0.01 → the information was bought at a coverage cost and E28/E29's warning
   applies after all.
4. **P4, guards.** `crates` ≥ **31.0** — the measured failure mode of every escape intervention so
   far (E33 30.81, E34 29.86). `think_max_ms` < 5 ms.
5. **P5, placebo, declared now.** If `PLB` moves `won` by more than **half** of `OPP`'s move, the
   entry is **inconclusive by construction**.

**Pre-committed magnitude:** `OPP` `won` **0.395-0.430**, follow-rate split ≥ 0.15, crates ≥ 31.0.
**Audit 7 flags the weak point in its own evidence and I am carrying it forward rather than hiding
it:** `far − near` is established on `score` (+0.316 [+0.079, +0.545]) but only **at the boundary on
`won` (+0.040 [−0.002, +0.082])**. `won` is still the primary, because switching to `score` now —
the metric the effect is strongest on — would be choosing the measure after seeing which one moved.

**Cost: 10 new runs, ~2.5 h, one of roughly 30 sweeps the remaining five weeks allow.**

---

### Result — 1000 rounds at validation seed 550731, n = 5 runs, paired to the E33 control

| @20 000 | score | **won** | crates | suicides | survived | kills |
|---|---|---|---|---|---|---|
| **ctl** | 3.719 ± 0.055 | 0.372 ± 0.009 | 32.13 | 0.616 | 0.333 | 0.221 |
| **OPP** | 3.726 ± 0.087 | 0.365 ± 0.024 | 32.10 | **0.422** | **0.525** | 0.220 |
| **PLB** (placebo) | 3.930 ± 0.177 | 0.385 ± 0.020 | **33.11** | 0.516 | 0.435 | 0.236 |

**P5 FAILS, and it is read first by design. The placebo moved more than the treatment.**
`won`: OPP **−0.0068**, PLB **+0.0126** — ratio **1.853** against a 0.5 threshold. **E36 is
INCONCLUSIVE BY CONSTRUCTION**, as pre-registered. Paired, PLB is the only arm with a significant
effect anywhere: **crates +0.980 [+0.340, +1.620]**, score +0.211 [−0.018, +0.440], `won` +0.013
[−0.010, +0.036].

**P1 REFUTED independently.** OPP `won` −0.007 [−0.031, +0.017].

**P2 REFUTED.** Follow-rate split between the near and far sub-families: **+0.029**, needing
≥ 0.15. **P3 PASSES** (all-zero-row share 0.00023 < 0.01, so the information cost no coverage —
E28/E29's deferral condition was correctly judged expired). **P4 PASSES** (crates 32.10 and 33.11,
both ≥ 31.0).

**My placebo was not a placebo, and that is the entry's real content.** `(x + y) mod 4` is
**position parity**, which on this board encodes the wall lattice — pillars sit at (even, even).
`scratchpad/survey/REPORT.md` records it as a *real feature* used deliberately by another project
("position parity (x,y even/odd → 3 cases), cheap encoding of the wall lattice"). I chose, as a
control, a feature the survey I commissioned had already catalogued as useful. **So E36 did not
test "does new information help?" against a null — it ran two features against each other, and the
one I labelled the control won.**

**The one number worth keeping:** giving the danger rows *structural* information (where the walls
are) is worth **+0.98 crates**, and giving them *opponent* information is worth nothing. That
inverts audit 7's ranking, which put opponent proximity first on the strength of a 6.1× death lift
— a lift that is real (verified) and still did not convert.

**And the agent did use the digit, just not on the predicted axis.** Follow rate by bucket:
**0.826** where no opponent is reachable, ~0.65 everywhere else. It conditions on "is anyone
around at all", not on how far away they are.

**Fourth replication of E33's finding.** OPP cut suicides 0.616 → **0.422** and raised survival
0.333 → **0.525** with `won` unchanged. E30 (passive), E31 (reckless), E33 (shaped safe), E34
(forced safe) and now E36 all agree: **on this board, survival does not convert into points.**

**Tool defect found and not yet fixed.** `scratchpad/deaths/collect.py` *reimplements*
`state_to_features` (lines 110-120) instead of calling it, and hardcodes `target_dist = DIST_NONE`
in the danger branch. Its docstring claims the opposite ("digits are computed by it, not by us"),
and that was true for every entry before this one — the reimplementation matched exactly until
E36 changed the danger branch. My first P2/P3 measurement was taken on pre-E36 rows and was void;
the numbers above come from a rollout through the agent's real `state_to_features`. **Every
forensic result from E28 on is unaffected, but the file must call the function it documents.**

### Correction, 2026-08-16 (audit 8) — three defects, two of them mine

Audit 8 reproduced **every number in the table above to four decimals**. The arithmetic is right;
the design and the verdict label are not. All three findings verified independently before being
written here.

**1. The arms and the control had different warm-start parents. E36 is not "E33 ctl verbatim".**
From `hyperparams.warm` across all 55 rung-4 runs:

| runs | warm parent | valued rows |
|---|---|---|
| E30 `T`, E31 `S0`, E33 all four arms, E35 all three — **45 runs** | `_rung2ship` | **2 364** |
| **E36 `OPP`, `PLB` — 10 runs** | **`_e36parent`** | **7 879** |

**E36 is the only rung-4 entry whose arms do not share the control's initial condition** — in
precisely the dimension E34 was about. My launcher did it and my entry claimed the opposite. It was
also avoidable: `_rung2ship` already carries 1 025 valued danger rows, so a matched parent could
have been broadcast from *it*.

Consequences, which differ per comparison:
- **P1's refutation is unharmed and is conservative** — `OPP` had the *richer* parent and still
  returned `won` −0.007. It stands as written.
- **PLB − ctl is confounded.** The +0.980 crates contains a parent effect of unknown size.
- **PLB − OPP is clean** — same parent, same 4-way split of the same idle digit, same run indices,
  same arenas. It is the only uncontaminated contrast E36 contains, and I never computed it:
  **score +0.203 [+0.046, +0.360], coins +0.122 [+0.034, +0.210], 5/5 seeds.**

**2. P5 was UNDEFINED, not failed — so "inconclusive by construction" is the wrong verdict.** The
ratio 0.0126 / 0.0068 = 1.853 is arithmetically right, but the denominator is `OPP`'s `won` move of
**−0.0068 with CI [−0.031, +0.017]** — indistinguishable from zero. **A ratio against a zero
denominator is undefined**, and the gate would have fired for almost any placebo value including
zero. A placebo test is informative only *conditional on the treatment having moved*, and P1 was
already refuted on its own, so P5 carried no information. **Corrected scoring: P1 refuted, P2
refuted, P3 and P4 passed, P5 undefined** — and PLB is an *exploratory positive* in a
pre-registered arm on a pre-registered metric (`crates` was P4's guard). The label I chose demoted
the one result in the entry that pointed forward, and I had an incentive to stop.

**3. The entry was underpowered on its own primary metric.** Paired SD of `won` differences across
nine arm × checkpoint sets is **0.0231**, so the 80 %-power minimum detectable effect is:

| n | MDE on `won` |
|---|---|
| **5** | **0.0446** |
| 10 | 0.0267 |
| 20 | 0.0177 |

**E36 pre-registered `won` 0.395-0.430, i.e. +0.023 to +0.058 — the lower two-thirds of its own
target range sat below its detection threshold.** E35's +0.018 to +0.058 has the same problem. And
the conversion rate is measurable: over 45 run-level points, `won ≈ −0.056 + 0.113 × score`
(r = 0.76), so **+0.211 score predicts +0.024 `won`** — 54 % of the n = 5 MDE.

**So "five pre-registered `won` negatives" partly measures the design rather than the
interventions.** That is a methods finding that recontextualises E33, E34, E35 and E36 at once, and
it belongs in the report ahead of any of them. **A rung-4 experiment expecting a moderate effect
needs n ≈ 10-20 seeds, not 5.**

**Two smaller corrections.** The placebo is described as having "matched marginals" — it is matched
in *arity* only (danger-step marginals PLB {0.316, 0.175, 0.325, 0.184} vs OPP {0.162, 0.197,
0.380, 0.261}). And the reading of P2's bucket 0 as *"it conditions on whether anyone is around at
all"* is about a quarter right: of the 16.2 % of danger steps in bucket 0, **73.4 % have opponents
alive but BFS-unreachable** behind crates, and only 26.6 % have none left. Bucket 0 is mostly a
statement about **board enclosure** — which points the same way as everything else in this entry.

**What audit 8 could not break.** PLB's crates effect survives every robustness test available:
5/5 seeds positive (min +0.237, paired p = 0.013), coherent secondaries (`coins` +0.137 p = 0.025,
`score` +0.211, and the arithmetic closes — 0.137 + 5 × 0.015 = 0.212), **monotone across
checkpoints** (+0.093 → +0.608 → +0.980, with the missing ctl@10 000 measured by the audit), and it
**replicates in an independent stream** — `CRATE_DESTROYED` in the training logs builds monotonically
and plateaus at **+0.94** against the evaluation's +0.98. The strongest case against it is also
recorded: **`OPP`'s own crates was +0.710 [+0.231, +1.190] at 10 000, 5/5 seeds, and −0.030 by
20 000** — a 5/5 CI-excluding-zero crates result in this exact design has evaporated once at n = 5.
What separates them is that PLB's is monotone where OPP's was a bump, and the training stream
discriminates them (PLB builds and holds, OPP decays to +0.02). Under a full 66-test Bonferroni
family PLB's crates does **not** survive (p ≈ 0.87); "post-hoc" is fair for the metric selection,
"likely false positive" is not supported.

- **Verdict: E36 is INCONCLUSIVE by its own pre-registered placebo rule, with P1 and P2 refuted
  and P3 and P4 passed.** Audit 7's premise — that no feature-map change had ever been tested on
  rung 4 — was correct and is now tested. The specific feature it ranked first does not convert.
- **What survives for the report.** Coverage does not bind (P3), so a feature *can* be added at
  zero state cost; opponent distance is not the feature; and a structural feature accidentally
  beat it. The eight-digit map is still the binding constraint, but the missing information is
  about **the board**, not about **the opponents** — which is the opposite of what five entries
  of death forensics implied.
- **Next, and the honest framing:** PLB's +0.98 crates is a **post-hoc** finding from an arm that
  existed to be a null. Chasing it directly is how a project talks itself into a false positive.
  If it is run, it must be as a *pre-registered* arm with a genuinely information-free control
  (a fixed random relabelling of the danger rows, not a function of position), and the stopping
  rule stands: this is one experiment, not a new programme.

---

## E35 — Thirty per cent of our score comes from an event priced at zero

- **Question:** `score = coins + 5·kills`, exactly (2.613 + 5 × 0.221 = 3.719). **Kills are 30 % of
  our score and `KILLED_OPPONENT` has been priced `0.0` in every rung-4 run** — an uncorrected
  script error of mine since E30, visible in every `.meta.json`. `train.py:234-236` argues in a
  comment that it should not be, and nobody acted.

  It has also never been tested under a *healthy* configuration. E26 arm H used `BM_KILL=25` with
  `STEP_COST=−0.1` and `CRATE=0.3` — precisely the setting E27 measured at earn/cost 0.70 and E31
  showed ratchets the argmax. `STEP_COST` has been 0 since E31.

  And the headroom is where kills are, not where coins are. `COIN_COUNT = 9` (`settings.py:28`),
  so the fair share is 2.25 and **we already take 2.613** — coin headroom is thin. Meanwhile
  **1.847 opponent deaths per round occur in our field and we are credited with 0.221 (12 %)**;
  ~82 % are their own suicides, so the pool is not free, but it is the only channel with 5×
  leverage. The ceiling arm reached its +0.68 as ⅓ coins and **⅔ kills**.

- **Change:** `BM_KILL`, and in one arm `BM_COIN`. Nothing else.

- **Design.** 5 seeds, `BM_RUN_INDEX` **100-104** — the same indices as the E33 control, so with
  `--seed 810731` the arenas and the exploration stream match and the comparison is **paired at the
  run level**. 20 000 episodes, checkpoints 5 000 / 10 000 / 20 000, otherwise E33 ctl verbatim.
  Control is E33 ctl, already measured (score 3.719, `won` 0.372, kills 0.221).

  | arm | change | rationale |
  |---|---|---|
  | **K5** | `BM_KILL=5` | one coin's worth at this table's scale |
  | **K25** | `BM_KILL=25` | the game's own 5:1 kill:coin ratio, given `BM_COIN=5` |
  | **PLB** | `BM_COIN` 5 → 7, `BM_KILL=0` | **placebo** — comparable Q-magnitude, no kill information |

  **The placebo is E32's carried-forward requirement #4, never honoured.** Without it, "pricing
  kills" and "any perturbation of a few Q-units" are not separable by this design, and E32 was
  withdrawn partly for that. **The middle dose is deliberately omitted** — E32's other instruction,
  after its middle dose was the one that collapsed.

- **Measurement:** 1000 rounds, ε = 0, `BM_TIE_TOL=0.0`, validation seed 550731, n = 5 runs,
  reported at @20 000.

### Prediction (written before the run)

1. **P1, primary.** K25's `won` at 20 000 beats the control's **0.372**, t-CI over the five runs
   excluding 0. **Magnitude: K25 0.39-0.43, K5 0.375-0.40.** `won` stays the primary because
   `MEASUREMENT.md` makes it the rung-4 ranking metric and E30-E34 all used it; switching to
   `score` now — the metric kills feed most directly — would be choosing the measure after knowing
   the mechanism.
2. **P2, mechanism.** `kills` is monotone in `BM_KILL`, **K25 in 0.30-0.42** (from 0.221). The
   bound is not arbitrary: 1.847 opponent deaths/round exist, we take 12 %, and ~82 % are their own
   suicides, so cheap conversion tops out near 0.35-0.45, nowhere near 1.8. **Refutation:** neither
   arm raises `kills` by ≥ 0.05 with a CI excluding 0 → the agent cannot convert a kill price into
   kills on these digits — digit 7 shares one bit between "a bomb here opens a crate" and "a bomb
   here catches an opponent" — and the answer is a feature, not a price.
3. **P3, the placebo, which is what makes this an experiment.** If PLB moves `won` by more than
   **half** of K25's move, the design cannot attribute the gain and **the entry is inconclusive by
   construction** — declared now, not after seeing it.
4. **P4, guard.** `crates` ≥ **30.0** (control 32.13) and `won` ≥ 0.34. The predicted failure mode
   is bomb-chasing: an agent paid 25 for a kill may abandon crates to hunt. **`suicides` is not a
   guard** — E33 falsified it by intervention and E30's P5 / E31's P4 are both retired.
5. **P5.** `think_max_ms` unchanged (~0.5 ms against the 500 ms limit), recorded because
   `AGENTS.md` says always watch it.

**Pre-committed magnitude.** K25 `won` **0.39-0.43** with `kills` 0.30-0.42; K5 `won`
**0.375-0.40**; PLB within noise of the control. **If K25 beats the control but PLB moves nearly as
much, I will report the entry as inconclusive** rather than claim the kill price — that is the
whole point of including it.

**Stopping rule, stated before the result.** E33 and E34 both failed their primaries, and the
escape-ceiling line is closed. **If E35 also fails P1, rung-4 optimisation stops** — 3.719 / 0.372
against a measured 0.283 bar already beats the reference, and five pre-registered negatives plus
the second-fixed-point diagnosis is a better Experiments chapter than a sixth attempt.

---

### Result — 1000 rounds at validation seed 550731, n = 5 runs, paired to the E33 control

| arm @20 000 | score | **won** | **kills** | coins | crates | suicides |
|---|---|---|---|---|---|---|
| **ctl** (E33) | 3.719 ± 0.055 | 0.372 ± 0.009 | 0.221 ± 0.016 | 2.61 | 32.13 | 0.616 |
| **K5** (`BM_KILL=5`) | 3.707 ± 0.179 | 0.367 ± 0.045 | 0.216 ± 0.023 | 2.63 | 32.20 | 0.653 |
| **K25** (`BM_KILL=25`) | 3.713 ± 0.109 | 0.359 ± 0.024 | 0.219 ± 0.014 | 2.62 | 32.17 | 0.651 |
| **PLB** (placebo) | 3.697 ± 0.087 | 0.368 ± 0.021 | 0.218 ± 0.013 | 2.61 | 32.06 | 0.702 |

Paired against the control, every arm on every metric: **`won` −0.005 / −0.013 / −0.004,
`score` −0.012 / −0.006 / −0.021, `kills` −0.005 / −0.002 / −0.003.** Not one CI excludes 0.
**Nothing moved.**

**P3 PASSES, and it is read first by design.** The placebo moved `won` by −0.0044 against K25's
−0.0130 — a ratio of **0.338**, inside the 0.5 threshold. So the design is *not* inconclusive and
the rest of the numbers may be interpreted. (Reading P1 first and the placebo second is how one
argues past a placebo; the order was fixed in advance.)

**P1 REFUTED.** K25's `won` is −0.013 [−0.044, +0.018].

**P2 REFUTED — and this is the entry's content.** `kills` moved by −0.002. **Paying 25 per kill,
five times the price of a coin, changed kill behaviour by nothing.**

**The reward demonstrably reached the learner**, so this is not a plumbing failure. Training reward
over the last 500 episodes: **ctl 26.82 → K25 31.31**, of which 3.59 is kill income. **A 17 %
increase in total reward, concentrated entirely on kills, produced no change in the policy.**

**The mechanism is the one P2's refutation clause named in advance.** The only action that can
produce a kill is `BOMB`, and **digit 7 is a single bit that fires for "a bomb here opens a crate"
*or* "a bomb here catches an opponent"** (the E26 HUNT change merged them). So the agent has no
representation in which "bomb for a kill" and "bomb for a crate" are different decisions. **The
price cannot be paid to a decision the state cannot express.**

> **Correction, 2026-08-16 (audit 7).** I first wrote that digit-7 rows "are dominated ~145:1 by
> crate opportunities (32.2 crates against 0.22 kills per round)". **That ratio is computed from
> *outcomes*, and it is the wrong quantity** — what determines whether a price can change a policy
> is how often the *state* occurs. Measured over 46 424 instrumented steps: of the 16.0 % of steps
> where digit 7 = 1, **58.5 % are crate-only and 35.9 % are opponent-only — 1.63 : 1, not 145 : 1**
> (verified independently). The conflation is real and the refutation of P2 stands; the reason I
> gave for it being unfixable was wrong by ~90×, and it made a splittable digit look hopeless.
> That is E37 in the queue, and it is E35's own refutation clause taken literally: *"the answer is
> a feature, not a price."*

- **Verdict: E35 FAILED its primary, with the placebo clean.** Not a failure of the reward — a
  failure of the *representation* to carry the distinction the reward is about.
- **The stopping rule fired, and was then reversed on evidence — see E36.** E33, E34 and E35 all
  failed their primaries, so rung-4 optimisation stopped as pre-registered. An audit commissioned
  to *break* that decision (`scratchpad/audit7/`) broke it: **not one of the five failed
  interventions changed the feature map**, and the condition that deferred the one ranked feature
  candidate (E28's coverage gate) expired at E30 and was never re-checked. The stop rested on an
  untested premise. **The shipped agent stands at score 3.719 / `won` 0.372** against a measured
  symmetric bar of 0.283 and `rule_based_agent`'s 3.254 / 0.286, and remains what ships unless E36
  beats it.
- **What the three negatives establish together**, which is the Experiments chapter rather than a
  footnote: the +0.68 escape ceiling is unreachable by reward (E19, E32, E33) *or* by initial
  conditions (E34), because the table has converged to a second Bellman fixed point that unforced
  learning independently re-derives (26.3 % agreement in danger rows learned from scratch); and the
  one large aligned reward channel left, kills, cannot be spent because one bit of the feature map
  conflates it with crates. **Both remaining routes are blocked by the eight-digit representation,
  not by the learning rule or the reward.**

---

## E34 — Is the ceiling a *reachable* fixed point?

- **Question:** four attacks on the escape gap have failed — E19 (potential-based shaping, −9.17
  crates), E32 (death price; saturates, middle dose collapses), E33 (action-conditioned shaping;
  follow rate moved 0.634 → 0.872 exactly as designed and `won` **fell**). Audit 6 explains all
  four at once, and the diagnosis is the entry's real content:

  **The table sits at its own Bellman fixed point on the decisive row.** Measured over 300 rounds
  of E31 s80 @20 000:

  | row, action | transitions | terminal | Q | empirical 1-step target under π | residual |
  |---|---|---|---|---|---|
  | **55060, DOWN** — the fatal one; digit 6 says UP | 1 457 | 74 | 7.867 | 7.780 | **−0.087** |
  | 59160, DOWN | 1 486 | 0 | 8.261 | 8.277 | +0.016 |
  | 35032, UP | 279 | 0 | 8.868 | 8.466 | −0.402 |

  Death fires on 74/1457 = **5.1 %** of visits and the row prices that correctly at ≈ −0.2. `DOWN`
  still wins. **Q-learning has not failed to converge — it converged to a worse policy.**

  **And a second fixed point demonstrably exists at these hyperparameters.** This is audit 5's own
  datum, whose significance I missed at the time: identical configuration, greedy-`DOWN` in row
  55060 on E31's five seeds and greedy-`UP` on two replications, **zero overlap** in suicides
  (0.587-0.747 against 0.500/0.533), with indistinguishable training curves.

  **A reward change moves *where* the fixed points are; it cannot move *which one you land in*.
  Only an initial-condition change can.** That is why every reward-side instrument failed, and it
  is the one class of instrument never tried.

- **Change:** the **initial condition only**. Each run warm-starts from its own E33 control table
  with the argmax re-pointed to digit 6 in every row where digit 5 > 0 and digit 6 ≠ 0 — the exact
  construction that measured the ceiling (`scratchpad/benedict/force_escape.py`, a parameterised
  version of `scratchpad/audit5/mkesc.py`). Then it trains the **E33 control configuration
  verbatim**: same rewards, γ, ε schedule, `BM_STEP_COST=0`, `BM_CRATE=1.0`, `--seed 810731`,
  20 000 episodes. **The objective is untouched.** The rule runs once, offline, before training —
  nothing rule-like runs at inference, and the artifact remains a Q-table read by `argmax`.

- **Design.** 5 runs, `BM_RUN_INDEX` **100-104** — deliberately the *same* indices as the E33
  control, because each E34 run starts from *that same run's* forced table. With `--seed 810731`
  every run shares one arena sequence and the exploration stream is matched, so the comparison is
  **paired at the run level** rather than arm-mean against arm-mean.

  Checkpoints **500 / 2 000 / 5 000 / 10 000 / 20 000**. The early ones matter more than usual:
  **the decay curve is the measurement**, not the endpoint.
- **Measurement:** 1000 rounds, ε = 0, `BM_TIE_TOL=0.0`, validation seed 550731, n = 5 runs.
  Control is E33 ctl, already measured at score 3.719 / `won` 0.372. Reference ceiling (the rule,
  untrained): score 4.399 / `won` 0.442.

### Prediction (written before the run)

1. **P1, primary.** `won` at 20 000 beats the control's **0.372**, t-CI over the five runs
   excluding 0. **Magnitude 0.390-0.425** — retaining 25-75 % of the ceiling's +0.070.
   **Refutation:** CI includes 0 → the forced policy is not a fixed point, the operator drains
   back to where the control sits, and **the +0.68 is unreachable by value learning on these eight
   digits.** That closes the line, and it is the most valuable negative still available here.
2. **P2, mechanism.** Escape-follow rate `P(greedy = digit 6 | digit 5 > 0)` per checkpoint. It
   starts at **1.000** by construction; the control is **0.634**. **Prediction: monotone decay
   settling in [0.70, 0.85].** **Refutation:** within 0.02 of 0.634 by 20 000 → the control's fixed
   point is uniquely attracting, and together with a failed P1 the line is finished.
3. **P3, the interpretation split, fixed now so it cannot be chosen afterwards.** Let *f* = follow
   rate at 20 000 and *m* = fraction of re-pointed rows whose argmax training moved back off
   digit 6.
   - *f* ∈ [0.70, 0.90] **and** *m* ≥ 0.30 → **the agent learned which re-pointings to keep.**
     Report as learned, with the initialisation declared, and cite it against the survey's
     precedents (E's pretraining curriculum, K's imitation scaffolding).
   - *f* > 0.97 **and** *m* < 0.05 → **training did nothing and the rule shipped by hand.** Report
     it as a hand-initialised policy **and do not ship it** — that is the `AGENTS.md` prohibition
     on a feature returning the best action, wearing a `.npy` extension.
   - Between: mixed, and both halves reported.

   This is E33's P3 done properly. That one presupposed a sign and became inapplicable; this one is
   evaluable under every outcome, including P1's refutation.
4. **P4, guard, with evidence behind it.** `crates` ≥ **31.0** (control 32.13). E33 measured the
   real failure mode of an escape intervention *during* training: the agent stops bombing
   (32.13 → 30.81 at follow rate 0.872). **The suicide guard is not reinstated** — E33 falsified it
   by intervention, and E30's P5 and E31's P4 are both retired as mis-specified.
5. **P5.** Suicides at 20 000 in **[0.25, 0.50]** (ceiling 0.292, control 0.616). If P1 passes with
   suicides *high*, the mechanism is not the claimed one and the entry is inconclusive whatever the
   score does.

**Named confound, declared in advance.** `warm_start` sets `visits = WARM_N = 100` for every row
carrying value, so the forced cells begin at α = 1/100^0.7 ≈ 0.04. Those values are *fabricated*
(max + 1.0), unlike every previous warm start where they were learned — so a pseudo-count of 100
asserts a confidence we do not have, and it biases P3 toward the "training did nothing" branch.
**If P3 lands there, WARM_N is the first thing to vary before concluding anything**, not the last.
Left at the default here so the run is comparable to E33 ctl.

**Pre-committed magnitude:** `won` **0.390-0.425**, follow rate **0.70-0.85**, crates ≥ 31.0. I
expect a partial retention — the forced policy holds where it is genuinely better and drains where
it is not, which is the outcome P3's middle band describes.

---

### Result — 1000 rounds at validation seed 550731, n = 5 runs (100-104), paired to the E33 control

| E34 checkpoint | score | **won** | suicides | survived | crates |
|---|---|---|---|---|---|
| *forced table, untrained (the ceiling)* | *4.399* | *0.442* | *0.292* | *0.660* | *34.15* |
| ep 500 | 3.936 ± 0.104 | 0.392 ± 0.013 | 0.406 | 0.552 | 32.27 |
| ep 2 000 | 3.920 ± 0.148 | 0.390 ± 0.024 | 0.436 | 0.525 | 32.14 |
| ep 5 000 | 3.830 ± 0.173 | 0.376 ± 0.020 | 0.419 | 0.533 | 32.15 |
| ep 10 000 | 3.838 ± 0.217 | 0.376 ± 0.045 | 0.404 | 0.556 | 32.05 |
| **ep 20 000 (primary)** | **3.564 ± 0.150** | **0.350 ± 0.025** | 0.380 | 0.583 | 29.86 |
| E33 control @20 000 | 3.719 ± 0.055 | 0.372 ± 0.009 | 0.616 | 0.333 | 32.13 |

Paired against the control: ep500 **won +0.020 [+0.003, +0.038]**, score **+0.217 [+0.060, +0.375]**;
ep2000 score +0.201 [+0.008, +0.395]; ep5000 and ep10000 not demonstrated;
**ep20000 won −0.022 [−0.049, +0.005], score −0.155 [−0.252, −0.058]**.

**P1 REFUTED at the pre-registered checkpoint.** The forced policy is worth something early and
loses it monotonically: 4.399 → 3.936 → 3.830 → 3.564, ending *below* the control it started from.

**P2 REFUTED, and this is where the entry becomes interesting.** I predicted the follow rate would
decay into [0.70, 0.85]. It sits at **0.938 ± 0.003** across all five seeds. Decomposed (seed 100):

| valued danger rows at ep 20 000 | n | still following digit 6 |
|---|---|---|
| re-pointed by the transform | 712 | **0.962** |
| already agreed in the base | 845 | 0.992 |
| **new rows, learned during E34** | 95 | **0.263** |

So the drop from 1.000 is almost entirely the 95 newly-learned rows, **not reversion**:
`m` = **0.030**, i.e. **97 % of the forced re-pointings survived 20 000 episodes.**

**Therefore my own refutation clause — "the operator drains back" — is wrong in detail, and the
detail is the finding.** The forced *policy* is preserved; what drains is the *benefit*. Bombs
placed are flat (31.92 → 32.10) while crates fall 32.27 → 29.86, so crates per bomb goes
**1.011 → 0.930**. **Training keeps the escape behaviour and degrades the bombing around it** —
the same signature E33 produced by reward, reached here by initialisation instead. Two instruments
of opposite kinds, one outcome: making the agent escape reliably makes it bomb worse.

**P3: the "rule shipped by hand" branch, and it decides the shipping question against us.** With
`f` = 0.938 and `m` = 0.030, the letter of the split is the middle band (`f` misses the > 0.97
threshold). The substance is not ambiguous: training moved 3 % of the forced rows. **At ep500 —
the only checkpoint that beats the control — `f` = 0.992 and `m` = 0.017, firmly inside the
don't-ship band.** So the one arm with a real advantage is the hand-written rule with 500 episodes
of polish, and the pre-registered rule says do not ship it. **Recorded and honoured.**

**P4 FAILED** at ep20000 (crates 29.86 < 31.0) — and passed at every earlier checkpoint, which is
the same story as P1. **P5 passes** (suicides 0.380, inside [0.25, 0.50]).

**The named confound is now the live question, exactly as pre-registered.** `WARM_N = 100` gives
the forced cells α ≈ 0.04 on fabricated values, so `m` ≈ 0 is partly guaranteed by construction. A
`WARM_N = 1` replication separates "the forced policy is a genuine second fixed point" from "α was
too small to move it". **But note this cuts against reachability either way:** if α was the reason
the argmax held, then a table that *can* move will move — toward the control's answer, since that
is what unforced learning produces (26.3 % agreement in the 95 rows learned from scratch).

**That 26.3 % is the most transferable number in the entry.** The disagreement with digit 6 is not
a stale artefact of the rung-2 warm start: **when rung-4 training meets a danger row with no prior,
it chooses against the escape direction three times in four.** The table's answer is a genuine
second optimum, arrived at independently, which is why four reward interventions (E19, E32, E33)
and now one initialisation intervention have all failed to hold the first one.

- **Verdict: E34 FAILED its primary.** Taken with E33, **the +0.68 ceiling is not reachable by
  value learning on these eight digits** — not by reward (E19, E32, E33) and not by initial
  conditions (E34). The line is closed, which is what P1's refutation clause pre-committed it to
  mean.
- **Next:** E35 (price `KILLED_OPPONENT`, still 0.0 since E30, with a placebo arm). **And the
  stopping rule stands: if E35 also fails its primary, stop optimising rung 4.** 3.719 / 0.372
  against a measured 0.283 bar already beats the reference, and five pre-registered negatives plus
  the fixed-point diagnosis is a better Experiments chapter than a sixth attempt.

---

## E33 — How much of the ceiling is tie-breaking?

- **Question:** audit 5 measured a ceiling. Forcing the argmax to the table's **own** escape digit
  whenever `own_danger > 0` — 200 rows, 17.4 % of steps, zero training — takes score 3.827 →
  **4.432** and `won` 0.390 → **0.448**, with suicides 0.677 → 0.304. That is headroom sitting in
  information the agent already has and does not use, and **35 % of all deaths are one row**
  (55060) where the table prefers `DOWN` by **0.055** and `UP` — what digit 6 says — is the only
  survivor.

  The rule itself cannot ship: `AGENTS.md` forbids a feature that returns the best action. The
  learnable form is a dense event that pays the agent for agreeing with digit 6 while in danger.

  **Measured baseline (300 rounds, E31 s80 @20 000):** 43.7 % of all steps are danger steps, the
  escape direction is known in 99.2 % of them, and the agent already follows it **61.2 %** of the
  time. So this is not teaching it to escape — it escapes correctly three times in five. It is
  about the other two.

- **The design question is the level, and it is not a magnitude search.** The shaping differential
  is `2r`, so **r selects which decisions get overridden**:

  | r | flips gaps below | what that means |
  |---|---|---|
  | **0.05** | 0.10 | near-ties only — row 55060's 0.055 flips, everything else keeps its own answer |
  | **0.20** | 0.40 | modest disagreements |
  | **0.80** | 1.60 | ≈ the mean danger-row gap (1.59-1.78) — i.e. *the rule*, in reward form |

  **That makes the sweep a measurement rather than a tuning exercise**, and it is the honest way
  to handle the machine-learning objection: if r = 0.05 captures most of the gain, the table's own
  judgement was right and it needed a tie-break. If only r = 0.80 works, the rule is doing the
  work and we should say so in the report rather than claim the agent learned it.

- **Change:** two custom events, `FOLLOWED_ESCAPE` / `IGNORED_ESCAPE`, fired in `train.py` when
  digit 5 > 0 and digit 6 > 0, priced `+r` / `−r`. `callbacks.py` untouched; `BM_ESCAPE` defaults
  to 0, so unset this is E31 bit-for-bit.

  **This must be defended in the report, not slipped in.** It is shaping on a feature the agent
  already carries, and the table stays free to override it — the agent must still learn when to
  bomb, when to be in danger at all, and where to go when safe. But it is close to the line, and
  P3 below is what decides how the report has to describe it.

- **Design.** 4 arms × 5 seeds (`BM_RUN_INDEX` **100-104**, unused), 20 000 episodes, checkpoints
  5 000 / 10 000 / 20 000, `BM_STEP_COST=0` and `BM_CRATE=1.0` as E31.

  | arm | `BM_ESCAPE` |
  |---|---|
  | **ctl** | 0 — **contemporaneous**, not E31's numbers |
  | **F005** | 0.05 |
  | **F020** | 0.20 |
  | **F080** | 0.80 |

  **`--seed 810731` is passed to `main.py` for the first time.** Audit 5 found no training run in
  E30-E32 was ever arena-seeded, contrary to what those entries claim.
- **Measurement:** 1000 rounds, ε = 0, validation seed 550731, n = 5 runs, reported at @20 000.

### Prediction (written before the run)

1. **P1, primary.** F020 or F080 beats the **contemporaneous** control on `won` at 20 000, t-CI
   over the five runs excluding 0. **Refutation:** neither does → the ceiling is not reachable by
   shaping and the remaining gap is a feature problem (the 49.3 % "an opponent took my escape
   tile" category, which `NB_CLEAR` cannot express).
2. **P2, mechanism.** The follow rate `P(action = digit 6 | in danger)` rises monotonically with r
   from the measured **0.612**, and exceeds **0.85** at F080. **Refutation:** flat in r → the
   events are not reaching the decision and P1's result, if any, is something else.
3. **P3, the interpretation split, pre-registered so it cannot be chosen afterwards.** Let
   `G(r) = won(r) − won(ctl)`. **If `G(0.05) ≥ 0.5 · G(0.80)`, the gain is tie-breaking** and the
   report says the table's judgement was sound where it was confident. **If `G(0.05) < 0.25 ·
   G(0.80)`, the rule is doing the work** and the report says so plainly, including in the
   Experiments chapter's discussion of what the model actually learned. Between those, it is
   mixed and both get reported.
4. **P4, guard.** No arm exceeds the ceiling — `won` at 20 000 below **0.448** and score below
   **4.432**. Beating a rule that has perfect access to the same digit would mean the shaping is
   doing something other than what it says, and I would want to find out what before believing it.
5. **P5, and this is a free replication.** The contemporaneous control's `suicides` at 20 000
   lands somewhere in **[0.45, 0.75]**, spanning both modes audit 5 found (E31's five runs gave
   0.587-0.747; two replications gave 0.500/0.533). With n = 5 fresh arena-seeded runs this
   settles whether E31's 0.690 was a property of the reward table or of five unlucky runs —
   **which is a result either way**, and the reason the control is worth its five runs.

**Pre-committed magnitude.** F020 `won` **0.39-0.43**, F080 **0.41-0.45** (approaching the ceiling
from below), F005 **0.38-0.41**. Follow rate F005 ≈ 0.68, F020 ≈ 0.78, F080 ≈ 0.90. I expect P3 to
land in the **mixed** band, i.e. tie-breaking is worth something real but not most of it.

---

### Result — 1000 rounds at validation seed 550731, n = 5 runs (100-104), first arena-seeded sweep

| @20 000 | score | **won** | suicides | survived | crates | **follow rate** |
|---|---|---|---|---|---|---|
| **ctl** | 3.719 ± 0.055 | **0.372 ± 0.009** | 0.616 | 0.333 | 32.13 | 0.634 |
| F005 | 3.702 ± 0.069 | 0.368 ± 0.011 | 0.568 | 0.384 | 32.22 | 0.671 |
| F020 | 3.717 ± 0.155 | 0.368 ± 0.025 | 0.613 | 0.344 | 31.62 | 0.692 |
| F080 | 3.523 ± 0.707 | 0.344 ± 0.076 | **0.496** | **0.451** | **30.81** | **0.872** |
| *ceiling (rule, not an arm)* | *4.399 ± 0.060* | *0.442 ± 0.010* | *0.292* | *0.660* | *34.15* | *1.000* |

**P2 CONFIRMED.** The follow rate rises monotonically 0.634 → 0.872 and clears 0.85 at F080.
**The intervention did exactly what it was designed to do.**

**P1 REFUTED.** `G(0.05) = −0.004`, `G(0.20) = −0.004`, `G(0.80) = −0.028`. No arm beats the
contemporaneous control.

**P3 is INAPPLICABLE, not "mixed".** Its ratio test presupposes `G(0.80) > 0`; every `G` is ≤ 0,
so the split cannot be evaluated. Recorded as written rather than reinterpreted to fit a sign it
did not anticipate.

**P4 passes** (nothing near the ceiling). **P5:** the control's suicides are **0.616**, inside the
pre-registered [0.45, 0.75] and *between* E31's 0.690 and audit 5's 0.500/0.533 — so the
bimodality is bounded but not resolved. The control otherwise **replicates E31** (`won` 0.372 vs
0.378, score 3.719 vs 3.813, flat 5 k → 20 k), so E31's headline survives arena seeding.

### The result is a negative one, and it is worth more than the arm

**Suicides were manipulated, not observed, and `won` did not follow.** Suicides 0.616 → 0.496 and
survival 0.333 → 0.451 while `won` *fell* 0.372 → 0.344. Audit 5's R3 argued from a correlation
(`won ~ suicides` slope +0.038 [−0.054, +0.143]); this is the intervention. **Survival is not
worth points on this board**, now shown three ways: E30 went passive and lost, E31 went reckless
and won, E33 was made safe deliberately and gained nothing.

**So the guard that "failed" in E30 (P5) and E31 (P4) was measuring something that does not
matter.** Both should be read as mis-specified guards, not agent defects. `rule_based_agent`
suiciding 0.533 in its own field was the tell and I did not take it.

### Why the rule works and the shaping does not — and the mistake was mine

Re-running audit 5's ceiling on **E33's own control** reproduces it exactly: **+0.680 score
[+0.611, +0.749], +0.070 won [+0.057, +0.083]**, five seeds out of five, against audit 5's
+0.606/+0.058 on different tables. So the ceiling is real and is not seed-specific.

**The discriminating number is `crates`.** The rule *raises* them 32.13 → 34.15; the shaping
*lowers* them 32.13 → 30.81, at nearly the same follow rate. The rule is applied at evaluation to
a table trained without it, so the agent keeps its bombing policy intact and only its escapes
improve — it bombs identically and lives longer, so it clears more crates. The shaping is applied
during **training**, so the agent learns that following digit 6 pays and takes escape steps in
states where it should be bombing.

**`AGENTS.md` states the reason outright: "Potential-based shaping (Ng et al. 1999) depends on
*states*, not actions."** `FOLLOWED_ESCAPE`/`IGNORED_ESCAPE` is conditioned on the **action
taken**, so it is not potential-based and it therefore *changes the optimal policy* — that is the
theorem, not a side effect. The crate loss is precisely what it predicts. I designed the arm
anyway, and the audit that checked the design did not catch it either.

- **Verdict: E33 FAILED its primary and is a clean negative result** — the causal chain was
  verified end to end and the outcome did not move. Both `BM_ESCAPE` and the suicide guard are
  retired.
- **E34 is the correct instrument, and it is already half-built.** Potential-based shaping,
  Φ(s) = −distance to safety while `own_danger > 0`. Provably policy-invariant, so it cannot cost
  crates the way this did, and `train.py` already carries `BM_SHAPE` and a `phi()` that is
  documented as "deliberately the safe branch only" — the danger branch is exactly what is
  missing. **Falsifier:** if potential-based shaping also fails to close the gap to the ceiling,
  then the gap is not learnable from these eight digits and the answer is a feature (the 49.3 %
  "an opponent took my escape tile" category) rather than a reward.

---

## E32 — Withdrawn before it ran, and replaced

**The design below was written, verified by an audit (`scratchpad/audit5/`), and abandoned. It
was never run.** Kept because the reasons it was wrong are worth more than the entry would have
been, and because the audit found a ceiling in the process.

**Why it was withdrawn**, in the order that matters:

1. **The control does not replicate (audit 5, R0).** E31's `suicides` = 0.690 is bimodal on the
   argmax of a single row (55060); two independent re-runs of the identical configuration give
   0.500 / 0.533 with zero overlap. E32 planned to measure both arms against that number. See the
   correction block in E31.
2. **Both levels were past saturation (R1).** Measured flip threshold over 83 fatal cells: median
   Δ = **1.08**, p90 = 5.17. `KILLED_SELF` at −10 and at −25 flip the *same* 90.4 % of cells. P3
   was written to read "K30 ≈ K15 on suicides" as *non-response*; it would in fact have been
   *saturation*, so a likely outcome was pre-registered to be interpreted backwards.
3. **A probe ran the arms and the middle dose is catastrophic.** 3 arms × 2 seeds × 20 000
   episodes against a *contemporaneous* control: K15 score **−1.240 [−1.490, −0.990]**, `won`
   **−0.160 [−0.210, −0.110]**, collapsing on both seeds into bomb-spam-and-hide (crates/bomb
   1.02 → 0.71). K30 was roughly neutral. **I pre-committed to K15 as the good arm at `won`
   0.36-0.40; it measured 0.245**, tripping my own P2 refutation. The dose-response is
   non-monotone with the *middle* dose breaking, and `suicides` — the primary — ranks the
   catastrophic arm as the better one.
4. **The motivation does not hold (R3).** Across the 15 existing E31 evaluations (suicides
   0.49-0.74), `won ~ suicides` has slope **+0.038 [−0.054, +0.143]** and `score ~ suicides`
   −0.180 [−0.594, +0.181] — no relationship either way. Forcing the single-cell flip directly
   gives suicides −0.116 (significant) with score −0.124 and `won` −0.027 (both n.s.). And
   `rule_based_agent` suicides **0.533** in its own field. **A lower suicide rate is not
   demonstrably worth anything**; E30's P5 and E31's P4 may both be mis-specified guards rather
   than agent defects.
5. **My arithmetic was wrong twice.** A death does not cost 5: the terminal update supplies no
   bootstrap, so it forfeits `V ≈ 5.98` as well — ≈ **11**. And a suicide's terminal reward is
   **−4.33**, not −5, because `update_bombs()` credits the killing bomb's crates *before*
   `evaluate_explosions()` kills the agent — 3.7 % of suicides are net **positive**.
6. **One thing I got right, recorded because the audit expected otherwise.** It predicted the
   penalty would ratchet the argmax the way the step cost did. Measured visit-weighted gap at
   20 000: ctl 2.218, K15 2.391, K30 **2.734** — it *widens*. A death penalty is action-*dependent*,
   so audit 4's ratchet argument does not apply, exactly as P4 argued a priori.

### What the audit found instead — a measured ceiling

**35.4 % of all deaths are one `(row, action)` pair.** Row 55060: in danger, escape digit says
`UP`, the table prefers `DOWN` by **0.055** out of Q ≈ 7.9. Replaying the opponents' recorded
moves, `cf_k = 1` for all 74 and the unique surviving action is `UP` — **the direction digit 6
already reports.** The 5 k → 20 k regression is one near-tie flipping in an upstream row (59160,
greedy goes `RIGHT` → `DOWN` by 0.079 on 4/5 seeds), routing **17.8×** more traffic into 55060.

Forcing the argmax to equal the table's **own** escape digit whenever `own_danger > 0` — 200 rows,
17.4 % of steps, **zero training** — pooled over 3 seeds × 300 paired arenas:

| | E31 | argmax forced to digit 6 |
|---|---|---|
| score | 3.827 | **4.432** (+0.606 [+0.362, +0.847]) |
| won | 0.390 | **0.448** (+0.058 [+0.014, +0.100]) |
| suicides | 0.677 | **0.304** |
| survived | 0.274 | **0.647** |

Not shippable — it is a rule, and `AGENTS.md` forbids a feature that returns the best action — but
it **bounds what is available from the information the agent already has**, and it exceeds every
number E32 pre-committed to.

Two further measured facts for whatever comes next: **49.3 % of deaths are "chose a safe tile, an
opponent took it"**, which `NB_CLEAR` structurally cannot express; and **7.5 bombs per round are
placed with `bomb_useful = 0`** — 26 % waste, never costed.

### Carried forward

1. **A contemporaneous control and `--seed` in every launcher.** Nothing downstream is
   interpretable until the control is re-established; the arena RNG has never been set.
2. **The escape-follow decision is the lever, not the price.** A dense `FOLLOWED_ESCAPE` /
   `IGNORED_ESCAPE` event while `own_danger > 0` is the learnable form of the ceiling above; the
   visit-weighted gap to overcome is measured at **1.59-1.78**, which sets the sweep. **This needs
   an explicit argument in the report** — rewarding agreement with a hand-computed direction is
   shaping, not a policy feature, but it is close enough to the line that it must be defended
   rather than slipped in.
3. **One appended binary digit: "digit 6's target tile is adjacent to an opponent"** (64 000 →
   128 000 rows, `warm_start` valid at factor 2). The 49.3 % category, which no existing candidate
   digit encodes.
4. **If the price question is revisited:** drop the middle dose, sweep small (Δ ≈ 1-3), pair every
   arm with a contemporaneous control, and add a **placebo arm** (e.g. `BM_COIN` 5 → 7) —
   otherwise "price" and "any perturbation ≥ 1 Q-unit" are not separable by the design.

---

## E31 — The step cost is not a cost, it is a ratchet on the argmax

- **Question:** E30 peaks at 5 000 episodes and decays monotonically to 20 000. I proposed two
  causes and **an independent audit refuted both** (`scratchpad/audit4/`; it states it had not
  opened this ledger when it formed its diagnosis). What it found instead, and what I verified
  myself before writing this:

  **The action gap collapses while the value function survives.** Visit-weighted over the rows the
  greedy policy actually occupies, `E[Q(best) − Q(second)]` falls **1.788 → 0.668** from 5 000 to
  20 000, and the share of decisions made at a gap below 10⁻³ rises **0.001 → 0.204** (my own
  rollout; the audit got 1.81 → 0.67 and 0.001 → 0.206 independently).

  **It is invisible without visit weighting.** Unweighted over all 4 167 valued rows the mean gap
  *rises*, 2.589 → 2.763. I checked that first and would have "refuted" the audit with it. This is
  the third time this project has been bitten by reading a table instead of a rollout.

  Two rows make it concrete (verified directly from the checkpoints):

  | row 12786 — crate adjacent, bomb in hand, only `BOMB` is right | greedy | gap |
  |---|---|---|
  | `rung2ship` | BOMB | 1.13 |
  | T s60 @5 000 | BOMB | 0.95 |
  | T s60 @10 000 | BOMB | 0.35 |
  | T s60 @20 000 | BOMB | **6.5 × 10⁻⁴** |
  | T s63 @20 000 | **LEFT** | **2.1 × 10⁻⁵** |

  At 20 000 on seed 63 the agent **walks away from a crate it is standing next to with a bomb in
  hand**, by two parts in a hundred thousand. In row 12774 it walks away from its own BFS target
  by 1.6 × 10⁻⁴. And the six actions converge **to each other at ≈ 5.20, not to zero** — this is
  not value decay, it is the *ordering* dissolving.

  **The consequence is period-2 cycling.** `act()` breaks ties by exact float equality, so a
  10⁻⁴ gap is float noise, the mirrored row picks the mirrored action, and the agent paces. Steps
  inside a ≥ 6-step 2-cycle: **0.000 (parent) → 0.001 (5 k) → 0.088 (10 k) → 0.462 (20 k)**, and
  100 % of those steps have `digit 6 > 0` — it has a target the whole time and ignores it. Pacing
  is safe and earns nothing, which is precisely the reported signature.

  **Mechanism.** Because V is near-constant (`corr(Q, G) = 0.25`; `E[Q]` sits at 5.2 ± 0.5 while
  the true return swings 13 points across a round), `max Q(s′) ≈ Q(s,a)` under the greedy action,
  so the update reduces to `Q ← Q − α(0.1 + (1−γ)Q)`. **That is a constant negative drive applied
  only to whichever action is currently top** — it is pushed under the runner-up, which becomes
  greedy and is pushed down in turn, and the six entries ratchet together. At `STEP_COST = 0` the
  drive is `−α(1−γ)Q`, proportional rather than additive, and does not close the gap.

  This is E27's "an action-independent constant dominates the fixed point", localised: it damages
  the **action gap**, not the return.

- **Not the cause, each refuted with a number.** Death penalty: contributes **+0.5** of the −9.6
  fall, and the collapse is complete in rows where death is impossible. Tie tolerance: `BM_TIE_TOL`
  destroys the cycles at evaluation and recovers **no** score (2.69 → 2.26 → 1.69 → 0.94 as it
  rises), and training with it does not stop the gap collapse — **do not ship `BM_TIE_TOL > 0`**.
  Stale warm-start cells: 0.04 % of visit-weighted greedy actions. Truncation-as-termination in
  `end_of_round`: **real**, confirmed to three decimals by the reward residual (`reward −
  reconstruction` = −0.1 × survival rate exactly), but bounded at ≲ 0.2 in Q units against a
  1.1-point collapse, and the wrong sign. **That is the bug I was one turn from spending fifteen
  training runs on.**

- **Change:** `BM_STEP_COST=0`. One existing switch, nothing else touched.

- **Design.** 5 seeds (`BM_RUN_INDEX` 80-84 — checked against E27's pilot, which used 60), 20 000
  episodes, checkpoints 5 000 / 10 000 / 20 000, otherwise E30 arm T verbatim. Control is E30 arm T,
  already measured. Audit 4's own probe (2 fresh seeds, 10 000 episodes) gives the prior: gap
  −3 % against the control's −24 %, `P(gap < 0.05)` 0.001 against 0.18, no cycles, and **higher
  discounted return scored under the control's own step-cost-bearing objective** (14.43 vs 8.94).
- **Measurement:** 1000 rounds, ε = 0, `BM_TIE_TOL=0.0`, validation seed 550731, 3 ×
  `rule_based_agent`, n = 5 training runs.

### Prediction (written before the run)

Taken from audit 4's pre-registration, since it proposed the arm and had the prior.

1. **P1, mechanism, primary.** Visit-weighted `E[gap]` at 20 000 is within **10 %** of its value at
   5 000, and `P(gap < 0.05)` stays **below 0.05**. Control: −63 % and 0.436. **Refutation:** the
   gap still falls > 25 % → the step cost is not the levelling force and the diagnosis is wrong.
2. **P2, symptom.** Steps in a ≥ 6-step 2-cycle at 20 000 stay **below 0.05** (control 0.462).
   **Refutation:** above 0.15 → the cycles have another source.
3. **P3, outcome.** **The peak-then-decay disappears**: score at 20 000 ≥ score at 5 000. And the
   stronger form — score at 20 000 beats E30 arm T @5 000's 3.625. **Refutation:** it still decays
   → the mechanism is real but not what costs the points.
4. **P4, guard, the one I expect to bite.** Removing the step cost removes all shortest-path
   pressure. `steps`-to-target and `WAITED` must not blow up, and `suicides` must not exceed
   E30 @5 000's 0.636. If P3 passes but the agent dawdles, the answer is a **potential-based**
   substitute (`BM_SHAPE` already exists) — a state function cancels out of the action gap the same
   way but does not ratchet the argmax.

**Pre-committed magnitude.** I expect the decay to flatten and the peak to move later, not a new
record: score at 20 000 in the range **3.4-3.9**. Given how badly I have called the last two, this
is the audit's prior rather than mine, and P1 is the claim I actually believe.

### Result — 1000 rounds at validation seed 550731, n = 5 training runs (80-84)

| arm | score | **won** | suicides | crates | survived | steps |
|---|---|---|---|---|---|---|
| E30 ctl @5 000 | 3.625 ± 0.103 | 0.347 ± 0.014 | 0.636 | 31.65 | 0.277 | 249 |
| E30 ctl @10 000 | 3.323 ± 0.531 | 0.313 ± 0.053 | 0.551 | 29.80 | 0.391 | 270 |
| E30 ctl @20 000 | 2.338 ± 0.546 | 0.205 ± 0.073 | 0.413 | 21.80 | 0.540 | 299 |
| **E31 S0 @5 000** | 3.796 ± 0.095 | 0.365 ± 0.024 | 0.508 | 32.39 | 0.394 | 260 |
| **E31 S0 @10 000** | 3.710 ± 0.086 | 0.365 ± 0.014 | 0.642 | 32.13 | 0.302 | 259 |
| **E31 S0 @20 000** | **3.813 ± 0.114** | **0.378 ± 0.020** | **0.690** | 32.38 | 0.258 | 252 |

**P3 CONFIRMED in both forms, and it is the result of the entry.** Score is **flat** —
3.796 / 3.710 / 3.813 — against the control's 3.625 → 2.338. The peak-then-decay is gone, and the
stronger form holds too: 3.813 beats the control's best-ever 3.625. `won` at 20 000 is
**0.378 [0.358, 0.399]**, every seed ≥ 0.360, decisively clear of the measured 0.283 bar. Against
the correct reference (`rule_based_agent` in slot 0 of its own field, 3.254 / 0.286) that is
**+0.56 score and +0.09 won** — where E30 @10 000's honest out-of-sample edge was +0.04.

**The methodological gain is worth as much as the points.** Because the curve is flat, @20 000 is
simply *the end of training*, not a checkpoint hunted for after seeing the numbers. E30's headline
needed the post-hoc-selection caveat; this one does not.

**P2 CONFIRMED.** Period-2 pacing at 20 000: **0.002** against the control's 0.418 (my
alternating-action metric; audit 4's position-cycle metric gave the control 0.462, so the two
agree). Threshold was 0.05.

**P1 PARTIAL — neither confirmed nor refuted, exactly as its three zones were written.** Pass was
"within 10 %", refutation ">25 %"; the visit-weighted gap fell **17.4 %** (per-seed −14.8 to
−19.4 %, fixed row set) against the control's −42 to −45 %. Its second clause passes emphatically:
`P(gap < 0.05)` is **0.000-0.017** against 0.31-0.34, and that sub-0.05 tail is what produces the
pacing. **So the step cost is the dominant driver of the collapse but not the only one.** Note
also the absolute levels: E31's gap is **2.75 at 5 000 where the control's was 1.79** — the step
cost was not merely eroding the margin over training, it was suppressing it from the start.

The 17.4 % residual is most likely audit 4's H2, which is already written down with a falsifier:
the feature map is phase-blind, `corr(Q, G) = 0.25`, so V averages the crate-rich opening with the
barren endgame and there is little true gap to defend. That is the pre-registered follow-up.

> **Correction, 2026-08-15 (audit 5, `scratchpad/audit5/`). The suicide number below is not
> established, and neither is the "world seed 810731" in E30's design.**
>
> 1. **No training run in E30, E31 or E32 was ever arena-seeded.** Neither launcher passes
>    `--seed`, so `main.py` never reset the world RNG. E30's "World seed 810731" and my statement
>    that "all five runs share one arena sequence" are both false: every run saw different arenas.
>    (This cuts in our favour on one point — the n = 5 t-intervals *do* cover arena variance, where
>    I claimed they did not — and against us on reproducibility, which is nil.)
> 2. **The suicide rate is bimodal on a single Q-cell.** Audit 5 re-ran this exact configuration
>    on two fresh seeds: identical warm-start parent (same md5), identical logged hyperparameters,
>    only `EXPERIMENT` differing. It reproduced ep5000 to three digits (gap 2.727 vs 2.728,
>    suicides 0.513 vs 0.508) and then landed at **suicides 0.500 / 0.533 and survived 0.480 /
>    0.407**, against these five runs' 0.587-0.747 and 0.197-0.363 — **zero overlap**. The
>    mechanism is one row: in **55060** all five runs here are greedy-`DOWN`, both replications are
>    greedy-`UP` (verified directly; margins 0.055 to 1.71, so it is a real bifurcation, not a
>    float tie). The *training* curves are indistinguishable — the split only appears at ε = 0.
>
> So P4's failure below may be a property of these five runs rather than of the reward table, and
> **anything comparing arms against it needs a contemporaneous control.** Score and `won` look
> stable across the replications; `suicides` and `survived` do not.

**P4 FAILED, and this is now the same guard failing twice running.** Suicides at 20 000 are
**0.690** against the 0.636 threshold — and they *rise* with training (0.508 → 0.642 → 0.690)
while survival *falls* (0.394 → 0.302 → 0.258). The dawdling half of the guard is fine (steps 252,
no blow-up), so removing shortest-path pressure did not make it lazy; it made it reckless. E30
became passive, E31 becomes aggressive, and **neither run has ever satisfied the suicide guard.**
In the same rounds we survive 0.258 against the opponents' 0.375-0.399: **we are winning on points
while dying most.**

**Predicted magnitude was right for once** — I pre-committed to 3.4-3.9 at 20 000 and it landed at
3.813. That is the audit's prior rather than mine, which is the honest attribution.

- **Verdict: E31 SUCCEEDED on its primary and failed its guard**, and confirms audit 4's mechanism
  in substance: an action-independent constant ratchets the argmax, and removing it removes both
  the ratchet and the decay. **`BM_STEP_COST=0` becomes the default for rung 4.**
- **Next, in order.** (1) The suicide guard has failed twice and is now the binding constraint —
  0.690 own-bomb deaths with `killed_by` already near floor. (2) The 17.4 % residual gap decay →
  audit 4's phase digit, with its stated falsifier. (3) `KILLED_OPPONENT` is still **0.0** — the
  highest-value action in the real game has never been priced (E30 script error, uncorrected).
  (4) `evaluate.py:258` still undercounts `killed_by`; every magnitude quoted from E28 on is
  inflated ~2× until it is fixed and the cited runs re-measured.

---

## E30 — Train in the field we are measured in, and test D₄'s surviving claim

- **Question:** three entries now converge on one conclusion. E28: the rung-4 deficit is a single
  number, `killed_by` +0.434 paired. The forensic: 73.4 % of deaths reach the last savable step in
  an **all-zero** row. E29: those rows are empty in **every orientation** — 89 % of each touched
  orbit was already trained, so they are not a sampling accident but a region of state space that
  solo training never enters. **The only thing that can fill them is playing against opponents.**

  E25–E27 trained with opponents and all failed, but E27 found why (the earnings/cost ratio) and
  measured the fix: `BM_CRATE=1.0` reached **3.107** on the `rule_based` field at 6 000 episodes
  against arm F's 2.793 there. What has never been tried is the obvious thing — **training in the
  `rule_based` field itself.** Every rung-3 run trained against `coin_collector_agent`, an
  opponent that never places a bomb, and was then measured against three that do. The rows that
  kill us cannot be visited in a field where nobody bombs.

- **Change:** two, one per arm, both in `train.py` only.
  - **T** — training field becomes 3 × `rule_based_agent`; `BM_CRATE=1.0`; warm start from the
    frozen table. `callbacks.py` untouched.
  - **TD** — additionally shares every TD update across the state's D₄ orbit (~5.95 cells), and
    warm-starts from the **folded** table so the initialisation is already self-consistent.

  **Why TD is worth its half of the compute.** E29 refuted D₄ as *coverage* and explicitly left
  its *convergence-speed* claim standing — the one E14 preserved for "if the larger table proves
  data-starved". Sharing gives each cell the samples of its whole orbit without changing what is
  representable. This is the form of D₄ that has never been tested, and it has been deferred since
  E14; testing it once ends the question either way.

- **Design.** 5 training seeds per arm (`BM_RUN_INDEX` **60–64**, never used), **20 000 episodes**,
  checkpoints at 5 000 / 10 000 / 20 000. Not 40 000: E27 measured the peak at ~6 000 and decay by
  40 000, so a longer run would buy only a worse table. World seed 810731. Everything else at its
  E27 value. Unit of analysis is the **training run, n = 5**, with t-intervals.
- **Measurement:** 1000 rounds, ε = 0, `BM_TIE_TOL=0.0`, validation seed **550731**, 3 ×
  `rule_based_agent`. **`won` is primary.** Control is arm F: `won` 0.167, score 2.822.

### Prediction (written before the run)

1. **P1, primary.** T @10 000 beats arm F on **`won`** (0.167), t-CI over the five runs excluding
   0. **Refutation:** CI includes 0 or is negative → training in the measured field still does not
   beat an untrained table, and rung 4 ships frozen exactly as rung 3 did.
2. **P2, mechanism, and the reason P1 is not just hope.** Re-running the forensic pipeline on T's
   table, **the share of deaths whose `t*` row is all-zero falls from 73.4 % to below 40 %.**
   **Refutation:** it stays above 60 % → the rows still are not being visited, and the diagnosis
   that opponents fill them is wrong rather than merely insufficient.
3. **P3, D₄'s surviving claim, in falsifiable form.** **TD @5 000 ≥ T @10 000** on `won` — i.e.
   sharing buys at least a 2× sample-efficiency factor (orbits average 5.95 members, so 2× is the
   conservative half of the range). **Refutation:** TD @5 000 < T @10 000 → sharing does not
   accelerate convergence, and D₄ is finished for this project in both of its forms.
4. **P4, the named risk.** TD @20 000 is **not worse** than T @20 000. If it is, the symmetry
   assumption is violated somewhere, and the first suspect is on record: `bfs_first_step` breaks
   distance ties by `DELTAS` order (45.5 / 30.6 / 13.5 / 10.5 %), so on a tied state the feature
   map is not equivariant and sharing writes one orientation's answer into another's cell. Both
   arms run at `BM_TIEBREAK=0` deliberately, so this arm is an *approximate* symmetry and the
   entry must say so whatever the result.
5. **P5, guard.** `suicides` does not exceed arm F's 0.411 upper CI (0.442). Rung 3's lesson is
   that learning aggression is exactly when an agent forgets to run from its own bomb, and arm F's
   suicide rate is currently *better* than `rule_based_agent`'s.

**Predicted magnitude, pre-committed so no outcome reads as a success.** I expect `won` **0.19 to
0.23** against the symmetric reference's 0.25 — a real improvement that still does not clear the
bar. Beating 0.25 in one step would be a surprise and should be checked for a harness error
before it is believed; the published state of the art on this task is ≈ 5.0 score with a
335-state table (`scratchpad/survey/REPORT.md`), and we are at 2.822 with 64 000.

### Result — commit a4872b1, 1000 rounds each at validation seed 550731, n = 5 training runs

t-intervals over the five runs (t₀.₉₇₅, df = 4), not per-round bootstrap.

| arm | score | **won** | suicides | killed_by | survived | kills |
|---|---|---|---|---|---|---|
| **F** frozen control | 2.890 | 0.178 | 0.403 | 0.519 | 0.078 | 0.119 |
| **S** folded, untrained | 1.178 | 0.058 | 0.613 | 0.346 | 0.041 | 0.051 |
| **T @5 000** | **3.625 ± 0.103** | **0.347 ± 0.014** | 0.636 ± 0.045 | 0.087 | 0.277 | 0.211 |
| **T @10 000** (primary) | 3.323 ± 0.531 | **0.313 ± 0.053** | 0.551 ± 0.117 | 0.058 | 0.391 | 0.188 |
| **T @20 000** | 2.338 ± 0.546 | 0.205 ± 0.073 | 0.413 ± 0.061 | 0.046 | 0.540 | 0.127 |
| **TD @5 000** | 1.876 ± 0.860 | 0.155 ± 0.089 | 0.545 | 0.048 | 0.407 | 0.108 |
| **TD @10 000** | 1.661 ± 0.530 | 0.125 ± 0.056 | 0.480 | 0.070 | 0.449 | 0.108 |
| **TD @20 000** | 2.131 ± 0.373 | 0.187 ± 0.047 | 0.515 | 0.037 | 0.448 | 0.136 |

**P1 CONFIRMED.** T @10 000 wins 0.313 against F's 0.178, and **every one of the five runs**
clears it (0.313 / 0.245 / 0.326 / 0.319 / 0.362). **This is the first time the agent beats
`rule_based_agent`:** the symmetric bar is `won` = 0.25, and in these rounds the three opponents
score **2.99-3.17** against the 3.77-3.84 they take off arm F. E28's headline — that we are worth
+0.59 to each of them — is not merely fixed but reversed.

**P2 CONFIRMED, and it is the load-bearing number.** Re-running E28's forensic classifier against
the trained table (`scratchpad/deaths/p2_check.py`, which reproduces E28's 73.4 % exactly on the
frozen table, so the two are the same object):

| at `t*` | F | T @10 000 |
|---|---|---|
| degenerate row — uniform draw over six actions | **73.4 %** | **3.1 %** |
| trained, every greedy action fatal | 25.9 % | **93.9 %** |
| deaths per 300 rounds | 282 | **163** |

Threshold was below 40 %. `killed_by` falls 0.519 → 0.058 and non-zero rows grow 2 364 → 3 859.
**The rows that were killing us are filled, and the failure mode has changed identity** — from
"empty row, coin flip" to "trained row, confidently wrong". That second category is exactly the
one the forensic ranked opponent-BFS-distance for (42.9 % within-row lift), and it is now 94 % of
what remains.

**P5 FAILED, and it is not a technicality.** Suicides at the primary are **0.551** against a 0.442
guard, and 0.636 at the peak checkpoint — up from F's 0.403, which was *better* than
`rule_based_agent`'s. **E30 bought its opponent-deaths with own-bomb deaths.** Rung 3 wrote down
that learning aggression is exactly when an agent forgets to run from its own bomb; this is that,
measured. The entry is a large win **and** a real regression, and both belong in the report.

**P3 REFUTED.** TD @5 000 wins 0.155 against T @10 000's 0.313 — sharing does not accelerate
convergence, it retards it.

**P4 passes, but only at 20 000 and confounded.** TD @20 000 (0.187 ± 0.047) overlaps T @20 000
(0.205 ± 0.073). TD is still climbing where T is collapsing, so the two converge — but arm S
prices the handicap TD carried: **the folded table scores 1.178 against the frozen table's
2.890.** Folding *halves* it.

**So my "the group acts exactly on this encoding" claim was wrong, and arm S is what caught it.**
The group *algebra* is exact — `d4.py --self-test` verifies closure, inverses and injectivity.
The *feature map* is not equivariant, because `bfs_first_step` breaks distance ties by `DELTAS`
order, and averaging orbits therefore destroys real information rather than pooling equivalent
information. This was written down as P4's named risk before the run; the number is worse than I
expected. **D₄ is now finished in all three of its forms** — refuted as coverage (E29: 279 rows
of 61 636), refuted as an exact symmetry (arm S: −59 % score), refuted as sample sharing (P3).
After four deferrals since E14, that question is closed.

**My predicted magnitude was too pessimistic and I pre-committed to checking that.** I wrote
"0.19 to 0.23 … beating 0.25 in one step would be a surprise and should be checked for a harness
error before it is believed." It reached 0.313. Checks run: the checkpoints differ from the
frozen table and from each other; `won` is `score == best`, not a survival proxy, and only 108 of
333 wins involved surviving the round; `think_max` 3.4 ms with zero steps over the limit; opponent
scores fall rather than our score being inflated. **The result stands.**

**The shape of the decay matters more than the peak.** From 5 000 to 20 000 episodes `survived`
rises monotonically 0.277 → 0.391 → 0.540 while `score` falls 3.625 → 2.338 and `kills` falls
0.211 → 0.127, and seed variance grows with it (`won` ± 0.014 at 5 000, ± 0.073 at 20 000).

> **Refuted, 2026-08-14 (audit 4, `scratchpad/audit4/`).** I wrote here that "the agent is
> converging on a passive survival policy" because "with `GOT_KILLED` at −5 against a −0.1 step
> cost, not dying dominates the return". **Both halves are wrong**, and an independent diagnosis
> that had not read this entry when it formed its view says so with numbers: the death term
> contributes **+0.5** of the −9.6 fall in training reward, against −5.1 from crates and −2.1 from
> coins; the collapse is *complete in rows where death is impossible*; and if death dominated,
> `sd[V]` would grow to separate safe from dangerous states — it *shrinks*. **Rising survival is a
> consequence of the real mechanism, not its cause.** See E31 for what it actually is. The E27
> half of my intuition — "an action-independent constant dominates" — was right; the "not dying"
> half was not.

### Ship-seed confirmation — and a correction to the bar

`@5 000` was chosen *after* seeing validation numbers, so it is confirmed at ship seed 990731,
1000 rounds, n = 5 runs. **The selection cost nothing:**

| T @5 000 | validation 550731 | ship 990731 |
|---|---|---|
| score | 3.625 ± 0.103 | **3.621 ± 0.117** |
| **won** | 0.347 ± 0.014 | **0.354 ± 0.012** |
| suicides | 0.636 | 0.621 |
| killed_by | 0.087 | 0.092 |
| survived | 0.277 | 0.287 |

Per-seed `won` on the ship seed: 0.355 / 0.365 / 0.338 / 0.354 / 0.357. In those same rounds we
beat all three opponents on both metrics — us 3.621 / 0.354, them 3.041-3.094 / 0.265-0.271.

**Correction, and it matters for how the claim is stated.** This entry and E28 both quote "the
symmetric bar is 0.25" on the reasoning that four identical policies split the wins. That is the
idealised value and it is wrong: **ties mean win rates sum to more than 1** — 1.157 in these
rounds — and E28's *measured* self-play figure is 0.266-0.300, mean **≈ 0.282**. So the margin is
0.354 against 0.282, **+0.072 and not +0.10.** Every "0.25" in E28-E30 should be read as ≈ 0.282.
The conclusion is unchanged; the size of it is not.

**And the win is fragile in a way the score hides.** We survive *less* than the reference
(0.287 against 0.380) and suicide far more (0.621 against ~0.47). We are ahead on points while
dying more often — which is the P5 failure restated, and it means the lead does not come from
playing more safely but from earning faster in the time we have.

### Correction, 2026-08-14, after an adversarial audit (`scratchpad/audit3/`)

An audit was commissioned with the sole remit of refuting this entry, given the CSVs and the
code and told to form its own numbers. It ran 11 new 1000-round evaluations and 4 instrumented
300-round runs. **Every arithmetic figure in the result table above reproduces exactly. The bar
they are measured against does not, and the headline is attached to the wrong checkpoint.** I
verified each load-bearing claim below myself before rewriting.

**1. The pre-registered primary does not beat `rule_based_agent`.** With the bar at its measured
0.283, `won` at @10 000 is **0.3130 ± 0.0528 → [0.260, 0.366]**, which *contains* the bar — at
the validation seed and again at the ship seed (0.3102). @5 000 is [0.334, 0.361] and excludes
it, at both seeds. So the claim is true of the checkpoint I selected post-hoc and **not** of the
one I pre-registered. P1 as literally written — "T @10 000 beats **arm F**" — is confirmed and
large; the invalid step was from "beats arm F" to "beats `rule_based_agent`".

**2. "Every one of the five runs clears it" is misleading.** True of arm F's 0.178, which is what
the sentence says; false of the `rule_based` bar, which is what the next sentence implies. At
@10 000 seed 61 scores **0.245 at both evaluation seeds** — a genuinely worse table, not noise.
4/5 at @10 000; 5/5 at @5 000.

**3. The opponents were not beaten down — arm F simply stopped feeding them.** Coins are a fixed
pool (8.96 in every field), so total score moves only through kills:

| field | total score | total kills | opponents' mean |
|---|---|---|---|
| arm F | 14.331 | 1.075 | 3.814 |
| T @10 000 | 12.718 | 0.749 | 3.102 |
| 4 × `rule_based` | 13.020 | 0.814 | 3.255 |

The audit prices it: opponents drop 0.745 each, their kill credit drops 2.16 points — **97 % of
the effect is arm F ceasing to be food.** So "E28's +0.59 is reversed" is wrong. Against the
correct reference — `rule_based_agent` in slot 0 of its own field — T @10 000's out-of-sample
score edge is **+0.04**, and T @5 000's is +0.40. Comparing opponents to their score against
arm F measures arm F, not us.

**4. `killed_by_opponent` is definitionally wrong in `tools/evaluate.py`, and I own the bug.**
`evaluate.py:258` computes `max(0, died − suicides)`. But `environment.py:243-257` evaluates each
explosion separately: an agent standing in *both* its own blast and an opponent's gets
`KILLED_SELF` **and** hands the opponent `KILLED_OPPONENT` (+5). It dies once, so the metric books
a pure suicide and the opponent's kill is invisible. **The undercount scales with bombs placed**,
which is exactly what differs between the arms being compared. Instrumented truth, per round:

| | metric says | true opponent-blast deaths | bombs/round |
|---|---|---|---|
| arm F | 0.513 | **0.577** | 16.0 |
| `rule_based` (4×rb) | 0.093 | **0.210** | 20.2 |
| T @10 000 | 0.070 | **0.137** | 28.7 |

**E28's "killed by opponents six times more often" is really 2.7×. E30's "0.519 → 0.058" (8.9×)
is really 0.577 → 0.137 (4.2×).** Direction and mechanism survive everywhere; every magnitude in
the E28-E30 chain is inflated roughly 2×.

**5. `BM_RUN_INDEX` 60-64 were not "never used".** E27's disclosed pilot ran `BM_CRATE=1.0` in the
`rule_based` field on **training seed 60** and validated on 550731. Arm T is that pilot extended.
The ship-seed replication mitigates it, but the design sentence is false as written.

**6. n = 5 understates the uncertainty.** All five runs share one arena sequence (world seed
810731); only exploration RNG and the unseeded opponents differ, so the t-interval covers
exploration but not arena sampling.

**7. What survived the attack, having been attacked properly.** @10 000 replicates out of sample
(0.3130 val → 0.3102 ship), so it is not a selection artefact — it is simply not far enough above
0.283. Training beating the frozen table is real and large at any bar. **An E30 agent does beat
`rule_based_agent`: T @5 000, `won` +0.067 over the measured bar with a CI excluding it on the
held-out seed, 5/5 seeds, score 3.65 against 3.25.** And P5's failure is *understated* — true
own-bomb death rate rises 0.370 (F) → 0.467 (@10 000) → **0.650** (@5 000). The checkpoint that
wins is the one that kills itself most.

**8. In-distribution by construction.** Arm T trains against 3 × `rule_based_agent` and is
measured against 3 × `rule_based_agent`, and its winning mechanism is denying a *scripted*
opponent its kills. That need not transfer to a tournament of unknown agents, and the report must
say so rather than let "beats `rule_based_agent`" stand unqualified.

- **Verdict after correction: E30 SUCCEEDED on P1 and P2, FAILED P5, and its headline needed
  re-attaching.** The honest claim is: **T @5 000 beats `rule_based_agent` on the held-out seed,
  5/5 seeds, selected post-hoc and confirmed** — not the pre-registered @10 000. First entry where
  training beat the frozen table on any rung above 2.
- **Next.** (a) The remaining deaths are 94 % trained-but-fatal — the forensic's C3 (opponent BFS
  distance, ×4 rows) now has a clean target and a measured lift, and coverage is no longer the
  binding constraint that argued against it. (b) The passivity gradient is a reward-balance
  question: `GOT_KILLED` −5 against `STEP_COST` −0.1. (c) Selection discipline — 5 000 was chosen
  *after* seeing validation numbers, so whatever ships must be confirmed at ship seed 990731
  before any number is quoted.

---

## E29 (stage 1) — Coverage, not features: fold the table by its symmetry group

- **Question:** E28 established that the rung-4 deficit is one number, `killed_by` (+0.434
  paired), and the forensic established that **73.4 % of deaths reach the last savable step in an
  all-zero Q-row**, where `act()` draws uniformly over six actions. That is a coverage problem,
  and every new digit makes coverage *worse*. Two routes exist. This entry takes the cheap one.

  **D₄ canonicalisation was the planned E14 and was dropped on evidence** — `cycle_dump` showed
  the two rows of an actual period-2 cycle are not related by any group element, so merging
  symmetric rows would not have fixed the failure it had been promoted to fix. That verdict
  stands and is not being relitigated. What was explicitly preserved was the *other* argument:
  "its sample-efficiency argument survives and becomes relevant again if the larger table proves
  data-starved." The forensic is that condition, measured: 2 364 trained rows of 64 000, and the
  rows we die in are the empty ones.

  So this is not the same experiment failing twice. E14 asked D₄ to break aliasing; E29 asks it
  to share data. Only the second claim was ever supported.

- **Why it is exact rather than approximate.** The state is direction-indexed throughout: digits
  1-4 are one per direction in `DELTAS` order, digit 6 is `direction + 1` with 0 reserved, and
  digits 5/7/8 are invariant. `DELTAS` is listed clockwise, so every group element is
  `g(d) = (s·d + k) mod 4` with `s ∈ {±1}`, `k ∈ {0..3}` — the whole 8-element group. The four
  movement actions permute with it; `WAIT` and `BOMB` are fixed points. Group algebra verified in
  `scratchpad/benedict/d4.py --self-test` (closure, inverses, injectivity on rows, identity acts
  trivially, `g` then `g⁻¹` restores the state).

- **Change:** **`callbacks.py` is untouched for arm S.** The fold averages each orbit over its
  *trained* members only — zero is the untrained sentinel, so averaging a trained row with an
  empty one would halve it rather than share it — and writes the result back into **every** orbit
  member. The table keeps its 64 000 × 6 shape and the lookup path is unchanged, so there is no
  new code in the tournament agent and no per-step cost. Runtime canonicalisation would be
  equivalent and strictly riskier. (Sharing updates *during training* does need the runtime
  version; that is stage 2.)

  **The one honest caveat, and the reason for the second factor.** `bfs_first_step` breaks
  distance ties by `DELTAS` order, measured at **45.5 / 30.6 / 13.5 / 10.5 %** — a bias with no
  counterpart in the game, and already flagged in the E14 note as something that "still has to be
  fixed before any of that". On a tied state the feature map is therefore **not** equivariant, and
  the fold merges rows the features treat differently. Rather than assume that is harmless, the
  tie-break is its own factor: `BM_TIEBREAK=1` draws a uniform permutation of `DELTAS` per BFS
  call, in both `bfs_first_step` and `escape_direction`. Default 0, so the shipped agent is
  unchanged until measured.

- **Design.** 2 × 2 on the frozen table, **no training runs at all** — every arm is a table
  transform and/or an environment switch, ~8 min per evaluation:

  | arm | table | tie-break |
  |---|---|---|
  | **F** | frozen | `DELTAS` order (control, already measured) |
  | **B** | frozen | uniform |
  | **S** | D₄-folded | `DELTAS` order |
  | **BS** | D₄-folded | uniform |

  1000 rounds, ship seed 990731, 3 × `rule_based_agent`, ε = 0, `BM_HUNT=1`.

- **Measurement:** task-4 preset. **`won` is primary** — `MEASUREMENT.md` makes it the ranking
  metric on rung 4, and E28 measured our relative deficit as worse on `won` (0.584) than on
  `score` (0.867).

### Prediction (written before the fold is run)

`d4.py --self-test` has been run and passes; **the orbit statistics have deliberately not been
looked at**, because P1 is about them.

1. **P1, the gate — decidable from the table alone, before a single game.** Of the **129 all-zero
   rows** that hold the forensic's 207 degenerate deaths, **≥ 40 % gain a value from the fold**
   (i.e. have at least one trained orbit sibling). **Refutation:** < 20 % → the empty rows are
   empty in every orientation, folding cannot touch the dominant failure, and stage 1 stops here
   without spending an evaluation.
2. **P2, primary.** BS beats F on **`won`** (0.167), paired CI excluding 0. **Refutation:** CI
   includes 0 or is negative.
3. **P3, mechanism — the one that makes P2 falsifiable rather than just hopeful.** The gain is
   from *folding*, not from the tie-break: `|won(B) − won(F)| < |won(BS) − won(F)|`. If B alone
   moves `won` as much as BS does, then P2 measured a tie-break repair and D₄ is again unproven.
4. **P4, the risk I expect to bite.** Folding **forces** symmetric Q-values in self-symmetric
   states, which manufactures exact ties precisely where the agent has no reason to prefer a
   direction — the period-2 cycling family. Guard: `suicides` does not rise and `invalid` does not
   worsen beyond its CI in BS. **If P2 passes and P4 fails, the entry is a trade, not a win, and
   must be reported as one.**
5. **P5, guard.** `think_max_ms` unchanged (~0.3 ms). The fold is offline; only the per-call
   permutation is new, and it is three shuffles of a 4-list per step.

**Predicted magnitude, so the entry cannot be scored as a success at any outcome:** if P1 lands
near 40 % and the sibling values are correct, roughly 40 % of 73.4 % of deaths get a real policy
instead of a 1-in-6 draw. That is worth a few points of survival, not a doubling — I expect
`won` +0.02 to +0.05, i.e. **0.19-0.22 against the symmetric reference's 0.25**. A result above
0.25 would be surprising and should be checked for a harness error before it is believed.

### Result — P1 REFUTED at the gate, no evaluation spent

```
trained_before 2364   trained_after 2643   gained 279
orbits_total  10750   orbits_trained  531
```

| | | |
|---|---|---|
| all-zero death rows that gain a value | **1 / 129** | **0.8 %** |
| deaths whose `t*` row gains a value | **1 / 207** | **0.5 %** |
| whole table: empty rows that gain a value | 279 / 61 636 | 0.45 % |

Threshold was ≥ 40 %, refutation below 20 %. **This is 0.8 %.** Arms B, S and BS are not run.

The check independently reproduces the forensic's 129 rows / 207 degenerate deaths from a
separate pickle, and confirms all 129 were untrained before the fold — so the two analyses agree
on the object being measured.

**Why, and it is a sharper result than the one I predicted.** 2 364 trained rows sit in only 531
orbits whose total membership is 2 643 — **89 % of every touched orbit was already trained.**
When the agent visits a state it visits that state's rotations and mirrors too, because the board
is symmetric and the agent moves in all four directions. The empty rows are therefore **not a
sampling accident that symmetry can repair: they are empty in every orientation.** They are a
region of state space that solo training never enters at all, in any symmetry class.

That strengthens the forensic's conclusion rather than merely failing to help it. The missing
rows exist only with opponents on the board, and **the only thing that can fill them is playing
against opponents.** Route (b) is closed; route (a) is now the whole plan.

**What is *not* refuted, and must not be reported as if it were.** This measures D₄ as a
**coverage** fix, which is what E29 promoted it for. It says nothing about D₄ as a
**convergence-speed** measure during training: orbits average 5.95 members, so folding updates
would give each canonical cell roughly 6× the samples even though the row count barely moves.
That is the original E14 sample-efficiency argument, it survives untouched, and it is a stage-2
question with its own test. `scratchpad/benedict/d4.py` is kept for it.

**Verdict: FAILED, at a cost of one table transform and zero evaluations.** The gate earned its
place — without it this entry would have spent four 1000-round runs to learn the same thing from
noisier evidence.

### Stage 2, sketched but not pre-registered

Train on rung 4 with E27's corrected reward scale (`BM_CRATE=1.0`), warm-started from whichever
table wins stage 1, 5 seeds. The forensic explains why this should work where E25-E27 did not:
the missing rows exist *only* with opponents present. E27 already measured a 6 000-episode
`BM_CRATE=1.0` table at **3.107** on the `rule_based` field against arm F's 2.793. Pre-registered
separately, after stage 1 decides the representation.

---

## E28 — Rung-4 baseline: we do not have an escape problem, we have a threat-blindness problem

- **Question:** rung 4 is `classic` against **3 × `rule_based_agent`** — the tournament setting.
  Before changing anything I need the floor and the ceiling on the *same* arenas, and I need to
  know **which of the two death causes** the gap lives in. That decides the whole rung: `suicides`
  is an escape-logic problem (features 1–5, the blast digits), `killed_by` is a positioning and
  threat-anticipation problem (information the map does not carry at all).

  The two numbers I already have say something I did not expect, and they say it loudly:

  | | arm F (the rung-3 ship) | `rule_based` in a field of itself |
  |---|---|---|
  | score | 2.757 | 3.290 |
  | won | 0.170 | 0.280 |
  | **suicides** | **0.377** | **0.507** |
  | **killed_by** | **0.553** | **0.093** |
  | total deaths | 0.930 | 0.610 |

  **We kill ourselves 26 % less often than `rule_based_agent` does, and are killed by opponents
  six times more often.** If that survives a paired re-measurement it inverts the priority I
  carried out of rung 3, where I wrote that "it dies to itself almost as often as to them" and
  filed escape logic as the rung-4 problem (`experiments/benedict_task3.md` §6.4). The correct
  reading of the same split is that our escape logic is *better* than the reference's, and the
  entire deficit is in deaths we do not cause.

  **The two numbers above are not comparable and that is exactly why this entry exists.** They
  come from different base seeds (990731 vs 20260731) and different n (1000 vs 300), so they
  share no arenas. The effect is far too large to be seeding noise, but "too large to be noise"
  is a guess, and E25 is the entry that records what my guesses are worth.

  Why it would be true, mechanistically: **digits 1–5 are all computed from bombs already on the
  board.** Digits 1–4 classify each neighbour as blocked / lethal this step / in a blast / clear,
  and digit 5 counts moves of grace on my own tile — every one of them reads `game_state['bombs']`
  and `explosion_map`. Nothing in the state conditions on a bomb that has *not been placed yet*.
  The agent can see a fuse; it cannot see a threat. `rule_based_agent` drops a bomb whenever an
  opponent is adjacent and then flees, so against it, reacting to placed bombs is reacting one
  step too late by construction — and 0.553 is what that looks like from the inside.

- **Change:** **none to the agent.** `agent_code/benedict_task4/` is `callbacks.py`, `train.py`
  and `q_table.npy` copied from `benedict_task3` — `q_table.npy` verified byte-identical, so this
  measures the rung-3 ship under its rung-4 name. A pure measurement entry; the first rung-4
  change is written after the three surveys land, not before.

- **Design.** Two evaluations on **identical arenas** — same base seed, same n, so
  `analyze.py --compare` is paired on the arena the way `MEASUREMENT.md` defines pairing for
  rung 3+ (arenas only; the provided opponents also shuffle with the stdlib RNG, which
  `evaluate.py` does not reach — §5.5 of `experiments/benedict_task3.md`):

  | label | field |
  |---|---|
  | **F4** | `benedict_task4` + 3 × `rule_based_agent` |
  | **R4** | 4 × `rule_based_agent` (the symmetric reference) |
  | **F4-noHUNT** | `benedict_task4` with `BM_HUNT=0` + 3 × `rule_based_agent` |

  1000 rounds, ship seed **990731**, ε = 0, `BM_TIE_TOL=0.0`, `BM_HUNT=1` (the default since
  E26). New tree `results/eval/task4_tournament/`. R4's reported row is agent slot 0; the other
  three slots are the spread of the same policy and bound the arena noise for free.

  The symmetric field is the honest ceiling: four identical policies split the wins, so
  `won ≈ 0.25` **is** the reference value and 0.280 is one slot's realisation of it. Beating
  `rule_based_agent` means `won > 0.25` in a field of three of them.

### Disclosure: the third arm was designed after a teardown of the opponent

The F4-noHUNT arm and P5 below were added after a subagent read `rule_based_agent/callbacks.py`
line by line (`scratchpad/rb_teardown/`) and I verified the load-bearing lines myself. That is
*design* input, not outcome data from E28 — no E28 number existed when it was written — but the
ledger says so rather than presenting a three-arm design as the original plan.

Two things came out of it that bear on this entry:

1. **`rule_based_agent` is not anticipatory either.** It never reads `others[i][2]`
   (`bombs_left`), never predicts opponent movement, and builds its danger model from
   `game_state['bombs']` alone — the same input class as our digits 1-5. So the anticipatory gap
   is a gap for *both* agents, and the question P1 poses gets sharper rather than easier: if
   neither side can see an unplaced bomb, **why do we die to opponents six times more often?**
2. **Its only offensive rule may be one we walk into on purpose** (`callbacks.py:173-176`,
   verified in the source):

   ```python
   if len(others) > 0:
       if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
           action_ideas.append('BOMB')
   ```

   Unconditional, no escape check, high in the proposal stack: **it bombs whenever anything
   stands next to it.** E26's HUNT fallback makes digit 6 walk toward the nearest opponent
   whenever no coin and no crate is reachable — which on a four-agent board is most of the round
   after ~step 140. HUNT was measured against `coin_collector_agent`, which never places a bomb
   at all, i.e. in a field where approaching an opponent is free. Against `rule_based_agent` the
   same feature may be walking us into a deterministic bomb for two thirds of every round, and
   the +1.595 that won rung 3 would not transfer.

### External calibration — what rung 4 is actually worth beating

From a prose-only survey of published solutions to this same course project
(`scratchpad/survey/REPORT.md`; ~90 repos swept, **only 8 with real written prose**, no source
code read). It changes no design and no prediction here — it sets the target, and one number in
it is uncomfortable:

- **The bar is ≈ 5.0 score against 3 × `rule_based_agent`** (Voß/Tiedl/Müller, MLE SS2024, 1000
  rounds). We are at 2.757. Their table has **335 states**; ours has 64 000.
- **Three independent ablations say aggressive state-space *reduction* beats richness** (72 → 12
  features; 10⁵ → 256 states; 2²⁰ → 2160 → 335). That is in direct tension with the teardown's
  proposal to append two digits and multiply our rows by six, and E28 cannot settle it — but the
  next entry has to address it rather than quietly pick a side.
- **Nobody in the corpus predicts a bomb that has not been placed yet.** The closest is one CNN
  "opponent can drop a bomb" channel, unablated. So if P1 holds, the feature it motivates is
  novel against the published work rather than a re-implementation.
- **Every number in that corpus is a bare mean — not one source reports a confidence interval**,
  so a 5.04 and our 2.757 are not as far apart as they look, and our paired-CI rule already puts
  the method ahead of the published prior work.
- One control we already own is worth more than I credited: `benedict_task3.md` §5.7 measured the
  best purely-feature policy our digits allow at 0.030 against the learned 4.377. The same control
  in the strongest published project came within 3 % of its learned agent (6.1 vs 6.3), and the
  author concluded the features were doing the work. **Ours says the opposite, with a much larger
  margin** — that belongs in the report next to their number.

### Disclosure: a death forensic ran before E28 and largely pre-empts P1 and P2

P1–P3 and the guard were committed at **98aedf7, 2026-08-14 09:44:57**, before any of the three
surveys returned — git fixes that ordering. They are nonetheless now **confirmatory rather than
exploratory**, and the entry says so instead of collecting a prediction it already knew.

A forensic diagnostic (`scratchpad/deaths/`, 300 rounds at seed **20260731** — not E28's ship
seed — 282 deaths over 41 945 alive steps, every death mechanically re-simulated and the killing
bomb's owner reproduced 282/282) gives, for the same agent:

| | forensic, seed 20260731 | R4 reference, same seed/n |
|---|---|---|
| suicides | 0.443 | 0.507 |
| killed_by | **0.497** | **0.093** |
| score | 2.850 | 3.290 |

So P1's gap is ≈ 0.40 and will almost certainly hold. **P2 holds too, but by 0.064, not the 0.130
the unpaired figures implied** — "we suicide 26 % less" was flattered by the seed mismatch, and
the honest version is "slightly less".

**More important: the forensic refutes the mechanism P1 was going to license.** I framed the gap
as *threat blindness* — information the state does not carry. Measured at the last savable step
of each death, it splits, and the halves need opposite fixes:

| at the last savable step `t*` | n | share |
|---|---|---|
| **Q-row is all-zero — all six actions tie and `act()` draws uniformly** | 207 | **73.4 %** |
| row is trained and every greedy action is fatal there | 73 | 25.9 % |
| not avoidable within 5 steps | 2 | 0.7 % |

**Only 26 % of deaths are a missing-information problem. 73 % are a missing-table-entry problem** —
the state is recognised perfectly, the row is empty, and the policy is a coin flip over six
actions. Those rows are 2.6 % of alive steps but 73.4 % of `t*` steps (28× enrichment), and 55.8 %
of every visit to one is the last savable step of a death. They exist only with opponents present,
and the shipped table was trained solo — this is `benedict_task3.md` §3 showing up as deaths.

Three further measured results, each of which kills a proposal I would otherwise have run:

- **"Is a bomb here survivable" buys nothing** — P(dead ≤ 4) is .033 with an escape and .035
  without. It was the teardown's second-ranked digit. Only **5 of 133** own-bomb deaths came from
  an unsurvivable drop: the agent does not bomb itself into corners, it drops a safe bomb and then
  fails to walk out.
- **Dead ends are *safer*, not more dangerous** (.020 against a .034 base). The intuition is
  backwards on this board.
- **93.3 % of deaths had a visible escape when the killing bomb appeared**, with three steps of
  warning. The agent is essentially never ambushed — which is what makes "it cannot see an
  unplaced bomb" the wrong diagnosis.

The one candidate that survives on both volume and lift is **opponent BFS distance, bucketed**:
at `t*` an opponent is within 2 tiles in 75 % of the trained-but-fatal deaths against a 10.5 %
base rate, and in the ambiguous rows the fatal and safe visits have *identical digits* — row 34110
was visited 482 times, and opponent distance was 2 in 71 % of fatal visits against 4 % of safe
ones. That is the E29 candidate, and it must be measured **together with a coverage metric**,
never alone, because §2 says coverage is the binding constraint and every new digit makes coverage
worse.

**None of this changes what E28 measures.** It changes what E28 is *for*: the paired arenas, the
HUNT arm, and P3 are the parts still carrying information.

- **Measurement:** task-4 preset — `score` / `won` / `kills` / `suicides` / `killed_by` /
  `think_ms` — plus `survived` and `crates`. `won` decides ranking, per `MEASUREMENT.md`.

### Prediction (written before the run)

Naming which number decides, because E25 scored four of six predictions "correct" on an agent
20 × worse and every one of them was a guard metric.

1. **P1, primary and decisive.** On paired arenas, F4's `killed_by` exceeds R4's by **≥ 0.30 per
   round**, CI excluding 0. **Refutation:** gap < 0.30 or CI includes 0 → the split above was an
   artefact of the seed/n mismatch, the rung-4 story is not "threat blindness", and the feature
   work aimed at anticipating opponents' bombs is aimed at nothing.
2. **P2, the counter-intuitive half, and the one that redirects effort.** F4's `suicides` is
   **lower** than R4's, CI excluding 0. **Refutation:** F4 ≥ R4 → escape logic is back on the
   table and rung 3's §6.4 stands as written.
3. **P3.** The relative deficit on `won` is larger than on `score`: `won_F/won_R < score_F/score_R`
   (the unpaired figures give 0.61 vs 0.84). Mechanism: dying at 0.93 deaths/round ends *our*
   scoring while three opponents keep collecting, so we lose rank faster than we lose points.
   **Refutation:** the ratios come out equal or inverted → deaths are not costing us rank, and
   `won` and `score` can be optimised as one target.
4. **P4.** F4-noHUNT's `killed_by` is **lower** than F4's, CI excluding 0 — i.e. the rung-3
   feature win is what feeds us to `callbacks.py:173-176`. **Refutation:** no difference, or the
   wrong sign → we end up adjacent to opponents for some other reason and P1's threat-blindness
   reading survives intact.

   **P4 is deliberately not primary, because it sets a trap.** HUNT also removed the
   `INVALID_ACTION` leak worth −24.25 per round on rung 3. If `killed_by` falls *and* `score`
   falls, the two effects are entangled and neither number decides on its own — the honest
   verdict then is "HUNT is doing two opposite things on rung 4" and the next entry has to
   separate them (a fallback objective that is neither `NO_TARGET` nor "walk at the enemy").
5. **P5, guard.** `think_max_ms` under 5 ms. Rung 4 is the tournament setting and the limit is
   500 ms on hardware far slower than this one; arm F measured 0.351 ms on rung 3, so anything
   near the guard means the copy is not the agent I think it is.

I expect F4's score to land within noise of the 2.757 already measured, since the agent is
byte-identical and only the seed pairing changes. **That is not a prediction, it is a smoke test
— if F4 comes back materially different from 2.757, the copy or the harness is wrong and no
other number in this entry may be read.**

### Result — commit 7d9057c, 1000 paired rounds each, ship seed 990731

Smoke test passes: F4 scores **2.822 [2.680, 2.970]**, and the CI contains the 2.757 measured
under the rung-3 name. The copy is the agent.

| | F4 | R4 (symmetric ref) | paired difference | |
|---|---|---|---|---|
| score | 2.822 | 3.254 | −0.432 [−0.644, −0.221] | WORSE |
| **won** | **0.167** | **0.286** | **−0.119 [−0.154, −0.084]** | WORSE |
| kills | 0.121 | 0.196 | −0.075 [−0.109, −0.041] | WORSE |
| **suicides** | **0.411** | **0.533** | **−0.122 [−0.164, −0.079]** | **BETTER** |
| **killed_by** | **0.525** | **0.091** | **+0.434 [+0.398, +0.469]** | WORSE |
| survived | 0.064 | 0.376 | −0.312 [−0.345, −0.279] | WORSE |
| invalid | 2.61 | 7.34 | −4.73 [−5.13, −4.33] | BETTER |
| think_max_ms | 0.3 | 1.4 | −1.166 | BETTER |

**P1 CONFIRMED** — +0.434, comfortably over the 0.30 threshold, CI excluding 0 by an order of
magnitude. **P2 CONFIRMED** — −0.122, and it is a real effect, not the seed artefact the forensic
suggested it might be (that run gave −0.064). **P3 CONFIRMED** — the relative deficit is 0.584 on
`won` against 0.867 on `score`; deaths cost rank faster than they cost points. **P5 guard passes**
at 0.3 ms, five times *faster* than `rule_based_agent`.

**P4 REFUTED.** `BM_HUNT=0` moves `killed_by` by −0.035 [−0.078, **+0.009**] — not demonstrated.
Walking at opponents is *not* what feeds us to `callbacks.py:173-176`. The pre-registered
refutation clause said this would leave P1's threat-blindness reading intact; it does not, because
the forensic had already refuted that reading on independent evidence (73.4 % of deaths are
all-zero rows). **Both of my mechanisms for the `killed_by` gap are now dead, and the forensic's
is the one standing.**

#### The result nobody asked for, flagged as post-hoc

Not pre-registered, so it is an observation and E29 must re-derive it, not cite it:

| noHUNT − HUNT | | |
|---|---|---|
| **won** | **+0.041 [+0.008, +0.074]** | **BETTER** |
| score | −0.076 [−0.252, +0.100] | no effect |
| survived | +0.086 [+0.060, +0.112] | BETTER |
| suicides | −0.051 [−0.093, −0.010] | BETTER |
| kills | −0.042 [−0.069, −0.016] | WORSE |
| invalid | +6.39 [+5.37, +7.42] | WORSE |

**Turning off the feature that won rung 3 improves the rung-4 ranking metric at no cost in score.**
HUNT bought +1.595 score against `coin_collector_agent`; here it buys 0.042 kills and pays 0.051
suicides, 0.086 survival and 0.041 `won`. It also still earns its keep on the invalid-action leak
(2.61 vs 9.00), so this is genuinely two opposite effects in one switch — exactly the entanglement
P4 was written to warn about.

**Do not flip the default on this.** The forensic gives a mechanism that predicts the sign will
change: HUNT steers the agent into opponent-adjacent states, and those are precisely the states
whose rows are all-zero, because the table was trained solo. HUNT's cost may be *entirely* a
coverage artefact. Re-measure it on a table that has actually seen rung 4 before deciding what
ships.

#### The number that should open the report's rung-4 section

`rule_based_agent` scores **3.847** in a field containing us and **3.254** in a field of four of
itself — **+0.59 each, +1.78 across the field**, with kills up 0.204 → 0.319. Our own mean is
2.822. **We do not merely lose to the reference agent; we are the reason it beats its own
baseline**, and we hand it more than half our own score in kill credit. `won` 0.167 against a
symmetric 0.25 follows directly.

Everything else about the agent is *ahead* of the reference: fewer suicides, 64 % fewer invalid
actions, five times faster. The entire deficit is one number, `killed_by`, and per the forensic
73 % of it is an empty Q-row drawing uniformly over six actions.

**Verdict: E28 SUCCEEDED as a measurement and killed both of the mechanisms I brought to it.**
Rung 4's problem is not that the agent cannot see — it is that the agent has never been in these
states. The next entry attacks coverage, not features.

- **Carried to E29.** Two coverage routes, and the forensic's ranked feature is explicitly *not*
  first: (a) train on rung 4 with E27's corrected reward scale, warm-started from the frozen
  table, which fills exactly the rows that only exist with opponents present — E27 already
  measured a 6 000-episode `BM_CRATE=1.0` table at **3.107** on the `rule_based` field against
  arm F's 2.793, and the forensic now explains *why* that should work; (b) 8-fold symmetry
  canonicalisation, ~8× coverage at zero information cost, against the survey's caution that one
  published attempt reduced 81 → 15 states with "no noticeable improvement". Opponent BFS
  distance (the forensic's C3, ×4 rows) comes **after** coverage, never alone, and always
  reported next to a coverage metric.

---

## E27 — The reward table was calibrated for a board the agent had to itself

- **Question:** three experiments (E25, E26) concluded that training in an opponent field
  destroys a competent policy, and E26 wrote that off as a property of *training*. The second
  audit (`scratchpad/audit2/`) shows it is a property of the **reward magnitudes**, which is the
  one axis none of the three varied — even though E16 had already measured that axis as worth
  45 crates on rung 2.

  The value function's dynamic range is set by gross earnings against the **action-independent**
  step cost. From the committed CSVs:

  | | earnings | costs (step + invalid + death) | earn/cost |
  |---|---|---|---|
  | rung 2, solo, 400 steps | +77.3 (coins 42.4, crates 35.0) | −40.1 | **1.93** |
  | rung 3, cc field, same reward table | +18.3 (coins 10.4, crates 7.9) | −26.1 | **0.70** |
  | rung 3, cc field, `CRATE_DESTROYED = 1.0` | +36.7 | −26.1 | **1.41** |

  Nine coins shared four ways cuts the coin stream 4× (8.47 → 2.09) and the crate stream 4.4×,
  while steps alive only fall 1.85× (400 → 216). **The table that produced rung 2's competent
  policy pays 2 : 1; the same table on rung 3 pays 0.7 : 1.** The return is then dominated by a
  constant −0.1 per step, and a constant is action-independent — so the fixed point is too. That
  is the flattening E25 measured and misattributed.

  `CRATE_DESTROYED` is the right knob because it is the only **dense, positive** reward
  attributable to a specific action (`BOMB`) that survives on a contested board: coins are
  capped at a ~2.25 fair share and kills are rare.

- **Change:** `STEP_COST` becomes an environment switch; nothing else. `callbacks.py` untouched.

### Disclosure: pilot data existed before these predictions were written

The audit ran single-seed screens of seven levers and a 3-seed replication; I then replicated
independently on an unused training seed (60), 6 000 episodes, 150 rounds at ε = 0, seed 550731:

| | cc field | `rule_based` field |
|---|---|---|
| `BM_CRATE=0.3` (= E26 arm H) | 0.660 (4.73 crates) | 0.953 (6.01 crates) |
| **`BM_CRATE=1.0`** | **2.940** (28.25 crates) | **3.107** (29.69 crates), won 0.220 |
| arm F, the frozen ship | 4.160 | 2.793, won 0.140 |

So a trained agent already beats the frozen ship **on the tournament field** at 6 000 episodes,
on every seed tried (audit: 3.72/3.22; mine: 3.107). It is still behind arm F on rung 3's own
field. These predictions are written knowing that, which is disclosed rather than hidden.

- **Design.** One factor — the earnings/cost ratio — reached by **two independent knobs**, so
  the *mechanism* is falsifiable and not just the effect:

  | arm | change | earn/cost |
  |---|---|---|
  | **C03** | `BM_CRATE=0.3` — E26 arm H exactly, the control | 0.70 |
  | **C10** | `BM_CRATE=1.0` | 1.41 |
  | **S03** | `BM_STEP_COST=-0.03`, crate at 0.3 | 1.53 |

  5 training seeds each (`BM_RUN_INDEX` **50–54**, never used), 40 000 episodes, cc field, world
  seed 810731, `BM_HUNT=1 BM_KILL=25`, everything else at its E26 value. **Primary checkpoint
  20 000**, pre-registered.
- **Measurement:** 300 rounds, ε = 0, **`BM_TIE_TOL=0.0`** — the shipped default. The tolerance
  is *not* tuned per arm: the audit measured that it triples a broken table (0.565 → 1.555) and
  does nothing for a healthy one (4.115 → 4.235), so tuning it would flatter precisely the arm
  that fails. Validation seed 550731; unit of analysis is the **training run, n = 5**.
  Rung-3 pairing is on arenas only (`MEASUREMENT.md`), so CIs are wider than fully-paired ones.

### Prediction (written before the run, after the pilot above)

1. **P1, primary and decisive.** C10 @20 000 beats **arm F's 4.160** on `score` in the cc field,
   t-CI over the five runs excluding 0. **Refutation:** CI includes 0 or is negative → training
   still does not beat the frozen table on its own rung. *If P1 fails the entry is FAILED
   regardless of P2–P6.*
2. **P2, the shipping decision, named in advance and reported whatever it says.** C10 @20 000
   beats **arm F's 2.793** in the 3× `rule_based` field, CI excluding 0; `won` reported alongside
   (F = 0.140). **Refutation:** CI includes 0 → the frozen table stays the ship *whatever P1
   says*, because rung 4 is what the tournament scores.
3. **P3, mechanism, decidable before any score is computed.** Visitation-weighted median decision
   margin at 20 000: C03 below 0.01 and C10 above 0.05, in ≥ 4 of 5 seeds each. **Refutation:**
   C10 and C03 indistinguishable → the margin collapse is not what the crate reward fixes, and
   any score effect has another cause.
4. **P4, mechanism, second knob.** S03 − C03 > 0 on `score` with a CI excluding 0, and S03
   between C03 and C10. **Refutation:** S03 ≤ C03 → the earnings/cost *ratio* is not the
   operative quantity, and the effect is specific to the crate reward being attributable to
   `BOMB`. That is a weaker and different claim and must be written as such.
5. **P5, horizon guard — the one I expect to bite.** C10 loses no more than 25 % of its 20 000
   `score` by 40 000. The pilot's decision margin already decays 0.704 → 0.323 between 3 200 and
   6 000 episodes. **Refutation:** it loses more → the crate reward *delays* the collapse rather
   than preventing it, 20 000 is a stopping rule rather than a converged result, and the entry
   must say so.
6. **P6, regression guards — explicitly not evidence of success.** `suicides` ≤ 0.60,
   `crates` ≥ 20, `think_max_ms` < 0.5 measured **serially**. E25's lesson is pre-registered
   here: **guards passing while `score` fails is a FAIL**, and no guard may be reported as a
   positive result.

- **Cost.** 15 runs × 40 000 episodes ≈ 7 h in two waves; ~60 evaluations ≈ 2 h at 5-way, plus
  ~10 min serial for think time. If that is too much, drop S03 to n = 3 (12 runs, ~5.5 h) — and
  say so in the entry, never silently, because S03 is what makes P3/P4 falsifiable.
### Results (300 rounds, ε = 0, `TIE_TOL` 0.0, seed 550731, n = 5 runs per arm)

| cc field @20 000 | arm F (frozen ship) | C03 (control) | C10 (crate 1.0) | S03 (step −0.03) |
|---|---|---|---|---|
| **score** | **4.160** | 0.852 [0.51, 1.20] | 1.777 [1.43, 2.12] | 2.986 [0.36, 5.62] |
| crates | 26.63 | 6.42 | 17.97 | 20.03 |
| survived | 0.400 | 0.741 | 0.849 | 0.833 |
| won | 0.370 | 0.067 | 0.159 | 0.317 |

**P1: C10 − F = −2.383 [−2.729, −2.036].** Decisively worse.
**P2: C10 − F on `rule_based` = −0.784 [−1.160, −0.408].** Also worse; S03 − F = +0.063
[−0.698, +0.825], not demonstrated.

### Verdict — **FAILED**. The mechanism is real, the remedy is not sufficient.

P1 was named decisive and P1 failed, so the entry is FAILED. **Nothing here ships; arm F remains
the rung-3 and rung-4 agent.**

**But the mechanism is confirmed, decisively and twice.** Paired over the five shared training
seeds, raising only the crate reward:

| C10 − C03 | cc @20 k | cc @40 k | rb @20 k |
|---|---|---|---|
| score | **+0.925** [+0.408, +1.443] | **+1.324** [+0.542, +2.106] | **+0.826** [+0.103, +1.549] |
| crates | **+11.5** [+5.3, +17.8] | **+9.3** [+4.4, +14.1] | **+12.1** [+5.4, +18.8] |

and the decision margin — the quantity the earnings/cost story predicts — orders exactly as the
ratio does (30 rounds, 3 seeds each, `scratchpad/audit2/d6_margins.py`):

| | earn/cost | margin med | Q(chosen) med | frac < 0.01 |
|---|---|---|---|---|
| C03 | 0.70 | 0.0002–0.0005 | 2.0 | 0.88–0.93 |
| C10 | 1.41 | 0.005–0.059 | 4.8 | 0.30–0.53 |
| S03 | 1.53 | 0.003–0.244 | 5.0 | 0.005–0.59 |
| **arm F** | — | **0.639** | 4.96 | 0.012 |

**So the audit's diagnosis was right and E25/E26's was wrong**: the collapse is the reward
magnitudes, the step cost is action-independent, and restoring the ratio restores both the value
level (Q 2.0 → 5.0) and the decision margin (250× to 1 000×). It simply does not restore *enough*
— the best arm is still 1.2 points short of a table that was never trained on this rung at all.

### Predictions

| # | claim | outcome |
|---|---|---|
| 1 | **primary**: C10 beats arm F on cc | **REFUTED**, −2.383 [−2.729, −2.036] |
| 2 | C10 beats arm F on `rule_based` | **REFUTED**, −0.784 [−1.160, −0.408] |
| 3 | C03 margin < 0.01 **and** C10 > 0.05 | **split**: C03 confirmed (3/3, ~0.0003); C10 clears 0.05 in only **1 of 3** — the threshold was miscalibrated, the *ordering* is exact |
| 4 | S03 − C03 > 0, CI excluding 0, S03 **between** C03 and C10 | **not demonstrated on the primary field** (+2.134 [−0.504, +4.772]); demonstrated on `rb` (+1.673 [+0.641, +2.706]); "between" **refuted** — S03 is *above* C10 |
| 5 | C10 retains ≥ 75 % of score from 20 k → 40 k | **PASS**, and my expectation was wrong: C10 retains **124.7 %**, S03 83.9 %, C03 103.4 %. The pilot's margin decay did not become a score decay |
| 6 | suicides ≤ 0.60, crates ≥ 20 | suicides ✓ (0.13–0.50); **crates ✗ for C10** (17.97), ✓ for S03 (20.03) |

P5 deserves a note: I wrote that it was "the one I expect to bite" and it did not bite at all.
The 6 000-episode pilot's margin decay (0.704 → 0.323) was **not** the start of a collapse; more
training helped every arm. That is the second time this rung that a pessimistic extrapolation
from a short pilot was wrong.

### Two things not to over-read, recorded so a later entry cannot mistake them for results

1. **S03's mean is one seed.** Per-seed cc @20 k: **6.707**, 2.673, 1.630, 1.783, 2.137. Seed 50
   alone beats arm F on *both* fields (6.707 vs 4.160; rb 3.700 vs 2.840, `won` 0.347 vs 0.167)
   and has by far the healthiest table (margin 0.244, only 0.5 % of steps below 0.01). The other
   four sit near 2. Reporting S03 as "matching arm F" would be the E23 error exactly: a mean of
   five carried by one cell. **Its CI [0.36, 5.62] is the honest summary.**
2. **`won` on `rule_based` is the one metric where a trained arm consistently beats arm F**, and
   it was **not** pre-registered. S03: 0.347 / 0.277 / 0.213 / 0.217 / 0.297 — **all five seeds
   above F's 0.167** — at a score of 2.903 against F's 2.840. `MEASUREMENT.md` says `won` matters
   more than mean score on rung 4, which makes this interesting and *not* claimable here. It is a
   hypothesis for E28 with a prediction written first, not a result of E27.

### What this settles, and what is left

- The rung-3 training collapse **has a known cause and a partial fix**. That is a real result for
  the report even though the arm lost: three entries blamed coverage, semantics, features and the
  learning rule, and the answer was one constant in a reward table calibrated for a board the
  agent had to itself.
- **A trained rung-3 agent still does not beat an untrained one.** Across E25, E26 and E27 —
  twenty-five training runs, four reward configurations, two feature maps — nothing has beaten
  the frozen rung-2 table read through the rung-3 map.
- **Remaining budget should go to rung 4 on the frozen table**, not to a fourth attempt at rung-3
  training, unless E28's `won` hypothesis survives its own pre-registration.

- **Verdict:** **FAILED** on its pre-registered primary. Mechanism confirmed (paired C10 − C03
  +0.93 score, +11.5 crates, margins 250× apart); remedy insufficient (best arm −1.17 against a
  table that never trained here). Arm F stays the ship.

---

## E26 — Give the agent an objective for the half of the round the board is empty

- **Question:** E25's audit (`scratchpad/audit/`) moved the diagnosis off both of my hypotheses.
  What actually drains the rung-3 reward stream is **`INVALID_ACTION` at −24.25 per round** —
  more than coins and crates earn together, and sitting unpriced in E24's own results table.
  96.3 % of it is `BOMB` pressed with no bomb available, 95.6 % of those in rows that *have*
  value (median margin 0.116, so learned rather than tie-breaking), and **99 % in exactly two
  rows, both with digit 6 = `NO_TARGET`**. `NO_TARGET` is common because four agents strip all
  122 crates by ~step 140: for the last two thirds of the round the state carries **no objective
  at all** (26.3 % of safe steps in the `coin_collector` field, against 0.43 % solo).

  The second fact is arithmetic. The 9 coins are shared four ways — a ~2.25 fair share, and the
  rung-2 table already banks 2.18. Since `score = coins + 5·kills`, **on rung 3 every further
  point of score has to come from kills**, and `KILLED_OPPONENT` is currently worth 0.

- **Change:** two, both behind switches that default to the pre-E26 behaviour.
  1. **`BM_HUNT`** (`callbacks.py`): digit 6 falls through to the nearest **opponent** when no
     coin and no crate is reachable, and digit 7 counts an opponent in blast range as a reason
     to bomb. **`FEATURE_SIZES` is unchanged** at `(4,4,4,4,5,5,2,5)` — no digit added, inserted
     or re-based — so `q_table_rung2ship` stays a valid parent at `factor = 1` and the parent's
     "walk that way" values are already the right prior for the repurposed rows.
  2. **Reward semantics** (`train.py`): `KILLED_SELF = 0`, `GOT_KILLED = −5`, so death is priced
     once. Provably identical to rung 2 on an empty board. `BM_KILL` adds `KILLED_OPPONENT`,
     **25** in arm H — the game's own 5:1 kill:coin ratio at this table's scale (E16 put the
     agent's coin at 5). A rescaling of a real game event, not an invented one.

- **No custom events, deliberately.** `MOVED_TOWARD_OPPONENT`, `BOMB_NEAR_OPPONENT`,
  `ESCAPED_BLAST` and friends were considered and rejected: all are action-dependent, all reward
  the attempt rather than the outcome, and all inject within-row noise of the same size as the
  margins they would widen (E19 measured sd 0.91). A hunting signal belongs in the **state**,
  where it is a fact about the board, not in the reward, where it is a bribe.

### Disclosure: I measured the untrained HUNT table before writing these predictions

Verifying that the switch was inert when off, I ran 100 rounds of both settings on the **frozen
rung-2 table, no retraining at all**, seed 550731, `coin_collector` field:

| | HUNT off | HUNT on |
|---|---|---|
| score | 2.610 | **4.340** |
| kills | 0.090 | **0.480** |
| invalid | 26.89 | **1.25** |
| crates | 28.06 | 26.40 |
| survived | 0.470 | 0.460 |
| think_max_ms | 0.285 | 0.318 |

**The feature change alone, with zero training, nearly doubles score and beats
`rule_based_agent`'s 2.753 in the same slot.** `invalid` falls 95 %, which is the mechanism the
audit predicted, at the magnitude it predicted. These predictions are therefore written *knowing
this*, which is disclosed rather than hidden — the trained arms are still unmeasured, and this
number is 100 rounds on one seed.

It also changes the design: **the untrained HUNT table becomes an arm of its own**, because E25's
lesson is that training in this field can make things worse, and a change that works without
training must be tested against training rather than assumed to benefit from it.

- **Design.** Primary metric **`score`**, 3× `coin_collector_agent`, 300 rounds at ε = 0, seed
  550731, n = 5 training runs (`BM_RUN_INDEX` 20–24, never used). Primary checkpoint **20 000**,
  pre-registered — both E25 arms halved on score between 20 k and 40 k, as did rung 2.

  | arm | training | `BM_HUNT` | `BM_KILL` |
  |---|---|---|---|
  | **F** | **none** — frozen `q_table_rung2ship` | 1 | — |
  | **S** | 40 000 in the cc field | 0 | 0 |
  | **H** | 40 000 in the cc field | 1 | 25 |

  Arm S is "E25 arm B done correctly" and is required, because E25's contrast was void.
  Everything else — ε 0.2→0.02, γ 0.99, α 1/N^0.7, `WARM_N` 100 — is **unchanged**, because the
  audit refuted the premise that any of them is the problem.

### Prediction (written before the trained runs, after the arm-F observation above)

1. **P1, primary and decisive.** Arm H at 20 000 beats the floor's 2.517 on `score`, paired CI
   over seeds excluding 0. **Refutation:** CI includes 0 → the hunt bundle does not pay once
   trained, and arm F ships instead. *If P1 fails the entry is FAILED regardless of P2–P6.*
2. **P2, the one I expect to lose.** Arm H beats **arm F** (4.340 on 100 rounds). Training in
   the field should add on top of the feature. **Refutation:** H ≤ F → training in company is
   actively harmful even with the objective fixed, the rung-3 agent is a *frozen rung-2 table
   with a rung-3 feature map*, and no further training happens on this rung.
3. **P3, mechanism.** `invalid` in both HUNT arms below **5.0**/round, from 25.3. Already 1.25
   untrained. **Refutation:** ≥ 15 → the `NO_TARGET` diagnosis is wrong and the aliasing needs
   an appended digit 9 = `bomb_possible` (radix 2, `np.repeat(parent, 2)`), which is E27.
4. **P4.** `kills` ≥ 0.30/round in arm H (floor 0.063, reference 0.150, arm F 0.480 untrained).
   **Refutation:** < 0.10 → pricing kills at 25 and pointing at opponents does not produce them,
   and rung 3's ceiling really is the coin fair share.
5. **P5, guard.** Arm S alone does **not** recover the floor: score < 1.5. The −10/−5 error was
   not what broke E25. **Refutation:** S ≥ 2.5 → the semantics error was the whole story and the
   feature change must be ablated before anything ships.
6. **P6, guard.** `suicides` ≤ 0.30 **and** `survived` ≥ 0.40 in arm H, and `crates` ≥ 15.
   **Refutation:** any breaks → aggression has cost the escape policy or overwritten crate
   behaviour, i.e. E25's collapse with a new cause.

`think_max_ms` must be re-measured **serially**; the batch figure is void at 5-way parallelism.

### Considered and rejected, with the reason (so the report has the negative decisions too)

- **Tuning ε_start or `WARM_N`** — E25's two named levers. Struck: ε = 0.2 kills the shipped
  table in 44 steps *with no opponents*, and crates/step *rises* to 0.25 through episode 1 000.
  There is no first-1 000-episode erasure to protect against.
- **Potential-based shaping** — E19 rejected it structurally, not for want of tuning: within-row
  sd(Φ) = 0.91 against action margins of the same size. An opponent-distance potential varies
  *more* within a row than crate distance does, so it is strictly worse than the version already
  refuted.
- **Appending digit 9 = `bomb_possible`** — would fix the invalid-`BOMB` aliasing directly, but
  doubles the table to 128 000 rows against a map that practises 2 364 of 64 000. The digit-6
  change removes the same invalid actions for free (measured: 26.89 → 1.25 untrained). Held as
  **E27, conditional on P3 failing** — which is what P3 is for.
- **Opponent-position digits** (relative bearing, "opponent adjacent", opponent count) — same
  sparsity argument; E20/E21 measured what a single extra digit costs before a warm start exists
  to fill it.
- **Training against `rule_based_agent`** — it is the rung-4 *measurement* target; training on it
  overfits one deterministic policy, and the floor survives only 0.143 there, a worse learning
  signal than the cc field.
- **Training longer** — 40 000 halves score against 20 000 in **both** E25 arms; E18 measured the
  same on rung 2.
- **Fixing the double terminal update in this entry** — real and worth fixing, but it changes the
  learning rule and would confound the feature arm. Its own entry, under a byte-identity control,
  or in every arm at once so it cancels.
- **Custom events** — see above; the reward table stays a map of events the tournament also
  generates, at the game's own ratios.

### Results (300 rounds, ε = 0, seed 550731, n = 5 runs per trained arm)

`coin_collector` field, primary checkpoint 20 000. Arm F is a **single frozen table**, so it
carries a per-round bootstrap CI; S and H carry t-intervals over five training runs. The two are
not interchangeable and are not differenced against each other as if they were.

| | **F** (frozen, no training) | S (trained) | H (trained) |
|---|---|---|---|
| **score** | **4.160** [3.74, 4.62] | 0.177 [0.12, 0.23] | 0.829 [0.60, 1.06] |
| kills | 0.420 | 0.004 | 0.063 |
| crates | 26.63 | 0.300 | 5.556 |
| invalid | 1.46 | 20.34 | 1.559 |
| survived | 0.400 | 0.926 | 0.749 |
| won | 0.370 | 0.003 | 0.058 |

**P1: arm H − floor = −1.688 [−1.916, −1.461].** Not "not demonstrated" — decisively worse.
Same at 40 000 (−1.787) and in the transfer field (−1.537).

### Verdict — **FAILED as pre-registered**, and the failure produced the rung-3 agent

P1 was named as decisive and P1 failed, so the entry is FAILED regardless of the rest. **The
trained arms are not shippable.** But the control that was added only because E25 had made me
suspicious of training at all is the result:

**Arm F — the feature change on the frozen rung-2 table, with no rung-3 training whatsoever.**
Confirmed on the held-out **ship seed 990731, 1000 rounds, run serially**:

| frozen table, `coin_collector` | HUNT off | HUNT on |
|---|---|---|
| **score** | 2.593 | **4.188** |
| kills | 0.088 | **0.420** |
| coins | 2.153 | 2.088 |
| crates | 27.19 | 26.24 |
| **invalid** | **24.21** | **1.43** |
| won | 0.228 | **0.367** |
| think_max_ms | 0.252 | **0.321** |

**Paired over 1 000 identical arenas: +1.595 [+1.342, +1.849].** It clears `rule_based_agent`'s
2.753 in the same slot. Against `peaceful_agent` it scores **17.93 and wins 100 % of rounds**;
against `rule_based_agent` 2.757 with `won` 0.170.

Every point of that comes from the mechanism the audit identified: `invalid` **24.21 → 1.43**, a
94 % drop, because digit 6 now has an answer for the two thirds of the round after the board is
stripped. Coins and crates are *unchanged* (2.15 → 2.09, 27.2 → 26.2) — the agent did not get
better at the rung-2 game, it stopped burning a reward per step on an impossible `BOMB`, and the
freed steps became kills (0.088 → 0.420).

**This is the first change in the project to move the primary metric by fixing a feature rather
than a reward or a hyperparameter**, and it needed no training at all.

### Predictions

| # | claim | outcome |
|---|---|---|
| 1 | **primary**: arm H beats the floor | **REFUTED**, −1.688 [−1.916, −1.461] |
| 2 | arm H beats arm F | **REFUTED** — 0.829 vs 4.160. I flagged this as the one I expected to lose |
| 3 | `invalid` < 5 in both HUNT arms | **correct** — F 1.46, H 1.56 (S, without the feature: 20.3) |
| 4 | kills ≥ 0.30 in arm H | **refuted** — 0.063 (though arm F reaches 0.420) |
| 5 | arm S alone does not recover the floor (< 1.5) | **correct** — 0.177. The −10/−5 semantics error was never the cure |
| 6 | H: suicides ≤ 0.30, survived ≥ 0.40, crates ≥ 15 | **partly refuted** — 0.152 ✓, 0.749 ✓, **crates 5.6** ✗ |

P3 and P5 together are the load-bearing pair: the invalid-action collapse happens **with the
feature and without training** (F) and fails to happen **with training and without the feature**
(S). That is the cleanest attribution in the ledger — one factor, both directions.

### Is this still machine learning? — the check the result demands

`AGENTS.md`'s hardest rule is that the model must *learn from* the features and that a feature
returning "the best action" is forbidden. An agent whose table was trained only on rung 2, now
carried by a digit that points at opponents, has to answer that. So
`scratchpad/benedict/feature_only_probe.py` builds the strongest policy the same digits allow —
escape if in danger, bomb if digit 7 fires, else walk digit 6 — and plays it in identical arenas:

| same table, same arenas, 300 rounds | score | kills | crates | survived |
|---|---|---|---|---|
| **learned** (Q-table reads the digits) | **4.377** | 0.460 | 26.20 | **0.417** |
| **features-only** (greedy on the same digits) | **0.030** | 0.000 | 4.25 | **0.000** |

**The purely-feature policy dies in every single round.** The features are necessary and nowhere
near sufficient: 146× on score, and the whole of survival. What the table contributes is *when
not to bomb* — a judgement no digit encodes, and the one thing rung 2 spent 20 000 episodes
learning. The digit says where the target is; the table decides whether going there is worth it.

### What this means for rung 3

**The rung-3 agent is the rung-2 table read through a rung-3 feature map, with no rung-3
training.** That is an unusual thing to ship and it is a *finding*, not a shortcut: two
independent experiments (E25, E26) now show that training in an opponent field destroys a
competent policy, and E26 isolates the cause well enough to say the field is not the problem
either — arm S trains in the same field with the same rewards and merely fails to improve, while
arm H trains with the *working* feature and loses 3.3 points of score against not training at all.

Open, in order:
1. **Flip `BM_HUNT`'s default to on** in `callbacks.py` so a bare checkout plays the rung-3
   agent. Held back deliberately until the ship-seed confirmation existed; it now does. *(Done,
   commit `9247948`.)*
2. **Why does training destroy it?** — see the correction below.
3. Rung 4 (`rule_based_agent`) is at 2.757 with `won` 0.170 — measured, not yet worked on.

> **Correction added 2026-08-13, from the second independent audit (`scratchpad/audit2/`).**
> Item 2 above framed the rung-3 question as a *training* question — "the learning rule itself,
> including the double terminal update, or 40 000 episodes of a dying policy at α = 1/N^0.7".
> **Both named suspects are dead and the framing was wrong.**
>
> - **The double terminal update fires in 100 % of rung-2 episodes** (rung 2 always survives) and
>   only 6–38 % of rung-3 ones. It is 1 update in ~397 and it produced the healthiest table in
>   the project. Real bug, wrong direction to explain rung 3.
> - **α = 1/N^0.55**, i.e. more plasticity, scores 0.987 against 1.393 for the unchanged
>   schedule. Not the lever. Cold start (no warm start at all) is 1.013 — also not the lever, and
>   with an *identical* margin collapse, which answers "is starting from a competent table in a
>   harder field actively worse" with **no**.
> - **The lever is the reward magnitudes**, the one axis E26 never varied. `CRATE_DESTROYED`
>   0.3 → 1.0 takes 6 000-episode training from 0.660 to **2.940** on the cc field and 0.953 to
>   **3.107** against `rule_based_agent` (my own replication, fresh seed 60). E27 tests it.
> - **Arm H did not collapse in training.** Its log is flat at 18.8 crates/episode from episode
>   2 000 to 40 000. Reporting only the ε = 0 evaluation was correct; concluding "arm H
>   collapses" was not — it trains a policy that is fine online and unreadable when frozen, which
>   is a *different* failure from arms A/B/S, whose logs do collapse to 1.4.
> - **Part of the collapse is a readout artefact.** `BM_TIE_TOL = 0.001` on arm H's frozen table
>   recovers **3.74 → 19.11 crates** — exactly what its training log records — while arm F, whose
>   margins are 0.669, is unaffected (4.115 → 4.235). So "every action really is worth the same,
>   and there is no gradient back to competence" (E25) is too strong: the ordering survives, it
>   is below the resolution of a deterministic argmax. `TIE_TOL` is a **diagnostic, never a fix**
>   — it triples a broken table and does nothing for a healthy one, i.e. it masks the symptom.
> - **Presentational:** P1's −1.688 is against the HUNT-**off** floor 2.517, while the results
>   table two lines above shows arm F at 4.160. Both are called "the floor" in this entry. The
>   gap to arm F is −3.331.

- **Verdict:** **FAILED** on its pre-registered primary. The control arm ships: **+1.595 score
  [+1.342, +1.849]** over the rung-2 floor on 1 000 held-out arenas, at 0.321 ms.

---

## E25 — Train in an opponent field, and tell the agent that being killed is bad

- **Question:** E24 established the floor and named the cause. Two of every three rung-3 deaths
  land in an **all-zero Q row** — 67.9 % against `coin_collector` and 60.5 % against
  `rule_based`, on base rates of 1.35 % and 3.44 % — so at the moment it dies the agent is
  choosing uniformly at random among six tied zeros. A row that was never updated cannot be
  repaired by a better feature; only by visiting it. That makes **training distribution**, not
  feature engineering, the first thing to change on this rung.

  Riding along is a plain inconsistency in the reward table that could not fire before, because
  rung 2 was trained with `--opponents none`:

  | event | reward | |
  |---|---|---|
  | `KILLED_SELF` | −5 | rung 1, load-bearing (E17) |
  | `GOT_KILLED` | **absent, i.e. 0** | never fired without opponents |

  Under that table, walking into *your* blast is free and walking into *mine* costs 5. Since
  `killed_by` is 0.340–0.453 per round, that is not a detail.

- **Change:** a new agent, `agent_code/benedict_task3/`, copied from `benedict_task2` at
  `<commit>`. The feature map is **untouched** — same eight digits, same 64 000 rows — so the
  shipped rung-2 table is a same-layout parent and `warm_start` transfers it row for row
  (`factor = 1`). Two things differ from rung 2: the training line-up has opponents in it, and
  `REWARDS` gains `e.GOT_KILLED` behind `BM_GOT_KILLED`.

- **Design.** The arms separate the two, because the death penalty is worthless without the
  field and the field may be sufficient without the penalty:

  | arm | `BM_GOT_KILLED` | isolates |
  |---|---|---|
  | baseline | — | the shipped rung-2 table, unretrained: **E24** |
  | `A` | 0 | training distribution alone |
  | `B` | −5 | training distribution + a symmetric death penalty |

  Five training seeds per arm, `BM_RUN_INDEX` **10–14** (never used on this project), 40 000
  episodes, warm-started from `q_table_rung2ship` with `WARM_N = 100`. Training field is
  **3× `coin_collector_agent`** — the ladder's hard rung-3 opponent, it bombs (and foreign
  bombs are ~90 % of the damage), but it does not hunt, so early training is not dominated by
  being out-played before anything is learnt. Measured cost 9.11 s per 200 episodes, so ~30 min
  a run; ten runs fit on the ten cores at once.
- **Measurement:** 300 rounds at ε = 0 on the **validation** seed **550731**, held out from
  E24's dev seed, at both the 20 000 and 40 000 checkpoints, in the training field
  (`coin_collector`) and in the transfer field (`rule_based`). Opponents are seeded per round
  (E24), so these are reproducible; the *training* runs are not — `main.py` does not seed the
  provided agents and nothing in `setup_training` can precede their own reseed. That is why
  every arm gets five seeds and is compared at the arm level, never on a single run.

  **Pre-registered before any table existed**, because E23 nearly reported a selected cell:
  - **The primary checkpoint is 40 000.** The mechanism under test is filling rows that were
    never visited, and that needs episodes. Rung 2's finding that 20 000 beats 40 000 was made
    on a saturated board with no opponents and does not transfer. The 20 000 point is reported
    as a curve, **not** selected over.
  - **The unit of analysis is the run, n = 5 per arm**, not the round. Arms A and B share
    `BM_RUN_INDEX` seed for seed, so B − A is a *paired* difference over five pairs.
  - The transfer field (`rule_based`) is a **secondary** measurement. Nothing is chosen on it;
    it only says whether the effect generalises off the training opponent.
  - `think_ms` from this batch is **void** — the evaluations run five at a time, which inflates
    per-step timing. The 0.5 s check must be re-run serially before anything ships.

### Prediction (written before the run)

1. **Arm A alone closes most of the survival gap in the training field.** The all-zero rows get
   visited, so the random-choice-at-death mechanism disappears whether or not deaths are priced.
   Predict arm-mean survival ≥ 0.60 against `coin_collector`, from the 0.433 floor, and the
   all-zero share at the fatal step to fall below 25 %. **Refutation:** survival stays within
   0.05 of the floor → visiting the rows is not sufficient and the state cannot express the
   situation, which promotes features (E26) over distribution.
2. **Arm B beats arm A on survival, and the paired arm difference clears zero.** −5 for
   `GOT_KILLED` prices the 0.34 deaths/round that currently cost nothing. Predict
   B − A ≥ +0.05 survival with a CI excluding 0 over the five seeds. **Refutation:** the CI
   contains 0 → the field does the work and the penalty is decoration, which is the cheaper
   result and worth knowing.
3. **Crates fall in both arms, and that is not a regression.** Rung 2's 116.6 came from 400
   uncontested steps. Predict 26.9 → no more than 35 in the training field: the board is shared
   four ways and the ceiling is ~30 per agent, which is what the three `coin_collector`
   opponents already achieve (31.6–32.4). **Refutation:** crates rise above 40 → I have
   mis-read the shared-board ceiling.
4. **`suicides` does not creep up.** The guard from rung 2. Both arms ≤ 0.227, the floor's
   value. **Refutation:** either arm above 0.30 → learning to live with opponents has cost the
   own-bomb escape, and `KILLED_SELF` needs rebalancing against `GOT_KILLED`.
5. **Kills stay near the floor in both arms.** Neither arm rewards a kill, and E24 showed kills
   are a by-product of bombing crates (1.893 against `peaceful` with no aggression term).
   Predict ≤ 0.15 in both — deliberately, so that `KILLED_OPPONENT` in E26+ has a clean
   baseline. **Refutation:** kills > 0.25 → surviving longer alone buys aggression, and the
   aggression term may never be needed.
6. **Transfer to `rule_based` is positive but much smaller.** Predict survival ≥ 0.25 there
   (floor 0.143) for the better arm — real movement, still far from the reference's 0.400.
   **Refutation:** transfer survival ≤ 0.16 → training against a non-hunting opponent does not
   generalise, and E26's field choice becomes the question rather than its features.

### Results (commit `0a54eb9` + the E25 agent, 300 rounds, ε = 0, seed 550731, n = 5 runs/arm)

The floor is the shipped rung-2 table **re-measured on 550731**, because E24's numbers are on
the dev seed and differencing across world seeds would have confounded everything. It lands
within 0.06 of E24 on every metric, which is a useful check that the floor is a property of the
agent and not of the seed.

**Training field (`coin_collector`), primary checkpoint 40 000:**

| | floor (rung 2) | arm A (`GOT_KILLED`=0) | arm B (=−5) |
|---|---|---|---|
| **score** | **2.517** | **0.081** [0.043, 0.118] | **0.118** [0.079, 0.157] |
| coins | 2.183 | 0.057 | 0.105 |
| crates | 27.34 | **0.119** | **0.138** |
| bombs | 27.98 | 7.54 | 24.47 |
| survived | 0.437 | 0.309 | **0.940** [0.917, 0.963] |
| killed by opp. | 0.343 | 0.677 | 0.039 |
| suicides | 0.220 | 0.014 | 0.021 |
| steps alive | 229.0 | 193.1 | 384.0 |

Transfer field (`rule_based`), arm B: survived **0.650**, score 0.273, crates 0.377.

**The 20 000 checkpoint, which this entry pre-registered as a curve and then failed to report
until the audit caught it:**

| | A@20k | A@40k | B@20k | B@40k |
|---|---|---|---|---|
| score | 0.159 | 0.081 | **0.220** | 0.118 |
| survived | **0.853** | 0.309 | 0.922 | 0.940 |

Both arms **halve on score between 20 k and 40 k**, and arm A@20k's survival of 0.853 satisfies
prediction 1's ≥ 0.60 at the checkpoint I left out. It does not change the verdict — 0.220 is
still 11× below the floor — but omitting a pre-registered quantity that flatters nothing is
still a selective report, and it is the third time in this ledger (E21's unevaluated 300 k
tables, E23's four-cell pooling) that a missing cell made a conclusion tidier.

### Verdict — **FAILED**, and for two independent reasons, one of them mine

**The agent learned to survive by refusing to play.** Arm B survives 94 % of rounds — better
than `rule_based_agent`'s 0.693 — while scoring **0.118 against the floor's 2.517**. It opens
**0.14 crates**. Rung 2's whole capability is gone.

**1. The reward semantics were wrong, and the error is mine.** `environment.py:264` adds
`GOT_KILLED` to *every* agent killed by a blast, and `:251` adds `KILLED_SELF` **on top** when
the bomb was its own. So arm B does not price death symmetrically at −5/−5 as the design says —
it prices a **suicide at −10** and an opponent's kill at −5. The correction is now in
`AGENTS.md`: put the whole penalty on `GOT_KILLED` and leave `KILLED_SELF` at 0, which is
*identical* to the rung-2 table on an empty board (a suicide fires both events there too) and
correct in company. **Arm B's numbers do not measure what this entry says they measure.**

**2. But the reward error is not what broke it.** Arm A carries the **untouched rung-2 reward
table** and collapses just as completely: 27.34 → 0.119 crates. The destroyer is *training in
the field*, not the death penalty.

> **Correction, from the independent audit of this entry (`scratchpad/audit/`).** I first wrote
> here that "the warm-started table meets rung 2's ε = 0.2 in a field that kills it in 40 steps,
> every single round", and that ~1 000 episodes of that erased the transfer. **Both halves are
> wrong, and I verified the refutation myself before committing this.**
>
> **(a) It is ε, not the field.** Rolling the *frozen* shipped table at fixed ε, no learning,
> 40 rounds (`scratchpad/audit/a4_eps.py`):
>
> | ε | field | steps/ep | suicides | survived | crates |
> |---|---|---|---|---|---|
> | 0.0 | solo | 400.0 | 0.000 | 1.000 | 116.72 |
> | 0.05 | solo | 120.3 | 0.975 | 0.025 | 38.85 |
> | **0.2** | **solo** | **44.2** | **1.000** | **0.000** | 14.57 |
> | 0.0 | 3× coin_collector | 214.0 | 0.400 | 0.375 | 27.05 |
> | 0.2 | 3× coin_collector | 32.0 | 0.975 | 0.000 | 10.88 |
>
> ε = 0.2 destroys the table **on an empty board**. The opponents are worth ~12 of the 44 steps.
> And ε 0.2 → 0.02 is *rung 2's own schedule*, which produced this table.
>
> **(b) The crate machinery is intact early and decays late.** Crates per **step** (the absolute
> count falls only because episodes are shorter):
>
> | episodes | ε | steps | crates | **crates/step** |
> |---|---|---|---|---|
> | 0–50 | 0.198 | 40.5 | 11.14 | **0.275** |
> | 500–1 000 | 0.138 | 50.8 | 12.64 | **0.249** |
> | 2 000–5 000 | 0.038 | 186.9 | 10.32 | 0.055 |
> | 10 000–20 000 | 0.020 | 287.2 | 2.92 | 0.010 |
> | 30 000–40 000 | 0.020 | 290.1 | 1.43 | **0.005** |
>
> At episode 1 000 it opens crates at **0.25/step, twice the floor's greedy 0.119**. The collapse
> runs from ~2 000 to 40 000, *after* ε has reached its floor, and is monotone. It is **slow
> forgetting under a bad gradient, not fast erasure by exploration** — and I read a falling
> absolute count as decay when the rate was rising. The table that contains the counter-evidence
> is the one I quoted.

**What actually drains the value.** Decomposing the floor policy's realised reward per round in
the `coin_collector` field:

| term | per round |
|---|---|
| `COIN_COLLECTED` +5 × 2.23 | +11.15 |
| `CRATE_DESTROYED` +0.3 × 27.67 | +8.30 |
| **`INVALID_ACTION` −1 × 24.25** | **−24.25** |
| `STEP_COST` −0.1 × 225.7 | −22.57 |
| `KILLED_SELF` −5 × 0.24 | −1.20 |

**`INVALID_ACTION` costs more than everything the agent earns**, and it sat in E24's own results
table one row below `crates` without being priced. **96.3 % of it is `BOMB` pressed with no bomb
available**, 95.6 % of those in rows that *have* value (median margin 0.1156 — learned, not
tie-breaking), and **99 % in exactly two rows, both with digit 6 = `NO_TARGET`**.

And `NO_TARGET` is common because **the board runs out**: four agents strip all 122 crates by
~step 140, after which the rung-2 state has **no objective at all** — 26.3 % of safe steps in
the `coin_collector` field against 0.43 % solo. The rung-2 table's answer in that row is "press
BOMB", at −1 a time, for the last two thirds of the round.

**The value function flattened.** `scratchpad/benedict/policy_probe.py`, 30 rounds, same field:

| | rung-2 table | arm B @40 k |
|---|---|---|
| Q(chosen), median | **+4.822** | **−1.177** |
| Q(BOMB) − Q(2nd), median, when BOMB wins | **0.1156** | **0.0003** |
| …of those, below 0.01 | 0.0 % | **99.8 %** |
| all-zero rows, share of steps | 1.54 % | **0.00 %** |
| BOMB chosen with a crate in range | 24.7 % | **0.5 %** |

*(An earlier version of this table carried "next to a crate (digit 7 = 1): 9.45 % vs 34.39 %"
and a sentence built on it. **Withdrawn** — digit 7 is `have_bomb AND bomb_hits_crate`, so it
measures bomb *possession*, not position: the floor holds a bomb on 26.6 % of steps because it
spends them, arm B almost always. Normalised by possession, both are ~31–34 % and the contrast
disappears.)*

So: **E24's diagnosis was right and the fix worked.** All-zero rows at the fatal step were the
problem; training in the field drove them from 1.54 % of steps to 0.00 %, and rows with value
went 2 364 → 4 931. It simply did not help, because the states are now *populated with
worthless values*: every decision is made on a margin of 3 × 10⁻⁴, 385× smaller than rung 2's
0.116. The agent stands next to a crate on a third of all steps and almost never bombs it, then
drops 24 bombs a round into empty space — not from ties (0.01 % of steps) but as the confident
argmax of a flat, negative value function. With no reward flowing, every action really is worth
the same, and there is no gradient back to competence. A self-reinforcing local optimum.

### Predictions — four of six "correct" while the agent got 20× worse

| # | claim | outcome |
|---|---|---|
| 1 | arm A closes the survival gap; all-zero < 25 % | **split**: all-zero → 0.00 % (right, decisively); arm A survival 0.437 → 0.309 (wrong) |
| 2 | B − A survival ≥ +0.05, CI excludes 0 | **numerically right** (+0.631 [+0.587, +0.676]) but **void** — confounded by the −10/−5 error |
| 3 | crates fall, no more than 35 | **refuted catastrophically** — 0.12, not 27 |
| 4 | suicides ≤ 0.227 | "correct" (0.014–0.021) and **meaningless**: it does not bomb, so it cannot suicide |
| 5 | kills ≤ 0.15 | "correct" (0.003) and meaningless for the same reason |
| 6 | transfer survival ≥ 0.25 | **correct** — arm B 0.650 on `rule_based` |

**The methodological lesson is the one worth carrying.** Four of six predictions came out
"correct" on an agent that is **twenty times worse on the primary metric**. Every one of those
four was a *guard* metric — suicides, kills, survival, coverage — and guards only bound the
failure modes you already imagined. Predictions 4 and 5 were satisfied by the agent abandoning
the behaviour they were meant to protect. A pre-registered metric set has to state which number
decides, and E25's did not: it named `score` as primary in the ladder and then spent five of six
predictions elsewhere.

### What E26 has to address

1. **Fix the reward semantics** — `GOT_KILLED = −5`, `KILLED_SELF = 0`. Not an arm; a
   correctness fix, and provably identical to rung 2 on an empty board.
2. **Give the agent an objective for the second half of the round.** Digit 6 is `NO_TARGET` on
   26 % of safe steps because the crates are gone by step ~140, and the table's answer there is
   an invalid `BOMB`. Letting digit 6 fall through to the nearest **opponent** costs no new
   digit — `FEATURE_SIZES` is unchanged, so the shipped table stays a valid parent — and the
   parent's values for "walk that way" are already the right prior.
3. **Price the only thing left to earn.** Coins are saturated: 9 shared four ways is a ~2.25
   fair share and the floor already banks 2.18. `score = coins + 5·kills`, so **on rung 3 every
   further point must come from kills**, and `KILLED_OPPONENT` is currently worth 0.

~~Protect the warm start (ε_start, `WARM_N`)~~ — **struck**: the correction above shows there is
no first-1 000-episode erasure to protect against. ~~The crate→coin chain is starved by
opponents~~ — **struck**: measured at `COIN_FOUND` 2.08 vs `COIN_COLLECTED` 2.23 per round, i.e.
the agent collects *more* coins than its own bombs reveal, at a rate above the board's 9/122.2
density. Both were my hypotheses and both are false.

**Bombing is not negative-EV either**, which kills the "the agent is behaving rationally" reading:
for the floor policy, gain per bomb = (0.3 × 27.67 + 5 × 2.23)/27.36 = **+0.711**, suicide cost
0.24 × 5/27.36 = −0.044, **net +0.67**. It only goes negative for an agent that is already
incompetent (arm B at episodes 5 000–10 000: −0.03) — a self-reinforcing local optimum in which
bombing pays iff you are good at it. Decisively, arm B's final policy is worth **−11.19**
discounted under **its own reward table** against **−0.589** for the table it started from: it
found a policy worse than its own initialisation, and worse than standing still (−9.82). That
is an optimisation failure, not a reward-design one, and no retuning of rewards explains it.

**The counterfactual that justifies E26.** Repricing the floor policy's *recorded* event stream
under alternative reward tables (discounted, γ = 0.99, `scratchpad/audit/a3_economics.py`):

| reward table | floor policy, cc field |
|---|---|
| rung-2 table as shipped | **+0.545** |
| …with the invalid-action bleed removed | **+2.439** |
| …and `KILLED_OPPONENT = +25` on top | **+3.08** (cc), **+15.14** (peaceful) |
| arm B's learned policy, under arm B's own table | −11.19 |
| "walk into a blast at step ~3" | ≈ −0.30 |

Two things follow. **The margin between competent play and immediate death is only ~0.6–1.0
discounted units** — the same order as the within-row aliasing noise E19 measured at sd 0.91,
which is why the policy is so easily talked out of playing. And **removing the invalid-action
bleed is worth 4× that entire margin**, from a single change that costs no new state. That is
the largest lever on the board and it is a *feature*, not a reward.

**Method note — this entry was audited by an independent session.** After E25 failed I handed a
fresh agent the repo, the raw CSVs and the trained tables, and asked it to recompute the
headline numbers *before* reading my conclusions and to be adversarial. It confirmed the
semantics error, the evaluation arithmetic and the flattened value function; it **refuted** my
central mechanism and my crate→coin premise, **overclaimed** two more, and found the invalid-action
drain, the board-empties-by-step-140 fact and the double terminal update. I then re-verified its
two load-bearing refutations myself (the ε table and the crates/step table above) before
rewriting this entry. Same pattern as the E19–E22 audit recorded in `benedict_task2.md` §6, and
it changed more verdicts than that one did.

**Latent defect found by the audit, not the cause of anything here but in the shipped training
code.** `end_of_round` is called with the same `last_game_state`, `last_action` and an
*uncleared* event list as the final `game_events_occurred`, so on a round that reaches
`MAX_STEPS` the terminal cell is updated **twice** — once bootstrapped, once not — and the last
step's reward is double-counted into the log. Rung 2 survived 1.000 of rounds, so this fired in
every one of its 20 000 training episodes. Fix separately, under a byte-identity control, or in
both arms at once so it cancels.

- **Verdict:** **FAILED** — survival bought at the cost of the entire scoring policy, on a
  measurement whose arm contrast was invalid. Nothing from E25 ships. The E24 diagnosis is
  *confirmed* (coverage was the problem, and it is now fixed); the remedy as specified is
  refuted.

---

## E24 — What the rung-2 agent does when there are opponents on the board

- **Question:** the opening measurement of rung 3, and deliberately not a change. Rung 3 has
  three holes and I cannot rank them from the armchair:

  1. **No opponent information in the state.** `state_to_features` is the rung-2 map verbatim,
     `TODO task 3+` untouched.
  2. **Kills and deaths by others are worth 0 in the learning signal.** `REWARDS`
     (`train.py:134`) is `{COIN_COLLECTED, CRATE_DESTROYED, INVALID_ACTION, WAITED,
     KILLED_SELF}` — no `KILLED_OPPONENT`, no `GOT_KILLED`. The *game* pays +5 for a kill; the
     agent is never told.
  3. **The danger digits assume a static board.** Digit 6's exit direction is computed from
     bombs and walls only, so an opponent standing in the escape corridor is invisible.

  Which of these binds depends on whether the failure mode is *death* or *passivity*, and those
  need opposite fixes. E09 is the precedent: the rung-1 → rung-2 transfer floor was 0.00 crates,
  and knowing that before building anything is what set rung 2's direction. The same move costs
  ~20 minutes here and no training at all.

- **Change:** none. Frozen `q_table.npy`, ε = 0, no `BM_*` set. This measures what the rung-2
  policy *does*, not what it could learn.

- **Three facts established before the run**, because they drive the predictions:
  - `peaceful_agent` (`agent_code/peaceful_agent/callbacks.py:8`) picks uniformly from the four
    moves, never bombs, never waits. Against it, **every bomb on the board is mine**, so its
    deaths are all attributable to me and `kills` measures accidental lethality with the
    aggression term switched off.
  - **Opponents block movement but are invisible to my pathing.** `tile_is_free`
    (`environment.py:121`) counts other agents *and* bombs as obstacles, while `bfs_first_step`
    searches `field` alone. A move onto an occupied tile is silently converted to a wait.
  - Rung-2 reference for the same table (1000 rounds, seed 990731): score **8.474**, crates
    **116.57**, bombs **45.27**, suicides **0.000**, survived **1.000**.

- **Design.** Two agents in the same slot × three opponent fields, 300 rounds, seed 20260731,
  ε = 0. The reference is `rule_based_agent` put in *my* slot against the identical field, so
  the comparison is paired arena for arena rather than against a number from another setup.

  | field | `--opponents` | what it isolates |
  |---|---|---|
  | 3× `peaceful_agent` | `peaceful` | pure survival + accidental kills; no foreign bombs |
  | 3× `coin_collector_agent` | `coin_collector` | competition for crates and coins, foreign bombs |
  | 3× `rule_based_agent` | `rule_based` | the rung-4 opponent, measured early for the gap |

- **Measurement:** `--preset task3` plus `killed_by` and `bombs`. Results in
  `results/eval/task3_opponents/` (mine) and `results/eval/baselines/` (the reference).

### Prediction (written before the run)

1. **Against `peaceful` the agent scores *higher* than its rung-2 8.47, from kills it never
   learnt to want.** 45.27 bombs per round, each blast covering ~10 tiles for 2 steps, is roughly
   900 tile-steps of lethal board against ~150 reachable free tiles over 400 steps — a random
   walker should not survive that. Predict **kills ≥ 1.0**, survived ≥ 0.95, and score ≥ 12
   (coins hold near 8.5 because `peaceful_agent` collects none). **Refutation:** kills < 0.5, or
   survived < 0.90.
2. **`invalid` rises in every field, and orders with how much the opponents are in the way.**
   Blocked steps are the direct, measurable consequence of pathing over `field` alone. Predict
   `invalid` above the rung-2 level in all three fields. **Refutation:** `invalid` unchanged
   against `peaceful` — then the blocking is rare enough to ignore and hole 3 is cheap.
3. **`crates` collapses against the two bombing fields, not against `peaceful`.** 122 crates
   shared four ways plus rounds that end early. Predict < 60 against `coin_collector` and
   `rule_based`, and ≥ 100 against `peaceful`. **Refutation:** ≥ 80 in a bombing field.
4. **The dominant death mode is `killed_by_opponent`, and `suicides` stays put.** The escape
   logic is priced by `KILLED_SELF` and measured at 0.000, and nothing about opponents changes
   my own bomb's blast. Predict suicides < 0.02 everywhere, `killed_by_opponent` ≈ 0 against
   `peaceful` and ≥ 0.3 against the bombing fields. **Refutation:** suicides ≥ 0.05 in any field
   — which would mean opponents break my escapes by standing in them, and would promote hole 3
   above hole 2.
5. **The gap to `rule_based_agent` is on `kills`, not on survival.** Predict the reference takes
   ≥ 2× my kills in the `peaceful` field, while my survival is within 0.05 of its. **Refutation:**
   the reference out-survives me by more than 0.1 — then the gap is defensive, not offensive.

**This is the prediction that picks E25.** Killed-by dominant → opponent features first.
Survives-but-never-kills → the reward table first.

### Results (commit `0a54eb9`, 300 rounds, seed 20260731, opponents seeded)

| | rung 2, alone | `peaceful` | `coin_collector` | `rule_based` |
|---|---|---|---|---|
| score | 8.474 | **17.680** | 2.460 | 2.947 |
| coins | 8.474 | 8.213 | 2.143 | 2.530 |
| kills | — | **1.893** | 0.063 | 0.083 |
| suicides | 0.000 | 0.073 | 0.227 | 0.403 |
| killed by opp. | — | 0.000 | 0.340 | 0.453 |
| survived | 1.000 | 0.927 | 0.433 | 0.143 |
| crates | 116.57 | 113.76 | 26.88 | 28.65 |
| invalid | 0.14 | 0.61 | 25.27 | 8.64 |
| steps alive | 399.9 | 389.9 | 225.7 | 163.1 |

`rule_based_agent` in the identical slot (score / kills / suicides / survived), paired arena for
arena: 21.517 / 2.767 / 0.140 / 0.860 (`peaceful`), 2.753 / 0.150 / 0.270 / 0.693
(`coin_collector`), 3.290 / 0.200 / 0.507 / 0.400 (`rule_based`).

**The paired comparison against that reference splits cleanly, and not the way the means read.**

| | `peaceful` | `coin_collector` | `rule_based` |
|---|---|---|---|
| score | −3.837 [−4.520, −3.150] | −0.293 [−0.623, **+0.030**] | −0.343 [−0.693, **+0.010**] |
| kills | −0.873 [−0.983, −0.767] | −0.087 [−0.140, −0.037] | −0.117 [−0.173, −0.060] |
| suicides | −0.067 (better) | −0.043 [−0.113, +0.027] | −0.103 (better) |
| survived | **+0.067** [+0.020, +0.113] | **−0.260** [−0.333, −0.187] | **−0.257** [−0.320, −0.193] |

In both bombing fields the score difference to `rule_based_agent` **includes zero** — I am not
demonstrably behind it on the primary metric — while survival is decisively worse by ~0.26 and
kills by ~0.1. Score parity is bought by being the better crate-and-coin collector for as long
as I stay alive, which is 163 steps against the tournament opponent's 224–240. That is a
different problem from "loses on score", and a more tractable one.

*(A first, pre-fix draw of this measurement — before the opponent-seeding correction below — gave
17.753 / 2.500 / 2.727 on score and 0.920 / 0.373 / 0.153 on survival. Every conclusion here is
unchanged; the two score comparisons were `WORSE` there and `no effect shown` here, which is the
noise the fix removes.)*

**Scorecard.** 1 correct (kills 1.923 ≥ 1.0, score 17.75, coins held; survival 0.920 undershot
the ≥ 0.95 I wrote but cleared the refutation). 2 correct in the number, **wrong in the
mechanism**. 3 correct. 4 **half refuted** by my own clause. 5 **refuted, and it reverses by
field**: 1.44× not ≥ 2×, and I *out-survive* the reference by +0.073 against `peaceful` while it
out-survives me by 0.297 and 0.200 in the bombing fields.

### Two premises in the question section above are wrong

Left standing as written, corrected here:

- **"No opponent information in the state" is false.** `state_to_features`
  (`callbacks.py:344`) already builds `occupied` from `bombs` *and* `others` and passes it to
  `neighbour_status` and `escape_direction`. Only `target_direction` ignores it, and a target
  direction pointing through a body costs a detour, not an invalid action. The data agrees:
  `invalid` against `peaceful` is 0.63 against a rung-2 baseline of 0.14 — half an action per
  round, small *because* the neighbour digits work.
- **"The danger digits assume a static board" is false.** `danger_map` (`callbacks.py:147`)
  iterates over every bomb on the board and the explosion map, and `escape_direction` is
  time-aware. Foreign bombs are already in the state.

So the machinery is present and the agent dies anyway. That moves the diagnosis off "missing
features" and onto the policy — which is what the post-mortem below tests.

### Bodies cost almost nothing; other people's bombs cost everything

`peaceful` isolates one variable, since `peaceful_agent` never bombs: agents as obstacles.
Bodies alone cost 0.073 survival and 2.8 crates. Adding foreign bombs costs a further 0.49–0.78.
**~90 % of the collapse is other agents' bombs, not blocking.**

The second split matters more. Against `rule_based`, `suicides` (0.403) is as large as
`killed_by` (0.453): **half my deaths are my own bomb**, from an agent measured at 0.000 alone.
And the asymmetry against `coin_collector` — same arenas, same bomb population — is stark:
I am killed by bombs 0.340 times per round, each opponent 0.030–0.067. **~7× more often**, while
collecting 0.063 kills to their 0.153–0.167.

### Post-mortem: the deaths are in rows the training never updated

`scratchpad/benedict/death_rows.py`, 100 rounds per field, same arenas. `callbacks.setup`
initialises with `np.zeros`, so an all-zero row is an exact test for "never updated by either
training stage" — **61 636 of 64 000 rows (96.3 %) are all-zero**, i.e. the rung-2 policy
practised 2 364 rows.

The number that matters is the enrichment over the **base rate**, not the raw share — a count is
not a rate, which is the E22 mistake:

| field | all-zero rows, all steps | all-zero rows, **at the fatal step** | enrichment |
|---|---|---|---|
| `peaceful` | 0.10 % | 0.00 % | — |
| `coin_collector` | 1.35 % | **67.86 %** | 50× |
| `rule_based` | 3.44 % | **60.49 %** | 18× |

Rows never visited by the solo ε = 0 rollout: base 0.58 / 2.45 / 5.71 %, at the fatal step
**83.3 / 92.9 / 96.3 %**.

The effect is far larger than the run-to-run noise: three draws of this script (two before the
opponent-seeding fix, one after) put the fatal-step figure at 71.9 / 65.5 / 67.9 % against
`coin_collector` and 64.0 / 71.4 / 60.5 % against `rule_based`, on base rates never above 3.5 %.

**In roughly two of every three deaths the Q-row is all zeros.** With `TIE_TOL = 0.0` all six
actions then tie exactly, `act` falls through to `policy_rng.choice(best)` — so the agent is
choosing **uniformly at random at the moment it dies**. That cannot be repaired by a better
feature; only by visiting the row.

The mechanism probe agrees and is independent of any visitation argument: in **27–42 %** of
deaths (noisy across draws) the agent was standing in a blast with `escape_direction` returning
`NO_TARGET` somewhere in the 4-step window — genuinely trapped. Against `peaceful` that is
**83–100 %** of the 6–12 deaths, with the fatal rows all *present* in the table but never
reached by the solo greedy policy (0.00 % all-zero at the fatal step, in every draw). So
`peaceful` deaths are rare-state deaths and bombing-field deaths are unvisited-state deaths —
two different failures that happen to share a metric.

Withdrawn from the first run of this script: a "foreign bomb on the board" figure. `game_state`
carries no owner for a bomb, so an agent **cannot tell its own bomb from anyone else's** — the
quantity is not measurable from the observation, which is itself a constraint on rung-3 feature
design. Only the total bomb count is; deaths cluster at 2–3 bombs on the board.

### Method: opponent behaviour is not reproducible, and the pairing is weaker from here on

`peaceful_agent:5`, `coin_collector_agent:68` and `rule_based_agent:69` all call
`np.random.seed()` **with no argument** in `setup`, reseeding the global legacy RNG from OS
entropy once per world. `evaluate.py`'s per-round `world.rng` reseed therefore pairs the
**arenas** but not the opponents. Re-running this diagnostic gave 12 vs 7 deaths against
`peaceful` and 84 vs 86 against `rule_based` on identical seeds.

Consequences: every rung-3 number is one draw, not a reproducible constant, and paired CIs no
longer remove all between-run variance. It is fixable in *our* harness without touching a
framework file — all three provided agents use the **global** `np.random`, while our agent uses
only seeded `default_rng` generators (`callbacks.py:403`, `train.py:189`), so a
`np.random.seed(base_seed + round_index)` beside the existing `world.rng` line restores pairing
for the opponents and cannot affect us. To be applied before E25.

### Verdict — **floor established**, and it re-picks E25 against what I pre-registered

> **Correction added 2026-08-12, from the independent audit of E24/E25 (`scratchpad/audit/`).**
> Two things in this entry are wrong and are left standing above with the correction here.
>
> 1. **"116.6 → 27 crates" is not a transfer failure.** The board holds 122.2 crates and four
>    agents strip it by roughly **step 140**; 27 *is* approximately the fair share, and the three
>    `coin_collector` opponents take 31–32 each. Reading it as capability loss framed the whole
>    rung as a survival problem.
> 2. **The real transfer failure was in this entry's own results table and I missed it.**
>    `invalid = 25.27`, one row below `crates`. It is worth **−25.27 reward per round** — more
>    than coins and crates earn together — and 99 % of it lives in two rows with digit 6 =
>    `NO_TARGET`. The agent has **no objective at all** for the last two thirds of the round:
>
>    | field | digit 6 = `NO_TARGET`, share of safe steps |
>    |---|---|
>    | solo | 0.43 % |
>    | 3× `peaceful_agent` | 0.46 % |
>    | 3× `rule_based_agent` | 10.91 % |
>    | 3× `coin_collector_agent` | **26.28 %** |
>
>    That is an **objective** problem, not a survival one, and it is what E26 acts on.
> 3. **The all-zero-row inference was pushed one step too far.** The statistic is real and
>    reproduces at 66 % (audit, independent instrument). But "*that cannot be repaired by a
>    better feature; only by visiting the row*" was tested by E25 and **failed** — all-zero rows
>    went to 0.00 % and the agent got 20× worse. A 50× enrichment *at the moment of death* is
>    also partly a consequence of dying in unusual states rather than a cause of it; no
>    intervention test separated the two, and the entry treats it as causal.

Not a verdict on a change: a floor. The rung-2 agent transfers as a **crate engine with no
survival policy in company** — 116.6 → 27–29 crates and 1.000 → 0.143 survival against the
tournament opponent. It is clearly behind `rule_based_agent` against `peaceful` (−3.84 score),
but in both bombing fields the score gap **is not demonstrated** (CIs include 0) while the
survival gap is (−0.26 in both). The deficit is time alive, not scoring rate.

My pre-registered rule ("killed-by dominant → features; survives-but-never-kills → rewards")
does not resolve: both death modes bind and kills are ~0. The numbers resolve it differently.
**Kills are nearly free once opponents exist** — 1.893 per round against `peaceful` with the
aggression term switched off, purely as a by-product of bombing crates. The bottleneck is not
*earning* kills but *surviving to bank them*: against `rule_based` I score 2.947 because I am
dead at step 163 of 400. Adding `KILLED_OPPONENT` to an agent that survives 14 % of rounds
optimises the wrong term.

Survival first, and there is a correctness argument rather than a tuning argument for it:
`REWARDS[KILLED_SELF] = -5` while `GOT_KILLED` is **absent, i.e. 0**. Under that table walking
into your bomb is free and walking into mine costs 5. The inconsistency was invisible until now
because with `--opponents none` the event cannot fire.

**E25 = add `GOT_KILLED` and retrain in an opponent field**, warm-started from the shipped
table; no new digits until that is measured. The two are inseparable — a penalty that never
fires is a no-op — so the arms are (A) retrain in the field, rewards unchanged, (B) retrain in
the field with `GOT_KILLED = -5`, against the shipped table as the untrained baseline. That
decomposes "training distribution" from "death penalty", which is exactly what the post-mortem
says is in question.

**Cost, measured rather than estimated** (200 episodes, `--train 1`): 9.11 s in the
`coin_collector` field, 8.24 s in `rule_based`, 1.56 s solo. Rounds end sooner with opponents,
so the extra per-step cost is largely offset and 40 000 episodes is ~30 min, not the ~1.7 h I
projected from per-step figures. Both fields are affordable; the field is a question about
learning signal only, and it is deferred to E26.

---

---

## E23 — `WARM_N` and the tie-break tolerance, on five training seeds never used before

- **Question:** the closing experiment for rung 2. Two loose ends, one batch, because they need
  the same thing — training runs that were not used to choose anything.
  1. **`WARM_N`.** The audit named the cause of `warm`'s instability: `WARM_N = 100` puts
     α = 0.0398 on transferred cells where the parent had settled to α ≈ 2.15e-4, so the warm
     start **raises the learning rate on converged values by 185×** and un-converges what it
     transfers. It was a number I picked to make the transfer survive its first update, never
     swept. The trade-off is visible in the arithmetic: at 100 000 pseudo-visits α = 3.16e-4,
     within 1.5× of the parent, so the values are preserved — but then the new digit can barely
     be learnt and the arm should collapse onto its parent's 97.31.
  2. **The tie-break.** `act` breaks ties on exact float equality, and the audit measured that
     this insurance never fires: the rows that absorb a collapsed policy sit at margins of
     1e-4 to 1e-2, never at 0. `BM_TIE_TOL = 0.01` was measured on the existing tables to lift
     the worst seed 63.97 → 76.20 and cut between-seed sd 16.77 → 11.99. But **`TOL` was chosen
     across those same five training seeds**, and the audit's own rule — a holdout must hold out
     the factor that was selected over — makes that a development number, not a result.

- **Change:** none to the learning rule. `BM_TIE_TOL` already exists (default 0.0, verified
  byte-identical: re-evaluating `warm_s2@100k` with the switch present reproduces the committed
  CSV in 300/300 rounds). This entry only *chooses* its value and tests it where it was not
  chosen.
- **Design.** Training seeds **`BM_RUN_INDEX` 5–9**, which have never been run on this project.
  Three arms × five seeds × 40 000 episodes (`warm` peaks at 40 000; 100 000 is spent past it):

  | arm | `BM_WARM_N` | α on transferred cells | vs the parent's 2.15e-4 |
  |---|---|---|---|
  | `wn100` | 100 (the E21 value) | 0.0398 | 185× |
  | `wn10k` | 10 000 | 1.58e-3 | 7× |
  | `wn100k` | 100 000 | 3.16e-4 | 1.5× |

  Every table is then evaluated **twice**, at `TIE_TOL` 0.0 and 0.01 — a paired within-table
  comparison, so the tie-break is tested on runs that played no part in choosing it.
  Coarse parents `e16_c5_k03_s{0..4}` are reused, run index *i* taking parent *i* − 5, so each
  parent appears once per arm. The parents are shared with E21; what is held out is the
  **training run**, which is the factor both questions were selected over.
- **Measurement:** 300 rounds, ε = 0, seed 20260731, at 20 000 and 40 000. Reference for the
  same configuration on the *old* seeds: `warm`@40 k **106.67 ± 9.70** dev, and the parent
  `c5_k03`@100 k **97.31 ± 12.54**.

### Prediction (written before the run)

1. **`WARM_N` is non-monotone, with 10 000 best.** `wn100k` lands within 5 crates of the parent's
   97.31 (values preserved, new digit unlearnable); `wn100` reproduces E21 at 100–110 with the
   large spread that made it unshippable; **`wn10k` ≥ 108 with sd < 9**. **Refutation:** all
   three arms within 5 crates of each other → `WARM_N` is not the lever, `warm`'s instability has
   another cause, and the 185× arithmetic explains nothing.
2. **`wn100k` collapses onto its parent in a checkable way, not just in the mean:** its 20 000 and
   40 000 checkpoints differ by less than 3 crates, because it is barely moving.
   **Refutation:** it moves more than `wn100` does over the same interval.
3. **The guard breach tracks `WARM_N`.** Arm-mean suicides: `wn100` ≈ 0.02 (E21's arm mean was
   0.019, not the 0.043–0.053 the entry quoted from one seed), `wn10k` and `wn100k` below 0.01.
   **Refutation:** `wn100k` ≥ `wn100` → the breach is not the transfer learning-rate.
4. **The tie-break replicates out of sample, on variance and not on the mean.** Over the 30
   (arm, seed, checkpoint) tables: worst-seed-per-cell improves by ≥ 5 crates, between-seed sd
   falls, and **the paired mean effect stays inside its CI** — I am explicitly *not* predicting a
   mean improvement. **Refutation:** worst seed improves by less than 2 crates, or sd does not
   fall → the dev-set result was selection over five seeds and the tie-break is dropped.
5. **The tie-break costs nothing where nothing is broken:** on tables already above 100 crates
   the paired difference is within ±2. **Refutation:** a healthy table loses more than 5 → 0.01
   is too wide and the value must be re-chosen.
6. **Guards:** `think_max_ms` ≤ 0.30 at `TOL = 0.01` (measured 0.12–0.19 on three tables, against
   a 500 ms budget); suicides ≤ 0.02 and survived ≥ 0.98 as **arm means**, reported per seed.

**Refutation condition for the whole entry:** `WARM_N` flat across the three arms **and** the
tie-break failing to replicate → rung 2 has no remaining tractable lever, the best table on the
board stays what it already is, and the honest close is to ship it and go to rung 3.

### Result (measured 2026-08-11) — rung 2 closes at reference parity

| arm | @20 000 | @40 000 |
|---|---|---|
| `wn100` (the E21 value) | **111.40 ± 4.84** | 104.03 ± 13.42 |
| `wn10k` | 55.22 ± 48.33 | 105.71 ± 6.66 |
| `wn100k` | 37.72 ± 25.10 | 44.50 ± 44.81 |
| incumbent `c5_k03` @100 k | — | 97.31 ± 12.54 |

**Prediction 1 is wrong, and backwards.** I predicted 10 000 best and 100 000 ≈ the parent.
Measured: **`WARM_N = 100` is the best of the three and raising it is harmful**, catastrophically
so at 100 000 (44.50, two seeds below 20). The mechanism reasoning was right at the table level
and wrong about its consequence — mean |Q − parent| after 40 000 episodes runs 0.0093 / 0.0017 /
0.0006 across the three arms, exactly the monotone ordering the α arithmetic predicts, so a high
pseudo-count really does preserve the parent. It just does not *help*: at initialisation the
children of a parent row are identical, so the greedy policy **is** the parent's, and a table
that can barely move cannot differentiate the five children the new digit created. It ends up
neither the parent nor a working fine-map agent. Prediction 2 fails with it (`wn100k` moves
+6.78 from 20 k → 40 k, `wn100` −7.37 — no smaller). Prediction 3 is unscored: training-time
`KILLED_SELF` is 0.62–0.65 in all three arms, which is the ε-greedy behaviour policy, not the
evaluated one; eval suicides are 0.007–0.016 everywhere, inside the guard.

**The tie-break replicates out of sample, and it is insurance rather than an improvement.**
Paired within each table, over the 30 tables of this batch:

| unit | effect | 95 % CI |
|---|---|---|
| per training run (n = 5) | **+13.84** | **[+7.17, +20.50]** |
| per arm × seed (n = 15) | +13.84 | [+1.10, +26.58] |
| worst seed per cell | 51.82 → 69.54 | [+2.86, +32.56] |
| between-seed sd | 23.86 → 18.88 | [−10.32, +0.37] |

It clears the bar on the mean *and* the worst seed at the run level — but the entire effect is
rescue: individual collapses go 5.57 → 60.65, 3.81 → 42.92, 11.82 → 65.71, while **on the
healthy `wn100` arm it is −1.06, CI [−3.05, +0.94]** — free where nothing is broken. Prediction
4 said there would be no mean effect; there is one, because a third of these tables were
collapsed. Prediction 5's refutation fires on its single-table criterion (`wn100_s7`@40 k loses
8.82) even though the arm-level effect is null. `think_max_ms` 0.12–0.22 against a 500 ms budget.

**The improvement over the incumbent is real but still not demonstrated, and I nearly reported a
selected cell as if it were.** Paired against the parent each run was warm-started from:

| cell | Δ vs parent | 95 % CI | t |
|---|---|---|---|
| seeds 5–9 @20 k (dev) | **+14.10** | [+4.46, +23.73] | **+2.87** |
| seeds 5–9 @20 k (held 550731) | **+16.32** | [+5.17, +27.48] | **+2.87** |
| seeds 5–9 @40 k | +6.73 | [−11.08, +24.53] | +0.74 |
| seeds 0–4 @20 k (E21) | +5.94 | [−14.06, +25.94] | +0.58 |
| seeds 0–4 @40 k (E21) | +9.36 | [−5.99, +24.70] | +1.20 |
| **both replicate runs averaged per parent, checkpoint-agnostic** | **+9.03** | **[−6.00, +24.06]** | **+1.18** |

The two significant rows are **one of four** seed-set × checkpoint cells. Averaging the two
independent replicate runs per parent — the correct pooling, since there are five independent
parents and ten children — gives **+9.03, not demonstrated**. Writing "DEMONSTRATED" off the
first cell I looked at would have been the same selection error the audit found in `warm`@40 k,
committed one entry later. **Recorded verdict: nicht gezeigt at n = 5, with a point estimate of
+9 crates.**

**What *is* settled is the ship table.** Selection on held-out 550731 among current-map tables
put `wn100_s5`@20 k first (116.92; dev 116.35). Confirmed on **990731, never used to select
anything, 1 000 rounds**:

> **116.57 crates · score 8.47 · suicides 0.000 · survived 1.000 · think_max 0.22 ms**

against `rule_based_agent`'s 116.43 and `coin_collector_agent`'s 116.26 on the same task, and
against the old map's `e13_s2`@10 k at **115.70** on that same arena set. Identical to two
decimals with `TIE_TOL` 0 and 0.01, which is the healthy-table result again.

**Verdict: BESSER for the ship table, nicht gezeigt for the configuration.** The fine map is no
longer behind the map it replaced — it is ahead of it on the one comparison that is properly
held out, and the shipped agent is at parity with the strongest reference on rung 2. The
proposed revert to the pre-E20 feature map is therefore **withdrawn**: E20's distance digit,
combined with the warm start at `WARM_N = 100` and an early stop, produces the best table this
project has measured. E20/E21/E22 remain three negative results on the way there, and the entry
below stands as written.

**Rung 2 is closed.** The remaining spread between training runs (± 4.8 at the best checkpoint,
worst seed 105.2) is small enough that seed selection on held-out data is a reasonable ship
procedure, and 116.6 is within noise of what two different rule-based references achieve — that
is the board running out of crates, not a policy ceiling worth another batch. Next work is
**rung 3**: `state_to_features` still has no opponent information at all.

### Measurement note (2026-08-11) — bomb placement quality, and why rung 2 is saturated

Prompted by watching the shipped agent play. Read-only probe over 40 rounds, every bomb it drops
(`scratchpad/benedict/` style probe, `blast_coords` at the moment of the drop):

| | crates |
|---|---|
| hit by the bomb actually dropped | **2.60** |
| best available from a tile 1 step away | 3.32 (**+0.72**) |
| best available within 2 steps | 3.66 (+1.07) |
| bombs already at the local optimum | 69 % (1 step) · 54 % (2 steps) |

The distribution is bimodal in the way it looks on screen: 429 of 1 799 bombs hit exactly one
crate, 219 hit four or more. **So 31 % of bombs are placed worse than a spot one step away** —
digit 7 is binary and cannot tell "hits 1" from "hits 4", which is the same gap the E16
correction above identifies as the reason a mis-priced crate reward degrades placement.

**But rung 2 cannot pay for fixing it.** The board holds **122.2 crates** and the agent destroys
**116.2 — 95.1 %**; coins 8.47 of 9. The entire remaining prize is **~6 crates**. What makes the
gap interesting is elsewhere: **99 % of rounds (993/1000) hit the 400-step limit**, so the agent
is stopped by the clock, not by capability. Bomb quality is therefore a *throughput* variable —
45 bombs at 2.60 clear what 37 would clear at 3.32, and each bomb costs ~4–5 steps of approach
and escape. The marginal rate is favourable but thin: the agent converts ~0.65 crates per step
today, and walking one extra step for +0.72 crates is barely above that.

**Decision: not tested on rung 2, folded into rung-3 feature work.** Three reasons. (1) The
prize is 5 % and we are already at `rule_based` parity. (2) E14 tested the counted digit (0/1/2/3+)
and it is a warning, not a green light — crates/bomb 2.43 → 2.69 (+10 %, against 2.9–3.4
predicted), mean unmoved, and the **ceiling fell** from 116.64 to 106.63. (3) The same efficiency
buys much more with opponents on the board, where the clock is tighter and coins come out of
crates. When it is tested, the pre-registered failure mode is E14's: **crates/bomb rises while
the best table falls below 116.6 → the same result twice, and the digit is dead.**

A second variant — retargeting digit 6 from "nearest crate" to "densest cluster" — is the
stronger lever but drifts closer to encoding the answer than the task rules are comfortable
with. The counted digit has no such problem: it describes the state and leaves the policy to be
learnt.

---

**Decision this entry was meant to settle.** The best table ever measured here is still
`e13_s2@10k` — **117.14 on 550731, 116.64 dev, 115.70 on 990731**, at or just above
`rule_based_agent`'s 116.43 — and it belongs to the **pre-E20 12 800-row map**. Nothing built on
the 64 000-row map has beaten it (best: `warm_s1@40k`, 113.46 held-out). If this entry does not
put the fine map clearly ahead, the correct close for rung 2 is to **revert the feature map to
its pre-E20 form and ship `e13_s2@10k`**, and to record E20/E21/E22 as three well-measured
negative results rather than carrying a map that costs crates.

---

## E22 — The α exponent, the one constant never swept

- **Question:** E21 finding 3 said the table never settles because the residual update and the
  deciding margin are the same size. Worked out properly, that has a crossover: a cell moves by
  |TD| / N^`ALPHA_EXP` per visit, the mean |TD| in late training is 0.31, and the margin that
  decided s0's corner cycle was 0.001, so

  | `ALPHA_EXP` | a cell keeps moving by more than 0.001 until |
  |---|---|
  | **0.7** (every experiment so far) | **N = 3 623 visits** |
  | 0.85 | N = 853 |
  | 1.0 | N = 310 |

  And the budget per cell over 100 000 episodes (~25 M updates) is **8 790** on the coarse map's
  474 used rows but only **4 960** on the fine map's 840. So at 0.7 the coarse map's busy cells
  finish well past the crossover and settle, while a large share of the fine map's are still
  above it when the run stops — which is a quantitative account of why the fine map churns and
  the coarse one did not, and of why E18 needed 200 000 to make the coarse map churn at all.
  `ALPHA_EXP` has been 0.7 since rung 1, picked because (0.5, 1] is where L26's two conditions
  both hold, and **never swept**. At 1.0 the crossover falls to 310 visits and essentially every
  used cell settles, with Σα = ∞ and Σα² < ∞ still satisfied — the other end of the same
  admissible interval, not a hack.

- **Change:** one constant becomes a switch. `ALPHA_EXP = float(os.environ.get("BM_ALPHA_EXP",
  0.7))`; it is already in the `TrainLogger` hyperparameters, so the metadata needs nothing.
  Three arms, all at exponent 1.0, all 100 000 episodes, five seeds, each paired against a run
  that already exists:

  | arm | switches | measured against |
  |---|---|---|
  | `a10` | `BM_ALPHA_EXP=1.0` | E20 `dist` (fine map, cold) |
  | `a10warm` | `BM_ALPHA_EXP=1.0`, `BM_WARM=...` | E21 `warm` (fine map, warm start) |
  | `a10coarse` | `BM_ALPHA_EXP=1.0`, `BM_ABLATE=target_dist` | the incumbent `c5_k03` |

  `a10coarse` reuses what E20 proved about `distnull`: pinning digit 8 is a *bijective
  relabeling*, so that arm is the incumbent's 12 800-row map exactly, and the α change can be
  measured on it without a second code state. The control that was worthless as a
  sample-dilution test is exactly the right tool here.

- **Agent:** `benedict_task2`, otherwise E21's cell · entry and the one-line change are one
  commit · suffixes `_e22_{a10,a10warm,a10coarse}_s<i>`.
- **Training:** 15 jobs × 100 000, world seed 810731 — about 4 h 45 at E20's throughput.
- **Measurement:** 300 rounds, ε = 0, seed 20260731, at 20 000 / 40 000 / 70 000 / 100 000.
  References: `dist` @100 k **91.92 ± 20.38**, `warm` @40 k **106.67 ± 9.70** (its peak) and
  @100 k **82.53 ± 32.69**, incumbent @100 k **97.31 ± 12.54**.

### Prediction (written before the run)

1. **Mechanism, static and decidable before any evaluation: `a10` ends with ≤ 5 % thin margins
   on 5/5 seeds** (`dist` @100 k: 17.4 / 3.9 / 6.2 / 1.9 / 2.6 %). **Refutation:** any seed
   above 10 % → the residual update is not what leaves margins thin, and E21's finding 3 falls
   with it.
2. **The decay stops.** `a10` at 100 000 ≥ its own 70 000 value on 5/5 seeds, and `a10warm`
   holds its peak: |100 k − 40 k| ≤ 10 crates on 5/5 (E21's `warm` lost 24 on the mean and 90
   on s2). **Refutation:** still decaying on two or more seeds → the instability has another
   cause, and the practical answer for this rung is early stopping rather than α.
3. **It will be slower early, and I say so in advance.** α = 1/N is the sample-average rule,
   which is right for a stationary target and too slow for a bootstrapped one: at N = 100 it is
   0.0031 against 0.0123 at exponent 0.7, four times smaller. **`a10` @40 k below `dist`'s
   76.87.** **Refutation:** `a10` is *faster* early too → α was never the binding constraint and
   the framing is wrong in both directions.
4. **`a10warm` @100 k ≥ 100** — the first configuration to still hold a good policy at 100 000.
   **Refutation:** below 92.
5. **`a10coarse` lands within ±6 of 97.31, with sd ≤ 12.54.** The incumbent's cells finish at
   8 790 visits, well past even the 0.7 crossover, so α should barely matter there. This arm is
   the safety control. **Refutation:** worse by more than 10 → exponent 1.0 is harmful in
   general, and arms 1–2 are confounded by that rather than informative about the fine map.
6. **Guards: `a10` and `a10coarse` clean (suicides ≤ 0.02, survived ≥ 0.98); `a10warm`
   violates them anyway.** E21's `warm` ran 0.043–0.053 suicides from a parent at 0.003, and
   that breach came from the transfer, not from α, so it should survive the α change unchanged.
   Predicting a guard *failure* on purpose — if `a10warm` comes back clean, the breach was an
   α interaction after all and E21's unexplained boldness has its answer.
7. **`a10warm` at its best checkpoint ≥ 106.67**, and it must reproduce on the held-out seed
   550731 (where `warm` @40 k gave 106.39) before it is claimed as anything.

**Fallback, decided in advance so it is not a post-hoc rescue:** if prediction 3 overshoots and
1.0 is so slow that `a10` is still climbing steeply at 100 000, the next value is **0.85**, not
a return to 0.7 — its crossover is 853 visits, still far below the fine map's 4 960 budget,
while allowing four times the early movement of 1.0.

**Refutation condition for the whole entry:** `a10` still decays **and** its thin margins do not
fall → the residual-update account is wrong, α is not the lever, and rung 2's answer is the one
E21 already produced empirically: take `warm` at 40 000, whose 106.39 on the held-out seed 550731
is the best validated number on this rung, accept that the checkpoint is a hyperparameter chosen
on held-out data, and move to rung 3.

### Result (2026-08-11) — refuted before the batch ran, by an independent audit

The 15-job batch was **never run**. An independent adversarial audit of E19–E22 (a separate
session, given the raw data and told to form its own numbers before reading these entries;
`scratchpad/AUDIT_E19_E22.md`) ran E22's own `a10` arm as a two-seed pilot instead — the real
`train.py`, `BM_ALPHA_EXP=1.0`, 100 000 episodes, world seed 810731, with its instrument
validated by reproducing `q_table_e20_dist_s0__ep20000.npy` byte for byte at 0.7.

| | crates dev | held 550731 | **thin %** | median \|Q\| |
|---|---|---|---|---|
| `dist` s0 @100 k (0.7) | 59.90 | 63.69 | 17.4 | 3.95 |
| **`a10` s0 @100 k (1.0)** | **9.29** | **11.35** | **41.5** | 4.12 |
| `dist` s4 @100 k (0.7) | 113.60 | 112.48 | 2.6 | 3.73 |
| **`a10` s4 @100 k (1.0)** | **25.65** | **25.05** | **39.9** | 4.47 |

**Prediction 1's refutation clause fires by 4× and 15×.** It asked for ≤ 5 % thin margins and
set refutation above 10 %; the measurement is 41.5 % and 39.9 %. Thin margins go **up**, and not
by scale compression — the median |Q| is unchanged or larger. The best 0.7 seed loses 88 crates.
Predictions 2, 4, 5 and 7 fall with it. **Verdict: SCHLECHTER, decisively, at a cost of one hour
instead of four and three quarters.**

**And the premise was wrong by four orders of magnitude.** The N ≈ 5 000 in E21 finding 3 and in
this entry's crossover table is 24.5 M updates ÷ (840 rows × 6) — *uniform visitation*. Measured
over 383 359 updates on a frozen table, the real distribution is skewed ~700×: unweighted median
N = 256, **visit-weighted median N = 173 454**. So the residual per-visit update has median
**3.1e-5**, and the visit-weighted margin of the greedy choice has median **0.24–0.48**. The
ratio is **12 000×, not 1×**. The entry's own crossover table never supported its conclusion
either: at 0.7 the crossover is N = 3 623 and both stated budgets (8 790 coarse, 4 960 fine) are
above it.

**This is the same error as E20's row 3007, one level up, and it is now a pattern:** reasoning
about a per-cell average when the quantity is distributed over a heavily skewed visitation. It is
written in my own notes as "static counts are not visitation" and I have now made it three times
in four entries. The rule that follows: **any claim about "a typical cell" must be visit-weighted
and measured, never divided out of a total.**

**Why 1/N is not "the other end of the same admissible interval".** Σα = ∞ and Σα² < ∞ guarantee
convergence for a *stationary* target. This target is measurably non-stationary — mean |TD| per
episode *rises* through training (`dist` s0: 0.226 @20 k → 0.315 @100 k). α = 1/N makes Q the
arithmetic mean of every target ever seen at that cell, weighting episode 1 as heavily as episode
100 000, which under a moving target drags every action in a row toward the same early-policy
mean. That is why margins got *thinner*. If α is ever swept again the interesting direction is
**0.55–0.6**, not 1.0.

---

## Corrections to E19 and E21, from the same audit (2026-08-11)

These change scored verdicts, so they go in the record rather than in a quiet edit.

**1. "Both arms decay" (E21) is wrong, and the fault is mine.** `CHECKPOINTS` wrote five
`ext`@300 000 tables and the evaluation loop I ran stopped at 200 000. Evaluated now:

| `ext` | 100 k | 150 k | 200 k | **300 k** |
|---|---|---|---|---|
| dev 20260731 | 91.92 | 90.61 | 83.57 | **97.59 ± 12.57** |
| held 550731 | — | — | — | **97.45 ± 12.05** |

The arm does not decay; it dips and recovers to its best checkpoint. **E21 prediction 2 was
scored "FAILED, refutation fired" on a number that was not the arm's endpoint** — at 300 000 the
s2/s3/s4 mean is 104.7 against the 103.7 they started from, so they did not fall. The score is
withdrawn: **not demonstrated either way**. Training to 300 000 and evaluating to 200 000 was a
choice, not a constraint, and the tables were on disk when the entry was written.

**2. Neither decay was established in the first place.** Paired at the run level (n = 5):
`warm` 40 k → 100 k **−24.13, CI [−73.1, +24.8]**; `ext` 100 k → 200 k **−8.35, CI [−19.9,
+3.2]**. 75 % of the `warm` decay is seed 2 alone; leave-one-out gives −7.6. E21's headline was
one seed and one unread checkpoint.

**3. E21 prediction 3 was unfalsifiable.** `np.repeat(coarse, 5)` puts **2 165–2 350** rows above
the ≥ 780 threshold *at initialisation*, 2.8–3.0× the line, before a single episode; training
moves the count by 5–41. Scoring it "CONFIRMED, emphatically" was scoring an identity. The
correct verdict is **not a prediction**. Finding 1 ("filling was never the deficit") still holds
— but on the flat-across-checkpoints evidence, not on this.

**4. E21's guard violation is one seed reported as the arm.** "suicides 0.043–0.053, survival
0.947–0.957" is seed 3 across checkpoints. Per seed at 40 000: 0.000 / 0.030 / 0.007 / **0.043** /
0.013. **Arm means 0.019 and 0.981 — inside both guards.** And it is not unexplained: `WARM_N=100`
sets α = 0.0398 on transferred cells where the parent was updating them at ≈ 2.15e-4, so the warm
start *raises* the learning rate on converged values by **185×**. It un-converges what it
transfers. E22's arm 2 would have confounded this, since 1.0 also changes that first-update α.

**5. `val550731` does not validate `warm`@40 k's checkpoint choice.** Selection shrinkage dev →
held-out is +0.28. A new *arena* seed costs nothing, because the checkpoint was selected on a
300-arena mean and arena noise is already averaged out; the factor selected over, and the dominant
variance component, is the **training seed**, which is identical in both sets. At the run level
the advantage over its own parent is +9.36 dev / +10.78 on 550731 / +10.68 on 990731, **CI
[−12.4, +31.1] / [−14.8, +36.4] / [−9.4, +30.8]** — not demonstrated on any of the three. E21's
finding 4 heading ("it survives a held-out arena set") overstates what that measurement can do.
**Also: `warm`@40 k has consumed 140 000 episodes, not 40 000** — its parent's 100 000 plus its
own 40 000 — and E21's table puts it in a "40 k" column beside genuine 40 k runs.

**6. E19 finding 2's magnitude is a partial sum of a telescoping series.** "−12.8 per round on
being in a blast … 2.5 deaths' worth of discouragement per round" sums F over the blast bucket
only. The three buckets cover all 400 steps and sum to **+1.94 per round**, which is what Ng et
al. predicts for γ < 1. The −12.8 on blast steps is offset by +24.8 on plain steps. **The
mechanism survives and is stated correctly as finding 3** — shaping does not telescope back into
the same aliased *rows*, so the row that chose `BOMB` is systematically debited — but "2.5 deaths
per round" is not a cost the agent ever pays, and that sentence should not be quoted in the report.

**7. Method changes adopted from the audit.**
- Report **run-level CIs at n = 5** alongside per-round paired ones, and say which question each
  answers: two *fixed tables* → paired over arenas; two *configurations* → paired over training
  runs. The pooled per-round bootstrap gives `warm`@40 k [+7.4, +11.4] by treating 5 runs as
  1 500 replicates; that is the wrong unit for a configuration claim.
- A **held-out set must hold out the factor that was selected over**. For a checkpoint or a
  hyperparameter chosen across seeds, that means held-out **training seeds** (`BM_RUN_INDEX` 5–9),
  not held-out arenas.
- **Write refutation clauses a plausible outcome can trigger.** Two here could not fire. Before
  committing an entry: ask what the arm produces at initialisation, and whether the opposite
  outcome is physically available.
- **Record `main.py --seed`** in the `TrainLogger` hyperparameters — the world seed currently
  survives only in the prose of these entries.

**What the audit found no fault with:** 80 of 80 official evaluations reproduced exactly under an
independent driver, as did both byte-identity controls (E19's no-op switch, E21's prediction 7).

---

## E21 — Is s0 slow or stuck? Longer training against a warm start

- **Question:** E20 left the finer map unconverged at 100 000 and its five seeds split into two
  populations. s2/s3/s4 filled ~800 of their ~840 used rows and are still climbing; s0 and s1
  filled ~485 — about what the *coarse* map fills — and s0 has 17.4 % near-ties, a curve that
  rose and fell (65.8 → 57.8 → 59.9), and a greedy cycle that turns on a **0.001** margin. Two
  different fixes follow from that, and they make different predictions, so this entry runs
  both: **more episodes** (if the empty rows are merely unvisited) against **a better
  starting point** (if they are visited but stuck in a near-tie the per-cell α can no longer
  move).
- **Change:** `train.py` only; `callbacks.py` is untouched, so the evaluated path is identical
  to E20's.

  | arm | switch | rounds |
  |---|---|---|
  | `ext` | — | 300 000 → checkpoints to **200 000** |
  | `warm` | `BM_WARM=_e16_c5_k03_s<i>__ep100000` | 100 000 |

  `warm` initialises every fine row from the coarse row it was split from. The distance digit
  is the least significant, so the children of parent *i* are rows 5*i*…5*i*+4 and `np.repeat`
  is exactly the right map — verified on 20 000 random states, `encode(f + (d,)) ==
  encode_coarse(f) · 5 + d` without exception. Each seed warm-starts from **its own** coarse
  seed, keeping the pairing intact (note s0 inherits from the coarse map's *best* seed, 108.0).

  **The pseudo-count is not optional.** α = 1/N(s,a)^0.7 is exactly **1** on a cell's first
  update, so a transferred value would be overwritten outright the first time the cell is
  touched and the arm would be a no-op with extra I/O. `BM_WARM_N` (default 100) credits the
  transferred cells with 100 prior visits, i.e. α starts at 0.0398 instead of 1. Only rows
  whose *parent* carried value are credited — 2 360 of 64 000 — so genuinely new rows keep a
  cold start and are not slowed to a twenty-fifth of their learning rate.

- **Agent:** `benedict_task2`, otherwise E20's cell exactly · entry and code are one commit.
- **Training:** five seeds, world seed 810731, new suffixes (`_e21_ext_s<i>`, `_e21_warm_s<i>`)
  — the E20 tables are not overwritten, and the new `save_table` guard would refuse anyway.
  Roughly 15 job-units of 100 000 against E20's 10, so ~5 h.
- **Measurement:** 300 rounds, ε = 0, seed 20260731. `ext` at 150 000 and 200 000 (≤ 100 000 is
  a byte-identical rerun of E20 and needs no evaluation); `warm` at 20 000 / 40 000 / 70 000 /
  100 000. References: **`dist` @100 k 91.92 ± 20.38** (per seed 59.9 · 88.5 · 93.3 · 104.3 ·
  113.6) and the **incumbent `c5_k03` @100 k 97.31 ± 12.54**.

### Prediction (written before the run)

1. **`ext` does *not* rescue s0: below 80 at 200 000** (it is at 59.9). The diagnosis says its
   rows are decided by margins of 0.001–0.05, and re-rolling those is a coin flip however long
   it runs. **Refutation:** s0 ≥ 90 → it was slow rather than stuck, the half-filled-table
   reading is wrong, and episode count is the whole story.
2. **`ext` keeps paying on s2/s3/s4: their mean ≥ 110 at 200 000** (now 103.7). **Refutation:**
   they *fall* → the fine map has E18's re-rolling pathology too, 100 000 was already near its
   optimum, and "unconverged, not worse" was the wrong reading of E20.
3. **`warm` fills the table — static check, decidable before any evaluation.** Rows with
   |Q|max > 2 ≥ 780 on 5/5 seeds at 100 000 (`dist`: 477 · 495 · 810 · 788 · 811).
   **Refutation:** any seed below 600 → the transfer did not take, and the first suspect is
   `BM_WARM_N` being too small to survive the ε = 0.2 phase.
4. **`warm` @100 k beats `dist` @100 k on both moments: mean ≥ 100 and sd < 10** (91.92 ±
   20.38), because the seeds that gain are the two that were half-empty. **Refutation:** mean
   below 92 → warm-starting does not help and the empty-row diagnosis is wrong.
5. **The decisive one: `warm` @100 k ≥ 97.31, the coarse table it started from.** If a map that
   is strictly finer, handed its parent's converged values, still cannot beat the parent, then
   the distance digit does not pay for itself under any training budget and the E20 line ends
   here. **Refutation:** below 97.31 → rung 2 feature work is finished and the incumbent ships.
6. **`warm` converges faster: `warm` @40 k ≥ `dist` @70 k (82.26).**
7. **Replication:** `q_table_e21_ext_s<i>__ep100000.npy` byte-identical to
   `q_table_e20_dist_s<i>__ep100000.npy`. Checkpoints are extra writes, not extra updates, so a
   longer run must reproduce a shorter one exactly. **Refutation:** any difference → the
   checkpoint machinery perturbs learning and every curve in this ledger is suspect.
8. **Guards:** suicides ≤ 0.02, survived ≥ 0.98, `think_max_ms` ≤ 0.30 on every reported
   checkpoint of both arms.

**Known risk, stated in advance:** `warm` still starts at ε = 0.2 and decays on the same
schedule, so the transferred values face 20 % random actions while α is at its smallest. That
is deliberate — changing ε as well would make a null result unattributable — but if
prediction 3 fails, the ε schedule is the second suspect after `BM_WARM_N`.

**Refutation condition for the whole entry:** `warm` fails prediction 5 **and** `ext` fails
both 1 and 2 → neither route fills the finer map, E20's measured resolution gain is not
reachable in practice, and rung 2 is done: the incumbent ships and the next work is rung 3
(`GOT_KILLED` into the reward table, then the transfer floor against `peaceful_agent`).

### Result (measured 2026-08-10)

| arm | 20 k | 40 k | 70 k | 100 k | 150 k | 200 k |
|---|---|---|---|---|---|---|
| incumbent `c5_k03` | 86.10 | 92.80 | 94.36 | **97.31 ± 12.54** | — | — |
| `dist` (E20) | 55.40 | 76.87 | 82.26 | 91.92 ± 20.38 | — | — |
| `ext` | *(= `dist`)* | | | | 90.61 ± 22.12 | 83.57 ± 19.96 |
| `warm` | 103.25 ± 15.29 | **106.67 ± 9.70** | 93.78 ± 19.14 | 82.53 ± 32.69 | — | — |

**Both arms decay.** `ext` falls 91.92 → 90.61 → 83.57; `warm` peaks at 40 000 and loses 24
crates by 100 000. Per seed, `warm` collapses spectacularly in places — s2 runs
114.7 → 115.2 → 80.5 → **24.9**, while s0 climbs 77.0 → 91.8 → **113.2** → 103.0 over the
same episodes. The seeds wander in and out of good policies *independently*.

**Finding 1 — the E20 s0 diagnosis was wrong, and this run refutes it.** I concluded that
s0's deficit was unfilled rows (477 of 851) and predicted a warm start would fix it by filling
them. The filling worked far beyond the prediction — **2 206–2 358 rows** carry value against
the ≥ 780 I asked for, three times the threshold, from episode zero — and the arm still
decays. Rows filled stay flat across the whole curve (s0: 2 275 → 2 286; s2: 2 354 → 2 358)
while crates swing by 90. **Filling the table is neither the deficit nor the fix.**

**Finding 2 — what governs performance is margin decisiveness, and it tracks almost
perfectly.** The fraction of well-posed rows decided by |margin| < 0.2, against crates, same
tables:

| checkpoint | s0 thin | s0 crates | s2 thin | s2 crates |
|---|---|---|---|---|
| 20 k | 24.5 % | 77.0 | 1.1 % | 114.7 |
| 40 k | 4.4 % | 91.8 | 1.5 % | 115.2 |
| 70 k | 1.1 % | **113.2** | 3.7 % | 80.5 |
| 100 k | 1.5 % | 103.0 | **15.1 %** | **24.9** |

The follow-digit-6 rate meanwhile sits at 62–65 % throughout and explains nothing. It is not
*what* the rows say, nor *how many* of them say anything — it is whether they say it
decisively.

**Finding 3 — the arithmetic of why it never settles.** At the end of training a busy cell has
N ≈ 5 000 visits, so α = 1/N^0.7 ≈ 0.0026, and the mean |TD error| in the last 10 000 episodes
is ≈ 0.30. **Each remaining update therefore moves a Q-value by ≈ 0.0008 — and the margin
deciding s0's corner cycle was 0.001.** The table is still stepping by roughly the size of the
decisions it is making. That is the whole pathology, it explains E18 (200 k re-rolled the
incumbent) and both arms here, and it says the fine map is worse only because more rows share
the same data, so more decisions live at that scale.

**Finding 4 — `warm` @40 k is the best configuration measured on this rung, and it survives a
held-out arena set.** 106.67 ± 9.70 on the evaluation seed; re-measured on **550731**, which
has never been used to choose anything, it gives **106.39 ± 10.02** against the incumbent's
95.61 there. The peak's *location* transfers too: 40 k > 70 k on both seed sets.

    warm @40k vs incumbent @100k, paired over 5 training seeds
      20260731 : +9.36   t = +1.20    -16.2 / +30.6 /  +2.8 / +11.4 / +18.2
      550731   : +10.78  t = +1.17    -19.3 / +33.2 /  +1.8 / +13.2 / +25.0

**Not demonstrated**, on both arena sets, for the same reason both times: four seeds gain 2–33
crates and s0 loses 19–20. A sign test on 4/5 does not reach 0.05 either. The effect is large,
reproducible across arenas, and still not established at n = 5 training seeds.

#### Predictions scored

1. **CONFIRMED.** `ext` does not rescue s0: 59.9 → 55.0 → **48.2**, worse, not better.
2. **FAILED, refutation fired.** s2/s3/s4 average 93.8 at 200 k against the ≥ 110 predicted and
   103.7 they started from — they *fell*. As the clause says: the fine map has E18's
   re-rolling pathology too, and 100 000 was already past its optimum.
3. **CONFIRMED, emphatically and uselessly.** 2 206–2 358 rows filled against ≥ 780 asked. See
   finding 1 — the prediction was right and the reasoning behind it was wrong.
4. **FAILED on both moments.** 82.53 ± 32.69 against ≥ 100 and sd < 10.
5. **Refutation fired, but the clause was mis-specified and its inference does not follow.**
   `warm` @100 k is 82.53, below the 97.31 line, which by the letter of the entry means "rung 2
   feature work is finished and the incumbent ships". But I pinned the comparison to 100 000
   *before* seeing that the arm peaks at 40 000, and at its own best checkpoint it beats the
   incumbent by ~10 crates on two independent arena sets. The honest reading is that the
   prediction tested the wrong point on the curve. **This is the third entry running in which
   a prediction was mis-specified rather than merely wrong** — E19's 4/5 dead zone, E20's
   corner row sized on the greedy instead of the training distribution, and now a checkpoint
   fixed before the curve's shape was known. The pattern is that I pin thresholds to numbers
   chosen from the *previous* experiment's geometry.
6. **CONFIRMED, emphatically.** `warm` @40 k = 106.67 against `dist` @70 k = 82.26.
7. **CONFIRMED.** All five `ext` tables byte-identical to `dist` at 100 000 — checkpoints are
   extra writes, not extra updates, and the curves in this ledger are safe from that.
8. **Guards: `warm` VIOLATED, `ext` clean.** `warm` runs suicides 0.043–0.053 against the
   ≤ 0.02 guard and survival 0.947–0.957 against ≥ 0.98 — inherited from a coarse parent whose
   own suicide rate is 0.003, so the transfer makes it *bolder* than its parent, which is not
   something the entry anticipated and is not yet explained. `ext` is spotless: 0.000 suicides,
   1.000 survival. `think_max_ms` ≤ 0.11 everywhere.

**Whole-entry refutation did not fire** — it required `ext` to fail predictions 1 *and* 2, and
1 was confirmed. Rung 2 is therefore not declared finished on these numbers.

**Verdict: nicht gezeigt — and the target has moved.** Neither longer training nor a warm start
demonstrably beats the incumbent at n = 5. But the question worth asking is no longer "does the
distance digit pay"; it is **"can this table be made to settle at all"**. Finding 3 says the
residual update size and the decisive margin are the same order of magnitude, which makes every
checkpoint on this map a sample from a random walk — and makes both "train longer" and "start
better" beside the point.

**Next — E22, the α exponent.** `ALPHA_EXP` has sat at 0.7 since rung 1, chosen because
(0.5, 1] is where L26's two convergence conditions both hold; it has never been swept. At 1.0,
α = 1/N and the residual update at N = 5 000 falls from 0.0008 to 0.00006 — thirteen times
smaller than the margins that are currently being re-rolled — while Σα = ∞ and Σα² < ∞ still
hold, so it is not a hack but the other end of the same admissible interval. That is a
one-constant experiment against a mechanism measured to three significant figures, and it
applies to the incumbent map as much as to the fine one, which makes it the first thing since
E15 that could move *both*. `warm` @40 k should be carried along as a second arm, since it is
the best table on the board and the α change is exactly what might let it hold its 40 k policy.

---

## E20 — The target distance as a feature digit

- **Question:** E19's finding 3 was that Φ varies *inside* a row — visit-weighted sd **1.28
  tiles** on the safe rows the policy actually uses — and that feeding it back as a *reward*
  injects noise of exactly that size into the very margins it was meant to widen. But the
  information is real and the agent cannot see it: one row saying "target is DOWN, all four
  neighbours clear, a bomb here pays" covers states where the target is one step away and
  states where it is six, and the value of stepping towards it is not the same in both.
  Shaping had to cancel between actions to stay honest; a **state digit** does not. Does the
  distance buy in the state space what it could not buy in the reward?

- **The design is measured, not guessed.** 40 greedy rounds of the incumbent, read-only,
  safe-with-target steps only (`scratchpad/benedict/` probes):

  | bucket scheme | residual within-row sd | used rows |
  |---|---|---|
  | none (incumbent) | 1.28 tiles | 144 |
  | `{1, 2+}` (base 3) | 1.23 | ×1.26 |
  | `{1, 2, 3+}` (base 4) | 0.89 | ×1.67 |
  | `{1, 2, 3, 4+}` (base 5) | 0.64 | ×2.08 |
  | **`{1, 2, 3–4, 5+}` (base 5)** | **0.48** | **×2.09** |

  The chosen split is not the obvious one: a uniform `{1,2,3,4+}` costs the same rows and
  only reaches 0.64, because the distance is not concentrated near 1 — over those rounds it
  spends 26.4 % of its steps at d = 1, 23.2 % at 2, 28.8 % at 3–4 and 21.6 % at 5+.
  **The nominal table growth is not the cost that matters:** the policy touches only **262
  of 12 800 rows (2.0 %)**, so 12 800 → 64 000 nominal is ×2.09 in rows that are actually
  learned.

- **Change:** `callbacks.py` gains digit 8, appended (not inserted) so every "digit 6" and
  "digit 7" in the entries below still means what it says. `bfs_first_step` returns
  `(direction, distance)` from the same traversal, so the two digits describe one objective
  by construction and cannot drift apart. `FEATURE_SIZES` becomes
  `(4, 4, 4, 4, 5, 5, 2, 5)`, N_STATES 64 000. No reward, no hyperparameter, no training
  change — `BM_SHAPE` stays in `train.py` at its default 0 so E19 remains reproducible.

  **Deliberately the safe branch only.** In a danger state digit 8 reads 0, because digit 5
  already carries the scarce resource there (moves of grace left), and the escape distance
  is bounded by it. Escape distance as its own digit is the obvious follow-up if this works;
  bundling it here would make a null result unattributable.

- **Honest caveat, measured before the run:** this will almost certainly **not** fix the
  corner-spawn row 3007 that E18 pinned. Over 40 rounds the agent is in that row 9 times
  (~22 % of rounds, one step each — it is a spawn row) at d = 2 in 7 of them, sd 0.47. The
  digit splits rows whose distance *varies*; that one's does not. E19 already severed the
  link between "row 3007 follows digit 6" and performance, so E20 is argued from aliasing in
  general, not from that row.

- **Agent:** `benedict_task2` · otherwise E16's winning cell (coin 5, crate 0.3, γ = 0.99,
  ε floor 0.02) · entry and code are one commit.
- **Training:** two arms × five seeds × 100 000 episodes, world seed 810731.

  | arm | switch | what it isolates |
  |---|---|---|
  | `dist` | — | the full change: finer rows *and* the new information |
  | `distnull` | `BM_ABLATE=target_dist` | digit 8 pinned to 0: the **same** information as the incumbent in a table of the **same** 64 000 rows — i.e. the pure sample-dilution cost, with no new information at all |

  `distnull` is the control that makes a null result readable. Without it, "E20 ties the
  baseline" cannot be split into "the distance is worthless" and "the distance is worth
  exactly what the extra rows cost".
- **Measurement:** 300 rounds, ε = 0, seed 20260731, at 40 000 and 100 000. Labels
  `benedict_q_e20_{dist,distnull}_s{0..4}__ep{40000,100000}__task2`. Baseline **c5_k03
  @100 k: 97.31 ± 12.54 crates** (@40 k: 92.80 ± 9.47). Old checkpoints are unreadable by
  design — the shape guard in `setup()` raises rather than misindexing.

### Prediction (written before the run)

1. **Crates @100 k, `dist`: 100–110**, i.e. the digit earns more than it costs.
   **Refutation:** below 92.8 → the resolution is not worth the dilution and feature
   refinement on this map is finished.
2. **The signature of a sample-efficiency trade, not a free win:** at 40 000 `dist` is
   **at or below** the baseline's 92.80 (I expect 88–94), and its 40 k → 100 k gain is
   **larger** than the baseline's +4.51. A finer map should start slower and end higher.
   **Refutation:** `dist` gains *less* from 40 k → 100 k than the baseline → whatever moved
   is not resolution, and prediction 1 succeeding would be luck.
3. **`distnull` lands below `dist` and below the baseline**, in the 88–96 band: same
   information, 5× the rows, so it pays the dilution and buys nothing.
   **Refutation:** `distnull` ≥ `dist` → the gain in 1 came from the table size (more
   free parameters, softer per-cell α) and *not* from the distance, which would make the
   whole feature story wrong and is the single most important thing this control can catch.
4. **Used rows grow ~2×, not 5×:** tables show 500–650 rows with a non-zero entry against
   the incumbent's ~262 measured on-policy. **Refutation:** above 900 → the fragmentation
   measurement did not transfer from the incumbent's trajectory to the trained one, and
   the sample-efficiency reasoning behind 2 and 3 is unfounded.
5. **Cycling does not get worse:** `loop_probe` confined rounds ≤ 5/20 on every `dist` seed
   (incumbent 1/20 and 3/20; E19's `shape02` 17/20). **Refutation:** > 5/20 on two or more
   seeds → the finer map creates new mirror-image degeneracies instead of breaking them.
6. **Row 3007 does not resolve** — still 3/5 or 4/5 seeds following digit 6, as now. This is
   a deliberately *negative* prediction; scoring it either way is informative.
   **Refutation:** 5/5 → the corner override was distance aliasing after all and I called it
   wrong above.
7. **Guards:** suicides ≤ 0.02, survived ≥ 0.98, `think_max_ms` ≤ 0.30. The BFS returns a
   depth it already computed, so think time should barely move; the table is 3.1 MB instead
   of 614 KB, which is still nothing against the 0.5 s budget.

**Refutation condition for the whole entry:** `dist` within noise of the baseline **and**
prediction 2's gain signature absent → the distance is not information this agent can use,
the within-row spread E19 measured is not load-bearing, and feature work on this rung is
done. In that case the next move is not another digit but **rung 3** — `GOT_KILLED` into the
reward table and the transfer floor against `peaceful_agent`.

### Result (measured 2026-08-10)

**The arm is not converged at 100 000, and that is the finding.** Crates per seed along the
whole curve (20 k and 70 k evaluated after the fact, the checkpoints were already on disk):

| episodes | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| 20 000 | 4.7 | 43.2 | 86.4 | 54.3 | 88.4 | 55.40 ± 34.53 |
| 40 000 | 65.8 | 80.9 | 87.2 | 66.9 | 83.6 | 76.87 ± 9.89 |
| 70 000 | 57.8 | 71.2 | 79.6 | 97.9 | 104.8 | 82.26 ± 19.25 |
| **100 000** | **59.9** | **88.5** | **93.3** | **104.3** | **113.6** | **91.92 ± 20.38** |
| baseline @100 k | 108.0 | 82.2 | 112.4 | 90.6 | 93.4 | 97.31 ± 12.54 |

Paired against the baseline at 100 k: **−5.39, t = −0.43 — not demonstrated**, with the
per-seed diffs running −48.1 / +6.3 / −19.1 / +13.7 / +20.2. Guards all held: suicides
≤ 0.013, survived ≥ 0.987, `think_max_ms` ≤ 0.09 (the table is 3.1 MB and costs nothing).
The bomb rate recovered to 39.17 against the baseline's 38.23 — E19's collapse does not
recur, as expected, since nothing about the reward changed.

**Finding 1 — the predicted data-hunger signature is exactly right, and it has not
finished.** `dist` starts far below the incumbent (76.87 vs 92.80 at 40 k) and gains
**+15.05** from 40 k to 100 k where the incumbent gains **+4.51**; the last 30 000 episodes
alone are worth +9.7. Four of five seeds are still climbing at the point the run stops, and
**s4 reaches 113.6 — above the best single table ever measured on this rung** (the
incumbent's s2 at 112.4) and within three crates of `rule_based_agent`'s 116.4. Stopping at
100 000 was a decision inherited from E16's *converged* map; it is the wrong stopping rule
for this one.

**Finding 2 — `distnull` was not a control, and it could not have been.** It reproduced the
incumbent **exactly**, all five seeds, every metric. That is not a coincidence and not a
bug: pinning digit 8 to 0 maps index → 5 · index, a bijection, so the table is five times
larger and the other four fifths are *never touched*. Verified directly —
`q_e20_distnull_s{i}[::5]` equals `q_e16_c5_k03_s{i}` element for element, and the remaining
rows sum to exactly 0.0. **Unused rows cost nothing**; dilution comes only from *splitting*
rows that were previously merged, which is what `dist` does and what `distnull` by
construction cannot. The reasoning behind prediction 3 — "same information, 5× the rows, so
it pays the dilution" — was simply wrong, and no run could have rescued it. What the arm
does buy is a bit-exact wiring check: the `bfs_first_step` refactor changed the return type
of a function on the evaluated path and introduced **zero** behavioural change. That is
worth having, but it must be reported as what it is.

**Finding 3 — s0 is a single broken seed carrying the whole deficit.** 4.7 → 65.8 → 57.8 →
59.9: it stalls after 40 000 while the others climb. It also has the most near-ties of any
seed (17.4 % of well-posed rows below 0.2, against 1.9–6.2 % for s1–s4) and the worst
rollout (`loop_probe` 12/20 rounds confined to ≤ 2 tiles, entered at step median **17** —
early, so genuinely giving up rather than finishing). Excluding it the arm averages 99.9
against the incumbent's 94.6 on the same four seeds. That is not a licence to drop it — it
is the reason the mean is where it is, and the thing to fix.

#### Predictions scored

1. **FAILED, refutation fired — but only just.** 91.92 against a refutation line of 92.8, a
   gap of 0.88 crates, which is one twentieth of the arm's own sd (20.38). It fired on the
   written rule; treating it as a decisive negative would be over-reading it.
2. **CONFIRMED on the claim, missed on the level.** The ordering held in both halves — 40 k
   at or below the baseline, and a larger 40 k → 100 k gain (+15.05 vs +4.51). I predicted
   88–94 at 40 k and it came in at 76.87, so I understated how much the finer map costs
   early.
3. **FAILED, and the control was invalid by construction** (finding 2). The written
   refutation clause — "`distnull` ≥ `dist` → the gain came from table size" — does fire
   arithmetically, but the inference behind it is void: `distnull` cannot separate
   information from table size because it is the incumbent under a different set of row
   labels. A valid control would have to *split* rows on something uninformative — a random
   hash digit of the same base — and that is what the next such experiment should use.
4. **Band missed, refutation not fired.** 832–851 rows used against a predicted 500–650
   (refutation > 900). The band was anchored to the 262 rows I measured on the *greedy*
   trajectory; a trained table is touched by 100 000 episodes of ε-exploration and uses 474
   even without the new digit. Against that the fragmentation is **×1.77**, close to the
   ×2.09 the design measurement predicted — the ratio was right, my baseline for it was not.
5. **Refutation fired.** Confined rounds 12 / 8 / 5 / 1 / 3 out of 20, i.e. two seeds above
   the ≤ 5 line. Read with the entry steps it is less uniform than the count suggests: s1's
   cycles start at step median 290, which is finishing, while s0's start at 17. The honest
   version is that cycling did not get worse *in general* but did get much worse *on s0*.
6. **Refutation fired — I was wrong, and instructively.** The corner spawn now follows
   digit 6 in **5/5** seeds at margins of +1.53 to +2.29, against 3/5 at −0.38 to +1.44
   before. I predicted it would not resolve, from an on-policy measurement showing the
   agent in that state at d = 2 in 78 % of visits with sd 0.47. The busiest variant in the
   trained tables is **d = 1**, not d = 2: the ε-exploring *training* distribution over that
   state is not the greedy *rollout* distribution, and I sized the prediction on the wrong
   one. The same trap as "static counts are not visitation", one level up.
7. **CONFIRMED.** All guards held, think time fell if anything.

**Whole-entry refutation: did not fire.** It required the gain signature to be absent, and
the signature is the clearest thing in the run. Feature work on this rung is not finished.

**Verdict: nicht gezeigt — and the experiment is incomplete rather than negative.** At a
matched 100 000 episodes the finer map is 5.39 crates behind (t = −0.43), so nothing is
demonstrated and the incumbent stands. But the comparison is not the one the entry set out
to make: it matches *episodes*, and the two maps are at different points on their curves —
one flat, one climbing at +9.7 per 30 000. The right question, which this run cannot answer,
is what `dist` is worth at convergence.

**Next — E21, and the E18 objection does not transfer.** Extend `dist` to 200 000 and
300 000 on the same five seeds. E18 measured that 200 000 *hurt* the incumbent (97.31 →
62.85, re-rolling near-tie rows), and that is the obvious objection to running longer — but
it was measured on a map that had already flattened, where extra episodes only churn settled
rows. This map is still climbing on 4/5 seeds, and its near-ties are concentrated in the one
seed that stalled. Falsifiable form: if `dist` at 200 000 behaves like E18's `ext` did, the
"unconverged, not worse" reading here is wrong and the incumbent is the end of this line.
s0 is the second target: it should be diagnosed before the run, not after, because if the
stall is an absorbing row rather than slow learning then more episodes will not touch it.

### s0, diagnosed (2026-08-10) — and it changes the E21 design

**Not absorbing.** Its TD error over the last 10 000 episodes is 0.306 against 0.302–0.363
for the other four, and its training-time crates hold at 47.5 → 51.6 → 48.2. It is still
learning; what has broken is the gap between the behaviour policy and the greedy extraction.

**The cycle, opened with `cycle_dump`** on the first round that collapses (world seed 810731,
step 17 of 400): tiles (1,1) ↔ (1,2), in the spawn corner.

| tile | row | target | Q | argmax |
|---|---|---|---|---|
| (1,1) | 15027 | `RIGHT`, bomb pays | `[-1.111, -0.042, -0.041, -1.109, -0.212, -0.127]` | `DOWN` |
| (1,2) | 51013 | `UP` | `[0.026, -1.069, 0.024, -1.065, -0.169, 0.021]` | `UP` |

At (1,1) the table picks `DOWN` (−0.041) over its own target `RIGHT` (−0.042) — **an override
by 0.001**. At (1,2) digit 6 genuinely points `UP`, back into the corner, and the table
follows it. So the cycle is half noise and half feature, and `cycle_dump`'s symmetry test
says the two rows are **not** D₄ images of each other: canonicalisation would not merge them
and would not break this.

**Why the margin is a thousandth is the actual finding.** Row 15027 across the five seeds:

| | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| Q(`RIGHT`) — the target | −0.042 | 1.403 | 3.266 | 2.663 | 3.093 |
| largest \|Q\| in the row | 1.111 | 1.920 | 3.266 | 3.358 | 3.093 |
| crates | 59.9 | 88.5 | 93.3 | 104.3 | 113.6 |

s0 has learned the two `INVALID_ACTION` penalties (−1.11 on the walled directions) and
nothing else: every legal action sits at the step cost. No crate value ever propagated back
into the corner. And it is systemic rather than two unlucky rows:

| seed | rows used | rows with \|Q\|max > 2 | thin margins | crates |
|---|---|---|---|---|
| s0 | 851 | **477** | 17.4 % | 59.9 |
| s1 | 846 | **495** | 3.9 % | 88.5 |
| s2 | 832 | 810 | 6.2 % | 93.3 |
| s3 | 845 | 788 | 1.9 % | 104.3 |
| s4 | 839 | 811 | 2.6 % | 113.6 |

All five fragmented into ~840 rows; s0 and s1 filled only ~485 of them — about what the
incumbent's *coarse* map fills (433–470 of 474). **They are running a 64 000-row map at the
resolution of the 12 800-row one**, and the unfilled half is exactly where the near-ties sit.

**Consequence for E21: "train longer" is the right fix for three seeds and the wrong one for
s0.** s2/s3/s4 are in the filling regime — monotone curves, ~95 % of their rows carrying
value — so more episodes should keep paying. s0 is not: its curve rose then fell
(65.8 → 57.8 → 59.9), which is E18's re-rolling signature, and re-rolling a row whose six
values lie within 0.05 of each other is a coin flip however long it runs.

So E21 becomes two arms rather than one:

| arm | change |
|---|---|
| `ext` | the same `dist` runs continued to 200 000 and 300 000 |
| `warm` | each fine row initialised from the coarse row it was split from — `q_fine[5i + d] = q_coarse[i]` for all four buckets, taken from the incumbent's converged `c5_k03` table — then trained normally |

`warm` attacks the measured deficit directly: the fine map's problem is *empty* rows, and the
coarse map already knows what those states are worth, up to the distinction the new digit
adds. It is coarse-to-fine value transfer, not a hack, and it costs one line at
`setup_training`. Together the two arms separate "s0 is slow" from "s0 is stuck": if `ext`
rescues it, it was slow; if only `warm` does, the near-ties never resolve on their own and
initialisation — not episode count — is the lever for this map.

---

## E19 — Potential-based shaping on digit 6's own goal

- **Question:** E17 showed that no feature is redundant, E18 that neither more episodes nor
  a lower ε floor closes the 12.5-crate seed spread — 200 000 episodes *re-rolled* the
  near-tie rows instead of settling them (best seed 112 → 5). What is left to change is the
  shape of the reward, not the length of the run. Made concrete: in the **512 rows where
  the question is well posed** (safe, digit 6 points somewhere, that neighbour reads
  `NB_CLEAR`, so the move is legal), the greedy action follows digit 6 in only **56–68 %**
  of the rows the run ever touched, and 8–20 % of those rows are decided by a margin below
  0.2 — the noise the per-cell α leaves behind at the end of training. Does a potential
  shaped on digit 6's *own* goal set widen those margins, and does that convert into crates?

  Baseline, the incumbent `e16_c5_k03` at 100 000 episodes (static table statistics, no new
  games; `scratchpad/benedict/target_follow.py`):

  | seed | crates | follows digit 6 | mean margin | thin (\|m\| < 0.2) | row 3007 |
  |---|---|---|---|---|---|
  | s0 | 107.98 | 68.1 % | +0.034 |  8.0 % | DOWN ✓ (+1.441) |
  | s1 |  82.21 | 56.0 % | −0.115 | 19.8 % | RIGHT ✗ (−0.127) |
  | s2 | 112.37 | 63.8 % | −0.372 |  8.6 % | DOWN ✓ (+0.092) |
  | s3 |  90.60 | 58.3 % | −0.332 | 14.8 % | DOWN ✓ (+0.438) |
  | s4 |  93.38 | 62.6 % | −0.039 | 11.3 % | RIGHT ✗ (−0.381) |

  Row 3007 = (UP `BLOCKED`, RIGHT `CLEAR`, DOWN `CLEAR`, LEFT `BLOCKED`, safe, target DOWN,
  bomb useful): the top-left corner spawn from the E18 diagnosis, entered in ~21 % of
  rounds. The two seeds that override it are the two weakest. Across the five seeds the
  follow rate and the crate count correlate at **r = 0.86 (t = 2.94, df = 3)** — that is
  *below* the 95 % threshold, so it is the motivation for this experiment, not evidence for
  its conclusion.

- **Change:** training only. `BM_SHAPE` (default 0 — every experiment before this one,
  bit-for-bit, including the BFS which is then never called) adds
  **F = γ·Φ(s′) − Φ(s)** to the TD target, with

  > Φ(s) = −`BM_SHAPE` · (BFS steps to the nearest reachable coin, else to the nearest
  > crate),

  i.e. the same goal set and the same "goals are tested but never expanded" rule
  `target_direction` uses, evaluated on the *safe* branch only — danger states keep the
  same Φ, because escaping is priced by `KILLED_SELF` and not by shaping. Φ(terminal) = 0,
  so the terminal update gets F = −Φ(s_last). Shaping enters the **update only**; the
  logged episode reward stays unshaped, so the learning curves remain comparable with
  E10–E18. `callbacks.py` is untouched — nothing in the evaluated path changes.

  Two bugs found and fixed before the run, both worth recording because both were invisible
  by reading:

  1. The terminal term was first written to reuse the cached Φ of the previous step. That
     cache is correct exactly when the agent **dies** — `send_game_events`
     (`environment.py:468`) skips a dead agent, so no step-update fires and the cache still
     holds Φ of the state `end_of_round` updates. On a *surviving* round the step-update
     does fire and leaves the cache holding Φ of the **post**-step state, while the cell
     being updated is the pre-step one. Probed on 40 rounds with a trained table: exact on
     23/23 deaths, **wrong on 16 of 17 survivals**, by up to 0.4 at `BM_SHAPE` = 0.2 — and
     survival is the common case here (≥ 0.987). Φ is now recomputed in `end_of_round`.
  2. `BM_SHAPE` was missing from the `TrainLogger` hyperparameters, so the arm's defining
     parameter would not have appeared in any `.meta.json` — the one thing that makes a
     training log reproducible from the commit.

- **Honesty note on the theorem.** Ng et al. 1999 guarantees policy invariance *for the MDP
  the agent learns in*. This agent learns in a 12 800-row aliasing of the game, where
  genuinely different situations share a row; there the guarantee does **not** hold and the
  shaping *does* move the fixed point. That is the entire hypothesis — it should move it
  towards digit 6 — but it means E19 has to be argued as a deliberate bias, not as free
  acceleration. If crates rise, the defensible sentence is "shaping biased the aliased
  solution towards the BFS target and that was worth N crates", never "shaping only sped up
  convergence to the same policy".

- **Agent:** `benedict_task2`, otherwise E16's winning cell (coin 5, crate 0.3, γ = 0.99,
  ε floor 0.02, 100 000 episodes) · entry, switch and both fixes are **one commit**, so the
  hash every eval `.meta.json` stamps *is* the code under test.
- **Training:** two arms × five seeds × 100 000 episodes, world seed 810731, plus a
  20 000-episode no-op control. Measured cost of the extra BFS: 65 → **67 µs per step
  (+3 %)** — on `classic` the crate branch terminates after one or two expansions — so the
  batch should cost about what E18's did (~3 h 15).

  | arm | `BM_SHAPE` | reasoning |
  |---|---|---|
  | `shape02` | 0.2 | a step towards the goal nets ≈ +0.2, twice the step cost and above every bad margin in the table (0.127 / 0.381) |
  | `shape10` | 1.0 | deliberately over-strong: ≈ ±1.0 per step, more than three times the crate reward — brackets the sweet spot from above |
  | `noop`    | unset | 20 000 episodes, s0 — must reproduce `q_e16_c5_k03_s0__ep20000` byte for byte |

- **Measurement:** 300 rounds, ε = 0, seed 20260731, at 40 000 and 100 000. Labels
  `benedict_q_e19_{shape02,shape10}_s{0..4}__ep{40000,100000}__task2`.
  Baseline **c5_k03 @100 k: 97.31 ± 12.54 crates, 7.04 score**.

### Prediction (written before the run)

1. **The follow rate rises to ≥ 85 % in all five `shape02` seeds** (baseline 56–68 %) and
   the thin-margin fraction drops below 5 % (baseline 8–20 %). This is a *static* check on
   the tables, decidable before a single evaluation round is played. **Refutation:** any
   seed still below 75 % → the shaping never reaches the rows the variance lives in, the
   margin theory of the spread dies with it, and D₄ canonicalisation (E20) becomes the next
   lever instead of shaping.
2. **Row 3007 follows DOWN in 5/5 seeds** (baseline 3/5). The weaker, one-row version of 1,
   on the row E18 pinned. **Refutation:** ≤ 3/5.
3. **Crates, `shape02`: 100–110 with sd < 5** (baseline 97.31 ± 12.54) and **worst seed
   ≥ 90** (baseline 82.21). The claim is about the *spread* first and the mean second.
   **Refutation of the variance claim:** sd ≥ 10. **Refutation of the mean claim:** below
   92.8, i.e. shaping costs crates outright.
4. **`shape10` ≤ `shape02`.** At Φ-scale 1.0 the shaping dominates every real reward except
   the coin, so the aliased fixed point is dictated by the BFS rather than by what bombing
   pays. **Refutation:** `shape10` beats `shape02` by more than 5 crates → the sweet spot
   lies above 1.0 and the margin-scale argument that picked 0.2 was wrong.
5. **The crate-consumption discontinuity is the main risk, and I predict it bites at 1.0
   but not at 0.2.** Blowing up the crate you were walking towards moves Φ from −0.2·1 to
   −0.2·d′; at d′ = 3 that is F = −0.394 on exactly the step that pays +0.3 per crate. In
   the true MDP it is repaid while walking to the next crate — that *is* the telescope —
   but the repayment accrues to *other* rows, and under aliasing it need not route back to
   the row that chose `BOMB`. Operational form: **bombs/episode ≥ 35 in `shape02`**
   (baseline 33.1–41.7), and **below 30 in `shape10`**. **Refutation:** `shape02` below 30
   → the potential is punishing the action it was built to support, and E20 has to give Φ a
   term in the remaining crate count rather than abandon shaping.
6. **Guards:** suicides ≤ 0.02, survived ≥ 0.98 at every reported checkpoint, both arms.
   **`think_max_ms` unchanged** — `callbacks.py` is not touched, so any movement here means
   something leaked into the evaluation path.
7. **No-op control:** `BM_SHAPE` unset, 20 000 episodes, s0 → a table byte-identical to
   `q_table_e16_c5_k03_s0__ep20000.npy`. **Refutation:** any difference at all → the switch
   is not inert and every pre-E19 number is in question. This is the check the E18 incident
   would have failed.

**Refutation condition for the whole entry:** prediction 1 holds (the follow rate rises as
designed) but prediction 3 fails on *both* clauses (mean and spread unmoved) → the near-tie
rows were never the mechanism behind the seed spread, E18's diagnosis was a correlation
mistaken for a cause, and reward shaping is not the lever for the remaining variance.

### Result (measured 2026-08-10)

The no-op control passed first: `BM_SHAPE` unset for 20 000 episodes reproduced
`q_table_e16_c5_k03_s0__ep20000.npy` **byte for byte**, so the switch is inert and no
pre-E19 number is affected by this commit.

| arm | crates @40 k | crates @100 k | worst seed | bombs | crates/bomb | suicides | survived | think_max |
|---|---|---|---|---|---|---|---|---|
| baseline `c5_k03` | 92.80 ± 9.47 | **97.31 ± 12.54** | 82.21 | 38.23 | 2.55 | 0.003 | 0.997 | 0.14 |
| `shape02` | 91.76 ± 7.33 | 88.13 ± 7.90 | 74.31 | 29.42 | 3.00 | 0.033 | 0.967 | 0.10 |
| `shape10` | 15.63 ± 3.41 | 11.47 ± 2.14 | 8.04 | 3.64 | 3.15 | 0.190 | 0.810 | 0.02 |

Per seed at 100 k — `shape02` 93.2 · 93.2 · **74.3** · 90.7 · 89.4 against the baseline's
108.0 · 82.2 · **112.4** · 90.6 · 93.4. Seed-paired: `shape02` **−9.17, t = −1.10**
(*not demonstrated*, and note the sign), `shape10` **−85.83, t = −13.87** (worse, decisively).

**Finding 1 — the loss is entirely a bomb-rate loss, and it is unanimous.** `crates =
bombs × crates-per-bomb` splits it cleanly: the bomb rate falls in **5/5** seeds (−5.5 % to
−39.1 %) while crates-per-bomb *rises* in **5/5** (2.36–2.75 → 2.81–3.13). The shaped agent
places better bombs and drops far fewer of them. Two unanimous 5/5 splits in opposite
directions is a stronger statement than the insignificant mean.

**Finding 2 — the potential fights the escape, and it is priced against the crate.**
Read-only rollout of the *baseline* greedy policy, accumulating F = γΦ(s′) − Φ(s) at
`BM_SHAPE` = 0.2 and bucketing by what the step did (60 rounds, seed 550731):

| step type | n | mean F | F < 0 on |
|---|---|---|---|
| escaping (in a blast) | 6 674 | **−0.1155** | 66.8 % |
| crate destroyed | 2 350 | **−0.2545** | 77.0 % |
| plain safe step | 14 976 | **+0.0992** | 14.2 % |

Per round that is −12.8 levied on being in a blast (against `KILLED_SELF` = −5, so **2.5
deaths' worth of discouragement per round**) and −10.0 on the very steps that pay
`CRATE_DESTROYED` = +0.3 — roughly a third of the crate reward cancelled at the moment it
is earned. It telescopes globally, exactly as the theorem says; what it does *not* do is
telescope back into the same rows. The entry's design note — "danger states keep the same Φ,
because escaping is priced by `KILLED_SELF`" — was **wrong**, and this is the error worth
keeping: Φ being unchanged *in* danger states does not make shaping absent *on* danger
transitions. F fires on every step, and escaping means walking away from the crate you just
bombed.

**Finding 3 — shaping injects within-row noise of exactly the size it was meant to remove.**
This is the general lesson. Q′(s,a) = Q(s,a) − Φ(s), so the offset cancels between actions
**within a state** — that is the invariance argument. A *row* of this table is not a state,
it is a bucket of states, and Φ varies inside the bucket. Measured over 40 rounds on the
rows visited ≥ 30 times: the within-row sd of the BFS distance is **0.91 tiles**
(visit-weighted), so the injected noise is `BM_SHAPE` × 0.91 =

- **0.18 at `shape02`** — the same size as the median action margin (+0.15…+0.39) the
  experiment set out to widen, and
- **0.91 at `shape10`** — two to six times it, which is why 67–77 % of `shape10`'s
  well-posed rows collapse to |margin| < 0.2 against 8–20 % in the baseline.

The design was self-defeating in its own currency: every unit of margin the shaping adds
between actions, it also adds as noise between the states sharing the row.

**Finding 4 — the period-2 cycle E13 removed is back.** `loop_probe`, 20 rounds, greedy:
baseline 1/20 and 3/20 rounds confined to ≤ 2 tiles (and those entered at step 296 — that is
finishing, not giving up); `shape02` **17/20** and 10/20, entered at step 249 and 147;
`shape10` **20/20 from step 14**. The action mix carries the signature — baseline is
balanced (21/21/21/20 %), `shape02` runs UP+DOWN at 61 % against LEFT+RIGHT at 27 %, and
`shape10` is a pure left–right oscillation (38.6 % / 38.3 %, `BOMB` 0.6 %). Flattened
margins plus aliasing is precisely E13's mirror-image failure, re-created by the fix.

#### Predictions scored

1. **FAILED, refutation fired.** Follow rate 69.0 / 68.1 / 70.8 / 73.7 / 69.6 % — all five
   below the 75 % refutation line, nowhere near the predicted ≥ 85 % (baseline 56–68 %).
   The thin-margin fraction did fall (8–20 % → 1.8–10.6 %) but only s3 met the < 5 % clause.
   By its own written terms: the shaping does not reach the rows the variance lives in.
2. **Neither confirmed nor refuted — the clause was written badly.** Row 3007 follows DOWN
   in **4/5** `shape02` seeds (s4 still overrides, and its margin got *worse*, −0.381 →
   −0.518). I predicted 5/5 and set refutation at ≤ 3/5, leaving 4/5 in a dead zone. That is
   a drafting fault, not a result. It is moot anyway: `shape10` gets **5/5** on this row and
   is the worst agent ever measured on this rung, which severs the link between "row 3007
   follows digit 6" and performance more decisively than any `shape02` number could.
3. **FAILED; the mean clause's refutation fired.** 88.13 at 100 k, below the 92.8 line →
   shaping costs crates. The variance clause did *not* refute (sd 7.90 < 10, down from
   12.54) but the narrowing is the wrong kind: it pulled the **top** down, not the bottom
   up. s1, the frozen seed, gained +11.0 — the one thing E18's diagnosis predicted — while
   s2, the champion, lost **−38.1**. Worst seed 74.31, below the baseline's own 82.21.
   Regression to the mean by destroying what the good seeds had learned.
4. **CONFIRMED, by an order of magnitude.** 11.47 vs 88.13.
5. **Mechanism CONFIRMED, threshold WRONG — and the refutation fired.** I predicted the
   crate-consumption penalty would bite at 1.0 but not at 0.2, with the refutation line at
   `shape02` bombs < 30. Measured **29.42**. It bit at both scales; I called the direction
   and missed the magnitude. `shape10` bombs 3.64, as predicted (< 30).
6. **Guards VIOLATED.** Suicides 0.033 (`shape02`) and 0.190 (`shape10`) against a ≤ 0.02
   guard; survived 0.967 and 0.810 against ≥ 0.98. Both follow from finding 2 —
   this is the first arm since E10 to make the agent worse at staying alive.
   `think_max_ms` unchanged (0.10 / 0.02 vs 0.14), as expected: `callbacks.py` untouched.
7. **CONFIRMED.** Byte-identical no-op control.

**Whole-entry refutation:** did not fire as written (it required prediction 1 to *hold*),
but the outcome is worse than the condition it described — prediction 1 failed *and* 3
failed, so shaping moved neither the rows nor the result in the intended direction.

**Verdict: SCHLECHTER — potential-based shaping on digit 6's goal is rejected at both
scales.** `shape10` is decisively worse (t = −13.87); `shape02` is not demonstrated as worse
on the mean (t = −1.10) but is worse on the worst seed, on suicides, on survival, on the
bomb rate in 5/5 seeds, and on cycling in the rollout probe. Nothing here recommends
carrying it forward. The incumbent stands: **`c5_k03` at 100 000 episodes, 97.31 ± 12.54.**

**What E19 is worth keeping for.** The negative result is more useful than the positive one
would have been, because finding 3 is a *design rule* rather than a fact about this
potential: **in an aliased tabular learner, any state-dependent shaping term is only safe
while its within-row variation stays below the action margins it is meant to widen.** Here
sd(Φ)/SHAPE = 0.91 tiles against margins of 0.15–0.39, so no scale of `BM_SHAPE` could have
worked — too small to matter, or large enough to drown the row. That kills shaping on *any*
distance-like potential for this feature map, not just this one.

It also points at the successor. If the problem is that Φ varies inside a row, the fix is to
stop hiding it: put a coarse distance-to-target digit **in the state** (0 / 1 / 2 / 3+),
which cuts the buckets along exactly the dimension Φ varies on, costs a factor of 4 in table
size (12 800 → 51 200), and needs no reward change at all. That is E20, and it is a better
motivated experiment than the D₄ canonicalisation E18 left on the list — D₄ merges states to
fight sample efficiency, whereas the measured problem here is that states are merged *too
aggressively already*.

#### Hazard found while analysing this run (no result affected)

A diagnostic that drives `BombeRLeWorld` with `train=True` while `BM_MODEL_SUFFIX` names a
**real** checkpoint will silently overwrite that checkpoint: `setup_training` registers
`atexit.register(save_table, self)`, and `MODEL_FILE` resolves to the named table. It
happened to `q_table_e16_c5_k03_s2__ep100000.npy` during this analysis. Restored from
`q_table_e18_ext_s2__ep100000.npy` — E18's `ext` arm is a byte-identical replica of
`c5_k03` at 100 k, verified here on the four untouched seeds (s0, s1, s3, s4 all `cmp`-clean)
— and confirmed end-to-end by re-running the 300-round evaluation: **112.37 crates, 40.90
bombs, round-for-round identical to the committed CSV in 300/300 rounds**. No recorded number
changed. The general lesson is the same shape as E18's: the training path writes to disk by
default, so a *probe* must either use a scratch suffix or `train=False`. A guard worth
adding: refuse to `save_table` onto a path that already existed at `setup_training` time and
was not created by this run.

---

## E18 — Train longer, and the ε floor below 0.02

- **Question:** two loose ends, one batch. (1) The winner's curve at 100 000 is "still
  rising" only as a point estimate — per-seed diffs 40 k → 100 k are [+13.4, 0.0, +4.8,
  −1.0, +5.4], t = 1.76, not demonstrated. Where is the optimum, and does brute force fix
  the frozen seed? The s1 diagnosis makes that concrete: its corner-spawn row 3007 holds
  `RIGHT` over its own target `DOWN` by a margin that shrank 0.35 → 0.13 over the last
  80 000 episodes — does it flip by 200 000? (2) E15 measured the ε floor one-sidedly:
  0.10 is harmful, 0.02 untested downwards. Q-learning is off-policy, so ε only shapes the
  data distribution; GLIE wants ε → 0 *slowly*, and the hot rows are frozen by per-cell α
  long before ε matters — the open question is whether the residual 0.64 training
  deaths/episode at the 0.02 floor cost anything, and whether rare rows starve at 0.
- **Change:** none to the agent. `CHECKPOINTS` gains 140 000 / 200 000. Three arms:

  | arm | rounds | `EPS_END` |
  |---|---|---|
  | `ext` | 200 000 | 0.02 (incumbent) |
  | `eps0005` | 100 000 | 0.005 |
  | `eps0` | 100 000 | 0 — exponential decay all the way; ε < 10⁻⁶ from ~25 000 on |

- **Agent:** `benedict_task2`, E16's winning cell otherwise · entry and `CHECKPOINTS`
  change are one commit · labels `benedict_q_e18_{ext,eps0005,eps0}_s{0..4}__ep*__task2`
- **Training:** five seeds, world seed 810731. Determinism makes `ext`'s first 100 000
  episodes byte-identical to E16's c5_k03 runs, so its ≤ 100 k checkpoints need no
  evaluation — only 140 k and 200 k are new numbers (and a hash comparison of the 100 k
  checkpoint against E16's is a free replication check).
- **Measurement:** 300 rounds, ε = 0, seed 20260731. `ext` at 140 k / 200 k; ε arms at
  40 k / 100 k. Baseline **c5_k03 @100 k: 97.31 ± 12.5 crates, 7.04 score**.

### Prediction (written before the run)

1. **`ext` @200 k: 99–107 crates**, a decelerating rise (92.80 → 97.31 over the last
   60 k). **Refutation:** above 110 means the curve is not decelerating and every
   "ceiling" statement since E15 was premature; below 92.8 means peak-then-decay is back
   at γ = 0.99 and E15's central conclusion falls.
2. **s1 flips row 3007 and jumps to ≥ 95.** The RIGHT−DOWN gap closed 0.35 → 0.13 while
   the row's values tripled; another 100 k should close it. **Refutation:** s1 still at
   82 ± 3 with the row unflipped — then the argmax starves its alternative of data at the
   ε floor, "train longer" is not a variance cure, and the structural fixes (E19 shaping,
   D₄) are the only routes to the worst seed.
3. **The ε floor is a plateau below 0.02: both ε arms within five-seed noise of 97.31 at
   100 k.** This completes E15's one-sided curve (0.10 harmful, 0.02 ↔ 0 flat). I expect
   `eps0` slightly lower with **at least one seed below 80** — a cycle-prone seed that
   greedy-only training never rescues. **Refutation:** `eps0` *beating* 0.02 by more than
   10 crates means late-training exploration noise was actively harmful, and the floor
   goes to 0 for rung 3.
4. **Mechanism read, from the training log alone:** `eps0`'s KILLED_SELF over the last
   10 000 episodes falls below 0.05 (0.02 floor: 0.64) — the behaviour policy converges to
   the greedy one it is measured as.
5. **Guards:** suicides ≤ 0.02, survived ≥ 0.98 at every reported checkpoint, all arms.
6. **`think_max_ms` unchanged** — nothing in the evaluation path changes.

**Refutation condition for the whole entry:** `ext` regresses below 92.8 *and* both ε arms
land within noise — training length and the floor both dead knobs, meaning the remaining
variance and the 19-crate gap are entirely structural (features/shaping), and E19 becomes
the only live lever on this rung.

### Incident note, before the results

The `train.py` changes this entry declares were never on disk: an unsaved editor buffer
meant commit 61cafc1's message claims a diff it does not contain, training ran with
`EXPERIMENT = "e17"` (logs live under `q_e17_{ext,eps0005,eps0}_*` names) and the old
`CHECKPOINTS`, so **no 140 000 / 200 000 checkpoints were written and the 140 k point is
lost**. The `ext` runs themselves are valid — 200 000 episodes confirmed in the logs, and
the final tables are the 200 k state, which is what the 200 k evaluations below measure.

The expensive half of the mistake: the first evaluation pass loaded the ten *nonexistent*
checkpoint files, and `setup()`'s missing-file fallback silently played an **all-zero
table** — 2.82 crates, 1.000 suicides, five byte-identical "seeds", i.e. a uniform-random
agent measured under a real label. Caught only because the numbers were absurd.
`callbacks.py` now raises on a missing table when not training (verified to fire); in the
tournament the table ships beside the file, so the guard can only trigger when something
is genuinely broken — which the submission pre-run should say loudly.

### Result

`crates`, five-seed mean ± std (ε arms at 100 k; `ext` at 200 k from the final tables):

| arm | crates | per seed | vs 0.02 @100 k (97.31 ± 12.5) |
|---|---|---|---|
| `eps0005` @100 k | **100.26 ± 7.79** | 94.4 · 108.4 · 107.9 · 91.2 · 99.3 | +2.95, t = +0.44 — within noise |
| `eps0` @100 k | 84.03 ± 4.07 | 85.4 · 79.4 · 80.2 · 88.9 · 86.2 | −13.27, t = −2.20, **all five seeds worse** |
| `ext` @200 k | **62.85 ± 44.40** | 90.4 · **24.5** · **5.4** · 100.6 · 93.4 | −34.46, t = −1.60 |

Guards: suicides ≤ 0.013, survived ≥ 0.987, `think_max_ms` ≤ 0.09 everywhere.

**Training to 200 000 is actively harmful: the peak-then-decay is back at γ = 0.99.** The
best seed of the whole configuration (s2, 112.37 at 100 k) collapsed to **5.43**, and the
frozen seed fell further (82.21 → 24.47). E15's addendum said "at γ = 0.99 there is no
peak to find"; there is — it merely sits past 40 000 instead of past 10 000. γ delays the
decay, it does not remove it. What survives is exactly E16's narrower restatement: the
variance is set by the size of the margins the argmax is decided by, and *any* knob that
leaves those margins thin leaves the cliff in place.

**Row 3007 shows the mechanism in one row.** Between 100 k and 200 k the corner-spawn row
did not converge toward its target — it *re-rolled*: s1 flipped RIGHT → BOMB (margin
0.012), s2 flipped DOWN → RIGHT at margin **0.000**, s4 stayed on its override. With
per-cell α at ~10⁻³ and true action gaps flattened by γ = 0.99, the argmax in a near-tie
row is a random walk on noise, and every extra 100 000 episodes is another draw that can
land on a cliff. **"Train longer" is not a variance cure — it is another spin of the same
wheel.**

**The ε answer: the floor stays. Decaying to 0 is measurably wrong.** `eps0` converges its
behaviour policy exactly as intended — KILLED_SELF in the last 10 000 training episodes is
**0.004** against the floor's 0.64 — and is *worse* for it: 4 of its 5 tables produce
byte-identical evaluations at 40 k and 100 k, i.e. **learning stops entirely once nothing
explores**, and all five seeds land below their 0.02 counterparts. The 0.64 deaths per
episode at the floor are not waste; they are the tuition that keeps rare rows alive.
`eps0005` is statistically indistinguishable from 0.02 (and incidentally the best mean and
tightest spread ever measured on this rung — not claimable at t = 0.44, noted for the
record). E15's one-sided curve is now two-sided: 0.10 harmful, 0.02 ↔ 0.005 flat, 0 harmful.

### Predictions, scored

1. **"`ext` @200 k: 99–107" — wrong, and the refutation clause fired as written:** 62.85,
   below 92.8. Peak-then-decay is back at γ = 0.99 and E15's "no peak to find" falls.
2. **"s1 flips row 3007 and jumps to ≥ 95" — wrong in the third way.** I wrote two
   branches (flips-and-recovers, stays-frozen) and the row took neither: it flipped *and*
   fell (24.47), while the best seed's same row flipped onto a 0.000-margin override and
   took 107 crates down with it. The churn, not the freeze, is the finding.
3. **"Both ε arms within noise; `eps0` slightly lower with ≥ 1 seed below 80" — half
   right.** `eps0005` within noise ✓; `eps0` −13.3 with 5/5 seeds worse is more than
   "slightly" (t = −2.20, short of the CI rule, unanimous in sign); the seed-below-80
   detail: 79.4 ✓. The refutation (`eps0` winning by > 10) did not fire.
4. **"`eps0` training suicides < 0.05" — right, emphatically.** 0.004 from 0.64.
5. **Guards — held everywhere** (worst: 0.013 / 0.987, at `ext` 200 k).
6. **`think_max_ms` unchanged — held** (0.09).

The whole-entry refutation does not fire: `ext` collapsed, but `eps0` is not within noise,
so the floor is not a dead knob — it has a wrong side.

### Verdict

**The operating point stands: 100 000 episodes, ε floor 0.02 (0.005 equally admissible).**
Both directions away from it are now measured as harmful — more training re-rolls the
near-tie rows and lost the best seed; zero exploration stops learning outright. The
checkpoint-and-measure discipline paid for itself a second time in one entry: shipping
"the final table" of the 200 k runs would have shipped 62.85 where 97.31 was available.

The deeper reading: every road on this rung now ends at the same wall. E17 says the
features all carry load; E18 says neither training length nor the floor moves the mean or
tames the churn. The disease is thin argmax margins in high-traffic rows — diagnosed in
the E13 post-mortem, seen live in row 3007's re-rolls — and the one untried lever that
attacks margins *directly* is potential-based shaping toward the target. **E19 is next and
it is now the main line, not an option.**

---

## E17 — The ablation panel E10 has owed since the rung transition

- **Question:** E10 changed five things at once and took on the obligation to decompose the
  bundle afterwards, "from a working agent rather than from a broken one". Seven experiments
  later the agent works and the configuration is settled (E16). What does each component
  actually contribute — and is any of them dead weight riding along since the transition?
- **Change:** none to the shipped agent. `BM_ABLATE` becomes an environment switch in
  `callbacks.py` (same pattern as `BM_MODEL_SUFFIX`: unset = the full map, the tournament path
  is untouched, an unknown arm name fails loudly). Each arm pins one component to a constant,
  so the table keeps its 12 800 rows and only the information content changes — table size is
  not a confound. Five removal arms, retrained from scratch:

  | arm | what is removed |
  |---|---|
  | `danger` | digits 1–4 collapse to blocked/clear, digit 5 pinned 0 — **nests: escape cannot fire either** |
  | `escape` | digit 6 no longer switches to the way out in a blast (digit 5 stays) |
  | `crate_target` | digit 6 silent without a visible coin (the E11 map) |
  | `bomb_digit` | digit 7 pinned 0 |
  | `nocrate` | `CRATE_DESTROYED = 0` (`BM_CRATE=0`, no code — the reward arm) |

- **Agent:** `benedict_task2` · the switch and this entry are one commit, so the hash every
  eval `.meta.json` stamps *is* the code under test · labels
  `benedict_q_e17_{danger,escape,crate_target,bomb_digit,nocrate}_s{0..4}__ep*__task2`
- **Training:** 100 000 rounds, five seeds, world seed 810731, γ = 0.99, ε floor 0.02,
  `COIN` 5 / `CRATE` 0.3 except in `nocrate` — the E16 winning cell minus one component each.
  Same seeds and world seed as E16, so every arm is seed-paired with the baseline.
- **Measurement:** 300 rounds, ε = 0, seed 20260731, checkpoints 40 000 and 100 000.
  **The four code arms are evaluated with `BM_ABLATE` set** — the shapes match either way, so
  forgetting it would not crash, it would silently measure an ablated table through the full
  map. Contributions via `analyze.py --ablation` (removal mode), per seed, against
  **c5_k03 @100 000: 97.31 ± 12.5 crates, 7.04 score**. Reference 116.26 / 8.50.

### Prediction (written before the run)

1. **`danger` is the largest contribution: crates < 15, suicides > 0.3.** Without danger
   digits nothing separates "three moves on the clock" from "walking into fire", and the
   nested loss of escape puts this near the E10 floor regime. **Refutation:** crates > 40
   means survival is learnable from the terminal −5 alone at γ = 0.99, and the E10 EV
   arithmetic was wrong about why the bundle was needed.
2. **`escape` collapses bombing, not survival: crates 5–30, bombs low, suicides < 0.1.**
   E10 measured this trap exactly — with danger digits but no way out, the agent learns
   bombing is negative-EV and survives by not playing (2.41 crates, γ = 0.9).
   **Refutation:** crates > 60 means the danger digits alone carry escape at γ = 0.99 and
   E11's attribution ("the binding constraint was the escape") was γ-specific.
3. **`crate_target`: crates 25–50, and the 2-cycle returns** (loop probe: most rounds
   confined again). E11 measured 26.19 without it at γ = 0.9. **Refutation:** crates > 70
   means the longer horizon substitutes for the target gradient — E15's tie-separation
   argument reaching further than I currently believe — and E13's "decisively better"
   was partly γ's work.
4. **`bomb_digit` is the arm I am least sure of: crates 60–85, `invalid` rises above 1.**
   Digit 7's load-bearing half may be the folded-in `bomb_possible` (E10's absorbing-row
   argument), not the crate information. **A null here is a good result, not a failure:**
   within noise of 97.31 means digit 7 is redundant given digit 6 and the reward, and the
   table halves for free.
5. **`nocrate`: crates 40–75.** Pre-run arithmetic worth recording: a random crate hides a
   coin with p ≈ 9/123, worth 5 · γ^~15 ≈ 4.3 by the time it is collected, so the *expected
   instrumental* value of opening a crate is ≈ 0.3 — the untuned `CRATE_DESTROYED` constant
   is almost exactly the expected discounted coin behind the crate. Removing it leaves that
   signal concentrated in the ~7 % of crates that actually pay, which should slow learning
   badly but not zero it.
6. **The ordering: `danger` < `escape` < `crate_target` < `nocrate` < `bomb_digit` ≤ full.**
   This is the entry's strongest falsifiable claim — five arms give 5! orderings and I am
   naming one.
7. **Guards, in the arms that keep the escape machinery** (`crate_target`, `bomb_digit`,
   `nocrate`): suicides ≤ 0.02, survived ≥ 0.98. In `danger` and `escape` the suicide rate
   is a finding, not a guard.
8. **`think_max_ms` unchanged (~0.2–0.3).** Every arm removes computation; none adds any.

**Refutation condition for the whole entry:** all five arms within five-seed noise of
97.31. That would mean no single component earns its place — the performance lives in the
reward scale and γ alone — and the feature-attribution story running from E10 to E16 is
wrong from the start.

### Result

25 runs (3 h 15 min in three batches), 50 evaluations at ε = 0, 300 rounds, seed 20260731,
plus a loop probe per arm on s2. Five-seed means at the 100 000 checkpoint:

| arm | `crates` | std | `score` | `bombs` | `suicides` | `survived` | `steps` | probe: tiles/round |
|---|---|---|---|---|---|---|---|---|
| **full (E16)** | **97.31** | 12.5 | 7.04 | 38.2 | 0.002 | 0.998 | 399 | 127 |
| `crate_target` | 21.17 | 10.3 | 1.505 | 27.9 | 0.025 | 0.975 | 394 | 6 |
| `escape` | 13.77 | 4.4 | 0.727 | 4.8 | **0.399** | 0.601 | 256 | 5 |
| `danger` | 3.25 | 0.5 | 0.003 | 1.1 | **0.995** | 0.005 | **7.6** | 2 |
| `bomb_digit` | 3.22 | 0.1 | 0.009 | 2.9 | **0.878** | 0.122 | 53 | 1 |
| `nocrate` | **0.50** | 0.5 | 0.010 | **0.18** | 0.000 | 1.000 | 400.0 | 3 |

(The full-map tiles/round figure is the shipped table's probe from the E13 era; the arms were
probed now, same seeds 810731–810750.)

**Removing any single component costs 78–99 % of the crates, and every arm fails in a
different way.** Paired per-seed contributions on s2: +98.0 to +111.2 crates, every CI far
from zero. The wiring hazard declared above was checked, not assumed: the `bomb_digit` s2
evaluation re-run with the switch set is round-for-round identical to the recorded one, and
the deliberately mis-wired control (switch unset) differs — 3.28 against 6.72 crates — so
these numbers measure the arms, not a feature mismatch.

The failure modes, from the probes:

- **`nocrate` is E10's inert regime, exactly.** 0.18 bombs, zero deaths, 400.0 steps,
  a 3-tile oscillation from step 1. It learned *never to bomb*.
- **`danger`** drops one bomb and stands in it: WAIT is 48 % of its actions, median round
  5 steps. Suicides *rose* with training, 0.868 → 0.995 from 40 000 to 100 000.
- **`bomb_digit`** is the surprise, and the checkpoints tell the story: at 40 000 it
  oscillates over two tiles and rarely bombs (2.8 % of actions, 400-step rounds, 0.115
  suicides); at 100 000 it runs the *same* oscillation with 10.4 % bombing and dies at
  step 5 (0.878). Training raised `Q(BOMB)` until the agent bombs inside its own loop.
- **`escape`** keeps bombing and cannot get out — 0.399 suicides, 256-step rounds.
- **`crate_target`** is the mildest removal and reproduces E11 (21.17 now, 26.19 then at
  γ = 0.9): it bombs where it stands (27.9 bombs) but travels 6 tiles a round.

### Two findings worth the panel's price

**`CRATE_DESTROYED = 0.3` is not reward shaping — it is the bootstrap through the death
barrier.** The `nocrate` agent's collapse is not weakness, it is the E10 circular trap
closing again: early bombs kill (escape not yet learned), so bombing goes negative-EV and
is abandoned *before a single coin payoff is ever sampled* — and with no bombs, no coin
ever becomes visible, so nothing remains to learn toward. The prediction-5 arithmetic
(expected instrumental value ≈ 0.3 per crate exists in principle) was right about the
signal and wrong about *reachability*: the agent never survives long enough to collect it.
Together with E16 this brackets the crate reward: 0 fails inert, 1.0 fails reckless, 0.3
works. The constant that was never tuned sits in the one window that functions.

**Training entrenches the broken configurations.** `danger` 0.868 → 0.995 suicides and
`bomb_digit` 0.115 → 0.878 from 40 000 to 100 000 — the E06/E12 pattern ("more training
makes an already-wrong policy permanent") in two new costumes. A report reader should take
from this that checkpointed measurement is not a luxury: the 40 000 numbers alone would
have ranked `bomb_digit` two tiers higher.

### Predictions, scored

1. **`danger`: crates < 15, suicides > 0.3 — right on both** (3.25, 0.995). "Largest
   contribution" — **wrong**: `nocrate` is larger.
2. **`escape`: crates 5–30 — right** (13.77). **The mechanism was inverted:** I predicted
   the E10 outcome — abstains and survives, suicides < 0.1 — and at γ = 0.99 it *keeps
   bombing and dies*, 0.399. The refutation clause (> 60) does not fire; E11's attribution
   stands.
3. **`crate_target`: 25–50 — near miss low** (21.17). The γ-substitution refutation (> 70)
   does not fire; E13's necessity claim stands.
4. **`bomb_digit`: 60–85 — wrong by a factor of 25** (3.22). I flagged it least-sure and
   still failed to imagine the failure. "A null here is a good result" got its answer:
   digit 7 is near-load-bearing and the table cannot halve.
5. **`nocrate`: 40–75 — wrong by a factor of ~100** (0.50). The worst prediction in this
   ledger, and the most instructive: I priced the signal and forgot to ask whether the
   agent would live to see it.
6. **The ordering — wrong at both ends, right in the middle.** Measured: `nocrate` 0.50 <
   `bomb_digit` 3.22 ≈ `danger` 3.25 < `escape` 13.77 < `crate_target` 21.17. The three
   feature-branch arms landed in the predicted order; the two arms I placed *closest* to
   the full agent are the two on the floor.
7. **Guards:** `nocrate` 0.000 ✓; `crate_target` 0.025 — a marginal fail of the ≤ 0.02
   threshold; `bomb_digit`'s guard is moot, the arm collapsed outright.
8. **`think_max_ms` ~0.2–0.3 — held in substance, missed as written:** worst step 5.2 ms
   across all 50 evaluations — a scheduler outlier, ~1 % of budget, and no arm adds
   computation.

The whole-entry refutation condition does not fire — by two orders of magnitude.

### Verdict

**Every component earns its place, and E10's bundling is vindicated by measurement.** The
rung-transition argument — "on this rung the components are not individually measurable,
each alone is a five-seed zero" — was made from necessity in E10 and is now a measured
fact: five removals, five distinct collapses, 78–99 % of the performance gone each time.
The feature map carries no dead weight, which also means the remaining gap to the
reference (97.31 against 116.26) will not come from pruning — it has to come from what the
map does *not yet* encode, or from the seeds that converge badly.

### What I do next

1. **Diagnose the frozen seed before touching the model.** The winner's s1 has sat at
   82.1 ± 0.2 crates from 20 000 through 100 000 while s0/s2 rose 25 crates — a specific,
   findable defect (`cycle_dump`, finding-7 margins against s2). What it turns out to be
   decides between D₄ canonicalisation and the densest-spot target as the next change.
2. **Background batch: 200 000-round extension of the winner, plus the ε-floor arms**
   (0.005 and decay-to-0) — settles "still rising at 100 000" honestly and closes the
   ε question with a measurement instead of an argument. Prediction before the run.
3. **Then rung 3.** `GOT_KILLED` enters the reward table before any opponent training.

---

## E16 — The reward scale at γ = 0.99, and the coin ablation six experiments late

- **Question:** with the variance closed by E15, the mean is a hard ceiling — every γ, every
  checkpoint, five seeds, 80 to 93 crates against the reference's 116.26. Is that ceiling set by
  the reward function or by the feature map? And, finally: is `COIN_COLLECTED = 5` better than
  the game's own +1?
- **Change:** no feature change. `COIN_COLLECTED` and `CRATE_DESTROYED` become environment
  switches and are swept 2 × 2. γ = 0.99, ε floor 0.02 (E15's settings), five seeds per cell —
  **20 runs**. `STEP_COST`, `INVALID_ACTION` and `KILLED_SELF` held fixed.

  | | `CRATE_DESTROYED` 0.3 | `CRATE_DESTROYED` 1.0 |
  |---|---|---|
  | `COIN_COLLECTED` 5 | E15's winner (baseline cell) | |
  | `COIN_COLLECTED` 1 | the game's own value | |

- **Agent:** `benedict_task2` · commit `bd52080` · labels
  `benedict_q_e16_c{1,5}_k{03,10}_s{0..4}__ep*__task2`
- **Training:** **100 000 rounds**, world seed 810731, checkpoints 20 000 / 40 000 / 70 000 /
  100 000. Longer than every previous entry because E15's addendum showed γ = 0.99 is still
  improving at 40 000 and has no peak to find.
- **Measurement:** 300 rounds, ε = 0, seed 20260731. Baseline **γ = 0.99 @40 000: 92.80 ± 9.47
  crates, 6.671 score**. Reference 116.26 / 8.50.

### Why the rewards are now *unknown* rather than merely untuned

`COIN_COLLECTED = 5` and `CRATE_DESTROYED = 0.3` were both guessed — the coin value in E01, the
crate value in E10 — and neither has ever been tested. E15 then multiplied the effective horizon
by ten, which rescales every reward against the per-step cost by the same factor. Whatever
balance those guesses happened to strike at γ = 0.9, it is not the balance in force now.

### Prediction (written before the run)

1. **`CRATE_DESTROYED` is the live knob and 1.0 beats 0.3: `crates` 100–115** in the winning
   cell, from 92.80. The gap to the reference is entirely in crates opened, not in coins
   collected — see prediction 2 — so the term that pays for opening them is where the ceiling
   should move. **Refutation:** if `crates` stays inside 88–97 for all four cells, the ceiling is
   the feature map and not the reward, and the next experiments are the densest-spot target and
   E14's counted digit 7 retested.
2. **`COIN_COLLECTED` shows no effect: |Δ| under 4 crates and under 0.4 score, CI containing
   zero.** Our crates are 74 % of the reference and our score is 73 % — they track almost
   exactly, which says the agent already converts revealed coins about as well as the reference
   does and the coin term is not what is binding. **A null here is the result**, and it closes an
   item deferred in E01, E05 and E07.
3. **No interaction.** The 2 × 2 should be additive to within noise; the two rewards act on
   different parts of the round. If they interact, the likely reason is that a larger crate
   reward changes how much time is left for coin collection, and that would be worth its own
   entry.
4. **Regression guard: `suicides` ≤ 0.02, `survived` ≥ 0.98 in every cell.** A larger crate
   reward is a direct incentive to bomb more and stand closer, and E14 already produced a
   0.140-suicide seed at the checkpoint with the most crates.
5. **The curve is still monotone at 100 000** in the baseline cell, or peaks between 40 000 and
   100 000. If it is still rising at 100 000, the round count stops being an experiment
   parameter and becomes a compute budget, and that should be said plainly in the report rather
   than reported as "converged".
6. **Larger rewards shrink nothing.** `crates` std stays in E15's range (3–10); the variance was
   a discounting artefact and the reward scale should not touch it. If the spread moves, γ was
   not the whole story.

**Refutation condition for the whole entry:** all four cells within noise of each other. That
would mean the reward scale is irrelevant over this range, the ceiling is representational, and
the remaining work on rung 2 is feature engineering rather than tuning.

### Result

20 runs at 100 000 episodes (~80 min each, two batches), 80 evaluations at ε = 0, 300 rounds,
seed 20260731. `crates`, five-seed mean ± std:

| `COIN` | `CRATE` | @20 000 | @40 000 | @70 000 | @100 000 |
|---|---|---|---|---|---|
| **5** | **0.3** | 86.10 ± 2.9 | 92.80 ± 9.5 | 94.36 ± 11.1 | **97.31 ± 12.5** |
| 5 | 1.0 | 36.67 ± 7.1 | 40.78 ± 12.3 | 45.82 ± 21.1 | 50.59 ± 22.0 |
| 1 | 0.3 | 52.32 ± 35.5 | 49.94 ± 37.6 | 51.74 ± 37.0 | 29.21 ± 12.0 |
| 1 | 1.0 | 42.06 ± 13.2 | 40.02 ± 14.9 | 44.42 ± 6.5 | 41.64 ± 10.0 |

Marginals at each cell's best checkpoint: **`CRATE` 0.3 → 1.0 costs 74.81 → 47.51**;
**`COIN` 5 → 1 costs 73.95 → 48.37**. Regression guard at those checkpoints:

| `COIN` | `CRATE` | `suicides` | `survived` |
|---|---|---|---|
| 5 | 0.3 | 0.002 | 0.998 |
| 5 | 1.0 | **0.071** | 0.929 |
| 1 | 0.3 | 0.003 | 0.997 |
| 1 | 1.0 | **0.081** | 0.919 |

**The reward table that was never tuned is the best of the four, and both alternatives are
roughly half as good.** But the entry is not a null: it settles two things that were open, and it
corrects E15.

### Both main predictions were inverted, and the reasons differ

**Raising `CRATE_DESTROYED` buys recklessness, not crates.** Suicides go 0.002 → 0.071 and
survival 0.998 → 0.929; a dead agent stops opening crates, so the metric being rewarded harder
*falls*. I predicted suicides ≤ 0.02 and got 3.5× that. The guard fired and it was the
explanation, not a footnote.

**Lowering `COIN_COLLECTED` to the game's +1 costs 45 crates**, where I predicted no effect at
all — my most confident prediction in the entry. The argument behind it was that our crates are
74 % of the reference and our score 73 %, so the coin term could not be what binds. That reasons
about *outcomes*; the reward acts on the *value function*. A 16.7 : 1 coin-to-crate ratio gives
the table far more dynamic range than 3.3 : 1, and the E13 post-mortem already established that
this policy is an argmax over near-ties — bigger rewards mean bigger margins mean fewer decisions
settled by noise. **`COIN_COLLECTED = 5` is not an over-weighted coin, it is what keeps the value
function separated.**

That also explains why the two knobs interact strongly (prediction 3, wrong): the coin effect is
+45 crates at `CRATE` 0.3 and +6 at `CRATE` 1.0. With the crate reward already destabilising the
policy, the coin term has nothing left to hold together. A factorial was the right design and two
sequential sweeps would have missed this.

### This revises E15

E15 concluded that the peak-then-decay and the long variance tail were **discounting** artefacts.
The `COIN` 1 / `CRATE` 0.3 cell runs at **γ = 0.99 and still has std 35.5**, the worst spread
anywhere in this ledger. So the correct statement is narrower and more useful:

> The variance is set by the **size of the margins the argmax is decided by**. γ and the reward
> scale both control that, and either one being wrong brings the long tail back.

γ = 0.99 was necessary and is not sufficient.

### Predictions, scored

1. **"`CRATE` 1.0 wins, 100–115 crates" — inverted.** 47.51 against 74.81.
2. **"`COIN` shows no effect, |Δ| under 4 crates" — inverted, by 45 crates.**
3. **"No interaction" — wrong.** +45 against +6 depending on the crate reward.
4. **Regression guard `suicides` ≤ 0.02 — failed in both `CRATE` 1.0 cells** (0.071, 0.081), and
   the failure *is* the mechanism behind prediction 1.
5. **"Still monotone at 100 000" — right.** 86.10 → 92.80 → 94.36 → 97.31, still climbing.
6. **"The reward scale will not touch the variance" — wrong**, see above.

The whole-entry refutation condition (all four cells within noise) does not fire; the cells differ
by more than a factor of two.

### Verdict

**Status quo confirmed, and two open items closed.**

- **The `COIN_COLLECTED` +5 versus the game's +1 ablation is settled**, six experiments after it
  was first deferred in E01 and dodged in E05 and E07: **+5 is better by ~45 crates**, and the
  reason is dynamic range rather than coin-seeking.
- **The reward is not the ceiling.** No cell in the grid beat the incumbent values, so the
  remaining gap to the reference is representational or a matter of training length.
- **100 000 episodes beats 40 000**: 97.31 ± 12.5 crates and 7.04 score, against 92.80 and 6.671
  — **84 % and 83 % of the reference**, from 29 % and 27 % at E10.

Per seed at 100 000: 107.98 · 82.21 · 112.37 · 90.60 · 93.38.

**Caveat carried from the snapshot-noise finding:** the ~7-crate checkpoint term applies to every
number here, so `c1_k03 s2` topping the single-table list at 115.32 is not distinguishable from
`c5_k03 s2`'s 112.37, and neither is separable from the shipped E13 s2 at 116.64. The shipped
model does not change, but it is now matched rather than ahead.

### What I do next

1. **Extend the winning cell to 200 000 episodes.** The curve is still rising at 100 000, so what
   I have been calling a ceiling may simply be an unfinished run. Five runs, one batch, ~2.7 h,
   and it decides whether the remaining 19 % gap is representational at all. **This has to be
   answered before any more feature work**, because every feature experiment since E13 has been
   measured against a number that was still moving.
2. **Then the ablation panel E10 owes**, which is the backbone of the report's most-weighted
   section and is now cheap: remove the escape branch, the crate target, `CRATE_DESTROYED`, and
   the danger digits, one at a time, from a configuration that works.
3. **Then rung 3.** `state_to_features` still has no opponent information — the `TODO task 3+` in
   `callbacks.py` is untouched — and that is where the tournament is decided.
4. **Deprioritised: the densest-spot target and E14's counted digit 7.** Both aim at the mean,
   and with the best seed already within noise of the reference the honest description of the
   remaining gap is reliability, not capability.

### Correction, 2026-08-10 — the suicide mechanism above is wrong

The sentence *"a dead agent stops opening crates, so the metric being rewarded harder falls —
the guard fired and it was the explanation, not a footnote"* does not survive the arithmetic,
and it has been repeated since. The guard did fire: suicides really do go 0.002 → 0.071, a
35-fold increase, and that is a real regression worth reporting. It is simply **not the
reason the crates fall**, and the check is one line — condition on the rounds the agent
survived to step 400:

| `c5` arm | overall | surviving rounds only | death rate |
|---|---|---|---|
| `k03` @100 k | 97.31 | 97.47 | 0.002 |
| `k10` @100 k | 50.59 | 51.70 | 0.071 |
| **deficit** | **46.72** | **45.77** | |

**98 % of the deficit is still there in rounds where nobody died** (98.4 % at 40 000, same
picture in the `c1` pair). Truncated episodes account for about one crate of forty-seven.
The original reasoning confused a *guard that fired* with a *mechanism that explains*, which
is exactly the failure mode the guards exist to avoid — and E19 repeated it a different way,
so this is a pattern and not a one-off.

What the surviving rounds actually show, same tables, same 400-step rounds:

| `c5` arm @100 k | bombs | crates per bomb |
|---|---|---|
| `k03` | 38.30 | **2.55** |
| `k10` | 47.41 | **1.09** |

The agent with the larger crate reward bombs **more** (+24 %) and gets **less than half** out
of each bomb. Raising `CRATE_DESTROYED` does not make it reckless about dying; it makes it
reckless about *where*. `BOMB` becomes attractive in rows where a bomb hits little, and digit
7 cannot object, because it is binary — "a bomb here hits at least one crate" reads the same
whether the answer is one crate or three. So the corrected statement is:

> A larger `CRATE_DESTROYED` degrades **bomb placement**, not survival. The suicide rise is a
> real but minor side effect; the crates are lost to bombs that were not worth dropping.

This also sharpens an idea E16 deprioritised as "aiming at the mean" (point 4 above, and
E14's counted digit): **make digit 7 count the crates in blast range** (0 / 1 / 2 / 3+)
rather than answer yes/no. The measurement above is the first direct evidence that the binary
digit is what lets a mis-priced bomb reward do damage, which is a better argument than the one
E14 was rejected on. Candidate for after E21.

---

## E15 — The first hyperparameter sweep: γ × ε floor, factorial

- **Question:** every feature change so far has produced the same side effect — performance peaks
  at some checkpoint and then decays — and the seeds that decay fastest are the ones that produce
  the variance now capping the result. **That phenomenon is invariant to the feature map**
  (E12 −21 %, E13 −44 %, E14 −24 %, three different representations), which is an argument that
  it is not caused by the representation. γ and the ε floor have never been varied on this rung;
  γ has not been varied since E01. Do they explain it?
- **Change:** no feature change. `GAMMA` and `EPS_END` become environment switches and are swept.
  **Runs on the E13 feature map** (binary digit 7, 12 800 rows), not E14's — E14 was not
  demonstrated, cost a doubled table and lost the reference-parity seed, and its one advantage
  (halved variance) is exactly what this entry is trying to obtain by other means. If E15 works,
  E14 can be retested on top of tuned hyperparameters, which is a better test of it than E14 was.
- **Design:** γ ∈ {0.9, 0.95, 0.99} × `EPS_END` ∈ {0.02, 0.10}, five seeds per cell — **30 runs**.
  α (`1/N^0.7`), the ε decay rate and the rewards are held fixed.
- **Agent:** `benedict_task2` · commit `5c644b4` · labels
  `benedict_q_e15_g{90,95,99}_e{02,10}_s{0..4}__ep*__task2`
  (the `g099_e002` arm's 5 000 and 40 000 checkpoints were evaluated one commit earlier,
  at `54e695b`; the agent code is identical between the two — the diff is this file.)
- **Training:** 40 000 rounds, world seed 810731, checkpoints 5 000 / 10 000 / 20 000 / 40 000.
- **Measurement:** 300 rounds, ε = 0, seed 20260731; evaluate the 10 000 and 20 000 checkpoints
  first and extend if a cell's optimum lands on an edge. Baseline **E13 @10 000: 83.56 ± 24.23
  crates, 5.990 score**, best seed 116.64. Reference 116.26 / 8.50.

### Why γ is the prime suspect

γ = 0.9 is an effective horizon of 1/(1−γ) ≈ **10 steps**. The BFS targets this agent navigates to
are routinely 10–30 steps away, and a coin 20 steps out is worth 5 × 0.9²⁰ = **0.61** against a
step cost of 0.1 per step. So the value function is close to flat over exactly the distances the
policy has to discriminate — the same flatness that let `BOMB` lose to walking on by 0.006 in the
E13 post-mortem. A longer horizon should separate states that are currently near-ties.

Why not simply γ → 1: with 400-step episodes and a dense crate reward, a near-undiscounted return
makes *every* state look similar for the opposite reason, and it converges far more slowly. The
sweep is there because I do not know which failure mode dominates.

### Prediction (written before the run)

1. **γ = 0.95 wins; its best cell reaches 95–115 crates**, five-seed mean, from 83.56.
   γ = 0.99 is **worse than 0.95** and possibly worse than 0.9, on slow convergence.
2. **The central prediction: `crates` std below 12 in the winning cell**, from 24.23. If the mean
   rises and the spread does not, γ is buying performance without touching the mechanism, and the
   next suspect is `ALPHA_EXP`.
3. **The peak-then-decay shrinks: loss from the best checkpoint to 40 000 under 20 %** in the
   winning cell (E13: −44 %). This is the phenomenon the entry exists to explain; if it is
   unchanged at every γ, then it is not a discounting effect and `ALPHA_EXP` is next.
4. **A higher ε floor lowers the mean slightly and lowers the spread**, at every γ: more
   exploration keeps rare rows refreshed (the mechanism E05 predicted and never got to test) at
   the cost of a behaviour policy further from greedy. Expected to be the weaker of the two knobs.
5. **`suicides` do not rise, and should fall** — 0.002 or below at γ = 0.95. Dying forfeits the
   discounted future, so a longer horizon makes `KILLED_SELF = −5` relatively *more* costly, not
   less. If suicides rise with γ, my sign is wrong somewhere and the reward scale needs checking
   before E16 touches it.
6. **`think_max_ms` unchanged at ~0.2.** γ never enters the feature computation; if this moves,
   something other than the intended knob changed.

**Refutation condition for the whole entry:** no cell beats 83.56 ± 24.23 by more than the
five-seed noise. That would mean the variance is not a discounting artefact, and the remaining
suspects are `ALPHA_EXP` (which sets *when* learning stops) and the reward scale (E16) — in that
order, because the decay-with-training pattern is a learning-rate signature before it is a reward
signature.

### Result

30 runs (69 min, three batches of ten), 60 evaluations at ε = 0, 300 rounds, seed 20260731.
`crates`, five-seed mean ± std:

| γ | ε floor | @10 000 | @20 000 |
|---|---|---|---|
| 0.90 | 0.02 | 83.56 ± 24.23 | 63.94 ± 36.77 |
| 0.90 | 0.10 | 36.99 ± 12.78 | 25.82 ± 10.38 |
| 0.95 | 0.02 | 80.25 ± 27.39 | 87.02 ± 11.21 |
| 0.95 | 0.10 | 40.36 ± 17.52 | 36.90 ± 11.78 |
| 0.99 | 0.02 | 84.34 ± 10.47 | **86.10 ± 2.93** |
| 0.99 | 0.10 | 48.58 ± 16.37 | 39.80 ± 12.07 |

`score` follows the same shape: 5.990 ± 1.845 at the baseline cell, **6.179 ± 0.255** at
γ = 0.99 / 20 000. `suicides` ≤ 0.013 and `survived` ≥ 0.987 in every cell.

Per seed, the two cells that matter:

| cell | s0 | s1 | s2 | s3 | s4 | mean | std | worst |
|---|---|---|---|---|---|---|---|---|
| γ 0.90 @10 000 *(= E13)* | 54.75 | 66.24 | **116.64** | 85.74 | 94.45 | 83.56 | 24.23 | 54.75 |
| γ 0.90 @20 000 | 29.70 | 38.70 | 114.86 | 90.22 | 46.19 | 63.94 | 36.77 | 29.70 |
| **γ 0.99 @20 000** | 85.65 | 82.07 | 85.30 | 90.02 | 87.47 | 86.10 | **2.93** | **82.07** |

**Same mean, 8.3× less spread, and the worst seed goes from 29.70 to 82.07.** Paired on the worst
seed of each cell: `crates` +55.95 [+49.88, +61.62], `score` +4.293 [+3.860, +4.713].

A free reproducibility check fell out of the design: the γ = 0.9 / ε = 0.02 cell at 10 000 is the
E13 configuration, and it reproduces to three decimals (83.561 ± 24.226 against 83.56 ± 24.23).

### Peak-then-decay was a discounting artefact

| γ | 10 000 → 20 000 |
|---|---|
| 0.90 | 83.56 → 63.94  (**−23.5 %**) |
| 0.95 | 80.25 → 87.02  (**+8.4 %**) |
| 0.99 | 84.34 → 86.10  (**+2.1 %**) |

The phenomenon that survived three separate feature maps — E12 −21 %, E13 −44 %, E14 −24 % —
disappears at γ = 0.99 and reverses at γ = 0.95. It was never about the representation. The
argument for looking here was precisely that *a phenomenon invariant to the feature map is
probably not caused by the feature map*, and that reasoning is the most transferable thing in
this entry.

Why: at γ = 0.9 the horizon is ~10 steps, so a target 10–30 steps away is discounted into the
noise and the ordering of actions is decided by whatever the last few updates did. Lengthening
the horizon gives distant outcomes enough weight to separate states that were previously ties —
the same flatness that let `BOMB` lose by 0.006 in the E13 post-mortem.

### Predictions, scored

1. **"γ = 0.95 wins, 95–115 crates; γ = 0.99 worse" — wrong twice.** γ = 0.99 is the best cell,
   and **no cell moved the mean out of the low 80s.** I predicted a mean effect and got a
   variance effect.
2. **"std below 12 in the winning cell" — right, and by a wide margin.** 2.93 against a predicted
   12 and a baseline of 24.23. This was the entry's central prediction.
3. **"Decay under 20 % at the winning γ" — right.** It reverses sign.
4. **"A higher ε floor costs a little mean and buys spread" — badly wrong.** ε = 0.10 roughly
   halves the crate count at every γ (36.99 against 83.56 at γ = 0.9) and is the worst setting in
   the grid. Its only defence is that it also shrinks the spread — of a much worse policy, which
   is not a trade worth making. **E07's open question is now answered for rung 2: the ε floor
   should stay at 0.02, and raising it is actively harmful.**
5. **"Suicides do not rise" — held.** 0.006 at γ = 0.99, survival ≥ 0.99 throughout.
6. **`think_max_ms` unchanged — held.**

The whole-entry refutation condition (no cell beats the baseline by more than five-seed noise)
technically fires *on the mean* — 86.10 against 83.56 is nothing. It does not fire on the metric
that turned out to matter, which I had listed as prediction 2 rather than as the headline.

### Addendum: γ = 0.99 never peaks

The 5 000 and 40 000 checkpoints of the winning cell were evaluated after the fact (they were
already on disk). The curve is **monotone increasing through the whole run**:

| ep | 5 000 | 10 000 | 20 000 | 40 000 |
|---|---|---|---|---|
| `crates` | 80.25 ± 4.59 | 84.34 ± 10.47 | 86.10 ± 2.93 | **92.80 ± 9.47** |
| `score` | 5.712 | 6.052 | 6.179 | **6.671** |

So 20 000 was not the optimum — it was the better of the two checkpoints I happened to evaluate,
and picking the best of a *subset* is how a boundary value gets reported as a maximum. E12 warned
about exactly this and I did it anyway.

At γ = 0.99 there is no peak to find: the optimum is at or beyond 40 000, and the round count is
now an open question rather than a settled one. The variance win is also partly checkpoint-bound
— std 2.93 at 20 000 but 9.47 at 40 000 — though both are far below γ = 0.9's 24–37.

### Verdict

**γ = 0.99 with ε floor 0.02, trained to at least 40 000 episodes.** At 40 000: 92.80 ± 9.47
crates and 6.671 score, against the baseline's 83.56 ± 24.23 and 5.990. The spread falls by a
factor of 2.6 at the matched checkpoint and by 8.3 at 20 000, and the worst seed goes from 29.70
to 82.07. Since E10 every result has been a distribution with a long bad tail; this closes it.
It also explains, and removes, a degradation that three previous entries wrongly attributed to
their own feature changes.

**The shipped model does not change.** E13 s2 still reaches 116.64 where γ = 0.99's best seed
manages 90.02, and §5.8 selects on validation and ships one table, so the ceiling is what ships.
That is worth flagging as a genuine tension rather than settling quietly: one configuration is a
coin flip between 29.70 and 116.64, the other is 82–90 every time. For a report, the second is
the better result; for a single-table tournament submission with a validation-seed selection, the
first still wins. It matters more from rung 4 on, when a bad draw cannot be re-rolled.

**The mean is now the binding constraint, and it is a hard ceiling**: every γ, every checkpoint,
five seeds — 80 to 87 crates, against the reference's 116.26. With the variance gone, that is a
property of the feature map and the reward function, not of luck.

### What I do next

1. **E16: the reward scale, at γ = 0.99, trained to 100 000 episodes.** Necessary rather than
   optional now — moving γ from 0.9 to 0.99 multiplies the effective horizon by ten, which
   changes the weight of every reward relative to the step cost by the same factor, and the
   reward table was never tuned even at γ = 0.9. It is also where the `COIN_COLLECTED` +5 against
   the game's +1 ablation finally belongs, six experiments after it was first deferred. The
   round-count question folds into it for free: the baseline cell of the 2×2 *is* the γ = 0.99
   configuration, so its curve out to 100 000 answers where the optimum is.
2. **Then retest E14's counted digit 7 on the tuned hyperparameters.** Its only measured benefit
   was variance reduction, which γ now provides for free; whether it buys *selectivity* is
   untested and is the question its own entry failed to answer.

## E14 — Count the crates in blast range instead of asking yes/no

- **Question:** the E13 post-mortem traced the 2-cycle to a margin: in the seeds that collapse,
  half of all bombing opportunities are decided by less than 0.01 between `BOMB` and walking on,
  against 0.207 in the seed that reaches reference parity. The proposed cause is aliasing — digit
  7 is binary, so a row where one crate is in range and a row where four are is **the same row**,
  and the learned `Q(BOMB)` is an average over both. Does splitting that row restore the margin,
  and does it reduce the run-to-run variance that is now the largest term in the result?
- **Change from E13:** exactly one digit. `bomb_useful` (2 values: would a bomb here open a
  crate, and do I have one) becomes `crates_in_blast` bucketed to **0 / 1 / 2 / 3+** (4 values),
  still 0 when no bomb is available. `FEATURE_SIZES` (4,4,4,4,5,5,2) → **(4,4,4,4,5,5,4)**, table
  12 800 → **25 600**. Digits 1–6, the escape branch, and the rewards are untouched.
- **Agent:** `benedict_task2` · commit `e6ead20` · labels `benedict_q_e14_s{0..4}__ep*__task2`
- **Training:** 40 000 rounds, five seeds, world seed 810731, checkpoints 5 000 / 10 000 /
  20 000 / 40 000. Kept at 40 000 rather than E13's optimum of 10 000 because the table doubles —
  see prediction 7.
- **Measurement:** 300 rounds, ε = 0, seed 20260731. Baseline **E13 @10 000**: `crates`
  83.56 ± 24.23, `score` 5.990, crates/bomb 2.43, `suicides` 0.002. Reference: 116.26 / 8.50 /
  3.07. Model selection on 550731, per `experiments/task1.md` §5.8.

### Where this came from

Not from a plan — from watching the agent play. Benedict noticed it dropping a bomb on a single
crate where moving one tile further would have caught three. That is the behavioural face of the
same thing three measurements were pointing at: crates per bomb 2.43 against the reference's
3.07, and 1.33 on the worst seed; the flat `BOMB` margin above; and a 2-cycle in which the agent
walks up to the crate its own target digit selected and then declines to bomb it by 0.001.

Worth recording that D₄ canonicalisation was the planned E14 and was dropped on evidence: the
`cycle_dump` diagnostic showed the two rows of an actual cycle are **not** related by any element
of the symmetry group, so merging symmetric rows would not have touched the failure it was
promoted to fix.

### Prediction (written before the run)

1. **`crates` 95–120**, five-seed mean, from 83.56. If every seed bombed like s2 (2.99 crates per
   bomb at ~37 bombs) the ceiling is ~110. **Refutation:** below 85 — no better than E13 — means
   the aliasing account of the flat margin is wrong, and the near-ties come from the reward scale
   rather than from the state.
2. **The central prediction: variance collapses. `crates` std < 12**, from 24.23. This is the
   entry's reason to exist. Splitting an aliased row is what turns a 0.006 margin into a decisive
   one, so fewer seeds should land on the wrong side of a coin flip. **If the mean rises and the
   spread does not, I have bought crates without fixing the mechanism**, and the next experiment
   is about the reward, not the features.
3. **crates per bomb 2.9–3.4**, from 2.43. It may exceed the reference's 3.07: the agent can now
   decline a one-crate spot and hold its bomb for a three-crate one, which `rule_based_agent`
   does not do.
4. **`table_check` finding 7 (the new one): median margin > 0.15 in every seed**, and fewer than
   15 % of bombing rows decided by less than 0.01 (E13: 0.006–0.207, and 5–56 %). Checkable in a
   second per seed, before any 300-round run.
5. **Loop probe: fewer than 5 of 20 rounds confined before 90 % of the round, in every seed.**
   E13's reference-parity seed manages 2/20; its collapsed seeds are at 19/20.
6. **Regression guard: `suicides` ≤ 0.01, `survived` ≥ 0.99.** The agent will now hold its bomb
   while it looks for a denser spot, which means more time standing next to crates in pockets.
7. **The optimal checkpoint moves later, to 20 000.** The table doubles, so it needs more data;
   E13 peaked at 10 000 and E12 at 20 000. **If it peaks at 40 000, the table is genuinely
   data-starved and D₄ canonicalisation comes back onto the list** — for its sample-efficiency
   argument, which survived the post-mortem, rather than for the cycle argument, which did not.

**Refutation condition for the whole entry:** `crates` < 85 *and* std > 20. That is E13's result
with a bigger table, and it would mean the margin is set by the reward function rather than by
what the state can distinguish — in which case the next experiment is `CRATE_DESTROYED` and the
step cost, and the long-overdue `COIN_COLLECTED` +5 against +1 ablation goes with it.

### Result

Five seeds × four checkpoints, 300 rounds each at ε = 0, seed 20260731:

| checkpoint | `crates` | std | `score` | `suicides` | `survived` | crates/bomb |
|---|---|---|---|---|---|---|
| 5 000 | 73.48 | **5.42** | 5.189 | 0.004 | 0.996 | 2.36 |
| 10 000 | 73.88 | 26.76 | 5.243 | 0.003 | 0.997 | 2.48 |
| **20 000** | **83.86** | 15.37 | **6.091** | **0.035** | **0.965** | 2.69 |
| 40 000 | 63.61 | 24.33 | 4.545 | 0.001 | 0.999 | 2.77 |

Baseline **E13 @10 000: 83.56 ± 24.23 crates, 5.990 score, 2.43 crates/bomb.** Reference:
116.26 / 8.50 / 3.07. `think_max_ms` 0.202.

**The mean did not move: 83.86 against 83.56.** The spread fell (15.37 against 24.23, and 5.42 at
the 5 000 checkpoint), and the ceiling came down with it — E13's best seed reached 116.64,
reference parity; E14's best single table is 106.63. On the validation seed 550731, every E14
table loses to the incumbent:

| | best E14 @5 000 | best E14 @20 000 | **E13 s2 @10 000 (shipped)** |
|---|---|---|---|
| `crates` @550731 | 78.9 | 103.2 | **117.1** |

**Verdict: not demonstrated on the primary metric, and the shipped model does not change.**

### The intervention worked on the table and not on the behaviour

The mechanism it was aimed at moved, and moved a lot. `table_check` finding 7 — the margin
between the best legal move and `BOMB` where a bomb pays off:

| | E13 | E14 |
|---|---|---|
| median margin, across seeds | 0.006 – 0.207 | **0.11 – 0.21** |
| share decided by less than 0.01 | 5 – 56 % | **2 – 14 %** |

The collapsed-seed regime — half of all bombing decisions settled by a coin flip — is gone. And
`crates`/bomb went 2.43 → 2.69. **A 10 % gain, against the 2.9–3.4 predicted.** The agent became
*decisive* without becoming *selective*.

**Why, and this is the finding worth keeping.** A local count tells the agent how good *here* is.
It does not tell it that *there* is better. Digit 6 still targets the **nearest crate**, so the
agent walks to the nearest crate and the count only lets it decide whether to bomb once it has
arrived. To act on the observation that prompted this experiment — walk one tile further and
catch three crates instead of one — the agent has to be *sent* to the denser spot. That is a
property of the target digit, not of the count. I put the fix in the wrong digit.

### Predictions, scored

1. **`crates` 95–120 — wrong.** 83.86, and the refutation clause I wrote ("below 85") fires by
   1.1 crates.
2. **"std < 12" — half right, and the half that held is the informative one.** 5.42 at 5 000
   (mean 73.48), 15.37 at the best checkpoint. I wrote: *"if the mean rises and the spread does
   not, I have bought crates without fixing the mechanism."* **The exact opposite happened** —
   the spread collapsed and the mean stood still, which says the mechanism moved and was not the
   thing holding the mean down.
3. **crates/bomb 2.9–3.4 — wrong.** 2.36–2.77.
4. **Margin > 0.15 in every seed, under 15 % noise — half right.** The noise half holds
   everywhere except s2 @40 000 (30.1 %); the median half holds only at 5 000, and only for four
   seeds of five.
5. **Loop probe under 5/20 confined — wrong.** 4–20 of 20 depending on seed and checkpoint.
6. **Regression guard — failed at one checkpoint, and it is the best one.** 0.004 / 0.996 at
   5 000 and 0.001 / 0.999 at 40 000, but **0.035 suicides and 0.965 survival at 20 000, driven
   by s3 at 0.140 / 0.860** — the same seed that tops the crate count at 106.63. The most
   productive table is also the one that dies in one round in seven. That is the aggression /
   safety trade `AGENTS.md` warns about arriving a rung earlier than expected.
7. **"The optimum moves later, to 20 000" — right**, on both `crates` and `score`. Worth noting
   because I talked myself out of it after seeing the margin gate peak at 5 000 and said so
   before the evaluations: the static gate pointed at the wrong checkpoint, the measurement
   pointed at the right one. **The gate ranks tables, it does not rank policies.**

The whole-entry refutation condition (`crates` < 85 **and** std > 20) does not fire: the crates
half does, the variance half does not.

### Verdict

**Not demonstrated.** 83.86 ± 15.37 against 83.56 ± 24.23 is no improvement in the mean, it costs
a doubled table, and it loses the reference-parity seed. Negative result, stays in the report —
and it is a useful one, because it separates two things that looked like one: the flat `BOMB`
margin was real and is now fixed, and it was **not** what was capping the crate count.

### What I do next

1. **E15: digit 6's crate branch targets the densest reachable bombing spot**, not the nearest
   crate — the fix for the observation E14 was supposed to address, in the digit that actually
   controls where the agent goes. This needs care about the "feature returns the best action"
   rule and the argument has to be made explicitly in the report: it is a pathfinding feature
   with a value criterion, and the agent still has to learn *when* to bomb, when to run, and when
   to chase a coin instead.
2. **Run E15 as two arms**, because whether E14 is worth keeping is now an open question rather
   than a settled one: arm A on the E13 base (binary digit 7), arm B on the E14 base (counted
   digit 7). If the count only pays off once the agent is *sent* to dense spots, arm B wins and
   E14 was a prerequisite rather than a failure. If the arms tie, digit 7 reverts to binary and
   the table halves. Two arms × 5 seeds run concurrently in the same wall clock.
3. **Watch `suicides` at every checkpoint from now on, not just at the reported one.** Prediction
   6 held at three checkpoints of four and failed at the one that mattered.

---

## E13 — Point digit 6 at the crate itself, not at a tile that can hit one

- **Question:** E10 measured that 99.7 % of free tiles on a fresh `classic` arena are
  crate-bombing positions, so `target_direction` — which returns `NO_TARGET` when the agent is
  *standing on* a target — reads 0 almost everywhere while the agent is safe. Does making the
  crate itself the goal restore the gradient, and how much of the remaining gap to the reference
  does that close?
- **Change from E11/E12:** exactly one. In the no-coins branch, the BFS goal becomes a **crate
  tile** (`field == 1`) rather than a free tile from which a bomb would reach one. Crates stay
  impassable — they are the goal, not the path — so the digit points *at* the crate the agent
  should walk up to and bomb. Coin branch unchanged, except that an unreachable coin now falls
  through to the crate branch instead of returning `NO_TARGET`. `FEATURE_SIZES`, the danger
  branch from E11, and the rewards are all untouched.
- **Agent:** `benedict_task2` · commit `b157c45` · labels `benedict_q_e13_s{0..4}__ep*__task2`
- **Training:** 40 000 rounds, five seeds, world seed 810731, checkpoints at
  **5 000 / 10 000 / 20 000 / 40 000**. Trained past E12's 20 000-round optimum on purpose — see
  prediction 5.
- **Measurement:** 300 rounds, ε = 0, seed 20260731, `--preset task2`, loop probe at every
  checkpoint. Baseline is **E12 at 20 000**, the best measured table: `crates` 33.32 ± 4.90,
  `score` 2.287, `suicides` 0.001, crates/bomb 1.91.

### Prediction (written before the run)

1. **`crates` 55–85**, five-seed mean, from 33.32. The bound: a bomb-and-escape cycle is ~7 steps,
   so 400 steps allow ~57 bombs, and at the current 1.91 crates per bomb that is ~109 — the
   agent is currently losing most of that to standing still, not to bombing badly.
   **Refutation:** below 40 means walking to crates was not the binding constraint and digit 7's
   inability to rank bombing spots (E14) dominates instead.
2. **The loop probe is the mechanism check: median distinct tiles per round > 60** (E12 at
   20 000: 25) and **fewer than 10 of 20 rounds confined to ≤ 2 tiles** (currently 20 of 20).
   If `crates` rises but this does not, the gain came from somewhere I have not identified and
   the entry's explanation is wrong even if its number is good.
3. **`score` 4–7**, from 2.287. More crates opened means more coins revealed, and the coin branch
   already works once they are visible.
4. **Regression guard: `suicides` ≤ 0.01 and `survived` ≥ 0.99.** The agent will now deliberately
   walk *up to* crates, so it spends far more time in dead ends and pockets — exactly the
   geometry where the escape BFS returns `NO_TARGET`. `AGENTS.md`'s warning is that learning
   aggression is when escape gets forgotten.
5. **The 20 000 → 40 000 degradation shrinks: `crates` at 40 000 ≥ `crates` at 20 000 − 3**
   (E12: −7.1, with four seeds of five worse). E12 attributed that decline to near-ties in the
   safe-branch rows hardening into a 2-cycle. If that diagnosis is right, removing the constant
   digit removes the degradation. **This is a test of E12's mechanism, not of E13's**, and it is
   the reason this run goes to 40 000 rather than stopping at the known optimum.
6. **New hazard, stated in advance: `table_check` finding 1 rises.** Digit 6 now points at a
   *blocked* tile whenever the agent is adjacent to its target crate, so a row can learn "walk
   into the crate" — invalid, state unchanged, absorbing. Predict finding 1 above E11's 0–3 but
   below 20, and **zero frozen spawns**. If spawns freeze, this change is a net loss regardless
   of what `crates` does.
7. **crates per bomb stays at 1.8–2.0.** E13 changes *where the agent goes*, not *how well it
   picks a spot*. Holding this constant is how I will know E13 and E14 are separable rather than
   two descriptions of one effect.

**Refutation condition for the whole entry:** `crates` < 40 *and* the loop probe unchanged. That
means the change did not do the thing it was designed to do, and the 99.7 % diagnosis — which is
the argument E10, E11 and E12 all lean on — is wrong about what the agent is actually missing.

### Result

Five seeds × four checkpoints, 300 rounds each at ε = 0, seed 20260731:

| checkpoint | `crates` | std | `score` | std | `suicides` | `survived` | crates/bomb |
|---|---|---|---|---|---|---|---|
| 5 000 | 80.61 | 29.85 | 5.773 | 2.28 | 0.008 | 0.992 | 2.44 |
| **10 000** | **83.56** | 24.23 | **5.990** | 1.85 | 0.002 | 0.998 | 2.43 |
| 20 000 | 63.94 | 36.77 | 4.565 | 2.77 | 0.002 | 0.998 | 2.89 |
| 40 000 | 46.69 | 44.11 | 3.358 | 3.22 | 0.001 | 0.999 | 2.40 |

Baseline (E12 at 20 000): 33.32 crates, 2.287 score. Reference (E08): 116.26 / 8.50.
`think_max_ms` 0.272 against a 500 ms budget.

**`crates` 33.32 → 83.56 and `score` 2.287 → 5.990, from one change to one digit.** That is 72 %
of the reference on crates and 70 % on score, against 29 % and 27 % before.

Paired, s0: E12 @20 000 → E13 @10 000, `crates` +18.677 [+13.193, +24.300] **BETTER**, `score`
+1.353 [+0.910, +1.790] **BETTER**.

### One seed reaches the reference

Seed 2 at 10 000 episodes, paired against `coin_collector_agent` over the same 300 arenas:

| Metric | reference | s2 @10 000 | diff | 95 % CI | verdict |
|---|---|---|---|---|---|
| `score` | 8.500 | **8.513** | +0.013 | [−0.160, +0.170] | **no effect shown** |
| `crates` | 116.26 | **116.64** | +0.377 | [−1.873, +2.363] | **no effect shown** |
| `bombs` | 37.83 | 39.04 | +1.210 | [+0.450, +1.897] | BETTER |
| `survived` | 1.000 | 0.997 | −0.003 | [−0.010, +0.000] | no effect shown |

**A 12 800-row tabular Q-table is statistically indistinguishable from the reference agent on
rung 2.** Not the mean of five seeds — one seed of five — which is the whole problem, below.

### Predictions, scored

1. **`crates` 55–85 — right.** 83.56 at the best checkpoint.
2. **Loop probe: "median distinct tiles > 60" — right** (112 at 5 000, 107 at 10 000, from 25).
   **"Fewer than 10 of 20 rounds confined" — unanswerable as written, because the metric was
   wrong.** The probe counted a round as confined even when the cycle started at step 396 of
   400, i.e. when the round had effectively ended, so it read 20/20 for tables that play the
   whole round. Fixed afterwards (`--stuck-before`, default 0.9) and re-validated against a
   known-bad table (E12 s2 @40 000: 19/20, entry step 4) and a known-good one (E13 s2 @10 000:
   2/20, entry step 319). **Changing a metric after seeing the data is exactly the move that
   invalidates a prediction**, so this half of prediction 2 does not count as confirmed, and the
   fix has to earn its trust on E14 instead.
3. **`score` 4–7 — right.** 5.990.
4. **Regression guard `suicides` ≤ 0.01, `survived` ≥ 0.99 — right.** 0.002 / 0.998, despite the
   agent now deliberately walking up to crates.
5. **"The 20 000 → 40 000 degradation shrinks" — wrong, it grew.** E12 lost 7.1 crates (−21 %);
   E13 loses 17.3 (−27 %), and the peak moved *earlier*, to 10 000. E12's mechanism claim is not
   refuted — the collapsed seeds are still 2-cycles, and s1 and s4 at 40 000 are confined to two
   tiles **from step 0** — but giving digit 6 a gradient does not prevent a row from acquiring an
   argmax that closes a cycle. It only made the productive phase longer before it happens.
6. **`table_check` finding 1 above 0–3, below 20, no frozen spawns — right.** 6–18 across all
   checkpoints, every seed passing the spawn gate. The new hazard I predicted (digit 6 pointing
   at a blocked crate) is real and small.
7. **"crates per bomb stays at 1.8–2.0" — wrong.** 2.4–2.9, up from 1.91. **E13 and E14 are not
   separable after all:** walking to crates does not only get the agent to more of them, it also
   makes it bomb from denser positions. So part of the headroom I attributed to digit 7 has
   already been collected, and E14's expected gain must be revised down before it is run.

### The round count is a property of the configuration, not of the setup

E12 concluded "20 000 rounds is the standard from E13 on". **One feature change moved the optimum
to 10 000**, and 20 000 now costs 24 % of the crates. So that verdict was over-generalised: the
optimal training length is not a constant of this project, it is a property of each
configuration, and the only way to know it is to checkpoint and measure. The cost of getting this
wrong is large and one-directional, so **checkpointing stays in every experiment from here on**
rather than being replaced by a fixed round count.

### Variance is now the binding constraint

| checkpoint | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| `crates` @10 000 | 54.75 | 66.24 | **116.64** | 85.74 | 94.45 |
| `crates` @40 000 | 98.64 | **10.82** | 28.35 | 89.41 | **6.23** |

Identical code, identical world seed, different exploration RNG: a factor of **2.1** between the
best and worst seed at the peak checkpoint, and a factor of **16** at 40 000. The bad seeds fail
the same way every bad seed on this project has failed since E04 — an absorbing 2-cycle — and at
40 000 two of them enter it at step 0.

**Model selection, done by the agreed rule.** `experiments/task1.md` §5.8 — select on the
held-out world seed 550731, report on 20260731 — applied to the five tables at the 10 000
checkpoint:

| | s0 | s1 | **s2** | s3 | s4 |
|---|---|---|---|---|---|
| `crates` @550731 (validation) | 56.76 | 62.52 | **117.14** | 85.77 | 93.76 |
| `crates` @20260731 (reported) | 54.75 | 66.24 | **116.64** | 85.74 | 94.45 |

The ranking is identical on both seeds and the values agree to within 4 %, so **the spread is a
property of the tables, not of the arenas** — s2 is genuinely a better policy, not a luckier
draw. §5.8 selects s2, and that table is now `agent_code/benedict_task2/q_table.npy`.

I nearly got this wrong in the other direction: I had E06's "always ship run index 0" in mind,
which §5.8 already superseded on rung 1 *because* index 0 turned out to be the worst of five
there. Blind index 0 would have shipped 54.75 crates instead of 116.64. Selecting a model on
held-out data and disclosing it is not cherry-picking; reporting the selection score as if it
were an unbiased estimate would be. **The headline stays the five-seed distribution
(83.56 ± 24.2); the shipped model's 116.64 is reported separately as a selected model.**

None of which makes the variance acceptable — it is now the largest single source of uncertainty
in the result, and it is what the next experiment attacks.

### Verdict

**BETTER, decisively, and rung 2 is within reach.** 83.56 crates / 5.990 score as a five-seed
mean, with one seed at reference parity, from 33.32 / 2.287. Suicides and survival unchanged at
0.002 / 0.998 — the E11 escape branch holds up under a policy that now seeks crates out.

The remaining gap to the reference is no longer a feature gap. It is **run-to-run variance with
an identified mechanism** (the 2-cycle), and that is what the next experiment has to attack.

### Post-mortem: what the 2-cycle actually is

Three entries in a row I named a mechanism for the cycle and was partly wrong, so rather than
guess a fourth time I opened one up (`scratchpad/benedict/cycle_dump.py`, written for this).
Dumping the two rows of the first early-collapsing round, for two collapsed tables:

| table | cycle | tile A | tile B |
|---|---|---|---|
| s1 @40 000 | (1,1)↔(2,1) | target `DOWN`, argmax `RIGHT`, Q(R)=0.163 Q(D)=0.163 | target `LEFT`, argmax `LEFT` |
| s4 @40 000 | (1,1)↔(1,2) | target `RIGHT`, argmax `DOWN`, Q(R)=−0.041 Q(D)=−0.032 | neighbours (3,0,0,0) — a **dead end**, only exit `UP`; Q(UP)=0.046, Q(BOMB)=0.045 |

**No D₄ element maps either pair of rows onto the other.** They are genuinely different states, so
canonicalising the table by its symmetry group would neither merge them nor touch this cycle —
which kills the argument I had promoted E14 on. Digit 6 does flip between the tiles, but that is
a symptom: in s4 it points `DOWN` into a crate the agent has walked up to and then declines to
bomb, by **0.001**.

That is the real finding, and it generalises. Margin between the best legal move and `BOMB`,
across all trained rows where a bomb would open a crate:

| table | median margin | share below 0.01 | spread over *legal* actions |
|---|---|---|---|
| s1 @40 000 (collapsed) | **0.0089** | 51 % | 0.201 |
| s4 @40 000 (collapsed) | **0.0061** | 56 % | 0.171 |
| s0 @10 000 (mediocre) | 0.1238 | 19 % | 0.272 |
| **s2 @10 000 (reference parity)** | **0.2067** | 5 % | 0.418 |

**The difference between a reference-parity agent and a collapsed one is whether `BOMB` beats
walking on by 0.2 or by 0.006.** In the collapsed tables half of all bombing opportunities are
decided by a margin under 0.01, i.e. by noise. This is E05b's row 409 again — a near-tie in a
high-traffic cell is not a small error, it is the policy — except that per-cell α converged these
cells honestly. The values really are nearly equal.

**Why they are nearly equal is state aliasing, and it is exactly what Benedict noticed while
watching the agent play**: it bombs a single crate where moving one tile further would have taken
three. Digit 7 is *binary*, so "one crate in range" and "four crates in range" are the same row.
The learned `Q(BOMB)` there is an average over both, which lands close to the value of walking
on — precisely the flat margin measured above. The behavioural observation, the crates-per-bomb
gap and the cycle all have one cause.

### What I do next

1. **E14: digit 7 becomes a bucketed count** (0 / 1 / 2 / 3+), 2 → 4 values, table 12 800 →
   25 600. Three independent lines of evidence now point at it: the behavioural observation, the
   crates-per-bomb gap (2.43 against the reference's 3.07, and 1.33 on the worst seed), and the
   margin table above. The prediction is not only more crates but **less variance**, since
   splitting an aliased row is what turns a 0.006 margin into a decisive one.
2. **D₄ is off the list for now.** The dump shows the cycling rows are not symmetry-related, so
   canonicalisation would not address the failure it was promoted for. Its sample-efficiency
   argument survives and becomes relevant again if E14's larger table proves data-starved — and
   the `bfs_first_step` tie-break bias (45.5 / 30.6 / 13.5 / 10.5 %) still has to be fixed before
   any of that.
3. **`table_check` finding 6 is measuring the wrong thing.** It takes `np.ptp` over the whole
   Q-row, which is dominated by the ≈ −1 that invalid actions carry, so it reported a "healthy"
   median spread of 1.5–2.7 for tables whose legal actions are separated by 0.17. Spread over
   *legal* actions is the diagnostic; the current one cannot distinguish s2 from s4.

---

## E12 — A learning curve measured at ε = 0, and what the round count should be

- **Question:** the E11 training curve rises to ~34 crates by episode 10 000 and then oscillates
  around it for 30 000 more. Is that convergence, or is the training curve hiding progress the
  greedy policy is still making? And what round count should every experiment after this one use?
- **Change:** none to the agent, the features or the rewards. `train.py` additionally writes the
  table at five checkpoints. Learning is untouched — the same updates in the same order — so this
  entry's answer transfers to E13 onward.
- **Agent:** `benedict_task2`, the E11 configuration exactly · commit `203fa13` · labels
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
what should ship is the best *measured* checkpoint. The team already has the rule for choosing
among trained models — `experiments/task1.md` §5.8: **select on the held-out world seed 550731,
report on 20260731, ties to the lowest index** — and it extends to checkpoints unchanged. Stated
here in full because I had been carrying E06's superseded "always run index 0" in my head:
**select the (checkpoint, seed) pair on 550731, report it on 20260731, and report the five-seed
distribution beside it.**

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
- **Agent:** `benedict_task2` · commit `b765202` (**-dirty**: the tree carried uncommitted
  changes when this was measured, so the hash alone does not pin the code) · labels
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
- **Agent:** `benedict_task2` · commit `c188bb9` (**-dirty**, see E11) · labels
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
- **Agent:** `benedict_task2` · commit `7799000` (**-dirty**, see E11) · label
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
  commit `7799000` · labels `ref_<agent>__task2`
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
