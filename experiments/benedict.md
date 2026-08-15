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
