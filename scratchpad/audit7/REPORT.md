# Audit 7 — the claim is half right, and the half that fails is the half that matters

Commissioned to **refute** the decision to stop optimising rung 4, not to ratify it. The agent's
own report writes were blocked by its harness; this is its returned deliverable. Its scripts and
data are on disk in this directory and every number is re-derivable from them.

**Independently verified before any action was taken:** the digit-8 pinning and the 64 %
unreachable-row count (from `callbacks.py` and a row enumeration), the digit-7 state-frequency
correction (1.63 : 1), the row-55060 visit split, and the broadcast counts (1 131 sources / 820
stale rows). See E35's correction block and E36 in `experiments/benedict.md`.

---

## Verdict

Split the claim in two:

> **(A)** Tabular Q on *this 8-digit feature map* is at its ceiling at 3.719 / `won` 0.372.
> **(B)** Further rung-4 optimisation is not worth the remaining five weeks.

**(A) survives — and now with a mechanism.  (B) breaks.** The evidence for (A) is five
interventions that all failed *because of the representation*, and **no feature-map change has ever
been trained on rung 4**. The stopping decision rests on an untested premise, and the test costs
**zero new table rows** and ~2.5 h of a ~30-sweep budget.

## 1 · The premise was never tested, and its blocking condition expired

E19/E32/E33 move the reward, E34 the initial condition, E35 a price. **Not one changed the feature
map.** The last feature change was **E26 (rung 3)** and it bought **+1.595 score**.

E28 ranked the missing feature — opponent BFS distance — and deferred it explicitly: *"comes
**after** coverage, never alone… every new digit makes coverage worse."* E29's coverage fix failed
at its gate; E30-E35 went to reward/init/price. **Nobody re-checked the deferral condition.**

| ε = 0 alive steps landing in an all-zero Q-row | |
|---|---|
| E28's solo-trained table | **0.026** (73.4 % of last-savable-steps) |
| shipped rung-4 table (n = 46 424 steps) | **0.0001** (4 steps in 46 424) |

Coverage is solved. 99 % of steps live in **703 rows**; the table holds 4 175 valued rows. The gate
has not held since E30.

## 2 · A structural blind spot exactly where the deaths are

In `state_to_features`, when `own_danger > 0` digit 6 is overwritten by the escape direction **and
digit 8 is pinned to `DIST_NONE`**. Measured: digit 8 = 0 on **20 370 of 20 370 danger steps**
(43.9 % of all steps). Consequently **40 960 of 64 000 rows (64 %) are structurally unreachable**,
and in the rows where essentially every death happens the state carries **no opponent information
at all**.

## 3 · The missing variable is opponent proximity — three ways

**(a) Deaths** (n = 164 deaths, 200 rounds, ship seed):

| nearest opponent at the last step before death | share | base rate |
|---|---|---|
| **BFS ≤ 2** | **0.933** | 0.153 |
| 3-5 | 0.055 | 0.288 |
| ≥ 6 | 0.012 | 0.395 |

**6.1× lift.** 94 % of these are own-bomb deaths (`died` 0.799, `suicides` 0.751 at n = 1000).

**(b) The decisive row is a mixture, not a fixed point.** Row 55060 — which audit 6's "second
Bellman fixed point" diagnosis and all of E34 rest on:

| row 55060, 809 visits | n | P(die ≤ 4) |
|---|---|---|
| opponent ≤ 2 | 148 | **0.365** |
| opponent 3-5 | 354 | **0.000** |
| opponent ≥ 6 | 301 | **0.000** |
| unreachable | 6 | 0.000 |

**Zero deaths in 655 visits**, permutation p < 1e-4. Not one state with a 5.1 % death rate priced
correctly — a **0 %-fatal sub-state (81 % of visits) glued to a 36 %-fatal one (18 %)**, with the
fixed point their average. Same pattern in 59160, 34110, 8720, 32160, 34210 (all p ≤ 0.017).
**Pooled over 41 rows: observed 39.81 vs null 3.68 ± 0.79, p < 0.0005; 58.5 % of the death mass
sits in rows the digit separates at p < 0.05.**

**(c) It changes what the optimal action is worth.** Audit 5's ceiling rule restricted to each half
(n = 1000, validation seed 550731, shipped table, paired arenas). `esc` reproduces the ledger's
ceiling (4.445 / 0.445 vs 4.399 / 0.442) — harness validated.

| arm (fires on) | score | won | died | **score per death avoided** |
|---|---|---|---|---|
| ctl | 3.742 | 0.381 | 0.799 | — |
| **near** — escape only when opponent ≤ 2 (7.5 % of steps) | 3.844 | 0.381 | 0.436 | **0.28** |
| **far** — escape only when opponent > 2 (26.9 % of steps) | 4.160 | 0.421 | 0.419 | **1.10** |
| esc — unconditional (34.0 %) | 4.445 | 0.445 | 0.334 | 1.51 |

Paired: `far − ctl` score **+0.418 [+0.178, +0.660]**; `near − ctl` +0.102 [−0.128, +0.332] (not
demonstrated). **`far − near`: score +0.316 [+0.079, +0.545], kills +0.042 [+0.003, +0.082], coins
+0.106 [+0.006, +0.207] — at statistically identical death reduction** (`died` −0.017 [−0.060,
+0.025]).

**This refines E33 rather than contradicting it.** E33 concluded *"survival is not worth points on
this board."* Conditionally: survival bought in the far sub-state is worth **1.10** per death
avoided, in the near sub-state **0.28**. E33's conclusion is an average over an interaction the
feature map hides — which is why paying for the escape everywhere (E33) and forcing it everywhere
(E34) both installed the behaviour and lost the benefit.

## 4 · The fix costs zero rows

Digit 8 is idle in every danger row, so giving it the opponent bucket **there only** costs no new
states: `FEATURE_SIZES` stays `(4,4,4,4,5,5,2,5)`, the shipped table remains a **factor-1**
warm-start parent, and the 1 131 valued `d8 = 0` danger rows broadcast into their four siblings
while 820 stale pre-E20 `d8 > 0` rows are overwritten. Measured cost **0.028 ms mean / 0.220 ms
max** per step against 500 ms.

**Exactly the move that won rung 3: E26 gave an idle digit a meaning and bought +1.595.** It also
answers the survey's "reduction beats richness" caution head-on — information at zero state cost.

## 5 · E35's stated mechanism is wrong by ~90×

E35 concluded the kill price cannot be spent because digit-7 rows *"are dominated ~145:1 by crate
opportunities (32.2 crates against 0.22 kills per round)."* That is computed from **outcomes**; what
determines whether a price changes a policy is **state frequency**. Over 46 424 steps:

| | share of all steps | share of digit-7 = 1 steps |
|---|---|---|
| crate only | 9.39 % | 58.5 % |
| **opponent only** | **5.77 %** | **35.9 %** |
| both | 0.89 % | 5.5 % |

**1.63 : 1, not 145 : 1.** The conflation is real; the reason given for it being unfixable does not
follow from the number cited.

## 6 · What did **not** break the claim

- **Learning-rule changes** (double Q, n-step, traces, α/ε schedules, optimistic init): the
  aliasing result argues *against* them — none enlarges the policy class, and no learning rule
  makes one row emit two actions. Audit 6 ranked these last; this audit supplies the reason.
- **"Don't bomb near opponents"** (n = 300, paired, with a random-dose placebo): score **−0.767
  [−1.107, −0.433]**, `kills` **−0.143 [−0.200, −0.087]**. Bombing near opponents is where our
  kills come from. The value is on the escape/movement side — which is where digit 8 is free.
- **My first hypothesis — that the near half is actively *harmful*** — **refuted.** −0.210 at
  n = 300 / ship seed, **+0.102 at n = 1000 / validation seed**. What replicates is the asymmetry
  in score-per-death-avoided, not a sign flip. Recorded as a refutation rather than reinterpreted.
- **The weak point of my own evidence:** `far − near` on **`won` is +0.040 [−0.002, +0.082] — not
  demonstrated.** The effect is established on `score`, at the boundary on `won`. E36 must be
  powered on `won`, and I say so rather than switching metric.
- **The stopping bar.** `won` 0.372 vs the 0.283 symmetric bar does beat `rule_based_agent` — but
  the survey's only directly comparable row (E, MLE SS24, tabular Q, **335 states**, 1000 rounds vs
  3× `rule_based`) is **5.04** against our 3.719. Beating the reference is a floor, not the
  tournament bar. Weaker evidence than §1-§4; I do not lean on it.

**Recorded independently:** rung-4 training *solved* the deficit E28 called the entire rung-4
problem — `killed_by` **0.525 → 0.048**. The remaining death mass is own-bomb deaths **taken next
to an opponent**. (`tools/evaluate.py:258` undercounts `killed_by` on overlapping blasts, so 0.048
is a lower bound — audit 6 defect (b), still open.)

## 7 · The experiment → became E36

`BM_OPPDIST`, default 0: in the `own_danger` branch set `target_dist` to a bucketed opponent BFS
distance {0 none/unreachable, 1 ≤2, 2 3-5, 3 ≥6}. `FEATURE_SIZES` untouched. Arms `ctl` (E33,
measured) · `OPP` · **`PLB`** carrying an information-free bucket of matched marginals
(`(x+y) mod 4`) — mandatory, because filling 40 960 dead rows is itself a perturbation.

Pre-registered predictions are in E36 in the ledger. Cost: 10 runs, ~2.5 h, **1 of ~30 available
sweeps.**

**Rank 2 — E37, split digit 7** (E35's own refutation clause: *"the answer is a feature, not a
price"*). Digit 7 from 2 to 3 values (none / crate / opponent-in-blast), 64 000 → 96 000 rows,
warm start by broadcast; re-run E35's `K25` on the split map. Caveat: opponents move during the
4-step fuse, so the bit is a weaker kill signal than a crate — the refinement is *"opponent in
blast **and** its escape is blocked."*

**Rank 3 — E32's carried-forward #3**, "digit 6's target tile is contestable", now with a
mechanism: 94 % of remaining deaths are own-bomb and 93.3 % happen with an opponent within 2, so
the opponent is interfering with the escape rather than bombing us.

**On report economics.** The aliasing measurement improves the Experiments chapter *whatever E36
returns*, because it replaces "a second Bellman fixed point exists" (true but incomplete) with one
mechanism that explains E19, E32, E33, E34 **and** E35 — measured, not argued. Stopping before
running a single feature experiment, on a rung where the last feature change was worth +1.595 and
where `AGENTS.md` says *"feature engineering beats model complexity… life-saving features"*, is the
weakest available version of that chapter.

**Artifacts** (this directory): `probe.py` → `probe_ship.pkl` (46 424 instrumented steps) ·
`analyse.py` (coverage, digit-7 decomposition, candidate lift) · `rows.py` (per-row permutation
tests) · `condrule.py` → `n300_*.json`, `cond_*.json` · `nobomb.py` → `nb_*.json` · `paired.py`.
