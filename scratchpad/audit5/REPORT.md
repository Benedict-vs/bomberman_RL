# Audit 5 — E32 (`BM_KILLED_SELF`) reviewed before the run

Phase 1 was done without opening `experiments/benedict.md`. Every number is **measured** (with its
n) unless marked inferred. Nothing in `agent_code/`, `experiments/`, `tools/` or the framework was
edited.

**Bottom line: do not run E32 as written.** Three separate reasons, each measured:
1. The control it compares against **does not replicate** (§2 R0).
2. Its arm K15 is **measurably harmful** — −1.24 score, −0.16 `won` against a matched
   contemporaneous control, 2/2 seeds (§3).
3. The suicides are an **argmax-ordering problem in the danger rows**, and fixing that ordering is
   worth **+0.61 score and +0.06 `won` with zero training** (§0).

Files created: `scratchpad/audit5/*` · `checkpoints/benedict_task4/q_table_a5*.npy` ·
`results/train/task4_tournament/benedict_task3__q_e31_a5*_s9?.csv` (+`.meta.json`; `TrainLogger`'s
`out_dir` is hardcoded, so probe logs land in the repo tree) · `scratchpad/deaths/cases.pkl` was
**overwritten** by re-running `analyse.py`.

---

## 0. The measurement that decides the diagnosis

Force the table's argmax to equal **its own escape digit (digit 6)** whenever `own_danger > 0`.
200 visited rows, 17.4 % of greedy steps, **no training at all** (`mktables.py`, `mkesc.py`).
Pooled over 3 tables (E31 seeds 80/81/83 @20 000) × 300 paired arenas, seed 550731:

| | E31 @20 000 | escape-obeying argmax | paired diff [95 % bootstrap] |
|---|---|---|---|
| **score** | 3.827 | **4.432** | **+0.606 [+0.362, +0.847]** |
| **won** | 0.390 | **0.448** | **+0.058 [+0.014, +0.100]** |
| suicides | 0.677 | **0.304** | −0.372 [−0.418, −0.328] |
| survived | 0.274 | **0.647** | +0.372 [+0.329, +0.414] |
| crates | 32.40 | 34.30 | +1.91 [+1.11, +2.71] |
| coins | 2.68 | 2.89 | +0.21 [+0.10, +0.33] |
| kills | 0.229 | 0.308 | +0.079 [+0.038, +0.119] |
| killed_by | 0.049 | 0.049 | ±0.000 |
| think_max | 6.85 ms | 5.74 ms | — |

Per seed, `suicides` −0.390 / −0.433 / −0.293 (all CIs exclude 0); `score` +0.790 / +0.750 / +0.277.
This override is **not shippable** — it is a rule, not a learned policy — but it is the ceiling
measurement: **20 000 episodes of Q-learning leave ≈ 0.6 score and 0.06 `won` on the table in 17 %
of steps, and every number E32 pre-commits to is beaten here at zero cost.**
(The `won` figure is the least robust: two identical 300-round evaluations of one table gave `won`
0.417 and 0.377, because `rule_based_agent` also shuffles with the *stdlib* RNG that `evaluate.py`
does not seed. `score`, `suicides` and `survived` are far outside that noise.)

---

## 1. Independent diagnosis

### 1.1 One row is 35 % of all deaths
300 rounds at seed 550731 with `q_table_e31_S0_s80__ep20000` (`scratchpad/deaths/collect.py`):
**209 deaths / 300 rounds, 191 own-bomb. 74 of the 209 (35.4 %) are the same `(row, action)`.**

    row 55060  digits (3,1,3,0, 1, 1, 0, 0)
      UP clear · RIGHT lethal · DOWN clear · LEFT blocked
      own_danger = 1 (move now or die) · escape digit = UP · bomb_useful = 0
    Q = [UP 7.812  RIGHT -4.474  DOWN 7.867  LEFT -5.727  WAIT -4.500  BOMB -5.619]
    greedy = DOWN, by 0.055 over UP.

`analyse.py`'s counterfactual replays the **opponents' actually recorded moves**: for all 74,
`cf_k = 1` and the unique surviving action is **`UP`** — the direction digit 6 already reported.
Across all deaths where one action would have saved the agent, that action **equals the escape
digit in 85/93 = 91.4 %** of cases.

### 1.2 Why `DOWN` kills, 74/74 (`why55060.py`)
| | |
|---|---|
| the agent's tile was in the killing blast | 74/74 |
| the tile it moved to was **not** in the blast | 74/74 |
| nearest opponent at Manhattan distance exactly 2 | 74/74 |
| that opponent moved **into the agent's target tile in the same step** | 74/74 |

The move was invalid (`tile_is_free`), the agent stayed put, its own bomb killed it. `NB_CLEAR` is
computed from positions at the *start* of the step and cannot express "an opponent two tiles away
can take this tile before I do". **Feature blindness, not mispricing.**

Whole-sample taxonomy (`blocking2.py`, `crosstab.py`; `C1` = the time-aware frozen-opponent
survivability simulator already in `scratchpad/deaths/candidates.py`):

| proximate cause at the death step | n | share | `C1` says an escape existed |
|---|---|---|---|
| chose a genuinely safe tile, **opponent moved into it** | 103 | **0.493** | 102/103 |
| stepped onto a tile whose own digit said `LETHAL` | 85 | 0.407 | 4/85 |
| `WAIT` / `BOMB` | 21 | 0.100 | 2/21 |

So **≈ 50 % of deaths were still avoidable at the death step** and **≈ 48 % were already lost** —
rows with `digit 6 = NO_TARGET`, `Q ≈ −5` on all six actions, `P(death | row, action) = 1.000`
(rows 1050, 300, 50, 16050, 17050, 4050, 4300: visits == deaths). **No reward constant repairs a
state where every action is death.**

### 1.3 The 5 k → 20 k regression is one near-tie flipping
| | ep5000 | ep10000 | ep20000 |
|---|---|---|---|
| row **59160** greedy (danger 3, escape UP) | `RIGHT` (1.99) | `RIGHT` (1.06) | **`DOWN` (0.079)**, 4/5 seeds |
| row 59160 visits / 300 greedy rounds | 202 | — | **1488** |
| row **55060** visits / 300 greedy rounds | 82 | — | **1457** (17.8×) |
| deaths / 300 rounds | 176 | — | 209 |
| "opponent took my escape tile" | 19 (0.108) | — | 101 (0.483) |
| `WAIT`/`BOMB` deaths | 74 (0.420) | — | 21 (0.100) |

Training 5 k → 20 k **fixed** the stand-still failure (−53 deaths) and **created** the
flee-into-a-contested-tile failure (+82). The routing change is a single argmax flip at margin
**0.079**. Row 55060's margin falls monotonically in all five E31 seeds
(3.91/3.37/4.13/2.99/3.87 → 0.055/0.319/0.494/1.711/0.614).
On top of this sits the residual gap collapse E31 called *P1 PARTIAL* (`gap.py`, visit-weighted,
fixed row set, 5 seeds): E[gap] **2.728 → 2.591 → 2.235**; visit mass with gap < 1.00
**0.113 → 0.218 → 0.372**.

### 1.4 Corrections to E32's arithmetic
**a. Death already costs ≈ 11, not 5.** Median V over valued rows at ep20 000 = **5.98**
(safe rows 7.04). Q-learning already prices the forfeited continuation, so the effective terminal
cost is `5 + V ≈ 11` ≈ 11 crates. E32's headline is ≈ 2× low.
**b. A suicide's terminal reward is not −5.** `update_bombs()` credits `CRATE_DESTROYED` to the
owner *before* `evaluate_explosions()` kills it. Measured over 191 suicides: the killing bomb
destroys **0.67 crates** on average, so at `BM_CRATE=1.0` the mean terminal reward is **−4.33** and
**3.7 % of suicides are net-positive at the terminal**.

### 1.5 Other measured facts
- **7.5 bombs per round are actually placed with `bomb_useful = 0`** — 22 % of all `BOMB` actions,
  26 % of bombs placed: they destroy nothing, catch nobody, and endanger only the agent.
  120 of the 479 rows whose greedy action is `BOMB` have `bomb_useful = 0`.
- 39.7 % of steps in danger with a valid escape digit take a different action.
- `INVALID_ACTION` costs 4.1/episode against death's 4.58 — comparable and never examined.
- Period-2 cycling in feature space 0.020 (E31 did fix audit-4's pathology).
- **The `killed_by` undercount does not bite here**: across 209 deaths the forensic recorded
  exactly 209 killers, one per death, so `max(0, died − suicides)` is exact for E31 @20 000.

---

## 2. Attack on E32 as designed, ranked

### R0 (new, and fatal). **The control does not replicate.**
I re-ran E31's *exact* configuration on two fresh seeds (90, 91). Verified identical: same
warm-start parent (`md5 412b8117…`), same `.meta.json` hyperparameters except `run_index`, and the
only code difference between `HEAD` and E31's commit `7005007` in `agent_code/benedict_task4/` is
the string `EXPERIMENT = "e30" → "e31"`. My control reproduces E31's visit-weighted gap at ep5000
to three digits (**2.727** vs **2.728**) and its ep5000 evaluation exactly (suicides 0.513 vs
0.508). Then, at ep20 000, evaluated identically (300 rounds, seed 550731, ε = 0):

| `suicides` @20 000 | per run |
|---|---|
| E31 seeds 80-84 | 0.697 · 0.747 · 0.697 · 0.587 · 0.723 → **0.690** |
| audit-5 ctl seeds 90-91 | 0.500 · 0.533 → **0.517** |

| `survived` @20 000 | per run |
|---|---|
| E31 seeds 80-84 | 0.263 · 0.197 · 0.230 · 0.363 · 0.213 → **0.253** |
| audit-5 ctl seeds 90-91 | 0.480 · 0.407 → **0.443** |

**Zero overlap on either metric.** The mechanism is exactly §1.3: in row 55060 at ep20 000 all five
E31 seeds are greedy-`DOWN` (fatal), and **both** of mine are greedy-`UP` (the surviving action),
by margins of 0.53 and 0.76. The *training* curves are indistinguishable
(`KILLED_SELF`/episode 0.843 vs 0.825, `SURVIVED_ROUND` 0.084 vs 0.101) — the bifurcation is
invisible until ε = 0. That is `MEASUREMENT.md`'s "a training curve is not a result" in a sharper
form: **the outcome of a 20 000-episode run is decided by which side of a sub-1.0 margin one Q-cell
lands on, and nothing in the experiment controls it.**

Consequences: E31's headline "suicides rise 0.508 → 0.642 → 0.690" is **not established**; E32's
premise ("the guard has failed twice") rests on it; and E32 plans to compare arms on seeds 90-94
against that control. An arm effect of 0.19 is exactly the size of the between-batch shift I just
measured. **No arm comparison is interpretable until the control is re-measured on ≥ 5 fresh runs.**

### R1. The saturation argument, and what the probe says
Extra penalty needed to flip each fatal cell, `ΔQ ≈ −P(death | s,a)·Δ` with `P` measured from the
rollout (`rowdiag.py`, 83 cells): percentiles 10/25/50/75/90 = **1.08 1.08 1.08 1.08 5.17**;
fraction flipped at Δ = 10: **0.904**; at Δ = 25: **0.904** — identical. The dominant cell needs
**Δ ≈ 1.1**; E32's arms are 10× and 25× past it.
**The probe confirms the consequence**: `suicides` K15 0.383 vs K30 0.438, difference
+0.055 [−0.003, +0.112] — no dose-response.
E32's P3 refutation clause (*"K30 ≈ K15 on suicides → the policy is not responding to the price at
all and P1's result, if any, was noise"*) **cannot distinguish saturation from non-response**, so
the most likely outcome is pre-registered to be read the wrong way. And the probe shows the arms
*are* responding — very strongly, and non-monotonically (§3).

### R2. P1's target is already met, free, by a measured checkpoint
Paired over 1000 arenas, E31 seeds averaged per arena (`paired.py`):

| | ep5000 | ep20000 | diff [95 %] |
|---|---|---|---|
| score | 3.796 | 3.813 | +0.017 [−0.092, +0.127] |
| won | 0.365 | 0.378 | +0.014 [−0.005, +0.033] |
| **suicides** | **0.508** | 0.690 | +0.182 [+0.162, +0.201] * |
| survived | 0.394 | 0.258 | −0.137 * |
| crates | 32.39 | 32.38 | −0.004 |

### R3. Fewer suicides, by itself, buys nothing
Over the 15 existing E31 evaluations (5 seeds × 3 checkpoints, 1000 rounds each; suicides span
0.493–0.739): `won ~ suicides` slope **+0.038 [−0.054, +0.143]**, `score ~ suicides` slope
**−0.180 [−0.594, +0.181]**. The single-cell flip of row 55060 → UP (3 tables × 300 arenas)
confirms it directly: suicides **−0.116 [−0.156, −0.076] ***, score **−0.124 [−0.366, +0.112]**,
won **−0.027 [−0.068, +0.016]**.
`P(kill ≥ 1 | suicide) = 0.192` vs `P(kill ≥ 1 | survived) = 0.323` (n = 5000) — it is *not*
trading its life for kills. And the symmetric bar is not zero: `rule_based_agent` suicides
**0.520–0.552** in its own field (n = 1000).
**`suicides` is a diagnostic, not an objective.** What earns points is the mechanism behind it
(§0), and that must be judged on `score`/`won`.

### R4. The diagnosis is wrong about *where* the mistake is
E32: *"the mistake is 1-3 steps earlier, in states that look ordinary … it is price."* Measured:
**49.3 %** of deaths are a mistake **at the death step**, in a state that looks ordinary because the
feature map cannot see the hazard; **40.7 %** are states where every action is death. E32's own
table has the evidence (`opponent distance at death median 2.0, ≤ 3 in 96.3 %`) and reads it as
"where HUNT sends it".

### R5. P4's a-priori argument — **I was wrong here and E32 is right**
I argued the penalty would ratchet the argmax because at ε = 0.02 it lands on the greedy action
98 % of the time, in death-adjacent rows that are 44.1 % of all steps. **Measured, ep20 000,
visit-weighted on a fixed row set:** ctl **2.218**, K15 **2.391**, K30 **2.734** — the penalty
*widens* the action gap. E32's "a death penalty is action-dependent, so it will not level the
gaps" survives. Prediction D refuted; I record it as such.
P4 as *written* is still nearly vacuous: in the control, `score@20 000 − score@5 000` is
+0.017 with CI [−0.092, +0.127]. State it as a CI against a threshold.

### R6. Confounds
1. **Training arenas are unseeded in E30, E31 and E32 alike.** No launcher passes `--seed`, so
   `BombeRLeWorld.__init__` receives `args.seed = None`. E30's entry claims *"World seed 810731"*
   and `train.py`'s docstring warns about it; both are stale, and `MEASUREMENT.md`'s "reproducible
   in its arenas" is false for these runs. This is a direct contributor to R0.
2. **The control is not re-run**; arms 90-94 vs E31's 80-84. Given R0 this is not acceptable.
3. **The reference bar is not arena-paired**: `ref_rule_based_agent__task4_rb_ship990731.meta.json`
   records `"seed": null`, while E31 is evaluated at 550731. E31's "+0.56 / +0.09 vs the correct
   reference" is two independent draws; E32 inherits the claim.
4. `e32_arms.sh` needs `EXPERIMENT = "e32"` hand-edited into `train.py` and reverted afterwards.
5. `BATCH=5` runs the arms **sequentially**: ~100 min, not 60.
6. Run-to-run evaluation noise at n = 300 is ≈ ±0.04 on `won` (stdlib RNG in `rule_based_agent`,
   unseeded by `evaluate.py` — `MEASUREMENT.md` already documents this).

### R7. `BM_CRATE=1.0` leaks into the tuned quantity
See §1.4b: the control's effective suicide price is **4.33**, the arms' **14.33** and **29.33**.

---

## 3. The probe — 3 arms × 2 seeds × 20 000 episodes, run before writing this section

`scratchpad/audit5/probe_arms.sh`, identical to `e31_arms.sh` except `BM_KILLED_SELF`; tables under
`BM_MODEL_SUFFIX=_a5*`. Evaluated at 300 rounds, seed 550731, ε = 0 (`eval_probe.sh`).
**Predictions were written before the runs finished** (A–D below).

### Per run, @20 000

| run | `BM_KILLED_SELF` | score | won | suicides | survived | crates | coins | kills | bombs |
|---|---|---|---|---|---|---|---|---|---|
| ctl s90 | 0 | 3.843 | 0.387 | 0.500 | 0.480 | 33.09 | 2.79 | 0.210 | 34.0 |
| ctl s91 | 0 | 3.760 | 0.423 | 0.533 | 0.407 | 31.69 | 2.59 | 0.233 | 31.7 |
| **K15 s90** | −10 | **2.513** | **0.233** | 0.413 | 0.543 | 27.99 | 2.08 | 0.087 | 38.6 |
| **K15 s91** | −10 | **2.610** | **0.257** | 0.353 | 0.620 | 28.24 | 2.16 | 0.090 | 40.7 |
| K30 s90 | −25 | 3.887 | 0.410 | 0.433 | 0.497 | 33.22 | 2.79 | 0.220 | 29.3 |
| K30 s91 | −25 | 4.213 | 0.420 | 0.443 | 0.497 | 33.59 | 2.95 | 0.253 | 30.4 |

Paired against the **matched contemporaneous control** (same commit, same batch), 300 arenas:

| @20 000 vs ctl | K15 | K30 |
|---|---|---|
| score | **−1.240 [−1.490, −0.990] *** | +0.248 [−0.033, +0.532] |
| won | **−0.160 [−0.210, −0.110] *** | +0.010 [−0.040, +0.062] |
| suicides | −0.133 [−0.190, −0.077] * | −0.078 [−0.135, −0.022] * |
| survived | +0.138 [+0.082, +0.195] * | +0.053 [−0.003, +0.110] |
| crates | −4.277 [−5.162, −3.367] * | +1.010 [+0.082, +1.955] * |
| coins | −0.573 [−0.695, −0.452] * | +0.173 [+0.045, +0.302] * |
| kills | −0.133 [−0.173, −0.093] * | +0.015 [−0.033, +0.065] |
| bombs | **+6.79 [+4.91, +8.67] *** | −3.02 [−4.66, −1.34] * |

**K30 − K15**: score **+1.488 [+1.213, +1.763] ***, won **+0.170 [+0.117, +0.223] ***,
suicides +0.055 [−0.003, +0.112].

### Reading
- **The dose-response is strongly non-monotone, and it is the *middle* dose that breaks.** K15 at
  ep20 000 collapses on **both** seeds into bomb-spam-and-hide: bombs **+6.8**, crates **−4.3**,
  coins −0.57, kills −0.13, survived **+0.14**, score **−1.24**, won **−0.16**. Crates per bomb
  fall from 1.02 to 0.71. E32 pre-commits to K15 as the *good* arm ("suicides 0.42-0.55, won
  0.36-0.40"); measured, it lands at suicides 0.383 and **won 0.245**, which is E32's own P2
  refutation condition.
- E32's pre-committed magnitude for K30 ("suicides below 0.35 but won below 0.34") is also wrong in
  the other direction: 0.438 and 0.415.
- **On `suicides` the two arms are indistinguishable** (+0.055 [−0.003, +0.112]) while differing by
  1.49 score — so `suicides` would have ranked K15 as the better arm. Optimising it directly is
  actively misleading here.
- **Neither arm improves `won`** against its matched control.

### My four predictions, scored
- **A** (both arms cut suicides; |K30 − K15| < 0.10): **confirmed** — −0.133 and −0.078, difference
  0.055 with a CI containing 0.
- **B** (`won` improves in neither arm beyond noise): **confirmed** — K30 +0.010 (ns), K15 −0.160
  (significantly worse).
- **C** (collateral over-caution = fewer bombs/crates, larger in K30): **refuted for K15**
  (bombs +6.8, crates −4.3) and correct only for K30 (bombs −3.0 *, crates +1.0 *). The damage at
  the middle dose is over-*bombing*, not over-caution. I did not predict this.
- **D** (the visit-weighted gap does not stay flat): **refuted** — ctl 2.218, K15 2.391, K30 2.734.
  The penalty widens the gap; E32's P4 argument is right and my R5 counter-argument is wrong.

Caveat, stated plainly: n = 2 per arm. Given R0 I cannot exclude that both K15 runs landed in a bad
mode by chance — but the K15 signature is identical on both seeds and is a coherent pathology, and
it is 1.24 score away from its own matched control.

---

## 4. Verdict and corrected design

**Verdict: do not run E32 as written.** Its diagnosis is wrong in mechanism (§1, R4), its control is
not reproducible (R0), its levels sit in a saturated regime for the mechanism it names (R1) while
producing a large *non-monotone* effect it does not anticipate (§3), its primary metric is
uncorrelated with the objective over the whole range in question (R3), and its pre-committed
magnitudes are wrong for both arms in opposite directions. One of its five predictions (P4's
a-priori argument) is right and I was wrong to attack it.

### What to run instead, in order

**E32a — re-establish the control (mandatory, no new mechanism).** 5 fresh seeds of the E31
configuration, 20 000 episodes, evaluated at 1000 rounds. Report the *distribution* of `suicides`,
`survived` and the greedy action in rows 55060/59160 per run, not the mean. Predicted (from my 2
runs + E31's 5): the ep20 000 outcome is bimodal, and the pooled `suicides` mean over 10 runs lands
near 0.64 with a spread of ≥ 0.20. If confirmed, E31's guard-failure claim and every arm comparison
against it must be re-stated. Add `--seed` to the launcher so this is diagnosable next time.

**E32b — the escape-following event (the real lever).** In `game_events_occurred`, when
`own_danger(old_state) > 0`, emit `FOLLOWED_ESCAPE` / `IGNORED_ESCAPE` according to whether
`self_action == ACTIONS[digit6 − 1]`, priced `+r` / `−r`. This is exactly the "custom event at the
escape step" E32 names as its own P1 fallback; the evidence says make it the first arm.
Scale, measured: the visit-weighted `Q(greedy) − Q(escape)` over the rows the override changes is
**1.59 / 1.73 / 1.78** across the three seeds, so sweep `r ∈ {0.25, 1.0, 2.5}` — 0.25 fixes only the
near-ties (0.05–0.5), 2.5 reproduces the full override. **Pre-registered target: the §0 ceiling,
score +0.61 and suicides 0.30.** Guard: `crates` and `bombs` (the K15 pathology).

**E32c — the contested-tile digit (the feature answer).** 49.3 % of deaths are "an opponent at
distance 2 took my escape tile". None of the nine candidates in `scratchpad/deaths/candidates.py`
encodes it (`C8` only counts opponents *already adjacent*). One **appended** binary digit — "the
tile in digit 6's direction is adjacent to an opponent" — keeps `warm_start` valid at factor 2
(64 000 → 128 000 rows). This is the change that removes the cause rather than re-weighting it.

**If the price question is kept at all:** drop K15 (measured harmful), pair every arm with a
contemporaneous control on the same seeds, and sweep small — `BM_KILLED_SELF ∈ {−1, −3, −25}` —
since the measured flip threshold is Δ ≈ 1.1. Add a **placebo arm** that perturbs an unrelated
reward by a comparable amount (e.g. `BM_COIN` 5 → 7): if that moves `suicides` as much, the
mechanism is perturbation of near-ties, not price. Without it, "price" and "any perturbation
≥ 1 Q-unit" are not distinguishable by this design.

**Two free write-ups that cost nothing:** (i) E31 @5000 already gives `suicides` 0.508 at
statistically identical `score`/`won` (R2); (ii) 7.5 bombs per round are placed with
`bomb_useful = 0` (§1.5) — a 26 % waste rate nobody has costed.
