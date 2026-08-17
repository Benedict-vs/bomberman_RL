# Audit 9 — E37 (`BM_D8` 4-arm × 15-seed sweep on rung 4)

Scope: analysis of existing data only. No training, no `evaluate.py`, nothing in
`results/` or `checkpoints/` written. All scratch work in `scratchpad/audit9/`.

---

## 0. Configuration checks (what I could NOT break)

Recorded first so the rest is read against a verified baseline.

**0.1 Every evaluation was run under the same `BM_D8` its table was trained with.**
Verified independently of the launcher, from `bm_env` in the 180 eval `.meta.json`
files (`results/eval/task4_tournament/benedict_q_e37_*.meta.json`):

| arm | `bm_env.BM_D8` at eval | files |
|---|---|---|
| ctl2 | `''` | 45 |
| PLB2 | `stripe` | 45 |
| PAR | `parity` | 45 |
| SHF | `shuffle` | 45 |

`missing bm_env: 0`. This is the failure mode `MEASUREMENT.md` warns about
(table read at the wrong indices) and it did not happen.

**0.2 All 60 training runs share the same warm parent and hyperparameters.**
From `results/train/task4_tournament/benedict_task3__q_e37_*.meta.json`:
`hyperparams.warm = "_e36parent"`, `warm_n = 100`, `alpha 0.1 / alpha_exp 0.7 /
alpha_mode visit`, `gamma 0.99`, `eps 0.2 → 0.02 decay 0.9995`, identical reward
table, `n_states 64000` — identical in all 60. `train_seed = 20260831 + (run_index - 100)`,
i.e. seed *s* is the same exploration stream in all four arms. This is the E36
defect (mismatched parents) genuinely fixed.

**0.3 The sweep straddles a commit, but the commit does not touch training code.**
`ctl2` seeds 100–109 ran at `efe16ed` (started 13:22:33 UTC); the other 50 runs ran at
`1c63585` (14:37 → 20:59 UTC). `git diff --stat efe16ed 1c63585` touches only
`MEASUREMENT.md`, `scratchpad/benedict/HANDOFF.md`, `scratchpad/benedict/e37_eval.sh`,
`tools/evaluate.py` (+14, the `bm_env` writer) and `tools/trainlog.py` (+6, same).
No change to `agent_code/**` or the framework. Working tree is clean against HEAD for
`agent_code/` and `tools/`, so the committed code is the code that ran.

**0.4 The numbers in the claim reproduce exactly.** From the 180 eval CSVs, our agent's
rows only, run means over 1000 rounds, paired t over 15 seeds
(`scratchpad/audit9/core.py` → `core.out`):

| PLB2 − ctl2, `score` | ep5000 | ep10000 | ep20000 |
|---|---|---|---|
| mean | +0.0973 | +0.1983 | **+0.2795** |
| 95 % t-CI | [−0.1089,+0.3036] | [+0.0543,+0.3422] | **[+0.0812,+0.4779]** |
| p | 0.3286 | 0.0105 | **0.0091** |
| seeds positive | 6/15 | 14/15 | **13/15** |

So the arithmetic is right and the pre-registered primary test is a genuine
pre-registered test. Everything below is about what those numbers mean.

**0.5 Other pre-registered guards.** All-zero-row share (P3-analogue): the author's own
`scratchpad/benedict/e37_coverage_*.out` gives 0.00000–0.00008, guard `< 0.01` passes.
`score = coins + 5·kills` holds exactly in all 180 files (0 violations).
No `Traceback` in any of the 60 training `.out` or 180 eval `.log` files.
All 60 training logs have exactly 20 000 episodes.

---

## 1. PART 1 — what is wrong with the claim

### 1.1 34 % of the headline effect comes from one collapsed control run

`ctl2` seed 105 is a catastrophic outlier at every checkpoint
(`scratchpad/audit9/perseed.out`):

| ctl2 `score` | ep5000 | ep10000 | ep20000 |
|---|---|---|---|
| s105 | 2.824 | 2.629 | **2.596** |
| other 14 seeds, mean | 3.682 | 3.708 | **3.688** |
| z of s105 | −3.5 | −13.2 | **−8.4** |

It is a real training collapse, not an evaluation artifact: in
`results/train/task4_tournament/benedict_task3__q_e37_ctl2_s105.csv` the mean training
score over episodes 15 000–20 000 is **1.992**, against 2.65–2.82 for all other 14 ctl2 runs.
Its crates are 27.87 vs ~32 (eval @20 000).

The 15 paired differences at ep20000 are, sorted:

```
-0.172 -0.042  0.025  0.133  0.138  0.167  0.211  0.224  0.230
 0.299  0.324  0.388  0.401  0.446  1.421   <- s105
```

The largest is **3.2× the second largest**. Consequences
(`scratchpad/audit9/robust.out`):

| PLB2 − ctl2 `score` | ep5000 | ep10000 | ep20000 |
|---|---|---|---|
| all 15 seeds | +0.0973 (p 0.33) | +0.1983 (p 0.011) | **+0.2795 (p 0.0091)** |
| drop s105 (14 pairs) | +0.0324 (p 0.68) | +0.1421 (p 0.0032) | **+0.1980 [+0.0967,+0.2993] (p 0.0010)** |
| drop 2 worst ctl2 runs (s105, s107) | −0.0388 (p 0.21) | +0.1438 (p 0.0055) | **+0.1834 (p 0.0025)** |

**The reported magnitude +0.280 is inflated by ~40 %.** The defensible point estimate is
**+0.18 to +0.20**. Note the effect gets *more* significant when the outlier is removed
(p 0.0091 → 0.0010) because the outlier inflates the variance more than the mean —
so this finding attacks the number, not the existence of the effect.

### 1.2 The confidence interval is not a valid 95 % interval

The 15 differences fail normality decisively: **Shapiro–Wilk p = 0.0009, skew +2.16,
excess kurtosis +5.17**. A one-sample t-CI on a sample with one point at 3.2× the next
is not calibrated. Distribution-free alternatives:

| test | result |
|---|---|
| paired t (as reported) | +0.2795 [+0.0812, +0.4779], p = 0.0091 |
| percentile bootstrap over seeds (20 000 draws) | [**+0.1331, +0.4815**] |
| Wilcoxon signed rank | p = 0.0015 |
| sign test, 13/15 | p = 0.0074 |

The interval should be quoted as the bootstrap one, or the estimate should be reported
outlier-trimmed. The lower bound +0.081 in particular is an artifact of fitting a normal
to a skewed sample — the bootstrap puts it at +0.133.

### 1.3 The monotonicity claim starts from a number that is entirely the outlier

"+0.097 → +0.198 → +0.280". Without s105 that is **+0.032 → +0.142 → +0.198**, and
without the two worst control runs **−0.039 → +0.144 → +0.183**. The ep5000 point is
not a small positive effect; it is zero. Monotonicity itself survives, but the sequence
does not start where the claim says it does — and at ep5000 the arm ordering is
PAR (+0.114) > SHF (+0.102) > PLB2 (+0.097), i.e. the "informative" arm is *last*.

Also: the three checkpoints are three views of the same 15 training runs, not three
replications. "Monotone across checkpoints" is one trajectory measured three times.

### 1.4 "Paired at run level" is a label with no content

Correlation between an arm's run mean and `ctl2`'s run mean at the same seed index,
`score` (`scratchpad/audit9/pairing.out`):

| | ep5000 | ep10000 | ep20000 |
|---|---|---|---|
| PLB2 | −0.472 | +0.441 | −0.216 |
| PAR | +0.044 | −0.169 | +0.112 |
| SHF | −0.042 | −0.058 | +0.301 |

Scattered about zero. `SD(paired) / SD(unpaired)` is 0.81–1.11 and **exceeds 1 in 5 of
the 9 cells** — pairing *adds* variance as often as it removes it. This is expected from
`MEASUREMENT.md`: the seed index fixes the training arena sequence and our own exploration
RNG, but `main.py` does not seed the three `rule_based_agent` opponents, so two runs at the
same index diverge. The seed index is a label, not a matched pair.

The test is still valid (a one-sample t on 15 differences), and the Welch unpaired t gives
p = 0.0043 at ep20000, so nothing collapses. But the design should not claim pairing as a
source of power.

### 1.5 …which makes the power calculation that justified n = 15 wrong by ~1.6×

The pre-registration chose n = 15 from a "paired SD of `score` differences over eight arm
sets" of **0.1825**, giving an 80 %-power MDE of 0.164. Measured *inside E37* over the
9 arm × checkpoint cells (`scratchpad/audit9/final.out`):

| | mean paired SD | 80 %-power MDE at n = 15 |
|---|---|---|
| pre-registered assumption | 0.1825 | 0.164 |
| **E37 as run, all 15 seeds** | **0.3287** | **0.256** |
| E37 excluding s105 | 0.2003 | 0.162 (n = 14) |

Against the pre-registered expected effect of ~0.20, E37 as run was **~60 % powered, not
80 %** — the same defect audit 8 found in E33–E36 and that this entry exists to correct.
The reason is visible in the second row: the SD used for planning was estimated from a
family that happened to contain no collapsed run. Run-level `score` is heavy-tailed
(occasional total collapse), so a single SD does not describe it and a
normal-theory power calculation on it is not conservative.

Practical consequence for the continuation rule: a confirmation sweep sized from 0.1825
will be underpowered again.

### 1.6 The pre-registered `think_max_ms < 5` guard fails as written

Global maximum of `think_max_ms` over the 180 000 evaluated rounds is **53.40 ms**, and
**389 rounds exceed 5 ms**. `think_over_limit` is 0 everywhere, so nothing is at risk
against the 500 ms tournament budget — but the guard as pre-registered ("`think_max_ms`
< 5") is violated and the entry must either say so or restate the guard as a mean
(mean `think_max_ms` ≈ 0.27 ms, which passes comfortably).

### 1.7 "A real effect of the feature change" is over-specified — the null arm moves too

In the **training stream** (independent of the evaluation; the stream the ledger used to
corroborate E36's PLB), averaged over episodes 15 000–20 000
(`scratchpad/audit9/train.out`):

| vs ctl2, training `score` | ep0–5k | ep5–10k | ep10–20k | ep15–20k |
|---|---|---|---|---|
| PLB2 | +0.055 (p 0.070) | +0.124 (p 0.0087) | +0.247 (p 0.0008, 15/15) | +0.280 (p 0.0005, 14/15) |
| PAR | +0.034 (p 0.21) | +0.053 (p 0.25) | +0.112 (p 0.041) | +0.137 (p 0.024, 12/15) |
| **SHF** | +0.035 (p 0.23) | +0.065 (p 0.13) | **+0.123 (p 0.047)** | **+0.146 (p 0.030, 13/15)** |

**`SHF` — the arm constructed to carry no information — beats the control on training
score by +0.146, p = 0.030, 13/15 seeds, and is indistinguishable from `PAR`.** In the
evaluation it falls to +0.079 (ns). So "changing digit 8 at all" buys something, and the
entry cannot attribute the whole of PLB2's move to the information content without
addressing why its own null moves in the independent stream it used to validate E36.
(The discriminating metric is `crates`, where SHF really is null — see Part 2 §2.3.)

### 1.8 Multiplicity, stated honestly in both directions

135 arm × checkpoint × metric contrasts against `ctl2` exist in this entry; **46 are
p < 0.05** where ~7 are expected under a global null. The primary (`score`, PLB2 − ctl2,
@20 000) is protected by pre-registration and Bonferroni over 135 is *not* the right
correction for it — I say this explicitly because audit 8 already had this argument.
But every *supporting* number in the claim (coins, crates, `won`, the checkpoint
trajectory) is drawn from that unprotected family, and `won` in particular
(+0.0237 [−0.0064,+0.0538], p = 0.113) does **not** clear its own bar.

### 1.9 What I could not break

- The arms are genuinely comparable — §0.1–0.3. Same parent, same pseudo-counts, same
  hyperparameters, same code, correct `BM_D8` at evaluation time.
- The effect survives every robustness test I could construct: Wilcoxon p = 0.0015;
  sign test p = 0.0074; outlier-deleted +0.198 p = 0.0010; two-outlier-deleted +0.183
  p = 0.0025; unpaired Welch p = 0.0043; bootstrap CI excludes 0 by a wider margin than
  the t-CI; leave-one-out range +0.198…+0.312 (no seed other than s105 moves it much).
- **Split-half over the 1000 evaluation arenas**: rounds 0–499 give +0.2664
  [+0.0618,+0.4710] 14/15, rounds 500–999 give +0.2927 [+0.0889,+0.4964] 12/15.
- It replicates in the **training stream** at +0.280 (14/15), and the *crates* channel
  replicates there at +1.367 with **15/15** seeds positive.
- The gain is broad over arenas, not one lucky map: pooling the 15 seeds, 59.1 % of the
  1000 arenas are positive, median +0.267 (the top 50 arenas do carry 48 % of the total
  gain, so the arena distribution is heavy-tailed too).

**Verdict on Part 1.** The effect is real and I could not make it go away. Three things
in the claim as stated are wrong: the **magnitude** (+0.280 should be ≈ +0.19), the
**interval** (t-CI on a Shapiro-p = 0.0009 sample; the bootstrap is [+0.133,+0.482]), and
the **first point of the monotone sequence** (+0.097 is the outlier, not an effect). Two
things in the surrounding design are wrong: **pairing buys nothing**, so the power
calculation that sized the sweep is off by ~1.6× and E37 was ~60 % powered on its own
primary; and the **`think_max_ms < 5` guard fails as written**.

---

## 2. PART 2 — my own explanation of what the four arms are doing

Written from the data before looking at any characterisation of the author's. The
evidence below is mostly from the **Q-tables**, which the entry never opened; every number
comes from `checkpoints/benedict_task4/q_table_e37_<arm>_s<seed>__ep20000.npy` compared
against the shared parent `q_table_e36parent.npy` (a row counts as *visited* iff it
differs from the parent). Scripts: `scratchpad/audit9/tables.out`,
`lattice.out`, `mech.out`.

### 2.1 The one thing the manipulation adds is the crossing/corridor distinction, and the table proves it directly

`FEATURE_SIZES = (4,4,4,4,5,5,2,5)`, so row index = base·5 + d8 and own_danger = (idx//50) % 5.
For `PLB2` and `PAR`, an **even** d8 means `x+y` even ⟺ both coordinates odd ⟺ a
**crossing** (4 free neighbours, own bomb covers 12 tiles); an **odd** d8 means a
**corridor** (2 free neighbours, own bomb covers 6 tiles). `SHF` is balanced within each
class, so its labels carry none of this.

Mean over visited danger rows of `max_a Q(crossing sibling) − max_a Q(corridor sibling)`,
same base, at ep20000, over 15 seeds each:

| arm | split | seeds with the same sign |
|---|---|---|
| **PLB2** | **−0.1096 ± 0.0175** | **15/15** |
| **PAR** | **−0.0555 ± 0.0250** | 13/15 |
| **SHF** | −0.0048 ± 0.0161 (CI includes 0) | 10/15 — chance |

A clean dose–response *on the information channel itself*, measured inside the learned
table rather than in the outcome, and it orders the arms exactly as the outcomes do.
`SHF` is a true null here by construction and comes out null.

The discriminating test within `PLB2` — its digit has one informative bit (parity) and one
arbitrary bit (which diagonal band):

```
|maxQ across lattice class|  = 1.0354
|maxQ within lattice class|  = 0.3898        ratio 2.66,  15/15 seeds,  p = 1.9e-17
```

**The informative bit separates the learned values 2.7× harder than the arbitrary bit,
in every single run.** This is the single strongest piece of evidence in the whole entry
and it does not depend on any outcome metric, any CI, or seed 105.

### 2.2 And the information is *conditional on the fuse*, which is why the control cannot express it

Same split, stratified by digit 5 (`own_danger` = moves left including this one):

| own_danger | PLB2: maxQ(crossing) − maxQ(corridor) | PAR |
|---|---|---|
| **1** (last move) | **+0.3230**, 15/15 **positive** | **+0.4682**, 15/15 positive |
| 2 | −0.2346, 15/15 negative | −0.2998, 15/15 negative |
| 3 | −0.1445, 15/15 negative | −0.0800, 12/15 negative |
| 4 | −0.2212, 15/15 negative | −0.1196, 14/15 negative |

**The sign flips with the timer, unanimously across seeds, and the flip is exactly the
board physics.** With one move left you need a free neighbour *right now*, and a crossing
has four exits against a corridor's two → crossing better. With two to four moves left the
problem is clearing the blast, and your own bomb at a crossing covers 12 tiles in four
directions while a corridor bomb covers 6 and lets you duck behind a pillar → crossing
much worse. `ctl2` pins d8, so it cannot represent either half; it learns the *average of
the two*, which is the same aliasing diagnosis E36 made about row 55060, in a different place.

### 2.3 What the agent does with it: bomb siting, not survival

The policy in **non-danger** rows — where digit 8 keeps its old meaning (coin distance) and
all four arms are semantically identical — barely moves. Share of visited non-danger rows
whose argmax is `BOMB`, ep20000:

```
ctl2 0.1810   PLB2 0.1790   PAR 0.1789   SHF 0.1790
```

So the agent is **not** more willing to bomb. What changes is where it is standing when it
does, and the outcome metrics say so (ep20000, run means over 15 seeds):

| | ctl2 | PLB2 | PAR | SHF |
|---|---|---|---|---|
| `bombs` | 31.65 | **29.39** | 30.80 | 29.92 |
| `crates` | 31.38 | **33.13** | 32.43 | 31.24 |
| **crates per bomb** | **0.9996** | **1.1291** | 1.0564 | 1.0456 |
| crates/bomb − ctl2 | — | **+0.1295 [+0.0609,+0.1982] 12/15** | +0.0568 [+0.0016,+0.1121] 13/15 | +0.0460 [−0.0206,+0.1126] 9/15 |

**PLB2 drops 2.26 fewer bombs and destroys 1.76 more crates.** And the score arithmetic
closes entirely on that route: `score = coins + 5·kills` exactly (verified in all 180
files), and +0.2239 coins + 5 × 0.0111 kills = **+0.2794**, against the observed +0.2795.
Survival contributes nothing (+0.0201, ns; `suicides` −0.027, ns). This is the *fifth*
independent confirmation of the ledger's "survival does not convert to points" — the
points come from crates, and the mechanism is where the bomb is placed, not whether the
agent lives.

The causal path is short: the moment you press `BOMB` you are in a *non-danger* row, and
its Q-value bootstraps from `max_a Q(s')` where `s'` is the *danger* row at the same tile.
Under `ctl2` that successor value is identical everywhere; under `PLB2`/`PAR` it is
position-dependent for the first time (§2.1–2.2). So the manipulation prices the tile you
are about to bomb from, which is exactly a bomb-siting signal — and a crossing bomb clears
12 tiles against a corridor's 6.

### 2.4 Why PLB2 > PAR, when both carry the same single bit

Not more information — more *learning rate*. `train.py:601` sets α = 1/visits^0.7 per
(state, action) cell, with a warm pseudo-count of 100 (`warm_start`, `WARM_N = 100`).
Splitting the danger rows 4 ways instead of 2 leaves each row's visit count 2× smaller for
the whole run, hence α about 2^0.7 = **1.62×** higher throughout. The same lattice bit is
therefore written into the table harder in 20 000 episodes. Measured split: 0.1096 vs
0.0555 — **a factor 1.97**, close to what that predicts. Visited danger rows at ep20000:

```
ctl2 1093    PAR 1274    PLB2 2309    SHF 3438
```

Two testable consequences, both consistent with the data: `PAR` should lag and still be
climbing at the last checkpoint (its `crates` advantage runs +0.56 → +0.30 → +1.06, and
its `score` +0.114 → +0.104 → +0.155, both rising at 20 000), and `PAR` should catch up
with a longer run.

**This is the confound the design does not fully control.** `SHF` is a matched null for
`PLB2` (both split 4 ways, so both get the same α inflation) — good. But there is **no
2-way null**, so `PAR − ctl2` mixes the lattice bit with a 1.62× α inflation, and H2's
threshold test ("`PAR` ≥ 0.5 × `PLB2`") is not a clean test of the mechanism. It is passed
on the full sample (0.155/0.280 = 0.55) and failed with the outlier removed
(0.081/0.198 = 0.41) — i.e. it is decided by seed 105.

### 2.5 Where SHF's residual gain comes from

`SHF` is not information-free in *effect*, only in *content*. Fragmenting the danger rows
raises α exactly as `PLB2` does, and it buys crates-per-bomb +0.046 and a training-score
gain of +0.146 (§1.7). What it does **not** buy is crates in the evaluation
(−0.14, ns) or coins (+0.057, ns) — and crates is precisely the channel the lattice bit
acts on. So the correct decomposition at ep20 000 is roughly:

- **α / fragmentation component** (common to any 4-way split) ≈ SHF ≈ +0.08 score, ~0 crates
- **lattice component** ≈ PLB2 − SHF = **+0.2005 [+0.0437,+0.3572] score**, **+1.893
  [+0.696,+3.090] crates, 13/15** — and this contrast needs no control arm at all, since
  PLB2 and SHF have identical arity, identical parent, identical α behaviour and differ
  only in whether the label correlates with the lattice.

**`PLB2 − SHF` is the cleanest single contrast in E37 and it is not the one being
reported.** It is immune to §1.1 (neither arm contains a collapsed run at ep20000 except
SHF's s107, which biases *against* the contrast), immune to the α confound, and immune to
"any 4-way split works". Excluding both collapsed runs (ctl2 s105 is irrelevant here;
SHF s107 = 2.930), PLB2 − SHF is +0.1240 on score and +1.397 on crates — smaller, still
positive, still the right shape.

### 2.6 The seed-105 question, answered rather than dismissed

`ctl2` has one collapsed run in 15 (s105, and s107 at 3.354 is halfway there); `SHF` has
one (s107 = 2.930); `PLB2` and `PAR` have none (minima 3.680 and 3.643). A collapse costs
~1.1 score in that run, i.e. ~0.073 on the arm mean, and at an apparent rate of ~1/15 per
arm the collapse lottery alone contributes ≈ 0.07 SD to each arm mean and ≈ 0.10 to a
difference. That is half the size of the claimed effect. **"The control got unlucky" is a
live alternative that n = 15 cannot exclude on its own** — but it is already answered by
the outlier-deleted estimate (+0.198, p = 0.0010) and by the training stream and the
table-level dose–response, none of which depend on s105. It is *not* answered by the
headline +0.280, which is where the collapse lands.

Whether the feature *prevents* collapses (0/30 in PLB2+PAR against 2/30 in ctl2+SHF) is
an interesting hypothesis and is completely untestable at this n.

### 2.7 My explanation, stated once, with confidence levels

> Pinning digit 8 in the danger branch makes the agent's own-bomb states an *average* over
> crossings and corridors. Un-pinning it lets the table learn that the value of standing in
> your own blast depends on the wall lattice, and — unanimously across 15 seeds — that the
> dependence *flips sign* with the fuse: crossings are better with one move left (4 exits)
> and worse with two or more (12-tile blast). That value flows back into `Q(s, BOMB)` at
> the tile you are about to bomb from, so the agent bombs from better tiles: 2.26 fewer
> bombs, 1.76 more crates, +0.13 crates per bomb, and the whole score gain arrives as
> coins. `PAR` carries the same bit and shows the same effect at about half strength
> because a 2-way split leaves the learning rate 1.6× lower than a 4-way one, not because
> the bit is worth less. `SHF` carries the split without the bit and gets the learning-rate
> half only — which is small, real, and visible in the training stream.

| claim | confidence |
|---|---|
| PLB2 beats ctl2 on `score` at ep20000 | **high** — survives every robustness test + an independent stream |
| the effect is ≈ +0.19, not +0.28 | **high** — §1.1, three deletion schemes agree |
| the channel is the crossing/corridor lattice bit | **high** — across/within ratio 2.66, 15/15, p = 1.9e-17; SHF null |
| the dependence is conditional on `own_danger` and flips sign | **high** — 15/15 seeds in every stratum |
| the pathway is bomb siting, not survival | **moderate–high** — crates/bomb +0.13, score arithmetic closes on coins; I could not roll the policy out to see *where* bombs are dropped (read-only audit) |
| PLB2 > PAR is a learning-rate artifact of 4-way vs 2-way splitting | **moderate** — the 1.97 vs predicted 1.62 factor fits, but there is no 2-way null arm to test it |
| SHF's residual is the same learning-rate effect | **moderate** — consistent, but SHF is also *spatially incoherent* (a random label per tile, so digit 8 jumps arbitrarily on every step), which is a second uncontrolled difference from PLB2's smooth (x+y) mod 4 |

### 2.8 Two design gaps worth recording for whatever comes next

1. **There is no 2-way null.** `SHF` nulls `PLB2`; nothing nulls `PAR`. A `parity`-arity
   shuffle (2 groups balanced within lattice class) would cost 15 runs and would make H2
   a real test instead of a threshold on a ratio that seed 105 decides.
2. **`SHF` is not only "position without the lattice", it is also spatially white noise.**
   `(x+y) mod 4` changes by ±1 on every step, so a trajectory sees a smooth label;
   `_SHUFFLE` changes arbitrarily, so consecutive states bootstrap from unrelated rows.
   If `SHF` had come out *negative* that would have been the explanation; as it came out
   slightly positive it does not change the conclusion, but "the true null" is doing two
   things at once and the entry should say so.

---

## 3. The analysis script `scratchpad/benedict/e37_analyse.py`

Run read-only at `--ep 20000`; it reproduces every number I computed independently.
Four problems, one of them decision-relevant.

### 3.1 The continuation decision rests on an undefined test — the E36 P5 error, repeated

The script prints:

```
H3  SHF - ctl2 on score CI includes 0  AND  SHF < PAR
      +0.079 [-0.111, +0.269]  9/15   p = 0.3880   SHF +0.079 < PAR +0.155
      -> HOLDS
...
  H1 HOLDS · H2 fails · H3 HOLDS
  -> CONTINUE to the confirmation sweep.
```

`H3` "holds" by **accepting a null**. Its observed 80 %-power MDE on `score` is
2.8 × 0.3437/√15 = **0.248** (0.267 with the t-based constant) — so "the CI includes 0"
tolerates an SHF effect **larger than PLB2's entire +0.28**. H3 has essentially no power
against the alternative it is supposed to exclude.

The script *knows* this: forty lines later it prints, for `won`,
*"null, but |effect| < MDE — UNDERPOWERED, not evidence of absence"*. It does not apply
the same caveat to H3, and **H3 is the leg that fires the continuation rule**
(`H1 and (H2 or H3)`). This is exactly the defect audit 8 overturned in E36 — a
pre-registered gate scored as PASS/FAIL when it is arithmetically undefined at the
available power — moved to a new place, and this time it is the *go* decision rather than
the *stop* decision.

The fix is the one the entry already knows: state H3 as an equivalence test with a
pre-registered margin (e.g. "SHF − ctl2 upper CI bound < 0.10"), not as `p > 0.05`.
Under that margin H3 would **fail** ( upper bound +0.269).

### 3.2 The `mde` constant is the large-sample one

Line 115: `"mde": 2.8 * sd / np.sqrt(n)`. 2.8 = 1.96 + 0.84 is the *z* version. At n = 15
the correct constant is `t₀.₉₇₅,₁₄ + t₀.₈₀,₁₄` = **3.013**, so the script **understates
every MDE it prints by 7.6 %** — in the direction that makes the design look better
powered than it is, in a script whose stated purpose is to stop that happening.

### 3.3 The pre-registered MDE table does not follow from the SD it quotes

The entry states a paired SD of `score` differences of **0.1825** and an MDE table of
0.352 / 0.210 / 0.164 at n = 5/10/15. Back-solving with the t-based constant, those three
numbers imply SD = 0.2117 / 0.2111 / 0.2108 — **not 0.1825**. With SD = 0.1825 the table
would read 0.303 / 0.182 / **0.142**. So the pre-registration's own two numbers are
mutually inconsistent by a factor 1.16. Either way, both are far below the SD E37
actually realised (0.329, §1.5).

### 3.4 The docstring asserts the pairing claim §1.4 refutes

Lines 13–16: *"arm and control share `BM_RUN_INDEX`, `--seed 810731` and therefore the
same arenas and exploration stream … That is what makes n = 15 worth what the entry claims
it is worth."* Measured, the run-level correlation between an arm and the control is
−0.47…+0.44 and `SD(paired)/SD(unpaired)` exceeds 1 in 5 of 9 cells. Sharing the arena
sequence does not survive 20 000 episodes of divergent training against unseeded opponents
(`MEASUREMENT.md`, "…but that guarantee ends when opponents train alongside us").

### 3.5 What the script gets right, and which the claim omits

The **P4 guards fail, and the script says so**:

```
   ctl2  won 0.360  crates 31.38  think_max 14.54 ms   FAIL: won 0.360 < 0.36; think_max 14.54 >= 5.0
   PLB2  won 0.383  crates 33.13  think_max 21.73 ms   FAIL: think_max 21.73 >= 5.0
    PAR  won 0.367  crates 32.43  think_max 41.53 ms   FAIL: think_max 41.53 >= 5.0
    SHF  won 0.361  crates 31.24  think_max 32.03 ms   FAIL: think_max 32.03 >= 5.0
```

All four arms fail `think_max_ms < 5` and the **control also fails `won ≥ 0.36`** (0.3595).
Nothing here endangers the tournament (`think_over_limit` = 0 in all 180 000 rounds; the
mean `think_max_ms` is ≈ 0.27 ms), but a pre-registered guard that fires must be written
into the ledger as fired, or restated.

### 3.6 H2's canned "refutation reading" is wrong on the mechanism

The script prints, when H2 fails: *"the gain rides on the arbitrary positional component,
i.e. it is overfitting to the training arenas."* The Q-tables say the opposite: inside
`PLB2`, the split of `max_a Q` **across** lattice classes is 1.0354 against **0.3898**
within them — ratio **2.66, 15/15 seeds, p = 1.9e-17** (§2.1). The lattice bit carries the
value separation; the arbitrary bit carries a third as much. H2 fails because `PAR` is a
2-way split and therefore learns the same bit at a 1.6× lower learning rate (§2.4), not
because the gain is positional overfitting.

H2 fails on both readings, incidentally: with all 15 seeds it needs +0.140 and gets +0.155
but with a CI touching 0; with s105 removed it needs +0.099 and gets +0.081.

---

## 4. How I would restate the claim

> `PLB2` beats the control `ctl2` on `score` at 20 000 episodes. Paired over 15 training
> seeds the difference is **+0.280 [+0.081, +0.478] (t), p = 0.0091, 13/15 positive**, but
> the 15 differences are strongly non-normal (Shapiro p = 0.0009, skew +2.16) because one
> control run collapsed (ctl2 s105: eval score 2.596 vs 3.688, training score 1.99 vs 2.72)
> and contributes 34 % of the mean. The robust estimate is **+0.198 [+0.097, +0.299],
> p = 0.0010** with that pair deleted, **+0.183** with the two worst control runs deleted;
> bootstrap over seeds gives [+0.133, +0.482], Wilcoxon p = 0.0015, sign test p = 0.0074.
> The effect builds with training (+0.03 → +0.14 → +0.20 outlier-deleted), replicates on
> the held-out half of the evaluation arenas (+0.266 / +0.293) and in the independent
> training stream (+0.280, 14/15). It arrives as coins, not survival:
> +0.224 coins + 5 × 0.011 kills = +0.279 of the +0.280, with 2.26 **fewer** bombs and 1.76
> **more** crates, i.e. **+0.130 crates per bomb**.
>
> The mechanism is the wall lattice. In the trained tables, `max_a Q` on danger rows splits
> by crossing-vs-corridor by −0.110 in `PLB2` (15/15 seeds), −0.056 in `PAR` (13/15) and
> −0.005 in `SHF` (null by construction, null in fact); within `PLB2` the split across
> lattice classes is 2.66× the split within them (15/15, p = 1.9e-17); and it **flips sign
> with the fuse** — crossings worth more with one move left (15/15), less with two to four
> (15/15). The control cannot represent either half.
>
> The clean contrast is **`PLB2 − SHF`** (identical arity, identical parent, identical
> per-row learning-rate dilution, differing only in whether the label tracks the lattice):
> **score +0.2005 [+0.0437, +0.3572], crates +1.893 [+0.696, +3.090], 13/15.**
> `PAR − ctl2` is not clean, because there is no 2-way null.
>
> `won` is +0.024 [−0.006, +0.054], below its own MDE; no tournament claim follows.
> H3 as written is an accept-the-null at ~0 % power (its 95 % CI tolerates +0.269, larger
> than PLB2's whole effect) and should not be used to fire the continuation rule.
> Guard `think_max_ms < 5` fails in all four arms (global max 53.4 ms over 180 000 rounds,
> `think_over_limit` = 0) and guard `won ≥ 0.36` fails in the control (0.3595).
> The realised paired SD is 0.329, not the 0.1825 the design was sized from, so E37 was
> ~60 % powered on its own primary — size the confirmation sweep from 0.33.

E36's headline also replicates and grows with the matched parent: `PLB − ctl` crates was
+0.980 [+0.340,+1.620] at n = 5 (confounded); `PLB2 − ctl2` crates here is
**+1.757 [+0.892, +2.622], 14/15**.

---

## 5. Files

All under `scratchpad/audit9/`. Nothing outside it was written; nothing in `results/`,
`checkpoints/`, `agent_code/`, `experiments/` or `tools/` was modified, and no game,
training run or `evaluate.py` invocation was executed.

| file | what it is |
|---|---|
| `load.py` | reads the 180 eval CSVs into `e37.pkl` (our agent's rows only) |
| `core.py` / `core.out` | per-arm levels and all paired contrasts vs ctl2, 15 metrics × 3 checkpoints |
| `perseed.out` | per-seed run means (score/crates/survived) — where seed 105 is visible |
| `robust.out` | leave-one-out, outlier deletion, Wilcoxon for score/crates/coins/won |
| `train.out` | training-log contrasts in four episode windows |
| `pairing.out` | pairing correlations, paired-vs-unpaired SD, multiplicity count |
| `checks.out` | score identity, think-time, normality, bootstrap, crates/bomb, `won` ties |
| `final.out` | realised paired SD and MDE, arena split-half, arena-level distribution, collapse census |
| `tables.out` | visited-row counts and argmax agreement across digit-8 siblings |
| `lattice.out` | crossing-vs-corridor split of `max_a Q`; BOMB attractiveness in non-danger rows |
| `mech.out` | across- vs within-lattice-class split; the same stratified by `own_danger` |
| `qtables.py` / `qtables.out` | the three Q-table analyses above, as one re-runnable script |

---

## 6. Reconciliation of the Part 2 divergence (added after coordinator review)

**Result up front: neither of us has an index bug, neither has a sign-convention bug, and
neither used visitation weighting. We computed two different quantities on two different
row sets. The coordinator's pooled estimator is confounded by update coverage, and the
confound is large enough to reverse the sign. My matched numbers stand; the arm ordering
`PLB2 > PAR` stands and is now measured on a common base set.**

I reproduced the coordinator's numbers exactly before arguing with them
(`scratchpad/audit9/recon.py` → `recon.out`, `recon1.out`, `recon2.out`):

| pooled `even − odd`, reachable d8 only | coordinator | audit 9 reproduction |
|---|---|---|
| PLB2 | +0.4419 | **+0.4419** |
| PAR | +0.6260 | **+0.6270** |
| SHF | +0.0009 | **+0.0011** |
| PLB2 by fuse 1/2/3/4 | +0.5525 / +0.3608 / +0.4225 / +0.3973 | **+0.5525 / +0.3608 / +0.4225 / +0.3973** |
| PAR by fuse 1/2/3/4 | +0.7816 / +0.5110 / +0.5971 / +0.5662 | **+0.7817 / +0.5150 / +0.5972 / +0.5658** |

So there is no ambiguity about what was computed. Index arithmetic agrees
(`idx = 5·k + d8`, `own_danger = (idx//50) % 5`), the checkpoint agrees (ep20000), the
lattice mapping agrees (even d8 ⟺ `x+y` even ⟺ both coords odd ⟺ crossing).

### 6.1 The one line that differs

- **Coordinator, method line:** *"danger rows where ALL d8 siblings are visited
  (`q[row].any(1)` for every sibling)"* → **`q[row].any(1)` is "this row carries value",
  not "training touched this row".** Every row of `q_table_e36parent.npy` that carried
  value is non-zero in **all five** siblings, because the parent is a factor-1
  `np.repeat` broadcast (`train.py:582`). So that filter selects the 1131 warm-valued
  danger bases and admits every sibling of them **whether or not the agent ever stood
  there**.
- **Audit 9, `qtables.py` line 25:** `visited(q) = |q − par|.sum(1) > 1e-12` — a row counts
  only if training **changed** it.

### 6.2 Why that flips the sign: coverage is confounded with lattice class

Within those 1131 warm bases, share of rows training actually updated (mean over 15 seeds,
`recon.out` §1):

| arm | f(even = crossing) | f(odd = corridor) | **f_even − f_odd** |
|---|---|---|---|
| **PLB2** | 0.752 | 0.207 | **+0.545** |
| **PAR** | 0.813 | 0.215 | **+0.597** |
| **SHF** | 0.708 | 0.748 | **−0.040** |

**Corridor siblings are ~3.6× less often reached than crossing siblings**, and this is
structural, not accidental: digits 1–4 are neighbour status, a corridor has **two
permanently BLOCKED neighbours** (the pillars) and a crossing has none, so most of the
1131 bases are geometrically crossing-only and their corridor siblings can never be
visited. That is the same 94.6 %-predictability the E37 pre-registration already measured
(H(lattice | digits 1–4) = 0.192 of 0.942 bits) — seen from the row side.

An unreached row still holds the parent value. Since **all five siblings of a base start at
the same parent value**, the parent term cancels out of the group means and the pooled
contrast is *identically*

```
mean(maxQ | even) − mean(maxQ | odd)  =  f_even · ū_even  −  f_odd · ū_odd
```

with `ū` = mean uplift (learned − parent) among updated rows. Checked numerically
(`recon.out` §2) — it reproduces to four decimals:

| arm | pooled (reported) | `f_e·ū_e − f_o·ū_o` | coverage-free `ū_e − ū_o` |
|---|---|---|---|
| PLB2 | +0.4419 | **+0.4419** | +0.1115 ± 0.0104 |
| PAR | +0.6270 | **+0.6270** | +0.1659 ± 0.0179 |
| SHF | +0.0011 | **+0.0011** | +0.0357 ± 0.0050 |

Training raises `max_a Q` by ū ≈ 0.66–0.99, so a coverage gap of +0.55 mechanically
manufactures a +0.44 "crossing advantage". **The pooled estimator is measuring how often
each lattice class was visited, multiplied by how much training moves a row.**

The per-fuse table (`recon2.out`) shows the same thing stratum by stratum: the coverage gap
is +0.67 / +0.54 / +0.48 / +0.44 at `own_danger` = 1/2/3/4, i.e. present and large at every
fuse level — which is why the pooled estimator is positive at every fuse level and cannot
see the sign flip underneath it.

### 6.3 The control falsifies the pooled estimator

`ctl2` pins d8, so its table contains **zero** lattice information by construction.
Applying the pooled estimator to `ctl2` s100 over all five siblings gives **+0.3088**
(`recon1.out`). Restricted to reachable d8 = [0] it is **undefined** — the odd group is
empty.

**An estimator that returns +0.31 on a table that cannot encode the effect, and that
cannot be computed at all on the control, has no null calibration.** `SHF` looks like a
clean null under it only because `SHF`'s coverage gap happens to be −0.04 instead of +0.55
(a random label spreads visits evenly across the four values) — it calibrates the
estimator by accident of balanced coverage, not because the estimator is unbiased.

### 6.4 Which computation is defensible, and why

The quantity the mechanism claim needs is: **holding digits 1–7 fixed, does the table hold
a different value for a crossing than for a corridor?** That requires
1. **matching on the base** (digits 1–7), because crossing-only and corridor-only bases are
   *different states* — a corridor base has two BLOCKED neighbours and is worth something
   different for that reason alone; and
2. **restricting to rows training reached**, because a row at its initialisation is not an
   estimate of anything.

The matched estimator satisfies both, and the parent value cancels **exactly** within a
base, so it needs no uplift correction. Its cost is that it is computable only on the
~16 % of bases that are genuinely ambiguous (183 of 1131 for PLB2, 180 for PAR) — but that
is precisely the population where the digit can add information, and outside it the digit
is redundant with digits 1–4 by construction.

**On visitation weighting: neither computation is weighted, because the weights do not
exist.** `train.py` saves `self.q` only (`np.save(checkpoint_file(round_no), self.q)`,
line 453); `self.visits` is never written to disk. Both estimators are therefore
*unweighted means over rows*, and the ledger must say so. A visitation-weighted version
would need a fresh rollout, which this audit is not permitted to run.

### 6.5 Re-derivation under the corrected method

Matched, updated rows only, over the same 1131-base warm set (`recon.out` §4), 15 seeds:

| arm | matched `maxQ(crossing) − maxQ(corridor)` | seeds | bases |
|---|---|---|---|
| **PLB2** | **−0.1088 ± 0.0190** | **15/15 negative** | 183 |
| **PAR** | **−0.0510 ± 0.0266** | 13/15 negative | 180 |
| **SHF** (null arm) | **+0.0087 ± 0.0045** | 13/15 positive | 840 |

Two honest caveats I did not state before:
- These reproduce my §2.1 numbers (−0.1096 / −0.0555 / −0.0048) to within 0.001–0.005, so
  §2.1 stands; §6 supersedes it only in rigour, not in value.
- **`SHF` is not exactly zero**: +0.0087 ± 0.0045 excludes 0. So the matched estimator
  carries a small residual positive bias of ≈ +0.009. Null-calibrated against it, PLB2 is
  **−0.118** and PAR **−0.060**. The bias is 12× smaller than PLB2's effect, whereas under
  the pooled estimator the bias (+0.44) is 4× *larger* than the signal.

**Across- vs within-lattice-class split in PLB2**, matched and updated-only
(`recon.out` §6): **|across| 1.0217 vs |within| 0.3218, ratio 3.17, 15/15 seeds,
p = 2.1e-21** — the corrected method *strengthens* it (was 2.66). This test is internal to
PLB2 and immune to the coverage confound in the first place, because d8 = 0 and d8 = 2 have
matched coverage (76.5 % vs 76.2 % on s100), as do 1 and 3 (21.0 % vs 20.7 %).

**The `own_danger` sign flip survives unchanged** (`recon.out` §7):

| fuse | PLB2 matched | PAR matched |
|---|---|---|
| 1 | **+0.3160**, 15/15 positive | **+0.4690**, 15/15 positive |
| 2 | −0.2335, 15/15 negative | −0.2830, 15/15 negative |
| 3 | −0.1588, 15/15 negative | −0.0800, 12/15 negative |
| 4 | −0.2011, 15/15 negative | −0.1222, 15/15 negative |

and `SHF` shows no such structure (−0.010 / +0.027 / +0.021 / +0.001).

### 6.6 The ordering — measured on identical bases

The coordinator is right that this matters, and right that the naive expectation is
PAR ≥ PLB2. Comparing the two on **the 159 bases where both arms have an updated row in
both lattice classes**, paired by seed (`recon.out` §5):

| | matched split | seeds |
|---|---|---|
| **PLB2** | **−0.0684** | **15/15 negative** |
| **PAR** | **−0.0136** | 8/15 negative (≈ null) |
| **PLB2 − PAR** | **−0.0548, p = 0.0009** | |

**PLB2 > PAR on the size of the lattice split, on identical bases, 15/15 seeds.** The
coordinator's PAR > PLB2 ordering is the coverage gap ordering (+0.597 for PAR vs +0.545
for PLB2), not a value-split ordering.

**But I must concede the coordinator's larger point.** On the common bases PAR's split is
close to null (−0.014, 8/15), weaker than the −0.051 it shows on its own 180-base set. So
the α-dilution argument in §2.4 is **load-bearing, not a supporting detail**, and I
overstated when I called the table-level dose–response an ordering that matches the
outcomes "exactly". The corrected statement is:

> The lattice split is present in `PLB2` (−0.109, 15/15), weak-to-absent in `PAR`
> (−0.051 on its own bases, −0.014 on bases shared with PLB2), and null in `SHF`
> (+0.009). `PLB2` expresses the same single bit ~2–5× more strongly than `PAR`, which is
> what a 4-way split predicts against a 2-way one under α = 1/visits^0.7 (2^0.7 = 1.62× the
> learning rate for the whole run), and which the coordinator's naive expectation —
> "PAR spends its whole digit on the bit, so PAR should split harder" — predicts the
> opposite of. That the data go against the naive expectation and with the α prediction is
> now the main evidence for §2.4, and it is the part of Part 2 that most needs a
> confirmation: **a 2-way `SHF`-style null arm would test it directly**, and it is the
> single cheapest addition to any follow-up sweep.

### 6.7 What this does and does not change

| claim | status after reconciliation |
|---|---|
| digit 8 carries crossing/corridor information in PLB2 | **stands** — matched −0.109, 15/15, null-calibrated −0.118 |
| across-class split ≫ within-class split in PLB2 (mechanism is the lattice bit, not the arbitrary bit) | **stands, strengthened** — ratio 3.17 (was 2.66), 15/15, p = 2.1e-21; immune to the confound by construction |
| the dependence flips sign with the fuse | **stands** — 15/15 at every stratum |
| SHF is null on the information channel | **stands**, with a stated residual bias of +0.009 |
| the split ordering is PLB2 > PAR > SHF | **stands on identical bases** (−0.068 / −0.014 / ~0), but PAR is weaker than I implied |
| PLB2 > PAR on *score* is explained by α-dilution rather than by more information | **now load-bearing rather than supporting**; predicted correctly, untested directly, needs a 2-way null arm |
| Part 1 (all of it) | untouched |

Nothing in Part 1 depends on any of this.
