# Task 2 — `classic`, no opponents: findings, and the agent task 3 starts from

Consolidated account of rung 2, drawn from `experiments/benedict.md` **E08–E23**. Unlike rung 1
this rung was developed by one person, so there is no second path to cross-check against —
instead the arc was audited near the end by an independent session given the raw data and told
to form its own numbers before reading the ledger (`§6`). That audit changed three verdicts and
is the reason several sections below read as corrections.

The ledger carries the prediction written before each run; this file is the synthesis. The most
useful sections for the report are **§4** (what actually moved the number, and what did not) and
**§6** (the method failures, which are more transferable than the agent).

---

## 1 · The rung, and what "solved" actually means

`classic` with `--opponents none`: `CRATE_DENSITY = 0.75`, `COIN_COUNT = 9`, one agent,
`MAX_STEPS = 400`. Coins are hidden inside crates, so the agent cannot score at all until it
learns to bomb, escape its own blast, and keep navigating.

**Reference measurement (E08)** — the provided agents, each alone, 300 rounds, seed 20260731:

| Agent | crates | score |
|---|---|---|
| `random_agent` | 3.01 | 0.00 |
| `peaceful_agent` | 0.00 | 0.00 |
| `coin_collector_agent` | **116.26** | **8.50** |
| `rule_based_agent` | **116.43** | 8.41 |

As on rung 1, the reference reframes what to measure — and it took until the end of the rung to
see how completely:

1. **The board holds 122.2 crates and 9 coins.** The two reference agents destroy 116.3–116.4 of
   them, i.e. **95 %**. `crates` is therefore not an open-ended target: the distance from 116 to
   122 is the entire remaining prize, and two independent rule-based agents landing within 0.2 of
   each other is the board running out, not a coincidence.
2. **The binding constraint is the clock, not capability.** 99 % of rounds (993 of 1000 for the
   shipped agent) hit the 400-step limit with ~6 crates still standing. Everything on this rung
   is ultimately a throughput question.
3. `peaceful_agent` scores exactly **0.00** — it never bombs, so it never opens a crate. That
   makes it a useless floor here and a *perfect* one on rung 3, where survival alone has value.
4. **`suicides` is the progress signal.** An agent that has learnt `BOMB` but not escape kills
   itself immediately; E09's transfer measurement is the extreme case.

**The transfer floor (E09) is 0.00 crates.** The finished rung-1 agent, unchanged, on `classic`:
it has no `BOMB` in its usable action set and no danger features, so it walks around a board it
cannot open. Rung 2 is not an extension of rung 1's agent; the feature map had to be rebuilt.

---

## 2 · The path, in one table

Every row is 300 rounds at ε = 0 on seed 20260731, five training seeds unless noted.

| # | change | crates | score | best seed |
|---|---|---|---|---|
| E09 | rung-1 agent, unchanged | 0.00 | 0.00 | — |
| E10 | first rung-2 map: danger digits, crate target, `BOMB` | 2.41 ± 2.27 | 0.11 | 4.99 |
| E11 | digit 6 points at the **exit** while in a blast | 26.19 ± 8.78 | 1.72 | 32.10 |
| E12 | learning curve at ε = 0; round count settled | 26.19 ± 8.78 | 1.72 | 32.10 |
| E13 | digit 6 targets the **crate itself**, not a tile that can hit one | 83.56 ± 24.23 | 5.99 | **116.64** |
| E14 | digit 7 counts crates in blast (0/1/2/3+) | 83.86 ± 15.37 | 6.09 | 106.63 |
| E15 | **γ = 0.9 → 0.99**, ε floor swept | 92.80 ± 9.47 | 6.67 | 107.61 |
| E16 | reward scale: coin 5, crate 0.3, 100 000 episodes | 97.31 ± 12.54 | 7.04 | 112.37 |
| E17 | ablation panel — no component is redundant | — | — | — |
| E18 | 200 000 episodes; ε floor below 0.02 | worse / flat | — | — |
| E19 | potential-based shaping | rejected | — | — |
| E20 | **distance-to-target digit** (8th digit, 64 000 rows) | 91.92 ± 20.38 | 6.66 | 113.60 |
| E21 | warm start from a coarse table; 300 000 episodes | 106.67 ± 9.70 | 7.80 | 115.20 |
| E22 | α exponent 0.7 → 1.0 | refuted before the batch | — | — |
| E23 | `WARM_N` sweep + tie-break tolerance | **111.40 ± 4.84** | **8.16** | **116.35** |

Three changes account for almost the whole climb: **E11** (escape direction, +24), **E13**
(target the crate, +57) and **E15/E16** (discounting and reward scale, +14). Everything after
E16 is variance work, and it took seven entries to convert a 15-crate spread into a 5-crate one.

---

## 3 · The agent that ships

`agent_code/benedict_task2/`, tabular Q-learning on eight hand-built digits.

**State — mixed radix `(4,4,4,4,5,5,2,5)` = 64 000 rows × 6 actions.**

| digit | meaning | values |
|---|---|---|
| 1–4 | each neighbour: blocked / lethal this step / in a blast / clear | 4 |
| 5 | moves of grace left on my own tile (0 = safe) | 5 |
| 6 | BFS first step to the objective — the **way out** while in a blast, otherwise the nearest **coin, else crate** | 5 |
| 7 | a bomb here would open a crate **and** I have one | 2 |
| 8 | how far the digit-6 target is: 1 / 2 / 3–4 / 5+ (0 = n/a) | 5 |

**Every hyperparameter, and what decided it.** None of these is a default carried in unexamined;
each row names the entry that set it. All are the defaults in `train.py`.

| setting | value | decided by |
|---|---|---|
| α | **1/N(s,a)^0.7** | Rung 1's largest single effect — constant α gave 20.78 ± 6.98 coins, per-cell 49.15 ± 1.23. A constant α meets neither of L26's convergence conditions, and an unsettled cell here is an absorbing deadlock, not a small error. The **exponent** was swept only in E22: 1.0 tripled thin margins and cost the best seed 88 crates, because the target is non-stationary (mean \|TD\| *rises* through training) and sample-averaging is the wrong rule for it. |
| γ | **0.99** | E15. At 0.9 the horizon is ~10 steps, shorter than the distance to most targets. Cut the crate std 24.2 → 2.9 and removed a peak-then-decay three earlier entries had blamed on their own feature changes. |
| ε | **0.2 → 0.02**, ×0.9995 | E15/E18. A floor of 0.10 halves performance; 0.005 is indistinguishable from 0.02; **0 stops learning** — 4 of 5 tables frozen from 40 000 on. The residual 0.64 deaths/episode are the tuition that keeps rare rows alive. |
| `COIN_COLLECTED` | **+5** | E16, the rung's most surprising result: the game's own +1 **costs 45 crates**. Not a claim that coins matter more — a 16.7:1 ratio against the crate reward is what gives the table the dynamic range that stops near-ties being settled by noise. |
| `CRATE_DESTROYED` | **+0.3** | E16. 1.0 halves the crate count, and *not* by killing the agent (98 % of the deficit is in rounds nobody died) — it bombs 24 % more often for 1.09 crates a bomb instead of 2.55. Rewarding the metric harder degrades placement. |
| `KILLED_SELF` | **−5** | Rung 1; removed every suicide at once, which is why `BOMB` can stay in the action set instead of being masked out. E17 confirms it is load-bearing. |
| `INVALID_ACTION` | −1 | Rung 1. An invalid move leaves the state unchanged, which is the absorbing-row failure mode. |
| step cost | −0.1 | Shortest-path pressure. Binding, since 99 % of rounds hit the 400-step cap. |
| `WARM_N` | **100** | E23. Not optional — α is exactly 1 on a cell's first update, so an uncredited transfer is overwritten immediately. 10 000 and 100 000 are both worse: the table then cannot differentiate the rows the new digit created. |
| episodes | **20 000** | E23. Longer is worse here (40 000 loses ~7 crates), and E18 measured 200 000 turning the coarse map from 97.31 into 62.85 by re-rolling near ties. |

### Coarse-to-fine — how the table is actually built, and why it is reusable

The shipped table is **not trained in one run.** It is a two-stage curriculum across two feature
maps, and this is the most transferable thing on the rung:

| stage | map | episodes | produces |
|---|---|---|---|
| 1 | 7 digits, **12 800 rows** (`BM_ABLATE=target_dist`) | 100 000 | the coarse parent |
| 2 | 8 digits, **64 000 rows** | **20 000** | the shipped table |

**Both stages run at HEAD.** Stage 1 does not need the pre-E20 commit: pinning the trailing
digit is a bijective relabeling, so an ablated run is the coarse agent embedded at stride 5 and
`BM_SAVE_PARENT=1` writes out `q[::5]` with the parent's layout in its sidecar. Verified by
training 20 000 episodes that way at HEAD and comparing against the committed
`q_table_e16_c5_k03_s0__ep20000.npy` — **identical**, with the untouched rows holding exactly
0.0. That check doubles as independent evidence that E21/E22/E23's defaults really are inert on
the training path.

Stage 2 initialises every row from the row it was split from — `np.repeat(coarse, 5)` — and
credits those cells with `WARM_N = 100` pseudo-visits so the first update does not erase them.

**Why not one stage.** Trained from scratch, the fine map reaches **91.92** crates after 100 000
episodes (E20). Warm-started it reaches **111.40** after 20 000 — better, and five times cheaper.
The fine map has ~840 used rows against the coarse map's ~474, so the same data is spread twice
as thin; the parent hands it values for those rows instead of making it rediscover them. The two
stages do different jobs: the coarse map learns **what to do**, the fine map learns **when the
distance changes the answer**.

**The constraint that makes it work: new digits must be *appended*, never inserted.** `encode`
is mixed radix, so appending multiplies every old index by the new radix — parent row *i*
becomes children *k·i … k·i+k−1*, which is exactly `np.repeat`. Insert a digit anywhere else and
the row count is unchanged, the divisibility check still passes, and **the mapping is silently
wrong**: measured on 5 000 random states, an inserted digit mis-maps **39 %** of them, with no
error raised. Digit 8 was appended for an unrelated reason — to keep "digit 6" meaning the same
thing in older entries — and that accident is what makes the warm start valid.

Since this is a silent-failure mode rather than a crash, `train.py` now writes a
`.layout.json` sidecar beside every table and `warm_start` refuses a parent whose
`FEATURE_SIZES` is not a prefix of the current one. Tables written before this exist without a
sidecar and produce a warning instead of a refusal.

**This is expected to be needed again on rungs 3 and 4.** Opponent features mean more digits,
which means a bigger and sparser table facing the same problem, and the shipped rung-2 table is
the natural parent for it. Append the opponent digits after digit 8 and the same machinery
applies unchanged.

**Shipped table** `q_table_e23_wn100_s5__ep20000`, selected on held-out seed 550731 and confirmed
on 990731 — an arena set used to choose nothing — over **1000 rounds**:

> **116.57 crates · score 8.47 · suicides 0.000 · survived 1.000 · think_max 0.22 ms**

against `rule_based_agent` 116.43 and `coin_collector_agent` 116.26. That is **95.1 % of the
122.2 crates on the board and 8.47 of 9 coins** — rung 2 is at reference parity and saturated.

**Caveat stated plainly:** the *table* is at parity; the *configuration* is not demonstrably
better than its 12 800-row parent (+9.03 crates, CI [−6.00, +24.06], n = 5 training runs). What
ships is a table selected on held-out data, which is a legitimate procedure and a weaker claim
than "this recipe beats the reference".

---

## 4 · What moved the number, and what did not

**Four things worked.**

1. **Give the agent a way out (E11, +24).** With `BOMB` in the action set and no escape
   direction, the agent bombs and dies. Digit 6 switching to the exit while in a blast is the
   single change that made bombing survivable.
2. **Target the crate, not a tile that can hit one (E13, +57).** The original rule — head for a
   tile from which a bomb would reach a crate — was satisfied almost everywhere: on a fresh
   `classic` arena **99.7 % of free tiles are already in range of some crate**, so the digit read
   0 nearly always, there was no gradient, and mirror-image rows pointed at each other. The agent
   oscillated between two tiles in 20 rounds out of 20.
3. **γ = 0.9 → 0.99 (E15).** At γ = 0.9 the effective horizon is ~10 steps, shorter than the
   distance to most targets. Raising it cut the between-seed spread from 24.2 to 2.9 and removed
   a peak-then-decay that three earlier entries had blamed on their own feature changes.
4. **A warm start plus an early stop (E21/E23).** Initialising the fine map from a converged
   coarse one and stopping at 20 000 episodes is worth ~10 crates and, more importantly, cuts the
   spread to ± 4.8.

**Five things did not, and the negatives are the more useful half.**

- **Potential-based shaping (E19)** — rejected at both scales. Ng et al.'s invariance holds for
  the MDP the agent learns in; this agent learns in a 12 800-row *aliasing* of the game, where a
  row is a bucket of states with different potentials. The offset cancels between actions within
  a state but not within a row, so shaping injects within-row noise of size `SHAPE · sd(Φ)` —
  measured at 0.91 tiles, i.e. **the same size as the action margins it was meant to widen**. No
  scale of the knob could have worked. Generalises to a design rule: *any state-dependent shaping
  term is only safe while its within-row variation stays below the margins it targets.*
- **The counted digit 7 (E14)** — crates/bomb rose 2.43 → 2.69 (+10 %, against 2.9–3.4
  predicted), the mean did not move, and the **ceiling fell** from 116.64 to 106.63.
- **More episodes (E18, E21)** — 200 000 re-rolls near-tie rows and made the incumbent *worse*
  (97.31 → 62.85). The fine map dips and recovers by 300 000, so "train longer" is not reliably
  anything.
- **α exponent → 1.0 (E22)** — refuted before its batch ran. 1/N is the sample-average rule,
  correct for a stationary target; this target is measurably non-stationary (mean |TD| *rises*
  through training), so it drags every action in a row toward the same early-policy mean.
  Thin margins **tripled** (17.4 % → 41.5 %) and the best seed lost 88 crates.
- **The distance digit on its own (E20)** — not demonstrated at matched episodes. It only pays
  once the warm start exists to fill the rows it creates.

**One thing is cheap insurance.** `act` broke ties on exact float equality, which never fires:
the rows that absorb a collapsed policy sit at margins of 1e-4 to 1e-2, never 0. `BM_TIE_TOL`
widens that to a band. Across 30 tables it is worth **+13.84 crates, CI [+7.17, +20.50]** — but
the entire effect is rescue (collapses of 5.57 → 60.65, 11.82 → 65.71), and on healthy tables it
is **−1.06, CI [−3.05, +0.94]**. Default is 0.0; worth enabling from rung 3, where unfamiliar
states will appear.

---

## 5 · Bomb placement — measured, and deliberately not fixed here

From watching the agent play. Over 1 799 bombs:

| | crates |
|---|---|
| hit by the bomb actually dropped | 2.60 |
| best available one step away | 3.32 (**+0.72**) |
| bombs already at the local optimum | 69 % |

**31 % of bombs are placed worse than a tile one step away** — 429 of them hit a single crate
while 219 hit four or more — and digit 7 is binary, so it cannot tell those apart. This is the
same gap the E16 correction identifies as the reason a mis-priced crate reward degrades
placement rather than survival.

It is **not fixed on this rung**, on evidence: the prize is ~6 crates (5 %), E14 already tried
the counted digit and lowered the ceiling, and the marginal rate is thin (the agent converts
~0.65 crates per step; walking one extra step for +0.72 is barely above that). Folded into rung-3
feature work, where the clock is tighter and coins come out of crates. **Pre-registered failure
mode: crates/bomb rises while the best table falls below 116.6 → the same result as E14 twice,
and the digit is dead.**

---

## 6 · Method — what the audit changed

An independent session audited E19–E22 with the raw data, instructed to compute its own numbers
before reading the entries. Its instrument reproduced **80 of 80** official evaluations exactly.
It found four things worth carrying forward, all now applied:

1. **Never divide a total by cells and call it typical.** E21/E22 rested on "a busy cell has
   N ≈ 5 000 visits", which is 24.5 M updates ÷ (840 rows × 6). Real visitation is skewed ~700×:
   unweighted median 256, **visit-weighted median 173 454**. The residual-update argument built on
   it was wrong by four orders of magnitude. This is the same error as reasoning about row
   statistics instead of visitation, and it was made **three times in four entries**.
2. **Match the statistical unit to the claim.** Two *fixed tables* → pair over arenas. Two
   *configurations* → pair over **training runs**, n = 5. The per-round bootstrap gives
   `warm`@40k [+7.4, +11.4] by treating 5 runs as 1500 replicates; the run-level CI is
   [−14.8, +36.4]. Same data, opposite verdict.
3. **A holdout must hold out the factor that was selected over.** Re-measuring a checkpoint choice
   on a new *arena* seed validates nothing — arena noise is already averaged out by 300 rounds,
   and the training seed, the dominant variance component, is identical in both sets. Selection
   shrinkage dev → held-out was +0.28. For a checkpoint or hyperparameter chosen across seeds,
   the holdout is **new training seeds**.
4. **Write refutation clauses a plausible outcome can trigger.** Two could not fire: E21's
   "≥ 780 rows filled" was already satisfied by `np.repeat` at initialisation, and E22's "slower
   early" had a refutation a strictly smaller α cannot produce.

**Withdrawn or corrected claims** — all in the ledger, and all belonging in the report as
corrections rather than being quietly dropped:

- **E16, "suicides explain the crate deficit".** They do not: **98 % of the deficit is present in
  rounds where nobody died.** The larger crate reward degrades *placement* (2.55 → 1.09
  crates/bomb) while bombing 24 % more often. A guard that fired was mistaken for a mechanism
  that explains.
- **E19, "2.5 deaths' worth of discouragement per round"** — a partial sum of a telescoping
  series over one bucket. All three buckets sum to +1.94/round, as Ng predicts for γ < 1. The
  mechanism survives; the magnitude does not.
- **E21, "both arms decay"** — the five 300 000-episode tables existed on disk and were never
  evaluated. They are the best in the arm (97.59 dev / 97.45 held-out), which reverses a scored
  prediction. Neither decay was statistically established anyway (CIs [−73, +25] and [−19.9,
  +3.2]), and 75 % of one was a single seed.
- **E21's guard breach** — reported as the arm, was one seed; arm means 0.019 suicides and 0.981
  survival are inside the guards. Cause identified: `WARM_N = 100` raises α on transferred cells
  by 185× over the parent, so the warm start *un-converges* what it copies.
- **E20's `distnull` control was invalid by construction** — pinning a digit is a bijective
  relabeling (`distnull[::5] == parent`, other rows exactly zero), so it could not price sample
  dilution. It is still a perfect wiring check, and E22 reused it legitimately as "the incumbent's
  map under a different α".
- **E23 nearly repeated the selection error it was written to avoid.** The parent-paired result
  was +14.10, t = 2.87 — significant on two arena sets, and **one of four** seed-set × checkpoint
  cells. Correct pooling over five independent parents gives +9.03, t = 1.18, not demonstrated.

**What held up:** every byte-identity control (E19's no-op switch, E21's continuation check,
E20's relabeling) reproduced exactly, and reproducibility was clean throughout — same commit plus
the same `BM_*` variables gives bit-identical tables.

---

## 7 · Evidence map

| what | where |
|---|---|
| per-experiment predictions, results, verdicts | `experiments/benedict.md` E08–E23 |
| evaluation CSVs + `.meta.json` | `results/eval/task2_crates/` |
| reference agents | `results/eval/baselines/ref_*__task2.csv` |
| training logs | `results/train/task2_crates/` (gitignored — reproducible from commit + `BM_*`) |
| Q-table checkpoints | `checkpoints/benedict_task2/` (gitignored, 707 files) |
| rollout / table diagnostics | `scratchpad/benedict/` — `loop_probe`, `cycle_dump`, `table_check`, `target_follow`, `bomb_quality` |
| curves | `tools/plot_checkpoints.py` (eval curves) · `tools/trainlog.py` (training curves) |

Held-out seeds: **550731** for model selection, **990731** for the final confirmation. Neither is
the evaluation seed 20260731; the training world seed is 810731.

---

## 8 · Carried into task 3

1. **`state_to_features` has no opponent information at all** — the `TODO task 3+` in
   `callbacks.py` is untouched. That is the whole of rung 3's feature work. **Append the new
   digits after digit 8**, so the shipped rung-2 table stays a valid warm-start parent (§3);
   inserting them would silently break the mapping and the guard in `warm_start` would be the
   only thing standing between that and a wasted batch.
2. **Order agreed:** `GOT_KILLED` into the reward table first, then measure the transfer floor of
   the shipped rung-2 agent against `peaceful_agent`, then add opponent digits. `peaceful_agent`
   is a genuine floor here in a way it never was on rung 2.
3. **The counted digit 7 / bomb-density work is queued** (§5), with E14's ceiling drop as its
   pre-registered failure mode.
4. **`suicides` changes role** — a progress signal on rung 2, a regression guard from rung 3 on.
   Learning aggression is exactly when an agent forgets to run from its own bomb. The shipped
   agent is at 0.000, so any movement is a regression.
5. **`BM_TIE_TOL = 0.01`** is measured, free on healthy tables, and worth enabling where new
   states appear.
6. **Time is the binding constraint** (99 % of rounds hit 400 steps). With opponents on the board
   it gets worse, so efficiency — not crate count — is the thing to optimise.

---

## 9 · Reproduction

**The shipped table, from a clean clone.** Every default in `train.py` is the shipped
configuration, so this is the whole recipe — the suffix exists only to avoid writing over
`q_table.npy`, which `save_table` refuses to do anyway:

```bash
BM_MODEL_SUFFIX=_repro uv run python main.py play --no-gui \
    --agents benedict_task2 --train 1 --n-rounds 20000 --seed 810731

cmp checkpoints/benedict_task2/q_table_repro.npy agent_code/benedict_task2/q_table.npy
```

Verified byte-identical. The warm-start parent
`checkpoints/benedict_task2/q_table_e16_c5_k03_s0__ep100000.npy` is the **one checkpoint kept
under version control** (614 KB) — an input to the shipped model, so that a clone can retrain it
immediately rather than after a 25-minute prerequisite run.

It is a convenience, not a dependency: **stage 1 rebuilds at HEAD**, no old commit required.

```bash
BM_ABLATE=target_dist BM_SAVE_PARENT=1 BM_WARM= BM_RUN_INDEX=0 \
  BM_MODEL_SUFFIX=_parent uv run python main.py play --no-gui \
  --agents benedict_task2 --train 1 --n-rounds 100000 --seed 810731
# -> q_table_parent__coarse.npy (12 800 rows) + its .layout.json;
#    then BM_WARM=_parent__coarse for stage 2.
```

`BM_WARM=` must be empty in that command, or the parent rebuild is itself warm-started from the
parent it is meant to replace. `BM_SAVE_PARENT` refuses to run without the ablation, since
without it the stride-5 rows are every fifth row of a *finer* table, which is not a coarse table
and would warm-start into nonsense.

**The sweep the shipped seed was selected from** — five training seeds in parallel, selection on
the held-out world seed 550731, confirmation on 990731:

```bash
for i in 5 6 7 8 9; do
  BM_QUIET_LOGS=1 BM_ARM=wn100 BM_RUN_INDEX=$i BM_MODEL_SUFFIX=_e23_wn100_s$i \
    uv run python main.py play --no-gui --agents benedict_task2 --train 1 \
    --n-rounds 40000 --seed 810731 &
done; wait

for i in 5 6 7 8 9; do
  BM_MODEL_SUFFIX=_e23_wn100_s${i}__ep20000 uv run python tools/evaluate.py \
    --agents benedict_task2 --opponents none --n-rounds 300 --seed 550731 \
    --label benedict_q_e23_wn100_s${i}__ep20000__task2_val550731 \
    --out-dir results/eval/task2_crates
done
```

`BM_RUN_INDEX` seeds the exploration RNG (`TRAIN_SEED + RUN_INDEX`), so a run is bit-identical
given the same commit and the same `BM_*` variables — the shipped seed is **5**, and the sweep it
was selected against is 5–9. `BM_MODEL_SUFFIX` resolves into `checkpoints/<agent>/`, so per-run
tables can never land in the submitted folder.
