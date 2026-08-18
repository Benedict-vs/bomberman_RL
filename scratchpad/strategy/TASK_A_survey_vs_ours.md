# Task A — mining `scratchpad/survey/REPORT.md` against what we built

All numbers below are either quoted from a named file or produced by a script in this directory.
Nothing was trained. Probes: `state_visits.py`, `target_type.py`, `trap_opportunity.py`,
`round_economy.py`, `won_anatomy.py`, `survival_value.py`, `phase_split.py`, `denial.py`.

---

## 0 · Headline: the premise of question 2 is false

**"191× the state space for 78 % of the score" is a category error, and correcting it flips the
whole reading of the survey's synthesis.**

`state_visits.py`, 200 greedy rounds of the shipped table against 3 × `rule_based_agent`,
56 875 alive steps:

| quantity | value |
|---|---|
| storage rows (`FEATURE_SIZES` product) | 64 000 |
| **distinct rows visited at ε = 0** | **1 407 (2.20 %)** |
| **effective rows (exp of visit entropy)** | **289.5** |
| rows covering 90 % of steps | 297 |
| rows covering 99 % of steps | 905 |
| rows visited ≥ 10 times | 487 |
| rows with any nonzero Q after training | 8 186 |
| visited rows that are all-zero | 5 |

Source E reports **335 states**. Our agent's visit-weighted effective state count is **290**, and
90 % of its steps fall in **297 rows**. On the axis the survey's synthesis is about — how many
situations does the abstraction actually distinguish — **we are already slightly *coarser* than
E, not 191× finer.** 64 000 is dense storage over a product space that is unreachable by
construction, exactly as `callbacks.py`'s own docstring says; it was never the abstraction size.

Three consequences, and they matter more than any individual feature below:

1. **"Aggressive state-space reduction beats richness" (survey rank 4: M 72→12, C 10⁵→256,
   E 2²⁰→2160→335) does not bind on us.** We are already at the small end of that range. The
   prior was collected on projects that were at 10⁵ and came down; we are at ~10².
2. **Rows are not the binding constraint. Visits are, and we have slack.** Training touches 8 186
   rows over ~5.6 M steps ≈ 680 updates/row-average. Tripling the reachable space is affordable
   in a way "our feature map is the binding constraint" assumed it was not.
3. So the E gap (5.04 vs 3.949) **cannot be explained by state-space size in either direction.**
   It has to be *what is in* the 300-odd states. §2 argues it is one specific thing.

Caveats, stated: 1 407 distinct rows is a 200-round sample, so the reachable set is larger — but
the tail is thin (99.9 % of steps in 1 351 rows). And the visit-entropy number is not what E means
by "335 states"; E's is presumably a reachable count. The fair pair of statements is *reachable:
ours ~1.4–4 k vs their 335 (≈4–10×)* and *effective: ours 290 vs their 335 (≈1×)*. Neither is 191×.

---

## 1 · What they encode that we do not

Read `agent_code/benedict_task4/callbacks.py`. Our eight digits are: 1–4 neighbour status
(blocked / lethal / in-blast / clear), 5 moves of grace on own tile, 6 BFS first step to the
objective (escape when in danger, else coin → crate → opponent), 7 one bit "bomb here pays off and
I have one", 8 distance bucket (safe rows) or `(x+y) % 4` (danger rows).

| corpus feature | evidence | do we have it | cost to add |
|---|---|---|---|
| BFS first step to nearest coin | **[ABLATED]** M | **yes**, digit 6 | — |
| BFS first step to nearest crate | bundle, E/K | **yes**, digit 6 fallback | — |
| BFS first step to nearest opponent | bundle, E/M | **yes**, digit 6 fallback (E26) | — |
| BFS first step to nearest **safe tile** (time-aware) | bundle, E — "cut our state space 6×" | **yes**, `escape_direction`, and it short-circuits digit 6 exactly as E's does | — |
| **type of the target alongside its direction** | **[ABLATED] — best-evidenced feature in the corpus** (K: dropping 2 bits → period-2 loops / waiting) | **NO** | see §1.1 |
| bomb countdown on own tile | asserted, C | **yes**, digit 5 (0–4) | — |
| per-direction "would this move kill me" | bundle, H/M | **yes**, digits 1–4 `NB_LETHAL` | — |
| blocked-direction bits | bundle, everywhere | **yes**, digits 1–4 `NB_BLOCKED` | — |
| position parity / wall lattice | bundle, X | **yes**, digit 8 danger rows (E37) | — |
| enemy bombs at all (X's 8-way array) | asserted, X, never trained with enemies | **yes and better** — `danger_map` is over *all* bombs, so digits 1–5 already carry enemy blasts with timers | — |
| **"is BOMB safe here"** (bomb available *and* a survivable escape exists) | bundle, K/E/H; E ran a whole extra pretraining cycle for it | no | **don't — measured near-constant, §1.2** |
| "is WAIT safe" | bundle, K | implicit in digits 1–5 | low value |
| crates destroyed if I bomb here (a count, not a bit) | bundle, M | **partial** — digit 7 is a bit, not a count | 1 digit |
| nearest dead end (relative coords) | C: bug, then helped | no; lattice bit is a coarse proxy | 2 digits, expensive |
| remaining-coins scalar (M's shivering fix) | **[ABLATED]** M, needed jointly with γ 0.85→0.6 | no | our loop pathology was fixed differently (E10) — low priority |
| free tiles per direction capped at 3 | **[DIRECTION — null]** C | no | C measured null; skip |
| opponents-alive count / "am I leading" | **[DIRECTION — negative]** E | no | E measured it *worse*; skip |
| 8-fold symmetry canonicalisation | H better curve, X "not noticeably" | tried, E30 D4-shared updates: not demonstrated | consistent with corpus; skip |

### 1.1 · Target type is the real hole, and it is large

`target_type.py`, 100 rounds, 16 487 safe steps — the type our digit 6 is currently pointing at:

| target type | share of safe steps |
|---|---|
| **opponent** (no coin and no crate reachable) | **39.1 %** |
| **crate** | **36.6 %** |
| **coin** | **21.4 %** |
| nothing | 3.0 % |

Three roughly equal classes, and **the table cannot tell them apart.** A row reading "digit 6 =
RIGHT, digit 8 = distance ≥ 5" pools *there is a coin five tiles right* (walk, collect, +1) with
*there is a crate five tiles right* (walk, bomb, retreat) with *there is an opponent five tiles
right* (the board is empty, approaching is optional and dangerous). Those three demand different
actions from the same row. This is K's ablated feature and it is our largest measured aliasing.

**It is not free but it is now affordable** (§0.3). Two shapes:

- **Zero new rows** — re-partition digit 8 in the *safe* rows, the same move E37 made in the
  danger rows. Five slots must hold type × distance; the joint has 13 populated cells, so
  something is dropped. Cheapest honest packing: `0 = none, 1 = coin, 2 = crate near (d ≤ 2),
  3 = crate far, 4 = opponent`. Keeps `FEATURE_SIZES`, so the current table stays a valid
  warm-start parent. Costs the coin/opponent distance resolution that E20 tuned.
- **One new digit of size 4**, 64 000 → 256 000 rows (12 MB `.npy`, still trivial). The parent is
  *not* lost: decode every old row, re-encode under the new radix and broadcast — a ~20-line
  remap. This project has never done a size-changing digit with remapping and has been treating
  "costs a digit" as prohibitive; §0 says it is not.

### 1.2 · The corpus's most-recommended missing feature is worthless *for us* — measured

K, E and H all carry "is `BOMB` safe here". `trap_opportunity.py`, 100 rounds, 10 293 armed steps:

- an escape from our own hypothetical bomb exists on **99.08 %** of armed steps;
- "bombing here is suicide" fires on **0.92 %**.

So the bit would be a constant. It cannot be what separates us from E. And it says something
sharper: **our 0.488 suicides/round are not caused by bombing when trapped — they are caused by
not walking the escape that exists.** That is consistent with E33 (paying for the escape step) and
E34 (forcing it) both failing, and it means the suicide problem is a *policy* problem in the danger
rows, not a missing danger feature. E37 acted on exactly those rows and is the only thing that
moved.

### 1.3 · What we have that the corpus does not

Worth writing up as contribution, not just gap-filling:

- **Enemy-bomb awareness by construction.** The survey's §"The subset that touches our failure
  mode" says dying to opponents' bombs is essentially unaddressed; X built the only feature for it
  and never trained against enemies. Our `danger_map` merges all bombs with their timers, so
  digits 1–5 are enemy-bomb-aware for free. Our `killed_by_opponent` is 0.048/round (and
  `evaluate.py:258` undercounts it ~2×, so call it ~0.1) against 0.488 suicides — we die to
  ourselves ~5–10× more often than to them. **The corpus's identified hole is not our failure mode.**
- **A bomb-independent structural digit.** The survey states verbatim that "nobody has a
  bomb-independent 'this tile is a corridor with k escape squares' digit". `(x+y) % 4` is one, at
  zero rows, with a measured effect (+0.255 score paired) and a mechanism measured *inside the
  Q-tables*. That is a genuine contribution against this corpus.
- **Confidence intervals.** No source in the corpus reports one.

---

## 2 · Where the 1.09 points to E could actually come from

Ranked by how much of the gap each could plausibly carry, with my confidence:

1. **Target type (§1.1) — most likely, medium confidence.** E carries separate first-step fields
   per goal, so their abstraction knows *what* it is walking toward even after the safe-tile
   short-circuit; ours multiplexes three goals into one direction digit. This is the one place the
   corpus's best-evidenced ablation lines up with a large measured aliasing in our agent.
2. **Their pretraining curriculum — possible, low confidence, and expensive to test.** γ = 0 over
   six hand-built scenarios, then the whole cycle again with the bomb bit pinned to 0, reporting
   every state populated before main training. Our coverage guard (all-zero rows 0.00008,
   `scratchpad/benedict/e37_coverage_PLB2.out`) says coverage of the *visited* space is solved —
   but that is circular, since the greedy policy avoids rows with no values. What we have never
   measured is coverage of the *reachable* space. Against it: E38 showed 15× more training makes
   us *worse*, so sample count is not our bottleneck, which makes a sample-generating curriculum an
   odd fix.
3. **Their reward table — unknown.** Rung 3 (E27) showed our reward scale was calibrated on a solo
   board and that fixing it was necessary but not sufficient. Nobody has re-derived the scale for
   rung 4. This is cheap to check and has never been checked on this rung.
4. **A measurement difference — genuinely unknown, and I cannot close it without their repo.**
   Their 5.04 is a bare mean with no CI. Ours carries a **±0.12 noise floor from the opponents'
   unseeded RNG alone** (`benedict_task4.md` §5.9) and was selected on a validation seed and
   confirmed held-out; the same table re-evaluated gives 3.828. If their figure is a best-of-seeds
   with no held-out confirmation, some of the gap is selection. Two things I would want and do not
   have: **what `rule_based_agent` scores in *their* field** (the calibration constant — in ours it
   is 3.254 and the *best of three* averages 5.315, so 5.04 is inside the range of a good
   rule-based round), and whether SS2024's `settings.py` matches SS2026's.

**Not on the list: state-space size** (§0), and **not "is BOMB safe"** (§1.2).

---

## 3 · What they tried that failed — the do-not-do list

| don't | who, and what happened |
|---|---|
| DQN on raw board planes without scaffolding | **N**: dueling double DQN → always-`WAIT` after 30 k episodes; PER, ε = 1.0, reward rescaling, buffer shrink all failed. **K** needed 100 000 episodes to make a CNN work. Direct support for our Model B go/no-go — and N is a free citation for "feature engineering beats model complexity" |
| "more information is better" feature vectors | **M**: 72 inverse-distance features did not converge in a full day, cut to 12. **C**: a 20-tile bitmask ≈ 10⁵ states, "impossible to visit". **E**: their 29-dim vector was *worse* than the 20-dim after main training |
| continuous per-direction danger | **E**: helped during imitation pretraining, worse after main training |
| global/aggregate spatial counts | **H**: coins-per-quadrant → period-2 cycles and walking past nearby coins; only shortest-path fixed it. Same shape as our `static-counts-are-not-visitation` note |
| free tiles per direction capped at 3 | **C**: null vs the plain 4-neighbour version. *This is the closest thing in the corpus to our E37 lattice bit and it measured null* — see §4 |
| opponents-alive count / "am I leading" | **E**: part of the vector that came out worse |
| bombing on sight | **C**: the aggressive variant "stopped collecting coins"; they shipped the conservative one. **Directly relevant to Task B** |
| symmetry canonicalisation as a win | **X**: 81 → 15 states, "performance didn't improve noticeably". Matches our E30 |
| sign conventions on relative-coordinate features | **C**: negating dead-end coords after bombing silently made the real dead end look free |
| distance-to-opponent alone as an offence feature | **M**: enough to flee, not to fight; their agent scored **3.2 vs 3× rule_based — below rule_based** |

---

## 4 · Where the survey contradicts `experiments/benedict_task4.md`

**One real tension, two corrections, and one thing the survey is *consistent* with that the ledger
did not claim.**

- **Tension — C's null on structural escape room vs our E37 win.** C measured "free tiles per
  direction, capped at 3" (256 states, an escape-room proxy) as *no difference* against the plain
  4-neighbour version. E37's lattice bit is a cheaper encoding of related information and is our
  only measured win on this rung (+0.255 [+0.039, +0.475]). Not a refutation — different agent,
  different arity, C reports no CI and no numbers — but it is the one place the corpus points the
  other way, and the report should say so rather than cite E37 as if the prior work agreed.
- **Correction — the `won`/`score` conversion in §6 is a ratio of means, not a marginal.** §6 uses
  `won ≈ 0.113 × score`. Recounting 2 000 rounds (`survival_value.py`) gives the actual marginal:
  **+0.088 [+0.076, +0.101] `won` per point of score**, flat from +1 to +5. The ratio of means is
  0.101. Using 0.113 inflates every `won`-side effect estimate by ~25 %; every §6 MDE argument
  survives the correction (it gets slightly *harder*, not easier).
- **Correction — "191× the state space"** (the framing in the handoff, not in the ledger): §0.
- **Consistent, and worth adding.** The ledger says survival does not convert into points and
  leaves the mechanism open. The survey does not address it, but our own data now does — see
  `TASK_B_argument.md` §1: the round's entire economy closes by step ~200, and every intervention
  that "bought survival" bought it in the empty half.

Nothing in the survey contradicts §2 (the lattice mechanism), §3's kill-price diagnosis, or §5's
method failures.

---

## 5 · What I would take, in order

1. **Target type as a digit** (§1.1). Best-evidenced feature in the corpus, largest measured
   aliasing in our agent, and §0 removes the row-cost objection. Two shapes, one of which is free.
2. **The representator control, already half-done.** `benedict_task3.md` §5.7 gives 0.030 for the
   strongest feature-only policy our digits allow, against the learned 4.377. K's equivalent came
   within 3 % of their learned agent (6.1 vs 6.3). **Ours is the stronger result and it belongs in
   the report as a direct comparison** — it is the cleanest available answer to "is this really
   machine learning". It needs re-running on the rung-4 digits to be quotable, which is one probe.
3. **Check the BFS tie-break** (survey design 4). E's BFS checks UP/DOWN before LEFT/RIGHT and
   their agent measurably moved up/down more. Ours orders `UP, RIGHT, DOWN, LEFT`
   (`callbacks.py:71`) and `bfs_first_step` returns the first goal found in that order. An action
   histogram is a five-minute probe and a leak here is a free, citable finding either way.
4. **Cite N as a negative result** for the Model B section, whatever Model B ends up doing.
5. **Do not adopt**: "is BOMB safe" (§1.2), opponents-alive/leading, continuous danger, quadrant
   counts, symmetry canonicalisation, dead-end coordinates.
