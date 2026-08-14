# Prior-work survey — hand-built features in published `bomberman_rl` projects

Prose-only survey (written reports, READMEs, blogs, dev logs) for E28 rung-4 planning.
**No source code was read or transcribed** — per instruction, features are described at the level
of "what information does this digit carry", and anything we adopt is implemented from scratch
and cited in our report.

Swept the upstream fork network (66 forks) plus ~90 name-matched repos. **Only ~8 contain real
written prose** — most are code dumps with the framework's default README. The corpus is far
thinner than the repo count suggests, but the few good sources are very good.

## Sources

| ID | Project | Course/year | Method | Prose |
|---|---|---|---|---|
| **K** | KunkelAlexander — [tabular blog](https://kunkelalexander.github.io/blog/computers-learning-bomberman-tabular-q/), [deep blog](https://kunkelalexander.github.io/blog/computers-learning-bomberman-deep-q/) | — | tabular Q (18-bit) + CNN DQN | **Best.** Only real feature ablation + a control agent |
| **E** | [lukevoss "Echo"](https://github.com/lukevoss/Bomberman_RL_2024) (Voß, Tiedl, Müller), report PDF in repo | **MLE SS2024** | tabular Q + PPO, same feature vector | **Closest match to our task.** 47 pp |
| **M** | [nickstr15 "Maverick"](https://github.com/nickstr15/bomberman), `MaverickReport.pdf` | FML 2021 | 1-hidden-layer NN (60) on 23 hand features | Excellent on feature *evolution* |
| **C** | [chbridges](https://github.com/chbridges/bomberman_agent), `report/report_final.pdf` | FML 2019 | Q-learning + LightGBM | Good on failed representations |
| **X** | [maxawake](https://github.com/maxawake/Bomberman-Reinforcement-Learning), `FML-Project-Report.pdf` | FML | tabular Q + DQN | Only source with an **enemy-bomb** feature |
| **H** | [hericks/FML](https://github.com/hericks/FML/tree/main/report/chapters) | FML | linear approx + n-step Sarsa | Symmetry + a measured feature failure |
| **N** | [nilskre/bomberman_rl_report](https://github.com/nilskre/bomberman_rl_report) | FML | dueling double DQN on raw board | **Documented total failure** |
| **A** | [add-IV](https://github.com/add-IV/bomberman_rl/tree/master/report), [Fjallripa log](https://github.com/Fjallripa/bomberman/blob/main/documentation/progress.md), [georggrab](https://github.com/georggrab/bomberman_rl), [Li-Jesse-Jiaze](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman) (3rd place SS24) | various | mixed | fragments |

External paper the good ones cite: [Kowalczyk et al. 2022, *Developing a Successful Bomberman
Agent*](https://arxiv.org/abs/2203.09608) (Beam Search, #1 of 2300 on CodinGame) — origin of E's
"shortest **survivable** path" idea.

---

## Evidence grades

**[ABLATED]** controlled comparison with a stated outcome · **[MEASURED-BUNDLE]** the agent
containing it was measured at ε≈0, feature not isolated · **[DIRECTION]** comparison run, no
number · **[ASSERTED]** claim only.

### Navigation

| Feature | Arity | Claimed effect | Evidence | Src |
|---|---|---|---|---|
| **First step of shortest path to nearest coin** (BFS, not L1/L2) | one-hot 4 dirs + "no path" → 5–6 | The load-bearing feature everywhere | **[ABLATED]** M tried L2 then L1: agent walks into the wall the coin sits behind, then waits. BFS fixed it | M, E, K, H |
| **Type of the nearest target** (none/enemy/crate/coin) *alongside* its direction | 2 bits | Lets the agent price what it is walking toward | **[ABLATED] — best-evidenced feature in the corpus.** K dropped these 2 bits (18→16): agent "caught in infinite up-down loops or just wait" | K |
| First step toward nearest crate | 5–6 (5th = BOMB if adjacent) | Crate-phase navigation | **[MEASURED-BUNDLE]** (E: 5.04) | E, K |
| First step toward nearest opponent | 5–6 (5th = BOMB if in blast) | Enables hunting | **[MEASURED-BUNDLE]**; M: distance-to-opponent alone suffices to *flee*, not to *fight* | E, M |
| Axis of larger residual distance (tiebreak) | +1–2 | Disambiguates (2,1) from (1,2), kills period-2 loops | **[ASSERTED]** | X |
| **Remaining coins / max remaining reward** as a scalar | 1 | Fixes the "shivering" pathology | **[ABLATED, indirect]** M: with 4 inverse-distance features the agent got 5–8/9 coins then froze; encoding the argmax slot as *remaining count* made "all endless loops vanish". Root cause: features cannot express remaining return, so at high γ the best linear fit is to do nothing. **Both** this and γ 0.85→0.6 were needed; neither alone | M |
| **Coins per board quadrant** (4 counts) | 4 small ints | Global density guidance | **[ABLATED — FAILED]** H: with few coins left the agent cannot pick a quadrant → period-2 cycles; also walks past nearby coins. Only shortest-path-to-coin fixed it | H |
| **Inverse distance to all N coins** | 36–72 | "more information is better" | **[ABLATED — FAILED]** M's 72-feature variant did not converge in a full day; cut to 12. C: a 20-tile bitmask ≈ 10⁵ states "impossible to visit in 1000 rounds × 400 steps" | M, C |

### Danger / survival (own bomb)

| Feature | Arity | Claimed effect | Evidence | Src |
|---|---|---|---|---|
| **Neighbour code including an explicit `DANGER` value** — "stepping here makes survival impossible given current bombs and timers" | 3 bits × 4 dirs | One digit that already encodes lookahead | **[MEASURED-BUNDLE]** in K's 18-bit agent (6.3) | K |
| **"Is `BOMB` safe here"** — bomb available **and** a survivable escape exists | 1 bit | Makes `BOMB` learnable at all | **[MEASURED-BUNDLE]**; it is half of E's reachable state space — they ran an entire extra pretraining cycle with it pinned to 0 to populate the other half | K, E, H |
| "Is `WAIT` safe" | 1 bit | Separates forced waiting from idling | **[MEASURED-BUNDLE]** (K); E prices it in rewards instead (+0.1 / −2) | K, E |
| Per-direction "would this move kill me" | 4–5 bits | Direct escape signal | **[MEASURED-BUNDLE]** | H, M |
| **First step toward the nearest *safe tile*** (BFS over survivable paths only) | 5–6, 5th = "surrounded, WAIT" | Turns escape into navigation | **[MEASURED-BUNDLE]** — E return this vector *first* and short-circuit their other three goals, cutting the state space 6× | E |
| **Bomb countdown on my own tile** | 0–4 | Without it the agent cannot time its escape | **[ASSERTED]**, mechanism airtight: C state "the agent is not able to flee… because it does not know when it will explode" | C, M |
| Explosion on an adjacent tile next turn | 4 bits | Stops running *into* death while not itself endangered | **[ASSERTED]** | M |
| **Per-direction continuous danger 0→1** | 5 continuous | "granular risk understanding" | **[DIRECTION — negative]** E: the 29-dim vector containing these "did not perform significantly better… after main training it showed to be **worse**". Helped only during imitation pretraining | E |
| Per-tile danger field, normalised | 17×17 plane | Danger as a continuous field for a CNN | **[ABLATED — FAILED]** N: the whole raw-board DQN converged to *always WAIT until killed* after 30k episodes | N |
| **Nearest dead end** (relative coords) | 2 | Best bomb spots *and* the classic death trap | **[ABLATED — bug, then FAILED]** C negated its coords after bombing to repel the agent; this silently made the real dead end look free. After the fix, dead-end deaths fell | C, M |
| **Free tiles per direction, capped at 3** (escape-room proxy) | 4⁴ = 256 | "more free tiles = more escape options" | **[DIRECTION — null]** C: "we could not see any difference" vs the plain 4-neighbour version; kept the cheaper one | C |

### Opponents — the thin part

| Feature | Arity | Claimed effect | Evidence | Src |
|---|---|---|---|---|
| **8-way direction to *all* bombs incl. diagonals**, distant ones pruned by L1 | 8 bits | **The only feature in the corpus built for enemy bombs.** Motivated by a worked "definite death" figure: escaping your own bomb diagonally walks you into an opponent's | **[ASSERTED]** — and their final agent trained *without* enemies anyway | X |
| Opponents merged into the crate channel; enemy bombs into the own-bomb channel | reuses digits | "trivially extends a crate agent to a fighting agent" | **[ASSERTED]** | X |
| Direction to nearest opponent + "opponent adjacent" flag | 5–6 + 1 | Enables offence | **[MEASURED-BUNDLE]**. C compared "adjacent" vs "same row/col within 5": the aggressive variant bombed on sight and **stopped collecting coins**; they shipped the conservative one | C, E, M |
| Opponents-alive count + "am I leading" | 1 + 1 | Adapt to the competitive situation | **[DIRECTION — negative]**, part of E's worse 29-dim vector | E |
| Opponent-position plane + "opponent can drop a bomb" flag | 2 CNN channels | Closest anyone gets to anticipating an unplaced bomb | **[MEASURED-BUNDLE]** in K's CNN (5.4, fewest suicides) | K |

### Board / structural

| Feature | Arity | Claimed effect | Evidence | Src |
|---|---|---|---|---|
| **Position parity** (x,y even/odd → 3 cases) | 3 | Cheap encoding of the wall lattice | **[MEASURED-BUNDLE]** in X's coin agent | X |
| 4 blocked-direction bits | 16 | Kills invalid actions | **[MEASURED-BUNDLE]** everywhere | C, X |
| **Crates destroyed if I bomb here** (−1 if none/suicide, +5 if opponent adjacent) | small int | Bomb-spot quality | **[MEASURED-BUNDLE]** | M |
| (inverse distance) × (crates destroyed) as one product | 4 scalars | Cheap surrogate after the full version failed | **[ABLATED]** M: full version failed after 10k games; the product "was a good equivalent" | M |
| **8-fold symmetry canonicalisation** | ÷8 states | Faster convergence | **[DIRECTION]** H: better curve, no number. X reduced 81→15 states — **"performance didn't improve noticeably"** | H, X |
| **Breadcrumbs** — own last-3 positions decaying to 0 | 1 CNN plane | Kills the stuck-in-loop failure | **[ASSERTED]** | georggrab |

---

## Ranked by evidence

1. **Target *type* alongside target direction** (K) — the only clean drop-two-bits ablation, and its failure mode (period-2 oscillation) is one we already track.
2. **BFS/real-path distance over L1/L2** (M) — ablated, unambiguous, diagnosable in one replay.
3. **A scalar encoding remaining return + matched γ** (M) — ablated jointly; *neither fix worked alone*.
4. **Aggressive state-space reduction beats richness** (M 72→12; C 10⁵→256; E 2²⁰→2160→335) — three independent ablations, all the same direction.
5. **Coins-per-quadrant is a trap** (H) — measured failure, same shape as our own `static-counts-are-not-visitation` note.
6. **Bigger feature vectors are not better** (E, 29 vs 20 dims) — direction only, but same algorithm, same training, and it came out *negative*.
7. **Bomb timer must be in the state** (C) — asserted, mechanism airtight.
8. Everything about opponents' bombs — **[ASSERTED]** at best.

---

## The subset that touches our failure mode

**Dying to opponents' bombs is essentially unaddressed in the prior work.** It is the clearest
hole in the corpus and it is exactly our hard setting.

- **Nobody predicts a bomb that has not been placed.** No source has "tile an opponent could bomb next step", "am I in an opponent's line of fire", or "opponent is armed and within blast distance". K's CNN *opponent-can-drop-bomb* flag is the closest, and it is not ablated.
- **K names the gap and leaves it open**, verbatim: their survivability check *"does **not** protect against being intentionally trapped by another agent or bombs being set off by other bombs"*.
- **X is the only one who builds for it** — the 8-way all-bombs array — but never trained it against enemies.
- **add-IV explicitly gave up**: their MCTS models opponents as always choosing `WAIT`, on CPU-time grounds.
- **Escape awareness always conditions on bombs already placed** (H's reachable-safe-tile bits, E's "surrounded → WAIT", K's survivable-continuation bit, C's measured-null free-tile counts). Nobody has a bomb-independent "this tile is a corridor with k escape squares" digit. K *asserts* their conv net discovers corridors and dead ends — a claim, not a measurement.
- **Only E splits the deaths** the way our `AGENTS.md` mandates for task 4 (a who-killed-whom confusion matrix over 250 rounds).

**Implication:** a digit answering *"could an opponent's bomb, placed right now, kill me here, and
do I have an exit that survives it"* is genuinely novel against this corpus and costs 1–3 bits in
tabular arity. That is a contribution to write up, not a re-implementation.

---

## Calibration: what typically happened

**Beating `rule_based_agent` is achievable, and tabular Q on good features is what repeatedly
does it.**

| Project | Method | Result |
|---|---|---|
| **E (MLE SS24)** | **tabular Q, 335 states** | **5.04 mean score, 1000 rounds vs 3× `rule_based_agent`** |
| E | PPO on the same 20-dim vector | 5.21; head-to-head vs their own Q agent 4.3 vs 4.2 over 100 rounds — almost certainly inside noise, no CI |
| K | tabular Q, 18 bits | 6.3 vs rule_based 4.5 (mixed field, 1000 games) |
| K | CNN DQN, 11 channels | 5.4 vs tabular 3.6 vs rule_based 2.4 — needed **100 000 episodes** |
| M | small NN, 23 features | **3.2** vs 3× rule_based over 3000 games — *below* rule_based; learned to flee, never to fight |
| N | dueling double DQN, raw board | **Total failure**: always-`WAIT` after 30k episodes; PER, ε=1.0, reward rescaling, buffer shrink all failed |
| H | linear + Sarsa(λ) | never got a reliable crate agent; never trained against opponents |
| Fjallripa | tabular Q | never past task 1; blockers were off-by-one bugs between action/state/reward, not features |
| add-IV | Q-table + DQN | both "stuck due to lack of long-distance vision"; only MCTS worked |

1. **Deep methods never beat `rule_based_agent` without scaffolding** — imitation pretraining, MCTS on top, or six-figure episode counts. Every unscaffolded DQN-on-raw-board here failed. Direct historical support for our Model B go/no-go.
2. **≈5 points/round against three rule-based agents is the bar.** E set it explicitly as "enough to have a chance of winning" and hit it with a **335-state** table.
3. **The features may be doing all the work.** K's decisive control: a **"representator"** — a rule-based agent that can see *only* their 18-bit feature vector — scored **6.1** against the learned agent's **6.3**. Q-learning added ~3 %. Their verdict: *"you reap what you sow."*

**Two caveats on every number above:** (a) **no source in this corpus reports a confidence
interval** — every figure is a bare mean, so our paired-CI rule already puts us ahead of the
published prior work methodologically; (b) field compositions differ (K's "classic" is 1×
rule_based + 1× peaceful + 1× representator, not 3× rule_based), so absolute scores are **not**
comparable across rows — only E's and M's are measured against three rule-based opponents.

---

## Five designs worth taking (as experiment designs, not code)

1. **The representator control (K).** Hand-write a rule-based agent restricted to our feature vector and `--compare` it. Converts "the model learns from the features" from assertion to measurement. **We already have a version of this** — `experiments/benedict_task3.md` §5.7: the strongest purely-feature policy our digits allow scores 0.030 against the learned 4.377 and dies every round. Ours is a *stronger* result than K's, whose control came within 3 %. Worth stating in the report against their number.
2. **The drop-two-bits ablation (K).** Remove target *type*, watch for period-2 oscillation. Cheap, decisive, failure mode already in our metric set.
3. **E's pretraining curriculum instead of raising ε.** Six hand-built scenarios (coins only / crates only / coins+opponents / crates+opponents / crates+coins / thinned classic), 200 rounds each (1000 for the last), α=0.9 and **γ=0 so pretraining learns immediate rewards only** — then the whole cycle again with the bomb-availability bit pinned to 0. They report every state populated and every action tried before main training (α=0.1, γ=0.8, 10k rounds). A far better answer to "rare states are never visited" than ε.
4. **Check our pathfinder's tie-break.** E's BFS checks UP/DOWN before LEFT/RIGHT; their agent measurably moved up/down more and covered **0.24 unique tiles per survived step vs rule_based's 0.28** (n=1000). A tie-break ordering leaks straight into the policy — worth an action-histogram check on ours.
5. **A free negative result to cite (N).** They deliberately refused path-to-coin and escape features on the grounds that these edge toward a rule-based agent, used raw board planes instead, and got always-`WAIT` after 30k episodes. This course's own strongest published evidence for "feature engineering beats model complexity", and a counterweight if we are ever accused of over-engineering features.

---

*Extracted report text used for this survey lived in the session scratchpad and is ephemeral;
every claim above is traceable to a URL in the source table. No repository was cloned and no
source code was read.*
