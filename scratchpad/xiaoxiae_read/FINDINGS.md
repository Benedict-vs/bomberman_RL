# xiaoxiae `BombermanML` — targeted read

Source: `scratchpad/external/xiaoxiae__BombermanML/` (GPL-3.0, one squashed commit `50b682f`).
All citations are `file:line` into **that** tree unless prefixed with `repo:`.
No code was copied into our repo. Nothing outside `scratchpad/xiaoxiae_read/` was written.

**Repo provenance caveat, up front:** the clone has exactly **one commit** (`50b682f "Wow"`), so
there is **no commit-message history** and no per-version rationale from git. The version story in
§4 is reconstructed from directory-to-directory diffs plus their report (`paper/main.tex`, 1185
lines, a full MLE final-project report from SS2023 — the single highest-value document in the repo).

---

## 0. The measured deficit, decomposed (our own CSVs, n = 1000, same slot, seed 990731)

| | ours (`benedict_task4`) | `binary_v6` | `bindist_v2` |
|---|---|---|---|
| score | 3.718 | **5.572** | **5.336** |
| coins | 2.633 | 3.372 | 3.216 |
| kills | 0.217 | 0.440 | 0.424 |
| suicides | 0.721 | 0.377 | 0.276 |
| killed_by_opponent | 0.061 | 0.077 | 0.083 |
| survived | 0.218 | 0.546 | 0.641 |
| **crates** | **32.16** | 25.09 | 25.10 |
| bombs | 28.15 | 27.31 | 25.44 |
| invalid | 5.67 | 2.81 | 2.88 |
| think_mean_ms | 0.046 | 1.654 | 1.821 |

(computed from `repo:results/eval/task4_tournament/{benedict_task4_shipped,ext_xiaoxiae_binary_v6_calib,ext_xiaoxiae_bindist_v2_calib}__task4_rb_ship990731.csv`)

Two things fall out of this that shape everything below:

1. **The gap splits ~60/40 kills/coins.** `score = coins + 5·kills` exactly. Their kill lead is
   +0.223 → **+1.115 score**; their coin lead is +0.739 → **+0.739 score**. Sum +1.854 = the gap.
2. **They convert crates into coins nearly twice as well as we do.** Coins per crate destroyed:
   ours **0.082**, theirs **0.134** (+64 %). We blow up **28 % more crates and collect 22 % fewer
   coins**. That is not a survival story dressed up — it is a *harvesting* story, and it is the
   single most actionable number in this document. It also survives the ledger's seven
   replications that "survival does not convert into points" (E30/E31/E33/E34/E36/E44/E46),
   because it is not about survival: it is about which objective the agent walks toward.

---

## 1. The two agents in one table

| | `binary_agent_v6` | `binary_distance_agent_v2` |
|---|---|---|
| model | MLP, 21 → 1024 → 1024 → 6, ReLU (`train.py:137-152`, `LAYER_SIZES` `:73`) | identical |
| params | 1 078 278 (verified by loading `target-model.pt`) | identical |
| input | **21 binary values** | same 21 slots, but the three objective blocks carry `1/sqrt(d+1)` instead of 1 (`train.py:516`) |
| layout | 4 blocks × 5 + 1 bit — see §2 | identical |
| discrete state count | ≤ **17⁴ × 2 = 167 042** reachable vectors (each block is empty, wait-only, or a non-empty subset of 4 directions), far fewer after masking | continuous-valued, but the same 167 042 *support patterns* |
| algorithm | DQN, one-step TD, online (one gradient step per env step), replay buffer + soft-updated target net (`train.py:155-188`, `:656-661`) | identical |
| hyperparams (final) | `BATCH=256 MEM=1000 GAMMA=0.99 EPS=0/0 TAU=1e-4 LR=1e-4 Adam` (`:64-73`) | `TAU=1e-3 LR=1e-6` (`:70-71`) |
| ε schedule | decays over **steps within a round**, not rounds: `EPS_END+(EPS_START-EPS_END)·exp(-step/EPS_DECAY)`, `EPS_DECAY=10` (`callbacks.py:26`). Final versions ship `EPS_START=EPS_END=0`, i.e. pure greedy fine-tuning | identical |
| training budget | not recorded numerically. Procedure is `train.py` (repo root) `--task complete --infinite 2`, i.e. **the last two stages repeated indefinitely with manual Elo-based checkpoint selection** (`paper/main.tex:763-772`). Their q-agent section quotes "more than 600 hours of training" (`paper:921`) but that is the *other* agent | same procedure, less of it (`paper:400`, `paper:862`) |
| policy at test | greedy argmax (`callbacks.py:33-38`) | identical |
| measured think time (our harness) | 1.65 ms mean, 8.0 ms max | 1.82 / 8.6 ms |

**Both are the same code family.** `binary_agent_v5` and `v6` are **byte-identical in code**
(`diff -r` reports only the two `.pt` files differ); so are `binary_distance_agent_v1` and `v2`.
The version numbers past v4 are **training checkpoints, not code variants**.

---

## 2. Their state, digit by digit, against our eight

Their layout, documented in their own source (`train.py:491-497`, and echoed in
`paper/main.tex:363-371`) and in the debug banner they print in manual mode (`callbacks.py:20-21`):

```
 0..4   direction to nearest COIN        u r d l wait
 5..9   direction to nearest CRATE       u r d l wait
10..14  direction to a tile from which a bomb ENDANGERS A PLAYER   u r d l "place now"
15..19  direction to SAFETY (all zero unless in danger)            u r d l wait
20      can I place a bomb AND survive it
```

### 2.1 Mechanism first — three things that make this *not* the same as our digit 6

**(a) Every objective is found by a BFS over *simulated future game states*, not over the board.**
`_directions_to_thing` (`train.py:345-417`) expands a queue of whole `Game` dicts; each expansion
calls `_next_game_state(state, action)` (`:229-310`) which moves the agent, decrements the
explosion map, ticks and detonates bombs (clearing crates to `0` at `:287`), and **returns `None`
if the mover would be standing in fire when the step resolves** (`:295-297`). Dead branches are
simply not expanded (`:411-412`). Consequences their paper states explicitly (`paper:391-392`):

- a tile that is currently on fire but clear by the time you arrive **is** traversable;
- a crate that will be destroyed before you arrive **is** traversable;
- a tile that will be lethal on arrival is **not**.

This applies to *all four* objectives. Our `target_direction` (`repo:agent_code/benedict_task4/callbacks.py:211-255`)
is a **static-board BFS that ignores bombs and fire entirely**; only `escape_direction`
(`:258-301`) is time-aware, and it only runs when `own_danger != 0`. So in every safe row our
digit 6 can point through a blast that is about to land, and treats a doomed crate as a wall.

**(b) The direction feature is multi-hot over *all tied shortest paths*.** When several first
steps reach the goal at the same distance, they backtrack the whole predecessor *set*
(`train.py:383-396`) and set every corresponding bit. Our digit 6 returns a single direction whose
tie-break is the `DELTAS` enumeration order in `bfs_first_step` — a fixed arbitrary bias baked
into the feature.

**(c) The "crate" and "enemy" goals are *firing positions*, not the object.**
`_is_near_crate` (`:329-334`) accepts a tile with a crate in one of the four **adjacent** cells;
`_is_near_enemy` (`:337-342`) accepts a tile whose blast coords contain an opponent. So block
5..9 answers "which way to a tile I could bomb a crate from" and block 10..14 "which way to a
tile I could bomb a player from" — and index 14 means "**you are standing on such a tile now,
drop it**" (`paper:389`). Ours targets the crate tile and the opponent tile themselves.

> Note against our own ledger: `callbacks.py:213-227` records that E10 rejected "head for a tile a
> bomb could hit a crate from" because 99.7 % of free tiles satisfy it. That is the **blast-range**
> (radius 3) predicate. Theirs is **adjacency** (radius 1), which is a materially sparser and
> different feature. E10 does not close it.

### 2.2 Side-by-side

| ours | theirs | verdict |
|---|---|---|
| **1-4** neighbour status: blocked / lethal-this-step / in-a-blast / clear (4 values each, 256 combos) | **nothing equivalent** | **we have this, they do not.** Their report lists it as a wanted future feature: *"the agent might benefit from a perfect vision of its neighbourhood, for example to identify a dangerous dead end"* (`paper:1165`). It is implicit in their search, never in the vector |
| **5** moves of grace on own tile, 0 = safe (5 values) | reduced to **one bit**: the safety block is all-zero iff `_is_in_danger` is false (`train.py:448-449`). No timer value anywhere | we carry strictly more here |
| **6** BFS first step to "the objective" — escape if in danger, else coin, else crate, else opponent. **One direction, no type** | **four separate blocks**, each with its own direction, all present simultaneously; the *type* is the block index | **the biggest single representational difference.** Our row cannot tell a 5-point coin from a 1-point crate; theirs is three orthogonal channels, priced 50 : 1 in the reward table |
| **7** one bit: a bomb here hits a crate *or* an opponent, and I have a bomb | **split in two**: block 10..14 index 14 = "a bomb here endangers a *player*" (masked by escapability, `:538`), and bit 20 = "I have a bomb and can escape it" (`:534-535`). Crate-bombing is index 9 of the crate block | theirs separates kill-opportunity from crate-opportunity; ours merges them into one bit |
| **8** distance bucket to digit 6's target (safe rows) / `(x+y)%4` lattice class (danger rows) | **no distance in v6**. `bindist_v2` puts `1/sqrt(d+1)` into the *magnitude* of the objective bits (`:516`) — one shared value per block, so it modulates importance but does not discriminate direction. **No lattice/parity feature at all** | we have a real distance digit; they only have a soft magnitude, and their own report calls the distance experiment a wash (`paper:1006`) |
| — | **safety block 15..19**, produced by a *separate* time-aware BFS with an "I am doomed, run furthest from the bomb" fallback (`:442-486`) | ours is folded into digit 6, so we cannot express "here is the escape **and** here is the coin" at the same time |
| — | **cross-block masking** (`:529-532`, `:538`) — see §3 | no analogue |

**Net:** their vector is not richer than ours in raw information — we carry neighbour tiles and a
fuse timer they do not. It is **better factored**: four independent, typed channels instead of one
overloaded direction, each computed by a search that already knows about time.

---

## 3. How they handle dying to their own bomb

**There is no action mask and no hard veto anywhere in the shipped code.** I looked for one
specifically. `act` (`callbacks.py:33-38`) is a bare `argmax` over the network output; nothing
filters, clamps or overrides it, in training or at test. The whole mechanism is (a) feature
masking, (b) two −500 rewards. Quoted mechanisms:

**(a) Objective directions are AND-ed with the safety directions** — `train.py:521-532`:

```python
if v := _directions_to_safety(game_state, include_unsafe=True):
    for i in v:
        feature_vector[i + 15] = 1
    # if we can get to safety by something other than waiting, don't wait
    if v != [4]:
        feature_vector[19] = 0
    # if we need to run away, mask other features to do that too
    for i in range(3):
        for j in range(5):
            feature_vector[j + 5 * i] &= feature_vector[j + 15]
```

So while in danger, the network is **never told that a coin lies in an unsafe direction**. It can
still pick that action — the mask is on the *state*, not the policy — but the input that would
motivate it has been zeroed. Their paper: *"it's essentially never useful for the agent to move in
such a way that will result in their death"* (`paper:394`).

**(b) Kill-opportunity is gated on escapability** — `train.py:534-538`:

```python
if game_state["self"][2] and _can_escape_after_placement(game_state):
    feature_vector[20] = 1
# feature 14 is 'place a bomb to kill player' so that needs to be masked with 20
feature_vector[14] &= feature_vector[20]
```

`_can_escape_after_placement` (`:313-321`) appends a hypothetical bomb at the agent's own tile and
asks whether `_directions_to_safety` returns anything.

**(c) The reward does the vetoing.** `DID_NOT_MOVE_TOWARD_SAFETY: -500` and
`DID_NOT_PLACE_USEFUL_BOMB: -500` are jointly the largest magnitudes in the table (`:53`, `:61`),
five to ten times the positive terms. And `DID_NOT_PLACE_USEFUL_BOMB` fires on **every** `BOMB`
press where the bomb is useless **or bit 20 is 0** (`:627-635`):

```python
if self_action == "BOMB" and old_game_state['self'][2]:
    if _is_bomb_useful(old_game_state) and state_list[20] == 1:
        ...  PLACED_SUPER_USEFUL_BOMB / PLACED_USEFUL_BOMB
    else:
        events.append(DID_NOT_PLACE_USEFUL_BOMB)
```

So "a bomb I cannot escape" is priced identically to "a bomb that hits nothing": −500. And every
subsequent step where the agent does not move along a set safety bit costs another −500.

**(d) A graceful-death fallback.** `_directions_to_safety(include_unsafe=True)` (`:442-486`): if
**no** safe state is reachable, it returns the first action of the branch that maximises Manhattan
distance to the nearest bomb (`:455-473`, `:483-484`). The vector is therefore never all-zero when
in danger — the agent always has advice, even when doomed.

**(e) A bug they document.** `:526-527` clears the safety block's WAIT bit whenever a *move*
reaches safety. Their own post-mortem (`paper:1150-1151`): this makes the agent flee when it had
time to drop a bomb first, *"making the agent more passive than it should be"*.

**Honest caveat for our purposes:** their measured suicide rate is 0.377/round against 3×
`rule_based` — **halved** relative to ours (0.721), not eliminated. And per our E46, survival on
this board does not convert into score. So (a)–(d) are almost certainly *not* where their +1.85
comes from; §0 says it comes from coins-per-crate and kills. I would not model this section as the
answer to the question that motivated the read.

---

## 4. What actually changed across their versions

Reconstructed by diffing directories, since git carries no history.

| step | change | evidence |
|---|---|---|
| `binary_agent_v1 → v2` | **nothing functional.** Adds `player_to_closest_bomb_distance` and a block of commented-out "furthest from bomb" code | diff is entirely comments + one dead function |
| `v2 → v3` | **nothing.** Removes two `TODO` comments | diff is 4 lines |
| **`v3 → v4`** | **the only substantive code change in the whole series.** (i) `_directions_to_coins` (single-goal, single predecessor chain) is generalised to `_directions_to_thing(state, thing)` with pluggable stop conditions and **tie-set backtracking**, giving multi-hot directions and the `_is_near_crate` / `_is_near_enemy` firing-position goals. (ii) Reward re-balance: `MOVED_TOWARD_CRATE 20 → 1`, `MOVED_TOWARD_PLAYER 10 → 25`, new `DID_NOT_MOVE_TOWARD_PLAYER: -10` | `agent_code/binary_agent_v{3,4}/train.py:39-62` and `:318-420` |
| `v4 → v5` | **hyperparameters only**: `EPS_START/END 0.10/0.05 → 0/0`, `TAU 1e-3 → 1e-4` | diff is 4 lines |
| `v5 → v6` | **no code change at all** — only the two `.pt` files differ. v6 is v5 trained longer | `diff -r` |
| `bindist_v1 → v2` | **no code change at all**, weights only | `diff -r` |
| `binary_agent` (unversioned) | is their working directory, currently holding the *distance* code with `TAU=1e-3, LR=1e-5` — **newer than v6, not older** | diff vs v6 |

**So the "15 variants" are not 15 experiments.** They are ~3 code states (pre-v4, v4, distance)
plus a hyperparameter annealing schedule plus checkpoints. Their stated method
(`paper:763-772`) is: run the curriculum, pick the best Elo checkpoint by hand, lower
`EPS_START/EPS_END/TAU/LR`, continue. That is manual annealing with model selection on the
evaluation metric — with no held-out seed, which by our standards means their published Elo
ranking is selected-on.

Their curriculum (`train.py:13-20` at repo root, narrated at `paper:441-461`):
`coin-heaven 100` → **`rule_based` on `empty` 1000** (combat with no crates) → `classic` solo 1000
→ `classic` vs `rule_based` 200 → **`classic` vs a frozen earlier version of itself** 200 →
**self-play** 200, then repeat the last two forever.

**Reward-design findings they report as negatives** (`paper:1008-1061`), worth having:
- hand-tuned small auxiliary rewards ("naïve"): *"performance seemed particularly volatile ... a
  sisyphean task ... we were unable to train a good agent in this way."*
- **potential-based shaping (Ng et al.): failed outright** — *"we could not produce an agent that
  performed better than random when facing crates"*, because a potential that rewards a
  well-placed bomb produces an action-independent penalty when it detonates. This independently
  reproduces our E19.
- the winning "shoehorn" table is the ±50/−100/−500 one; it learns coins *slower* at first and
  overtakes later (`paper:1092`, fig. `coin_collection_*`).
- **No terminal game rewards at all**: no `COIN_COLLECTED`, `KILLED_OPPONENT`, `CRATE_DESTROYED`,
  `GOT_KILLED`, `KILLED_SELF`. Their stated reason (`paper:851-853`): one-step TD cannot bridge the
  4-step fuse, so they restricted themselves to events with immediate causality. They name n-step
  TD as the fix they did not have time for (`paper:1159`).

**Two documentation discrepancies I would not trust the paper on:**
- `paper:398` and `:994-1004` describe the distance variant as `1/d`. The shipped code is
  `1/np.sqrt(d + 1)` (`binary_distance_agent_v2/train.py:516`).
- `paper:784` annotates `EPS_DECAY` as steps, `paper:794-796` then argues for "per-round" decay in
  the same breath. The code decays on `game_state["step"]` (`callbacks.py:26`), i.e. within a round.

---

## 5. Ranked portable ideas

Ranked by (value against **our measured** deficit) × (cheapness on a 64 000-row table). I checked
each against `experiments/benedict.md` so nothing already closed is re-proposed; the three levers
`NEXT_STEPS.md` §3 still lists open are 3.1 target type, 3.2 rung-4 reward recalibration, 3.5 the
truncation bug (per E46's corrected "what this closes" table, `experiments/benedict.md:187-193`).

**1. Re-price coins against crates, and drop the death penalty — lever 3.2. [high confidence, ~zero code]**
Their table pays coin-following **50 / −100** and crate-following **1 / 0** — an effective 100:1
coin:crate weight — and pays **nothing for dying** and **nothing for killing**. Ours pays
`COIN=5, CRATE=1.0` (5:1), `GOT_KILLED=−5`, `KILL=0`
(`repo:agent_code/benedict_task4/train.py:186-215`). §0 says we destroy 28 % more crates for 22 %
fewer coins, i.e. we are *over-paid for crates*; and our own ledger has seven replications that
survival does not convert to score, yet we still pay −5 to die. Every knob is already an
environment variable (`BM_COIN`, `BM_CRATE`, `BM_GOT_KILLED`, `BM_KILL`) — this is a sweep, not a
change. **Reason: it is the one place where their design and our own accumulated negative results
point the same direction, and it costs nothing to test.**

**2. Put the target *type* into the state — lever 3.1. [high confidence, zero extra rows if done as a re-partition]**
Their three objective channels are separate; our digit 6 gives a direction with no type, so one
row mixes "walk to a 5-point coin" with "walk to a 1-point crate" with "walk to an opponent", and
the table must price a single action for all three. Digit 8 in the **safe** rows currently holds a
4-way distance bucket — re-partition it as e.g. `{none, coin-near, coin-far, other-near, other-far}`
and the row count does not change at all. **Reason: it is the biggest structural difference
between the two representations, it is the corpus's best-evidenced feature per
`TASK_A_survey_vs_ours.md` §5.1, and it directly targets the coins-per-crate gap in §0.**
*(Caveat: this trades away distance resolution; a 3-value type digit as a real 8th→9th digit costs
3× the table instead, warm-startable from the current one.)*

**3. Make the objective BFS time-aware. [medium-high confidence, self-contained code change, zero rows]**
Port the *idea* of `_next_game_state`: in `target_direction`, refuse to expand a tile that will be
lethal when the agent arrives (depth-indexed against our existing `danger_map`), and *allow* a
tile that is currently burning but clear on arrival. We already have exactly this logic in
`escape_direction` (`callbacks.py:282-299`) — it just never runs in safe rows. **Reason: cheapest
correctness fix in the list; our digit 6 currently routes coin paths through blasts and treats
about-to-die crates as walls.** Expected direct survival gain is small (our `killed_by_opponent`
is only 0.061), but the *direction* it returns changes, which is the coin-efficiency channel.

**4. Crate goal = a tile adjacent to a crate; opponent goal = a tile whose blast reaches them. [medium confidence, zero rows]**
`_is_near_crate` / `_is_near_enemy` (`train.py:329-342`). Ours targets the crate/opponent tile
itself, so the agent walks to the object and only then discovers it must stop. **Reason: E10
rejected the radius-3 version of this, not the radius-1 version; the distinction is real and the
change is ten lines.** Test alongside #3, they share the BFS.

**5. Break digit-6 ties toward a `NB_CLEAR` neighbour. [medium confidence, ~10 lines, zero rows]**
Their tie-set is multi-hot; we cannot afford 16 values per direction digit (3.2× rows), but we can
stop letting `DELTAS` order decide. When several first steps tie at the shortest distance, prefer
one whose digit 1-4 status is `NB_CLEAR`. **Reason: a free, principled tie-break that removes a
fixed arbitrary bias from the feature; cheap enough to bundle with #3.**

**6. Self-play against a frozen earlier checkpoint. [medium confidence, expensive]**
Their curriculum's last two stages are "vs a previous version of yourself" and "vs yourself",
repeated indefinitely (`train.py:13-20`). E44 closed the *provided-opponent* training distribution;
self-play against our own frozen table is adjacent but not the same experiment. **Reason: the only
training-side idea of theirs that our ledger has not already tested — but it is a multi-hour sweep,
so it should queue behind #1-#5.**

**7. Escapability as a *state bit* rather than a veto. [LOW confidence — I would not run this]**
Their bit 20 is the non-gating form of what our E46 vetoed. Two reasons to demote it: (a) E46
measured the underlying predicate ("an escape exists at all") as **99.08 % constant** on our board,
so as a digit it would be near-degenerate; (b) E46 showed the mechanism does exactly what it says
(−0.131 suicides, +0.050 survival) and still **loses 0.283 score**, because tight bombs are the
productive ones. **Reason to list it anyway: it is the most-cited "safety" mechanism in their code,
and someone will propose it — this is the note that says it is already priced.**

**Explicitly NOT recommended: their per-step "did you follow the advice" reward (±50/−100 per
objective per step).** E33 tested precisely this for the escape channel (`FOLLOWED_ESCAPE` /
`IGNORED_ESCAPE`, already wired at `repo:agent_code/benedict_task4/train.py:159-161, 213-214`,
default off) and it failed; E34 tested forcing it. It is also uncomfortably close to our own hard
rule against a feature that returns the best action — theirs moves the violation from the feature
into the reward, and I would not want to defend that in the report.

---

## 6. What is not portable, and why

- **The network itself.** 1 078 278 parameters generalising across a 21-dimensional input is the
  whole reason their factored representation is affordable. A tabular agent gets no generalisation
  between rows; every combination must be visited. This is the fundamental asymmetry and it caps
  how much of §2 transfers.
- **Multi-hot direction blocks.** Four blocks × (17 patterns) is 167 042 support patterns — the
  same *order* as our 64 000, so it is not that their state is "continuous and huge". But as a
  mixed-radix table it would be 17⁴×2 rows with our digits 1-5 *on top*, which is not affordable.
  Only the single-direction projection (#5 above) is portable.
- **`1/sqrt(d+1)` feature magnitudes** (`bindist_v2`). A tabular row index cannot carry a
  continuous value; and their own report says the distance variant did not significantly change
  performance (`paper:1006`), and our harness measures it 0.236 *below* v6.
- **Replay buffer, target network, soft updates, per-step gradient descent, `SmoothL1Loss`,
  gradient clipping.** All DQN stability machinery with no tabular analogue; our α is a
  visit-count schedule.
- **Their evaluation methodology (Elo, `elo.py`).** Strictly worse than what we already have:
  Elo over 100-round duels, model selection on the same metric, no held-out seed, no paired CI, no
  fragility check. Their headline "837:9:154 against the default agents" (`paper:1139`) is a
  selected-on-evaluation number.
- **`train.py --infinite` with manual checkpoint picking.** Same objection — the selection is on
  the reported metric.
- **Their optimisation write-up** (`paper:935-959`: `copy.copy` over `deepcopy` = 10×, LRU-caching
  `get_blast_coords` and `state_to_features` = further 2×) is real and well-measured, but it exists
  to make a per-step forward-simulating search affordable. At 0.046 ms/step we do not have that
  problem. If we adopt #3/#4 our think time rises toward their 1.7 ms — still ~30× under the
  tournament limit even after de-rating for the reference Ryzen 2600.

---

## 7. Things I could not determine

- **Total training budget for `binary_v6`.** The repo records no episode count for the binary
  agents. `paper:921`'s "more than 600 hours" refers to the *q-agent*. The `plot.txt` in
  `binary_distance_agent_v2/` holds only the last 101 rounds of a single stage.
- **Whether their v4 reward re-balance or their v4 feature change caused the improvement.** The two
  ship in the same diff and no per-change measurement exists.
- **Any per-version performance numbers.** `paper` fig. `elo.pdf` and `stacked.pdf` are compiled
  PDFs of aggregate results; `elo/stats.json` contains only the four base agents' calibration
  ratings plus a long match log, not a per-version table.

---

## 8. Fidelity check on what we measured

Verified before trusting §0:

- `repo:agent_code/ext_xiaoxiae_binary_v6/_agent_defs.py` is **byte-identical** to
  `binary_agent_v6/train.py` (`diff` is empty) — the install renamed the file so the framework
  cannot load training callbacks, and changed nothing else.
- `target-model.pt` md5 matches the upstream file (`9dad4724…`).
- The installed `callbacks.py` differs from upstream only by dropping the `self.train` branch;
  `act` is the same bare `argmax` over the **target** network — which is what upstream also loads
  when not training (`binary_agent_v6/callbacks.py:11-13`).

So the 5.572 in §0 is their shipped agent, unmodified, and the "no action mask" finding in §3
holds for the binary we measured, not only for the source we read.
