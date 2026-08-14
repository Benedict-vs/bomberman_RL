# Teardown of `rule_based_agent` — what it conditions on that our 8 digits do not

Produced for E28 (rung-4 planning) by a subagent reading `agent_code/rule_based_agent/callbacks.py`
line by line against our feature map. The load-bearing line references were re-verified in the
source by hand before anything here was used — the claims marked **verified** below were read
back out of the file directly; the rest is the subagent's reading and is labelled as such.

---

## 1 · The headline, which inverts the premise E28 started from

**`rule_based_agent` is not anticipatory about opponent bombs either.** It never reads
`others[i][2]` (`bombs_left`), never predicts opponent movement, and builds its danger model
(`bomb_map`, lines 105-109) from `game_state['bombs']` alone — the same input class as our
digits 1-5.

It has exactly one rule that conditions on an opponent beyond "that tile is occupied"
(**verified**, `callbacks.py:173-176`):

```python
if len(others) > 0:
    if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
        action_ideas.append('BOMB')
```

Unconditional. No escape check. High in the proposal stack.

Two consequences:

1. The anticipatory gap is a gap for **both** agents. Closing it is not catching up to the
   reference, it is overtaking it.
2. Because the rule is deterministic, a feature of the form "an armed opponent is adjacent and
   its hypothetical bomb covers my tile" is a near-**certainty-level prediction of what
   `rule_based` does next step**. That is the highest-value new digit on the list (§4.1).

**The E28 hypothesis this generated:** E26's HUNT fallback makes digit 6 walk toward the nearest
opponent whenever no coin and no crate is reachable — most of the round after ~step 140 on a
four-agent board. HUNT was measured against `coin_collector_agent`, which never places a bomb.
Against `rule_based_agent` it may walk us into the rule above, deterministically, for two thirds
of every round. Tested as E28 arm F4-noHUNT.

## 2 · Proposal priority (lines 137-210)

`action_ideas` is a stack; line 203-204 pops until it finds a valid action, so **later appends
win**. Ascending priority:

| # | lines | proposal |
|---|---|---|
| 0 | 139-140 | four directions, `shuffle`d (stdlib `random`, unseeded) |
| 1 | 162-168 | BFS first step toward nearest target, else `WAIT` |
| 2 | 171-172 | `BOMB` if standing in a dead end |
| 3 | 173-176 | `BOMB` if an opponent is within Manhattan 1 |
| 4 | 178-179 | `BOMB` if BFS says "stay put" and a crate is adjacent |
| 5 | 182-196 | flee each bomb in row/col within 3 — straight run-away appended first, then both perpendiculars, so **perpendicular outranks straight** |
| 6 | 198-200 | if standing **on** a bomb, re-append the shuffled four directions → **uniformly random valid move, overriding everything** (**verified**: `action_ideas.extend(action_ideas[:4])`) |

The flee loop iterates `bombs` in placement order, so the **most recently placed bomb dictates
the escape direction**.

## 3 · Information table

`?` legend: **YES** = our digits represent it · **PART** = represented but conflated, or only in
a sub-case · **NO** = blind.

| # | information | lines | how it is used | ours | comment |
|---|---|---|---|---|---|
| 1 | own `(x,y)` | 100 | only via `coordinate_history` | NO | not worth it — translation invariance is what keeps the table at 10⁵ rows |
| 2 | own `bombs_left` | 100, 134 | gates `BOMB` validity independent of usefulness | **PART** | folded into digit 7 as `have_bomb AND useful`; `7=0` cannot separate "no bomb" from "bomb, nothing to hit" |
| 3 | scores | 100, 103 | **unused** | NO | endgame risk posture only; low value, needs a lead bucket |
| 4 | bombs → `bomb_map`, **no wall occlusion** | 105-109 | only `bomb_map[d] > 0` at 124 | **YES, ours is better** | our `danger_map` respects stone occlusion via `blast_coords` |
| 5 | per-neighbour blast *timer* | 105-109 | unused (only the `>0` test) | PART | arity 6 per neighbour → 6⁴ = 1296 vs 256, ×5.06 rows. Rejected on arity; the reference does not use it either |
| 6 | `explosion_map` | 124 | `< 1` validity | YES | — |
| 7 | others as obstacles | 125, 158-160 | invalid move; blocked in BFS only while ignoring others | **PART** | conflated with wall/crate/bomb — cannot separate "permanent wall" from "agent that moves next step" |
| 8 | **others' `bombs_left`** | never read | — | **NO (both blind)** | after bombing, `rule_based` is disarmed 5-6 steps (`environment.py:203-206`) — a safe window neither agent can see |
| 9 | **could an opponent's hypothetical bomb reach me** | never computed | — | **NO — the gap** | → D10 |
| 10 | opponent at Manhattan ≤ 1 | 173-176 | → `BOMB`, no escape check | PART | our digit 7 fires for an opponent anywhere in blast (up to 3 tiles), so "adjacent" and "3 away in line" share a row |
| 11 | `len(others)` alive | 150, 174 | gates hunting mode | NO | arity 4, cheap, low expected value |
| 12 | **dead-end tile set** | 145-146 | both a BFS *target* class and a `BOMB` trigger | PART | "I am in a dead end" is representable (3 of digits 1-4 = blocked); "the tile ahead is one", and dead ends as a destination class, are not. Note: pillars sit at (even,even), so a free tile's stone-only degree is 2 or 4, never 1 — every dead end is crate-induced, so bombing one always destroys ≥1 crate |
| 13 | **escape-route count / is my own bomb survivable** | never computed | — | **NO (both blind)** | → D9 |
| 14 | crates | 147 | BFS target | YES (6, 7, 8) | — |
| 15 | coins | 148 | BFS target, **no priority over crates or dead ends** | **YES, ours is better** | digit 6 prioritises coins strictly |
| 17 | BFS distance to objective | 29-37, 161 | nearest target only; **never compared against a bomb timer** | YES (digit 8) — but pinned to `DIST_NONE` while escaping | — |
| 18 | **loop / revisit memory** (`coordinate_history`, 20 deep) | 112-116 | 3 visits → `ignore_others_timer = 5` | **NO — our state is strictly Markov** | → D12; targets our documented period-2 oscillation |
| 19 | bomb-drop memory (`bomb_history`, 5) | 71, 134, 208 | forbids re-bombing a tile | PART | mostly already suppressed by digit 7 self-clearing; skip |
| 20 | mode `ignore_others_timer` | 74, 112-116, 150, 158 | hunt vs farm | NO | a mode digit is a policy switch in disguise — rejected, §4.6 |
| 21 | stochastic tie-breaking | 45, 69, 140 | random BFS neighbour order + random direction baseline | `policy_rng`, `TIE_TOL=0.0` | not a feature; it is why `rule_based` is only *partly* beatable deterministically |

## 4 · Where `rule_based_agent` is exploitable

Ranked by what a learnable policy can extract.

**4.1 · Its first move after dropping a bomb is uniformly random** (lines 197-200, **verified**).
Standing on its own bomb it re-appends the shuffled directions on top of everything, so the pop
at 204 is a uniform draw over valid directions — the flee logic at 182-196 is overridden at
exactly the step that matters. Where one direction escapes and another does not, it dies with
probability ≈ ½. This is consistent with the measured `suicides` 0.507 in self-play, and it means
its own bombs are a free kill source for an agent that simply survives longer.

**4.2 · It bombs on adjacency with no escape check** (173-176 + 171-172, against the complete
absence of escape-feasibility anywhere in the file). Two exploits: it bombs *predictably*
whenever we stand at Manhattan 1; and it is then disarmed for 5-6 steps and cannot threaten us
at all. Nothing in our 8 digits sees the disarmed window.

**4.3 · Its escape direction is deterministic** (186-196). Perpendiculars are appended after the
straight run-away so turning a corner outranks running; within them the second append wins —
**`RIGHT` over `LEFT`** for a bomb in its column, **`DOWN` over `UP`** for a bomb in its row.
Our freshly-placed bomb dictates its flight. Line 125 makes an agent-occupied tile invalid, so
the destination can be denied. Overridden by 4.1 whenever it stands on its own bomb.

**4.4 · Its target BFS is blind to bombs and explosions** (line 157, `free_space = arena == 0`).
Bombs are dropped from the *target* list but never from the *traversable* set. So `d` can be a
tile `valid_actions` rejects, no `if d == …` branch matches, and the whole target-seeking layer
silently evaporates — leaving the shuffled random directions as the highest valid proposal.
**Bombs near `rule_based` do not just threaten it, they degrade its navigation to a random walk.**

**4.5 · It can trap itself with two bombs** — the flee rules never check the destination against
a *different* bomb. Needs a third agent's bomb; real on a 4-player board, not schedulable by us.

**4.6 · No coin priority** (line 148: `targets = coins + dead_ends + crates`, one BFS, equal
weight). At `CRATE_DENSITY = 0.75` the nearest crate is almost always ≤ 2 tiles, so it is a
purely *local* crate farmer that will never walk 6 tiles for a coin. **Our digit 6 already holds
this edge** — a measured asymmetry for the report, not a change to make.

**4.7 · It never waits out a blast** — `WAIT` is proposed only when there is no target at all
(166-168). With 4.3, a bomb that seals a corridor *forces* a move in a nameable direction.

**4.8 · Its `bomb_map` ignores stone occlusion** (line 107, `range(-3,4)`, no wall break). Bites
once: it refuses to step onto a tile a wall actually shields. Minor, but *our* `danger_map` is
correct here and the report can claim it.

**4.9 · Loop-breaking only addresses opponents** — 112-116 alters targets and `free_space` only
w.r.t. `others`. A loop from two mirror-symmetric crate targets is untouched; only the `shuffle`
breaks it.

## 5 · Candidate digits

Baseline `FEATURE_SIZES = (4,4,4,4,5,5,2,5)` = **64 000 rows × 6**. All are **appended**, so the
frozen table stays a valid parent under `np.repeat(parent, k, axis=0)` — no re-basing. Timing is
not a constraint: measured `think_max_ms` is 0.321 ms against a 500 ms limit.

**These are candidates, not decisions.** Nothing here enters the agent without its own entry and
a paired measurement — the survey and the death forensics feed the same shortlist.

### D10 — armed-opponent threat on my tile — arity 3 → 192 000 rows

```
0  no armed opponent's hypothetical bomb covers my tile
1  some armed opponent's bomb-if-dropped-now would cover my tile
2  ... and that opponent is at Manhattan distance 1
```

`(x,y) in blast_coords(ox,oy)` per armed other; ≤ 3 × 13 tile tests, negligible.

*Feature, not policy:* it reports a **hazard field** — the same class of object as `danger_map`,
sourced from potential rather than placed bombs. It ranks no action. In state 2 the right answer
is sometimes flee, sometimes bomb first to deny the kill, sometimes step out of the line, and
which one depends on digits 1-5.

Ranked first: it is the precondition of the reference's only offensive rule, so it is also a
prediction of its next action; level 0 for a *disarmed* opponent is the whole of exploit 4.2; and
it repairs the perverse interaction HUNT introduced, where digit 7 rewards standing inside mutual
blast range with no notion that the other agent shoots too.

### D9 — is a bomb here survivable — arity 2 → 128 000 rows (arity 3 → 192 000)

Arity-3 variant `0 / exactly one first move escapes / ≥2 escape` — the 1-vs-2 split is exactly
"an opponent can body-block my only exit", a real mechanic given line 125.

`danger2 = np.minimum(danger, BOMB_TIMER on blast_coords(x,y,field))`, then
`escape_direction(...) != NO_TARGET`. The timing already matches the existing convention: the
bomb is created at timer 4, `update_bombs` decrements after the agent acts, so it detonates at
the end of step *t+4* and the agent moves at *t+1…t+4* — what `escape_direction`'s
`danger[nx,ny] <= depth` test encodes. One extra bounded BFS, ~+50 % on `state_to_features`.

*Feature, not policy — the nearest to the line, so it must be argued explicitly in the report:*
it answers "does a safe tile exist from here under a counterfactual", a reachability property of
the state, in the same family as digit 5 and digit 6's escape branch, and the task hints
recommend exactly this ("in blast radius / can I escape"). The honest risk is the joint row
`(digit7=1, D9=1)`, where "bomb" is right often enough that a grader could read the pair as an
oracle. Mitigation: keep D9 at arity 2, leave digit 7 alone, and **report the measured action
distribution in that row** — if it is not ~100 % `BOMB`, the pair demonstrably is not a policy.

Ranked second: the reference is blind here too, so this is an overtake rather than a catch-up,
and it attacks `suicides`, our regression guard.

**Suggested bundle: D10 + D9 = 384 000 rows** (2.3 M cells, 18 MB float64), then ablate each with
the existing `BM_ABLATE` machinery per bundle-then-ablate. Sparsity makes this less alarming than
it looks: 2 364 rows carry value today, ~475 visited per rollout, and `np.repeat` starts every
new row at its parent's values — so evaluating the frozen table under the new digits is a
well-defined experiment on day one, the manoeuvre that won E26.

### D12 — where I came from — arity 5 (×5), or arity 2 (×2)

`0 = none / 1-4 = direction back to the previous tile`; arity-2 variant "digit 6 points back
where I came from". The analogue of `coordinate_history`, aimed at the period-2 oscillation.
Two flags: it needs per-round state on `self`, so `state_to_features` stops being a pure function
of `game_state` (plumbing in both files plus a round reset like lines 95-97); and it makes the
MDP formally non-Markov in the environment's state — fine for a POMDP-style feature, but it has
to be argued. Arity 5 on top of the bundle is 1.92 M rows: too much. Arity 2 or defer.

### D13 — bomb available — arity 2 (×2)

Separates "no bomb" from "bomb, nothing worth hitting". Cost ~0. The clean fix — re-basing digit
7 to arity 3 at *no* extra rows — **breaks `np.repeat` parentage**, so it is out under the
append-only constraint. As an appended digit, `(digit7=1, D13=0)` is unreachable, so 25 % of new
rows are dead. Low priority now that HUNT drained the invalid-`BOMB` leak (24.21 → 1.43).

### D15 — opponent distance, always on — arity 4 (×4)

Opponent distance currently reaches digit 8 only through digit 6's third fallback, i.e. only once
coins *and* crates are exhausted. An always-on `none / ≤2 / 3-5 / 6+` would separate "farm" from
"fight" posture without a mode variable. Real but second-order; ×4 on the bundle is prohibitive.
Parked.

### Rejected

- **Per-neighbour blast timers** — ×5.06 rows for information the reference does not use either.
- **A hunt/farm mode digit** — a switch over *behaviours*, not an observation; the closest thing
  here to a forbidden policy feature. D15 is the legitimate version.
- **"Can I win the bomb race against this opponent"** — a minimax evaluation returning an action
  preference. Forbidden. D10 is the observation it would be built from.
- **Dead ends as a digit-6 target class** — copies the reference's objective function, and it is
  *worse* than ours: coins lose to crates in that scheme (4.6 is an edge we already hold).
- **Absolute position** — destroys the translation invariance that keeps the table small.

## 6 · One free observation for the report

Digit 8 is pinned to `DIST_NONE` whenever digit 5 ≠ 0 (the escape branch), so a fifth of its row
budget is structurally dead. Reclaiming it requires re-basing, which the append-only constraint
excludes — but it is worth a sentence as a known inefficiency of the mixed-radix layout, and it
is the natural home for an escape-route count if a re-base ever happens.
