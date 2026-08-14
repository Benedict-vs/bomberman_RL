# Forensic analysis of `benedict_task4` deaths vs 3 × `rule_based_agent`

**n = 300 rounds, `classic`, seed 20260731 → 282 deaths over 41 945 alive steps.**
Agent is `agent_code/benedict_task4/`, byte-identical to the rung-3 ship, `BM_HUNT` at default.
Scripts in this directory.

**Route:** not replays. `BombeRLeWorld` driven directly the way `tools/evaluate.py` does (same
per-round reseeding of `world.rng` *and* `np.random`), with a subclass recording which explosion
covered our tile — `evaluate_explosions` discards the killer, and a replay `.pt` stores only
arena/coins/actions/permutations, so replays would have needed re-simulation anyway.

## 0 · Mechanical vs interpreted

`sim.py` re-implements one `do_step`; `validate.py` replays all 282 deaths through it and
**reproduces the death step and the killing bomb's owner in 282/282**. Feature digits are
recomputed by importing the agent's own `callbacks.py`; **the action actually played was in the
greedy set of the recomputed row in 280/280 cases.** So the counterfactuals below are mechanical,
not narrative.

Two stated assumptions: `survivable()` is a space-time BFS against bombs *already on the board*
with opponents frozen — so "survivable" means "a way out was already visible" and "not
survivable" is a hard verdict; the `cf_k` counterfactual replays the opponents' recorded actions,
so **k = 1 is exact** and k > 1 is optimistic.

Sanity against the committed reference (`benedict_q_e26_F_frozen__task3_rb_ship990731.csv`,
1000 rounds): suicides **0.443** vs its 0.409 [0.379, 0.439]; killed_by **0.497** vs 0.519
[0.488, 0.550]; score 2.850 vs 2.757 [2.616, 2.897]. Inside the CIs.

## 1 · Taxonomy

| cause | n | share | steps forfeited | avoidable at last step (k=1) |
|---|---|---|---|---|
| opponent's bomb, escape existed when it landed | 146 | 51.8 % | 274.8 | 80 % |
| own bomb, escape botched after a **survivable** drop | 128 | 45.4 % | 279.5 | 41 % |
| own bomb, the drop itself was already fatal | 5 | 1.8 % | — | — |
| opponent's bomb, already trapped when it landed | 3 | 1.1 % | 250.3 | 67 % |

Two numbers reframe the problem:

- **93.3 % of deaths had a way out when the killing bomb became visible**, with 3 steps of warning
  in 245/282 and 4 in the rest. **The agent is essentially never ambushed** — it watches a bomb
  tick for three steps and then dies to it.
- **Only 5 of 133 own-bomb deaths came from an unsurvivable drop.** The agent does not bomb itself
  into corners; it drops a safe bomb and fails to walk out.

Smallest surviving divergence: k=1 **61.3 %**, k=2 25.5 %, k=3 4.3 %, k=4 5.0 %, k=5 3.2 %,
none 0.7 %.

**Cost.** corr(score, alive_steps) = 0.42, slope 0.0103 score/step; alive 139.8 of 400 steps.
Survived rounds (n=18) score 4.61, died rounds 2.74. Opponent kills gift **2.49 points/round** to
the field — 87 % of our own mean score of 2.85, and the direct cause of `won` = 0.17 against their
~0.32 each. Own-bomb deaths gift nothing but forfeit 124 steps/round.

## 2 · The decisive result — at `t*`, the last savable step

| at `t*` | n | share |
|---|---|---|
| **Q-row is degenerate — all six actions tie, `act()` draws uniformly** | 207 | **73.4 %** |
| row is trained and **every** greedy action is fatal here | 73 | 25.9 % |
| not avoidable within 5 steps | 2 | 0.7 % |

Mean **P(survive | shipped policy at `t*`) = 0.318**.

Degenerate rows are all-zero — the table has 2 364 non-zero rows of 64 000. They are **2.6 % of
all alive steps but 73.4 % of `t*` steps — 28× enrichment.** 129 such rows hold the 207 deaths,
were entered 371 times in total, and **55.8 % of every visit to one is the last savable step of a
death.**

So "can the 8 digits distinguish this from a safe state?" splits in two, and the halves need
**opposite** fixes:

- **73 % — yes, perfectly.** The state gets its own recognisable row, fatal on over half its
  visits. What is missing is a **table entry, not information.** Those rows exist only with
  opponents present, and the shipped table was trained solo on rung 2 — this is the price of
  `benedict_task3.md` §3, showing up as deaths.
- **26 % — no.** The state maps onto a well-visited, mostly-safe row whose learned answer kills
  here.

Only **4.6 %** of savable deaths needed a move into a tile digits 1–4 called `NB_BLOCKED` (an
opponent vacating mid-step via the random activation permutation — luck the agent cannot see).
**95.4 % had a survivor it could see.**

## 3 · The five cases worth reading

**A — round 0, step 184, the modal death.** Own bomb at (1,7) timer 0, agent at (3,7). Digits:
`UP=clear RIGHT=LETHAL DOWN=in-blast LEFT=LETHAL, d5=1, d6=UP` → row 54310. The only surviving
action is `UP`; **digit 6 says `UP`.** Row 54310 is all-zero, so the greedy set is all six actions
and the tie-break rolled `LEFT`. P(survive) = 1/6. Set-up: it bombed from a tile with one free
neighbour, walked *along* its own blast corridor for two steps, and at d−1 played `BOMB` with no
bomb left (a wasted `INVALID_ACTION`).

**B — round 3, step 65, digit 6 against digit 7.** Opponent bomb at (6,11) timer 0, agent adjacent
at (7,11). `UP` and `DOWN` both survive; d6 = `UP`; **d7 `bomb_useful` = 1** because an opponent is
in blast range — true, and fatal, because bombing costs the one move left. Row 55065 all-zero →
tie-break picked `BOMB`. **The map creates this trap:** digit 7 is computed unconditionally, so
the row says "run" and "bomb" simultaneously and nothing encodes that bombing forfeits the escape
move.

**C — round 19, step 83, the one the digits genuinely cannot see.** Agent at (11,8) in a N–S
corridor, own bomb at (11,10), `rule_based_agent_2` two tiles up at (11,7). Row 34110 =
`(2,0,2,0,2,1,0,0)`; the table's greedy is `UP`, the survivor is `DOWN`. `UP` runs at the opponent,
which is still there next step. **This is the exact comparison the brief asked for:** row 34110 was
visited 482 times, 14 ending in death within 4 steps, and the fatal and safe visits have
**identical digits** — the row *is* the state. What differs is outside the map: **opponent BFS
distance = 2 in 71 % of the fatal visits against 4 % of the safe ones.** Same pattern in row 8620
`(0,2,0,2,2,2,0,0)`: 365 visits, 9 deaths, distance 2 in 63 % fatal vs 7 % safe.

**D — round 42, step 73, bodies read as walls.** Agent at (7,8), own bomb below, opponent above.
Row 100 = `(0,0,0,0,2,0,0,0)` — "every neighbour blocked, two moves of grace, no escape"; trained,
answers `WAIT`, and `WAIT` dies. Three of the four "blocked" neighbours are **bodies, not walls** —
`NB_BLOCKED` conflates wall/crate/bomb/agent and only the agent moves. Row 100 alone holds 9 of the
73 trained-but-fatal deaths. (This case is in the 4.6 % where the survivor depended on permutation
luck.)

**E — the own-bomb aggregate.** 128 of 133 follow case A's shape. Digit 6 is `NO_TARGET` at the
death step in 79/133 — but that is **not** the escape BFS being over-conservative: of the 110
deaths with `d6 = NO_TARGET` while `d5 > 0`, only **6** were still survivable at that point. When
digit 6 gives up, the agent really is dead; the error was 1–3 steps earlier.

## 4 · Ranked missing state information

Base rate of "dead within 4 steps" = **0.0336** over 41 945 steps.

| candidate | P(dead ≤ 4) by value | fires on | within-row lift\* |
|---|---|---|---|
| **C3 opponent BFS distance** (1 / 2 / 3–4 / ≥5) | .138 / .149 / .073 / **.010** | 26 % | **42.9 %** |
| **C8 neighbours blocked by a body** (0 / 1 / 2) | .028 / .134 / **.480** | 5.0 % | — |
| **C5 armed opponent's bomb would cover me** | .028 / **.107** | 7.2 % | 14.8 % |
| **C6 …and it would leave no way out** | .032 / **.827** | 0.23 % | — |
| C7 distance to nearest never-blasted tile | .017/.046/.071/.065 | 41 % | 3.5 % |
| C4 free neighbours (1/2/3/4) | **.020**/.033/.031/.043 | — | 7.9 % |
| C1 correct time-aware "any escape at all" | **.935** / .029 | 0.5 % | 0 % |
| C2 "if I bomb now, do I still get out" | .033 / .035 | 36 % | 3.5 % |
| C9 my bomb is on the board | .039 / .031 | 63 % | — |

\* share of fatal steps in ≥ 200-visit rows that the candidate moves into a ≥ 3×-fatal cell — i.e.
signal the eight digits do *not* already carry.

1. **Opponent BFS distance, bucketed.** Fixes the trained-but-fatal rows — **73 deaths, 25.9 %,
   ≈ 0.24/round, ≈ 1.2 points/round gifted.** The only candidate with both volume and lift; at
   `t*` an opponent is within 2 tiles in **75 %** of trained-fatal deaths against a **10.5 %** base
   rate. Cost: reuse the BFS digit 6 already runs; ×4 rows.
2. **Split `NB_BLOCKED` into static vs opponent body.** Fixes the boxed-in family (row 100 = 9
   deaths); one body-blocked neighbour raises the death rate 2.8 % → 13.4 %, two → 48.0 %. Present
   in 23 % of trained-fatal `t*` states vs 5 % base. Costs no new digit — digits 1–4 gain a fifth
   value, 4⁴ → 5⁴. **Note this is a re-base, not an append.**
3. **"An armed opponent already covers my tile" (C5).** Fires one step before the killing bomb in
   **76/149 (51 %)** of opponent-bomb deaths, quadrupling the death rate. Its sharpened form C6 is
   the highest-precision signal in the whole run — 82.7 % fatal on 98 steps, 10 false alarms in
   41 847 safe steps — but recall ≈ 5 %.
4. **Three explicit negative results.**
   - **"Is it safe to bomb here" (C2) buys nothing** — .033 vs .035. The obvious life-saving
     feature is already unnecessary, because only 5 of 133 own-bomb deaths came from an
     unsurvivable drop.
   - **Dead ends are *safer*, not riskier** (C4 = 1: .020 against a .034 base). The intuition is
     backwards on this board.
   - **A correct time-aware escape test (C1) has no warning value** — it fires on 200 steps, 102 of
     which are the death step itself.

**And the largest single item is not a feature at all.** 73.4 % of deaths reach `t*` in an all-zero
row where the policy is a uniform draw over six actions. No new digit helps there — every extra
digit multiplies the row count and makes coverage *worse*. That fix is of a different kind:
symmetry canonicalisation (≈ 8× on required coverage), warm-starting those rows, or a fallback for
a row with no learned values — the last weighed against the `AGENTS.md` rule that the model must
*learn from* the features.

## 5 · Caveats

One run, one seed; binomial SE ≈ 3 % at n = 282, so the 73/26 split is robust and the 1.1 % bucket
is not. Rung-3+ evaluations are not round-reproducible (`rule_based_agent` shuffles with the
unseeded stdlib RNG) — re-runs gave 277/280/282 deaths with the same structure, not the same
rounds. `survivable()` never re-opens a route a blast clears and freezes opponents (conservative);
`cf_k` for k > 1 lets opponents ignore our divergence (optimistic). "Warning = 3 steps" counts from
first visibility — a bomb dropped at step *t* first appears at *t+1* with timer 3.

**C3/C5/C6/C8 are measured as *predictors*, not as trained features.** Separation is necessary, not
sufficient — and §2 says row coverage is already the binding constraint. So the next experiment
must add C3 (or fold digits 1–4 to five values) **together with a coverage measure**, never alone.
