# Audit 4 — diagnosis of the E30 arm-T peak-then-decay

Commissioned with the raw data and **no** statement of my own hypothesis, precisely so the
diagnosis would be independent. The agent states it read `AGENTS.md`, `MEASUREMENT.md`,
`agent_code/benedict_task4/{callbacks,train}.py`, `environment.py` and `agents.py` first and did
**not** open `experiments/benedict.md` until the whole diagnosis was complete and every number
measured. Its conclusion is not the one that was in the ledger.

**The agent's scripts were not written to disk** (it reported paths it never created); this
report is its returned analysis, preserved verbatim in substance. What *does* survive on disk is
its probe evidence: `train/` here (12 training logs, moved out of `results/train/task4_tournament/`
where they were mislabelled `q_e30_a4*`), and the probe tables
`checkpoints/benedict_task4/q_table_a4{ctl,tie,step0}_s7{0,1}__ep{5000,10000}.npy` (gitignored).
Six duplicate final-state tables were deleted after verifying they were byte-identical to their
`__ep10000` checkpoints.

**Independently verified before the ledger was rewritten** (see E31): the visit-weighted gap
collapse, both named rows, and the direction of the causal probe.

---

## 1 · Headline

The decay is **convergence, not divergence** — but to a degenerate fixed point. The driver is the
**−0.1 per-step cost**, which is action-independent and therefore contributes nothing to the
quantity that decides the policy, while acting as a constant downward ratchet on whichever action
is currently greedy.

1. **Decision margins collapse.** Visit-weighted top1−top2 gap in rows the greedy policy visits
   falls **1.81 → 1.34 → 0.67** over 5 k → 10 k → 20 k; the share of steps decided below 10⁻³
   rises **0.001 → 0.057 → 0.206**. All five seeds.
2. **A collapsed gap plus a deterministic argmax is a 2-cycle.** `act()` breaks ties by exact
   float equality, so where LEFT and RIGHT differ by 1.6 × 10⁻⁴ the argmax is float noise, the
   mirrored row picks the mirrored action, and the agent paces. Steps inside a ≥ 6-step period-2
   cycle: **0.000 (parent) → 0.001 (5 k) → 0.088 (10 k) → 0.462 (20 k)**, median run 50 steps,
   max 297 of 400.
3. **Pacing is safe and unproductive** — exactly the reported signature: `survived` ↑,
   crates/coins/kills/score ↓, longer episodes, undiscounted training reward → 0.
4. **Causal test:** remove the step cost and the collapse does not happen (§7).

Not "the reward's optimum is passive and the learner is correctly finding it": the greedy policy's
own **discounted training return** falls too, 14.04 → 11.86 → 9.71. A policy worth 14.0 exists, so
the later tables are worse under their own criterion.

## 2 · Reward decomposition (measured, reconstruction matches logged reward to ±0.02)

| episodes | reward | coins ×5 | crates ×1 | death ×(−5) | wait | invalid | step cost | steps | survived |
|---|---|---|---|---|---|---|---|---|---|
| 0–1 k   |  2.56 |  3.52 | 11.67 | −5.00 | −0.45 | −3.09 |  −4.10 |  41 | 0.000 |
| 4–5 k   |  9.59 |  9.98 | 25.76 | −4.60 | −1.31 | −4.34 | −15.90 | 159 | 0.081 |
| 9–10 k  |  7.22 |  9.83 | 25.39 | −4.31 | −1.84 | −3.93 | −17.90 | 179 | 0.139 |
| 19–20 k | −0.05 |  7.91 | 20.67 | −4.12 | −1.48 | −3.98 | −19.04 | 190 | 0.176 |

Peak → end: crates **−5.1**, coins **−2.1**, step cost −3.1, **death penalty +0.5**. The fall is
*earnings*, not a longer-episode artefact.

**Wiring check.** `reward − reconstruction` is 0.000 / −0.008 / −0.017 = exactly
`−0.1 × survived_rate`. That is the fingerprint of truncation-as-termination in `end_of_round`:
on a round reaching `MAX_STEPS`, the agent is alive, the last transition gets **two** updates on
the same cell — the correct bootstrapped one, then `td_target = reward` with V = 0. Size:
0.176 × 20 000 ≈ 3 500 such updates against ~3.8 M step updates on rows whose α = 1/N^0.7 is
~10⁻³·⁵ → supportable steady-state bias **≲ 0.2**. Real, worth fixing, an order of magnitude too
small to be the phenomenon and the wrong sign to explain rising survival.

## 3 · Does the trained table earn more of its own reward? (ε = 0, paired arenas, seed 550731)

| table | score | **disc. return** | undisc. | crates | steps alive | survived | BOMB share |
|---|---|---|---|---|---|---|---|
| `rung2ship` (warm start) | 2.47 | **13.36** | 15.24 | 25.9 | 127 | 0.04 | 0.114 |
| T s60 @5 000  | 3.99 | **14.39** |  7.14 | 32.5 | 253 | 0.35 | 0.128 |
| T s60 @10 000 | 3.34 | **11.81** |  4.50 | 29.6 | 273 | 0.43 | 0.110 |
| T s60 @20 000 | 2.69 | **9.73**  | −0.12 | 25.3 | 297 | 0.58 | 0.066 |
| T s63 @5 000  | 4.51 | **14.54** | 10.67 | 32.2 | 250 | 0.28 | — |
| T s63 @20 000 | 1.96 | **6.19**  | −8.62 | 21.8 | 298 | 0.54 | — |

`V(s₀)`: 4.68 → 6.04 → 5.86 → 5.41. Both the realised objective and the table's own estimate fall.

## 4 · What breaks

### 4a · The value function is a constant

T s60 @20 000, 100 greedy rounds. Only **10.3 %** of steps carry any reward beyond the step cost.

| phase | E[Q] | sd[Q] | E[G] | sd[G] | bias |
|---|---|---|---|---|---|
| 1–50 | 5.63 | 0.85 | +5.53 | 7.08 | +0.10 |
| 100–150 | 5.39 | 0.85 | −4.63 | 4.19 | +10.02 |
| 200–250 | 5.20 | 0.51 | −7.99 | 2.35 | **+13.19** |
| 300–400 | 5.14 | 0.46 | −4.41 | 2.33 | +9.55 |
| all | 5.31 | 0.71 | −3.53 | 6.11 | +8.84 · **corr(Q,G) = 0.251** |

The estimate is **5.2 ± 0.5 whatever is happening** while the truth swings 13 points. With
γ = 0.99 the horizon is a third of an episode, and nothing in the eight digits records crate
stock, coins left or step number — so V *cannot* express phase, and the same cells average the
crate-rich opening with the barren endgame. **The enabling condition:** there is very little true
action gap to defend.

### 4b · Gap collapse, all five seeds (fixed row set, pooled visit weights)

| checkpoint | E[gap₁₂] | P(gap<10⁻³) | P(gap<0.05) | sd[V] |
|---|---|---|---|---|
| `rung2ship` | 1.216 | 0.025 | 0.029 | 1.167 |
| @5 000 | 1.670–1.743 | 0.001–0.002 | 0.002–0.026 | 0.964–0.989 |
| @10 000 | 1.299–1.354 | 0.001–0.039 | 0.033–0.087 | 0.885–0.915 |
| @20 000 | 0.894–0.946 | 0.102–0.180 | 0.352–0.412 | 0.841–0.861 |

**Graded by credit distance:**

| decision type (share of steps) | E[gap] parent → 5 k → 20 k | P(gap<10⁻³) @20 k |
|---|---|---|
| escape a blast, grace>0 (36.4 %) | 1.89 → 2.44 → **1.66** | 0.001 |
| safe, navigate, target ≤2 (10.3 %) | 1.12 → 1.64 → 0.73 | 0.000 |
| safe, a bomb here pays (18.0 %) | 0.74 → 1.31 → 0.51 | 0.098 |
| safe, navigate, target ≥3 (33.5 %) | 0.75 → 1.08 → **0.25** | **0.281** |

Decisions with an immediate consequence keep their margin. Distant payoffs lose it.
`P(BOMB greedy | bomb pays)` falls 0.638 → 0.386.

### 4c · Named rows

**Row 12786** — corridor, safe, crate directly below at distance 1, bomb in hand. Only `BOMB` is
correct.

| table | UP | RIGHT | DOWN | LEFT | WAIT | **BOMB** | gap |
|---|---|---|---|---|---|---|---|
| `rung2ship` | 2.975 | 3.639 | 2.928 | 3.596 | 3.637 | **4.769** | 1.1e+00 |
| T s60 @5 000 | 3.874 | 4.880 | 3.839 | 4.730 | 4.563 | **5.832** | 9.5e−01 |
| T s60 @10 000 | 4.115 | 5.461 | 4.152 | 5.309 | 4.872 | **5.812** | 3.5e−01 |
| T s60 @20 000 | 4.189 | 5.20418 | 4.161 | 5.20356 | 5.065 | **5.20483** | **6.5e−04** |
| T s63 @20 000 | 4.227 | 5.276951 | 4.187 | **5.277000** | 5.106 | 5.276979 | **2.1e−05** |

At 20 000 on seed 63 the greedy action is **LEFT** — it walks away from an adjacent crate with a
bomb in hand, by 2 × 10⁻⁵.

**Row 12774** — corridor, BFS target 5+ tiles RIGHT, no bomb payoff. T s60 @20 000:
RIGHT = 4.9785740, **LEFT = 4.9787376** — LEFT wins by 1.6 × 10⁻⁴, i.e. the table prefers walking
away from its own target. This one row absorbs **5 680** of ~33 000 rolled-out steps.

Not stale warm-start cells: **0.04 %** of visit-weighted greedy actions are never-updated cells.

### 4d · Cycles (120 greedy rounds)

| table | steps in ≥6-step 2-cycle | rounds with one | cycle-row mean gap | P(gap<0.05 ∣ in cycle) |
|---|---|---|---|---|
| `rung2ship` | **0.000** | 0.000 | — | — |
| T s60 @5 000 | 0.001 | 0.025 | 1.94 | 0.00 |
| T s60 @10 000 | 0.088 | 0.442 | 0.082 | **0.89** |
| T s60 @20 000 | **0.462** | 0.925 | 0.171 | 0.64 |
| T s63 @20 000 | 0.163 | 0.742 | 0.079 | **0.94** |

100 % of cycle steps have `digit 6 > 0` — a target the whole time, ignored. Two details fall out:
`WAITED = −0.1` **on top of** the step cost makes waiting cost 0.2 and pacing 0.1, which is why
the degenerate policy paces rather than stands still; and a long corridor with target bucket "5+"
maps several tiles to the *same* row, closing the loop as a self-loop with no gradient.

## 5 · The proximate fix is not the fix (important negative)

`BM_TIE_TOL` at evaluation on T s60 @20 000:

| TIE_TOL | score | disc. return | crates | longest 2-cycle mean/med/max |
|---|---|---|---|---|
| 0.0 (shipped) | **2.69** | **9.73** | 25.3 | 84.8 / 50 / 297 |
| 0.02 | 2.26 | 8.39 | 25.9 | 9.1 / 6 / 101 |
| 0.10 | 1.69 | 4.44 | 19.4 | 5.2 / 4 / 36 |
| 0.50 | 0.94 | −0.39 | 13.2 | 4.3 / 4 / 10 |

Tie-breaking destroys the cycles and recovers **no** score. Training with `BM_TIE_TOL=0.05` is
worse on both seeds and does not stop the collapse. **Do not ship `BM_TIE_TOL > 0`.**

## 6 · Ranked hypotheses

**H1 — accepted, causally tested.** The action-independent −0.1 step cost levels action gaps in
high-traffic rows; below ~10⁻³ the deterministic argmax is float noise and the greedy policy locks
into 2-cycles. *Mechanism:* because V is near-constant, `max Q(s′) ≈ Q(s,a)` under the greedy
action, so the update reduces to `Q ← Q − α(0.1 + (1−γ)Q)` — a constant negative drive applied
**only to whichever action is top**. It is pushed below the runner-up, which becomes greedy and is
pushed down in turn. At `STEP_COST = 0` the drive is `−α(1−γ)Q`, proportional not additive.

**H2 — accepted, enabling condition.** The feature map is phase-blind, so there is almost no true
action gap to defend (§4a).

**H3 — accepted, contributing.** `CRATE_DESTROYED` — the main income — fires 4 steps late, by
which time the agent must be *outside* the blast, so credit smears over ordinary safe rows instead
of attaching to `BOMB`. BOMB-greedy share in `(grace=0, bombu=1)` falls 0.554 → 0.320; mean
`Q_BOMB − best other` −0.99 → −2.06. Contrast: `COIN_COLLECTED` fires in the same step, and
coin-seeking never broke.

**H4 — accepted, minor.** Truncation-as-termination. Real, ≲0.2 Q-units, wrong sign.

**H5 — refuted.** "The optimum is passive and Q-learning is finding it": greedy discounted return
falls 14.04 → 9.71.

**H6 — refuted as *the* cause.** "`GOT_KILLED` = −5 dominates" (the ledger's reading): the death
term contributes **+0.5** of the −9.6 fall; the collapse is complete in rows where death is
impossible (row 12774 has no bomb anywhere, grace = 0); pure-navigation rows (32 % of steps) go
1.06 → **0.30**; and if death dominated, sd[V] would grow — it shrinks. Rising survival is a
consequence of pacing, not its cause.

**H7 — refuted.** Deterministic tie-breaking is not the bug (§5).

**H8 — refuted.** Stale warm-start cells: 0.04 %.

## 7 · The causal probe

3 arms × 2 fresh seeds (70, 71) × 10 000 episodes, everything else E30 arm T verbatim.

**Margin, fixed row set, 5 k → 10 k:**

| arm | seed 70 | seed 71 | change |
|---|---|---|---|
| ctl | 1.545 → 1.167 | 1.700 → 1.423 | −24 %, −16 % |
| tie | 1.621 → 1.235 | 1.599 → 1.150 | −24 %, −28 % |
| **step0** | **2.593 → 2.515** | **2.490 → 2.422** | **−3 %, −3 %** |

`P(gap < 0.05)` at 10 k: ctl 0.181 / 0.034 · tie 0.068 / 0.043 · **step0 0.001 / 0.001**.

**Head to head at 10 000, seed 70, ε = 0, 100 paired arenas** (discounted return scored with the
−0.1 step cost in **both** cases, i.e. on the control's own objective):

| | score | **disc. return** | crates | crates/bomb | alive | survived | 2-cycle share | longest run |
|---|---|---|---|---|---|---|---|---|
| `a4step0` | **3.51** | **14.43** | **32.3** | 1.11 | 249 | 0.28 | 0.115 | 2.8 / 2 / **6** |
| `a4ctl` | 2.40 | 8.94 | 24.5 | 0.77 | 303 | 0.52 | 0.227 | 24.4 / 4 / **319** |

**Caveats stated by the agent:** 2 seeds, 10 000 episodes, and control seed 71 had not yet
collapsed at 10 000 — the collapse is decisive on all five original seeds only by 20 000. `step0`
also removes shortest-path pressure, so it must be re-checked at 20 000 before it is believed as a
*fix* rather than as a mechanism test.

## 8 · Recommended next experiment → became E31

`BM_STEP_COST = 0`, 5 seeds, 20 000 episodes, checkpoints 5 k/10 k/20 k, otherwise E30 arm T
verbatim. If shortest-path pressure turns out to be needed, the follow-up is a **potential-based**
substitute (`BM_SHAPE` exists) rather than a flat per-step constant, because a state function
cancels out of the action gap the same way but does not ratchet the argmax.

Also recommended and carried forward separately:

- **H2 arm**, only if P1 passes and P3 fails: append a 4-bucket board-depletion/phase digit.
  Falsifier: `corr(Q,G)` stays below 0.5 and late-game bias stays above +10.
- **H3 arm**, independent: attach crate credit to the act of bombing via a custom event at `BOMB`
  carrying the crates its blast would clear, priced so per-episode crate income is unchanged.
  Falsifier: BOMB premium in `(grace=0, bombu=1)` stays below 0.2.
- **H4 one-line fix regardless:** `end_of_round` should bootstrap when the round ended at
  `MAX_STEPS` rather than by death (`last_game_state["step"] >= s.MAX_STEPS` and no `GOT_KILLED`).

**Do not run:** `BM_TIE_TOL > 0`, more episodes at the current configuration (converging, not
under-trained), or another D₄ arm (E30 closed it).
