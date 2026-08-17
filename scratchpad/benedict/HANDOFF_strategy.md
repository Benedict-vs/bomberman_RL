# Handoff prompt — prior-work review and rung-4 strategy

Paste everything below the line into a fresh session.

---

You are picking up a Bomberman RL university project at
`/Users/benedictvonschubert/Projects/bomberman_RL`. Read `AGENTS.md` first — it is the project
spec and it binds you. `CLAUDE.md` adds tool-specific notes. This session is **analysis and
discussion, not a training sweep**: I want to review prior work and decide strategy, not launch
another experiment today.

## Hard rules

- **Never run `git add`, `git commit`, `git push`, and never edit `.gitignore`.** I commit,
  always. Propose changes; I apply them.
- **Do not edit `agent_code/**` unless I explicitly ask in this session.** Exceptions are granted
  per task and expire.
- Never edit the framework files (`environment.py`, `agents.py`, `items.py`, `settings.py`,
  `main.py`) — they are reset to upstream for the tournament.
- You may write freely to `scratchpad/`. `experiments/*.md`, `AGENTS.md`, `MEASUREMENT.md` and
  `tools/` are yours to edit **when I ask**, not on your own initiative.
- **`experiments/benedict.md` is ~7 800 lines. Do NOT read it whole — it has stalled an agent.**
  Read lines 1–500 with `offset`/`limit`; that covers E38 back to E36.

## Where things stand

**A new agent shipped today (E37).** `agent_code/benedict_task4/q_table.npy` is a tabular
Q-learning table over an 8-digit hand-built state. On the held-out seed 990731, 1000 rounds
against 3 × `rule_based_agent`: **score 3.949, `won` 0.406, kills 0.226, suicides 0.488**, against
`rule_based_agent`'s 3.254 / 0.286 in the same slot. The win was a *feature*: digit 8 now carries
the wall lattice (`(x+y) % 4`) in the rows where the agent is inside a blast.

Read these, in this order, and they are short:

1. `experiments/benedict_task4.md` — the rung-4 consolidation. What ships, the mechanism measured
   in the Q-tables, eight failed interventions, nine method failures, and the limitations.
2. `experiments/benedict_task3.md` (162 lines) — the previous rung, including why training
   *destroyed* the policy there and a feature won instead.
3. `scratchpad/survey/REPORT.md` (161 lines) — see task A.

**E38 finished and closed the last open hyperparameter.** 5 seeds × 300 000 episodes, every
checkpoint evaluated at ε = 0. The training horizon is an inverted U and **20 000 sits at the
peak**:

| episodes | 5k | 10k | **20k** | 40k | 80k | 160k | 300k |
|---|---|---|---|---|---|---|---|
| score | 3.722 | 3.834 | **3.976** | 3.810 | 3.848 | 3.637 | 3.206 |

Training to 300 000 costs **−0.770 [−1.355, −0.185], 0 of 5 seeds improving**; going earlier than
20 000 is significantly worse (@20k − @5k = +0.172 [+0.111, +0.234], 14/15). **So do not propose
"train it longer" or "train it less" — both are measured and both are worse.**

Three things from it that matter for strategy:

- **The loss runs through bomb siting**: crates per bomb 1.182 → 0.988 over the horizon, against
  the +0.130 the shipped feature bought. Bomb *placement quality* is the sensitive quantity on this
  rung, more than survival or aggression.
- **It is not margin erosion.** Rows already updated at 20 000 get *sharper* (thin-margin share
  0.37×). Whatever degrades is not the table losing its decisions.
- **Within-run checkpoint contrasts have SD 0.155 against 0.358 for between-arm contrasts** — 2.3×
  more precise, because everything except training length is shared. If a question can be framed as
  "how long" rather than "which arm", it is far cheaper to answer. Worth remembering when you cost
  out any design.

**Deadlines:** agent code **21.09.2026**, report **28.09.2026** (~4 000 words per team member).
The scientific method is the main grading criterion and "Experiments and Results" is the most
important section. **At least two different models must be described in the report** — my line is
tabular Q-learning end to end, so if nobody on the team has a working second model that is a
grading problem, not a polish problem.

---

## Task A — the prior-work survey

A while back I commissioned a survey of published work on this exact course project. It lives at
`scratchpad/survey/REPORT.md`: ~8 real sources with written prose out of ~150 repos scanned,
including a 47-page report from the same course two years ago and one project that reached ≈5.0
score with a **335-state** table — against our 64 000 rows.

I want you to **mine it against what we actually built**. Specifically:

1. **What did they encode that we do not?** For each candidate, say what information the digit
   carries, whether our eight digits already carry it (read `agent_code/benedict_task4/callbacks.py`
   — it is now clean and short), and what it would cost us in rows. Our feature map is the binding
   constraint, so a feature that costs zero new rows is worth far more than one that multiplies the
   table.
2. **The 335-state result is the one to start on, and I have checked it is comparable.** Source E
   (the same course, SS2024) reports **5.04 mean score over 1000 rounds against 3 ×
   `rule_based_agent`** — our exact protocol — with a **335-state** tabular table. We score 3.949
   with **64 000 rows**. That is 191× the state space for 78 % of the score.

   The survey's own synthesis says the same thing three ways: *"aggressive state-space reduction
   beats richness (M 72→12; C 10⁵→256; E 2²⁰→2160→335) — three independent ablations, all the same
   direction."* Our project has been adding information to a large table; theirs shrank.

   So the question is not whether they are better — it is **what their 335 states encode that our
   64 000 do not**, and whether the gap is the abstraction or something else entirely (their
   pretraining curriculum, their reward table, a measurement difference I have not spotted). Work
   out which, quantify it, and say how confident you are. **This is the single highest-value
   question in the session**, because if state-space reduction is the answer then almost everything
   in `experiments/benedict_task4.md` is optimising the wrong axis.

   Two specific things in the survey to weigh: E's **"first step toward the nearest *safe* tile"**
   (BFS over survivable paths only), which they say cut their state space 6× and which they return
   *before* their other goals; and their **pretraining curriculum** — six hand-built scenarios with
   γ = 0 so it learns immediate rewards only, then the whole cycle again with the bomb-availability
   bit pinned to 0, reporting every state populated before main training. That is a different
   answer to "rare states are never visited" than the ε floor we use.
3. **What did they try that failed?** Failures are more useful to us than successes right now,
   because we have five weeks and cannot afford a dead end.
4. Anything in the survey that **contradicts** a conclusion in `experiments/benedict_task4.md`.
   Say so plainly if you find one.

Do not copy code or transcribe implementations — anything we adopt we implement ourselves and cite.
The survey was written under that constraint and you should keep it.

## Task B — strategy, which is what I actually want to argue about

The question I keep coming back to: **should the agent hunt opponents rather than farm coins?**
Concretely — bombing *towards* another agent to force a kill, instead of treating opponents as
something to avoid while collecting.

Relevant facts from our own measurements. **Challenge any of them; several have been overturned by
audits before:**

- `score = coins + 5 × kills`, exactly. Kills are ~30 % of our score.
- Coins look close to saturated: 9 on the board shared four ways is a ~2.25 fair share, and we
  already take **2.82**. Kills are **0.226** out of ~1.85 opponent deaths per round in our field —
  but ~82 % of those are the opponents' own suicides, so the pool is not free.
- **Pricing kills did not work** (E35). `KILLED_OPPONENT` at 5 and at 25 moved kills by −0.002
  while training reward rose 26.82 → 31.31 with 17 % of it kill income. The reward reached the
  learner and the policy ignored it. The diagnosis was representational: digit 7 is **one bit**
  shared between "a bomb here opens a crate" and "a bomb here catches an opponent", so the price
  cannot reach the decision.
- **Six separate interventions that bought survival did not convert it into points.** The one
  change that did pay worked through *bomb siting* — 2.26 fewer bombs, 1.76 more crates.
- `won` (rank within the round) matters more than mean score for the tournament, and we have never
  demonstrated a `won` improvement: the minimum detectable effect on it is larger than any effect
  we can produce.

Questions I want a real argument on, not a summary:

1. **Is kill-hunting the right target at all**, or is `won` won by surviving to the end while
   others blow themselves up? We have never measured what actually decides `won` in our rounds.
   That looks like a gap.
2. **If kills are the target, what is the minimum representational change** that makes them
   reachable? E35 says a price alone cannot work. Splitting digit 7 into crate-bomb and
   opponent-bomb costs a digit — how many rows, and is there a zero-cost encoding the way the
   lattice bit was?
3. **What would the experiment be**, and can it be pre-registered honestly given our measured
   noise? Read `experiments/benedict_task4.md` §5 before answering — the paired SD is 0.358 on
   score, a 1000-round evaluation has a ±0.12 noise floor from the unseeded opponents, and four
   earlier entries were underpowered on their own primary metric. **Any design you propose must
   state its MDE.**
4. Anything with a better expected value than either, given five weeks and roughly 25 sweeps of
   remaining compute.

## How I want you to work

- **Argue, do not summarise.** I can read the files myself. I want the conclusion, the number it
  rests on, and what would falsify it.
- **Verify before asserting.** Eight audits have run on this project and seven overturned a claim I
  was confident about — including the shipped agent being the wrong table for four experiments, a
  metric that undercounted by ~2×, and a placebo that turned out to be a real feature. If you make
  a load-bearing claim, name the file and the number.
- **Say when you do not know.** "This needs a measurement we have not taken" is a useful answer.
- Short diagnostic probes on existing data are welcome. Rolling out a policy or reading the Q-table
  is cheap and often decisive. Launching a training sweep is not — propose it, do not run it.
- Write anything substantial to `scratchpad/strategy/` as you go, rather than holding it all for a
  final message.

Start with Task A, since it is bounded, then come to me with Task B before writing much — I want to
argue about the framing before you invest in one direction.
