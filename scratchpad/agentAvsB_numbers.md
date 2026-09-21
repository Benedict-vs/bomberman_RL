# Agent A vs Agent B — measured head-to-head numbers

Purpose: verified numbers for the report's Agent-A-vs-Agent-B comparison section.
Not the report text itself — just the measured facts, per instructions.

- Agent A = `agent_code/Agent_A/` = byte-identical copy of `agent_code/benedict_task4/`
  (checked: `diff -q` on `callbacks.py` and `q_table.npy` both empty). This is the E37
  table (`q_table_e37_PLB2_s106__ep20000.npy`) plus the E51 certain-death move filter,
  which is **on by default** in the current code (no env var needed).
- Agent B = `agent_code/Agent_B/` = loads `Agent_B_model.pt`, sha256
  `d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`, byte-identical to
  `agent_code/dqn_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt` and
  `agent_code/ben_task4/archived_models/ben_task4_mixed_kill_v1_2000ep_seed11.pt` — the
  2000-episode "Mixed-Kill" DQN checkpoint, seed 11.
- Tools used: `tools/evaluate.py` (never modified), `tools/analyze.py` (never modified).
  `BM_QUIET_LOGS=1` set on every run. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` set on
  Agent-B runs only (external to the agent's code, to stop 3 concurrent PyTorch
  processes from oversubscribing threads — does not change the model or its output).

**Bottom line on what existed vs. what was run:** none of the three requested
evaluations (Agent A **and** Agent B, same conditions) pre-existed in any form.
`results/eval/task4_tournament/` had Agent-A-only data at seed 990731 for both fields
(some stale — pre-E51-filter — some current-equivalent), and Agent-B-only data at the
*default* seed (20260731), never 990731, never against binary_v6 as a 3-copy field. All
five underlying runs (Agent A ×2, Agent B ×2, plus the evaluation-3 shared board) were
executed fresh in this session. Details and the verification of the two "reportedly"
figures follow.

## Pre-flight: verifying the two "reportedly" numbers against the CSVs

Before running anything new, checked what already exists under `results/eval/`.

**Finding 1 — the two committed `benedict_task4_shipped_e37__task4_*_ship990731.csv`
files (commits `6671c04`, Aug 16, and `5819f0f`, Aug 18) predate the E51 death filter.**
`167bbaa` ("E51 ships: death filter on by default") landed **2026-08-23**, a week after
both. Diffing `callbacks.py` at `2e3f57f` (Aug 22, the repaired pre-ship E51 commit,
run with `BM_DEATH_FILTER=1` explicitly) against current `HEAD` shows the **only**
difference is the default value of that one env var (`"0"` → `"1"`) plus comments — so
`2e3f57f` with the flag on is byte-for-byte the same policy as current Agent A, while
the two `shipped_e37` CSVs are the **pre-filter** policy. `git diff 167bbaa HEAD --
agent_code/benedict_task4/callbacks.py` is empty (fully identical).

Computed from the actual CSVs (`tools/analyze.py --preset task4`), `benedict_task4` row,
seed 990731, n=1000 for the shipped_e37 files, n=4000 for the e51_1 (filter-on) files:

| Source | Filter | Field | n | score mean [95% CI] |
|---|---|---|---|---|
| `benedict_task4_shipped_e37__task4_rb_ship990731.csv` | OFF (pre-E51) | 3×rule_based | 1000 | 3.828 [3.661, 3.994] |
| `benedict_q_e51_1__task4_rb_ship990731.csv` | ON (=current) | 3×rule_based | 4000 | 4.050 [3.962, 4.139] |
| `benedict_task4_shipped_e37__task4_ext_xiaoxiae_binary_v6_ship990731.csv` | OFF (pre-E51) | 3×binary_v6 | 1000 | 2.611 [2.474, 2.752] |
| `benedict_q_e51_1__task4_ext_xiaoxiae_binary_v6_ship990731.csv` | ON (=current) | 3×binary_v6 | 4000 | 2.877 [2.800, 2.956] |

Paired filter-on vs filter-off (same arenas, n=4000, `tools/analyze.py --compare`):
score **+0.121 [+0.007, +0.236]** vs rule_based, **+0.099 [+0.067, +0.132]** vs
binary_v6, both verdict BETTER, neither flagged fragile.

**Verdict on the two numbers given in the task — both now pinned down exactly, not just
bounded.** After running the fresh, same-batch evaluations below, I went back and
checked both "reportedly" numbers bit-for-bit against the first 1000 rows (seeds
990731-991730) of the two n=4000 `benedict_q_e51_1__task4_*_ship990731.csv` files
(commit `2e3f57f`, confirmed above to be behaviourally identical to current Agent A):

- **"4.022" vs 3×rule_based** = the exact mean score of `benedict_q_e51_1__task4_rb_
  ship990731.csv`'s first 1000 rounds (computed value: **4.022**, matching to three
  decimals). This is a real, reproducible number — it is *not* a match for the
  pre-filter 3.828, confirming it describes the current, filtered agent, and it is
  exactly the first-1000-round subset of an existing committed file, not a rounding or
  a misremembering.
- **"2.651" vs 3×binary_v6** = the exact mean of `benedict_q_e51_1__task4_ext_
  xiaoxiae_binary_v6_ship990731.csv`'s first 1000 rounds (computed value: **2.651**,
  exact match to three decimals).
  My initial read of this (comparing unpaired CIs against the n=4000 full-file mean of
  2.877) wrongly concluded this must be the pre-filter number — **that conclusion was
  wrong and is corrected below**: it is the current, filtered agent, and the apparent
  gap to 2.877 is pure sampling variation across *different, non-overlapping arenas*
  (seeds 990731-991730 vs the full 990731-994730), not a code or filter difference, and
  not cross-process noise either — see the bit-exact check right below.

**The bit-exact check (this is the important methodological finding):** I compared my
fresh runs, round by round, against the first 1000 rows of the two `e51_1` files
(same seed range, confirmed-identical code, but a *different process*, 4 weeks apart,
different commit hash):

| Field | Old (`e51_1`, first 1000, commit `2e3f57f`) | Fresh (this session, commit `4d80416`) | Rounds with every field bit-identical |
|---|---|---|---|
| 3×rule_based, seed 990731 | mean score 4.022 | mean score 4.015 | **13 / 1000 (1.3%)** |
| 3×binary_v6, seed 990731 | mean score 2.651 | mean score 2.651 | **1000 / 1000 (100%)** |

This is a direct, empirical confirmation of the asymmetry the task description asserts:
the 3×binary_v6 field really is bit-exact deterministic across independent process
invocations (`ext_xiaoxiae_binary_v6`'s `act()` is a frozen-model argmax with no RNG
anywhere in its inference path — checked `agent_code/ext_xiaoxiae_binary_v6/callbacks.py`
and `_agent_defs.py`: the only `random.sample` call is in the training-only replay
buffer, never reached by `evaluate.py`), while the 3×rule_based field is not — only
1.3% of rounds reproduce here (same qualitative conclusion as the project's earlier
20-23% figures for other agent pairs; the exact rate is comparison-specific). Agent A's
own tie-breaking RNG (`agent_code/Agent_A/callbacks.py:565`,
`self.policy_rng = np.random.default_rng(POLICY_SEED)` with `POLICY_SEED = 20260731`
hard-coded) is fixed-seeded and therefore not itself a noise source. So: for the
binary_v6 field, "same batch" is provably not necessary for reproducibility (any two
runs of the same code at the same seed match exactly) — it was still run fresh here to
follow the instruction and to get Agent B in an identical-provenance CSV. For the
rule_based field, "same batch" is exactly as necessary as `AGENTS.md` says, and neither
the old 4.022 figure nor my fresh 4.015 should be treated as *the* number — both sit
inside the documented ±0.12-ish noise band, which is why the paired comparison in
evaluation (1) below is against a same-batch Agent A run, not the old CSV.

## Evaluations run (all fresh, this session, in parallel background jobs, same commit)

Repo state at launch: `HEAD=4d80416`, working tree dirty (untracked `Agent_A/`, `Agent_B/`,
`benedict_cnn/`, `data/`, `submission/` — none touched by this task).

Status:

- [x] (1) Agent A vs 3×rule_based, seed 990731, n=1000 → `AgentA__task4_rb_ship990731`
- [x] (1) Agent B vs 3×rule_based, seed 990731, n=1000 → `AgentB__task4_rb_ship990731`
- [x] (2) Agent A vs 3×binary_v6, seed 990731, n=1000 → `AgentA__task4_ext_xiaoxiae_binary_v6_ship990731`
- [x] (2) Agent B vs 3×binary_v6, seed 990731, n=1000 → `AgentB__task4_ext_xiaoxiae_binary_v6_ship990731`
- [x] (3) Agent A + Agent B + 2×rule_based, one board, seed 20260731 (project default,
      not specified as "held-out" in the task), n=1000 → `AgentA_AgentB_2rb__task4_headtohead_seed20260731`

All 5 evaluations completed successfully, 0 crashes, 0 steps over the 0.5 s think-time
limit in any file (checked explicitly below).

(all written to `results/eval/task4_tournament/`, matching the existing naming
convention in that directory; none of these existed before this session — checked by
grep across `results/eval/` for `990731` restricted to DQN-related dirs, and by name
across `task4_tournament/` before launching. All 5 launched concurrently as background
jobs at commit `4d80416` (working tree already had the untracked `Agent_A/`, `Agent_B/`
dirs shown in git status; nothing else touched), `BM_QUIET_LOGS=1` on all,
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` added on Agent-B runs only, to stop 3 concurrent
CPU PyTorch processes from each grabbing all cores — external env, does not touch the
agent's code or weights. Machine has 10 cores; 5 processes ran concurrently, each
pinned near 100% of one core (checked with `ps aux` mid-run), so no evidence of
scheduling contention inflating think_ms.)

## Evaluation (1) — Agent B vs 3×rule_based, seed 990731

Both agents run **fresh, this session, same commit (`4d80416-dirty`)**, in separate
processes (unavoidable: each occupies the single "our" slot against 3 copies of the
opponent, so they cannot share one match). `results/eval/task4_tournament/`:
`AgentA__task4_rb_ship990731.csv/.meta.json` (wall clock 304.3 s) and
`AgentB__task4_rb_ship990731.csv/.meta.json` (wall clock 347.9 s), both n=1000,
base_seed=990731, scenario=classic.

Solo means (bootstrap 95% CI over rounds, unpaired), `tools/analyze.py --preset task4`:

| Agent | score | won | kills | suicides | killed_by | think_max_ms |
|---|---|---|---|---|---|---|
| Agent_A | 4.015 [3.848, 4.185] | 0.398 [0.368,0.428] | 0.237 [0.209,0.266] | 0.468 [0.437,0.499] | 0.050 [0.037,0.064] | 0.4–0.5 |
| Agent_B | 3.915 [3.751, 4.082] | 0.433 [0.402,0.463] | 0.200 [0.175,0.226] | 0.392 [0.362,0.423] | 0.061 [0.047,0.076] | 0.5 |

(coins/survived solo means are the "A mean"/"B mean" columns of the paired table right
below — not repeated here since n_a=n_b=1000 fully-overlapping seeds make them the
same numbers either way.)

**Paired difference (B − A), same seeds, n=1000 paired rounds**, `tools/analyze.py
--compare AgentA__task4_rb_ship990731.csv AgentB__task4_rb_ship990731.csv`:

| Metric | A mean | B mean | Paired diff (B−A) [95% CI] | t | sign-flip p | Verdict | fragile? |
|---|---|---|---|---|---|---|---|
| score | 4.015 | 3.915 | −0.100 [−0.328, +0.125] | −0.86 | 0.397 | no effect shown | no |
| coins | 2.830 | 2.915 | +0.085 [−0.023, +0.193] | +1.55 | 0.128 | no effect shown | no |
| kills | 0.237 | 0.200 | −0.037 [−0.075, +0.001] | −1.89 | 0.066 | no effect shown | no |
| suicides | 0.468 | 0.392 | −0.076 [−0.121, −0.032] | −3.38 | 0.0009 | BETTER (B) | no |
| killed_by opp. | 0.050 | 0.061 | +0.011 [−0.009, +0.031] | +1.05 | 0.345 | no effect shown | no |
| survived | 0.482 | 0.547 | +0.065 [+0.021, +0.110] | +2.89 | 0.0044 | BETTER (B) | no |
| think_max_ms | 0.4 | 0.5 | +0.038 [−0.003, +0.078] | +1.86 | 0.062 | no effect shown | no |

**Verdict (1):** Score difference is **not** statistically demonstrated (paired
−0.100, 95% CI crosses 0, magnitude smaller than the project's documented ±0.12
same-agent noise floor). Agent B suicides significantly less and survives
significantly more than Agent A in this field; neither translates into a demonstrated
score edge at n=1000. Both agents' `think_max_ms` are far below the 500 ms limit.

## Evaluation (2) — Agent B vs 3×binary_v6, seed 990731

Both fresh, this session, same commit (`4d80416-dirty`). `results/eval/task4_tournament/`:
`AgentA__task4_ext_xiaoxiae_binary_v6_ship990731.csv/.meta.json` (wall clock 614.3 s)
and `AgentB__task4_ext_xiaoxiae_binary_v6_ship990731.csv/.meta.json` (wall clock 675.6 s),
both n=1000, base_seed=990731, scenario=classic. As shown above, Agent A's numbers here
are **bit-exact reproductions** of the existing `benedict_q_e51_1` file's first 1000
rounds (1000/1000 rounds identical) — this field needed no "same batch" precaution, but
Agent B was still run fresh alongside it as instructed.

Solo means (bootstrap 95% CI over rounds, unpaired), `tools/analyze.py --preset task4`:

| Agent | score | won | kills | suicides | killed_by | think_max_ms (mean-of-round-max) |
|---|---|---|---|---|---|---|
| Agent_A | 2.651 [2.511, 2.797] | 0.186 [0.163,0.211] | 0.094 [0.075,0.114] | 0.546 [0.516,0.577] | 0.125 [0.105,0.146] | 0.2 |
| Agent_B | 2.295 [2.184, 2.408] | 0.120 [0.100,0.140] | 0.050 [0.036,0.064] | 0.544 [0.513,0.574] | 0.151 [0.130,0.174] | 0.3 |

**Paired difference (B − A), same seeds, n=1000 paired rounds**, `tools/analyze.py
--compare AgentA__task4_ext_xiaoxiae_binary_v6_ship990731.csv
AgentB__task4_ext_xiaoxiae_binary_v6_ship990731.csv`:

| Metric | A mean | B mean | Paired diff (B−A) [95% CI] | t | sign-flip p | Verdict | fragile? |
|---|---|---|---|---|---|---|---|
| score | 2.651 | 2.295 | −0.356 [−0.529, −0.190] | −4.12 | 0.0001 | **WORSE (B)** | no |
| coins | 2.181 | 2.045 | −0.136 [−0.238, −0.037] | −2.64 | 0.0091 | WORSE (B) | no |
| kills | 0.094 | 0.050 | −0.044 [−0.069, −0.020] | −3.59 | 0.0006 | WORSE (B) | no |
| suicides | 0.546 | 0.544 | −0.002 [−0.045, +0.039] | −0.09 | 0.964 | no effect shown | no |
| killed_by opp. | 0.125 | 0.151 | +0.026 [−0.004, +0.057] | +1.65 | 0.112 | no effect shown | no |
| survived | 0.329 | 0.305 | −0.024 [−0.064, +0.016] | −1.19 | 0.252 | no effect shown | no |
| think_max_ms | 0.2 | 0.3 | +0.105 [+0.089, +0.122] | +12.25 | 0.0000 | WORSE (B), tiny magnitude | no |

**Verdict (2):** Agent A scores significantly higher than Agent B against 3×binary_v6:
paired **−0.356 [−0.529, −0.190]** (B−A), 95% CI clearly excludes 0, not fragile. This
is the one evaluation of the three where a score difference is actually demonstrated,
and because this field is bit-exact reproducible (see above), it is also the most
trustworthy of the three score comparisons. Driven by both fewer coins and fewer kills
for B, not by suicides (statistically identical, ~0.545 for both) — i.e. Agent B is not
dying more to its own bombs here, it is simply less productive per surviving step
against this particular opponent. Both agents' `think_max_ms` are far below the 500 ms
limit (global max across all 1000 rounds: Agent A 2.377 ms, Agent B 2.272 ms — see the
timing table at the end of this file); `ext_xiaoxiae_binary_v6` itself runs noticeably
hotter (up to ~127 ms observed in the quick on-screen summary) but that is the
opponent's own budget, not ours, and still clears 500 ms by 4×.

## Evaluation (3) — Agent A and Agent B on the same board, + 2×rule_based

Single match, seed 20260731 (project default — the task did not name a seed for this
one and it is not the "held-out" 990731 used in (1)/(2), so the default was used
rather than inventing a reason to pick another). `results/eval/task4_tournament/
AgentA_AgentB_2rb__task4_headtohead_seed20260731.csv/.meta.json`, n=1000,
commit `4d80416-dirty`, wall clock 300.8 s.

This is a **true within-round pairing** — both agents played the literal same 1000
games, not just the same seeds replayed in separate processes — so there is no
stdlib-RNG cross-process noise floor to worry about here at all. Computed by loading
the one CSV and calling `tools/analyze.py`'s own `paired_effect()` on the two agents'
rows directly (script: `scratchpad/paired_same_board.py`; reuses the tool's
`load`/`paired_effect`/bootstrap machinery, no separate scoring logic):

| Metric | Agent_A mean | Agent_B mean | Paired diff (B−A) [95% CI] | t | sign-flip p | Verdict | fragile? |
|---|---|---|---|---|---|---|---|
| score | 3.641 | 3.554 | −0.087 [−0.323, +0.152] | −0.72 | 0.477 | no effect shown | no |
| coins | 2.601 | 2.664 | +0.063 [−0.075, +0.206] | +0.87 | 0.397 | no effect shown | no |
| kills | 0.208 | 0.178 | −0.030 [−0.067, +0.008] | −1.58 | 0.125 | no effect shown | no |
| suicides | 0.377 | 0.468 | +0.091 [+0.048, +0.135] | +4.04 | 0.0001 | WORSE (B) | no |
| killed_by opp. | 0.078 | 0.105 | +0.027 [+0.001, +0.052] | +2.08 | 0.044 | WORSE (B), borderline | no |
| survived | 0.545 | 0.427 | −0.118 [−0.163, −0.072] | −5.02 | 0.0000 | WORSE (B) | no |
| think_max_ms | 0.470 | 0.513 | +0.043 [+0.012, +0.079] | +2.50 | 0.0087 | WORSE (B), tiny magnitude | no |

Also on this board, for context (not part of any "reportedly" claim):
`rule_based_agent_0` scored 2.651 [2.502, 2.807] here — numerically identical to the
"reportedly 2.651" figure quoted for Agent A vs binary_v6 in the task brief, but this
is a **coincidence**: different agent (a rule_based opponent, not Agent A), different
field (2×rule_based, not 3×binary_v6), different seed. Flagging it only so it is not
mistaken for corroboration of that number — see evaluation (2) above for the actual,
bit-exact-verified check of the real "2.651" claim.

**Verdict (3):** Score difference between the two agents sharing a board is **not**
statistically demonstrated (paired −0.087, 95% CI crosses 0). Agent B suicides
*more* and survives *less* than Agent A here — the **opposite direction** from
evaluation (1) (where B suicided less / survived more against 3×rule_based). Both
are real, non-fragile, statistically significant paired effects, just measured in
different fields (3×RB alone vs. sharing a board with a second strong bomber +
2×RB) — see "things that contradict a single protocol" below.

## think_max_ms — global maximum per run (not the mean-of-round-max shown above)

`analyze.py`'s printed "Think max (ms)" column is the *mean, across rounds, of each
round's own maximum* — useful for a CI but not the number the 0.5 s tournament limit
actually bounds. Pulled the true global maximum (max over every row's
`think_max_ms`) and the `think_over_limit` sum directly from each CSV:

| Run | Agent | global max think_max_ms | steps over 0.5 s limit |
|---|---|---|---|
| vs 3×rule_based, seed 990731 | Agent_A | 9.410 ms | 0 |
| vs 3×rule_based, seed 990731 | Agent_B | 8.271 ms | 0 |
| vs 3×binary_v6, seed 990731 | Agent_A | 2.377 ms | 0 |
| vs 3×binary_v6, seed 990731 | Agent_B | 2.272 ms | 0 |
| head-to-head, seed 20260731 | Agent_A | 2.449 ms | 0 |
| head-to-head, seed 20260731 | Agent_B | 9.308 ms | 0 |

Every figure is 2-3 orders of magnitude under the 500 ms tournament limit; zero
over-limit steps anywhere for either agent, in any of the 6 (agent, field) cells. No
think-time concern for Agent B relative to Agent A.

## Summary — one-line verdict per evaluation

| # | Comparison | n | seed | Paired score diff (B−A) [95% CI] | Excludes 0? | Which scored higher |
|---|---|---|---|---|---|---|
| 1 | Agent B vs Agent A, each vs 3×rule_based | 1000 | 990731 | −0.100 [−0.328, +0.125] | No | Neither demonstrated — Agent A nominally +0.100, not significant |
| 2 | Agent B vs Agent A, each vs 3×binary_v6 | 1000 | 990731 | −0.356 [−0.529, −0.190] | **Yes** | **Agent A**, by 0.356, significant and not fragile |
| 3 | Agent A vs Agent B, same board, + 2×rule_based | 1000 | 20260731 | −0.087 [−0.323, +0.152] | No | Neither demonstrated — Agent A nominally +0.087, not significant |

None of the three `--compare` runs were flagged `(fragile)` by `analyze.py` on any
metric reported here.

**Overall:** across all three evaluations Agent A never scores significantly *worse*
than Agent B, and in one of the three (vs 3×binary_v6, the one field that is bit-exact
reproducible end to end) Agent A scores significantly *better*. The other two
comparisons (vs 3×rule_based alone, and sharing a board vs 3×rule_based split 2-2) do
not demonstrate a score difference at n=1000 either way.

## Things that contradict "these two agents can be compared under one protocol"

- **The two opponent fields are not equally trustworthy for a given sample size, and
  this is now measured, not just asserted.** Round-for-round reproducibility across
  independent processes at the same seed: **100%** vs 3×binary_v6, **1.3%** vs
  3×rule_based (13/1000 rounds bit-identical to the old `e51_1` run). A score
  difference of a given size is far easier to demonstrate in the binary_v6 field
  (evaluation 2 reached significance at n=1000) than in the rule_based field
  (evaluations 1 and 3 did not, at the same n) — this is a property of the opponent's
  RNG usage, not of which agent is being measured.
- **Agent A and Agent B are architecturally incomparable in one respect that matters
  for future same-protocol work: only Agent A has a fixed, hard-coded tie-break seed
  (`POLICY_SEED = 20260731` in `agent_code/Agent_A/callbacks.py`).** This makes Agent
  A's own contribution to any evaluation fully deterministic given the arena and
  opponents. I did not find an equivalent check needed for Agent B (its forward pass is
  a deterministic argmax over `model_result.max(1)[1]`, no sampling), so both agents are
  in fact internally deterministic — the entire noise floor documented in this project
  comes from the *provided opponents*, never from either of our own two agents.
- **Evaluations (1) and (3) put Agent B in front of qualitatively different fields**
  (3 unfamiliar rule_based opponents vs. sharing a board with a second strong,
  bomb-dropping agent it never trained against) **and get opposite-signed, both
  statistically significant, results on suicides/killed_by/survived** (B safer in (1),
  B less safe in (3)). A single "Agent B's safety relative to Agent A" number would be
  misleading; which field you measure in changes the sign, not just the magnitude.
- **Neither agent's development field is symmetric with the fields measured here.**
  Checked `agent_code/ben_task4/train.py`'s `TRAINING_OPPONENTS` table: the
  `mixed_kill_v1` arm (Agent B's training run) trains against
  `peaceful_agent, rule_based_agent, rule_based_agent` — one peaceful agent plus two
  (not three) rule_based agents, never binary_v6. Agent A's table was tuned and
  selected against `rule_based_agent` throughout (the whole E28-E51 ledger in
  `experiments/benedict_task4.md` is rule_based-centric, always 3 copies). So
  evaluation (1) (3×rule_based) is close to Agent A's own tuning field but only
  approximates Agent B's (2 of 3 opponents match, the peaceful-agent slot does not);
  evaluation (2) (3×binary_v6) is off-distribution for both agents equally, which if
  anything makes its verdict *more* comparable between them, not less.
- **The DQN's own README/spec drifted from the plan**: `AGENTS.md` describes "Model B —
  DQN on the raw board (7 channels × 17×17)"; the shipped `Agent_B/model.py` takes
  `input_channels=11`. Not a measurement problem (the model that exists is the one that
  was evaluated), just a documentation mismatch worth a one-line mention if the report
  cites the original plan.
- **Neither of the two committed pre-existing "shipped_e37" CSVs for Agent A (seed
  990731, n=1000, commits `6671c04`/`5819f0f`) represents the currently shipped Agent
  A** — both predate the E51 death filter by about a week and score ~0.1-0.2 lower than
  the current agent. Anyone pulling a "measured Agent A score" from `results/eval/
  task4_tournament/` by filename pattern alone (`shipped_e37` sounds authoritative) would
  silently grab the wrong number; the filter-inclusive numbers live in the `e51_1`-labelled
  files (`BM_DEATH_FILTER=1`) or in the fresh `AgentA__*` files from this session.
