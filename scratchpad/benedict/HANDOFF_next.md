# Session handoff — paste everything below the rule into a fresh session

---

You are continuing a Bomberman RL university project at
`/Users/benedictvonschubert/Projects/bomberman_RL`. Read `AGENTS.md` first — it is the project
spec and it binds you. `CLAUDE.md` adds tool-specific notes.

## Read these before proposing anything

1. **`scratchpad/strategy/NEXT_STEPS.md`** — the ranked plan from the last session. Start here.
2. `scratchpad/strategy/README.md` — index: every headline number with the probe that produced it.
3. `scratchpad/strategy/TASK_A_survey_vs_ours.md` — published prior work mined against our features.
4. `scratchpad/strategy/TASK_B_argument.md` — the strategy argument; §6 tests the hunt hypothesis,
   §7 measures its ceiling.
5. `experiments/benedict_task4.md` — the rung-4 consolidation: what ships, the mechanism, eight
   failed interventions, nine method failures, limitations.

**`experiments/benedict.md` is ~8 000 lines. Never read it whole — it has stalled an agent.** Read
lines 1–500 with `offset`/`limit`; that covers E38 back to E36.

**The shipped agent:** `agent_code/benedict_task4/` — tabular Q-learning, 8-digit hand-built state.
3.949 score / `won` 0.406 on held-out seed 990731 vs 3 × `rule_based_agent` (which scores 3.254).

## The workflow — follow this exactly

**Benedict runs everything and commits everything. You never touch git.**

1. **You draft code changes in chat**; he applies them. `agent_code/**` is his — do not edit it
   unless he asks in this session. Exceptions are granted per task and expire.
2. **Never** `git add`/`commit`/`push`, and never edit `.gitignore` — propose, don't apply.
3. Never edit the framework files (`environment.py`, `agents.py`, `items.py`, `settings.py`,
   `main.py`); they are reset to upstream for the tournament.
4. **You own** `AGENTS.md`, `MEASUREMENT.md`, `tools/`, `experiments/*.md`, `scratchpad/`.
5. **You write the prediction into `experiments/benedict.md` yourself, before the run**, matching
   the style already in the file. Never hand him ledger text to paste.
6. **You give him the commands** — a launcher in `scratchpad/benedict/` (`eNN_arms.sh`), an eval
   launcher (`eNN_eval.sh`), and the git commands as a copy-paste block with a short message.
7. **You finish the ledger entry** with the result, scoring every pre-registered prediction **as
   written**, including the ones that fail. Never quietly redefine a statistic that failed.
8. If he says "watch it and evaluate when done", you may run the sweep-watcher and the evaluations
   yourself. That permission is **per-run**.

**Sweep scripts live in `scratchpad/benedict/` and get committed with the results — never `/tmp`.**
A launcher deleted on reboot is not evidence. An eval launcher must export the same `callbacks.py`
switches its arm trained with, derived from the same case block as the training launcher.

## How the thinking is done here

**Delegate reasoning to subagents, not just searching** — what experiment comes next, why something
failed, whether a conclusion is safe. Nine audits have run; **eight overturned a claim the author
was confident about.**

- **Ask it to break the claim, not check it.** "Find what is wrong with X" and "is X right?"
  produce visibly different work.
- **Don't state your hypothesis when the question is diagnostic.** Agreement is only evidence if it
  was reached independently.
- Tell it the hard rules verbatim; it writes only to its own `scratchpad/auditN/`; it **analyses,
  it does not run the experiment**. Bound its ledger read and ask for the report on disk
  incrementally — agents have returned paths they never created.
- **Verify its load-bearing claims yourself before acting.** Audits have been wrong too.

## Measurement rules that have each been learned the expensive way

- **Never change `--seed`.** Validation 550731, held-out ship 990731, training 810731. 1000 rounds
  for anything reported.
- **A change counts only if the paired 95 % CI excludes 0.** Negative results stay in the report.
- **State the MDE before running.** Between-seed score SD is 0.249 → n = 15 MDE is 0.254; ≈0.11
  with a collapse screen. Four rung-4 entries pre-registered targets below their own detection
  threshold. **Never pre-register `won` as primary** — its MDE is 0.029 against a conversion of
  +0.088 `won` per score point, so anything under +0.33 score is unreadable by construction.
- **A 1000-round evaluation has a ±0.12 noise floor on `score`** from the opponents' unseeded
  stdlib RNG. A single evaluation is confirmation, never the effect.
- **A training curve is not a result.** Nothing is claimed until ε = 0 through `evaluate.py`. This
  has bitten six times, most recently a "passivity" reading that vanished at ε = 0.
- **Beware pooled statistics over a growing population** — they measure the population. Two
  separate errors of this exact shape have been caught, one day apart.
- **Ceiling-test before building a feature.** `scratchpad/strategy/hunt_ceiling.py` is the pattern:
  an oracle policy over the shipped table driven through the provided `user_agent`, no training,
  paired arenas, straight into `analyze.py --compare`. It settled the hunt question in an afternoon
  at a resolution a 15-seed sweep could not reach. **If the oracle doesn't clear the sweep's MDE,
  the feature cannot.**

## Open items the last session deliberately left

- **The `won`-vs-score correction is APPLIED** (2026-08-17), so do not redo it. Verified against
  `final_project.pdf` §3 verbatim — *"multiple episodes of the game will be played to determine a
  winner by total score"* — and the words *rank* / *per-round* / *wins the round* appear nowhere in
  the spec. `AGENTS.md`, `tools/evaluate.py` and `benedict_task4.md` §6 now say score is primary and
  `won` is a secondary; the conversion is the measured marginal **+0.088 [+0.076, +0.101]** per
  point, not the ratio-of-means 0.113 that was there before.
  **Still open:** the PDF never spells out the tournament format beyond "a tournament between all
  trained agents", so the sentence is about a *game*. One message on `#final-project-questions`
  closes it. Ask before the report locks.
- **`NEXT_STEPS.md` §0.2 is unactioned and cheap:** the course explicitly sanctions downloading
  other teams' agents on `#final-project-beat-my-agent`. Every one of the 412 committed rung-4
  evaluations is against `rule_based_agent`, and §1 of that file argues a real opponent field
  outranks any feature change.
- **The hunt ceiling has no ledger entry.** It is a complete, pre-registerable negative and belongs
  in `experiments/benedict.md` as its own entry.
- **None of that session's work has been audited.**

Deadlines: agent code **21.09.2026**, report **28.09.2026** (~4 000 words per member). The
scientific method is the main grading criterion. **Two models must be described** — Benedict's line
is tabular throughout, so confirm someone on the team has the second.

Start by reading `NEXT_STEPS.md` and telling Benedict what you would do first and why, before
touching anything.
