# Session handoff — paste this into a fresh session

You are picking up a Bomberman RL university project at
`/Users/benedictvonschubert/Projects/bomberman_RL`. Read `AGENTS.md` and `MEASUREMENT.md` first —
they are the project spec and bind you. `CLAUDE.md` adds tool-specific notes.

**Do not read `experiments/benedict.md` whole — it is ~7 500 lines and reading it has stalled an
agent.** Read **lines 1–420** (`Read` with offset 1, limit 420), which covers E37 back to E35.
`experiments/benedict_task3.md` (162 lines) is safe whole. The audit reports in
`scratchpad/audit3/` … `scratchpad/audit8/REPORT.md` and `scratchpad/survey/REPORT.md` are the
supporting evidence and are safe whole.

---

## The working agreement — follow this exactly

**Benedict runs everything and commits everything. You never touch git.**

1. **You draft**, in chat: the code change, where it goes, and why. He applies it. Exceptions are
   granted per-task and expire — do not assume one carries over.
2. **You write the prediction into `experiments/benedict.md` yourself**, before the run, matching
   the style already in the file. Never hand him ledger text to paste.
3. **You give him the git commands.** Short message, no author trailer. **You never run `git add`,
   `git commit`, `git push`, and never edit `.gitignore`** — propose those, never apply them.
4. **You give him the training commands**, usually a launcher in `scratchpad/benedict/`.
5. **You give him the evaluation commands.** He pastes the output; you diagnose.
6. **You finish the ledger entry** with the result, scoring every pre-registered prediction as
   written — including the ones that fail.

He often says "watch it and evaluate when done" — then you may run the sweep-watcher and the
evaluations yourself. That permission is per-run.

**Files he owns:** `agent_code/**` (especially `callbacks.py` and `train.py`).
**Files you own:** `AGENTS.md`, `MEASUREMENT.md`, `tools/`, `experiments/*.md`, `scratchpad/`.

## The thinking is delegated — this is the part that has repeatedly saved the project

**Hand the *reasoning* to a subagent, not just the searching.** Specifically: what experiment
should come next, why did something fail, and is a conclusion safe. Eight audits have run and
**seven overturned a claim I was confident about** — including the shipped agent being the wrong
table for four experiments, a metric that undercounted by ~2×, a "second Bellman fixed point" that
was one aliased row, and a placebo that was a real feature.

Rules for briefing one, all learned expensively:

- **Ask it to break the claim, not check it.** "Find what is wrong with X" and "is X right?"
  produce visibly different work.
- **Do not state your hypothesis when the question is diagnostic.** Have it form one from the data,
  then compare. Agreement is only evidence if it was reached independently.
- **Tell it the hard rules verbatim:** never `git add`/`commit`/`push`, never edit `.gitignore`,
  never edit `agent_code/**`, `experiments/**`, `AGENTS.md`, `MEASUREMENT.md`, or the framework
  files (`environment.py`, `agents.py`, `items.py`, `settings.py`, `main.py`). It writes only to
  its own `scratchpad/auditN/`.
- **It analyses; it does not run the experiment.** Short diagnostic probes and rollouts are fine
  (`BM_MODEL_SUFFIX=_aN*`, never overwriting `q_table_e3*` or `q_table_rung2ship`), but the
  training sweep and the reported evaluations are the experiment itself and belong to the
  documented workflow above.
- **Bound its ledger read** (offset/limit) and **ask for the report on disk incrementally** —
  five agents returned paths they never created and one stalled losing everything. Expect to have
  to write its report to `scratchpad/auditN/REPORT.md` yourself from the returned text.
- **Verify its load-bearing claims yourself** before acting. Audits have been wrong too — audit 8's
  own recommended sample size was underpowered by its own argument.

## Where things stand

**Shipped agent:** `agent_code/benedict_task4/q_table.npy` = E33 arm ctl seed 104. Score **3.719 /
`won` 0.372** at validation seed 550731; **3.694 / 0.372** confirmed on held-out ship seed 990731.
Beats `rule_based_agent` (3.254 / 0.286) and the measured symmetric bar (`won` 0.283). Published
prior work on this same course project reaches ≈5.0 with a **335-state** table.

**E37 is running or about to run** — 60 runs, ~11 h, four arms testing whether E36's accidental
placebo effect is real and whether its mechanism is the wall lattice. The entry, predictions and
stop rules are at the top of `experiments/benedict.md`. **It has a pre-committed continuation rule:
only if H1 and (H2 or H3) hold does anything else run; otherwise rung 4 closes.**

**Open defects, all recorded in the ledger:**
- `tools/evaluate.py:258` undercounts `killed_by_opponent` — `died − suicides` misses deaths where
  own and enemy blasts overlap, and the undercount scales with bombs placed. Every such magnitude
  since E28 is inflated ~2×. Fixing it means re-running the cited evaluations.
- `scratchpad/deaths/collect.py` reimplements `state_to_features` instead of calling it and
  hardcodes `target_dist = DIST_NONE`; correct for everything before E36, wrong after.
- `end_of_round` treats truncation at `MAX_STEPS` as termination (no bootstrap) on 33 % of rounds;
  measured at ≈8.8 Q units, never fixed alone.
- ~~`.meta.json` does not record the `BM_*` arm variables~~ — **fixed 2026-08-16** in both writers
  (`tools/evaluate.py`, `tools/trainlog.py` now record a `bm_env` dict). Runs before that date have
  no `bm_env`. E37 straddles the fix: batch 1 is `ctl2` seeds **100-109** (launched 06:22, old
  module in memory, no `bm_env`); `ctl2` 110-114 and all of `PLB2`/`PAR`/`SHF` have it. So the
  gap is 10 of 60 runs but **10 of the control arm's 15**. See `MEASUREMENT.md`.
- ~~`agent_code/benedict_task4/callbacks.py` may still carry `# <-- restore these three lines`~~ —
  removed with the E37 edits, verified absent 2026-08-16.

## Sweep scripts go in `scratchpad/benedict/`, never `/tmp`

Learned 2026-08-16. "Did E36's evaluations export `BM_OPPDIST`?" was unanswerable from the repo —
the eval launcher had been written to `/tmp/e36_watch.sh`, and only survived by luck. The answer
was yes, but a switch that lives in `callbacks.py` changes **which row a state indexes**, so
getting it wrong would have invalidated the arm rather than just annoying the reader.

So: **every watcher and eval launcher lives beside its `*_arms.sh` and is committed with the
results.** Naming is `eNN_arms.sh` / `eNN_eval.sh`. The eval launcher must set the same
`callbacks.py` switches its arm trained with, derived from the *same case block* as the training
launcher so the two cannot drift.

**Project deliverables:** agent code due **21.09.2026**, report due **28.09.2026** (~4 000 words per
team member; "Experiments and Results" is the most important section and the scientific method is
the main grading criterion). Two models must be described. `docker build .` has never been run.
`experiments/benedict.md` is long enough that a consolidation like `experiments/benedict_task3.md`
should be written for rung 4 before the report.

**The single most transferable finding, which outranks any individual entry:** paired SD of `won`
differences is 0.023, so **n = 5 has an 80 %-power MDE of 0.045** — and E33, E34, E35 and E36 all
pre-registered `won` targets *below* that. "Five pre-registered negatives" partly measures the
design, not the interventions. Rung-4 experiments need n ≈ 15–20.
