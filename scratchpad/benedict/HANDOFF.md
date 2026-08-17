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

**E37 is COMPLETE (2026-08-16) and is the first positive feature-map result on rung 4.** 60 runs,
180 evaluations, full entry with audit 9's corrections at the top of `experiments/benedict.md`.

- **H1 holds.** `PLB2` `(x+y)%4` − `ctl2` on score **+0.28 raw, +0.20 after removing one collapsed
  control seed** (s105, a genuine training collapse worth ~29 % of the headline). Significant under
  t, bootstrap, Wilcoxon and sign test; monotone across checkpoints; replicated in the training
  stream. **Report `PLB2` − `SHF` = +0.2005 [+0.0437, +0.3572] as the primary contrast** — same
  arity, same parent, same α dilution, immune to the outlier.
- **Mechanism, measured in the Q-tables:** the lattice bit does ~3× the work of the arbitrary bit
  (across/within split 1.123 vs 0.361, 15/15), conditional on the fuse (crossings worth more at
  `own_danger` = 1, less at 2/3/4), and it pays through **bomb siting** (crates/bomb +0.130), not
  survival. Sixth replication that survival does not convert.
- **Do NOT obey the continuation rule as written.** It fires on H1 ∧ (H2 ∨ H3) and the supplying
  leg is H3 — an accept-the-null test whose CI tolerates 96 % of the treatment effect. Size any
  follow-up from the **realised** paired SD of 0.358 (n ≈ 30-40 on score; `won` is unreachable),
  and add a 2-way null to test the α-dilution explanation for `PLB2` > `PAR`.
- **Pairing at run level bought nothing** (corr −0.47…+0.44). Unseeded training opponents destroy
  it. Every rung-4 power calculation that assumed pairing was wrong.

Tools written for it, all reusable: `e37_analyse.py` (scores pre-registered hypotheses
mechanically; validated by reproducing E36's published numbers exactly), `e37_status.py`,
`e37_coverage.py`, `e37_eval.sh`.

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
